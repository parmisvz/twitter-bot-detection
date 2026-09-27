"""Offline P0/P1 extraction from identical frozen hidden states; development only."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import threading
import time
from uuid import uuid4
import numpy as np
import pandas as pd
import torch
from src.baselines import text_digest
from src.performance_diagnosis import digest, require, write_json, Inputs
from scripts.pilot_xlmr_provenance import REVISION, PHASE, EMBED, PINNED, differences, agreement

MODEL_HASHES={
    'config.json':'d66ed8cd4f2a93b358c245e50736fa389ed4f35c0bae7aad0b32abb20c62b579',
    'model.safetensors':'6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb',
    'sentencepiece.bpe.model':'cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865',
    'tokenizer.json':'a898ea75433890f6610f4e470b8ebeb0c21dce5c8dd61f892eb09eb5919d2e2c',
    'tokenizer_config.json':'994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce'}

def snapshot():
    p=Path.home()/'.cache/huggingface/hub/models--xlm-roberta-base/snapshots'/REVISION
    require(all(digest(p/name)==h for name,h in MODEL_HASHES.items()),'Pinned snapshot differs')
    return p

def inputs(root):
    reader=Inputs(root)
    for rel,h in PINNED.items():reader.allow(root/rel,h)
    phase=reader.read(root/PHASE/'manifest.json');em=reader.read(root/EMBED/'manifest.json')
    result={}
    for split,alias in [('train','train'),('validation','val')]:
        text=reader.registered(root/PHASE,phase,f'splits/text/{split}_text.csv')
        require(text_digest(text)==em['ordered_text_sha256'][split],'Ordered text changed')
        keys=reader.registered(root/EMBED,em,f'{split}_keys.csv').user_key
        arrays={}
        for kind,rel,h in [('tweets',f'final_data/embeddings/xlmr/{alias}_tweet_embeddings.npy',em['source_sha256'][f'final_data/embeddings/xlmr/{alias}_tweet_embeddings.npy']),
                           ('users',EMBED+f'/{split}_embeddings.npy',em['artifacts'][f'{split}_embeddings.npy']['sha256'])]:
            require(digest(root/rel)==h,'Cache hash differs');reader.accessed[rel]=h
            arrays[kind]=np.load(root/rel,allow_pickle=False)
        require(arrays['tweets'].shape==(len(text),768) and np.isfinite(arrays['tweets']).all(),'Invalid tweet cache')
        groups=text.groupby('user_key',sort=True).indices
        require(keys.tolist()==list(groups),'Legacy key order differs')
        np.testing.assert_array_equal(np.vstack([arrays['tweets'][i].mean(axis=0) for i in groups.values()]),arrays['users'])
        result[split]={'text':text,'keys':keys,'legacy':arrays,'groups':groups}
    return result,reader.accessed

def masks(ids,attention,returned,special_ids):
    designated=torch.isin(ids,torch.tensor(special_ids,dtype=ids.dtype,device=ids.device))|returned.bool()
    p0=attention.bool();p1=p0&~designated
    require(p0.any(dim=1).all().item() and p1.any(dim=1).all().item(),'No usable P1 tokens; stop for policy approval')
    require(not (p1&designated).any().item() and not (p1&~p0).any().item(),'Invalid special-token exclusion')
    return p0,p1,designated

def pool(hidden,mask):
    expanded=mask.unsqueeze(-1).expand(hidden.size()).float()
    return (hidden*expanded).sum(dim=1)/expanded.sum(dim=1).clamp(min=1e-9)

def tokenize(tokenizer,text):
    e=tokenizer(text.text_model.tolist(),padding='max_length',truncation=True,max_length=128,return_special_tokens_mask=True,return_tensors='pt')
    a,b,s=masks(e['input_ids'],e['attention_mask'],e['special_tokens_mask'],tokenizer.all_special_ids)
    return e,a,b,s

def extract(root):
    from transformers import AutoTokenizer,AutoModel
    import psutil
    root=Path(root).resolve();data,source_hashes=inputs(root);path=snapshot()
    torch.set_num_threads(1);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    output=root/'final_data/embedding_versions/token_pooling_v1'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    print('EMBEDDING_VERSION:',output,flush=True)
    proc=psutil.Process();baseline=proc.memory_info().rss;peak=[baseline];stop=threading.Event()
    def monitor():
        while not stop.wait(.05):peak[0]=max(peak[0],proc.memory_info().rss)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    t=time.perf_counter();tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True,use_fast=True)
    encoded={s:tokenize(tokenizer,d['text']) for s,d in data.items()};tokenize_seconds=time.perf_counter()-t
    for split,(e,p0,p1,special) in encoded.items():
        np.savez_compressed(output/f'{split}_tokens.npz',input_ids=e['input_ids'].numpy(),attention_mask=e['attention_mask'].numpy(),
            returned_special_mask=e['special_tokens_mask'].numpy(),p0_mask=p0.numpy(),p1_mask=p1.numpy(),designated_special_mask=special.numpy())
    t=time.perf_counter();model=AutoModel.from_pretrained(str(path),local_files_only=True,attn_implementation='eager').float().cpu().eval()
    for p in model.parameters():p.requires_grad_(False)
    load_seconds=time.perf_counter()-t
    require(not model.training and not any(p.requires_grad for p in model.parameters()),'Encoder not frozen')
    def forward(e,start,stop):
        with torch.no_grad():return model(input_ids=e['input_ids'][start:stop],attention_mask=e['attention_mask'][start:stop]).last_hidden_state
    t=time.perf_counter();warm=forward(encoded['train'][0],0,16);warm_seconds=time.perf_counter()-t
    timings=[];summaries=[];agreements={};norms=[];mask_rows=[]
    for split,d in data.items():
        e,p0,p1,special=encoded[split];arrays={'P0':[],'P1':[]};audit_ids=[];audit_hidden=[]
        # Hidden-state audit covers first/last row and one row with each special ID present in attention.
        audit=set([0,len(d['text'])-1])
        for sid in tokenizer.all_special_ids:
            candidates=torch.nonzero(((e['input_ids']==sid)&p0).any(dim=1)).flatten()
            if len(candidates):audit.add(int(candidates[0]))
        start_split=time.perf_counter()
        for start in range(0,len(d['text']),16):
            end=min(start+16,len(d['text']));t=time.perf_counter();hidden=forward(e,start,end)
            for arm,mask in [('P0',p0),('P1',p1)]:arrays[arm].append(pool(hidden,mask[start:end]).numpy())
            for i in sorted(audit&set(range(start,end))):audit_ids.append(i);audit_hidden.append(hidden[i-start].numpy())
            if split=='train' and start==0:torch.testing.assert_close(hidden,warm,atol=0,rtol=0)
            elapsed=time.perf_counter()-t;timings.append({'split':split,'start':start,'tweets':end-start,'seconds':elapsed})
            if start%128==0 or end==len(d['text']):print(f'{split}: {end}/{len(d["text"])} tweets; elapsed {time.perf_counter()-start_split:.1f}s',flush=True)
        arrays={a:np.vstack(v) for a,v in arrays.items()}
        np.savez_compressed(output/f'{split}_hidden_audit.npz',rows=np.array(audit_ids),hidden=np.stack(audit_hidden))
        ledger=d['text'][['user_key','source_row','text_key']].copy();ledger.insert(0,'vector_row',np.arange(len(ledger)))
        ledger['p0_tokens']=p0.sum(1).numpy();ledger['p1_tokens']=p1.sum(1).numpy();ledger['excluded_special_tokens']=(p0&special).sum(1).numpy()
        ledger.to_csv(output/f'{split}_tweet_keys.csv',index=False);pd.DataFrame({'user_key':d['keys']}).to_csv(output/f'{split}_user_keys.csv',index=False)
        user_arrays={a:np.vstack([v[idx].mean(axis=0) for idx in d['groups'].values()]) for a,v in arrays.items()}
        for arm in ['P0','P1']:
            for level,values in [('tweet',arrays[arm]),('user',user_arrays[arm])]:
                np.save(output/f'{split}_{arm}_{level}.npy',values,allow_pickle=False)
                keys=ledger[['user_key','vector_row']] if level=='tweet' else pd.DataFrame({'user_key':d['keys']})
                norms.append(keys.assign(split=split,arm=arm,level=level,l2_norm=np.linalg.norm(values.astype(np.float64),axis=1)))
        for level,fresh,old,control,treat in [('tweet',arrays['P0'],d['legacy']['tweets'],arrays['P0'],arrays['P1']),('user',user_arrays['P0'],d['legacy']['users'],user_arrays['P0'],user_arrays['P1'])]:
            agreements[split+'_'+level]=agreement(fresh,old)
            for comparison,a,b in [('P0_vs_legacy',fresh,old),('P1_vs_P0',treat,control)]:
                f=differences(a,b);keys=ledger[['user_key','vector_row']] if level=='tweet' else pd.DataFrame({'user_key':d['keys']})
                pd.concat([keys.reset_index(drop=True),f],axis=1).to_csv(output/f'{split}_{level}_{comparison}.csv',index=False)
                for metric in f.columns:summaries.append({'split':split,'level':level,'comparison':comparison,'metric':metric,**f[metric].describe(percentiles=[.05,.5,.95,.99]).to_dict()})
        for sid,token in zip(tokenizer.all_special_ids,tokenizer.all_special_tokens):
            mask_rows.append({'split':split,'special_id':sid,'token':token,'attended_occurrences':int(((e['input_ids']==sid)&p0).sum()),'retained_P1':int(((e['input_ids']==sid)&p1).sum())})
    stop.set();thread.join();peak[0]=max(peak[0],proc.memory_info().rss)
    pd.DataFrame(timings).to_csv(output/'batch_timings.csv',index=False);pd.DataFrame(summaries).to_csv(output/'representation_difference_summary.csv',index=False)
    pd.concat(norms,ignore_index=True).to_csv(output/'embedding_norms.csv',index=False);pd.DataFrame(mask_rows).to_csv(output/'special_token_counts.csv',index=False)
    write_json(output/'agreement.json',agreements)
    write_json(output/'runtime.json',{'tokenization_seconds':tokenize_seconds,'model_load_seconds':load_seconds,'warmup_seconds':warm_seconds,
        'forward_and_dual_pooling_seconds':sum(t['seconds'] for t in timings),'tweets':sum(len(d['text']) for d in data.values()),
        'baseline_rss_bytes':baseline,'peak_sampled_rss_bytes':peak[0],'native_peak_working_set_bytes':getattr(proc.memory_info(),'peak_wset',None),
        'memory_scope':'whole process including tokenization and accumulated output; 50ms RSS sampling; native peak covers process lifetime',
        'representation_array_bytes':sum(p.stat().st_size for p in output.glob('*.npy'))})
    write_json(output/'provenance.json',{'revision':REVISION,'tokenizer_revision':REVISION,'model_hashes':MODEL_HASHES,'inputs_sha256':source_hashes,
        'text_digests':{s:text_digest(d['text']) for s,d in data.items()},'special_ids':tokenizer.all_special_ids,'special_tokens':tokenizer.all_special_tokens,
        'P1_definition':'attention AND NOT (ID in tokenizer.all_special_ids OR tokenizer returned special_tokens_mask)',
        'settings':{'max_length':128,'batch_size':16,'dtype':'float32','device':'cpu','threads':1,'attention':'eager','padding':'max_length','eval':True,'gradients':False,'hidden':'final','user_pooling':'arithmetic_mean'},
        'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','tokenizers','numpy','pandas','psutil','safetensors']},
        'python':platform.python_version(),'hardware':{'platform':platform.platform(),'processor':platform.processor(),'ram_bytes':psutil.virtual_memory().total,'cpu_count':psutil.cpu_count(),'cuda_available':torch.cuda.is_available()},
        'test_access':False,'same_forward_pass':True,'warmup_repeat_exact':True})
    (output/'xlmr_pooling.py').write_bytes(Path(__file__).read_bytes())
    require(all(digest(root/p)==h for p,h in source_hashes.items()),'Source changed');snapshot()
    write_json(output/'manifest.json',{'complete':True,'schema_version':'token_pooling_embeddings_v1','artifacts':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()}})
    return output

def verify(root,output):
    from transformers import AutoTokenizer
    root,output=Path(root).resolve(),Path(output).resolve();manifest=json.loads((output/'manifest.json').read_text())
    require(manifest['complete'],'Incomplete embeddings')
    for rel,meta in manifest['artifacts'].items():require(digest(output/rel)==meta['sha256'],'Embedding artifact differs')
    data,hashes=inputs(root);prov=json.loads((output/'provenance.json').read_text());require(hashes==prov['inputs_sha256'],'Input provenance differs')
    tokenizer=AutoTokenizer.from_pretrained(str(snapshot()),local_files_only=True,use_fast=True)
    require(tokenizer.all_special_ids==prov['special_ids'],'Tokenizer specials differ')
    saved_ag=json.loads((output/'agreement.json').read_text())
    for split,d in data.items():
        e,p0,p1,special=tokenize(tokenizer,d['text']);saved=np.load(output/f'{split}_tokens.npz',allow_pickle=False)
        for name,v in [('input_ids',e['input_ids']),('attention_mask',e['attention_mask']),('returned_special_mask',e['special_tokens_mask']),('p0_mask',p0),('p1_mask',p1),('designated_special_mask',special)]:np.testing.assert_array_equal(saved[name],v.numpy())
        keys=pd.read_csv(output/f'{split}_tweet_keys.csv',dtype={'user_key':str})
        require(keys.user_key.tolist()==d['text'].user_key.tolist() and keys.text_key.tolist()==d['text'].text_key.tolist(),'Tweet mapping differs')
        require(pd.read_csv(output/f'{split}_user_keys.csv',dtype=str).user_key.tolist()==d['keys'].tolist(),'User order differs')
        audit=np.load(output/f'{split}_hidden_audit.npz',allow_pickle=False)
        arrays={}
        for arm,mask in [('P0',p0),('P1',p1)]:
            v=np.load(output/f'{split}_{arm}_tweet.npy',allow_pickle=False);u=np.load(output/f'{split}_{arm}_user.npy',allow_pickle=False)
            require(v.shape==(len(d['text']),768) and np.isfinite(v).all(),'Invalid vectors')
            np.testing.assert_array_equal(u,np.vstack([v[i].mean(axis=0) for i in d['groups'].values()]))
            for i,h in zip(audit['rows'],audit['hidden']):
                expected=h[mask[int(i)].numpy()].astype(np.float64).mean(axis=0)
                np.testing.assert_allclose(v[i],expected,atol=1e-5,rtol=1e-5)
            arrays[arm]=(v,u)
        for j,level in enumerate(['tweet','user']):
            legacy=d['legacy']['tweets' if j==0 else 'users'];require(saved_ag[split+'_'+level]==agreement(arrays['P0'][j],legacy),'Legacy comparison differs')
            for label,a,b in [('P0_vs_legacy',arrays['P0'][j],legacy),('P1_vs_P0',arrays['P1'][j],arrays['P0'][j])]:
                expected=differences(a,b);f=pd.read_csv(output/f'{split}_{level}_{label}.csv')
                pd.testing.assert_frame_equal(expected,f[expected.columns],check_dtype=False,atol=1e-12,rtol=1e-10)
    return {'train_tweets':len(data['train']['text']),'validation_tweets':len(data['validation']['text']),'all_token_masks_retokenized':True,'saved_hidden_pooling_audited':True,'test_access':False,'P0_legacy_agreement':saved_ag}
