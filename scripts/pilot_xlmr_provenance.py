"""Training-only pinned XLM-R extraction provenance/runtime pilot; offline only."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import threading
import time
from uuid import uuid4
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REVISION='e73636d4f797dec63c3081bb6ed5c7b0bb3f2089'
PHASE='final_data/versions/preprocessing_v2_e861084d1a722a21'
EMBED='final_data/embedding_versions/xlmr_baselines_v2'
PINNED={PHASE+'/manifest.json':'594a2defef59d663ef171020b7cd1ba75aa321c30e7a7d09b57b560563c109c9',
        EMBED+'/manifest.json':'d68e263c41d308d350aa3a139cbb1fb9f2b703ca9ee393b4a389081009eef07d'}

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(p,value):
    Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def checked(root,rel,expected,inputs):
    p=(root/rel).resolve()
    assert p.is_relative_to(root) and not any(x in p.stem.lower() for x in ['test','validation','val_']), 'Forbidden data input'
    assert sha(p)==expected, 'Input hash mismatch: '+rel
    inputs[rel]=expected
    return p

def load_training(root):
    inputs={}
    manifests={rel:json.loads(checked(root,rel,h,inputs).read_text()) for rel,h in PINNED.items()}
    phase,em=manifests[PHASE+'/manifest.json'],manifests[EMBED+'/manifest.json']
    rel=PHASE+'/splits/text/train_text.csv'
    text=pd.read_csv(checked(root,rel,phase['artifacts']['splits/text/train_text.csv']['sha256'],inputs),dtype={'user_key':str})
    digest=hashlib.sha256(text[['user_key','text_model']].to_json(orient='values',force_ascii=False).encode()).hexdigest()
    assert digest==em['ordered_text_sha256']['train']
    rel='final_data/embeddings/xlmr/train_tweet_embeddings.npy'
    vectors=np.load(checked(root,rel,em['source_sha256'][rel],inputs),allow_pickle=False)
    rel=EMBED+'/train_keys.csv'
    keys=pd.read_csv(checked(root,rel,em['artifacts']['train_keys.csv']['sha256'],inputs),dtype={'user_key':str}).user_key
    rel=EMBED+'/train_embeddings.npy'
    users=np.load(checked(root,rel,em['artifacts']['train_embeddings.npy']['sha256'],inputs),allow_pickle=False)
    assert vectors.shape==(len(text),768) and np.isfinite(vectors).all()
    groups=text.groupby('user_key',sort=True).indices
    assert keys.tolist()==list(groups) and keys.is_unique
    np.testing.assert_array_equal(np.vstack([vectors[i].mean(axis=0) for i in groups.values()]),users)
    assert text.text_model.fillna('').str.strip().ne('').all()
    return text,vectors,keys,users,inputs,em['limitation']

def select(text,lengths):
    counts=text.groupby('user_key').size()
    categories={'single_tweet':counts.index[counts.eq(1)],'2_to_10_tweets':counts.index[counts.between(2,10)],
                '11_to_50_tweets':counts.index[counts.between(11,50)]}
    bins={'short_le32':lengths<=32,'middle_33_111':(lengths>=33)&(lengths<=111),
          'near_limit_112_128':(lengths>=112)&(lengths<=128),'truncated_gt128':lengths>128}
    categories.update({k:text.loc[v,'user_key'].unique() for k,v in bins.items()})
    chosen=set();reasons={}
    for name,candidates in categories.items():
        ordered=sorted(set(candidates),key=lambda k:(hashlib.sha256(('xlmr_provenance_v1|'+k).encode()).hexdigest(),k))
        assert len(ordered)>=2,'Insufficient candidates for '+name
        reasons[name]=ordered[:2];chosen.update(ordered[:2])
    rows=np.flatnonzero(text.user_key.isin(chosen))
    assert all(np.any(mask[rows]) for mask in bins.values())
    return rows,sorted(chosen),reasons

def differences(fresh,legacy):
    assert fresh.shape==legacy.shape and np.isfinite(fresh).all() and np.isfinite(legacy).all()
    a,b=fresh.astype(np.float64),legacy.astype(np.float64);d=a-b
    denom=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1);assert (denom>0).all()
    return pd.DataFrame({'cosine_similarity':np.clip((a*b).sum(axis=1)/denom,-1,1),
        'l2_distance':np.linalg.norm(d,axis=1),'max_absolute_difference':np.abs(d).max(axis=1),
        'mean_signed_difference':d.mean(axis=1),'fresh_l2_norm':np.linalg.norm(a,axis=1),
        'legacy_l2_norm':np.linalg.norm(b,axis=1),'relative_l2_distance':np.linalg.norm(d,axis=1)/np.linalg.norm(b,axis=1)})

def agreement(fresh,legacy):
    # Declared tolerances, not thresholds selected from observed results.
    d=differences(fresh,legacy)
    exact=np.array_equal(fresh,legacy)
    near=bool(np.allclose(fresh,legacy,atol=1e-5,rtol=1e-5) and d.cosine_similarity.min()>=.999999)
    return {'classification':'exact' if exact else 'near-exact' if near else 'materially different under declared tolerance',
            'exact':exact,'within_declared_tolerance':near,'atol':1e-5,'rtol':1e-5,
            'min_cosine':float(d.cosine_similarity.min()),'max_coordinate_difference':float(d.max_absolute_difference.max())}

def run():
    import torch
    from transformers import AutoTokenizer,AutoModel
    import psutil
    root=ROOT.resolve();text,legacy,legacy_keys,legacy_users,input_hashes,limitation=load_training(root)
    snapshot=Path.home()/'.cache/huggingface/hub/models--xlm-roberta-base/snapshots'/REVISION
    model_hashes={p.name:sha(p) for p in sorted(snapshot.iterdir()) if p.is_file()}
    assert {'model.safetensors','config.json','tokenizer.json','tokenizer_config.json','sentencepiece.bpe.model'}<=set(model_hashes)
    torch.set_num_threads(1);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    process=psutil.Process();baseline_rss=process.memory_info().rss;peak=[baseline_rss];stop=threading.Event()
    def sample_memory():
        while not stop.wait(.05):peak[0]=max(peak[0],process.memory_info().rss)
    monitor=threading.Thread(target=sample_memory,daemon=True);monitor.start()
    t=time.perf_counter();tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,use_fast=True)
    # Inspect lengths without truncation only for selection; no model inference here.
    lengths=np.asarray([len(x) for x in tokenizer(text.text_model.tolist(),padding=False,truncation=False)['input_ids']])
    selection_seconds=time.perf_counter()-t
    rows,keys,reasons=select(text,lengths);subset=text.iloc[rows].copy().reset_index(drop=True)
    output=root/'results/versions/xlmr_provenance_pilot_v1'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    ledger=subset[['user_key','source_row','text_key']].copy();ledger.insert(0,'legacy_vector_row',rows)
    ledger['untruncated_tokens']=lengths[rows];ledger['text_model_sha256']=[hashlib.sha256(s.encode()).hexdigest() for s in subset.text_model]
    ledger['token_length_group']=pd.cut(lengths[rows],[0,32,111,128,np.inf],labels=['short_le32','middle_33_111','near_limit_112_128','truncated_gt128']).astype(str)
    ledger.to_csv(output/'subset_tweets.csv',index=False)
    pd.DataFrame({'user_key':keys}).to_csv(output/'subset_users.csv',index=False)
    write(output/'selection.json',{'rule':'two lowest SHA256(xlmr_provenance_v1|user_key) per predeclared stratum; union users; ALL their prepared tweets in original global row order',
        'strata':reasons,'selected_users':len(keys),'selected_tweets':len(rows),'random_sampling':False,
        'training_ordered_text_sha256':hashlib.sha256(text[['user_key','text_model']].to_json(orient='values',force_ascii=False).encode()).hexdigest(),
        'subset_ordered_text_sha256':hashlib.sha256(subset[['user_key','text_model']].to_json(orient='values',force_ascii=False).encode()).hexdigest()})
    print('OUTPUT_VERSION:',output,flush=True);print('Selected',len(keys),'complete users /',len(rows),'tweets',flush=True)
    t=time.perf_counter();model=AutoModel.from_pretrained(str(snapshot),local_files_only=True,attn_implementation='eager').float().cpu().eval()
    for p in model.parameters():p.requires_grad_(False)
    load_seconds=time.perf_counter()-t
    assert model.config.hidden_size==768 and not model.training and not any(p.requires_grad for p in model.parameters())
    def extract(strings):
        encoded=tokenizer(strings,padding='max_length',truncation=True,max_length=128,return_tensors='pt')
        with torch.no_grad():
            hidden=model(**encoded).last_hidden_state
            mask=encoded['attention_mask'].unsqueeze(-1).expand(hidden.size()).float()
            return ((hidden*mask).sum(dim=1)/mask.sum(dim=1).clamp(min=1e-9)).cpu().numpy()
    t=time.perf_counter();warm=extract(subset.text_model.iloc[:16].tolist());warm_seconds=time.perf_counter()-t
    batches=[];fresh=[];t0=time.perf_counter()
    for start in range(0,len(subset),16):
        strings=subset.text_model.iloc[start:start+16].tolist();t=time.perf_counter();v=extract(strings);elapsed=time.perf_counter()-t
        fresh.append(v);batches.append({'start':start,'tweets':len(strings),'seconds':elapsed})
        print(f'Extracted {start+len(strings)}/{len(subset)} tweets; batch {elapsed:.2f}s',flush=True)
    extraction_seconds=time.perf_counter()-t0;fresh=np.vstack(fresh)
    repeated=agreement(fresh[:len(warm)],warm)
    np.save(output/'fresh_tweet_vectors.npy',fresh,allow_pickle=False)
    groups=subset.groupby('user_key',sort=True).indices
    assert list(groups)==keys
    user_vectors=np.vstack([fresh[i].mean(axis=0) for i in groups.values()])
    np.save(output/'fresh_user_vectors.npy',user_vectors,allow_pickle=False)
    idx=pd.Index(legacy_keys).get_indexer(keys);assert (idx>=0).all()
    tweets=pd.concat([ledger,differences(fresh,legacy[rows])],axis=1)
    users=pd.concat([pd.DataFrame({'user_key':keys,'tweet_count':[len(i) for i in groups.values()]}),differences(user_vectors,legacy_users[idx])],axis=1)
    tweets.to_csv(output/'tweet_comparison.csv',index=False);users.to_csv(output/'user_comparison.csv',index=False)
    pd.DataFrame(batches).to_csv(output/'batch_timings.csv',index=False)
    summary=[]
    for level,f in [('tweet',tweets),('user',users)]:
        for metric in ['cosine_similarity','l2_distance','max_absolute_difference','relative_l2_distance','mean_signed_difference']:
            summary.append({'level':level,'metric':metric,**f[metric].describe(percentiles=[.05,.5,.95,.99]).to_dict()})
    pd.DataFrame(summary).to_csv(output/'difference_distributions.csv',index=False)
    tweets.groupby('token_length_group')[['cosine_similarity','l2_distance','max_absolute_difference']].agg(['count','mean','median','max']).to_csv(output/'differences_by_length.csv')
    d=fresh.astype(np.float64)-legacy[rows].astype(np.float64)
    pd.DataFrame({'coordinate':np.arange(768),'mean_signed_difference':d.mean(axis=0),'mean_absolute_difference':np.abs(d).mean(axis=0),'max_absolute_difference':np.abs(d).max(axis=0)}).to_csv(output/'coordinate_differences.csv',index=False)
    stop.set();monitor.join();peak[0]=max(peak[0],process.memory_info().rss)
    native_peak=getattr(process.memory_info(),'peak_wset',None)
    rate=extraction_seconds/len(rows);total_tweets=len(text)+2618
    write(output/'runtime.json',{'selection_tokenization_seconds':selection_seconds,'model_load_seconds':load_seconds,'warmup_seconds':warm_seconds,
        'extraction_seconds':extraction_seconds,'seconds_per_tweet':rate,'amortized_seconds_per_complete_user':extraction_seconds/len(keys),
        'peak_sampled_process_rss_bytes':peak[0],'process_rss_before_model_bytes':baseline_rss,'native_peak_working_set_bytes':native_peak,
        'memory_poll_seconds':.05,'memory_measure':'whole process RSS, not model-only; native peak includes process lifetime',
        'fresh_array_storage_bytes':sum((output/p).stat().st_size for p in ['fresh_tweet_vectors.npy','fresh_user_vectors.npy']),
        'full_train_tweets':len(text),'validation_tweets_from_prior_audit_not_loaded':2618,'estimated_full_train_validation_seconds':rate*total_tweets,
        'estimated_full_including_load_warmup_seconds':rate*total_tweets+load_seconds+warm_seconds,
        'estimate_limit':'linear CPU fixed128 batch16 estimate; excludes full output/report writing; pilot selected for coverage, not random representativeness'})
    write(output/'provenance.json',{'model':'xlm-roberta-base','model_revision':REVISION,'tokenizer_revision':REVISION,'snapshot_path':str(snapshot),
        'snapshot_file_sha256':model_hashes,'inputs_sha256':input_hashes,'legacy_provenance_limitation':limitation,
        'definition':{'max_length':128,'batch_size':16,'padding':'max_length','token_pooling':'final_hidden_state_attention_mask_mean','special_tokens_included':True,'padding_excluded':True,'user_pooling':'arithmetic_mean_all_prepared_tweets','dtype':'float32','device':'cpu','gradients':False,'eval':True,'attention_implementation':'eager','tokenizer_class':type(tokenizer).__name__},
        'historical_execution_details_unknown':['encoder revision','package versions','hardware','attention backend','batch composition for subset'],
        'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','tokenizers','numpy','pandas','psutil','safetensors']},
        'hardware':{'platform':platform.platform(),'processor':platform.processor(),'logical_cpus':psutil.cpu_count(),'physical_cpus':psutil.cpu_count(logical=False),'ram_bytes':psutil.virtual_memory().total,'torch_threads':torch.get_num_threads(),'cuda_available':torch.cuda.is_available()},
        'no_test_or_validation_data_loaded':True,'no_training':True})
    write(output/'agreement.json',{'tweet':agreement(fresh,legacy[rows]),'user':agreement(user_vectors,legacy_users[idx]),'warmup_repeat':repeated})
    (output/'pilot_xlmr_provenance.py').write_bytes(Path(__file__).read_bytes())
    assert all(sha(root/p)==h for p,h in input_hashes.items()),'Training inputs changed'
    assert all(sha(snapshot/p)==h for p,h in model_hashes.items()),'Model files changed'
    write(output/'manifest.json',{'complete':True,'scope':'training_only_provenance_runtime_pilot','artifacts':{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return output

def verify(output):
    output=Path(output).resolve();manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['complete']
    for p,h in manifest['artifacts'].items():assert sha(output/p)==h
    text,legacy,keys,users,inputs,_=load_training(ROOT.resolve())
    provenance=json.loads((output/'provenance.json').read_text());assert inputs==provenance['inputs_sha256']
    snapshot=Path(provenance['snapshot_path'])
    assert snapshot.name==REVISION and all(sha(snapshot/p)==h for p,h in provenance['snapshot_file_sha256'].items())
    ledger=pd.read_csv(output/'subset_tweets.csv',dtype={'user_key':str})
    rows=ledger.legacy_vector_row.to_numpy();subset=text.iloc[rows]
    assert np.array_equal(np.flatnonzero(text.user_key.isin(ledger.user_key)),rows),'Incomplete selected users'
    assert subset.user_key.tolist()==ledger.user_key.tolist()
    assert ledger.text_model_sha256.tolist()==[hashlib.sha256(s.encode()).hexdigest() for s in subset.text_model]
    fresh=np.load(output/'fresh_tweet_vectors.npy',allow_pickle=False);fresh_users=np.load(output/'fresh_user_vectors.npy',allow_pickle=False)
    groups=subset.reset_index(drop=True).groupby('user_key',sort=True).indices
    saved_keys=pd.read_csv(output/'subset_users.csv',dtype=str).user_key.tolist();assert list(groups)==saved_keys
    np.testing.assert_array_equal(fresh_users,np.vstack([fresh[i].mean(axis=0) for i in groups.values()]))
    idx=pd.Index(keys).get_indexer(saved_keys)
    for filename,a,b in [('tweet_comparison.csv',fresh,legacy[rows]),('user_comparison.csv',fresh_users,users[idx])]:
        expected=differences(a,b);saved=pd.read_csv(output/filename)
        pd.testing.assert_frame_equal(expected,saved[expected.columns],check_dtype=False,atol=1e-12,rtol=1e-10)
    ag=json.loads((output/'agreement.json').read_text())
    # Preserve the first immutable run; normalize its abbreviated field name in memory only.
    for entry in ag.values():
        if 'allclose_atol1e5_rtol1e5' in entry:
            entry['within_declared_tolerance']=entry.pop('allclose_atol1e5_rtol1e5')
    assert ag['tweet']==agreement(fresh,legacy[rows]) and ag['user']==agreement(fresh_users,users[idx])
    return {'verified_users':len(saved_keys),'verified_tweets':len(rows),'agreement':ag,'classifier_training':False,'test_validation_data_loaded':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-run');args=parser.parse_args()
    output=Path(args.verify_run) if args.verify_run else run()
    print(json.dumps(verify(output),indent=2),flush=True)
