"""Approved contrastive_v1: fixed P1, train-only SupCon, validation-only selection."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import threading
import time
from uuid import uuid4
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.baselines import seed_all
from src.temporal_statistics import make_loader
from src.checkpoint_weight_experiment import state_hash, weights_from_training
from src.performance_diagnosis import Inputs, digest, write_json, require, plt, save_figure
from src.token_pooling import inputs as p1_inputs, scored, preserved_hashes
from src.text_aggregation import model_data
from src.contrastive_core import (backbone, head, primary, auxiliary, projected, supcon, rng_hash,
                                  equivalence, diagnostic_arrays, representation_stats, view_diagnostic)

POLICIES = ['best_unweighted_validation_loss', 'last']
METRICS = ['bot_f1', 'bot_precision', 'bot_recall', 'roc_auc', 'average_precision', 'brier', 'validation_unweighted_loss']


def load(root, config, reader):
    require(config['arms'] == ['C0', 'C1'] and config['seeds'] == [42,43,44]
            and config['training'] == {'epochs':50,'batch_size':32,'lr':.001,'weight_decay':.0001}
            and config['lambda'] == .05 and config['temperature'] == .1 and config['projection'] == [80,32,16]
            and config['normalization_epsilon'] == 1e-12 and config['threshold'] == .5
            and config['selection'] == POLICIES[0] and config['dropout'] == .3 and config['torch_threads'] == 1
            and config['expected_parameters'] == {'C0':16897,'C1':20017}
            and config['scope'] == 'frozen_P1_train_only_supcon_no_test_no_graph_no_pseudolabel_no_promotion'
            and config['contrastive_reduction'] == 'equal_mean_of_class_anchor_means'
            and config['single_class_batch'] == 'zero_contrastive_keep_classification'
            and config['head_seed_rule'] == '10000 + run_seed'
            and config['auxiliary_seed_rule'] == '20000 + 100000 * run_seed + zero_based_step', 'Approved protocol changed')
    rp = root / config['reference_manifest']
    reader.allow(rp, config['reference_manifest_sha256'])
    manifest = reader.read(rp)
    previous = reader.registered(rp.parent, manifest, 'provenance/resolved_config.json')
    require(manifest['complete'] and previous['embedding_manifest_sha256'] == config['embedding_manifest_sha256'], 'P1 reference differs')
    data, mapping, reps, counts, _, _, _ = p1_inputs(root, previous, reader)
    data = model_data(data, reps, 'P1')
    require(list(data) == ['train','validation'] and len(data['train']['y']) == 671 and len(data['validation']['y']) == 144, 'Population changed')
    require(set(data['train']['users'].index).isdisjoint(data['validation']['users'].index), 'Identity overlap')
    require(int(data['train']['y'].sum()) == 131 and int(data['validation']['y'].sum()) == 28, 'Label counts changed')
    return data, mapping, counts, rp.parent, manifest


def registered_torch(root, directory, manifest, rel, reader):
    path = directory / rel
    require(digest(path) == manifest['artifacts'][rel]['sha256'] and 'test' not in path.stem.lower(), 'Checkpoint source changed')
    reader.accessed[path.relative_to(root).as_posix()] = digest(path)
    return torch.load(path, map_location='cpu', weights_only=True)


def train(data, mapping, config, arm, seed):
    seed_all(seed, 1)
    model = backbone(mapping)
    initial = deepcopy(model.state_dict())
    initial_rng = rng_hash()
    projector = head(seed) if arm == 'C1' else None
    require(rng_hash() == initial_rng, 'Head initialization changed primary RNG')
    initial_head = deepcopy(projector.state_dict()) if projector is not None else None
    parameters = list(model.parameters()) + (list(projector.parameters()) if projector is not None else [])
    require(sum(p.numel() for p in parameters) == config['expected_parameters'][arm], 'Parameter count changed')
    spec = config['training']
    weight = weights_from_training(data['train']['y'])['full_train_ratio']
    optimizer = torch.optim.AdamW(parameters, lr=spec['lr'], weight_decay=spec['weight_decay'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], dtype=torch.float32))
    loaders = {s: make_loader(p, spec, seed, s) for s,p in data.items()}
    best, best_epoch, best_state, best_head = float('inf'), None, None, None
    history, batches, orders = [], [], []
    order_hash = hashlib.sha256()
    primary_rng_hash = hashlib.sha256()
    step = 0
    for epoch in range(1, 51):
        row = {'epoch':epoch}
        seen = []
        batch_contrast, batch_joint, batch_ratio = [], [], []
        for split, loader in loaders.items():
            model.train(split == 'train')
            state_before = state_hash(model.state_dict()) if split == 'validation' else None
            total, unweighted, n = 0., 0., 0
            with torch.set_grad_enabled(split == 'train'):
                for x,y,idx in loader:
                    if split == 'train':
                        seen.extend(idx.tolist())
                        order_hash.update(idx.numpy().tobytes())
                        optimizer.zero_grad()
                        rng_before = rng_hash()
                        logits,h = primary(model,x)
                        rng_after = rng_hash()
                        primary_rng_hash.update((rng_before + rng_after).encode())
                    else:
                        logits = model(x)
                    classification = criterion(logits,y)
                    if split == 'train':
                        bots = int(y.sum())
                        single = bots in [0,len(y)]
                        contrast = torch.zeros((), dtype=logits.dtype)
                        if projector is not None:
                            h2 = auxiliary(model,x,seed,step)
                            contrast = supcon(projected(projector,h)[1], projected(projector,h2)[1], y.flatten(), config['temperature'])
                        loss = classification + config['lambda'] * contrast if projector is not None else classification
                        require(torch.isfinite(loss).item(), 'Nonfinite training loss')
                        loss.backward()
                        optimizer.step()
                        raw = float(contrast.detach())
                        ratio = config['lambda'] * raw / float(classification.detach())
                        batches.append({'epoch':epoch,'step':step,'n':len(y),'humans':len(y)-bots,'bots':bots,
                            'single_class':single,'contrastive_skipped':bool(projector is not None and single),
                            'human_positive_views':2*(len(y)-bots)-1 if bots<len(y) else 0,
                            'bot_positive_views':2*bots-1 if bots else 0,
                            'human_negative_views':2*bots,'bot_negative_views':2*(len(y)-bots),
                            'classification_loss':float(classification.detach()),'contrastive_loss':raw,
                            'weighted_contrastive':config['lambda']*raw,'joint_loss':float(loss.detach()),
                            'contrastive_classification_ratio':ratio,'primary_rng_before':rng_before,'primary_rng_after':rng_after,
                            'batch_user_indices_sha256':hashlib.sha256(idx.numpy().tobytes()).hexdigest()})
                        batch_contrast.append(raw);batch_joint.append(float(loss.detach()));batch_ratio.append(ratio)
                        step += 1
                    else:
                        unweighted += nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='sum').item()
                    total += classification.item()
                    n += len(x)
            row[split+'_weighted_loss'] = total / len(loader)
            if split == 'validation':
                row['validation_unweighted_loss'] = unweighted / n
                require(state_before == state_hash(model.state_dict()), 'Validation updated state')
        require(sorted(seen) == list(range(671)), 'Omitted/repeated training user')
        orders.append(hashlib.sha256(np.asarray(seen,dtype=np.int64).tobytes()).hexdigest())
        row.update(train_contrastive_loss=float(np.mean(batch_contrast)), train_joint_loss=float(np.mean(batch_joint)),
                   train_weighted_contrastive=config['lambda']*float(np.mean(batch_contrast)),
                   train_mean_batch_ratio=float(np.mean(batch_ratio)),
                   train_ratio_of_mean_losses=config['lambda']*float(np.mean(batch_contrast))/row['train_weighted_loss'])
        require(np.isfinite(list(row.values())).all(), 'Nonfinite history')
        history.append(row)
        if row['validation_unweighted_loss'] < best:
            best,best_epoch = row['validation_unweighted_loss'],epoch
            best_state = deepcopy(model.state_dict())
            best_head = deepcopy(projector.state_dict()) if projector is not None else None
        if epoch % 25 == 0:
            print(f'{arm} seed {seed} epoch {epoch}: validation BCE {row["validation_unweighted_loss"]:.6f}',flush=True)
    states = {'initial':(initial,initial_head),POLICIES[0]:(best_state,best_head), 'last':(deepcopy(model.state_dict()),deepcopy(projector.state_dict()) if projector is not None else None)}
    info = {'arm':arm,'seed':seed,'parameter_count':config['expected_parameters'][arm],'inference_parameters':16897,
            'pos_weight':weight,'epochs':{POLICIES[0]:best_epoch,'last':50},'initial_state_sha256':state_hash(initial),
            'batch_order_sha256':order_hash.hexdigest(),'epoch_order_sha256':orders,'primary_rng_sha256':primary_rng_hash.hexdigest(),
            'training_user_keys':data['train']['users'].index.tolist(),'checkpoints':{}}
    return states,pd.DataFrame(history),pd.DataFrame(batches),info


def summaries(table):
    summary=table.groupby(['arm','policy','cohort'])[METRICS].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    effects=[]
    for r in table[table.arm.eq('C1')].itertuples():
        c=table[table.arm.eq('C0') & table.seed.eq(r.seed) & table.policy.eq(r.policy) & table.cohort.eq(r.cohort)].iloc[0]
        effects.append({'seed':r.seed,'policy':r.policy,'cohort':r.cohort,**{'delta_'+m:float(getattr(r,m)-c[m]) for m in METRICS}})
    return summary.reset_index(),pd.DataFrame(effects)


def decision(table):
    _,effect=summaries(table)
    f=effect[effect.cohort.eq('all') & effect.policy.eq(POLICIES[0])]
    passed=bool((f.delta_bot_f1>0).all() and f.delta_average_precision.mean()>=0)
    return {'criterion_metrics_pass':passed,'all_three_f1_improved':bool((f.delta_bot_f1>0).all()),
            'mean_delta_f1':float(f.delta_bot_f1.mean()),'mean_delta_ap':float(f.delta_average_precision.mean()),
            'requires_independent_verification':True,'promoted_model':None,'interpretation':'development_only_not_statistical_proof'}


def diagnostics(output, name, policy, model, projector, data, seed, write=True):
    summaries_out=[]
    for split,pop in data.items():
        arrays=diagnostic_arrays(model,projector,pop)
        for space,a in arrays.items():
            stem=f'{name}_{policy}_{split}_{space}'
            s,coordinates,spectrum=representation_stats(a,pop['y'])
            summaries_out.append({'run':name,'policy':policy,'split':split,'space':space,**s})
            if write:
                np.save(output/f'representations/{stem}.npy',a,allow_pickle=False)
                coordinates.to_csv(output/f'diagnostics/{stem}_coordinates.csv',index=False)
                spectrum.to_csv(output/f'diagnostics/{stem}_spectrum.csv',index=False)
            else:
                np.testing.assert_array_equal(a,np.load(output/f'representations/{stem}.npy',allow_pickle=False))
                for suffix,expected in [('coordinates',coordinates),('spectrum',spectrum)]:
                    pd.testing.assert_frame_equal(expected,pd.read_csv(output/f'diagnostics/{stem}_{suffix}.csv'),check_dtype=False,atol=1e-10,rtol=1e-8)
    views=view_diagnostic(model,projector,data['train'],seed)
    path=output/f'diagnostics/{name}_{policy}_training_dropout_views.csv'
    if write:views.to_csv(path,index=False)
    else:pd.testing.assert_frame_equal(views,pd.read_csv(path,dtype={'user_key':str}),check_dtype=False,atol=1e-10,rtol=1e-8)
    return summaries_out


def figures(output,table,histories):
    for policy,short in [(POLICIES[0],'selected'),('last','final')]:
        fig,axes=plt.subplots(2,3,figsize=(11,7))
        for i,arm in enumerate(['C0','C1']):
            for j,seed in enumerate([42,43,44]):
                r=table[table.arm.eq(arm)&table.seed.eq(seed)&table.policy.eq(policy)&table.cohort.eq('all')].iloc[0]
                cm=np.array([[r.tn,r.fp],[r.fn,r.tp]],dtype=int);ax=axes[i,j]
                ax.imshow(cm,cmap='Blues',vmin=0,vmax=116)
                for a in range(2):
                    for b in range(2):ax.text(b,a,str(cm[a,b]),ha='center',va='center',color='white' if cm[a,b]>58 else 'black')
                ax.set(title=f'{arm} seed {seed}, epoch {int(r.epoch)}',xticks=[0,1],yticks=[0,1],xticklabels=['Human','Bot'],yticklabels=['Human','Bot'],xlabel='Predicted',ylabel='True')
        save_figure(fig,output/f'figures/confusion_{short}.png')
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ax,(name,h) in zip(axes.flat,histories.items()):
        ax.plot(h.epoch,h.train_weighted_loss,label='Training weighted BCE')
        ax.plot(h.epoch,h.validation_unweighted_loss,label='Validation unweighted BCE')
        ax.axvline(int(h.loc[h.validation_unweighted_loss.idxmin(),'epoch']),ls=':',color='black')
        ax.set(title=name,xlabel='Epoch',ylabel='Loss (different objectives)');ax.legend(fontsize=8)
    save_figure(fig,output/'figures/classification_curves.png')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for name,h in histories.items():
        if name.startswith('C1'):
            for ax,column in zip(axes,['train_contrastive_loss','train_weighted_contrastive','train_ratio_of_mean_losses']):
                ax.plot(h.epoch,h[column],label=name);ax.set(xlabel='Epoch',title=column);ax.legend(fontsize=8)
    save_figure(fig,output/'figures/contrastive_curves.png')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,m in zip(axes,['bot_f1','average_precision','brier']):
        for seed in [42,43,44]:
            f=table[table.seed.eq(seed)&table.policy.eq(POLICIES[0])&table.cohort.eq('all')].set_index('arm').loc[['C0','C1']]
            ax.plot([0,1],f[m],marker='o',label=f'seed {seed}')
        ax.set(xticks=[0,1],xticklabels=['C0','C1'],title=m);ax.legend()
    save_figure(fig,output/'figures/paired_metrics.png')


def run(root,config_path='configs/contrastive_v1.json'):
    import psutil
    root=Path(root).resolve();reader=Inputs(root)
    reader.allow(root/config_path,digest(root/config_path));config=reader.read(root/config_path)
    before=preserved_hashes(root,config['output_root'])
    data,mapping,counts,reference,refmanifest=load(root,config,reader)
    output=root/config['output_root']/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid4().hex[:8])
    for folder in ['models','histories','predictions','provenance','diagnostics','representations','figures']:(output/folder).mkdir(parents=True,exist_ok=False)
    print('OUTPUT_VERSION:',output,flush=True)
    write_json(output/'provenance/resolved_config.json',config);write_json(output/'provenance/feature_mapping.json',mapping)
    for split,pop in data.items():
        pop['users'][['label_binary']].join(pop['masks']).to_csv(output/f'provenance/{split}_population.csv')
    regression=[]
    for seed in config['seeds']:
        loader=make_loader(data['train'],config['training'],seed,'train')
        batches=[(x,y) for x,y,_ in loader]
        regression.append(equivalence(mapping,batches,seed,540/131))
    write_json(output/'lambda_zero_regression.json',regression)
    print('PASS: lambda=0 exact logits, state/BN buffers, and primary RNG for 21 updates per seed',flush=True)
    start=time.perf_counter();proc=psutil.Process();peak=[proc.memory_info().rss];stop=threading.Event()
    def monitor():
        while not stop.wait(.05):peak[0]=max(peak[0],proc.memory_info().rss)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    rows=[];diags=[];runs={};histories={};runtimes=[]
    for seed in config['seeds']:
        for arm in config['arms']:
            t=time.perf_counter();states,h,b,info=train(data,mapping,config,arm,seed)
            runtimes.append({'arm':arm,'seed':seed,'training_seconds':time.perf_counter()-t})
            name=f'{arm}_seed{seed}';histories[name]=h
            h.to_csv(output/f'histories/{name}.csv',index=False);b.to_csv(output/f'histories/{name}_batches.csv',index=False)
            if arm=='C0':
                old=reader.registered(reference,refmanifest,f'histories/P1_seed{seed}.csv')
                pd.testing.assert_frame_equal(h[old.columns],old,check_dtype=False,atol=1e-7,rtol=1e-7)
                require(info['batch_order_sha256']==refmanifest['runs'][f'P1_seed{seed}']['batch_order_sha256'],'Historical P1 ordering changed')
            for policy,(state,projection_state) in states.items():
                model=backbone(mapping);model.load_state_dict(state)
                projector=head(seed) if arm=='C1' else None
                if projector is not None:projector.load_state_dict(projection_state)
                payload={'state_dict':state,'projection_state_dict':projection_state,'arm':arm,'seed':seed,'policy':policy,
                         'epoch':0 if policy=='initial' else info['epochs'][policy],'mapping':mapping,
                         'training_user_keys':info['training_user_keys'],'threshold':.5,'pos_weight':540/131,
                         'lambda':config['lambda'],'temperature':config['temperature'],
                         'embedding_manifest_sha256':config['embedding_manifest_sha256'],
                         'reference_manifest_sha256':config['reference_manifest_sha256']}
                rel=f'models/{name}_{policy}.pt';torch.save(payload,output/rel)
                if policy!='initial':
                    info['checkpoints'][policy]=rel
                    f,scores,_=scored(model,data,counts,config,info,policy);rows.extend(scores)
                    f.to_csv(output/f'predictions/{name}_{policy}_validation.csv',index=False)
                    if arm=='C0':
                        previous=registered_torch(root,reference,refmanifest,f'models/P1_seed{seed}_{policy}.pt',reader)
                        require(state_hash(state)==state_hash(previous['state_dict']),'C0 did not exactly reproduce P1 weights')
                diags.extend(diagnostics(output,name,policy,model,projector,data,seed))
            runs[name]=info
        a,b=[runs[f'{arm}_seed{seed}'] for arm in ['C0','C1']]
        for key in ['initial_state_sha256','batch_order_sha256','epoch_order_sha256','primary_rng_sha256','training_user_keys','pos_weight']:
            require(a[key]==b[key],'Pairing changed: '+key)
    table=pd.DataFrame(rows);summary,effects=summaries(table)
    for name,f in [('validation_metrics',table),('seed_summary',summary),('paired_effects',effects),('representation_diagnostics',pd.DataFrame(diags)),('training_runtime',pd.DataFrame(runtimes))]:f.to_csv(output/(name+'.csv'),index=False)
    figures(output,table,histories)
    stop.set();thread.join();peak[0]=max(peak[0],proc.memory_info().rss)
    write_json(output/'runtime.json',{'training_seconds':sum(r['training_seconds'] for r in runtimes),'training_export_diagnostics_seconds':time.perf_counter()-start,
        'peak_sampled_process_rss_bytes':peak[0],'native_peak_working_set_bytes':getattr(proc.memory_info(),'peak_wset',None),'sampling_interval_seconds':.05})
    write_json(output/'development_decision.json',decision(table));write_json(output/'training_metadata.json',runs)
    write_json(output/'provenance/parsed_inputs.json',reader.accessed);write_json(output/'provenance/protected_files.json',before)
    for file in ['src/contrastive.py','src/contrastive_core.py','src/token_pooling.py','src/text_aggregation.py','src/temporal_statistics.py','src/checkpoint_weight_experiment.py','src/performance_diagnosis.py','src/baselines.py','scripts/run_contrastive.py']:
        (output/'provenance'/Path(file).name).write_bytes((root/file).read_bytes())
    write_json(output/'provenance/environment.json',{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,'pandas':pd.__version__,
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'cpu_threads':1,'test_access':False,'language_model_loaded':False,
        'diagnostic_covariance':'centered, population denominator N; effective rank=exp(entropy of normalized covariance eigenvalues)',
        'primary_rng':'historical stream preserved; head/auxiliary fork_rng; diagnostics after training on copies',
        'loss_history':'historical mean of batch means; validation selection uses unweighted user mean; ratio reports batch means and ratio of means'})
    require(before==preserved_hashes(root,config['output_root']),'Protected files changed')
    require(all(digest(root/p)==sha for p,sha in reader.accessed.items()),'Consumed input changed')
    write_json(output/'manifest.json',{'schema_version':'contrastive_v1','complete':True,'promoted_model':None,'runs':runs,
        'artifacts':{p.relative_to(output).as_posix():{'sha256':digest(p)} for p in sorted(output.rglob('*')) if p.is_file()}})
    return output


def verify(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    manifest=json.loads((output/'manifest.json').read_text())
    require(manifest['complete'] and manifest['promoted_model'] is None,'Invalid experiment')
    for rel,meta in manifest['artifacts'].items():require(digest(output/rel)==meta['sha256'],'Artifact changed: '+rel)
    config=json.loads((output/'provenance/resolved_config.json').read_text());reader=Inputs(root)
    data,mapping,counts,ref,refmanifest=load(root,config,reader)
    rows=[];diags=[];regression=[]
    for seed in config['seeds']:
        loader=make_loader(data['train'],config['training'],seed,'train')
        regression.append(equivalence(mapping,[(x,y) for x,y,_ in loader],seed,540/131))
        loader=make_loader(data['train'],config['training'],seed,'train');schedule=[];order=hashlib.sha256();epochs=[]
        for epoch in range(1,51):
            seen=[]
            for _,y,idx in loader:
                seen.extend(idx.tolist());order.update(idx.numpy().tobytes())
                bots=int(y.sum());schedule.append((epoch,len(y),bots,hashlib.sha256(idx.numpy().tobytes()).hexdigest()))
            require(sorted(seen)==list(range(671)),'Training membership mismatch')
            epochs.append(hashlib.sha256(np.asarray(seen,dtype=np.int64).tobytes()).hexdigest())
        pair=[]
        for arm in config['arms']:
            name=f'{arm}_seed{seed}';info=manifest['runs'][name];pair.append(info)
            require(info['batch_order_sha256']==order.hexdigest() and info['epoch_order_sha256']==epochs,'Batch schedule changed')
            h=pd.read_csv(output/f'histories/{name}.csv');b=pd.read_csv(output/f'histories/{name}_batches.csv')
            require(h.epoch.tolist()==list(range(1,51)) and len(b)==1050,'Incomplete training')
            require(list(zip(b.epoch,b.n,b.bots,b.batch_user_indices_sha256))==schedule,'Batch audit changed')
            require(np.array_equal(b.single_class,b.bots.eq(0)|b.bots.eq(b.n)), 'Single-class audit differs')
            require(np.array_equal(b.contrastive_skipped,b.single_class if arm=='C1' else np.zeros(len(b),dtype=bool)),'Skip policy differs')
            require((b.loc[b.single_class,'contrastive_loss']==0).all(),'Single-class contrastive loss nonzero')
            require(np.array_equal(b.human_positive_views,np.where(b.bots<b.n,2*(b.n-b.bots)-1,0))
                    and np.array_equal(b.bot_positive_views,np.where(b.bots>0,2*b.bots-1,0))
                    and np.array_equal(b.human_negative_views,2*b.bots) and np.array_equal(b.bot_negative_views,2*(b.n-b.bots)), 'Pair count audit differs')
            np.testing.assert_allclose(b.weighted_contrastive,.05*b.contrastive_loss,atol=1e-12,rtol=1e-10)
            np.testing.assert_allclose(b.contrastive_classification_ratio,b.weighted_contrastive/b.classification_loss,atol=1e-10,rtol=1e-8)
            np.testing.assert_allclose(b.joint_loss,b.classification_loss+b.weighted_contrastive,atol=1e-7,rtol=1e-7)
            require(info['training_user_keys']==data['train']['users'].index.tolist(),'Training identity mismatch')
            seed_all(seed,1);initial=backbone(mapping)
            require(state_hash(initial.state_dict())==info['initial_state_sha256'],'Initial backbone changed')
            if arm=='C0':
                old=reader.registered(ref,refmanifest,f'histories/P1_seed{seed}.csv')
                pd.testing.assert_frame_equal(h[old.columns],old,check_dtype=False,atol=1e-7,rtol=1e-7)
            for policy in ['initial']+POLICIES:
                payload=torch.load(output/f'models/{name}_{policy}.pt',map_location='cpu',weights_only=True)
                expected_epoch=0 if policy=='initial' else (50 if policy=='last' else int(h.loc[h.validation_unweighted_loss.idxmin(),'epoch']))
                require(payload['epoch']==expected_epoch and payload['arm']==arm and payload['seed']==seed and payload['policy']==policy
                        and payload['mapping']==mapping and payload['training_user_keys']==info['training_user_keys']
                        and payload['threshold']==.5 and payload['pos_weight']==540/131 and payload['lambda']==.05 and payload['temperature']==.1
                        and payload['embedding_manifest_sha256']==config['embedding_manifest_sha256']
                        and payload['reference_manifest_sha256']==config['reference_manifest_sha256'],'Checkpoint provenance changed')
                model=backbone(mapping);model.load_state_dict(payload['state_dict'],strict=True)
                projector=head(seed) if arm=='C1' else None
                if projector is not None:projector.load_state_dict(payload['projection_state_dict'],strict=True)
                else:require(payload['projection_state_dict'] is None,'Control contains projection head')
                require(sum(p.numel() for p in model.parameters())+(sum(p.numel() for p in projector.parameters()) if projector is not None else 0)==config['expected_parameters'][arm],'Capacity changed')
                if policy=='initial':require(state_hash(model.state_dict())==info['initial_state_sha256'],'Initial snapshot changed')
                else:
                    before=state_hash(model.state_dict())
                    f,scores,losses=scored(model,data,counts,config,info,policy);rows.extend(scores)
                    require(before==state_hash(model.state_dict()),'Evaluation updated model')
                    pd.testing.assert_frame_equal(f,pd.read_csv(output/f'predictions/{name}_{policy}_validation.csv',dtype={'user_key':str}),check_dtype=False,atol=1e-7,rtol=1e-6)
                    np.testing.assert_allclose(losses,h.set_index('epoch').loc[expected_epoch,['validation_weighted_loss','validation_unweighted_loss']].to_numpy(dtype=float),atol=1e-7,rtol=1e-7)
                    if arm=='C0':
                        previous=registered_torch(root,ref,refmanifest,f'models/P1_seed{seed}_{policy}.pt',reader)
                        require(state_hash(model.state_dict())==state_hash(previous['state_dict']),'C0 not exact historical P1')
                diags.extend(diagnostics(output,name,policy,model,projector,data,seed,write=False))
        for key in ['initial_state_sha256','batch_order_sha256','epoch_order_sha256','primary_rng_sha256','training_user_keys','pos_weight']:
            require(pair[0][key]==pair[1][key],'Pairing mismatch')
        bc=[pd.read_csv(output/f'histories/{arm}_seed{seed}_batches.csv') for arm in ['C0','C1']]
        for col in ['primary_rng_before','primary_rng_after','batch_user_indices_sha256']:
            require(bc[0][col].equals(bc[1][col]),'Primary stochastic behavior differs')
    require(regression==json.loads((output/'lambda_zero_regression.json').read_text()),'Lambda-zero regression differs')
    table=pd.read_csv(output/'validation_metrics.csv');keys=['arm','seed','policy','cohort']
    pd.testing.assert_frame_equal(pd.DataFrame(rows).sort_values(keys).reset_index(drop=True),table.sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-7,rtol=1e-6)
    for name,f in zip(['seed_summary','paired_effects'],summaries(table)):
        pd.testing.assert_frame_equal(f,pd.read_csv(output/(name+'.csv')),check_dtype=False,atol=1e-10,rtol=1e-8)
    pd.testing.assert_frame_equal(pd.DataFrame(diags),pd.read_csv(output/'representation_diagnostics.csv'),check_dtype=False,atol=1e-9,rtol=1e-7)
    saved_decision=json.loads((output/'development_decision.json').read_text());calculated=decision(table)
    for key in ['mean_delta_f1','mean_delta_ap']:np.testing.assert_allclose(saved_decision.pop(key),calculated.pop(key),atol=1e-12,rtol=1e-10)
    require(saved_decision==calculated,'Decision changed')
    parsed=json.loads((output/'provenance/parsed_inputs.json').read_text())
    require(not any('test' in Path(p).stem.lower() or '/graph/' in p for p in parsed),'Forbidden input')
    require(all(digest(root/p)==sha for p,sha in parsed.items()),'Input artifact changed')
    return {'training_runs':6,'selected_final_checkpoints_reloaded':12,'initial_snapshots_reloaded':6,
            'keyed_predictions_reproduced':12,'representation_arrays_reproduced':len(diags),
            'lambda_zero_regression':regression,'historical_P1_C0_states_exact':True,
            'primary_rng_and_batches_paired':True,'test_access':False,
            'consistent_development_improvement':bool(decision(table)['criterion_metrics_pass']), 'promoted_model':None}
