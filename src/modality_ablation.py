"""Fixed development branch ablations; no test access or model promotion."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.baselines import seed_all
from src.contrastive import load as reference_inputs, registered_torch
from src.contrastive_core import backbone
from src.temporal_statistics import make_loader
from src.checkpoint_weight_experiment import state_hash
from src.performance_diagnosis import Inputs, digest, require, write_json, plt, save_figure
from src.token_pooling import scored, preserved_hashes

ARMS = {'M0':('profile',), 'M1':('text',), 'M2':('profile','temporal'),
        'M3':('profile','text'), 'M4':('profile','temporal','text')}
COUNTS = {'M0':3329,'M1':12881,'M2':4081,'M3':16145,'M4':16897}
DIMS = {'profile':48,'temporal':16,'text':16}
PAIRS = [('M2','M0'),('M3','M0'),('M4','M2'),('M4','M3'),('M4','M0')]
POLICIES = ['best_unweighted_validation_loss','last']
METRICS = ['bot_f1','bot_precision','bot_recall','roc_auc','average_precision','brier','validation_unweighted_loss']


class Reduced(nn.Module):
    def __init__(self, template, arm, seed):
        super().__init__()
        self.arm=arm
        self.register_buffer('other_indices',template.base.other_indices.clone())
        self.register_buffer('temporal_indices',template.base.temporal_indices.clone())
        if 'profile' in ARMS[arm]:self.profile=deepcopy(template.base.other_encoder)
        if 'temporal' in ARMS[arm]:self.temporal=deepcopy(template.base.temporal_encoder)
        if 'text' in ARMS[arm]:self.text=deepcopy(template.text_projection)
        # Same classifier-specific seed across arms; real fan-in determines scale.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(300000+seed)
            first=nn.Linear(sum(DIMS[b] for b in ARMS[arm]),32)
        self.classifier=nn.Sequential(first,nn.ReLU(),nn.Dropout(.3),deepcopy(template.base.classifier[3]))

    def forward(self,x):
        parts={}
        # Preserve historical relative evaluation order for retained branches.
        if 'text' in ARMS[self.arm]:
            mask=x[:,815:816]
            parts['text']=self.text(torch.where(mask.bool(),x[:,47:815],torch.zeros_like(x[:,47:815])))*mask
        if 'profile' in ARMS[self.arm]:parts['profile']=self.profile(x[:,self.other_indices])
        if 'temporal' in ARMS[self.arm]:
            mask=x[:,40:47]
            values=torch.where(mask.bool(),x[:,self.temporal_indices],torch.zeros_like(mask))
            parts['temporal']=self.temporal(torch.cat([values,mask],1))*mask.any(1,keepdim=True)
        return self.classifier(torch.cat([parts[b] for b in ARMS[self.arm]],1))


def build(mapping,arm,seed):
    seed_all(seed,1)
    template=backbone(mapping)
    rng=torch.get_rng_state().clone()
    model=template if arm=='M4' else Reduced(template,arm,seed)
    require(torch.equal(rng,torch.get_rng_state()),'Reduced construction consumed primary RNG')
    require(sum(p.numel() for p in model.parameters())==COUNTS[arm],'Capacity mismatch')
    return model


def initialization(model,arm,seed):
    encoders=({'profile':model.base.other_encoder,'temporal':model.base.temporal_encoder,'text':model.text_projection}
              if arm=='M4' else {b:getattr(model,b) for b in ARMS[arm]})
    classifier=model.base.classifier if arm=='M4' else model.classifier
    return {'encoders':{b:state_hash(m.state_dict()) for b,m in encoders.items()},
            'first_classifier':state_hash(classifier[0].state_dict()),
            'final_classifier':state_hash(classifier[3].state_dict()),
            'first_fan_in':classifier[0].in_features,
            'classifier_rng_seed':None if arm=='M4' else 300000+seed,
            'full_state':state_hash(model.state_dict())}


def invariance(model,arm,pop,mapping):
    model.eval();before=state_hash(model.state_dict())
    x=torch.tensor(pop['x'][:32],dtype=torch.float32)
    corrupt=x.clone()
    if 'profile' not in ARMS[arm]:corrupt[:,mapping['other_indices']]=float('nan')
    if 'temporal' not in ARMS[arm]:
        corrupt[:,mapping['temporal_indices']]=float('nan');corrupt[:,40:47]=float('nan')
    if 'text' not in ARMS[arm]:corrupt[:,47:816]=float('nan')
    with torch.no_grad():
        torch.testing.assert_close(model(x),model(corrupt),atol=0,rtol=0)
        masked=x.clone();masked[:,40:47]=0;masked[:,815]=0
        absent=masked.clone();absent[:,mapping['temporal_indices']]=float('nan');absent[:,47:815]=float('nan')
        torch.testing.assert_close(model(masked),model(absent),atol=0,rtol=0)
        require(torch.isfinite(model(absent)).all().item(),'Missingness produced nonfinite predictions')
    require(before==state_hash(model.state_dict()),'Diagnostic updated state')


def train(data,mapping,config,arm,seed):
    model=build(mapping,arm,seed);initial=deepcopy(model.state_dict());init=initialization(model,arm,seed)
    spec=config['training'];optimizer=torch.optim.AdamW(model.parameters(),lr=spec['lr'],weight_decay=spec['weight_decay'])
    weight=float((data['train']['y']==0).sum()/(data['train']['y']==1).sum())
    require(weight==540/131,'Training weight changed')
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight],dtype=torch.float32))
    loaders={s:make_loader(p,spec,seed,s) for s,p in data.items()}
    history=[];order=hashlib.sha256();epochs=[];best=float('inf');best_epoch=None;best_state=None
    for epoch in range(1,51):
        row={'epoch':epoch};seen=[]
        for split,loader in loaders.items():
            model.train(split=='train');before=state_hash(model.state_dict()) if split=='validation' else None
            total=0.;unweighted=0.;n=0
            with torch.set_grad_enabled(split=='train'):
                for x,y,idx in loader:
                    if split=='train':
                        seen.extend(idx.tolist());order.update(idx.numpy().tobytes());optimizer.zero_grad()
                    logits=model(x);loss=criterion(logits,y)
                    if split=='train':loss.backward();optimizer.step()
                    else:unweighted+=nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='sum').item()
                    total+=loss.item();n+=len(x)
            row[split+'_weighted_loss']=total/len(loader)
            if split=='validation':
                row['validation_unweighted_loss']=unweighted/n
                require(before==state_hash(model.state_dict()),'Validation changed model')
        require(sorted(seen)==list(range(671)),'Training population changed')
        epochs.append(hashlib.sha256(np.asarray(seen,dtype=np.int64).tobytes()).hexdigest())
        require(np.isfinite(list(row.values())).all(),'Nonfinite loss');history.append(row)
        if row['validation_unweighted_loss']<best:
            best=row['validation_unweighted_loss'];best_epoch=epoch;best_state=deepcopy(model.state_dict())
    print(f'{arm} seed {seed}: selected epoch {best_epoch}, BCE {best:.6f}',flush=True)
    return {'initial':initial,POLICIES[0]:best_state,'last':deepcopy(model.state_dict())},pd.DataFrame(history),{
        'arm':arm,'seed':seed,'parameter_count':COUNTS[arm],'pos_weight':weight,'initialization':init,
        'epochs':{POLICIES[0]:best_epoch,'last':50},'batch_order_sha256':order.hexdigest(),'epoch_order_sha256':epochs,
        'training_user_keys':data['train']['users'].index.tolist()}


def summaries(table):
    allrows=table[table.cohort.eq('all')]
    summary=allrows.groupby(['arm','policy'])[METRICS].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    effects=[];decisions=[]
    for a,b in PAIRS:
        for policy in POLICIES:
            paired=[]
            for seed in [42,43,44]:
                r=allrows[allrows.seed.eq(seed)&allrows.policy.eq(policy)].set_index('arm')
                item={'comparison':a+'-'+b,'seed':seed,'policy':policy,'parameter_delta':COUNTS[a]-COUNTS[b],
                      **{'delta_'+m:float(r.loc[a,m]-r.loc[b,m]) for m in METRICS}}
                effects.append(item);paired.append(item)
            if policy==POLICIES[0]:
                decisions.append({'comparison':a+'-'+b,'all_three_f1_improve':all(r['delta_bot_f1']>0 for r in paired),
                    'mean_delta_f1':float(np.mean([r['delta_bot_f1'] for r in paired])),
                    'mean_delta_ap':float(np.mean([r['delta_average_precision'] for r in paired])),
                    'metrics_pass':bool(all(r['delta_bot_f1']>0 for r in paired) and np.mean([r['delta_average_precision'] for r in paired])>=0),
                    'requires_verification':True,'automatic_promotion':False})
    return summary.reset_index(),pd.DataFrame(effects),decisions


def load(root,config,reader):
    require(config['arms']==list(ARMS) and config['seeds']==[42,43,44] and config['expected_parameters']==COUNTS
            and config['training']=={'epochs':50,'batch_size':32,'lr':.001,'weight_decay':.0001}
            and config['threshold']==.5 and config['selection']==POLICIES[0]
            and config['classifier_seed_rule']=='300000 + seed','Protocol changed')
    p=root/config['reference_config'];reader.allow(p,config['reference_config_sha256']);c=reader.read(p)
    return reference_inputs(root,c,reader)


def plots(output,table,histories):
    for policy,short in [(POLICIES[0],'selected'),('last','final')]:
        fig,axes=plt.subplots(5,3,figsize=(11,16))
        for i,arm in enumerate(ARMS):
            for j,seed in enumerate([42,43,44]):
                r=table[table.arm.eq(arm)&table.seed.eq(seed)&table.policy.eq(policy)&table.cohort.eq('all')].iloc[0]
                cm=np.array([[r.tn,r.fp],[r.fn,r.tp]],dtype=int);ax=axes[i,j];ax.imshow(cm,cmap='Blues',vmin=0,vmax=116)
                for a in range(2):
                    for b in range(2):ax.text(b,a,str(cm[a,b]),ha='center',va='center',color='white' if cm[a,b]>58 else 'black')
                ax.set(title=f'{arm}, seed {seed}, epoch {int(r.epoch)}',xticks=[0,1],yticks=[0,1],xticklabels=['Human','Bot'],yticklabels=['Human','Bot'],xlabel='Predicted',ylabel='True')
        save_figure(fig,output/f'figures/confusion_{short}.png')
    fig,axes=plt.subplots(5,3,figsize=(13,16))
    for i,arm in enumerate(ARMS):
        for j,seed in enumerate([42,43,44]):
            h=histories[f'{arm}_seed{seed}'];ax=axes[i,j]
            ax.plot(h.epoch,h.train_weighted_loss,label='Train weighted BCE');ax.plot(h.epoch,h.validation_unweighted_loss,label='Validation BCE')
            ax.axvline(int(h.loc[h.validation_unweighted_loss.idxmin(),'epoch']),ls=':',color='black')
            ax.set(title=f'{arm}, seed {seed}',xlabel='Epoch',ylabel='Loss');ax.legend(fontsize=7)
    save_figure(fig,output/'figures/learning_curves.png')


def run(root,config_path='configs/modality_ablation_v1.json'):
    root=Path(root).resolve();reader=Inputs(root);reader.allow(root/config_path,digest(root/config_path));config=reader.read(root/config_path)
    before=preserved_hashes(root,config['output_root']);data,mapping,counts,ref,rm=load(root,config,reader)
    output=root/config['output_root']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    for folder in ['models','histories','predictions','provenance','figures']:(output/folder).mkdir(parents=True,exist_ok=False)
    print('OUTPUT_VERSION:',output,flush=True)
    write_json(output/'provenance/resolved_config.json',config);write_json(output/'provenance/feature_mapping.json',mapping)
    for split,pop in data.items():pop['users'][['label_binary']].join(pop['masks']).to_csv(output/f'provenance/{split}_population.csv')
    rows=[];runs={};histories={};timings=[]
    # Gate acceptance on historical reproduction; run M4 first for each seed.
    for seed in config['seeds']:
        for arm in ['M4','M0','M1','M2','M3']:
            start=time.perf_counter();states,h,info=train(data,mapping,config,arm,seed)
            timings.append({'arm':arm,'seed':seed,'seconds':time.perf_counter()-start})
            name=f'{arm}_seed{seed}';runs[name]=info;histories[name]=h;h.to_csv(output/f'histories/{name}.csv',index=False)
            if arm=='M4':
                old=reader.registered(ref,rm,f'histories/P1_seed{seed}.csv')
                pd.testing.assert_frame_equal(h,old,check_dtype=False,atol=1e-7,rtol=1e-7)
            for policy,state in states.items():
                model=build(mapping,arm,seed);model.load_state_dict(state,strict=True);invariance(model,arm,data['train'],mapping)
                payload={'state_dict':state,'arm':arm,'seed':seed,'policy':policy,'mapping':mapping,
                    'epoch':0 if policy=='initial' else info['epochs'][policy],'config':config,'initialization':info['initialization']}
                torch.save(payload,output/f'models/{name}_{policy}.pt')
                if policy=='initial':continue
                f,scores,_=scored(model,data,counts,config,info,policy);rows.extend(scores)
                f['included_branches']='+'.join(ARMS[arm]);f.to_csv(output/f'predictions/{name}_{policy}_validation.csv',index=False)
                if arm=='M4':
                    previous=registered_torch(root,ref,rm,f'models/P1_seed{seed}_{policy}.pt',reader)
                    require(state_hash(state)==state_hash(previous['state_dict']),'M4 is not exact P1')
    table=pd.DataFrame(rows);summary,effects,decisions=summaries(table)
    for name,f in [('validation_metrics',table),('seed_summary',summary),('paired_effects',effects),('runtime',pd.DataFrame(timings))]:f.to_csv(output/(name+'.csv'),index=False)
    write_json(output/'development_decisions.json',decisions);plots(output,table,histories)
    write_json(output/'provenance/parsed_inputs.json',reader.accessed);write_json(output/'provenance/protected_files.json',before)
    for file in ['src/modality_ablation.py','scripts/run_modality_ablation.py','src/text_aggregation.py','src/temporal_statistics.py','src/token_pooling.py','src/contrastive.py','src/contrastive_core.py','src/checkpoint_weight_experiment.py','src/performance_diagnosis.py','src/baselines.py']:
        (output/'provenance'/Path(file).name).write_bytes((root/file).read_bytes())
    write_json(output/'provenance/environment.json',{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,'pandas':pd.__version__,'device':'cpu','threads':1,'test_access':False})
    require(before==preserved_hashes(root,config['output_root']),'Protected files changed')
    write_json(output/'manifest.json',{'complete':True,'schema_version':'modality_ablation_v1','runs':runs,'promoted_model':None,
        'artifacts':{p.relative_to(output).as_posix():{'sha256':digest(p)} for p in sorted(output.rglob('*')) if p.is_file()}})
    return output


def verify(root,output):
    root=Path(root).resolve();output=Path(output).resolve();manifest=json.loads((output/'manifest.json').read_text())
    for rel,meta in manifest['artifacts'].items():require(digest(output/rel)==meta['sha256'],'Artifact changed: '+rel)
    config=json.loads((output/'provenance/resolved_config.json').read_text());reader=Inputs(root)
    data,mapping,counts,ref,rm=load(root,config,reader);rows=[]
    for seed in [42,43,44]:
        template=build(mapping,'M4',seed);template_init=initialization(template,'M4',seed)
        loader=make_loader(data['train'],config['training'],seed,'train');order=hashlib.sha256();epochs=[]
        for epoch in range(50):
            seen=[]
            for _,_,idx in loader:seen.extend(idx.tolist());order.update(idx.numpy().tobytes())
            require(sorted(seen)==list(range(671)),'Invalid schedule');epochs.append(hashlib.sha256(np.asarray(seen,dtype=np.int64).tobytes()).hexdigest())
        for arm in ARMS:
            name=f'{arm}_seed{seed}';info=manifest['runs'][name];model=build(mapping,arm,seed)
            require(initialization(model,arm,seed)==info['initialization'],'Initialization changed')
            for branch,sha in info['initialization']['encoders'].items():require(sha==template_init['encoders'][branch],'Encoder pairing differs')
            require(info['initialization']['final_classifier']==template_init['final_classifier'],'Final classifier not paired')
            require(info['batch_order_sha256']==order.hexdigest() and info['epoch_order_sha256']==epochs,'Batch schedule differs')
            require(info['training_user_keys']==data['train']['users'].index.tolist(),'Training users differ')
            h=pd.read_csv(output/f'histories/{name}.csv');require(h.epoch.tolist()==list(range(1,51)),'Incomplete history')
            require(info['epochs'][POLICIES[0]]==int(h.loc[h.validation_unweighted_loss.idxmin(),'epoch']),'Selection changed')
            if arm=='M4':
                old=reader.registered(ref,rm,f'histories/P1_seed{seed}.csv');pd.testing.assert_frame_equal(h,old,check_dtype=False,atol=1e-7,rtol=1e-7)
                require(info['batch_order_sha256']==rm['runs'][f'P1_seed{seed}']['batch_order_sha256'],'Historical schedule changed')
            for policy in ['initial']+POLICIES:
                payload=torch.load(output/f'models/{name}_{policy}.pt',map_location='cpu',weights_only=True)
                require(payload['config']==config and payload['mapping']==mapping and payload['initialization']==info['initialization']
                    and payload['arm']==arm and payload['seed']==seed and payload['policy']==policy
                    and payload['epoch']==(0 if policy=='initial' else info['epochs'][policy]),'Checkpoint metadata differs')
                model=build(mapping,arm,seed);model.load_state_dict(payload['state_dict'],strict=True)
                invariance(model,arm,data['train'],mapping)
                if policy=='initial':
                    require(state_hash(model.state_dict())==info['initialization']['full_state'],'Initial snapshot differs');continue
                before=state_hash(model.state_dict());f,scores,losses=scored(model,data,counts,config,info,policy);rows.extend(scores)
                require(before==state_hash(model.state_dict()),'Evaluation updated state')
                f['included_branches']='+'.join(ARMS[arm])
                pd.testing.assert_frame_equal(f,pd.read_csv(output/f'predictions/{name}_{policy}_validation.csv',dtype={'user_key':str}),check_dtype=False,atol=1e-7,rtol=1e-6)
                np.testing.assert_allclose(losses,h.set_index('epoch').loc[info['epochs'][policy],['validation_weighted_loss','validation_unweighted_loss']].to_numpy(dtype=float),atol=1e-7,rtol=1e-7)
                if arm=='M4':
                    previous=registered_torch(root,ref,rm,f'models/P1_seed{seed}_{policy}.pt',reader)
                    require(state_hash(model.state_dict())==state_hash(previous['state_dict']),'M4 reproduction failed')
    table=pd.read_csv(output/'validation_metrics.csv');keys=['arm','seed','policy','cohort']
    pd.testing.assert_frame_equal(pd.DataFrame(rows).sort_values(keys).reset_index(drop=True),table.sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-7,rtol=1e-6)
    summary,effects,decisions=summaries(table)
    for name,f in [('seed_summary',summary),('paired_effects',effects)]:pd.testing.assert_frame_equal(f,pd.read_csv(output/(name+'.csv')),check_dtype=False,atol=1e-10,rtol=1e-8)
    saved=json.loads((output/'development_decisions.json').read_text())
    for a,b in zip(saved,decisions):
        for key in ['mean_delta_f1','mean_delta_ap']:np.testing.assert_allclose(a.pop(key),b.pop(key),atol=1e-12,rtol=1e-10)
    require(saved==decisions,'Decision changed')
    parsed=json.loads((output/'provenance/parsed_inputs.json').read_text())
    require(not any('test' in Path(p).stem.lower() or '/graph/' in p for p in parsed),'Forbidden input')
    require(all(digest(root/p)==sha for p,sha in parsed.items()),'Input changed')
    return {'runs':15,'selected_final_checkpoints_reloaded':30,'initial_snapshots':15,'keyed_predictions_reproduced':30,
        'exact_M4_P1':True,'paired_encoders_and_final_layer':True,'excluded_and_missing_input_invariance':True,
        'artifact_hashes':len(manifest['artifacts']),'test_access':False,'promoted_model':None}
