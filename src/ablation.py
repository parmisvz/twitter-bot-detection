"""Cohort-controlled branch removal; original pairwise architectures, no refitting."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import importlib.metadata
import platform
import subprocess
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.baselines import (SPLITS, align, artifact_inventory, checked_artifact, evaluate_registry,
    keyed, notebook_source_hash, read_json, require, seed_all, sha256, write_json)
from src import multimodal as mm

VARIANTS = {'tabular_text':('tabular','text'), 'text_graph':('text','graph'), 'tabular_graph':('tabular','graph')}
DIMS = {'tabular':64,'text':128,'graph':32}


class AblationClassifier(nn.Module):
    """Only retained branches exist. Three-argument adapter does not use removed input."""
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        branches = {'tabular':mm.TabularEncoder,'text':mm.TextEncoder,'graph':mm.GraphEncoder}
        for name in VARIANTS[variant]:
            setattr(self,name+'_encoder',branches[name]())
        self.classifier = nn.Sequential(nn.Linear(sum(DIMS[n] for n in VARIANTS[variant]),64),
            nn.ReLU(),nn.Dropout(.3),nn.Linear(64,2))
    def forward(self, tabular, text, graph):
        inputs = dict(tabular=tabular,text=text,graph=graph)
        return self.classifier(torch.cat([getattr(self,n+'_encoder')(inputs[n]) for n in VARIANTS[self.variant]],dim=1))


def load_inputs(root, config):
    require(config['variants']==list(VARIANTS),'Unapproved variant policy')
    require(config['comparison_type']=='cohort_controlled_branch_removal_not_capacity_controlled','Unapproved comparison policy')
    inputs = mm.load_inputs(root,config)
    phase2,samples,_,_,_,_ = inputs
    path = Path(root)/config['reference_manifest']
    require(sha256(path)==config['reference_manifest_sha256'],'Reference manifest changed')
    reference = read_json(path)
    require(reference['complete'] and reference['schema_version']=='multimodal_v2','Invalid full reference')
    rc = read_json(checked_artifact(path.parent,reference,'provenance/resolved_config.json'))
    for key in ('training','seed','device','torch_threads','expected_counts','encoder','protocol'):
        require(config[key]==rc[key],f'Reference setting differs: {key}')
    require(reference['feature_order']==phase2['feature_order'],'Reference feature order differs')
    for key,value in reference['input_manifest_hashes'].items():
        require(config[key]==value,'Reference input version differs')
    require(reference['training']['training_user_keys']==samples['train']['keys'].tolist(),'Reference training keys differ')
    counts = torch.bincount(samples['train']['dataset'].tensors[3],minlength=2)
    require(reference['training']['class_counts']==counts.tolist() and
            reference['training']['class_weights']==(counts.sum()/(2*counts)).float().tolist(),'Reference class weights differ')
    primary = {}
    model,payload = mm.load_checkpoint(checked_artifact(path.parent,reference,
        reference['training']['checkpoints'][reference['training']['selection']]))
    require(payload['feature_order']==phase2['feature_order'] and payload['input_manifest_hashes']==reference['input_manifest_hashes'],'Reference checkpoint contract differs')
    for split in SPLITS:
        keys = pd.read_csv(checked_artifact(path.parent,reference,f'provenance/{split}_eligible_keys.csv'),dtype={'user_key':str}).user_key.tolist()
        require(keys==samples[split]['keys'].tolist(),'Reference cohort/order differs')
        frame = pd.read_csv(checked_artifact(path.parent,reference,f"predictions/multimodal_{reference['training']['selection']}_{split}.csv"),dtype={'user_key':str,'prediction':'Int64'})
        rebuilt = mm.predictions(model,phase2['populations'][split],samples[split],split,config)
        pd.testing.assert_frame_equal(frame[rebuilt.columns],rebuilt,check_dtype=False,atol=1e-7,rtol=0)
        primary[split] = frame
    return inputs,reference,primary


def train_model(train, validation, config, variant):
    seed_all(config['seed'],config['torch_threads'])
    model = AblationClassifier(variant)
    counts = torch.bincount(train['dataset'].tensors[3],minlength=2)
    require((counts>0).all().item(),'Both training classes required')
    weights = (counts.sum()/(2*counts)).float()
    spec = config['training']
    optimizer = torch.optim.AdamW(model.parameters(),lr=spec['lr'],weight_decay=spec['weight_decay'])
    batches = mm.loader(train,config,True)
    history,best,best_epoch,best_loss = [],None,None,float('inf')
    for epoch in range(1,spec['epochs']+1):
        model.train()
        losses=[]
        for b in batches:
            optimizer.zero_grad()
            loss=nn.functional.cross_entropy(model(*b[:3]),b[3],weight=weights)
            require(torch.isfinite(loss).item(),'Nonfinite training loss')
            loss.backward(); optimizer.step(); losses.append(loss.item())
        vl=mm.validation_loss(model,validation,config,weights)
        require(np.isfinite(vl),'Nonfinite validation loss')
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_loss=vl))
        if vl<best_loss:
            best,best_epoch,best_loss=deepcopy(model.state_dict()),epoch,vl
        if epoch%20==0: print(f'{variant} epoch {epoch}: validation={vl:.6f}',flush=True)
    return {'best_validation_loss':best,'last':deepcopy(model.state_dict())},pd.DataFrame(history),dict(
        best_epoch=best_epoch,last_epoch=spec['epochs'],selection=spec['selection'],class_counts=counts.tolist(),
        class_weights=weights.tolist(),training_user_keys=train['keys'].tolist(),
        parameter_count=sum(p.numel() for p in model.parameters()))


def predictions(model,pop,sample,split,config):
    frame=mm.predictions(model,pop,sample,split,config)
    required=np.ones(len(frame),dtype=bool)
    if 'text' in VARIANTS[model.variant]: required &= frame.has_tweet_text.eq(1).to_numpy()
    if 'graph' in VARIANTS[model.variant]: required &= frame.has_graph.eq(1).to_numpy()
    frame['required_modalities_available']=required
    frame['in_fixed_ablation_cohort']=frame.eligible
    frame['status']=np.where(frame.eligible,'predicted',np.where(required,'outside_fixed_ablation_cohort','missing_required_modality'))
    frame['missing_required_modalities']=[','.join(n for n,mask in [('text','has_tweet_text'),('graph','has_graph')]
        if n in VARIANTS[model.variant] and row[mask]==0) or 'none' for _,row in frame.iterrows()]
    frame['model']=model.variant
    return frame


def load_checkpoint(path):
    p=torch.load(path,map_location='cpu',weights_only=True)
    m=AblationClassifier(p['model_variant']);m.load_state_dict(p['state_dict'],strict=True)
    return m.eval(),p


def compare(primary,reference,config):
    rows,keys=[],[]
    for split in ('validation','test'):
        ref=keyed(reference[split],'reference')
        cohort=sorted(ref.index[ref.eligible])
        require(len(cohort)==config['expected_counts'][split],'Comparison cohort differs')
        for name,frame in [('full_fusion',reference[split])]+[(v,primary[v][split]) for v in VARIANTS]:
            candidate=align(keyed(frame,name),ref,'canonical population')
            require(candidate.label_binary.eq(ref.label_binary).all() and candidate.eligible.eq(ref.eligible).all(),'Comparison labels/cohort differ')
            require(candidate.loc[cohort,'status'].eq('predicted').all(),'Missing cohort prediction')
            metric,_=evaluate_registry(candidate.loc[cohort].reset_index(),'all_users')
            metric['cohort']='fixed_complete_modality_cohort'
            rows.append(dict(model=name,split=split,comparison_type=config['comparison_type'],**metric))
        keys.extend(dict(split=split,user_key=k) for k in cohort)
    result=pd.DataFrame(rows)
    for metric in ('bot_f1','roc_auc','average_precision'):
        baseline=result[result.model.eq('full_fusion')].set_index('split')[metric]
        result['delta_'+metric]=result[metric]-result.split.map(baseline)
    return result,pd.DataFrame(keys)


def save_figures(comparison,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for split in ('validation','test'):
        data=comparison[comparison.split.eq(split)]
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        for ax,metric,label in zip(axes,['bot_f1','roc_auc','average_precision'],['Bot F1','ROC-AUC','Average precision']):
            ax.bar(data.model,data[metric],color=['#555555','#2878b5','#e69138','#3b9154'])
            ax.set_ylim(0,1);ax.set_ylabel(label);ax.tick_params(axis='x',rotation=35)
        fig.suptitle(f'{split.capitalize()}: fixed cohort n={int(data.evaluated_n.iloc[0])}; single seed, not capacity-controlled')
        fig.savefig(directory/f'{split}_ablation.png',dpi=180)
        fig.savefig(directory/f'{split}_ablation.pdf')
        plt.close(fig)


class AblationRun:
    def __init__(self,root,config_path='configs/ablation_v2.json'):
        self.root=Path(root).resolve();self.config=read_json(self.root/config_path)
        seed_all(self.config['seed'],self.config['torch_threads'])
        inputs,self.reference,self.reference_predictions=load_inputs(self.root,self.config)
        self.phase2,self.samples,self.embedding_manifest,self.inputs,self.coverage,self.overlap=inputs
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        self.output_dir=self.root/self.config['output_root']/(stamp+'_'+uuid4().hex[:8])
        self.output_dir.mkdir(parents=True,exist_ok=False)
        for folder in ('models','histories','predictions','reports','provenance','figures'):(self.output_dir/folder).mkdir()
        write_json(self.output_dir/'provenance/resolved_config.json',self.config)
        write_json(self.output_dir/'provenance/reference_manifest.json',self.reference)
        write_json(self.output_dir/'provenance/phase2_manifest.json',self.phase2['manifest'])
        write_json(self.output_dir/'provenance/embedding_manifest.json',self.embedding_manifest)
        for name,(_,manifest) in self.inputs.items():write_json(self.output_dir/f'provenance/{name}_manifest.json',manifest)
        self.coverage.to_csv(self.output_dir/'reports/coverage.csv',index=False)
        self.overlap.to_csv(self.output_dir/'reports/shared_support.csv',index=False)
        for split,pop in self.phase2['populations'].items():
            pop['observed'].to_csv(self.output_dir/f'provenance/{split}_observation_masks.csv')
            pd.DataFrame({'user_key':self.samples[split]['keys']}).to_csv(self.output_dir/f'provenance/{split}_cohort.csv',index=False)
        sources=['src/ablation.py','src/multimodal.py','src/baselines.py','src/graph_baseline.py','src/graph_tabular.py']
        for s in sources:(self.output_dir/'provenance'/Path(s).name).write_bytes((self.root/s).read_bytes())
        write_json(self.output_dir/'provenance/run.json',dict(python=platform.python_version(),platform=platform.platform(),
            packages={n:importlib.metadata.version(n) for n in ('torch','numpy','pandas','scikit-learn','torch-geometric','matplotlib')},
            seed=self.config['seed'],device='cpu',torch_threads=self.config['torch_threads'],deterministic_algorithms=True,
            source_sha256={s:sha256(self.root/s) for s in sources},config_sha256=sha256(self.root/config_path),
            notebook_source_sha256=notebook_source_hash(self.root/'analysis/07_ablation_study.ipynb'),
            git_commit_at_execution=subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root,text=True).strip(),
            git_status_at_execution=subprocess.check_output(['git','status','--porcelain'],cwd=self.root,text=True),
            limitations=['Fixed complete-modality cohorts, not full-population deployment.',
              'Branch removal changes parameter count. Same seed does not mean identical retained weights or dropout draws.',
              'Tabular combines profile, behavior and temporal statistics; removing it is not metadata-only ablation.',
              'Temporal placeholders and observation masks unchanged; no new imputation or temporal exclusion.',
              'Graph statistics MLP, no GNN. Gold-isolated graphs share support identities.',
              'One seed, six test bots, previously inspected test data; no significance or generalization claim.',self.embedding_manifest['limitation']]))
        self.info={};self.states={};self.histories={}

    def fit(self,variant):
        require(variant in VARIANTS and variant not in self.info,'Invalid/already fitted variant')
        states,history,info=train_model(self.samples['train'],self.samples['validation'],self.config,variant)
        self.states[variant]=states;self.histories[variant]=history;self.info[variant]=info
        history.to_csv(self.output_dir/f'histories/{variant}.csv',index=False)
        info['checkpoints']={}
        for snapshot,state in states.items():
            relative=f'models/{variant}_{snapshot}.pt'
            torch.save(dict(state_dict=state,model_variant=variant,checkpoint_variant=snapshot,
                epoch=info['best_epoch'] if snapshot=='best_validation_loss' else info['last_epoch'],
                feature_order=self.phase2['feature_order'],retained_branches=list(VARIANTS[variant]),
                input_manifest_hashes={k:v for k,v in self.config.items() if k.endswith('_sha256')}),self.output_dir/relative)
            info['checkpoints'][snapshot]=relative
        write_json(self.output_dir/f'models/{variant}_metadata.json',info)
        return history

    def evaluate_export(self):
        require(set(self.info)==set(VARIANTS) and not (self.output_dir/'manifest.json').exists(),'Incomplete/already exported run')
        rows,details,primary,reloads=[],{}, {},[]
        for variant,info in self.info.items():
            primary[variant]={}
            for snapshot,relative in info['checkpoints'].items():
                model,_=load_checkpoint(self.output_dir/relative)
                memory=AblationClassifier(variant).eval();memory.load_state_dict(self.states[variant][snapshot])
                for split,pop in self.phase2['populations'].items():
                    frame=predictions(model,pop,self.samples[split],split,self.config)
                    pd.testing.assert_frame_equal(frame,predictions(memory,pop,self.samples[split],split,self.config),check_exact=True)
                    reloads.append(dict(model=variant,checkpoint=snapshot,split=split,identical=True))
                    frame['primary_checkpoint']=snapshot==info['selection']
                    frame.to_csv(self.output_dir/f'predictions/{variant}_{snapshot}_{split}.csv',index=False)
                    if snapshot==info['selection']:primary[variant][split]=frame
                    for cohort in ('all_users','fixed_ablation_cohort'):
                        metric,detail=mm.metrics_for(frame,cohort)
                        rows.append(dict(model=variant,checkpoint_variant=snapshot,primary_checkpoint=snapshot==info['selection'],split=split,**metric))
                        details[f'{variant}/{snapshot}/{split}/{cohort}']=detail
        self.comparison,keys=compare(primary,self.reference_predictions,self.config)
        self.comparison.to_csv(self.output_dir/'reports/ablation_comparison.csv',index=False)
        keys.to_csv(self.output_dir/'reports/matched_user_keys.csv',index=False)
        counts={v:i['parameter_count'] for v,i in self.info.items()}
        counts['full_fusion']=sum(p.numel() for p in mm.MultimodalFusionClassifier().parameters())
        write_json(self.output_dir/'reports/parameter_counts.json',counts)
        save_figures(self.comparison,self.output_dir/'figures')
        metrics=pd.DataFrame(rows);metrics.to_csv(self.output_dir/'metrics.csv',index=False)
        write_json(self.output_dir/'reports/classification_reports.json',details)
        write_json(self.output_dir/'reports/checkpoint_reload.json',reloads)
        write_json(self.output_dir/'manifest.json',dict(schema_version='ablation_v2',complete=True,identity_key='user_key',
            feature_order=self.phase2['feature_order'],models=self.info,comparison_type=self.config['comparison_type'],
            input_manifest_hashes={k:v for k,v in self.config.items() if k.endswith('_sha256')},
            cohort_policy='identical 193/42/40 users across full and all removed-branch models',
            artifacts=artifact_inventory(self.output_dir)))
        return metrics


def verify_run(root,output_dir):
    directory=Path(output_dir);manifest=read_json(directory/'manifest.json')
    require(manifest['complete'],'Incomplete run')
    for rel in manifest['artifacts']:checked_artifact(directory,manifest,rel)
    config=read_json(directory/'provenance/resolved_config.json')
    seed_all(config['seed'],config['torch_threads'])
    inputs,reference,reference_predictions=load_inputs(root,config)
    phase2,samples,_,_,coverage,overlap=inputs
    require(manifest['feature_order']==phase2['feature_order'],'Feature order differs')
    hashes={k:v for k,v in config.items() if k.endswith('_sha256')}
    require(manifest['input_manifest_hashes']==hashes,'Input hashes differ')
    pd.testing.assert_frame_equal(coverage,pd.read_csv(directory/'reports/coverage.csv'),check_dtype=False)
    pd.testing.assert_frame_equal(overlap,pd.read_csv(directory/'reports/shared_support.csv'),check_dtype=False)
    for split,pop in phase2['populations'].items():
        require(pd.read_csv(directory/f'provenance/{split}_cohort.csv').user_key.tolist()==samples[split]['keys'].tolist(),'Saved cohort differs')
        pd.testing.assert_frame_equal(pd.read_csv(directory/f'provenance/{split}_observation_masks.csv').set_index('user_key'),pop['observed'],check_dtype=False)
    rows,primary=[],{}
    details=read_json(directory/'reports/classification_reports.json')
    for variant in VARIANTS:
        info=manifest['models'][variant];primary[variant]={}
        require(info==read_json(directory/f'models/{variant}_metadata.json'),'Training metadata differs')
        for k in ('training_user_keys','class_counts','class_weights','selection','last_epoch'):
            require(info[k]==reference['training'][k],f'Training contract differs: {k}')
        history=pd.read_csv(directory/f'histories/{variant}.csv')
        require(history.epoch.tolist()==list(range(1,201)),'Incomplete history')
        require(info['best_epoch']==int(history.loc[history.validation_loss.idxmin(),'epoch']),'Wrong best epoch')
        for snapshot,relative in info['checkpoints'].items():
            model,payload=load_checkpoint(directory/relative)
            epoch=info['best_epoch'] if snapshot=='best_validation_loss' else info['last_epoch']
            require(payload['epoch']==epoch and payload['model_variant']==variant and payload['checkpoint_variant']==snapshot and payload['feature_order']==phase2['feature_order'] and payload['input_manifest_hashes']==hashes,'Checkpoint contract differs')
            loss=mm.validation_loss(model,samples['validation'],config,torch.tensor(info['class_weights']))
            require(np.isclose(loss,history.loc[history.epoch.eq(epoch),'validation_loss'].iloc[0],atol=1e-7,rtol=1e-7),'Checkpoint loss differs')
            for split,pop in phase2['populations'].items():
                expected=predictions(model,pop,samples[split],split,config)
                saved=pd.read_csv(directory/f'predictions/{variant}_{snapshot}_{split}.csv',dtype={'user_key':str,'prediction':'Int64'})
                pd.testing.assert_frame_equal(saved[expected.columns],expected,check_dtype=False,atol=1e-7,rtol=0)
                require(saved.primary_checkpoint.eq(snapshot==info['selection']).all(),'Primary flag differs')
                if snapshot==info['selection']:primary[variant][split]=saved
                for cohort in ('all_users','fixed_ablation_cohort'):
                    metric,detail=mm.metrics_for(saved,cohort)
                    rows.append(dict(model=variant,checkpoint_variant=snapshot,primary_checkpoint=snapshot==info['selection'],split=split,**metric))
                    require(details[f'{variant}/{snapshot}/{split}/{cohort}']==detail,'Classification report differs')
    pd.testing.assert_frame_equal(pd.DataFrame(rows),pd.read_csv(directory/'metrics.csv'),check_dtype=False,atol=1e-10,rtol=1e-10)
    comparison,keys=compare(primary,reference_predictions,config)
    pd.testing.assert_frame_equal(comparison,pd.read_csv(directory/'reports/ablation_comparison.csv'),check_dtype=False,atol=1e-10,rtol=1e-10)
    pd.testing.assert_frame_equal(keys,pd.read_csv(directory/'reports/matched_user_keys.csv'),check_dtype=False)
    return dict(verified_artifacts=len(manifest['artifacts']),reloaded_checkpoints=6,reference_checkpoint_reloaded=True,
        verified_prediction_files=18,verified_metric_rows=len(rows),comparison_rows=len(comparison),matched_counts=keys.groupby('split').size().to_dict())
