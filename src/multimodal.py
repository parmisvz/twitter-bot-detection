"""Unchanged concatenation baseline with explicit split-local modality contracts."""
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
from torch.utils.data import DataLoader, TensorDataset
from src.baselines import (SPLITS, align, artifact_inventory, checked_artifact, evaluate_registry,
    keyed, load_phase2, load_embeddings, notebook_source_hash, read_json, require, seed_all, sha256, write_json)
from src.graph_baseline import load_graph_inputs
from src.graph_tabular import pinned_manifest


class TabularEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(40, 64), nn.ReLU(), nn.Dropout(.3))
    def forward(self, x):
        return self.encoder(x)


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(768, 128), nn.ReLU(), nn.Dropout(.3))
    def forward(self, x):
        return self.encoder(x)


class GraphEncoder(nn.Module):
    """Historical degree-statistics MLP, not a GNN."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Dropout(.3))
    def forward(self, x):
        return self.encoder(x)


class MultimodalFusionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.tabular_encoder = TabularEncoder()
        self.text_encoder = TextEncoder()
        self.graph_encoder = GraphEncoder()
        self.classifier = nn.Sequential(nn.Linear(224, 64), nn.ReLU(), nn.Dropout(.3), nn.Linear(64, 2))
    def forward(self, tabular, text, graph):
        return self.classifier(torch.cat([self.tabular_encoder(tabular), self.text_encoder(text),
                                         self.graph_encoder(graph)], dim=1))


def assemble(pop, embedding, mapping, degrees):
    """Only eligible real vectors enter tensors; all users remain in the registry."""
    masks = pop['masks']
    keys = masks.index[masks.has_graph.eq(1) & masks.has_tweet_text.eq(1)]
    require(embedding['keys'].is_unique and mapping.user_key.is_unique, 'Duplicate modality keys')
    require(mapping.node_index.tolist() == list(range(len(mapping))), 'Invalid degree row mapping')
    require(set(embedding['keys']) == set(masks.index[masks.has_tweet_text.eq(1)]), 'Tweet availability mismatch')
    ti = embedding['keys'].get_indexer(keys)
    gi = pd.Index(mapping.user_key).get_indexer(keys)
    require((ti >= 0).all() and (gi >= 0).all(), 'Eligible user lacks real modality')
    ui = pop['users'].index.get_indexer(keys)
    require((ui >= 0).all(), 'User alignment failure')
    arrays = [pop['x'][ui], embedding['x'][ti], degrees[gi].numpy()]
    require([a.shape for a in arrays] == [(len(keys), 40), (len(keys), 768), (len(keys), 3)], 'Input dimensions differ')
    require(all(np.isfinite(a).all() for a in arrays), 'Nonfinite inputs; no imputation permitted')
    return {'keys': keys, 'dataset': TensorDataset(*(torch.tensor(a, dtype=torch.float32) for a in arrays),
              torch.tensor(pop['y'][ui], dtype=torch.long))}


def load_inputs(root, config):
    require(config['training'] == dict(epochs=200,batch_size=32,lr=.001,weight_decay=.0001,
            threshold=.5,selection='best_validation_loss'), 'Unapproved training change')
    require(config['device'] == 'cpu' and config['expected_counts'] == dict(train=193,validation=42,test=40), 'Unapproved execution/cohort change')
    phase2 = load_phase2(root, config)
    require(len(phase2['feature_order']) == 40, 'Expected 40 manifest features')
    embeddings, em = load_embeddings(root, config, phase2)
    graphs, _, _, overlap = load_graph_inputs(root, config)
    manifests = {name: pinned_manifest(root, config, name) for name in ('graph','baseline','graph_tabular')}
    path, gm = manifests['graph']
    require(gm['graph_protocol'] == phase2['manifest']['graph_protocol'], 'Graph protocol differs')
    samples, rows = {}, []
    for split in SPLITS:
        spec = gm['graph_artifacts'][split]
        mapping = pd.read_csv(checked_artifact(path.parent, gm, spec['nodes']), dtype={'user_key':str})
        tensors = torch.load(checked_artifact(path.parent, gm, spec['tensors']), map_location='cpu', weights_only=True)
        pd.testing.assert_frame_equal(mapping, graphs[split]['mapping'], check_dtype=False)
        for k in ('x','edge_index','y','eligible_mask'):
            torch.testing.assert_close(tensors[k], graphs[split]['data'][k], rtol=0, atol=0)
        pop = phase2['populations'][split]
        samples[split] = assemble(pop, embeddings[split], mapping, tensors['x'])
        require(len(samples[split]['keys']) == config['expected_counts'][split], 'Cohort changed')
        m = pop['masks']
        rows.append(dict(split=split,canonical_users=len(m),eligible_users=len(samples[split]['keys']),
            has_graph=int(m.has_graph.sum()),has_tweet_text=int(m.has_tweet_text.sum()),
            has_temporal=int(m.has_temporal.sum()),temporal_complete=int(m.temporal_complete.sum())))
    return phase2, samples, em, manifests, pd.DataFrame(rows), overlap


def loader(sample, config, shuffle=False):
    return DataLoader(sample['dataset'], batch_size=config['training']['batch_size'], shuffle=shuffle,
                      generator=torch.Generator().manual_seed(config['seed']))


def validation_loss(model, sample, config, weights):
    model.eval()
    with torch.no_grad():
        values = [nn.functional.cross_entropy(model(*b[:3]), b[3], weight=weights).item()
                  for b in loader(sample, config)]
    return float(np.mean(values))  # Preserve the historical mean of batch losses.


def train_model(train, validation, config):
    seed_all(config['seed'], config['torch_threads'])
    model = MultimodalFusionClassifier()
    counts = torch.bincount(train['dataset'].tensors[3], minlength=2)
    require((counts > 0).all().item(), 'Both training classes required')
    weights = (counts.sum() / (2 * counts)).float()
    spec = config['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['lr'], weight_decay=spec['weight_decay'])
    train_loader = loader(train, config, True)
    history, best, best_epoch, best_loss = [], None, None, float('inf')
    for epoch in range(1, spec['epochs'] + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(*batch[:3]), batch[3], weight=weights)
            require(torch.isfinite(loss).item(), 'Nonfinite training loss')
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        vl = validation_loss(model, validation, config, weights)
        require(np.isfinite(vl), 'Nonfinite validation loss')
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_loss=vl))
        if vl < best_loss:
            best, best_epoch, best_loss = deepcopy(model.state_dict()), epoch, vl
        if epoch % 20 == 0:
            print(f'Epoch {epoch}: train={np.mean(losses):.6f}, validation={vl:.6f}', flush=True)
    return {'best_validation_loss':best,'last':deepcopy(model.state_dict())}, pd.DataFrame(history), dict(
        best_epoch=best_epoch,last_epoch=spec['epochs'],selection=spec['selection'],class_counts=counts.tolist(),
        class_weights=weights.tolist(),training_user_keys=train['keys'].tolist())


def predictions(model, pop, sample, split, config):
    frame = pop['masks'].copy()
    frame['label_binary'] = pop['users'].label_binary
    frame['eligible'] = frame.has_graph.eq(1) & frame.has_tweet_text.eq(1)
    frame['status'] = np.select([frame.eligible,frame.has_graph.eq(0)&frame.has_tweet_text.eq(0),frame.has_graph.eq(0)],
        ['predicted','missing_graph_and_tweet_text','missing_graph'],default='missing_tweet_text')
    frame['bot_probability'] = np.nan
    frame['prediction'] = pd.Series(pd.NA,index=frame.index,dtype='Int64')
    model.eval()
    with torch.no_grad():
        p = np.concatenate([torch.softmax(model(*b[:3]),dim=1)[:,1].numpy() for b in loader(sample,config)]) if len(sample['keys']) else np.array([])
    frame.loc[sample['keys'],'bot_probability'] = p
    frame.loc[sample['keys'],'prediction'] = (p >= config['training']['threshold']).astype(int)
    require(frame.bot_probability.notna().eq(frame.eligible).all(), 'Prediction eligibility mismatch')
    frame['model'],frame['split'] = 'multimodal_fusion',split
    return frame.reset_index()


def metrics_for(frame, cohort):
    selected = frame if cohort == 'all_users' else frame[frame.eligible]
    result, detail = evaluate_registry(selected,'all_users')
    result['cohort'] = cohort
    return result, detail


def load_checkpoint(path):
    payload = torch.load(path,map_location='cpu',weights_only=True)
    model = MultimodalFusionClassifier()
    model.load_state_dict(payload['state_dict'],strict=True)
    return model.eval(),payload


def comparison(root, config, primary):
    sources = {n:pinned_manifest(root,config,n) for n in ('baseline','graph','graph_tabular')}
    rows, key_rows = [], []
    for split in ('validation','test'):
        ref = keyed(primary[split],'fusion predictions')
        keys = sorted(ref.index[ref.eligible])
        require(len(keys) == config['expected_counts'][split], 'Matched cohort changed')
        candidates = {'multimodal_fusion':ref}
        for name in ('tabular_lr','tabular_mlp','text_lr','text_mlp','gcn_degree_only','gcn_tabular'):
            prefix = 'graph' if name == 'gcn_degree_only' else 'graph_tabular' if name == 'gcn_tabular' else 'baseline'
            path,m = sources[prefix]
            variant = m['models'][name]['selected'] if prefix == 'baseline' else m['training']['selection']
            stem = 'gcn' if name == 'gcn_degree_only' else name
            candidates[name] = keyed(pd.read_csv(checked_artifact(path.parent,m,f'predictions/{stem}_{variant}_{split}.csv'),dtype={'user_key':str}),name)
        for name, candidate in candidates.items():
            frame = align(candidate,ref,'comparison users')
            require(frame.label_binary.eq(ref.label_binary).all(), 'Comparison labels differ')
            for mask in ('has_graph','has_tweet_text','has_temporal'):
                require(frame[mask].eq(ref[mask]).all(),'Comparison modality mask differs')
            require(frame.loc[keys,'status'].eq('predicted').all(), 'Unsupported matched user')
            metric,_ = evaluate_registry(frame.loc[keys].reset_index(),'all_users')
            metric['cohort'] = 'graph_and_tweet_available_exact_keys'
            rows.append(dict(model=name,split=split,comparison_type=config['comparison_type'],**metric))
        key_rows.extend(dict(split=split,user_key=k) for k in keys)
    return pd.DataFrame(rows),pd.DataFrame(key_rows)


class MultimodalRun:
    def __init__(self, root, config_path='configs/multimodal_v2.json'):
        self.root = Path(root).resolve()
        self.config = read_json(self.root/config_path)
        self.phase2,self.samples,self.embedding_manifest,self.inputs,self.coverage,self.overlap = load_inputs(self.root,self.config)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        self.output_dir = self.root/self.config['output_root']/(stamp+'_'+uuid4().hex[:8])
        self.output_dir.mkdir(parents=True,exist_ok=False)
        for name in ('models','histories','predictions','reports','provenance'):
            (self.output_dir/name).mkdir()
        self.coverage.to_csv(self.output_dir/'reports/coverage.csv',index=False)
        self.overlap.to_csv(self.output_dir/'reports/shared_support.csv',index=False)
        write_json(self.output_dir/'provenance/resolved_config.json',self.config)
        for name,m in [('phase2',self.phase2['manifest']),('embedding',self.embedding_manifest)]+[(n,m) for n,(_,m) in self.inputs.items()]:
            write_json(self.output_dir/f'provenance/{name}_manifest.json',m)
        for split,pop in self.phase2['populations'].items():
            pop['observed'].to_csv(self.output_dir/f'provenance/{split}_observation_masks.csv')
            pd.DataFrame({'user_key':self.samples[split]['keys']}).to_csv(self.output_dir/f'provenance/{split}_eligible_keys.csv',index=False)
        sources = ['src/multimodal.py','src/baselines.py','src/graph_baseline.py','src/graph_tabular.py']
        for source in sources:
            (self.output_dir/'provenance'/Path(source).name).write_bytes((self.root/source).read_bytes())
        write_json(self.output_dir/'provenance/run.json',dict(python=platform.python_version(),platform=platform.platform(),
            packages={n:importlib.metadata.version(n) for n in ('torch','numpy','pandas','scikit-learn','torch-geometric')},
            source_sha256={s:sha256(self.root/s) for s in sources},config_sha256=sha256(self.root/config_path),
            notebook_source_sha256=notebook_source_hash(self.root/'analysis/06_multimodal_fusion.ipynb'),
            git_commit_at_execution=subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.root,text=True).strip(),
            git_status_at_execution=subprocess.check_output(['git','status','--porcelain'],cwd=self.root,text=True),
            seed=self.config['seed'],device='cpu',torch_threads=self.config['torch_threads'],deterministic_algorithms=True,
            limitations=[self.embedding_manifest['limitation'],
              'Degree MLP, not GNN. Concatenation, no attention, temporal branch or contrastive loss.',
              'Temporal zeros are Phase-2 storage placeholders, not estimates; observation masks retained outside model inputs.',
              'Restricted coverage: missing graph/tweets yield null predictions. No fake vectors or fallback classifier.',
              'Graph degrees use isolated gold cohorts but shared support identities; not fully node-disjoint.',
              'Descriptive comparisons differ in training populations/settings; six test bots, one seed, previously inspected test data.']))
        self.info = None

    def fit(self):
        require(self.info is None,'Already trained; create a new run')
        self.states,self.history,self.info = train_model(self.samples['train'],self.samples['validation'],self.config)
        self.history.to_csv(self.output_dir/'histories/multimodal.csv',index=False)
        self.info['checkpoints'] = {}
        for variant,state in self.states.items():
            relative = f'models/multimodal_{variant}.pt'
            torch.save(dict(state_dict=state,variant=variant,epoch=self.info['best_epoch'] if variant=='best_validation_loss' else self.info['last_epoch'],
                feature_order=self.phase2['feature_order'],input_dimensions=[40,768,3],
                input_manifest_hashes={k:v for k,v in self.config.items() if k.endswith('_sha256')}),self.output_dir/relative)
            self.info['checkpoints'][variant] = relative
        write_json(self.output_dir/'models/training_metadata.json',self.info)
        return self.history

    def evaluate_export(self):
        require(not (self.output_dir/'manifest.json').exists(),'Run already complete')
        rows,details,primary,reload_checks = [],{},{},[]
        for variant,relative in self.info['checkpoints'].items():
            model,_ = load_checkpoint(self.output_dir/relative)
            memory = MultimodalFusionClassifier().eval()
            memory.load_state_dict(self.states[variant])
            for split,pop in self.phase2['populations'].items():
                frame = predictions(model,pop,self.samples[split],split,self.config)
                expected = predictions(memory,pop,self.samples[split],split,self.config)
                pd.testing.assert_frame_equal(frame,expected,check_exact=True)
                reload_checks.append(dict(variant=variant,split=split,identical=True))
                frame['primary_checkpoint'] = variant==self.info['selection']
                frame.to_csv(self.output_dir/f'predictions/multimodal_{variant}_{split}.csv',index=False)
                if variant==self.info['selection']:
                    primary[split] = frame
                for cohort in ('all_users','graph_and_tweet_available'):
                    metric,detail = metrics_for(frame,cohort)
                    rows.append(dict(model='multimodal_fusion',checkpoint_variant=variant,primary_checkpoint=variant==self.info['selection'],split=split,**metric))
                    details[f'{variant}/{split}/{cohort}'] = detail
        self.comparison,keys = comparison(self.root,self.config,primary)
        self.comparison.to_csv(self.output_dir/'reports/descriptive_matched_comparison.csv',index=False)
        keys.to_csv(self.output_dir/'reports/matched_user_keys.csv',index=False)
        metrics = pd.DataFrame(rows)
        metrics.to_csv(self.output_dir/'metrics.csv',index=False)
        write_json(self.output_dir/'reports/classification_reports.json',details)
        write_json(self.output_dir/'reports/checkpoint_reload.json',reload_checks)
        write_json(self.output_dir/'manifest.json',dict(schema_version='multimodal_v2',complete=True,identity_key='user_key',
            feature_order=self.phase2['feature_order'],training=self.info,eligibility='has_graph AND has_tweet_text',
            input_manifest_hashes={k:v for k,v in self.config.items() if k.endswith('_sha256')},
            downstream_contract='Use exact manifest inputs and selected checkpoint; retain full canonical registry and unsupported nulls. No missing-modality deployment support yet.',
            artifacts=artifact_inventory(self.output_dir)))
        return metrics


def verify_run(root, output_dir):
    directory = Path(output_dir)
    m = read_json(directory/'manifest.json')
    require(m['complete'],'Incomplete run')
    for relative in m['artifacts']:
        checked_artifact(directory,m,relative)
    config = read_json(directory/'provenance/resolved_config.json')
    seed_all(config['seed'],config['torch_threads'])
    phase2,samples,_,_,coverage,overlap = load_inputs(root,config)
    require(m['feature_order']==phase2['feature_order'],'Feature order differs')
    hashes = {k:v for k,v in config.items() if k.endswith('_sha256')}
    require(m['input_manifest_hashes']==hashes,'Input hashes differ')
    pd.testing.assert_frame_equal(coverage,pd.read_csv(directory/'reports/coverage.csv'),check_dtype=False)
    pd.testing.assert_frame_equal(overlap,pd.read_csv(directory/'reports/shared_support.csv'),check_dtype=False)
    info = m['training']
    require(info==read_json(directory/'models/training_metadata.json'),'Training metadata differs')
    counts = torch.bincount(samples['train']['dataset'].tensors[3],minlength=2)
    weights = (counts.sum()/(2*counts)).float()
    require(info['class_counts']==counts.tolist() and info['class_weights']==weights.tolist(),'Training weights differ')
    require(info['training_user_keys']==samples['train']['keys'].tolist(),'Training keys differ')
    history = pd.read_csv(directory/'histories/multimodal.csv')
    require(history.epoch.tolist()==list(range(1,201)),'Incomplete history')
    require(info['best_epoch']==int(history.loc[history.validation_loss.idxmin(),'epoch']),'Wrong best epoch')
    saved_metrics = pd.read_csv(directory/'metrics.csv')
    saved_details = read_json(directory/'reports/classification_reports.json')
    primary,rows = {},[]
    for variant,relative in info['checkpoints'].items():
        model,payload = load_checkpoint(directory/relative)
        epoch = info['best_epoch'] if variant=='best_validation_loss' else info['last_epoch']
        require(payload['variant']==variant and payload['epoch']==epoch and payload['feature_order']==m['feature_order'] and payload['input_manifest_hashes']==hashes,'Checkpoint metadata differs')
        loss = validation_loss(model,samples['validation'],config,weights)
        require(np.isclose(loss,history.loc[history.epoch.eq(epoch),'validation_loss'].iloc[0],atol=1e-7,rtol=1e-7),'Reload loss differs')
        for split,pop in phase2['populations'].items():
            expected = predictions(model,pop,samples[split],split,config)
            saved = pd.read_csv(directory/f'predictions/multimodal_{variant}_{split}.csv',dtype={'user_key':str,'prediction':'Int64'})
            pd.testing.assert_frame_equal(saved[expected.columns],expected,check_dtype=False,atol=1e-7,rtol=0)
            require(saved.primary_checkpoint.eq(variant==info['selection']).all(),'Primary flag differs')
            if variant==info['selection']:
                primary[split] = saved
            for cohort in ('all_users','graph_and_tweet_available'):
                metric,detail = metrics_for(saved,cohort)
                rows.append(dict(model='multimodal_fusion',checkpoint_variant=variant,primary_checkpoint=variant==info['selection'],split=split,**metric))
                require(saved_details[f'{variant}/{split}/{cohort}']==detail,'Classification report differs')
    pd.testing.assert_frame_equal(pd.DataFrame(rows),saved_metrics,check_dtype=False,atol=1e-10,rtol=1e-10)
    comp,keys = comparison(root,config,primary)
    pd.testing.assert_frame_equal(comp,pd.read_csv(directory/'reports/descriptive_matched_comparison.csv'),check_dtype=False,atol=1e-10,rtol=1e-10)
    pd.testing.assert_frame_equal(keys,pd.read_csv(directory/'reports/matched_user_keys.csv'),check_dtype=False)
    return dict(verified_artifacts=len(m['artifacts']),reloaded_checkpoints=2,verified_prediction_files=6,
        verified_metric_rows=len(rows),descriptive_comparison_rows=len(comp),matched_counts=keys.groupby('split').size().to_dict())
