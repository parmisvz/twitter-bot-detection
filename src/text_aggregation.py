"""Paired frozen tweet-vector aggregation; no language-model or graph inference."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score

from src.baselines import seed_all, text_digest
from src.temporal_statistics import TemporalExperimentModel, load_inputs as temporal_inputs, make_loader, POLICIES
from src.checkpoint_weight_experiment import state_hash, weights_from_training, score, validation_losses
from src.performance_diagnosis import Inputs, digest, require, write_json, preserved_hashes, metrics, plt, save_figure

ARMS = ['T0_mean', 'T1_max']

# Training loop copied from temporal_statistics; only the model factory changes.
def train_one(data, mapping, config, arm, seed, weight):
    seed_all(seed, config["torch_threads"])
    model = TemporalTextModel(arm, mapping, config["dropout"])
    count = sum(p.numel() for p in model.parameters())
    require(count == config["expected_parameters"][arm], "Architecture parameter count differs")
    initial_hash = state_hash(model.state_dict())
    spec = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], dtype=torch.float32))
    loaders = {split: make_loader(pop, spec, seed, split) for split, pop in data.items()}
    best, best_epoch, best_state = float("inf"), None, None
    history, order_hash, epoch_order_hashes = [], hashlib.sha256(), []
    for epoch in range(1, spec["epochs"] + 1):
        row = {"epoch": epoch}
        seen = []
        for split, loader in loaders.items():
            model.train(split == "train")
            total, unweighted, n = 0., 0., 0
            with torch.set_grad_enabled(split == "train"):
                for x, y, indices in loader:
                    if split == "train":
                        seen.extend(indices.tolist())
                        order_hash.update(indices.numpy().tobytes())
                        optimizer.zero_grad()
                    logits = model(x)
                    loss = criterion(logits, y)
                    if split == "train":
                        loss.backward()
                        optimizer.step()
                    else:
                        unweighted += nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="sum").item()
                    total += loss.item()
                    n += len(x)
            row[split + "_weighted_loss"] = total / len(loader)
            if split == "validation":
                row["validation_unweighted_loss"] = unweighted / n
        require(sorted(seen) == list(range(len(data["train"]["y"]))), "Training epoch omitted or repeated a user")
        epoch_order_hashes.append(hashlib.sha256(np.asarray(seen, dtype=np.int64).tobytes()).hexdigest())
        require(np.isfinite(list(row.values())).all(), "Nonfinite losses")
        history.append(row)
        if row["validation_unweighted_loss"] < best:
            best, best_epoch, best_state = row["validation_unweighted_loss"], epoch, deepcopy(model.state_dict())
        if epoch % 25 == 0:
            print(f"{arm}, seed {seed}, epoch {epoch}: validation BCE {row['validation_unweighted_loss']:.4f}", flush=True)
    return {config["selection"]: best_state, "last": deepcopy(model.state_dict())}, pd.DataFrame(history), {
        "arm": arm, "seed": seed, "pos_weight": weight, "parameter_count": count,
        "initial_state_sha256": initial_hash, "batch_order_sha256": order_hash.hexdigest(),
        "epoch_order_sha256": epoch_order_hashes, "epochs": {config["selection"]: best_epoch, "last": spec["epochs"]}}


class TemporalTextModel(nn.Module):
    def __init__(self, arm, mapping, dropout=.3):
        super().__init__()
        require(arm in ARMS, 'Unknown aggregation arm')
        self.base = TemporalExperimentModel('temporal_branch', mapping, dropout)
        self.text_projection = nn.Sequential(nn.Linear(768, 16), nn.ReLU(), nn.Dropout(dropout))
        self.base.classifier = nn.Sequential(nn.Linear(80, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        require(x.ndim == 2 and x.shape[1] == 816, 'Expected temporal47 + text768 + availability1')
        other, temporal, mask = self.base.split_inputs(x[:, :47])
        available = x[:, 815:816]
        require(((available == 0) | (available == 1)).all().item(), 'Invalid tweet mask')
        values = torch.where(available.bool(), x[:, 47:815], torch.zeros_like(x[:, 47:815]))
        text = self.text_projection(values) * available
        return self.base.classifier(torch.cat([self.base.other_encoder(other), self.base.temporal_representation(temporal, mask), text], dim=1))


def aggregate(text, vectors, keys, mask):
    require(keys.is_unique and keys.notna().all(), 'Invalid user keys')
    require(vectors.shape == (len(text), 768) and np.isfinite(vectors).all(), 'Invalid cached tweet vectors')
    require(text.user_key.notna().all() and set(text.user_key) == set(keys[mask]), 'Tweet coverage mismatch')
    groups = text.groupby('user_key', sort=True).indices
    result = {a: np.full((len(keys), 768), np.nan, dtype=np.float32) for a in ARMS}
    count = np.zeros(len(keys), dtype=np.int64)
    for i, key in enumerate(keys):
        if key in groups:
            v = vectors[groups[key]]
            result['T0_mean'][i], result['T1_max'][i], count[i] = v.mean(axis=0), v.max(axis=0), len(v)
    require(np.array_equal(count > 0, mask), 'Counts and availability differ')
    np.testing.assert_array_equal(result['T0_mean'][count == 1], result['T1_max'][count == 1])
    return result, count


def array(reader, relative, sha):
    path = reader.root / relative
    require(path.resolve().is_relative_to(reader.root) and 'test' not in path.stem.lower(), 'Forbidden vector input')
    require(digest(path) == sha, 'Vector hash mismatch')
    reader.accessed[relative] = sha
    return np.load(path, allow_pickle=False)


def inputs(root, config, reader):
    require(config['arms'] == ARMS and config['seeds'] == [42, 43, 44] and config['threshold'] == .5
            and config['selection'] == POLICIES[0] and config['dropout'] == .3 and config['torch_threads'] == 1
            and config['expected_parameters'] == dict.fromkeys(ARMS, 16897)
            and config['scope'] == 'cached_tweets_only_train_validation_no_test_no_graph_no_regeneration_no_promotion', 'Protocol changed')
    path = root / config['temporal_manifest']
    reader.allow(path, config['temporal_manifest_sha256']); temporal_manifest = reader.read(path)
    original = reader.registered(path.parent, temporal_manifest, 'provenance/resolved_config.json')
    require(config['training'] == original['training'] and temporal_manifest['complete'], 'Training protocol differs')
    data, mapping, _ = temporal_inputs(root, original, reader)
    require(mapping == temporal_manifest['feature_mapping'], 'Temporal feature mapping differs')
    reference = reader.registered(path.parent, temporal_manifest, 'validation_metrics.csv')
    reference = reference[reference.arm.eq('temporal_branch')].copy()
    phase_path = root / original['phase2_manifest']; phase = reader.read(phase_path)
    ep = root / config['embedding_manifest']
    reader.allow(ep, config['embedding_manifest_sha256']); em = reader.read(ep)
    require(em['phase2_manifest_sha256'] == original['phase2_manifest_sha256'] and em['encoder'] == {
        'model': 'xlm-roberta-base', 'dimension': 768, 'max_length': 128, 'batch_size': 16,
        'token_pooling': 'attention_mask_mean', 'user_pooling': 'unweighted_tweet_mean', 'frozen': True}, 'Embedding contract differs')
    representations, counts, texts = {}, {}, {}
    for split, pop in data.items():
        text = reader.registered(phase_path.parent, phase, f'splits/text/{split}_text.csv')
        require(text_digest(text) == em['ordered_text_sha256'][split], 'Ordered prepared text differs from cache registration')
        require(text.text_model.fillna('').str.strip().ne('').all(), 'Empty tweet')
        alias = 'train' if split == 'train' else 'val'
        rel = f'final_data/embeddings/xlmr/{alias}_tweet_embeddings.npy'
        v = array(reader, rel, em['source_sha256'][rel])
        keys = pop['users'].index
        reps, count = aggregate(text, v, keys, pop['masks'].has_tweet_text.to_numpy(dtype=bool))
        saved_keys = reader.registered(ep.parent, em, f'{split}_keys.csv').user_key
        rel = (ep.parent / f'{split}_embeddings.npy').relative_to(root).as_posix()
        saved = array(reader, rel, em['artifacts'][f'{split}_embeddings.npy']['sha256'])
        require(saved_keys.is_unique and set(saved_keys) == set(keys[count > 0]), 'Cached user identity differs')
        idx = keys.get_indexer(saved_keys)
        np.testing.assert_array_equal(reps['T0_mean'][idx], saved)
        representations[split], counts[split], texts[split] = reps, count, text
    require(not set(texts['train'].text_key) & set(texts['validation'].text_key), 'Exact text leakage')
    return data, mapping, representations, counts, texts, reference, em['limitation']


def model_data(data, reps, arm):
    return {s: {**p, 'x': np.concatenate([p['x'], reps[s][arm], p['masks'].has_tweet_text.to_numpy(dtype=np.float32)[:, None]], axis=1)} for s, p in data.items()}


def representation_registry(pop, reps, count, arm):
    f = pop['masks'][['has_tweet_text']].copy()
    f['tweet_count'], f['aggregation'] = count, arm
    f['status'] = np.where(count > 0, 'observed_cached_tweets', 'missing_tweet_text')
    f['l2_norm'] = np.linalg.norm(reps[arm].astype(np.float64), axis=1)
    return f.reset_index()


def cohort_rows(frame):
    rows = []
    for cohort, keep in [('all', np.ones(len(frame), bool)), ('one_tweet', frame.tweet_count.eq(1)),
                         ('2_to_10', frame.tweet_count.between(2, 10)), ('11_to_50', frame.tweet_count.between(11, 50)),
                         ('missing_text', frame.has_tweet_text.eq(0))]:
        f = frame.loc[keep]
        identity = {k: frame.iloc[0][k] for k in ['arm', 'seed', 'policy', 'epoch']}
        row = {**identity, 'cohort': cohort, 'n': len(f), 'bots': int(f.label_binary.sum())}
        if len(f):
            row.update(metrics(f.label_binary, f.bot_probability, .5))
            two = f.label_binary.nunique() == 2
            row.update(roc_auc=roc_auc_score(f.label_binary, f.bot_probability) if two else np.nan,
                       average_precision=average_precision_score(f.label_binary, f.bot_probability) if two else np.nan,
                       brier=float(np.mean((f.bot_probability - f.label_binary)**2)), metric_status='defined' if two else 'single_class_ranking_undefined')
        else:
            row['metric_status'] = 'empty_cohort'
        rows.append(row)
    return rows


def summarize(table):
    names = ['bot_f1', 'bot_precision', 'bot_recall', 'roc_auc', 'average_precision', 'brier']
    summary = table.groupby(['arm', 'policy', 'cohort'])[names].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    effects = []
    for r in table[table.arm.eq('T1_max')].itertuples():
        ref = table[table.arm.eq('T0_mean') & table.seed.eq(r.seed) & table.policy.eq(r.policy) & table.cohort.eq(r.cohort)]
        require(len(ref) == 1, 'Missing control')
        effects.append({'seed': r.seed, 'policy': r.policy, 'cohort': r.cohort, **{'delta_' + n: float(getattr(r, n) - ref.iloc[0][n]) for n in names}})
    return summary.reset_index(), pd.DataFrame(effects)


def scored(model, data, counts, config, info, policy):
    frame, _ = score(model, data, config, arm=info['arm'], seed=info['seed'], policy=policy, epoch=info['epochs'][policy])
    frame['tweet_count'] = counts['validation']
    frame['text_status'] = np.where(frame.has_tweet_text.eq(1), 'observed_cached_tweets', 'masked_absent')
    weighted, common = validation_losses(model, data['validation'], config['training']['batch_size'], info['pos_weight'])
    rows = cohort_rows(frame)
    for r in rows:
        r.update(parameter_count=info['parameter_count'], validation_weighted_loss=weighted, validation_unweighted_loss=common)
    return frame, rows, (weighted, common)


def run(root, config_path='configs/text_aggregation_v1.json'):
    root = Path(root).resolve(); reader = Inputs(root)
    reader.allow(root / config_path, digest(root / config_path)); config = reader.read(root / config_path)
    before = preserved_hashes(root, config['output_root'])
    data, mapping, reps, counts, texts, reference, limitation = inputs(root, config, reader)
    output = root / config['output_root'] / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid4().hex[:8])
    for name in ['representations', 'models', 'histories', 'predictions', 'provenance', 'figures']:
        (output / name).mkdir(parents=True, exist_ok=False)
    write_json(output / 'provenance/resolved_config.json', config)
    write_json(output / 'provenance/feature_mapping.json', mapping)
    write_json(output / 'provenance/embedding_provenance.json', {'limitation': limitation, 'regenerated': False,
        'representation_rows': 'canonical user order; missing vectors remain NaN, masked before projection',
        'T0_mean': 'numpy float32 arithmetic mean over all prepared tweets', 'T1_max': 'coordinate-wise maximum over identical rows',
        'normalization': 'none', 'ordered_text_sha256': {s: text_digest(t) for s,t in texts.items()}})
    reference.to_csv(output / 'temporal_reference_validation.csv', index=False)
    norms = []
    for split, pop in data.items():
        pop['users'][['label_binary']].join(pop['masks'][['has_tweet_text','has_temporal']]).to_csv(output / f'provenance/{split}_population.csv')
        texts[split][['user_key','source_row','text_key']].assign(vector_row=np.arange(len(texts[split]))).to_csv(output / f'provenance/{split}_tweet_rows.csv', index=False)
        for arm in ARMS:
            np.save(output / f'representations/{split}_{arm}.npy', reps[split][arm], allow_pickle=False)
            f = representation_registry(pop, reps[split], counts[split], arm)
            f.to_csv(output / f'representations/{split}_{arm}_keys.csv', index=False)
            norms.append(f.assign(split=split))
    pd.concat(norms, ignore_index=True).to_csv(output / 'embedding_norms.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, split in zip(axes, ['train', 'validation']):
        values = [np.linalg.norm(reps[split][a][counts[split] > 0].astype(np.float64), axis=1) for a in ARMS]
        ax.boxplot(values, tick_labels=['Mean', 'Maximum'])
        ax.set(title=split, ylabel='User embedding L2 norm')
    save_figure(fig, output / 'figures/embedding_norms.png')
    rows, runs, histories = [], {}, {}
    weight = weights_from_training(data['train']['y'])['full_train_ratio']
    for seed in config['seeds']:
        for arm in ARMS:
            print(f'Training {arm}, seed {seed}', flush=True)
            prepared = model_data(data, reps, arm)
            states, h, info = train_one(prepared, mapping, config, arm, seed, weight)
            info['training_user_keys'] = data['train']['users'].index.tolist(); info['checkpoints'] = {}
            name = f'{arm}_seed{seed}'
            h.to_csv(output / f'histories/{name}.csv', index=False); histories[name] = h
            for policy, state in states.items():
                model = TemporalTextModel(arm, mapping); model.load_state_dict(state)
                f, scores, _ = scored(model, prepared, counts, config, info, policy)
                f.to_csv(output / f'predictions/{name}_{policy}_validation.csv', index=False); rows.extend(scores)
                rel = f'models/{name}_{policy}.pt'
                torch.save({'state_dict': state, 'arm': arm, 'seed': seed, 'policy': policy, 'epoch': info['epochs'][policy],
                    'mapping': mapping, 'threshold': .5, 'pos_weight': weight, 'training_user_keys': info['training_user_keys'],
                    'embedding_manifest_sha256': config['embedding_manifest_sha256'], 'temporal_manifest_sha256': config['temporal_manifest_sha256']}, output / rel)
                info['checkpoints'][policy] = rel
            runs[name] = info
        pair = [runs[f'{a}_seed{seed}'] for a in ARMS]
        for key in ['initial_state_sha256','batch_order_sha256','epoch_order_sha256','training_user_keys','parameter_count','pos_weight']:
            require(pair[0][key] == pair[1][key], 'Pairing differs: ' + key)
    table = pd.DataFrame(rows); summary, effects = summarize(table)
    for name,f in [('validation_metrics',table),('seed_summary',summary),('paired_effects',effects)]: f.to_csv(output / (name + '.csv'), index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ['bot_f1','roc_auc','average_precision']):
        for seed in config['seeds']:
            f=table[table.seed.eq(seed)&table.policy.eq(POLICIES[0])&table.cohort.eq('all')].set_index('arm').loc[ARMS]
            ax.plot([0,1],f[metric],marker='o',label=f'seed {seed}')
        ax.set(xticks=[0,1],xticklabels=['Mean','Maximum'],ylabel=metric,ylim=(0,1));ax.legend()
    save_figure(fig,output/'figures/paired_metrics.png')
    fig, axes=plt.subplots(2,3,figsize=(12,7))
    for ax,(name,h) in zip(axes.flat,histories.items()):
        ax.plot(h.epoch,h.validation_unweighted_loss,label='validation BCE');ax.axvline(runs[name]['epochs'][POLICIES[0]],linestyle=':',color='black');ax.set(title=name,xlabel='Epoch');ax.legend()
    save_figure(fig,output/'figures/learning_curves.png')
    write_json(output/'training_metadata.json',runs)
    write_json(output/'provenance/parsed_inputs.json',reader.accessed)
    write_json(output/'provenance/protected_files.json',before)
    for file in ['src/text_aggregation.py','src/temporal_statistics.py','src/checkpoint_weight_experiment.py','src/performance_diagnosis.py','src/baselines.py','scripts/run_text_aggregation.py']:
        (output/'provenance'/Path(file).name).write_bytes((root/file).read_bytes())
    write_json(output/'provenance/environment.json',{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,'pandas':pd.__version__,
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'torch_threads':torch.get_num_threads(),
        'test_inputs_parsed':False,'language_model_loaded':False,'threshold_tuned':False})
    require(before == preserved_hashes(root,config['output_root']), 'Protected files changed')
    write_json(output/'manifest.json',{'complete':True,'schema_version':config['schema_version'],'promoted_model':None,'runs':runs,
        'protected_files_verified':len(before),'artifacts':{p.relative_to(output).as_posix():{'sha256':digest(p)} for p in sorted(output.rglob('*')) if p.is_file()}})
    return output


def verify(root, output):
    root,output=Path(root).resolve(),Path(output).resolve()
    manifest=json.loads((output/'manifest.json').read_text());require(manifest['complete'] and manifest['promoted_model'] is None,'Incomplete run')
    for rel,meta in manifest['artifacts'].items(): require(digest(output/rel)==meta['sha256'],'Artifact differs: '+rel)
    config=json.loads((output/'provenance/resolved_config.json').read_text());reader=Inputs(root)
    data,mapping,reps,counts,texts,reference,_=inputs(root,config,reader)
    pd.testing.assert_frame_equal(reference.reset_index(drop=True),pd.read_csv(output/'temporal_reference_validation.csv'),check_dtype=False,atol=1e-10,rtol=1e-10)
    for split,pop in data.items():
        for arm in ARMS:
            np.testing.assert_array_equal(reps[split][arm],np.load(output/f'representations/{split}_{arm}.npy',allow_pickle=False))
            pd.testing.assert_frame_equal(representation_registry(pop,reps[split],counts[split],arm),pd.read_csv(output/f'representations/{split}_{arm}_keys.csv',dtype={'user_key':str}),check_dtype=False,atol=1e-10,rtol=1e-10)
    rows=[]
    for seed in config['seeds']:
        pair=[manifest['runs'][f'{a}_seed{seed}'] for a in ARMS]
        for key in ['initial_state_sha256','batch_order_sha256','epoch_order_sha256','training_user_keys','parameter_count','pos_weight']:
            require(pair[0][key]==pair[1][key],'Pairing differs: '+key)
        loader=make_loader(data['train'],config['training'],seed,'train');orders=[];combined=hashlib.sha256()
        for _ in range(50):
            seen=[]
            for _,_,idx in loader: seen.extend(idx.tolist());combined.update(idx.numpy().tobytes())
            require(sorted(seen)==list(range(len(data['train']['y']))),'Invalid batch membership')
            orders.append(hashlib.sha256(np.asarray(seen,dtype=np.int64).tobytes()).hexdigest())
        require(pair[0]['epoch_order_sha256']==orders and pair[0]['batch_order_sha256']==combined.hexdigest(),'Shuffle differs')
        for arm,info in zip(ARMS,pair):
            seed_all(seed,1); initial=TemporalTextModel(arm,mapping)
            require(state_hash(initial.state_dict())==info['initial_state_sha256'] and info['parameter_count']==16897,'Initialization/capacity differs')
            name=f'{arm}_seed{seed}';h=pd.read_csv(output/f'histories/{name}.csv')
            require(h.epoch.tolist()==list(range(1,51)),'Incomplete history')
            prepared=model_data(data,reps,arm)
            for policy,rel in info['checkpoints'].items():
                payload=torch.load(output/rel,map_location='cpu',weights_only=True)
                epoch=50 if policy=='last' else int(h.loc[h.validation_unweighted_loss.idxmin(),'epoch'])
                require(payload['arm']==arm and payload['seed']==seed and payload['policy']==policy and payload['epoch']==epoch==info['epochs'][policy]
                    and payload['mapping']==mapping and payload['threshold']==.5 and payload['pos_weight']==weights_from_training(data['train']['y'])['full_train_ratio']
                    and payload['training_user_keys']==data['train']['users'].index.tolist()
                    and payload['embedding_manifest_sha256']==config['embedding_manifest_sha256'] and payload['temporal_manifest_sha256']==config['temporal_manifest_sha256'],'Checkpoint provenance differs')
                initial.load_state_dict(payload['state_dict'],strict=True)
                frame,scores,losses=scored(initial,prepared,counts,config,info,policy);rows.extend(scores)
                pd.testing.assert_frame_equal(frame,pd.read_csv(output/f'predictions/{name}_{policy}_validation.csv',dtype={'user_key':str}),check_dtype=False,atol=1e-7,rtol=1e-6)
                np.testing.assert_allclose(losses,h.set_index('epoch').loc[epoch,['validation_weighted_loss','validation_unweighted_loss']].to_numpy(dtype=float),atol=1e-7,rtol=1e-7)
    table=pd.read_csv(output/'validation_metrics.csv');keys=['arm','seed','policy','cohort']
    pd.testing.assert_frame_equal(pd.DataFrame(rows).sort_values(keys).reset_index(drop=True),table.sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-7,rtol=1e-6)
    for name,frame in zip(['seed_summary','paired_effects'],summarize(table)):
        pd.testing.assert_frame_equal(frame,pd.read_csv(output/(name+'.csv')),check_dtype=False,atol=1e-10,rtol=1e-10)
    parsed=json.loads((output/'provenance/parsed_inputs.json').read_text())
    require(not any('test' in Path(p).stem.lower() or '/graph/' in p or 'description' in Path(p).stem.lower() for p in parsed),'Forbidden input parsed')
    require(all(digest(root/p)==sha for p,sha in parsed.items()),'Parsed input changed')
    return {'training_runs':6,'checkpoints_reloaded':12,'keyed_representation_matrices':4,'single_tweet_users_verified':{s:int((v==1).sum()) for s,v in counts.items()},'test_inputs_parsed':0}
