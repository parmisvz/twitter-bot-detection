"""Controlled optional-modality gating on frozen scaled GCN inputs."""
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

from src.gcn_integration import (initialized_model as initialize_base, evaluate,
    prediction_registry, POLICIES, registered_path)
from src.graph_scaling import inputs as load_scaled, scored_rows
from src.temporal_statistics import make_loader
from src.checkpoint_weight_experiment import state_hash, weights_from_training
from src.performance_diagnosis import Inputs, digest, preserved_hashes, require, write_json, plt, save_figure


class GatedFusion(nn.Module):
    def __init__(self, model, arm):
        super().__init__()
        require(arm in ('concat', 'gated'), 'Unknown fusion arm')
        self.model, self.arm = model, arm
        if arm == 'gated':
            self.gate = nn.Linear(99, 3)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def graph_representation(self, graph):
        return self.model.graph_representation(graph)

    def components(self, x, indices, graph, representation=None):
        other, temporal, mask = self.model.base.split_inputs(x)
        other_h = self.model.base.other_encoder(other)
        temporal_h = self.model.base.temporal_representation(temporal, mask)
        h = self.graph_representation(graph) if representation is None else representation
        available = torch.stack([torch.ones(len(x), dtype=torch.bool), mask.any(dim=1), graph['available'][indices]], dim=1)
        graph_h = h[indices] * available[:, 2:3]
        combined = torch.cat([other_h, temporal_h, graph_h], dim=1)
        weights = available.to(combined.dtype)
        if self.arm == 'gated':
            weights = 2 * self.gate(torch.cat([combined, weights], dim=1)).sigmoid() * weights
        fused = torch.cat([other_h * weights[:, :1], temporal_h * weights[:, 1:2], graph_h * weights[:, 2:3]], dim=1)
        return fused, weights

    def forward(self, x, indices, graph, representation=None):
        fused, _ = self.components(x, indices, graph, representation)
        return self.model.base.classifier(fused)


def initialized_model(arm, mapping, seed):
    base = initialize_base('gcn', mapping, seed)
    rng = torch.get_rng_state()
    model = GatedFusion(base, arm)
    # Gate allocation must not alter dropout RNG or paired training batches.
    torch.set_rng_state(rng)
    return model


def train_one(data, graphs, mapping, spec, arm, seed, weight):
    """Same loop as gcn_integration.train_one; only model factory differs."""
    model = initialized_model(arm, mapping, seed)
    info = {'arm': arm, 'seed': seed, 'parameter_count': sum(p.numel() for p in model.parameters()),
            'initial_common_sha256': state_hash(model.model.state_dict()), 'pos_weight': weight,
            'training_user_keys': data['train']['users'].index.tolist()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['lr'], weight_decay=spec['weight_decay'])
    loader = make_loader(data['train'], spec, seed, 'train')
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], dtype=torch.float32))
    best, history, orders = float('inf'), [], []
    for epoch in range(1, spec['epochs'] + 1):
        model.train()
        total, seen = 0., []
        for x, y, indices in loader:
            seen.extend(indices.tolist())
            optimizer.zero_grad()
            loss = criterion(model(x, indices, graphs['train']), y)
            require(torch.isfinite(loss).item(), 'Nonfinite training loss')
            loss.backward()
            optimizer.step()
            total += loss.item()
        require(sorted(seen) == list(range(len(data['train']['y']))), 'Training user omitted/repeated')
        orders.append(hashlib.sha256(np.asarray(seen, dtype=np.int64).tobytes()).hexdigest())
        _, weighted, common = evaluate(model, data['validation'], graphs['validation'], spec, weight)
        history.append({'epoch': epoch, 'train_weighted_loss': total / len(loader),
                        'validation_weighted_loss': weighted, 'validation_unweighted_loss': common})
        if common < best:
            best, best_epoch, best_state = common, epoch, deepcopy(model.state_dict())
        if epoch % 10 == 0:
            print(f'{arm} seed {seed}, epoch {epoch}: validation BCE {common:.4f}', flush=True)
    info.update({'epoch_order_sha256': orders, 'epochs': {POLICIES[0]: best_epoch, 'last': spec['epochs']}})
    return {POLICIES[0]: best_state, 'last': deepcopy(model.state_dict())}, pd.DataFrame(history), info


def inputs(root, config, reader):
    require(config['seeds'] == [42, 43, 44] and config['arms'] == ['concat', 'gated']
            and config['threshold'] == .5 and config['gate'] == '2_sigmoid_linear_99_to_3_zero_initialized_times_availability'
            and config['scope'] == 'train_only_validation_only_no_test_no_threshold_tuning_no_promotion', 'Protocol changed')
    path = root / config['source_manifest']
    reader.allow(path, config['source_manifest_sha256'])
    source = reader.read(path)
    require(source['complete'] and source['schema_version'] == 'graph_scaling_v1', 'Invalid reference')
    scaling_config = reader.registered(path.parent, source, 'provenance/resolved_config.json')
    loaded = load_scaled(root, scaling_config, reader)
    _, _, _, data, mapping, _, graphs, training, fitted, coverage = loaded
    require(fitted == source['fitted_graph_scale'], 'Scale changed')
    for pop in data.values():
        require(pop['masks'].has_profile.eq(1).all(), 'Missing profile requires a separately approved policy')
    return path, source, data, mapping, graphs, training['training'], fitted, coverage


def gates(model, pop, graph, spec):
    model.eval()
    rows = []
    with torch.no_grad():
        h = model.graph_representation(graph)
        for start in range(0, len(pop['y']), spec['batch_size']):
            stop = min(start + spec['batch_size'], len(pop['y']))
            x = torch.tensor(pop['x'][start:stop], dtype=torch.float32)
            _, weights = model.components(x, torch.arange(start, stop), graph, h)
            rows.extend(weights.double().tolist())
    f = pd.DataFrame(rows, columns=['other_gate', 'temporal_gate', 'graph_gate'], index=pop['users'].index)
    f['has_graph'] = graph['available'].numpy().astype(int)
    f['has_temporal'] = (pop['x'][:, 40:47] != 0).any(axis=1).astype(int)
    require(f.loc[f.has_graph.eq(0), 'graph_gate'].eq(0).all() and f.loc[f.has_temporal.eq(0), 'temporal_gate'].eq(0).all(), 'Missing branch active')
    return f.reset_index()


def summaries(table):
    measured = ['bot_f1', 'bot_precision', 'bot_recall', 'roc_auc', 'average_precision', 'brier']
    summary = table.groupby(['arm', 'policy', 'cohort'])[measured].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    effects = []
    for row in table[table.arm.eq('gated')].itertuples():
        ref = table[table.arm.eq('concat') & table.seed.eq(row.seed) & table.policy.eq(row.policy) & table.cohort.eq(row.cohort)]
        require(len(ref) == 1, 'Missing paired control')
        effects.append({'seed': row.seed, 'policy': row.policy, 'cohort': row.cohort,
                        **{'delta_' + m: float(getattr(row, m) - ref.iloc[0][m]) for m in measured}})
    return summary.reset_index(), pd.DataFrame(effects)


def run(root, config_path='configs/gated_fusion_v1.json'):
    root = Path(root).resolve()
    reader = Inputs(root)
    reader.allow(root / config_path, digest(root / config_path))
    config = reader.read(root / config_path)
    before = preserved_hashes(root, config['output_root'])
    source_path, source, data, mapping, graphs, spec, fitted, coverage = inputs(root, config, reader)
    output = root / config['output_root'] / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid4().hex[:8])
    for name in ['models', 'histories', 'predictions', 'gates', 'provenance', 'figures']:
        (output / name).mkdir(parents=True, exist_ok=False)
    write_json(output / 'provenance/resolved_config.json', config)
    write_json(output / 'provenance/training_config.json', spec)
    write_json(output / 'provenance/feature_mapping.json', mapping)
    write_json(output / 'provenance/fitted_graph_scale.json', fitted)
    coverage.to_csv(output / 'graph_coverage.csv', index=False)
    weight = weights_from_training(data['train']['y'])['full_train_ratio']
    rows, runs = [], {}
    for seed in config['seeds']:
        for arm in config['arms']:
            print(f'Training {arm}, seed {seed}', flush=True)
            states, history, info = train_one(data, graphs, mapping, spec, arm, seed, weight)
            require(info['parameter_count'] == config['expected_parameters'][arm], 'Capacity changed')
            reference = source['runs'][f'scaled_gcn_seed{seed}']
            require(info['epoch_order_sha256'] == reference['epoch_order_sha256'], 'Frozen batch pairing changed')
            if arm == 'gated':
                require(info['initial_common_sha256'] == runs[f'concat_seed{seed}']['initial_common_sha256'], 'Common initial weights differ')
            name = f'{arm}_seed{seed}'
            history.to_csv(output / f'histories/{name}.csv', index=False)
            info['checkpoints'] = {}
            for policy, state in states.items():
                model = initialized_model(arm, mapping, seed)
                model.load_state_dict(state)
                epoch = info['epochs'][policy]
                p, weighted, common = evaluate(model, data['validation'], graphs['validation'], spec, weight)
                if arm == 'concat':
                    ref = torch.load(registered_path(reader, source_path.parent, source, reference['checkpoints'][policy]), map_location='cpu', weights_only=True)
                    require(epoch == reference['epochs'][policy], 'Control selected epoch differs')
                    for k, v in ref['state_dict'].items():
                        torch.testing.assert_close(state['model.' + k], v, atol=0, rtol=0)
                f = prediction_registry(data['validation'], p, arm=arm, seed=seed, policy=policy, epoch=epoch)
                f.to_csv(output / f'predictions/{name}_{policy}.csv', index=False)
                rows.extend(scored_rows(f, info['parameter_count'], weighted, common))
                for split, pop in data.items():
                    gates(model, pop, graphs[split], spec).to_csv(output / f'gates/{name}_{policy}_{split}.csv', index=False)
                relative = f'models/{name}_{policy}.pt'
                torch.save({'state_dict': state, 'arm': arm, 'seed': seed, 'policy': policy, 'epoch': epoch,
                            'mapping': mapping, 'scale': fitted, 'threshold': .5, 'pos_weight': weight,
                            'training_user_keys': info['training_user_keys'], 'source_manifest_sha256': config['source_manifest_sha256']}, output / relative)
                info['checkpoints'][policy] = relative
            runs[name] = info
    table = pd.DataFrame(rows)
    summary, effects = summaries(table)
    for name, f in [('validation_metrics', table), ('seed_summary', summary), ('paired_effects', effects)]:
        f.to_csv(output / (name + '.csv'), index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, cohort in zip(axes, ['all', 'graph_available', 'graph_unavailable']):
        for seed in config['seeds']:
            f = table[table.seed.eq(seed) & table.policy.eq(POLICIES[0]) & table.cohort.eq(cohort)].set_index('arm').loc[config['arms']]
            ax.plot([0, 1], f.bot_f1, marker='o', label=f'seed {seed}')
        ax.set(xticks=[0, 1], xticklabels=['Concatenation', 'Gated'], title=cohort, ylabel='Bot F1', ylim=(0, 1)); ax.legend()
    save_figure(fig, output / 'figures/paired_f1.png')
    write_json(output / 'training_metadata.json', runs)
    write_json(output / 'provenance/parsed_inputs.json', reader.accessed)
    write_json(output / 'provenance/protected_files.json', before)
    for file in ['src/gated_fusion.py', 'src/gcn_integration.py', 'src/graph_scaling.py', 'src/temporal_statistics.py', 'scripts/run_gated_fusion.py']:
        (output / 'provenance' / Path(file).name).write_bytes((root / file).read_bytes())
    write_json(output / 'provenance/environment.json', {'python': platform.python_version(), 'torch': str(torch.__version__),
        'numpy': np.__version__, 'pandas': pd.__version__, 'threads': torch.get_num_threads(), 'test_inputs_parsed': False,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True),
        'assignment_projection': ['user_key', 'split'], 'threshold_tuned': False})
    require(before == preserved_hashes(root, config['output_root']), 'Protected files changed')
    write_json(output / 'manifest.json', {'complete': True, 'schema_version': config['schema_version'], 'promoted_model': None, 'runs': runs,
        'source_manifest_sha256': config['source_manifest_sha256'], 'protected_files_verified': len(before),
        'artifacts': {p.relative_to(output).as_posix(): {'sha256': digest(p)} for p in sorted(output.rglob('*')) if p.is_file()}})
    return output


def verify(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest = json.loads((output / 'manifest.json').read_text())
    require(manifest['complete'] and manifest['promoted_model'] is None, 'Incomplete run')
    for rel, meta in manifest['artifacts'].items():
        require(digest(output / rel) == meta['sha256'], 'Artifact changed: ' + rel)
    config = json.loads((output / 'provenance/resolved_config.json').read_text())
    reader = Inputs(root)
    source_path, source, data, mapping, graphs, spec, fitted, _ = inputs(root, config, reader)
    rows, count = [], 0
    weight = weights_from_training(data['train']['y'])['full_train_ratio']
    for name, info in manifest['runs'].items():
        require(info['parameter_count'] == config['expected_parameters'][info['arm']], 'Capacity differs')
        require(info['initial_common_sha256'] == manifest['runs'][f"concat_seed{info['seed']}"]['initial_common_sha256'], 'Initial pairing differs')
        history = pd.read_csv(output / f'histories/{name}.csv')
        require(history.epoch.tolist() == list(range(1, 51)), 'History incomplete')
        require(info['epoch_order_sha256'] == source['runs'][f"scaled_gcn_seed{info['seed']}"]['epoch_order_sha256'], 'Batch order differs')
        require(info['training_user_keys'] == data['train']['users'].index.tolist(), 'Training keys differ')
        for policy, rel in info['checkpoints'].items():
            payload = torch.load(output / rel, map_location='cpu', weights_only=True)
            epoch = 50 if policy == 'last' else int(history.loc[history.validation_unweighted_loss.idxmin(), 'epoch'])
            require(payload['epoch'] == info['epochs'][policy] == epoch and payload['mapping'] == mapping and payload['scale'] == fitted
                    and payload['threshold'] == .5 and payload['pos_weight'] == weight and payload['training_user_keys'] == info['training_user_keys']
                    and payload['source_manifest_sha256'] == config['source_manifest_sha256'], 'Checkpoint provenance differs')
            model = initialized_model(info['arm'], mapping, info['seed'])
            model.load_state_dict(payload['state_dict'], strict=True)
            if info['arm'] == 'concat':
                reference = source['runs'][f"scaled_gcn_seed{info['seed']}"]
                frozen = torch.load(registered_path(reader, source_path.parent, source, reference['checkpoints'][policy]), map_location='cpu', weights_only=True)
                require(epoch == reference['epochs'][policy], 'Frozen selection differs')
                for key, value in frozen['state_dict'].items():
                    torch.testing.assert_close(payload['state_dict']['model.' + key], value, atol=0, rtol=0)
            p, weighted, common = evaluate(model, data['validation'], graphs['validation'], spec, weight)
            np.testing.assert_allclose([weighted, common], history.set_index('epoch').loc[epoch, ['validation_weighted_loss', 'validation_unweighted_loss']].to_numpy(dtype=float), atol=1e-7, rtol=1e-7)
            f = prediction_registry(data['validation'], p, arm=info['arm'], seed=info['seed'], policy=policy, epoch=epoch)
            pd.testing.assert_frame_equal(f, pd.read_csv(output / f'predictions/{name}_{policy}.csv', dtype={'user_key': str}), check_dtype=False, atol=1e-7, rtol=1e-6)
            rows.extend(scored_rows(f, info['parameter_count'], weighted, common))
            for split, pop in data.items():
                pd.testing.assert_frame_equal(gates(model, pop, graphs[split], spec), pd.read_csv(output / f'gates/{name}_{policy}_{split}.csv', dtype={'user_key': str}, float_precision='round_trip'), check_dtype=False, atol=1e-7, rtol=1e-6)
            count += 1
    table = pd.read_csv(output / 'validation_metrics.csv')
    keys = ['arm', 'seed', 'policy', 'cohort']
    pd.testing.assert_frame_equal(pd.DataFrame(rows).sort_values(keys).reset_index(drop=True), table.sort_values(keys).reset_index(drop=True), check_dtype=False, atol=1e-7, rtol=1e-6)
    for name, frame in zip(['seed_summary', 'paired_effects'], summaries(table)):
        pd.testing.assert_frame_equal(frame, pd.read_csv(output / (name + '.csv')), check_dtype=False, atol=1e-10, rtol=1e-10)
    parsed = json.loads((output / 'provenance/parsed_inputs.json').read_text())
    require(not any('test' in Path(p).stem.lower() for p in parsed) and all(digest(root / p) == sha for p, sha in parsed.items()), 'Inputs changed/test parsed')
    require(count == 12, 'Missing checkpoints')
    return {'checkpoints_reloaded': count, 'keyed_validation_files': count, 'keyed_gate_files': 2 * count, 'test_inputs_parsed': 0}
