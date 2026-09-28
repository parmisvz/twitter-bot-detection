"""Small supervised-contrastive auxiliary path; the primary model is unchanged."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.baselines import seed_all
from src.text_aggregation import TemporalTextModel
from src.performance_diagnosis import require
from src.checkpoint_weight_experiment import state_hash


def backbone(mapping):
    # Preserve even the historical constructor's RNG consumption.
    return TemporalTextModel('T0_mean', mapping, .3)


def head(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(10000 + seed)
        return nn.Sequential(nn.Linear(80, 32), nn.ReLU(), nn.Linear(32, 16))


def rng_hash():
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


def primary(model, x):
    """Capture the classifier input without rewriting or reordering forward()."""
    captured = []
    handle = model.base.classifier.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        logits = model(x)
    finally:
        handle.remove()
    require(len(captured) == 1 and captured[0].shape[1] == 80, 'Fused interface changed')
    return logits, captured[0]


def fused(model, x):
    """Exact original branch order: text, profile/behavior, temporal."""
    other, temporal, mask = model.base.split_inputs(x[:, :47])
    available = x[:, 815:816]
    require(((available == 0) | (available == 1)).all().item(), 'Invalid text mask')
    values = torch.where(available.bool(), x[:, 47:815], torch.zeros_like(x[:, 47:815]))
    text = model.text_projection(values) * available
    return torch.cat([model.base.other_encoder(other), model.base.temporal_representation(temporal, mask), text], 1)


@contextmanager
def no_batchnorm_updates(model):
    modules = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    require(all(m.training for m in modules), 'Auxiliary BN requires training mode')
    flags = [m.track_running_stats for m in modules]
    before = {n: t.clone() for n, t in model.named_buffers()}
    try:
        for m in modules:
            m.track_running_stats = False
        yield
    finally:
        for m, flag in zip(modules, flags):
            m.track_running_stats = flag
        for n, value in model.named_buffers():
            torch.testing.assert_close(value, before[n], atol=0, rtol=0)


def auxiliary(model, x, seed, step):
    before = rng_hash()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20000 + 100000 * seed + step)
        with no_batchnorm_updates(model):
            h = fused(model, x)
    require(rng_hash() == before, 'Auxiliary forward consumed primary RNG')
    return h


def pair_masks(labels):
    require(labels.ndim == 1 and ((labels == 0) | (labels == 1)).all().item(), 'Invalid training labels')
    y = labels.repeat(2)
    other = ~torch.eye(len(y), dtype=torch.bool, device=y.device)
    positives = (y[:, None] == y[None, :]) & other
    negatives = (y[:, None] != y[None, :]) & other
    require(positives.any(1).all().item(), 'Missing paired view')
    return y, other, positives, negatives


def supcon(first, second, labels, temperature=.1):
    """Mean of positive log probabilities, then equal class-anchor means."""
    require(first.shape == second.shape and first.shape == (len(labels), 16), 'Projection shape differs')
    y, other, pos, neg = pair_masks(labels)
    if not neg.any().item():
        return (first.sum() + second.sum()) * 0
    z = torch.cat([first, second], 0)
    logits = z @ z.T / temperature
    logdenom = torch.logsumexp(logits.masked_fill(~other, -torch.inf), dim=1)
    logprob = logits - logdenom[:, None]
    per_anchor = -(logprob * pos).sum(1) / pos.sum(1)
    return .5 * (per_anchor[y == 0].mean() + per_anchor[y == 1].mean())


def projected(projector, h):
    raw = projector(h)
    return raw, nn.functional.normalize(raw, p=2, dim=1, eps=1e-12)


def equivalence(mapping, batches, seed, weight):
    """Paired multi-step lambda=0 regression on training inputs only; exact CPU equality."""
    trajectories = []
    for treatment in [False, True]:
        seed_all(seed, 1)
        model = backbone(mapping).train()
        initial_rng = rng_hash()
        projector = head(seed) if treatment else None
        require(rng_hash() == initial_rng, 'Head initialization consumed primary RNG')
        parameters = list(model.parameters()) + (list(projector.parameters()) if treatment else [])
        optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.0001)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], dtype=torch.float32))
        rows = []
        for step, (x, y) in enumerate(batches):
            optimizer.zero_grad()
            before_rng = rng_hash()
            logits, h = primary(model, x)
            after_rng = rng_hash()
            loss = criterion(logits, y)
            if treatment:
                h2 = auxiliary(model, x, seed, step)
                _, z1 = projected(projector, h)
                _, z2 = projected(projector, h2)
                loss = loss + 0. * supcon(z1, z2, y.flatten())
            loss.backward()
            optimizer.step()
            rows.append((logits.detach().clone(), state_hash(model.state_dict()), before_rng, after_rng, rng_hash()))
        trajectories.append(rows)
    maximum = 0.
    for control, treatment in zip(*trajectories):
        torch.testing.assert_close(control[0], treatment[0], atol=0, rtol=0)
        maximum = max(maximum, float((control[0] - treatment[0]).abs().max()))
        require(control[1:] == treatment[1:], 'Lambda-zero updates, buffers, or RNG differ')
    return {'seed': seed, 'steps': len(batches), 'atol': 0., 'rtol': 0., 'max_logit_difference': maximum,
            'state_including_bn_exact': True, 'primary_rng_exact': True, 'head_rng_isolated': True}


def representation_stats(values, labels):
    """Float64 population summaries; no fitting or selection based on diagnostics."""
    a = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    require(a.ndim == 2 and np.isfinite(a).all(), 'Nonfinite representation')
    n, d = a.shape
    norms = np.linalg.norm(a, axis=1)
    unit = a / np.maximum(norms[:, None], 1e-12)
    cos = unit @ unit.T
    off = ~np.eye(n, dtype=bool)
    centered = a - a.mean(0)
    variance = (centered ** 2).mean(0)
    eigen = np.maximum(np.linalg.eigvalsh(centered.T @ centered / n), 0)[::-1]
    p = eigen[eigen > 0] / eigen.sum() if eigen.sum() > 1e-20 else np.array([])
    rank = float(np.exp(-(p * np.log(p)).sum())) if len(p) else 0.
    summary = {'n': n, 'dimension': d, 'effective_rank': rank, 'total_variance': float(variance.sum()),
               'mean_coordinate_variance': float(variance.mean()), 'mean_norm': float(norms.mean()),
               'min_norm': float(norms.min()), 'max_norm': float(norms.max()),
               'zero_vector_rate': float((norms <= 1e-12).mean()),
               'inactive_coordinate_rate': float((a == 0).all(0).mean()),
               'near_constant_coordinate_rate': float((np.sqrt(variance) <= 1e-6).mean()),
               'zero_activation_rate': float((a == 0).mean()),
               'off_diagonal_cosine': float(cos[off].mean()) if n > 1 else None}
    centroids, scatter = {}, {}
    for c in [0, 1]:
        keep = labels == c
        same = keep[:, None] & keep[None, :] & off
        if keep.any():
            centroids[c] = a[keep].mean(0)
            scatter[c] = float(((a[keep] - centroids[c]) ** 2).sum(1).mean())
            summary[f'class{c}_within_scatter'] = scatter[c]
            summary[f'class{c}_zero_vector_rate'] = float((norms[keep] <= 1e-12).mean())
            summary[f'class{c}_mean_norm'] = float(norms[keep].mean())
        summary[f'class{c}_same_class_cosine'] = float(cos[same].mean()) if same.any() else None
    if len(centroids) == 2:
        distance = float(np.linalg.norm(centroids[0] - centroids[1]))
        within = .5 * (scatter[0] + scatter[1])
        summary.update(centroid_distance=distance, balanced_within_scatter=within,
                       separation_over_scatter=distance ** 2 / max(within, 1e-12),
                       between_class_cosine=float(cos[labels[:, None] != labels[None, :]].mean()))
    coordinates = pd.DataFrame({'coordinate': np.arange(d), 'mean': a.mean(0), 'variance_ddof0': variance,
                                'std_ddof0': np.sqrt(variance), 'zero_rate': (a == 0).mean(0)})
    for c in [0, 1]:
        coordinates[f'class{c}_variance_ddof0'] = a[labels == c].var(0) if (labels == c).any() else np.nan
    spectrum = pd.DataFrame({'component': np.arange(d), 'covariance_eigenvalue': eigen})
    return summary, coordinates, spectrum


def diagnostic_arrays(model, projector, population):
    model.eval()
    if projector is not None:
        projector.eval()
    before = state_hash(model.state_dict())
    pieces = []
    with torch.no_grad():
        for start in range(0, len(population['x']), 32):
            pieces.append(fused(model, torch.tensor(population['x'][start:start+32], dtype=torch.float32)))
        h = torch.cat(pieces)
        arrays = {'fused': h.numpy()}
        if projector is not None:
            raw, z = projected(projector, h)
            arrays.update(projection_raw=raw.numpy(), projection_normalized=z.numpy())
    require(before == state_hash(model.state_dict()), 'Evaluation updated model state')
    return arrays


def view_diagnostic(model, projector, population, seed):
    # Diagnostic uses a copy and a separate RNG context; it cannot change saved state.
    m = deepcopy(model).train()
    p = deepcopy(projector) if projector is not None else None
    x = torch.tensor(population['x'][:32], dtype=torch.float32)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(9000000 + seed)
        _, h1 = primary(m, x)
        h2 = auxiliary(m, x, seed, 9000000)
        pairs = {'fused': (h1, h2)}
        if p is not None:
            pairs['projection_normalized'] = (projected(p, h1)[1], projected(p, h2)[1])
        rows = []
        for space, (a, b) in pairs.items():
            cosine = nn.functional.cosine_similarity(a.double(), b.double(), dim=1, eps=1e-12).numpy()
            distance = (a.double() - b.double()).norm(dim=1).numpy()
            for key, y, c, l2 in zip(population['users'].index[:32], population['y'][:32], cosine, distance):
                rows.append({'user_key': key, 'label_binary': int(y), 'space': space,
                             'cosine_similarity': float(c), 'l2_distance': float(l2)})
    return pd.DataFrame(rows)
