# Controlled cached XLM-R aggregation experiment

## Approved design (recorded before training)

T0 is the arithmetic mean of each user's prepared cached tweet vectors. T1 is
their coordinate-wise maximum. Same rows, canonical `user_key` order, 768 output
dimensions, no embedding normalization. No new LM inference, tokenizer, pooling
inside the LM, encoder download, fine-tuning, graph input, gating, contrastive loss,
description substitution or pseudo-labeling. The negative gated experiment remains
unpromoted. All earlier artifacts and frozen splits remain unchanged.

Both arms use the same 16,897-parameter temporal-plus-text model: existing 33→48
profile/behavioral encoder (BatchNorm/ReLU/dropout), existing 14→16 temporal encoder
(seven values plus observation masks), 768→16 text projection (ReLU/dropout), then
concatenation80→32→1 classifier (ReLU/dropout). Dropout .3. All classifier/encoder
weights are trained from scratch, paired exactly by seed. The cached LM is frozen.
The temporal-only reference (4,081 parameters) is context, not a capacity-matched
ablation. T0 versus T1 changes only user aggregation.

Fixed seeds42/43/44; 50 epochs; batch32; AdamW lr.001, weight decay.0001; CPU one
thread; BCE positive weight540/131 from training users. All671 training and144
validation users retained. Earliest minimum user-mean unweighted validation BCE
selects checkpoints; save final checkpoints too. Threshold .5, no tuning or test
parsing. The training loop is copied from the temporal reference with only its
model factory changed. Record and reconstruct initial state and batch-order hashes.

Validate exact T0 cache reproduction; same tweet membership/order for both arms;
single-tweet T0/T1 equality; explicit `has_tweet_text`; missing vectors saved as
unavailable/NaN and masked before the projection. A projection bias must never
become an observed missing-text representation. Existing temporal masks remain.

Coverage: training12,976 tweets/671 users, validation2,618 tweets/144 users;
single-tweet users152/28. Every current development user has tweets; synthetic
masking tests check software behavior, not real missingness performance. The
18,331 true-unlabeled users have no prepared tweets and are not used here.

The cache manifest pins arrays and ordered text but lacks the original encoder
revision and full row-to-vector generation provenance. Current equality checks
cannot recover that history. Preserve this limitation in each new run manifest's
registered provenance. No embeddings are regenerated in this phase.

Report F1, precision, recall, AUC, AP, Brier, confusion counts, seed variation,
selected/final differences, learning curves and keyed vector L2 norms. Norms are
diagnostics only; do not add normalization after observing results. Compare
1, 2–10, 11–50 tweet cohorts descriptively; empty and single-class ranking metrics
are explicitly undefined. The primary selected-checkpoint T1−T0 comparison is
called consistently favorable only if F1 improves in all three seeds and mean AP
does not decrease. This is not a significance test or promotion rule.

## Reproduction

```powershell
.\env\Scripts\python.exe -X utf8 -m unittest tests.test_text_aggregation tests.test_temporal_statistics -v
.\env\Scripts\python.exe -X utf8 scripts/run_text_aggregation.py --execute-notebook
.\env\Scripts\python.exe -X utf8 scripts/run_text_aggregation.py --verify-run results/versions/text_aggregation_v1/RUN_DIRECTORY
```

Outputs include four keyed representation matrices, shared tweet-row ledgers,
explicit masks/counts/status/norms, six histories, twelve best/final checkpoints,
keyed predictions, metrics/cohort tables, figures, hashes and code provenance.
Fresh-kernel and independent verification must pass before committing.

## Completed results

Run: `results/versions/text_aggregation_v1/20260923T050904839674Z_311fdde2`.
All six classifier runs completed. No language model was loaded. Selected epochs
were T0:12/13/41 and T1:12/27/41 for seeds42/43/44 respectively.

### Selected checkpoints: identical 144-user validation population

Three-seed mean; F1 uncertainty is sample SD across seeds, not population uncertainty.

| Model | Bot F1 | Precision | Recall | ROC-AUC | AP | Brier |
|---|---:|---:|---:|---:|---:|---:|
| T0 mean | 0.473 ± 0.025 | 0.380 | 0.631 | 0.757 | 0.536 | 0.182 |
| T1 maximum | 0.483 ± 0.029 | 0.393 | 0.631 | 0.752 | 0.531 | 0.178 |
| Frozen temporal reference (context) | 0.479 ± 0.013 | 0.378 | 0.655 | 0.768 | 0.553 | — |

The temporal reference is copied from its pinned validation artifact; adding text
also changes capacity and initialization relative to that reference. It is not the
primary controlled comparison. T1's small mean F1 advantage does not establish
superiority over the temporal model, whose mean AUC/AP remain higher.

| Seed | T0 F1 | T1 F1 | Difference | T0 TN/FP/FN/TP | T1 TN/FP/FN/TP |
|---|---:|---:|---:|---|---|
| 42 | 0.4500 | 0.4500 | 0.0000 | 82/34/10/18 | 82/34/10/18 |
| 43 | 0.4675 | 0.5000 | +0.0325 | 85/31/10/18 | 90/26/10/18 |
| 44 | 0.5000 | 0.5000 | 0.0000 | 93/23/11/17 | 93/23/11/17 |

Mean paired F1 difference is +0.0108, entirely from seed43. Mean AP falls by
0.0048 and AUC by 0.0046. Mean Brier improves by 0.0041. **The prespecified
consistent-improvement criterion fails:** two F1 ties and lower mean AP.
This is mixed evidence, not a promoted replacement.

At epoch50, mean F1 is T0:0.468 versus T1:0.468 (unrounded0.468010/0.467508),
with AP0.520 versus0.510. The selected-epoch gain does not persist as an
end-of-training gain. Seed43 selected different epochs under the fixed rule;
the comparison therefore includes the aggregation's effect on checkpoint selection.
No epoch was selected using F1 and no threshold was changed.

### Descriptive tweet-count cohorts

| Validation cohort | Users | Bots | T0 mean F1 | T1 mean F1 | T0 mean AP | T1 mean AP |
|---|---:|---:|---:|---:|---:|---:|
| One tweet | 28 | 1 | 0.217 | 0.206 | 0.292 | 0.208 |
| 2–10 tweets | 55 | 12 | 0.553 | 0.555 | 0.639 | 0.627 |
| 11–50 tweets | 61 | 15 | 0.462 | 0.481 | 0.540 | 0.529 |

The one-tweet cohort contains only one bot; avoid conclusions about aggregation
quality from those unstable metrics. Its T0/T1 representations are identical, but
predictions can differ because the classifiers learn from different representations
of other training users. Equal input vectors do not imply equal learned models.
The missing-text validation cohort is empty and its metrics are explicitly undefined.

### Embedding magnitude

| Split | Rule | Mean L2 norm | Median L2 norm | Min | Max |
|---|---|---:|---:|---:|---:|
| Train | T0 | 18.9073 | 18.9101 | 18.4643 | 19.0861 |
| Train | T1 | 18.8748 | 18.8803 | 18.3327 | 19.0861 |
| Validation | T0 | 18.9050 | 18.9061 | 18.7052 | 19.0352 |
| Validation | T1 | 18.8824 | 18.8839 | 18.5299 | 19.1192 |

Mean per-user T1/T0 norm ratio is0.998280 in training and0.998804 in validation.
There is no large norm inflation here. Coordinate-wise maximum need not increase
L2 norm because signed negative coordinates may move toward zero. These norm
summaries alone do not establish representation diversity or collapse. No
normalization or additional representation transformation was introduced.

### Verification and preservation

- Eight targeted/temporal regression tests passed.
- Fresh-kernel execution and separate-process verification passed.
- All twelve checkpoints reproduced keyed predictions, metrics and selected losses.
- All four representation matrices reproduced; T0 equals the registered user cache.
- Single-tweet equality verified for152 training and28 validation users.
- T0/T1 initial-state hashes, every epoch's user-order hash, training keys, parameter
  counts and class weights match; batch orders reconstructed independently.
- Missing-text/temporal invariance and checkpoint snapshot isolation passed tests.
- Sixty-three hashed artifacts saved. At completion1,158 protected files matched;
  subsequent executed-notebook output is the sole expected change (1,157 unchanged).
- No test data parsed, graph inputs used, LM inference performed or model promoted.
- All three generated figures were visually inspected and the notebook has no errors.

## Next approval boundary

Keep both arms as reproducible experiments and retain T0 as the current cached-text
control. Do not treat the maximum arm as a consistent improvement. Before further
classifier changes, propose a training-only, pinned-XLM-R extraction provenance
and runtime pilot: compare fresh mean-pooled vectors against the legacy cache,
record numerical agreement and estimate full extraction cost. This requires the
separate approval already specified by the user. No fresh extraction, token-pooling
experiment, ParsBERT download, or encoder fine-tuning was performed here.
