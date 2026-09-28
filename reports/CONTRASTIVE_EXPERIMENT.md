# contrastive_v1: controlled supervised contrastive experiment

## Outcome

Completed 2026-09-28. **Negative controlled development experiment; do not promote C1.**
Selected-checkpoint Bot F1 decreased in all three paired seeds, and mean average
precision decreased. All protocol and independent reproduction checks passed.
Retain the approved P1 classification-only reference. No hyperparameter, threshold,
batch, checkpoint policy, or representation was changed in response to the results.
No test data were accessed. The modality-ablation study was not started.

Run: `results/versions/contrastive_v1/20260928T051235934399Z_41464a56`.
Manifest SHA256:
`dd5c88eb406c7ddac2ee133350333a851aa59332f6af01c72d2ba7cbda377ce7`.

## Approved protocol and implementation

Both arms use the original model constructor and forward method, the same frozen
P1 XLM-R user vectors, Phase-2 transformations/feature order, and temporal masks.
The 80-dimensional concatenation contains 48 profile/behavioral, 16 temporal, and
16 projected text coordinates. It is captured immediately before the existing
80->32->1 classifier without rewriting the primary forward path. Existing ReLU,
BatchNorm and dropout 0.3 remain unchanged.

- **C0:** weighted classification BCE only; 16,897 parameters.
- **C1:** same inference model plus 80->32->ReLU->16 head and L2 normalization
  (epsilon 1e-12). The head adds 3,120 parameters: 20,017 training parameters,
  still 16,897 at inference. The head is absent from classifier inference.
- Two views use only existing branch dropout. The second pass stops at the fused
  representation; classification uses only the primary pass.
- Each anchor's positives are every other same-label view, including its paired
  view. Negatives are opposite-label views. The anchor is excluded from its own
  denominator. No memory bank, prototypes, unlabeled identities, augmentation,
  modality dropout, graph, gating, fine-tuning, or new language model.
- Temperature 0.1. Mean positive log probability per anchor, followed by an equal
  mean of human-anchor and bot-anchor means. Denominator contains all other views.
  This corrects anchor-count dominance, not every denominator-imbalance effect.
- Joint loss = classification BCE + **0.05 * contrastive loss**. No schedule or sweep.
- Positive BCE weight is 540/131, derived only from training labels.
- 671 training users and 144 validation users (28 validation bots); seeds 42/43/44;
  50 epochs; batch 32, final batch 31; AdamW lr 0.001, weight decay 0.0001; one CPU
  thread. No dropped users or changed sampling. Each run has 1,050 updates.
- Select the earliest minimum user-mean unweighted validation BCE, never joint
  loss or F1. Threshold 0.5. Selected and final checkpoints are saved separately.
- Validation is eval/no-grad, contains no contrastive pairing, and has explicit
  before/after state-hash checks. It never contributes representation-training gradients.

### RNG and BatchNorm isolation

The original constructor's RNG consumption is preserved. Head initialization uses
`10000 + seed` inside `torch.random.fork_rng`. Auxiliary dropout uses
`20000 + 100000*seed + zero_based_step` in its own forked RNG context. Primary RNG
hashes before/after every training forward match C0/C1 for all 3,150 paired updates.

During the auxiliary pass, existing BatchNorm is still in training mode but temporarily
has `track_running_stats=False`: it uses the current minibatch statistics without
updating running mean, running variance or `num_batches_tracked`. Flags are restored
in a finally block. Every auxiliary call asserts exact equality of all model buffers
before/after and verifies that the primary RNG state was restored. Both views receive
gradients; the second view is not detached. No doubled BatchNorm batch is used.

### Single-class and one-bot batches

| Seed | Training batches | Single-class batches | One-bot batches |
|---|---:|---:|---:|
| 42 | 1,050 | 0 | 7 |
| 43 | 1,050 | 1 | 5 |
| 44 | 1,050 | 3 | 6 |

The four single-class batches have exactly zero contrastive loss; their usual
classification update is retained. No user is removed. In one-bot batches, the
other view of that bot is a valid positive. Per-batch pair counts are saved.

## Validation results

Same 144 users throughout. Values are mean +/- sample SD across seeds, not
confidence intervals or independent validation-population replicates.

### Selected checkpoints: primary comparison

| Metric | C0: classification only | C1: plus contrastive |
|---|---:|---:|
| Bot F1 | 0.494172 +/- 0.028423 | 0.482692 +/- 0.025619 |
| Precision | 0.402980 +/- 0.038937 | 0.387319 +/- 0.029774 |
| Recall | 0.642857 +/- 0.035714 | 0.642857 +/- 0.035714 |
| ROC-AUC | 0.756671 +/- 0.006620 | 0.755747 +/- 0.005852 |
| Average precision | 0.548662 +/- 0.016240 | 0.540540 +/- 0.003452 |
| Brier score | 0.177880 +/- 0.015646 | 0.180416 +/- 0.012328 |
| Unweighted validation BCE | 0.541210 +/- 0.022535 | 0.543888 +/- 0.018754 |

| Seed | C0 epoch | C1 epoch | C0 F1 | C1 F1 | Delta F1 | Delta AP | Delta AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 12 | 12 | 0.461538 | 0.455696 | -0.005842 | +0.006050 | +0.000616 |
| 43 | 27 | 27 | 0.513514 | 0.506667 | -0.006847 | -0.010480 | -0.006773 |
| 44 | 41 | 38 | 0.507463 | 0.485714 | -0.021748 | -0.019935 | +0.003387 |

Mean delta F1 is **-0.011479**, and mean delta AP is **-0.008122**. Both requirements
of the approved improvement criterion fail. C1 predicts one additional false
positive in seeds 42 and 43 and three additional false positives in seed 44,
without changing the number of detected bots at selected checkpoints.

### Selected and final confusion matrices

Entries are `[[TN, FP], [FN, TP]]`, true human/bot rows and predicted human/bot columns.

| Seed | C0 selected | C1 selected | C0 final | C1 final |
|---|---|---|---|---|
| 42 | [[84,32],[10,18]] | [[83,33],[10,18]] | [[73,43],[12,16]] | [[79,37],[11,17]] |
| 43 | [[89,27],[9,19]] | [[88,28],[9,19]] | [[90,26],[10,18]] | [[88,28],[11,17]] |
| 44 | [[94,22],[11,17]] | [[91,25],[11,17]] | [[95,21],[11,17]] | [[96,20],[11,17]] |

### Final epoch: secondary comparison

| Metric | C0 | C1 |
|---|---:|---:|
| Bot F1, mean +/- SD | 0.460989 +/- 0.081045 | 0.467821 +/- 0.054251 |
| Precision | 0.375882 | 0.384017 |
| Recall | 0.607143 | 0.607143 |
| ROC-AUC | 0.737890 | 0.736145 |
| Average precision | 0.520536 | 0.528337 |
| Brier | 0.192516 | 0.191309 |
| Unweighted BCE | 0.606176 | 0.607356 |

Final-epoch paired F1 changes: +0.046818, -0.034247, +0.007925. This is mixed evidence,
not a reason to switch the prespecified checkpoint rule. Both arms' final mean F1
is below their selected mean F1. Histories retain all losses; training weighted BCE
and validation unweighted BCE are different objectives and their raw gap is not a
calibrated generalization estimate. Cohort-specific metrics are saved for one tweet,
2-10 tweets, 11-50 tweets, and missing text. They do not govern the decision.

## Representation diagnostics

All diagnostic inference is dropout-disabled. Training and validation summaries
are separate; labels are used for descriptive diagnostics, not new representation
training or checkpoint selection. Arrays are keyed by the corresponding population
CSV. Fused vectors are 80-dimensional; projected vectors are 16-dimensional.

Effective rank is exp(entropy of normalized eigenvalues of the centered population
covariance). Coordinate variances use denominator N. Every coordinate's mean,
variance, SD, zero rate and class-specific variance is exported, as is the full
covariance spectrum. A small effective rank alone does not prove global collapse.

### Selected fused representations: means across seeds

| Population / diagnostic | C0 | C1 |
|---|---:|---:|
| Training effective rank | 22.8578 | 22.8311 |
| Training mean coordinate variance | 0.241125 | 0.236806 |
| Training inactive-coordinate rate | 0.13333 | 0.12917 |
| Training human within-class scatter | 18.9164 | 18.5658 |
| Training bot within-class scatter | 17.4912 | 17.1246 |
| Training centroid distance | 2.0361 | 2.0492 |
| Training squared separation / balanced scatter | 0.2280 | 0.2354 |
| Validation effective rank | 20.7295 | 20.7194 |
| Validation mean coordinate variance | 0.235478 | 0.231781 |
| Validation inactive-coordinate rate | 0.14167 | 0.14167 |
| Validation human within-class scatter | 17.8101 | 17.4641 |
| Validation bot within-class scatter | 20.8022 | 20.7105 |
| Validation centroid distance | 1.6826 | 1.6852 |
| Validation squared separation / balanced scatter | 0.14794 | 0.14960 |

Zero-vector rates are zero in both arms, all populations and all recorded stages.
Selected fused effective rank and variance remain substantial; there is no evidence
of global fused-vector collapse. Inactive-coordinate rates reflect ReLU coordinates
that are zero for every user in that population, not missing users or modalities.
The small fused-space differences do not translate into better validation metrics.

### Treatment projection: representation learning did occur

| Normalized projection diagnostic | Training selected | Validation selected | Training final | Validation final |
|---|---:|---:|---:|---:|
| Effective rank | 4.6529 | 4.4157 | 4.3853 | 4.5878 |
| Total variance | 0.08918 | 0.08983 | 0.13856 | 0.13130 |
| Human within-class scatter | 0.07341 | 0.08137 | 0.09440 | 0.11441 |
| Bot within-class scatter | 0.03028 | 0.08083 | 0.03510 | 0.14335 |
| Centroid distance | 0.38057 | 0.23038 | 0.59523 | 0.26735 |
| Squared separation / balanced scatter | 2.87969 | 0.65575 | 5.49263 | 0.56828 |
| Between-class cosine | 0.87118 | 0.89156 | 0.75785 | 0.83516 |

At initialization, normalized-projection effective rank averages 7.4651 on training
and 6.9431 on validation. The head becomes more concentrated, while training class
separation grows much more strongly than validation separation. Nonzero variance
and nonzero-vector counts rule out complete collapse in these recorded summaries,
but do not establish that the projected geometry is useful for classification.
The auxiliary head is discarded at inference. Its changes must not be presented
as proof that the classifier's fused representation improved.

### Same-user dropout views: training-only diagnostic

Use the first 32 canonical training users, copied models and isolated RNG, never
validation users. Means below combine the fixed diagnostic users across seeds.

| Space / arm | Initial cosine | Selected cosine | Final cosine |
|---|---:|---:|---:|
| Fused C0 | 0.69463 | 0.69331 | 0.70265 |
| Fused C1 | 0.69463 | 0.69534 | 0.70087 |
| Normalized projection C1 | 0.82439 | 0.97120 | 0.96010 |

The head learns view consistency, but fused-space view similarity changes little.
High projected similarity alone is not evidence of useful discrimination: different
users also occupy a relatively narrow angular region. Per-user cosine/L2 values
are saved for all three diagnostic stages.

## Loss behavior and effective contribution

| Seed | Raw contrastive loss, epoch 1 -> 50 | Weighted term, epoch 1 -> 50 | Ratio of mean contrastive contribution to mean BCE, epoch 1 -> 50 |
|---|---|---|---|
| 42 | 4.3550 -> 3.8010 | 0.21775 -> 0.19005 | 0.20067 -> 0.34424 |
| 43 | 4.4191 -> 3.8049 | 0.22096 -> 0.19024 | 0.19959 -> 0.31441 |
| 44 | 4.3667 -> 3.7923 | 0.21833 -> 0.18961 | 0.19799 -> 0.39098 |

The raw auxiliary loss decreases. Its relative contribution grows as classification
loss decreases; lambda 0.05 does not mean the contrastive term is 5% of BCE. Across
individual batches the ratio reaches approximately 1.014 in seed 42. These are
loss-magnitude ratios, not gradient-magnitude ratios. Zero-contribution single-class
batches are included in the epoch means. Both the ratio of mean losses and the
mean of per-batch ratios are saved with distinct names. No adjustment was made.

## Equivalence, reproducibility and preservation

- Nine focused/regression tests passed, including an independently coded scalar
  contrastive-loss calculation, both-view gradients, one-bot/single-class handling,
  missingness invariance, dropout isolation, collapse diagnostics and head removal.
- Before training, lambda=0 machinery was tested on one complete training epoch
  (21 updates) for each seed. **Maximum primary-logit difference is exactly zero**;
  full backbone/classifier state hashes, including BatchNorm buffers/counters, and
  primary RNG states match at every update. Declared tolerance: atol=rtol=0.
- C0's selected/final weights exactly equal the historical P1 checkpoints for all
  seeds; all historical classification-loss histories reproduce within 1e-7.
- C0/C1 initialization hashes, primary RNG sequences, every batch key-order hash,
  and all epoch-order hashes match. Auxiliary buffers/RNG are checked every call.
- Fresh-kernel execution completed all three code cells with no errors.
- Independent process verification passed: 12 selected/final checkpoints plus
  six initial snapshots reloaded; 12 keyed predictions, losses/metrics, 72
  representation arrays, coordinate/spectrum diagnostics and training view pairs
  reproduced. Lambda=0 regressions passed again in that independent process.
- All 306 artifacts passed SHA256 verification. Parsed input hashes remain unchanged.
  During execution all 66 protected source/config/notebook files were unchanged;
  afterward only the new executed notebook differs due to saved outputs. No test
  data files were opened, including for preservation hashing.
- Existing P1, legacy and other experiment artifacts were not overwritten. The
  unrelated pre-existing Notebook 06 edit remains untouched and outside this commit.

Training all six runs took **63.35 seconds**; training/export/diagnostics/figures
took **70.21 seconds** (excludes initial loading, preflight regression and independent
verification). Peak sampled process RSS was **519,454,720 bytes**, approximately
495.39 MiB. Complete run storage was 11,496,460 bytes. No XLM-R inference was run.

## Interpretation and limits

This experiment answers a narrow question about the approved supervised contrastive
regularizer at temperature 0.1 and lambda 0.05 with dropout views. It does not show
that all contrastive methods are ineffective. It is not self-supervised BotSSCL.
The auxiliary head adds training parameters but no inference capacity. Annotation
uncertainty, same-class behavioral diversity, class imbalance in the denominator,
only three seeds, 28 validation bots, and repeated validation use limit conclusions.
The current development population has no naturally missing text/temporal modality,
so functional masking tests do not establish missing-modality predictive robustness.

Retain C1 as a negative controlled experiment. Retain P1/C0 as the fixed development
reference. Do not promote a final checkpoint or tune temperature/lambda after these
results. The next modality study requires separate approval and is not implemented.

## Reproduction and files

- `configs/contrastive_v1.json`: pinned source and approved settings.
- `src/contrastive_core.py`: isolated auxiliary path, loss, regression and diagnostics.
- `src/contrastive.py`: paired training, export, fixed decision and verification.
- `analysis/improvements/10_supervised_contrastive.ipynb`: executed experiment.
- `scripts/run_contrastive.py`: fresh kernel, completed-run replay and independent verification.
- `tests/test_contrastive.py`: focused numerical and protocol checks.

```powershell
.\env\Scripts\python.exe -B -X utf8 -m unittest tests.test_contrastive -v
.\env\Scripts\python.exe -B -X utf8 scripts/run_contrastive.py --execute-notebook
.\env\Scripts\python.exe -B -X utf8 scripts/run_contrastive.py --verify-run results/versions/contrastive_v1/20260928T051235934399Z_41464a56
```

Use `--execute-notebook --resume-run results/versions/contrastive_v1/RUN_DIRECTORY`
to verify/replay a completed run without retraining. The default notebook execution
creates a new version. All old versions remain intact.
