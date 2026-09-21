# Temporal-statistics encoder experiment

## Approved protocol

This experiment tests whether a dedicated temporal-statistics MLP improves
validation performance compared with nearly parameter-matched controls. It
implements only the approved temporal experiment. It adds no Transformer,
attention, learned fusion gate, GNN representation, contrastive objective,
pseudo-labels, new data, threshold tuning, or missingness augmentation.

New notebook: `analysis/improvements/03_temporal_statistics_encoder.ipynb`.
Configuration: `configs/temporal_statistics_v1.json`.
Output parent: `results/versions/temporal_statistics_v1/`.

### Inputs and feature partition

Use the pinned Phase-2 manifest and its training-fitted transformations. The
canonical 40-feature order is retained; explicit index mappings partition it into
33 other features and seven temporal statistics. The latter, in manifest order:

1. `time_between_tweets`
2. `tweet_frequency`
3. `min_tweets_per_hour`
4. `min_tweets_per_day`
5. `max_tweets_per_hour`
6. `max_tweets_per_day`
7. `max_occurence_of_same_gap`

The remaining 33 comprise 20 profile and 13 behavioral features, in their
original relative order. There is no duplication across branches. Account age,
growth rates and other retained fields may still contain time-related information;
the control is named "without seven statistics", not "without all temporal signal".

All models accept a common interface containing the 40 transformed values and
the seven temporal observation masks. A/B consume the same observed information.
C uses only the 33 retained values and ignores the seven values and masks.
The mapping is recorded in checkpoint payloads and separate provenance.

### Architecture

| Arm | Architecture | Trainable parameters |
| --- | --- | ---: |
| A: flat masked | 47 -> 49 -> 32 -> 1 | 4,083 |
| B: temporal branch | 33 -> 48 plus 14 -> 16; concatenate 64 -> 32 -> 1 | 4,081 |
| C: without seven statistics | 33 -> 59 -> 32 -> 1 | 4,077 |

Each hidden layer uses ReLU and dropout 0.3. A/C use BatchNorm after the first
linear layer; B uses BatchNorm after the 33->48 non-temporal layer only. The
temporal branch has no BatchNorm. The final layer emits one bot logit.

Primary comparison: **B minus A**. This tests the whole branch design with the
same information and almost equal parameter counts, not equal functions or
optimization dynamics. Secondary comparison: **B minus C**, which changes both
information and architecture.

### Missingness

Missing temporal input slots retain their Phase-2 observation masks and are
zeroed for computation, not interpreted as measured values or newly estimated.
The temporal hidden representation is forced to zero when all seven masks are
zero. This is a deterministic availability operation, not learned gated fusion.
Every canonical user remains supported by the non-temporal path. No user is
removed because text, graph, or temporal information is missing.

All 671 training and 144 validation users currently have all seven statistics
observed. Therefore missingness handling is tested functionally with synthetic
unit-test inputs, not by adding artificial records or missingness to training.
These checks cannot demonstrate real-world performance under naturally missing
temporal data. The original formulas/windows of the imported statistics remain
unverified; no complete timeline is assumed.

### Training and evaluation

- Nine runs: A/B/C crossed with seeds 42/43/44, trained from scratch.
- Frozen 671 labeled training users; identical 144 validation users for every arm.
- Weighted BCE with positive weight 540/131, derived from training labels only.
- AdamW: learning rate 0.001, weight decay 0.0001; batch 32; 50 epochs.
- Dedicated training shuffle generator seeded with the run seed; separate
  validation generator seeded with seed+100000. Both are independent of model
  initialization and dropout RNG. Every epoch includes each training user once.
- Actual epoch/batch hashes must match across arms. Different architectures do
  not share initial tensors or dropout trajectories; no such pairing is claimed.
- Primary checkpoint: minimum user-mean unweighted validation BCE, earliest
  epoch on ties. Final epoch is retained as secondary. Independent state copies.
- Fixed decision threshold 0.5; no threshold search.
- Save per-seed Bot F1/precision/recall, ROC-AUC, AP, confusion counts, losses,
  means/sample standard deviations, and paired differences. Standard deviations
  over three seeds are not population confidence intervals or significance tests.
- The same validation set selects and reports checkpoints: these are exploratory
  development results, not unbiased generalization estimates.
- Step-2 full-weight common-criterion MLP results remain a historical reference.
  Its architecture and shuffle schedule differ, so it is not the primary control.

The parser is restricted to allowlisted train/validation artifacts. It does not
parse test labels/predictions or unlabeled records. Prior files, including test
artifacts, may be hashed as opaque bytes to verify preservation. No model or
policy is automatically promoted to the frozen baseline or final test protocol.

## Results (2026-09-21)

Verified run: `results/versions/temporal_statistics_v1/20260921T090006150868Z_61ce9da6`.
The same 144 validation users (28 bots) are evaluated throughout, at threshold
0.5. Primary results use the prespecified minimum unweighted validation BCE
checkpoint. Values are means across seeds 42/43/44; F1 includes sample standard
deviation, not a confidence interval.

| Arm | Bot F1 mean +/- SD | Precision | Recall | ROC-AUC | Average Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: flat masked | 0.4732 +/- 0.0435 | 0.3745 | 0.6429 | 0.7552 | 0.5457 |
| B: temporal branch | 0.4786 +/- 0.0132 | 0.3776 | 0.6548 | 0.7679 | 0.5529 |
| C: without seven statistics | 0.4411 +/- 0.0137 | 0.3359 | 0.6429 | 0.7551 | 0.5521 |

### Primary comparison: B minus A

| Seed | F1 difference | Precision difference | Recall difference | AUC difference | AP difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | -0.0453 | -0.0327 | -0.0714 | +0.0080 | +0.0033 |
| 43 | +0.0483 | +0.0250 | +0.1071 | +0.0243 | +0.0090 |
| 44 | +0.0132 | +0.0170 | 0.0000 | +0.0055 | +0.0093 |

The branch increases mean F1 by only **0.0054**. F1 improves in two seeds and
declines in one. AUC and AP improve in all three seeds, by means of approximately
0.0126 and 0.0072. Its across-seed F1 standard deviation is smaller in these
three runs, but three seeds are insufficient to establish a general stability
advantage. This is a modest development-set signal, not a substantial or
statistically established improvement.

The selected epochs are A: **12/28/16**, B: **12/21/16**, and C: **12/28/28**.
No epoch or threshold was selected using test data.

### Secondary comparison and checkpoint sensitivity

B's selected-checkpoint F1 exceeds C's by 0.0390, 0.0193 and 0.0541 across the
three seeds, for a mean difference of **0.0375**. However, mean AP is almost
identical (0.5529 versus 0.5521), and the comparison changes both information and
architecture. It does not show that the seven statistics independently cause
the entire F1 difference or that all temporal signal has been removed in C.

At the final epoch, mean F1 is A **0.4647**, B **0.4408**, and C **0.4265**.
B is below A in all three final-epoch comparisons. Its favorable primary
comparison therefore depends on the prespecified validation selection policy;
the final-epoch result must remain visible and is not discarded.

The earlier Step-2 full-weight reference has mean F1 0.4619 under its common
criterion, but differs in parameterization and shuffle schedule. It is retained
in `historical_reference_validation.csv` as context, not used to claim a
controlled temporal-branch gain.

### Verification

- Four focused tests pass: feature partition/parameter counts, equal-information
  contract, temporal-value/mask isolation and absent-branch handling, seeded
  training repeats, paired batches, selected epochs and independent snapshots.
- Fresh-kernel execution completed all seven notebook cells with zero errors.
- All **18 checkpoints** reload, reproduce their recorded losses and keyed
  validation predictions, and reproduce the summary and paired-effect tables.
- Independent verification reconstructs the 50-epoch shuffle schedule for each
  seed and confirms that all three arms used exactly that schedule without
  dropping or duplicating users within an epoch.
- Separate-process verification passed and all **67 output artifacts** have
  verified hashes. All **655 protected files** remained unchanged during the run.
  After saving the new executed notebook, an independent preservation check
  confirmed the other **654 files** remained unchanged.
- No test labels/predictions were parsed; no threshold tuning, missingness
  augmentation, architecture additions beyond this experiment, or model promotion
  occurred.

## Recommendation

Preserve B as a thesis-aligned candidate alongside A. The separate branch is
implemented and tested, but the current F1 evidence does not justify declaring
it superior or replacing the baseline. The small ranking-metric gains motivate
further controlled study; they do not solve the overall performance limitation.

Review these results before proceeding to the next approved architectural
stage. Optional GCN representation integration, learned gating, and contrastive
learning remain unimplemented here and must be separate experiments. Preserve
both the flat and branched references so later gains are not attributed to a
temporal benefit that this experiment has not established conclusively.

## Reproduction

```powershell
.\env\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_temporal_statistics.py -v
.\env\Scripts\python.exe -X utf8 scripts/run_temporal_statistics.py --execute-notebook
.\env\Scripts\python.exe -X utf8 scripts/run_temporal_statistics.py --verify-run results/versions/temporal_statistics_v1/RUN_DIRECTORY
```

Each execution creates a unique versioned directory with 18 checkpoints and keyed
validation prediction files, nine histories, metric/effect tables, figures,
feature mappings, training identities, observation masks, source snapshots,
environment metadata, input/output hashes and protected-file inventory.
