# Step 2: controlled checkpoint selection and class weights

## Protocol fixed before training

Scope: the existing **tabular MLP only**, trained from scratch on the frozen
671 labeled training users. No architecture change, new feature, imputation,
preprocessing refit, graph change, text experiment or pseudo-label training.
All 144 validation users are retained. Test labels and predictions are never
parsed; prior artifacts are hashed only as opaque bytes for preservation checks.

The new notebook is `analysis/improvements/02_checkpoint_and_class_weight.ipynb`.
Inputs and policy are pinned by `configs/improvement_step2_v1.json`. Outputs
are new directories under `results/versions/improvement_step2_v1/`.

### Fixed training conditions

- Phase-2 manifest feature order: the existing 40 features and training-fitted
  transforms. Per-feature observation and modality masks remain explicit.
- Architecture: 40 -> 64 (BatchNorm/ReLU/dropout) -> 32 (ReLU/dropout) -> 1;
  dropout 0.3, exactly the Notebook-03 implementation.
- AdamW, learning rate 0.001, weight decay 0.0001, batch size 32, 50 epochs.
- Paired seeds: **42, 43, 44**, CPU, one PyTorch thread, deterministic algorithms.
- Decision threshold: **0.5**, fixed for every arm and checkpoint.
- No class sampling or extra data. No users removed for missing modalities.

### Training-weight intervention

The negative-to-positive ratio is computed only from the labeled training users:
540 / 131 = approximately 4.1221. The three fixed arms use BCE positive weights:

| Arm | Weight rule |
| --- | --- |
| Unweighted | 1.0 |
| Reduced weight | sqrt(540 / 131), approximately 2.0303 |
| Baseline weight | 540 / 131, approximately 4.1221 |

Each weight arm starts from identical initial parameters for a given seed and
uses identical batch order and random-number consumption. Initial-state hashes,
batch-order hashes, and final RNG-state hashes verify this pairing. Training
from scratch avoids confounding weight effects with an already weighted teacher.

Changing positive weight also changes loss/gradient scale under ordinary BCE's
mean reduction. This study estimates the effect of that implemented training
policy; it does not isolate a pure class-preference mechanism independent of
optimization. AdamW and other hyperparameters remain unchanged.

### Checkpoint policies

Every training run saves three independently copied checkpoint states:

1. **Last:** epoch 50; fixed training length across weight arms.
2. **Minimum weighted validation loss:** the run's training weight and historical
   mean-of-batch-means convention. This preserves the baseline compatibility
   comparison, but numerical losses must not rank arms with different weights.
3. **Minimum unweighted validation BCE:** common user-mean BCE across all arms.
   This is the primary checkpoint criterion for comparing weights.

Ties choose the earliest epoch. No early termination is introduced; the full
50-epoch trajectory remains available. The comparison includes three weights x
three seeds x three checkpoint policies = 27 checkpoint evaluations.

The full-weight seed-42 control runs first. Its complete weighted loss history
and final/weighted-best validation predictions must reproduce the pinned
Notebook-03 run before interpreting the remaining experiment.

### Comparison and interpretation

- Report per-seed and mean/sample-standard-deviation Bot F1, precision, recall,
  ROC-AUC, and Average Precision on exactly the same validation user keys.
- Report paired weight effects against the full-weight arm under the same
  checkpoint policy, and paired checkpoint effects against the last epoch under
  the same weight and seed. Do not compare the best seed of one arm to another.
- Fixed-epoch comparisons hold selection constant. Common-criterion comparisons
  include the effect of each training trajectory on checkpoint selection.
- Three seeds measure optimization variability, not sampling confidence intervals
  or evidence of statistical significance.
- Validation users select checkpoints and supply these reported metrics. These
  are exploratory development results with selection bias, not new test results.
- Weighted validation loss, unweighted validation BCE, ranking metrics, and
  thresholded F1 have different objectives. A common BCE criterion is not a
  guarantee of the largest F1.
- No checkpoint, weight, or threshold is automatically promoted. The existing
  baseline and final test protocol remain unchanged.

## Results (2026-09-20)

Verified run: `results/versions/improvement_step2_v1/20260920T114151695086Z_3c7db074`.
Every result below uses the same **144 validation users / 28 annotated bots**,
with threshold fixed at 0.5. No test result was computed.

### Primary comparison: common unweighted-BCE checkpoint criterion

Values are means across the three paired seeds; F1 includes sample standard
deviation. These are not confidence intervals.

| Positive weight | Bot F1 mean +/- SD | Precision | Recall | ROC-AUC | Average Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.0000 | 0.2807 +/- 0.0222 | 0.5563 | 0.1905 | 0.7444 | 0.4496 |
| 2.0303 (sqrt ratio) | 0.4658 +/- 0.0092 | 0.4458 | 0.4881 | 0.7533 | 0.4760 |
| 4.1221 (full ratio) | 0.4619 +/- 0.0259 | 0.3689 | 0.6190 | 0.7591 | 0.5045 |

Reduced weighting changes mean F1 by only **+0.0039** relative to full weighting,
while reducing recall by **0.1310** and increasing precision by **0.0769**.
Its paired F1 differences are -0.0143, +0.0237, and +0.0024 for seeds 42/43/44.
It lowers AP in all three seeds. This is a precision/recall trade-off, not
convincing evidence that reduced weighting is generally better.

Removing weighting increases precision but detects only about 19% of validation
bots on average under this checkpoint policy and fixed threshold. This supports
retaining class weighting for the current operating point; it does not prove
that unweighted models would remain inferior under a separately evaluated
threshold/calibration policy.

### Checkpoint-selection sensitivity

| Weight | Last-epoch F1 mean | Weighted-loss-selected F1 mean | Common-BCE-selected F1 mean |
| --- | ---: | ---: | ---: |
| 1.0000 | 0.4235 | 0.2461 | 0.2807 |
| 2.0303 | 0.4802 | 0.4403 | 0.4658 |
| 4.1221 | 0.4465 | 0.4583 | 0.4619 |

The reduced-weight last checkpoint has the highest mean F1 in this table, but
this was a secondary comparison and is not automatically promoted. Choosing
whichever weight/checkpoint combination has the largest number after inspection
would introduce another layer of validation selection.

For full weighting, weighted-loss selection improves F1 versus the last epoch
by +0.0584 and +0.0232 in seeds 42/43, but lowers it by -0.0464 in seed 44.
It increases recall and ROC-AUC in all three seeds. Thus the favorable original
seed-42 checkpoint result does not translate into a uniform F1 improvement.

The common criterion selects full-weight epochs **16, 19, 37**, reduced-weight
epochs **15, 12, 13**, and unweighted epochs **14, 10, 9**. For the unweighted
arm, the two loss-selection policies can still choose different epochs because
historical mean-of-batch-means loss weights the final smaller batch differently
from the common user-mean loss. This is an explicit aggregation distinction,
not a change in training or a checkpoint persistence error.

### Baseline reproduction and verification

- Full-weight seed 42 reproduced the original 50-epoch weighted loss history,
  epoch-9 weighted-best selection, and both original best/final validation
  prediction sets within the declared numerical tolerances.
- All nine runs use 671 training users only. Initial-state, batch-order, and
  final RNG hashes agree across the three weight arms within each seed.
- All **27 checkpoints** reload, reproduce their selected epoch losses and keyed
  validation predictions, and reproduce the 27 metric rows and summary/effect tables.
- The fresh kernel completed all seven notebook cells without errors. Two
  focused tests pass, covering training-derived weights, paired stochastic
  controls, deterministic repeats, selection correctness, and independent states.
- Separate-process verification passed; **82 output artifacts** have verified
  hashes. All **569 protected files** remained unchanged during training/export.
  After saving the new executed notebook, a separate preservation check confirmed
  the other **568 files** remained unchanged.
- No test labels/predictions, unlabeled data, or pseudo-labels were parsed or used.

## Recommendation

This experiment has not produced a substantial F1 improvement. It clarifies
that class weighting helps recall at threshold 0.5, while checkpoint policy has
seed-dependent effects on F1. The small differences do not justify claiming a
new best model or removing the full positive weight.

For a subsequent approved architectural experiment, retain the full training
ratio as a reference and hold the common checkpoint criterion fixed across its
arms. This keeps the effect of the new component distinguishable from additional
weight/selection changes. The reduced-weight arm remains a documented
precision-oriented alternative rather than a discarded experiment.

The next architectural step may be the separately designed temporal-statistics
encoder; no such component has been implemented here. Do not add GCN fusion,
gating, contrastive learning, or pseudo-labeling at the same time. Any later
promotion to the final evaluation protocol requires an explicit decision.

## Reproduction

```powershell
.\env\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_checkpoint_weight_experiment.py -v
.\env\Scripts\python.exe -X utf8 scripts/run_checkpoint_weight_experiment.py --execute-notebook
.\env\Scripts\python.exe -X utf8 scripts/run_checkpoint_weight_experiment.py --verify-run results/versions/improvement_step2_v1/RUN_DIRECTORY
```

Versioned outputs include all checkpoint states, histories, keyed validation
predictions, train/validation identity and mask provenance, per-seed metrics,
summary/paired-effect tables, figures, source snapshots, input/output hashes,
environment details, and the preservation inventory. Existing artifacts and
earlier improvement runs are retained.
