# Improvement phase, Step 1: validation-only performance diagnosis

## Scope and scientific boundary

This is a separate versioned analysis of the reproducible baseline. It changes
no existing model, feature, split, class weight, checkpoint-selection policy,
decision threshold, or final test protocol. It performs no training. Sources are
explicitly pinned in `configs/improvement_step1_v1.json`; incomplete and historical
runs are not discovered automatically.

The notebook is `analysis/improvements/01_performance_diagnosis.ipynb`.
The implementation parses only registered training metadata/histories, canonical
train/validation user tables and masks, and validation prediction registries.
It does not load combined metrics files, test labels, or test predictions.
Prior files, including test artifacts, are hashed as opaque bytes solely to
verify preservation. An input allowlist rejects test-data parsing.

Twenty saved checkpoint prediction sets from twelve model experiments are
analyzed. The supervised teacher is already the baseline tabular MLP and is not
counted twice. Semi-supervised best/final learning histories exist, but only
selected-checkpoint validation predictions were saved; final-checkpoint scores
are explicitly outside this analysis rather than fabricated.

## Threshold protocol

- Analyze the saved decision threshold 0.5 and a fixed grid of 101 thresholds
  from 0 to 1 at increments of 0.01.
- Diagnostic maximum-F1 ties prefer proximity to 0.5, then the lower threshold.
- Export every grid result, confusion counts, precision/recall, coverage, and
  precision-maximizing operating points at minimum recall 0.50, 0.75, and 0.90.
- A diagnostic maximum is selected and scored on the same validation users. It
  is optimistically biased and is not an unbiased performance estimate.
- No diagnostic threshold is promoted, exported as a deployment policy, or
  applied to test data. Threshold changes cannot improve ROC-AUC or AP because
  they do not change score ranking.
- Preserve all 144 validation users per prediction registry, including explicit
  unsupported statuses. Report the full 144-user and matched 42-user cohorts
  separately. The latter has just seven annotated bots.

## Main findings: full validation population

All rows below use the same 144 users, including 28 annotated bots. These are
validation results, not replacement test results.

| Model / checkpoint | F1 at 0.5 | Precision at 0.5 | Recall at 0.5 | Diagnostic threshold | Apparent maximum F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tabular LR | 0.4773 | 0.3500 | 0.7500 | 0.51 | 0.4828 |
| Tabular MLP, primary last epoch | 0.4416 | 0.3469 | 0.6071 | 0.73 | 0.5185 |
| Tabular MLP, saved minimum-validation-loss epoch | 0.5000 | 0.3750 | 0.7500 | 0.53 | 0.5122 |
| Text LR | 0.4330 | 0.3043 | 0.7500 | 0.54 | 0.5128 |
| Text MLP, primary minimum-validation-loss epoch | 0.3636 | 0.2261 | 0.9286 | 0.55 | 0.4675 |
| Labeled-only continuation | 0.4800 | 0.3830 | 0.6429 | 0.73 | 0.4912 |
| Pseudo-label student | 0.4390 | 0.3333 | 0.6429 | 0.46 | 0.4598 |

The always-bot reference has F1 0.3256 on this population. This is a context
reference, not a trained model or an acceptable classifier.

### Checkpoint choice matters

The tabular MLP's validation loss reaches 0.9181 at epoch 9 and rises to 1.3081
at epoch 50, while training loss decreases from 1.0969 to 0.5310. Its epoch-9
checkpoint has validation AUC 0.7802 versus 0.7346 at epoch 50 and F1 0.5000
versus 0.4416 at the same 0.5 threshold. The existing primary remains epoch 50.
A subsequent experiment should explicitly consider validation-loss selection;
this analysis does not silently replace the original experiment.

### A higher F1 can sacrifice bot recall

For the primary tabular MLP, moving from 0.5 to the diagnostic 0.73 threshold:

- False positives decrease from 32 to 12.
- True positives decrease from 17 to 14.
- Precision increases from 0.3469 to 0.5385.
- Recall decreases from 0.6071 to 0.5000.

Therefore, reporting only the F1 increase would hide a meaningful trade-off.
The epoch-9 checkpoint detects 21 of the 28 bots at threshold 0.5, with 35 false
positives. The choice depends on the research objective and cannot be settled
by cherry-picking the largest validation F1.

### Threshold tuning alone is insufficient

The apparent full-population F1 maxima remain around 0.46–0.52. The selected
text MLP predicts 115 of 144 users as bots at threshold 0.5, producing 89 false
positives. Raising its threshold to 0.55 reduces false positives to 31 but also
reduces true positives from 26 to 18. This supports a decision-policy diagnosis,
not a claim that its underlying text representation is already sufficient.

### Pseudo-labeling has no demonstrated validation advantage

At the preserved threshold, the student has F1 0.4390 versus 0.4800 for the
labeled-only continuation control. Its AUC/AP are also lower on validation.
Its minimum validation loss occurs at continuation epoch 1 and worsens by
epoch 50. The current validation evidence does not justify escalating
semi-supervised complexity or claiming a reliable improvement.

## Graph-related cohort: promising ranking, fragile threshold estimates

The following use the same 42 validation users, including seven bots:

| Model / selected checkpoint | F1 at 0.5 | AUC | AP | Diagnostic threshold | Apparent maximum F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Degree-only GCN | 0.2174 | 0.4776 | 0.2440 | 0.49 | 0.2857 |
| GCN + Tabular | 0.4348 | 0.8653 | 0.6854 | 0.61 | 0.6250 |
| Full current fusion | 0.4762 | 0.8980 | 0.7324 | 0.79 | 0.7692 |
| Tabular + Text ablation | 0.5455 | 0.9061 | 0.7554 | 0.78 | 0.7692 |

For fusion, the diagnostic threshold retains five true positives and reduces
false positives from nine to one. This is a potentially useful threshold signal,
but selecting it on seven bots makes the estimate unstable and optimistic.
It is not evidence of 0.77 generalization F1 or superiority over full-population
models. The matched-primary table also re-scores all eligible baseline models
on exactly these users; training populations still differ.

The degree-only GCN has near-chance AUC and its selected-checkpoint maximum F1
equals the always-bot reference (0.2857). Threshold changes cannot repair its
weak ordering of users. The Text+Graph branch-removal variant also performs
poorly. Future GNN integration therefore needs an explicit no-graph/degree
reference rather than an assumption that a graph branch will help.

## Learning curves and class weights

- **Text MLP:** minimum validation loss at epoch 2 (1.1036), rising to 3.2009
  at epoch 50, while training loss falls to 0.1300. This pattern is consistent
  with overfitting; train/eval mode and loss aggregation differences prevent
  interpreting the raw gap as a precise generalization estimate.
- **Tabular+Text ablation:** minimum validation loss at epoch 35 (0.3868),
  rising to 1.7415 at epoch 200 while training loss falls to 0.0316.
- **Graph+Tabular:** minimum validation loss is at epoch 200, the training
  boundary. There is no observed earlier validation-loss minimum in this run;
  this does not authorize extending training automatically.
- **Full fusion:** minimum validation loss is at epoch 139. The last checkpoint
  has higher F1 at threshold 0.5 but worse AUC/AP than the selected checkpoint.
  Weighted loss, ranking metrics and fixed-threshold F1 optimize different things.
- Loss values are not directly comparable across model families: objectives,
  class weights, aggregation conventions, populations, and train/eval modes differ.

The bot-to-human loss-weight ratio is approximately **4.122** for the full
labeled training population (540 humans / 131 bots), and **5.226** for the
graph cohort (162 humans / 31 bots). These ratios are verified against training
labels only. Weighted objectives favor bot errors more strongly; the learned
scores should not automatically be interpreted as calibrated posterior probabilities.

The pseudo student retains the 4.122 positive-class weight even though its
selected pseudo population has 195 users of each class. Confidence weights and
the coefficient 0.5 also apply; equal class counts do not imply equal effective
loss contributions. This is an approved existing policy, not a newly discovered
leak or a policy changed by this analysis.

**There is no unweighted or reduced-weight training control in these artifacts.**
Consequently, the class-weight effect cannot be causally quantified by changing
thresholds or comparing different architectures. A separate approved controlled
experiment would be needed to assess it.

## Recommendation after Step 1

1. Review the checkpoint/precision-recall findings before adding architecture.
2. Consider an explicitly versioned selection-policy experiment using the same
   architecture, and, if desired, a small controlled class-weight comparison.
   No such training has been run here.
3. Any internal cross-validation must fit preprocessing on each fold's training
   users; the existing transforms fitted on all 671 training users cannot serve
   as fold-specific transforms.
4. If moving to architectural work, change one component at a time and preserve
   the agreed order and explicit missing-modality handling. No architecture
   improvement follows automatically from this report.
5. Keep the test protocol unchanged. These analyses cannot make the historically
   inspected test set an untouched holdout again.

## Reproduction

Verified run: `results/versions/improvement_step1_v1/20260920T111205869424Z_61a29ac9`.
Fresh-kernel execution completed all seven notebook cells without errors.
Three focused tests pass, including explicit test-input rejection, identity and
missingness checks, and threshold metrics checked against scikit-learn.
Separate-process verification reproduced the 20 checkpoint summaries and 2,020
threshold rows, checked exported predictions against their original validation
sources, and verified all 18 output artifact hashes. All 529 protected files
were unchanged during analysis; after saving the new executed notebook, an
independent check confirmed the other 528 files were still unchanged.

```powershell
.\env\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_performance_diagnosis.py -v
.\env\Scripts\python.exe -X utf8 scripts/run_performance_diagnosis.py --execute-notebook
.\env\Scripts\python.exe -X utf8 scripts/run_performance_diagnosis.py --verify-run results/versions/improvement_step1_v1/RUN_DIRECTORY
```

Each execution creates a new output directory. It saves CSV diagnostics, three
figures, source/config/environment provenance, parsed-input hashes, a protected
file inventory, and an output manifest. Baseline artifacts and previous diagnosis
runs are retained. Generated results remain local under the existing ignore policy.
