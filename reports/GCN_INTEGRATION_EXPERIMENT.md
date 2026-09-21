# Learned GCN representation integration — prespecified experiment

## Problem and scope

Notebook 06 embeds three degree statistics with an MLP, not a GCN. Notebook 04
has a real GCN but predicts only graph-covered users. This new, separate experiment
tests an optional learned graph representation alongside the temporal-statistics
model. Existing experiments and preprocessing remain frozen. It does not yet
integrate tweet embeddings, attention, contrastive learning or pseudo-labels.

## Arms and architecture

All arms use the Phase-2 40-feature order, seven temporal observation masks and
the previous 33/7 feature partition. The common branches are 33→48 (BatchNorm,
ReLU, dropout 0.3) and 14→16 (ReLU, dropout 0.3), with explicit temporal masking.

1. Temporal anchor: concatenate 48+16, then 64→32→1; unchanged previous model.
2. Degree MLP control: graph degrees 3→64→32 using linear layers, ReLU and
   dropout 0.3 after the first layer; concatenate 48+16+32, then 96→32→1.
3. GCN: same widths, activations, dropout and classifier as arm 2, replacing
   only the two graph linear operations with GCNConv message passing.

The graph arms share initial common weights and corresponding graph weights,
parameter counts, batch orders and dropout RNG schedules. GCN versus degree MLP
is the primary controlled comparison. GCN versus the smaller temporal anchor
changes capacity and available information, so it is a secondary comparison.
This does not establish the temporal branch as superior to the prior flat model.

Graph inputs are the existing three raw symmetrized degree counts [d,d,2d],
without new node features, support metadata or trainable identity embeddings.
The graph branch uses every observed node for computation; a canonical user's
representation is attached only when Phase-2 has_graph=1. Node membership,
owner availability and supervised eligibility remain separate fields. All other
users get an explicitly masked, zero graph representation and still receive a
prediction from the non-graph branches. Zeros represent absence, not observations.

## Graph and training protocol

Rebuild train and validation graphs from registered Phase-2 edges and owner
evidence; require exact equality with Notebook-04 node maps/tensors. Read only
user_key and split from the frozen assignment file to check forbidden endpoints;
test labels, features, graphs and predictions are never parsed. Support nodes
may overlap across split graphs. This is gold-isolated inference on a separate
validation graph, not a fully node-disjoint inductive benchmark. Message passing
within each supplied split graph is transductive; no graph is merged across splits.

Train from scratch, end to end, on all 671 training users (540 humans, 131 bots),
including 193 graph-available users. No auxiliary graph supervision/pretraining.
Recompute the training graph representation each minibatch, keeping gradients.
Validation runs only in eval/no_grad mode on its own graph. Seeds 42/43/44,
50 epochs, batch 32, AdamW lr 0.001, decay 0.0001, CPU one thread; positive
class weight 540/131 from the complete training population. Select earliest
minimum user-mean unweighted validation BCE; also save final epoch. Threshold
0.5 is fixed. No tuning, test scoring or automatic model promotion.

## Evaluation and verification

Save best/final checkpoints, histories, keyed predictions for all 144 validation
users, input hashes, environment, feature mapping, node provenance and graph
coverage. Report Bot F1, precision, recall, ROC-AUC, AP and confusion counts for
the same all-user cohort, 42 graph-available users and 102 graph-unavailable
users across every arm. Report seed means/sample SD and paired differences.
The graph cohort contains only seven validation bots: uncertainty is substantial.

Test isolation, masking, gradient propagation, paired initialization, topology
sensitivity and checkpoint independence before real training. Fresh-kernel
execution and independent checkpoint reload must reproduce predictions and
metrics. Verify old artifact hashes. Repeated validation use means results are
development evidence, not an unbiased final generalization estimate.

Temporal formulas/windows remain unverified, and all train/validation temporal
statistics are observed. Synthetic missingness checks establish software behavior
only. Raw degree scaling and redundant degree channels are deliberately preserved;
changing them would be a separate feature experiment.

## Reproduction and artifact interpretation

```powershell
.\env\Scripts\python.exe -X utf8 -m unittest tests.test_gcn_integration tests.test_temporal_statistics -v
.\env\Scripts\python.exe -X utf8 scripts/run_gcn_integration.py --execute-notebook
.\env\Scripts\python.exe -X utf8 scripts/run_gcn_integration.py --verify-run results/versions/gcn_integration_v1/RUN_DIRECTORY
```

Each execution creates a new version. A directory without a complete manifest
is an incomplete attempt, not a research result. The initial interrupted attempt
is preserved; its reporting helper was corrected before the completed experiment.

Node-provenance `supervised_eligible` retains Notebook 04's **graph-branch**
eligibility meaning. It does not determine eligibility for the joint classifier:
all 671 training users contribute to the classifier loss. `tabular_attached=false`
means tabular features are not attached to graph nodes; canonical users still
receive their Phase-2 features through the non-graph branches.

The two graph arms each have 7,441 parameters, versus 4,081 in the anchor.
The verified train/validation graphs share 4,035 support identities and no gold
identities. Assignment reading is restricted to `user_key` and `split`; support
overlap is disclosed rather than described as a fully node-disjoint protocol.

## Completed run and results

Completed version: `results/versions/gcn_integration_v1/20260921T103149397425Z_0df4d0e8`.
Results below come from its `validation_metrics.csv`, `seed_summary.csv` and
`paired_effects.csv`. The earlier incomplete attempt is not included.

All-user validation results (144 users, 28 bots), selected by minimum unweighted
validation BCE, fixed threshold 0.5. Values are means across seeds 42/43/44;
F1 uncertainty is sample SD across seeds, not a generalization confidence interval.

| Arm | Bot F1 ± SD | Precision | Recall | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|
| Temporal anchor | 0.4786 ± 0.0132 | 0.3776 | 0.6548 | 0.7679 | 0.5529 |
| Degree-MLP fusion | 0.3878 ± 0.0582 | 0.3012 | 0.5476 | 0.6884 | 0.3726 |
| GCN fusion | 0.4283 ± 0.0913 | 0.3319 | 0.6071 | 0.7155 | 0.4356 |

**Primary comparison:** GCN minus degree MLP has mean F1 difference +0.0406,
ROC-AUC +0.0271 and AP +0.0630. This is inconsistent across seeds: F1 differences
are −0.0422, +0.1579 and +0.0061. The mean is driven substantially by seed 43;
three seeds do not establish a reliable improvement.

**Secondary comparison:** GCN minus temporal anchor has mean F1 difference
−0.0502, ROC-AUC −0.0523 and AP −0.1173. F1 differences are −0.1420, +0.0250
and −0.0337. The AP difference is negative for all three seeds. The graph branch
does not justify replacing the temporal reference in this experiment.

### Matched coverage cohorts

Mean selected-checkpoint Bot F1 ± sample SD; every arm uses exactly the same
user keys within each row.

| Validation cohort | Users / bots | Temporal anchor | Degree MLP | GCN |
|---|---:|---:|---:|---:|
| All users | 144 / 28 | 0.4786 ± 0.0132 | 0.3878 ± 0.0582 | 0.4283 ± 0.0913 |
| Graph available | 42 / 7 | 0.6005 ± 0.0212 | 0.2738 ± 0.2865 | 0.3861 ± 0.3375 |
| Graph unavailable | 102 / 21 | 0.4446 ± 0.0114 | 0.3943 ± 0.0277 | 0.4213 ± 0.0596 |

Missing-graph users are predicted in every run, but their performance can still
change because the non-graph branches and classifier are jointly trained.
Masking is a functionality guarantee, not a guarantee of unchanged accuracy.

An important operating-point observation: GCN seed 42 predicts all 42
graph-available users as human at 0.5 (TN=35, FP=0, FN=7, TP=0), despite subgroup
ROC-AUC 0.9510 and AP 0.8626. Degree MLP seed 43 also predicts that entire subgroup
as human, despite AUC 0.9551. Ranking and fixed-threshold decisions differ sharply
in these small cohorts. This motivates score-distribution and calibration
diagnosis; it does not prove that threshold tuning will fix overall performance,
and no thresholds were tuned in this experiment.

### Checkpoint sensitivity

| Arm | Selected epochs, seeds 42 / 43 / 44 | Final-epoch mean F1 ± SD | Final-epoch AUC | Final-epoch AP |
|---|---|---:|---:|---:|
| Temporal anchor | 12 / 21 / 16 | 0.4408 ± 0.0188 | 0.7294 | 0.5183 |
| Degree MLP | 5 / 2 / 42 | 0.4167 ± 0.0620 | 0.6955 | 0.4626 |
| GCN | 2 / 18 / 42 | 0.4239 ± 0.0207 | 0.7223 | 0.4805 |

The selected checkpoint minimizes BCE, not F1. Final-epoch means also favor the
temporal anchor over both graph arms; no alternative checkpoint was promoted.
The per-seed curves and confusion counts remain in the versioned artifacts.

## Verification outcome

- Fresh-kernel execution completed all nine training runs and reloaded all
  18 selected/final checkpoints, reproducing all 18 keyed prediction files.
- All three temporal-anchor seeds reproduced the prior temporal experiment's
  metrics and selected epochs.
- All 71 registered output artifacts passed hash verification. All 726 protected
  files remained unchanged during training and notebook verification.
- A post-run inference guard handles an entirely graph-unavailable population
  using explicit zero representations, without creating placeholder nodes/edges.
  It does not change parameter counts or the computation for the populated graphs
  used in this run. The run retains its original training-source snapshot.
- Six GCN integration tests and four temporal regression tests passed, including
  the new empty-graph case. A separate-process verification with the updated
  model reloaded all 18 checkpoints and reproduced the recorded predictions.
- After saving the executed notebook and adding that guard, an independent
  preservation check confirmed the other 724 protected files were unchanged.
- Test labels, features, graphs and predictions were never parsed. Identity-only
  split assignments were used solely to audit held-out endpoint isolation.

## Decision and proposed next step

Keep the existing temporal and flat references frozen; do not promote this GCN
fusion as the final model. Preserve the experiment as evidence about this specific
degree-only integration, not as evidence that GNNs in general cannot help.

Before adding learned gates or contrastive loss, inspect validation-only score
distributions by graph availability, branch representation magnitudes, and the
interaction between checkpoint selection and the fixed operating threshold.
Raw graph degrees are unscaled and redundant, so input scaling is one plausible
factor, but its causal role has not been established. If the diagnosis supports
it, preregister a separate train-fitted graph-scaling experiment with the same
MLP/GCN controls, or a separately justified calibration experiment. Do not bundle
these changes or select them using test results.
