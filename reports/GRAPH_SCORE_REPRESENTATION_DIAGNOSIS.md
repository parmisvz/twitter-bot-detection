# Graph score distribution and representation diagnosis

## Conclusion

The frozen experiment shows a coverage-related score offset, strongly unequal
branch magnitudes in some checkpoints, and a completely inactive graph branch
in one final checkpoint. These findings justify a controlled graph-input scaling
experiment before adding gates or contrastive learning. They do not prove that
scaling will improve F1, or that every graph model is unsuitable for this dataset.

Verified diagnosis: `results/versions/graph_diagnosis_v1/20260922T050323622224Z_abbeded5`.
Frozen source: `results/versions/gcn_integration_v1/20260921T103149397425Z_0df4d0e8`.

## Scope and methods

- Inspect all 18 selected/final checkpoints, with all 671 training and 144
  validation users per checkpoint. Training scores are in-sample diagnostics.
- Preserve Phase-2 identities, feature order, observation masks and separate
  graphs. Do not parse test labels, features, graphs or predictions. Read only
  user_key/split from frozen assignments for endpoint isolation checks.
- Keep threshold 0.5 fixed. No training, calibration fitting, threshold search,
  preprocessing refit, checkpoint reselection or model promotion.
- Summarize probabilities/logits by class and graph availability; calculate
  Brier score, unweighted BCE, ten-bin reliability and binned ECE.
- Exactly decompose overall ROC-AUC into bot/human pairs within and across
  graph-availability groups. Weighted contributions must sum to overall AUC.
- Measure branch L2 and RMS magnitudes, first-fusion-layer weighted RMS, zero
  activation fractions and realized signed logit contributions.
- Recompute outputs after zeroing graph representations as an inference
  perturbation. This can be out of distribution and is not a retrained ablation.

All comparisons retain individual seeds. Three runs do not create three
independent copies of the validation users. The graph-covered validation cohort
contains only seven bots; repeated validation use limits generalization claims.

## 1. Coverage-related score offset

For GCN seed 42's selected checkpoint (epoch 2), validation probabilities are:

| Group | Users | Minimum | Median | Maximum |
|---|---:|---:|---:|---:|
| Graph available, human | 35 | 0.1357 | 0.2139 | 0.2628 |
| Graph available, bot | 7 | 0.2520 | 0.2678 | 0.2906 |
| Graph unavailable, human | 81 | 0.2520 | 0.5087 | 0.6351 |
| Graph unavailable, bot | 21 | 0.4232 | 0.5322 | 0.6242 |

Every graph-covered bot is below 0.5, despite within-covered ROC-AUC 0.9510.
The same checkpoint's overall AUC is only 0.6275. The exact pair decomposition
explains how those observations coexist:

| Bot availability | Human availability | Pair weight in overall AUC | Pairwise AUC |
|---|---|---:|---:|
| No graph | No graph | 0.5237 | 0.6249 |
| No graph | Graph | 0.2263 | 1.0000 |
| Graph | No graph | 0.1746 | 0.0123 |
| Graph | Graph | 0.0754 | 0.9510 |

A graph-covered bot almost never outranks a graph-unavailable human. Conversely,
every graph-unavailable bot outranks every graph-covered human. Availability is
strongly associated with the score level in this checkpoint.

This severe pattern is not universal. For selected GCN seeds 43 and 44, the
graph-bot versus no-graph-human pair AUCs are 0.7302 and 0.7284. The failure is
seed/checkpoint dependent, rather than evidence of an identity alignment bug.
No checkpoint has an exactly saturated probability of 0 or 1 in these exports.

## 2. Graph magnitudes can dominate the first fusion layer

The graph arm has 32 outputs, the other profile/behavior branch 48, and the
temporal branch 16. RMS permits a dimension-normalized comparison; L2 is also
exported. The following are medians over the same 42 graph-covered validation
users at the selected checkpoint:

| Arm | Seed | Graph RMS | Other RMS | Temporal RMS | Weighted graph / other-branches ratio | Actual graph logit effect |
|---|---:|---:|---:|---:|---:|---:|
| Degree MLP | 42 | 37.7246 | 0.6184 | 0.4007 | 26.02 | −0.3425 |
| Degree MLP | 43 | 45.3533 | 0.6020 | 0.2109 | 51.94 | −1.0266 |
| Degree MLP | 44 | 25.3683 | 0.6978 | 0.5595 | 6.81 | +0.8070 |
| GCN | 42 | 26.7174 | 0.5864 | 0.3878 | 24.22 | −1.1655 |
| GCN | 43 | 6.5191 | 0.6851 | 0.3498 | 3.25 | +0.0761 |
| GCN | 44 | 5.3855 | 0.6781 | 0.4571 | 1.53 | +0.2399 |

For each user, the weighted ratio is
`RMS(W_graph h_graph) / (RMS(W_other h_other) + RMS(W_temporal h_temporal))`;
the table reports the median of those ratios. All weighted terms have width 32.
This ratio concerns pre-ReLU fusion contributions, not a probability share.
The actual graph logit effect is `original_logit - zero_graph_logit`, recomputing
the classifier's nonlinear activations for the zero-graph input.

Seed 42's large graph branch therefore does more than have a large raw norm:
it contributes a large first-layer signal and suppresses bot logits for covered
users. Its median effect is −1.3088 on covered training users and −1.1655 on
covered validation users, so the phenomenon is already visible in training.

The exported signed logit terms use the actual ReLU activation mask and sum to
the observed logit, including bias terms. They are algebraic descriptions, not
causal attributions. Their sign can differ from the nonlinear zero-graph effect.

## 3. A final checkpoint has a completely inactive graph branch

At GCN seed 43's selected checkpoint, the graph representation is nonzero for
all covered users. At its final checkpoint:

- All **193/193** graph-covered training users have zero graph vectors.
- All **42/42** graph-covered validation users have zero graph vectors.
- All 32 output coordinates are zero for these users.
- Zeroing the graph representation changes no logits for these users.

These users have `has_graph=1`; their zero vectors are learned inactivity after
ReLU, not missing-modality placeholders. The other final GCN checkpoints also
have sparse graph vectors: seeds 42 and 44 retain two nonzero coordinates per
covered user. This is an observed optimization/representation failure mode; the
diagnosis does not establish its cause or the epoch when inactivity began.

No graph-missing user has a nonzero graph vector or a nonzero graph-removal
effect in any diagnosed checkpoint. Their accuracy can nevertheless differ
between trained models because the remaining branches/classifier were jointly
optimized with the graph branch earlier in training.

## 4. The graph inputs reflect the partial, owner-centric graph

The three input channels are exactly `[d, d, 2d]` in both permitted graphs.
They contain one underlying degree statistic, not three independent structural
features. These are degrees in the observed split graph, not full Twitter/X
follower counts.

| Population | Train median d | Validation median d |
|---|---:|---:|
| Graph-covered canonical users | 326 | 311 |
| Non-gold support nodes | 1 | 1 |
| All observed graph nodes | 1 | 1 |

The contrast is consistent with owner-centric sampling. Owner and support nodes
have very different observed degree distributions. It should not be interpreted
as complete network structure or an established bot-specific pattern.

## 5. Calibration evidence is mixed; threshold changes alone are insufficient

Mean all-user validation diagnostics over the three selected checkpoints:

| Arm | Brier ↓ | Unweighted BCE ↓ | Ten-bin ECE ↓ | ROC-AUC ↑ | AP ↑ |
|---|---:|---:|---:|---:|---:|
| Temporal anchor | 0.1843 | 0.5429 | 0.2058 | 0.7679 | 0.5529 |
| Degree MLP | 0.2083 | 0.6049 | 0.2335 | 0.6884 | 0.3726 |
| GCN | 0.1945 | 0.5699 | 0.2039 | 0.7155 | 0.4356 |

GCN has slightly lower ten-bin ECE than the anchor but worse Brier, BCE and
ranking metrics overall. On graph-covered users, GCN's mean Brier is 0.1238,
versus the anchor's 0.1313, despite worse F1 at 0.5. Poor fixed-threshold recall
does not by itself prove poor probability calibration. ECE is bin dependent and
unreliable as a standalone selection metric in these small subgroups.

Graph removal in GCN seed 42 illustrates the trade-off: it moves all seven
covered bots above 0.5, but also creates **13 false positives** among covered
humans. This is not a validated replacement model. A global threshold adjustment
does not change score rankings or revive a zero representation.

## Recommended next controlled experiment — not implemented

Test **graph-input magnitude normalization only**, before new fusion gates or
contrastive loss. Use one positive scalar derived without labels from the 193
graph-available training users:

`s = sqrt(mean(X_train_available ** 2))`, across their three raw degree channels.

Then apply `X_scaled = X_raw / s` to every observed node in each permitted graph.
Use the same saved training-derived scalar on validation and, only during a
separately authorized final evaluation, test graphs. Do not fit anything on
validation/support populations from other splits. Retain the raw graph tensors
and record the fit user_keys, scalar, source hashes and transformation policy.

Why this first: one scalar changes input magnitude while preserving channel
ratios, ordering, relative differences and architecture. Adding log compression,
centering, learned normalization or a new activation at the same time would test
additional changes. The intervention may improve optimization stability; it is
not guaranteed to improve detection or prevent branch inactivity.

Proposed comparison:

1. Reuse verified frozen raw-degree MLP and GCN controls and temporal reference.
2. Train scaled-degree MLP and scaled-degree GCN from scratch with paired seeds
   42/43/44, matching initial weights and minibatch orders to the raw controls.
3. Preserve 50 epochs, existing AdamW settings, training-only class weight,
   minimum unweighted validation-BCE selection, and threshold 0.5.
4. Keep all 671 training users and the same 144/42/102 validation cohorts. Keep
   graph masks, graph topology, all Phase-2 tabular transforms and architecture
   fixed; no learned gates, pseudo-labels or text changes.
5. Compare scaling effects within each architecture, then GCN versus MLP under
   the scaled condition. Report seed variability, inactive-vector counts,
   weighted branch magnitude, pairwise AUC, Brier/AP/F1 and missing-graph results.

Do not fit group-specific calibrators on seven validation bots or promote the
best seed. Preserve the test protocol and retain the temporal/flat references.
This proposal requires the next phase's approval before implementation/training.

## Artifacts, verification and reproduction

- Fresh-kernel execution diagnosed 18 checkpoints and produced 36 keyed
  train/validation files, six summary tables, raw-degree/coverage tables,
  verification counts, two figures and provenance/source snapshots.
- All 60 registered artifacts passed hash verification and separate-process
  summary reconstruction. Maximum realized-logit reconstruction error:
  **0.00000190735**. Saved validation decisions and probabilities were checked
  against the frozen experiment, and checkpoint state hashes did not change.
- Three diagnostic tests and six existing GCN regression tests passed. Tests
  include exact numeric CSV round trips, logit decomposition, zero missing-graph
  effects, AUC ties and reliability-bin endpoints.
- The first extraction exposed short float32 CSV formatting. It is preserved
  as an unverified diagnostic attempt and excluded from this report. The verified
  version widens float32 values exactly to float64 before export/statistics and
  uses round-trip CSV parsing. No checkpoint or prediction rule changed.
- All 817 protected files stayed unchanged during the verified run. The executed
  diagnosis notebook is the only intended later change among those files.
- Test data was not scored, no model was trained, and no threshold/calibrator or
  preprocessing transformation was fitted. No existing experiment was modified.

```powershell
.\env\Scripts\python.exe -X utf8 -m unittest tests.test_graph_diagnosis tests.test_gcn_integration -v
.\env\Scripts\python.exe -X utf8 scripts/run_graph_diagnosis.py --execute-notebook
.\env\Scripts\python.exe -X utf8 scripts/run_graph_diagnosis.py --verify-run results/versions/graph_diagnosis_v1/20260922T050323622224Z_abbeded5
```
