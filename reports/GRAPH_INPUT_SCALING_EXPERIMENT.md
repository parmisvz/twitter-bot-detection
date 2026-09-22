# Controlled graph-input scalar experiment

## Approved question and intervention

Does one training-derived graph input scale improve the raw-degree MLP/GCN
fusion experiment's detection performance and representation stability?

Fit `s = sqrt(mean(X ** 2))` over the three raw degree channels of the 193
graph-available canonical training users, without using labels. Apply `X / s`
to all observed nodes in each permitted graph. Retain all raw tensors. No log
compression, centering, extra features, learned normalization, activation changes,
gates or contrastive learning are included.

## Controls and training

The frozen source is
`results/versions/gcn_integration_v1/20260921T103149397425Z_0df4d0e8`.
Reload its raw degree-MLP, raw GCN and temporal-reference checkpoints and verify
their predictions. Train only the scaled MLP and scaled GCN: seeds 42/43/44,
50 epochs, original 7,441-parameter architectures, identical initial weights and
actual minibatch orders to their raw counterparts. Reuse the existing trainer.
All 671 training users remain included, positive weight 540/131, AdamW learning
rate 0.001 and weight decay 0.0001, batch 32, CPU one thread. Select the earliest
minimum unweighted validation BCE and also preserve final-epoch checkpoints.
Threshold remains 0.5.

Tabular preprocessing, the exact Phase-2 feature order, temporal masks, split
assignments and graph topology stay fixed. Split graphs are not merged. Missing
graph users still receive predictions through the non-graph branches. Only
identity/split columns of the assignment file are read for isolation auditing;
test labels, features, graph tensors and predictions are not parsed.

## Evaluation and verification

Compare scaled GCN versus raw GCN (primary), scaled MLP versus raw MLP,
scaled GCN versus scaled MLP, and scaled GCN versus the temporal reference.
The last comparison also changes capacity/information and is secondary.
Use the identical 144-user validation population and its fixed 42 graph-covered
and 102 graph-unavailable cohorts. Report Bot F1, precision, recall, ROC-AUC,
AP, confusion counts, seed variation and selected/final checkpoint differences.
Inspect graph-vector inactivity, weighted branch magnitudes and graph-removal
logit effects; the removal operation is an inference diagnostic, not a retrained
ablation.

Store the fitted scalar and fitting identities, transformed node features,
12 new checkpoints, six histories, predictions from all 30 new/frozen checkpoints,
60 keyed train/validation diagnostic files, metrics, hashes and source provenance.
Fresh-kernel execution and independent verification must reproduce model outputs,
scalar fitting, preserved graph tensors, raw/scaled pairing and histories.

Scaling changes optimization and effective regularization despite unchanged
capacity. Validation is reused for development and checkpoint selection; three
seeds measure limited optimization variability. The covered validation cohort
has only seven bots. No automatic model promotion or test evaluation is allowed.

## Reproduction

The executed notebook now reloads the completed version below. To train a new
version, run `scripts/run_graph_scaling.py` without arguments. To verify and
display an existing version in a fresh kernel, use `--execute-notebook --resume-run
results/versions/graph_scaling_v1/RUN_DIRECTORY`.

```powershell
.\env\Scripts\python.exe -X utf8 -m unittest tests.test_graph_scaling tests.test_graph_diagnosis tests.test_gcn_integration -v
.\env\Scripts\python.exe -X utf8 scripts/run_graph_scaling.py --execute-notebook
.\env\Scripts\python.exe -X utf8 scripts/run_graph_scaling.py --verify-run results/versions/graph_scaling_v1/RUN_DIRECTORY
```

## Completed results

Run: `results/versions/graph_scaling_v1/20260922T052601567701Z_4a1137a7`.
The training-only scalar was **454.3006810563794**. All six runs completed
50 epochs. Both scaled arms selected epochs 12, 18 and 20 for seeds 42, 43 and 44.

Selected-checkpoint validation results, three-seed mean (F1 shows sample SD):

| Model | Bot F1 | Precision | Recall | ROC-AUC | AP | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Temporal reference | 0.479 ± 0.013 | 0.378 | 0.655 | 0.768 | 0.553 | 0.184 |
| Raw degree MLP | 0.388 ± 0.058 | 0.301 | 0.548 | 0.688 | 0.373 | 0.208 |
| Scaled degree MLP | 0.440 ± 0.036 | 0.346 | 0.607 | 0.763 | 0.546 | 0.188 |
| Raw GCN | 0.428 ± 0.091 | 0.332 | 0.607 | 0.716 | 0.436 | 0.194 |
| Scaled GCN | 0.431 ± 0.037 | 0.339 | 0.595 | 0.762 | 0.531 | 0.188 |

All rows use the same 144 validation users and threshold 0.5. Paired GCN scaling
changes mean F1 by only +0.003, AUC by +0.047 and AP by +0.096. F1 improves for
seed 42 but declines for seeds 43 and 44: this is not a consistent F1 improvement.
Scaled GCN remains below scaled degree MLP and the temporal reference on mean
F1 and AP. These findings do not establish a benefit from message passing.

| Matched validation cohort | Temporal F1 | Raw MLP F1 | Scaled MLP F1 | Raw GCN F1 | Scaled GCN F1 |
|---|---:|---:|---:|---:|---:|
| Graph available, 42 users | 0.600 | 0.274 | 0.520 | 0.386 | 0.492 |
| Graph unavailable, 102 users | 0.445 | 0.394 | 0.420 | 0.421 | 0.417 |

Covered-user F1 variability decreases substantially, but covered-user mean AUC
falls from 0.890 to 0.835 for GCN. Better fixed-threshold decisions do not imply
better within-cohort ranking. The exported AUC pair decomposition separates
within-coverage and cross-coverage ranking, avoiding that conflation.

Scaling addresses numerical branch dominance: selected GCN median weighted graph
magnitude ratios are 0.718, 0.927 and 0.863, versus raw 24.218, 3.253 and 1.529.
All scaled GCN selected and final checkpoints have zero completely inactive
covered validation representations. Graph contributions still lower covered-user
logits (selected median removal effects -0.456, -0.391 and -0.659).

Final-epoch mean F1 is 0.401 for scaled MLP and 0.389 for scaled GCN, below their
selected checkpoints (0.440 and 0.431). Their final Brier scores also worsen to
0.207 and 0.213. Continued fitting does not resolve the validation weakness;
preserve the predeclared checkpoint policy rather than selecting a favorable seed.

## Verification and recommendation

Training completed in a fresh kernel. The first post-training verification stopped
on a CSV dtype mismatch: the temporal reference's inapplicable nullable boolean
diagnostic was read as floating NaN. Explicit nullable-boolean loading fixed the
verifier only; no checkpoint, metric, prediction or training artifact was rewritten.
The run retains the original training-source snapshots. Fresh-kernel replay and
independent process verification then passed: 12 new and 18 frozen checkpoints,
30 validation prediction files and 60 keyed diagnostic files reproduced. All 12
targeted/regression tests passed. No test input was parsed.

The run verified 942 protected files before completion. Subsequent intentional
changes are the new experiment's verifier and executed notebook only; all prior
baseline artifacts remain unchanged. The runner now supports replay without
retraining. The figure was visually inspected.

Do not promote the graph model. Keep the temporal reference and flat control,
and preserve scaling as a documented graph optimization experiment. The next
proposed architecture experiment is a small missingness-aware gated fusion versus
a matched concatenation control, with fixed inputs, splits, training settings and
three seeds. Predeclare its capacity comparison and validation criteria before
training; test data stays sealed. This requires separate approval. Current evidence
does not show that a gate will outperform the temporal reference, and graph inputs
remain limited partial-graph degree information. Contrastive learning should follow
only as a separate experiment after that decision.
