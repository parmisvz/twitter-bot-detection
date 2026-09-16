# Phase 4 — protocol-correct, reproducible degree-only GCN

## Scope and preserved work

Notebook `analysis/04_graph_baseline.ipynb` has been refactored after approval of
its inspection plan. Its original contents, including outputs, were copied before
editing to `analysis/archive/04_graph_baseline_pre_v2.ipynb`. Legacy graph tensors,
node mappings, features, results and other notebooks remain historical artifacts.

The active notebook has 19 cells. Reusable construction, training, persistence and
verification functions live in `src/graph_baseline.py`; generic artifact/metric
helpers are imported from the unchanged Phase-3 module. Configuration is explicit
in `configs/graph_baseline_v2.json`.

## Corrections and their scientific purpose

| Original behavior | Correction | Expected benefit |
|---|---|---|
| One global graph for all splits | Three separate permitted Phase-2 graphs | Held-out gold topology cannot enter training |
| Class weights calculated from all graph labels | Weights from eligible training users only | Remove direct validation/test label leakage |
| Case-sensitive node creation, lowercased screen-name matching | Canonical `user_key` and deterministic split-specific mappings | Remove alias collisions and identity ambiguity |
| Shallow best-state copy | Independent `deepcopy` snapshots, best and final both persisted | Recover intended validation selection |
| Graph membership treated as modality availability | Use Phase-2 `has_graph`; report node-only evidence separately | Make the evaluation population match the accepted modality policy |
| Unsupported users absent from outputs | Full canonical prediction registries with explicit null predictions | Preserve user populations and expose coverage |
| No model weights/history/predictions/provenance | Versioned, hashed run artifacts and independent reload checks | Make results inspectable and reproducible |
| Baselines evaluated on different populations | Re-score saved primary baselines and GCN on common supported keys | Enable valid cohort-matched comparisons |

These are correctness and protocol changes, not claims of improved predictive
performance. No architecture search, threshold tuning, new structural features,
normalization of input degrees or encoder improvements were introduced.

## Preserved GCN methodology

- GCNConv 3 → 64 → 32, ReLU after each layer, dropout 0.3 after the first layer,
  linear classifier 32 → 2.
- Raw, unscaled degree features after reverse-edge concatenation:
  `[symmetrized_in_degree, symmetrized_out_degree, symmetrized_total_degree]`.
  These are exactly `[d, d, 2*d]`; they remain redundant by design in this pass.
- Reverse-edge concatenation retains multiplicity for reciprocal observed edges.
  No silent coalescing. Degrees count message-edge entries before self-loops.
- Standard GCN normalization/self-loops, with `cached=False` explicitly set on
  both layers so a validation graph cannot reuse training adjacency normalization.
- Weighted cross-entropy; class weights `N_train / (2 * count_class_train)`.
- AdamW learning rate 0.01, weight decay 0.0005; full-batch training, 200 epochs;
  no early stopping. Minimum validation loss selects the primary checkpoint.
- Bot probability threshold 0.5, preserving the original `>=` convention.
- New reproducibility controls: seed 42 per training run, deterministic CPU
  operations, one PyTorch thread, recorded package/source/config versions.

Reverse connections and GCN self-loops are computational message-passing
operations, not claims that additional follower relationships were observed.

## Phase-2 inputs and graph protocol

Selected version: `preprocessing_v2_e861084d1a722a21`.
Manifest SHA256: `594a2defef59d663ef171020b7cd1ba75aa321c30e7a7d09b57b560563c109c9`.

Only registered split graph edges, edge-owner provenance, canonical users,
modality masks, frozen assignments and graph audit summaries are consumed.
No legacy graph, global audit graph or unlabeled-owner graph enters training.
No raw-data preprocessing is refitted.

Before constructing tensors, validate artifact hashes, canonical unique keys,
labels and masks, edge/owner consistency, absence of observed self-edges or
duplicate directed relations, and absence of other-split gold endpoints.
Observed support endpoints outside the canonical user table are retained; they
need no metadata, tweets or labels for degree-only message passing.

| Split | Canonical users | Eligible owners | Eligible humans / bots | Gold nodes including node-only evidence | All graph nodes | Directed observed edges |
|---|---:|---:|---:|---:|---:|---:|
| Train | 671 | 193 | 162 / 31 | 573 | 42,954 | 60,924 |
| Validation | 144 | 42 | 35 / 7 | 84 | 11,436 | 13,006 |
| Test | 144 | 41 | 35 / 6 | 89 | 10,453 | 12,536 |

After appending reversed edges, message-passing entry counts are
121,848 / 26,012 / 25,072. The corresponding repeated-entry counts are
3,964 / 336 / 1,210. These reflect the explicitly retained historical convention.

Gold users are isolated across the three graphs. Support account overlap is
4,035 train/validation, 3,919 train/test, and 1,660 validation/test. Therefore this
is held-out-gold evaluation with shared support, **not fully node-disjoint**.
Held-out accounts are evaluated jointly within their cohort graph, not one at a
time against a fixed deployment graph. Graph features are recomputed separately
from each permitted graph; no global degree statistics are used.

## Missing modalities and evaluation populations

The primary loss/evaluation mask is `has_graph=1`. A node without the required
owner evidence may support message passing but has no training/evaluation label
in this experiment. Support and unsupported gold label slots are -1.

All 959 canonical users remain across the prediction files. Graph-unavailable
users have null predictions, `status=missing_graph_modality`, and evidence status
`node_only` or `no_permitted_node`. No missing user receives an invented vector,
edge or neighborhood. Missing text/temporal information does not filter the GCN
cohort; graph eligibility alone determines this branch's availability.

`all_users` retains the complete split denominator while scores describe supported
users and coverage is explicit. `graph_available` scores the eligible cohort.
Training scores are apparent training fit. Reports include bot precision, recall,
F1, ROC-AUC, average precision, accuracy, confusion matrices and classification
reports. Undefined ranking metrics are null. Average precision is not labelled
trapezoidal PR-AUC.

The graph-available test cohort contains only six bots. One seeded run does not
establish uncertainty or generalization across graph-collection conditions.
Historical global-graph results used 657/138/140 labeled nodes and therefore are
not directly comparable with these 193/42/41 primary cohorts. The old model was
also unseeded, its class weights used held-out labels, and its best-state snapshot
was defective. The historical test set has already been inspected.

## Versioned artifacts and downstream handoff

Each execution writes a new directory under `results/versions/graph_baseline_v2/`.

- `manifest.json`: input hash, protocol, eligibility, feature order, graph paths,
  checkpoints, training keys and complete artifact hashes.
- `graphs/`: three canonical key/index CSVs and tensor-only PyTorch bundles.
  Each bundle includes features, message edges, labels and the eligibility mask;
  it loads using `weights_only=True` without a serialized PyG object.
- `models/`: best-validation and final model state dictionaries, architecture,
  epoch/input-version metadata, training-only class weights and training keys.
  These are inference checkpoints, not optimizer-resume checkpoints.
- `histories/`: all 200 training and validation losses.
- `predictions/`: keyed full-population predictions for both snapshots and all
  three splits, including masks, evidence/status and checkpoint selection.
- `metrics.csv` and `reports/`: both evaluation cohorts, classification reports,
  graph coverage, shared support, exact reload comparisons and matched baselines.
- `provenance/`: resolved configuration, Phase-2 manifest, both implementation
  source snapshots, source/notebook/config hashes, package versions, seed, device,
  git state at execution and explicit limitations.

The comparison pins the completed Phase-3 run
`baselines_v2/20260916T103951409018Z_7fa575b7` and uses its declared primary
checkpoints. Saved prediction hashes, full user sets, labels and graph masks are
checked. All five models are scored on the same common supported keys; these keys
are exported. No baseline model is retrained and no checkpoint is selected using
the comparison scores.

Notebooks 05–07 still read legacy global graph artifacts. Their migration must
load the new explicit graph manifest, preserve separate graph objects, and align
each graph's features through its own `user_key` mapping. A local node index must
never be used as an account identity or reused across split graphs. Existing
legacy paths were intentionally not repointed or overwritten.

Generated runs remain local and Git-ignored under the existing versioned-results
rule. The notebook, backup, configuration, module, validation script, tests,
archive documentation and this report are tracked.

## Reproduction

From the repository root, using the existing environment:

```powershell
.\env\Scripts\python.exe -m unittest discover -s tests -p test_graph_baseline.py -v
.\env\Scripts\python.exe scripts/validate_graph_baseline.py --write-executed
```

Run All in notebook 04 is also supported from the root or `analysis/` after the
selected Phase-2/3 outputs exist. The validator protects prior data, models,
results, archives and other notebooks against changes. It independently
reconstructs saved graph tensors, reloads both checkpoints, verifies checkpoint
validation losses against history and recomputes predictions, metrics and
baseline comparisons.

For a saved run, `scripts/validate_graph_baseline.py --verify-run RELATIVE_RUN_PATH`
performs verification in a separate process without training or starting Jupyter.

## Verified run and observed results

Completed run:
`results/versions/graph_baseline_v2/20260916T113841559166Z_2dd11f1e/`.

- Five targeted graph tests passed: forbidden endpoints/owners, preserved edge
  multiplicity and missing users, held-out-label independence of training weights
  and a training update, independent best-state snapshots, and no cross-graph
  adjacency-normalization caching.
- Fresh-kernel execution passed all 19 notebook cells and the full 200 epochs.
- Both checkpoints reloaded with exactly matching predictions in this CPU run.
- All three saved graph tensors/mappings matched reconstruction from Phase-2.
- Six keyed prediction files, 12 metric rows and 10 matched-comparison rows were
  independently checked; 28 exported artifacts passed hash verification.
- Both saved checkpoints reproduced the validation loss recorded for their epoch.
- The standalone `--verify-run` command also passed in a separate Python process.
- All 220 protected files remained unchanged, including Phase-2 and Phase-3
  artifacts, legacy data/results/models, archives and other notebooks.
- Training class counts were 162 humans / 31 bots; weights approximately
  0.595679 / 3.112903. The primary checkpoint is epoch 56; epoch 200 is retained
  as the secondary final snapshot.

The primary GCN test metrics on 41 graph-available users (35 humans / 6 bots):

| Bot precision | Bot recall | Bot F1 | ROC-AUC | Average precision |
|---:|---:|---:|---:|---:|
| 0.162162 | 1.000000 | 0.279070 | 0.595238 | 0.238547 |

These results indicate weak discrimination and many false positives; no model or
threshold was changed in response. They are a reproducible reference for later
research, not evidence that the degree-only GCN improves detection.

The all-model matched cohort contains 42 validation users and **40 test users**.
One of the 41 graph-available test users has no tweet representation, so that user
remains in the full GCN registry but is outside the common five-model comparison.
The exact cohort keys and each model's recomputed metrics are saved together.

To verify this completed run without retraining:

```powershell
.\env\Scripts\python.exe scripts/validate_graph_baseline.py --verify-run results/versions/graph_baseline_v2/20260916T113841559166Z_2dd11f1e
```

No manual execution is required to complete this phase. Fresh-kernel validation
used execution outside the Codex filesystem sandbox because this Windows
installation's Jupyter connection-file permission setup requires it.
