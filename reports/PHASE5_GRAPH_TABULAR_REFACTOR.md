# Notebook 05 — Graph + Tabular compatibility refactor

## Scope and preserved work

The original `analysis/05_graph_tabular.ipynb` was backed up before editing to
`analysis/archive/05_graph_tabular_pre_v2.ipynb`, including outputs and metadata.
The active notebook has 19 cells. Implementation is in `src/graph_tabular.py`,
configuration in `configs/graph_tabular_v2.json`, regression tests in
`tests/test_graph_tabular.py`, and kernel/reload validation in
`scripts/validate_graph_tabular.py`. Prior preprocessing, baseline and graph
modules are imported unchanged. Legacy artifacts are not overwritten.

This phase preserves the original model and deliberately restricted support-node
feature policy. It corrects protocol and reproducibility, not architecture.
Policy guards reject changes to the approved architecture, hyperparameters,
attachment policy, initialization and comparison populations instead of silently
relaxing the user's conditions.

## Original experiment and corrections

The old notebook concatenated all 959 gold users' tabular features onto the
historical global graph. Training therefore accessed validation/test topology
and tabular features through message passing. Its training masks selected
657/138/140 users across the three splits, rather than the accepted owner-available
cohorts. A case-alias collision attached features to 936 nodes for 935 identities.

The original **training-only class weights and `deepcopy` best-state selection
were already correct** and are preserved. Missing model persistence, unseeded
training, a hardcoded result CSV, and absent keyed predictions/provenance are
corrected. The old preprocessing columns happened to match the new 40-column
order, but exclusion-based column selection was replaced with the explicit
Phase-2 manifest order. Phase-2's corrected `description_length` transformation
is inherited; historical results are not directly comparable.

## Approved architecture and training settings

Input-level concatenation: three degree columns plus 40 tabular columns.

| Component | Preserved setting |
|---|---|
| Input width | Exactly 43 |
| First convolution | GCNConv 43 → 64, ReLU |
| Second convolution | GCNConv 64 → 32, ReLU |
| Classifier | Linear 32 → 2 |
| Dropout / batch normalization | Neither is present |
| Loss | Class-weighted cross-entropy |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 |
| Training | Full-batch, 200 epochs, no early stopping |
| Primary checkpoint | Minimum validation loss |
| Secondary checkpoint | Final epoch |
| Classification threshold | Bot probability ≥ 0.5 |

Train from scratch with seed 42, deterministic CPU operations and one PyTorch
thread. No Graph-only model weights are loaded for initialization. Standard GCN
normalization/self-loops are preserved; `cached=False` prevents adjacency reuse
between graphs. Computational reversals/self-loops are not observed Twitter
relationships. Degree columns retain the Notebook-04 order and raw `[d,d,2d]`
values, including reciprocal-edge multiplicity.

## Pinned inputs and isolation

Selected inputs:

- Phase-2: `preprocessing_v2_e861084d1a722a21`.
- Notebook-03: `baselines_v2/20260916T103951409018Z_7fa575b7`.
- Notebook-04: `graph_baseline_v2/20260916T113841559166Z_2dd11f1e`.

Configuration pins all three manifest paths and SHA256 values. The loader checks
Phase-2 identities, labels, masks, exact feature order and training-only fitted
preprocessing provenance. It then verifies each saved Notebook-04 node mapping
and tensor against reconstruction from the accepted Phase-2 edge/owner files.
The saved Notebook-04 tensors are the graph source after those checks.

Each split remains a separate graph with its own canonical `user_key` mapping.
No other-split gold endpoints are permitted. No global audit graph or additional
unlabeled-owner edges are added. Shared support identities are documented, not
misrepresented as fully node-disjoint evaluation. Inference is joint within each
held-out cohort graph.

## Approved tabular attachment semantics

Only `manifest["feature_order"]` selects the 40 tabular inputs. No imputer or
scaler is fitted here, including on the smaller graph-eligible training subset.
The Phase-2 transformations fitted on all 671 training users are consumed as-is.
Canonical observation masks are exported; missing temporal placeholders retain
their Phase-2 semantics without additional imputation or user removal.

For each graph:

1. Attach transformed tabular rows to permitted **same-split gold nodes only**.
   This includes node-only gold context users who are not supervised targets.
2. Other support nodes retain a 40-dimensional zero block. These are strictly
   **unpopulated slots, not measured zeros**. Even known non-gold metadata is
   deliberately unused in this compatibility experiment.
3. Export `tabular_attached`, `tabular_provenance` and `tabular_source_artifact`
   separately from the 43 input columns. Never infer attachment from numerical
   values: an attached transformed row can legitimately contain zeros.
4. Keep graph membership, `has_graph` and supervised eligibility distinct in
   exports. Non-gold support `has_graph` is unspecified in the gold-cohort
   sidecar, not asserted as an observed absence; its eligibility is false.

| Split | All users | Graph nodes | Attached gold-node tabular blocks | Unpopulated support blocks | Supervised/evaluation targets |
|---|---:|---:|---:|---:|---:|
| Train | 671 | 42,954 | 573 | 42,381 | 193 |
| Validation | 144 | 11,436 | 84 | 11,352 | 42 |
| Test | 144 | 10,453 | 89 | 10,364 | 41 |

Only eligible training labels determine the class weights: 162 humans / 31 bots.
The formula is unchanged, `N_train / (2 * count_class_train)`. The trainer accepts
training and validation graphs only; test features/labels never enter its inputs.

## Missing graph users and full-population reporting

Both checkpoints save one prediction file per split containing **every canonical
user**. Unsupported users receive `status=missing_graph_modality` with null
probability and prediction. Node-only context and absence of a permitted node are
distinguished. A graph-absent canonical user has an available tabular table row,
but no graph slot to attach it to; its provenance says so explicitly.

No fallback classifier, invented graph neighborhood or support metadata is added.
Missing tweets or temporal information never filter an otherwise eligible target.
`supervised_eligible` describes the declared training/evaluation target mask,
which follows `has_graph` in this experiment; it is not synonymous with node
membership or tabular attachment.

Metrics include bot precision, recall, F1, ROC-AUC, average precision, accuracy,
class counts and coverage, with confusion matrices/classification reports.
`all_users` retains the canonical denominator while metrics describe supported
predictions. `graph_available` uses the explicit eligible population. Training
metrics are apparent fit scores. All metrics are computed, never transcribed.

## Exact matched-cohort descriptive comparisons

Tabular LR, Tabular MLP, Graph-only and Graph+Tabular are scored on **exactly the
same 42 validation and 41 test user keys**. The keys come from the accepted
Phase-2 graph-available cohort and are exported. Each comparator's saved primary
predictions are hash-verified and checked for key, label and mask agreement.
Missing predictions trigger failure, never silent cohort intersection/shrinkage.
There is no text-availability restriction and no reduction to the earlier
five-model comparison's 40 test users.

Every comparison row is labelled `descriptive_not_controlled_ablation`:

- Tabular baselines trained on 671 users; graph models train on 193.
- Graph-only has dropout 0.3, lr 0.01 and weight decay 0.0005; Graph+Tabular
  preserves no dropout, lr 0.001 and weight decay 0.0001.

Thus any score difference does not isolate the causal effect of adding tabular
features. No comparator is retrained or selected by test results. The test cohort
has only six bots, and one seeded run does not quantify uncertainty. Historical
test data has already been examined.

## Versioned outputs and downstream contract

Each execution writes a new directory under `results/versions/graph_tabular_v2/`:

- `manifest.json`: all input-manifest hashes, combined feature order, approved
  policies, training keys/checkpoints and complete output hashes.
- `graphs/`: split-specific node/provenance CSVs and tensor-only bundles containing
  `x`, edges, labels, eligibility, and a separate attachment tensor. The model
  forward pass accepts only `x` and edges; attachment is not an input feature.
- `models/`: independent best/final state dictionaries, architecture, epochs,
  ordered inputs and source hashes; separate training metadata. These are
  inference checkpoints, not optimizer-resume checkpoints.
- `histories/`: all 200 training and validation losses.
- `predictions/`: six full-population keyed registries with eligibility,
  attachment/evidence provenance and explicit missing-prediction status.
- `metrics.csv`: 12 rows for two checkpoints, three splits and two cohorts.
- `reports/`: coverage, shared support, classification reports, reload checks,
  eight descriptive comparison rows and the exact matched keys.
- `provenance/`: resolved config, all three source manifests, ordered features,
  canonical observation masks, three implementation snapshots, environment,
  seed/device, source hashes, git state and scientific limitations.

Later stages must load an explicit run manifest, retain graph separation and
honor the documented attachment semantics. No support-node metadata, graph
features, GAT/GraphSAGE or fusion gates were introduced. Notebooks 06 onward and
all prior implementations/configurations remain unchanged. Generated runs remain
local under the existing Git-ignored versioned-results directory.

## Reproduction and validation

From the repository root:

```powershell
.\env\Scripts\python.exe -m unittest discover -s tests -p test_graph_tabular.py -v
.\env\Scripts\python.exe scripts/validate_graph_tabular.py --write-executed
```

Run All in the notebook also works from the root or `analysis/` with the existing
project kernel and pinned Phase-2/3/4 outputs available. Execution creates a new
run directory; it does not overwrite prior experiments.

`scripts/validate_graph_tabular.py --verify-run RELATIVE_RUN_PATH` independently
verifies an existing run without retraining or starting Jupyter. Verification
reconstructs all attached features, checks sidecar provenance and hashes, reloads
both checkpoints, reproduces their recorded validation losses, and recomputes
predictions, metrics and exact-cohort comparisons.

## Verified execution and results

Completed run:
`results/versions/graph_tabular_v2/20260916T120656079973Z_8609c94f/`.

- Six targeted regression tests passed. They compare the refactored architecture
  numerically against the archived notebook's class definition, verify feature
  order/key alignment and support-slot semantics, reject held-out tabular rows,
  retain users with missing modalities, check reproducible training and independent
  checkpoint snapshots, and reject changes to approved policies.
- Fresh-kernel execution passed all 19 cells and all 200 training epochs.
- Both saved checkpoints reproduced their keyed predictions exactly and their
  recorded validation losses within verification tolerance.
- All three saved graph/feature bundles matched reconstruction from the pinned
  Notebook-04 and Phase-2 artifacts, including attachment provenance.
- Verified 35 output artifacts, six keyed prediction files, 12 metric rows and
  eight descriptive comparison rows on exactly 42 validation and 41 test keys.
- All 257 protected prior files remained unchanged, including raw/legacy data,
  prior versioned outputs, other notebooks, backups, preprocessing implementations
  and their configurations.
- Best validation loss occurred at epoch **200**. Best and final checkpoints are
  saved separately; they legitimately contain the same weights in this run.

Descriptive test results on the identical 41-user cohort (35 humans, six bots):

| Model | Bot precision | Bot recall | Bot F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|
| Tabular LR | 0.307692 | 0.666667 | 0.421053 | 0.852381 | 0.689712 |
| Tabular MLP | 0.272727 | 0.500000 | 0.352941 | 0.790476 | 0.540288 |
| Graph-only | 0.162162 | 1.000000 | 0.279070 | 0.595238 | 0.238547 |
| Graph+Tabular | 0.294118 | 0.833333 | 0.434783 | 0.871429 | 0.763305 |

These are observed descriptive results, not a controlled ablation or a claim of
statistically established improvement. No hyperparameters, architecture,
thresholds, feature definitions or approved conditions were changed in response
to the results. The full 144-user test registry retains the 103 unsupported users
with explicit null predictions; graph-branch coverage is 41/144.

No manual training is required to complete this phase. To verify the saved run:

```powershell
.\env\Scripts\python.exe scripts/validate_graph_tabular.py --verify-run results/versions/graph_tabular_v2/20260916T120656079973Z_8609c94f
```

Fresh-kernel verification ran outside the Codex filesystem sandbox because
Jupyter's Windows connection-file permission setup requires it on this machine.
