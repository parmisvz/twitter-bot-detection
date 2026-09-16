# Phase 3: reproducible baseline compatibility pass

## Scope

Notebook `analysis/03_baseline_models.ipynb` now consumes the accepted Phase-2
manifest and a separately registered embedding bundle. No architecture, feature
definition, split assignment, threshold or hyperparameter search was introduced.
The complete original notebook, including outputs, was copied before editing to
`analysis/archive/03_baseline_models_pre_v2.ipynb`.

The notebook has 20 cells covering configuration, coverage, text provenance,
four separate training experiments, computed evaluation, reload verification,
and downstream handoff. Implementation is shared through `src/baselines.py`.

## Scientific problems and corrections

| Previous behavior | Scientific impact | Compatibility correction |
|---|---|---|
| Absolute workstation path and legacy split directories | Input version ambiguous | Explicit Phase-2 manifest path and pinned SHA256 in `configs/baselines_v2.json` |
| Select all columns except three exclusions | New metadata/masks could become unintended predictors | Select exactly the manifest feature order; no additional mask inputs |
| Informal key/label alignment | Row-position mistakes can corrupt labels | Validate canonical unique `user_key`, exact populations, disjoint splits and labels; join by key |
| Missing tweet user absent from text evaluation outputs | Different evaluation denominators hidden | Full user registry, null unsupported predictions, explicit coverage and matched-cohort results |
| No model persistence | Results cannot be independently reproduced from checkpoints | Save both LR estimators and both final/best snapshots of each MLP |
| Mutable best-state reference | Text model did not retain the intended validation checkpoint | `deepcopy` snapshots; primary best-validation text model, separately retained final model |
| Unseeded neural training | Random trajectory not reproducible | Seed Python/NumPy/PyTorch per experiment; deterministic CPU execution, one torch thread |
| Manually overwritten text-MLP metrics | Reported numbers may not describe trained weights | Compute all metrics from current predictions and verify again from disk |
| `PR_AUC` actually computed average precision | Ambiguous metric interpretation | Name it `average_precision`; preserve the estimator |
| Results repeatedly overwritten | Experiment provenance lost | Unique, non-overwriting run directories and artifact hashes |

These changes improve traceability and validity; they are not expected to
guarantee higher scores.

## Methodology preserved

- Both logistic regressions: `class_weight="balanced"`, `max_iter=1000`,
  `random_state=42`; effective sklearn parameters and iteration counts saved.
- Tabular MLP: 40 → 64 → 32 → 1, first-layer BatchNorm, ReLU, dropout 0.3;
  weighted BCE; AdamW lr 0.001, weight decay 0.0001; batch 32, 50 epochs.
- Text MLP: 768 → 256 → 64 → 1, same normalization/activation/dropout pattern;
  weighted BCE; AdamW lr 0.001, original implicit weight decay 0.01;
  batch 32, 50 epochs.
- Positive-class weights come only from each experiment's training users.
- Epoch loss is the original arithmetic mean of batch losses, including the
  smaller final batch. No early stopping or threshold tuning.
- Tabular MLP primary checkpoint remains epoch 50. Text MLP primary checkpoint
  is the intended minimum-validation-loss snapshot. In the validated run this
  is epoch 2; tabular best-validation snapshot is epoch 9 but remains secondary.
- Decision threshold stays 0.5. Tabular LR retains sklearn `.predict()`; the
  other three retain the original probability `>= 0.5` convention.
- Frozen XLM-R representations retain 128-token truncation, batch 16 extraction,
  attention-mask mean token pooling and unweighted mean tweet pooling per user.
  No text encoder fine-tuning, description substitution, or new embeddings.

## Inputs and limitations

Selected Phase-2 version: `preprocessing_v2_e861084d1a722a21`.
Manifest SHA256: `594a2defef59d663ef171020b7cd1ba75aa321c30e7a7d09b57b560563c109c9`.

The runtime reads only registered Phase-2 and embedding artifacts. Legacy paths
exist solely in the explicit one-time migration configuration. Registration
checks all ordered `(user_key, text_model)` records against Phase-2; checks finite
768-dimensional vectors and exact user keys; and recomputes cached user means
from cached tweet vectors. The new bundle uses CSV key sidecars rather than
pickled NumPy object keys. The sole legacy pickle read is confined to migration
of the trusted locally generated key files.

**Embedding limitation:** historical encoder revision and original row-to-vector
provenance were not recorded. Current row equality and pooling checks cannot
recover those facts. The bundle pins the actual cached representations, not a
verified historical Hugging Face revision. Fresh-kernel verification retrains
the four classifiers; it does not rerun XLM-R inference. Registration never
silently downloads or replaces an encoder. A future encoder extraction study
must be separately versioned and revision-pinned.

| Split | Users | Humans / bots | Tweet available | Temporal statistics available | Graph available |
|---|---:|---:|---:|---:|---:|
| Train | 671 | 540 / 131 | 671 | 671 | 193 |
| Validation | 144 | 116 / 28 | 144 | 144 | 42 |
| Test | 144 | 116 / 28 | 143 | 144 | 41 |

All canonical users remain. Missing tweets produce no invented vectors or text
predictions. Having a description is insufficient for a tweet-based baseline.
Graph masks are retained for analysis and never gate baseline inclusion.

Temporal statistics are imported observations, not complete timelines. Their
upstream windows remain unverified. The canonical 19K source supplies all seven
temporal values for the current gold users, unlike the less-complete annotation
source. Missing values in other compatible inputs retain the Phase-2 zero
placeholder with observation/modality masks; no further imputation or user
removal occurs here. Masks are saved for interpretation, not added to the
unchanged baseline input dimension. These classifiers do not learn a new
missingness gate in this pass.

The Phase-2 `description_length` correction changes some tabular values relative
to the legacy preprocessing. Historical MLPs were unseeded and the text checkpoint
was defective. Therefore new scores must not be presented as exact reproductions
of historical results. The previously evaluated test set is not newly untouched.

## Outputs and downstream contract

Validated run:
`results/versions/baselines_v2/20260916T103951409018Z_7fa575b7/`.

- `manifest.json`: complete artifact hashes, input-manifest hashes, feature order,
  model metadata, checkpoint paths and primary selections.
- `models/`: two LR estimators, four MLP snapshots, and per-model metadata with
  actual training keys/class counts. Torch files contain architecture and state
  dictionaries; they are inference checkpoints, not optimizer-resume checkpoints.
- `histories/`: 50-epoch loss CSVs for each MLP; LR iteration/parameter/warning
  JSONs. sklearn does not expose a full LR epoch-loss history.
- `predictions/`: 18 keyed files (six checkpoints × three splits), retaining all
  canonical users and masks. Missing text predictions remain null and identified.
- `metrics.csv`: 36 computed rows (six checkpoints × three splits × two cohorts).
  Primary and secondary checkpoint rows are distinguished explicitly.
- `reports/`: coverage, confusion matrices, classification reports, and reload
  comparisons. `average_precision` is not trapezoidal PR-AUC.
- `provenance/`: resolved config, both input manifests, source snapshot, source
  and notebook-source hashes, package/platform/seed/device information, git
  state at execution, observation masks, and methodological limitations.

`all_users` preserves the split denominator; scores refer to supported users and
coverage is explicit. `tweet_available` compares all four models on the same
users (143 on test). Training metrics are apparent fit metrics. Later evaluation
must load an explicit run manifest and its declared primary checkpoints; do not
select a latest directory or the highest test score. Phase-2 remains the data
source for notebook 04; baseline predictions are not new training features.

Generated bundles/runs are retained locally and ignored by git. Configuration,
code, notebook outputs, backup and this report are committed. No legacy exports
or Phase-2 files were changed.

## Verification and commands

Fresh kernel passed all 20 notebook cells using `bot_detection_env`:

- All four models trained using their original settings; both MLPs ran 50 epochs.
- Six persisted checkpoints reloaded correctly.
- Eighteen keyed prediction files and 36 metric rows independently recomputed.
- Forty-four output artifacts verified by hash.
- Reloaded probabilities matched exactly in this CPU run (maximum difference 0).
- All 174 protected files remained unchanged, including Phase-2 outputs, legacy
  data/results/models, registered embeddings, archives and other notebooks.
- Four targeted regression tests passed: key alignment, artifact corruption,
  missing-modality retention/coverage, and independent checkpoint snapshots with
  repeatable seeded training.

No manual execution is required to complete this phase. For later reproduction
from the repository root:

```powershell
.\env\Scripts\python.exe scripts/register_baseline_embeddings.py
.\env\Scripts\python.exe -m unittest discover -s tests -p test_baselines.py -v
.\env\Scripts\python.exe scripts/validate_baselines.py --write-executed
```

Registration reuses the already verified bundle without writing to it. Run All
in notebook 03 is also supported, from the root or `analysis/` directory, after
registration. Every execution creates a new result directory.

To reload and verify the saved run without retraining or starting Jupyter:

```powershell
.\env\Scripts\python.exe scripts/validate_baselines.py --verify-run results/versions/baselines_v2/20260916T103951409018Z_7fa575b7
```

The validation script uses the local Jupyter kernel. On this Windows installation,
Jupyter's connection-file permission setup requires execution outside the Codex
filesystem sandbox; the requested fresh-kernel validation used that permission.
