# Notebook 06: reproducible multimodal compatibility experiment

## Scope and preserved methodology

The original notebook is backed up at `analysis/archive/06_multimodal_fusion_pre_v2.ipynb`.
The architecture remains tabular 40→64, frozen cached text 768→128, and degree
statistics 3→32, each with ReLU and dropout 0.3. Concatenation produces 224 inputs
for a 224→64→2 classifier with ReLU/dropout 0.3. The graph branch is an MLP, not a GNN.
Weighted cross-entropy uses only eligible training labels. AdamW uses lr 0.001,
weight decay 0.0001, batch 32 and 200 epochs. Loss history preserves arithmetic
mean batch losses. Minimum validation loss selects the primary checkpoint;
threshold remains 0.5. Best/final states are independent deep copies.

## Corrections and scientific impact

- Consume pinned Phase-2 tables with exact manifest feature order and no refitting.
- Validate registered XLM-R embeddings by keys, hashes and prepared text provenance.
  Historical encoder revision and original vector provenance remain incompletely known.
- Consume Notebook-04 split-local degree tensors after checking mappings and all
  tensors against Phase-2 reconstruction. Global graph artifacts are unused.
- Distinguish graph membership from `has_graph`; tweet availability is
  `has_tweet_text`, not the broader description-or-tweet `has_text` mask.
- Eligible populations are explicitly 193/42/40; full registries retain 671/144/144.
  Missing modalities produce explicit statuses and null predictions, never fabricated vectors.
- Use user_key alone. Metadata IDs, labels, mappings and masks never become predictors.
- Eliminate shallow checkpoint copies, stale hardcoded metrics, unkeyed predictions,
  implicit artifact selection, unseeded execution and overwriting historical results.
- Report average precision with its correct name; export confusion matrices and
  classification reports for both checkpoints and full/supported cohorts.

## Coverage and temporal strategy

Train: 162 humans + 31 bots. Validation: 35 humans + 7 bots.
Supported test: 34 humans + 6 bots. The graph-eligible test user without retained
text stays in the canonical registry with missing_tweet_text status.
All current gold users have seven observed temporal statistics in the canonical
source. This does not imply complete timelines or verified original measurement
windows. No new imputation or exclusion based on temporal availability occurs.
Phase-2 observation masks and zero storage placeholders retain their meaning;
the unchanged tabular MLP does not have a separate temporal encoder or learned
missingness gate. This is a restricted-coverage baseline, not the final
missing-modality-capable thesis framework.

Graphs isolate gold splits but share support identities. Degrees retain the
Notebook-04 unscaled redundant representation and reciprocal-edge multiplicity.
No graph weights or support metadata enter this model; there is no message passing.
Held-out cohort topology is available for feature extraction at evaluation;
this should not be represented as fully node-disjoint deployment.

## Comparisons and downstream handoff

Tabular LR/MLP, Text LR/MLP, Graph-only, Graph+Tabular and fusion are re-scored
on exactly the same 42 validation and 40 test keys. All seven rows per split
are descriptive comparisons, not controlled ablations: training populations,
architectures and optimizer settings differ. Notebook-05's 42/41 comparison is
preserved separately. Six test bots, one seed and previously inspected test data
limit inference; no claims of statistically established improvement are justified.

Notebook 07 still reads legacy result files; Notebook 08 still reads legacy
fusion checkpoints. They remain untouched and require explicit subsequent
manifest migration. Future attention, contrastive, temporal and missing-modality
changes must be separate experiments.

## Files and outputs

- `configs/multimodal_v2.json`: pinned inputs and original training settings.
- `src/multimodal.py`: loaders, original architecture, training, export and verification.
- `analysis/06_multimodal_fusion.ipynb`: executable research narrative.
- `scripts/validate_multimodal.py`: protected-artifact fresh kernel and standalone verification.
- `tests/test_multimodal.py`: identity/missingness, archived architecture equivalence,
  seeded training and independent checkpoint regression checks.

Each execution creates a unique `results/versions/multimodal_v2/` run with two
inference checkpoints (not optimizer-resume checkpoints), 200-epoch history,
six full keyed prediction registries, 12 metric rows, 14 comparison rows,
matched keys, classification reports, coverage, observation masks, source/config
snapshots, input manifests, environment/git provenance and artifact hashes.
Legacy files and prior versioned experiments are preserved.

## Reproduction

```powershell
.\env\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_multimodal.py -v
.\env\Scripts\python.exe -X utf8 scripts/validate_multimodal.py --write-executed
.\env\Scripts\python.exe -X utf8 scripts/validate_multimodal.py --verify-run results/versions/multimodal_v2/RUN_DIRECTORY
```

On this Windows installation Jupyter fresh-kernel execution needs connection-file
permission setup outside the filesystem sandbox. No download or encoder inference
is needed; the registered representations are deliberately reused.

## Verified execution

Run: `results/versions/multimodal_v2/20260916T175232735080Z_1dd8a17c/`.

- Three regression tests passed, including numerical equivalence with archived architecture.
- Fresh kernel passed all 17 cells and all 200 epochs.
- All 296 protected prior files remained unchanged.
- Both checkpoints reproduced their predictions exactly; recorded validation losses were reproduced.
- Verified 34 artifacts, six prediction registries, 12 metric rows and 14 descriptive comparison rows.
- Matched cohorts are exactly 42 validation / 40 test users; best epoch is 139.
- Original notebook backup matches the pre-refactor Git version including outputs.

Descriptive fusion test results: bot precision 0.285714, recall 0.666667,
F1 0.400000, ROC-AUC 0.877451, average precision 0.718750.
The full test registry retains all 144 users with 104 explicit unsupported nulls.
These results do not establish superiority over the other models.

No manual training is required for this completed compatibility pass.

Standalone verification also passed in a separate Python process, reloading both checkpoints and rechecking all 34 artifacts and matched comparisons.
