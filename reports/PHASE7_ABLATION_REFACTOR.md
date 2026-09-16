# Notebook 07: cohort-controlled branch-removal study

## Approved scope

The original notebook is preserved in `analysis/archive/07_ablation_study_pre_v2.ipynb`.
The three original pairwise architectures are retained:

| Variant | Branch widths | Classifier |
|---|---|---|
| Tabular + Text | 40→64 and 768→128 | 192→64→2 |
| Text + Graph statistics | 768→128 and 3→32 | 160→64→2 |
| Tabular + Graph statistics | 40→64 and 3→32 | 96→64→2 |

All use ReLU and dropout 0.3, weighted cross-entropy, AdamW lr 0.001,
weight decay 0.0001, batch 32, 200 epochs, arithmetic mean batch losses,
minimum-validation-loss checkpoint selection, and threshold 0.5.
The graph branch is an MLP over degree statistics, not a GNN.
Tabular+Graph here is distinct from Notebook-05's GCN experiment.

## Scientific correction

Historical training populations differed by variant and test cohorts ranged
from 139 to 143 users. Merely matching test users would not control training
population differences. The approved primary study fixes training/validation/test
keys to Notebook-06's 193/42/40 complete-modality cohort for all variants.
Training labels are 162 human / 31 bot, validation 35/7, and test 34/6.

The full reference is pinned to Notebook-06 run
`results/versions/multimodal_v2/20260916T175232735080Z_1dd8a17c/`.
The loader verifies matching input hashes, feature order, settings, ordered cohort
keys and training-only class weights, and reloads the selected reference model to
reproduce its keyed predictions. No full model is chosen using test scores.

The reference and every variant use the same seed and deterministic loader
policy. Retained layers do not necessarily have identical initial weights or
dropout draws across differently shaped models. Branch removal reduces parameter
count; exported parameter counts make this explicit. The study controls cohorts
and training policy but is not capacity-controlled. One seed and six test bots
cannot establish robust significance or generalization. Historical test users
have already been inspected.

## Data contract and missingness

- Exact Phase-2 manifest feature order; transforms remain fitted on the original
  training population. No refitting on the 193-user subset.
- Registered frozen text embeddings, keyed by user_key; unresolved historical
  encoder revision/provenance limitations remain documented.
- Notebook-04 degree tensors checked against split-local Phase-2 reconstruction;
  no global graph input, held-out gold topology exposure to training, inferred
  edges, new graph features or support metadata.
- Graph membership, owner evidence and has_graph remain distinct. Support
  identities can overlap between split graphs; these are not fully node-disjoint.
- Full canonical registries retain 671/144/144 users. Users outside the fixed
  cohort receive null predictions, with separate indicators for cohort membership
  and required-modality availability. The status distinguishes design exclusion
  from a missing modality actually required by the variant.
- Missing temporal statistics never define exclusion. Observation masks and
  Phase-2 placeholders remain unchanged; no new imputation occurs.
- All current canonical gold users have seven observed temporal statistics, not
  complete timelines or documented original measurement windows.
- Removing the tabular branch removes profile, behavioral AND temporal features;
  it must not be described as a metadata-only ablation.

## Implementation and artifacts

`configs/ablation_v2.json` pins inputs and policy. `src/ablation.py` implements the
three retained architectures, shared training, explicit registries, reference
validation, exports and reload verification. Prior pipeline modules are unchanged.
Notebook 07 presents the protocol, separate experiment cells, computed comparisons,
figures and downstream handoff. `tests/test_ablation.py` checks archive-equivalent
logits, independence from removed inputs, missingness/design statuses and seeded
training with independent checkpoint snapshots.

Each execution creates a unique `results/versions/ablation_v2/` run containing:

- Six best/final inference checkpoints and three 200-epoch histories.
- Eighteen keyed prediction registries and 36 computed metric rows.
- Eight comparison rows: full fusion plus three removals, on validation and test.
- Deltas calculated as variant minus full fusion, and exact matched user keys.
- Classification reports, confusion matrices, coverage and parameter counts.
- Validation/test PNG and PDF charts computed from matched results.
- Input manifests, source snapshots, config, observation masks, environment,
  seed/device/git provenance and artifact hashes.

No legacy files or earlier versioned outputs are replaced. These checkpoints
support inference, not optimizer-state training resumption. Notebook 08 and later
stages remain untouched. Final evaluation must explicitly consume the new manifest.

## Reproduction

```powershell
.\env\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_ablation.py -v
.\env\Scripts\python.exe -X utf8 scripts/validate_ablation.py --write-executed
.\env\Scripts\python.exe -X utf8 scripts/validate_ablation.py --verify-run results/versions/ablation_v2/RUN_DIRECTORY
```

Jupyter's Windows connection-file setup requires fresh-kernel execution outside
the filesystem sandbox on this installation. No encoder download/inference is needed.

## Verified execution

Run: `results/versions/ablation_v2/20260916T180713858800Z_cd3b2070/`.

- Three regression tests passed, covering all three architectures, removed-input
  independence, explicit registry statuses, repeatability and checkpoint independence.
- Fresh kernel passed all 21 cells and 200 epochs per variant (600 total).
- All 334 protected prior files remained unchanged.
- Six pairwise checkpoints and the pinned full-fusion reference reloaded correctly.
- Verified 61 artifacts, 18 prediction registries, 36 metrics and eight comparison rows.
- Matched keys are exactly 42 validation / 40 test; training keys match all 193 reference users.
- Best epochs: Tabular+Text 35, Text+Graph statistics 37, Tabular+Graph statistics 198.
- Backup equals the original Git notebook including outputs.
- Saved PNG charts were visually inspected; labels and limitations are readable.

Observed results on the shared 40-user test cohort (34 humans, six bots):

| Model | Bot F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|
| Full fusion reference | 0.400000 | 0.877451 | 0.718750 |
| Tabular + Text | 0.500000 | 0.872549 | 0.695489 |
| Text + Graph statistics | 0.260870 | 0.534314 | 0.230181 |
| Tabular + Graph statistics | 0.400000 | 0.862745 | 0.710039 |

Full fusion does not improve every metric in this run. These findings describe
this fixed-cohort, single-seed branch-removal experiment, not statistically
established modality effects. No architecture or threshold was changed in
response to the test results. All 144 test users remain in each registry;
104 users receive explicit design-exclusion or missing-required-modality statuses.

No manual training is needed for this completed phase.

Standalone verification passed in a separate Python process, including all six ablation checkpoints, the full reference, and all 61 output artifacts.
