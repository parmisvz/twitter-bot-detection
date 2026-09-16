# Phase 2 — reproducible preprocessing v2

## What changed

Notebook 02 now orchestrates a continuous raw-to-final pipeline implemented in
`src/preprocessing.py`. Its pre-refactor working copy, including the user's kernel
metadata and old outputs, is preserved byte-for-byte in
`analysis/archive/02_data_preprocessing_and_feature_engineering_pre_v2.ipynb`.

The four raw CSVs, `configs/preprocessing_v2.json`, and
`configs/frozen_user_split_v1.csv` are the data/configuration inputs. Runtime does
not load old `final_data` tables, graphs, masks, or preprocessing objects. The split
configuration is an explicit copy of the previous assignments, with the original
path and hash recorded as provenance. Invalid assignments stop execution instead
of triggering a new split.

Numeric IDs are metadata only. Loading their existing tokens as strings cannot
recover precision lost upstream. Every join, split, and output alignment uses
`user_key` only.

## Scientific decisions and definition changes

- Retain the historical 40-feature order, log1p choices, and robust scaling.
  This is a fixed historical design, not a new feature-selection experiment.
- Account features come from the raw 19K source for both labeled and unlabeled
  users; annotations come only from the labeled source. Source comparisons and
  per-user temporal provenance explain the coverage difference between sources.
- Keep profile, behavioral, and temporal groups disjoint. Text and graph have
  separate prepared artifacts and availability masks.
- Preserve nonbinary annotation exclusions and the previously documented corrupt
  account exclusion. Missing modalities are never an account-exclusion rule.
- Unknown description/name values now remain unknown in derived features rather
  than becoming zero. Observed values retain the original formulas.
- Followers/friends retains its original positive-denominator formula and zero-
  denominator fallback. A missing or invalid denominator now produces a missing
  ratio, rather than silently falling through to the follower count.
- Negative measurements in declared nonnegative fields, infinities, unknown
  booleans, and parse failures become explicitly reported missing observations.
  They are not silently clipped or treated as valid zeros.
- Preserve the existing text decontamination and tweet-cap choices. Retain every
  user even when no usable tweet remains. Descriptions and tweets stay separate.
- Build directed graphs only from observed lists. Deduplicate model edges by
  endpoints and relation, retain owner provenance separately, and report invalid
  tokens/self-edges. No inferred edges are introduced.

All these decisions are explicit in the versioned policy, registry, and notebook.
The new matrices are not silently substituted into historical checkpoints.

## Temporal missingness contract

Canonical tables keep NaNs. Temporal features are never statistically imputed.

`has_temporal` means at least one finite, valid observed timing statistic;
`temporal_complete` means all seven; `temporal_observed_count` records the count.
Per-feature `observed__*` indicators describe availability before any numerical
placeholder. The per-user provenance table distinguishes annotated-source and
canonical-source observations.

Temporal scalers fit only observed training values. A missing value receives zero
only in the separate scaled numerical matrix, alongside its observation mask.
That is a storage placeholder, not an estimated time statistic. A completely
missing temporal branch must be masked out by downstream models. Features with
no training observations are disabled, recorded in the registry, and retain their
schema position; held-out data cannot establish their transformations.

Non-temporal numeric filling uses training medians; binary filling uses the
training mode. Both also retain observation masks. Training, validation, test,
and unlabeled users use the same fitted preprocessor.

## Graph protocol

Gold-user graphs are centered on observed owners from the relevant split and
exclude gold endpoints from other splits. Support/external users may be shared,
so this is not a fully node-disjoint split. The unlabeled-owner graph excludes
validation/test gold endpoints.

`has_graph_owner_record`, `has_graph_owner`, `has_graph_node`, and `has_graph`
describe different evidence. `has_graph` requires an observed owner neighborhood
and retained incident evidence in the permitted graph. Graph-unavailable users
remain in all user tables. Global edge exports are audit/provenance artifacts,
not permission to train on the full graph. Notebook 04 must honor this contract.

## Output layout and preservation

Each run exports to `final_data/versions/preprocessing_v2_<fingerprint>/`.
The fingerprint includes raw/configuration/split/code hashes and dependency
versions. Identical artifacts are accepted on rerun; different existing content
is never overwritten. The final manifest marks a complete run. A failed run may
leave an incomplete version without a manifest; do not consume it.

| Location inside a version | Purpose |
|---|---|
| `users/` | Canonical users, gold binary users, true unlabeled users; original missingness retained |
| `splits/user_split_assignment.csv` | Preserved supervised assignments |
| `splits/users/` | User tables for train, validation, test, and unlabeled populations |
| `splits/tabular/` | Aligned finite feature matrices; feature order comes from manifest |
| `splits/masks/` | Per-feature observation and modality masks |
| `splits/text/` | Prepared tweets and separately sourced descriptions |
| `splits/graph/` | Permitted directed graph edges and owner provenance |
| `graph/` | Global observed graph evidence for audit only |
| `provenance/` | Temporal availability/source records by user |
| `audit/` | Source, validity, exclusion, coverage, transformation, and integrity reports |
| `preprocessing/` | Fitted object and readable fitted statistics |
| `config/` | Resolved policy and feature registry |
| `manifest.json` | Input/output hashes, runtime, feature order, and downstream contract |

Generated versions remain on disk and are Git-ignored to avoid duplicating large
datasets in commits. Source code, configuration, frozen assignments, backup, and
verification instructions are tracked. No old artifacts are deleted.

## How to run manually

No manual action is required while the refactor is being verified. To reproduce
later, run from the repository root using the existing project environment:

```powershell
.\env\Scripts\python.exe -m unittest discover -s tests -p test_preprocessing.py -v
.\env\Scripts\python.exe scripts/validate_preprocessing.py --write-executed
```

The second command executes notebook 02 in a fresh kernel from `analysis/`,
protects legacy files with before/after hashes, verifies all new artifact hashes,
reloads the fitted preprocessor, and saves notebook outputs only after success.
Use `--from-root` to check execution from the repository root. The default kernel
is the existing `bot_detection_env`; `--kernel NAME` supports another registered
kernel containing the project dependencies.

Alternatively, open notebook 02, select the project environment, restart the
kernel, and Run All. It supports launch from either root or `analysis/`. No model
download, API access, or model training is required. Do not run archived notebooks
as pipeline stages.

## Notebook 03 compatibility

Notebook 03 has not been modified or trained. It must explicitly choose a completed
version, load the manifest's feature order, and align masks by user_key. Additional
metadata and masks must not accidentally become predictive features. Any deliberate
use of missingness indicators should be declared in the experiment configuration.
The old baseline paths still refer to preserved historical data; selecting v2 is
part of Phase 3, not an implicit side effect of this refactor.

Limitations remain: imported feature formulas/windows are incompletely documented,
modalities are incompletely observed, labels are human judgments, the gold set is
small and imbalanced, and the historical test set has already been inspected.

## Verified run

Completed output: `final_data/versions/preprocessing_v2_e861084d1a722a21/`.

- Five targeted tests passed, including a raw-only temporary fixture with absent
  modalities, invalid split rejection, observed-only temporal fitting, and refusal
  to overwrite different versioned content.
- Fresh-kernel verification passed all 22 notebook cells and 34 integrity checks.
- All 60 output artifacts passed hash/reload verification. Reloaded preprocessing
  reproduced the saved matrices for train, validation, test, and unlabeled users.
- A repeated run produced identical versioned content. All 105 protected legacy
  files remained unchanged during the final verification.
- Preserved 959 binary users (772 human, 187 bot), 18,331 true-unlabeled users,
  and the original 671/144/144 split assignments.
- Temporal coverage: 772/959 in the annotated raw source; 959/959 in matching raw
  19K observations; 18,331/18,331 in the final unlabeled population. No estimated
  temporal values were introduced. Synthetic missing-temporal users were retained.
- Text counts match the historical policy: 12,976/2,618/3,006 retained tweets.
  All 144 test users remain, including the one without retained tweet text.
- Training graph has 60,924 edges after removing 12 duplicate endpoint/relation
  entries; validation/test retain 13,006/12,536 edges. Every split has zero
  forbidden cross-split gold nodes.
- Notebook 03 and all model checkpoints/results remain unchanged.
