# Phase 1: audit-only dataset exploration

## Scope

The active notebook is `analysis/01_dataset_exploration_and_audit.ipynb`.
This refactor changes dataset exploration only. It does not regenerate processed
data, final data, splits, embeddings, checkpoints, predictions, or model results.
Notebook 02 and later notebooks are not modified in this phase.

The full previous notebook is preserved under `analysis/archive/`, including its
existing outputs. The older `DATASET_AUDIT_AND_PREPROCESSING.md` remains historical
documentation and has not been rewritten to imply that Phase 2 is already fixed.

## Problems addressed

| Previous behavior | Scientific impact | Current behavior |
|---|---|---|
| Audit exported graphs and a pseudo-label pool | Mixed descriptive analysis with production preprocessing | No data/model exports; only notebook outputs |
| Relative paths assumed one launch directory | Fresh executions depended on the working directory | Detect project root from root or analysis directory |
| IDs loaded as floating point | Additional precision loss could obscure identity problems | Preserve raw account-ID tokens as strings; report scientific notation |
| Case-sensitive overlap checks | Inconsistent denominators and account matching | Diagnostic normalized keys with collision and malformed-handle reports |
| All annotation categories entered imbalance summary | Binary task imbalance was obscured | Report all categories and binary population separately |
| Timing and other behavior mixed in temporal coverage | Claimed temporal availability was ambiguous | Seven timing fields separated from other behavior |
| Profile presence assumed; graph presence conflated | Missing-modality claims were poorly defined | Finite-value availability and separate graph owner/list/neighbor flags |
| Larger user table treated as unlabeled | Annotated accounts could be counted as pseudo-label candidates | Set-based reconciliation excluding every annotated category |
| Mutable source DataFrames accumulated audit features | Later schema counts no longer represented raw inputs | Separate diagnostic views and raw-frame integrity checks |
| No formal provenance/limitations section | Results were difficult to reproduce and overinterpretation was possible | File hashes, environment, Git state, limitation table, handoff checks |

## Audit semantics

- Normalized handles are diagnostic keys, not a new production identity policy.
- Repeated tweet authors are expected. Repeated account records and conflicting
  annotations are reported separately.
- Tweet IDs are not compared with account IDs.
- Account-ID spelling differences are not proof of distinct accounts; lost
  upstream precision cannot be restored by changing a dtype.
- Coverage aggregates duplicate records with `any` only to measure whether an
  observation exists. It does not choose a canonical record.
- Text availability distinguishes a raw tweet row, nonblank raw tweet text,
  description, and the provided cleaned-description field.
- Profile/temporal availability means at least one finite value in the listed
  group. Complete temporal coverage is reported separately. Neither establishes
  completeness of historical observations.
- Graph-list parsing measures observations only. No edges, node indices, or graph
  features are generated, and missing lists are not interpreted as zero degree.
- Full-data quality summaries do not authorize full-data feature selection or
  fitting of transformations.
- Historical exclusions are read only for reconciliation and explicitly marked
  as unavailable if the optional exclusion file does not exist.

## Formal limitations

The notebook contains a limitation matrix covering sample size and imbalance,
annotation uncertainty, sampling/collection bias, Persian and multilingual text,
unequal modality availability, incomplete networks, limited temporal evidence,
identity ambiguity, undocumented feature derivations, overlapping/related users,
historical test reuse, and offline/compute constraints. Each limitation states its
scientific consequence and the required downstream treatment.

## Compatibility boundary

The current notebook 02 reads historical graph statistics and edges from
`final_data/graph/`. Those files remain untouched. Notebook 01 checks their
presence and required columns but no longer creates them.

This preserves the existing checkout's inputs. It does **not** establish a
raw-to-final clean rebuild for the old notebook 02. Phase 2 must generate these
artifacts from raw observations itself, and own canonical users, feature policy,
modality masks, frozen splits, and training-fitted transforms.

Optional final labeled/unlabeled tables are compared by unique key sets, including
duplicate-key checks. Missing historical artifacts do not prevent raw-data audit.
No later notebook must depend on notebook 01's Python variables.

## Verification

Use the project Python environment and its registered Jupyter kernel:

```powershell
.\env\Scripts\python.exe scripts/validate_dataset_audit.py
.\env\Scripts\python.exe scripts/validate_dataset_audit.py --from-analysis
```

The validator executes all cells in a fresh kernel, checks identity/population
invariants, coverage denominators, graph-parser cases, and legacy handoff schemas.
It hashes raw/processed/final data, models, results, and other notebooks before
and after execution and fails if any protected file changes. Notebook outputs
are saved only with `--write-executed` after successful verification.

There is no training or hyperparameter selection in this audit, so a random seed
does not affect its results. Execution timestamp, platform, and Git working-tree
state are expected to vary between runs; source hashes and measured counts should
remain stable for unchanged inputs.

### Verification result for the current source snapshot

- Fresh-kernel execution passed from both the repository root and `analysis/`.
- The saved notebook contains 31 cells, six inline figures, and no error outputs.
- All 103 protected data/artifact/notebook files were byte-identical before and
  after execution.
- Reconciliation found 1,103 annotated accounts, 959 binary accounts, and 19,435
  unique accounts in the larger source. Removing overlap with all annotated
  categories yields 18,332 candidates; the recorded historical exclusion accounts
  for the final difference to 18,331.
- Existing final labeled/unlabeled key sets matched the audited populations;
  historical graph artifacts passed required-column checks.
- Notebook 02 was not executed. Its complete raw-data rebuild remains Phase-2 work.

The raw binary annotated source has finite temporal statistics for 772 of 959
users (140/187 bots and 632/772 humans). This differs from the broader availability
in historical final tables. Phase 2 must document any cross-source enrichment and
its conflict-resolution rules; raw-source coverage must not be confused with
post-preprocessing coverage.
