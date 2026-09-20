# Notebook 08: leakage-safe tabular semi-supervised experiment

## Scope

The historical Notebook 08 is retained in
`analysis/archive/08_semi_supervised_learning_pre_v2.ipynb`. Its results are not
valid held-out semi-supervised results: it combines all 959 labeled accounts in
student training, substitutes profile descriptions for the teacher's tweet
embeddings, uses a global graph mapping incorrectly, selects by training loss,
and overwrites the supervised checkpoint.

Phase-2 establishes that the 18,331 true unlabeled users have **zero retained
tweet-text records**. The accepted full-fusion teacher therefore has no valid
unlabeled cohort. This replacement is deliberately a tabular-only experiment,
not an implementation or claim of semi-supervised multimodal fusion.

## Fixed protocol

- The teacher is the pinned Notebook-03 `tabular_mlp` `last` checkpoint, whose
  primary status, architecture, feature order, training keys and frozen-split
  predictions are verified before pseudo-labeling.
- All Phase-2 tabular values use the exact 40-column manifest order and original
  train-fitted transforms. No preprocessing is fitted on labeled subsets or
  unlabeled data.
- Pseudo-label inference is strictly on 18,331 identity-disjoint unlabeled users.
  Validation/test users and labels cannot enter inference, selection or training.
- Threshold acceptance uses symmetric confidence >= 0.95. The complete pool is
  exported without deletion. It contains 13,782 accepted human and 195 accepted
  bot labels in this pinned run.
- The approved class-balanced selection keeps every 195 accepted bots and the
  top 195 accepted humans, ranked by confidence descending then `user_key`
  ascending. It is deterministic; selection seed 42 and no-sampling policy are
  stored in configuration/provenance.
- The pseudo student and labeled-only continuation control begin from identical
  teacher weights and run 50 extra epochs with the exact Notebook-03 tabular
  MLP architecture (40->64->32->1, BatchNorm, ReLU, dropout 0.3), AdamW,
  batch size 32, lr 0.001 and weight decay 0.0001.
- The student combines each labeled loss with confidence-weighted pseudo loss
  multiplied by the approved coefficient 0.5. Paired deterministic pseudo batches
  prevent the 18K pool from dominating only through population size.
- Checkpoint selection is best validation loss. Test data remains evaluation-only.

## Interpretation

The comparison has three models: supervised teacher, labeled-only continuation
control, and pseudo student. They evaluate identical frozen validation/test
populations. The control separates pseudo-label influence from the effect of
additional optimization, but neither continuation is a new untouched test study.
Confidence is not a calibration guarantee. One seed, small bot counts, human
annotation uncertainty and already-inspected held-out data limit conclusions.

Temporal values remain the Phase-2 transformed observations/placeholders, with
observation masks exported. No full timeline is inferred and no user is excluded
because temporal values are unavailable. Description text is not substituted for
missing tweets. A later description-based or missing-modality multimodal method
must be a separately designed and named experiment.

## Outputs and verification

Every run creates `results/versions/semi_supervised_v2/` with full pseudo-label
pool and selection status, separate control/student best and last checkpoints,
histories, keyed full-population predictions, computed metrics, class reports,
training-population comparison, frozen input manifests, observation masks,
source/config/git provenance and artifact hashes. Teacher, control and student
checkpoints remain separate; no legacy file is overwritten.

```powershell
.\env\Scripts\python.exe -X utf8 -m unittest discover -s tests -p test_semi_supervised.py -v
.\env\Scripts\python.exe -X utf8 scripts/validate_semi_supervised.py --write-executed
.\env\Scripts\python.exe -X utf8 scripts/validate_semi_supervised.py --verify-run results/versions/semi_supervised_v2/RUN_DIRECTORY
```

Jupyter fresh-kernel execution needs the Windows connection-file permission setup
outside the filesystem sandbox on this machine.

## Verified result (2026-09-20)

Run: `results/versions/semi_supervised_v2/20260920T104058299599Z_da452759`.
Fresh-kernel execution passed all 14 cells and preserved all 415 protected prior
files. Verification checked 34 hashed artifacts, four checkpoint reloads, nine
keyed prediction files and nine recomputed metric rows. Three focused tests pass,
including deterministic cap selection under confidence ties and reordered rows.
Run provenance includes Python, operating-system and package versions.

The teacher and labeled-only control train on 671 annotated users (540 humans,
131 bots). The student adds 195 pseudo-humans and 195 pseudo-bots, totaling 1,061
unique training users. The `human_labeled` population-table column means
human-annotated users of both classes. All 18,331 pseudo-label candidates remain
in the full pool: 13,782 humans and 195 bots pass confidence, 13,587 accepted humans
are excluded by the cap, and 4,354 candidates fail confidence. No held-out user
enters pseudo-label selection or training.

Both loss terms retain the positive-class weight from labeled training data;
the selected pseudo population is balanced by count, not by effective loss weight.
Pseudo batches cycle to pair with labeled updates and also update BatchNorm
running statistics. Confidence weights multiply per-example pseudo losses before
averaging; the resulting term has coefficient 0.5.

All models below evaluate the same 144 test users, including 28 annotated bots.

| Model | Bot precision | Bot recall | Bot F1 | ROC-AUC | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Supervised teacher | 0.3333 | 0.5000 | 0.4000 | 0.7365 | 0.4324 |
| Labeled-only continuation | 0.2683 | 0.3929 | 0.3188 | 0.7315 | 0.4292 |
| Pseudo-label student | 0.3409 | 0.5357 | 0.4167 | 0.7482 | 0.4403 |

The student's small observed gain is descriptive; this single-seed result does
not establish statistical significance or multimodal semi-supervised improvement.
