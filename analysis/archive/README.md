# Historical notebooks

`01_dataset_exploration_and_audit_legacy.ipynb` preserves the working copy of
notebook 01 immediately before the Phase-1 refactor, including outputs and the
existing kernel metadata. It records earlier experiments and export logic.

Do not include this archive in sequential pipeline execution. In particular, its
export cells write legacy preprocessing files. The active notebook 01 is an
audit-only notebook; notebook 02 will own production data preparation.

`02_data_preprocessing_and_feature_engineering_pre_v2.ipynb` preserves the complete
working copy immediately before the Phase-2 refactor. Its paths and exports are
historical; do not execute it as part of the versioned pipeline.

`03_baseline_models_pre_v2.ipynb` is the byte-for-byte backup before the Phase-3
compatibility refactor. It preserves the original architectures, outputs, and
legacy export logic. Do not run its overwrite/export cells in the new pipeline.
