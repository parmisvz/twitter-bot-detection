# Historical notebooks

`01_dataset_exploration_and_audit_legacy.ipynb` preserves the working copy of
notebook 01 immediately before the Phase-1 refactor, including outputs and the
existing kernel metadata. It records earlier experiments and export logic.

Do not include this archive in sequential pipeline execution. In particular, its
export cells write legacy preprocessing files. The active notebook 01 is an
audit-only notebook; notebook 02 will own production data preparation.
