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

`04_graph_baseline_pre_v2.ipynb` preserves notebook 04 before the Phase-4 GCN
protocol refactor. It uses the historical global graph, all-label class weights,
and a shallow best-checkpoint copy. Keep its results as historical experiments;
do not run its artifact-overwrite cells in the versioned pipeline.

`05_graph_tabular_pre_v2.ipynb` preserves notebook 05 before its compatibility
refactor. It uses the global graph with all gold users' tabular features, although
class weights and deep-copy checkpoint selection are already training-correct.
Its hardcoded metric export is historical and must not overwrite versioned runs.

## Notebook 06

`06_multimodal_fusion_pre_v2.ipynb` preserves the original notebook and outputs. Global degree inputs, silent cohort filtering, shallow best-state copies and legacy comparisons are historical limitations. Do not run it as a current pipeline stage.

## Notebook 07

`07_ablation_study_pre_v2.ipynb` preserves the original pairwise experiments and outputs. Its differing cohorts, global graph statistics and shallow best-state snapshots are historical limitations. The current fixed-cohort study is separately versioned; do not run this archive as a pipeline stage.

## Notebook 08

`08_semi_supervised_learning_pre_v2.ipynb` preserves the original notebook and outputs. Its student training includes held-out labeled users, substitutes descriptions for tweet embeddings and overwrites a supervised checkpoint. Retain it as historical evidence; do not execute its export cells or treat its metrics as valid held-out results. The active notebook uses the separately versioned tabular pseudo-label experiment.
