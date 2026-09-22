# Missingness-aware gated fusion experiment

## Prespecified question

Can learned, user-dependent branch weights improve the scaled-GCN concatenation
experiment? This changes fusion only. Use the frozen scaling experiment
`graph_scaling_v1/20260922T052601567701Z_4a1137a7` and verify its manifest hash.
Recreate train/validation graphs through the existing Phase-2 contract, using
the same training-fitted scalar 454.3006810563794. Never merge split graphs.

## Architecture and controls

The existing encoders produce 48 profile/behavioral, 16 temporal-statistics and
32 GCN coordinates. No text branch is introduced. Concatenation is the control.
The treatment computes three gates from the 96 coordinates plus three availability
indicators: `2 * sigmoid(W [h; availability] + b) * availability`.
Multiply each branch by its gate, concatenate, and use the unchanged classifier.
Initialize W and b to zero: observed branches initially have weight one, absent
branches zero. Gates range from zero to two without softmax renormalization.

The profile/behavioral branch is required to have profile coverage for every
current user; missing profile triggers an error, not a silent assumption. Temporal
availability means any of the seven statistics is observed; the existing per-feature
masks still apply. Graph availability remains `has_graph`, separate from node
membership. No user is removed for missing temporal or graph data. Imported temporal
statistic formulas/windows remain uncertain; this is not a sequence encoder.

Concatenation has 7,441 trainable parameters; gating has 7,741 (+300, about 4%).
This is a matched-input/training comparison, not an exact parameter-matched ablation.
Shared weights and dropout RNG begin identically. A gain could partly reflect
capacity; no claim isolates gating from parameter count. Gate weights are not
causal modality importance.

## Training and evaluation

Train both arms from scratch for seeds 42/43/44. Fixed 50 epochs, AdamW lr .001,
decay .0001, batch 32, dropout .3, CPU one thread, weighted BCE with training-only
positive weight 540/131. All 671 training users remain included. Select earliest
minimum user-mean unweighted validation BCE; save final checkpoints too.
Concatenation must exactly reproduce the existing scaled-GCN checkpoint tensors,
and both arms must match its actual minibatch orders. No new hyperparameter search.

Primary comparison is gated minus concatenation Bot F1 at threshold .5, using all
144 validation users; also report precision, recall, AUC, AP, Brier, confusion
counts, paired seed variation, and matched covered42/unavailable102 cohorts.
Keep final-epoch results visible. Report keyed gates for train and validation,
including absence and saturation. Temporal-reference results remain contextual.

Prespecified interpretation: call the result a consistent validation improvement
only if selected-checkpoint all-user F1 improves in all three paired seeds and
mean AP does not decrease. Otherwise report mixed/negative evidence. This is a
development decision rule, not a significance test or an automatic promotion rule.

No test labels, features, graphs or predictions are parsed. Only identity/split
columns of frozen assignments are read for isolation checks. No test tuning or
automatic promotion. Three seeds quantify limited optimization variation; repeated
validation use and seven graph-covered validation bots limit generalization claims.

## Reproduction

```powershell
.\env\Scripts\python.exe -X utf8 -m unittest tests.test_gated_fusion tests.test_graph_scaling -v
.\env\Scripts\python.exe -X utf8 scripts/run_gated_fusion.py --execute-notebook
.\env\Scripts\python.exe -X utf8 scripts/run_gated_fusion.py --verify-run results/versions/gated_fusion_v1/RUN_DIRECTORY
```

Fresh-kernel execution, independent checkpoint reload, hash preservation and keyed
prediction/gate reproduction are required. New outputs are versioned; all older
experiments remain unchanged. No original notebook is edited.

## Completed experiment

Version: `results/versions/gated_fusion_v1/20260922T104310152569Z_0f07576a`.
All six runs completed 50 epochs. The three concatenation controls reproduced
the frozen scaled-GCN selected and final checkpoint tensors **exactly**.
Selected epochs were 12/18/20 for concatenation and 12/16/20 for gating.

### Selected-checkpoint results

Three-seed means on the same 144 validation users; F1 includes sample SD.

| Model | Bot F1 | Precision | Recall | ROC-AUC | AP | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Concatenation | 0.431 ± 0.037 | 0.339 | 0.595 | 0.762 | 0.531 | 0.188 |
| Gated | 0.418 ± 0.031 | 0.330 | 0.571 | 0.752 | 0.531 | 0.191 |
| Frozen temporal reference (context) | 0.479 ± 0.013 | 0.378 | 0.655 | 0.768 | 0.553 | 0.184 |

Temporal-reference values are from the pinned scaling source's `seed_summary.csv`;
that comparison also changes graph information and capacity, so it is contextual.

| Seed | Concatenation F1 | Gated F1 | Paired F1 change | Concatenation TN/FP/FN/TP | Gated TN/FP/FN/TP |
|---|---:|---:|---:|---|---|
| 42 | 0.4096 | 0.3951 | -0.0146 | 78/38/11/17 | 79/37/12/16 |
| 43 | 0.4737 | 0.4533 | -0.0204 | 86/30/10/18 | 86/30/11/17 |
| 44 | 0.4110 | 0.4054 | -0.0056 | 86/30/13/15 | 85/31/13/15 |

All three paired F1 changes are negative. Mean F1 change is -0.0135, AUC
change -0.0104, AP change -0.0008, and Brier change +0.0023 (higher is worse).
The prespecified improvement criterion fails. AP improves in seeds 42/44 but
declines in seed 43; a rounded mean of 0.531 for both arms hides that small decrease.

| Cohort | Concatenation F1 | Gated F1 | Concatenation AP | Gated AP |
|---|---:|---:|---:|---:|
| Graph available: 42 users, 7 bots | 0.492 | 0.469 | 0.662 | 0.649 |
| Graph unavailable: 102 users, 21 bots | 0.417 | 0.406 | 0.512 | 0.504 |

At the final epoch, gated mean F1 is 0.409 versus concatenation 0.389. However,
gated final AP is lower (0.477 versus 0.491), and both final F1 values are below
their respective selected-checkpoint means. Do not switch checkpoint policy to
make the treatment look favorable. Full histories and metrics remain available.

### Gate behavior

Median selected-checkpoint gate weights among the 42 graph-covered validation users:

| Seed | Profile/behavioral | Temporal statistics | Graph |
|---|---:|---:|---:|
| 42 | 1.836 | 0.957 | 0.789 |
| 43 | 1.900 | 1.458 | 0.835 |
| 44 | 1.959 | 1.516 | 0.664 |

The gates learn nontrivial weights and reduce the graph branch's scale. For the
covered cohort, profile/behavioral weights exceed 1.9 in 3/20/42 users for seeds
42/43/44: some gates approach their upper bound of two. No available-branch
selected validation gate is below 0.1. This describes learned scaling, not causal
importance; the encoders and classifier also adapt jointly.

All 102 graph-unavailable validation users have graph gate exactly zero for every
checkpoint. All actual training/validation users have the seven temporal statistics
observed; therefore empirical performance under temporal absence is not estimated.
Synthetic tests verify masking and prediction invariance when temporal values and
graph inputs are unavailable. Those tests are software checks, not fabricated users
in the research dataset. Current profile absence is explicitly unsupported and
checked at load time; no user is removed to satisfy coverage assumptions.

### Verification

- Fresh-kernel training, scoring and checkpoint verification passed.
- Separate-process verification reproduced 12 checkpoint predictions and 24 keyed
  train/validation gate files, histories, metrics, summaries and paired effects.
- All six control checkpoints matched their frozen counterparts exactly.
- Fifteen targeted and regression tests passed (6 gating/scaling and 9 graph/diagnosis).
- The run registered 72 hashed artifacts and verified 1,082 protected files before
  completion. The subsequent executed notebook is the only expected protected-file
  change; previous artifacts are unchanged.
- No test inputs were parsed. No threshold was tuned and no model was promoted.
- The comparison figure was visually inspected; the executed notebook has no errors.

## Conclusion and next decision

Preserve this as a negative controlled experiment. Do not promote this gated model
or claim that added fusion complexity improves the current dataset. Keep the
temporal reference and its flat control as the stronger existing development
references. The result applies to this gate, these features and this protocol;
it does not establish that all gated architectures fail.

Following the agreed experiment order, the next proposed step is a separately
specified contrastive-learning experiment using the stronger graph-independent
reference, rather than stacking a new loss on the weaker gated model. Before
training, define the positive-pair construction, valid augmentations, projection
head, loss coefficient, and matched supervised control. Missing modalities must
remain masked, and validation/test identities must stay out of representation
training. That is a new experiment requiring approval; none is implemented here.
