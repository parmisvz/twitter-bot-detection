# Controlled modality/branch ablation: modality_ablation_v1

## Outcome

All 15 approved runs completed. M4 reproduces the frozen P1/C0 development reference. At selected checkpoints, M4 meets the prespecified development improvement criterion against M0, M2 and M3. M2 versus M0 and M3 versus M0 do not meet it. No model was automatically promoted; no final test evaluation was started.

This is a development branch study, not a pure causal information ablation. Adding a branch changes information, parameter count, classifier dimensionality and the stochastic training trajectory. Three seeds on one repeatedly used validation population are not independent population replications or statistical proof.

## Definitions and architecture

M0 is profile/behavioral, not pure profile metadata. Its 33 features include tweet-derived numeric statistics (word/hashtag/language counts, tweet types, engagement), description-derived numeric features, account age and growth rates. Removing XLM-R removes semantic text representations, not all tweet-derived information. Removing the temporal branch removes only the seven designated statistics:

- time_between_tweets
- tweet_frequency
- min_tweets_per_hour
- min_tweets_per_day
- max_tweets_per_hour
- max_tweets_per_day
- max_occurence_of_same_gap

Profile branch: Linear(33,48), BatchNorm1d(48), ReLU, Dropout(0.3): 1,728 parameters.
Temporal branch: seven masked values plus seven observation indicators, Linear(14,16), ReLU, Dropout(0.3), deterministic availability masking: 240 parameters.
Text branch: frozen P1 XLM-R vector, Linear(768,16), ReLU, Dropout(0.3), deterministic availability masking: 12,304 parameters.
Classifier: Linear(D,32), ReLU, Dropout(0.3), Linear(32,1). Sigmoid converts its logit into a probability.

| Arm | Included branches | D | Trainable parameters |
| --- | --- | --- | --- |
| M0 | Profile/behavioral | 48 | 3,329 |
| M1 | Frozen P1 semantic text | 16 | 12,881 |
| M2 | Profile/behavioral + designated temporal statistics | 64 | 4,081 |
| M3 | Profile/behavioral + frozen P1 semantic text | 64 | 16,145 |
| M4 | All three retained branches | 80 | 16,897 |

Frozen XLM-R weights are not trainable parameters. No language model was loaded or run.

## Approved initialization adjustment

For every seed, build the historical untrained M4 constructor exactly, including its historical RNG consumption. M4 uses that model directly. Reduced arms copy all compatible retained encoder parameters and BatchNorm buffers and the final 32-to-1 classifier layer. Their first classifier layer is freshly initialized by PyTorch Linear using its actual fan-in (48, 16, or 64), inside an isolated CPU RNG context with seed `300000 + run_seed` (300042, 300043, 300044). No M4 first-layer weight slicing is used.

All smaller arms use the same classifier-specific seed for a given run seed. M2 and M3 consequently have identical initial 64-to-32 tensors, but their input blocks have different meanings. The first classifier layer is not parameter-paired across unequal dimensions. Construction leaves the primary training RNG at the historical M4 post-construction state. Encoder, first-classifier, final-classifier and full-model state hashes are recorded separately. No trained weights initialize any arm.

Compatible branch initializations and minibatches are paired. Dropout masks across different architectures are not claimed to be identical: omitted branches change RNG consumption. No new stochastic mechanism was introduced to force artificial dropout pairing.

## Fixed protocol and inputs

- 671 training users: 540 humans, 131 bots. Same 144 validation users, including 28 bots.
- Seeds 42, 43, 44; 50 epochs; batch size 32 (last training batch 31); no dropped users.
- AdamW lr 0.001, weight decay 0.0001; positive BCE weight 540/131; dropout 0.3.
- Earliest minimum user-mean unweighted validation BCE checkpoint; threshold 0.5. Final epoch 50 retained separately.
- Validation uses evaluation mode and no gradients; model-state checks guard against validation updates.
- Original Phase-2 train-fitted transformations, manifest feature order, identities and frozen splits.
- Frozen P1: pinned multilingual XLM-R, attention-mask token mean excluding designated special tokens, arithmetic mean across each user's prepared tweets.
- No test access, graph, gating, contrastive loss, pseudo-labels, threshold tuning, LM fine-tuning, new encoder or augmentation.

Pinned Phase-2 manifest SHA256: `594a2defef59d663ef171020b7cd1ba75aa321c30e7a7d09b57b560563c109c9`.
Pinned embedding manifest SHA256: `fc42fc1a7c4ad9f055976f9887448faf4a095351e226980c163bd8f326f27289`.
Historical P1 experiment manifest SHA256: `711eeaa0dd0a37d8162fb34cc8acc84dcb6d3f7feed12f95c91719275563902b`.

Missing text is sanitized before projection and its output is masked; missing temporal values use observation masks, with an entirely unavailable branch forced to zero. Users remain in all arms. Experimental branch exclusion is recorded separately from data missingness. M1 on absent text uses the classifier on its zero representation, retaining the user. The development population has no naturally absent text/temporal modality, so functional masking checks do not demonstrate missing-modality predictive robustness.

## Selected-checkpoint results

Mean +/- sample SD across three seeds, not confidence intervals. All metrics use the same 144 validation users.


| Arm | Bot F1 | Precision | Recall | ROC-AUC | AP | Brier | BCE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | 0.447731 +/- 0.017696 | 0.348148 +/- 0.025660 | 0.630952 +/- 0.041239 | 0.760365 +/- 0.002861 | 0.546035 +/- 0.031749 | 0.189730 +/- 0.008559 | 0.553903 +/- 0.019849 |
| M1 | 0.420982 +/- 0.039277 | 0.353753 +/- 0.014016 | 0.523810 +/- 0.089879 | 0.713978 +/- 0.003746 | 0.348574 +/- 0.010091 | 0.189190 +/- 0.015062 | 0.564898 +/- 0.036786 |
| M2 | 0.438498 +/- 0.041989 | 0.348058 +/- 0.046741 | 0.595238 +/- 0.020620 | 0.753797 +/- 0.007705 | 0.530429 +/- 0.025744 | 0.191522 +/- 0.007565 | 0.558389 +/- 0.016299 |
| M3 | 0.434377 +/- 0.019763 | 0.357410 +/- 0.019130 | 0.559524 +/- 0.074345 | 0.745587 +/- 0.008360 | 0.545469 +/- 0.023750 | 0.183537 +/- 0.012811 | 0.551339 +/- 0.018629 |
| M4 | 0.494172 +/- 0.028423 | 0.402980 +/- 0.038937 | 0.642857 +/- 0.035714 | 0.756671 +/- 0.006620 | 0.548662 +/- 0.016240 | 0.177880 +/- 0.015646 | 0.541210 +/- 0.022535 |

## Final epoch 50: secondary comparison


| Arm | Bot F1 | Precision | Recall | ROC-AUC | AP | Brier | BCE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M0 | 0.421922 +/- 0.014986 | 0.324275 +/- 0.022264 | 0.607143 +/- 0.035714 | 0.734195 +/- 0.013443 | 0.526840 +/- 0.033598 | 0.207542 +/- 0.022538 | 0.616028 +/- 0.060045 |
| M1 | 0.446086 +/- 0.013616 | 0.342790 +/- 0.016379 | 0.642857 +/- 0.061859 | 0.710591 +/- 0.003330 | 0.345143 +/- 0.003013 | 0.206369 +/- 0.015681 | 0.603757 +/- 0.038338 |
| M2 | 0.403742 +/- 0.043948 | 0.313528 +/- 0.048044 | 0.571429 +/- 0.035714 | 0.722393 +/- 0.002619 | 0.493755 +/- 0.018139 | 0.212328 +/- 0.017622 | 0.637826 +/- 0.043319 |
| M3 | 0.413482 +/- 0.011902 | 0.335352 +/- 0.032533 | 0.547619 +/- 0.054554 | 0.727114 +/- 0.002910 | 0.503646 +/- 0.059794 | 0.196220 +/- 0.023491 | 0.617942 +/- 0.065114 |
| M4 | 0.460989 +/- 0.081045 | 0.375882 +/- 0.092667 | 0.607143 +/- 0.035714 | 0.737890 +/- 0.027935 | 0.520536 +/- 0.022614 | 0.192516 +/- 0.037245 | 0.606176 +/- 0.095931 |


Final checkpoints do not replace the selected-checkpoint decision. In particular, M1 has higher mean final F1 than selected F1, but this does not authorize changing its selection rule. Training weighted BCE and validation unweighted BCE are different objectives; their raw gap is not a calibrated generalization estimate.

## All five prespecified primary comparisons

Criterion: selected Bot F1 strictly increases in all three paired seeds AND mean AP does not decrease, with all verification checks passing. No automatic promotion.

| Comparison | Added parameters | F1 delta seed42 | seed43 | seed44 | Mean delta F1 | Mean delta AP | Criterion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M2-M0 | 752 | -0.005500 | +0.013120 | -0.035318 | -0.009233 | -0.015606 | Does not pass |
| M3-M0 | 12816 | +0.008637 | -0.049087 | +0.000389 | -0.013353 | -0.000565 | Does not pass |
| M4-M2 | 12816 | +0.019980 | +0.034640 | +0.112401 | +0.055674 | +0.018233 | Pass |
| M4-M3 | 752 | +0.005842 | +0.096847 | +0.076693 | +0.059794 | +0.003193 | Pass |
| M4-M0 | 13568 | +0.014480 | +0.047760 | +0.077083 | +0.046441 | +0.002627 | Pass |


### Interpretation

- **M2 vs M0:** adding designated temporal statistics alone gives mixed F1 changes and lower AP in all seeds; no consistent improvement.
- **M3 vs M0:** adding semantic text alone gives mixed F1 changes and a slightly lower mean AP; no consistent improvement.
- **M4 vs M2:** adding semantic text given profile/behavioral and temporal improves F1 and AP in all seeds; passes the development criterion.
- **M4 vs M3:** adding temporal statistics given profile/behavioral and text improves F1 in all seeds and mean AP; passes. AP still decreases for seed 43.
- **M4 vs M0:** the complete retained combination improves F1 in all seeds and mean AP; passes. AP still decreases for seed 43.
- **M1:** standalone semantic text is weaker in mean selected F1 and AP than M0 or M4. This does not mean text is useless: M4-M2 supports incremental benefit in the combined architecture.

M4 has the best mean selected F1, precision, recall, AP, Brier and BCE among these five arms. M0 has slightly higher mean ROC-AUC than M4. Therefore, this is not an across-all-metrics dominance claim.

The pattern is compatible with complementary branch information in the complete architecture, but it does not establish causal synergy: capacity, first-layer dimensionality, initialization scale and stochastic optimization also differ. M2 here is newly initialized under this study's pairing policy and is not claimed to reproduce the older temporal-only experiment. All prior experiments remain intact.

## Per-seed selected/final results and confusion matrices

Confusion entries are [[TN, FP], [FN, TP]]. Full precision/recall/AUC/AP/Brier/BCE and all paired metric differences are in the versioned CSVs.

### Selected


| Arm | Seed | Epoch | Bot F1 | AP | Confusion matrix |
| --- | --- | --- | --- | --- | --- |
| M0 | 42 | 13 | 0.447059 | 0.509413 | [[78,38],[9,19]] |
| M0 | 43 | 20 | 0.465753 | 0.565790 | [[88,28],[11,17]] |
| M0 | 44 | 26 | 0.430380 | 0.562901 | [[82,34],[11,17]] |
| M1 | 42 | 46 | 0.422535 | 0.338321 | [[88,28],[13,15]] |
| M1 | 43 | 39 | 0.459459 | 0.348907 | [[87,29],[11,17]] |
| M1 | 44 | 41 | 0.380952 | 0.358495 | [[93,23],[16,12]] |
| M2 | 42 | 12 | 0.441558 | 0.501391 | [[84,32],[11,17]] |
| M2 | 43 | 18 | 0.478873 | 0.539441 | [[90,26],[11,17]] |
| M2 | 44 | 23 | 0.395062 | 0.550456 | [[79,37],[12,16]] |
| M3 | 42 | 14 | 0.455696 | 0.524418 | [[83,33],[10,18]] |
| M3 | 43 | 28 | 0.416667 | 0.571216 | [[87,29],[13,15]] |
| M3 | 44 | 41 | 0.430769 | 0.540774 | [[93,23],[14,14]] |
| M4 | 42 | 12 | 0.461538 | 0.531972 | [[84,32],[10,18]] |
| M4 | 43 | 27 | 0.513514 | 0.549603 | [[89,27],[9,19]] |
| M4 | 44 | 41 | 0.507463 | 0.564410 | [[94,22],[11,17]] |

### Final


| Arm | Seed | Epoch | Bot F1 | AP | Confusion matrix |
| --- | --- | --- | --- | --- | --- |
| M0 | 42 | 50 | 0.404762 | 0.496677 | [[77,39],[11,17]] |
| M0 | 43 | 50 | 0.432432 | 0.520792 | [[86,30],[12,16]] |
| M0 | 44 | 50 | 0.428571 | 0.563052 | [[78,38],[10,18]] |
| M1 | 42 | 50 | 0.454545 | 0.342366 | [[76,40],[8,20]] |
| M1 | 43 | 50 | 0.430380 | 0.344717 | [[82,34],[11,17]] |
| M1 | 44 | 50 | 0.453333 | 0.348347 | [[86,30],[11,17]] |
| M2 | 42 | 50 | 0.409639 | 0.486379 | [[78,38],[11,17]] |
| M2 | 43 | 50 | 0.444444 | 0.480465 | [[88,28],[12,16]] |
| M2 | 44 | 50 | 0.357143 | 0.514420 | [[75,41],[13,15]] |
| M3 | 42 | 50 | 0.400000 | 0.434878 | [[76,40],[11,17]] |
| M3 | 43 | 50 | 0.422535 | 0.532685 | [[88,28],[13,15]] |
| M3 | 44 | 50 | 0.417910 | 0.543376 | [[91,25],[14,14]] |
| M4 | 42 | 50 | 0.367816 | 0.521480 | [[73,43],[12,16]] |
| M4 | 43 | 50 | 0.500000 | 0.497465 | [[90,26],[10,18]] |
| M4 | 44 | 50 | 0.515152 | 0.542664 | [[95,21],[11,17]] |


## Verification and artifacts

- Two focused tests passed, covering all arms/seeds, exact parameter counts, encoder/final-layer pairing, actual fan-in first-layer initialization, primary RNG preservation, deterministic rebuilds, excluded-input invariance and missingness invariance.
- Fresh-kernel execution completed the full 15-run experiment and verification.
- Separate-process verification reloads all 30 selected/final checkpoints plus 15 initial snapshots, reproduces 30 keyed validation predictions and metrics, and verifies hashes, initialization pairing, complete epoch schedules, selection policy, missingness and excluded-input invariance.
- All six M4 selected/final state dictionaries exactly match historical P1 (zero tolerance), including BatchNorm buffers. Historical P1 histories reproduce within 1e-7 serialization tolerance. A separate check confirms all M4 initial full-state hashes and selected epochs exactly match historical P1.
- All 115 manifest-listed artifacts are SHA256 verified. Consumed input hashes are checked; no test files are opened even for preservation hashing.
- No old artifact is overwritten. The pre-existing Notebook 06 edit is untouched and excluded from this change.

The first fresh-kernel attempt (`20260928T054809793986Z`) completed training but failed while serializing a NumPy Boolean in the decision summary. Its partial outputs remain intact. Only the export Boolean conversion was corrected; all 15 runs were rerun unchanged in the accepted version. The partial run is not a second set of independent experiments and is not used for conclusions.

Expected files include config, module, runner, focused tests, executed notebook, and this report. Versioned results contain initial/selected/final checkpoints, keyed predictions, histories, all metrics and paired comparisons, runtime, population masks, initialization hashes, source snapshots, figures and input provenance. The manifest's run metadata records separate encoder/first-layer/final-layer hashes. Plot panels were visually checked.


Accepted run: `results/versions/modality_ablation_v1/20260928T054959860328Z`.

Manifest SHA256: `77ffafccf6d9635ea5da787ed8a86c78303ff744082df57c956a22dc9af43730`.

Total training time for 15 runs: 62.52 seconds (excludes loading, export and verification).


## Reproduction

```powershell
.\env\Scripts\python.exe -B -X utf8 -m unittest tests.test_modality_ablation -v
.\env\Scripts\python.exe -B -X utf8 scripts/run_modality_ablation.py --execute-notebook
.\env\Scripts\python.exe -B -X utf8 scripts/run_modality_ablation.py --verify-run results/versions/modality_ablation_v1/20260928T054959860328Z
```

Default execution creates a new version. `--execute-notebook --resume-run results/versions/modality_ablation_v1/RUN_DIRECTORY` verifies and displays an existing completed version without retraining.

Stop here. P1/M4 remains the current development reference. Final evaluation has not started.
