# Controlled P0/P1 token-pooling experiment

## Result

Completed extraction on 2026-09-23 and fresh-kernel classifier execution plus
independent verification on 2026-09-27. **P1 meets the prespecified development
criterion, but is not promoted.** Its selected-checkpoint Bot F1 improves in all
three paired seeds and mean AP increases. The gain is modest and does not persist
consistently at epoch 50. No test data were accessed and no threshold was tuned.

### Selected checkpoints: all 144 validation users (28 bots)

Values are mean +/- sample SD across seeds 42, 43, 44. These quantify seed
variability, not confidence intervals for population performance.

| Metric | P0: include specials | P1: exclude specials |
|---|---:|---:|
| Bot F1 | 0.478829 +/- 0.025865 | 0.494172 +/- 0.028423 |
| Bot precision | 0.387486 +/- 0.039562 | 0.402980 +/- 0.038937 |
| Bot recall | 0.630952 +/- 0.020620 | 0.642857 +/- 0.035714 |
| ROC-AUC | 0.758210 +/- 0.007995 | 0.756671 +/- 0.006620 |
| Average precision | 0.544713 +/- 0.008544 | 0.548662 +/- 0.016240 |
| Brier score (lower is better) | 0.181758 +/- 0.015302 | 0.177880 +/- 0.015646 |

| Seed | P0 epoch | P1 epoch | P0 F1 | P1 F1 | Paired F1 change | Paired AP change |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 12 | 12 | 0.450000 | 0.461538 | +0.011538 | -0.003370 |
| 43 | 21 | 27 | 0.486486 | 0.513514 | +0.027027 | +0.002877 |
| 44 | 41 | 41 | 0.500000 | 0.507463 | +0.007463 | +0.012339 |

Mean paired F1 change is +0.015343 and mean AP change is +0.003948.
ROC-AUC decreases slightly on average. Selection uses the same earliest-minimum
unweighted validation BCE rule in both arms; it does not force the same epoch.

### Selected and final checkpoint confusion matrices

Each entry is `[[TN, FP], [FN, TP]]`; rows are true human/bot, columns predicted
human/bot. Every checkpoint uses threshold 0.5 and the same validation users.

| Seed | P0 selected | P1 selected | P0 epoch 50 | P1 epoch 50 |
|---|---|---|---|---|
| 42 | [[82,34],[10,18]] | [[84,32],[10,18]] | [[76,40],[12,16]] | [[73,43],[12,16]] |
| 43 | [[88,28],[10,18]] | [[89,27],[9,19]] | [[90,26],[10,18]] | [[90,26],[10,18]] |
| 44 | [[93,23],[11,17]] | [[94,22],[11,17]] | [[96,20],[11,17]] | [[95,21],[11,17]] |

Final-epoch mean metrics:

| Arm | F1 | Precision | Recall | ROC-AUC | AP | Brier |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 0.468010 | 0.384755 | 0.607143 | 0.733580 | 0.520573 | 0.189741 |
| P1 | 0.460989 | 0.375882 | 0.607143 | 0.737890 | 0.520536 | 0.192516 |

Final paired F1 changes are -0.013136, 0, and -0.007925. The prespecified criterion
is applied to selected checkpoints, not whichever policy gives the best result.
Validation curves are noisy; seed 42 deteriorates markedly after its early minimum.
Seed 43 selects different epochs across pooling arms. Seed 44 reaches its minimum
late. All 50-epoch histories and both checkpoint policies are preserved.

### Descriptive tweet-count cohorts

Mean selected-checkpoint metrics; these are small descriptive comparisons,
not independently powered subgroup tests. Cohorts partition validation users.

| Tweets/user | Users | Bots | P0 F1 | P1 F1 | P0 AP | P1 AP |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 28 | 1 | 0.207407 | 0.216667 | 0.319444 | 0.291667 |
| 2-10 | 55 | 12 | 0.569052 | 0.584010 | 0.650423 | 0.648349 |
| 11-50 | 61 | 15 | 0.469403 | 0.485483 | 0.543416 | 0.557779 |

The one-tweet cohort has only one bot, so its metrics are especially unstable.
No train/validation user lacks prepared tweet text in this version; missing-text
masking remains implemented and unit-tested, but this experiment does not establish
empirical robustness to missing text. Empty-cohort metrics are explicitly undefined.

The temporal-only reference remains context: F1 0.478561, AP 0.552904, AUC 0.767857.
Its capacity and available information differ, so this is not a controlled ablation
and the P1 result does not establish superiority over that reference.

## Representation verification and measurements

All 12,976 training tweets/671 users and 2,618 validation tweets/144 users are included.
P0 versus legacy is **near-exact, not bit-identical**, in all four comparisons.
Every comparison passes the original combined absolute/relative tolerance and cosine
criterion; no vectors or tolerances were corrected after observing differences.

| Population | Mean L2 distance | Maximum coordinate difference | Minimum cosine |
|---|---:|---:|---:|
| Train tweets | 1.527152e-6 | 1.955032e-5 | 0.9999999999981023 |
| Train users | 1.011602e-6 | 3.814697e-6 | 0.9999999999999688 |
| Validation tweets | 1.516373e-6 | 3.814697e-6 | 0.9999999999999530 |
| Validation users | 9.684480e-7 | 3.814697e-6 | 0.9999999999999910 |

The largest training coordinate difference exceeds the absolute tolerance alone,
but passes `abs(fresh-old) <= 1e-5 + 1e-5*abs(old)`. Mean signed coordinate differences
are tiny (approximately -2.27e-11 to 8.47e-11 across the four populations); this
does not reveal a meaningful common directional shift. Full per-row differences
and mean/std/min/5th/50th/95th/99th/max summaries are saved.

Near-exact embeddings do not guarantee identical optimization trajectories or
checkpoint selection: fresh P0 seed 43 selects epoch 21, whereas the earlier cached
T0 experiment selected epoch 13. This experiment therefore uses the freshly trained
P0 as its paired control; historical T0 metrics are not substituted. No correction
of either cache was attempted.

### P0/P1 norm and distance summaries

Norms below are mean +/- sample SD. Distances compare P1 with P0 from the shared
forward pass. No normalization was introduced to eliminate the norm differences.

| Population | P0 norm | P1 norm | Mean cosine | Mean L2 distance |
|---|---:|---:|---:|---:|
| Train tweets | 18.925646 +/- 0.081188 | 19.033196 +/- 0.041461 | 0.9995712 | 0.463399 |
| Train users | 18.907315 +/- 0.076823 | 19.023456 +/- 0.040525 | 0.9994755 | 0.495463 |
| Validation tweets | 18.923926 +/- 0.075286 | 19.032137 +/- 0.041477 | 0.9996372 | 0.459223 |
| Validation users | 18.904999 +/- 0.062583 | 19.019358 +/- 0.032836 | 0.9996439 | 0.478393 |

P1 consistently increases average norms and changes direction slightly. Training
tweet/user minimum cosine is 0.9562295; validation minima are 0.9879677 (tweet) and
0.9980717 (user). Thus changes are not identical across users and tweets.

All five designated IDs were checked: `<s>`, `</s>`, `<unk>`, `<pad>`, `<mask>`.
Each tweet has its two boundary tokens excluded. In addition, P1 excludes 991
training and 206 validation attended unknown tokens. No attended padding/mask tokens
were present and no designated special token was retained. Every tweet retained at
least one content token. Retokenization reproduced all saved input IDs and masks;
independent float64 pooling of saved hidden-state audit rows agreed within tolerance;
all user means reproduced exactly from their saved tweet vectors.

### Runtime, storage, and provenance

- Encoder forward passes plus both pooling reductions: 6,024.294 seconds (100.405
  minutes), 0.38632 seconds/tweet or 7.392 seconds/user averaged over 815 users.
- Tokenization: 22.789 seconds; model loading: 14.408; warm-up: 16.763. Sum of measured
  stages: 6,078.253 seconds (101.304 minutes). This is not a measurement of complete
  wall time including imports, hashing, saving, and verification. The failed attempt
  is excluded from these successful-run timings.
- Peak sampled RSS: 2,013,782,016 bytes (1.8755 GiB); native peak working set:
  2,018,942,976 bytes (1.8803 GiB). Memory is process-wide, not encoder-only.
- Representation arrays: 100,817,920 bytes. Complete embedding bundle:
  115,184,261 bytes. Classifier bundle: 12,135,309 bytes.
- Python 3.11.9; PyTorch 2.14.0; Transformers 5.17.0; tokenizers 0.23.2;
  NumPy 2.4.6; pandas 3.0.5. CPU float32, one thread, no CUDA, approximately 7.88 GiB
  system RAM. Exact platform, library versions, model/tokenizer hashes and prepared
  input hashes are recorded in the embedding provenance file.

## Completed artifacts and verification

Embedding bundle:
`final_data/embedding_versions/token_pooling_v1/20260923T103253436409Z_f2cfb2f4`

Classifier bundle:
`results/versions/token_pooling_v1/20260927T131239185953Z_31718179`

Manifest SHA256 values, respectively:

```text
fc42fc1a7c4ad9f055976f9887448faf4a095351e226980c163bd8f326f27289
711eeaa0dd0a37d8162fb34cc8acc84dcb6d3f7feed12f95c91719275563902b
```

The embedding manifest covers 32 artifacts; the classifier manifest covers 67.
Artifacts include separate tweet/user arrays, keyed ledgers, masks, comparison and
norm distributions, histories, 12 best/final checkpoints, 12 keyed validation
prediction files, cohort metrics, paired differences, source/config snapshots,
provenance, and PNG/PDF figures. No model was promoted.

Six unit tests passed. All four code cells executed in a fresh kernel without error.
A separate process repeated full representation verification and independently
reloaded all 12 checkpoints, reproducing keyed predictions, losses and metrics.
All paired initial-state hashes and all 50 epoch batch-order hashes match.
The 62 protected source/config/notebook files were unchanged during training; after
notebook serialization, the only changed protected file is this new executed notebook.
Consumed legacy input hashes remain unchanged. The unrelated pre-existing edit to
`analysis/06_multimodal_fusion.ipynb` is left untouched and excluded from this commit.

The same validation set selects checkpoints and supports these development results.
Repeated experiments can overfit validation, and three seeds are not statistical
proof. No test estimate or automatic promotion is claimed. Further pooling,
Persian encoders, fine-tuning, or contrastive experiments require separate approval.

## Prespecified scope

P0: final hidden states, attention-mask mean, padding excluded, specials included.
P1: same forward pass, mean over attended tokens excluding every tokenizer-designated
special ID plus any returned special-token mask positions. This includes unknown
and literal mask-token IDs, not only boundary tokens. Stop if any tweet has no
remaining content tokens. Both aggregate prepared tweets by arithmetic mean.

Pinned local XLM-R revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, all five
snapshot file hashes checked against the provenance pilot. Eval/no-grad, CPU float32,
one thread, eager attention, batch16, fixed128 padding/truncation. Prepared training
and validation texts, ordering, user keys, normalization, deduplication and selection
are unchanged. No test files are opened, even for preservation hashing.

Store both tweet/user representations under a new versioned embedding directory.
Save input IDs, attention/special masks, keyed ledgers, hidden-state audit samples,
legacy comparisons, norms, differences, runtime, process memory and hashes. Independent
verification retokenizes every row, checks all exclusion masks, reproduces user means,
and independently averages saved hidden states for audited rows. P0 is checked against
all legacy development vectors using the pilot's exact/near-exact rule: allclose
atol1e-5/rtol1e-5 plus minimum cosine0.999999. Report mismatches without correction.

The classifier is the existing16,897-parameter temporal-plus-text model. Six runs,
P0/P1 with seeds42/43/44, all671 training and144 validation users,50epochs,batch32,
AdamW lr.001/decay.0001,dropout.3,training-only positive weight540/131. Initial states
and actual batch orders must match. Select earliest minimum unweighted validation BCE;
retain final checkpoints. Threshold.5, no tuning. Same user-mean aggregation and
missing-text mask in both arms. The temporal-only reference is contextual.

Report F1, precision, recall, AUC, AP, Brier, confusion matrices, paired seed effects,
selected/final results, learning curves, tweet/user norms, P1−P0 vector cosine/L2
differences, and descriptive1/2–10/11–50 tweet cohorts. Consistent improvement requires
strictly positive F1 changes in all three seeds and mean AP not decreasing. This is
a development criterion, not significance evidence or automatic promotion.

No graph, gating, contrastive loss, descriptions, pseudo-labels, ParsBERT, CLS pooling,
fine-tuning, sequence-length change, or legacy overwrite. Stop after this experiment.

## Reproduction

```powershell
.\env\Scripts\python.exe -B -X utf8 -m unittest tests.test_token_pooling tests.test_text_aggregation -v
.\env\Scripts\python.exe -B -X utf8 scripts/run_token_pooling.py --execute-notebook
.\env\Scripts\python.exe -B -X utf8 scripts/run_token_pooling.py --verify-run results/versions/token_pooling_v1/RUN_DIRECTORY
```

Use `--reuse-embeddings final_data/embedding_versions/token_pooling_v1/RUN_DIRECTORY`
with notebook execution only for a completed bundle; it is verified before new
classifier training. No fresh extraction is silently substituted for a supplied bundle.

## Execution recovery

The first fresh-kernel attempt (`20260923T100923973541Z_162af7cc`) stopped during
training-tweet extraction after a Windows ZeroMQ assertion, `No buffer space
available [10055]`. Its incomplete directory is preserved and has no completion
manifest. No classifier was trained in that attempt.

The retry uses `--extract-only` to run the identical encoder implementation outside
Jupyter's messaging transport. The completed bundle is then passed through
`--execute-notebook --reuse-embeddings ...` for fresh-kernel representation
verification and classifier execution. This changes execution orchestration only;
no representation definition, training policy, or population is changed.
