# Training-only XLM-R provenance and runtime pilot

## Outcome

Fresh pinned XLM-R vectors are **near-exact**, not bit-identical, to the legacy
cache on the selected training subset. Neither cache was corrected or overwritten.
The differences are consistent with small floating-point execution differences;
this is an interpretation, not proof of the original revision or hardware.
Full train+validation extraction appears practical as an offline CPU job, with a
linear estimate of about **2 hours 12 minutes** under the measured settings.
Full extraction and token-pooling experiments have not been started.

Version: `results/versions/xlmr_provenance_pilot_v1/20260923T052217323306Z_4479f965`.

## Scope and selection

Only the existing prepared training texts, training tweet vectors, training user
vectors/keys and their manifests were read. No validation/test data or labels were
loaded. The 2,618 validation-tweet count used for extrapolation comes from the prior
audit and is explicitly identified as such in `runtime.json`. No classifier or
language model was trained. No download, preprocessing change or cache replacement.

Selection is deterministic and label-free: take the two user keys with the lowest
SHA256(`xlmr_provenance_v1|user_key`) in each stratum, union the selected users,
then retain **all** their prepared tweets in the original training-row order.
Strata are one tweet, 2–10 tweets, 11–50 tweets, and users containing tweets in
each untruncated-token group below. No random sampling or text edits.

The union contains **9 users and 248 tweets**. Their tweet counts, in sorted key
order, are 1, 50, 1, 50, 50, 5, 6, 35, 50. Complete users are essential: comparing
a partial fresh user mean to a full legacy mean would be invalid.

| Token count including special tokens | Pilot tweets |
|---|---:|
| At most 32 | 62 |
| 33–111 | 172 |
| 112–128 | 3 |
| More than 128 | 11 |

`subset_tweets.csv` stores original vector-row index, `user_key`, source-row
identifier, normalized text key, text hash and token count. Original tweet numeric
IDs are not used as identity. `subset_users.csv` fixes user-vector row order.
`selection.json` stores the selection reasons and ordered training/subset text hashes.
Input file hashes, including the Phase-2 manifest that pins preprocessing and tweet
selection, are in `provenance.json`. Historical row-to-vector provenance remains
incomplete; successful comparisons do not retroactively fill in that history.

## Pinned extraction

- Model and tokenizer revision: `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`.
- Local `xlm-roberta-base` snapshot only; offline/local-files-only loading.
- Model evaluation mode, parameters frozen, `torch.no_grad()`.
- CPU float32, one PyTorch thread, deterministic algorithms, batch size 16.
- Fixed padding/truncation to 128 tokens; final hidden states, 768 dimensions.
- Attention-mask mean pooling: padding excluded, special tokens included.
- Arithmetic mean over every prepared tweet for each selected user.
- Explicit eager attention backend; actual tokenizer class `XLMRobertaTokenizer`.

The historical package versions, hardware, backend and revision are unknown. The
pilot pins those execution details prospectively. Subset batch composition can
differ from historical full-extraction batches; sequence padding remains fixed.
The unused MLM-head weights reported when loading the base encoder do not enter
the final hidden-state extraction. The long-sequence warning occurred during
untruncated length inspection; model inputs were truncated to 128.

| Snapshot file | SHA256 |
|---|---|
| model.safetensors | `6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb` |
| config.json | `d66ed8cd4f2a93b358c245e50736fa389ed4f35c0bae7aad0b32abb20c62b579` |
| tokenizer.json | `a898ea75433890f6610f4e470b8ebeb0c21dce5c8dd61f892eb09eb5919d2e2c` |
| tokenizer_config.json | `994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce` |
| sentencepiece.bpe.model | `cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865` |

Versions: PyTorch2.14.0, Transformers5.17.0, tokenizers0.23.2, NumPy2.4.6,
pandas3.0.5, safetensors0.8.0, psutil7.2.2. Hardware: Windows build26100,
Intel64 Family6 Model142 Stepping11, 4 physical/8 logical CPUs, approximately
7.88 GiB RAM, no CUDA GPU. Full environment is recorded in `provenance.json`.

## Numerical comparison

Before extraction, agreement was defined as exact array equality, otherwise
near-exact if every coordinate passes `atol=1e-5, rtol=1e-5` and minimum cosine
is at least0.999999. Values outside those tolerances would be reported as
materially different under that declared criterion. No tolerance was fitted to results.

| Measure | Tweets (248) | Users (9) |
|---|---:|---:|
| Minimum cosine similarity | 0.9999999999999811 | 0.9999999999999984 |
| Mean cosine similarity | 0.9999999999999977 | 0.9999999999999996 |
| Mean L2 distance | 1.425e-6 | 6.514e-7 |
| Median L2 distance | 1.171e-6 | 5.224e-7 |
| 95th-percentile L2 distance | 2.531e-6 | 1.550e-6 |
| Maximum L2 distance | 4.041e-6 | 2.038e-6 |
| Maximum absolute coordinate difference | 3.815e-6 | 1.907e-6 |
| Maximum relative L2 distance | 2.142e-7 | 1.071e-7 |
| Classification | Near-exact | Near-exact |

Full count/mean/SD/min/5th/median/95th/99th/max distributions are saved in
`difference_distributions.csv`; every individual comparison is keyed in the
tweet/user comparison files. No claim about downstream prediction equality is made.

### Systematic differences

The mean signed difference across tweet coordinates is +1.289e-10; 126 tweets
have positive mean signed differences and122 negative. The largest absolute
coordinate-wise mean bias is1.154e-7 (coordinate741). There is no evidence of a
material global shift at these scales; tiny coordinate-specific biases remain.

Mean tweet L2 distances by length are1.809e-6 (short),1.315e-6 (middle),
1.337e-6 (near limit), and1.001e-6 (truncated). Differences do not increase
monotonically with length or spike above the truncation limit. Boundary groups
are small, so this is descriptive evidence, not a formal absence-of-bias test.
`coordinate_differences.csv` and `differences_by_length.csv` retain the diagnostics.

The first16 tweets were extracted twice (warm-up and timed pass), producing
**bit-identical fresh vectors**. This supports repeatability in the pinned current
environment. Near-exact legacy agreement applies to this purposive subset, not
proof of equality for every training/validation user or the historical environment.

## Runtime and storage

| Measurement | Result |
|---|---:|
| Training length inspection and selection tokenization | 8.92 s |
| Model loading | 10.84 s |
| Warm-up, first16 tweets | 12.49 s |
| Timed extraction, 248 tweets | 125.24 s |
| Extraction per tweet | 0.505 s |
| Amortized extraction per selected complete user | 13.92 s |
| Peak sampled whole-process RSS | 1.44 GiB |
| Native peak process working set | 1.72 GiB |
| Fresh tweet+user array files | 789,760 bytes (0.753 MiB) |
| Entire version directory at completion | 1,126,472 bytes (1.074 MiB) |
| Estimated15,594 train+validation tweets | 7,874.83 s (2.19 h) |
| Estimate including one load and warm-up | 7,898.15 s (about2 h12 min) |

Peak RSS was sampled every50 ms; the native Windows peak covers the entire
process lifetime. Neither measure is model-only memory. Per-user timing is
amortized, not an individual-user latency benchmark, and depends on tweet counts.
The fixed128-token CPU batch setting makes a per-tweet extrapolation useful, but
background load, thermal behavior, I/O and batch tails can change actual time.
The estimate excludes complete output/report generation and is not a guarantee.

## Verification and stop boundary

Two unit tests passed for deterministic complete-user selection and numerical
metrics/tolerances. A separate Python process verified all15 registered artifact
hashes, model/tokenizer hashes, unchanged accessed training inputs, complete selected
user membership, fresh user means and saved comparison statistics. Training-only
tokenizer/selection replay independently checked selected rows, lengths and reasons.
Legacy arrays were opened read-only and their hashes remained unchanged.

No test/validation data was read, no classifier was trained, and no representation
was corrected. No full extraction, token-pooling change, ParsBERT download, or
fine-tuning has been performed. **Stop here pending separate approval.**

Reproduce the verification without fresh inference:

```powershell
.\env\Scripts\python.exe -B -X utf8 scripts/pilot_xlmr_provenance.py --verify-run results/versions/xlmr_provenance_pilot_v1/20260923T052217323306Z_4479f965
```

Running the script without `--verify-run` starts another training-only pilot in a
new version directory; it never starts full extraction.
