# Dataset Audit, Preprocessing, and Feature Engineering Report

## Project Overview

This project develops a multimodal framework for Twitter/X bot detection using user profile information, behavioral and temporal features, tweet content, and partially available follower/following graph information.

The preprocessing pipeline was designed with three main goals:

1. Construct a reliable binary bot/human gold-standard dataset.
2. Preserve multiple modalities without requiring every user to have all modalities.
3. Prevent data leakage between training, validation, and test sets.

The complete preprocessing workflow is implemented in two notebooks:

- `01_dataset_exploration_and_audit.ipynb`
- `02_data_preprocessing_and_feature_engineering.ipynb`

---

# 1. Notebook 01 — Dataset Exploration and Audit

## 1.1 Input Datasets

Four raw datasets were inspected.

| Dataset | Shape | Role |
|---|---:|---|
| `1000user_sheet - 1k_users.csv` | 1,103 × 101 | Manually labeled user dataset |
| `1000user_sheet - tweets_meta_data.csv` | 75,913 × 38 | Raw tweet-level data |
| `all_users_new - 19k.csv` | 19,510 × 75 | Main user-level profile and behavioral dataset |
| `all_users_new - follower_following.csv` | 3,453 × 4 | Collected follower/following information |

The 1K labeled dataset contains five final annotation categories:

| Label | Users |
|---|---:|
| Human | 772 |
| Bot | 187 |
| Can't find | 58 |
| Unverified | 57 |
| News Agent | 29 |
| **Total** | **1,103** |

The main binary classification task uses only the verified `Human` and `Bot` labels, resulting in:

**959 Gold Binary Users = 772 Human + 187 Bot**

---

## 1.2 User Identity Normalization

A canonical user identifier named `user_key` was constructed from `screen_name` by:

- removing leading `@`;
- stripping whitespace;
- converting usernames to lowercase.

After normalization:

| Dataset | Valid Unique Users |
|---|---:|
| Labeled dataset | 1,103 |
| 19K user dataset | 19,435 |
| Tweet dataset | 1,099 |
| Graph owner dataset | 3,359 |

All 1,103 labeled users were found inside the valid 19K user dataset.

Therefore:

**19,435 valid users - 1,103 labeled users = 18,332 initially unlabeled users**

A later data-quality check removed one corrupted unlabeled record, resulting in a final clean unlabeled pool of **18,331 users**.

---

## 1.3 Modality Availability

The binary Gold dataset contains 959 users.

All 959 users have profile, behavioral/temporal, and raw tweet information.

Graph information is only available for a subset.

| Modality | Binary Gold Users |
|---|---:|
| Profile | 959 / 959 |
| Behavioral / Temporal | 959 / 959 |
| Raw Tweets | 959 / 959 |
| Description | 915 / 959 |
| Graph owner information | 276 / 959 |

Among the 276 graph-enabled binary users:

- 232 are Human;
- 44 are Bot.

This partial graph coverage motivated a multimodal design in which graph information is treated as an optional modality rather than a mandatory model input.

---

## 1.4 Tweet Dataset Audit

The raw tweet dataset contains:

**75,913 tweet records from 1,099 users**

All 959 users used for the binary classification task have at least one tweet record.

Tweet-level information includes text, language, publication time, engagement counts, tweet type, reply metadata, mentions, and other metadata.

Tweets were intentionally kept as a separate modality instead of being flattened into the user-level table.

This prevents user duplication and allows the future text encoder to independently process tweet content.

---

## 1.5 Graph Dataset Audit

Follower/following information was converted into a directed graph representation.

The resulting graph contains:

**796,485 directed `follows` edges**

with approximately:

**230,791 unique graph nodes**

The processed graph files are:

- `graph_edges.csv`
- `graph_user_statistics.csv`

The graph contains many nodes that are not part of the Gold dataset because followers and followed accounts can exist outside the collected user population.

A user appearing somewhere in the graph is not automatically considered to have a complete graph modality. The `has_graph` flag is based on whether follower/following information was directly collected for that user.

---

## 1.6 Leakage and Feature Provenance Audit

Columns representing external detector outputs, LLM judgments, annotation decisions, or tagger information were explicitly excluded from the predictive feature space.

Examples include:

`Botometer Score`, `LLM(ChatGPT)`, `LLM(Gemini)`, `Grok 4`, tagger names, probabilities, reasons, and manually generated annotation questions.

These fields were excluded because they could introduce target leakage or artificially inflate model performance.

The final class label was taken exclusively from the manually determined final-label column of the 1K labeled dataset.

For binary classification:

- Human → `0`
- Bot → `1`

---

# 2. Notebook 02 — Data Preprocessing and Feature Engineering

## 2.1 Canonical User Dataset

The 19K dataset was selected as the canonical source for profile, behavioral, and temporal features because its shared columns showed very high consistency with the 1K dataset and generally provided better coverage.

Labels were joined from the 1K Gold dataset using the normalized `user_key`.

The resulting operational datasets were:

| Dataset | Users |
|---|---:|
| Binary labeled users | 959 |
| Clean true unlabeled users | 18,331 |
| Complete multimodal binary subset | 276 |

---

## 2.2 Feature Construction and Leakage Removal

The initial candidate feature space contained profile, behavioral, temporal, and derived user-level features.

Five additional safe features were reconstructed consistently for both labeled and unlabeled users:

- `description_length`
- `followers_friend_ratio`
- `num_digits_in_name`
- `num_digits_in_username`
- `url_in_description`

After removing unusable or effectively constant features, the initial predictive tabular space contained **42 approved features**.

The following unusable features were removed:

`fast_followers_count`, `verified`, `is_translator`, `want_retweets`, and `translator_type`.

---

## 2.3 Data Quality Control

One unlabeled user was identified as corrupted because the same Twitter/X ID-like value was repeated across many unrelated numerical features.

Instead of attempting unreliable imputation, the complete row was excluded.

The unlabeled pool therefore changed from:

**18,332 → 18,331 users**

A data-quality exclusion file was stored to preserve reproducibility.

---

# 3. Train / Validation / Test Split

The 959 Gold Binary users were split at the **user level**, not at the tweet level.

The split was stratified simultaneously by:

`label_binary × has_graph`

This preserves both class balance and graph availability across all three subsets.

| Split | Users | Human | Bot | Graph Owners |
|---|---:|---:|---:|---:|
| Train | 671 | 540 | 131 | 193 |
| Validation | 144 | 116 | 28 | 42 |
| Test | 144 | 116 | 28 | 41 |

The final ratio is approximately:

- Train: 70%
- Validation: 15%
- Test: 15%

No user appears in more than one split.

The split was frozen and saved so that all subsequent experiments use exactly the same users.

---

# 4. Tweet Split and Text Leakage Prevention

Tweets were not randomly divided.

Instead, every tweet inherited the split of its corresponding user.

The original tweet distribution associated with Gold Binary users was:

| Split | Tweet Rows |
|---|---:|
| Train | 45,443 |
| Validation | 13,968 |
| Test | 11,464 |

This guarantees that tweets belonging to the same account never appear in multiple data splits.

---

## 4.1 Text Cleaning

Minimal text normalization was used to preserve semantic content.

The pipeline performs:

- Unicode NFKC normalization;
- normalization of Persian characters;
- URL replacement with a generic `URL` token;
- mention replacement with `@USER`;
- whitespace normalization.

Hashtags, emojis, Persian text, and English text are preserved.

---

## 4.2 Tweet Deduplication

Tweet IDs were not used for deduplication because large Twitter/X identifiers were stored using floating-point/scientific notation, which can lose integer precision.

Instead, exact normalized text was used.

Duplicate text from the same user was removed while identical text from different users was initially preserved.

Cross-split exact-text decontamination was then performed:

- Validation texts already appearing in Train were removed.
- Test texts appearing in Train or Validation were removed.

After preprocessing, exact text overlap between Train, Validation, and Test is **zero**.

---

## 4.3 Maximum Tweets per User

Tweet counts were highly imbalanced across users.

To prevent highly active users from dominating the text representation, a maximum of:

**50 most recent unique tweets per user**

was selected.

The final text datasets contain:

| Split | Tweets | Users with Text |
|---|---:|---:|
| Train | 12,976 | 671 |
| Validation | 2,618 | 144 |
| Test | 3,006 | 143 |

One Test user had no usable text after exact-text decontamination. The user was retained in the frozen Test set and the text modality is treated as missing for that account.

---

# 5. XLM-R Tokenization Policy

The multilingual `xlm-roberta-base` tokenizer was used to analyze true token lengths.

Training-set token statistics were:

| Statistic | Tokens |
|---|---:|
| Median | 43 |
| P90 | 87 |
| P95 | 96 |
| P97 | 104 |
| P99 | 219.8 |

Using `max_length = 128` truncates only approximately:

- 1.91% of Train tweets;
- 1.60% of Validation tweets;
- 2.46% of Test tweets.

Therefore the final text configuration is:

- **Encoder:** `xlm-roberta-base`
- **Maximum tweets/user:** `50`
- **Maximum sequence length:** `128`
- **Truncation:** enabled
- **Padding:** dynamic per batch

---

# 6. Tabular Preprocessing

Missing values were extremely limited.

Only three numerical features contained missing values:

- `followers_count`
- `friends_count`
- `followers_friend_ratio`

Median imputation was fitted **only on the Train set**.

No imputer statistics were learned from Validation or Test.

---

## 6.1 Redundancy Analysis

Feature redundancy was evaluated using Train only.

`status_count` was found to be an exact duplicate of `statuses_count`.

`normal_followers_count` was nearly identical to `followers_count`, with approximately 99.85% identical rows and nearly perfect correlation.

The redundant features removed were therefore:

- `status_count`
- `normal_followers_count`

This reduced the predictive feature space from:

**42 → 40 final tabular features**

The final feature space consists of:

**33 continuous features + 7 effective binary features**

---

## 6.2 Distribution Transformation

Most numerical Twitter/X features exhibited strong right-skewed distributions.

Of the final continuous features, 30 were transformed using:

`log1p(x)`

The transformation was selected based only on the Train distribution.

Binary features were not log-transformed.

---

## 6.3 Scaling

Because many variables contained substantial outliers even after transformation, `RobustScaler` was selected.

The scaler was fitted **only on Train** and then applied without refitting to Validation and Test.

The final tabular model input therefore contains:

**40 dimensions**

with no missing or infinite values.

---

# 7. Inductive Graph Construction

The original graph contained cross-split Gold-user connections.

Among 4,600 Gold-to-Gold graph edges:

**48.57% were cross-split edges**

Using the complete graph directly could therefore expose Validation/Test structure during model training.

To prevent this, leakage-safe **inductive graph splits** were created.

Cross-split Gold-user neighbors were removed while external and unlabeled neighboring nodes were preserved as structural context.

The final graph subsets are:

| Split | Graph Owners | Edges |
|---|---:|---:|
| Train | 193 | 60,936 |
| Validation | 42 | 13,006 |
| Test | 41 | 12,536 |

No Gold user from Validation or Test appears as a neighboring Gold node inside the Train graph, and equivalent restrictions are applied to the other splits.

---

# 8. Final Operational Data Structure

The final project data is organized into separate synchronized modalities rather than one flattened dataset.

```text
final_data/
│
├── users/
│   ├── modeling_labeled_binary.csv
│   ├── modeling_true_unlabeled.csv
│   └── modeling_complete_multimodal_binary.csv
│
├── tweets/
│   └── tweets_meta_data.csv
│
├── graph/
│   ├── graph_edges.csv
│   └── graph_user_statistics.csv
│
├── config/
│   ├── model_feature_policy.json
│   ├── data_quality_exclusions.json
│   ├── tabular_preprocessing_policy.json
│   ├── text_preprocessing_policy.json
│   └── preprocessing_manifest.json
│
└── splits/
    ├── users/
    ├── tabular/
    ├── tweets/
    ├── text/
    └── graph/
```

**All modalities are synchronized using the canonical `user_key`.**

This canonical key is used to align user-level tabular data, tweet records, text-encoder inputs, graph-owner information, graph splits, labels, and modality-availability masks without relying on potentially imprecise numeric Twitter/X IDs.

---

# 9. Leakage Prevention Summary

The preprocessing pipeline explicitly prevents several forms of data leakage.

| Potential Leakage | Prevention |
|---|---|
| Same user in multiple splits | User-level split |
| Tweets from one user across splits | Tweets inherit user split |
| Exact duplicate text across splits | Cross-split text decontamination |
| Validation/Test statistics in imputation | Imputer fitted on Train only |
| Validation/Test statistics in scaling | Scaler fitted on Train only |
| Feature-selection leakage | Redundancy analysis performed on Train only |
| Graph structure leakage | Inductive graph construction |
| LLM/external detector leakage | Detector outputs excluded from predictive features |

---

# 10. Final Preprocessing State

At the end of Notebook 02, the complete preprocessing pipeline is frozen.

The final supervised experimental setup is:

- **959 Gold Binary users**
- **Train:** 671
- **Validation:** 144
- **Test:** 144
- **Tabular:** 40 features
- **Text Encoder:** XLM-RoBERTa
- **Tweets per user:** maximum 50
- **Token length:** 128
- **Graph evaluation:** inductive

The remaining **18,331 unlabeled users** are not used in the initial supervised split. They are reserved for later pseudo-labeling and semi-supervised experiments.

Any preprocessing applied to these users must reuse the already frozen Train-fitted preprocessing objects without refitting.

---

## Conclusion

Notebooks 01 and 02 establish a leakage-aware and reproducible multimodal data pipeline for Twitter/X bot detection.

The resulting dataset preserves profile, behavioral, temporal, textual, and graph information while explicitly handling missing modalities.

The frozen data splits and preprocessing policies provide a stable foundation for subsequent experiments including baseline classification, multimodal fusion, contrastive learning, graph knowledge integration, graph-to-language-model distillation, and semi-supervised pseudo-labeling.
