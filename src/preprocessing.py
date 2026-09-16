"""Versioned, user-key-only preprocessing for notebook 02.

Nothing is written until export(). Legacy final_data is never an input. Canonical
tables retain missing values; finite matrices have separate observation masks.
"""
from __future__ import annotations

import ast
import gzip
import hashlib
import html
import importlib.metadata
import io
import json
import platform
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_user_key(series):
    value = series.astype("string").str.strip().str.lstrip("@").str.lower()
    return value.mask(value.isin(["", "nan", "none", "null", "<na>"]))


def nonblank(series):
    return series.notna() & series.astype("string").str.strip().ne("").fillna(False)


def clean_text(value):
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", html.unescape(str(value)))
    value = value.replace("ي", "ی").replace("ى", "ی").replace("ك", "ک")
    value = re.sub(r"https?://\S+|www\.\S+", " URL ", value, flags=re.I)
    value = re.sub(r"(?<!\w)@\w+", " @USER ", value)
    return re.sub(r"\s+", " ", value).strip()


def derive_features(accounts):
    """Retain legacy formulas; preserve unknown inputs rather than invent zeros."""
    out = accounts.copy()
    description = out["description"].astype("string")
    out["description_length"] = description.str.len().astype(float)
    for source, target in [("name", "num_digits_in_name"), ("screen_name", "num_digits_in_username")]:
        out[target] = out[source].map(lambda v: np.nan if pd.isna(v) else sum(c.isdigit() for c in str(v)))
    out["url_in_description"] = description.str.contains(r"https?://|www\.", case=False, regex=True).astype("Float64")
    followers = pd.to_numeric(out["followers_count"], errors="coerce")
    friends = pd.to_numeric(out["friends_count"], errors="coerce")
    valid = np.isfinite(followers) & np.isfinite(friends) & followers.ge(0) & friends.ge(0)
    ratio = pd.Series(np.nan, index=out.index, dtype=float)
    positive = valid & friends.gt(0)
    ratio.loc[positive] = followers.loc[positive] / friends.loc[positive]
    ratio.loc[valid & friends.eq(0)] = followers.loc[valid & friends.eq(0)]
    out["followers_friend_ratio"] = ratio
    return out


def validate_values(frame, config, source):
    """Deterministic validation only; no fitting, filling, or clipping."""
    result = pd.DataFrame(index=frame.index)
    reports = []
    true = {"true", "1", "1.0", "yes", "y"}
    false = {"false", "0", "0.0", "no", "n"}
    for col in config["feature_order"]:
        raw = frame[col] if col in frame else pd.Series(np.nan, index=frame.index)
        if col in config["binary_features"]:
            value = raw.map(lambda v: 1.0 if str(v).strip().lower() in true else
                            0.0 if str(v).strip().lower() in false else np.nan)
        else:
            value = pd.to_numeric(raw, errors="coerce").astype(float)
        infinite = pd.Series(np.isinf(value.to_numpy(dtype=float)), index=frame.index)
        negative = value.lt(0) if col in config["nonnegative_features"] else pd.Series(False, index=frame.index)
        value = value.mask(infinite | negative)
        result[col] = value.astype(float)
        reports.append({"source": source, "feature": col, "rows": len(frame),
                        "raw_missing": int(raw.isna().sum()), "negative": int(negative.sum()),
                        "infinite": int(infinite.sum()),
                        "nonnull_invalid": int((raw.notna() & value.isna()).sum()),
                        "valid_observed": int(value.notna().sum())})
    return result, pd.DataFrame(reports)


def inspect_neighbor_list(value):
    """Parse observed handles, reporting invalid tokens instead of guessing edges."""
    if pd.isna(value) or str(value).strip().lower() in {"", "nan", "none", "null", "<na>"}:
        return [], "missing", 0
    text = str(value).strip()
    if text.startswith(("[", "{", "(")):
        try:
            values = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return [], "malformed", 0
        if not isinstance(values, (list, tuple)) or any(not isinstance(v, str) for v in values):
            return [], "unsupported", 0
    else:
        values = text.split(",")
    tokens = [v.strip().lstrip("@").lower() for v in values if v.strip()]
    valid = [v for v in tokens if re.fullmatch(r"[a-z0-9_]{1,15}", v)]
    return valid, "nonempty" if tokens else "empty", len(tokens) - len(valid)


def fit_preprocessor(training, config):
    """Fit every statistic on training observations only, including temporal scale."""
    order = config["feature_order"]
    if training.empty:
        raise ValueError("No training users")
    bundle = {"version": config["version"], "feature_order": order,
              "training_user_keys": sorted(training.index.tolist()), "statistics": {}, "scalers": {}}
    for col in order:
        observed = training[col].dropna().astype(float)
        temporal = col in config["groups"]["temporal"]
        binary = col in config["binary_features"]
        stat = {"training_observations": len(observed), "enabled": not observed.empty,
                "temporal": temporal, "binary": binary, "log1p": col in config["log1p_features"],
                "fill_value": None, "center": None, "scale": None}
        if observed.empty:
            bundle["statistics"][col] = stat
            continue
        if binary:
            stat["fill_value"] = float(observed.mode().min())
        elif temporal:
            # No imputed temporal observations enter the scaler fit.
            fitting = observed
        else:
            stat["fill_value"] = float(observed.median())
            fitting = training[col].fillna(stat["fill_value"])
        if not binary:
            fitting = np.log1p(fitting) if stat["log1p"] else fitting
            scaler = RobustScaler().fit(np.asarray(fitting).reshape(-1, 1))
            bundle["scalers"][col] = scaler
            stat["center"], stat["scale"] = float(scaler.center_[0]), float(scaler.scale_[0])
        bundle["statistics"][col] = stat
    return bundle


def transform_features(frame, bundle):
    """Return finite inputs AND observation masks. Never mutate canonical data."""
    matrix = pd.DataFrame(index=frame.index)
    observed = frame[bundle["feature_order"]].notna().astype("int8")
    for col in bundle["feature_order"]:
        stat = bundle["statistics"][col]
        if not stat["enabled"]:
            matrix[col] = 0.0
            continue
        value = frame[col].copy()
        if not stat["temporal"]:
            value = value.fillna(stat["fill_value"])
        if stat["log1p"]:
            value = np.log1p(value)
        if not stat["binary"]:
            value = pd.Series(bundle["scalers"][col].transform(value.to_numpy().reshape(-1, 1)).ravel(), index=frame.index)
        # This placeholder never becomes an observed value or an estimated time statistic.
        matrix[col] = value.fillna(0.0)
    return matrix, observed.add_prefix("observed__")


class PreprocessingRun:
    def __init__(self, root, config_path="configs/preprocessing_v2.json"):
        self.root = Path(root).resolve()
        self.config_path = self.root / config_path
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.reports = {}
        self.tables = {}
        self.stage = "configured"
        order = self.config["feature_order"]
        groups = sum(self.config["groups"].values(), [])
        if len(order) != len(set(order)) or len(groups) != len(set(groups)) or set(order) != set(groups):
            raise ValueError("Feature groups must form an exact nonoverlapping partition")
        forbidden = sum(self.config["excluded_features"].values(), [])
        if set(order) & set(forbidden) or any(k in order for k in ["id", "user_key", "label_binary"]):
            raise ValueError("Forbidden predictive input")

    def load_sources(self):
        self.raw = {}
        self.input_hashes = {}
        for name, relative in self.config["raw_files"].items():
            path = self.root / relative
            self.input_hashes[relative] = sha256(path)
            df = pd.read_csv(path, dtype={"id": "string", "screen_name": "string"}, low_memory=False)
            df["user_key"] = normalize_user_key(df["screen_name"])
            self.raw[name] = df
        for relative in [str(self.config_path.relative_to(self.root)), self.config["split_file"], "src/preprocessing.py"]:
            self.input_hashes[relative] = sha256(self.root / relative)
        self.environment = {"python": platform.python_version(), **{
            p: importlib.metadata.version(p) for p in ["pandas", "numpy", "scikit-learn", "joblib"]}}
        digest = hashlib.sha256(json.dumps({"inputs": self.input_hashes, "environment": self.environment}, sort_keys=True).encode()).hexdigest()[:16]
        self.output_dir = self.root / self.config["output_parent"] / f"{self.config['version']}_{digest}"
        self.reports["source_inventory"] = pd.DataFrame([
            {"source": name, "rows": len(df), "unique_keys": df.user_key.nunique(),
             "missing_key_rows": int(df.user_key.isna().sum()),
             "repeated_key_rows": int((df.user_key.notna() & df.user_key.duplicated(keep=False)).sum())}
            for name, df in self.raw.items()])
        self.stage = "loaded"
        return self.reports["source_inventory"]

    def build_users(self):
        if self.stage != "loaded":
            raise RuntimeError("Load raw sources first")
        excluded = self.config["quality_exclusions"]
        self.tables["audit/missing_identity_accounts.csv"] = self.raw["accounts"].loc[self.raw["accounts"].user_key.isna()]
        self.tables["audit/quality_excluded_accounts.csv"] = self.raw["accounts"].loc[self.raw["accounts"].user_key.isin(excluded)]
        a = self.raw["annotated"].copy()
        if a.user_key.isna().any() or a.user_key.duplicated().any():
            raise ValueError("Annotated identities missing/duplicated: explicit resolution required")
        accounts = self.raw["accounts"].loc[self.raw["accounts"].user_key.notna() & ~self.raw["accounts"].user_key.isin(excluded)].copy()
        if accounts.user_key.duplicated().any():
            raise ValueError("Duplicate account keys: explicit resolution required")
        invalid = ~accounts.user_key.str.fullmatch(r"[a-z0-9_]{1,15}", na=False)
        if invalid.any():
            raise ValueError(f"Unexpected malformed account keys: {accounts.loc[invalid, 'user_key'].tolist()}")
        if not set(a.user_key) <= set(accounts.user_key):
            raise ValueError("Annotated user absent from canonical source; do not silently drop")
        self.annotation = a.set_index("user_key")
        accounts = accounts.set_index("user_key").sort_index()
        measured = derive_features(accounts)
        values, quality = validate_values(measured, self.config, "canonical_19k")
        reference = ["id", "screen_name", "name", "description", "clean_description", "created_at", "last_updated"]
        users = accounts.reindex(columns=reference).join(values)
        users["label_raw"] = self.annotation[self.config["label_column"]].reindex(users.index)
        users["label_binary"] = users.label_raw.map(self.config["label_map"]).astype("Int64")
        annotated = users.index.isin(self.annotation.index)
        self.tables["audit/annotation_exclusions.csv"] = users.loc[annotated & users.label_binary.isna()].reset_index()
        users = users.loc[~annotated | users.label_binary.notna()].copy()
        self.users = users
        self.values = values.reindex(users.index)
        self.reports["value_validation"] = quality
        self.reports["population"] = pd.DataFrame([
            {"population": "binary_labeled", "users": int(users.label_binary.notna().sum())},
            {"population": "true_unlabeled", "users": int(users.label_binary.isna().sum())},
            {"population": "all_annotated", "users": len(a)}])
        temporal = self.config["groups"]["temporal"]
        raw_annotated_values, _ = validate_values(self.annotation, self.config, "annotated")
        gold_keys = users.index[users.label_binary.notna()]
        temporal_provenance = pd.DataFrame(index=users.index)
        temporal_provenance["canonical_source"] = self.config["canonical_source"]
        temporal_provenance["source_row_present_annotated"] = users.index.isin(self.annotation.index)
        for col in temporal:
            temporal_provenance[f"annotated_observed__{col}"] = raw_annotated_values[col].reindex(users.index).notna()
            temporal_provenance[f"canonical_observed__{col}"] = self.values[col].notna()
            temporal_provenance[f"source__{col}"] = np.where(self.values[col].notna(), "raw_19k", "missing_or_invalid")
        self.tables["provenance/temporal_by_user.csv"] = temporal_provenance.reset_index()
        source_coverage = []
        for name, view in [("annotated_binary_raw", raw_annotated_values.reindex(gold_keys)),
                           ("canonical_binary_raw19k", self.values.reindex(gold_keys)),
                           ("canonical_unlabeled_raw19k", self.values.loc[users.label_binary.isna()])]:
            counts = view[temporal].notna().sum(axis=1)
            source_coverage.append({"source": name, "users": len(view), "any_temporal": int(counts.gt(0).sum()),
                                    "complete_temporal": int(counts.eq(len(temporal)).sum()), "absent_temporal": int(counts.eq(0).sum())})
        self.reports["temporal_source_coverage"] = pd.DataFrame(source_coverage)
        comparisons = []
        for col in self.config["feature_order"]:
            left = raw_annotated_values[col].reindex(gold_keys)
            right = self.values[col].reindex(gold_keys)
            both = left.notna() & right.notna()
            comparisons.append({"feature": col, "compared": int(both.sum()),
                "disagreements": int((~np.isclose(left[both], right[both], rtol=1e-5, atol=1e-8)).sum()),
                "only_19k_observed": int((left.isna() & right.notna()).sum()),
                "rule": "19K remains canonical regardless of label; no cross-source filling"})
        self.reports["source_comparison"] = pd.DataFrame(comparisons)
        # These indicators always describe observations BEFORE statistical filling.
        for group, columns in self.config["groups"].items():
            self.users[f"has_{'behavior' if group == 'behavioral' else group}"] = self.values[columns].notna().any(axis=1).astype("int8")
        self.users["temporal_observed_count"] = self.values[temporal].notna().sum(axis=1).astype("int8")
        self.users["temporal_complete"] = self.users.temporal_observed_count.eq(len(temporal)).astype("int8")
        self.stage = "users"
        return self.reports["population"]

    def freeze_splits(self):
        if self.stage != "users":
            raise RuntimeError("Build canonical users first")
        assignments = pd.read_csv(self.root / self.config["split_file"], dtype={"user_key": "string"})
        if assignments.user_key.isna().any() or assignments.user_key.duplicated().any():
            raise ValueError("Frozen split keys missing or duplicated")
        if not assignments.user_key.equals(normalize_user_key(assignments.user_key)):
            raise ValueError("Frozen split keys are not normalized")
        assignments = assignments.set_index("user_key")
        gold = self.users.loc[self.users.label_binary.notna()]
        if set(assignments.index) != set(gold.index):
            raise ValueError("Frozen split does not exactly cover gold users; no automatic resplit")
        if not assignments.label_binary.astype(int).equals(gold.label_binary.reindex(assignments.index).astype(int)):
            raise ValueError("Frozen labels disagree with raw annotations")
        if set(assignments.split) != {"train", "validation", "test"}:
            raise ValueError("Frozen split names invalid")
        for split, frame in assignments.groupby("split"):
            if set(frame.label_binary) != {0, 1}:
                raise ValueError(f"Both classes required in {split}")
        self.assignments = assignments
        self.users["split"] = assignments.split.reindex(self.users.index).fillna("unlabeled")
        self.reports["split_summary"] = self.users.groupby(["split", "label_binary"], dropna=False).size().rename("users").reset_index()
        self.stage = "split"
        return self.reports["split_summary"]

    def prepare_text(self):
        if self.stage != "split":
            raise RuntimeError("Freeze user splits first")
        tweets = self.raw["tweets"].copy()
        tweets["source_row"] = np.arange(len(tweets))
        self.tables["audit/tweets_outside_population.csv"] = tweets.loc[~tweets.user_key.isin(self.users.index), ["source_row", "user_key"]]
        tweets = tweets.loc[tweets.user_key.isin(self.users.index)].copy()
        self.users["has_raw_tweets"] = self.users.index.isin(tweets.user_key).astype("int8")
        tweets["split"] = tweets.user_key.map(self.users.split)
        tweets["text_model"] = tweets.text.map(clean_text)
        tweets["text_key"] = tweets.text_model.str.lower().str.strip()
        tweets["published_at_parsed"] = pd.to_datetime(tweets.published_at, errors="coerce", utc=True, format="mixed")
        self.reports["tweet_time_quality"] = pd.DataFrame([{"rows": len(tweets), "missing_or_invalid_timestamp": int(tweets.published_at_parsed.isna().sum())}])
        keep = ["user_key", "source_row", "published_at", "published_at_parsed", "text", "text_model", "text_key"]
        seen = set()
        rows = []
        prepared = []
        for split in ["train", "validation", "test", "unlabeled"]:
            original = tweets.loc[tweets.split.eq(split)]
            data = original.loc[original.text_model.str.len().gt(0)].sort_values(
                ["user_key", "published_at_parsed", "source_row"], ascending=[True, False, True], kind="stable")
            data = data.drop_duplicates(["user_key", "text_key"], keep="first")
            dedup_count = len(data)
            if split != "train":
                data = data.loc[~data.text_key.isin(seen)]
            # Match historical decontamination before tweet capping.
            seen.update(data.text_key)
            decontaminated = len(data)
            data = data.groupby("user_key", sort=False).head(self.config["text"]["max_tweets_per_user"])
            self.tables[f"splits/text/{split}_text.csv"] = data[keep].reset_index(drop=True)
            prepared.extend(data.user_key.unique())
            rows.append({"split": split, "raw_rows": len(original), "after_within_user_dedup": dedup_count,
                         "after_decontamination": decontaminated, "retained_tweets": len(data),
                         "users_with_text": data.user_key.nunique(),
                         "all_users": int(self.users.split.eq(split).sum())})
        self.users["has_tweet_text"] = self.users.index.isin(prepared).astype("int8")
        self.users["has_description_raw"] = nonblank(self.users.description).astype("int8")
        raw_ok = nonblank(self.users.description)
        provided_ok = nonblank(self.users.clean_description)
        text = self.users.description.where(raw_ok, self.users.clean_description.where(provided_ok))
        self.users["description_model"] = text.map(clean_text)
        self.users["description_source"] = np.where(raw_ok, "raw_description", np.where(provided_ok, "provided_clean_description", "missing"))
        self.users["has_description"] = self.users.description_model.str.len().gt(0).astype("int8")
        self.users["has_text"] = (self.users.has_tweet_text.eq(1) | self.users.has_description.eq(1)).astype("int8")
        self.reports["text_preparation"] = pd.DataFrame(rows)
        self.stage = "text"
        return self.reports["text_preparation"]

    def prepare_graph(self):
        if self.stage != "text":
            raise RuntimeError("Prepare text before graph")
        observations, parse_rows, owner_records = [], [], set()
        for row_number, row in enumerate(self.raw["graph"].itertuples(index=False)):
            owner = row.user_key
            if pd.isna(owner) or not re.fullmatch(r"[a-z0-9_]{1,15}", owner):
                parse_rows.append({"source_row": row_number, "owner_key": owner, "field": "owner", "state": "invalid_owner", "invalid_tokens": 1, "self_edges": 0})
                continue
            owner_records.add(owner)
            for field in ["followers", "following"]:
                neighbors, state, invalid = inspect_neighbor_list(getattr(row, field))
                parse_rows.append({"source_row": row_number, "owner_key": owner, "field": field,
                                   "state": state, "invalid_tokens": invalid, "self_edges": neighbors.count(owner)})
                for neighbor in neighbors:
                    if neighbor == owner:
                        continue
                    source, target = (neighbor, owner) if field == "followers" else (owner, neighbor)
                    observations.append((source, target, "follows", owner))
        columns = ["source_key", "target_key", "relation", "owner_key"]
        observations = pd.DataFrame(observations, columns=columns).drop_duplicates().sort_values(columns).reset_index(drop=True)
        edges = observations.drop_duplicates(columns[:3])[columns[:3]].reset_index(drop=True)
        self.tables["graph/edge_observations.csv.gz"] = observations
        self.tables["graph/directed_edges_audit_only.csv.gz"] = edges
        self.reports["graph_parse"] = pd.DataFrame(parse_rows)
        self.users["has_graph_owner_record"] = self.users.index.isin(owner_records).astype("int8")
        self.users["has_graph_owner"] = self.users.index.isin(observations.owner_key).astype("int8")
        summaries = []
        self.users["has_graph"] = 0
        self.users["has_graph_node"] = 0
        for split in ["train", "validation", "test", "unlabeled"]:
            keys = set(self.users.index[self.users.split.eq(split)])
            allowed_gold = set(self.assignments.index[self.assignments.split.eq("train" if split == "unlabeled" else split)])
            forbidden = set(self.assignments.index) - allowed_gold
            obs = observations.loc[observations.owner_key.isin(keys) &
                ~observations.source_key.isin(forbidden) & ~observations.target_key.isin(forbidden)]
            model_edges = obs[columns[:3]].drop_duplicates().reset_index(drop=True)
            nodes = set(model_edges.source_key) | set(model_edges.target_key)
            population_index = self.users.index[self.users.split.eq(split)]
            self.users.loc[population_index, "has_graph_node"] = population_index.isin(nodes).astype(int)
            self.users.loc[population_index, "has_graph"] = (
                self.users.loc[population_index, "has_graph_owner"].eq(1)
                & population_index.isin(nodes)).astype(int)
            self.tables[f"splits/graph/{split}_graph_edges.csv.gz"] = model_edges
            self.tables[f"splits/graph/{split}_edge_owners.csv.gz"] = obs.reset_index(drop=True)
            summaries.append({"split": split, "owner_users": len(set(obs.owner_key)), "nodes": len(nodes),
                              "edges": len(model_edges), "forbidden_gold_nodes": len(nodes & forbidden)})
        self.reports["graph_split_summary"] = pd.DataFrame(summaries)
        self.stage = "graph"
        return self.reports["graph_split_summary"]

    def fit_transform(self):
        if self.stage != "graph":
            raise RuntimeError("Prepare modalities first")
        training = self.values.loc[self.users.split.eq("train")]
        self.preprocessor = fit_preprocessor(training, self.config)
        self.matrices = {}
        for split in ["train", "validation", "test", "unlabeled"]:
            keys = self.assignments.index[self.assignments.split.eq(split)] if split != "unlabeled" else self.users.index[self.users.split.eq(split)]
            source = self.users.reindex(keys)
            matrix, observed = transform_features(self.values.reindex(keys), self.preprocessor)
            self.matrices[split] = matrix
            self.tables[f"splits/users/{split}_users.csv"] = source.reset_index()
            self.tables[f"splits/tabular/{split}_tabular.csv"] = source[["screen_name", "label_binary"]].join(matrix).reset_index()
            self.tables[f"splits/masks/{split}_observation_masks.csv"] = observed.reset_index()
            mask_columns = [c for c in source if c.startswith("has_") or c in ["temporal_complete", "temporal_observed_count"]]
            self.tables[f"splits/masks/{split}_modality_masks.csv"] = source[mask_columns].reset_index()
            self.tables[f"splits/text/{split}_descriptions.csv"] = source[["description_model", "description_source", "has_description"]].reset_index()
        self.reports["fitted_statistics"] = pd.DataFrame.from_dict(self.preprocessor["statistics"], orient="index").rename_axis("feature").reset_index()
        temporal = self.config["groups"]["temporal"]
        self.reports["temporal_by_split_label"] = self.users.groupby(["split", "label_binary"], dropna=False).agg(
            users=("has_temporal", "size"), available=("has_temporal", "sum"), complete=("temporal_complete", "sum")).reset_index()
        self.tables["users/canonical_users.csv"] = self.users.reset_index()
        self.tables["users/labeled_binary.csv"] = self.users.loc[self.users.label_binary.notna()].reset_index()
        self.tables["users/true_unlabeled.csv"] = self.users.loc[self.users.label_binary.isna()].reset_index()
        self.tables["splits/user_split_assignment.csv"] = self.assignments.reset_index()
        self.stage = "transformed"
        return self.reports["temporal_by_split_label"]

    def validate(self):
        if self.stage not in {"transformed", "validated"}:
            raise RuntimeError("Transform all populations first")
        gold = set(self.users.index[self.users.label_binary.notna()])
        unlabeled = set(self.users.index[self.users.label_binary.isna()])
        expected = set(self.annotation.index[self.annotation[self.config["label_column"]].isin(self.config["label_map"])])
        temporal = self.config["groups"]["temporal"]
        checks = {
            "gold_users_preserved": gold == expected,
            "unique_user_keys": self.users.index.is_unique and not self.users.index.isna().any(),
            "unlabeled_disjoint_all_annotations": not bool(unlabeled & set(self.annotation.index)),
            "frozen_split_preserved": set(self.assignments.index) == gold,
            "training_fit_keys_exact": set(self.preprocessor["training_user_keys"]) == set(self.assignments.index[self.assignments.split.eq("train")]),
            "temporal_mask_before_fill": self.users.has_temporal.eq(self.values[temporal].notna().any(axis=1).astype(int)).all(),
            "temporal_canonical_missingness_preserved": self.users[temporal].isna().equals(self.values[temporal].isna()),
            "no_cross_split_gold_graph_nodes": self.reports["graph_split_summary"].forbidden_gold_nodes.eq(0).all(),
            "source_files_unchanged": all(sha256(self.root / p) == h for p, h in self.input_hashes.items()),
        }
        text_sets = []
        for split in ["train", "validation", "test", "unlabeled"]:
            matrix = self.matrices[split]
            users = self.tables[f"splits/users/{split}_users.csv"]
            masks = self.tables[f"splits/masks/{split}_observation_masks.csv"]
            checks[f"{split}_all_users_retained"] = len(matrix) == int(self.users.split.eq(split).sum())
            checks[f"{split}_finite_matrix"] = bool(np.isfinite(matrix.to_numpy()).all())
            checks[f"{split}_feature_order"] = matrix.columns.tolist() == self.config["feature_order"]
            checks[f"{split}_aligned_keys"] = matrix.index.tolist() == users.user_key.tolist() == masks.user_key.tolist()
            observed = masks.set_index("user_key").reindex(matrix.index)
            checks[f"{split}_temporal_placeholders_masked"] = all(
                matrix.loc[observed[f"observed__{col}"].eq(0), col].eq(0).all() for col in temporal)
            text = self.tables[f"splits/text/{split}_text.csv"]
            checks[f"{split}_text_belongs_to_users"] = set(text.user_key) <= set(users.user_key)
            text_sets.append(set(text.text_key))
        checks["cross_split_exact_text_disjoint"] = all(not a & b for i, a in enumerate(text_sets) for b in text_sets[i + 1:])
        self.reports["validation"] = pd.DataFrame([{"check": k, "passed": bool(v)} for k, v in checks.items()])
        if not all(checks.values()):
            raise AssertionError(self.reports["validation"].loc[~self.reports["validation"].passed].to_dict("records"))
        self.stage = "validated"
        return self.reports["validation"]

    def export(self):
        if self.stage != "validated":
            raise RuntimeError("Validation must pass before export")
        artifacts = {}
        base = self.output_dir.resolve()
        if not base.is_relative_to((self.root / "final_data/versions").resolve()):
            raise ValueError("Output must remain in final_data/versions")

        def write(relative, content):
            path = base / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_bytes() != content:
                    raise RuntimeError(f"Refusing to overwrite different versioned artifact: {path}")
            else:
                path.write_bytes(content)
            artifacts[relative] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}

        def write_json(relative, value):
            write(relative, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))

        for name, report in self.reports.items():
            self.tables[f"audit/{name}.csv"] = report
        for relative, frame in self.tables.items():
            content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
            if relative.endswith(".gz"):
                content = gzip.compress(content, mtime=0)
            write(relative, content)
            artifacts[relative]["rows"] = len(frame)
        write_json("config/resolved_policy.json", self.config)
        write_json("preprocessing/fitted_statistics.json", {k: v for k, v in self.preprocessor.items() if k != "scalers"})
        buffer = io.BytesIO()
        joblib.dump(self.preprocessor, buffer)
        write("preprocessing/preprocessor.joblib", buffer.getvalue())
        registry = []
        for col in self.config["feature_order"]:
            group = next(g for g, cols in self.config["groups"].items() if col in cols)
            definition_key = "name_digit_counts" if col in {"num_digits_in_name", "num_digits_in_username"} else col
            registry.append({"feature": col, "group": group,
                "type": "binary" if col in self.config["binary_features"] else "continuous",
                "source": "derived_from_raw19k" if col in self.config["derived_features"] else "raw19k",
                "derivation": self.config["definition_changes"].get(definition_key, "Imported statistic; upstream formula/window not verified"),
                "enabled": self.preprocessor["statistics"][col]["enabled"]})
        write_json("config/feature_registry.json", registry)
        manifest = {"version": self.config["version"], "inputs": self.input_hashes,
                    "environment": self.environment, "artifacts": artifacts.copy(),
                    "identity_key": "user_key", "fit_scope": "train_only", "feature_order": self.config["feature_order"],
                    "graph_protocol": self.config["graph"], "missingness": self.config["missingness"],
                    "limitations": ["Upstream temporal formulas/windows unverified", "Partial graph and tweet observations",
                                    "Historical test reuse", "Legacy feature choices retained; no claim of new blind selection"],
                    "downstream_contract": "Load feature_order explicitly. Join masks by user_key. Gate missing modalities. Never use audit-only global graph for training."}
        # Manifest written last marks a completed run. It excludes its own recursive hash.
        write_json("manifest.json", manifest)
        self.manifest = manifest
        return self.output_dir


def verify_export(output_dir):
    """Reload every export: verify bytes, users, mask alignment, and fitted inference."""
    base = Path(output_dir)
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    for relative, info in manifest["artifacts"].items():
        if sha256(base / relative) != info["sha256"]:
            raise AssertionError(f"Artifact hash mismatch: {relative}")
    bundle = joblib.load(base / "preprocessing/preprocessor.joblib")
    for split in ["train", "validation", "test", "unlabeled"]:
        users = pd.read_csv(base / f"splits/users/{split}_users.csv", dtype={"user_key": "string"}).set_index("user_key")
        saved = pd.read_csv(base / f"splits/tabular/{split}_tabular.csv", dtype={"user_key": "string"}).set_index("user_key")
        masks = pd.read_csv(base / f"splits/masks/{split}_observation_masks.csv", dtype={"user_key": "string"}).set_index("user_key")
        actual, actual_masks = transform_features(users, bundle)
        if not users.index.equals(saved.index) or not users.index.equals(masks.index):
            raise AssertionError(f"Key alignment failed: {split}")
        np.testing.assert_allclose(saved[manifest["feature_order"]], actual, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(masks, actual_masks)
    return {"verified_artifacts": len(manifest["artifacts"]), "reloaded_transform_populations": 4}
