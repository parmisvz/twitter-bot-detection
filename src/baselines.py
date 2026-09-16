"""Phase-3 compatibility pass: unchanged baseline families, explicit artifact contracts.

No preprocessing is fitted here. A legacy embedding import is an explicit migration,
not a fallback in the training loader. All predictions are joined by user_key.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import warnings
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SPLITS = ("train", "validation", "test")
MASKS = ("has_profile", "has_temporal", "has_behavior", "has_tweet_text", "has_text",
         "has_description", "has_graph", "temporal_complete")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
                          + "\n", encoding="utf-8")


def checked_artifact(directory, manifest, relative):
    require(relative in manifest["artifacts"], f"Unregistered artifact: {relative}")
    path = (directory / relative).resolve()
    require(path.is_relative_to(directory.resolve()), f"Artifact escapes directory: {relative}")
    require(sha256(path) == manifest["artifacts"][relative]["sha256"],
            f"Artifact hash mismatch: {relative}")
    return path


def keyed(frame, name):
    require("user_key" in frame, f"{name}: user_key missing")
    keys = frame.user_key
    require(keys.notna().all() and keys.is_unique, f"{name}: null/duplicate user_key")
    require(keys.eq(keys.str.strip().str.lstrip("@").str.lower()).all()
            and keys.str.len().gt(0).all(), f"{name}: noncanonical user_key")
    return frame.set_index("user_key")  # Uniqueness checked above; no deprecated pandas argument.


def align(frame, users, name):
    require(set(frame.index) == set(users.index), f"{name}: user_key population mismatch")
    return frame.loc[users.index]


def load_phase2(root, config):
    """Verify registered inputs, identities, labels, splits and modality evidence."""
    root = Path(root)
    path = root / config["phase2_manifest"]
    require(sha256(path) == config["phase2_manifest_sha256"], "Phase-2 manifest hash mismatch")
    manifest = read_json(path)
    require(manifest["identity_key"] == "user_key" and manifest["fit_scope"] == "train_only",
            "Unsupported Phase-2 identity/fit scope")
    order = manifest["feature_order"]
    require(len(order) == len(set(order)) and len(order) > 0, "Invalid feature_order")
    require(not set(order) & {"user_key", "id", "label_binary", "screen_name"}, "Identity/target feature")
    directory = path.parent

    def csv(relative):
        return pd.read_csv(checked_artifact(directory, manifest, relative), dtype={"user_key": str})

    assignment = keyed(csv("splits/user_split_assignment.csv"), "assignment")
    stats = read_json(checked_artifact(directory, manifest, "preprocessing/fitted_statistics.json"))
    require(stats["feature_order"] == order, "Fitted feature order differs from manifest")
    registry = read_json(checked_artifact(directory, manifest, "config/feature_registry.json"))
    temporal = [r["feature"] for r in registry if r["group"] == "temporal" and r["feature"] in order]
    populations = {}
    for split in SPLITS:
        users = keyed(csv(f"splits/users/{split}_users.csv"), split)
        expected = assignment[assignment.split.eq(split)]
        expected = align(expected, users, f"{split} assignment")
        require(users.label_binary.isin([0, 1]).all(), f"{split}: nonbinary labels")
        require(users.label_binary.eq(expected.label_binary).all(), f"{split}: assignment labels differ")
        tab = align(keyed(csv(f"splits/tabular/{split}_tabular.csv"), "tabular"), users, "tabular")
        require(tab.label_binary.eq(users.label_binary).all(), f"{split}: tabular labels differ")
        masks = align(keyed(csv(f"splits/masks/{split}_modality_masks.csv"), "masks"), users, "masks")
        observed = align(keyed(csv(f"splits/masks/{split}_observation_masks.csv"), "observed"), users, "observed")
        require(masks[list(MASKS)].isin([0, 1]).all().all(), f"{split}: invalid modality mask")
        require(masks[list(MASKS)].eq(users[list(MASKS)]).all().all(), f"{split}: masks disagree with users")
        obs_columns = ["observed__" + f for f in order]
        require(observed[obs_columns].isin([0, 1]).all().all(), f"{split}: invalid observation mask")
        require(np.array_equal(observed[obs_columns].to_numpy(), users[order].notna().to_numpy()),
                f"{split}: observation masks disagree with canonical observations")
        counts = observed[["observed__" + f for f in temporal]].sum(axis=1)
        require(masks.has_temporal.eq(counts.gt(0).astype(int)).all()
                and masks.temporal_observed_count.eq(counts).all()
                and masks.temporal_complete.eq(counts.eq(len(temporal)).astype(int)).all(),
                f"{split}: temporal coverage disagrees with observations")
        text = csv(f"splits/text/{split}_text.csv")
        require(text.user_key.notna().all() and set(text.user_key) <= set(users.index),
                f"{split}: text key outside split")
        require(text.text_model.notna().all() and text.text_model.str.strip().ne("").all(),
                f"{split}: empty prepared tweet text")
        require(set(text.user_key) == set(masks.index[masks.has_tweet_text.eq(1)]),
                f"{split}: tweet mask disagrees with text")
        require(masks.has_text.eq((masks.has_tweet_text | masks.has_description)).all(),
                f"{split}: has_text union disagrees")
        x = tab.loc[:, order].to_numpy(dtype=np.float64)
        require(np.isfinite(x).all(), f"{split}: nonfinite Phase-2 tensor; do not refit/impute here")
        # Missing temporal entries are explicit Phase-2 zero placeholders, not new estimates.
        for feature in temporal:
            missing = observed["observed__" + feature].eq(0)
            require(tab.loc[missing, feature].eq(0).all(), f"{split}: missing temporal placeholder changed")
        populations[split] = {"users": users, "masks": masks, "observed": observed,
                              "text": text, "x": x, "y": users.label_binary.to_numpy(dtype=int)}
    keys = [set(populations[s]["users"].index) for s in SPLITS]
    require(all(not keys[i] & keys[j] for i in range(3) for j in range(i)), "Split overlap")
    require(set.union(*keys) == set(assignment.index), "Assignment population not exhausted")
    require(set(stats["training_user_keys"]) == keys[0], "Preprocessing fit users differ from training")
    checked_artifact(directory, manifest, "preprocessing/preprocessor.joblib")
    return {"manifest": manifest, "manifest_path": path, "populations": populations,
            "feature_order": order, "temporal_features": temporal}


def text_digest(frame):
    payload = frame[["user_key", "text_model"]].to_json(orient="values", force_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def artifact_inventory(directory):
    return {p.relative_to(directory).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "manifest.json"}


def register_legacy_embeddings(root, config):
    """One-time, explicit cache migration. Historical encoder revision remains unknown.

    Validate ordered legacy/Phase-2 texts and reproduce cached user means from tweet
    vectors. This cannot prove the historical encoder's revision or row provenance.
    """
    root = Path(root)
    data = load_phase2(root, config)
    target = root / config["embedding_manifest"]
    if target.exists():
        load_embeddings(root, config, data)
        return target
    require(not target.parent.exists(), "Incomplete embedding directory exists; inspect before retrying")
    migration = config["legacy_embedding_import"]
    source_hashes = {}
    arrays = {}
    digests = {}
    for split, alias in migration["split_aliases"].items():
        text_path = root / migration["text_dir"] / f"{split}_text.csv"
        old_text = pd.read_csv(text_path, dtype={"user_key": str})
        text = data["populations"][split]["text"]
        require(text_digest(old_text) == text_digest(text), f"{split}: ordered legacy text differs")
        source_hashes[text_path.relative_to(root).as_posix()] = sha256(text_path)
        paths = {name: root / migration["embedding_dir"] / f"{alias}_{name}.npy"
                 for name in ("tweet_embeddings", "user_embeddings", "user_keys")}
        for path in paths.values():
            source_hashes[path.relative_to(root).as_posix()] = sha256(path)
        tweets = np.load(paths["tweet_embeddings"], allow_pickle=False)
        users = np.load(paths["user_embeddings"], allow_pickle=False)
        # Sole legacy pickle read: locally trusted keys written by the archived notebook.
        keys = np.load(paths["user_keys"], allow_pickle=True).reshape(-1).astype(str)
        require(tweets.shape == (len(text), config["encoder"]["dimension"])
                and np.isfinite(tweets).all(), f"{split}: invalid tweet embeddings")
        groups = text.groupby("user_key", sort=True).indices
        expected_keys = list(groups)
        require(keys.tolist() == expected_keys, f"{split}: cached user key order differs")
        pooled = np.vstack([tweets[indices].mean(axis=0) for indices in groups.values()])
        require(users.shape == pooled.shape and np.isfinite(users).all(), f"{split}: invalid user embeddings")
        np.testing.assert_allclose(users, pooled, rtol=1e-6, atol=1e-7,
                                   err_msg=f"{split}: cached user pooling differs")
        arrays[split] = (keys, users)
        digests[split] = text_digest(text)
    archive = root / migration["archive_notebook"]
    source_hashes[migration["archive_notebook"]] = sha256(archive)
    target.parent.mkdir(parents=True, exist_ok=False)
    for split, (keys, users) in arrays.items():
        np.save(target.parent / f"{split}_embeddings.npy", users, allow_pickle=False)
        pd.DataFrame({"user_key": keys}).to_csv(target.parent / f"{split}_keys.csv", index=False)
    write_json(target, {
        "schema_version": "baseline_embeddings_v1", "source": "validated_legacy_cache_import",
        "phase2_manifest_sha256": config["phase2_manifest_sha256"], "encoder": config["encoder"],
        "encoder_revision": None, "ordered_text_sha256": digests, "source_sha256": source_hashes,
        "limitation": "Historical encoder revision and original row-to-vector provenance were not recorded. "
                      "Checks establish current text equality, key alignment and tweet-to-user pooling only. "
                      "No fresh language-model inference was performed.",
        "artifacts": artifact_inventory(target.parent)})
    load_embeddings(root, config, data)
    return target


def load_embeddings(root, config, data):
    """Read only the registered bundle; no fallback to legacy files or fabricated vectors."""
    path = Path(root) / config["embedding_manifest"]
    require(path.exists(), "Embedding bundle absent. Run scripts/register_baseline_embeddings.py first.")
    require(sha256(path) == config["embedding_manifest_sha256"], "Embedding manifest hash mismatch")
    manifest = read_json(path)
    require(manifest["phase2_manifest_sha256"] == config["phase2_manifest_sha256"], "Embedding dataset mismatch")
    require(manifest["encoder"] == config["encoder"], "Embedding encoder/pooling settings mismatch")
    result = {}
    for split in SPLITS:
        pop = data["populations"][split]
        require(text_digest(pop["text"]) == manifest["ordered_text_sha256"][split], "Embedding text mismatch")
        keys = keyed(pd.read_csv(checked_artifact(path.parent, manifest, f"{split}_keys.csv"),
                                 dtype={"user_key": str}), "embedding keys").index
        vectors = np.load(checked_artifact(path.parent, manifest, f"{split}_embeddings.npy"), allow_pickle=False)
        expected = set(pop["masks"].index[pop["masks"].has_tweet_text.eq(1)])
        require(set(keys) == expected, f"{split}: embedding users differ from tweet availability")
        require(vectors.shape == (len(keys), config["encoder"]["dimension"])
                and np.isfinite(vectors).all(), f"{split}: invalid embedding matrix")
        result[split] = {"keys": keys, "x": vectors,
                         "y": pop["users"].loc[keys, "label_binary"].to_numpy(dtype=int)}
    return result, manifest


class BaselineMLP(nn.Module):
    """Exact original topology for both tabular (64/32) and text (256/64)."""
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        a, b = hidden_dims
        self.network = nn.Sequential(nn.Linear(input_dim, a), nn.BatchNorm1d(a), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(a, b), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(b, 1))

    def forward(self, x):
        return self.network(x)


def seed_all(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def train_mlp(train, validation, spec, seed, threads, device):
    """No test input; fixed 50-epoch protocol. Independent snapshots avoid state aliasing."""
    seed_all(seed, threads)
    model = BaselineMLP(train["x"].shape[1], spec["hidden_dims"], spec["dropout"]).to(device)
    positives = int(train["y"].sum())
    require(0 < positives < len(train["y"]), "Training needs both classes")
    weight = (len(train["y"]) - positives) / positives
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"])
    loaders = {}
    for name, sample in (("train", train), ("validation", validation)):
        tensors = TensorDataset(torch.tensor(sample["x"], dtype=torch.float32),
                                torch.tensor(sample["y"].reshape(-1, 1), dtype=torch.float32))
        require(len(tensors) > 0, f"Empty {name} cohort")
        require(name != "train" or len(tensors) % spec["batch_size"] != 1,
                "BatchNorm would see a singleton; do not silently discard a user")
        loaders[name] = DataLoader(tensors, batch_size=spec["batch_size"], shuffle=name == "train")
    history, best_loss, best_epoch, best_state = [], float("inf"), None, None
    for epoch in range(1, spec["epochs"] + 1):
        losses = {}
        for name, loader in loaders.items():
            model.train(name == "train")
            total = 0.0
            with torch.set_grad_enabled(name == "train"):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    if name == "train":
                        optimizer.zero_grad()
                    loss = criterion(model(x), y)
                    if name == "train":
                        loss.backward()
                        optimizer.step()
                    total += loss.item()
            losses[name + "_loss"] = total / len(loader)  # Preserve original batch-mean convention.
        require(np.isfinite(list(losses.values())).all(), "Nonfinite training loss")
        history.append({"epoch": epoch, **losses})
        if losses["validation_loss"] < best_loss:
            best_loss, best_epoch = losses["validation_loss"], epoch
            best_state = deepcopy(model.state_dict())
        if epoch % 10 == 0:
            print(f"epoch {epoch}/{spec['epochs']}: {losses}", flush=True)
    last_state = deepcopy(model.state_dict())
    states = {"last": last_state, "best_validation_loss": best_state}
    model.load_state_dict(states[spec["selection"]])
    return model, states, pd.DataFrame(history), {"best_epoch": best_epoch, "last_epoch": spec["epochs"],
                                                "pos_weight": weight, "selected": spec["selection"]}


def probabilities(model, x, batch_size=32, device="cpu"):
    if isinstance(model, LogisticRegression):
        return model.predict_proba(x)[:, 1]
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            values.append(torch.sigmoid(model(batch)).cpu().numpy().ravel())
    return np.concatenate(values) if values else np.empty(0)


def prediction_registry(population, keys, probabilities_, predictions, model_name, split):
    """Retain every user, including unsupported users, without inserting text vectors."""
    users, masks = population["users"], population["masks"]
    require(len(keys) == len(set(keys)) and set(keys) <= set(users.index), "Invalid prediction keys")
    p = np.asarray(probabilities_)
    pred = np.asarray(predictions)
    require(len(keys) == len(p) == len(pred), "Prediction length mismatch")
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() and np.isin(pred, [0, 1]).all(),
            "Invalid probabilities/predictions")
    registry = users[["label_binary"]].join(masks[list(MASKS) + ["temporal_observed_count"]])
    registry["bot_probability"] = pd.Series(p, index=keys, dtype=float)
    registry["prediction"] = pd.Series(pred, index=keys).astype("Int64")
    registry["status"] = np.where(registry.bot_probability.notna(), "predicted", "missing_tweet_text")
    missing = registry.bot_probability.isna()
    require(not missing.any() or (model_name.startswith("text_")
                                 and registry.loc[missing, "has_tweet_text"].eq(0).all()),
            "Unexplained missing predictions")
    registry["model"], registry["split"] = model_name, split
    return registry.reset_index()


def evaluate_registry(registry, cohort):
    frame = registry if cohort == "all_users" else registry[registry.has_tweet_text.eq(1)]
    supported = frame[frame.status.eq("predicted")]
    result = {"cohort": cohort, "population_n": len(frame), "evaluated_n": len(supported),
              "missing_predictions_n": len(frame) - len(supported),
              "coverage": len(supported) / len(frame) if len(frame) else None}
    if supported.empty:
        return result, {}
    y = supported.label_binary.to_numpy(dtype=int)
    pred = supported.prediction.to_numpy(dtype=int)
    p = supported.bot_probability.to_numpy()
    result.update({"human_n": int((y == 0).sum()), "bot_n": int(y.sum()),
                   "accuracy": accuracy_score(y, pred), "bot_precision": precision_score(y, pred, zero_division=0),
                   "bot_recall": recall_score(y, pred, zero_division=0), "bot_f1": f1_score(y, pred, zero_division=0),
                   "roc_auc": roc_auc_score(y, p) if len(set(y)) == 2 else None,
                   "average_precision": average_precision_score(y, p) if len(set(y)) == 2 else None})
    detail = {"confusion_matrix_labels": [0, 1], "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
              "classification_report": classification_report(y, pred, labels=[0, 1],
                    target_names=["Human", "Bot"], zero_division=0, output_dict=True)}
    return result, detail


def load_checkpoint(path, device="cpu"):
    """Reload trusted local estimators; torch payload contains tensors and primitives only."""
    path = Path(path)
    if path.suffix == ".joblib":
        return joblib.load(path)
    payload = torch.load(path, map_location=device, weights_only=True)
    model = BaselineMLP(**payload["architecture"]).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


def decisions(model, x, probability, name, threshold):
    # Preserve original tabular sklearn.predict tie convention; other models used >= 0.5.
    return model.predict(x) if name == "tabular_lr" else (probability >= threshold).astype(int)


def notebook_source_hash(path):
    notebook = read_json(path)
    source = [(c["cell_type"], "".join(c["source"])) for c in notebook["cells"]]
    return hashlib.sha256(json.dumps(source, ensure_ascii=False).encode("utf-8")).hexdigest()


class BaselineRun:
    """One immutable output directory per execution; training cells can be inspected separately."""
    def __init__(self, root, config_path="configs/baselines_v2.json"):
        self.root = Path(root).resolve()
        self.config = read_json(self.root / config_path)
        self.data = load_phase2(self.root, self.config)
        self.embeddings, self.embedding_manifest = load_embeddings(self.root, self.config, self.data)
        require(self.config["device"] == "cpu", "This reproducible first pass is pinned to CPU")
        seed_all(self.config["seed"], self.config["torch_threads"])
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.output_dir = self.root / self.config["output_root"] / (stamp + "_" + uuid4().hex[:8])
        self.output_dir.mkdir(parents=True, exist_ok=False)
        for folder in ("models", "histories", "predictions", "reports", "provenance"):
            (self.output_dir / folder).mkdir()
        self.models, self.metric_rows, self.details = {}, [], {}
        self.reload_checks = []
        self.completed = False
        self.coverage = pd.DataFrame([
            {"split": s, "users": len(p["users"]), "human": int((p["y"] == 0).sum()),
             "bot": int(p["y"].sum()), **{m: int(p["masks"][m].sum()) for m in MASKS},
             "missing_temporal_values": int((p["observed"][["observed__" + f for f in self.data["temporal_features"]]] == 0).sum().sum())}
            for s, p in self.data["populations"].items()])
        self.coverage.to_csv(self.output_dir / "reports/coverage.csv", index=False)
        write_json(self.output_dir / "provenance/resolved_config.json", self.config)
        write_json(self.output_dir / "provenance/phase2_manifest.json", self.data["manifest"])
        write_json(self.output_dir / "provenance/embedding_manifest.json", self.embedding_manifest)
        code_path = self.root / "src/baselines.py"
        (self.output_dir / "provenance/baselines_source.py").write_bytes(code_path.read_bytes())
        packages = {name: importlib.metadata.version(name)
                    for name in ("numpy", "pandas", "scikit-learn", "torch", "joblib", "nbformat", "nbclient")}
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.root, text=True).strip()
            git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=self.root, text=True)
        except (OSError, subprocess.CalledProcessError):
            git_commit, git_status = None, "unavailable"
        self.provenance = {"created_utc": stamp, "python": platform.python_version(),
            "platform": platform.platform(), "packages": packages, "device": self.config["device"],
            "torch_threads": torch.get_num_threads(), "deterministic_algorithms": True,
            "seed_per_experiment": self.config["seed"], "git_commit_at_execution": git_commit,
            "git_status_at_execution": git_status, "source_sha256": sha256(code_path),
            "notebook_source_sha256": notebook_source_hash(self.root / "analysis/03_baseline_models.ipynb"),
            "config_sha256": sha256(self.root / config_path),
            "phase2_manifest_sha256": sha256(self.data["manifest_path"]),
            "embedding_manifest_sha256": sha256(self.root / self.config["embedding_manifest"]),
            "feature_order": self.data["feature_order"], "identity_key": "user_key",
            "training_scope": "Frozen train users only; validation used only for text best-checkpoint selection. No test selection.",
            "metric_policy": "Bot=1. AP is average_precision_score, not trapezoidal PR-AUC. No threshold tuning.",
            "missingness_policy": "All users retained in prediction registries. Tabular consumes Phase-2 values and explicit observation masks are saved, not appended to the 40 inputs. Missing temporal zero placeholders remain documented, never imputed again. Text models abstain when has_tweet_text=0, even if description exists. No graph required.",
            "comparability_limits": ["Phase-2 corrected missing description_length handling; inputs differ from historical tabular experiments.",
                "Historical MLP runs were unseeded. Seeded reruns do not reproduce their random trajectory.",
                "deepcopy restores intended text best-validation checkpoint; historical reference aliasing actually kept final weights.",
                "Historical test data has already been evaluated; this is not a newly untouched test set.",
                self.embedding_manifest["limitation"]]}
        write_json(self.output_dir / "provenance/run.json", self.provenance)
        for split, population in self.data["populations"].items():
            population["observed"].to_csv(self.output_dir / f"provenance/{split}_observation_masks.csv")

    def sample(self, name, split):
        if name.startswith("text_"):
            return self.embeddings[split]
        population = self.data["populations"][split]
        return {"keys": population["users"].index, "x": population["x"], "y": population["y"]}

    def fit(self, name):
        require(not self.completed, "Run is finalized; create another run")
        require(name in ("tabular_lr", "tabular_mlp", "text_lr", "text_mlp"), "Unknown baseline")
        require(name not in self.models, "Experiment already saved; start a new run rather than overwrite")
        train, validation = self.sample(name, "train"), self.sample(name, "validation")
        train_keys = set(self.data["populations"]["train"]["users"].index)
        require(set(train["keys"]) <= train_keys, "Training includes nontraining users")
        seed_all(self.config["seed"], self.config["torch_threads"])
        checkpoint_models = {}
        if name.endswith("_lr"):
            model = LogisticRegression(**self.config["logistic_regression"])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(train["x"], train["y"])
            messages = [str(w.message) for w in caught]
            for message in messages:
                print(f"{name}: {message}", flush=True)
            relative = f"models/{name}.joblib"
            joblib.dump(model, self.output_dir / relative)
            checkpoint_models["fitted"] = (relative, model)
            info = {"selected": "fitted", "effective_parameters": model.get_params(),
                    "n_iter": model.n_iter_.tolist(), "warnings": messages,
                    "history_note": "sklearn LR exposes iteration count, not an epoch loss history."}
            write_json(self.output_dir / f"histories/{name}.json", info)
        else:
            spec = self.config[name]
            model, states, history, info = train_mlp(train, validation, spec,
                self.config["seed"], self.config["torch_threads"], self.config["device"])
            history.to_csv(self.output_dir / f"histories/{name}.csv", index=False)
            architecture = {"input_dim": train["x"].shape[1], "hidden_dims": spec["hidden_dims"], "dropout": spec["dropout"]}
            for variant, state in states.items():
                relative = f"models/{name}_{variant}.pt"
                torch.save({"state_dict": state, "architecture": architecture, "model_name": name,
                            "variant": variant, "epoch": info["last_epoch"] if variant == "last" else info["best_epoch"],
                            "feature_order": self.data["feature_order"] if name.startswith("tabular") else None,
                            "phase2_manifest_sha256": self.config["phase2_manifest_sha256"]}, self.output_dir / relative)
                snapshot = BaselineMLP(**architecture)
                snapshot.load_state_dict(state)
                checkpoint_models[variant] = (relative, snapshot.eval())
            info["specification"] = spec
        info["training_user_keys"] = list(train["keys"])
        info["training_class_counts"] = {"human": int((train["y"] == 0).sum()), "bot": int(train["y"].sum())}
        info["checkpoints"] = {v: p for v, (p, _) in checkpoint_models.items()}
        self.models[name] = info
        for variant, (relative, model) in checkpoint_models.items():
            restored = load_checkpoint(self.output_dir / relative)
            for split in SPLITS:
                sample = self.sample(name, split)
                p = probabilities(model, sample["x"])
                reloaded_p = probabilities(restored, sample["x"])
                np.testing.assert_allclose(reloaded_p, p, rtol=0, atol=1e-7)
                predicted = decisions(model, sample["x"], p, name, self.config["threshold"])
                np.testing.assert_array_equal(decisions(restored, sample["x"], reloaded_p, name, self.config["threshold"]), predicted)
                registry = prediction_registry(self.data["populations"][split], sample["keys"], p, predicted, name, split)
                registry["checkpoint_variant"] = variant
                registry["primary_checkpoint"] = variant == info["selected"]
                registry.to_csv(self.output_dir / f"predictions/{name}_{variant}_{split}.csv", index=False)
                for cohort in ("all_users", "tweet_available"):
                    metric, detail = evaluate_registry(registry, cohort)
                    self.metric_rows.append({"model": name, "checkpoint_variant": variant,
                        "primary_checkpoint": variant == info["selected"], "split": split, **metric})
                    self.details[f"{name}/{variant}/{split}/{cohort}"] = detail
                self.reload_checks.append({"model": name, "variant": variant, "split": split,
                                           "users_checked": len(p), "max_abs_probability_difference": float(np.max(np.abs(p - reloaded_p))) if len(p) else 0.0})
        write_json(self.output_dir / f"models/{name}_metadata.json", info)
        print(f"Saved {name}; primary checkpoint={info['selected']}", flush=True)
        return info

    def finalize(self):
        require(not self.completed, "Run already finalized")
        require(set(self.models) == {"tabular_lr", "tabular_mlp", "text_lr", "text_mlp"}, "Incomplete baseline run")
        metrics = pd.DataFrame(self.metric_rows)
        metrics.to_csv(self.output_dir / "metrics.csv", index=False)
        write_json(self.output_dir / "reports/classification_reports.json", self.details)
        write_json(self.output_dir / "reports/checkpoint_reload.json", self.reload_checks)
        write_json(self.output_dir / "manifest.json", {"schema_version": "baselines_v2", "complete": True,
            "phase2_manifest_sha256": self.config["phase2_manifest_sha256"],
            "embedding_manifest_sha256": self.provenance["embedding_manifest_sha256"],
            "feature_order": self.data["feature_order"], "models": self.models,
            "artifacts": artifact_inventory(self.output_dir)})
        self.completed = True
        return metrics


def verify_run(root, output_dir):
    """Independent disk reload: verify hashes, cohorts, all checkpoints and computed metrics."""
    directory = Path(output_dir)
    manifest = read_json(directory / "manifest.json")
    require(manifest["complete"], "Run not complete")
    for relative in manifest["artifacts"]:
        checked_artifact(directory, manifest, relative)
    config = read_json(directory / "provenance/resolved_config.json")
    torch.set_num_threads(config["torch_threads"])
    data = load_phase2(root, config)
    embeddings, _ = load_embeddings(root, config, data)
    require(sha256(Path(root) / config["embedding_manifest"]) == manifest["embedding_manifest_sha256"],
            "Registered embedding manifest changed")
    metrics = pd.read_csv(directory / "metrics.csv")
    details = read_json(directory / "reports/classification_reports.json")
    checked = 0
    for name, metadata in manifest["models"].items():
        expected_train = embeddings["train"]["keys"] if name.startswith("text_") else data["populations"]["train"]["users"].index
        require(metadata["training_user_keys"] == list(expected_train), f"{name}: training key provenance mismatch")
        if name.endswith("_mlp"):
            history = pd.read_csv(directory / f"histories/{name}.csv")
            best_epoch = int(history.loc[history.validation_loss.idxmin(), "epoch"])
            require(best_epoch == metadata["best_epoch"], "Best checkpoint epoch differs from history")
            require(len(history) == config[name]["epochs"], "Incomplete training history")
        for variant, relative in metadata["checkpoints"].items():
            model = load_checkpoint(directory / relative)
            for split in SPLITS:
                pop = data["populations"][split]
                sample = embeddings[split] if name.startswith("text_") else {"keys": pop["users"].index, "x": pop["x"]}
                probability = probabilities(model, sample["x"])
                predicted = decisions(model, sample["x"], probability, name, config["threshold"])
                expected = prediction_registry(pop, sample["keys"], probability, predicted, name, split)
                saved = pd.read_csv(directory / f"predictions/{name}_{variant}_{split}.csv", dtype={"user_key": str, "prediction": "Int64"})
                pd.testing.assert_frame_equal(saved[expected.columns], expected, check_dtype=False, rtol=0, atol=1e-7)
                require(saved.primary_checkpoint.eq(variant == metadata["selected"]).all(), "Prediction selection mismatch")
                for cohort in ("all_users", "tweet_available"):
                    actual, detail = evaluate_registry(saved, cohort)
                    row = metrics[(metrics.model == name) & (metrics.checkpoint_variant == variant)
                                  & (metrics.split == split) & (metrics.cohort == cohort)]
                    require(len(row) == 1, "Missing/duplicate metrics row")
                    for key, value in actual.items():
                        saved_value = row.iloc[0][key]
                        if isinstance(value, str):
                            require(saved_value == value, "Metric cohort mismatch")
                        elif value is None:
                            require(pd.isna(saved_value), "Undefined metric mismatch")
                        else:
                            require(np.isclose(saved_value, value, atol=1e-10, rtol=1e-10), f"Metric differs: {key}")
                    require(detail == details[f"{name}/{variant}/{split}/{cohort}"], "Classification report differs")
                checked += 1
    require(len(metrics) == checked * 2, "Unexpected metric rows")
    return {"verified_artifacts": len(manifest["artifacts"]), "reloaded_checkpoints": sum(len(m["checkpoints"]) for m in manifest["models"].values()),
            "verified_prediction_files": checked, "verified_metric_rows": len(metrics)}
