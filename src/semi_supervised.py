"""Leakage-safe, tabular pseudo-label experiment for Notebook 08.

The unlabeled data has no retained tweet text, so this deliberately is not a
multimodal-fusion experiment.  It reuses the exact Notebook-03 tabular MLP.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import importlib.metadata
import platform
import subprocess

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.baselines import (BaselineMLP, SPLITS, align, artifact_inventory, checked_artifact,
    evaluate_registry, keyed, load_phase2, notebook_source_hash, probabilities, read_json,
    require, seed_all, sha256, write_json)


def _unlabeled(root, config, phase2):
    """Load only registered Phase-2 unlabeled data; no refit or text substitution."""
    directory = Path(root) / Path(config["phase2_manifest"]).parent
    manifest = phase2["manifest"]
    def csv(relative):
        return pd.read_csv(checked_artifact(directory, manifest, relative), dtype={"user_key": str})
    users = keyed(csv("splits/users/unlabeled_users.csv"), "unlabeled users")
    tabular = keyed(csv("splits/tabular/unlabeled_tabular.csv"), "unlabeled tabular")
    masks = keyed(csv("splits/masks/unlabeled_modality_masks.csv"), "unlabeled masks")
    observed = keyed(csv("splits/masks/unlabeled_observation_masks.csv"), "unlabeled observed")
    tabular, masks, observed = (align(frame, users, name) for frame, name in
                                ((tabular, "unlabeled tabular"), (masks, "unlabeled masks"),
                                 (observed, "unlabeled observed")))
    order = phase2["feature_order"]
    require(len(users) == config["expected"]["unlabeled"], "Unexpected unlabeled population")
    # Schema retains the label column for table compatibility, but it must contain no label.
    require(("label_binary" not in users or users.label_binary.isna().all())
            and ("label_binary" not in tabular or tabular.label_binary.isna().all()),
            "Unlabeled labels present")
    require(masks.has_tweet_text.eq(0).all(), "Unlabeled tweet-text contract changed")
    require(np.isfinite(tabular.loc[:, order].to_numpy(dtype=np.float32)).all(), "Unlabeled tabular has nonfinite values")
    return {"users": users, "masks": masks, "observed": observed,
            "x": tabular.loc[:, order].to_numpy(dtype=np.float32)}


def load_inputs(root, config):
    require(config["teacher_model"] == "tabular_mlp" and config["teacher_checkpoint_variant"] == "last",
            "This approved experiment requires the selected tabular MLP teacher")
    require(config["student_initialization"] == "pinned_teacher_state", "Unexpected student initialization")
    require(config["pseudo_label"] == {"confidence_threshold": .95, "pseudo_loss_coefficient": .5,
            "class_balanced_cap": 195, "human_selection": "confidence_desc_then_user_key_asc_top_k",
            "bot_selection": "retain_all_accepted", "sampling": "none_deterministic_ranking",
            "selection_seed": 42}, "Pseudo-label policy change requires approval")
    phase2 = load_phase2(root, config)
    unlabeled = _unlabeled(root, config, phase2)
    gold = set().union(*(set(phase2["populations"][s]["users"].index) for s in SPLITS))
    require(not gold & set(unlabeled["users"].index), "Labeled/unlabeled identity overlap")
    path = Path(root) / config["baseline_manifest"]
    require(sha256(path) == config["baseline_manifest_sha256"], "Baseline manifest hash mismatch")
    baseline = read_json(path)
    require(baseline["complete"] and baseline["phase2_manifest_sha256"] == config["phase2_manifest_sha256"],
            "Teacher source uses different inputs")
    require(baseline["feature_order"] == phase2["feature_order"], "Teacher feature order differs")
    model_meta = baseline["models"]["tabular_mlp"]
    require(model_meta["selected"] == "last", "Pinned supervised teacher selection changed")
    teacher_path = checked_artifact(path.parent, baseline, model_meta["checkpoints"]["last"])
    payload = torch.load(teacher_path, map_location="cpu", weights_only=True)
    require(payload["architecture"] == {"input_dim": 40, "hidden_dims": [64, 32], "dropout": .3},
            "Teacher architecture differs")
    teacher = BaselineMLP(**payload["architecture"]).eval()
    teacher.load_state_dict(payload["state_dict"], strict=True)
    train = phase2["populations"]["train"]
    require(model_meta["training_user_keys"] == train["users"].index.tolist(), "Teacher training keys differ")
    require(len(train["users"]) == config["expected"]["labeled_train"], "Train population differs")
    # Reproduce registered supervised predictions before using the teacher for pseudo labels.
    for split in SPLITS:
        registered = pd.read_csv(checked_artifact(path.parent, baseline,
            f"predictions/tabular_mlp_last_{split}.csv"), dtype={"user_key": str, "prediction": "Int64"})
        p = probabilities(teacher, phase2["populations"][split]["x"], config["training"]["batch_size"])
        expected = registered.set_index("user_key").loc[phase2["populations"][split]["users"].index]
        # CSV round-trip plus float32 inference can differ from the historical
        # in-memory values by a few 1e-7 units; this is not a model change.
        np.testing.assert_allclose(expected.bot_probability.to_numpy(), p, rtol=1e-6, atol=1e-6)
    return phase2, unlabeled, baseline, teacher, payload, teacher_path


def pseudo_pool(teacher, unlabeled, config):
    p = probabilities(teacher, unlabeled["x"], config["training"]["batch_size"])
    frame = unlabeled["users"].index.to_frame(index=False)
    frame["bot_probability"] = p
    frame["human_probability"] = 1 - p
    frame["pseudo_label"] = (p >= config["threshold"]).astype(int)
    frame["confidence"] = np.maximum(p, 1 - p)
    threshold = config["pseudo_label"]["confidence_threshold"]
    frame["accepted_by_threshold"] = frame.confidence.ge(threshold)
    frame["selection_status"] = "rejected_below_confidence"
    frame.loc[frame.accepted_by_threshold, "selection_status"] = "accepted_not_selected_class_cap"
    cap = config["pseudo_label"]["class_balanced_cap"]
    bots = frame[(frame.accepted_by_threshold) & frame.pseudo_label.eq(1)].sort_values(
        ["confidence", "user_key"], ascending=[False, True])
    humans = frame[(frame.accepted_by_threshold) & frame.pseudo_label.eq(0)].sort_values(
        ["confidence", "user_key"], ascending=[False, True])
    require(len(bots) == config["expected"]["accepted_pseudo_bot"], "Accepted pseudo-bot count changed")
    require(len(bots) <= cap and len(humans) >= cap, "Class-balanced selection unavailable")
    chosen = pd.concat([bots, humans.head(cap)]).sort_values("user_key")
    frame.loc[frame.user_key.isin(chosen.user_key), "selection_status"] = "selected_for_student"
    selected = frame[frame.selection_status.eq("selected_for_student")].copy().reset_index(drop=True)
    require(selected.pseudo_label.value_counts().to_dict() == {0: cap, 1: len(bots)}, "Selected class balance differs")
    return frame, selected


def _batches(n, batch_size, seed, epoch):
    """Deterministic shuffled partitions, with no singleton BatchNorm batch."""
    indices = np.random.default_rng(seed + epoch).permutation(n)
    chunks = [chunk for chunk in np.array_split(indices, range(batch_size, n, batch_size)) if len(chunk)]
    if len(chunks) > 1 and len(chunks[-1]) == 1:
        last = chunks.pop()
        prior = chunks.pop()
        split = len(prior) // 2
        chunks.extend([np.concatenate([prior[:split], last]), prior[split:]])
    require(all(len(chunk) > 1 for chunk in chunks), "BatchNorm singleton batch")
    return chunks


def _loss(model, x, y, pos_weight, sample_weight=None):
    values = nn.functional.binary_cross_entropy_with_logits(model(x), y.reshape(-1, 1),
        pos_weight=pos_weight, reduction="none")
    return values.mean() if sample_weight is None else (values.reshape(-1) * sample_weight).mean()


def train_continuation(initial_state, labeled, pseudo, validation, config, mode):
    """Same initialized model/label passes; only student receives paired pseudo batches."""
    require(mode in ("labeled_control", "pseudo_student"), "Unknown continuation mode")
    seed_all(config["seed"], config["torch_threads"])
    model = BaselineMLP(40, [64, 32], .3)
    model.load_state_dict(deepcopy(initial_state), strict=True)
    n_pos = int(labeled["y"].sum())
    pos_weight = torch.tensor([(len(labeled["y"]) - n_pos) / n_pos], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"],
                                  weight_decay=config["training"]["weight_decay"])
    x_l, y_l = (torch.tensor(labeled[k], dtype=torch.float32) for k in ("x", "y"))
    x_p = torch.tensor(pseudo["x"], dtype=torch.float32)
    y_p = torch.tensor(pseudo["y"], dtype=torch.float32)
    w_p = torch.tensor(pseudo["confidence"], dtype=torch.float32)
    x_v, y_v = (torch.tensor(validation[k], dtype=torch.float32) for k in ("x", "y"))
    history, best, best_loss, best_epoch = [], None, float("inf"), None
    pseudo_draws = 0
    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train(); labeled_losses=[]; pseudo_losses=[]
        lb = _batches(len(x_l), config["training"]["batch_size"], config["seed"], epoch)
        pb = _batches(len(x_p), config["training"]["batch_size"], config["seed"] + 100000, epoch)
        for step, idx in enumerate(lb):
            optimizer.zero_grad()
            labeled_loss = _loss(model, x_l[idx], y_l[idx], pos_weight)
            pseudo_loss = torch.tensor(0.)
            if mode == "pseudo_student":
                pidx = pb[step % len(pb)]
                pseudo_loss = _loss(model, x_p[pidx], y_p[pidx], pos_weight, w_p[pidx])
                pseudo_draws += len(pidx)
            loss = labeled_loss + config["pseudo_label"]["pseudo_loss_coefficient"] * pseudo_loss
            require(torch.isfinite(loss).item(), "Nonfinite continuation loss")
            loss.backward(); optimizer.step()
            labeled_losses.append(labeled_loss.item()); pseudo_losses.append(pseudo_loss.item())
        model.eval()
        with torch.no_grad():
            val_loss = _loss(model, x_v, y_v, pos_weight).item()
        row = {"epoch": epoch, "labeled_loss": float(np.mean(labeled_losses)),
               "pseudo_loss": float(np.mean(pseudo_losses)), "validation_loss": val_loss}
        history.append(row)
        if val_loss < best_loss:
            best, best_loss, best_epoch = deepcopy(model.state_dict()), val_loss, epoch
        if epoch % 10 == 0: print(f"{mode} epoch {epoch}: {row}", flush=True)
    return {"best_validation_loss": best, "last": deepcopy(model.state_dict())}, pd.DataFrame(history), {
        "best_epoch": best_epoch, "last_epoch": config["training"]["epochs"], "selection": config["training"]["selection"],
        "pos_weight": float(pos_weight.item()), "labeled_training_user_keys": labeled["keys"].tolist(),
        "pseudo_training_user_keys": pseudo["keys"].tolist() if mode == "pseudo_student" else [],
        "pseudo_draws": pseudo_draws, "initialization": "pinned_teacher_state"}


def registry(model, population, split, config, name):
    p = probabilities(model, population["x"], config["training"]["batch_size"])
    frame = population["users"][["label_binary"]].join(population["masks"])
    frame["bot_probability"] = p
    frame["prediction"] = pd.Series((p >= config["threshold"]).astype(int), index=frame.index, dtype="Int64")
    frame["status"] = "predicted"
    frame["model"], frame["split"] = name, split
    return frame.reset_index()


class SemiSupervisedRun:
    def __init__(self, root, config_path="configs/semi_supervised_v2.json"):
        self.root, self.config = Path(root).resolve(), read_json(Path(root) / config_path)
        seed_all(self.config["seed"], self.config["torch_threads"])
        self.phase2, self.unlabeled, self.baseline, self.teacher, self.teacher_payload, self.teacher_path = load_inputs(self.root, self.config)
        self.pool, self.selected = pseudo_pool(self.teacher, self.unlabeled, self.config)
        self.labeled = {"keys": self.phase2["populations"]["train"]["users"].index,
                        "x": self.phase2["populations"]["train"]["x"].astype(np.float32),
                        "y": self.phase2["populations"]["train"]["y"].astype(np.float32)}
        positions = self.unlabeled["users"].index.get_indexer(self.selected.user_key)
        self.pseudo = {"keys": self.selected.user_key.to_numpy(), "x": self.unlabeled["x"][positions],
                       "y": self.selected.pseudo_label.to_numpy(dtype=np.float32),
                       "confidence": self.selected.confidence.to_numpy(dtype=np.float32)}
        self.validation = {"x": self.phase2["populations"]["validation"]["x"].astype(np.float32),
                           "y": self.phase2["populations"]["validation"]["y"].astype(np.float32)}
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.output_dir = self.root / self.config["output_root"] / f"{stamp}_{uuid4().hex[:8]}"
        self.output_dir.mkdir(parents=True, exist_ok=False)
        for folder in ("models", "histories", "predictions", "reports", "provenance", "pseudo_labels"):
            (self.output_dir / folder).mkdir()
        write_json(self.output_dir / "provenance/resolved_config.json", self.config)
        write_json(self.output_dir / "provenance/phase2_manifest.json", self.phase2["manifest"])
        write_json(self.output_dir / "provenance/baseline_manifest.json", self.baseline)
        self.pool.to_csv(self.output_dir / "pseudo_labels/full_teacher_pseudo_label_pool.csv", index=False)
        self.selected.to_csv(self.output_dir / "pseudo_labels/selected_for_student.csv", index=False)
        self._write_setup()

    def _write_setup(self):
        accepted = self.pool[self.pool.accepted_by_threshold].pseudo_label.value_counts().rename_axis("pseudo_label").reset_index(name="users")
        selected = self.selected.pseudo_label.value_counts().rename_axis("pseudo_label").reset_index(name="users")
        accepted["stage"], selected["stage"] = "accepted_by_threshold", "selected_for_training"
        pd.concat([accepted, selected]).to_csv(self.output_dir / "reports/pseudo_label_distribution.csv", index=False)
        pd.DataFrame([{"population": "supervised_teacher", "human_labeled": 671, "pseudo_human": 0, "pseudo_bot": 0,
                       "validation_labels_in_training": False, "test_labels_in_training": False},
                      {"population": "labeled_control", "human_labeled": 671, "pseudo_human": 0, "pseudo_bot": 0,
                       "validation_labels_in_training": False, "test_labels_in_training": False},
                      {"population": "pseudo_student", "human_labeled": 671, "pseudo_human": 195, "pseudo_bot": 195,
                       "validation_labels_in_training": False, "test_labels_in_training": False}]).to_csv(
                           self.output_dir / "reports/training_populations.csv", index=False)
        for split, pop in self.phase2["populations"].items():
            pop["observed"].to_csv(self.output_dir / f"provenance/{split}_observation_masks.csv")
        self.unlabeled["observed"].to_csv(self.output_dir / "provenance/unlabeled_observation_masks.csv")
        sources = ["src/semi_supervised.py", "src/baselines.py"]
        for source in sources:
            (self.output_dir / "provenance" / Path(source).name).write_bytes((self.root / source).read_bytes())
        write_json(self.output_dir / "provenance/run.json", {"seed": self.config["seed"], "device": "cpu",
            "python_version": platform.python_version(), "platform": platform.platform(),
            "package_versions": {name: importlib.metadata.version(name)
                                 for name in ("numpy", "pandas", "torch", "scikit-learn", "nbformat", "nbclient")},
            "torch_threads": self.config["torch_threads"], "deterministic_algorithms": True,
            "source_sha256": {s: sha256(self.root / s) for s in sources},
            "config_sha256": sha256(self.root / "configs/semi_supervised_v2.json"),
            "notebook_source_sha256": notebook_source_hash(self.root / "analysis/08_semi_supervised_learning.ipynb"),
            "teacher_checkpoint_sha256": sha256(self.teacher_path),
            "git_commit_at_execution": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.root, text=True).strip(),
            "git_status_at_execution": subprocess.check_output(["git", "status", "--porcelain"], cwd=self.root, text=True),
            "limitations": ["Tabular-only pseudo-label experiment: unlabeled users have no retained tweet text; not multimodal fusion.",
                "Confidence threshold is not a calibration guarantee; class-balanced cap was prespecified after unlabeled teacher diagnostics.",
                "Temporal values are Phase-2 transformed observations/placeholders; no new temporal imputation or exclusion.",
                "Single seed, annotation uncertainty and previously inspected validation/test data limit inference."]})

    def fit(self):
        initial = self.teacher_payload["state_dict"]
        self.states, self.histories, self.info = {}, {}, {}
        for name in ("labeled_control", "pseudo_student"):
            states, history, info = train_continuation(initial, self.labeled, self.pseudo, self.validation, self.config, name)
            self.states[name], self.histories[name], self.info[name] = states, history, info
            history.to_csv(self.output_dir / f"histories/{name}.csv", index=False)
            info["checkpoints"] = {}
            for variant, state in states.items():
                rel = f"models/{name}_{variant}.pt"
                torch.save({"state_dict": state, "architecture": self.teacher_payload["architecture"], "model": name,
                    "variant": variant, "epoch": info["best_epoch"] if variant == "best_validation_loss" else info["last_epoch"],
                    "feature_order": self.phase2["feature_order"], "pseudo_policy": self.config["pseudo_label"]}, self.output_dir / rel)
                info["checkpoints"][variant] = rel
            write_json(self.output_dir / f"models/{name}_metadata.json", info)

    def export(self):
        rows, details, primary = [], {}, {}
        models = {"supervised_teacher": self.teacher}
        for name, info in self.info.items():
            payload = torch.load(self.output_dir / info["checkpoints"][info["selection"]], map_location="cpu", weights_only=True)
            model = BaselineMLP(**payload["architecture"]).eval(); model.load_state_dict(payload["state_dict"])
            models[name] = model
        for name, model in models.items():
            primary[name] = {}
            for split, pop in self.phase2["populations"].items():
                frame = registry(model, pop, split, self.config, name)
                frame.to_csv(self.output_dir / f"predictions/{name}_{split}.csv", index=False)
                primary[name][split] = frame
                metric, detail = evaluate_registry(frame, "all_users")
                rows.append({"model": name, "split": split, **metric})
                details[f"{name}/{split}"] = detail
        metrics = pd.DataFrame(rows); metrics.to_csv(self.output_dir / "metrics.csv", index=False)
        write_json(self.output_dir / "reports/classification_reports.json", details)
        comparisons = []
        for split in ("validation", "test"):
            ref = keyed(primary["supervised_teacher"][split], "teacher")
            for name, frame in primary.items():
                candidate = align(keyed(frame[split], name), ref, "comparison")
                metric, _ = evaluate_registry(candidate.reset_index(), "all_users")
                comparisons.append({"model": name, "split": split, "comparison_type": "same_frozen_population_continuation_control",
                                    **metric})
        pd.DataFrame(comparisons).to_csv(self.output_dir / "reports/supervised_control_student_comparison.csv", index=False)
        write_json(self.output_dir / "manifest.json", {"schema_version": "semi_supervised_tabular_v2", "complete": True,
            "identity_key": "user_key", "feature_order": self.phase2["feature_order"], "teacher": {"checkpoint": str(self.teacher_path.relative_to(self.root)),
            "checkpoint_sha256": sha256(self.teacher_path)}, "student": self.info.get("pseudo_student"),
            "continuations": self.info,
            "pseudo_policy": self.config["pseudo_label"], "artifacts": artifact_inventory(self.output_dir)})
        return metrics


def verify_run(root, output_dir):
    directory, manifest = Path(output_dir), read_json(Path(output_dir) / "manifest.json")
    require(manifest["complete"], "Incomplete run")
    for relative in manifest["artifacts"]: checked_artifact(directory, manifest, relative)
    config = read_json(directory / "provenance/resolved_config.json")
    seed_all(config["seed"], config["torch_threads"])
    phase2, unlabeled, baseline, teacher, payload, teacher_path = load_inputs(root, config)
    pool, selected = pseudo_pool(teacher, unlabeled, config)
    saved = pd.read_csv(directory / "pseudo_labels/full_teacher_pseudo_label_pool.csv", dtype={"user_key": str})
    pd.testing.assert_frame_equal(saved, pool, check_dtype=False, atol=1e-7, rtol=0)
    pd.testing.assert_frame_equal(pd.read_csv(directory / "pseudo_labels/selected_for_student.csv", dtype={"user_key": str}), selected,
                                  check_dtype=False, atol=1e-7, rtol=0)
    require(set(selected.user_key).isdisjoint(set().union(*(set(phase2["populations"][s]["users"].index) for s in SPLITS))), "Pseudo/gold overlap")
    expected_distribution = {0: 195, 1: 195}
    require(selected.pseudo_label.value_counts().sort_index().to_dict() == expected_distribution,
            "Pseudo-label class balance differs")
    labeled = {"keys": phase2["populations"]["train"]["users"].index,
               "x": phase2["populations"]["train"]["x"].astype(np.float32),
               "y": phase2["populations"]["train"]["y"].astype(np.float32)}
    positions = unlabeled["users"].index.get_indexer(selected.user_key)
    pseudo = {"keys": selected.user_key.to_numpy(), "x": unlabeled["x"][positions],
              "y": selected.pseudo_label.to_numpy(dtype=np.float32), "confidence": selected.confidence.to_numpy(dtype=np.float32)}
    validation = {"x": phase2["populations"]["validation"]["x"].astype(np.float32),
                  "y": phase2["populations"]["validation"]["y"].astype(np.float32)}
    training_populations = pd.read_csv(directory / "reports/training_populations.csv")
    expected_training = {"supervised_teacher": (671, 0, 0), "labeled_control": (671, 0, 0),
                         "pseudo_student": (671, 195, 195)}
    for name, counts in expected_training.items():
        row = training_populations.set_index("population").loc[name]
        require(tuple(row[["human_labeled", "pseudo_human", "pseudo_bot"]]) == counts
                and not row.validation_labels_in_training and not row.test_labels_in_training,
                "Training population report differs")
    primary = {"supervised_teacher": {}}
    for split, pop in phase2["populations"].items():
        expected = registry(teacher, pop, split, config, "supervised_teacher")
        saved = pd.read_csv(directory / f"predictions/supervised_teacher_{split}.csv", dtype={"user_key": str, "prediction": "Int64"})
        pd.testing.assert_frame_equal(saved, expected, check_dtype=False, atol=1e-6, rtol=0)
        primary["supervised_teacher"][split] = saved
    pos_weight = torch.tensor([(len(labeled["y"]) - int(labeled["y"].sum())) / int(labeled["y"].sum())], dtype=torch.float32)
    rows = []
    for name in ("labeled_control", "pseudo_student"):
        info = manifest["continuations"][name]
        require(info == read_json(directory / f"models/{name}_metadata.json"), "Continuation metadata differs")
        require(info["labeled_training_user_keys"] == labeled["keys"].tolist(), "Labeled training keys differ")
        require(not set(info["labeled_training_user_keys"]) & set(phase2["populations"]["validation"]["users"].index)
                and not set(info["labeled_training_user_keys"]) & set(phase2["populations"]["test"]["users"].index),
                "Held-out user in continuation training")
        require((name == "pseudo_student" and info["pseudo_training_user_keys"] == pseudo["keys"].tolist())
                or (name == "labeled_control" and not info["pseudo_training_user_keys"]), "Pseudo training keys differ")
        history = pd.read_csv(directory / f"histories/{name}.csv")
        require(history.epoch.tolist() == list(range(1, 51)) and info["best_epoch"] == int(history.loc[history.validation_loss.idxmin(), "epoch"]),
                "Continuation history differs")
        primary[name] = {}
        for variant, relative in info["checkpoints"].items():
            snapshot = torch.load(directory / relative, map_location="cpu", weights_only=True)
            require(snapshot["model"] == name and snapshot["variant"] == variant and snapshot["feature_order"] == phase2["feature_order"],
                    "Checkpoint contract differs")
            model = BaselineMLP(**snapshot["architecture"]).eval(); model.load_state_dict(snapshot["state_dict"], strict=True)
            epoch = info["best_epoch"] if variant == "best_validation_loss" else info["last_epoch"]
            loss = _loss(model, torch.tensor(validation["x"]), torch.tensor(validation["y"]), pos_weight).item()
            require(np.isclose(loss, history.loc[history.epoch.eq(epoch), "validation_loss"].iloc[0], atol=1e-7, rtol=1e-7),
                    "Reloaded validation loss differs")
            if variant == info["selection"]:
                for split, pop in phase2["populations"].items():
                    expected = registry(model, pop, split, config, name)
                    saved = pd.read_csv(directory / f"predictions/{name}_{split}.csv", dtype={"user_key": str, "prediction": "Int64"})
                    pd.testing.assert_frame_equal(saved, expected, check_dtype=False, atol=1e-6, rtol=0)
                    primary[name][split] = saved
    for name, splits in primary.items():
        for split, frame in splits.items():
            metric, _ = evaluate_registry(frame, "all_users")
            rows.append({"model": name, "split": split, **metric})
    pd.testing.assert_frame_equal(pd.DataFrame(rows), pd.read_csv(directory / "metrics.csv"), check_dtype=False, atol=1e-10, rtol=1e-10)
    return {"verified_artifacts": len(manifest["artifacts"]), "full_pseudo_pool": len(pool),
            "selected_pseudo": len(selected), "selected_distribution": selected.pseudo_label.value_counts().sort_index().to_dict(),
            "reloaded_checkpoints": 4, "verified_prediction_files": 9, "verified_metric_rows": len(rows)}
