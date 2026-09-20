"""Step 1: read frozen train metadata/histories and validation predictions only.

No model training, inference, test parsing, checkpoint promotion or baseline writes.
Test files may be hashed as opaque bytes solely to prove preservation.
"""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class Inputs:
    """Explicit parsing allowlist; deliberately never calls a general Phase-2 loader."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.allowed = {}
        self.accessed = {}

    def allow(self, path, sha):
        path = Path(path).resolve()
        require(path.is_relative_to(self.root), "Input path escapes repository")
        self.allowed[path] = sha

    def read(self, path):
        path = Path(path).resolve()
        require(path in self.allowed, "Input not explicitly allowlisted")
        require("test" not in path.stem.lower(), "Test data must never be parsed")
        require(digest(path) == self.allowed[path], "Input hash mismatch: " + str(path))
        self.accessed[path.relative_to(self.root).as_posix()] = self.allowed[path]
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        return pd.read_csv(path, dtype={"user_key": str})

    def registered(self, directory, manifest, relative):
        require(relative in manifest["artifacts"], "Unregistered source artifact")
        path = (directory / relative).resolve()
        require(path.is_relative_to(directory.resolve()), "Artifact path escapes source")
        self.allow(path, manifest["artifacts"][relative]["sha256"])
        return self.read(path)


def preserved_hashes(root, output_root):
    """Freeze existing data/artifacts, notebooks, and prior pipeline code/configs."""
    root = Path(root).resolve()
    output_root = (root / output_root).resolve()
    paths = set()
    for folder in ("data", "processed_data", "final_data", "models", "results", "analysis", "src", "configs"):
        paths.update(p for p in (root / folder).rglob("*") if p.is_file()
                     and not p.resolve().is_relative_to(output_root)
                     and "__pycache__" not in p.parts and ".ipynb_checkpoints" not in p.parts)
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(paths)}


def validate_predictions(frame, users):
    require(frame.user_key.notna().all() and frame.user_key.is_unique, "Invalid prediction identities")
    require(set(frame.user_key) == set(users.index), "Prediction population differs")
    frame = frame.set_index("user_key").loc[users.index]
    require(frame.split.eq("validation").all(), "Non-validation prediction input")
    require(frame.label_binary.eq(users.label_binary).all(), "Prediction labels differ")
    supported = frame.status.eq("predicted")
    require(frame.bot_probability.notna().eq(supported).all(), "Missingness/status differs")
    require(frame.loc[~supported, "prediction"].isna().all(), "Unsupported users have decisions")
    require(np.isfinite(frame.loc[supported, "bot_probability"]).all()
            and frame.loc[supported, "bot_probability"].between(0, 1).all(), "Invalid probability")
    require(frame.loc[supported, "prediction"].isin([0, 1]).all(), "Invalid saved decision")
    return frame


def metrics(y, p, threshold):
    y = np.asarray(y)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp = int(((y == 0) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())
    fn, tp = int(((y == 1) & (pred == 0)).sum()), int(((y == 1) & (pred == 1)).sum())
    return {"threshold": float(threshold), "bot_precision": tp / (tp + fp) if tp + fp else 0.,
            "bot_recall": tp / (tp + fn) if tp + fn else 0.,
            "bot_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "predicted_bots": int(pred.sum())}


def sweep(y, p, config):
    grid = config["threshold_grid"]
    return pd.DataFrame([metrics(y, p, t) for t in np.linspace(grid["start"], grid["stop"], grid["points"])])


def diagnostic_best(table):
    return table.assign(distance=(table.threshold - .5).abs()).sort_values(
        ["bot_f1", "distance", "threshold"], ascending=[False, True, True]).iloc[0].drop("distance")


def model_catalog(source, manifest):
    if source in ("baselines", "ablation"):
        for model in manifest["models"]:
            yield model, f"models/{model}_metadata.json", f"histories/{model}.csv"
    elif source == "semi_supervised":
        for model in manifest["continuations"]:
            yield model, f"models/{model}_metadata.json", f"histories/{model}.csv"
    else:
        model = {"graph": "gcn", "graph_tabular": "gcn_tabular", "fusion": "multimodal"}[source]
        yield model, "models/training_metadata.json", f"histories/{model}.csv"


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def analyze(root, config, reader, output):
    phasepath = root / config["phase2_manifest"]
    reader.allow(phasepath, config["phase2_manifest_sha256"])
    phase = reader.read(phasepath)
    require(phase["identity_key"] == "user_key" and phase["fit_scope"] == "train_only", "Phase-2 contract differs")
    populations, masks = {}, {}
    for split in ("train", "validation"):
        populations[split] = reader.registered(phasepath.parent, phase, f"splits/users/{split}_users.csv").set_index("user_key")
        masks[split] = reader.registered(phasepath.parent, phase, f"splits/masks/{split}_modality_masks.csv").set_index("user_key")
        require(populations[split].index.is_unique and masks[split].index.is_unique
                and set(populations[split].index) == set(masks[split].index), "Invalid canonical identity table")
        masks[split] = masks[split].loc[populations[split].index]
    require(set(populations["train"].index).isdisjoint(populations["validation"].index), "Train/validation overlap")
    rows, curves, trades, histories, weights, coverage, registries, primary = [], [], [], [], [], [], [], {}
    loss_plots = []
    for source, spec in config["sources"].items():
        path = root / spec["manifest"]
        reader.allow(path, spec["sha256"])
        manifest = reader.read(path)
        require(manifest["complete"], "Incomplete source run")
        if "feature_order" in manifest:
            expected_order = phase["feature_order"]
            if source == "graph_tabular":
                expected_order = ["symmetrized_in_degree", "symmetrized_out_degree", "symmetrized_total_degree"] + expected_order
            require(manifest["feature_order"] == expected_order, "Source feature order differs")
        resolved = reader.registered(path.parent, manifest, "provenance/resolved_config.json")
        require(resolved["phase2_manifest_sha256"] == config["phase2_manifest_sha256"], "Source preprocessing differs")
        for model, meta_rel, hist_rel in model_catalog(source, manifest):
            identity = f"{source}/{model}"
            meta = reader.registered(path.parent, manifest, meta_rel)
            selected = meta.get("selected", meta.get("selection"))
            train = populations["train"]
            keys = meta.get("training_user_keys", meta.get("labeled_training_user_keys"))
            if keys is None:
                keys = train.index[masks["train"].has_graph.eq(1)].tolist()
            require(len(keys) == len(set(keys)) and set(keys) <= set(train.index), "Invalid training keys")
            counts = train.loc[keys, "label_binary"].value_counts()
            human, bot = int(counts[0]), int(counts[1])
            ratio = human / bot
            recorded_ratio = meta.get("pos_weight", ratio)
            if "class_weights" in meta:
                recorded_ratio = meta["class_weights"][1] / meta["class_weights"][0]
            require(np.isclose(ratio, recorded_ratio), "Class weights disagree with training population")
            weights.append({"model": identity, "labeled_train_human": human, "labeled_train_bot": bot,
                            "bot_to_human_loss_weight": float(recorded_ratio),
                            "loss_type": "weighted_CE" if "class_weights" in meta else
                            ("weighted_BCE" if "pos_weight" in meta else "balanced_logistic_loss"),
                            "pseudo_users": len(meta.get("pseudo_training_user_keys", [])),
                            "causal_weight_effect": "not_identifiable_without_controlled_retraining"})
            if hist_rel in manifest["artifacts"]:
                h = reader.registered(path.parent, manifest, hist_rel)
                best = h.loc[h.validation_loss.idxmin()]
                require(int(best.epoch) == meta["best_epoch"], "History best checkpoint mismatch")
                histories.append({"model": identity, "selected_variant": selected, "best_epoch": int(best.epoch),
                                  "last_epoch": int(h.iloc[-1].epoch), "best_validation_loss": float(best.validation_loss),
                                  "last_validation_loss": float(h.iloc[-1].validation_loss),
                                  "last_minus_best_validation_loss": float(h.iloc[-1].validation_loss-best.validation_loss),
                                  "epochs_after_best": int(h.iloc[-1].epoch-best.epoch),
                                  "training_loss_first": float(h.iloc[0].get("train_loss", h.iloc[0].get("labeled_loss"))),
                                  "training_loss_last": float(h.iloc[-1].get("train_loss", h.iloc[-1].get("labeled_loss")))})
                loss_plots.append((identity, h, int(best.epoch)))
            for variant in meta["checkpoints"]:
                checkpoint = meta["checkpoints"][variant]
                require(checkpoint in manifest["artifacts"] and
                        digest(path.parent / checkpoint) == manifest["artifacts"][checkpoint]["sha256"],
                        "Checkpoint artifact hash mismatch")
                rel = (f"predictions/{model}_validation.csv" if source == "semi_supervised"
                       else f"predictions/{model}_{variant}_validation.csv")
                if source == "semi_supervised" and variant != selected:
                    continue  # Final checkpoints have no saved validation predictions; never fabricate them.
                frame = validate_predictions(reader.registered(path.parent, manifest, rel), populations["validation"])
                valid = frame.status.eq("predicted")
                expected = pd.Series(True, index=frame.index)
                if source in ("graph", "graph_tabular", "fusion", "ablation"):
                    expected &= masks["validation"].has_graph.eq(1)
                if source in ("fusion", "ablation") or model.startswith("text_"):
                    expected &= masks["validation"].has_tweet_text.eq(1)
                require(valid.equals(expected), "Eligibility contract differs")
                sample = frame.loc[valid]
                y, p = sample.label_binary.to_numpy(), sample.bot_probability.to_numpy()
                require(set(y) == {0, 1}, "Two classes required for threshold diagnosis")
                table = sweep(y, p, config)
                best = diagnostic_best(table)
                ref = metrics(y, p, config["reference_threshold"])
                require(np.array_equal(sample.prediction, (p >= config["reference_threshold"]).astype(int)),
                        "Saved decisions differ from reference threshold; review required")
                info = {"model": identity, "variant": variant, "primary": variant == selected,
                        "validation_n": len(y), "validation_bots": int(y.sum()),
                        "cohort": "all_validation" if valid.all() else "graph_and_text_available_validation"}
                rows.append({**info, **ref, "roc_auc": roc_auc_score(y, p), "average_precision": average_precision_score(y, p),
                             "diagnostic_threshold": float(best.threshold), "apparent_max_f1": float(best.bot_f1),
                             "diagnostic_precision": float(best.bot_precision), "diagnostic_recall": float(best.bot_recall),
                             "diagnostic_fp": int(best.fp), "diagnostic_fn": int(best.fn),
                             "status": "same_validation_selection_and_scoring_optimistic_not_promoted"})
                curves.append(table.assign(**info))
                for target in config["minimum_recall_targets"]:
                    options = table[table.bot_recall >= target].copy()
                    pick = options.sort_values(["bot_precision", "bot_f1", "threshold"], ascending=[False, False, False]).iloc[0]
                    trades.append({**info, "minimum_recall": target, **pick.to_dict(), "status": "diagnostic_only"})
                coverage.append({**info, "canonical_users": len(frame), "unsupported_users": int((~valid).sum()),
                                 "coverage": float(valid.mean())})
                registries.append(frame.reset_index()[["user_key", "label_binary", "bot_probability", "prediction", "status"]].assign(
                    source_model=identity, source_variant=variant))
                if variant == selected:
                    primary[identity] = frame
    tables = {"checkpoint_threshold_summary": pd.DataFrame(rows), "threshold_sweep": pd.concat(curves, ignore_index=True),
              "precision_recall_tradeoffs": pd.DataFrame(trades), "learning_curve_summary": pd.DataFrame(histories),
              "class_weight_audit": pd.DataFrame(weights), "coverage": pd.DataFrame(coverage),
              "validation_prediction_registry": pd.concat(registries, ignore_index=True)}
    # Explicitly match primary models on the same users; never rank across cohorts.
    matched, trivial = [], []
    for cohort, keys in {"all_validation": populations["validation"].index,
                         "graph_and_text_available_validation": masks["validation"].index[
                             masks["validation"].has_graph.eq(1) & masks["validation"].has_tweet_text.eq(1)]}.items():
        y_ref = populations["validation"].loc[keys, "label_binary"].to_numpy()
        for label in (0, 1):
            trivial.append({"cohort": cohort, "reference": "always_bot" if label else "always_human",
                            "n": len(y_ref), "bots": int(y_ref.sum()),
                            **metrics(y_ref, np.full(len(y_ref), label), .5)})
        for identity, frame in primary.items():
            sample = frame.loc[keys]
            if not sample.status.eq("predicted").all():
                continue
            y, p = sample.label_binary.to_numpy(), sample.bot_probability.to_numpy()
            matched.append({"cohort": cohort, "model": identity, "n": len(y), "bots": int(y.sum()),
                            **metrics(y, p, .5), "roc_auc": roc_auc_score(y, p),
                            "average_precision": average_precision_score(y, p), "comparison": "descriptive_not_controlled_ablation"})
    tables["matched_primary_validation"] = pd.DataFrame(matched)
    tables["trivial_decision_references"] = pd.DataFrame(trivial)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    fig, axes = plt.subplots(5, 2, figsize=(12, 18))
    for ax, (identity, h, epoch) in zip(axes.flat, loss_plots):
        for column in ("train_loss", "labeled_loss", "pseudo_loss", "validation_loss"):
            if column in h:
                ax.plot(h.epoch, h[column], label=column)
        ax.axvline(epoch, color="black", linestyle=":", label="min validation loss")
        ax.set(title=identity, xlabel="Epoch", ylabel="Recorded weighted loss")
        ax.legend(fontsize=7)
    save_figure(fig, output / "figures/learning_curves.png")
    for cohort in ("all_validation", "graph_and_text_available_validation"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for identity, frame in primary.items():
            n = frame.bot_probability.notna().sum()
            if (cohort == "all_validation") != (n == len(frame)):
                continue
            s = frame[frame.status.eq("predicted")]
            precision, recall, _ = precision_recall_curve(s.label_binary, s.bot_probability)
            axes[0].step(recall, precision, where="post", label=identity)
            t = tables["threshold_sweep"]
            t = t[t.model.eq(identity) & t.primary]
            axes[1].plot(t.threshold, t.bot_f1, label=identity)
        axes[0].set(title=f"Validation PR: {cohort}", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.05))
        fig.suptitle("Frozen primary checkpoints only — diagnostic, no promotion", fontsize=11)
        axes[1].set(title="Validation threshold sensitivity (apparent)", xlabel="Threshold", ylabel="Bot F1", xlim=(0, 1), ylim=(0, 1))
        axes[1].axvline(.5, linestyle=":", color="black")
        axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
        save_figure(fig, output / f"figures/{cohort}.png")
    return tables


def run(root, config_path="configs/improvement_step1_v1.json"):
    root = Path(root).resolve()
    reader = Inputs(root)
    reader.allow(root / config_path, digest(root / config_path))
    config = reader.read(root / config_path)
    require(config["policy"] == "validation_only_descriptive_no_training_no_checkpoint_or_threshold_promotion", "Unsupported analysis policy")
    before = preserved_hashes(root, config["output_root"])
    output = root / config["output_root"] / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    (output / "figures").mkdir()
    (output / "provenance").mkdir()
    try:
        tables = analyze(root, config, reader, output)
        write_json(output / "provenance/resolved_config.json", config)
        write_json(output / "provenance/parsed_inputs.json", reader.accessed)
        write_json(output / "provenance/protected_files.json", before)
        for source in ("src/performance_diagnosis.py", "scripts/run_performance_diagnosis.py"):
            (output / "provenance" / Path(source).name).write_bytes((root / source).read_bytes())
        write_json(output / "provenance/environment.json", {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "matplotlib")},
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
            "test_labels_parsed": False, "training_performed": False, "threshold_promoted": False})
    finally:
        require(before == preserved_hashes(root, config["output_root"]), "A protected prior file changed")
    artifacts = {p.relative_to(output).as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
                 for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "manifest.json", {"schema_version": config["schema_version"], "complete": True,
        "scope": "train_histories_and_validation_only", "protected_files_verified": len(before),
        "artifacts": artifacts, "source_manifests": config["sources"],
        "limitations": ["Threshold maxima use the same validation users for selection and scoring; optimistic diagnostics only.",
            "Seven validation bots in graph cohort; descriptive comparisons, not reliable evidence of superiority.",
            "No unweighted training control: causal class-weight effects cannot be estimated.",
            "Weighted losses and train/eval modes differ; loss gaps alone do not prove overfitting.",
            "Semi-supervised last-checkpoint validation predictions were not saved; only selected predictions are analyzed.",
            "Previously inspected validation and test data do not become fresh holdouts."]})
    return output, tables


def verify(root, directory):
    """Independent recomputation from allowlisted source inputs; writes no files."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    require(manifest["complete"], "Incomplete diagnosis")
    for rel, meta in manifest["artifacts"].items():
        require(digest(directory / rel) == meta["sha256"], "Output hash mismatch: " + rel)
    inputs = json.loads((directory / "provenance/parsed_inputs.json").read_text())
    require(not any("test" in Path(p).stem.lower() for p in inputs), "Test input recorded")
    for path, sha in inputs.items():
        require(digest(Path(root) / path) == sha, "Source changed: " + path)
    registry = pd.read_csv(directory / "validation_prediction_registry.csv")
    summary = pd.read_csv(directory / "checkpoint_threshold_summary.csv")
    curves = pd.read_csv(directory / "threshold_sweep.csv")
    config = json.loads((directory / "provenance/resolved_config.json").read_text())
    for row in summary.itertuples():
        frame = registry[registry.source_model.eq(row.model) & registry.source_variant.eq(row.variant)]
        require(frame.user_key.is_unique and len(frame) == 144, "Output registry population differs")
        s = frame[frame.status.eq("predicted")]
        t = sweep(s.label_binary, s.bot_probability, config)
        saved = curves[curves.model.eq(row.model) & curves.variant.eq(row.variant)][t.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(t, saved, check_dtype=False, atol=1e-12, rtol=1e-12)
        best = diagnostic_best(t)
        ref = metrics(s.label_binary, s.bot_probability, .5)
        # Independent sklearn cross-check, outside the vectorized grid calculation.
        prediction = (s.bot_probability.to_numpy() >= .5).astype(int)
        np.testing.assert_array_equal(confusion_matrix(s.label_binary, prediction, labels=[0, 1]).ravel(),
                                      [ref[k] for k in ("tn", "fp", "fn", "tp")])
        require(np.isclose(ref["bot_f1"], f1_score(s.label_binary, prediction, zero_division=0)), "Independent F1 mismatch")
        family, model = row.model.split("/", 1)
        filename = (f"predictions/{model}_validation.csv" if family == "semi_supervised"
                    else f"predictions/{model}_{row.variant}_validation.csv")
        source_path = Path(config["sources"][family]["manifest"]).parent / filename
        require(source_path.as_posix() in inputs, "Prediction source was not allowlisted")
        source = pd.read_csv(Path(root) / source_path, dtype={"user_key": str}).set_index("user_key")
        columns = ["label_binary", "bot_probability", "prediction", "status"]
        pd.testing.assert_frame_equal(frame.set_index("user_key")[columns].sort_index(), source[columns].sort_index(),
                                      check_dtype=False, atol=1e-12, rtol=1e-12)
        for key, value in ref.items():
            require(np.isclose(getattr(row, key), value), "Summary mismatch: " + key)
        require(np.isclose(row.diagnostic_threshold, best.threshold) and np.isclose(row.apparent_max_f1, best.bot_f1), "Diagnostic threshold mismatch")
        require(np.isclose(row.roc_auc, roc_auc_score(s.label_binary, s.bot_probability)) and
                np.isclose(row.average_precision, average_precision_score(s.label_binary, s.bot_probability)), "Ranking metric mismatch")
    return {"artifacts": len(manifest["artifacts"]), "validation_checkpoint_rows": len(summary),
            "threshold_rows": len(curves), "test_inputs_parsed": 0,
            "protected_files_at_execution": manifest["protected_files_verified"]}
