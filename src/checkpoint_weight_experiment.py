"""Step 2: controlled tabular MLP class weights and checkpoint selection.

Only training/validation artifacts are parsed. Existing preprocessing and baseline
files are immutable. Paired seeds hold initialization and RNG consumption fixed.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, roc_auc_score

from src.baselines import BaselineMLP, MASKS, align, keyed, seed_all, probabilities
from src.performance_diagnosis import Inputs, digest, metrics, preserved_hashes, require, write_json, plt, save_figure


POLICIES = ("last", "best_weighted_validation_loss", "best_unweighted_validation_loss")


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def weights_from_training(y):
    y = np.asarray(y)
    require(set(y) == {0, 1}, "Both training classes required")
    ratio = float((y == 0).sum() / (y == 1).sum())
    return {"unweighted": 1.0, "sqrt_train_ratio": float(np.sqrt(ratio)), "full_train_ratio": ratio}


def load_inputs(root, config, reader):
    path = root / config["phase2_manifest"]
    reader.allow(path, config["phase2_manifest_sha256"])
    manifest = reader.read(path)
    require(manifest["identity_key"] == "user_key" and manifest["fit_scope"] == "train_only", "Preprocessing contract differs")
    order = manifest["feature_order"]
    require(len(order) == len(set(order)) == 40, "Expected exact 40-feature order")
    stats = reader.registered(path.parent, manifest, "preprocessing/fitted_statistics.json")
    registry = reader.registered(path.parent, manifest, "config/feature_registry.json")
    temporal = [r["feature"] for r in registry if r["group"] == "temporal" and r["feature"] in order]
    require(stats["feature_order"] == order, "Fitted feature order differs")
    data = {}
    for split in ("train", "validation"):
        users = keyed(reader.registered(path.parent, manifest, f"splits/users/{split}_users.csv"), split)
        tab = align(keyed(reader.registered(path.parent, manifest, f"splits/tabular/{split}_tabular.csv"), "tabular"), users, split)
        masks = align(keyed(reader.registered(path.parent, manifest, f"splits/masks/{split}_modality_masks.csv"), "masks"), users, split)
        observed = align(keyed(reader.registered(path.parent, manifest, f"splits/masks/{split}_observation_masks.csv"), "observed"), users, split)
        require(users.label_binary.isin([0, 1]).all() and tab.label_binary.eq(users.label_binary).all(), "Labels differ")
        require(len(users) == config["expected"][f"{split}_users"], "Population changed")
        require(int(users.label_binary.sum()) == config["expected"][f"{split}_bots"], "Class count changed")
        require(masks[list(MASKS)].isin([0, 1]).all().all() and masks[list(MASKS)].eq(users[list(MASKS)]).all().all(), "Modality masks differ")
        obs = observed[["observed__" + f for f in order]]
        require(obs.isin([0, 1]).all().all() and np.array_equal(obs.to_numpy(), users[order].notna().to_numpy()), "Observation masks differ")
        count = observed[["observed__" + f for f in temporal]].sum(axis=1)
        require(masks.has_temporal.eq(count.gt(0)).all() and masks.temporal_observed_count.eq(count).all()
                and masks.temporal_complete.eq(count.eq(len(temporal))).all(), "Temporal coverage differs")
        for f in temporal:
            require(tab.loc[observed["observed__" + f].eq(0), f].eq(0).all(), "Temporal placeholder changed")
        x = tab[order].to_numpy(dtype=np.float32)
        require(np.isfinite(x).all(), "Nonfinite inputs; refitting/imputation forbidden")
        data[split] = {"users": users, "masks": masks, "observed": observed, "x": x,
                       "y": users.label_binary.to_numpy(dtype=np.float32)}
    train_keys = data["train"]["users"].index.tolist()
    require(set(train_keys).isdisjoint(data["validation"]["users"].index), "Train/validation identity overlap")
    require(set(stats["training_user_keys"]) == set(train_keys), "Preprocessing fit users differ")
    bpath = root / config["baseline_manifest"]
    reader.allow(bpath, config["baseline_manifest_sha256"])
    baseline = reader.read(bpath)
    require(baseline["complete"] and baseline["phase2_manifest_sha256"] == config["phase2_manifest_sha256"]
            and baseline["feature_order"] == order, "Baseline contract differs")
    meta = reader.registered(bpath.parent, baseline, "models/tabular_mlp_metadata.json")
    require(meta["training_user_keys"] == train_keys, "Baseline training order differs")
    references = {"history": reader.registered(bpath.parent, baseline, "histories/tabular_mlp.csv"), "metadata": meta}
    for variant in ("last", "best_validation_loss"):
        references[variant] = reader.registered(bpath.parent, baseline, f"predictions/tabular_mlp_{variant}_validation.csv")
    require(config["architecture"] == {"input_dim": 40, "hidden_dims": [64, 32], "dropout": .3}, "Architecture change forbidden")
    require(config["training"] == {k: meta["specification"][k] for k in ("epochs", "batch_size", "lr", "weight_decay")}, "Training hyperparameters changed")
    return data, order, references


def validation_losses(model, population, batch_size, weight):
    """Reproduce historical batch-mean weighted loss and common user-mean BCE."""
    model.eval()
    weighted, unweighted, n = [], 0., 0
    with torch.no_grad():
        for start in range(0, len(population["x"]), batch_size):
            x = torch.tensor(population["x"][start:start + batch_size], dtype=torch.float32)
            y = torch.tensor(population["y"][start:start + batch_size], dtype=torch.float32).reshape(-1, 1)
            logits = model(x)
            weighted.append(nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor([weight], dtype=torch.float32)).item())
            unweighted += nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="sum").item()
            n += len(x)
    return float(np.mean(weighted)), unweighted / n


def train_one(data, config, seed, weight):
    seed_all(seed, config["torch_threads"])
    model = BaselineMLP(**config["architecture"])
    initial_hash = state_hash(model.state_dict())
    spec = config["training"]
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"])
    loaders = {}
    for split in ("train", "validation"):
        pop = data[split]
        require(len(pop["x"]) % spec["batch_size"] != 1, "Singleton BatchNorm batch; do not drop users")
        dataset = TensorDataset(torch.tensor(pop["x"], dtype=torch.float32), torch.tensor(pop["y"], dtype=torch.float32).reshape(-1, 1),
                                torch.arange(len(pop["x"])))
        loaders[split] = DataLoader(dataset, batch_size=spec["batch_size"], shuffle=split == "train")
    states, best, epochs, history = {}, {"weighted": float("inf"), "unweighted": float("inf")}, {}, []
    order_hash = hashlib.sha256()
    for epoch in range(1, spec["epochs"] + 1):
        row = {"epoch": epoch}
        for split, loader in loaders.items():
            model.train(split == "train")
            total, unweighted, count = 0., 0., 0
            with torch.set_grad_enabled(split == "train"):
                for x, y, indices in loader:
                    if split == "train":
                        optimizer.zero_grad()
                        order_hash.update(indices.numpy().tobytes())
                    logits = model(x)
                    loss = criterion(logits, y)
                    if split == "train":
                        loss.backward()
                        optimizer.step()
                    else:
                        unweighted += nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="sum").item()
                    total += loss.item()
                    count += len(x)
            row[split + "_weighted_loss"] = total / len(loader)
            if split == "validation":
                row["validation_unweighted_loss"] = unweighted / count
        require(np.isfinite(list(row.values())).all(), "Nonfinite training history")
        history.append(row)
        for kind in best:
            value = row[f"validation_{kind}_loss"]
            if value < best[kind]:
                best[kind], epochs[kind] = value, epoch
                states[f"best_{kind}_validation_loss"] = deepcopy(model.state_dict())
        if epoch % 25 == 0:
            print(f"seed={seed} weight={weight:.6f} epoch={epoch}: validation BCE={row['validation_unweighted_loss']:.4f}", flush=True)
    states["last"] = deepcopy(model.state_dict())
    return states, pd.DataFrame(history), {"initial_state_sha256": initial_hash, "batch_order_sha256": order_hash.hexdigest(),
        "final_rng_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        "epochs": {"last": spec["epochs"], **{f"best_{k}_validation_loss": v for k, v in epochs.items()}}}


def score(model, data, config, **identity):
    pop = data["validation"]
    p = probabilities(model, pop["x"], config["training"]["batch_size"])
    registry = pop["users"][["label_binary"]].join(pop["masks"])
    registry["bot_probability"], registry["prediction"] = p, (p >= config["threshold"]).astype(int)
    registry["status"], registry["split"] = "predicted", "validation"
    for k, v in identity.items():
        registry[k] = v
    y = pop["y"]
    return registry.reset_index(), {**identity, "n": len(y), "bots": int(y.sum()),
        **metrics(y, p, config["threshold"]), "roc_auc": roc_auc_score(y, p), "average_precision": average_precision_score(y, p)}


def summarize(table):
    measured = ["bot_f1", "bot_precision", "bot_recall", "roc_auc", "average_precision", "epoch"]
    summary = table.groupby(["weight_rule", "policy"])[measured].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(c) for c in summary.columns]
    effects = []
    for row in table.itertuples():
        reference = table[(table.seed == row.seed) & (table.weight_rule == "full_train_ratio") & (table.policy == row.policy)].iloc[0]
        last = table[(table.seed == row.seed) & (table.weight_rule == row.weight_rule) & (table.policy == "last")].iloc[0]
        for kind, ref in (("weight_vs_full_same_policy", reference), ("checkpoint_vs_last_same_weight", last)):
            effects.append({"seed": row.seed, "weight_rule": row.weight_rule, "policy": row.policy, "effect": kind,
                            **{f"delta_{m}": float(getattr(row, m)-ref[m]) for m in measured if m != "epoch"}})
    return summary.reset_index(), pd.DataFrame(effects)


def run(root, config_path="configs/improvement_step2_v1.json"):
    root = Path(root).resolve()
    reader = Inputs(root)
    reader.allow(root / config_path, digest(root / config_path))
    config = reader.read(root / config_path)
    require(config["scope"] == "train_only_optimization_validation_only_selection_no_test_no_threshold_tuning_no_promotion", "Invalid scope")
    require(config["seeds"] == [42, 43, 44] and config["checkpoint_policies"] == list(POLICIES)
            and config["threshold"] == .5 and config["device"] == "cpu", "Prespecified protocol differs")
    before = preserved_hashes(root, config["output_root"])
    output = root / config["output_root"] / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("models", "histories", "predictions", "provenance", "figures"):
        (output / folder).mkdir()
    try:
        data, order, references = load_inputs(root, config, reader)
        weights = weights_from_training(data["train"]["y"])
        require(list(weights) == config["weight_rules"], "Weight grid differs")
        # Materialize the protocol before any training.
        write_json(output / "provenance/resolved_config.json", config)
        write_json(output / "provenance/resolved_weights.json", weights)
        write_json(output / "provenance/parsed_inputs.json", reader.accessed)
        for split, pop in data.items():
            pop["users"][["label_binary"]].join(pop["masks"]).to_csv(output / f"provenance/{split}_population.csv")
            pop["observed"].to_csv(output / f"provenance/{split}_observation_masks.csv")
        results, runs, histories = [], {}, {}
        # Reproduction control comes first; all seeds use the same fixed arm ordering.
        for seed in config["seeds"]:
            for rule in ("full_train_ratio", "sqrt_train_ratio", "unweighted"):
                print(f"Training {rule}, seed {seed}", flush=True)
                states, history, info = train_one(data, config, seed, weights[rule])
                name = f"{rule}_seed{seed}"
                history.to_csv(output / f"histories/{name}.csv", index=False)
                info.update({"seed": seed, "weight_rule": rule, "pos_weight": weights[rule],
                    "training_user_keys": data["train"]["users"].index.tolist(), "checkpoints": {}})
                if seed == 42 and rule == "full_train_ratio":
                    expected = references["history"].rename(columns={"train_loss": "train_weighted_loss", "validation_loss": "validation_weighted_loss"})
                    pd.testing.assert_frame_equal(history[expected.columns], expected, check_dtype=False, atol=1e-7, rtol=1e-7)
                    require(info["epochs"]["best_weighted_validation_loss"] == references["metadata"]["best_epoch"], "Baseline epoch not reproduced")
                for policy in POLICIES:
                    model = BaselineMLP(**config["architecture"])
                    model.load_state_dict(states[policy], strict=True)
                    registry, row = score(model, data, config, seed=seed, weight_rule=rule, policy=policy, epoch=info["epochs"][policy])
                    weighted, common = validation_losses(model, data["validation"], config["training"]["batch_size"], weights[rule])
                    row.update({"pos_weight": weights[rule], "validation_weighted_loss": weighted, "validation_unweighted_loss": common})
                    if seed == 42 and rule == "full_train_ratio" and policy in ("last", "best_weighted_validation_loss"):
                        old = references["last" if policy == "last" else "best_validation_loss"].set_index("user_key")
                        new = registry.set_index("user_key").loc[old.index]
                        np.testing.assert_allclose(new.bot_probability, old.bot_probability, atol=1e-6, rtol=1e-6)
                        np.testing.assert_array_equal(new.prediction, old.prediction)
                    relative = f"models/{name}_{policy}.pt"
                    torch.save({"state_dict": states[policy], "architecture": config["architecture"], "feature_order": order,
                        "seed": seed, "weight_rule": rule, "pos_weight": weights[rule], "policy": policy,
                        "epoch": info["epochs"][policy], "threshold": .5, "phase2_manifest_sha256": config["phase2_manifest_sha256"],
                        "training_user_keys": info["training_user_keys"]}, output / relative)
                    info["checkpoints"][policy] = relative
                    registry.to_csv(output / f"predictions/{name}_{policy}_validation.csv", index=False)
                    results.append(row)
                runs[name], histories[name] = info, history
            paired = [v for v in runs.values() if v["seed"] == seed]
            for field in ("initial_state_sha256", "batch_order_sha256", "final_rng_sha256"):
                require(len({v[field] for v in paired}) == 1, "Paired stochastic control failed: " + field)
        table = pd.DataFrame(results)
        summary, effects = summarize(table)
        table.to_csv(output / "validation_metrics.csv", index=False)
        summary.to_csv(output / "seed_summary.csv", index=False)
        effects.to_csv(output / "paired_effects.csv", index=False)
        write_json(output / "training_metadata.json", runs)
        fig, axes = plt.subplots(3, 3, figsize=(13, 10))
        for row, rule in enumerate(config["weight_rules"]):
            for col, seed in enumerate(config["seeds"]):
                h = histories[f"{rule}_seed{seed}"]
                ax = axes[row, col]
                ax.plot(h.epoch, h.validation_unweighted_loss, label="Common validation BCE")
                ax.plot(h.epoch, h.validation_weighted_loss, label="Weighted validation loss", alpha=.7)
                ax.axvline(runs[f"{rule}_seed{seed}"]["epochs"][config["primary_comparison_policy"]], color="black", linestyle=":")
                ax.set(title=f"{rule}, seed {seed}", xlabel="Epoch", ylabel="Validation loss")
                ax.legend(fontsize=7)
        save_figure(fig, output / "figures/checkpoint_curves.png")
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, metric in zip(axes, ("bot_f1", "bot_precision", "bot_recall")):
            for seed in config["seeds"]:
                s = table[(table.seed == seed) & (table.policy == config["primary_comparison_policy"])].set_index("weight_rule").loc[config["weight_rules"]]
                ax.plot(range(3), s[metric], marker="o", label=f"seed {seed}")
            ax.set(xticks=range(3), xticklabels=["1.0", "sqrt ratio", "full ratio"], xlabel="Positive-class weight", ylabel=metric, ylim=(0, 1))
            ax.legend()
        fig.suptitle("Same validation users; common checkpoint criterion; threshold 0.5")
        save_figure(fig, output / "figures/paired_weight_effects.png")
        write_json(output / "provenance/protected_files.json", before)
        for rel in ("src/checkpoint_weight_experiment.py", "src/baselines.py", "src/performance_diagnosis.py", "scripts/run_checkpoint_weight_experiment.py"):
            (output / "provenance" / Path(rel).name).write_bytes((root / rel).read_bytes())
        write_json(output / "provenance/environment.json", {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "pandas", "scikit-learn", "matplotlib")},
            "torch_threads": config["torch_threads"], "deterministic_algorithms": True,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
            "test_data_parsed": False, "unlabeled_data_used": False, "threshold_tuned": False, "baseline_control_reproduced": True})
    finally:
        require(before == preserved_hashes(root, config["output_root"]), "Protected prior files changed")
    artifacts = {p.relative_to(output).as_posix(): {"sha256": digest(p)} for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "manifest.json", {"complete": True, "schema_version": config["schema_version"],
        "feature_order": order, "protected_files_verified": len(before), "runs": runs, "artifacts": artifacts,
        "primary_comparison_policy": config["primary_comparison_policy"], "promoted_model": None,
        "limitations": ["Shared previously inspected validation set used for selection and reporting; descriptive development results only.",
            "Three seeds measure optimization variability, not population uncertainty or statistical significance.",
            "Weight-specific validation losses are not comparable across arms; common user-mean BCE is the primary checkpoint criterion.",
            "Fixed-epoch comparisons isolate the training-weight treatment; checkpoint comparisons additionally include the selection policy.",
            "No test evaluation, threshold tuning, new architecture, or pseudo-label training performed."]})
    return output, table, summary


def verify(root, output):
    root, output = Path(root).resolve(), Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    require(manifest["complete"] and manifest["promoted_model"] is None, "Invalid completed experiment")
    for path, meta in manifest["artifacts"].items():
        require(digest(output / path) == meta["sha256"], "Artifact hash mismatch: " + path)
    config = json.loads((output / "provenance/resolved_config.json").read_text())
    seed_all(42, config["torch_threads"])
    reader = Inputs(root)
    data, order, refs = load_inputs(root, config, reader)
    weights = weights_from_training(data["train"]["y"])
    rows = []
    for name, info in manifest["runs"].items():
        require(info["training_user_keys"] == data["train"]["users"].index.tolist(), "Training keys differ")
        require(not set(info["training_user_keys"]) & set(data["validation"]["users"].index), "Validation in training")
        require(info["pos_weight"] == weights[info["weight_rule"]], "Weight not derived from training labels")
        h = pd.read_csv(output / f"histories/{name}.csv")
        require(h.epoch.tolist() == list(range(1, config["training"]["epochs"] + 1)), "History epochs differ")
        for policy, path in info["checkpoints"].items():
            snapshot = torch.load(output / path, map_location="cpu", weights_only=True)
            epoch = config["training"]["epochs"] if policy == "last" else int(h.loc[h[
                "validation_weighted_loss" if policy == "best_weighted_validation_loss" else "validation_unweighted_loss"].idxmin(), "epoch"])
            require(snapshot["epoch"] == info["epochs"][policy] == epoch, "Checkpoint selection differs")
            require(snapshot["architecture"] == config["architecture"] and snapshot["feature_order"] == order
                    and snapshot["training_user_keys"] == info["training_user_keys"] and snapshot["pos_weight"] == info["pos_weight"]
                    and snapshot["seed"] == info["seed"] and snapshot["policy"] == policy
                    and snapshot["threshold"] == .5 and snapshot["phase2_manifest_sha256"] == config["phase2_manifest_sha256"], "Checkpoint contract differs")
            model = BaselineMLP(**snapshot["architecture"])
            model.load_state_dict(snapshot["state_dict"], strict=True)
            frame, row = score(model, data, config, seed=info["seed"], weight_rule=info["weight_rule"], policy=policy, epoch=epoch)
            saved = pd.read_csv(output / f"predictions/{name}_{policy}_validation.csv", dtype={"user_key": str})
            pd.testing.assert_frame_equal(frame, saved, check_dtype=False, rtol=1e-6, atol=1e-7)
            weighted, common = validation_losses(model, data["validation"], config["training"]["batch_size"], info["pos_weight"])
            recorded = h.set_index("epoch").loc[epoch]
            np.testing.assert_allclose([weighted, common], [recorded.validation_weighted_loss, recorded.validation_unweighted_loss], atol=1e-7, rtol=1e-7)
            row.update({"pos_weight": info["pos_weight"], "validation_weighted_loss": weighted, "validation_unweighted_loss": common})
            rows.append(row)
            if name == "full_train_ratio_seed42" and policy in ("last", "best_weighted_validation_loss"):
                ref = refs["last" if policy == "last" else "best_validation_loss"].set_index("user_key")
                np.testing.assert_allclose(frame.set_index("user_key").loc[ref.index, "bot_probability"], ref.bot_probability, atol=1e-6, rtol=1e-6)
    expected = pd.DataFrame(rows).sort_values(["seed", "weight_rule", "policy"]).reset_index(drop=True)
    saved = pd.read_csv(output / "validation_metrics.csv").sort_values(["seed", "weight_rule", "policy"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, saved, check_dtype=False, rtol=1e-6, atol=1e-7)
    summary, effects = summarize(saved)
    pd.testing.assert_frame_equal(summary, pd.read_csv(output / "seed_summary.csv"), check_dtype=False, atol=1e-10, rtol=1e-10)
    # Effect row order follows original export order rather than sorted metric rows.
    effects = effects.sort_values(["seed", "weight_rule", "policy", "effect"]).reset_index(drop=True)
    saved_effects = pd.read_csv(output / "paired_effects.csv").sort_values(["seed", "weight_rule", "policy", "effect"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(effects, saved_effects, check_dtype=False, atol=1e-10, rtol=1e-10)
    for seed in config["seeds"]:
        paired = [v for v in manifest["runs"].values() if v["seed"] == seed]
        require(len(paired) == 3, "Missing paired arm")
        for field in ("initial_state_sha256", "batch_order_sha256", "final_rng_sha256"):
            require(len({v[field] for v in paired}) == 1, "Pairing differs")
    parsed = json.loads((output / "provenance/parsed_inputs.json").read_text())
    require(not any("test" in Path(p).stem.lower() for p in parsed), "Test input parsed")
    return {"training_runs": len(manifest["runs"]), "checkpoints_reloaded": len(rows),
            "validation_prediction_files": len(rows), "test_inputs_parsed": 0,
            "baseline_control_reproduced": True, "protected_files_at_execution": manifest["protected_files_verified"]}
