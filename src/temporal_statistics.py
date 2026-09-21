"""Approved capacity-matched temporal-statistics experiment; no test-data parsing."""
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

from src.baselines import seed_all
from src.checkpoint_weight_experiment import load_inputs as load_reference_inputs, score, state_hash, validation_losses, weights_from_training
from src.performance_diagnosis import Inputs, digest, preserved_hashes, require, write_json, plt, save_figure


ARMS = ("flat_masked", "temporal_branch", "without_seven_statistics")
POLICIES = ("best_unweighted_validation_loss", "last")


def feature_mapping(order, temporal):
    require(len(order) == len(set(order)) == 40, "Expected 40 unique Phase-2 features")
    require(len(temporal) == len(set(temporal)) == 7 and set(temporal) <= set(order), "Invalid temporal group")
    require([f for f in order if f in temporal] == temporal, "Temporal manifest order differs")
    other = [f for f in order if f not in temporal]
    require(len(other) == 33 and not set(other) & set(temporal), "Feature groups overlap")
    return {"feature_order": order, "temporal_features": temporal, "other_features": other,
            "temporal_indices": [order.index(f) for f in temporal], "other_indices": [order.index(f) for f in other],
            "input_order": order + ["observed__" + f for f in temporal],
            "temporal_mask_indices": list(range(40, 47))}


class TemporalExperimentModel(nn.Module):
    """All arms accept the same 47-column interface; C ignores temporal values/masks."""
    def __init__(self, arm, mapping, dropout=.3):
        super().__init__()
        require(arm in ARMS, "Unknown experiment arm")
        self.arm = arm
        self.register_buffer("temporal_indices", torch.tensor(mapping["temporal_indices"], dtype=torch.long))
        self.register_buffer("other_indices", torch.tensor(mapping["other_indices"], dtype=torch.long))
        if arm == "temporal_branch":
            self.other_encoder = nn.Sequential(nn.Linear(33, 48), nn.BatchNorm1d(48), nn.ReLU(), nn.Dropout(dropout))
            self.temporal_encoder = nn.Sequential(nn.Linear(14, 16), nn.ReLU(), nn.Dropout(dropout))
            self.classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
        else:
            inputs, hidden = (47, 49) if arm == "flat_masked" else (33, 59)
            self.network = nn.Sequential(nn.Linear(inputs, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
                                         nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def split_inputs(self, x):
        require(x.ndim == 2 and x.shape[1] == 47, "Expected 47-column model interface")
        mask = x[:, 40:47]
        require(torch.all((mask == 0) | (mask == 1)).item(), "Nonbinary temporal observation mask")
        values = torch.where(mask.bool(), x[:, self.temporal_indices], torch.zeros_like(mask))
        return x[:, self.other_indices], values, mask

    def temporal_representation(self, values, mask):
        hidden = self.temporal_encoder(torch.cat([values, mask], dim=1))
        # Deterministic availability masking, not learned gated fusion.
        return hidden * mask.any(dim=1, keepdim=True).to(hidden.dtype)

    def forward(self, x):
        other, temporal, mask = self.split_inputs(x)
        if self.arm == "without_seven_statistics":
            return self.network(other)
        if self.arm == "flat_masked":
            clean = x[:, :40].clone()
            clean[:, self.temporal_indices] = temporal
            return self.network(torch.cat([clean, mask], dim=1))
        hidden = torch.cat([self.other_encoder(other), self.temporal_representation(temporal, mask)], dim=1)
        return self.classifier(hidden)


def load_inputs(root, config, reader):
    # Reuse the unchanged train/validation-only validation contract. Its architecture
    # field describes the historical reference, not one of the new experiment arms.
    bridge = {**config, "architecture": config["reference_architecture"]}
    data, order, _ = load_reference_inputs(root, bridge, reader)
    path = root / config["phase2_manifest"]
    manifest = reader.read(path)
    registry = reader.registered(path.parent, manifest, "config/feature_registry.json")
    actual = [f for f in order if any(r["feature"] == f and r["group"] == "temporal" for r in registry)]
    require(actual == config["temporal_features"], "Temporal feature policy differs from Phase-2")
    mapping = feature_mapping(order, actual)
    for pop in data.values():
        masks = pop["observed"][["observed__" + f for f in actual]].to_numpy(dtype=np.float32)
        pop["x"] = np.concatenate([pop["x"], masks], axis=1)
    step2path = root / config["step2_manifest"]
    reader.allow(step2path, config["step2_manifest_sha256"])
    previous = reader.read(step2path)
    require(previous["complete"] and previous["feature_order"] == order, "Step-2 reference differs")
    reference = reader.registered(step2path.parent, previous, "validation_metrics.csv")
    reference = reference[reference.weight_rule.eq("full_train_ratio") & reference.policy.eq(config["selection"])].copy()
    require(sorted(reference.seed.tolist()) == config["seeds"], "Reference seeds differ")
    reference["comparison_role"] = "historical_only_different_architecture_and_shuffle_schedule"
    return data, mapping, reference


def make_loader(pop, spec, seed, split):
    require(len(pop["x"]) > 1 and (split != "train" or len(pop["x"]) % spec["batch_size"] != 1), "Singleton training batch; cannot drop users")
    tensors = TensorDataset(torch.tensor(pop["x"], dtype=torch.float32),
                            torch.tensor(pop["y"], dtype=torch.float32).reshape(-1, 1), torch.arange(len(pop["x"])))
    generator = torch.Generator().manual_seed(seed if split == "train" else seed + 100000)
    return DataLoader(tensors, batch_size=spec["batch_size"], shuffle=split == "train", generator=generator)


def train_one(data, mapping, config, arm, seed, weight):
    seed_all(seed, config["torch_threads"])
    model = TemporalExperimentModel(arm, mapping, config["dropout"])
    count = sum(p.numel() for p in model.parameters())
    require(count == config["expected_parameters"][arm], "Architecture parameter count differs")
    initial_hash = state_hash(model.state_dict())
    spec = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], dtype=torch.float32))
    loaders = {split: make_loader(pop, spec, seed, split) for split, pop in data.items()}
    best, best_epoch, best_state = float("inf"), None, None
    history, order_hash, epoch_order_hashes = [], hashlib.sha256(), []
    for epoch in range(1, spec["epochs"] + 1):
        row = {"epoch": epoch}
        seen = []
        for split, loader in loaders.items():
            model.train(split == "train")
            total, unweighted, n = 0., 0., 0
            with torch.set_grad_enabled(split == "train"):
                for x, y, indices in loader:
                    if split == "train":
                        seen.extend(indices.tolist())
                        order_hash.update(indices.numpy().tobytes())
                        optimizer.zero_grad()
                    logits = model(x)
                    loss = criterion(logits, y)
                    if split == "train":
                        loss.backward()
                        optimizer.step()
                    else:
                        unweighted += nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="sum").item()
                    total += loss.item()
                    n += len(x)
            row[split + "_weighted_loss"] = total / len(loader)
            if split == "validation":
                row["validation_unweighted_loss"] = unweighted / n
        require(sorted(seen) == list(range(len(data["train"]["y"]))), "Training epoch omitted or repeated a user")
        epoch_order_hashes.append(hashlib.sha256(np.asarray(seen, dtype=np.int64).tobytes()).hexdigest())
        require(np.isfinite(list(row.values())).all(), "Nonfinite losses")
        history.append(row)
        if row["validation_unweighted_loss"] < best:
            best, best_epoch, best_state = row["validation_unweighted_loss"], epoch, deepcopy(model.state_dict())
        if epoch % 25 == 0:
            print(f"{arm}, seed {seed}, epoch {epoch}: validation BCE {row['validation_unweighted_loss']:.4f}", flush=True)
    return {config["selection"]: best_state, "last": deepcopy(model.state_dict())}, pd.DataFrame(history), {
        "arm": arm, "seed": seed, "pos_weight": weight, "parameter_count": count,
        "initial_state_sha256": initial_hash, "batch_order_sha256": order_hash.hexdigest(),
        "epoch_order_sha256": epoch_order_hashes, "epochs": {config["selection"]: best_epoch, "last": spec["epochs"]}}


def summarize(table):
    measured = ["bot_f1", "bot_precision", "bot_recall", "roc_auc", "average_precision", "epoch"]
    summary = table.groupby(["arm", "policy"])[measured].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(c) for c in summary.columns]
    effects = []
    for row in table[table.arm.eq("temporal_branch")].itertuples():
        for control, role in (("flat_masked", "primary"), ("without_seven_statistics", "secondary")):
            ref = table[table.arm.eq(control) & table.seed.eq(row.seed) & table.policy.eq(row.policy)].iloc[0]
            effects.append({"seed": row.seed, "policy": row.policy, "comparison": f"temporal_branch_minus_{control}",
                            "role": role, **{f"delta_{m}": float(getattr(row, m) - ref[m]) for m in measured if m != "epoch"}})
    return summary.reset_index(), pd.DataFrame(effects)


def validate_config(config):
    require(config["arms"] == list(ARMS) and config["seeds"] == [42, 43, 44], "Approved arms/seeds differ")
    require(config["selection"] == POLICIES[0] and config["threshold"] == .5 and config["dropout"] == .3,
            "Approved selection/threshold/dropout differs")
    require(config["device"] == "cpu" and config["torch_threads"] == 1 and
            config["scope"] == "train_only_optimization_validation_only_selection_no_test_no_threshold_tuning_no_promotion",
            "Approved scope differs")


def run(root, config_path="configs/temporal_statistics_v1.json"):
    root = Path(root).resolve()
    reader = Inputs(root)
    reader.allow(root / config_path, digest(root / config_path))
    config = reader.read(root / config_path)
    validate_config(config)
    before = preserved_hashes(root, config["output_root"])
    output = root / config["output_root"] / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    for name in ("models", "histories", "predictions", "provenance", "figures"):
        (output / name).mkdir()
    try:
        data, mapping, reference = load_inputs(root, config, reader)
        weight = weights_from_training(data["train"]["y"])["full_train_ratio"]
        write_json(output / "provenance/resolved_config.json", config)
        write_json(output / "provenance/feature_mapping.json", mapping)
        write_json(output / "provenance/parsed_inputs.json", reader.accessed)
        reference.to_csv(output / "historical_reference_validation.csv", index=False)
        coverage = []
        for split, pop in data.items():
            pop["users"][["label_binary"]].join(pop["masks"]).to_csv(output / f"provenance/{split}_population.csv")
            pop["observed"].to_csv(output / f"provenance/{split}_observation_masks.csv")
            coverage.append({"split": split, "users": len(pop["y"]), "temporal_available": int(pop["masks"].has_temporal.sum()),
                             "temporal_complete": int(pop["masks"].temporal_complete.sum())})
        pd.DataFrame(coverage).to_csv(output / "temporal_coverage.csv", index=False)
        rows, runs, histories = [], {}, {}
        for seed in config["seeds"]:
            for arm in config["arms"]:
                print(f"Training {arm}, seed {seed}", flush=True)
                states, history, info = train_one(data, mapping, config, arm, seed, weight)
                name = f"{arm}_seed{seed}"
                history.to_csv(output / f"histories/{name}.csv", index=False)
                info["training_user_keys"] = data["train"]["users"].index.tolist()
                info["checkpoints"] = {}
                for policy in POLICIES:
                    model = TemporalExperimentModel(arm, mapping, config["dropout"])
                    model.load_state_dict(states[policy], strict=True)
                    frame, row = score(model, data, config, seed=seed, arm=arm, policy=policy, epoch=info["epochs"][policy])
                    weighted, common = validation_losses(model, data["validation"], config["training"]["batch_size"], weight)
                    row.update({"pos_weight": weight, "parameter_count": info["parameter_count"],
                                "validation_weighted_loss": weighted, "validation_unweighted_loss": common})
                    relative = f"models/{name}_{policy}.pt"
                    torch.save({"state_dict": states[policy], "arm": arm, "seed": seed, "mapping": mapping,
                        "dropout": config["dropout"], "policy": policy, "epoch": info["epochs"][policy],
                        "pos_weight": weight, "threshold": .5, "training_user_keys": info["training_user_keys"],
                        "phase2_manifest_sha256": config["phase2_manifest_sha256"]}, output / relative)
                    info["checkpoints"][policy] = relative
                    frame.to_csv(output / f"predictions/{name}_{policy}_validation.csv", index=False)
                    rows.append(row)
                runs[name], histories[name] = info, history
            paired = [v for v in runs.values() if v["seed"] == seed]
            require(len({v["batch_order_sha256"] for v in paired}) == 1, "Batch ordering differs across arms")
            require(all(v["epoch_order_sha256"] == paired[0]["epoch_order_sha256"] for v in paired), "Epoch ordering differs")
        table = pd.DataFrame(rows)
        summary, effects = summarize(table)
        for name, frame in (("validation_metrics", table), ("seed_summary", summary), ("paired_effects", effects)):
            frame.to_csv(output / f"{name}.csv", index=False)
        write_json(output / "training_metadata.json", runs)
        fig, axes = plt.subplots(3, 3, figsize=(13, 10))
        for r, arm in enumerate(config["arms"]):
            for c, seed in enumerate(config["seeds"]):
                name = f"{arm}_seed{seed}"
                h = histories[name]
                axes[r, c].plot(h.epoch, h.validation_unweighted_loss, label="Validation BCE")
                axes[r, c].axvline(runs[name]["epochs"][config["selection"]], color="black", linestyle=":")
                axes[r, c].set(title=f"{arm}, seed {seed}", xlabel="Epoch", ylabel="Unweighted validation loss")
        save_figure(fig, output / "figures/validation_curves.png")
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, metric in zip(axes, ("bot_f1", "bot_precision", "bot_recall")):
            for seed in config["seeds"]:
                f = table[table.seed.eq(seed) & table.policy.eq(config["selection"])].set_index("arm").loc[config["arms"]]
                ax.plot(range(3), f[metric], marker="o", label=f"seed {seed}")
            ax.set(xticks=range(3), xticklabels=["Flat masked", "Temporal branch", "Without 7 stats"], ylabel=metric, ylim=(0, 1))
            ax.legend()
        fig.suptitle("Validation only; fixed threshold 0.5; common checkpoint criterion")
        save_figure(fig, output / "figures/paired_arm_results.png")
        write_json(output / "provenance/protected_files.json", before)
        sources = ("src/temporal_statistics.py", "src/checkpoint_weight_experiment.py", "src/performance_diagnosis.py",
                   "src/baselines.py", "scripts/run_temporal_statistics.py")
        for source in sources:
            (output / "provenance" / Path(source).name).write_bytes((root / source).read_bytes())
        write_json(output / "provenance/environment.json", {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "pandas", "scikit-learn", "matplotlib")},
            "torch_threads": config["torch_threads"], "deterministic_algorithms": True,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
            "test_data_parsed": False, "threshold_tuned": False, "missingness_augmentation": False,
            "pairing": "Same seeds and batch orders; different architectures do not share initial tensors or dropout trajectories."})
    finally:
        require(before == preserved_hashes(root, config["output_root"]), "Protected prior files changed")
    artifacts = {p.relative_to(output).as_posix(): {"sha256": digest(p)} for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "manifest.json", {"schema_version": config["schema_version"], "complete": True, "promoted_model": None,
        "feature_mapping": mapping, "runs": runs, "artifacts": artifacts, "protected_files_verified": len(before),
        "limitations": ["Shared validation selection/reporting; no unbiased test estimate or significance claim.",
            "All training/validation temporal statistics observed; functional missingness handling is not evidence of natural missingness robustness.",
            "Imported temporal-statistic formulas and measurement windows remain unverified; no timeline or temporal Transformer is inferred.",
            "Near-matched parameter counts do not equate architectures or optimization; B versus A tests the complete branch design.",
            "B versus C changes both information and architecture; other features retain time-related signals.",
            "Historical Step-2 reference has a different architecture and shuffle schedule; it is descriptive only."]})
    return output, table, summary


def verify(root, output):
    root, output = Path(root).resolve(), Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    require(manifest["complete"] and manifest["promoted_model"] is None, "Invalid experiment manifest")
    for rel, meta in manifest["artifacts"].items():
        require(digest(output / rel) == meta["sha256"], "Artifact hash mismatch: " + rel)
    config = json.loads((output / "provenance/resolved_config.json").read_text())
    validate_config(config)
    seed_all(42, config["torch_threads"])
    reader = Inputs(root)
    data, mapping, reference = load_inputs(root, config, reader)
    require(mapping == manifest["feature_mapping"] == json.loads((output / "provenance/feature_mapping.json").read_text()), "Feature mapping differs")
    pd.testing.assert_frame_equal(reference.reset_index(drop=True), pd.read_csv(output / "historical_reference_validation.csv"), check_dtype=False, atol=1e-12, rtol=1e-12)
    weight = weights_from_training(data["train"]["y"])["full_train_ratio"]
    require(manifest["runs"] == json.loads((output / "training_metadata.json").read_text()), "Training metadata differs")
    rows = []
    for name, info in manifest["runs"].items():
        require(info["training_user_keys"] == data["train"]["users"].index.tolist() and info["pos_weight"] == weight, "Training identities or weight differ")
        h = pd.read_csv(output / f"histories/{name}.csv")
        require(h.epoch.tolist() == list(range(1, config["training"]["epochs"] + 1)), "History epochs differ")
        require(info["parameter_count"] == config["expected_parameters"][info["arm"]], "Parameter count differs")
        for policy, relative in info["checkpoints"].items():
            payload = torch.load(output / relative, map_location="cpu", weights_only=True)
            epoch = config["training"]["epochs"] if policy == "last" else int(h.loc[h.validation_unweighted_loss.idxmin(), "epoch"])
            require(payload["epoch"] == info["epochs"][policy] == epoch and payload["mapping"] == mapping and
                    payload["arm"] == info["arm"] and payload["seed"] == info["seed"] and payload["policy"] == policy and
                    payload["training_user_keys"] == info["training_user_keys"] and payload["pos_weight"] == weight and
                    payload["threshold"] == .5 and payload["dropout"] == config["dropout"] and
                    payload["phase2_manifest_sha256"] == config["phase2_manifest_sha256"], "Checkpoint contract differs")
            model = TemporalExperimentModel(info["arm"], mapping, config["dropout"])
            for key in ("other_indices", "temporal_indices"):
                torch.testing.assert_close(payload["state_dict"][key], getattr(model, key), rtol=0, atol=0)
            model.load_state_dict(payload["state_dict"], strict=True)
            require(sum(p.numel() for p in model.parameters()) == info["parameter_count"], "Reloaded architecture count differs")
            frame, row = score(model, data, config, seed=info["seed"], arm=info["arm"], policy=policy, epoch=epoch)
            saved = pd.read_csv(output / f"predictions/{name}_{policy}_validation.csv", dtype={"user_key": str})
            pd.testing.assert_frame_equal(frame, saved, check_dtype=False, atol=1e-7, rtol=1e-6)
            weighted, common = validation_losses(model, data["validation"], config["training"]["batch_size"], weight)
            rec = h.set_index("epoch").loc[epoch]
            np.testing.assert_allclose([weighted, common], [rec.validation_weighted_loss, rec.validation_unweighted_loss], atol=1e-7, rtol=1e-7)
            row.update({"pos_weight": weight, "parameter_count": info["parameter_count"], "validation_weighted_loss": weighted, "validation_unweighted_loss": common})
            rows.append(row)
    sort = ["seed", "arm", "policy"]
    table = pd.read_csv(output / "validation_metrics.csv")
    pd.testing.assert_frame_equal(pd.DataFrame(rows).sort_values(sort).reset_index(drop=True), table.sort_values(sort).reset_index(drop=True),
                                  check_dtype=False, atol=1e-7, rtol=1e-6)
    summary, effects = summarize(table)
    pd.testing.assert_frame_equal(summary, pd.read_csv(output / "seed_summary.csv"), check_dtype=False, atol=1e-10, rtol=1e-10)
    pd.testing.assert_frame_equal(effects, pd.read_csv(output / "paired_effects.csv"), check_dtype=False, atol=1e-10, rtol=1e-10)
    for seed in config["seeds"]:
        paired = [i for i in manifest["runs"].values() if i["seed"] == seed]
        require({i["arm"] for i in paired} == set(ARMS) and len(paired) == 3, "Missing/duplicate arm")
        require(len({i["batch_order_sha256"] for i in paired}) == 1, "Paired order hashes differ")
        # Reconstruct each actual epoch order using only the dedicated generator.
        loader = make_loader(data["train"], config["training"], seed, "train")
        orders, combined = [], hashlib.sha256()
        for _ in range(config["training"]["epochs"]):
            order = []
            for _, _, indices in loader:
                order.extend(indices.tolist()); combined.update(indices.numpy().tobytes())
            require(sorted(order) == list(range(len(data["train"]["y"]))), "Invalid reconstructed batch order")
            orders.append(hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes()).hexdigest())
        require(all(i["epoch_order_sha256"] == orders and i["batch_order_sha256"] == combined.hexdigest() for i in paired), "Shuffle schedule not reproducible")
    parsed = json.loads((output / "provenance/parsed_inputs.json").read_text())
    require(not any("test" in Path(p).stem.lower() for p in parsed), "Test input parsed")
    for path, sha in parsed.items():
        require(digest(root / path) == sha, "Parsed source changed")
    return {"training_runs": len(manifest["runs"]), "checkpoints_reloaded": len(rows), "validation_prediction_files": len(rows),
            "test_inputs_parsed": 0, "protected_files_at_execution": manifest["protected_files_verified"]}
