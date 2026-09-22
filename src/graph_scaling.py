"""One train-fitted graph input scalar; reuse the frozen architecture and trainer."""
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
from uuid import uuid4

import numpy as np
import pandas as pd
import torch

from src.gcn_integration import (load_inputs, initialized_model, train_one, evaluate, prediction_registry,
                               cohort_metrics, registered_path, POLICIES, validate_config as validate_source)
from src.graph_diagnosis import representations, auc_pairs
from src.checkpoint_weight_experiment import weights_from_training, state_hash
from src.performance_diagnosis import Inputs, digest, preserved_hashes, require, write_json, plt, save_figure


def fit_scale(train_graph):
    available = train_graph["available"]
    indices = train_graph["canonical_node_index"][available]
    require(len(indices) > 0 and (indices >= 0).all().item(), "Invalid scale-fit population")
    values = train_graph["data"].x[indices].double()
    require(values.shape[1] == 3 and torch.isfinite(values).all().item(), "Invalid graph channels")
    scalar = values.square().mean().sqrt().item()
    require(np.isfinite(scalar) and scalar > 0, "Invalid fitted scale")
    return {"scalar": scalar, "fit_user_keys": train_graph["users"].index[available.numpy()].tolist(),
            "fit_users": len(indices), "fit_channels": 3, "fit_dtype": "float64", "transform_dtype": "float32",
            "formula": "sqrt(mean(X_train_available ** 2))", "uses_labels": False,
            "fit_values_sha256": state_hash({"fit_values": values})}


def scale_graphs(graphs, scalar):
    require(np.isfinite(scalar) and scalar > 0, "Scale must be finite and positive")
    result = {}
    for split, graph in graphs.items():
        require(split in ("train", "validation"), "Only development graphs authorized")
        copied = {**graph, "data": graph["data"].clone()}
        copied["data"].x = graph["data"].x / scalar
        require(torch.isfinite(copied["data"].x).all().item(), "Nonfinite scaled input")
        for key in ("edge_index", "y", "eligible_mask"):
            torch.testing.assert_close(getattr(copied["data"], key), getattr(graph["data"], key), atol=0, rtol=0)
        result[split] = copied
    return result


def inputs(root, config, reader):
    path = root / config["source_manifest"]
    reader.allow(path, config["source_manifest_sha256"])
    source = reader.read(path)
    require(source["complete"] and source["schema_version"] == "gcn_integration_v1", "Invalid source experiment")
    source_config = reader.registered(path.parent, source, "provenance/resolved_config.json")
    validate_source(source_config)
    data, mapping, graphs, training_config, _, coverage, _ = load_inputs(root, source_config, reader)
    require(mapping == source["feature_mapping"], "Feature mapping mismatch")
    fitted = fit_scale(graphs["train"])
    require(fitted["fit_users"] == config["expected_fit_users"] == 193, "Scale-fit population changed")
    return path, source, source_config, data, mapping, graphs, scale_graphs(graphs, fitted["scalar"]), training_config, fitted, coverage


def diagnostics(model, data, graphs, identity):
    frames, rows = {}, []
    for split, pop in data.items():
        f = representations(model, pop, graphs[split])
        for k, value in {**identity, "split": split}.items():
            f[k] = value
        frames[split] = f
        covered = f[f.has_graph.eq(1)]
        graph_arm = identity["arm"] != "temporal_anchor"
        row = {**identity, "split": split, "covered_users": len(covered),
               "graph_zero_vectors": int(covered.graph_rms.eq(0).sum()) if graph_arm else None,
               "graph_rms_median": float(covered.graph_rms.median()) if graph_arm else None,
               "graph_effect_median": float(covered.graph_logit_effect.median()) if graph_arm else None}
        if graph_arm:
            denominator = covered.other_weighted_rms + covered.temporal_weighted_rms
            require(denominator.gt(0).all(), "Undefined weighted magnitude ratio")
            row["weighted_graph_ratio_median"] = float((covered.graph_weighted_rms / denominator).median())
        else:
            row["weighted_graph_ratio_median"] = None
        rows.append(row)
    return frames, rows


def summaries(table):
    metrics = ["bot_f1", "bot_precision", "bot_recall", "roc_auc", "average_precision", "brier"]
    summary = table.groupby(["arm", "policy", "cohort"])[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    effects = []
    comparisons = [("scaled_gcn", "gcn"), ("scaled_degree_mlp", "degree_mlp"),
                   ("scaled_gcn", "scaled_degree_mlp"), ("scaled_gcn", "temporal_anchor")]
    for treatment, control in comparisons:
        for row in table[table.arm.eq(treatment)].itertuples():
            ref = table[table.arm.eq(control) & table.seed.eq(row.seed) & table.policy.eq(row.policy) & table.cohort.eq(row.cohort)]
            require(len(ref) == 1, "Missing or duplicate matched control")
            effects.append({"comparison": treatment + "_minus_" + control, "seed": row.seed, "policy": row.policy,
                            "cohort": row.cohort, **{"delta_" + m: float(getattr(row, m) - ref.iloc[0][m]) for m in metrics}})
    return summary.reset_index(), pd.DataFrame(effects)


def scored_rows(frame, parameter_count, weighted, common):
    rows = cohort_metrics(frame)
    for row in rows:
        group = frame if row["cohort"] == "all" else frame[frame.has_graph.eq(int(row["cohort"] == "graph_available"))]
        row.update({"parameter_count": parameter_count, "validation_weighted_loss": weighted,
                    "validation_unweighted_loss": common,
                    "brier": float(np.mean((group.bot_probability.to_numpy() - group.label_binary.to_numpy()) ** 2))})
    return rows


def validate_config(config):
    require(config["arms"] == ["degree_mlp", "gcn"] and config["seeds"] == [42, 43, 44] and config["threshold"] == .5,
            "Approved arms/seeds/threshold changed")
    require(config["scale_rule"] == "sqrt_mean_squared_three_channels_training_graph_available_canonical_users"
            and config["scope"] == "graph_input_scalar_only_no_other_feature_changes_no_test_no_threshold_tuning_no_promotion",
            "Approved scale or scope changed")


def run(root, config_path="configs/graph_scaling_v1.json"):
    root = Path(root).resolve()
    reader = Inputs(root)
    reader.allow(root / config_path, digest(root / config_path))
    config = reader.read(root / config_path)
    validate_config(config)
    before = preserved_hashes(root, config["output_root"])
    output = root / config["output_root"] / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    for name in ("models", "histories", "predictions", "diagnostics", "provenance", "figures"):
        (output / name).mkdir()
    try:
        source_path, source, source_config, data, mapping, raw, scaled, training, fitted, coverage = inputs(root, config, reader)
        weight = weights_from_training(data["train"]["y"])["full_train_ratio"]
        spec = training["training"]
        print(f"Training-only graph RMS scale: {fitted['scalar']:.12f}; fit users: {fitted['fit_users']}", flush=True)
        write_json(output / "provenance/resolved_config.json", config)
        write_json(output / "provenance/training_config.json", training)
        write_json(output / "provenance/feature_mapping.json", mapping)
        write_json(output / "provenance/fitted_graph_scale.json", fitted)
        coverage.to_csv(output / "graph_coverage.csv", index=False)
        for split in data:
            data[split]["users"][["label_binary"]].join(data[split]["masks"]).to_csv(output / f"provenance/{split}_population.csv")
            # Store versioned derived node features only; original graph tensors are untouched.
            torch.save({"x": scaled[split]["data"].x, "scalar": fitted["scalar"],
                        "node_keys": raw[split]["mapping"].user_key.tolist()}, output / f"provenance/{split}_scaled_node_features.pt")
        rows, runs, branch_rows, pair_rows = [], {}, [], []
        def record(model, graphs, arm, base_arm, seed, policy, epoch):
            p, weighted, common = evaluate(model, data["validation"], graphs["validation"], spec, weight)
            frame = prediction_registry(data["validation"], p, arm=arm, seed=seed, policy=policy, epoch=epoch)
            frame.to_csv(output / f"predictions/{arm}_seed{seed}_{policy}_validation.csv", index=False)
            rows.extend(scored_rows(frame, sum(p.numel() for p in model.parameters()), weighted, common))
            keyed, branch = diagnostics(model, data, graphs, {"arm": arm, "seed": seed, "policy": policy, "epoch": epoch})
            for split, values in keyed.items():
                values.to_csv(output / f"diagnostics/{arm}_seed{seed}_{policy}_{split}.csv", index=False)
                pair_rows.extend({"arm": arm, "seed": seed, "policy": policy, "epoch": epoch, "split": split, **r}
                                 for r in auc_pairs(frame if split == "validation" else values))
            branch_rows.extend(branch)
            return frame
        # Re-score frozen raw controls; never retrain or overwrite their checkpoints.
        for name, info in source["runs"].items():
            for policy in POLICIES:
                payload = torch.load(registered_path(reader, source_path.parent, source, info["checkpoints"][policy]), map_location="cpu", weights_only=True)
                model = initialized_model(info["arm"], mapping, info["seed"])
                model.load_state_dict(payload["state_dict"], strict=True)
                f = record(model, raw, info["arm"], info["arm"], info["seed"], policy, info["epochs"][policy])
                saved = reader.registered(source_path.parent, source, f"predictions/{name}_{policy}_validation.csv")
                pd.testing.assert_frame_equal(f, saved, check_dtype=False, atol=1e-7, rtol=1e-6)
        for seed in config["seeds"]:
            for arm in config["arms"]:
                print(f"Training scaled_{arm}, seed {seed}", flush=True)
                states, history, info = train_one(data, scaled, mapping, spec, arm, seed, weight, training["torch_threads"])
                reference = source["runs"][f"{arm}_seed{seed}"]
                for key in ("initial_normalized_sha256", "epoch_order_sha256", "parameter_count", "pos_weight"):
                    require(info[key] == reference[key], "Raw/scaled pairing changed: " + key)
                name = f"scaled_{arm}_seed{seed}"
                history.to_csv(output / f"histories/{name}.csv", index=False)
                info.update({"training_user_keys": data["train"]["users"].index.tolist(), "scalar": fitted["scalar"], "checkpoints": {}})
                for policy in POLICIES:
                    model = initialized_model(arm, mapping, seed)
                    model.load_state_dict(states[policy], strict=True)
                    record(model, scaled, "scaled_" + arm, arm, seed, policy, info["epochs"][policy])
                    relative = f"models/{name}_{policy}.pt"
                    torch.save({"state_dict": states[policy], "arm": arm, "seed": seed, "policy": policy,
                        "epoch": info["epochs"][policy], "mapping": mapping, "fitted_graph_scale": fitted,
                        "pos_weight": weight, "threshold": .5, "training_user_keys": info["training_user_keys"],
                        "source_manifest_sha256": config["source_manifest_sha256"]}, output / relative)
                    info["checkpoints"][policy] = relative
                runs[name] = info
        table = pd.DataFrame(rows)
        summary, effects = summaries(table)
        for name, frame in (("validation_metrics", table), ("seed_summary", summary), ("paired_effects", effects),
                            ("branch_diagnostics", pd.DataFrame(branch_rows)), ("auc_pair_decomposition", pd.DataFrame(pair_rows))):
            frame.to_csv(output / (name + ".csv"), index=False)
        write_json(output / "training_metadata.json", runs)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        order = ["temporal_anchor", "degree_mlp", "scaled_degree_mlp", "gcn", "scaled_gcn"]
        for ax, cohort in zip(axes, ("all", "graph_available", "graph_unavailable")):
            for seed in config["seeds"]:
                f = table[table.seed.eq(seed) & table.policy.eq(POLICIES[0]) & table.cohort.eq(cohort)].set_index("arm").loc[order]
                ax.plot(range(5), f.bot_f1, marker="o", label=f"seed {seed}")
            ax.set(xticks=range(5), xticklabels=["Temporal", "Raw MLP", "Scaled MLP", "Raw GCN", "Scaled GCN"], ylabel="Bot F1", title=cohort, ylim=(0, 1))
            ax.tick_params(axis="x", labelrotation=30); ax.legend()
        save_figure(fig, output / "figures/paired_f1.png")
        write_json(output / "provenance/parsed_inputs.json", reader.accessed)
        write_json(output / "provenance/protected_files.json", before)
        for file in ("src/graph_scaling.py", "src/gcn_integration.py", "src/graph_diagnosis.py", "src/temporal_statistics.py",
                     "src/checkpoint_weight_experiment.py", "src/graph_baseline.py", "src/baselines.py", "src/performance_diagnosis.py", "scripts/run_graph_scaling.py"):
            (output / "provenance" / Path(file).name).write_bytes((root / file).read_bytes())
        write_json(output / "provenance/environment.json", {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "pandas": pd.__version__, "threads": torch.get_num_threads(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
            "test_labels_features_graphs_predictions_parsed": False, "assignment_projection": ["user_key", "split"],
            "threshold_tuned": False, "only_intervention": "one training-derived graph input scale"})
    finally:
        require(before == preserved_hashes(root, config["output_root"]), "Protected files changed")
    write_json(output / "manifest.json", {"schema_version": config["schema_version"], "complete": True, "promoted_model": None,
        "runs": runs, "fitted_graph_scale": fitted, "source_manifest_sha256": config["source_manifest_sha256"],
        "protected_files_verified": len(before), "artifacts": {p.relative_to(output).as_posix(): {"sha256": digest(p)} for p in sorted(output.rglob("*")) if p.is_file()}})
    return output, table, summary


def verify(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    require(manifest["complete"] and manifest["promoted_model"] is None, "Incomplete experiment")
    for rel, meta in manifest["artifacts"].items():
        require(digest(output / rel) == meta["sha256"], "Output hash mismatch: " + rel)
    config = json.loads((output / "provenance/resolved_config.json").read_text())
    validate_config(config)
    reader = Inputs(root)
    source_path, source, _, data, mapping, raw, scaled, training, fitted, coverage = inputs(root, config, reader)
    require(fitted == manifest["fitted_graph_scale"] == json.loads((output / "provenance/fitted_graph_scale.json").read_text()), "Scale mismatch")
    weight = weights_from_training(data["train"]["y"])["full_train_ratio"]
    require(manifest["runs"] == json.loads((output / "training_metadata.json").read_text()) and len(manifest["runs"]) == 6, "Training metadata mismatch")
    for split in data:
        payload = torch.load(output / f"provenance/{split}_scaled_node_features.pt", map_location="cpu", weights_only=True)
        require(payload["scalar"] == fitted["scalar"] and payload["node_keys"] == raw[split]["mapping"].user_key.tolist(), "Scaled node mapping mismatch")
        torch.testing.assert_close(payload["x"], scaled[split]["data"].x, atol=0, rtol=0)
    rows, branch_rows, pair_rows, count = [], [], [], 0
    for condition, runs, base, graph_set in (("raw", source["runs"], source_path.parent, raw), ("scaled", manifest["runs"], output, scaled)):
        for name, info in runs.items():
            arm = info["arm"] if condition == "raw" else "scaled_" + info["arm"]
            history = None
            if condition == "scaled":
                history = pd.read_csv(output / f"histories/{name}.csv")
                require(history.epoch.tolist() == list(range(1, training["training"]["epochs"] + 1)), "History incomplete")
                ref = source["runs"][f"{info['arm']}_seed{info['seed']}"]
                for key in ("initial_normalized_sha256", "epoch_order_sha256", "parameter_count", "pos_weight"):
                    require(info[key] == ref[key], "Pairing differs: " + key)
                require(info["training_user_keys"] == data["train"]["users"].index.tolist(), "Training keys differ")
            for policy, relative in info["checkpoints"].items():
                path = registered_path(reader, source_path.parent, source, relative) if condition == "raw" else base / relative
                payload = torch.load(path, map_location="cpu", weights_only=True)
                epoch = info["epochs"][policy]
                if condition == "scaled":
                    expected_epoch = training["training"]["epochs"] if policy == "last" else int(history.loc[history.validation_unweighted_loss.idxmin(), "epoch"])
                    require(epoch == expected_epoch and payload["fitted_graph_scale"] == fitted and payload["mapping"] == mapping
                            and payload["source_manifest_sha256"] == config["source_manifest_sha256"]
                            and payload["training_user_keys"] == info["training_user_keys"], "Checkpoint provenance differs")
                require(payload["arm"] == info["arm"] and payload["seed"] == info["seed"] and payload["policy"] == policy
                        and payload["epoch"] == epoch and payload["threshold"] == .5 and payload["pos_weight"] == weight, "Checkpoint policy differs")
                model = initialized_model(info["arm"], mapping, info["seed"])
                model.load_state_dict(payload["state_dict"], strict=True)
                p, weighted, common = evaluate(model, data["validation"], graph_set["validation"], training["training"], weight)
                f = prediction_registry(data["validation"], p, arm=arm, seed=info["seed"], policy=policy, epoch=epoch)
                saved = pd.read_csv(output / f"predictions/{arm}_seed{info['seed']}_{policy}_validation.csv", dtype={"user_key": str})
                pd.testing.assert_frame_equal(f, saved, check_dtype=False, atol=1e-7, rtol=1e-6)
                rows.extend(scored_rows(f, sum(p.numel() for p in model.parameters()), weighted, common))
                if history is not None:
                    rec = history.set_index("epoch").loc[epoch]
                    np.testing.assert_allclose([weighted, common], [rec.validation_weighted_loss, rec.validation_unweighted_loss], atol=1e-7, rtol=1e-7)
                keyed, branch = diagnostics(model, data, graph_set, {"arm": arm, "seed": info["seed"], "policy": policy, "epoch": epoch})
                branch_rows.extend(branch)
                for split, values in keyed.items():
                    saved = pd.read_csv(output / f"diagnostics/{arm}_seed{info['seed']}_{policy}_{split}.csv", dtype={"user_key": str, "zero_graph_prediction_flip": "boolean"}, float_precision="round_trip")
                    pd.testing.assert_frame_equal(values, saved, check_dtype=False, atol=1e-7, rtol=1e-6)
                    pair_rows.extend({"arm": arm, "seed": info["seed"], "policy": policy, "epoch": epoch, "split": split, **r}
                                     for r in auc_pairs(f if split == "validation" else values))
                count += 1
    table = pd.read_csv(output / "validation_metrics.csv")
    keys = ["arm", "seed", "policy", "cohort"]
    pd.testing.assert_frame_equal(pd.DataFrame(rows).sort_values(keys).reset_index(drop=True), table.sort_values(keys).reset_index(drop=True), check_dtype=False, atol=1e-7, rtol=1e-6)
    for name, expected in zip(("seed_summary", "paired_effects"), summaries(table)):
        pd.testing.assert_frame_equal(expected, pd.read_csv(output / (name + ".csv")), check_dtype=False, atol=1e-10, rtol=1e-10)
    pd.testing.assert_frame_equal(pd.DataFrame(branch_rows), pd.read_csv(output / "branch_diagnostics.csv"), check_dtype=False, atol=1e-7, rtol=1e-6)
    pd.testing.assert_frame_equal(pd.DataFrame(pair_rows), pd.read_csv(output / "auc_pair_decomposition.csv"), check_dtype=False, atol=1e-10, rtol=1e-10)
    parsed = json.loads((output / "provenance/parsed_inputs.json").read_text())
    require(not any("test" in Path(p).stem.lower() for p in parsed) and all(digest(root / p) == sha for p, sha in parsed.items()), "Parsed inputs changed or test artifact parsed")
    return {"new_training_runs": 6, "scaled_checkpoints_reloaded": 12, "frozen_checkpoints_reloaded": count - 12,
            "validation_prediction_files": count, "keyed_diagnostic_files": count * 2, "test_inputs_parsed": 0,
            "fit_users": fitted["fit_users"], "scalar": fitted["scalar"], "protected_files_at_execution": manifest["protected_files_verified"]}
