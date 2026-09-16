"""Notebook-05 compatibility experiment: original 43-input GCN on permitted graphs.

Support tabular blocks are unpopulated slots, not measured zero-valued metadata.
Only same-split gold tables can populate those slots; attachment indicators are
provenance and never become model inputs. Phase-2/3/4 modules remain unchanged.
"""
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import platform
from pathlib import Path
import subprocess
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from src.baselines import (SPLITS, align, artifact_inventory, checked_artifact,
                           evaluate_registry, keyed, load_phase2, notebook_source_hash,
                           read_json, require, seed_all, sha256, write_json)
from src.graph_baseline import (graph_metrics, graph_predictions, load_graph_inputs,
                                training_class_weights)


def validate_policy(config):
    """Fail rather than silently relaxing the user's approved experimental conditions."""
    require(config["architecture"] == {"input_dim": 43, "hidden_dim": 64, "second_dim": 32}, "Architecture change requires approval")
    require(config["training"] == {"epochs": 200, "lr": .001, "weight_decay": .0001,
            "selection": "best_validation_loss", "threshold": .5}, "Hyperparameter change requires approval")
    require(config["tabular_attachment"] == "same_split_gold_only"
            and config["support_tabular_policy"] == "zero_unpopulated_slots_not_observations"
            and config["indicator_in_model_input"] is False, "Attachment policy change requires approval")
    require(config["initialization"] == "from_scratch" and config["device"] == "cpu", "Unsupported initialization/device")
    require(config["comparison_type"] == "descriptive_not_controlled_ablation"
            and config["comparison_models"] == ["tabular_lr", "tabular_mlp", "gcn_degree_only", "gcn_tabular"]
            and config["expected_comparison_counts"] == {"validation": 42, "test": 41}, "Comparison policy change requires approval")


def attach_tabular(graph, tabular, feature_order, split):
    """Keyed, split-local assignment; no other population's features accepted here."""
    require(len(feature_order) == 40 and len(set(feature_order)) == 40, "Expected manifest's 40 unique features")
    require(not set(feature_order) & {"id", "user_key", "label_binary", "screen_name", "tabular_attached"}, "Identity/target/indicator in feature order")
    require(tabular.index.is_unique and tabular.index.notna().all(), "Invalid tabular identity index")
    tabular = align(tabular, graph["users"], f"{split} tabular keys")
    values = tabular.loc[:, feature_order].to_numpy(dtype=np.float32)
    require(np.isfinite(values).all(), "Invalid Phase-2 values; no refitting or additional imputation allowed")
    mapping = graph["mapping"].copy()
    require(mapping.user_key.is_unique and mapping.node_index.tolist() == list(range(len(mapping))), "Invalid node mapping")
    data = graph["data"]
    require(data.x.shape == (len(mapping), 3), "Graph source must supply exactly three degree features")
    positions = tabular.index.get_indexer(mapping.user_key)
    attached = positions >= 0
    block = np.zeros((len(mapping), 40), dtype=np.float32)
    block[attached] = values[positions[attached]]
    result = data.clone()
    result.x = torch.cat([data.x, torch.from_numpy(block)], dim=1)
    result.tabular_attached = torch.tensor(attached, dtype=torch.bool)
    require(result.x.shape[1] == 43, "Indicators must not enter model input")
    require(not result.eligible_mask[~result.tabular_attached].any().item(), "Eligible target has no same-split tabular row")
    provenance = mapping.copy()
    provenance["split"] = split
    provenance["graph_node_member"] = True
    provenance["is_same_split_gold"] = attached
    provenance["has_graph"] = mapping.user_key.map(graph["masks"].has_graph).astype("Int64")
    provenance["supervised_eligible"] = data.eligible_mask.numpy()
    provenance["tabular_attached"] = attached.astype(int)
    provenance["tabular_provenance"] = np.where(attached, "phase2_same_split_gold", "non_gold_support_unpopulated_slots")
    provenance["tabular_source_artifact"] = np.where(attached, f"splits/tabular/{split}_tabular.csv", "")
    return {**graph, "data": result, "node_provenance": provenance}


def pinned_manifest(root, config, prefix):
    path = Path(root) / config[prefix + "_manifest"]
    require(sha256(path) == config[prefix + "_manifest_sha256"], f"{prefix} manifest hash mismatch")
    manifest = read_json(path)
    require(manifest["complete"] and manifest["phase2_manifest_sha256"] == config["phase2_manifest_sha256"],
            f"{prefix} manifest uses different Phase-2 inputs")
    return path, manifest


def load_inputs(root, config):
    validate_policy(config)
    phase2 = load_phase2(root, config)  # Validates train-only fitted objects; never fits them.
    expected, _, coverage, overlap = load_graph_inputs(root, config)
    graph_path, graph_manifest = pinned_manifest(root, config, "graph")
    baseline_path, baseline_manifest = pinned_manifest(root, config, "baseline")
    require(graph_manifest["eligibility"] == "has_graph" and graph_manifest["graph_protocol"] == phase2["manifest"]["graph_protocol"], "Graph handoff protocol differs")
    graphs = {}
    for split in SPLITS:
        artifact = graph_manifest["graph_artifacts"][split]
        mapping = pd.read_csv(checked_artifact(graph_path.parent, graph_manifest, artifact["nodes"]), dtype={"user_key": str})
        tensors = torch.load(checked_artifact(graph_path.parent, graph_manifest, artifact["tensors"]), map_location="cpu", weights_only=True)
        require(set(tensors) == {"x", "edge_index", "y", "eligible_mask"}, "Unexpected graph tensor schema")
        pd.testing.assert_frame_equal(mapping, expected[split]["mapping"], check_dtype=False)
        for name, tensor in tensors.items():
            torch.testing.assert_close(tensor, expected[split]["data"][name], rtol=0, atol=0)
        # Consume saved Notebook-04 tensors, after checking them against Phase-2.
        source = {**expected[split], "mapping": mapping, "data": Data(**tensors)}
        pop = phase2["populations"][split]
        require(list(pop["users"].index) == list(source["users"].index), "Graph/tabular canonical order mismatch")
        tabular = pd.DataFrame(pop["x"], index=pop["users"].index, columns=phase2["feature_order"])
        graphs[split] = attach_tabular(source, tabular, phase2["feature_order"], split)
    coverage = coverage.copy()
    coverage["tabular_attached_nodes"] = [int(graphs[s]["data"].tabular_attached.sum()) for s in coverage.split]
    coverage["unpopulated_support_blocks"] = coverage.nodes - coverage.tabular_attached_nodes
    return graphs, phase2, graph_manifest, baseline_manifest, coverage, overlap


class GCNTabularClassifier(nn.Module):
    """Original Notebook-05 topology: no dropout, batch normalization or fusion branch."""
    def __init__(self, input_dim=43, hidden_dim=64, second_dim=32):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, cached=False, normalize=True, add_self_loops=True)
        self.conv2 = GCNConv(hidden_dim, second_dim, cached=False, normalize=True, add_self_loops=True)
        self.classifier = nn.Linear(second_dim, 2)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.classifier(F.relu(self.conv2(x, edge_index)))


def train_model(train_graph, validation_graph, config):
    """Initialize from scratch; only train labels define loss weights. No test argument."""
    seed_all(config["seed"], config["torch_threads"])
    model = GCNTabularClassifier(**config["architecture"])
    weights, counts = training_class_weights(train_graph)
    require(validation_graph.eligible_mask.any().item(), "Empty validation cohort")
    criterion = nn.CrossEntropyLoss(weight=weights)
    spec = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"])
    history, best_state, best_epoch, best_loss = [], None, None, float("inf")
    for epoch in range(1, spec["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_graph.x, train_graph.edge_index)
        loss = criterion(logits[train_graph.eligible_mask], train_graph.y[train_graph.eligible_mask])
        require(torch.isfinite(loss).item(), "Nonfinite training loss")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(validation_graph.x, validation_graph.edge_index)
            val_loss = criterion(val_logits[validation_graph.eligible_mask], validation_graph.y[validation_graph.eligible_mask]).item()
        require(np.isfinite(val_loss), "Nonfinite validation loss")
        history.append({"epoch": epoch, "train_loss": loss.item(), "validation_loss": val_loss})
        if val_loss < best_loss:
            best_state, best_epoch, best_loss = deepcopy(model.state_dict()), epoch, val_loss
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{spec['epochs']}: train={loss.item():.6f}, validation={val_loss:.6f}", flush=True)
    return {"best_validation_loss": best_state, "last": deepcopy(model.state_dict())}, pd.DataFrame(history), {
        "best_epoch": best_epoch, "last_epoch": spec["epochs"], "selection": spec["selection"],
        "class_counts": counts.tolist(), "class_weights": weights.tolist(), "initialization": "from_scratch"}


class _GraphCall:
    """Adapt only the call signature to the shared keyed prediction helper."""
    def __init__(self, model):
        self.model = model

    def eval(self):
        self.model.eval()

    def __call__(self, data):
        return self.model(data.x, data.edge_index)


def predictions(model, graph, split, threshold):
    registry = graph_predictions(_GraphCall(model), graph, split, threshold)
    registry["model"] = "gcn_tabular"
    provenance = graph["node_provenance"].set_index("user_key")
    registry["supervised_eligible"] = registry.user_key.map(provenance.supervised_eligible).fillna(False).astype(bool)
    registry["tabular_attached"] = registry.user_key.map(provenance.tabular_attached).fillna(0).astype(int)
    registry["tabular_provenance"] = registry.user_key.map(provenance.tabular_provenance).fillna("canonical_tabular_available_but_no_graph_node")
    require(registry.supervised_eligible.eq(registry.has_graph.eq(1)).all(), "Eligibility differs from declared mask")
    return registry


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = GCNTabularClassifier(**payload["architecture"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval(), payload


def descriptive_comparison(root, config, primary):
    graph_path, graph_manifest = pinned_manifest(root, config, "graph")
    baseline_path, baseline_manifest = pinned_manifest(root, config, "baseline")
    rows, key_rows = [], []
    for split in ("validation", "test"):
        reference = keyed(primary[split], "Graph+Tabular predictions")
        keys = sorted(reference.index[reference.has_graph.eq(1)])
        require(len(keys) == config["expected_comparison_counts"][split], "Required matched cohort changed; approval needed")
        candidates = {"gcn_tabular": reference}
        for name in ("tabular_lr", "tabular_mlp"):
            selected = baseline_manifest["models"][name]["selected"]
            relative = f"predictions/{name}_{selected}_{split}.csv"
            candidates[name] = keyed(pd.read_csv(checked_artifact(baseline_path.parent, baseline_manifest, relative), dtype={"user_key": str}), name)
        variant = graph_manifest["training"]["selection"]
        relative = f"predictions/gcn_{variant}_{split}.csv"
        candidates["gcn_degree_only"] = keyed(pd.read_csv(checked_artifact(graph_path.parent, graph_manifest, relative), dtype={"user_key": str}), "Graph-only")
        for name in config["comparison_models"]:
            frame = align(candidates[name], reference, "comparison population")
            require(frame.label_binary.eq(reference.label_binary).all() and frame.has_graph.eq(reference.has_graph).all(), "Comparison labels/masks differ")
            # Fail if a selected user is unsupported; do not silently shrink the cohort.
            require(frame.loc[keys, "status"].eq("predicted").all()
                    and frame.loc[keys, ["bot_probability", "prediction"]].notna().all().all(), "Requested matched user lacks predictions")
            metric, _ = evaluate_registry(frame.loc[keys].reset_index(), "all_users")
            metric["cohort"] = "phase2_graph_available_exact_keys"
            rows.append({"model": name, "split": split, "comparison_type": config["comparison_type"], **metric})
        key_rows.extend({"split": split, "user_key": k} for k in keys)
    return pd.DataFrame(rows), pd.DataFrame(key_rows)


class GraphTabularRun:
    def __init__(self, root, config_path="configs/graph_tabular_v2.json"):
        self.root = Path(root).resolve()
        self.config = read_json(self.root / config_path)
        (self.graphs, self.phase2, self.graph_manifest, self.baseline_manifest,
         self.coverage, self.support_overlap) = load_inputs(self.root, self.config)
        seed_all(self.config["seed"], self.config["torch_threads"])
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.output_dir = self.root / self.config["output_root"] / (stamp + "_" + uuid4().hex[:8])
        self.output_dir.mkdir(parents=True, exist_ok=False)
        for folder in ("graphs", "models", "histories", "predictions", "reports", "provenance"):
            (self.output_dir / folder).mkdir()
        self.coverage.to_csv(self.output_dir / "reports/coverage.csv", index=False)
        self.support_overlap.to_csv(self.output_dir / "reports/shared_support.csv", index=False)
        for split, graph in self.graphs.items():
            graph["node_provenance"].to_csv(self.output_dir / f"graphs/{split}_nodes.csv", index=False)
            torch.save({name: graph["data"][name] for name in ("x", "edge_index", "y", "eligible_mask", "tabular_attached")},
                       self.output_dir / f"graphs/{split}_tensors.pt")
            self.phase2["populations"][split]["observed"].to_csv(self.output_dir / f"provenance/{split}_observation_masks.csv")
        write_json(self.output_dir / "provenance/resolved_config.json", self.config)
        for name, manifest in (("phase2", self.phase2["manifest"]), ("graph", self.graph_manifest), ("baseline", self.baseline_manifest)):
            write_json(self.output_dir / f"provenance/{name}_manifest.json", manifest)
        self.feature_order = self.graph_manifest["degree_feature_order"] + self.phase2["feature_order"]
        require(len(self.feature_order) == 43, "Input contract is not 43 features")
        write_json(self.output_dir / "provenance/model_feature_order.json", self.feature_order)
        sources = ["src/graph_tabular.py", "src/graph_baseline.py", "src/baselines.py"]
        for source in sources:
            (self.output_dir / "provenance" / Path(source).name).write_bytes((self.root / source).read_bytes())
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.root, text=True).strip()
            status = subprocess.check_output(["git", "status", "--porcelain"], cwd=self.root, text=True)
        except (OSError, subprocess.CalledProcessError):
            commit, status = None, "unavailable"
        self.provenance = {"created_utc": stamp, "python": platform.python_version(), "platform": platform.platform(),
            "packages": {n: importlib.metadata.version(n) for n in ("torch", "torch-geometric", "numpy", "pandas", "scikit-learn")},
            "seed": self.config["seed"], "device": self.config["device"], "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": True, "source_sha256": {s: sha256(self.root / s) for s in sources},
            "config_sha256": sha256(self.root / config_path), "git_commit_at_execution": commit, "git_status_at_execution": status,
            "notebook_source_sha256": notebook_source_hash(self.root / "analysis/05_graph_tabular.ipynb"),
            "feature_order": self.feature_order,
            "limitations": ["Support zero slots are intentionally unpopulated, not observed metadata; indicators remain outside x.",
                "Known non-gold support metadata is deliberately unused to preserve the original baseline feature policy.",
                "Same-split node-only gold users carry tabular context, but has_graph availability alone defines supervised eligibility.",
                "Separate gold cohorts share support identities; this is not fully node-disjoint, and inference is joint within each cohort.",
                "Degree features remain redundant [d,d,2d], unscaled, with reverse-edge multiplicity preserved.",
                "Phase-2 transformations are unchanged and fitted on all 671 training users, not refitted on the 193 graph-eligible subset.",
                "The degree-only model has different dropout/optimizer settings; tabular baselines have a larger training population.",
                "Matched comparisons are descriptive, not controlled ablations or evidence isolating the causal effect of adding tabular inputs.",
                "Historical test data was already inspected; the supported test cohort contains only six bots. One seed does not quantify uncertainty."]}
        write_json(self.output_dir / "provenance/run.json", self.provenance)
        self.info, self.complete = None, False

    def fit(self):
        require(self.info is None, "Already fitted; start a new run rather than overwrite")
        self.states, self.history, self.info = train_model(self.graphs["train"]["data"], self.graphs["validation"]["data"], self.config)
        self.history.to_csv(self.output_dir / "histories/gcn_tabular.csv", index=False)
        self.info["training_user_keys"] = sorted(self.graphs["train"]["masks"].index[self.graphs["train"]["masks"].has_graph.eq(1)])
        self.info["checkpoints"] = {}
        for variant, state in self.states.items():
            relative = f"models/gcn_tabular_{variant}.pt"
            epoch = self.info["best_epoch"] if variant == "best_validation_loss" else self.info["last_epoch"]
            torch.save({"state_dict": state, "architecture": self.config["architecture"], "variant": variant, "epoch": epoch,
                        "feature_order": self.feature_order, "class_weights": self.info["class_weights"],
                        "phase2_manifest_sha256": self.config["phase2_manifest_sha256"],
                        "graph_manifest_sha256": self.config["graph_manifest_sha256"]}, self.output_dir / relative)
            self.info["checkpoints"][variant] = relative
        write_json(self.output_dir / "models/training_metadata.json", self.info)
        return self.history

    def evaluate_and_export(self):
        require(self.info is not None and not self.complete, "Fit before exporting; never overwrite a complete run")
        rows, details, reload_checks, primary = [], {}, [], {}
        for variant, relative in self.info["checkpoints"].items():
            restored, _ = load_checkpoint(self.output_dir / relative)
            original = GCNTabularClassifier(**self.config["architecture"])
            original.load_state_dict(self.states[variant])
            for split, graph in self.graphs.items():
                frame = predictions(restored, graph, split, self.config["training"]["threshold"])
                pd.testing.assert_frame_equal(frame, predictions(original, graph, split, self.config["training"]["threshold"]), check_exact=True)
                reload_checks.append({"checkpoint": variant, "split": split, "exact_reload_match": True})
                frame["checkpoint_variant"] = variant
                frame["primary_checkpoint"] = variant == self.info["selection"]
                frame.to_csv(self.output_dir / f"predictions/gcn_tabular_{variant}_{split}.csv", index=False)
                if variant == self.info["selection"]:
                    primary[split] = frame
                for cohort in ("all_users", "graph_available"):
                    metric, detail = graph_metrics(frame, cohort)
                    rows.append({"model": "gcn_tabular", "checkpoint_variant": variant,
                                 "primary_checkpoint": variant == self.info["selection"], "split": split, **metric})
                    details[f"{variant}/{split}/{cohort}"] = detail
        self.comparison, keys = descriptive_comparison(self.root, self.config, primary)
        self.comparison.to_csv(self.output_dir / "reports/descriptive_matched_comparison.csv", index=False)
        keys.to_csv(self.output_dir / "reports/matched_user_keys.csv", index=False)
        metrics = pd.DataFrame(rows)
        metrics.to_csv(self.output_dir / "metrics.csv", index=False)
        write_json(self.output_dir / "reports/classification_reports.json", details)
        write_json(self.output_dir / "reports/checkpoint_reload.json", reload_checks)
        write_json(self.output_dir / "manifest.json", {"schema_version": "graph_tabular_v2", "complete": True,
            "identity_key": "user_key", "phase2_manifest_sha256": self.config["phase2_manifest_sha256"],
            "graph_manifest_sha256": self.config["graph_manifest_sha256"], "baseline_manifest_sha256": self.config["baseline_manifest_sha256"],
            "feature_order": self.feature_order, "eligibility": "has_graph", "training": self.info,
            "tabular_attachment": self.config["tabular_attachment"], "support_tabular_policy": self.config["support_tabular_policy"],
            "comparison_type": self.config["comparison_type"],
            "graph_artifacts": {s: {"nodes": f"graphs/{s}_nodes.csv", "tensors": f"graphs/{s}_tensors.pt"} for s in SPLITS},
            "downstream_contract": "Keep split graphs separate. x has exactly 3 degree + 40 Phase-2 tabular columns; tabular_attached is sidecar provenance only. Unsupported canonical users retain null predictions.",
            "artifacts": artifact_inventory(self.output_dir)})
        self.complete = True
        return metrics


def verify_run(root, output_dir):
    directory = Path(output_dir)
    manifest = read_json(directory / "manifest.json")
    require(manifest["complete"], "Incomplete run")
    for relative in manifest["artifacts"]:
        checked_artifact(directory, manifest, relative)
    config = read_json(directory / "provenance/resolved_config.json")
    seed_all(config["seed"], config["torch_threads"])
    graphs, phase2, graph_manifest, _, coverage, overlap = load_inputs(root, config)
    feature_order = graph_manifest["degree_feature_order"] + phase2["feature_order"]
    require(manifest["feature_order"] == feature_order, "Feature order mismatch")
    pd.testing.assert_frame_equal(coverage, pd.read_csv(directory / "reports/coverage.csv"), check_dtype=False)
    pd.testing.assert_frame_equal(overlap, pd.read_csv(directory / "reports/shared_support.csv"), check_dtype=False)
    for split, graph in graphs.items():
        tensors = torch.load(directory / f"graphs/{split}_tensors.pt", weights_only=True, map_location="cpu")
        require(set(tensors) == {"x", "edge_index", "y", "eligible_mask", "tabular_attached"}, "Tensor schema differs")
        for key, value in tensors.items():
            torch.testing.assert_close(value, graph["data"][key], rtol=0, atol=0)
        saved_nodes = pd.read_csv(directory / f"graphs/{split}_nodes.csv", dtype={"user_key": str, "has_graph": "Int64"})
        # CSV represents intentionally empty support artifact paths as NA.
        saved_nodes["tabular_source_artifact"] = saved_nodes.tabular_source_artifact.fillna("")
        pd.testing.assert_frame_equal(saved_nodes, graph["node_provenance"], check_dtype=False)
        graph["data"] = Data(**tensors)
    info = manifest["training"]
    require(info == read_json(directory / "models/training_metadata.json"), "Training metadata mismatch")
    weights, counts = training_class_weights(graphs["train"]["data"])
    require(info["class_counts"] == counts.tolist() and info["class_weights"] == weights.tolist(), "Weights not derived from training labels")
    require(info["training_user_keys"] == sorted(graphs["train"]["masks"].index[graphs["train"]["masks"].has_graph.eq(1)]), "Training key provenance differs")
    history = pd.read_csv(directory / "histories/gcn_tabular.csv")
    require(len(history) == config["training"]["epochs"] and history.epoch.tolist() == list(range(1, len(history) + 1)), "Incomplete training history")
    require(info["best_epoch"] == int(history.loc[history.validation_loss.idxmin(), "epoch"]), "Wrong best epoch")
    metrics = pd.read_csv(directory / "metrics.csv")
    details = read_json(directory / "reports/classification_reports.json")
    primary, checked = {}, 0
    for variant, relative in info["checkpoints"].items():
        model, payload = load_checkpoint(directory / relative)
        epoch = info["best_epoch"] if variant == "best_validation_loss" else info["last_epoch"]
        require(payload["variant"] == variant and payload["epoch"] == epoch and payload["feature_order"] == feature_order, "Checkpoint metadata mismatch")
        require(payload["phase2_manifest_sha256"] == config["phase2_manifest_sha256"] and payload["graph_manifest_sha256"] == config["graph_manifest_sha256"], "Checkpoint input version mismatch")
        with torch.no_grad():
            val = graphs["validation"]["data"]
            loss = nn.CrossEntropyLoss(weight=weights)(model(val.x, val.edge_index)[val.eligible_mask], val.y[val.eligible_mask]).item()
        require(np.isclose(loss, history.loc[history.epoch.eq(epoch), "validation_loss"].iloc[0], atol=1e-7, rtol=1e-7), "Checkpoint validation loss differs from history")
        for split, graph in graphs.items():
            expected = predictions(model, graph, split, config["training"]["threshold"])
            saved = pd.read_csv(directory / f"predictions/gcn_tabular_{variant}_{split}.csv", dtype={"user_key": str, "prediction": "Int64"})
            pd.testing.assert_frame_equal(saved[expected.columns], expected, check_dtype=False, atol=1e-7, rtol=0)
            require(saved.primary_checkpoint.eq(variant == info["selection"]).all(), "Primary checkpoint flag differs")
            if variant == info["selection"]:
                primary[split] = saved
            for cohort in ("all_users", "graph_available"):
                actual, detail = graph_metrics(saved, cohort)
                row = metrics[(metrics.checkpoint_variant == variant) & (metrics.split == split) & (metrics.cohort == cohort)]
                require(len(row) == 1, "Missing/duplicate metric row")
                for key, value in actual.items():
                    recorded = row.iloc[0][key]
                    if isinstance(value, str):
                        require(recorded == value, "Metric cohort mismatch")
                    elif value is None:
                        require(pd.isna(recorded), "Undefined metric mismatch")
                    else:
                        require(np.isclose(recorded, value, atol=1e-10, rtol=1e-10), f"Metric differs: {key}")
                require(details[f"{variant}/{split}/{cohort}"] == detail, "Classification report mismatch")
            checked += 1
    require(len(metrics) == checked * 2, "Unexpected metric rows")
    comparison, keys = descriptive_comparison(root, config, primary)
    pd.testing.assert_frame_equal(comparison, pd.read_csv(directory / "reports/descriptive_matched_comparison.csv"), check_dtype=False, atol=1e-10, rtol=1e-10)
    pd.testing.assert_frame_equal(keys, pd.read_csv(directory / "reports/matched_user_keys.csv"), check_dtype=False)
    return {"verified_artifacts": len(manifest["artifacts"]), "reconstructed_graphs": len(graphs),
            "reloaded_checkpoints": len(info["checkpoints"]), "verified_prediction_files": checked,
            "verified_metric_rows": len(metrics), "descriptive_comparison_rows": len(comparison),
            "matched_counts": keys.groupby("split").size().to_dict()}
