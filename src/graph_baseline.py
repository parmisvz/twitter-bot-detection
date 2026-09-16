"""Protocol-correct degree-only GCN; preserve the historical architecture and settings.

Graphs are independent per split, with shared support identities but no shared
gold targets. Training never receives the test graph. Missing graph users remain
in keyed prediction registries, not fabricated graph neighborhoods.
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
                           evaluate_registry, keyed, notebook_source_hash, read_json,
                           require, seed_all, sha256, write_json)

GRAPH_MASKS = ["has_graph_owner_record", "has_graph_owner", "has_graph_node", "has_graph",
               "has_profile", "has_tweet_text", "has_temporal", "has_text"]
EDGE_COLUMNS = ["source_key", "target_key", "relation"]


def validate_edges(edges, owners, users, assignment, split):
    """Check owner evidence and held-out target isolation before constructing tensors."""
    require(split in SPLITS, "Only supervised split graphs are authorized here")
    require(not edges.empty, f"{split}: empty permitted graph")
    for column in ["source_key", "target_key"]:
        values = edges[column]
        require(values.notna().all() and values.str.fullmatch(r"[a-z0-9_]{1,15}").all(),
                f"{split}: noncanonical endpoint")
    require(edges.relation.eq("follows").all(), "Unsupported graph relation")
    require(not edges.duplicated(EDGE_COLUMNS).any(), "Duplicate observed directed edge")
    require(not edges.source_key.eq(edges.target_key).any(), "Observed self edge forbidden")
    nodes = set(edges.source_key) | set(edges.target_key)
    forbidden = set(assignment.index[~assignment.split.eq(split)])
    require(not nodes & forbidden, f"{split}: forbidden cross-split gold endpoint")
    require(owners.owner_key.notna().all() and set(owners.owner_key) <= set(users.index),
            f"{split}: owner outside canonical split")
    require((owners.owner_key.eq(owners.source_key) | owners.owner_key.eq(owners.target_key)).all(),
            "Owner must be an endpoint of its observed relation")
    require(not owners.duplicated(EDGE_COLUMNS + ["owner_key"]).any(), "Duplicate owner evidence")
    require(set(map(tuple, edges[EDGE_COLUMNS].to_numpy())) ==
            set(map(tuple, owners[EDGE_COLUMNS].to_numpy())), "Edge/owner evidence mismatch")
    return nodes


def build_graph(edges, users, masks):
    """Keep only observed nodes; reverse connections are computational, not new evidence."""
    keys = sorted(set(edges.source_key) | set(edges.target_key))
    node_index = pd.Index(keys, name="user_key")
    mapping = pd.DataFrame({"user_key": keys, "node_index": np.arange(len(keys))})
    src = node_index.get_indexer(edges.source_key)
    dst = node_index.get_indexer(edges.target_key)
    require((src >= 0).all() and (dst >= 0).all(), "Missing endpoint mapping")
    edge_index = torch.tensor(np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])]), dtype=torch.long)
    outgoing = torch.bincount(edge_index[0], minlength=len(keys))
    incoming = torch.bincount(edge_index[1], minlength=len(keys))
    x = torch.stack([incoming, outgoing, incoming + outgoing], dim=1).float()
    eligible_keys = masks.index[masks.has_graph.eq(1)]
    eligible_indices = node_index.get_indexer(eligible_keys)
    require((eligible_indices >= 0).all(), "Eligible graph user has no permitted graph node")
    labels = torch.full((len(keys),), -1, dtype=torch.long)
    eligible_mask = torch.zeros(len(keys), dtype=torch.bool)
    labels[eligible_indices] = torch.tensor(users.loc[eligible_keys, "label_binary"].to_numpy(dtype=np.int64))
    eligible_mask[eligible_indices] = True
    graph = Data(x=x, edge_index=edge_index, y=labels, eligible_mask=eligible_mask)
    return {"data": graph, "mapping": mapping, "users": users, "masks": masks}


def load_graph_inputs(root, config):
    root = Path(root)
    path = root / config["phase2_manifest"]
    require(sha256(path) == config["phase2_manifest_sha256"], "Phase-2 manifest hash mismatch")
    manifest = read_json(path)
    require(manifest["identity_key"] == "user_key", "Unsupported identity contract")
    require(manifest["graph_protocol"]["protocol"] == config["protocol"], "Graph protocol mismatch")
    require(config["eligibility"] == "has_graph", "First pass requires Phase-2 graph availability")
    require(config["gcn_settings"] == {"cached": False, "normalize": True, "add_self_loops": True},
            "First pass requires original GCN normalization without shared adjacency caching")

    def csv(relative):
        return pd.read_csv(checked_artifact(path.parent, manifest, relative), dtype={"user_key": str})

    assignment = keyed(csv("splits/user_split_assignment.csv"), "split assignments")
    require(set(assignment.split) == set(SPLITS), "Invalid frozen split names")
    summary = csv("audit/graph_split_summary.csv").set_index("split")
    graphs, rows = {}, []
    for split in SPLITS:
        users = keyed(csv(f"splits/users/{split}_users.csv"), split)
        expected = align(assignment[assignment.split.eq(split)], users, "frozen split")
        require(users.label_binary.isin([0, 1]).all()
                and users.label_binary.eq(expected.label_binary).all(), "Canonical label mismatch")
        masks = align(keyed(csv(f"splits/masks/{split}_modality_masks.csv"), "masks"), users, "masks")
        require(masks[GRAPH_MASKS].isin([0, 1]).all().all()
                and masks[GRAPH_MASKS].eq(users[GRAPH_MASKS]).all().all(), "Modality masks invalid/inconsistent")
        edges = csv(f"splits/graph/{split}_graph_edges.csv.gz")
        owners = csv(f"splits/graph/{split}_edge_owners.csv.gz")
        nodes = validate_edges(edges, owners, users, assignment, split)
        membership = users.index.isin(nodes)
        require(np.array_equal(masks.has_graph_node.to_numpy(), membership), "Node membership mask differs")
        require(masks.has_graph.eq(masks.has_graph_owner.eq(1) & membership).all(), "Graph eligibility differs")
        require((masks.has_graph_owner <= masks.has_graph_owner_record).all(), "Owner evidence without owner record")
        require(set(owners.owner_key) <= set(masks.index[masks.has_graph_owner.eq(1)]), "Owner availability missing")
        require(len(nodes) == summary.loc[split, "nodes"] and len(edges) == summary.loc[split, "edges"]
                and owners.owner_key.nunique() == summary.loc[split, "owner_users"]
                and summary.loc[split, "forbidden_gold_nodes"] == 0, "Graph audit counts differ")
        graphs[split] = build_graph(edges, users, masks)
        edge_index = graphs[split]["data"].edge_index.numpy().T
        eligible = users.loc[masks.has_graph.eq(1), "label_binary"]
        rows.append({"split": split, "all_users": len(users), "graph_available": len(eligible),
                     "eligible_humans": int(eligible.eq(0).sum()), "eligible_bots": int(eligible.eq(1).sum()),
                     "node_only_users": int((masks.has_graph_node.eq(1) & masks.has_graph.eq(0)).sum()),
                     "nodes": len(nodes), "directed_observed_edges": len(edges),
                     "message_passing_entries": len(edge_index),
                     "repeated_message_entries": len(edge_index) - len(np.unique(edge_index, axis=0)),
                     "forbidden_gold_nodes": 0})
    gold = set(assignment.index)
    support = {s: set(g["mapping"].user_key) - gold for s, g in graphs.items()}
    overlap = [{"left": a, "right": b, "shared_support_nodes": len(support[a] & support[b])}
               for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]]
    return graphs, manifest, pd.DataFrame(rows), pd.DataFrame(overlap)


class GCNClassifier(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, second_dim=32, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, cached=False, normalize=True, add_self_loops=True)
        self.conv2 = GCNConv(hidden_dim, second_dim, cached=False, normalize=True, add_self_loops=True)
        self.classifier = nn.Linear(second_dim, 2)
        self.dropout = dropout

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(F.relu(self.conv2(x, data.edge_index)))


def training_class_weights(train_graph):
    labels = train_graph.y[train_graph.eligible_mask]
    require(len(labels) > 0 and torch.isin(labels, torch.tensor([0, 1])).all().item(), "Invalid training labels")
    counts = torch.bincount(labels, minlength=2)
    require((counts > 0).all().item(), "Both classes required among eligible training users")
    return (counts.sum() / (2 * counts)).float(), counts


def train_gcn(train_graph, validation_graph, config):
    """No test graph parameter; weights depend solely on eligible training labels."""
    require(config["device"] == "cpu", "Reproducible first pass is configured for CPU")
    seed_all(config["seed"], config["torch_threads"])
    model = GCNClassifier(**config["architecture"])
    weights, counts = training_class_weights(train_graph)
    require(validation_graph.eligible_mask.any().item(), "Empty graph-available validation cohort")
    criterion = nn.CrossEntropyLoss(weight=weights)
    spec = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"])
    history, best_state, best_epoch, best_loss = [], None, None, float("inf")
    for epoch in range(1, spec["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_graph)
        loss = criterion(logits[train_graph.eligible_mask], train_graph.y[train_graph.eligible_mask])
        require(torch.isfinite(loss).item(), "Nonfinite GCN training loss")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(validation_graph)
            val_loss = criterion(val_logits[validation_graph.eligible_mask], validation_graph.y[validation_graph.eligible_mask]).item()
        require(np.isfinite(val_loss), "Nonfinite validation loss")
        history.append({"epoch": epoch, "train_loss": loss.item(), "validation_loss": val_loss})
        if val_loss < best_loss:
            best_state, best_epoch, best_loss = deepcopy(model.state_dict()), epoch, val_loss
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{spec['epochs']}: train={loss.item():.6f}, validation={val_loss:.6f}", flush=True)
    states = {"best_validation_loss": best_state, "last": deepcopy(model.state_dict())}
    return states, pd.DataFrame(history), {"best_epoch": best_epoch, "last_epoch": spec["epochs"],
        "class_counts": counts.tolist(), "class_weights": weights.tolist(), "selection": spec["selection"]}


def graph_predictions(model, graph, split, threshold):
    model.eval()
    data = graph["data"]
    with torch.no_grad():
        probabilities = torch.softmax(model(data), dim=1)[:, 1][data.eligible_mask].numpy()
    require(np.isfinite(probabilities).all(), "Nonfinite graph probabilities")
    keys = graph["mapping"].loc[data.eligible_mask.numpy(), "user_key"].tolist()
    registry = graph["users"][["label_binary"]].join(graph["masks"][GRAPH_MASKS])
    require(set(keys) == set(registry.index[registry.has_graph.eq(1)]), "Prediction eligibility mismatch")
    registry["bot_probability"] = pd.Series(probabilities, index=keys, dtype=float)
    registry["prediction"] = pd.Series((probabilities >= threshold).astype(int), index=keys).astype("Int64")
    registry["status"] = np.where(registry.has_graph.eq(1), "predicted", "missing_graph_modality")
    registry["graph_evidence"] = np.where(registry.has_graph.eq(1), "owner_and_incident_edges",
                                           np.where(registry.has_graph_node.eq(1), "node_only", "no_permitted_node"))
    registry["model"], registry["split"] = "gcn_degree_only", split
    return registry.reset_index()


def graph_metrics(registry, cohort):
    require(cohort in ("all_users", "graph_available"), "Unknown graph cohort")
    frame = registry if cohort == "all_users" else registry.loc[registry.has_graph.eq(1)]
    metrics, detail = evaluate_registry(frame, "all_users")
    metrics["cohort"] = cohort
    return metrics, detail


def load_gcn_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = GCNClassifier(**payload["architecture"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval(), payload


def matched_comparison(root, config, registries):
    """Re-score saved primary baselines and GCN on one explicit common supported cohort."""
    path = Path(root) / config["baseline_manifest"]
    require(sha256(path) == config["baseline_manifest_sha256"], "Baseline comparison manifest hash mismatch")
    manifest = read_json(path)
    require(manifest["complete"] and manifest["phase2_manifest_sha256"] == config["phase2_manifest_sha256"],
            "Baseline comparison uses a different population/version")
    rows, cohort_rows = [], []
    for split in ("validation", "test"):
        graph = keyed(registries[split], "graph predictions")
        candidates = {"gcn_degree_only": graph}
        common = set(graph.index[graph.status.eq("predicted")])
        for name, metadata in manifest["models"].items():
            relative = f"predictions/{name}_{metadata['selected']}_{split}.csv"
            frame = keyed(pd.read_csv(checked_artifact(path.parent, manifest, relative),
                                      dtype={"user_key": str}), name)
            frame = align(frame, graph, "baseline comparison")
            require(frame.label_binary.eq(graph.label_binary).all(), "Comparison labels differ")
            require(frame.has_graph.eq(graph.has_graph).all(), "Comparison graph masks differ")
            common &= set(frame.index[frame.status.eq("predicted")])
            candidates[name] = frame
        require(common, "Empty matched comparison cohort")
        keys = sorted(common)
        cohort_rows.extend({"split": split, "user_key": k} for k in keys)
        for name, frame in candidates.items():
            metric, _ = evaluate_registry(frame.loc[keys].reset_index(), "all_users")
            metric["cohort"] = "graph_available_and_all_baselines_supported"
            rows.append({"model": name, "split": split, **metric})
    return pd.DataFrame(rows), pd.DataFrame(cohort_rows)


class GraphBaselineRun:
    def __init__(self, root, config_path="configs/graph_baseline_v2.json"):
        self.root = Path(root).resolve()
        self.config = read_json(self.root / config_path)
        self.graphs, self.phase2_manifest, self.coverage, self.support_overlap = load_graph_inputs(self.root, self.config)
        require(self.config["training"]["selection"] == "best_validation_loss", "First pass preserves intended best-validation selection")
        seed_all(self.config["seed"], self.config["torch_threads"])
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.output_dir = self.root / self.config["output_root"] / (stamp + "_" + uuid4().hex[:8])
        self.output_dir.mkdir(parents=True, exist_ok=False)
        for directory in ("graphs", "models", "histories", "predictions", "reports", "provenance"):
            (self.output_dir / directory).mkdir()
        self.coverage.to_csv(self.output_dir / "reports/graph_coverage.csv", index=False)
        self.support_overlap.to_csv(self.output_dir / "reports/shared_support.csv", index=False)
        for split, graph in self.graphs.items():
            graph["mapping"].to_csv(self.output_dir / f"graphs/{split}_nodes.csv", index=False)
            data = graph["data"]
            torch.save({"x": data.x, "edge_index": data.edge_index, "y": data.y,
                        "eligible_mask": data.eligible_mask}, self.output_dir / f"graphs/{split}_tensors.pt")
        write_json(self.output_dir / "provenance/resolved_config.json", self.config)
        write_json(self.output_dir / "provenance/phase2_manifest.json", self.phase2_manifest)
        sources = ["src/graph_baseline.py", "src/baselines.py"]
        for source in sources:
            (self.output_dir / "provenance" / Path(source).name).write_bytes((self.root / source).read_bytes())
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.root, text=True).strip()
            status = subprocess.check_output(["git", "status", "--porcelain"], cwd=self.root, text=True)
        except (OSError, subprocess.CalledProcessError):
            commit, status = None, "unavailable"
        self.provenance = {"created_utc": stamp, "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "torch-geometric", "numpy", "pandas", "scikit-learn")},
            "seed": self.config["seed"], "device": self.config["device"], "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": True, "source_sha256": {s: sha256(self.root / s) for s in sources},
            "config_sha256": sha256(self.root / config_path),
            "notebook_source_sha256": notebook_source_hash(self.root / "analysis/04_graph_baseline.ipynb"),
            "git_commit_at_execution": commit, "git_status_at_execution": status,
            "protocol": self.phase2_manifest["graph_protocol"],
            "limitations": ["Gold targets are held out across graphs, but support accounts overlap; not fully node-disjoint.",
                "Validation/test inference uses the cohort graph jointly, not isolated per-account deployment.",
                "Owner-available cohorts differ from historical global-node cohorts; scores are not directly comparable.",
                "Three degree columns are redundant [d,d,2d], unscaled and measured on partial observed split graphs.",
                "Reverse message connections retain reciprocal-edge multiplicity; GCN self-loops are computational only.",
                "Graph-unavailable users remain in registries with no prediction, even when node-only evidence exists.",
                "Small graph-available test cohort and few bots; a single seeded run does not quantify uncertainty.",
                "Historical test data was already evaluated. Original GCN was unseeded and its best-state copy was defective."]}
        write_json(self.output_dir / "provenance/run.json", self.provenance)
        self.info = None
        self.complete = False

    def fit(self):
        require(self.info is None, "Run already fitted; start a new run instead of overwriting")
        states, self.history, self.info = train_gcn(self.graphs["train"]["data"], self.graphs["validation"]["data"], self.config)
        self.history.to_csv(self.output_dir / "histories/gcn.csv", index=False)
        self.info["training_user_keys"] = sorted(self.graphs["train"]["masks"].index[self.graphs["train"]["masks"].has_graph.eq(1)])
        self.info["checkpoints"] = {}
        for variant, state in states.items():
            relative = f"models/gcn_{variant}.pt"
            epoch = self.info["best_epoch"] if variant == "best_validation_loss" else self.info["last_epoch"]
            torch.save({"state_dict": state, "architecture": self.config["architecture"],
                        "epoch": epoch, "variant": variant, "class_weights": self.info["class_weights"],
                        "phase2_manifest_sha256": self.config["phase2_manifest_sha256"]}, self.output_dir / relative)
            self.info["checkpoints"][variant] = relative
        self.states = states
        write_json(self.output_dir / "models/training_metadata.json", self.info)
        return self.history

    def evaluate_and_export(self):
        require(self.info is not None and not self.complete, "Fit once before finalizing once")
        rows, details, reload_checks, primary = [], {}, [], {}
        for variant, relative in self.info["checkpoints"].items():
            model, _ = load_gcn_checkpoint(self.output_dir / relative)
            original = GCNClassifier(**self.config["architecture"])
            original.load_state_dict(self.states[variant])
            for split, graph in self.graphs.items():
                registry = graph_predictions(model, graph, split, self.config["training"]["threshold"])
                expected = graph_predictions(original, graph, split, self.config["training"]["threshold"])
                pd.testing.assert_frame_equal(registry, expected, check_exact=True)
                reload_checks.append({"checkpoint": variant, "split": split, "exact_reload_match": True})
                registry["checkpoint_variant"] = variant
                registry["primary_checkpoint"] = variant == self.info["selection"]
                registry.to_csv(self.output_dir / f"predictions/gcn_{variant}_{split}.csv", index=False)
                if variant == self.info["selection"]:
                    primary[split] = registry
                for cohort in ("all_users", "graph_available"):
                    metric, detail = graph_metrics(registry, cohort)
                    rows.append({"model": "gcn_degree_only", "checkpoint_variant": variant,
                                 "primary_checkpoint": variant == self.info["selection"], "split": split, **metric})
                    details[f"{variant}/{split}/{cohort}"] = detail
        metrics = pd.DataFrame(rows)
        metrics.to_csv(self.output_dir / "metrics.csv", index=False)
        self.comparison, matched_keys = matched_comparison(self.root, self.config, primary)
        self.comparison.to_csv(self.output_dir / "reports/matched_baseline_comparison.csv", index=False)
        matched_keys.to_csv(self.output_dir / "reports/matched_comparison_keys.csv", index=False)
        write_json(self.output_dir / "reports/classification_reports.json", details)
        write_json(self.output_dir / "reports/checkpoint_reload.json", reload_checks)
        write_json(self.output_dir / "manifest.json", {"schema_version": "graph_baseline_v2", "complete": True,
            "phase2_manifest_sha256": self.config["phase2_manifest_sha256"], "identity_key": "user_key",
            "graph_protocol": self.phase2_manifest["graph_protocol"], "eligibility": "has_graph",
            "degree_feature_order": self.config["degree_feature_order"], "training": self.info,
            "graph_artifacts": {s: {"nodes": f"graphs/{s}_nodes.csv", "tensors": f"graphs/{s}_tensors.pt"} for s in SPLITS},
            "downstream_contract": "Load graphs separately using split-specific user_key mappings; do not merge gold cohorts or reuse global legacy indices. Only has_graph users receive graph predictions.",
            "artifacts": artifact_inventory(self.output_dir)})
        self.complete = True
        return metrics


def verify_graph_run(root, output_dir):
    """Independent disk reload plus graph reconstruction and evaluation from pinned inputs."""
    directory = Path(output_dir)
    manifest = read_json(directory / "manifest.json")
    require(manifest["complete"], "Incomplete graph run")
    for relative in manifest["artifacts"]:
        checked_artifact(directory, manifest, relative)
    config = read_json(directory / "provenance/resolved_config.json")
    seed_all(config["seed"], config["torch_threads"])
    graphs, _, coverage, overlap = load_graph_inputs(root, config)
    pd.testing.assert_frame_equal(coverage, pd.read_csv(directory / "reports/graph_coverage.csv"), check_dtype=False)
    pd.testing.assert_frame_equal(overlap, pd.read_csv(directory / "reports/shared_support.csv"), check_dtype=False)
    for split, graph in graphs.items():
        tensors = torch.load(directory / f"graphs/{split}_tensors.pt", weights_only=True, map_location="cpu")
        for key, value in tensors.items():
            torch.testing.assert_close(value, graph["data"][key], rtol=0, atol=0)
        pd.testing.assert_frame_equal(graph["mapping"], pd.read_csv(directory / f"graphs/{split}_nodes.csv"), check_dtype=False)
        # Prediction verification uses saved graph tensors after validating the reconstruction.
        graph["data"] = Data(**tensors)
    info = manifest["training"]
    require(info == read_json(directory / "models/training_metadata.json"), "Training metadata differs")
    weights, counts = training_class_weights(graphs["train"]["data"])
    require(counts.tolist() == info["class_counts"] and weights.tolist() == info["class_weights"], "Class weights differ from train-only labels")
    train_keys = sorted(graphs["train"]["masks"].index[graphs["train"]["masks"].has_graph.eq(1)])
    require(info["training_user_keys"] == train_keys, "Training keys differ")
    history = pd.read_csv(directory / "histories/gcn.csv")
    require(len(history) == config["training"]["epochs"] and history.epoch.tolist() == list(range(1, len(history) + 1)), "Incomplete history")
    require(int(history.loc[history.validation_loss.idxmin(), "epoch"]) == info["best_epoch"], "Best epoch differs from history")
    metrics = pd.read_csv(directory / "metrics.csv")
    details = read_json(directory / "reports/classification_reports.json")
    primary, checked = {}, 0
    for variant, relative in info["checkpoints"].items():
        model, payload = load_gcn_checkpoint(directory / relative)
        epoch = info["best_epoch"] if variant == "best_validation_loss" else info["last_epoch"]
        require(payload["epoch"] == epoch and payload["variant"] == variant, "Checkpoint selection metadata differs")
        require(payload["phase2_manifest_sha256"] == config["phase2_manifest_sha256"], "Checkpoint input version differs")
        with torch.no_grad():
            val_graph = graphs["validation"]["data"]
            loss = nn.CrossEntropyLoss(weight=weights)(model(val_graph)[val_graph.eligible_mask], val_graph.y[val_graph.eligible_mask]).item()
        require(np.isclose(loss, history.loc[history.epoch.eq(epoch), "validation_loss"].iloc[0], atol=1e-7, rtol=1e-7), "Checkpoint does not reproduce its recorded validation loss")
        for split, graph in graphs.items():
            expected = graph_predictions(model, graph, split, config["training"]["threshold"])
            saved = pd.read_csv(directory / f"predictions/gcn_{variant}_{split}.csv", dtype={"user_key": str, "prediction": "Int64"})
            pd.testing.assert_frame_equal(saved[expected.columns], expected, check_dtype=False, rtol=0, atol=1e-7)
            require(saved.primary_checkpoint.eq(variant == info["selection"]).all(), "Primary selection mismatch")
            if variant == info["selection"]:
                primary[split] = saved
            for cohort in ("all_users", "graph_available"):
                metric, detail = graph_metrics(saved, cohort)
                row = metrics[(metrics.checkpoint_variant == variant) & (metrics.split == split) & (metrics.cohort == cohort)]
                require(len(row) == 1, "Missing/duplicate metric row")
                for key, value in metric.items():
                    actual = row.iloc[0][key]
                    if isinstance(value, str):
                        require(actual == value, "Metric cohort mismatch")
                    elif value is None:
                        require(pd.isna(actual), "Undefined metric mismatch")
                    else:
                        require(np.isclose(actual, value, atol=1e-10, rtol=1e-10), f"Metric differs: {key}")
                require(details[f"{variant}/{split}/{cohort}"] == detail, "Classification report differs")
            checked += 1
    require(len(metrics) == checked * 2, "Unexpected metrics count")
    comparison, comparison_keys = matched_comparison(root, config, primary)
    pd.testing.assert_frame_equal(comparison, pd.read_csv(directory / "reports/matched_baseline_comparison.csv"), check_dtype=False, atol=1e-10, rtol=1e-10)
    pd.testing.assert_frame_equal(comparison_keys, pd.read_csv(directory / "reports/matched_comparison_keys.csv"), check_dtype=False)
    return {"verified_artifacts": len(manifest["artifacts"]), "reloaded_checkpoints": len(info["checkpoints"]),
            "reconstructed_graphs": len(graphs), "verified_prediction_files": checked,
            "verified_metric_rows": len(metrics), "matched_comparison_rows": len(comparison)}
