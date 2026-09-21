"""End-to-end optional graph representations; train/validation only, immutable inputs."""
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
from torch_geometric.nn import GCNConv
from sklearn.metrics import average_precision_score, roc_auc_score

from src.baselines import seed_all
from src.graph_baseline import build_graph, validate_edges, GRAPH_MASKS
from src.temporal_statistics import TemporalExperimentModel, load_inputs as load_temporal, make_loader
from src.checkpoint_weight_experiment import weights_from_training, state_hash
from src.performance_diagnosis import Inputs, digest, metrics, preserved_hashes, require, write_json, plt, save_figure

ARMS = ("temporal_anchor", "degree_mlp", "gcn")
POLICIES = ("best_unweighted_validation_loss", "last")


def registered_path(reader, directory, manifest, relative):
    path = (directory / relative).resolve()
    require(path.is_relative_to(directory.resolve()) and "test" not in path.stem.lower(), "Forbidden artifact")
    expected = manifest["artifacts"][relative]["sha256"]
    require(digest(path) == expected, "Artifact hash mismatch: " + relative)
    reader.accessed[path.relative_to(reader.root).as_posix()] = expected
    return path


def load_inputs(root, config, reader):
    temporal_path = root / config["temporal_config"]
    reader.allow(temporal_path, digest(temporal_path))
    reference_config = reader.read(temporal_path)
    require(reference_config["seeds"] == config["seeds"] == [42, 43, 44], "Seed policy changed")
    require(reference_config["selection"] == POLICIES[0] and reference_config["threshold"] == .5, "Selection changed")
    data, mapping, _ = load_temporal(root, reference_config, reader)
    temporal_path = root / config["temporal_manifest"]
    reader.allow(temporal_path, config["temporal_manifest_sha256"])
    temporal_manifest = reader.read(temporal_path)
    require(temporal_manifest["complete"] and temporal_manifest["feature_mapping"] == mapping, "Temporal reference mismatch")
    original_config = reader.registered(temporal_path.parent, temporal_manifest, "provenance/resolved_config.json")
    require(original_config == reference_config, "Temporal configuration no longer matches frozen reference")
    historical = reader.registered(temporal_path.parent, temporal_manifest, "validation_metrics.csv")
    phase_path = root / reference_config["phase2_manifest"]
    phase = reader.read(phase_path)
    require(phase["graph_protocol"]["protocol"] == config["protocol"], "Graph protocol mismatch")
    graph_path = root / config["graph_manifest"]
    reader.allow(graph_path, config["graph_manifest_sha256"])
    graph_manifest = reader.read(graph_path)
    require(graph_manifest["complete"] and graph_manifest["phase2_manifest_sha256"] == reference_config["phase2_manifest_sha256"], "Graph reference mismatch")
    # Projection is explicit: do not deserialize the assignment's label column.
    assignment_path = registered_path(reader, phase_path.parent, phase, "splits/user_split_assignment.csv")
    assignment = pd.read_csv(assignment_path, usecols=["user_key", "split"], dtype=str).set_index("user_key")
    require(assignment.index.is_unique and assignment.index.notna().all() and set(assignment.split) == {"train", "validation", "test"}, "Invalid identity-only assignments")
    coverage, graphs = [], {}
    for split, pop in data.items():
        require(set(pop["users"].index) == set(assignment.index[assignment.split.eq(split)]), "Assignment identities differ")
        edges = reader.registered(phase_path.parent, phase, f"splits/graph/{split}_graph_edges.csv.gz")
        owners = reader.registered(phase_path.parent, phase, f"splits/graph/{split}_edge_owners.csv.gz")
        nodes = validate_edges(edges, owners, pop["users"], assignment, split)
        masks = pop["masks"]
        require(masks[GRAPH_MASKS].isin([0, 1]).all().all() and masks[GRAPH_MASKS].eq(pop["users"][GRAPH_MASKS]).all().all(), "Graph masks differ")
        require(masks.has_graph_node.eq(masks.index.isin(nodes)).all(), "Node membership mask differs")
        require(masks.has_graph.eq(masks.has_graph_owner.eq(1) & masks.has_graph_node.eq(1)).all(), "Graph eligibility differs")
        graph = build_graph(edges, pop["users"], masks)
        saved_nodes = reader.registered(graph_path.parent, graph_manifest, f"graphs/{split}_nodes.csv")
        pd.testing.assert_frame_equal(graph["mapping"], saved_nodes, check_dtype=False)
        saved = torch.load(registered_path(reader, graph_path.parent, graph_manifest, f"graphs/{split}_tensors.pt"), map_location="cpu", weights_only=True)
        for name in ("x", "edge_index", "y", "eligible_mask"):
            torch.testing.assert_close(getattr(graph["data"], name), saved[name], rtol=0, atol=0)
        index = pd.Index(graph["mapping"].user_key).get_indexer(pop["users"].index)
        available = masks.has_graph.to_numpy(dtype=bool)
        require((index[available] >= 0).all() and int(available.sum()) == config["expected_graph_users"][split], "Graph coverage changed")
        graph["canonical_node_index"] = torch.tensor(index, dtype=torch.long)
        graph["available"] = torch.tensor(available, dtype=torch.bool)
        graph["provenance"] = graph["mapping"].copy()
        graph["provenance"]["is_same_split_gold"] = graph["mapping"].user_key.isin(pop["users"].index)
        graph["provenance"]["has_graph"] = graph["mapping"].user_key.map(masks.has_graph).astype("Int64")
        graph["provenance"]["supervised_eligible"] = graph["data"].eligible_mask.numpy()
        graph["provenance"]["node_features"] = "observed_symmetrized_degrees_only"
        graph["provenance"]["tabular_attached"] = False
        graphs[split] = graph
        coverage.append({"split": split, "canonical_users": len(index), "graph_available": int(available.sum()),
                         "graph_unavailable": int((~available).sum()), "graph_nodes": len(nodes), "observed_directed_edges": len(edges),
                         "eligible_bots": int(pop["y"][available].sum()), "forbidden_gold_endpoints": 0})
    shared = set(graphs["train"]["mapping"].user_key) & set(graphs["validation"]["mapping"].user_key)
    require(not shared & set(assignment.index), "Shared gold graph nodes")
    return data, mapping, graphs, reference_config, historical, pd.DataFrame(coverage), len(shared)


class GraphIntegrationModel(nn.Module):
    def __init__(self, arm, mapping, dropout=.3):
        super().__init__()
        require(arm in ARMS, "Invalid arm")
        self.arm = arm
        self.base = TemporalExperimentModel("temporal_branch", mapping, dropout)
        if arm != "temporal_anchor":
            self.base.classifier = nn.Sequential(nn.Linear(96, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
            layer = (lambda a, b: GCNConv(a, b, cached=False, normalize=True, add_self_loops=True)) if arm == "gcn" else nn.Linear
            self.graph1, self.graph2 = layer(3, 64), layer(64, 32)
            self.graph_dropout = nn.Dropout(dropout)

    def graph_representation(self, graph):
        if self.arm == "temporal_anchor":
            return None
        x, edges = graph["data"].x, graph["data"].edge_index
        # A graph-less population needs no placeholder node or fabricated edge.
        # Populated train/validation graphs follow the original computation below.
        if not graph["available"].any().item():
            return x.new_zeros((len(graph["available"]), 32))
        h = self.graph1(x, edges) if self.arm == "gcn" else self.graph1(x)
        h = self.graph_dropout(torch.relu(h))
        h = torch.relu(self.graph2(h, edges) if self.arm == "gcn" else self.graph2(h))
        idx, available = graph["canonical_node_index"], graph["available"]
        require(torch.all(idx[available] >= 0).item(), "Available user lacks graph node")
        # Index -1 is never interpreted as a user's node. Mask before fusion.
        return torch.where(available[:, None], h[idx.clamp_min(0)], torch.zeros((len(idx), 32), dtype=h.dtype))

    def forward(self, x, indices, graph, representation=None):
        if self.arm == "temporal_anchor":
            return self.base(x)
        other, temporal, mask = self.base.split_inputs(x)
        other_h = self.base.other_encoder(other)
        temporal_h = self.base.temporal_representation(temporal, mask)
        graph_h = self.graph_representation(graph) if representation is None else representation
        return self.base.classifier(torch.cat([other_h, temporal_h, graph_h[indices]], dim=1))


def initialized_model(arm, mapping, seed, threads=1):
    seed_all(seed, threads)
    if arm != "gcn":
        return GraphIntegrationModel(arm, mapping)
    reference = GraphIntegrationModel("degree_mlp", mapping)
    rng = torch.get_rng_state()
    model = GraphIntegrationModel("gcn", mapping)
    model.base.load_state_dict(reference.base.state_dict())
    with torch.no_grad():
        for name in ("graph1", "graph2"):
            linear, conv = getattr(reference, name), getattr(model, name)
            conv.lin.weight.copy_(linear.weight)
            conv.bias.copy_(linear.bias)
    torch.set_rng_state(rng)
    return model


def normalized_initial_hash(model):
    return state_hash({k.replace(".lin.weight", ".weight"): v for k, v in model.state_dict().items()})


def evaluate(model, pop, graph, spec, weight):
    model.eval()
    weighted, common, probabilities = [], 0., []
    with torch.no_grad():
        representation = model.graph_representation(graph)
        for start in range(0, len(pop["y"]), spec["batch_size"]):
            stop = min(start + spec["batch_size"], len(pop["y"]))
            x = torch.tensor(pop["x"][start:stop], dtype=torch.float32)
            y = torch.tensor(pop["y"][start:stop], dtype=torch.float32).reshape(-1, 1)
            logits = model(x, torch.arange(start, stop), graph, representation)
            require(torch.isfinite(logits).all().item(), "Nonfinite logits")
            weighted.append(nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor([weight])).item())
            common += nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="sum").item()
            probabilities.extend(logits.sigmoid().flatten().tolist())
    return np.asarray(probabilities), float(np.mean(weighted)), common / len(pop["y"])


def train_one(data, graphs, mapping, spec, arm, seed, weight, threads=1):
    model = initialized_model(arm, mapping, seed, threads)
    info = {"arm": arm, "seed": seed, "parameter_count": sum(p.numel() for p in model.parameters()),
            "initial_normalized_sha256": normalized_initial_hash(model), "pos_weight": weight}
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"])
    loader = make_loader(data["train"], spec, seed, "train")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], dtype=torch.float32))
    best, history, orders = float("inf"), [], []
    for epoch in range(1, spec["epochs"] + 1):
        model.train()
        total, seen = 0., []
        for x, y, indices in loader:
            seen.extend(indices.tolist())
            optimizer.zero_grad()
            loss = criterion(model(x, indices, graphs["train"]), y)
            require(torch.isfinite(loss).item(), "Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += loss.item()
        require(sorted(seen) == list(range(len(data["train"]["y"]))), "Training user omitted/repeated")
        orders.append(hashlib.sha256(np.asarray(seen, dtype=np.int64).tobytes()).hexdigest())
        _, weighted, common = evaluate(model, data["validation"], graphs["validation"], spec, weight)
        history.append({"epoch": epoch, "train_weighted_loss": total / len(loader),
                        "validation_weighted_loss": weighted, "validation_unweighted_loss": common})
        if common < best:
            best, best_epoch, best_state = common, epoch, deepcopy(model.state_dict())
        if epoch % 10 == 0:
            print(f"{arm} seed {seed}, epoch {epoch}: validation BCE {common:.4f}", flush=True)
    info.update({"epoch_order_sha256": orders, "epochs": {POLICIES[0]: best_epoch, "last": spec["epochs"]}})
    return {POLICIES[0]: best_state, "last": deepcopy(model.state_dict())}, pd.DataFrame(history), info


def prediction_registry(pop, p, **identity):
    frame = pop["users"][["label_binary"]].join(pop["masks"]).reset_index()
    require(len(p) == len(frame) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid prediction array")
    frame["bot_probability"], frame["prediction"] = p, (p >= .5).astype(int)
    frame["status"], frame["split"] = "predicted", "validation"
    frame["graph_representation_status"] = np.where(frame.has_graph.eq(1), "available", "masked_absent")
    if identity["arm"] == "temporal_anchor":
        frame["graph_representation_status"] = "not_used_by_arm"
    for key, value in identity.items():
        frame[key] = value
    return frame


def cohort_metrics(frame):
    rows = []
    for cohort, mask in (("all", np.ones(len(frame), dtype=bool)), ("graph_available", frame.has_graph.eq(1)),
                         ("graph_unavailable", frame.has_graph.eq(0))):
        group = frame.loc[mask]
        require(len(group) > 0 and group.label_binary.nunique() == 2, "Degenerate evaluation cohort")
        identity = {k: group.iloc[0][k] for k in ("arm", "seed", "policy", "epoch")}
        rows.append({**identity, "cohort": cohort, "n": len(group), "bots": int(group.label_binary.sum()),
                     **metrics(group.label_binary, group.bot_probability, .5),
                     "roc_auc": float(roc_auc_score(group.label_binary, group.bot_probability)),
                     "average_precision": float(average_precision_score(group.label_binary, group.bot_probability))})
    return rows


def summarize(table):
    measured = ["bot_f1", "bot_precision", "bot_recall", "roc_auc", "average_precision"]
    summary = table.groupby(["arm", "policy", "cohort"])[measured].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    effects = []
    for row in table[table.arm.eq("gcn")].itertuples():
        for control in ("degree_mlp", "temporal_anchor"):
            ref = table[table.arm.eq(control) & table.seed.eq(row.seed) & table.policy.eq(row.policy) & table.cohort.eq(row.cohort)].iloc[0]
            effects.append({"seed": row.seed, "policy": row.policy, "cohort": row.cohort, "comparison": "gcn_minus_" + control,
                            **{"delta_" + m: float(getattr(row, m) - ref[m]) for m in measured}})
    return summary.reset_index(), pd.DataFrame(effects)


def validate_config(config):
    require(config["arms"] == list(ARMS) and config["seeds"] == [42, 43, 44], "Arms/seeds changed")
    require(config["graph_input_dim"] == 3 and config["graph_hidden_dims"] == [64, 32] and config["fusion_hidden_dim"] == 32
            and config["graph_dropout"] == .3, "Architecture changed")
    require(config["gcn_settings"] == {"cached": False, "normalize": True, "add_self_loops": True}, "GCN protocol changed")
    require(config["graph_availability"] == "has_graph" and config["training_mode"] == "end_to_end_full_split_graph_per_training_minibatch"
            and config["scope"] == "train_only_optimization_validation_only_selection_no_test_no_threshold_tuning_no_promotion", "Scope changed")


def run(root, config_path="configs/gcn_integration_v1.json"):
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
        data, mapping, graphs, reference, historical, coverage, shared = load_inputs(root, config, reader)
        spec = reference["training"]
        weight = weights_from_training(data["train"]["y"])["full_train_ratio"]
        write_json(output / "provenance/resolved_config.json", config)
        write_json(output / "provenance/training_config.json", reference)
        write_json(output / "provenance/feature_mapping.json", mapping)
        write_json(output / "provenance/parsed_inputs.json", reader.accessed)
        write_json(output / "provenance/assignment_projection.json", {"columns": ["user_key", "split"], "purpose": "forbidden endpoint audit only", "label_column_parsed": False})
        coverage.to_csv(output / "graph_coverage.csv", index=False)
        historical.to_csv(output / "historical_temporal_validation.csv", index=False)
        for split, pop in data.items():
            pop["users"][["label_binary"]].join(pop["masks"]).to_csv(output / f"provenance/{split}_population.csv")
            graphs[split]["provenance"].to_csv(output / f"provenance/{split}_nodes.csv", index=False)
        runs, rows = {}, []
        for seed in config["seeds"]:
            for arm in ARMS:
                print(f"Training {arm}, seed {seed}", flush=True)
                states, history, info = train_one(data, graphs, mapping, spec, arm, seed, weight, reference["torch_threads"])
                name = f"{arm}_seed{seed}"
                history.to_csv(output / f"histories/{name}.csv", index=False)
                info["training_user_keys"] = data["train"]["users"].index.tolist()
                info["checkpoints"] = {}
                for policy in POLICIES:
                    model = initialized_model(arm, mapping, seed)
                    model.load_state_dict(states[policy], strict=True)
                    p, weighted, common = evaluate(model, data["validation"], graphs["validation"], spec, weight)
                    frame = prediction_registry(data["validation"], p, arm=arm, seed=seed, policy=policy, epoch=info["epochs"][policy])
                    for row in cohort_metrics(frame):
                        row.update({"parameter_count": info["parameter_count"], "validation_weighted_loss": weighted,
                                    "validation_unweighted_loss": common})
                        rows.append(row)
                    relative = f"models/{name}_{policy}.pt"
                    torch.save({"state_dict": states[policy], "arm": arm, "seed": seed, "mapping": mapping,
                                "policy": policy, "epoch": info["epochs"][policy], "pos_weight": weight, "threshold": .5,
                                "training_user_keys": info["training_user_keys"], "phase2_manifest_sha256": reference["phase2_manifest_sha256"],
                                "graph_manifest_sha256": config["graph_manifest_sha256"]}, output / relative)
                    frame.to_csv(output / f"predictions/{name}_{policy}_validation.csv", index=False)
                    info["checkpoints"][policy] = relative
                runs[name] = info
            paired = [i for i in runs.values() if i["seed"] == seed]
            require(all(i["epoch_order_sha256"] == paired[0]["epoch_order_sha256"] for i in paired), "Unpaired batches")
            graph_arms = [i for i in paired if i["arm"] != "temporal_anchor"]
            require(len({i["parameter_count"] for i in graph_arms}) == 1 and len({i["initial_normalized_sha256"] for i in graph_arms}) == 1, "Graph arms not paired")
        table = pd.DataFrame(rows)
        summary, effects = summarize(table)
        for name, frame in (("validation_metrics", table), ("seed_summary", summary), ("paired_effects", effects)):
            frame.to_csv(output / f"{name}.csv", index=False)
        write_json(output / "training_metadata.json", runs)
        fig, axes = plt.subplots(3, 3, figsize=(13, 10))
        for r, arm in enumerate(ARMS):
            for c, seed in enumerate(config["seeds"]):
                name = f"{arm}_seed{seed}"
                h = pd.read_csv(output / f"histories/{name}.csv")
                axes[r, c].plot(h.epoch, h.validation_unweighted_loss)
                axes[r, c].axvline(runs[name]["epochs"][POLICIES[0]], color="black", linestyle=":")
                axes[r, c].set(title=f"{arm}, seed {seed}", xlabel="Epoch", ylabel="Validation BCE")
        save_figure(fig, output / "figures/validation_curves.png")
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, cohort in zip(axes, ("all", "graph_available", "graph_unavailable")):
            for seed in config["seeds"]:
                f = table[table.seed.eq(seed) & table.policy.eq(POLICIES[0]) & table.cohort.eq(cohort)].set_index("arm").loc[list(ARMS)]
                ax.plot(range(3), f.bot_f1, marker="o", label=f"seed {seed}")
            ax.set(xticks=range(3), xticklabels=["Temporal", "Degree MLP", "GCN"], title=cohort, ylabel="Bot F1", ylim=(0, 1))
            ax.legend()
        save_figure(fig, output / "figures/paired_cohort_f1.png")
        write_json(output / "provenance/protected_files.json", before)
        for source in ("src/gcn_integration.py", "src/temporal_statistics.py", "src/graph_baseline.py", "src/checkpoint_weight_experiment.py",
                       "src/performance_diagnosis.py", "src/baselines.py", "scripts/run_gcn_integration.py"):
            (output / "provenance" / Path(source).name).write_bytes((root / source).read_bytes())
        write_json(output / "provenance/environment.json", {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "torch-geometric", "numpy", "pandas", "scikit-learn", "matplotlib")},
            "torch_threads": reference["torch_threads"], "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
            "test_labels_features_graphs_predictions_parsed": False, "test_identities_used_for_isolation_only": True,
            "shared_train_validation_support_nodes": shared, "threshold_tuned": False})
    finally:
        require(before == preserved_hashes(root, config["output_root"]), "Protected prior files changed")
    artifacts = {p.relative_to(output).as_posix(): {"sha256": digest(p)} for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output / "manifest.json", {"schema_version": config["schema_version"], "complete": True, "promoted_model": None,
        "feature_mapping": mapping, "runs": runs, "artifacts": artifacts, "protected_files_verified": len(before),
        "limitations": ["Validation reused for checkpoint selection and development; no unbiased final test estimate.",
            "Graph-available validation has 42 users and seven bots; small-cohort uncertainty.",
            "Support nodes can overlap graphs; gold-isolated protocol is not fully node-disjoint.",
            "GCN versus temporal anchor changes parameter count and available information.",
            "Temporal formulas/windows unverified; no natural missing temporal statistics in training/validation.",
            "Degree channels are raw, redundant [d,d,2d]; no semantic support features."]})
    return output, table, summary


def verify(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    require(manifest["complete"] and manifest["promoted_model"] is None, "Invalid experiment manifest")
    for rel, meta in manifest["artifacts"].items():
        require(digest(output / rel) == meta["sha256"], "Artifact hash mismatch: " + rel)
    config = json.loads((output / "provenance/resolved_config.json").read_text())
    validate_config(config)
    reader = Inputs(root)
    data, mapping, graphs, reference, historical, coverage, shared = load_inputs(root, config, reader)
    require(mapping == manifest["feature_mapping"], "Feature mapping mismatch")
    require(reference == json.loads((output / "provenance/training_config.json").read_text()), "Training config mismatch")
    pd.testing.assert_frame_equal(coverage, pd.read_csv(output / "graph_coverage.csv"), check_dtype=False)
    weight, rows = weights_from_training(data["train"]["y"])["full_train_ratio"], []
    require(manifest["runs"] == json.loads((output / "training_metadata.json").read_text()), "Metadata mismatch")
    for name, info in manifest["runs"].items():
        require(info["training_user_keys"] == data["train"]["users"].index.tolist() and info["pos_weight"] == weight, "Training population/weight mismatch")
        history = pd.read_csv(output / f"histories/{name}.csv")
        require(history.epoch.tolist() == list(range(1, reference["training"]["epochs"] + 1)), "Invalid history")
        initial = initialized_model(info["arm"], mapping, info["seed"])
        require(normalized_initial_hash(initial) == info["initial_normalized_sha256"] and sum(p.numel() for p in initial.parameters()) == info["parameter_count"], "Initial model mismatch")
        for policy, relative in info["checkpoints"].items():
            payload = torch.load(output / relative, map_location="cpu", weights_only=True)
            epoch = reference["training"]["epochs"] if policy == "last" else int(history.loc[history.validation_unweighted_loss.idxmin(), "epoch"])
            expected = {"arm": info["arm"], "seed": info["seed"], "mapping": mapping, "policy": policy, "epoch": epoch,
                        "pos_weight": weight, "threshold": .5, "training_user_keys": info["training_user_keys"],
                        "phase2_manifest_sha256": reference["phase2_manifest_sha256"], "graph_manifest_sha256": config["graph_manifest_sha256"]}
            require(all(payload[k] == v for k, v in expected.items()) and info["epochs"][policy] == epoch, "Checkpoint contract mismatch")
            model = initialized_model(info["arm"], mapping, info["seed"])
            for key in ("base.temporal_indices", "base.other_indices"):
                torch.testing.assert_close(payload["state_dict"][key], model.state_dict()[key], rtol=0, atol=0)
            model.load_state_dict(payload["state_dict"], strict=True)
            p, weighted, common = evaluate(model, data["validation"], graphs["validation"], reference["training"], weight)
            frame = prediction_registry(data["validation"], p, arm=info["arm"], seed=info["seed"], policy=policy, epoch=epoch)
            saved = pd.read_csv(output / f"predictions/{name}_{policy}_validation.csv", dtype={"user_key": str})
            pd.testing.assert_frame_equal(frame, saved, check_dtype=False, atol=1e-7, rtol=1e-6)
            rec = history.set_index("epoch").loc[epoch]
            np.testing.assert_allclose([weighted, common], [rec.validation_weighted_loss, rec.validation_unweighted_loss], atol=1e-7, rtol=1e-7)
            for row in cohort_metrics(frame):
                row.update({"parameter_count": info["parameter_count"], "validation_weighted_loss": weighted, "validation_unweighted_loss": common})
                rows.append(row)
    table = pd.read_csv(output / "validation_metrics.csv")
    keys = ["arm", "seed", "policy", "cohort"]
    pd.testing.assert_frame_equal(pd.DataFrame(rows).sort_values(keys).reset_index(drop=True), table.sort_values(keys).reset_index(drop=True), check_dtype=False, atol=1e-7, rtol=1e-6)
    summary, effects = summarize(table)
    pd.testing.assert_frame_equal(summary, pd.read_csv(output / "seed_summary.csv"), check_dtype=False, atol=1e-10, rtol=1e-10)
    pd.testing.assert_frame_equal(effects, pd.read_csv(output / "paired_effects.csv"), check_dtype=False, atol=1e-10, rtol=1e-10)
    for seed in config["seeds"]:
        paired = [v for v in manifest["runs"].values() if v["seed"] == seed]
        require({v["arm"] for v in paired} == set(ARMS) and len(paired) == 3, "Missing arm")
        loader = make_loader(data["train"], reference["training"], seed, "train")
        orders = []
        for _ in range(reference["training"]["epochs"]):
            seen = [i for _, _, ids in loader for i in ids.tolist()]
            require(sorted(seen) == list(range(len(data["train"]["y"]))), "Invalid epoch coverage")
            orders.append(hashlib.sha256(np.asarray(seen, dtype=np.int64).tobytes()).hexdigest())
        require(all(v["epoch_order_sha256"] == orders for v in paired), "Batch schedule mismatch")
        pair = [v for v in paired if v["arm"] != "temporal_anchor"]
        require(len({v["initial_normalized_sha256"] for v in pair}) == len({v["parameter_count"] for v in pair}) == 1, "Graph pairing mismatch")
    # The anchor must reproduce the prior temporal experiment, not drift silently.
    anchor = table[table.arm.eq("temporal_anchor") & table.cohort.eq("all")].sort_values(["seed", "policy"])
    prior = historical[historical.arm.eq("temporal_branch")].sort_values(["seed", "policy"])
    for metric in ("bot_f1", "bot_precision", "bot_recall", "roc_auc", "average_precision", "epoch"):
        np.testing.assert_allclose(anchor[metric], prior[metric], atol=1e-7, rtol=1e-7, err_msg="Temporal anchor drift: " + metric)
    parsed = json.loads((output / "provenance/parsed_inputs.json").read_text())
    require(not any("test" in Path(p).stem.lower() for p in parsed), "Test artifact parsed")
    require(all(digest(root / p) == sha for p, sha in parsed.items()), "Source changed")
    return {"training_runs": len(manifest["runs"]), "checkpoints_reloaded": len(manifest["runs"]) * 2,
            "validation_prediction_files": len(manifest["runs"]) * 2, "test_inputs_parsed": 0,
            "identity_only_assignment_projection": True, "temporal_anchor_reproduced": True,
            "protected_files_at_execution": manifest["protected_files_verified"]}
