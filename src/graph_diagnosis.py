"""Frozen score and representation diagnostics. No optimizer, fitting or test scoring."""
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.gcn_integration import load_inputs, initialized_model, registered_path
from src.checkpoint_weight_experiment import state_hash
from src.performance_diagnosis import Inputs, digest, metrics, preserved_hashes, require, write_json, plt, save_figure

IDENTITY = ["arm", "seed", "policy", "split"]
COHORTS = ("all", "graph_available", "graph_unavailable")
BRANCHES = ("other", "temporal", "graph")


def cohort(frame, name):
    return frame if name == "all" else frame[frame.has_graph.eq(int(name == "graph_available"))]


def quantiles(values, prefix):
    a = np.asarray(values, dtype=float)
    require(len(a) > 0 and np.isfinite(a).all(), "Invalid distribution")
    return {prefix + "_" + k: float(v) for k, v in zip(
        ("min", "q05", "q25", "median", "q75", "q95", "max", "mean"),
        [*np.quantile(a, [0, .05, .25, .5, .75, .95, 1]), a.mean()])}


def auc_pairs(frame):
    """Exact decomposition of overall AUC into within/cross-availability pairs."""
    bots, humans = frame[frame.label_binary.eq(1)], frame[frame.label_binary.eq(0)]
    denominator = len(bots) * len(humans)
    require(denominator > 0, "Both classes required")
    rows = []
    for b in (0, 1):
        for h in (0, 1):
            p = bots.loc[bots.has_graph.eq(b), "bot_probability"].to_numpy()
            q = humans.loc[humans.has_graph.eq(h), "bot_probability"].to_numpy()
            require(len(p) > 0 and len(q) > 0, "Empty availability/class group")
            value = float(((p[:, None] > q).astype(float) + .5 * (p[:, None] == q)).mean())
            count = len(p) * len(q)
            rows.append({"bot_has_graph": b, "human_has_graph": h, "pairs": count,
                         "pair_weight": count / denominator, "pair_auc": value,
                         "weighted_auc_contribution": value * count / denominator})
    np.testing.assert_allclose(sum(r["weighted_auc_contribution"] for r in rows),
                               roc_auc_score(frame.label_binary, frame.bot_probability), atol=1e-12, rtol=0)
    return rows


def reliability(frame, bins=10):
    ids = np.minimum((frame.bot_probability.to_numpy() * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        group = frame.iloc[np.flatnonzero(ids == b)]
        rows.append({"bin": b, "left": b / bins, "right": (b + 1) / bins, "n": len(group),
                     "mean_probability": float(group.bot_probability.mean()) if len(group) else None,
                     "bot_fraction": float(group.label_binary.mean()) if len(group) else None})
    require(sum(r["n"] for r in rows) == len(frame), "Reliability bins omit users")
    return rows


@torch.no_grad()
def representations(model, pop, graph):
    """Decompose the realized logit, conditioning on the observed ReLU activation mask.

    Signed terms are an algebraic decomposition, not causal feature attributions.
    Zero-graph predictions separately recompute the nonlinear activation mask.
    """
    model.eval()
    before = state_hash(model.state_dict())
    x = torch.tensor(pop["x"], dtype=torch.float32)
    other, temporal, mask = model.base.split_inputs(x)
    hs = {"other": model.base.other_encoder(other),
          "temporal": model.base.temporal_representation(temporal, mask)}
    if model.arm != "temporal_anchor":
        hs["graph"] = model.graph_representation(graph)
        require(torch.all(hs["graph"][~graph["available"]] == 0).item(), "Missing graph representation nonzero")
    first, last = model.base.classifier[0], model.base.classifier[3]
    joined = torch.cat(list(hs.values()), dim=1)
    pre = first(joined)
    activation = (pre > 0).float()
    logits = model.base.classifier(joined).flatten()
    expected = model(x, torch.arange(len(x)), graph).flatten()
    torch.testing.assert_close(logits, expected, atol=1e-4, rtol=1e-5)
    frame = pop["users"][["label_binary"]].join(pop["masks"][["has_graph", "has_graph_node", "has_temporal"]]).reset_index()
    frame["logit"] = logits.numpy()
    frame["bot_probability"] = logits.sigmoid().numpy()
    frame["prediction"] = (frame.bot_probability >= .5).astype(int)
    bias = ((first.bias * activation) @ last.weight.T).flatten() + last.bias.flatten()
    reconstructed = bias.clone()
    frame["bias_logit_term"] = bias.numpy()
    frame["active_head_units"] = activation.sum(1).numpy()
    offset = 0
    weighted_parts = {}
    for name, h in hs.items():
        width = h.shape[1]
        weighted = h @ first.weight[:, offset:offset + width].T
        term = ((weighted * activation) @ last.weight.T).flatten()
        weighted_parts[name] = weighted
        frame[name + "_l2"] = h.norm(dim=1).numpy()
        frame[name + "_rms"] = h.square().mean(1).sqrt().numpy()
        frame[name + "_weighted_rms"] = weighted.square().mean(1).sqrt().numpy()
        frame[name + "_logit_term"] = term.numpy()
        frame[name + "_zero_fraction"] = (h == 0).float().mean(1).numpy()
        reconstructed += term
        offset += width
    for name in set(BRANCHES) - set(hs):
        for suffix in ("l2", "rms", "weighted_rms", "logit_term", "zero_fraction"):
            frame[name + "_" + suffix] = np.nan  # Absent architecture branch, not a measured zero.
    frame["logit_reconstruction_error"] = (reconstructed - logits).abs().numpy()
    torch.testing.assert_close(reconstructed, logits, atol=2e-4, rtol=1e-5)
    if "graph" in hs:
        zero_joined = torch.cat([hs["other"], hs["temporal"], torch.zeros_like(hs["graph"])], dim=1)
        zero_logits = model.base.classifier(zero_joined).flatten()
        frame["zero_graph_logit"] = zero_logits.numpy()
        frame["zero_graph_probability"] = zero_logits.sigmoid().numpy()
        frame["graph_logit_effect"] = (logits - zero_logits).numpy()
        frame["zero_graph_prediction_flip"] = ((zero_logits >= 0) != (logits >= 0)).numpy()
        torch.testing.assert_close(zero_logits[~graph["available"]], logits[~graph["available"]], atol=0, rtol=0)
    else:
        frame["zero_graph_logit"] = np.nan
        frame["zero_graph_probability"] = np.nan
        frame["graph_logit_effect"] = np.nan
        frame["zero_graph_prediction_flip"] = pd.Series([pd.NA] * len(frame), dtype="boolean")
    require(before == state_hash(model.state_dict()), "Diagnostic inference mutated checkpoint state")
    # Preserve every float32 value exactly as a float64 before CSV serialization
    # and summary arithmetic; pandas otherwise emits short float32 decimals.
    floating = frame.select_dtypes(include="floating").columns
    frame[floating] = frame[floating].astype(np.float64)
    return frame


def tables(frame, bins):
    summaries, distributions, calibration, pairs, magnitudes, perturbations = [], [], [], [], [], []
    for values, f in frame.groupby(IDENTITY, sort=True):
        identity = dict(zip(IDENTITY, values))
        for row in auc_pairs(f):
            pairs.append({**identity, **row})
        for name in COHORTS:
            g = cohort(f, name)
            y, p = g.label_binary.to_numpy(), g.bot_probability.to_numpy()
            rel = reliability(g, bins)
            ece = sum(r["n"] / len(g) * abs(r["mean_probability"] - r["bot_fraction"]) for r in rel if r["n"])
            # Stable BCE computed directly from stored, un-clipped logits.
            loss = np.logaddexp(0, g.logit.to_numpy()) - y * g.logit.to_numpy()
            summaries.append({**identity, "cohort": name, "n": len(g), "bots": int(y.sum()),
                **metrics(y, p, .5), "roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
                "brier": float(np.mean((p - y) ** 2)), "unweighted_log_loss": float(loss.mean()), "ece_10_bins": ece,
                "saturated_zero_count": int((p == 0).sum()), "saturated_one_count": int((p == 1).sum())})
            calibration.extend({**identity, "cohort": name, **r} for r in rel)
            for label, cls in g.groupby("label_binary"):
                distributions.append({**identity, "cohort": name, "label": int(label), "n": len(cls),
                                      **quantiles(cls.bot_probability, "probability"), **quantiles(cls.logit, "logit")})
                if identity["arm"] != "temporal_anchor":
                    original = cls.bot_probability.ge(.5)
                    zero = cls.zero_graph_probability.ge(.5)
                    perturbations.append({**identity, "cohort": name, "label": int(label), "n": len(cls),
                        **quantiles(cls.graph_logit_effect, "original_minus_zero_graph_logit"),
                        **quantiles(cls.bot_probability - cls.zero_graph_probability, "original_minus_zero_graph_probability"),
                        "flipped_to_bot": int((~original & zero).sum()), "flipped_to_human": int((original & ~zero).sum()),
                        "original_errors": int(original.ne(label).sum()), "zero_graph_errors": int(zero.ne(label).sum())})
                for branch in BRANCHES:
                    if cls[branch + "_rms"].isna().all():
                        continue
                    row = {**identity, "cohort": name, "label": int(label), "branch": branch, "n": len(cls)}
                    for suffix in ("rms", "l2", "weighted_rms", "logit_term", "zero_fraction"):
                        row.update(quantiles(cls[branch + "_" + suffix], suffix))
                    magnitudes.append(row)
    return {"score_metrics": pd.DataFrame(summaries), "score_distributions": pd.DataFrame(distributions),
            "reliability_bins": pd.DataFrame(calibration), "auc_pair_decomposition": pd.DataFrame(pairs),
            "representation_distributions": pd.DataFrame(magnitudes), "inference_perturbation_summary": pd.DataFrame(perturbations)}


def raw_degrees(graphs):
    rows = []
    for split, graph in graphs.items():
        x = graph["data"].x.numpy()
        np.testing.assert_array_equal(x[:, 0], x[:, 1])
        np.testing.assert_array_equal(x[:, 2], 2 * x[:, 0])
        idx = graph["canonical_node_index"][graph["available"]].numpy()
        support = ~graph["provenance"].is_same_split_gold.to_numpy()
        for name, values in (("all_observed_nodes", x[:, 0]), ("non_gold_support_nodes", x[support, 0]),
                             ("graph_available_canonical_users", x[idx, 0])):
            rows.append({"split": split, "population": name, "n": len(values), **quantiles(values, "degree")})
    return pd.DataFrame(rows)


def figures(frame, output):
    selected = frame[frame.split.eq("validation") & frame.policy.eq("best_unweighted_validation_loss")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for r, arm in enumerate(("degree_mlp", "gcn")):
        for c, seed in enumerate((42, 43, 44)):
            group = selected[selected.arm.eq(arm) & selected.seed.eq(seed)]
            for (available, label), g in group.groupby(["has_graph", "label_binary"]):
                p = np.sort(g.bot_probability)
                axes[r, c].step(p, np.arange(1, len(p) + 1) / len(p), where="post",
                    label=f"{'Graph' if available else 'No graph'} / {'Bot' if label else 'Human'} (n={len(p)})")
            axes[r, c].axvline(.5, color="black", linestyle=":")
            axes[r, c].set(title=f"{arm}, seed {seed}", xlabel="Bot probability", ylabel="Empirical cumulative fraction", xlim=(0, 1), ylim=(0, 1))
            axes[r, c].legend(fontsize=7)
    save_figure(fig, output / "figures/score_ecdf.png")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for r, arm in enumerate(("degree_mlp", "gcn")):
        for c, seed in enumerate((42, 43, 44)):
            g = selected[selected.arm.eq(arm) & selected.seed.eq(seed) & selected.has_graph.eq(1)]
            axes[r, c].boxplot([np.log1p(g[b + "_weighted_rms"]) for b in BRANCHES], tick_labels=BRANCHES)
            axes[r, c].set(title=f"{arm}, seed {seed}", ylabel="log(1 + weighted RMS)")
    fig.suptitle("Graph-available validation users; first fusion-layer contributions (n=42)")
    save_figure(fig, output / "figures/weighted_representation_scale.png")


def run(root, config_path="configs/graph_diagnosis_v1.json"):
    root = Path(root).resolve()
    reader = Inputs(root)
    reader.allow(root / config_path, digest(root / config_path))
    config = reader.read(root / config_path)
    require(config["scope"] == "frozen_checkpoint_diagnosis_no_training_no_calibration_fit_no_threshold_selection_no_test_no_promotion"
            and config["threshold"] == .5 and config["reliability_bins"] == 10 and config["splits"] == ["train", "validation"]
            and config["policies"] == ["best_unweighted_validation_loss", "last"], "Diagnostic scope changed")
    before = preserved_hashes(root, config["output_root"])
    output = root / config["output_root"] / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    for name in ("provenance", "figures", "keyed"):
        (output / name).mkdir()
    try:
        source_path = root / config["source_manifest"]
        reader.allow(source_path, config["source_manifest_sha256"])
        source = reader.read(source_path)
        require(source["complete"] and source["schema_version"] == "gcn_integration_v1", "Invalid frozen experiment")
        source_config = reader.registered(source_path.parent, source, "provenance/resolved_config.json")
        data, mapping, graphs, _, _, coverage, shared = load_inputs(root, source_config, reader)
        require(mapping == source["feature_mapping"], "Feature mapping changed")
        frames, counts = [], []
        for name, info in source["runs"].items():
            for policy in config["policies"]:
                path = registered_path(reader, source_path.parent, source, info["checkpoints"][policy])
                payload = torch.load(path, weights_only=True, map_location="cpu")
                require(payload["mapping"] == mapping and payload["arm"] == info["arm"] and payload["seed"] == info["seed"]
                        and payload["policy"] == policy and payload["threshold"] == .5
                        and payload["epoch"] == info["epochs"][policy]
                        and payload["training_user_keys"] == data["train"]["users"].index.tolist(), "Checkpoint contract mismatch")
                model = initialized_model(info["arm"], mapping, info["seed"])
                model.load_state_dict(payload["state_dict"], strict=True)
                for split in config["splits"]:
                    f = representations(model, data[split], graphs[split])
                    for k, v in {"arm": info["arm"], "seed": info["seed"], "policy": policy, "epoch": payload["epoch"], "split": split}.items():
                        f[k] = v
                    require(f.user_key.is_unique and f.user_key.tolist() == data[split]["users"].index.tolist(), "Identity order changed")
                    if split == "validation":
                        saved = reader.registered(source_path.parent, source, f"predictions/{name}_{policy}_validation.csv")
                        require(f.user_key.tolist() == saved.user_key.tolist() and f.label_binary.tolist() == saved.label_binary.tolist(), "Saved prediction identities differ")
                        np.testing.assert_allclose(f.bot_probability, saved.bot_probability, atol=1e-6, rtol=1e-5)
                        np.testing.assert_array_equal(f.prediction, saved.prediction)
                    f.to_csv(output / f"keyed/{name}_{policy}_{split}.csv", index=False)
                    counts.append({"arm": info["arm"], "seed": info["seed"], "policy": policy, "split": split, "n": len(f),
                                   "max_logit_reconstruction_error": float(f.logit_reconstruction_error.max())})
                    frames.append(f)
                print(f"Diagnosed {name}, {policy}", flush=True)
        frame = pd.concat(frames, ignore_index=True)
        results = tables(frame, config["reliability_bins"])
        for name, table in results.items():
            table.to_csv(output / (name + ".csv"), index=False)
        raw_degrees(graphs).to_csv(output / "raw_degree_distributions.csv", index=False)
        coverage.to_csv(output / "graph_coverage.csv", index=False)
        pd.DataFrame(counts).to_csv(output / "verification_counts.csv", index=False)
        figures(frame, output)
        write_json(output / "provenance/resolved_config.json", config)
        write_json(output / "provenance/parsed_inputs.json", reader.accessed)
        write_json(output / "provenance/protected_files.json", before)
        write_json(output / "provenance/protocol.json", {"assignment_columns": ["user_key", "split"],
            "test_labels_features_graphs_predictions_parsed": False, "training_performed": False, "calibration_fitted": False,
            "threshold_selected": False, "inference_zero_graph_role": "diagnostic perturbation; not a retrained ablation",
            "contribution_interpretation": "exact realized-logit decomposition conditional on observed ReLU mask; not causal",
            "train_scores_role": "in-sample diagnostics only", "shared_support_nodes": shared})
        for file in ("src/graph_diagnosis.py", "src/gcn_integration.py", "src/temporal_statistics.py", "src/graph_baseline.py",
                     "src/checkpoint_weight_experiment.py", "src/performance_diagnosis.py", "src/baselines.py", "scripts/run_graph_diagnosis.py"):
            (output / "provenance" / Path(file).name).write_bytes((root / file).read_bytes())
        write_json(output / "provenance/environment.json", {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "threads": torch.get_num_threads(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)})
    finally:
        require(before == preserved_hashes(root, config["output_root"]), "Frozen inputs or existing files changed")
    write_json(output / "manifest.json", {"schema_version": config["schema_version"], "complete": True,
        "source_manifest_sha256": config["source_manifest_sha256"], "checkpoint_count": 18, "keyed_files": len(frames),
        "protected_files_verified": len(before), "promoted_model": None,
        "artifacts": {p.relative_to(output).as_posix(): {"sha256": digest(p)} for p in sorted(output.rglob("*")) if p.is_file()}})
    return output, results


def verify(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    require(manifest["complete"] and manifest["promoted_model"] is None, "Incomplete diagnosis")
    for rel, meta in manifest["artifacts"].items():
        require(digest(output / rel) == meta["sha256"], "Diagnostic artifact hash mismatch")
    parsed = json.loads((output / "provenance/parsed_inputs.json").read_text())
    require(not any("test" in Path(p).stem.lower() for p in parsed), "Test artifact parsed")
    require(all(digest(root / p) == sha for p, sha in parsed.items()), "Source artifact changed")
    paths = sorted((output / "keyed").glob("*.csv"))
    require(len(paths) == manifest["keyed_files"] == 36, "Missing keyed diagnostics")
    frame = pd.concat([pd.read_csv(p, dtype={"user_key": str}, float_precision="round_trip") for p in paths], ignore_index=True)
    require(not frame.duplicated(IDENTITY + ["user_key"]).any(), "Duplicate diagnostic identity")
    for values, g in frame.groupby(IDENTITY):
        require(len(g) == (671 if values[-1] == "train" else 144), "Diagnostic cohort changed")
        if values[0] != "temporal_anchor":
            absent = g.has_graph.eq(0)
            require(g.loc[absent, "graph_rms"].eq(0).all() and g.loc[absent, "graph_logit_effect"].eq(0).all(), "Missing graph effect nonzero")
    for name, expected in tables(frame, 10).items():
        saved = pd.read_csv(output / (name + ".csv"))
        pd.testing.assert_frame_equal(expected.reset_index(drop=True), saved.reset_index(drop=True), check_dtype=False, atol=1e-9, rtol=1e-8)
    return {"artifacts_verified": len(manifest["artifacts"]), "checkpoints_diagnosed": manifest["checkpoint_count"],
            "keyed_files_verified": len(paths), "test_inputs_parsed": 0, "training_performed": False,
            "max_logit_reconstruction_error": float(frame.logit_reconstruction_error.max())}
