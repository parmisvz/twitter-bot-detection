"""Approved Notebook-05 conditions, including equivalence with its archived model."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

from src.graph_baseline import GRAPH_MASKS, build_graph, training_class_weights
from src.graph_tabular import GCNTabularClassifier, attach_tabular, predictions, train_model, validate_policy

ROOT = Path(__file__).resolve().parents[1]


def fixture(prefix="train"):
    keys = [prefix + "_a", prefix + "_b", prefix + "_context", prefix + "_absent"]
    users = pd.DataFrame({"label_binary": [0, 1, 0, 1]}, index=pd.Index(keys, name="user_key"))
    masks = pd.DataFrame(1, index=users.index, columns=GRAPH_MASKS)
    masks.loc[keys[2:], ["has_graph", "has_graph_owner", "has_graph_owner_record"]] = 0
    masks.loc[keys[3], "has_graph_node"] = 0
    masks.loc[keys[1], ["has_temporal", "has_tweet_text"]] = 0
    edges = pd.DataFrame({"source_key": [keys[0], keys[1], keys[0]],
                          "target_key": ["support", "support", keys[2]], "relation": ["follows"] * 3})
    graph = build_graph(edges, users, masks)
    columns = [f"f{i:02d}" for i in range(40)]
    # Include an actually attached all-zero row, to prove availability cannot be inferred from values.
    tabular = pd.DataFrame(np.arange(160).reshape(4, 40) / 100, index=users.index, columns=columns)
    tabular.loc[keys[0]] = 0
    return graph, tabular, columns


class GraphTabularTests(unittest.TestCase):
    def test_manifest_order_key_alignment_and_unpopulated_slots(self):
        graph, tabular, columns = fixture()
        joined = attach_tabular(graph, tabular.iloc[::-1, ::-1], columns, "train")
        x = joined["data"].x
        nodes = joined["mapping"].set_index("user_key").node_index
        self.assertEqual(x.shape[1], 43)
        torch.testing.assert_close(x[:, :3], graph["data"].x, rtol=0, atol=0)
        for key in tabular.index.intersection(nodes.index):
            np.testing.assert_array_equal(x[nodes[key], 3:].numpy(), tabular.loc[key, columns].to_numpy(dtype=np.float32))
        self.assertTrue(torch.equal(x[nodes["support"], 3:], torch.zeros(40)))
        self.assertFalse(joined["data"].tabular_attached[nodes["support"]].item())
        self.assertTrue(joined["data"].tabular_attached[nodes["train_a"]].item())
        self.assertFalse(joined["data"].eligible_mask[nodes["train_context"]].item())
        self.assertTrue(joined["data"].tabular_attached[nodes["train_context"]].item())
        self.assertNotIn("train_absent", nodes.index)
        provenance = joined["node_provenance"].set_index("user_key")
        self.assertEqual(provenance.loc["support", "tabular_provenance"], "non_gold_support_unpopulated_slots")
        self.assertTrue(pd.isna(provenance.loc["support", "has_graph"]))

    def test_heldout_rows_are_rejected_and_heldout_changes_cannot_change_train_input(self):
        graph, tabular, columns = fixture()
        val_graph, val_tabular, _ = fixture("val")
        with self.assertRaisesRegex(ValueError, "population mismatch"):
            attach_tabular(graph, pd.concat([tabular, val_tabular]), columns, "train")
        original = attach_tabular(graph, tabular, columns, "train")["data"].x.clone()
        val_tabular[:] = 1e6
        attach_tabular(val_graph, val_tabular, columns, "validation")
        after = attach_tabular(graph, tabular, columns, "train")["data"].x
        torch.testing.assert_close(original, after, rtol=0, atol=0)

    def test_missing_graph_registry_and_temporal_text_missingness(self):
        graph, tabular, columns = fixture()
        graph = attach_tabular(graph, tabular, columns, "train")
        frame = predictions(GCNTabularClassifier(), graph, "train", .5).set_index("user_key")
        self.assertEqual(len(frame), 4)
        self.assertEqual(frame.loc["train_b", "status"], "predicted")  # No text/timing filtering.
        self.assertTrue(frame.loc[["train_context", "train_absent"], "prediction"].isna().all())
        self.assertTrue(frame.loc[["train_context", "train_absent"], "bot_probability"].isna().all())
        self.assertEqual(frame.loc["train_context", "tabular_attached"], 1)
        self.assertEqual(frame.loc["train_absent", "tabular_attached"], 0)
        self.assertEqual(frame.loc["train_absent", "tabular_provenance"], "canonical_tabular_available_but_no_graph_node")

    def test_architecture_matches_archived_notebook_logits(self):
        notebook = json.loads((ROOT / "analysis/archive/05_graph_tabular_pre_v2.ipynb").read_text(encoding="utf-8"))
        definitions = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                for node in ast.parse("".join(cell["source"])).body:
                    if isinstance(node, ast.ClassDef) and node.name == "GCNTabularClassifier":
                        definitions.append(node)
        self.assertEqual(len(definitions), 1)
        scope = {"nn": nn, "F": F, "GCNConv": GCNConv}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "archived_model", "exec"), scope)
        legacy = scope["GCNTabularClassifier"]().eval()
        current = GCNTabularClassifier().eval()
        current.load_state_dict(legacy.state_dict(), strict=True)
        graph, tabular, columns = fixture()
        data = attach_tabular(graph, tabular, columns, "train")["data"]
        with torch.no_grad():
            torch.testing.assert_close(current(data.x, data.edge_index), legacy(data.x, data.edge_index), rtol=0, atol=0)
        self.assertFalse(any(isinstance(m, (nn.Dropout, nn.BatchNorm1d)) for m in current.modules()))

    def test_seeded_training_and_independent_checkpoint_snapshots(self):
        graph, tabular, columns = fixture()
        data = attach_tabular(graph, tabular, columns, "train")["data"]
        config = json.loads((ROOT / "configs/graph_tabular_v2.json").read_text())
        config["training"]["epochs"] = 3  # Synthetic unit test, not the immutable research run policy.
        states, history, info = train_model(data, data, config)
        repeated, repeated_history, repeated_info = train_model(data, data, config)
        pd.testing.assert_frame_equal(history, repeated_history)
        self.assertEqual(info, repeated_info)
        for variant in states:
            for key in states[variant]:
                torch.testing.assert_close(states[variant][key], repeated[variant][key], rtol=0, atol=0)
        best = deepcopy(states["best_validation_loss"])
        for key in states["last"]:
            self.assertNotEqual(states["last"][key].data_ptr(), states["best_validation_loss"][key].data_ptr())
            states["last"][key].add_(10)
            torch.testing.assert_close(states["best_validation_loss"][key], best[key], rtol=0, atol=0)
        model = GCNTabularClassifier().eval()
        model.load_state_dict(best)
        with torch.no_grad():
            loss = nn.CrossEntropyLoss(weight=training_class_weights(data)[0])(
                model(data.x, data.edge_index)[data.eligible_mask], data.y[data.eligible_mask]).item()
        self.assertAlmostEqual(loss, history.validation_loss.min(), places=7)

    def test_unapproved_policy_changes_fail(self):
        config = json.loads((ROOT / "configs/graph_tabular_v2.json").read_text())
        validate_policy(config)
        for key, value in [("indicator_in_model_input", True), ("tabular_attachment", "all_support"),
                           ("expected_comparison_counts", {"validation": 42, "test": 40})]:
            changed = deepcopy(config)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_policy(changed)


if __name__ == "__main__":
    unittest.main()
