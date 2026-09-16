"""Graph protocol and persistence regression tests; synthetic data only."""
from copy import deepcopy
import unittest

import numpy as np
import pandas as pd
import torch

from src.graph_baseline import (GRAPH_MASKS, GCNClassifier, build_graph, graph_predictions,
                                train_gcn, training_class_weights, validate_edges)


def fixture(prefix="train"):
    keys = [prefix + "_a", prefix + "_b", prefix + "_node", prefix + "_absent"]
    users = pd.DataFrame({"label_binary": [0, 1, 0, 1]}, index=pd.Index(keys, name="user_key"))
    masks = pd.DataFrame(1, index=users.index, columns=GRAPH_MASKS)
    masks.loc[keys[2:], ["has_graph", "has_graph_owner", "has_graph_owner_record"]] = 0
    masks.loc[keys[3], "has_graph_node"] = 0
    edges = pd.DataFrame({"source_key": [keys[0], keys[1], keys[0], "support"],
                          "target_key": ["support", "support", keys[2], keys[0]],
                          "relation": ["follows"] * 4})
    owners = edges.copy()
    owners["owner_key"] = [keys[0], keys[1], keys[0], keys[0]]
    return edges, owners, users, masks


class GraphProtocolTests(unittest.TestCase):
    def test_cross_split_gold_and_invalid_owner_are_rejected(self):
        edges, owners, users, masks = fixture()
        assignment = pd.DataFrame({"split": ["train"] * len(users) + ["test"]},
                                   index=[*users.index, "test_gold"])
        validate_edges(edges, owners, users, assignment, "train")
        bad_edges, bad_owners = edges.copy(), owners.copy()
        bad_edges.loc[0, "target_key"] = "test_gold"
        bad_owners.loc[0, "target_key"] = "test_gold"
        with self.assertRaisesRegex(ValueError, "cross-split"):
            validate_edges(bad_edges, bad_owners, users, assignment, "train")
        bad_owners = owners.copy()
        bad_owners.loc[0, "owner_key"] = "test_gold"
        with self.assertRaisesRegex(ValueError, "owner outside"):
            validate_edges(edges, bad_owners, users, assignment, "train")

    def test_degree_multiplicity_and_missing_graph_users_are_explicit(self):
        edges, _, users, masks = fixture()
        graph = build_graph(edges, users, masks)
        data = graph["data"]
        self.assertEqual(data.edge_index.shape[1], 8)
        self.assertEqual(torch.unique(data.edge_index, dim=1).shape[1], 6)
        torch.testing.assert_close(data.x[:, 0], data.x[:, 1])
        torch.testing.assert_close(data.x[:, 2], 2 * data.x[:, 0])
        self.assertNotIn("train_absent", set(graph["mapping"].user_key))
        registry = graph_predictions(GCNClassifier(), graph, "train", .5).set_index("user_key")
        self.assertEqual(len(registry), len(users))
        self.assertEqual(registry.loc["train_node", "graph_evidence"], "node_only")
        self.assertEqual(registry.loc["train_absent", "graph_evidence"], "no_permitted_node")
        self.assertTrue(registry.loc[["train_node", "train_absent"], "bot_probability"].isna().all())
        self.assertTrue(registry.loc[["train_node", "train_absent"], "prediction"].isna().all())

    def test_heldout_labels_do_not_change_training_weights_or_training_step(self):
        edges, _, users, masks = fixture()
        training = build_graph(edges, users, masks)["data"]
        ve, _, vu, vm = fixture("val")
        validation = build_graph(ve, vu, vm)["data"]
        weights, counts = training_class_weights(training)
        config = {"seed": 42, "torch_threads": 1, "device": "cpu",
                  "architecture": {"input_dim": 3, "hidden_dim": 8, "second_dim": 4, "dropout": .3},
                  "training": {"epochs": 1, "lr": .01, "weight_decay": .0005, "selection": "best_validation_loss"}}
        a, _, _ = train_gcn(training, validation, config)
        validation.y[validation.eligible_mask] = 1 - validation.y[validation.eligible_mask]
        b, _, _ = train_gcn(training, validation, config)
        for key in a["last"]:
            torch.testing.assert_close(a["last"][key], b["last"][key], rtol=0, atol=0)
        unchanged, unchanged_counts = training_class_weights(training)
        torch.testing.assert_close(weights, unchanged)
        torch.testing.assert_close(counts, unchanged_counts)
        # Training graph label slots for unsupported nodes must not affect weighting.
        training.y[~training.eligible_mask] = 1
        torch.testing.assert_close(weights, training_class_weights(training)[0])

    def test_best_state_is_independent_and_saved_epoch_loss_matches(self):
        edges, _, users, masks = fixture()
        training = build_graph(edges, users, masks)["data"]
        config = {"seed": 42, "torch_threads": 1, "device": "cpu",
                  "architecture": {"input_dim": 3, "hidden_dim": 8, "second_dim": 4, "dropout": .3},
                  "training": {"epochs": 4, "lr": .01, "weight_decay": .0005, "selection": "best_validation_loss"}}
        states, history, info = train_gcn(training, training, config)
        snapshot = deepcopy(states["best_validation_loss"])
        for key in states["last"]:
            self.assertNotEqual(states["last"][key].data_ptr(), states["best_validation_loss"][key].data_ptr())
            states["last"][key].add_(10)
            torch.testing.assert_close(states["best_validation_loss"][key], snapshot[key], rtol=0, atol=0)
        model = GCNClassifier(**config["architecture"])
        model.load_state_dict(snapshot)
        model.eval()
        with torch.no_grad():
            loss = torch.nn.CrossEntropyLoss(weight=training_class_weights(training)[0])(
                model(training)[training.eligible_mask], training.y[training.eligible_mask]).item()
        self.assertAlmostEqual(loss, history.validation_loss.min(), places=7)
        self.assertEqual(info["best_epoch"], int(history.loc[history.validation_loss.idxmin(), "epoch"]))

    def test_validation_graph_does_not_reuse_training_normalization(self):
        edges, _, users, masks = fixture()
        graph = build_graph(edges, users, masks)["data"]
        altered = graph.clone()
        altered.edge_index = graph.edge_index[:, :2]
        torch.manual_seed(42)
        model = GCNClassifier().eval()
        with torch.no_grad():
            model(graph)
            after_training_graph = model(altered)
            fresh = GCNClassifier().eval()
            fresh.load_state_dict(model.state_dict())
            direct = fresh(altered)
        torch.testing.assert_close(after_training_graph, direct, rtol=0, atol=0)
        self.assertFalse(model.conv1.cached)
        self.assertFalse(model.conv2.cached)


if __name__ == "__main__":
    unittest.main()
