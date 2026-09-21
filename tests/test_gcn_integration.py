from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.gcn_integration import initialized_model, normalized_initial_hash, train_one, evaluate, cohort_metrics, summarize
from src.temporal_statistics import feature_mapping
from src.graph_baseline import validate_edges

ROOT = Path(__file__).resolve().parents[1]


class TestGCNIntegration(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        config = json.loads((ROOT / "configs/temporal_statistics_v1.json").read_text())
        order = json.loads((ROOT / "configs/preprocessing_v2.json").read_text())["feature_order"]
        self.mapping = feature_mapping(order, config["temporal_features"])
        self.graph = {"data": Data(x=torch.tensor([[1., 1., 2.], [2., 2., 4.], [1., 1., 2.]]),
                                  edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])),
                      "canonical_node_index": torch.tensor([0, 1, -1, 2]), "available": torch.tensor([True, True, False, False])}
        rng = np.random.default_rng(7)
        x = rng.normal(size=(4, 47)).astype(np.float32)
        x[:, 40:] = 1
        self.pop = {"x": x, "y": np.array([0, 1, 0, 1], dtype=np.float32)}

    def test_paired_initialization_and_rng(self):
        a = initialized_model("degree_mlp", self.mapping, 42)
        rng_a = torch.get_rng_state().clone()
        b = initialized_model("gcn", self.mapping, 42)
        self.assertEqual(normalized_initial_hash(a), normalized_initial_hash(b))
        torch.testing.assert_close(rng_a, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))

    def test_topology_effect_and_missing_graph_invariance(self):
        x = torch.tensor(self.pop["x"])
        disconnected = deepcopy(self.graph)
        disconnected["data"].edge_index = torch.empty((2, 0), dtype=torch.long)
        for arm in ("degree_mlp", "gcn"):
            model = initialized_model(arm, self.mapping, 42).eval()
            with torch.no_grad():
                rep = model.graph_representation(self.graph)
                self.assertTrue(torch.equal(rep[2:], torch.zeros(2, 32)))
                original = model(x, torch.arange(4), self.graph)
                altered = model(x, torch.arange(4), disconnected)
            torch.testing.assert_close(original[2:], altered[2:], rtol=0, atol=0)
            if arm == "degree_mlp":
                torch.testing.assert_close(original, altered, rtol=0, atol=0)
            else:
                self.assertFalse(torch.equal(original[:2], altered[:2]))
            model.zero_grad()
            model(x, torch.arange(4), self.graph).sum().backward()
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.graph1.parameters()), 0)

    def test_cross_split_gold_edges_rejected(self):
        edges = pd.DataFrame({"source_key": ["trainuser"], "target_key": ["heldout"], "relation": ["follows"]})
        owners = edges.assign(owner_key="trainuser")
        users = pd.DataFrame(index=pd.Index(["trainuser"], name="user_key"))
        assignment = pd.DataFrame({"split": ["train", "test"]}, index=["trainuser", "heldout"])
        with self.assertRaisesRegex(ValueError, "cross-split gold"):
            validate_edges(edges, owners, users, assignment, "train")

    def test_entirely_graph_unavailable_population(self):
        empty = {"data": Data(x=torch.empty((0, 3)), edge_index=torch.empty((2, 0), dtype=torch.long)),
                 "canonical_node_index": torch.full((4,), -1, dtype=torch.long),
                 "available": torch.zeros(4, dtype=torch.bool)}
        populated_but_unavailable = deepcopy(self.graph)
        populated_but_unavailable["available"][:] = False
        for arm in ("degree_mlp", "gcn"):
            model = initialized_model(arm, self.mapping, 42).eval()
            with torch.no_grad():
                rep = model.graph_representation(empty)
                torch.testing.assert_close(rep, torch.zeros(4, 32), rtol=0, atol=0)
                x = torch.tensor(self.pop["x"])
                output = model(x, torch.arange(4), empty)
                reference = model(x, torch.arange(4), populated_but_unavailable)
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
                self.assertEqual(tuple(output.shape), (4, 1))
                self.assertTrue(torch.isfinite(output).all())

    def test_metrics_include_ranking_scores_and_matched_cohorts(self):
        rows = []
        for arm in ("temporal_anchor", "degree_mlp", "gcn"):
            frame = pd.DataFrame({"label_binary": [0, 1, 0, 1], "has_graph": [1, 1, 0, 0],
                                  "bot_probability": [.1, .8, .2, .9], "arm": arm, "seed": 42,
                                  "policy": "best_unweighted_validation_loss", "epoch": 1})
            measured = cohort_metrics(frame)
            self.assertEqual([r["n"] for r in measured], [4, 2, 2])
            for row in measured:
                self.assertEqual(row["roc_auc"], 1.)
                self.assertEqual(row["average_precision"], 1.)
            rows.extend(measured)
        summary, effects = summarize(pd.DataFrame(rows))
        self.assertEqual(len(summary), 9)
        self.assertTrue(effects.delta_bot_f1.eq(0).all())

    def test_training_repeatability_snapshot_and_reload(self):
        data = {"train": self.pop, "validation": self.pop}
        graphs = {"train": self.graph, "validation": self.graph}
        spec = {"epochs": 2, "batch_size": 2, "lr": .001, "weight_decay": .0001}
        a, h, info = train_one(data, graphs, self.mapping, spec, "gcn", 42, 1.)
        b, h2, info2 = train_one(data, graphs, self.mapping, spec, "gcn", 42, 1.)
        pd.testing.assert_frame_equal(h, h2)
        self.assertEqual(info, info2)
        best = a["best_unweighted_validation_loss"]
        for k in best:
            torch.testing.assert_close(best[k], b["best_unweighted_validation_loss"][k], rtol=0, atol=0)
            self.assertNotEqual(best[k].data_ptr(), a["last"][k].data_ptr())
        model = initialized_model("gcn", self.mapping, 42)
        model.load_state_dict(best)
        p, _, loss = evaluate(model, self.pop, self.graph, spec, 1.)
        self.assertEqual(len(p), 4)
        self.assertAlmostEqual(loss, h.validation_unweighted_loss.min(), places=7)
        model.train()
        x = torch.tensor(self.pop["x"])
        logits = model(x, torch.arange(4), self.graph)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor(self.pop["y"]).reshape(-1, 1))
        loss.backward()
        # No graph tensor is an optimization parameter or mutated by forward.
        self.assertFalse(self.graph["data"].x.requires_grad)


if __name__ == "__main__":
    unittest.main()
