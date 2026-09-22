from copy import deepcopy
import unittest
import numpy as np
import pandas as pd
import torch

from tests import test_gcn_integration as fixtures
from src.graph_scaling import fit_scale, scale_graphs
from src.gcn_integration import train_one


class TestGraphScaling(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.TestGCNIntegration()
        fixture.setUp()
        self.fixture = fixture
        self.graph = deepcopy(fixture.graph)
        self.graph["users"] = pd.DataFrame(index=pd.Index(["a", "b", "c", "d"], name="user_key"))
        self.graph["data"].y = torch.tensor([0, 1, -1])
        self.graph["data"].eligible_mask = torch.tensor([True, True, False])

    def test_fit_uses_only_available_canonical_training_nodes(self):
        before = self.graph["data"].x.clone()
        fit = fit_scale(self.graph)
        self.assertEqual(fit["fit_user_keys"], ["a", "b"])
        self.assertAlmostEqual(fit["scalar"], np.sqrt(5.))
        changed = deepcopy(self.graph)
        changed["data"].x[2] = 1e8
        changed["data"].y[:] = 1
        self.assertEqual(fit, fit_scale(changed))
        torch.testing.assert_close(before, self.graph["data"].x, atol=0, rtol=0)

    def test_same_transform_all_nodes_no_topology_or_mask_changes(self):
        graphs = {"train": self.graph, "validation": deepcopy(self.graph)}
        graphs["validation"]["data"].x *= 100
        scalar = fit_scale(graphs["train"])["scalar"]
        scaled = scale_graphs(graphs, scalar)
        self.assertAlmostEqual(scaled["train"]["data"].x[:2].double().square().mean().sqrt().item(), 1., places=6)
        for split in graphs:
            torch.testing.assert_close(scaled[split]["data"].x, graphs[split]["data"].x / scalar, atol=0, rtol=0)
            torch.testing.assert_close(scaled[split]["data"].edge_index, graphs[split]["data"].edge_index, atol=0, rtol=0)
            torch.testing.assert_close(scaled[split]["available"], graphs[split]["available"], atol=0, rtol=0)
            self.assertNotEqual(scaled[split]["data"].x.data_ptr(), graphs[split]["data"].x.data_ptr())
        with self.assertRaises(ValueError):
            scale_graphs({"test": self.graph}, scalar)

    def test_paired_initialization_and_batch_orders_with_changed_input_scale(self):
        graphs = {"train": self.graph, "validation": deepcopy(self.graph)}
        scaled = scale_graphs(graphs, fit_scale(self.graph)["scalar"])
        data = {"train": self.fixture.pop, "validation": self.fixture.pop}
        spec = {"epochs": 2, "batch_size": 2, "lr": .001, "weight_decay": .0001}
        for arm in ("degree_mlp", "gcn"):
            _, _, raw_info = train_one(data, graphs, self.fixture.mapping, spec, arm, 42, 1.)
            _, _, scaled_info = train_one(data, scaled, self.fixture.mapping, spec, arm, 42, 1.)
            for key in ("initial_normalized_sha256", "epoch_order_sha256", "parameter_count", "pos_weight"):
                self.assertEqual(raw_info[key], scaled_info[key])


if __name__ == "__main__":
    unittest.main()
