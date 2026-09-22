import unittest
from io import StringIO
from copy import deepcopy
import numpy as np
import pandas as pd
import torch

from tests import test_gcn_integration as graph_fixture
from src.gcn_integration import initialized_model
from src.graph_diagnosis import auc_pairs, reliability, representations


class TestGraphDiagnosis(unittest.TestCase):
    def test_auc_pair_decomposition_with_ties(self):
        frame = pd.DataFrame({"label_binary": [1, 1, 0, 0], "has_graph": [1, 0, 1, 0],
                              "bot_probability": [.3, .8, .1, .8]})
        rows = auc_pairs(frame)
        self.assertEqual(sum(r["pairs"] for r in rows), 4)
        self.assertAlmostEqual(sum(r["weighted_auc_contribution"] for r in rows), .625)
        self.assertEqual(next(r["pair_auc"] for r in rows if r["bot_has_graph"] == 1 and r["human_has_graph"] == 0), 0.)

    def test_reliability_includes_probability_boundaries(self):
        frame = pd.DataFrame({"label_binary": [0, 1, 1], "bot_probability": [0., .5, 1.]})
        bins = reliability(frame)
        self.assertEqual(sum(r["n"] for r in bins), 3)
        self.assertEqual([bins[i]["n"] for i in (0, 5, 9)], [1, 1, 1])
        self.assertIsNone(bins[1]["mean_probability"])

    def test_logit_decomposition_missingness_and_state_immutability(self):
        fixture = graph_fixture.TestGCNIntegration()
        fixture.setUp()
        pop = deepcopy(fixture.pop)
        index = pd.Index(["a", "b", "c", "d"], name="user_key")
        pop["users"] = pd.DataFrame({"label_binary": pop["y"].astype(int)}, index=index)
        pop["masks"] = pd.DataFrame({"has_graph": [1, 1, 0, 0], "has_graph_node": [1, 1, 0, 1], "has_temporal": 1}, index=index)
        for arm in ("temporal_anchor", "degree_mlp", "gcn"):
            model = initialized_model(arm, fixture.mapping, 42)
            state = deepcopy(model.state_dict())
            result = representations(model, pop, fixture.graph)
            numeric = result.select_dtypes(include="floating").columns
            self.assertTrue(all(result[c].dtype == np.float64 for c in numeric))
            restored = pd.read_csv(StringIO(result.to_csv(index=False)), float_precision="round_trip")
            np.testing.assert_array_equal(result[numeric].to_numpy(), restored[numeric].to_numpy())
            self.assertEqual(result.user_key.tolist(), list(index))
            self.assertLess(result.logit_reconstruction_error.max(), 1e-5)
            if arm == "temporal_anchor":
                self.assertTrue(result.graph_rms.isna().all())
            else:
                self.assertTrue(result.loc[result.has_graph.eq(0), "graph_logit_effect"].eq(0).all())
                self.assertTrue(result.loc[result.has_graph.eq(0), "graph_rms"].eq(0).all())
            for name, value in state.items():
                torch.testing.assert_close(value, model.state_dict()[name], atol=0, rtol=0)
            self.assertTrue(np.isfinite(result.bot_probability).all())


if __name__ == "__main__":
    unittest.main()
