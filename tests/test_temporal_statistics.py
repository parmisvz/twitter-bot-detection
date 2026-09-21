from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from src.temporal_statistics import ARMS, TemporalExperimentModel, feature_mapping, train_one
from src.checkpoint_weight_experiment import validation_losses

ROOT = Path(__file__).resolve().parents[1]


class TestTemporalStatistics(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/temporal_statistics_v1.json").read_text())
        order = json.loads((ROOT / "configs/preprocessing_v2.json").read_text())["feature_order"]
        self.mapping = feature_mapping(order, self.config["temporal_features"])
        torch.set_num_threads(1)

    def test_exact_partition_and_parameter_counts(self):
        indices = self.mapping["other_indices"] + self.mapping["temporal_indices"]
        self.assertEqual(sorted(indices), list(range(40)))
        for arm in ARMS:
            model = TemporalExperimentModel(arm, self.mapping)
            self.assertEqual(sum(p.numel() for p in model.parameters()), self.config["expected_parameters"][arm])
        with self.assertRaises(ValueError):
            feature_mapping(self.mapping["feature_order"], self.config["temporal_features"][:-1])

    def test_missing_temporal_values_cannot_become_observations(self):
        x = torch.randn(4, 47)
        x[:, 40:] = 1
        x[0, 40:] = 0
        x[1, 40] = 0
        corrupt = x.clone()
        corrupt[0, self.mapping["temporal_indices"]] = float("nan")
        corrupt[1, self.mapping["temporal_indices"][0]] = 999999.
        for arm in ARMS:
            model = TemporalExperimentModel(arm, self.mapping).eval()
            with torch.no_grad():
                normal, altered = model(x), model(corrupt)
            torch.testing.assert_close(normal, altered, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(altered).all())
            self.assertEqual(tuple(altered.shape), (4, 1))
            if arm == "temporal_branch":
                _, values, mask = model.split_inputs(corrupt)
                hidden = model.temporal_representation(values, mask)
                torch.testing.assert_close(hidden[0], torch.zeros(16), rtol=0, atol=0)

    def test_information_contract_and_removed_feature_control(self):
        x = torch.randn(4, 47); x[:, 40:] = 1
        a = TemporalExperimentModel("flat_masked", self.mapping).eval()
        b = TemporalExperimentModel("temporal_branch", self.mapping).eval()
        for left, right in zip(a.split_inputs(x), b.split_inputs(x)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        c = TemporalExperimentModel("without_seven_statistics", self.mapping).eval()
        changed = x.clone(); changed[:, self.mapping["temporal_indices"]] = 1000.; changed[:, 40:] = 0
        torch.testing.assert_close(c(x), c(changed), rtol=0, atol=0)
        x[0, 40] = .5
        with self.assertRaises(ValueError):
            b(x)

    def test_paired_batches_reproducibility_selection_and_snapshots(self):
        config = deepcopy(self.config)
        config["training"]["epochs"], config["training"]["batch_size"] = 3, 4
        rng = np.random.default_rng(5)
        x = rng.normal(size=(8, 47)).astype(np.float32); x[:, 40:] = 1
        data = {"train": {"x": x, "y": np.array([0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)},
                "validation": {"x": x[:4], "y": np.array([0, 0, 0, 1], dtype=np.float32)}}
        trials = [train_one(data, self.mapping, config, arm, 42, 3.) for arm in ARMS]
        self.assertEqual(len({info["batch_order_sha256"] for _, _, info in trials}), 1)
        self.assertTrue(all(info["epoch_order_sha256"] == trials[0][2]["epoch_order_sha256"] for _, _, info in trials))
        states, history, info = trials[1]
        _, repeated, repeated_info = train_one(data, self.mapping, config, "temporal_branch", 42, 3.)
        pd.testing.assert_frame_equal(history, repeated)
        self.assertEqual(info, repeated_info)
        key = config["selection"]
        self.assertEqual(info["epochs"][key], int(history.loc[history.validation_unweighted_loss.idxmin(), "epoch"]))
        model = TemporalExperimentModel("temporal_branch", self.mapping)
        model.load_state_dict(states[key])
        values = validation_losses(model, data["validation"], 4, 3.)
        row = history.set_index("epoch").loc[info["epochs"][key]]
        np.testing.assert_allclose(values, [row.validation_weighted_loss, row.validation_unweighted_loss], atol=1e-7)
        saved = deepcopy(states[key])
        for value in states["last"].values(): value.add_(1)
        for name in saved: torch.testing.assert_close(saved[name], states[key][name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
