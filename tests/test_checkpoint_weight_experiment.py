from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from src.checkpoint_weight_experiment import train_one, weights_from_training, validation_losses
from src.baselines import BaselineMLP

ROOT = Path(__file__).resolve().parents[1]


class TestCheckpointWeightExperiment(unittest.TestCase):
    def test_weight_rules_use_training_counts(self):
        self.assertEqual(weights_from_training([0, 0, 0, 0, 1]),
                         {"unweighted": 1., "sqrt_train_ratio": 2., "full_train_ratio": 4.})
        with self.assertRaises(ValueError):
            weights_from_training([0, 0])

    def test_paired_training_and_independent_checkpoint_states(self):
        config = json.loads((ROOT / "configs/improvement_step2_v1.json").read_text())
        config["training"]["epochs"], config["training"]["batch_size"] = 3, 4
        x = np.arange(320, dtype=np.float32).reshape(8, 40) / 320
        data = {"train": {"x": x, "y": np.array([0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)},
                "validation": {"x": x[:4], "y": np.array([0, 0, 0, 1], dtype=np.float32)}}
        states, history, info = train_one(data, config, 42, 3.)
        _, other_history, other_info = train_one(data, config, 42, 1.)
        for field in ("initial_state_sha256", "batch_order_sha256", "final_rng_sha256"):
            self.assertEqual(info[field], other_info[field])
        self.assertFalse(history.equals(other_history))
        _, repeated, repeated_info = train_one(data, config, 42, 3.)
        pd.testing.assert_frame_equal(history, repeated)
        self.assertEqual(info, repeated_info)
        for kind in ("weighted", "unweighted"):
            key = f"best_{kind}_validation_loss"
            self.assertEqual(info["epochs"][key], int(history.loc[history[f"validation_{kind}_loss"].idxmin(), "epoch"]))
            model = BaselineMLP(**config["architecture"])
            model.load_state_dict(states[key])
            values = validation_losses(model, data["validation"], 4, 3.)
            record = history.set_index("epoch").loc[info["epochs"][key]]
            np.testing.assert_allclose(values, [record.validation_weighted_loss, record.validation_unweighted_loss], atol=1e-7)
        copy = deepcopy(states["best_weighted_validation_loss"])
        for value in states["last"].values():
            value.add_(1)
        for key in copy:
            torch.testing.assert_close(copy[key], states["best_weighted_validation_loss"][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
