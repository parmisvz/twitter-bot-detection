"""Research-contract regressions for Phase 3; no fitting on the real test population."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from src.baselines import (MASKS, align, checked_artifact, evaluate_registry, keyed,
                           prediction_registry, sha256, train_mlp)


class BaselineContractTests(unittest.TestCase):
    def test_alignment_is_by_key_and_rejects_extra_or_duplicate_users(self):
        users = keyed(pd.DataFrame({"user_key": ["a", "b"]}), "users")
        other = keyed(pd.DataFrame({"user_key": ["b", "a"], "value": [20, 10]}), "other")
        self.assertEqual(align(other, users, "other").value.tolist(), [10, 20])
        with self.assertRaisesRegex(ValueError, "population mismatch"):
            align(other.iloc[:1], users, "other")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            keyed(pd.DataFrame({"user_key": ["a", "a"]}), "duplicate")

    def test_missing_text_user_remains_explicit_and_metrics_show_coverage(self):
        users = keyed(pd.DataFrame({"user_key": ["a", "b", "c"], "label_binary": [0, 1, 1]}), "users")
        masks = pd.DataFrame(1, index=users.index, columns=list(MASKS))
        masks["temporal_observed_count"] = [7, 0, 7]
        masks.loc["b", ["has_temporal", "temporal_complete"]] = 0
        masks.loc["c", "has_tweet_text"] = 0  # Description present still does not supply tweet vectors.
        registry = prediction_registry({"users": users, "masks": masks}, ["a", "b"],
                                       [.1, .9], [0, 1], "text_lr", "test")
        self.assertEqual(len(registry), 3)
        self.assertEqual(registry.iloc[2].status, "missing_tweet_text")
        self.assertTrue(pd.isna(registry.iloc[2].bot_probability))
        self.assertTrue(pd.isna(registry.iloc[2].prediction))
        self.assertEqual(registry.iloc[1].status, "predicted")  # Missing temporal does not drop user.
        metric, detail = evaluate_registry(registry, "all_users")
        self.assertEqual(metric["population_n"], 3)
        self.assertEqual(metric["evaluated_n"], 2)
        self.assertEqual(metric["coverage"], 2 / 3)
        self.assertEqual(detail["confusion_matrix"], [[1, 0], [0, 1]])
        with self.assertRaisesRegex(ValueError, "Unexplained missing"):
            prediction_registry({"users": users, "masks": masks}, ["a"], [.1], [0], "text_lr", "test")

    def test_artifact_corruption_stops_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = directory / "data.csv"
            path.write_text("user_key\na\n")
            manifest = {"artifacts": {"data.csv": {"sha256": sha256(path)}}}
            self.assertEqual(checked_artifact(directory, manifest, "data.csv"), path.resolve())
            path.write_text("user_key\nb\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                checked_artifact(directory, manifest, "data.csv")

    def test_checkpoint_snapshots_are_independent_and_seeded_training_repeats(self):
        rng = np.random.default_rng(7)
        sample = {"x": rng.normal(size=(12, 4)), "y": np.array([0, 1] * 6)}
        spec = {"hidden_dims": [8, 4], "dropout": .3, "epochs": 3, "batch_size": 4,
                "lr": .001, "weight_decay": .01, "selection": "best_validation_loss"}
        model, states, history, info = train_mlp(sample, sample, spec, 42, 1, "cpu")
        frozen = deepcopy(states)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(10)
        for variant in states:
            for key in states[variant]:
                torch.testing.assert_close(states[variant][key], frozen[variant][key], rtol=0, atol=0)
                self.assertNotEqual(states["last"][key].data_ptr(), states["best_validation_loss"][key].data_ptr())
        _, repeated_states, repeated_history, repeated_info = train_mlp(sample, sample, spec, 42, 1, "cpu")
        pd.testing.assert_frame_equal(history, repeated_history)
        self.assertEqual(info, repeated_info)
        for variant in states:
            for key in states[variant]:
                torch.testing.assert_close(states[variant][key], repeated_states[variant][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
