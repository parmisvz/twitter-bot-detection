import json
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from src.semi_supervised import _batches, pseudo_pool, train_continuation


ROOT = Path(__file__).resolve().parents[1]


class ConstantTeacher:
    """Small deterministic model with the BaselineMLP inference interface."""
    def eval(self):
        return self
    def __call__(self, x):
        return x[:, :1]


class TestSemiSupervised(unittest.TestCase):
    def config(self):
        return json.loads((ROOT / "configs/semi_supervised_v2.json").read_text())

    def test_balanced_selection_is_deterministic_and_retains_full_pool(self):
        config = self.config()
        config["expected"]["unlabeled"] = 396
        config["expected"]["accepted_pseudo_bot"] = 195
        keys = [f"u{i:03d}" for i in range(396)]
        # Excess tied humans exercise the cap; the final user fails confidence.
        x = np.r_[np.full((200, 40), -5.), np.full((195, 40), 5.), np.zeros((1, 40))].astype(np.float32)
        unlabeled = {"users": pd.DataFrame(index=pd.Index(keys, name="user_key")), "x": x}
        pool, selected = pseudo_pool(ConstantTeacher(), unlabeled, config)
        self.assertEqual(len(pool), 396)
        self.assertEqual(len(selected), 390)
        self.assertEqual(selected.pseudo_label.value_counts().to_dict(), {0: 195, 1: 195})
        self.assertEqual(pool.selection_status.eq("selected_for_student").sum(), 390)
        # Ranking tie uses user_key, and the original pool retains every rejected/selected row.
        self.assertTrue(pool.user_key.is_unique)
        self.assertEqual(selected.loc[selected.pseudo_label.eq(0), "user_key"].tolist(), keys[:195])
        self.assertEqual(pool.selection_status.eq("accepted_not_selected_class_cap").sum(), 5)
        self.assertEqual(pool.selection_status.eq("rejected_below_confidence").sum(), 1)
        reverse = {"users": unlabeled["users"].iloc[::-1], "x": x[::-1].copy()}
        _, reordered = pseudo_pool(ConstantTeacher(), reverse, config)
        self.assertEqual(set(selected.user_key), set(reordered.user_key))

    def test_batch_partitions_never_drop_or_duplicate_a_user(self):
        for size in (31, 32, 33, 671, 390):
            batches = _batches(size, 32, 42, 1)
            values = np.concatenate(batches)
            self.assertEqual(len(values), size)
            self.assertEqual(len(set(values)), size)
            self.assertTrue(all(len(batch) > 1 for batch in batches))

    def test_control_and_student_are_seeded_and_states_independent(self):
        config = self.config()
        config["training"]["epochs"] = 3
        config["expected"]["accepted_pseudo_bot"] = 2
        labeled = {"keys": np.array(["a", "b", "c", "d"]), "x": np.arange(160, dtype=np.float32).reshape(4, 40) / 160,
                   "y": np.array([0, 1, 0, 1], dtype=np.float32)}
        pseudo = {"keys": np.array(["p0", "p1", "p2", "p3"]), "x": labeled["x"] + .1,
                  "y": np.array([0, 1, 0, 1], dtype=np.float32), "confidence": np.ones(4, dtype=np.float32)}
        from src.baselines import BaselineMLP
        state = BaselineMLP(40, [64, 32], .3).state_dict()
        states, history, info = train_continuation(state, labeled, pseudo, labeled, config, "pseudo_student")
        repeated, repeated_history, repeated_info = train_continuation(state, labeled, pseudo, labeled, config, "pseudo_student")
        pd.testing.assert_frame_equal(history, repeated_history)
        self.assertEqual(info, repeated_info)
        best = deepcopy(states["best_validation_loss"])
        for key in states["last"]:
            states["last"][key].add_(1)
            torch.testing.assert_close(states["best_validation_loss"][key], best[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
