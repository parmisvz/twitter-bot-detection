import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.performance_diagnosis import Inputs, diagnostic_best, digest, metrics, sweep, validate_predictions


class TestPerformanceDiagnosis(unittest.TestCase):
    def test_allowlist_rejects_test_even_when_registered(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            p = root / "test_users.csv"
            p.write_text("user_key,label_binary\nu,1\n")
            reader = Inputs(root)
            reader.allow(p, digest(p))
            with self.assertRaisesRegex(ValueError, "Test data"):
                reader.read(p)
            with self.assertRaisesRegex(ValueError, "allowlisted"):
                reader.read(root / "anything.csv")
            self.assertEqual(reader.accessed, {})

    def test_sweep_handles_extremes_and_preserves_ranking(self):
        table = sweep(np.array([0, 0, 1, 1]), np.array([.1, .2, .7, .9]),
                      {"threshold_grid": {"start": 0, "stop": 1, "points": 101}})
        self.assertTrue(table.predicted_bots.diff().dropna().le(0).all())
        self.assertEqual(table.iloc[0].predicted_bots, 4)
        self.assertEqual(table.iloc[-1].predicted_bots, 0)
        self.assertEqual(diagnostic_best(table).threshold, .5)
        self.assertEqual(metrics([0, 1], [0, 1], .5)["bot_f1"], 1)
        for t in (0, .2, .5, .7, 1):
            y, p = np.array([0, 0, 1, 1]), np.array([.1, .2, .7, .9])
            actual = metrics(y, p, t)
            for name, function in [("bot_f1", f1_score), ("bot_precision", precision_score), ("bot_recall", recall_score)]:
                self.assertAlmostEqual(actual[name], function(y, p >= t, zero_division=0))

    def test_registry_preserves_unsupported_and_rejects_identity_or_label_changes(self):
        users = pd.DataFrame({"label_binary": [0, 1]}, index=pd.Index(["a", "b"], name="user_key"))
        frame = pd.DataFrame({"user_key": ["b", "a"], "label_binary": [1, 0],
            "split": ["validation", "validation"], "status": ["missing_graph", "predicted"],
            "bot_probability": [np.nan, .2], "prediction": [np.nan, 0]})
        aligned = validate_predictions(frame, users)
        self.assertEqual(aligned.index.tolist(), ["a", "b"])
        self.assertTrue(pd.isna(aligned.loc["b", "bot_probability"]))
        for column, value in [("label_binary", [0, 0]), ("user_key", ["a", "a"]),
                              ("bot_probability", [.8, .2]), ("split", ["test", "validation"])]:
            broken = frame.copy()
            broken[column] = value
            with self.assertRaises(ValueError):
                validate_predictions(broken, users)


if __name__ == "__main__":
    unittest.main()
