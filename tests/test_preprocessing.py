"""Scientific invariants, including naturally absent modalities and held-out isolation."""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import (PreprocessingRun, derive_features, fit_preprocessor,
                               inspect_neighbor_list, transform_features, verify_export)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/preprocessing_v2.json").read_text(encoding="utf-8"))


class PreprocessingTests(unittest.TestCase):
    def test_temporal_missingness_and_train_only_fitting(self):
        data = pd.DataFrame(1.0, index=pd.Index(["train_a", "train_b"], name="user_key"), columns=CONFIG["feature_order"])
        temporal = CONFIG["groups"]["temporal"]
        data.loc["train_b", temporal] = np.nan
        data.loc[:, temporal[0]] = np.nan  # No training observations for one feature.
        bundle = fit_preprocessor(data, CONFIG)
        query = data.copy()
        query.loc["train_a", temporal[0]] = 99999.0
        result, masks = transform_features(query, bundle)
        self.assertFalse(bundle["statistics"][temporal[0]]["enabled"])
        self.assertIsNone(bundle["statistics"][temporal[1]]["fill_value"])
        self.assertEqual(bundle["statistics"][temporal[1]]["training_observations"], 1)
        self.assertTrue(result.loc["train_b", temporal].eq(0).all())
        self.assertTrue(masks.loc["train_b", ["observed__" + c for c in temporal]].eq(0).all())
        self.assertTrue(data.loc["train_b", temporal].isna().all())
        before = copy.deepcopy(bundle["statistics"])
        transform_features(query * 1_000_000, bundle)
        self.assertEqual(before, bundle["statistics"])

    def test_unknown_denominator_is_not_observed_zero(self):
        data = pd.DataFrame({"followers_count": [10, 10, 10], "friends_count": [2, 0, np.nan],
            "description": ["", "www.example.com", None], "name": ["name2", "name", None],
            "screen_name": ["u1", "u2", "u3"]})
        result = derive_features(data)
        self.assertEqual(result.followers_friend_ratio.iloc[0], 5)
        self.assertEqual(result.followers_friend_ratio.iloc[1], 10)
        self.assertTrue(pd.isna(result.followers_friend_ratio.iloc[2]))
        self.assertEqual(result.description_length.iloc[0], 0)
        self.assertTrue(pd.isna(result.description_length.iloc[2]))
        self.assertTrue(pd.isna(result.url_in_description.iloc[2]))

    def test_graph_parser_does_not_invent_edges(self):
        self.assertEqual(inspect_neighbor_list(None), ([], "missing", 0))
        self.assertEqual(inspect_neighbor_list("[]"), ([], "empty", 0))
        self.assertEqual(inspect_neighbor_list("[broken")[1], "malformed")
        self.assertEqual(inspect_neighbor_list("Alice, @Bob, invalid handle"), (["alice", "bob"], "nonempty", 1))

    def make_fixture(self, root):
        config = copy.deepcopy(CONFIG)
        (root / "configs").mkdir()
        (root / "src").mkdir()
        shutil.copyfile(ROOT / "src/preprocessing.py", root / "src/preprocessing.py")
        accounts = pd.DataFrame(1.0, index=range(8), columns=CONFIG["feature_order"])
        accounts["screen_name"] = [f"user{i}" for i in range(8)]
        accounts["id"] = "1.23E+18"  # Same lossy numeric token; never an identity.
        accounts["name"] = "test"
        accounts["description"] = "observed description"
        accounts["clean_description"] = "observed description"
        accounts["created_at"] = "2020-01-01"
        accounts["last_updated"] = "2025-01-01"
        accounts.loc[[5, 7], CONFIG["groups"]["temporal"]] = np.nan
        accounts.loc[[5, 7], ["description", "clean_description"]] = None
        annotated = accounts.iloc[:6].copy()
        annotated[CONFIG["label_column"]] = ["human(2)", "bot(1)"] * 3
        tweets = pd.DataFrame({"id": ["1", "2"], "screen_name": ["user0", "user1"],
                               "text": ["متن اول", "متن دوم"], "published_at": ["2024-01-01"] * 2})
        graph = pd.DataFrame({"id": ["1"], "screen_name": ["user0"],
                              "followers": ["user6, user2"], "following": ["user0"]})
        for name, frame in [("annotated", annotated), ("accounts", accounts), ("tweets", tweets), ("graph", graph)]:
            path = root / config["raw_files"][name]
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
        split = pd.DataFrame({"user_key": [f"user{i}" for i in range(6)], "label_binary": [0, 1] * 3,
                              "has_graph": [1, 0, 0, 0, 0, 0],
                              "split": ["train", "train", "validation", "validation", "test", "test"]})
        split.to_csv(root / config["split_file"], index=False)
        (root / "configs/preprocessing_v2.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    def test_raw_only_end_to_end_and_missing_users_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root)
            self.assertFalse((root / "final_data").exists())
            run = PreprocessingRun(root)
            run.load_sources(); run.build_users(); run.freeze_splits()
            run.prepare_text(); run.prepare_graph(); run.fit_transform(); run.validate()
            self.assertEqual(len(run.users), 8)
            for key in ["user5", "user7"]:
                self.assertEqual(run.users.loc[key, "has_temporal"], 0)
                self.assertEqual(run.users.loc[key, "has_text"], 0)
                self.assertEqual(run.users.loc[key, "has_graph"], 0)
            self.assertEqual(len(run.matrices["test"]), 2)
            self.assertEqual(len(run.matrices["unlabeled"]), 2)
            path = run.export()
            self.assertEqual(verify_export(path)["reloaded_transform_populations"], 4)
            self.assertEqual(run.export(), path)  # Exact rerun is safe.
            protected = path / "config/resolved_policy.json"
            protected.write_text("do not overwrite", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
                run.export()
            self.assertEqual(protected.read_text(), "do not overwrite")

    def test_invalid_split_fails_instead_of_resplitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_fixture(root)
            path = root / CONFIG["split_file"]
            split = pd.read_csv(path)
            split.loc[0, "label_binary"] = 1
            split.to_csv(path, index=False)
            run = PreprocessingRun(root)
            run.load_sources(); run.build_users()
            with self.assertRaisesRegex(ValueError, "labels disagree"):
                run.freeze_splits()


if __name__ == "__main__":
    unittest.main()
