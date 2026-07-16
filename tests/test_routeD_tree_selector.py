import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression

from projects.mmp_tracker.mmp_tracker.hypothesis_scorer import (
    MultiThresholdHypothesisScorer,
)
from projects.mmp_tracker.mmp_tracker.routeD_selector import load_routeD_selector


class TestRouteDTreeSelector(unittest.TestCase):
    def test_tree_selector_loads_and_returns_valid_shapes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            scorer = MultiThresholdHypothesisScorer(
                feature_dim=12, hidden_dim=8, threshold_count=5
            )
            scorer_path = root / "scorer.pt"
            torch.save(
                {
                    "kind": "routeD_multithreshold_utility_scorer",
                    "feature_dim": 12,
                    "threshold_count": 5,
                    "thresholds_px": [1.0, 2.0, 4.0, 8.0, 16.0],
                    "config": {"hidden_dim": 8},
                    "model_state": scorer.state_dict(),
                    "feature_mean": torch.zeros(12),
                    "feature_std": torch.ones(12),
                },
                scorer_path,
            )

            rng = np.random.default_rng(17)
            tree_x = rng.normal(size=(256, 54)).astype(np.float32)
            tree_y = (tree_x[:, 0] + tree_x[:, 1] > 0).astype(np.int64)
            tree = ExtraTreesClassifier(
                n_estimators=20,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=17,
            ).fit(tree_x, tree_y)
            raw_probability = tree.predict_proba(tree_x)[:, 1]
            isotonic = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            isotonic.fit(raw_probability, tree_y)

            controller_path = root / "controller.pkl"
            with controller_path.open("wb") as handle:
                pickle.dump(
                    {
                        "kind": "routeD_kubric_extratrees_controller",
                        "tree": tree,
                        "isotonic": isotonic,
                        "policy": {
                            "probability_threshold": 0.5,
                            "min_p1_margin": -1.0,
                            "min_coarse_gain": -1.0,
                            "min_total_gain": -1.0,
                        },
                        "selected_tree_config": {},
                        "thresholds_px": (1.0, 2.0, 4.0, 8.0, 16.0),
                        "train_partitions": {},
                        "scorer_bundle_payload": torch.load(
                            scorer_path, map_location="cpu", weights_only=False
                        ),
                        "train_cache": "synthetic",
                    },
                    handle,
                )

            selector = load_routeD_selector(
                controller_path, device="cpu", threshold=0.5
            )
            features = torch.randn(2, 3, 6, 12)
            points = torch.rand(2, 3, 6, 2)
            valid = torch.ones(2, 3, 6, dtype=torch.bool)
            fallback = points[..., 0, :]
            output = selector.select(
                features,
                points,
                valid,
                fallback_points=fallback,
            )

            self.assertEqual(tuple(output["points"].shape), (2, 3, 2))
            self.assertEqual(tuple(output["index"].shape), (2, 3))
            self.assertEqual(tuple(output["gate_probability"].shape), (2, 3))
            self.assertEqual(
                tuple(output["threshold_probabilities"].shape), (2, 3, 6, 5)
            )
            self.assertTrue(bool(torch.isfinite(output["points"]).all()))
            self.assertTrue(
                bool(torch.isfinite(output["gate_probability"]).all())
            )
            self.assertTrue(
                bool(((output["index"] >= 0) & (output["index"] < 6)).all())
            )


if __name__ == "__main__":
    unittest.main()
