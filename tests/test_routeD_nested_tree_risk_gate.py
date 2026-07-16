import unittest

import torch

from scripts.audit_routeD_nested_tree_risk_gate import (
    fit_tree,
    parse_max_features,
    predict_probability,
    select_tree_config,
)


def _examples(rows: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(rows, 8, generator=generator)
    target = (features[:, 0] + 0.5 * features[:, 1] > 0).bool()
    return {"features": features, "target": target}


class TestRouteDNestedTreeRiskGate(unittest.TestCase):
    def test_parse_max_features_accepts_supported_values(self):
        self.assertEqual(parse_max_features("sqrt"), "sqrt")
        self.assertEqual(parse_max_features("log2"), "log2")
        self.assertEqual(parse_max_features("0.75"), 0.75)

    def test_tree_probability_is_finite_and_bounded(self):
        examples = _examples(128, 17)
        model = fit_tree(
            examples["features"],
            examples["target"],
            n_estimators=20,
            min_samples_leaf=2,
            max_features="sqrt",
            seed=17,
            n_jobs=1,
        )
        probability = predict_probability(model, examples["features"])
        self.assertEqual(tuple(probability.shape), (128,))
        self.assertTrue(bool(torch.isfinite(probability).all()))
        self.assertTrue(
            bool(((probability >= 0.0) & (probability <= 1.0)).all())
        )

    def test_tree_config_selection_returns_a_candidate(self):
        fit = _examples(160, 29)
        validation = _examples(96, 43)
        selected, rows = select_tree_config(
            fit,
            validation,
            n_estimators=20,
            min_samples_leaf_values=[2, 4],
            max_features_values=["sqrt"],
            seed=17,
            n_jobs=1,
        )
        self.assertEqual(len(rows), 2)
        self.assertIn(selected["min_samples_leaf"], {2, 4})
        self.assertEqual(selected["max_features"], "sqrt")
        self.assertGreaterEqual(selected["auc"], 0.0)
        self.assertLessEqual(selected["auc"], 1.0)
        self.assertGreaterEqual(selected["average_precision"], 0.0)
        self.assertLessEqual(selected["average_precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
