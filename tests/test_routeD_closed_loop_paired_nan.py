import unittest

import numpy as np

from scripts.analyze_routeD_closed_loop_paired import (
    paired_bootstrap,
    safe_correlation,
)


class TestRouteDClosedLoopPairedNaN(unittest.TestCase):
    def test_paired_bootstrap_uses_only_finite_video_pairs(self):
        result = paired_bootstrap(
            np.array([0.1, np.nan, -0.05, np.inf]),
            seed=17,
            resamples=200,
        )
        self.assertEqual(result["videos"], 2)
        self.assertAlmostEqual(result["mean"], 0.025)
        self.assertEqual(result["positive_videos"], 1)
        self.assertEqual(result["negative_videos"], 1)

    def test_safe_correlation_drops_nonfinite_pairs(self):
        value = safe_correlation(
            np.array([1.0, 2.0, np.nan, 4.0]),
            np.array([2.0, 4.0, 3.0, 8.0]),
        )
        self.assertAlmostEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
