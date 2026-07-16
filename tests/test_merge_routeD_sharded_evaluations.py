import json
import tempfile
import unittest
from pathlib import Path

from scripts.merge_routeD_sharded_evaluations import main


class TestMergeRouteDShardedEvaluations(unittest.TestCase):
    def test_merge_rejects_duplicate_video_names(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            rows = []
            for shard_index in range(2):
                result = root / f"shard_{shard_index:02d}.result.json"
                result.write_text(json.dumps({
                    "evaluation_mode": "closed_loop_routeD",
                    "closed_loop_state_updated": True,
                    "independent_baseline_model": True,
                    "baseline_has_routeD_selector": False,
                    "threshold": 0.5,
                    "seed": 17,
                    "dataset": "tapvid_kinetics",
                    "query_mode": "first",
                    "input_resolution": [256, 256],
                    "metric_resolution": [256, 256],
                    "fusion_strength": None,
                    "max_switch_distance_px": None,
                    "profile_p1_tolerance": 0.0,
                    "profile_min_coarse_gain": 0.0,
                    "profile_min_total_gain": 0.0,
                    "per_sample": [{
                        "sample": 0,
                        "video_name": "duplicate",
                        "global_selection_rate": 0.0,
                        "baseline": {"AJ": 0.1, "OA": 1.0, "delta_avg": 0.2},
                        "local": {"AJ": 0.1, "OA": 1.0, "delta_avg": 0.2},
                        "routeD_open": {"AJ": 0.2, "OA": 1.0, "delta_avg": 0.3},
                        "routeD_closed": {"AJ": 0.3, "OA": 1.0, "delta_avg": 0.4},
                    }],
                }))
                rows.append({
                    "shard_index": shard_index,
                    "source_num_samples": 1,
                    "expected_result": str(result),
                })
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({
                "expected_video_count": 2,
                "claim_boundary": "test",
                "source_root": str(root),
                "shards": rows,
            }))
            import sys
            old_argv = sys.argv
            sys.argv = ["merge", "--protocol", str(protocol), "--output", str(root / "out.json")]
            try:
                with self.assertRaisesRegex(ValueError, "not unique"):
                    main()
            finally:
                sys.argv = old_argv


if __name__ == "__main__":
    unittest.main()
