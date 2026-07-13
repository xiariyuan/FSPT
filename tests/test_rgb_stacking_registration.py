import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from datasets import TAPVidRGBStackingDataset, get_dataset
from projects.mmp_tracker.train_mmp import resolve_dataset


def _write_tiny_rgb_stacking(root: Path) -> None:
    time, height, width = 3, 4, 5
    video = np.zeros((time, height, width, 3), dtype=np.uint8)
    video[..., 0] = 17
    points_xy = np.asarray(
        [
            [[0.10, 0.20], [0.20, 0.30], [0.30, 0.40]],
            [[0.70, 0.60], [0.60, 0.50], [0.50, 0.40]],
        ],
        dtype=np.float32,
    )
    occluded = np.zeros((2, time), dtype=bool)
    with (root / "tapvid_rgb_stacking.pkl").open("wb") as handle:
        pickle.dump(
            [{"video": video, "points": points_xy, "occluded": occluded}],
            handle,
            protocol=4,
        )


class TestRGBStackingRegistration(unittest.TestCase):
    def test_get_dataset_aliases_and_coordinate_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_tiny_rgb_stacking(root)
            for alias in ("rgb_stacking", "rgbstacking", "stacking", "tapvid_rgb_stacking"):
                dataset = get_dataset(
                    alias,
                    str(root),
                    split="validation",
                    backend="sharded_pkl",
                    annotation_file="tapvid_rgb_stacking.pkl",
                    query_mode="first",
                )
                self.assertIsInstance(dataset, TAPVidRGBStackingDataset)
                sample = dataset[0]
                self.assertEqual(tuple(sample["video"].shape), (3, 3, 4, 5))
                np.testing.assert_allclose(
                    sample["target_points"][0, 0].numpy(),
                    np.asarray([0.20, 0.10], dtype=np.float32),
                )
                np.testing.assert_allclose(
                    sample["query_points"][0].numpy(),
                    np.asarray([0.0, 0.20, 0.10], dtype=np.float32),
                )

    def test_resolve_dataset_does_not_inject_sharded_backend(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _write_tiny_rgb_stacking(root)
            config = {
                "data": {
                    "validation": {
                        "dataset": "tapvid_rgb_stacking",
                        "root": str(root),
                        "split": "validation",
                        "backend": "sharded_pkl",
                        "query_mode": "first",
                        "first_frame_query": False,
                    }
                }
            }
            dataset, limit = resolve_dataset(config, "validation", train=False)
            self.assertIsInstance(dataset, TAPVidRGBStackingDataset)
            self.assertIsNone(limit)
            self.assertEqual(len(dataset), 1)


if __name__ == "__main__":
    unittest.main()
