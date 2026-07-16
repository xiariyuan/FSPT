from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from scripts.audit_tapvid_kinetics_package_identity import sample_matches_group


def jpeg_bytes(height: int, width: int) -> bytes:
    image = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


class TestTapVidKineticsIdentityAudit(unittest.TestCase):
    def test_official_generator_transform_matches(self):
        height, width = 72, 128
        csv_points = np.asarray(
            [[[[0.0, 0.0], [0.25, 0.75], [1.0, 1.0]]]], dtype=np.float64
        ).reshape(1, 3, 2)
        occ = np.asarray([[True, False, False]])
        stored = csv_points.copy()
        stored[..., 0] = (stored[..., 0] * width - 0.5) / width
        stored[..., 1] = (stored[..., 1] * height - 0.5) / height
        sample = {
            "video": np.asarray([jpeg_bytes(height, width)] * 3),
            "points": stored,
            "occluded": occ,
        }
        group = {"key": ("vid", 0, 10), "points": csv_points, "occluded": occ}
        matched, diff, size = sample_matches_group(sample, group)
        self.assertTrue(matched)
        self.assertEqual(diff, 0.0)
        self.assertEqual(size, (height, width))

    def test_occlusion_mismatch_is_rejected(self):
        height, width = 72, 128
        csv_points = np.zeros((1, 3, 2), dtype=np.float64)
        stored = csv_points.copy()
        stored[..., 0] = (stored[..., 0] * width - 0.5) / width
        stored[..., 1] = (stored[..., 1] * height - 0.5) / height
        sample = {
            "video": np.asarray([jpeg_bytes(height, width)] * 3),
            "points": stored,
            "occluded": np.asarray([[False, False, False]]),
        }
        group = {
            "key": ("vid", 0, 10),
            "points": csv_points,
            "occluded": np.asarray([[True, False, False]]),
        }
        matched, _, _ = sample_matches_group(sample, group)
        self.assertFalse(matched)


if __name__ == "__main__":
    unittest.main()
