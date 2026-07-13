import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from datasets.tapvid_kubric import _generate_sparse_tracks_from_kubric


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "preprocess_kubric_tfrecord_no_tf.py"
SPEC = importlib.util.spec_from_file_location("preprocess_kubric_tfrecord_no_tf", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TestKubricNoTensorFlowPreprocess(unittest.TestCase):
    def test_bottom_right_query_stays_visible(self):
        time, height, width = 3, 4, 5
        flow = np.zeros((time, height, width, 2), dtype=np.float32)
        segmentation = np.zeros((time, height, width), dtype=np.uint8)
        segmentation[:, height - 1, width - 1] = 1

        query, tracks, occluded = _generate_sparse_tracks_from_kubric(
            forward_flow=flow,
            backward_flow=flow,
            segmentations=segmentation,
            num_points=16,
            rng=np.random.RandomState(7),
        )

        query_t = np.rint(query[:, 0]).astype(np.int64)
        point_index = np.arange(query.shape[0])
        self.assertFalse(occluded[point_index, query_t].any())
        self.assertTrue(np.isfinite(tracks).all())
        self.assertLessEqual(float(tracks.max()), 1.0)
        self.assertGreaterEqual(float(tracks.min()), 0.0)
        np.testing.assert_allclose(
            tracks[point_index, query_t],
            np.ones((tracks.shape[0], 2), dtype=np.float32),
            atol=1e-6,
        )

    def test_kubric_flow_uses_delta_row_delta_col_order(self):
        time, height, width = 4, 20, 30
        forward = np.zeros((time, height, width, 2), dtype=np.float32)
        backward = np.zeros_like(forward)
        forward[..., 0] = 1.0
        forward[..., 1] = 2.0
        backward[..., 0] = -1.0
        backward[..., 1] = -2.0

        query, tracks, occluded = _generate_sparse_tracks_from_kubric(
            forward_flow=forward,
            backward_flow=backward,
            segmentations=None,
            num_points=512,
            rng=np.random.RandomState(19),
        )

        tracks_px = tracks.copy()
        tracks_px[..., 0] *= float(height - 1)
        tracks_px[..., 1] *= float(width - 1)
        query_t = np.rint(query[:, 0]).astype(np.int64)
        observed = []
        for point_index, q_t in enumerate(query_t):
            for t in range(int(q_t), time - 1):
                if not occluded[point_index, t] and not occluded[point_index, t + 1]:
                    observed.append(tracks_px[point_index, t + 1] - tracks_px[point_index, t])
            for t in range(1, int(q_t) + 1):
                if not occluded[point_index, t] and not occluded[point_index, t - 1]:
                    observed.append(tracks_px[point_index, t] - tracks_px[point_index, t - 1])

        self.assertGreater(len(observed), 100)
        np.testing.assert_allclose(
            np.asarray(observed),
            np.tile(np.asarray([[1.0, 2.0]], dtype=np.float32), (len(observed), 1)),
            atol=1e-4,
        )

    def test_dynamic_example_and_tfrecord_reader(self):
        example = MODULE.EXAMPLE_CLASS()
        example.features.feature["metadata/num_frames"].int64_list.value.append(1)
        payload = example.SerializeToString()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tiny.tfrecord"
            with path.open("wb") as handle:
                handle.write(struct.pack("<Q", len(payload)))
                handle.write(b"\x00\x00\x00\x00")
                handle.write(payload)
                handle.write(b"\x00\x00\x00\x00")
            records = list(MODULE.iter_tfrecord(path))

        self.assertEqual(records, [payload])

    def test_png_sequence_decode_rgb(self):
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        rgb[..., 0] = 11
        rgb[..., 1] = 22
        rgb[..., 2] = 33
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".png", bgr)
        self.assertTrue(ok)

        decoded = MODULE.decode_png_sequence([encoded.tobytes()], color=True)
        self.assertEqual(decoded.shape, (1, 2, 3, 3))
        np.testing.assert_array_equal(decoded[0], rgb)


if __name__ == "__main__":
    unittest.main()
