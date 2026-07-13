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
