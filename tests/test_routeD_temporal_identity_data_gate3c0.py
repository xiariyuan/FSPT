import numpy as np

from scripts.package_routeD_temporal_identity_data_gate3c0_v0 import (
    _array_sha256,
    _sample_digest,
    _sample_identity,
)


def test_array_hash_binds_dtype_shape_and_content():
    first = _array_sha256(np.asarray([[1, 2]], dtype=np.int32))
    same = _array_sha256(np.asarray([[1, 2]], dtype=np.int32))
    changed_dtype = _array_sha256(np.asarray([[1, 2]], dtype=np.int64))
    changed_shape = _array_sha256(np.asarray([1, 2], dtype=np.int32))
    assert first == same
    assert first != changed_dtype
    assert first != changed_shape


def test_sample_digest_and_raw_identity_are_deterministic():
    sample = {
        "video": np.zeros((2, 3, 3, 3), dtype=np.uint8),
        "target_points": np.ones((1, 2, 2), dtype=np.float32),
        "source_tfrecord": "source-00001",
        "source_record_index": 7,
    }
    assert _sample_digest(sample) == _sample_digest(dict(reversed(sample.items())))
    assert _sample_identity(sample) == ("source-00001", 7)
