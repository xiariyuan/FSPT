import copy

import pytest
import torch

from scripts.build_routeD_geometry_causal_input_cache_gate3a_v1 import (
    ALLOWED_TENSOR_KEYS,
    _assert_teacher_free,
)
from scripts.package_routeD_geometry_candidate_cache_gate3a_v1 import nested_exact


def _causal_payload():
    tensors = {
        key: torch.tensor([index], dtype=torch.float32)
        for index, key in enumerate(ALLOWED_TENSOR_KEYS)
    }
    return {"tensors": tensors}


def test_causal_input_allowlist_accepts_only_registered_fields():
    _assert_teacher_free(_causal_payload())


def test_causal_input_allowlist_rejects_teacher_tensor():
    payload = _causal_payload()
    payload["tensors"]["teacher_coordinate"] = torch.zeros(1)
    with pytest.raises(ValueError):
        _assert_teacher_free(payload)


def test_candidate_cache_nested_exact_detects_single_value_drift():
    left = {"x": [torch.tensor([1.0, 2.0]), {"valid": True}]}
    right = copy.deepcopy(left)
    assert nested_exact(left, right)
    right["x"][0][1] = 3.0
    assert not nested_exact(left, right)
