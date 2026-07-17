from pathlib import Path

import torch

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_feature_cache import (
    CMCP_FEATURE_CACHE_SCHEMA_VERSION,
    feature_quantization_audit,
    validate_quantization_audit,
    verify_feature_cache_artifact,
)


def test_feature_quantization_audit_passes_normalized_features():
    torch.manual_seed(5)
    value = torch.nn.functional.normalize(torch.randn(3, 8, 7, 9), dim=1)
    half = value.half()
    audit = feature_quantization_audit(value.float(), half)
    validate_quantization_audit(audit)
    assert audit["max_abs"] < 5e-4
    assert audit["min_cosine"] > 0.99999


def test_feature_cache_artifact_verifies_hashes(tmp_path: Path):
    feature = torch.randn(2, 4, 5, 6).half()
    artifact = {
        "schema_version": CMCP_FEATURE_CACHE_SCHEMA_VERSION,
        "provenance": {
            "protocol_sha256": "protocol",
            "base_sidecar_sha256": "base",
        },
        "feature_maps_f16": feature,
        "feature_maps_f16_sha256": tensor_sha256(feature),
    }
    path = tmp_path / "feature.pt"
    torch.save(artifact, path)
    loaded = verify_feature_cache_artifact(
        path,
        expected_protocol_sha256="protocol",
        expected_base_sidecar_sha256="base",
    )
    assert torch.equal(loaded["feature_maps_f16"], feature)


def test_feature_cache_artifact_rejects_hash_drift(tmp_path: Path):
    feature = torch.randn(2, 4, 5, 6).half()
    path = tmp_path / "feature.pt"
    torch.save(
        {
            "schema_version": CMCP_FEATURE_CACHE_SCHEMA_VERSION,
            "provenance": {},
            "feature_maps_f16": feature,
            "feature_maps_f16_sha256": "wrong",
        },
        path,
    )
    try:
        verify_feature_cache_artifact(path)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("hash drift was not rejected")
