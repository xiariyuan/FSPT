from __future__ import annotations

import json
from pathlib import Path

import torch

from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    PROTOCOL_SCHEMA_VERSION,
    aggregate_partition_sidecars,
    evaluate_qualification_gates,
    partition_indices,
    validate_protocol,
)


def _protocol():
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "source_manifests": {
            "a": {"path": "/tmp/a", "sha256": "x", "num_samples": 8},
            "b": {"path": "/tmp/b", "sha256": "y", "num_samples": 4},
        },
        "partitions": {
            "fit": {"source": "a", "inclusive_ranges": [[0, 3]], "expected_count": 4},
            "model_validation": {"source": "a", "indices": [4, 5], "expected_count": 2},
            "pilot_excluded": {"source": "b", "indices": [0], "expected_count": 1},
            "qualification": {"source": "b", "indices": [1, 2], "expected_count": 2},
            "final_holdout": {"source": "b", "indices": [3], "expected_count": 1},
        },
        "pilot_exclusion_partition": "pilot_excluded",
        "final_holdout_partition": "final_holdout",
    }


def test_protocol_expands_and_rejects_overlap():
    protocol = _protocol()
    validate_protocol(protocol)
    assert partition_indices(protocol, "fit") == (0, 1, 2, 3)
    protocol["partitions"]["model_validation"]["indices"] = [3, 4]
    try:
        validate_protocol(protocol)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("overlapping partitions must fail")


def _sidecar(path: Path, *, native_offset: float, candidate_offset: float, aj_gain: float):
    points, frames, candidates = 2, 4, 2
    gt_yx = torch.tensor(
        [
            [[0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5]],
            [[0.6, 0.2], [0.6, 0.3], [0.6, 0.4], [0.6, 0.5]],
        ],
        dtype=torch.float32,
    )
    gt_xy = gt_yx[..., [1, 0]] * 255.0
    native = gt_xy + native_offset
    alternative = gt_xy + candidate_offset
    coords = torch.stack([native, alternative], dim=-2)
    oracle = alternative if abs(candidate_offset) < abs(native_offset) else native
    artifact = {
        "schema_version": "routeD_musr_cotracker3_stage0_adapter_v1",
        "tensors": {
            "candidate_coords_xy_px": coords,
            "candidate_valid_mask": torch.ones(points, frames, candidates, dtype=torch.bool),
            "native_coords_xy_px": native,
            "oracle_coords_xy_px": oracle,
            "gt_tracks_yx": gt_yx,
            "native_visibility": torch.ones(points, frames, dtype=torch.bool),
            "gt_occluded": torch.zeros(points, frames, dtype=torch.bool),
            "query_points_tyx": torch.tensor([[0.0, 0.2, 0.2], [0.0, 0.6, 0.2]]),
        },
        "audit": {
            "gain_points": {"AJ": aj_gain, "delta_average": aj_gain + 1.0},
            "visible_evaluation_rows": 6,
        },
    }
    torch.save(artifact, path)


def test_aggregate_sidecars_preserves_native_parity_and_oracle_gain(tmp_path):
    first = tmp_path / "a.pt"
    second = tmp_path / "b.pt"
    _sidecar(first, native_offset=20.0, candidate_offset=0.0, aj_gain=10.0)
    _sidecar(second, native_offset=18.0, candidate_offset=0.0, aj_gain=8.0)
    summary = aggregate_partition_sidecars([first, second])
    assert summary["videos"] == 2
    assert summary["routing_disabled_native_parity_all"]
    assert summary["gain_points"]["AJ"] > 0.0
    assert summary["gain_points"]["delta_average"] > 0.0
    assert summary["per_video_AJ_gain_points"]["median"] == 8.0
    assert all(value > 0.0 for value in summary["threshold_hit_gain_points"].values())


def test_gate_never_passes_partial_partition():
    summary = {
        "routing_disabled_native_parity_all": True,
        "gain_points": {"AJ": 9.0, "delta_average": 10.0},
        "per_video_AJ_gain_points": {
            "median": 8.0,
            "gain_at_least_1_fraction": 1.0,
        },
        "threshold_hit_gain_points": {"1": 1.0, "2": 1.0, "4": 1.0, "8": 1.0, "16": 1.0},
    }
    gates = {
        "pooled_AJ_gain_points_min": 3.0,
        "pooled_delta_gain_points_min": 4.0,
        "median_video_AJ_gain_points_min": 2.0,
        "video_AJ_gain_at_least_1_fraction_min": 2 / 3,
        "each_threshold_gain_points_gt": 0.0,
    }
    result = evaluate_qualification_gates(summary, gates, complete=False)
    assert not result["pass"]
    assert not result["checks"]["complete_partition"]
    assert "FORBIDDEN" in result["decision"]


def test_repository_protocol_is_valid():
    path = Path("configs/routeD_musr_kubric_cache_protocol_v0.json")
    protocol = json.loads(path.read_text())
    validate_protocol(protocol)
    assert partition_indices(protocol, "pilot_excluded") == (0,)
    assert partition_indices(protocol, "candidate_qualification") == tuple(range(1, 16))
    assert partition_indices(protocol, "final_holdout") == tuple(range(16, 32))
