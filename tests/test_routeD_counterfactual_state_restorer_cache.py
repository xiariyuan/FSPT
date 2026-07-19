from pathlib import Path

import torch

from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restoration import (
    CoTrackerOnlineStateSnapshot,
    snapshots_exact,
)
from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer_cache import (
    CSRR_CACHE_SCHEMA_VERSION,
    nested_tensor_hashes,
    select_gate2_rows,
    snapshot_from_cache_dict,
    snapshot_to_cache_dict,
    verify_csrr_cache_artifact,
)


def _snapshot() -> CoTrackerOnlineStateSnapshot:
    torch.manual_seed(10)
    return CoTrackerOnlineStateSnapshot(
        predictor_n=64,
        predictor_queries=torch.randn(1, 64, 3),
        online_ind=8,
        online_track_feat=tuple(torch.randn(1, 1, 64, 128) for _ in range(4)),
        online_track_support=tuple(torch.randn(1, 49, 64, 128) for _ in range(4)),
        online_coords_predicted=torch.randn(1, 16, 64, 2),
        online_vis_predicted=torch.randn(1, 16, 64),
        online_conf_predicted=torch.randn(1, 16, 64),
    )


def test_snapshot_cache_roundtrip_exact_and_unaliased():
    snapshot = _snapshot()
    restored = snapshot_from_cache_dict(snapshot_to_cache_dict(snapshot))
    assert snapshots_exact(snapshot, restored)
    assert snapshot.online_coords_predicted.data_ptr() != restored.online_coords_predicted.data_ptr()


def test_gate2_row_selection_is_thresholded_and_ranked():
    native = torch.zeros(4, 8, 2)
    gt = torch.zeros(4, 24, 2)
    # GT is normalized yx; create future x offsets 20, 30, 1, 2 px.
    gt[0, 16:24, 1] = 20 / 255
    gt[1, 16:24, 1] = 30 / 255
    gt[2, 16:24, 1] = 1 / 255
    gt[3, 16:24, 1] = 2 / 255
    occluded = torch.zeros(4, 24, dtype=torch.bool)
    rows = select_gate2_rows(
        native_future_coords_xy_px=native,
        gt_tracks_yx=gt,
        gt_occluded=occluded,
        original_query_frames=torch.zeros(4),
        failure_error_min_px=16,
        clean_error_max_px=4,
        min_future_visible_frames=4,
        per_class_cap=2,
    )
    assert rows["failure_point_indices"].tolist() == [1, 0]
    assert rows["clean_point_indices"].tolist() == [2, 3]


def test_artifact_verification_detects_hash_tamper(tmp_path: Path):
    snapshot = _snapshot()
    exact = snapshot_to_cache_dict(snapshot)
    points = torch.tensor([1, 2], dtype=torch.long)
    model_tensors = {
        "continuation_video_u8": torch.zeros(16, 3, 256, 256, dtype=torch.uint8),
        "point_indices": points,
        "apply_target": torch.tensor([1.0, 0.0]),
        "trajectory_features": torch.zeros(2, 8, 9, dtype=torch.float16),
        "native_commit_coordinates_normalized_xy": torch.zeros(2, 2, dtype=torch.float16),
        "teacher_commit_coordinates_normalized_xy": torch.zeros(2, 2, dtype=torch.float16),
        "native_visibility_probability": torch.zeros(2, dtype=torch.float16),
        "native_confidence_probability": torch.zeros(2, dtype=torch.float16),
        "teacher_visibility_probability": torch.zeros(2, dtype=torch.float16),
        "teacher_confidence_probability": torch.zeros(2, dtype=torch.float16),
        "frame_feature_pyramid": [torch.zeros(128, 2, 2, dtype=torch.float16) for _ in range(4)],
        "native_track_feat": [torch.zeros(2, 1, 128, dtype=torch.float16) for _ in range(4)],
        "native_track_support": [torch.zeros(2, 49, 128, dtype=torch.float16) for _ in range(4)],
        "teacher_track_feat": [torch.zeros(2, 1, 128, dtype=torch.float16) for _ in range(4)],
        "teacher_track_support": [torch.zeros(2, 49, 128, dtype=torch.float16) for _ in range(4)],
    }
    hashed = {"exact_native_state": exact, "model_tensors": model_tensors}
    artifact = {
        "schema_version": CSRR_CACHE_SCHEMA_VERSION,
        "partition": "smoke",
        "source_index": 8,
        "provenance": {"config_sha256": "abc"},
        "exact_native_state": exact,
        "model_tensors": model_tensors,
        "tensor_hashes": nested_tensor_hashes(hashed),
        "quantization": {"pass": True},
    }
    path = tmp_path / "video.pt"
    torch.save(artifact, path)
    verify_csrr_cache_artifact(path, expected_partition="smoke", expected_source_index=8, expected_config_sha256="abc")
    artifact["model_tensors"]["trajectory_features"][0, 0, 0] = 1
    torch.save(artifact, path)
    try:
        verify_csrr_cache_artifact(path)
    except ValueError as error:
        assert "hash" in str(error)
    else:
        raise AssertionError("tampered artifact was accepted")
