from __future__ import annotations

import torch

from projects.mmp_tracker.mmp_tracker.routeD_raw_representation import (
    RAW_V1_CANDIDATE_FEATURE_DIM,
    RAW_V1_STATE_FEATURE_DIM,
    build_raw_v1_candidate_features_for_frame,
    build_raw_v1_state_features,
    sample_candidate_centered_correlation_patches,
)


def test_raw_candidate_contract_and_invalid_zeroing():
    torch.manual_seed(3)
    fmap = torch.randn(128, 9, 11)
    support = torch.randn(2, 128)
    coords = torch.tensor(
        [[[4.0, 5.0], [8.0, 10.0], [12.0, 16.0]], [[2.0, 3.0], [7.0, 8.0], [14.0, 15.0]]]
    )
    structured = torch.randn(2, 3, 64)
    valid = torch.tensor([[True, True, False], [True, False, True]])
    output = build_raw_v1_candidate_features_for_frame(
        structured_candidate_features=structured,
        candidate_valid_mask=valid,
        candidate_xy_px=coords,
        support_features=support,
        fmap=fmap,
        input_height=32,
        input_width=32,
    )
    assert output.shape == (2, 3, RAW_V1_CANDIDATE_FEATURE_DIM)
    assert torch.equal(output[~valid], torch.zeros_like(output[~valid]))
    assert torch.allclose(output[..., :64][valid], structured[valid])
    assert torch.isfinite(output).all()


def test_correlation_patch_center_matches_direct_correlation():
    torch.manual_seed(5)
    fmap = torch.randn(128, 8, 8)
    support = torch.randn(1, 128)
    coords = torch.tensor([[[12.0, 20.0]]])
    patches = sample_candidate_centered_correlation_patches(
        fmap,
        support,
        coords,
        input_height=32,
        input_width=32,
    )
    normalized_fmap = torch.nn.functional.normalize(fmap, dim=0)
    normalized_support = torch.nn.functional.normalize(support[0], dim=0)
    corr = torch.einsum("d,dhw->hw", normalized_support, normalized_fmap)
    x = coords[0, 0, 0] / 31.0 * 7.0
    y = coords[0, 0, 1] / 31.0 * 7.0
    grid = torch.tensor([[[[2.0 * x / 7.0 - 1.0, 2.0 * y / 7.0 - 1.0]]]])
    expected = torch.nn.functional.grid_sample(
        corr[None, None], grid, mode="bilinear", padding_mode="border", align_corners=True
    )[0, 0, 0, 0]
    assert torch.allclose(patches[0, 0, 12], expected, atol=2e-6, rtol=2e-6)


def test_state_contract_repeats_full_track_feature():
    structured = torch.randn(3, 4, 32)
    track = torch.randn(3, 128)
    output = build_raw_v1_state_features(structured, track)
    assert output.shape == (3, 4, RAW_V1_STATE_FEATURE_DIM)
    assert torch.equal(output[..., :32], structured)
    assert torch.equal(output[:, 0, 32:], track)
    assert torch.equal(output[:, -1, 32:], track)


def test_raw_feature_replay_is_exact():
    torch.manual_seed(9)
    args = dict(
        structured_candidate_features=torch.randn(2, 3, 64),
        candidate_valid_mask=torch.ones(2, 3, dtype=torch.bool),
        candidate_xy_px=torch.rand(2, 3, 2) * 31.0,
        support_features=torch.randn(2, 128),
        fmap=torch.randn(128, 8, 8),
        input_height=32,
        input_width=32,
    )
    first = build_raw_v1_candidate_features_for_frame(**args)
    second = build_raw_v1_candidate_features_for_frame(**args)
    assert torch.equal(first, second)


def test_frozen_contract_accepts_representation_only_change():
    from projects.mmp_tracker.mmp_tracker.routeD_raw_representation import (
        FROZEN_SHARED_TENSOR_KEYS,
        verify_raw_v1_frozen_contract,
    )

    base = {key: torch.zeros(2, 3) for key in FROZEN_SHARED_TENSOR_KEYS}
    base["candidate_coords_xy_px"] = torch.zeros(2, 3, 4, 2)
    base["candidate_scores"] = torch.zeros(2, 3, 4)
    base["candidate_valid_mask"] = torch.ones(2, 3, 4, dtype=torch.bool)
    base["source_ids"] = torch.zeros(2, 3, 4, dtype=torch.long)
    base["native_coords_xy_px"] = torch.zeros(2, 3, 2)
    base["native_visibility_probability"] = torch.zeros(2, 3)
    base["native_confidence_probability"] = torch.zeros(2, 3)
    base["native_joint_probability"] = torch.zeros(2, 3)
    base["native_visibility"] = torch.zeros(2, 3, dtype=torch.bool)
    base["query_points_tyx"] = torch.zeros(2, 3)
    base["gt_tracks_yx"] = torch.zeros(2, 3, 2)
    base["gt_occluded"] = torch.zeros(2, 3, dtype=torch.bool)
    base["oracle_candidate_index"] = torch.zeros(2, 3, dtype=torch.long)
    base["oracle_coords_xy_px"] = torch.zeros(2, 3, 2)
    base["oracle_error_px"] = torch.zeros(2, 3)
    base["native_error_px"] = torch.zeros(2, 3)
    base["local_peak"] = torch.zeros(2, 3)
    base["local_margin"] = torch.zeros(2, 3)
    base["local_entropy"] = torch.zeros(2, 3)
    base["search_cell_count"] = torch.zeros(2, 3, dtype=torch.int32)
    base["candidate_features"] = torch.zeros(2, 3, 4, 64)
    base["state_features"] = torch.zeros(2, 3, 32)
    raw = {key: value.clone() for key, value in base.items()}
    raw["candidate_features"] = torch.zeros(2, 3, 4, RAW_V1_CANDIDATE_FEATURE_DIM)
    raw["state_features"] = torch.zeros(2, 3, RAW_V1_STATE_FEATURE_DIM)
    hashes = verify_raw_v1_frozen_contract(base, raw)
    assert set(hashes) == set(FROZEN_SHARED_TENSOR_KEYS)


def test_frozen_contract_rejects_coordinate_drift():
    from projects.mmp_tracker.mmp_tracker.routeD_raw_representation import (
        FROZEN_SHARED_TENSOR_KEYS,
        verify_raw_v1_frozen_contract,
    )

    base = {key: torch.zeros(1) for key in FROZEN_SHARED_TENSOR_KEYS}
    base.update({
        "candidate_coords_xy_px": torch.zeros(1, 1, 1, 2),
        "candidate_scores": torch.zeros(1, 1, 1),
        "candidate_valid_mask": torch.ones(1, 1, 1, dtype=torch.bool),
        "source_ids": torch.zeros(1, 1, 1, dtype=torch.long),
        "native_coords_xy_px": torch.zeros(1, 1, 2),
        "native_visibility_probability": torch.zeros(1, 1),
        "native_confidence_probability": torch.zeros(1, 1),
        "native_joint_probability": torch.zeros(1, 1),
        "native_visibility": torch.zeros(1, 1, dtype=torch.bool),
        "query_points_tyx": torch.zeros(1, 3),
        "gt_tracks_yx": torch.zeros(1, 1, 2),
        "gt_occluded": torch.zeros(1, 1, dtype=torch.bool),
        "oracle_candidate_index": torch.zeros(1, 1, dtype=torch.long),
        "oracle_coords_xy_px": torch.zeros(1, 1, 2),
        "oracle_error_px": torch.zeros(1, 1),
        "native_error_px": torch.zeros(1, 1),
        "local_peak": torch.zeros(1, 1),
        "local_margin": torch.zeros(1, 1),
        "local_entropy": torch.zeros(1, 1),
        "search_cell_count": torch.zeros(1, 1, dtype=torch.int32),
        "candidate_features": torch.zeros(1, 1, 1, 64),
        "state_features": torch.zeros(1, 1, 32),
    })
    raw = {key: value.clone() for key, value in base.items()}
    raw["candidate_features"] = torch.zeros(1, 1, 1, RAW_V1_CANDIDATE_FEATURE_DIM)
    raw["state_features"] = torch.zeros(1, 1, RAW_V1_STATE_FEATURE_DIM)
    raw["candidate_coords_xy_px"][0, 0, 0, 0] = 1.0
    try:
        verify_raw_v1_frozen_contract(base, raw)
    except ValueError as exc:
        assert "candidate_coords_xy_px" in str(exc)
    else:
        raise AssertionError("coordinate drift must be rejected")
