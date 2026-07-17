from __future__ import annotations

import numpy as np
import torch

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import (
    CANDIDATE_FEATURE_DIM,
    LOCAL_CORRELATION_SOURCE_ID,
    NATIVE_SOURCE_ID,
    STATE_FEATURE_DIM,
    build_state_feature,
    gather_candidate_coordinates,
    local_correlation_candidates,
    make_first_visible_queries,
    oracle_candidate_indices,
    tensor_sha256,
)


def test_first_visible_queries_filter_never_visible_and_use_yx_contract():
    tracks = np.array(
        [
            [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
            [[0.5, 0.6], [0.6, 0.7], [0.7, 0.8]],
            [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]],
        ],
        dtype=np.float32,
    )
    occluded = np.array(
        [[True, False, False], [False, False, True], [True, True, True]], dtype=bool
    )
    queries, filtered_tracks, filtered_occ = make_first_visible_queries(tracks, occluded)
    np.testing.assert_allclose(queries, [[1.0, 0.2, 0.3], [0.0, 0.5, 0.6]])
    assert filtered_tracks.shape == (2, 3, 2)
    assert filtered_occ.shape == (2, 3)


def test_local_search_is_deterministic_and_preserves_native_candidate_zero():
    fmap = torch.zeros(4, 5, 5)
    fmap[0] = 1.0
    fmap[:, 2, 3] = torch.tensor([1.0, 1.0, 0.0, 0.0])
    fmap[:, 1, 1] = torch.tensor([1.0, 1.0, 0.0, 0.0])
    support = torch.tensor([1.0, 1.0, 0.0, 0.0])
    kwargs = dict(
        fmap=fmap,
        support_feature=support,
        native_xy_px=torch.tensor([8.0, 8.0]),
        previous_native_xy_px=torch.tensor([7.0, 8.0]),
        native_visibility_probability=0.8,
        native_confidence_probability=0.9,
        input_height=17,
        input_width=17,
        radius_px=16.0,
        topk=3,
    )
    first = local_correlation_candidates(**kwargs)
    second = local_correlation_candidates(**kwargs)
    assert torch.equal(first.candidate_xy_px, second.candidate_xy_px)
    assert torch.equal(first.candidate_features, second.candidate_features)
    assert first.candidate_features.shape == (4, CANDIDATE_FEATURE_DIM)
    assert first.source_ids.tolist() == [
        NATIVE_SOURCE_ID,
        LOCAL_CORRELATION_SOURCE_ID,
        LOCAL_CORRELATION_SOURCE_ID,
        LOCAL_CORRELATION_SOURCE_ID,
    ]
    assert torch.equal(first.candidate_xy_px[0], kwargs["native_xy_px"])
    assert first.candidate_valid.all()


def test_state_feature_contract_has_32_dimensions():
    feature = build_state_feature(
        native_xy_px=torch.tensor([20.0, 30.0]),
        previous_native_xy_px=torch.tensor([18.0, 29.0]),
        previous_previous_native_xy_px=torch.tensor([17.0, 27.0]),
        visibility_probability=0.7,
        confidence_probability=0.8,
        frame_index=4,
        frame_count=10,
        query_frame=2,
        online_track_feature=torch.arange(40, dtype=torch.float32),
        input_height=256,
        input_width=256,
        radius_px=64.0,
    )
    assert feature.shape == (STATE_FEATURE_DIM,)
    assert torch.isfinite(feature).all()


def test_oracle_never_selects_invalid_and_gather_matches():
    candidates = torch.tensor(
        [
            [[[0.0, 0.0], [5.0, 5.0], [1.0, 1.0]]],
            [[[3.0, 3.0], [2.0, 2.0], [9.0, 9.0]]],
        ]
    )
    valid = torch.tensor([[[True, False, True]], [[True, True, False]]])
    gt = torch.tensor([[[1.2, 1.1]], [[2.1, 2.2]]])
    index, best, native = oracle_candidate_indices(candidates, valid, gt)
    assert index.tolist() == [[2], [1]]
    assert torch.all(best < native)
    gathered = gather_candidate_coordinates(candidates, index)
    torch.testing.assert_close(gathered, torch.tensor([[[1.0, 1.0]], [[2.0, 2.0]]]))


def test_tensor_hash_includes_shape_and_content():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b = a.reshape(3, 2)
    c = a.clone(); c[0, 0] = 1.0
    assert tensor_sha256(a) == tensor_sha256(a.clone())
    assert tensor_sha256(a) != tensor_sha256(b)
    assert tensor_sha256(a) != tensor_sha256(c)
