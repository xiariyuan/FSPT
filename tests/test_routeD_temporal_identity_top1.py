import torch

from projects.mmp_tracker.mmp_tracker.routeD_temporal_identity_features import (
    STATIC_FEATURE_CHANNELS,
    TEMPORAL_FEATURE_CHANNELS,
)
from projects.mmp_tracker.mmp_tracker.routeD_temporal_identity_top1 import (
    build_full_bank_top1_features,
    build_shortlist_top1_features,
)


def _inputs(rows=3, candidates=129):
    generator = torch.Generator().manual_seed(7)
    temporal = torch.randn(
        rows, candidates, 16, len(TEMPORAL_FEATURE_CHANNELS), generator=generator
    )
    static = torch.randn(
        rows, candidates, len(STATIC_FEATURE_CHANNELS), generator=generator
    )
    valid = torch.ones(rows, candidates, dtype=torch.bool)
    query = torch.tensor([0, 3, 7][:rows], dtype=torch.long)
    return temporal, static, query, valid


def test_top1_feature_shapes_and_native_role():
    temporal, static, query, valid = _inputs()
    full = build_full_bank_top1_features(
        temporal_features=temporal,
        static_features=static,
        query_frames=query,
        valid_mask=valid,
    )
    assert full.shape == (3, 129, 98)
    result = build_shortlist_top1_features(
        temporal_features=temporal,
        static_features=static,
        query_frames=query,
        valid_mask=valid,
    )
    assert result["selected_indices"].shape == (3, 9)
    assert result["candidate_features"].shape == (3, 9, 102)
    assert torch.equal(result["selected_indices"][:, 0], torch.zeros(3, dtype=torch.long))
    assert torch.all(result["candidate_features"][:, 0, -1] == 1.0)
    assert torch.all(result["candidate_features"][:, 1:, -1] == 0.0)


def test_top1_features_are_deterministic_and_teacher_free():
    temporal, static, query, valid = _inputs()
    first = build_shortlist_top1_features(
        temporal_features=temporal,
        static_features=static,
        query_frames=query,
        valid_mask=valid,
    )
    second = build_shortlist_top1_features(
        temporal_features=temporal.clone(),
        static_features=static.clone(),
        query_frames=query.clone(),
        valid_mask=valid.clone(),
    )
    assert torch.equal(first["selected_indices"], second["selected_indices"])
    assert torch.equal(first["candidate_features"], second["candidate_features"])


def test_top1_empty_rows_preserve_contract():
    temporal, static, query, valid = _inputs(rows=0)
    result = build_shortlist_top1_features(
        temporal_features=temporal,
        static_features=static,
        query_frames=query,
        valid_mask=valid,
    )
    assert result["selected_indices"].shape == (0, 9)
    assert result["candidate_features"].shape == (0, 9, 102)
