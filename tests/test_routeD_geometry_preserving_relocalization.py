import pytest
import torch
import torch.nn.functional as F

from projects.mmp_tracker.mmp_tracker.routeD_geometry_preserving_relocalization import (
    cotracker_to_unfold_support_order,
    geometry_preserving_pyramid_score,
    geometry_preserving_support_score,
    robust_multi_anchor_consensus,
)


def _support_at(frame, center_xy, radius):
    _, channels, height, width = frame.shape
    cx, cy = center_xy
    rows = []
    for x_offset in range(-radius, radius + 1):
        for y_offset in range(-radius, radius + 1):
            x = min(max(cx + x_offset, 0), width - 1)
            y = min(max(cy + y_offset, 0), height - 1)
            rows.append(frame[0, :, y, x])
    return torch.stack(rows)[None]


def test_support_order_maps_x_major_cotracker_to_y_major_unfold():
    assert cotracker_to_unfold_support_order(1).tolist() == [
        0,
        3,
        6,
        1,
        4,
        7,
        2,
        5,
        8,
    ]


def test_geometry_score_peaks_at_exact_asymmetric_patch_center():
    generator = torch.Generator().manual_seed(9)
    frame = F.normalize(torch.randn(1, 32, 9, 11, generator=generator), dim=1)
    center = (7, 3)
    support = _support_at(frame, center, radius=1)
    score = geometry_preserving_support_score(
        frame, support, support_radius=1, trim_fraction=0.0, row_chunk_size=1
    )
    flat_index = int(score[0].argmax())
    observed = (flat_index % frame.shape[-1], flat_index // frame.shape[-1])
    assert observed == center
    assert float(score[0, center[1], center[0]]) == pytest.approx(1.0, abs=1e-6)


def test_geometry_score_matches_clamped_border_support():
    generator = torch.Generator().manual_seed(11)
    frame = F.normalize(torch.randn(1, 24, 8, 8, generator=generator), dim=1)
    support = _support_at(frame, (0, 0), radius=1)
    score = geometry_preserving_support_score(
        frame, support, support_radius=1, trim_fraction=0.0
    )
    assert int(score[0].argmax()) == 0
    assert float(score[0, 0, 0]) == pytest.approx(1.0, abs=1e-6)


def test_trimmed_geometry_score_tolerates_one_corrupt_token():
    generator = torch.Generator().manual_seed(13)
    frame = F.normalize(torch.randn(1, 32, 9, 9, generator=generator), dim=1)
    support = _support_at(frame, (4, 4), radius=1)
    support[:, 0] = -support[:, 0]
    untrimmed = geometry_preserving_support_score(
        frame, support, support_radius=1, trim_fraction=0.0
    )
    trimmed = geometry_preserving_support_score(
        frame, support, support_radius=1, trim_fraction=1.0 / 9.0
    )
    assert float(trimmed[0, 4, 4]) > float(untrimmed[0, 4, 4])


def test_pyramid_and_multi_anchor_interfaces_have_fixed_shapes():
    generator = torch.Generator().manual_seed(17)
    frame0 = F.normalize(torch.randn(1, 16, 8, 8, generator=generator), dim=1)
    frame1 = F.normalize(torch.randn(1, 16, 4, 4, generator=generator), dim=1)
    support0 = torch.cat([_support_at(frame0, (3, 3), 1)] * 2)
    support1 = torch.cat([_support_at(frame1, (2, 2), 1)] * 2)
    result = geometry_preserving_pyramid_score(
        [frame0, frame1],
        [support0, support1],
        common_height=6,
        common_width=7,
        support_radius=1,
        trim_fraction=0.0,
    )
    assert result["fused_score_map"].shape == (2, 6, 7)
    anchors = torch.stack(
        [result["fused_score_map"], result["fused_score_map"] - 0.1, result["fused_score_map"] + 2]
    )
    consensus = robust_multi_anchor_consensus(anchors, mode="median")
    assert torch.allclose(consensus, result["fused_score_map"])
