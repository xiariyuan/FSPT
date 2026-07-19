"""Nonparametric geometry-preserving support matching for Route-D.

Unlike the Gate 2 restorer, this interface never attention-pools the 7x7
CoTracker support into one vector. Every support token is compared to the target
feature at its corresponding spatial offset, then token evidence is aggregated
robustly. No selector, action head, or learned parameter is defined here.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


GEOMETRY_MATCHER_SCHEMA_VERSION = "routeD_geometry_preserving_matcher_interface_v0"


def cotracker_to_unfold_support_order(support_radius: int) -> torch.Tensor:
    """Map CoTracker's x-major support order to PyTorch unfold's y-major order."""
    radius = int(support_radius)
    if radius < 0:
        raise ValueError("support radius must be non-negative")
    width = 2 * radius + 1
    # CoTracker constructs meshgrid(x, y, indexing="ij"): x is the outer
    # dimension and y is inner. Unfold flattens kernel rows: y outer, x inner.
    return torch.tensor(
        [y * width + x for x in range(width) for y in range(width)],
        dtype=torch.long,
    )


def geometry_preserving_support_score(
    frame_feature: torch.Tensor,
    anchor_support: torch.Tensor,
    *,
    support_radius: int = 3,
    trim_fraction: float = 0.125,
    row_chunk_size: int = 8,
) -> torch.Tensor:
    """Score every candidate center while retaining support-token geometry.

    Args:
        frame_feature: Shared target feature map ``[1,C,H,W]``.
        anchor_support: Per-row CoTracker support in x-major order ``[R,S,C]``.
        support_radius: CoTracker support radius; ``S=(2r+1)^2``.
        trim_fraction: Fraction removed from each tail before token averaging.
        row_chunk_size: Bounds the materialized ``R x S x H x W`` score tensor.

    Returns:
        Robust cosine score maps with shape ``[R,H,W]``.

    Replicate padding matches the clamped border semantics used by deterministic
    CoTracker memory re-extraction. Trimming is symmetric and fixed; it prevents
    a small number of occluded or corrupted tokens from dominating the score.
    """
    if frame_feature.ndim != 4 or frame_feature.shape[0] != 1:
        raise ValueError("frame feature must have shape [1,C,H,W]")
    if anchor_support.ndim != 3:
        raise ValueError("anchor support must have shape [R,S,C]")
    rows, support_tokens, channels = anchor_support.shape
    if channels != frame_feature.shape[1]:
        raise ValueError("frame/support channel mismatch")
    radius = int(support_radius)
    kernel = 2 * radius + 1
    if support_tokens != kernel * kernel:
        raise ValueError("support token count does not match support radius")
    if not 0.0 <= float(trim_fraction) < 0.5:
        raise ValueError("trim fraction must be in [0, 0.5)")
    if int(row_chunk_size) <= 0:
        raise ValueError("row chunk size must be positive")
    if rows == 0:
        return frame_feature.new_empty(
            (0, frame_feature.shape[-2], frame_feature.shape[-1])
        )

    height, width = frame_feature.shape[-2:]
    normalized_frame = F.normalize(
        frame_feature.float(), p=2, dim=1, eps=1.0e-8
    )
    padded = F.pad(
        normalized_frame,
        (radius, radius, radius, radius),
        mode="replicate",
    )
    unfolded = F.unfold(padded, kernel_size=kernel, padding=0, stride=1)
    # Unfold layout is [1,C,S,H*W]. Reorder kernel positions into CoTracker's
    # x-major support order before token-wise comparison.
    patches = unfolded.view(1, channels, support_tokens, height * width)[0]
    order = cotracker_to_unfold_support_order(radius).to(patches.device)
    patches = patches[:, order].permute(1, 0, 2).contiguous()  # [S,C,L]
    patches = F.normalize(patches, p=2, dim=1, eps=1.0e-8)

    normalized_anchor = F.normalize(
        anchor_support.float(), p=2, dim=-1, eps=1.0e-8
    )
    trim = int(support_tokens * float(trim_fraction))
    if support_tokens - 2 * trim <= 0:
        raise ValueError("trim removes every support token")
    output = []
    for start in range(0, rows, int(row_chunk_size)):
        anchor = normalized_anchor[start : start + int(row_chunk_size)]
        token_scores = torch.einsum("rsc,scl->rsl", anchor, patches)
        if trim > 0:
            token_scores = token_scores.sort(dim=1).values[:, trim:-trim]
        robust = token_scores.mean(dim=1)
        output.append(robust.view(-1, height, width))
    return torch.cat(output, dim=0).to(dtype=frame_feature.dtype)


def geometry_preserving_pyramid_score(
    frame_feature_pyramid: Sequence[torch.Tensor],
    anchor_support_pyramid: Sequence[torch.Tensor],
    *,
    common_height: int = 64,
    common_width: int = 64,
    support_radius: int = 3,
    trim_fraction: float = 0.125,
    row_chunk_size: int = 8,
    level_weights: Sequence[float] | None = None,
) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
    """Compute per-level maps and a fixed weighted multi-level mean."""
    if len(frame_feature_pyramid) != len(anchor_support_pyramid):
        raise ValueError("frame/support pyramid level mismatch")
    if not frame_feature_pyramid:
        raise ValueError("feature pyramid must be non-empty")
    levels = len(frame_feature_pyramid)
    if level_weights is None:
        weights = [1.0 / levels] * levels
    else:
        if len(level_weights) != levels:
            raise ValueError("level weight count mismatch")
        total = float(sum(float(value) for value in level_weights))
        if total <= 0.0 or any(float(value) < 0.0 for value in level_weights):
            raise ValueError("level weights must be non-negative with positive sum")
        weights = [float(value) / total for value in level_weights]

    maps = []
    common = []
    expected_rows = int(anchor_support_pyramid[0].shape[0])
    for frame, support in zip(frame_feature_pyramid, anchor_support_pyramid):
        if int(support.shape[0]) != expected_rows:
            raise ValueError("support pyramid row mismatch")
        score = geometry_preserving_support_score(
            frame,
            support,
            support_radius=support_radius,
            trim_fraction=trim_fraction,
            row_chunk_size=row_chunk_size,
        )
        maps.append(score)
        common.append(
            F.interpolate(
                score[:, None],
                size=(int(common_height), int(common_width)),
                mode="bilinear",
                align_corners=True,
            )[:, 0]
        )
    weight_tensor = common[0].new_tensor(weights).view(levels, 1, 1, 1)
    stacked = torch.stack(common, dim=0)
    fused = (stacked * weight_tensor).sum(dim=0)
    return {
        "level_score_maps": tuple(maps),
        "common_level_score_maps": tuple(common),
        "fused_score_map": fused,
    }


def robust_multi_anchor_consensus(
    anchor_score_maps: torch.Tensor,
    *,
    mode: str = "median",
) -> torch.Tensor:
    """Fuse independently frozen anchor maps without learned anchor attention.

    ``anchor_score_maps`` has shape ``[A,R,H,W]``. Median is the default because
    one stale anchor cannot dominate a three-anchor bank.
    """
    if anchor_score_maps.ndim != 4 or anchor_score_maps.shape[0] == 0:
        raise ValueError("anchor maps must have shape [A,R,H,W]")
    if mode == "median":
        return anchor_score_maps.median(dim=0).values
    if mode == "minimum":
        return anchor_score_maps.min(dim=0).values
    if mode == "mean":
        return anchor_score_maps.mean(dim=0)
    raise ValueError(f"unsupported anchor consensus mode: {mode}")
