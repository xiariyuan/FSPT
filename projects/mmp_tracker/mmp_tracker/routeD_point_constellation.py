"""Local point-constellation descriptors for long-occlusion identity audits.

This module intentionally contains no learned selector, visibility override, or
tracker-state writeback.  It asks a narrower falsifiable question: does the
local spatial arrangement of backbone features distinguish the correct
post-occlusion point better than the centre-point appearance alone?
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .routeD_safe_redetection import sample_batched_map


POINT_CONSTELLATION_SCHEMA_VERSION = "routeD_point_constellation_v0"


_DEFAULT_OFFSETS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (-4.0, 0.0),
    (4.0, 0.0),
    (0.0, -4.0),
    (0.0, 4.0),
    (-8.0, -8.0),
    (8.0, -8.0),
    (-8.0, 8.0),
    (8.0, 8.0),
    (-16.0, 0.0),
    (16.0, 0.0),
    (0.0, -16.0),
    (0.0, 16.0),
)


@dataclass(frozen=True)
class PointConstellationConfig:
    """Geometry of a fixed causal local constellation in 256-raster pixels."""

    input_height: int = 256
    input_width: int = 256
    offsets_xy_px: tuple[tuple[float, float], ...] = _DEFAULT_OFFSETS
    exclude_center_from_direct: bool = True

    def __post_init__(self) -> None:
        if self.input_height <= 1 or self.input_width <= 1:
            raise ValueError("input dimensions must exceed one pixel")
        if len(self.offsets_xy_px) < 3:
            raise ValueError("a constellation needs at least three sample points")
        if tuple(self.offsets_xy_px[0]) != (0.0, 0.0):
            raise ValueError("offset zero must be the centre point")
        if len(set(tuple(row) for row in self.offsets_xy_px)) != len(self.offsets_xy_px):
            raise ValueError("constellation offsets must be unique")


def offset_tensor(
    config: PointConstellationConfig,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(config.offsets_xy_px, device=device, dtype=dtype)


def constellation_coordinates(
    coords_xy_px: torch.Tensor,
    config: PointConstellationConfig,
) -> torch.Tensor:
    """Expand BxKx2 centres to BxKxPx2 bounded constellation coordinates."""
    if coords_xy_px.ndim != 3 or coords_xy_px.shape[-1] != 2:
        raise ValueError("coords_xy_px must have shape (B,K,2)")
    offsets = offset_tensor(config, device=coords_xy_px.device, dtype=coords_xy_px.dtype)
    coordinates = coords_xy_px[:, :, None, :] + offsets[None, None, :, :]
    x = coordinates[..., 0].clamp(0.0, float(config.input_width - 1))
    y = coordinates[..., 1].clamp(0.0, float(config.input_height - 1))
    return torch.stack((x, y), dim=-1)


def sample_point_constellations(
    feature_map: torch.Tensor,
    coords_xy_px: torch.Tensor,
    config: PointConstellationConfig = PointConstellationConfig(),
) -> torch.Tensor:
    """Sample normalized local constellations, returning BxKxPxD."""
    if feature_map.ndim != 4:
        raise ValueError("feature_map must have shape (B,D,H,W)")
    if coords_xy_px.shape[0] != feature_map.shape[0]:
        raise ValueError("feature-map and coordinate batch sizes differ")
    coordinates = constellation_coordinates(coords_xy_px, config)
    batch, candidates, points, _ = coordinates.shape
    sampled = sample_batched_map(
        feature_map,
        coordinates.reshape(batch, candidates * points, 2),
        # sample_batched_map only needs the input raster fields.
        config,  # type: ignore[arg-type]
    )
    sampled = sampled.reshape(batch, candidates, points, feature_map.shape[1])
    return F.normalize(sampled.float(), dim=-1, eps=1e-12)


def direct_correspondence_similarity(
    reference: torch.Tensor,
    candidates: torch.Tensor,
    *,
    exclude_center: bool = True,
) -> torch.Tensor:
    """Corresponding-offset cosine similarity, returning BxK."""
    if reference.ndim != 3 or candidates.ndim != 4:
        raise ValueError("reference must be BxPxD and candidates BxKxPxD")
    if reference.shape[0] != candidates.shape[0] or reference.shape[1:] != candidates.shape[2:]:
        raise ValueError("reference and candidate constellation shapes are incompatible")
    point_similarity = (reference[:, None] * candidates).sum(dim=-1)
    if exclude_center:
        if point_similarity.shape[-1] <= 1:
            raise ValueError("cannot exclude centre from a one-point constellation")
        point_similarity = point_similarity[..., 1:]
    return point_similarity.mean(dim=-1)


def _self_similarity_signature(constellations: torch.Tensor) -> torch.Tensor:
    if constellations.ndim < 3:
        raise ValueError("constellations must end with point and feature dimensions")
    points = constellations.shape[-2]
    if points < 3:
        raise ValueError("at least three points are required for a structure signature")
    matrix = torch.matmul(constellations, constellations.transpose(-1, -2))
    upper = torch.triu_indices(points, points, offset=1, device=constellations.device)
    signature = matrix[..., upper[0], upper[1]]
    signature = signature - signature.mean(dim=-1, keepdim=True)
    return F.normalize(signature.float(), dim=-1, eps=1e-12)


def structural_self_similarity(
    reference: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """Cosine agreement between local self-similarity graphs, returning BxK."""
    if reference.ndim != 3 or candidates.ndim != 4:
        raise ValueError("reference must be BxPxD and candidates BxKxPxD")
    if reference.shape[0] != candidates.shape[0] or reference.shape[1:] != candidates.shape[2:]:
        raise ValueError("reference and candidate constellation shapes are incompatible")
    reference_signature = _self_similarity_signature(reference)
    candidate_signature = _self_similarity_signature(candidates)
    return (reference_signature[:, None] * candidate_signature).sum(dim=-1)


def combined_constellation_score(
    center_score: torch.Tensor,
    direct_score: torch.Tensor,
    structure_score: torch.Tensor,
    *,
    direct_weight: float,
    structure_weight: float,
) -> torch.Tensor:
    """Combine three bounded evidence terms without hidden thresholds."""
    if center_score.shape != direct_score.shape or center_score.shape != structure_score.shape:
        raise ValueError("all candidate scores must share shape (B,K)")
    if direct_weight < 0.0 or structure_weight < 0.0:
        raise ValueError("constellation weights must be non-negative")
    return center_score + float(direct_weight) * direct_score + float(structure_weight) * structure_score


def candidate_distance_utility(
    candidate_coords_xy_px: torch.Tensor,
    gt_coord_xy_px: torch.Tensor,
    thresholds_px: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 16.0),
    weights: Sequence[float] = (0.10, 0.15, 0.20, 0.25, 0.30),
) -> torch.Tensor:
    """Ground-truth audit utility only; never used as an inference feature."""
    if candidate_coords_xy_px.ndim != 3 or candidate_coords_xy_px.shape[-1] != 2:
        raise ValueError("candidate coordinates must have shape (B,K,2)")
    if gt_coord_xy_px.shape != (candidate_coords_xy_px.shape[0], 2):
        raise ValueError("ground-truth coordinates must have shape (B,2)")
    if len(thresholds_px) != len(weights) or not thresholds_px:
        raise ValueError("threshold and weight lengths must match and be non-empty")
    distance = torch.linalg.norm(candidate_coords_xy_px - gt_coord_xy_px[:, None], dim=-1)
    threshold = torch.tensor(thresholds_px, device=distance.device, dtype=distance.dtype)
    weight = torch.tensor(weights, device=distance.device, dtype=distance.dtype)
    weight = weight / weight.sum()
    return ((distance[..., None] <= threshold).to(distance.dtype) * weight).sum(dim=-1)
