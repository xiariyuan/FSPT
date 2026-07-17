"""Raw CoTracker feature representation for Route-D MUSR P0e.

This module upgrades only the candidate/state representation. Candidate
coordinates, validity, source identities, native state, and oracle labels are
owned by the frozen stage-0 cache and must remain byte-identical.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .cotracker3_stage0_adapter import sample_feature_at_xy


RAW_V1_CANDIDATE_FEATURE_DIM = 601
RAW_V1_STATE_FEATURE_DIM = 160
RAW_V1_PATCH_SIZE = 5
RAW_V1_BACKBONE_FEATURE_DIM = 128
RAW_V1_SCHEMA_VERSION = "routeD_musr_cotracker3_raw_representation_v1"


@dataclass(frozen=True)
class RawRepresentationShapes:
    candidate_feature_dim: int = RAW_V1_CANDIDATE_FEATURE_DIM
    state_feature_dim: int = RAW_V1_STATE_FEATURE_DIM
    patch_size: int = RAW_V1_PATCH_SIZE
    backbone_feature_dim: int = RAW_V1_BACKBONE_FEATURE_DIM


def _validate_backbone_dim(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.shape[-1] != RAW_V1_BACKBONE_FEATURE_DIM:
        raise ValueError(
            f"{name} must have final dimension {RAW_V1_BACKBONE_FEATURE_DIM}, "
            f"got {tensor.shape[-1]}"
        )


def sample_candidate_centered_correlation_patches(
    fmap: torch.Tensor,
    support_features: torch.Tensor,
    candidate_xy_px: torch.Tensor,
    *,
    input_height: int,
    input_width: int,
    patch_size: int = RAW_V1_PATCH_SIZE,
) -> torch.Tensor:
    """Sample support-correlation patches around candidate coordinates.

    Args:
        fmap: normalized current-frame features, ``(D,Hf,Wf)``.
        support_features: normalized query features, ``(N,D)``.
        candidate_xy_px: candidate positions, ``(N,K,2)`` in input pixels.

    Returns:
        Correlation patches ``(N,K,patch_size*patch_size)``. Border values use
        border padding, matching the candidate feature sampler.
    """
    if fmap.ndim != 3:
        raise ValueError("fmap must have shape (D,Hf,Wf)")
    if support_features.ndim != 2:
        raise ValueError("support_features must have shape (N,D)")
    if candidate_xy_px.ndim != 3 or candidate_xy_px.shape[-1] != 2:
        raise ValueError("candidate_xy_px must have shape (N,K,2)")
    if candidate_xy_px.shape[0] != support_features.shape[0]:
        raise ValueError("point count mismatch")
    if patch_size <= 0 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer")
    if fmap.shape[0] != support_features.shape[1]:
        raise ValueError("feature channel mismatch")

    fmap = F.normalize(fmap.float(), dim=0, eps=1.0e-12)
    support = F.normalize(support_features.float(), dim=-1, eps=1.0e-12)
    point_count, candidate_count = candidate_xy_px.shape[:2]
    _, feature_height, feature_width = fmap.shape
    correlation = torch.einsum("nd,dhw->nhw", support, fmap)

    xy = candidate_xy_px.to(device=fmap.device, dtype=fmap.dtype)
    feature_x = xy[..., 0] / float(max(input_width - 1, 1)) * float(
        max(feature_width - 1, 1)
    )
    feature_y = xy[..., 1] / float(max(input_height - 1, 1)) * float(
        max(feature_height - 1, 1)
    )
    radius = patch_size // 2
    offsets = torch.arange(-radius, radius + 1, device=fmap.device, dtype=fmap.dtype)
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    sample_x = feature_x[..., None, None] + offset_x
    sample_y = feature_y[..., None, None] + offset_y
    grid_x = 2.0 * sample_x / float(max(feature_width - 1, 1)) - 1.0
    grid_y = 2.0 * sample_y / float(max(feature_height - 1, 1)) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    # Stack K patches vertically in the output grid for one batched call.
    grid = grid.reshape(point_count, candidate_count * patch_size, patch_size, 2)
    sampled = F.grid_sample(
        correlation.unsqueeze(1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[:, 0].reshape(
        point_count, candidate_count, patch_size * patch_size
    )


def build_raw_v1_candidate_features_for_frame(
    *,
    structured_candidate_features: torch.Tensor,
    candidate_valid_mask: torch.Tensor,
    candidate_xy_px: torch.Tensor,
    support_features: torch.Tensor,
    fmap: torch.Tensor,
    input_height: int,
    input_width: int,
) -> torch.Tensor:
    """Build the 601-D raw-v1 candidate tokens for one frame."""
    if structured_candidate_features.ndim != 3:
        raise ValueError("structured_candidate_features must have shape (N,K,64)")
    if structured_candidate_features.shape[-1] != 64:
        raise ValueError("raw-v1 requires the frozen 64-D structured candidate feature")
    if candidate_valid_mask.shape != structured_candidate_features.shape[:2]:
        raise ValueError("candidate_valid_mask shape mismatch")
    if candidate_xy_px.shape != structured_candidate_features.shape[:2] + (2,):
        raise ValueError("candidate_xy_px shape mismatch")
    if support_features.shape[0] != structured_candidate_features.shape[0]:
        raise ValueError("support point count mismatch")
    _validate_backbone_dim(support_features, name="support_features")
    if fmap.shape[0] != RAW_V1_BACKBONE_FEATURE_DIM:
        raise ValueError("unexpected fmap channel count")

    point_count, candidate_count = candidate_valid_mask.shape
    current = sample_feature_at_xy(
        fmap,
        candidate_xy_px,
        input_height=input_height,
        input_width=input_width,
    )
    _validate_backbone_dim(current, name="candidate features")
    support = F.normalize(support_features.float(), dim=-1, eps=1.0e-12)
    support = support[:, None, :].expand(point_count, candidate_count, -1)
    current = F.normalize(current.float(), dim=-1, eps=1.0e-12)
    patches = sample_candidate_centered_correlation_patches(
        fmap,
        support_features,
        candidate_xy_px,
        input_height=input_height,
        input_width=input_width,
    )
    feature = torch.cat(
        [
            structured_candidate_features.float(),
            support,
            current,
            support - current,
            support * current,
            patches,
        ],
        dim=-1,
    )
    if feature.shape[-1] != RAW_V1_CANDIDATE_FEATURE_DIM:
        raise RuntimeError(f"raw-v1 candidate dimension drift: {feature.shape[-1]}")
    return torch.where(
        candidate_valid_mask.unsqueeze(-1), feature, torch.zeros_like(feature)
    )


def build_raw_v1_state_features(
    structured_state_features: torch.Tensor,
    online_track_features: torch.Tensor,
) -> torch.Tensor:
    """Append full level-0 online track state to the frozen 32-D state token.

    ``structured_state_features`` is ``(N,T,32)`` and
    ``online_track_features`` is ``(N,128)``.
    """
    if structured_state_features.ndim != 3 or structured_state_features.shape[-1] != 32:
        raise ValueError("structured_state_features must have shape (N,T,32)")
    if online_track_features.ndim != 2:
        raise ValueError("online_track_features must have shape (N,128)")
    if online_track_features.shape[0] != structured_state_features.shape[0]:
        raise ValueError("state point count mismatch")
    _validate_backbone_dim(online_track_features, name="online_track_features")
    repeated = online_track_features.float()[:, None, :].expand(
        -1, structured_state_features.shape[1], -1
    )
    state = torch.cat([structured_state_features.float(), repeated], dim=-1)
    if state.shape[-1] != RAW_V1_STATE_FEATURE_DIM:
        raise RuntimeError(f"raw-v1 state dimension drift: {state.shape[-1]}")
    return state

FROZEN_SHARED_TENSOR_KEYS = (
    "candidate_coords_xy_px",
    "candidate_scores",
    "candidate_valid_mask",
    "source_ids",
    "native_coords_xy_px",
    "native_visibility_probability",
    "native_confidence_probability",
    "native_joint_probability",
    "native_visibility",
    "query_points_tyx",
    "gt_tracks_yx",
    "gt_occluded",
    "oracle_candidate_index",
    "oracle_coords_xy_px",
    "oracle_error_px",
    "native_error_px",
    "local_peak",
    "local_margin",
    "local_entropy",
    "search_cell_count",
)


def verify_raw_v1_frozen_contract(
    base_tensors: dict[str, torch.Tensor],
    raw_tensors: dict[str, torch.Tensor],
) -> dict[str, str]:
    """Verify that raw-v1 changes representation and nothing else."""
    from .cotracker3_stage0_adapter import tensor_sha256

    for key in FROZEN_SHARED_TENSOR_KEYS:
        if key not in base_tensors or key not in raw_tensors:
            raise KeyError(f"missing frozen tensor key: {key}")
        if not torch.equal(base_tensors[key], raw_tensors[key]):
            raise ValueError(f"raw-v1 drifted frozen tensor: {key}")
    if base_tensors["candidate_features"].shape[-1] != 64:
        raise ValueError("base candidate representation is not the frozen 64-D contract")
    if base_tensors["state_features"].shape[-1] != 32:
        raise ValueError("base state representation is not the frozen 32-D contract")
    if raw_tensors["candidate_features"].shape[-1] != RAW_V1_CANDIDATE_FEATURE_DIM:
        raise ValueError("raw-v1 candidate feature dimension mismatch")
    if raw_tensors["state_features"].shape[-1] != RAW_V1_STATE_FEATURE_DIM:
        raise ValueError("raw-v1 state feature dimension mismatch")
    if not torch.equal(
        raw_tensors["candidate_features"][..., :64],
        base_tensors["candidate_features"].float(),
    ):
        raise ValueError("raw-v1 did not preserve the structured candidate prefix")
    if not torch.equal(
        raw_tensors["state_features"][..., :32],
        base_tensors["state_features"].float(),
    ):
        raise ValueError("raw-v1 did not preserve the structured state prefix")
    return {key: tensor_sha256(raw_tensors[key]) for key in FROZEN_SHARED_TENSOR_KEYS}
