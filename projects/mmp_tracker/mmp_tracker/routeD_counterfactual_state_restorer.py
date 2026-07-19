"""Structured counterfactual state re-extraction for Route-D Gate 2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .routeD_counterfactual_state_restoration import CoTrackerOnlineStateSnapshot


CSRR_SCHEMA_VERSION = "routeD_counterfactual_state_restorer_gate2_v0"


def input_xy_to_model_xy(
    coordinates_xy: torch.Tensor,
    *,
    input_height: int,
    input_width: int,
    model_height: int,
    model_width: int,
) -> torch.Tensor:
    scale = coordinates_xy.new_tensor(
        [
            float(model_width - 1) / float(max(input_width - 1, 1)),
            float(model_height - 1) / float(max(input_height - 1, 1)),
        ]
    )
    return coordinates_xy * scale


def model_xy_to_input_xy(
    coordinates_xy: torch.Tensor,
    *,
    input_height: int,
    input_width: int,
    model_height: int,
    model_width: int,
) -> torch.Tensor:
    scale = coordinates_xy.new_tensor(
        [
            float(input_width - 1) / float(max(model_width - 1, 1)),
            float(input_height - 1) / float(max(model_height - 1, 1)),
        ]
    )
    return coordinates_xy * scale


def extract_cotracker_observed_feature_pyramid(
    model: Any,
    video: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Reproduce CoTracker3's normalized feature pyramid for an observed clip.

    Encoding the complete observed clip in one batch matches the official online
    forward path and avoids small CUDA convolution differences caused by changing
    the fnet batch size.
    """
    if video.ndim != 5 or video.shape[2] != 3:
        raise ValueError("video must have shape [B,T,3,H,W]")
    batch, frames, channels, height, width = video.shape
    model_height, model_width = map(int, model.model_resolution)
    resized = F.interpolate(
        video.float().reshape(batch * frames, channels, height, width),
        size=(model_height, model_width),
        mode="bilinear",
        align_corners=True,
    )
    normalized = 2.0 * (resized / 255.0) - 1.0
    fmaps = model.fnet(normalized)
    fmaps = F.normalize(fmaps, p=2, dim=1, eps=1.0e-12)
    fmaps = fmaps.reshape(
        batch, frames, fmaps.shape[1], fmaps.shape[2], fmaps.shape[3]
    )
    levels: list[torch.Tensor] = [fmaps]
    current = fmaps
    for _ in range(int(model.corr_levels) - 1):
        pooled = F.avg_pool2d(
            current.reshape(
                batch * frames, current.shape[2], current.shape[3], current.shape[4]
            ),
            2,
            stride=2,
        )
        current = pooled.reshape(
            batch, frames, pooled.shape[1], pooled.shape[2], pooled.shape[3]
        )
        levels.append(current)
    return tuple(levels)


def extract_cotracker_frame_feature_pyramid(
    model: Any,
    frame: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Convenience wrapper for a single frame; clip extraction is parity-preferred."""
    if frame.ndim != 4:
        raise ValueError("frame must have shape [B,3,H,W]")
    return extract_cotracker_observed_feature_pyramid(model, frame[:, None])


def reextract_cotracker_memory(
    model: Any,
    feature_pyramid: Sequence[torch.Tensor],
    coordinates_input_xy: torch.Tensor,
    *,
    input_height: int,
    input_width: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Sample coherent track feature/support memory at input-raster coordinates."""
    if coordinates_input_xy.ndim != 3 or coordinates_input_xy.shape[-1] != 2:
        raise ValueError("coordinates must have shape [B,N,2]")
    if len(feature_pyramid) != int(model.corr_levels):
        raise ValueError("feature-pyramid level mismatch")
    model_height, model_width = map(int, model.model_resolution)
    model_xy = input_xy_to_model_xy(
        coordinates_input_xy,
        input_height=input_height,
        input_width=input_width,
        model_height=model_height,
        model_width=model_width,
    )
    queried_frames = torch.zeros(
        coordinates_input_xy.shape[:2],
        dtype=torch.long,
        device=coordinates_input_xy.device,
    )
    track_features: list[torch.Tensor] = []
    track_supports: list[torch.Tensor] = []
    for level, fmap in enumerate(feature_pyramid):
        if fmap.ndim != 5 or fmap.shape[1] != 1:
            raise ValueError("each feature level must have shape [B,1,C,H,W]")
        track_feat, track_support = model.get_track_feat(
            fmap,
            queried_frames,
            model_xy / float(model.stride * (2**level)),
            support_radius=int(model.corr_radius),
        )
        track_features.append(track_feat)
        track_supports.append(track_support)
    return tuple(track_features), tuple(track_supports)


def build_csrr_trajectory_features(
    snapshot: CoTrackerOnlineStateSnapshot,
    *,
    point_indices: torch.Tensor,
    frame_start: int = 8,
    frame_end_inclusive: int = 15,
    input_height: int = 256,
    input_width: int = 256,
    model_height: int = 384,
    model_width: int = 512,
) -> torch.Tensor:
    """Build the frozen nine-dimensional causal trajectory representation.

    Per frame the channels are normalized ``xy`` (2), normalized framewise
    ``dxy`` (2), visibility/confidence probabilities (2), normalized original
    query ``xy`` (2), and normalized distance from the original query (1).
    """
    indices = point_indices.long().to(snapshot.online_coords_predicted.device)
    frame_slice = slice(int(frame_start), int(frame_end_inclusive) + 1)
    model_xy = snapshot.online_coords_predicted[0, frame_slice, indices].permute(1, 0, 2)
    input_xy = model_xy_to_input_xy(
        model_xy,
        input_height=input_height,
        input_width=input_width,
        model_height=model_height,
        model_width=model_width,
    )
    scale = input_xy.new_tensor(
        [float(max(input_width - 1, 1)), float(max(input_height - 1, 1))]
    )
    normalized_xy = input_xy / scale
    delta = torch.zeros_like(normalized_xy)
    delta[:, 1:] = normalized_xy[:, 1:] - normalized_xy[:, :-1]
    visibility = torch.sigmoid(
        snapshot.online_vis_predicted[0, frame_slice, indices].permute(1, 0)
    )[..., None]
    confidence = torch.sigmoid(
        snapshot.online_conf_predicted[0, frame_slice, indices].permute(1, 0)
    )[..., None]
    query_model_xy = snapshot.predictor_queries[0, indices, 1:3]
    query_input_xy = model_xy_to_input_xy(
        query_model_xy,
        input_height=input_height,
        input_width=input_width,
        model_height=model_height,
        model_width=model_width,
    )
    normalized_query = query_input_xy / scale
    query_sequence = normalized_query[:, None].expand(-1, normalized_xy.shape[1], -1)
    query_distance = torch.linalg.vector_norm(normalized_xy - query_sequence, dim=-1, keepdim=True)
    features = torch.cat(
        [normalized_xy, delta, visibility, confidence, query_sequence, query_distance],
        dim=-1,
    )
    if features.shape[-1] != 9:
        raise RuntimeError("unexpected CSRR trajectory dimension")
    return features


@dataclass(frozen=True)
class CSRRConfig:
    latent_dim: int = 128
    projection_dim: int = 32
    common_height: int = 64
    common_width: int = 64
    trajectory_frames: int = 8
    trajectory_dim: int = 9
    temperature: float = 0.10


class CounterfactualStructuredReextractionRestorer(nn.Module):
    """Small global matcher that predicts commit coordinate and no-op action."""

    def __init__(self, config: CSRRConfig | None = None) -> None:
        super().__init__()
        self.config = config or CSRRConfig()
        c = self.config
        self.support_attention = nn.Linear(c.latent_dim, 1)
        self.channel_norm = nn.LayerNorm(c.latent_dim)
        self.channel_projection = nn.Linear(c.latent_dim, c.projection_dim, bias=False)
        self.level_fusion = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        trajectory_input = c.trajectory_frames * c.trajectory_dim
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(trajectory_input, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        # trajectory embedding + max prob + entropy + top margin + correction xy
        self.action_head = nn.Sequential(
            nn.Linear(32 + 3 + 2, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _support_query(self, support: torch.Tensor) -> torch.Tensor:
        if support.ndim != 3 or support.shape[-1] != self.config.latent_dim:
            raise ValueError("support must have shape [B,49,C]")
        weights = torch.softmax(self.support_attention(support).squeeze(-1), dim=-1)
        pooled = torch.sum(weights[..., None] * support, dim=1)
        projected = self.channel_projection(self.channel_norm(pooled))
        return F.normalize(projected, p=2, dim=-1, eps=1.0e-8)

    def _correlation_map(
        self,
        frame_feature: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        if frame_feature.ndim != 4 or frame_feature.shape[1] != self.config.latent_dim:
            raise ValueError("frame feature must have shape [B,C,H,W]")
        query = self._support_query(support)
        frame = frame_feature.permute(0, 2, 3, 1)
        frame = self.channel_projection(self.channel_norm(frame))
        frame = F.normalize(frame, p=2, dim=-1, eps=1.0e-8)
        if frame.shape[0] == 1 and query.shape[0] > 1:
            score = torch.einsum("hwc,bc->bhw", frame[0], query)[:, None]
        elif frame.shape[0] == query.shape[0]:
            score = torch.einsum("bhwc,bc->bhw", frame, query)[:, None]
        else:
            raise ValueError("frame/support batch mismatch")
        return F.interpolate(
            score,
            size=(self.config.common_height, self.config.common_width),
            mode="bilinear",
            align_corners=True,
        )

    def forward(
        self,
        *,
        trajectory_features: torch.Tensor,
        native_support_pyramid: Sequence[torch.Tensor],
        frame_feature_pyramid: Sequence[torch.Tensor],
        native_commit_coordinates_xy: torch.Tensor,
        input_height: int = 256,
        input_width: int = 256,
    ) -> dict[str, torch.Tensor]:
        c = self.config
        if trajectory_features.shape[1:] != (c.trajectory_frames, c.trajectory_dim):
            raise ValueError("trajectory feature shape mismatch")
        if len(native_support_pyramid) != 4 or len(frame_feature_pyramid) != 4:
            raise ValueError("CSRR requires four pyramid levels")
        maps = []
        for support, frame in zip(native_support_pyramid, frame_feature_pyramid):
            if frame.ndim == 5:
                if frame.shape[1] != 1:
                    raise ValueError("frame pyramid must contain one commit frame")
                frame = frame[:, 0]
            maps.append(self._correlation_map(frame, support))
        fused_logits = self.level_fusion(torch.cat(maps, dim=1))[:, 0]
        flat_logits = fused_logits.flatten(1) / float(c.temperature)
        probabilities = torch.softmax(flat_logits, dim=-1)
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, float(input_height - 1), c.common_height, device=fused_logits.device, dtype=fused_logits.dtype),
            torch.linspace(0.0, float(input_width - 1), c.common_width, device=fused_logits.device, dtype=fused_logits.dtype),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        predicted_xy = probabilities @ grid
        max_probability = probabilities.max(dim=-1).values
        entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(dim=-1)
        entropy = entropy / torch.log(probabilities.new_tensor(float(probabilities.shape[-1])))
        top2 = torch.topk(probabilities, k=2, dim=-1).values
        top_margin = top2[:, 0] - top2[:, 1]
        trajectory_embedding = self.trajectory_encoder(trajectory_features.flatten(1))
        correction = (predicted_xy - native_commit_coordinates_xy) / native_commit_coordinates_xy.new_tensor(
            [float(max(input_width - 1, 1)), float(max(input_height - 1, 1))]
        )
        action_input = torch.cat(
            [trajectory_embedding, max_probability[:, None], entropy[:, None], top_margin[:, None], correction],
            dim=-1,
        )
        action = self.action_head(action_input)
        return {
            "fused_logits": fused_logits,
            "spatial_probability": probabilities.view(-1, c.common_height, c.common_width),
            "predicted_coordinates_xy": predicted_xy,
            "apply_logit": action[:, 0],
            "visibility_residual": action[:, 1],
            "confidence_residual": action[:, 2],
            "map_max_probability": max_probability,
            "map_entropy": entropy,
            "map_top_margin": top_margin,
        }


def _deterministic_bilinear_sample_shared_map(
    feature_map: torch.Tensor, coordinates_xy: torch.Tensor
) -> torch.Tensor:
    """Sample a shared ``[1,C,H,W]`` map without CUDA grid-sampler backward.

    The feature map is frozen. Gathered corner values therefore act as constants
    and gradients flow only through the bilinear weights into coordinates.
    Border padding and align-corners pixel coordinates match CoTracker.
    """
    if feature_map.ndim != 4 or feature_map.shape[0] != 1:
        raise ValueError("deterministic sampler requires a shared [1,C,H,W] map")
    if coordinates_xy.ndim != 3 or coordinates_xy.shape[-1] != 2:
        raise ValueError("coordinates must have shape [B,R,2]")
    _, channels, height, width = feature_map.shape
    x = coordinates_xy[..., 0].clamp(0.0, float(width - 1))
    y = coordinates_xy[..., 1].clamp(0.0, float(height - 1))
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)
    wx = x - x0.to(x.dtype)
    wy = y - y0.to(y.dtype)
    flat = feature_map[0].detach().reshape(channels, height * width)

    def gather(ix: torch.Tensor, iy: torch.Tensor) -> torch.Tensor:
        linear = (iy * width + ix).reshape(-1)
        values = flat[:, linear].transpose(0, 1)
        return values.reshape(*ix.shape, channels)

    top_left = gather(x0, y0)
    top_right = gather(x1, y0)
    bottom_left = gather(x0, y1)
    bottom_right = gather(x1, y1)
    return (
        top_left * ((1.0 - wx) * (1.0 - wy))[..., None]
        + top_right * (wx * (1.0 - wy))[..., None]
        + bottom_left * ((1.0 - wx) * wy)[..., None]
        + bottom_right * (wx * wy)[..., None]
    )


def deterministic_reextract_cotracker_memory(
    feature_pyramid: Sequence[torch.Tensor],
    coordinates_input_xy: torch.Tensor,
    *,
    input_height: int = 256,
    input_width: int = 256,
    model_height: int = 384,
    model_width: int = 512,
    stride: int = 4,
    support_radius: int = 3,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Deterministic differentiable equivalent of single-frame get_track_feat."""
    if coordinates_input_xy.ndim != 2 or coordinates_input_xy.shape[-1] != 2:
        raise ValueError("coordinates must have shape [B,2]")
    model_xy = input_xy_to_model_xy(
        coordinates_input_xy,
        input_height=input_height,
        input_width=input_width,
        model_height=model_height,
        model_width=model_width,
    )
    offsets = torch.linspace(
        -float(support_radius),
        float(support_radius),
        2 * int(support_radius) + 1,
        device=coordinates_input_xy.device,
        dtype=coordinates_input_xy.dtype,
    )
    xgrid, ygrid = torch.meshgrid(offsets, offsets, indexing="ij")
    # This ordering matches CoTracker get_support_points: stack([t,xgrid,ygrid]).
    offset_xy = torch.stack([xgrid, ygrid], dim=-1).reshape(-1, 2)
    track_features: list[torch.Tensor] = []
    track_supports: list[torch.Tensor] = []
    center_index = offset_xy.shape[0] // 2
    for level, feature in enumerate(feature_pyramid):
        if feature.ndim == 5:
            if feature.shape[1] != 1:
                raise ValueError("training feature pyramid must contain one frame")
            feature = feature[:, 0]
        center = model_xy / float(int(stride) * (2**level))
        support_coordinates = center[:, None] + offset_xy[None]
        sampled = _deterministic_bilinear_sample_shared_map(
            feature, support_coordinates
        )
        track_features.append(sampled[:, center_index][:, None, None])
        track_supports.append(sampled[:, :, None])
    return tuple(track_features), tuple(track_supports)


def apply_reextracted_state_action(
    snapshot: CoTrackerOnlineStateSnapshot,
    *,
    point_indices: torch.Tensor,
    predicted_coordinates_input_xy: torch.Tensor,
    apply_mask: torch.Tensor,
    reextracted_track_features: Sequence[torch.Tensor],
    reextracted_track_supports: Sequence[torch.Tensor],
    input_height: int,
    input_width: int,
    model_height: int,
    model_width: int,
    commit_frame: int = 15,
    visibility_residual: torch.Tensor | None = None,
    confidence_residual: torch.Tensor | None = None,
    write_probability: bool = True,
    write_memory: bool = True,
) -> CoTrackerOnlineStateSnapshot:
    """Apply structured re-extraction to selected point slots; false rows are exact no-op."""
    indices = point_indices.long().detach().cpu()
    mask = apply_mask.bool().detach().cpu()
    if indices.ndim != 1 or mask.shape != indices.shape:
        raise ValueError("point indices/apply mask shape mismatch")
    if predicted_coordinates_input_xy.shape != (indices.numel(), 2):
        raise ValueError("predicted coordinate shape mismatch")
    coords = snapshot.online_coords_predicted.detach().clone()
    vis = snapshot.online_vis_predicted.detach().clone()
    conf = snapshot.online_conf_predicted.detach().clone()
    feat = tuple(None if value is None else value.detach().clone() for value in snapshot.online_track_feat)
    support = tuple(None if value is None else value.detach().clone() for value in snapshot.online_track_support)
    if not mask.any():
        return CoTrackerOnlineStateSnapshot(
            predictor_n=snapshot.predictor_n,
            predictor_queries=snapshot.predictor_queries.detach().clone(),
            online_ind=snapshot.online_ind,
            online_track_feat=feat,
            online_track_support=support,
            online_coords_predicted=coords,
            online_vis_predicted=vis,
            online_conf_predicted=conf,
        )
    applied_rows = torch.where(mask)[0]
    native_indices = indices[applied_rows]
    model_xy = input_xy_to_model_xy(
        predicted_coordinates_input_xy[applied_rows].to(coords.device),
        input_height=input_height,
        input_width=input_width,
        model_height=model_height,
        model_width=model_width,
    )
    coords[0, int(commit_frame), native_indices.to(coords.device)] = model_xy
    if write_probability and visibility_residual is not None:
        selected_vis = vis[0, int(commit_frame), native_indices.to(vis.device)]
        probability = torch.sigmoid(selected_vis)
        updated_probability = (
            probability + visibility_residual[applied_rows].to(vis.device)
        ).clamp(1.0e-5, 1.0 - 1.0e-5)
        vis[0, int(commit_frame), native_indices.to(vis.device)] = torch.logit(
            updated_probability
        )
    if write_probability and confidence_residual is not None:
        selected_conf = conf[0, int(commit_frame), native_indices.to(conf.device)]
        probability = torch.sigmoid(selected_conf)
        updated_probability = (
            probability + confidence_residual[applied_rows].to(conf.device)
        ).clamp(1.0e-5, 1.0 - 1.0e-5)
        conf[0, int(commit_frame), native_indices.to(conf.device)] = torch.logit(
            updated_probability
        )
    feat_rows = list(feat)
    support_rows = list(support)
    if write_memory:
        for level, (new_feat, new_support) in enumerate(
            zip(reextracted_track_features, reextracted_track_supports)
        ):
            if feat_rows[level] is None or support_rows[level] is None:
                raise ValueError("live state memory unavailable")
            feat_rows[level][:, :, native_indices.to(feat_rows[level].device)] = (
                new_feat[:, :, applied_rows].to(feat_rows[level].device)
            )
            support_rows[level][:, :, native_indices.to(support_rows[level].device)] = (
                new_support[:, :, applied_rows].to(support_rows[level].device)
            )
    return CoTrackerOnlineStateSnapshot(
        predictor_n=snapshot.predictor_n,
        predictor_queries=snapshot.predictor_queries.detach().clone(),
        online_ind=snapshot.online_ind,
        online_track_feat=tuple(feat_rows),
        online_track_support=tuple(support_rows),
        online_coords_predicted=coords,
        online_vis_predicted=vis,
        online_conf_predicted=conf,
    )
