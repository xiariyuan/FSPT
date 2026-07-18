"""End-to-end query-token lifecycle for recurrent Tracking-Any-Point models.

ReQueryTAP keeps an immutable exact-point identity memory separate from the
active transient TAPNext state.  A global locator predicts where the physical
point has reappeared.  Respawn retires the old transient state and invokes the
same trainable TAPNext backbone with a new query; no hidden state is copied or
injected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


REQUERYTAP_SCHEMA_VERSION = "requerytap_token_lifecycle_v0"


@dataclass
class ReQueryTAPState:
    """One strictly query-independent lifecycle state."""

    track_state: Any
    identity_memory: torch.Tensor  # B,1,D
    original_query_points: torch.Tensor  # B,1,3 [t,y,x]
    generation: int
    frames_since_respawn: int


@dataclass
class ReQueryTAPStepOutput:
    selected_tracks: torch.Tensor
    selected_track_logits: torch.Tensor
    selected_visibility_logits: torch.Tensor
    state: ReQueryTAPState
    native_tracks: torch.Tensor
    native_track_logits: torch.Tensor
    native_visibility_logits: torch.Tensor
    fresh_tracks: torch.Tensor | None
    fresh_track_logits: torch.Tensor | None
    fresh_visibility_logits: torch.Tensor | None
    rebind_coordinate_yx: torch.Tensor | None
    rebind_patch_logits: torch.Tensor | None
    respawn_logit: torch.Tensor | None
    respawn_applied: bool


class ExactPointGlobalLocator(nn.Module):
    """Global identity-to-patch locator with differentiable soft coordinates."""

    def __init__(
        self,
        backbone_width: int,
        *,
        identity_dim: int = 128,
        patch_size: int = 8,
        image_size: tuple[int, int] = (256, 256),
        temperature: float = 10.0,
    ) -> None:
        super().__init__()
        if identity_dim <= 0 or patch_size <= 0 or temperature <= 0:
            raise ValueError("identity_dim, patch_size, and temperature must be positive")
        if image_size[0] % patch_size or image_size[1] % patch_size:
            raise ValueError("image size must be divisible by patch size")
        self.identity_dim = int(identity_dim)
        self.patch_size = int(patch_size)
        self.image_size = tuple(int(value) for value in image_size)
        self.temperature = float(temperature)
        self.identity_projection = nn.Linear(backbone_width, identity_dim)
        self.image_projection = nn.Linear(backbone_width, identity_dim)
        self.offset_head = nn.Sequential(
            nn.Linear(identity_dim * 2, identity_dim),
            nn.GELU(),
            nn.Linear(identity_dim, 2),
        )
        self.respawn_head = nn.Sequential(
            nn.Linear(identity_dim * 2 + 3, identity_dim),
            nn.GELU(),
            nn.Linear(identity_dim, 1),
        )
        height = self.image_size[0] // self.patch_size
        width = self.image_size[1] // self.patch_size
        y = (torch.arange(height, dtype=torch.float32) + 0.5) * self.patch_size
        x = (torch.arange(width, dtype=torch.float32) + 0.5) * self.patch_size
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self.register_buffer("patch_centers_yx", torch.stack([yy, xx], dim=-1).reshape(-1, 2))

    def encode_identity(self, sampled_backbone_feature: torch.Tensor) -> torch.Tensor:
        if sampled_backbone_feature.ndim != 3 or sampled_backbone_feature.shape[1] != 1:
            raise ValueError("identity feature must have shape B,1,C")
        return F.normalize(self.identity_projection(sampled_backbone_feature), dim=-1)

    def forward(
        self,
        identity_memory: torch.Tensor,
        image_tokens: torch.Tensor,
        *,
        native_coordinate_yx: torch.Tensor,
        native_visibility_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if identity_memory.ndim != 3 or identity_memory.shape[1] != 1:
            raise ValueError("identity memory must have shape B,1,D")
        if image_tokens.ndim != 3:
            raise ValueError("image tokens must have shape B,N,C")
        if native_coordinate_yx.shape != (identity_memory.shape[0], 1, 2):
            raise ValueError("native coordinate must have shape B,1,2")
        if native_visibility_logit.shape != (identity_memory.shape[0], 1, 1):
            raise ValueError("native visibility logit must have shape B,1,1")
        image_embedding = F.normalize(self.image_projection(image_tokens), dim=-1)
        patch_logits = torch.einsum("bqd,bnd->bqn", identity_memory, image_embedding)
        patch_logits = patch_logits * self.temperature
        probability = F.softmax(patch_logits.float(), dim=-1).to(patch_logits.dtype)
        centers = self.patch_centers_yx.to(device=probability.device, dtype=probability.dtype)
        coarse = torch.einsum("bqn,nd->bqd", probability, centers)
        attended = torch.einsum("bqn,bnd->bqd", probability, image_embedding)
        offset_input = torch.cat([identity_memory, attended], dim=-1)
        offset = torch.tanh(self.offset_head(offset_input)) * (self.patch_size / 2.0)
        coordinate = coarse + offset
        coordinate_y = coordinate[..., 0].clamp(0.5, self.image_size[0] - 0.5)
        coordinate_x = coordinate[..., 1].clamp(0.5, self.image_size[1] - 0.5)
        coordinate = torch.stack([coordinate_y, coordinate_x], dim=-1)
        distance = torch.linalg.norm(coordinate - native_coordinate_yx, dim=-1, keepdim=True)
        distance = distance / float(max(self.image_size))
        visibility_probability = torch.sigmoid(native_visibility_logit)
        gate_input = torch.cat(
            [identity_memory, attended, native_visibility_logit, visibility_probability, distance],
            dim=-1,
        )
        respawn_logit = self.respawn_head(gate_input)
        return coordinate, patch_logits, respawn_logit


class ReQueryTAP(nn.Module):
    """Lifecycle model around a trainable single-query TAPNext backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        identity_dim: int = 128,
        respawn_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if not hasattr(backbone, "lin_proj") or not hasattr(backbone, "image_pos_emb"):
            raise ValueError("backbone must expose TAPNext lin_proj and image_pos_emb")
        if respawn_threshold <= 0.0 or respawn_threshold >= 1.0:
            raise ValueError("respawn threshold must lie strictly within (0,1)")
        self.backbone = backbone
        self.respawn_threshold = float(respawn_threshold)
        width = int(backbone.lin_proj.out_channels)
        patch_size = int(backbone.lin_proj.kernel_size[0])
        image_size = tuple(int(value) for value in backbone.image_size)
        self.locator = ExactPointGlobalLocator(
            width,
            identity_dim=identity_dim,
            patch_size=patch_size,
            image_size=image_size,
        )

    @staticmethod
    def _validate_video(video: torch.Tensor) -> None:
        if video.ndim != 5 or video.shape[1] != 1 or video.shape[-1] != 3:
            raise ValueError("online ReQueryTAP expects B,1,H,W,3 video")

    @staticmethod
    def _validate_single_query(query_points: torch.Tensor) -> None:
        if query_points.ndim != 3 or query_points.shape[1:] != (1, 3):
            raise ValueError("ReQueryTAP enforces one independent query per model call")

    def patch_tokens(self, video: torch.Tensor) -> torch.Tensor:
        self._validate_video(video)
        batch, time, height, width, channels = video.shape
        image = video.permute(0, 1, 4, 2, 3).reshape(batch * time, channels, height, width)
        tokens = self.backbone.lin_proj(image)
        tokens = tokens.flatten(2).transpose(1, 2).reshape(batch, time, -1, tokens.shape[1])
        position = self.backbone.image_pos_emb.to(device=tokens.device, dtype=tokens.dtype)
        return tokens[:, 0] + position

    def sample_query_feature(
        self, image_tokens: torch.Tensor, query_coordinate_yx: torch.Tensor
    ) -> torch.Tensor:
        batch, token_count, channels = image_tokens.shape
        grid_height = self.backbone.image_size[0] // self.backbone.patch_size[0]
        grid_width = self.backbone.image_size[1] // self.backbone.patch_size[1]
        if token_count != grid_height * grid_width:
            raise ValueError("image-token count differs from the TAPNext patch grid")
        feature_map = image_tokens.transpose(1, 2).reshape(batch, channels, grid_height, grid_width)
        y = query_coordinate_yx[..., 0] / float(self.backbone.image_size[0]) * 2.0 - 1.0
        x = query_coordinate_yx[..., 1] / float(self.backbone.image_size[1]) * 2.0 - 1.0
        grid = torch.stack([x, y], dim=-1).reshape(batch, 1, 1, 2)
        sampled = F.grid_sample(feature_map, grid, mode="bilinear", align_corners=False)
        return sampled[:, :, 0, 0].unsqueeze(1)

    def initialize(
        self, video: torch.Tensor, query_points: torch.Tensor
    ) -> ReQueryTAPStepOutput:
        self._validate_video(video)
        self._validate_single_query(query_points)
        tracks, track_logits, visibility_logits, track_state = self.backbone(
            video=video, query_points=query_points
        )
        image_tokens = self.patch_tokens(video)
        sampled = self.sample_query_feature(image_tokens, query_points[..., 1:3])
        identity_memory = self.locator.encode_identity(sampled)
        state = ReQueryTAPState(
            track_state=track_state,
            identity_memory=identity_memory,
            original_query_points=query_points.detach().clone(),
            generation=0,
            frames_since_respawn=0,
        )
        return ReQueryTAPStepOutput(
            selected_tracks=tracks,
            selected_track_logits=track_logits,
            selected_visibility_logits=visibility_logits,
            state=state,
            native_tracks=tracks,
            native_track_logits=track_logits,
            native_visibility_logits=visibility_logits,
            fresh_tracks=None,
            fresh_track_logits=None,
            fresh_visibility_logits=None,
            rebind_coordinate_yx=None,
            rebind_patch_logits=None,
            respawn_logit=None,
            respawn_applied=False,
        )

    def step(
        self,
        video: torch.Tensor,
        state: ReQueryTAPState,
        *,
        allow_respawn: bool = True,
        force_respawn: bool = False,
        forced_query_coordinate_yx: torch.Tensor | None = None,
    ) -> ReQueryTAPStepOutput:
        self._validate_video(video)
        native_tracks, native_logits, native_visibility, native_state = self.backbone(
            video=video, state=state.track_state
        )
        if not allow_respawn:
            next_state = ReQueryTAPState(
                track_state=native_state,
                identity_memory=state.identity_memory,
                original_query_points=state.original_query_points,
                generation=state.generation,
                frames_since_respawn=state.frames_since_respawn + 1,
            )
            return ReQueryTAPStepOutput(
                selected_tracks=native_tracks,
                selected_track_logits=native_logits,
                selected_visibility_logits=native_visibility,
                state=next_state,
                native_tracks=native_tracks,
                native_track_logits=native_logits,
                native_visibility_logits=native_visibility,
                fresh_tracks=None,
                fresh_track_logits=None,
                fresh_visibility_logits=None,
                rebind_coordinate_yx=None,
                rebind_patch_logits=None,
                respawn_logit=None,
                respawn_applied=False,
            )

        image_tokens = self.patch_tokens(video)
        native_coordinate = native_tracks[:, -1]
        native_visibility_last = native_visibility[:, -1]
        rebind_coordinate, patch_logits, respawn_logit = self.locator(
            state.identity_memory,
            image_tokens,
            native_coordinate_yx=native_coordinate,
            native_visibility_logit=native_visibility_last,
        )
        spawn_coordinate = (
            forced_query_coordinate_yx
            if forced_query_coordinate_yx is not None
            else rebind_coordinate
        )
        if spawn_coordinate.shape != native_coordinate.shape:
            raise ValueError("forced query coordinate must have shape B,1,2")
        apply_respawn = force_respawn or bool(
            (torch.sigmoid(respawn_logit).detach() >= self.respawn_threshold).all().item()
        )
        fresh_tracks = fresh_logits = fresh_visibility = fresh_state = None
        if apply_respawn or self.training:
            fresh_query = torch.cat(
                [
                    torch.zeros(
                        spawn_coordinate.shape[0],
                        1,
                        1,
                        device=spawn_coordinate.device,
                        dtype=spawn_coordinate.dtype,
                    ),
                    spawn_coordinate,
                ],
                dim=-1,
            )
            fresh_tracks, fresh_logits, fresh_visibility, fresh_state = self.backbone(
                video=video, query_points=fresh_query
            )
        if apply_respawn:
            selected_tracks = fresh_tracks
            selected_logits = fresh_logits
            selected_visibility = fresh_visibility
            selected_state = fresh_state
            generation = state.generation + 1
            age = 0
        else:
            selected_tracks = native_tracks
            selected_logits = native_logits
            selected_visibility = native_visibility
            selected_state = native_state
            generation = state.generation
            age = state.frames_since_respawn + 1
        next_state = ReQueryTAPState(
            track_state=selected_state,
            identity_memory=state.identity_memory,
            original_query_points=state.original_query_points,
            generation=generation,
            frames_since_respawn=age,
        )
        return ReQueryTAPStepOutput(
            selected_tracks=selected_tracks,
            selected_track_logits=selected_logits,
            selected_visibility_logits=selected_visibility,
            state=next_state,
            native_tracks=native_tracks,
            native_track_logits=native_logits,
            native_visibility_logits=native_visibility,
            fresh_tracks=fresh_tracks,
            fresh_track_logits=fresh_logits,
            fresh_visibility_logits=fresh_visibility,
            rebind_coordinate_yx=rebind_coordinate,
            rebind_patch_logits=patch_logits,
            respawn_logit=respawn_logit,
            respawn_applied=apply_respawn,
        )
