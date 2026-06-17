"""
CoTracker-based Refiner for FSPT

Route A: Use a strong point-tracking base (CoTracker3) and learn a lightweight
refinement module that injects frequency/semantic/occlusion priors.

Key ideas:
- Base tracker produces coarse tracks (pixel space, x/y) + visibility.
- We sample our own geometric/semantic features along the (coarse/refined) tracks.
- We run SemanticEnhancedLFD + TemporalTransformer + residual heads to refine tracks/visibility.
- Occlusion predictor optionally refines both visibility and positions.

The design keeps CoTracker as an optional dependency:
  pip install -e baselines/cotracker
  (and download checkpoints into baselines/cotracker/checkpoints)
"""

from __future__ import annotations

import logging
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import cv2 as _cv2
import numpy as _np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .freq_semantic_fusion import SemanticEnhancedLFD
from .occlusion_predictor import FrequencyAwareOcclusionPredictor
from .point_tracker import GeometricBackbone, TemporalTransformer, _has_valid_band_features
from .semantic_encoder import CLIP_AVAILABLE, SemanticEncoder

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_local_cotracker_path() -> None:
    """
    Make the vendored CoTracker checkout importable without requiring a manual
    `pip install -e baselines/cotracker` step.
    """
    cotracker_root = _project_root() / "baselines" / "cotracker"
    if cotracker_root.is_dir():
        root_str = str(cotracker_root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    try:
        return bool(value)
    except Exception:
        return bool(default)


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return getattr(cfg, key)
    except Exception:
        return default


def _resolve_checkpoint_path(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    ckpt = Path(str(path_value))
    if ckpt.is_absolute():
        return str(ckpt)
    return str((_project_root() / ckpt).resolve())


def _ensure_video_uint8_range(video: torch.Tensor) -> torch.Tensor:
    """
    CoTracker expects video in [0,255] float (it internally maps to [-1, 1] via /255).
    Our datasets typically output [0,1]. We detect and scale when necessary.
    """
    if not isinstance(video, torch.Tensor):
        raise TypeError(f"video must be a torch.Tensor, got {type(video)}")
    if video.numel() == 0:
        return video
    vmax = float(video.max())
    if vmax <= 1.5:
        return video * 255.0
    return video


def _normalized_yx_to_grid_xy(points_yx: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Convert normalized [y, x] coordinates to grid_sample [x, y] coordinates in [-1, 1].

    This keeps sampling consistent with datasets normalized by (H, W).
    """
    h = max(int(height), 1)
    w = max(int(width), 1)

    y_px = (points_yx[..., 0] * float(h)).clamp(0.0, float(h - 1))
    x_px = (points_yx[..., 1] * float(w)).clamp(0.0, float(w - 1))

    if h > 1:
        y_grid = (y_px / float(h - 1)) * 2.0 - 1.0
    else:
        y_grid = torch.zeros_like(y_px)
    if w > 1:
        x_grid = (x_px / float(w - 1)) * 2.0 - 1.0
    else:
        x_grid = torch.zeros_like(x_px)

    return torch.stack([x_grid, y_grid], dim=-1)


def _confidence_gate(conf: torch.Tensor, threshold: float, power: float = 1.0) -> torch.Tensor:
    """
    Compute a confidence-based gating factor in [0, 1] with *no dead zone*.

    Previous designs that used (conf - threshold) could produce exact zeros when
    confidence is below threshold, which can block gradients and stall learning.

    This gate uses a simple ramp:
      gate = clamp(conf / threshold, 0, 1)          (threshold > 0)
      gate = clamp(conf, 0, 1)                      (threshold <= 0)

    Args:
        conf: confidence tensor (any shape)
        threshold: gating threshold in [0, 1] (clamped)
        power: optional exponent to sharpen/soften the gate (>=0 recommended)
    """
    if not isinstance(conf, torch.Tensor):
        conf = torch.as_tensor(conf)

    threshold_f = float(threshold)
    threshold_f = 0.0 if threshold_f < 0 else (1.0 if threshold_f > 1 else threshold_f)

    if threshold_f <= 0.0:
        gate = conf.clamp(0.0, 1.0)
    else:
        gate = (conf / threshold_f).clamp(0.0, 1.0)

    power_f = float(power)
    if power_f != 1.0:
        gate = gate ** power_f
    return gate


def _compute_reappearance_mask(
    base_visibility: torch.Tensor,
    query_t: torch.Tensor,
    *,
    min_occlusion_len: int,
    frames_after: int,
    vis_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Detect "re-appearance after a long occlusion" events using the *base* tracker visibility.

    This is a lightweight heuristic used by Route A paper experiments focusing on
    re-localization after long occlusion. We intentionally base it on the base tracker
    (CoTracker3) visibility so it can be used at inference without ground-truth.

    Args:
        base_visibility: (B, N, T) float visibility in [0, 1] (base tracker).
        query_t: (B, N) query frame indices.
        min_occlusion_len: minimum consecutive occlusion frames before a reappearance triggers.
        frames_after: apply the mask for this many visible frames after reappearance.
        vis_threshold: visibility threshold for "visible" vs "occluded".

    Returns:
        mask_nt: (B, N, T) bool mask indicating frames where a re-localization step
            should be allowed.
    """
    if not isinstance(base_visibility, torch.Tensor) or base_visibility.dim() != 3:
        raise ValueError(
            f"base_visibility must be a torch.Tensor with shape (B,N,T), got {type(base_visibility)}"
        )
    if not isinstance(query_t, torch.Tensor) or query_t.dim() != 2:
        raise ValueError(f"query_t must be a torch.Tensor with shape (B,N), got {type(query_t)}")

    B, N, T = base_visibility.shape
    if query_t.shape != (B, N):
        raise ValueError(
            f"query_t shape mismatch: expected {(B, N)}, got {tuple(query_t.shape)}"
        )

    min_len = max(int(min_occlusion_len), 1)
    frames_after = max(int(frames_after), 1)
    th = float(vis_threshold)

    device = base_visibility.device
    t_idx = torch.arange(T, device=device).view(1, 1, T)  # (1,1,T)
    after_query = t_idx >= query_t.clamp(0, T - 1).unsqueeze(-1)  # (B,N,T)

    visible = (base_visibility >= th) & after_query
    occluded = (~visible) & after_query

    # Compute the occlusion run-length up to each frame.
    run = torch.zeros((B, N), device=device, dtype=torch.int32)
    run_len = torch.zeros((B, N, T), device=device, dtype=torch.int32)
    for t in range(T):
        oc_t = occluded[:, :, t]
        run = torch.where(oc_t, run + 1, torch.zeros_like(run))
        run_len[:, :, t] = run

    # For a "reappearance" at frame t, we care about the occlusion length ending at t-1.
    run_before = torch.cat(
        [torch.zeros((B, N, 1), device=device, dtype=run_len.dtype), run_len[:, :, :-1]],
        dim=-1,
    )  # (B,N,T)
    reappear = visible & (run_before >= min_len)

    if frames_after <= 1:
        return reappear

    # Extend the mask across the next few visible frames after reappearance.
    out = torch.zeros_like(reappear)
    countdown = torch.zeros((B, N), device=device, dtype=torch.int32)
    for t in range(T):
        start = reappear[:, :, t]
        countdown = torch.where(start, torch.full_like(countdown, frames_after), countdown)
        out[:, :, t] = (countdown > 0) & visible[:, :, t]
        countdown = torch.where(countdown > 0, countdown - 1, countdown)
        countdown = torch.where(visible[:, :, t], countdown, torch.zeros_like(countdown))
    return out


def _convert_query_points_to_cotracker(
    query_points: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Convert FSPT query_points (t, y, x) to CoTracker queries (t, x, y) in pixel coords.
    FSPT uses y/x in [0,1] normalized; CoTracker expects pixel coordinates.
    """
    if query_points.dim() != 3 or query_points.shape[-1] != 3:
        raise ValueError(f"query_points must be (B, N, 3), got {tuple(query_points.shape)}")
    t = query_points[..., 0]
    coords = query_points[..., 1:3]
    coords_max = float(coords.max()) if coords.numel() else 0.0

    if coords_max <= 1.5:
        # normalized [0,1] -> pixels
        y = (coords[..., 0] * float(max(height, 1))).clamp(0.0, float(max(height - 1, 0)))
        x = (coords[..., 1] * float(max(width, 1))).clamp(0.0, float(max(width - 1, 0)))
    else:
        # already pixels in (y,x)
        y = coords[..., 0].clamp(0.0, float(max(height - 1, 0)))
        x = coords[..., 1].clamp(0.0, float(max(width - 1, 0)))

    queries = torch.stack([t, x, y], dim=-1)
    return queries


def _convert_tracks_from_cotracker(
    tracks_xy: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Convert CoTracker tracks (x,y) pixels with shape (B, T, N, 2)
    to FSPT tracks (y,x) normalized with shape (B, N, T, 2).
    """
    if tracks_xy.dim() != 4 or tracks_xy.shape[-1] != 2:
        raise ValueError(f"tracks_xy must be (B, T, N, 2), got {tuple(tracks_xy.shape)}")
    x = tracks_xy[..., 0]
    y = tracks_xy[..., 1]
    denom_h = float(max(height, 1))
    denom_w = float(max(width, 1))
    tracks_y = (y / denom_h).clamp(0.0, 1.0)
    tracks_x = (x / denom_w).clamp(0.0, 1.0)
    tracks_yx = torch.stack([tracks_y, tracks_x], dim=-1)  # (B, T, N, 2)
    return tracks_yx.permute(0, 2, 1, 3).contiguous()  # (B, N, T, 2)


class ResidualTrackDecoder(nn.Module):
    """Predict residual track updates and visibility from point features."""

    def __init__(
        self,
        dim: int = 256,
        hidden_dim: int = 128,
        delta_scale: float = 0.2,
        use_gate: bool = False,
        gate_init_bias: float = -4.0,
        max_step: Optional[float] = None,
    ):
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.use_gate = bool(use_gate)
        self.gate_init_bias = float(gate_init_bias)
        self.max_step = float(max_step) if max_step is not None else None

        self.position_head = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.gate_head: Optional[nn.Module] = None
        if self.use_gate:
            self.gate_head = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        self.visibility_head = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def init_identity(self) -> None:
        """Initialize residual decoder to behave like an identity mapping."""
        last = self.position_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.constant_(last.weight, 0.0)
            if last.bias is not None:
                nn.init.constant_(last.bias, 0.0)
        if self.gate_head is not None:
            gate_last = self.gate_head[-1]
            if isinstance(gate_last, nn.Linear):
                nn.init.constant_(gate_last.weight, 0.0)
                if gate_last.bias is not None:
                    nn.init.constant_(gate_last.bias, float(self.gate_init_bias))

    def forward(
        self,
        features: torch.Tensor,
        base_tracks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, T, N, C)
            base_tracks: (B, N, T, 2) normalized [y, x]
        Returns:
            tracks: (B, N, T, 2)
            visibility: (B, N, T)
        """
        if features.dim() != 4:
            raise ValueError(f"features must be (B,T,N,C), got {tuple(features.shape)}")
        if base_tracks.dim() != 4:
            raise ValueError(f"base_tracks must be (B,N,T,2), got {tuple(base_tracks.shape)}")

        delta = self.position_head(features)  # (B, T, N, 2) in y/x order
        delta = delta.permute(0, 2, 1, 3).contiguous()  # (B, N, T, 2)
        step = delta * float(self.delta_scale)
        if self.gate_head is not None:
            gate_logits = self.gate_head(features).squeeze(-1)  # (B, T, N)
            gate = torch.sigmoid(gate_logits).permute(0, 2, 1).unsqueeze(-1)  # (B, N, T, 1)
            step = step * gate
        if self.max_step is not None and self.max_step > 0:
            step = torch.clamp(step, -self.max_step, self.max_step)

        tracks = base_tracks + step
        tracks = torch.clamp(tracks, 0.0, 1.0)

        visibility = self.visibility_head(features).squeeze(-1)  # (B, T, N)
        visibility = visibility.permute(0, 2, 1).contiguous()  # (B, N, T)
        return tracks, visibility


class LocalTrackCorrelation(nn.Module):
    """
    Lightweight local matching around the current track estimate.

    For each point, compute a local correlation map between:
      - a reference feature (from the point at its query frame, or a neighbor frame), and
      - a window of features around the point's current location in each frame.

    The correlation map is embedded and added to the refiner token features.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        window_size: int = 7,
        corr_dim: int = 64,
        hidden_dim: int = 128,
        scale: float = 1.0,
        normalize: bool = True,
        temperature: float = 1.0,
        projection: str = "learned",
        step_mode: str = "softargmax",
        confidence_mode: str = "maxprob",
        prob_power: float = 1.0,
        reference_mode: str = "query",
        corr_mode: str = "point",
        query_window_size: Optional[int] = None,
        all_pair_chunk: int = 8,
        learned_step_gate: bool = False,
        learned_step_gate_hidden_dim: int = 32,
        learned_step_gate_init_bias: float = 0.0,
        freq_enabled: bool = False,
        freq_cutoff: float = 0.3,
        freq_scale: float = 1.0,
        freq_out_mode: str = "add",
        freq_input: str = "prob",
        freq_step_mode: str = "raw",
        freq_msf_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if window_size % 2 != 1 or window_size < 1:
            raise ValueError(f"window_size must be odd and >=1, got {window_size}")
        if corr_dim < 1:
            raise ValueError(f"corr_dim must be >=1, got {corr_dim}")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.window_size = int(window_size)
        self.corr_dim = int(corr_dim)
        self.hidden_dim = int(hidden_dim)
        self.scale = float(scale)
        self.normalize = bool(normalize)
        self.temperature = float(temperature)

        self.step_mode = str(step_mode).lower().strip()
        if self.step_mode in ("soft", "softarg", "expected", "expectation", "softargmax"):
            self.step_mode = "softargmax"
        elif self.step_mode in ("hard", "argmax", "max"):
            self.step_mode = "argmax"
        if self.step_mode not in ("softargmax", "argmax"):
            raise ValueError(
                "step_mode must be one of {softargmax, argmax}, "
                f"got {step_mode!r}"
            )
        self.prob_power = float(prob_power)
        if not math.isfinite(self.prob_power) or self.prob_power <= 0:
            raise ValueError(f"prob_power must be finite and > 0, got {prob_power}")
        self.confidence_mode = str(confidence_mode).lower().strip()
        if self.confidence_mode not in ("maxprob", "max_prob", "shift", "shift_prob"):
            raise ValueError(
                "confidence_mode must be one of {maxprob, shift}, "
                f"got {confidence_mode!r}"
            )

        self.reference_mode = str(reference_mode).lower().strip()
        if self.reference_mode in ("", "default"):
            self.reference_mode = "query"
        if self.reference_mode not in ("query", "prev", "next", "adjacent"):
            raise ValueError(
                "reference_mode must be one of {query, prev, next, adjacent}, "
                f"got {reference_mode!r}"
            )

        self.corr_mode = str(corr_mode).lower().strip()
        if self.corr_mode in ("", "default"):
            self.corr_mode = "point"
        if self.corr_mode in ("point", "center", "single"):
            self.corr_mode = "point"
        elif self.corr_mode in ("all_pair", "allpair", "4d", "locotrack", "patch"):
            self.corr_mode = "all_pair"
        if self.corr_mode not in ("point", "all_pair"):
            raise ValueError(f"corr_mode must be one of {{point, all_pair}}, got {corr_mode!r}")

        if query_window_size is None:
            self.query_window_size = int(window_size)
        else:
            self.query_window_size = int(query_window_size)
        if self.query_window_size % 2 != 1 or self.query_window_size < 1:
            raise ValueError(f"query_window_size must be odd and >=1, got {query_window_size}")

        self.all_pair_chunk = int(all_pair_chunk)
        if self.all_pair_chunk < 1:
            raise ValueError(f"all_pair_chunk must be >= 1, got {all_pair_chunk}")

        self.projection = str(projection).lower().strip()
        if self.projection in ("", "default"):
            self.projection = "learned"
        if self.projection in ("learned", "proj", "projection", "conv", "conv1x1", "linear"):
            self.map_proj = nn.Conv2d(self.in_dim, self.corr_dim, kernel_size=1)
            self.query_proj = nn.Linear(self.in_dim, self.corr_dim)
        elif self.projection in ("identity", "none", "raw", "off"):
            if self.corr_dim != self.in_dim:
                raise ValueError(
                    "local_correlation.projection='identity' requires corr_dim == in_dim "
                    f"(got corr_dim={self.corr_dim}, in_dim={self.in_dim})."
                )
            self.map_proj = nn.Identity()
            self.query_proj = nn.Identity()
        else:
            raise ValueError(
                "projection must be one of {learned, identity}, "
                f"got {projection!r}"
            )

        K = self.window_size * self.window_size
        self.embed = nn.Sequential(
            nn.Linear(K, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

        radius = self.window_size // 2
        offsets = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                offsets.append((dy, dx))
        self.register_buffer(
            "_offsets_yx",
            torch.tensor(offsets, dtype=torch.float32),
            persistent=False,
        )

        query_radius = self.query_window_size // 2
        query_offsets = []
        for dy in range(-query_radius, query_radius + 1):
            for dx in range(-query_radius, query_radius + 1):
                query_offsets.append((dy, dx))
        self.register_buffer(
            "_query_offsets_yx",
            torch.tensor(query_offsets, dtype=torch.float32),
            persistent=False,
        )

        self.learned_step_gate = bool(learned_step_gate)
        self.step_gate_mlp: Optional[nn.Module] = None
        if self.learned_step_gate:
            gate_hidden = max(4, int(learned_step_gate_hidden_dim))
            self.step_gate_mlp = nn.Sequential(
                nn.Linear(4, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, 1),
            )
            nn.init.constant_(self.step_gate_mlp[-1].bias, float(learned_step_gate_init_bias))

        # --- Optional: frequency-aware matching (FAMNet-inspired) ---
        # We treat the ws×ws correlation map as a tiny "matching image" in displacement space.
        # A 2D FFT split lets us dampen domain-specific frequency reliance and use the mid-band
        # as a stable structural prior to guide low/high residual fusion.
        self.freq_enabled = bool(freq_enabled)
        self.freq_cutoff = float(freq_cutoff)
        if not (0.0 < self.freq_cutoff < 0.5):
            raise ValueError(f"freq_cutoff must be in (0,0.5), got {freq_cutoff}")
        self.freq_scale = float(freq_scale)
        if not math.isfinite(self.freq_scale) or self.freq_scale < 0:
            raise ValueError(f"freq_scale must be finite and >= 0, got {freq_scale}")

        self.freq_out_mode = str(freq_out_mode).lower().strip()
        if self.freq_out_mode in ("", "default"):
            self.freq_out_mode = "add"
        if self.freq_out_mode not in ("add", "replace"):
            raise ValueError(f"freq_out_mode must be one of {{add, replace}}, got {freq_out_mode!r}")

        self.freq_input = str(freq_input).lower().strip()
        if self.freq_input in ("", "default"):
            self.freq_input = "prob"
        if self.freq_input not in ("prob", "logits"):
            raise ValueError(f"freq_input must be one of {{prob, logits}}, got {freq_input!r}")

        self.freq_step_mode = str(freq_step_mode).lower().strip()
        if self.freq_step_mode in ("", "default"):
            self.freq_step_mode = "raw"
        if self.freq_step_mode not in ("raw", "mid", "msf"):
            raise ValueError(f"freq_step_mode must be one of {{raw, mid, msf}}, got {freq_step_mode!r}")

        self.freq_embed_low: Optional[nn.Module] = None
        self.freq_embed_mid: Optional[nn.Module] = None
        self.freq_embed_high: Optional[nn.Module] = None
        self.freq_msf_gate: Optional[nn.Module] = None
        if self.freq_enabled:
            K = self.window_size * self.window_size
            self.freq_embed_low = nn.Sequential(
                nn.Linear(K, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.out_dim),
            )
            self.freq_embed_mid = nn.Sequential(
                nn.Linear(K, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.out_dim),
            )
            self.freq_embed_high = nn.Sequential(
                nn.Linear(K, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.out_dim),
            )

            gate_hidden = max(4, int(freq_msf_hidden_dim))
            self.freq_msf_gate = nn.Sequential(
                nn.Linear(self.out_dim, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, 2),
            )

            ws = int(self.window_size)
            radius = ws // 2
            yy, xx = torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")
            dist = torch.sqrt((yy - radius).float() ** 2 + (xx - radius).float() ** 2)
            max_radius = math.sqrt(float(radius * radius + radius * radius)) + 1.0e-6
            low_r = max_radius * float(self.freq_cutoff)
            high_r = max_radius * float(1.0 - self.freq_cutoff)

            low_mask = (dist <= low_r).float()
            high_mask = (dist >= high_r).float()
            mid_mask = ((dist > low_r) & (dist < high_r)).float()

            # Masks operate in the shifted FFT domain (DC at center).
            self.register_buffer("_freq_low_mask", low_mask.view(1, 1, ws, ws), persistent=False)
            self.register_buffer("_freq_mid_mask", mid_mask.view(1, 1, ws, ws), persistent=False)
            self.register_buffer("_freq_high_mask", high_mask.view(1, 1, ws, ws), persistent=False)

    def _fft_split_bands(self, corr_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split a ws×ws correlation map into low/mid/high frequency components using 2D FFT masks.

        Args:
            corr_map: (B, 1, ws, ws) real tensor
        Returns:
            low, mid, high: each (B, 1, ws, ws) real tensors whose sum reconstructs corr_map.
        """
        if corr_map.dim() != 4:
            raise ValueError(f"corr_map must be (B,1,ws,ws), got {tuple(corr_map.shape)}")
        ws = int(self.window_size)
        if corr_map.shape[-2] != ws or corr_map.shape[-1] != ws:
            raise ValueError(
                f"corr_map spatial dims must be (ws,ws)=({ws},{ws}), got {tuple(corr_map.shape[-2:])}"
            )

        # FFT kernels are not consistently autocast-safe across PyTorch/CUDA builds.
        # Force fp32 to avoid mixed-precision runtime errors and keep the spectral
        # split numerically stable.
        corr_map_f = corr_map.float()
        fft = torch.fft.fftshift(torch.fft.fft2(corr_map_f, dim=(-2, -1)), dim=(-2, -1))
        low_fft = fft * self._freq_low_mask.to(device=fft.device, dtype=corr_map_f.dtype)
        mid_fft = fft * self._freq_mid_mask.to(device=fft.device, dtype=corr_map_f.dtype)
        high_fft = fft * self._freq_high_mask.to(device=fft.device, dtype=corr_map_f.dtype)

        low = torch.fft.ifft2(torch.fft.ifftshift(low_fft, dim=(-2, -1)), dim=(-2, -1)).real
        mid = torch.fft.ifft2(torch.fft.ifftshift(mid_fft, dim=(-2, -1)), dim=(-2, -1)).real
        high = torch.fft.ifft2(torch.fft.ifftshift(high_fft, dim=(-2, -1)), dim=(-2, -1)).real
        return low, mid, high

    def forward(
        self,
        feature_map: torch.Tensor,
        center_point_features: torch.Tensor,
        tracks_bt: torch.Tensor,
        query_t: torch.Tensor,
        return_step: bool = False,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            feature_map: (B, T, H, W, C) float
            center_point_features: (B, T, N, C) sampled at tracks_bt
            tracks_bt: (B, T, N, 2) normalized [y,x] in [0,1]
            query_t: (B, N) int indices in [0, T-1]
        Returns:
            corr_emb: (B, T, N, out_dim)
        """
        if feature_map.dim() != 5:
            raise ValueError(f"feature_map must be (B,T,H,W,C), got {tuple(feature_map.shape)}")
        if center_point_features.dim() != 4:
            raise ValueError(
                f"center_point_features must be (B,T,N,C), got {tuple(center_point_features.shape)}"
            )
        if tracks_bt.dim() != 4:
            raise ValueError(f"tracks_bt must be (B,T,N,2), got {tuple(tracks_bt.shape)}")

        B, T, H, W, C = feature_map.shape
        N = tracks_bt.shape[2]
        K = self.window_size * self.window_size

        batch_idx = torch.arange(B, device=center_point_features.device).view(B, 1).expand(B, N)
        point_idx = torch.arange(N, device=center_point_features.device).view(1, N).expand(B, N)
        query_t = query_t.clamp(0, T - 1)
        query_feats = None
        query_feats_bt = None
        if self.reference_mode == "query":
            query_feats = center_point_features[batch_idx, query_t, point_idx]  # (B,N,C)
            query_feats = self.query_proj(query_feats)
            if self.normalize:
                query_feats = F.normalize(query_feats, dim=-1)
        else:
            # Use a per-frame reference feature:
            # - prev: always use features from frame t-1
            # - next: always use features from frame t+1
            # - adjacent: use the neighbor frame with respect to query_t:
            #     t > query_t -> t-1 (forward tracking)
            #     t < query_t -> t+1 (backward tracking)
            #     t == query_t -> query_t (identity)
            projected = self.query_proj(center_point_features)  # (B,T,N,corr_dim)
            if self.normalize:
                projected = F.normalize(projected, dim=-1)

            if self.reference_mode == "prev":
                query_feats_bt = torch.cat([projected[:, :1], projected[:, :-1]], dim=1)
            elif self.reference_mode == "next":
                query_feats_bt = torch.cat([projected[:, 1:], projected[:, -1:]], dim=1)
            else:
                t_idx = torch.arange(T, device=projected.device).view(1, T, 1).expand(B, T, N)
                q = query_t.view(B, 1, N).expand(B, T, N)
                ref_idx = torch.where(t_idx > q, t_idx - 1, torch.where(t_idx < q, t_idx + 1, q))
                ref_idx = ref_idx.clamp(0, T - 1).long()
                query_feats_bt = torch.gather(
                    projected,
                    dim=1,
                    index=ref_idx.unsqueeze(-1).expand(B, T, N, self.corr_dim),
                )

        fmap = feature_map.reshape(B * T, H, W, C).permute(0, 3, 1, 2)  # (B*T,C,H,W)
        fmap = self.map_proj(fmap)  # (B*T,corr_dim,H,W)
        if self.normalize:
            fmap = F.normalize(fmap, dim=1)

        center_grid = _normalized_yx_to_grid_xy(tracks_bt, H, W)  # (B,T,N,2) in (x,y) grid coords
        center_grid = center_grid.reshape(B * T, N, 1, 2)  # (B*T,N,1,2)

        if H > 1:
            y_step = 2.0 / float(H - 1)
        else:
            y_step = 0.0
        if W > 1:
            x_step = 2.0 / float(W - 1)
        else:
            x_step = 0.0

        offsets = self._offsets_yx.to(device=center_grid.device, dtype=center_grid.dtype)  # (K,2) dy,dx
        offset_grid = torch.stack(
            [offsets[:, 1] * x_step, offsets[:, 0] * y_step],
            dim=-1,
        ).view(1, 1, K, 2)

        grid = center_grid + offset_grid  # (B*T,N,K,2)
        patch = F.grid_sample(fmap, grid, mode="bilinear", align_corners=True)  # (B*T,corr_dim,N,K)
        patch = patch.reshape(B, T, self.corr_dim, N, K)  # (B,T,corr_dim,N,K)

        if self.corr_mode == "all_pair":
            if self.reference_mode != "query":
                raise ValueError(
                    "corr_mode='all_pair' currently requires reference_mode='query' "
                    f"(got reference_mode={self.reference_mode!r})."
                )
            ref_yx = tracks_bt[batch_idx, query_t, point_idx]  # (B,N,2) normalized y/x
            ref_grid = _normalized_yx_to_grid_xy(ref_yx, H, W).reshape(B * N, 1, 1, 2)  # (B*N,1,1,2)

            bt_query = (batch_idx * int(T) + query_t).reshape(B * N).long()
            fmap_query = fmap[bt_query]  # (B*N,corr_dim,H,W)

            q_offsets = self._query_offsets_yx.to(device=ref_grid.device, dtype=ref_grid.dtype)  # (Kq,2) dy,dx
            qK = q_offsets.shape[0]
            q_offset_grid = torch.stack([q_offsets[:, 1] * x_step, q_offsets[:, 0] * y_step], dim=-1).view(
                1, 1, qK, 2
            )
            q_grid = ref_grid + q_offset_grid  # (B*N,1,qK,2)
            q_patch = F.grid_sample(fmap_query, q_grid, mode="bilinear", align_corners=True)  # (B*N,corr_dim,1,qK)
            q_patch = q_patch.reshape(B, N, self.corr_dim, qK)  # (B,N,corr_dim,qK)

            corr_max = None
            chunk = min(int(self.all_pair_chunk), int(qK))
            for start in range(0, int(qK), int(chunk)):
                q_chunk = q_patch[..., start : start + chunk]  # (B,N,C,chunk)
                corr_chunk = torch.einsum("bnck,btcnm->btnkm", q_chunk, patch)  # (B,T,N,chunk,K)
                corr_chunk = corr_chunk.amax(dim=-2)  # (B,T,N,K)
                corr_max = corr_chunk if corr_max is None else torch.maximum(corr_max, corr_chunk)
            corr = corr_max
        elif query_feats_bt is not None:
            corr = torch.einsum("btnc,btcnk->btnk", query_feats_bt, patch)
        else:
            corr = torch.einsum("bnc,btcnk->btnk", query_feats, patch)
        if not self.normalize:
            corr = corr / math.sqrt(float(self.corr_dim))
        temp = max(float(self.temperature), 1.0e-6)
        corr = corr / temp

        corr_flat = corr.reshape(B * T * N, K)
        emb = self.embed(corr_flat).reshape(B, T, N, self.out_dim)
        emb = emb * float(self.scale)

        if self.freq_enabled and self.freq_scale > 0 and self.freq_embed_mid is not None:
            if self.freq_input == "prob":
                corr_map = torch.softmax(corr, dim=-1).reshape(B * T * N, 1, self.window_size, self.window_size)
            else:
                corr_map = corr.reshape(B * T * N, 1, self.window_size, self.window_size)

            low_map, mid_map, high_map = self._fft_split_bands(corr_map)
            low_flat = low_map.reshape(B * T * N, K)
            mid_flat = mid_map.reshape(B * T * N, K)
            high_flat = high_map.reshape(B * T * N, K)

            low_emb = self.freq_embed_low(low_flat).reshape(B, T, N, self.out_dim)
            mid_emb = self.freq_embed_mid(mid_flat).reshape(B, T, N, self.out_dim)
            high_emb = self.freq_embed_high(high_flat).reshape(B, T, N, self.out_dim)

            # MSF-style fusion: mid-band guides how much low/high residual we keep.
            msf_gate = self.freq_msf_gate(mid_emb.reshape(B * T * N, self.out_dim)).reshape(B, T, N, 2)
            msf_gate = torch.softmax(msf_gate, dim=-1)
            fused_emb = mid_emb + msf_gate[..., 0:1] * low_emb + msf_gate[..., 1:2] * high_emb

            if self.freq_out_mode == "replace":
                emb = fused_emb * float(self.freq_scale)
            else:
                emb = emb + fused_emb * float(self.freq_scale)

        if not return_step:
            return emb

        # Convert correlation scores into an expected offset (soft-argmax).
        # Offsets are in feature-map pixel units; convert to input-normalized y/x
        # by dividing by feature-map (H, W). This aligns with how we sample features:
        # y_feature = y_norm * H_feature.
        prob = torch.softmax(corr, dim=-1)  # (B,T,N,K)
        if abs(self.prob_power - 1.0) > 1e-6:
            prob = prob.clamp(min=1.0e-8) ** self.prob_power
            prob = prob / prob.sum(dim=-1, keepdim=True).clamp(min=1.0e-8)

        if self.freq_enabled and self.freq_step_mode in ("mid", "msf"):
            prob_map = prob.reshape(B * T * N, 1, self.window_size, self.window_size)
            low_map, mid_map, high_map = self._fft_split_bands(prob_map)

            if self.freq_step_mode == "mid":
                prob_eff_map = mid_map
            else:
                mid_flat = mid_map.reshape(B * T * N, K)
                low_flat = low_map.reshape(B * T * N, K)
                high_flat = high_map.reshape(B * T * N, K)
                if self.freq_embed_mid is not None and self.freq_msf_gate is not None:
                    mid_emb = self.freq_embed_mid(mid_flat).reshape(B, T, N, self.out_dim)
                    low_emb = self.freq_embed_low(low_flat).reshape(B, T, N, self.out_dim)
                    high_emb = self.freq_embed_high(high_flat).reshape(B, T, N, self.out_dim)
                    msf_gate = self.freq_msf_gate(mid_emb.reshape(B * T * N, self.out_dim)).reshape(B, T, N, 2)
                    msf_gate = torch.softmax(msf_gate, dim=-1)
                    low_w = msf_gate[..., 0:1].reshape(B * T * N, 1, 1, 1)
                    high_w = msf_gate[..., 1:2].reshape(B * T * N, 1, 1, 1)
                else:
                    low_w = 0.5
                    high_w = 0.5
                prob_eff_map = mid_map + low_map * low_w + high_map * high_w

            prob_eff_map = prob_eff_map.clamp(min=0.0)
            prob_eff = prob_eff_map.reshape(B * T * N, K)
            prob_eff = prob_eff / prob_eff.sum(dim=-1, keepdim=True).clamp(min=1.0e-8)
            prob = prob_eff.reshape(B, T, N, K)
        offsets = self._offsets_yx.to(device=prob.device, dtype=prob.dtype)  # (K,2) dy,dx
        if self.step_mode == "argmax":
            idx = prob.argmax(dim=-1)  # (B,T,N)
            expected_yx = offsets[idx]  # (B,T,N,2)
        else:
            expected_yx = torch.einsum("btnk,kd->btnd", prob, offsets)  # (B,T,N,2) dy,dx

        denom = torch.tensor([float(H), float(W)], device=prob.device, dtype=prob.dtype).view(1, 1, 1, 2)
        step_bt = expected_yx / denom  # (B,T,N,2) normalized [dy,dx] in y/x

        if self.learned_step_gate and self.step_gate_mlp is not None:
            max_prob = prob.max(dim=-1).values  # (B,T,N)
            center_idx = K // 2
            center_prob = prob[..., center_idx]  # (B,T,N)
            shift_prob = (max_prob - center_prob).clamp(min=0.0)
            entropy = -(prob.clamp(min=1.0e-8) * prob.clamp(min=1.0e-8).log()).sum(dim=-1)  # (B,T,N)
            entropy = entropy / math.log(float(K) + 1.0e-6)
            radius = float(self.window_size // 2) + 1.0e-6
            expected_norm = torch.norm(expected_yx, dim=-1) / radius  # (B,T,N)

            gate_in = torch.stack([shift_prob, max_prob, entropy, expected_norm], dim=-1)  # (B,T,N,4)
            gate = self.step_gate_mlp(gate_in.reshape(B * T * N, 4)).reshape(B, T, N)
            gate = torch.sigmoid(gate).clamp(0.0, 1.0)
            step_bt = step_bt * gate.unsqueeze(-1)
        step = step_bt.permute(0, 2, 1, 3).contiguous()  # (B,N,T,2)

        confidence = prob.max(dim=-1).values  # (B,T,N)
        if self.confidence_mode in ("shift", "shift_prob"):
            center_idx = K // 2
            confidence = (confidence - prob[..., center_idx]).clamp(min=0.0)
        confidence = confidence.permute(0, 2, 1).contiguous()  # (B,N,T)

        if return_logits:
            return emb, step, confidence, corr
        return emb, step, confidence


class CorrelationMapUpdateHead(nn.Module):
    """
    Predict a refinement step from a local correlation map (RAFT-style update).

    Why this exists
    ---------------
    A soft-argmax step (expected offset) can be overly conservative and can
    average multi-modal peaks. Recent matching-based trackers (e.g., optical
    flow and point tracking families) often feed the *full correlation volume*
    into an update network.

    This head consumes the K=ws*ws correlation logits for each (frame, point),
    converts them to a probability map, and predicts:
      - a step in feature-pixel units (dy, dx) within [-radius, radius]
      - a per-step gate in [0, 1]
    """

    def __init__(
        self,
        window_size: int,
        hidden_dim: int = 64,
        gate_init_bias: float = -1.5,
    ) -> None:
        super().__init__()
        if window_size % 2 != 1 or window_size < 1:
            raise ValueError(f"window_size must be odd and >=1, got {window_size}")
        self.window_size = int(window_size)
        self.radius = int(window_size // 2)

        # A tiny CNN is enough since ws is small (7 or 11).
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(64, max(8, int(hidden_dim))),
            nn.GELU(),
            nn.Linear(max(8, int(hidden_dim)), 3),  # dy, dx, gate
        )
        nn.init.constant_(self.mlp[-1].bias, 0.0)
        self.mlp[-1].bias.data[2] = float(gate_init_bias)

    def forward(self, corr_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            corr_logits: (B, T, N, K) where K=ws*ws
        Returns:
            step_yx_cells: (B, T, N, 2) in feature-pixel units (dy, dx)
            gate: (B, T, N) in [0,1]
        """
        if corr_logits.dim() != 4:
            raise ValueError(f"corr_logits must be (B,T,N,K), got {tuple(corr_logits.shape)}")

        B, T, N, K = corr_logits.shape
        ws = self.window_size
        if K != ws * ws:
            raise ValueError(
                f"corr_logits last dim must be K=ws*ws={ws*ws}, got K={K} (ws={ws})"
            )

        prob = torch.softmax(corr_logits.reshape(B * T * N, K), dim=-1)
        prob_map = prob.reshape(B * T * N, 1, ws, ws)

        feat = self.cnn(prob_map).reshape(B * T * N, 64)
        out = self.mlp(feat).reshape(B, T, N, 3)

        step = torch.tanh(out[..., :2]) * float(self.radius)
        gate = torch.sigmoid(out[..., 2]).clamp(0.0, 1.0)
        return step, gate


class CoTrackerBase(nn.Module):
    """A small wrapper that runs CoTracker predictors and converts coordinate conventions."""

    def __init__(
        self,
        checkpoint: Optional[str],
        offline: bool = True,
        window_len: int = 60,
        v2: bool = False,
        require: bool = True,
    ):
        super().__init__()
        self.require = bool(require)
        self.offline = bool(offline)
        self.window_len = int(window_len)
        self.v2 = bool(v2)

        ckpt = _resolve_checkpoint_path(checkpoint)
        self.checkpoint = ckpt

        self._predictor = None
        self._load_predictor_if_available()

    def _load_predictor_if_available(self):
        _ensure_local_cotracker_path()
        try:
            from cotracker.predictor import CoTrackerPredictor, CoTrackerOnlinePredictor
        except Exception as exc:
            if self.require:
                raise ImportError(
                    "CoTracker is required but not available. "
                    "Install it on the server:\n"
                    "  bash scripts/download_baselines.sh\n"
                    "or manually:\n"
                    "  git clone https://github.com/facebookresearch/co-tracker.git baselines/cotracker\n"
                    "  pip install -e baselines/cotracker\n"
                ) from exc
            logger.warning(f"CoTracker not available, base tracker disabled: {exc}")
            self._predictor = None
            return

        if not self.checkpoint:
            raise ValueError(
                "model.cotracker.checkpoint is required for cotracker_refiner."
            )

        predictor_cls = CoTrackerPredictor if self.offline else CoTrackerOnlinePredictor
        try:
            self._predictor = predictor_cls(
                checkpoint=self.checkpoint,
                offline=self.offline,
                v2=self.v2,
                window_len=self.window_len,
            )
        except TypeError:
            # CoTracker3 API may only accept checkpoint as positional arg
            try:
                self._predictor = predictor_cls(
                    checkpoint=self.checkpoint,
                )
            except TypeError:
                self._predictor = predictor_cls(self.checkpoint)
        self._predictor.eval()

    @torch.no_grad()
    def forward(
        self,
        video: torch.Tensor,
        query_points: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video: (B, T, 3, H, W) float (usually [0,1])
            query_points: (B, N, 3) [t, y, x] (usually normalized)
        Returns:
            tracks: (B, N, T, 2) normalized [y, x]
            visibility: (B, N, T) float in {0,1}
        """
        if self._predictor is None:
            # Fallback for local/dev environments when CoTracker is not installed.
            # This produces trivial "static" tracks so downstream code paths can run.
            B, T, _, _, _ = video.shape
            coords = query_points[..., 1:3]
            coords_max = float(coords.max()) if coords.numel() else 0.0
            if coords_max > 1.5:
                # pixel -> normalized
                height = int(video.shape[-2])
                width = int(video.shape[-1])
                denom_h = float(max(height, 1))
                denom_w = float(max(width, 1))
                coords = torch.stack([coords[..., 0] / denom_h, coords[..., 1] / denom_w], dim=-1)
                coords = coords.clamp(0.0, 1.0)
            tracks = coords.unsqueeze(2).expand(B, coords.shape[1], T, 2).contiguous()
            visibility = torch.ones((B, coords.shape[1], T), device=video.device, dtype=video.dtype)
            if return_features:
                return tracks, visibility, {}
            return tracks, visibility

        B, T, C, H, W = video.shape
        video_cotracker = _ensure_video_uint8_range(video)
        queries = _convert_query_points_to_cotracker(query_points, H, W)

        cotracker_fmaps = None
        hook_handle = None
        captured_chunks: List[torch.Tensor] = []
        if return_features:
            model = getattr(self._predictor, "model", None)
            fnet = getattr(model, "fnet", None) if model is not None else None

            if isinstance(fnet, nn.Module):
                def _hook(_module, _inputs, output):
                    if not isinstance(output, torch.Tensor):
                        return
                    out = output.detach()
                    if out.dim() != 4:
                        return
                    if out.shape[0] % max(int(B), 1) != 0:
                        return
                    t_chunk = out.shape[0] // int(B)
                    captured_chunks.append(out.reshape(B, t_chunk, out.shape[1], out.shape[2], out.shape[3]))

                hook_handle = fnet.register_forward_hook(_hook)

        # CoTracker returns:
        #   tracks: (B, T, N, 2) in pixel (x,y)
        #   visibilities: (B, T, N) bool (thresholded)
        try:
            tracks_xy, vis = self._predictor(video_cotracker, queries=queries)
        finally:
            if hook_handle is not None:
                try:
                    hook_handle.remove()
                except Exception:
                    pass

        tracks = _convert_tracks_from_cotracker(tracks_xy, H, W)
        visibility = vis.permute(0, 2, 1).float().contiguous()

        if return_features and captured_chunks:
            # captured_chunks: list[(B, T_chunk, C, Hf, Wf)]
            try:
                fmaps = torch.cat(captured_chunks, dim=1)  # (B, T, C, Hf, Wf)
                fmaps = fmaps.permute(0, 1, 3, 4, 2).contiguous()  # (B,T,Hf,Wf,C)
                denom = torch.sqrt(
                    torch.clamp((fmaps * fmaps).sum(dim=-1, keepdim=True), min=1.0e-12)
                )
                cotracker_fmaps = (fmaps / denom).contiguous()
            except Exception:
                cotracker_fmaps = None

        if return_features:
            info: Dict[str, object] = {}
            if isinstance(cotracker_fmaps, torch.Tensor):
                info["cotracker_fmaps"] = cotracker_fmaps
            return tracks, visibility, info

        return tracks, visibility


class CoTrackerFSPTRefiner(nn.Module):
    """
    Hybrid tracker: CoTracker base + FSPT refinement heads.

    The refiner is designed to be trainable while the base tracker can be frozen.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._frozen_submodules: Set[str] = set()

        cotracker_cfg = getattr(config, "cotracker", None)
        if cotracker_cfg is None:
            raise ValueError("Missing model.cotracker config for cotracker_refiner.")

        require_cotracker = _as_bool(getattr(cotracker_cfg, "require", True), default=True)
        self.base_tracker = CoTrackerBase(
            checkpoint=str(getattr(cotracker_cfg, "checkpoint", "")) if getattr(cotracker_cfg, "checkpoint", None) else None,
            offline=_as_bool(getattr(cotracker_cfg, "offline", True), default=True),
            window_len=_as_int(getattr(cotracker_cfg, "window_len", 60), 60),
            v2=_as_bool(getattr(cotracker_cfg, "v2", False), default=False),
            require=require_cotracker,
        )
        self.base_tracker.eval()
        for param in self.base_tracker.parameters():
            param.requires_grad = False

        # Geometric backbone for refinement features
        self.geo_backbone = GeometricBackbone(
            backbone_type=config.backbone.type,
            pretrained=config.backbone.pretrained,
            pretrained_path=getattr(config.backbone, "pretrained_path", None),
            freeze_stages=getattr(config.backbone, "freeze_stages", None),
            freeze_bn=getattr(config.backbone, "freeze_bn", False),
            output_dim=config.temporal.dim,
            use_multiscale=getattr(config.backbone, "use_multiscale", True),
        )

        # Semantic encoder (optional)
        if CLIP_AVAILABLE and config.semantic.enabled:
            self.semantic_encoder = SemanticEncoder(
                clip_model=config.clip.model,
                output_dim=config.temporal.dim,
                freeze=config.clip.freeze,
                cache_dir=getattr(config.clip, "cache_dir", None),
            )
            self.use_semantic = True
        else:
            self.semantic_encoder = None
            self.use_semantic = False
            logger.warning("Semantic encoder disabled (CLIP not available or disabled in config)")
        self._semantic_spatial_available = True
        self._warned_semantic_fallback = False

        # Optional CLIP feature cache (recommended for evaluation-only on server)
        self.semantic_cache = None
        self.semantic_cache_only_eval = True
        self.semantic_cache_mode = "spatial"
        semantic_cfg = getattr(config, "semantic", None)
        cache_cfg = getattr(semantic_cfg, "cache", None) if semantic_cfg is not None else None
        if self.use_semantic and cache_cfg is not None and _as_bool(getattr(cache_cfg, "enabled", False), default=False):
            try:
                from utils.tensor_cache import DiskTensorCache

                root_dir = getattr(cache_cfg, "root_dir", "outputs/semantic_cache")
                write = _as_bool(getattr(cache_cfg, "write", True), default=True)
                self.semantic_cache_only_eval = _as_bool(getattr(cache_cfg, "only_eval", True), default=True)
                self.semantic_cache_mode = str(getattr(cache_cfg, "mode", "spatial")).lower().strip()
                keep_in_memory_cfg = getattr(cache_cfg, "keep_in_memory", None)
                if keep_in_memory_cfg is None:
                    keep_in_memory = self.semantic_cache_mode != "spatial"
                else:
                    keep_in_memory = _as_bool(keep_in_memory_cfg, default=True)
                self.semantic_cache = DiskTensorCache(
                    root_dir=str(root_dir),
                    enabled=True,
                    write=write,
                    keep_in_memory=keep_in_memory,
                )
            except Exception as exc:
                logger.warning(f"Semantic cache disabled (init failed): {exc}")

        # Frequency-semantic fusion (optional)
        if config.frequency.enabled:
            fusion_type = getattr(config.semantic, "fusion_type", "cross_attention")
            if not self.use_semantic:
                fusion_type = "none"
            modulation_type = getattr(config.semantic, "modulation_type", "multiplicative")
            if not self.use_semantic:
                modulation_type = "none"
            self.freq_semantic = SemanticEnhancedLFD(
                geo_dim=config.temporal.dim,
                semantic_dim=config.temporal.dim,
                num_bands=config.frequency.num_bands,
                kernel_size=getattr(config.frequency, "kernel_size", 7),
                fusion_type=fusion_type,
                modulation_type=modulation_type,
                use_fixed_laplacian=getattr(config.frequency, "use_fixed_laplacian", False),
                ortho_weight=getattr(config.frequency, "ortho_weight", 0.0),
                feature_ortho_weight=getattr(config.frequency, "feature_ortho_weight", 0.0),
            )
            self.use_frequency = True
        else:
            self.freq_semantic = None
            self.use_frequency = False

        # Temporal module
        self.temporal_transformer = TemporalTransformer(
            dim=config.temporal.dim,
            num_layers=config.temporal.num_layers,
            num_heads=config.temporal.num_heads,
            dropout=config.temporal.dropout,
            max_window_size=getattr(config.temporal, "max_window_size", None),
            use_flash_attention=getattr(config.temporal, "use_flash_attention", True),
        )

        # Residual decoder
        refiner_cfg = getattr(config, "refiner", None)
        online_recovery_cfg = _cfg_get(refiner_cfg, "online_recovery", None)
        if online_recovery_cfg is None:
            online_recovery_cfg = _cfg_get(refiner_cfg, "recovery", None)
        self.online_recovery_requested = _as_bool(
            _cfg_get(online_recovery_cfg, "enabled", False), default=False
        )
        online_recovery_trigger_cfg = _cfg_get(online_recovery_cfg, "trigger", None)
        online_recovery_memory_cfg = _cfg_get(online_recovery_cfg, "memory_bank", None)
        online_recovery_search_cfg = _cfg_get(online_recovery_cfg, "search", None)
        online_recovery_verifier_cfg = _cfg_get(online_recovery_cfg, "verifier", None)
        online_recovery_retracking_cfg = _cfg_get(online_recovery_cfg, "retracking", None)
        delta_scale = _as_float(getattr(refiner_cfg, "delta_scale", 0.2) if refiner_cfg is not None else 0.2, 0.2)
        hidden_dim = _as_int(getattr(refiner_cfg, "hidden_dim", 128) if refiner_cfg is not None else 128, 128)
        gate_cfg = _cfg_get(refiner_cfg, "delta_gate", None)
        use_delta_gate = _as_bool(_cfg_get(gate_cfg, "enabled", False), default=False)
        gate_init_bias = _as_float(_cfg_get(gate_cfg, "init_bias", -4.0), -4.0)

        max_step = None
        max_step_value = _as_float(_cfg_get(refiner_cfg, "delta_max_step", 0.0), 0.0)
        if max_step_value > 0:
            max_step = max_step_value
        self.delta_max_total = None
        max_total_value = _as_float(_cfg_get(refiner_cfg, "delta_max_total", 0.0), 0.0)
        if max_total_value > 0:
            self.delta_max_total = max_total_value

        # Optional: adapt the trust-region clamp based on correlation confidence.
        # This allows the refiner to make larger corrections only when matching is confident,
        # while keeping low-confidence points close to the base tracker.
        adaptive_max_total_cfg = _cfg_get(refiner_cfg, "delta_max_total_adaptive", None)
        self.delta_max_total_adaptive_enabled = _as_bool(
            _cfg_get(adaptive_max_total_cfg, "enabled", False), default=False
        )
        self.delta_max_total_adaptive_min_scale = _as_float(
            _cfg_get(adaptive_max_total_cfg, "min_scale", 0.25), 0.25
        )
        self.delta_max_total_adaptive_max_scale = _as_float(
            _cfg_get(adaptive_max_total_cfg, "max_scale", 1.0), 1.0
        )
        self.delta_max_total_adaptive_threshold = _as_float(
            _cfg_get(adaptive_max_total_cfg, "threshold", 0.15), 0.15
        )
        self.delta_max_total_adaptive_power = _as_float(
            _cfg_get(adaptive_max_total_cfg, "power", 1.0), 1.0
        )
        if not math.isfinite(self.delta_max_total_adaptive_min_scale) or self.delta_max_total_adaptive_min_scale < 0:
            self.delta_max_total_adaptive_min_scale = 0.0
        if not math.isfinite(self.delta_max_total_adaptive_max_scale) or self.delta_max_total_adaptive_max_scale < 0:
            self.delta_max_total_adaptive_max_scale = 1.0
        if self.delta_max_total_adaptive_max_scale < self.delta_max_total_adaptive_min_scale:
            self.delta_max_total_adaptive_max_scale = self.delta_max_total_adaptive_min_scale
        if not math.isfinite(self.delta_max_total_adaptive_threshold) or self.delta_max_total_adaptive_threshold < 0:
            self.delta_max_total_adaptive_threshold = 0.15
        if not math.isfinite(self.delta_max_total_adaptive_power) or self.delta_max_total_adaptive_power <= 0:
            self.delta_max_total_adaptive_power = 1.0

        base_jitter_cfg = _cfg_get(refiner_cfg, "base_jitter", None)
        self.base_jitter_enabled = _as_bool(_cfg_get(base_jitter_cfg, "enabled", False), default=False)
        self.base_jitter_prob = _as_float(_cfg_get(base_jitter_cfg, "prob", 1.0), 1.0)
        self.base_jitter_std_px = _as_float(_cfg_get(base_jitter_cfg, "std_px", 0.0), 0.0)
        self.base_jitter_max_px = _as_float(_cfg_get(base_jitter_cfg, "max_px", 0.0), 0.0)
        self.base_jitter_exclude_query_frame = _as_bool(
            _cfg_get(base_jitter_cfg, "exclude_query_frame", True), default=True
        )
        self.base_jitter_mode = str(_cfg_get(base_jitter_cfg, "mode", "constant")).lower().strip()

        base_occ_cfg = _cfg_get(refiner_cfg, "base_occlusion_drift", None)
        self.base_occ_drift_enabled = _as_bool(_cfg_get(base_occ_cfg, "enabled", False), default=False)
        self.base_occ_drift_prob = _as_float(_cfg_get(base_occ_cfg, "prob", 1.0), 1.0)
        self.base_occ_drift_min_len = _as_int(_cfg_get(base_occ_cfg, "min_len", 20), 20)
        self.base_occ_drift_max_len = _as_int(_cfg_get(base_occ_cfg, "max_len", 20), 20)
        self.base_occ_drift_after_frames = _as_int(_cfg_get(base_occ_cfg, "after_frames", 8), 8)
        self.base_occ_drift_std_px = _as_float(_cfg_get(base_occ_cfg, "drift_std_px", 0.0), 0.0)
        self.base_occ_drift_max_px = _as_float(_cfg_get(base_occ_cfg, "drift_max_px", 0.0), 0.0)
        self.base_occ_drift_exclude_query_frame = _as_bool(
            _cfg_get(base_occ_cfg, "exclude_query_frame", True), default=True
        )
        self.base_occ_drift_force_visible_after = _as_bool(
            _cfg_get(base_occ_cfg, "force_visible_after", True), default=True
        )
        self.base_occ_drift_occluded_value = _as_float(_cfg_get(base_occ_cfg, "occluded_value", 0.0), 0.0)
        self.base_occ_drift_visible_value = _as_float(_cfg_get(base_occ_cfg, "visible_value", 1.0), 1.0)
        self.base_occ_drift_mode = str(_cfg_get(base_occ_cfg, "drift_mode", "constant")).lower().strip()
        if self.base_occ_drift_prob < 0:
            self.base_occ_drift_prob = 0.0
        if self.base_occ_drift_prob > 1:
            self.base_occ_drift_prob = 1.0
        if self.base_occ_drift_min_len < 1:
            self.base_occ_drift_min_len = 1
        if self.base_occ_drift_max_len < self.base_occ_drift_min_len:
            self.base_occ_drift_max_len = self.base_occ_drift_min_len
        if self.base_occ_drift_after_frames < 1:
            self.base_occ_drift_after_frames = 1

        self.residual_decoder = ResidualTrackDecoder(
            dim=config.temporal.dim,
            hidden_dim=hidden_dim,
            delta_scale=delta_scale,
            use_gate=use_delta_gate,
            gate_init_bias=gate_init_bias,
            max_step=max_step,
        )

        # Optional policy gate: learn when to trust the refiner vs the base tracker.
        policy_gate_cfg = _cfg_get(refiner_cfg, "policy_gate", None)
        self.policy_gate_enabled = _as_bool(_cfg_get(policy_gate_cfg, "enabled", False), default=False)
        self.policy_gate_apply = str(_cfg_get(policy_gate_cfg, "apply", "post")).lower().strip()
        self.policy_gate_extra: List[str] = []
        self.policy_gate_head: Optional[nn.Module] = None
        if self.policy_gate_enabled:
            extras_cfg = _cfg_get(policy_gate_cfg, "extra_features", [])
            if isinstance(extras_cfg, str):
                extras_list = [extras_cfg]
            elif isinstance(extras_cfg, (list, tuple)):
                extras_list = list(extras_cfg)
            else:
                extras_list = []
            extras = []
            for item in extras_list:
                if item is None:
                    continue
                name = str(item).lower().strip()
                if name:
                    extras.append(name)
            valid_extras = {"corr_conf", "relocal_conf", "base_vis", "delta_norm"}
            self.policy_gate_extra = [e for e in extras if e in valid_extras]
            gate_hidden = max(4, int(_cfg_get(policy_gate_cfg, "hidden_dim", 64)))
            gate_in_dim = int(config.temporal.dim) + len(self.policy_gate_extra)
            self.policy_gate_head = nn.Sequential(
                nn.Linear(gate_in_dim, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, 1),
            )
            gate_init_bias = _as_float(_cfg_get(policy_gate_cfg, "init_bias", -1.0), -1.0)
            gate_last = self.policy_gate_head[-1]
            if isinstance(gate_last, nn.Linear):
                nn.init.constant_(gate_last.weight, 0.0)
                if gate_last.bias is not None:
                    nn.init.constant_(gate_last.bias, float(gate_init_bias))

        # Optional relocalization acceptor: learn whether to keep global
        # re-localization updates on hard frames, instead of always committing.
        relocal_accept_cfg = _cfg_get(refiner_cfg, "relocal_acceptor", None)
        if relocal_accept_cfg is None:
            relocal_accept_cfg = _cfg_get(refiner_cfg, "verifier", None)
        self.relocal_acceptor_enabled = _as_bool(
            _cfg_get(relocal_accept_cfg, "enabled", False),
            default=False,
        )
        self.relocal_acceptor_extra: List[str] = []
        self.relocal_acceptor_head: Optional[nn.Module] = None
        self.relocal_acceptor_apply_mode = str(
            _cfg_get(relocal_accept_cfg, "apply_mode", "soft")
        ).lower().strip()
        if self.relocal_acceptor_apply_mode in ("binary", "hard_threshold"):
            self.relocal_acceptor_apply_mode = "hard"
        if self.relocal_acceptor_apply_mode not in ("soft", "hard"):
            self.relocal_acceptor_apply_mode = "soft"
        self.relocal_acceptor_threshold = _as_float(
            _cfg_get(relocal_accept_cfg, "threshold", 0.5), 0.5
        )
        if not math.isfinite(self.relocal_acceptor_threshold):
            self.relocal_acceptor_threshold = 0.5
        self.relocal_acceptor_threshold = max(0.0, min(1.0, float(self.relocal_acceptor_threshold)))
        if self.relocal_acceptor_enabled:
            extras_cfg = _cfg_get(relocal_accept_cfg, "extra_features", [])
            if isinstance(extras_cfg, str):
                extras_list = [extras_cfg]
            elif isinstance(extras_cfg, (list, tuple)):
                extras_list = list(extras_cfg)
            else:
                extras_list = []
            extras = []
            for item in extras_list:
                if item is None:
                    continue
                name = str(item).lower().strip()
                if name:
                    extras.append(name)
            valid_extras = {"corr_conf", "relocal_conf", "base_vis", "delta_norm", "relocal_mask"}
            self.relocal_acceptor_extra = [e for e in extras if e in valid_extras]
            accept_hidden = max(4, int(_cfg_get(relocal_accept_cfg, "hidden_dim", 64)))
            accept_in_dim = int(config.temporal.dim) + len(self.relocal_acceptor_extra)
            self.relocal_acceptor_head = nn.Sequential(
                nn.Linear(accept_in_dim, accept_hidden),
                nn.GELU(),
                nn.Linear(accept_hidden, 1),
            )
            accept_init_bias = _as_float(_cfg_get(relocal_accept_cfg, "init_bias", -2.0), -2.0)
            accept_last = self.relocal_acceptor_head[-1]
            if isinstance(accept_last, nn.Linear):
                nn.init.constant_(accept_last.weight, 0.0)
                if accept_last.bias is not None:
                    nn.init.constant_(accept_last.bias, float(accept_init_bias))
        # Paper-facing aliases. Keep the old "relocal_acceptor" name for compatibility,
        # but expose the module as a verifier in logs/configs.
        self.verifier_enabled = self.relocal_acceptor_enabled
        self.verifier_head = self.relocal_acceptor_head
        self.verifier_threshold = self.relocal_acceptor_threshold
        self.verifier_apply_mode = self.relocal_acceptor_apply_mode

        local_corr_cfg = _cfg_get(refiner_cfg, "local_correlation", None)
        self.local_correlation: Optional[LocalTrackCorrelation] = None
        feature_source = str(_cfg_get(refiner_cfg, "feature_source", "geo")).lower().strip()
        if feature_source in ("cotracker", "base", "cotracker_fnet"):
            feature_source = "cotracker"
        else:
            feature_source = "geo"
        self.refiner_feature_source = feature_source

        self.local_corr_feature_level = str(_cfg_get(local_corr_cfg, "feature_level", "fused")).lower().strip()
        if self.local_corr_feature_level not in ("", "fused", "output", "default", "c2"):
            logger.warning(
                f"Unsupported refiner.local_correlation.feature_level={self.local_corr_feature_level!r}; "
                "fallback to 'fused'. (Supported: 'fused', 'c2')"
            )
            self.local_corr_feature_level = "fused"
        corr_feature_source = str(_cfg_get(local_corr_cfg, "feature_source", self.refiner_feature_source)).lower().strip()
        if corr_feature_source in ("cotracker", "base", "cotracker_fnet"):
            corr_feature_source = "cotracker"
        else:
            corr_feature_source = "geo"
        self.local_corr_feature_source = corr_feature_source
        self.local_corr_use_step = False
        self.local_corr_step_scale = 1.0
        self.local_corr_step_gate = "none"
        self.local_corr_gate_threshold = 0.15
        self.local_corr_gate_power = 1.0
        self.local_corr_gate_residual = False
        self.local_corr_apply_step_before_residual = False
        # Gating can be helpful at evaluation time (do-no-harm), but can also
        # prevent position modules from learning if the confidence signal is
        # initially low. Default to eval-only gating; enable *_train flags to
        # apply the same gate during training.
        self.local_corr_gate_step_train = False
        self.local_corr_gate_residual_train = False
        self.local_corr_residual_gate_threshold = 0.15
        self.local_corr_residual_gate_power = 1.0
        self.local_corr_map_update: Optional[CorrelationMapUpdateHead] = None
        self.local_corr_map_update_enabled = False
        self.local_corr_map_update_step_scale = 1.0
        if _as_bool(_cfg_get(local_corr_cfg, "enabled", False), default=False):
            freq_cfg = _cfg_get(local_corr_cfg, "freq", None)
            msf_cfg = _cfg_get(freq_cfg, "msf", None)
            freq_enabled = _as_bool(_cfg_get(freq_cfg, "enabled", False), default=False)
            freq_cutoff = _as_float(_cfg_get(freq_cfg, "cutoff", 0.3), 0.3)
            freq_scale = _as_float(_cfg_get(freq_cfg, "scale", 1.0), 1.0)
            freq_out_mode = str(_cfg_get(freq_cfg, "out_mode", "add"))
            freq_input = str(_cfg_get(freq_cfg, "input", "prob"))
            freq_step_mode = str(_cfg_get(freq_cfg, "step_mode", "raw"))
            freq_msf_hidden_dim = _as_int(_cfg_get(msf_cfg, "hidden_dim", 64), 64)
            corr_mode = str(_cfg_get(local_corr_cfg, "corr_mode", "point"))
            query_window_size = _cfg_get(local_corr_cfg, "query_window_size", None)
            all_pair_chunk = _as_int(_cfg_get(local_corr_cfg, "all_pair_chunk", 8), 8)

            learned_gate_cfg = _cfg_get(local_corr_cfg, "learned_step_gate", None)
            learned_gate_enabled = _as_bool(_cfg_get(learned_gate_cfg, "enabled", False), default=False)
            learned_gate_hidden = _as_int(_cfg_get(learned_gate_cfg, "hidden_dim", 32), 32)
            learned_gate_init_bias = _as_float(_cfg_get(learned_gate_cfg, "init_bias", 0.0), 0.0)
            self.local_correlation = LocalTrackCorrelation(
                in_dim=config.temporal.dim,
                out_dim=config.temporal.dim,
                window_size=_as_int(_cfg_get(local_corr_cfg, "window_size", 7), 7),
                corr_dim=_as_int(_cfg_get(local_corr_cfg, "corr_dim", 64), 64),
                hidden_dim=_as_int(_cfg_get(local_corr_cfg, "hidden_dim", hidden_dim), hidden_dim),
                scale=_as_float(_cfg_get(local_corr_cfg, "scale", 1.0), 1.0),
                normalize=_as_bool(_cfg_get(local_corr_cfg, "normalize", True), default=True),
                temperature=_as_float(_cfg_get(local_corr_cfg, "temperature", 1.0), 1.0),
                projection=str(_cfg_get(local_corr_cfg, "projection", "learned")),
                step_mode=str(_cfg_get(local_corr_cfg, "step_mode", "softargmax")),
                confidence_mode=str(_cfg_get(local_corr_cfg, "confidence_mode", "maxprob")),
                prob_power=_as_float(_cfg_get(local_corr_cfg, "prob_power", 1.0), 1.0),
                reference_mode=str(_cfg_get(local_corr_cfg, "reference_mode", "query")),
                corr_mode=corr_mode,
                query_window_size=query_window_size,
                all_pair_chunk=all_pair_chunk,
                learned_step_gate=learned_gate_enabled,
                learned_step_gate_hidden_dim=learned_gate_hidden,
                learned_step_gate_init_bias=learned_gate_init_bias,
                freq_enabled=freq_enabled,
                freq_cutoff=freq_cutoff,
                freq_scale=freq_scale,
                freq_out_mode=freq_out_mode,
                freq_input=freq_input,
                freq_step_mode=freq_step_mode,
                freq_msf_hidden_dim=freq_msf_hidden_dim,
            )
            self.local_corr_use_step = _as_bool(_cfg_get(local_corr_cfg, "use_step", False), default=False)
            self.local_corr_step_scale = _as_float(_cfg_get(local_corr_cfg, "step_scale", 1.0), 1.0)
            self.local_corr_step_gate = str(_cfg_get(local_corr_cfg, "step_gate", "none")).lower().strip()
            # If enabled, apply correlation step *before* the learned residual update.
            # This keeps the confidence/offset semantics aligned (corr_conf was computed at
            # the same center tracks) and avoids "step computed at A, applied after moving to B".
            self.local_corr_apply_step_before_residual = _as_bool(
                _cfg_get(local_corr_cfg, "apply_step_before_residual", False), default=False
            )
            self.local_corr_gate_threshold = _as_float(_cfg_get(local_corr_cfg, "gate_threshold", 0.15), 0.15)
            self.local_corr_gate_power = _as_float(_cfg_get(local_corr_cfg, "gate_power", 1.0), 1.0)
            self.local_corr_gate_residual = _as_bool(
                _cfg_get(local_corr_cfg, "gate_residual", False), default=False
            )
            self.local_corr_gate_step_train = _as_bool(
                _cfg_get(local_corr_cfg, "gate_step_train", False), default=False
            )
            self.local_corr_gate_residual_train = _as_bool(
                _cfg_get(local_corr_cfg, "gate_residual_train", False), default=False
            )
            self.local_corr_residual_gate_threshold = _as_float(
                _cfg_get(local_corr_cfg, "residual_gate_threshold", self.local_corr_gate_threshold),
                self.local_corr_gate_threshold,
            )
            self.local_corr_residual_gate_power = _as_float(
                _cfg_get(local_corr_cfg, "residual_gate_power", self.local_corr_gate_power),
                self.local_corr_gate_power,
            )

            # Optional: learn an update step from the full correlation map (RAFT-style).
            map_update_cfg = _cfg_get(local_corr_cfg, "map_update", None)
            self.local_corr_map_update_enabled = _as_bool(
                _cfg_get(map_update_cfg, "enabled", False), default=False
            )
            self.local_corr_map_update_step_scale = _as_float(
                _cfg_get(map_update_cfg, "step_scale", 1.0), 1.0
            )
            if self.local_corr_map_update_enabled:
                map_hidden = _as_int(_cfg_get(map_update_cfg, "hidden_dim", 64), 64)
                map_gate_bias = _as_float(_cfg_get(map_update_cfg, "gate_init_bias", -1.5), -1.5)
                self.local_corr_map_update = CorrelationMapUpdateHead(
                    window_size=getattr(self.local_correlation, "window_size", 7),
                    hidden_dim=map_hidden,
                    gate_init_bias=map_gate_bias,
                )

        # Optional: a second (fine) local-correlation module for multi-scale refinement.
        # This matters in Route A where the main refiner feature_source can be CoTracker fnet
        # features (dim=128, stride ~8). A fine correlation head can still leverage a higher
        # resolution geo-backbone pyramid level (e.g., ResNet c2, stride 4) for sub-pixel fixes.
        local_corr_fine_cfg = _cfg_get(refiner_cfg, "local_correlation_fine", None)
        self.local_correlation_fine: Optional[LocalTrackCorrelation] = None
        self.local_corr_fine_feature_level = "c2"
        self.local_corr_fine_feature_source = "geo"
        self.local_corr_fine_use_step = False
        self.local_corr_fine_step_scale = 1.0
        self.local_corr_fine_step_gate = "none"
        self.local_corr_fine_gate_threshold = 0.15
        self.local_corr_fine_gate_power = 1.0
        self.local_corr_fine_apply_step_before_residual = False
        self.local_corr_fine_gate_step_train = False
        self.local_corr_fine_map_update: Optional[CorrelationMapUpdateHead] = None
        self.local_corr_fine_map_update_enabled = False
        self.local_corr_fine_map_update_step_scale = 1.0
        self._warned_local_corr_fine_channel_mismatch = False

        if _as_bool(_cfg_get(local_corr_fine_cfg, "enabled", False), default=False):
            fine_feature_level = str(_cfg_get(local_corr_fine_cfg, "feature_level", "c2")).lower().strip()
            if fine_feature_level in ("", "default", "output", "fused"):
                fine_feature_level = "fused"
            if fine_feature_level not in ("fused", "c2", "c3", "c4", "c5"):
                logger.warning(
                    f"Unsupported refiner.local_correlation_fine.feature_level={fine_feature_level!r}; "
                    "fallback to 'c2'. (Supported: fused, c2, c3, c4, c5)"
                )
                fine_feature_level = "c2"
            self.local_corr_fine_feature_level = fine_feature_level

            fine_feature_source = str(_cfg_get(local_corr_fine_cfg, "feature_source", "geo")).lower().strip()
            if fine_feature_source in ("cotracker", "base", "cotracker_fnet"):
                fine_feature_source = "cotracker"
            else:
                fine_feature_source = "geo"
            self.local_corr_fine_feature_source = fine_feature_source

            fine_in_dim_value = _cfg_get(local_corr_fine_cfg, "in_dim", None)
            fine_in_dim: Optional[int] = None
            if fine_in_dim_value is not None:
                fine_in_dim = _as_int(fine_in_dim_value, 0)

            if fine_in_dim is None or fine_in_dim < 1:
                # Infer channels from the timm backbone feature pyramid when possible.
                if fine_feature_source == "geo" and fine_feature_level in ("c2", "c3", "c4", "c5"):
                    channels = getattr(self.geo_backbone, "backbone_channels", None)
                    level_to_idx = {"c2": 0, "c3": 1, "c4": 2, "c5": 3}
                    idx = level_to_idx.get(fine_feature_level, None)
                    if isinstance(channels, (list, tuple)) and idx is not None and idx < len(channels):
                        try:
                            fine_in_dim = int(channels[idx])
                        except Exception:
                            fine_in_dim = None

            if fine_in_dim is None or fine_in_dim < 1:
                logger.warning(
                    "refiner.local_correlation_fine.enabled=True but could not infer in_dim; "
                    "disabling fine correlation."
                )
            else:
                fine_freq_cfg = _cfg_get(local_corr_fine_cfg, "freq", None)
                fine_msf_cfg = _cfg_get(fine_freq_cfg, "msf", None)
                fine_freq_enabled = _as_bool(_cfg_get(fine_freq_cfg, "enabled", False), default=False)
                fine_freq_cutoff = _as_float(_cfg_get(fine_freq_cfg, "cutoff", 0.3), 0.3)
                fine_freq_scale = _as_float(_cfg_get(fine_freq_cfg, "scale", 1.0), 1.0)
                fine_freq_out_mode = str(_cfg_get(fine_freq_cfg, "out_mode", "add"))
                fine_freq_input = str(_cfg_get(fine_freq_cfg, "input", "prob"))
                fine_freq_step_mode = str(_cfg_get(fine_freq_cfg, "step_mode", "raw"))
                fine_freq_msf_hidden_dim = _as_int(_cfg_get(fine_msf_cfg, "hidden_dim", 64), 64)
                fine_corr_mode = str(_cfg_get(local_corr_fine_cfg, "corr_mode", "point"))
                fine_query_window_size = _cfg_get(local_corr_fine_cfg, "query_window_size", None)
                fine_all_pair_chunk = _as_int(_cfg_get(local_corr_fine_cfg, "all_pair_chunk", 8), 8)

                learned_gate_cfg = _cfg_get(local_corr_fine_cfg, "learned_step_gate", None)
                learned_gate_enabled = _as_bool(_cfg_get(learned_gate_cfg, "enabled", False), default=False)
                learned_gate_hidden = _as_int(_cfg_get(learned_gate_cfg, "hidden_dim", 32), 32)
                learned_gate_init_bias = _as_float(_cfg_get(learned_gate_cfg, "init_bias", 0.0), 0.0)

                self.local_correlation_fine = LocalTrackCorrelation(
                    in_dim=fine_in_dim,
                    out_dim=config.temporal.dim,
                    window_size=_as_int(_cfg_get(local_corr_fine_cfg, "window_size", 7), 7),
                    corr_dim=_as_int(_cfg_get(local_corr_fine_cfg, "corr_dim", 64), 64),
                    hidden_dim=_as_int(_cfg_get(local_corr_fine_cfg, "hidden_dim", hidden_dim), hidden_dim),
                    scale=_as_float(_cfg_get(local_corr_fine_cfg, "scale", 1.0), 1.0),
                    normalize=_as_bool(_cfg_get(local_corr_fine_cfg, "normalize", True), default=True),
                    temperature=_as_float(_cfg_get(local_corr_fine_cfg, "temperature", 1.0), 1.0),
                    projection=str(_cfg_get(local_corr_fine_cfg, "projection", "learned")),
                    step_mode=str(_cfg_get(local_corr_fine_cfg, "step_mode", "softargmax")),
                    confidence_mode=str(_cfg_get(local_corr_fine_cfg, "confidence_mode", "maxprob")),
                    prob_power=_as_float(_cfg_get(local_corr_fine_cfg, "prob_power", 1.0), 1.0),
                    reference_mode=str(_cfg_get(local_corr_fine_cfg, "reference_mode", "query")),
                    corr_mode=fine_corr_mode,
                    query_window_size=fine_query_window_size,
                    all_pair_chunk=fine_all_pair_chunk,
                    learned_step_gate=learned_gate_enabled,
                    learned_step_gate_hidden_dim=learned_gate_hidden,
                    learned_step_gate_init_bias=learned_gate_init_bias,
                    freq_enabled=fine_freq_enabled,
                    freq_cutoff=fine_freq_cutoff,
                    freq_scale=fine_freq_scale,
                    freq_out_mode=fine_freq_out_mode,
                    freq_input=fine_freq_input,
                    freq_step_mode=fine_freq_step_mode,
                    freq_msf_hidden_dim=fine_freq_msf_hidden_dim,
                )
                self.local_corr_fine_use_step = _as_bool(
                    _cfg_get(local_corr_fine_cfg, "use_step", False), default=False
                )
                self.local_corr_fine_step_scale = _as_float(_cfg_get(local_corr_fine_cfg, "step_scale", 1.0), 1.0)
                self.local_corr_fine_step_gate = str(_cfg_get(local_corr_fine_cfg, "step_gate", "none")).lower().strip()
                self.local_corr_fine_apply_step_before_residual = _as_bool(
                    _cfg_get(local_corr_fine_cfg, "apply_step_before_residual", False), default=False
                )
                self.local_corr_fine_gate_threshold = _as_float(_cfg_get(local_corr_fine_cfg, "gate_threshold", 0.15), 0.15)
                self.local_corr_fine_gate_power = _as_float(_cfg_get(local_corr_fine_cfg, "gate_power", 1.0), 1.0)
                self.local_corr_fine_gate_step_train = _as_bool(
                    _cfg_get(local_corr_fine_cfg, "gate_step_train", False), default=False
                )

                # Optional: learn an update step from the fine correlation map (RAFT-style).
                map_update_cfg = _cfg_get(local_corr_fine_cfg, "map_update", None)
                self.local_corr_fine_map_update_enabled = _as_bool(
                    _cfg_get(map_update_cfg, "enabled", False), default=False
                )
                self.local_corr_fine_map_update_step_scale = _as_float(
                    _cfg_get(map_update_cfg, "step_scale", 1.0), 1.0
                )
                if self.local_corr_fine_map_update_enabled:
                    map_hidden = _as_int(_cfg_get(map_update_cfg, "hidden_dim", 64), 64)
                    map_gate_bias = _as_float(_cfg_get(map_update_cfg, "gate_init_bias", -1.5), -1.5)
                    self.local_corr_fine_map_update = CorrelationMapUpdateHead(
                        window_size=getattr(self.local_correlation_fine, "window_size", 7),
                        hidden_dim=map_hidden,
                        gate_init_bias=map_gate_bias,
                    )

        # Optional: fuse coarse+fine correlation steps by confidence weighting.
        fusion_cfg = _cfg_get(refiner_cfg, "local_correlation_fusion", None)
        self.local_corr_fusion_enabled = bool(
            self.local_correlation is not None and self.local_correlation_fine is not None
        )
        self.local_corr_fusion_enabled = _as_bool(
            _cfg_get(fusion_cfg, "enabled", self.local_corr_fusion_enabled),
            default=self.local_corr_fusion_enabled,
        )
        self.local_corr_fusion_power = _as_float(_cfg_get(fusion_cfg, "confidence_power", 1.0), 1.0)
        self.local_corr_fusion_eps = _as_float(_cfg_get(fusion_cfg, "eps", 1.0e-6), 1.0e-6)
        if not math.isfinite(self.local_corr_fusion_power) or self.local_corr_fusion_power <= 0:
            self.local_corr_fusion_power = 1.0
        if not math.isfinite(self.local_corr_fusion_eps) or self.local_corr_fusion_eps <= 0:
            self.local_corr_fusion_eps = 1.0e-6

        # Optional: frequency-guided fusion (paper-facing ablation knob).
        # When enabled and the frequency module is on, we compute a per-point/time
        # high-frequency ratio from the LFD band features and use it to bias the
        # coarse/fine step fusion:
        #   - low-frequency -> prefer coarse (CoTracker fnet / stride~8)
        #   - high-frequency -> prefer fine (geo c2 / stride~4)
        freq_fusion_cfg = _cfg_get(refiner_cfg, "frequency_guided_fusion", None)
        self.freq_guided_fusion_enabled = _as_bool(
            _cfg_get(freq_fusion_cfg, "enabled", False), default=False
        )
        self.freq_guided_fusion_source = str(_cfg_get(freq_fusion_cfg, "source", "lfd_band_energy")).lower().strip()
        if self.freq_guided_fusion_source in ("", "default"):
            self.freq_guided_fusion_source = "lfd_band_energy"
        if self.freq_guided_fusion_source in ("velocity", "motion", "track_velocity", "tracks"):
            self.freq_guided_fusion_source = "track_velocity"
        elif self.freq_guided_fusion_source in ("feature_diff", "feature_derivative", "temporal_derivative"):
            self.freq_guided_fusion_source = "feature_diff"
        else:
            self.freq_guided_fusion_source = "lfd_band_energy"

        self.freq_guided_fusion_invert = _as_bool(_cfg_get(freq_fusion_cfg, "invert", False), default=False)
        self.freq_guided_fusion_detach = _as_bool(_cfg_get(freq_fusion_cfg, "detach", True), default=True)
        self.freq_guided_fusion_strength = _as_float(_cfg_get(freq_fusion_cfg, "strength", 1.0), 1.0)
        if not math.isfinite(self.freq_guided_fusion_strength):
            self.freq_guided_fusion_strength = 1.0
        self.freq_guided_fusion_strength = 0.0 if self.freq_guided_fusion_strength < 0 else (
            1.0 if self.freq_guided_fusion_strength > 1.0 else self.freq_guided_fusion_strength
        )
        self.freq_guided_fusion_power = _as_float(_cfg_get(freq_fusion_cfg, "power", 1.0), 1.0)
        self.freq_guided_fusion_eps = _as_float(_cfg_get(freq_fusion_cfg, "eps", 1.0e-6), 1.0e-6)
        self.freq_guided_fusion_low_band = _as_int(_cfg_get(freq_fusion_cfg, "low_band", 0), 0)
        self.freq_guided_fusion_high_band = _as_int(_cfg_get(freq_fusion_cfg, "high_band", -1), -1)
        self.freq_guided_fusion_velocity_threshold = _as_float(
            _cfg_get(freq_fusion_cfg, "velocity_threshold", 0.01), 0.01
        )
        if self.freq_guided_fusion_power <= 0 or not math.isfinite(self.freq_guided_fusion_power):
            self.freq_guided_fusion_power = 1.0
        if self.freq_guided_fusion_eps <= 0 or not math.isfinite(self.freq_guided_fusion_eps):
            self.freq_guided_fusion_eps = 1.0e-6

        # Optional: occlusion-triggered re-localization step (paper-facing).
        # This is designed for the "re-localization after long occlusion" story:
        #   - Use base visibility to detect re-appearance after long occlusion.
        #   - Run a global matching step (coarse feature map) at those frames only.
        reloc_cfg = _cfg_get(refiner_cfg, "relocalization", None)
        if reloc_cfg is None and self.online_recovery_requested:
            reloc_cfg = online_recovery_cfg
        self.relocalization_enabled = _as_bool(_cfg_get(reloc_cfg, "enabled", False), default=False)
        if self.online_recovery_requested:
            self.relocalization_enabled = True
        self.relocalization_feature_source = str(
            _cfg_get(
                online_recovery_search_cfg,
                "feature_source",
                _cfg_get(reloc_cfg, "feature_source", self.local_corr_feature_source),
            )
        ).lower().strip()
        if self.relocalization_feature_source in ("base", "cotracker_fnet"):
            self.relocalization_feature_source = "cotracker"
        if self.relocalization_feature_source not in ("cotracker", "geo", "dino"):
            logger.warning(
                "Unsupported refiner.relocalization.feature_source="
                f"{self.relocalization_feature_source!r}; disabling relocalization."
            )
            self.relocalization_enabled = False

        # DINO recovery extractor (lazy init). The recovery-only DINO branch
        # uses raw normalized DINO features for cosine search, so we keep the
        # extractor here but do not route through local_correlation projections.
        self._dino_recovery_extractor = None
        self._dino_proj = None
        self._learned_recovery_head = None
        if self.relocalization_feature_source == "dino" and self.relocalization_enabled:
            dino_weights = str(_cfg_get(online_recovery_search_cfg, "dino_weights", ""))
            self._dino_recovery_weights = dino_weights or None
            self._dino_recovery_stride = _as_int(
                _cfg_get(online_recovery_search_cfg, "dino_stride", 14), 14
            )
            dino_dim = _as_int(_cfg_get(online_recovery_search_cfg, "dino_dim", 384), 384)
            self._dino_out_dim = dino_dim
            self._dino_proj_target_dim = None

            # Learned head integration
            self._use_learned_head = _as_bool(
                _cfg_get(online_recovery_search_cfg, "use_learned_head", False), False
            )
            self._learned_head_checkpoint = str(
                _cfg_get(online_recovery_search_cfg, "learned_head_checkpoint", "")
            ) or None
            self._learned_head_loaded = False

        self.relocalization_feature_level = str(
            _cfg_get(reloc_cfg, "feature_level", self.local_corr_feature_level)
        ).lower().strip()
        if self.relocalization_feature_level in ("", "default"):
            self.relocalization_feature_level = "fused"

        self.relocalization_downsample = _as_int(
            _cfg_get(
                online_recovery_search_cfg,
                "downsample",
                _cfg_get(reloc_cfg, "downsample", 2),
            ),
            2,
        )
        self.relocalization_downsample = 1 if self.relocalization_downsample < 1 else self.relocalization_downsample
        self.relocalization_min_occlusion_len = _as_int(
            _cfg_get(
                online_recovery_trigger_cfg,
                "min_occlusion_len",
                _cfg_get(reloc_cfg, "min_occlusion_len", 20),
            ),
            20,
        )
        self.relocalization_frames_after = _as_int(
            _cfg_get(
                online_recovery_trigger_cfg,
                "frames_after",
                _cfg_get(reloc_cfg, "frames_after", 3),
            ),
            3,
        )
        self.relocalization_vis_threshold = _as_float(
            _cfg_get(
                online_recovery_trigger_cfg,
                "vis_threshold",
                _cfg_get(reloc_cfg, "vis_threshold", 0.5),
            ),
            0.5,
        )
        self.relocalization_template_mode = str(
            _cfg_get(
                online_recovery_memory_cfg,
                "template_mode",
                _cfg_get(reloc_cfg, "template_mode", "query"),
            )
        ).lower().strip()
        if self.relocalization_template_mode in ("", "default"):
            self.relocalization_template_mode = "query"
        if self.relocalization_template_mode in ("memory", "memory_bank", "visible_memory"):
            self.relocalization_template_mode = "visible_bank"
        if self.relocalization_template_mode not in ("query", "last_visible", "visible_bank"):
            logger.warning(
                "Unsupported refiner.relocalization.template_mode="
                f"{self.relocalization_template_mode!r}; falling back to 'query'."
            )
            self.relocalization_template_mode = "query"
        self.relocalization_template_bank_size = _as_int(
            _cfg_get(
                online_recovery_memory_cfg,
                "template_bank_size",
                _cfg_get(reloc_cfg, "template_bank_size", 1),
            ),
            1,
        )
        if self.relocalization_template_bank_size < 1:
            self.relocalization_template_bank_size = 1
        if self.relocalization_template_bank_size > 16:
            logger.warning(
                "refiner.relocalization.template_bank_size too large "
                f"({self.relocalization_template_bank_size}); clamping to 16."
            )
            self.relocalization_template_bank_size = 16
        self.relocalization_template_bank_stride = _as_int(
            _cfg_get(
                online_recovery_memory_cfg,
                "template_bank_stride",
                _cfg_get(reloc_cfg, "template_bank_stride", 1),
            ),
            1,
        )
        if self.relocalization_template_bank_stride < 1:
            self.relocalization_template_bank_stride = 1
        if self.relocalization_template_bank_size > 1 and self.relocalization_template_mode not in ("last_visible", "visible_bank"):
            logger.warning(
                "refiner.relocalization.template_bank_size>1 is only supported when "
                "template_mode is 'last_visible' or 'visible_bank'; disabling template bank."
            )
            self.relocalization_template_bank_size = 1

        confirm_cfg = _cfg_get(reloc_cfg, "confirm", None)
        self.relocalization_confirm_enabled = _as_bool(_cfg_get(confirm_cfg, "enabled", False), default=False)
        self.relocalization_confirm_dist_threshold = _as_float(_cfg_get(confirm_cfg, "dist_threshold", 0.02), 0.02)
        if not math.isfinite(self.relocalization_confirm_dist_threshold) or self.relocalization_confirm_dist_threshold < 0:
            self.relocalization_confirm_dist_threshold = 0.02
        self.relocalization_confirm_allow_single_frame = _as_bool(
            _cfg_get(confirm_cfg, "allow_single_frame", False),
            default=False,
        )
        default_reloc_temp = float(getattr(self.local_correlation, "temperature", 1.0)) if self.local_correlation is not None else 1.0
        self.relocalization_temperature = _as_float(
            _cfg_get(reloc_cfg, "temperature", default_reloc_temp), default_reloc_temp
        )
        self.relocalization_conf_threshold = _as_float(_cfg_get(reloc_cfg, "conf_threshold", 0.15), 0.15)
        self.relocalization_gate_power = _as_float(_cfg_get(reloc_cfg, "gate_power", 2.0), 2.0)
        self.relocalization_margin_threshold = _as_float(_cfg_get(reloc_cfg, "margin_threshold", 0.0), 0.0)
        if not math.isfinite(self.relocalization_margin_threshold) or self.relocalization_margin_threshold < 0:
            self.relocalization_margin_threshold = 0.0
        self.relocalization_margin_power = _as_float(_cfg_get(reloc_cfg, "margin_power", 1.0), 1.0)
        if not math.isfinite(self.relocalization_margin_power) or self.relocalization_margin_power <= 0:
            self.relocalization_margin_power = 1.0
        self.relocalization_shift_threshold = _as_float(_cfg_get(reloc_cfg, "shift_threshold", 0.0), 0.0)
        if not math.isfinite(self.relocalization_shift_threshold) or self.relocalization_shift_threshold < 0:
            self.relocalization_shift_threshold = 0.0
        self.relocalization_shift_power = _as_float(_cfg_get(reloc_cfg, "shift_power", 1.0), 1.0)
        if not math.isfinite(self.relocalization_shift_power) or self.relocalization_shift_power <= 0:
            self.relocalization_shift_power = 1.0
        self.relocalization_step_mode = str(_cfg_get(reloc_cfg, "step_mode", "softargmax")).lower().strip()
        if self.relocalization_step_mode in ("hard", "argmax", "max"):
            self.relocalization_step_mode = "argmax"
        else:
            self.relocalization_step_mode = "softargmax"
        self.relocalization_apply_only_first_iter = _as_bool(
            _cfg_get(reloc_cfg, "apply_only_first_iter", True), default=True
        )
        self.relocalization_max_step = _as_float(_cfg_get(reloc_cfg, "max_step", 0.0), 0.0)
        if not math.isfinite(self.relocalization_max_step) or self.relocalization_max_step <= 0:
            self.relocalization_max_step = 0.0
        self.relocalization_max_total = _as_float(_cfg_get(reloc_cfg, "max_total", 0.0), 0.0)
        if not math.isfinite(self.relocalization_max_total) or self.relocalization_max_total <= 0:
            self.relocalization_max_total = 0.0
        self.relocalization_topk = _as_int(_cfg_get(reloc_cfg, "topk", 1), 1)
        if self.relocalization_topk < 1:
            self.relocalization_topk = 1
        if self.relocalization_topk > 64:
            logger.warning(
                "refiner.relocalization.topk too large "
                f"({self.relocalization_topk}); clamping to 64."
            )
            self.relocalization_topk = 64
        candidate_fusion_cfg = _cfg_get(reloc_cfg, "candidate_fusion", None)
        self.relocalization_candidate_fusion_enabled = _as_bool(
            _cfg_get(candidate_fusion_cfg, "enabled", False),
            default=False,
        )
        self.relocalization_candidate_fusion_temperature = _as_float(
            _cfg_get(candidate_fusion_cfg, "temperature", 0.5),
            0.5,
        )
        if (
            not math.isfinite(self.relocalization_candidate_fusion_temperature)
            or self.relocalization_candidate_fusion_temperature <= 0
        ):
            self.relocalization_candidate_fusion_temperature = 0.5
        self.relocalization_candidate_fusion_min_weight = _as_float(
            _cfg_get(candidate_fusion_cfg, "min_weight", 0.0),
            0.0,
        )
        if (
            not math.isfinite(self.relocalization_candidate_fusion_min_weight)
            or self.relocalization_candidate_fusion_min_weight < 0
        ):
            self.relocalization_candidate_fusion_min_weight = 0.0
        self.relocalization_temporal_sigma = _as_float(_cfg_get(reloc_cfg, "temporal_sigma", 0.0), 0.0)
        if not math.isfinite(self.relocalization_temporal_sigma) or self.relocalization_temporal_sigma <= 0:
            self.relocalization_temporal_sigma = 0.0
        self.relocalization_temporal_power = _as_float(_cfg_get(reloc_cfg, "temporal_power", 1.0), 1.0)
        if not math.isfinite(self.relocalization_temporal_power) or self.relocalization_temporal_power <= 0:
            self.relocalization_temporal_power = 1.0
        self.relocalization_temporal_min_prev_conf = _as_float(
            _cfg_get(reloc_cfg, "temporal_min_prev_conf", 0.0), 0.0
        )
        if not math.isfinite(self.relocalization_temporal_min_prev_conf):
            self.relocalization_temporal_min_prev_conf = 0.0
        self.relocalization_temporal_min_prev_conf = max(
            0.0, min(1.0, float(self.relocalization_temporal_min_prev_conf))
        )
        self.online_recovery_mode = "disabled"
        if self.online_recovery_requested:
            self.online_recovery_mode = "tracker_conditioned"
        elif self.relocalization_enabled:
            self.online_recovery_mode = "legacy_relocalization"

        # Optional: dedicated residual branch for long-occlusion re-localization.
        # This branch is explicitly masked to re-appearance frames, so it can be
        # more aggressive without globally harming easy regions.
        reloc_res_cfg = _cfg_get(refiner_cfg, "relocalization_residual", None)
        self.relocalization_residual_enabled = _as_bool(
            _cfg_get(reloc_res_cfg, "enabled", False), default=False
        )
        self.relocalization_residual_hidden_dim = _as_int(
            _cfg_get(reloc_res_cfg, "hidden_dim", config.temporal.dim), config.temporal.dim
        )
        if self.relocalization_residual_hidden_dim < 8:
            self.relocalization_residual_hidden_dim = 8
        self.relocalization_residual_scale = _as_float(
            _cfg_get(reloc_res_cfg, "scale", 0.02), 0.02
        )
        if not math.isfinite(self.relocalization_residual_scale):
            self.relocalization_residual_scale = 0.02
        self.relocalization_residual_scale = max(0.0, float(self.relocalization_residual_scale))
        self.relocalization_residual_gate_init_bias = _as_float(
            _cfg_get(reloc_res_cfg, "gate_init_bias", -2.0), -2.0
        )
        if not math.isfinite(self.relocalization_residual_gate_init_bias):
            self.relocalization_residual_gate_init_bias = -2.0
        self.relocalization_residual_conf_threshold = _as_float(
            _cfg_get(reloc_res_cfg, "conf_threshold", 0.0), 0.0
        )
        if not math.isfinite(self.relocalization_residual_conf_threshold):
            self.relocalization_residual_conf_threshold = 0.0
        self.relocalization_residual_conf_power = _as_float(
            _cfg_get(reloc_res_cfg, "conf_power", 1.0), 1.0
        )
        if not math.isfinite(self.relocalization_residual_conf_power) or self.relocalization_residual_conf_power <= 0:
            self.relocalization_residual_conf_power = 1.0
        self.relocalization_residual_max_step = _as_float(
            _cfg_get(reloc_res_cfg, "max_step", 0.0), 0.0
        )
        if not math.isfinite(self.relocalization_residual_max_step) or self.relocalization_residual_max_step <= 0:
            self.relocalization_residual_max_step = 0.0
        self.relocalization_residual_apply_only_first_iter = _as_bool(
            _cfg_get(reloc_res_cfg, "apply_only_first_iter", False), default=False
        )
        self.relocalization_residual_require_mask = _as_bool(
            _cfg_get(reloc_res_cfg, "require_mask", True), default=True
        )
        self.relocalization_residual_tanh = _as_bool(
            _cfg_get(reloc_res_cfg, "tanh", True), default=True
        )
        self.relocalization_residual_delta_head: Optional[nn.Module] = None
        self.relocalization_residual_gate_head: Optional[nn.Module] = None
        self.relocalization_residual_module: Optional[nn.Module] = None
        if self.relocalization_residual_enabled:
            hidden = int(self.relocalization_residual_hidden_dim)
            self.relocalization_residual_delta_head = nn.Sequential(
                nn.Linear(config.temporal.dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 2),
            )
            self.relocalization_residual_gate_head = nn.Sequential(
                nn.Linear(config.temporal.dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            self.relocalization_residual_module = nn.ModuleList(
                [self.relocalization_residual_delta_head, self.relocalization_residual_gate_head]
            )

        # Optional: frame-wise global init (TAPIR-style).
        frame_cfg = _cfg_get(refiner_cfg, "framewise_init", None)
        self.framewise_init_enabled = _as_bool(_cfg_get(frame_cfg, "enabled", False), default=False)
        self.framewise_init_feature_source = str(
            _cfg_get(frame_cfg, "feature_source", self.relocalization_feature_source)
        ).lower().strip()
        if self.framewise_init_feature_source in ("base", "cotracker_fnet"):
            self.framewise_init_feature_source = "cotracker"
        if self.framewise_init_feature_source not in ("cotracker", "geo"):
            logger.warning(
                "Unsupported refiner.framewise_init.feature_source="
                f"{self.framewise_init_feature_source!r}; disabling framewise_init."
            )
            self.framewise_init_enabled = False
        self.framewise_init_feature_level = str(
            _cfg_get(frame_cfg, "feature_level", self.relocalization_feature_level)
        ).lower().strip()
        if self.framewise_init_feature_level in ("", "default"):
            self.framewise_init_feature_level = "fused"
        self.framewise_init_downsample = _as_int(
            _cfg_get(frame_cfg, "downsample", self.relocalization_downsample),
            self.relocalization_downsample,
        )
        self.framewise_init_downsample = 1 if self.framewise_init_downsample < 1 else self.framewise_init_downsample
        self.framewise_init_template_mode = str(
            _cfg_get(frame_cfg, "template_mode", "query")
        ).lower().strip()
        if self.framewise_init_template_mode not in ("query", "last_visible"):
            logger.warning(
                "Unsupported refiner.framewise_init.template_mode="
                f"{self.framewise_init_template_mode!r}; falling back to 'query'."
            )
            self.framewise_init_template_mode = "query"
        self.framewise_init_exclude_query_frame = _as_bool(
            _cfg_get(frame_cfg, "exclude_query_frame", True), default=True
        )
        self.framewise_init_after_query = _as_bool(
            _cfg_get(frame_cfg, "after_query", True), default=True
        )
        self.framewise_init_stride = _as_int(_cfg_get(frame_cfg, "stride", 1), 1)
        if self.framewise_init_stride < 1:
            self.framewise_init_stride = 1
        self.framewise_init_apply_prob = _as_float(_cfg_get(frame_cfg, "apply_prob", 1.0), 1.0)
        self.framewise_init_apply_prob = max(0.0, min(1.0, float(self.framewise_init_apply_prob)))
        self.framewise_init_conf_threshold = _as_float(_cfg_get(frame_cfg, "conf_threshold", 0.0), 0.0)
        if not math.isfinite(self.framewise_init_conf_threshold) or self.framewise_init_conf_threshold < 0:
            self.framewise_init_conf_threshold = 0.0
        self.framewise_init_gate_power = _as_float(_cfg_get(frame_cfg, "gate_power", 1.0), 1.0)
        if not math.isfinite(self.framewise_init_gate_power) or self.framewise_init_gate_power <= 0:
            self.framewise_init_gate_power = 1.0

        # Optional: inject base-tracker priors into transformer tokens (paper-friendly ablation knob).
        prior_cfg = _cfg_get(refiner_cfg, "prior_features", None)
        self.prior_features_enabled = _as_bool(_cfg_get(prior_cfg, "enabled", False), default=False)
        self.prior_features_use_base_visibility = _as_bool(
            _cfg_get(prior_cfg, "use_base_visibility", True), default=True
        )
        self.prior_features_use_delta = _as_bool(_cfg_get(prior_cfg, "use_delta", True), default=True)
        self.prior_features_use_delta_norm = _as_bool(
            _cfg_get(prior_cfg, "use_delta_norm", False), default=False
        )
        self.prior_features_detach_delta = _as_bool(
            _cfg_get(prior_cfg, "detach_delta", True), default=True
        )
        self.prior_features_scale = _as_float(_cfg_get(prior_cfg, "scale", 1.0), 1.0)
        prior_hidden_dim = _as_int(_cfg_get(prior_cfg, "hidden_dim", hidden_dim), hidden_dim)
        prior_dim = 0
        if self.prior_features_use_base_visibility:
            prior_dim += 1
        if self.prior_features_use_delta:
            prior_dim += 2
            if self.prior_features_use_delta_norm:
                prior_dim += 1
        self.prior_features_mlp: Optional[nn.Module] = None
        if self.prior_features_enabled and prior_dim > 0:
            self.prior_features_mlp = nn.Sequential(
                nn.Linear(prior_dim, prior_hidden_dim),
                nn.GELU(),
                nn.Linear(prior_hidden_dim, config.temporal.dim),
            )
        elif self.prior_features_enabled:
            logger.warning("refiner.prior_features.enabled=True but no prior signals selected; disabling.")
            self.prior_features_enabled = False

        # Occlusion predictor (optional)
        occlusion_enabled = bool(config.occlusion.enabled)
        if occlusion_enabled and not self.use_frequency:
            logger.warning("Occlusion predictor disabled because frequency module is off.")
            occlusion_enabled = False
        if occlusion_enabled:
            self.occlusion_predictor = FrequencyAwareOcclusionPredictor(
                dim=config.temporal.dim,
                semantic_dim=config.temporal.dim,
                num_bands=config.frequency.num_bands,
                use_semantic_consistency=bool(
                    config.occlusion.use_semantic_propagation and self.use_semantic
                ),
                use_low_freq_prediction=getattr(config.occlusion, "use_low_freq_prediction", True),
            )
            self.use_occlusion = True
        else:
            self.occlusion_predictor = None
            self.use_occlusion = False
        # Track fusion policy for occlusion branch:
        # - train: default off to avoid trajectory predictor interfering with position supervision
        # - eval: default on to keep occlusion-aware refinement behavior
        self.occlusion_track_fusion_train = bool(
            getattr(config.occlusion, "track_fusion_train", False)
        )
        self.occlusion_track_fusion_eval = bool(
            getattr(config.occlusion, "track_fusion_eval", True)
        )
        self._warned_missing_band_features = False

        # Iterative refinement
        self.num_refinement_iters = _as_int(
            getattr(config, "num_refinement_iters", getattr(config.temporal, "num_refinement_iters", 1)),
            1,
        )
        if self.num_refinement_iters < 1:
            self.num_refinement_iters = 1

        self._init_weights()
        self.residual_decoder.init_identity()
        self._init_relocalization_residual_identity()
        self._apply_trainability_from_config(refiner_cfg)
        self._active_stage_idx: Optional[int] = None

        visibility_prior_cfg = _cfg_get(refiner_cfg, "visibility_prior", None)
        self.visibility_prior_mode = str(_cfg_get(visibility_prior_cfg, "mode", "none")).lower().strip()
        self.visibility_prior_alpha = _as_float(_cfg_get(visibility_prior_cfg, "alpha", 0.5), 0.5)
        self.visibility_prior_strength = _as_float(_cfg_get(visibility_prior_cfg, "strength", 1.0), 1.0)
        self.visibility_prior_eps = _as_float(_cfg_get(visibility_prior_cfg, "eps", 1e-4), 1e-4)
        self.visibility_prior_force_base_when_occlusion_disabled = _as_bool(
            _cfg_get(visibility_prior_cfg, "force_base_when_occlusion_disabled", True),
            default=True,
        )

        # Optional: re-track after global re-localization (long-occlusion story).
        #
        # Motivation:
        # - Stage-3 relocalization can find a better position at re-appearance, but
        #   local refinement may not propagate that correction through the remaining frames.
        # - A strong base tracker (CoTracker3) can often track well once re-initialized
        #   at a better location. Therefore, we can restart the base tracker at a
        #   relocalized frame and splice the tail of the trajectory.
        retrack_cfg = _cfg_get(refiner_cfg, "retracking", None)
        self.retracking_enabled = _as_bool(_cfg_get(retrack_cfg, "enabled", False), default=False)
        self.retracking_apply_in_train = _as_bool(_cfg_get(retrack_cfg, "apply_in_train", False), default=False)
        self.retracking_selection = str(_cfg_get(retrack_cfg, "selection", "first")).lower().strip()
        if self.retracking_selection not in ("first", "max_conf"):
            self.retracking_selection = "first"
        self.retracking_overwrite_mode = str(_cfg_get(retrack_cfg, "overwrite_mode", "to_end")).lower().strip()
        if self.retracking_overwrite_mode not in ("to_end", "window"):
            self.retracking_overwrite_mode = "to_end"
        self.retracking_overwrite_window = _as_int(_cfg_get(retrack_cfg, "overwrite_window", 0), 0)
        if self.retracking_overwrite_window < 0:
            self.retracking_overwrite_window = 0
        self.retracking_min_conf: Optional[float] = None
        min_conf_raw = _cfg_get(retrack_cfg, "min_conf", None)
        if min_conf_raw is not None:
            try:
                self.retracking_min_conf = float(min_conf_raw)
            except Exception:
                self.retracking_min_conf = None
        if self.online_recovery_requested and online_recovery_retracking_cfg is not None:
            self.retracking_enabled = _as_bool(
                _cfg_get(online_recovery_retracking_cfg, "enabled", self.retracking_enabled),
                default=self.retracking_enabled,
            )
            self.retracking_apply_in_train = _as_bool(
                _cfg_get(online_recovery_retracking_cfg, "apply_in_train", self.retracking_apply_in_train),
                default=self.retracking_apply_in_train,
            )
            overwrite_mode = str(
                _cfg_get(online_recovery_retracking_cfg, "overwrite_mode", self.retracking_overwrite_mode)
            ).lower().strip()
            if overwrite_mode in ("to_end", "window"):
                self.retracking_overwrite_mode = overwrite_mode
            self.retracking_overwrite_window = _as_int(
                _cfg_get(
                    online_recovery_retracking_cfg,
                    "overwrite_window",
                    self.retracking_overwrite_window,
                ),
                self.retracking_overwrite_window,
            )
            if self.retracking_overwrite_window < 0:
                self.retracking_overwrite_window = 0
            min_conf_raw = _cfg_get(online_recovery_retracking_cfg, "min_conf", self.retracking_min_conf)
            if min_conf_raw is not None:
                try:
                    self.retracking_min_conf = float(min_conf_raw)
                except Exception:
                    pass
        self.online_recovery_summary = {
            "requested": bool(self.online_recovery_requested),
            "mode": str(self.online_recovery_mode),
            "feature_source": str(self.relocalization_feature_source),
            "template_mode": str(self.relocalization_template_mode),
            "template_bank_size": int(self.relocalization_template_bank_size),
            "min_occlusion_len": int(self.relocalization_min_occlusion_len),
            "frames_after": int(self.relocalization_frames_after),
            "vis_threshold": float(self.relocalization_vis_threshold),
            "retracking_enabled": bool(self.retracking_enabled),
            "use_learned_head": bool(getattr(self, "_use_learned_head", False)),
        }
        if self.retracking_min_conf is not None:
            if (not math.isfinite(self.retracking_min_conf)) or self.retracking_min_conf < 0:
                self.retracking_min_conf = None

        focus_cfg = _cfg_get(refiner_cfg, "focus_corrections", None)
        self.focus_corrections_enabled = _as_bool(_cfg_get(focus_cfg, "enabled", False), default=False)
        # By default, focus corrections apply to both train and eval, matching the historical behavior.
        # For most paper runs we recommend `apply_in_train: false` so the heuristic masking does not
        # block gradients during training, while still affecting eval outputs/metrics.
        self.focus_corrections_apply_in_train = _as_bool(
            _cfg_get(focus_cfg, "apply_in_train", True), default=True
        )
        self.focus_corrections_non_focus_scale = _as_float(
            _cfg_get(focus_cfg, "non_focus_scale", 1.0), 1.0
        )
        if not math.isfinite(self.focus_corrections_non_focus_scale):
            self.focus_corrections_non_focus_scale = 1.0
        self.focus_corrections_non_focus_scale = max(
            0.0, min(1.0, float(self.focus_corrections_non_focus_scale))
        )
        self.focus_corrections_use_relocal_mask = _as_bool(
            _cfg_get(focus_cfg, "use_relocal_mask", True), default=True
        )
        self.focus_corrections_use_corr_conf = _as_bool(
            _cfg_get(focus_cfg, "use_corr_conf", True), default=True
        )
        self.focus_corrections_corr_threshold = _as_float(
            _cfg_get(focus_cfg, "corr_threshold", 0.25), 0.25
        )
        if not math.isfinite(self.focus_corrections_corr_threshold):
            self.focus_corrections_corr_threshold = 0.25
        self.focus_corrections_use_relocal_conf = _as_bool(
            _cfg_get(focus_cfg, "use_relocal_conf", True), default=True
        )
        self.focus_corrections_relocal_threshold = _as_float(
            _cfg_get(focus_cfg, "relocal_threshold", 0.2), 0.2
        )
        if not math.isfinite(self.focus_corrections_relocal_threshold):
            self.focus_corrections_relocal_threshold = 0.2
        self.focus_corrections_include_occluded = _as_bool(
            _cfg_get(focus_cfg, "include_occluded", False), default=False
        )
        self.focus_corrections_occluded_threshold = _as_float(
            _cfg_get(focus_cfg, "occluded_threshold", 0.5), 0.5
        )
        if not math.isfinite(self.focus_corrections_occluded_threshold):
            self.focus_corrections_occluded_threshold = 0.5
        self.focus_corrections_occluded_threshold = max(
            0.0, min(1.0, float(self.focus_corrections_occluded_threshold))
        )

        # Optional query-level focusing for long-occlusion specialization.
        # If a query experiences a long occlusion run (estimated from the base
        # tracker's visibility), allow full-strength corrections on all frames
        # for that query. This makes it easier to preserve long-occ gains while
        # being conservative on easy/short-occlusion queries.
        self.focus_corrections_use_long_occlusion_query = _as_bool(
            _cfg_get(focus_cfg, "use_long_occlusion_query", False), default=False
        )
        self.focus_corrections_long_occlusion_len = _as_int(
            _cfg_get(focus_cfg, "long_occlusion_len", 30), 30
        )
        if self.focus_corrections_long_occlusion_len < 1:
            self.focus_corrections_long_occlusion_len = 1
        self.focus_corrections_long_occlusion_vis_threshold = _as_float(
            _cfg_get(focus_cfg, "long_occlusion_vis_threshold", 0.5), 0.5
        )
        if not math.isfinite(self.focus_corrections_long_occlusion_vis_threshold):
            self.focus_corrections_long_occlusion_vis_threshold = 0.5
        self.focus_corrections_long_occlusion_vis_threshold = max(
            0.0, min(1.0, float(self.focus_corrections_long_occlusion_vis_threshold))
        )
        self.focus_corrections_long_occlusion_mode = str(
            _cfg_get(focus_cfg, "long_occlusion_mode", "max_run")
        ).lower().strip()
        if self.focus_corrections_long_occlusion_mode not in ("max_run", "reappearance"):
            self.focus_corrections_long_occlusion_mode = "max_run"
        self.focus_corrections_long_occlusion_fill_gaps = _as_int(
            _cfg_get(focus_cfg, "long_occlusion_fill_gaps", 0), 0
        )
        if self.focus_corrections_long_occlusion_fill_gaps < 0:
            self.focus_corrections_long_occlusion_fill_gaps = 0
        # Optional: tighten the occlusion run definition using corr_conf.
        # This is useful because base visibility can be noisy: some "occluded"
        # spans are false positives where the tracker is actually confident.
        self.focus_corrections_long_occlusion_use_corr_conf_for_occ = _as_bool(
            _cfg_get(focus_cfg, "long_occlusion_use_corr_conf_for_occ", False), default=False
        )
        self.focus_corrections_long_occlusion_occ_corr_threshold = _as_float(
            _cfg_get(focus_cfg, "long_occlusion_occ_corr_threshold", 0.10), 0.10
        )
        if not math.isfinite(self.focus_corrections_long_occlusion_occ_corr_threshold):
            self.focus_corrections_long_occlusion_occ_corr_threshold = 0.10
        self.focus_corrections_long_occlusion_occ_corr_threshold = max(
            0.0, min(1.0, float(self.focus_corrections_long_occlusion_occ_corr_threshold))
        )
        # Optional: apply long-occlusion focus only after re-appearance.
        #
        # Motivation:
        # - Query-level focus ("apply on all frames") preserves long-occ gains but can
        #   slightly harm global metrics because it also perturbs easy frames within
        #   the same query.
        # - Restricting focus to frames AFTER a long occlusion run ends can reduce
        #   false-positive damage while keeping the intended relocalization effect.
        #
        # Semantics:
        # - 0: legacy behavior (focus applies to the entire query trajectory)
        # - -1: apply from re-appearance to end-of-video
        # - >0: apply a window of K frames after each re-appearance
        self.focus_corrections_long_occlusion_apply_window_after_reappear = _as_int(
            _cfg_get(focus_cfg, "long_occlusion_apply_window_after_reappear", 0), 0
        )
        if self.focus_corrections_long_occlusion_apply_window_after_reappear < -1:
            self.focus_corrections_long_occlusion_apply_window_after_reappear = -1
        # Optional: also allow corrections on a small window BEFORE the long-occlusion run starts.
        #
        # Motivation:
        # - Stage-3 relocalization relies on templates from the *last visible* frames before occlusion.
        # - If we only allow corrections after re-appearance, those template frames remain "base-only",
        #   which can weaken the relocalization step.
        #
        # Semantics:
        # - 0: disabled
        # - >0: for each detected long-occlusion run, allow corrections on the K frames immediately
        #       preceding the occlusion start (clamped to >= query_t+1).
        self.focus_corrections_long_occlusion_apply_window_before_occlusion = _as_int(
            _cfg_get(focus_cfg, "long_occlusion_apply_window_before_occlusion", 0), 0
        )
        if self.focus_corrections_long_occlusion_apply_window_before_occlusion < 0:
            self.focus_corrections_long_occlusion_apply_window_before_occlusion = 0
        apply_mode_raw = _cfg_get(focus_cfg, "long_occlusion_apply_mode", None)
        if apply_mode_raw is None:
            # Backward-compatible default:
            # - if a non-zero "window_after_reappear" is set, assume the user wants
            #   after-reappearance focusing.
            # - otherwise, keep the legacy query-level focusing.
            self.focus_corrections_long_occlusion_apply_mode = (
                "after_reappear"
                if self.focus_corrections_long_occlusion_apply_window_after_reappear != 0
                else "query"
            )
        else:
            self.focus_corrections_long_occlusion_apply_mode = str(apply_mode_raw).lower().strip()
        if self.focus_corrections_long_occlusion_apply_mode not in (
            "query",
            "occluded",
            "after_reappear",
            "occluded_or_after_reappear",
        ):
            self.focus_corrections_long_occlusion_apply_mode = "query"

        # Optional: choose the signal used to detect "re-appearance" for AFTER-REAPPEAR focusing.
        #
        # Motivation:
        # - Base visibility can be noisy and may fail to flip back to visible after GT
        #   re-appearance, which makes after-reappear masks empty and collapses gains.
        # - Correlation / relocalization confidence can provide a more direct cue for
        #   "we have re-acquired the point", even when base visibility stays low.
        #
        # Values:
        # - "visibility" (default): base_visibility >= long_occlusion_vis_threshold
        # - "corr_conf": corr_conf_nt >= (long_occlusion_reappear_threshold or corr_threshold)
        # - "relocal_conf": relocal_conf_nt >= (long_occlusion_reappear_threshold or relocal_threshold)
        reappear_src_raw = _cfg_get(focus_cfg, "long_occlusion_reappear_source", None)
        if reappear_src_raw is None:
            self.focus_corrections_long_occlusion_reappear_source = "visibility"
        else:
            src = str(reappear_src_raw).lower().strip()
            if src in ("visibility", "base_visibility", "vis", "base_vis"):
                src = "visibility"
            elif src in ("corr_conf", "corr", "corrconf", "corr_confidence"):
                src = "corr_conf"
            elif src in ("relocal_conf", "relocal", "relocalization_conf", "relocalization"):
                src = "relocal_conf"
            else:
                src = "visibility"
            self.focus_corrections_long_occlusion_reappear_source = src

        # Optional explicit threshold for the chosen re-appearance source.
        # If unset / non-finite, we fall back to a sensible default per source.
        self.focus_corrections_long_occlusion_reappear_threshold = _as_float(
            _cfg_get(focus_cfg, "long_occlusion_reappear_threshold", float("nan")),
            float("nan"),
        )

        # Safety default:
        # If occlusion is disabled, visibility is typically *unsupervised* and can
        # collapse, catastrophically harming OA/AJ. In that case, default to using
        # the base tracker's visibility unless explicitly overridden.
        if (
            (not self.use_occlusion)
            and self.visibility_prior_force_base_when_occlusion_disabled
            and self.visibility_prior_mode in ("none", "", "off", "disabled")
        ):
            self.visibility_prior_mode = "blend"
            self.visibility_prior_alpha = 1.0
            logger.warning(
                "Occlusion disabled but refiner.visibility_prior.mode is 'none'; "
                "forcing visibility_prior.mode='blend', alpha=1.0 to reuse base visibility "
                "and avoid OA collapse. Set refiner.visibility_prior.force_base_when_occlusion_disabled=false "
                "to override."
            )

        self.apply_refiner_stage(epoch=0)

    def _maybe_apply_base_jitter(
        self,
        tracks: torch.Tensor,
        query_t: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Inject small synthetic noise into base tracks during training to avoid the trivial
        'do nothing' optimum when the base tracker is already strong.

        This is a *training-only* perturbation: it should never affect evaluation metrics
        computed on clean CoTracker base tracks.
        """
        if not self.training:
            return tracks
        if not (self.base_jitter_enabled and self.base_jitter_std_px > 0):
            return tracks
        if self.base_jitter_prob <= 0:
            return tracks
        if tracks.dim() != 4:
            return tracks
        if query_t is None or not isinstance(query_t, torch.Tensor) or query_t.dim() != 2:
            return tracks

        B, N, T, _ = tracks.shape
        denom_h = float(max(int(height), 1))
        denom_w = float(max(int(width), 1))

        std_y = float(self.base_jitter_std_px) / denom_h
        std_x = float(self.base_jitter_std_px) / denom_w
        max_y = float(self.base_jitter_max_px) / denom_h if self.base_jitter_max_px > 0 else 0.0
        max_x = float(self.base_jitter_max_px) / denom_w if self.base_jitter_max_px > 0 else 0.0

        if self.base_jitter_mode in ("per_frame", "frame", "independent"):
            noise = torch.randn((B, N, T, 2), device=tracks.device, dtype=tracks.dtype)
        else:
            # Default: constant offset per point across time (more like drift/bias).
            noise = (
                torch.randn((B, N, 1, 2), device=tracks.device, dtype=tracks.dtype)
                .expand(B, N, T, 2)
                .clone()
            )

        scale = torch.tensor([std_y, std_x], device=tracks.device, dtype=tracks.dtype).view(1, 1, 1, 2)
        noise = noise * scale
        if max_y > 0:
            noise[..., 0].clamp_(-max_y, max_y)
        if max_x > 0:
            noise[..., 1].clamp_(-max_x, max_x)

        if self.base_jitter_exclude_query_frame:
            batch_idx = torch.arange(B, device=tracks.device).view(B, 1).expand(B, N)
            point_idx = torch.arange(N, device=tracks.device).view(1, N).expand(B, N)
            qt = query_t.clamp(0, T - 1)
            noise[batch_idx, point_idx, qt, :] = 0.0

        if self.base_jitter_prob < 1.0:
            p = float(self.base_jitter_prob)
            p = 0.0 if p < 0 else (1.0 if p > 1 else p)
            apply_mask = (torch.rand((B, N, 1, 1), device=tracks.device) < p).to(dtype=tracks.dtype)
            noise = noise * apply_mask

        return torch.clamp(tracks + noise, 0.0, 1.0)

    def _maybe_apply_base_occlusion_drift(
        self,
        base_tracks: torch.Tensor,
        base_visibility: torch.Tensor,
        query_t: torch.Tensor,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training-only augmentation for the Route-A "re-localization after long occlusion" story.

        We simulate a long occlusion run (visibility forced low) followed by a re-appearance,
        and simultaneously inject a *large* drift into the base tracks right after re-appearance.
        This prevents the trivial "base is already perfect" regime on synthetic data and provides
        a strong learning signal for the global re-localization step.
        """
        if not self.training:
            return base_tracks, base_visibility
        if not self.base_occ_drift_enabled:
            return base_tracks, base_visibility
        if (
            not isinstance(base_tracks, torch.Tensor)
            or not isinstance(base_visibility, torch.Tensor)
            or base_tracks.dim() != 4
            or base_visibility.dim() != 3
        ):
            return base_tracks, base_visibility
        if query_t is None or not isinstance(query_t, torch.Tensor) or query_t.dim() != 2:
            return base_tracks, base_visibility
        if self.base_occ_drift_prob <= 0:
            return base_tracks, base_visibility
        if self.base_occ_drift_std_px <= 0 and self.base_occ_drift_max_px <= 0:
            return base_tracks, base_visibility

        B, N, T, _ = base_tracks.shape
        if base_visibility.shape != (B, N, T):
            return base_tracks, base_visibility

        tracks = base_tracks.clone()
        vis = base_visibility.clone()

        denom_h = float(max(int(height), 1))
        denom_w = float(max(int(width), 1))
        std_y = float(self.base_occ_drift_std_px) / denom_h
        std_x = float(self.base_occ_drift_std_px) / denom_w
        max_y = float(self.base_occ_drift_max_px) / denom_h if self.base_occ_drift_max_px > 0 else 0.0
        max_x = float(self.base_occ_drift_max_px) / denom_w if self.base_occ_drift_max_px > 0 else 0.0

        apply = torch.rand((B, N), device=tracks.device) < float(self.base_occ_drift_prob)
        min_len = int(self.base_occ_drift_min_len)
        max_len = int(self.base_occ_drift_max_len)
        after_frames = int(self.base_occ_drift_after_frames)
        occ_val = float(self.base_occ_drift_occluded_value)
        vis_val = float(self.base_occ_drift_visible_value)

        for b in range(B):
            for n in range(N):
                if not bool(apply[b, n]):
                    continue
                tq = int(query_t[b, n].clamp(0, T - 1).item())
                if tq >= T - 3:
                    continue

                # Sample occlusion length, ensure there is at least 1 frame left for re-appearance.
                occ_len = int(torch.randint(min_len, max_len + 1, (1,), device=tracks.device).item())
                if occ_len < 1:
                    continue
                start_min = tq + 1
                start_max = T - occ_len - 2  # end <= T-2 => reappear at end+1 exists
                if start_max < start_min:
                    continue
                occ_start = int(torch.randint(start_min, start_max + 1, (1,), device=tracks.device).item())
                occ_end = occ_start + occ_len - 1
                if occ_end >= T - 1:
                    continue

                # Force a long occlusion run.
                vis[b, n, occ_start : occ_end + 1] = occ_val

                # Force re-appearance visibility so the heuristic trigger definitely fires.
                ra_start = occ_end + 1
                ra_end = min(occ_end + after_frames, T - 1)
                if self.base_occ_drift_force_visible_after:
                    vis[b, n, ra_start : ra_end + 1] = vis_val

                # Large drift right after re-appearance (to mimic re-lock failure).
                if self.base_occ_drift_mode in ("per_frame", "frame", "independent"):
                    noise = torch.randn((ra_end - ra_start + 1, 2), device=tracks.device, dtype=tracks.dtype)
                else:
                    noise = torch.randn((1, 2), device=tracks.device, dtype=tracks.dtype).expand(
                        ra_end - ra_start + 1, 2
                    )
                scale = torch.tensor([std_y, std_x], device=tracks.device, dtype=tracks.dtype).view(1, 2)
                noise = noise * scale
                if max_y > 0:
                    noise[:, 0].clamp_(-max_y, max_y)
                if max_x > 0:
                    noise[:, 1].clamp_(-max_x, max_x)
                tracks[b, n, ra_start : ra_end + 1, :] = tracks[b, n, ra_start : ra_end + 1, :] + noise

        # Optionally keep query-frame alignment clean (will also be enforced later).
        if self.base_occ_drift_exclude_query_frame:
            batch_idx = torch.arange(B, device=tracks.device).view(B, 1).expand(B, N)
            point_idx = torch.arange(N, device=tracks.device).view(1, N).expand(B, N)
            qt = query_t.clamp(0, T - 1)
            tracks[batch_idx, point_idx, qt, :] = base_tracks[batch_idx, point_idx, qt, :]
            vis[batch_idx, point_idx, qt] = base_visibility[batch_idx, point_idx, qt]

        tracks = torch.clamp(tracks, 0.0, 1.0)
        vis = vis.clamp(0.0, 1.0)
        return tracks, vis

    def _get_relocalization_feature_map(
        self,
        *,
        cotracker_fmaps: Optional[torch.Tensor],
        geo_backbone_features: Optional[torch.Tensor],
        geo_pyramid: Optional[Dict[str, torch.Tensor]],
        video: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not self.relocalization_enabled:
            return None
        if self.relocalization_feature_source == "cotracker":
            return cotracker_fmaps if isinstance(cotracker_fmaps, torch.Tensor) else None

        if self.relocalization_feature_source == "dino":
            return self._get_dino_feature_map(video)

        # geo features
        level = str(self.relocalization_feature_level).lower().strip()
        if level in ("c2", "c3", "c4", "c5") and isinstance(geo_pyramid, dict):
            fmap = geo_pyramid.get(level, None)
            if isinstance(fmap, torch.Tensor) and fmap.dim() == 5:
                return fmap
        if isinstance(geo_backbone_features, torch.Tensor) and geo_backbone_features.dim() == 5:
            return geo_backbone_features
        return None

    def _get_dino_feature_map(self, video: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Compute a channel-last DINOv2 feature map for each frame in the video.

        Args:
            video: (B, T, 3, H, W) uint8 or float tensor

        Returns:
            (B, T, H_feat, W_feat, D) normalized feature tensor, or None
        """
        if video is None:
            return None
        if self._dino_recovery_extractor is None:
            from models.recovery_features import DINORecoveryExtractor
            self._dino_recovery_extractor = DINORecoveryExtractor(
                weights_path=self._dino_recovery_weights
            )

        B, T, C, H, W = video.shape
        device = video.device

        # Process frame-by-frame to manage memory
        frames_flat = video.reshape(B * T, C, H, W)  # (B*T, 3, H, W)
        feat_list = []
        for i in range(B * T):
            frame_uint8 = frames_flat[i]  # (3, H, W) in [0,255]
            frame_hwc = frame_uint8.permute(1, 2, 0)  # (H, W, 3)
            feat = self._dino_recovery_extractor.feature_map(frame_hwc, device)  # (D, h, w)
            feat_list.append(feat)

        # Stack as channel-last to match the rest of the recovery path:
        # _sample_features() and _compute_global_relocalization_step() both
        # expect (B, T, H, W, C).
        feat_all = torch.stack(feat_list, dim=0).to(device=device, dtype=torch.float32)
        feat_all = feat_all.permute(0, 2, 3, 1).contiguous()  # (B*T, h, w, D)
        h, w, d = int(feat_all.shape[1]), int(feat_all.shape[2]), int(feat_all.shape[3])
        feat_all = F.normalize(feat_all, dim=-1)
        feat_all = feat_all.reshape(B, T, h, w, d)
        return feat_all

    def _ensure_learned_head(self, device: torch.device) -> bool:
        """Load the learned recovery head if not already loaded."""
        if self._learned_recovery_head is not None:
            return True
        if not self._use_learned_head:
            return False
        if not self._learned_head_checkpoint:
            logger.warning("use_learned_head=True but no checkpoint specified")
            return False
        ckpt_path = Path(self._learned_head_checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = _project_root() / ckpt_path
        if not ckpt_path.exists():
            logger.warning(f"Learned head checkpoint not found: {ckpt_path}")
            return False

        from models.online_recovery_head import OnlineRecoveryHead
        head = OnlineRecoveryHead(feat_dim=self._dino_out_dim, hidden_dim=128, num_layers=3)
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        head.load_state_dict(state, strict=False)
        self._learned_recovery_head = head.to(device).eval()
        for p in self._learned_recovery_head.parameters():
            p.requires_grad = False
        self._learned_head_loaded = True
        logger.info(f"Learned recovery head loaded from {ckpt_path}")
        return True

    def _apply_learned_recovery_head(
        self,
        *,
        feature_map: torch.Tensor,
        query_features: torch.Tensor,
        tracks: torch.Tensor,
        mask_nt: torch.Tensor,
        video: Optional[torch.Tensor] = None,
        base_visibility: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use the learned recovery head with fully aligned inputs.

        Aligned with offline training contract:
        1. Search crop: cv2.getRectSubPix centered on current track position
        2. Search feature map: DINOv2 on search crop, normalized per-channel
        3. Support descriptor: DINOv2 on support patches, mean-pooled
        4. Tracker vis: real visibility signal
        5. Local->global: exact crop geometry
        """
        import cv2 as _cv2

        B, N, T, _ = tracks.shape
        device = tracks.device
        step_out = torch.zeros_like(tracks)
        conf_out = torch.zeros_like(tracks[..., 0])

        if not mask_nt.any():
            return step_out, conf_out

        head = self._learned_recovery_head
        dino_ext = self._dino_recovery_extractor

        if video is None or dino_ext is None:
            return step_out, conf_out

        _, _, _, H_img, W_img = video.shape
        search_crop_size = 224

        for b in range(B):
            for t in range(T):
                idx = torch.nonzero(mask_nt[b, :, t], as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    continue

                # Get RGB frame as numpy for cv2 cropping
                frame_np = video[b, t].permute(1, 2, 0).cpu().numpy().astype(_np.uint8)

                for j in range(idx.numel()):
                    n_idx = idx[j].item()
                    current_pos = tracks[b, n_idx, t]
                    cx = current_pos[0].item() * W_img
                    cy = current_pos[1].item() * H_img

                    # --- Search crop via cv2.getRectSubPix (matches offline exactly) ---
                    search_crop = _cv2.getRectSubPix(frame_np, (search_crop_size, search_crop_size), (cx, cy))

                    # --- DINO feature map on search crop ---
                    search_t = torch.from_numpy(search_crop).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    search_t = F.interpolate(search_t, size=(dino_ext.img_size, dino_ext.img_size),
                                             mode="bilinear", align_corners=False).to(device)
                    with torch.no_grad():
                        sfm = dino_ext.model(search_t)[-1]
                        sfm = F.normalize(sfm.float(), dim=1)
                    search_fmap = sfm  # (1, D, h, w) on device

                    # --- Support descriptor from visible-bank features ---
                    if query_features.dim() == 4:
                        supp_desc = query_features[b, n_idx].mean(dim=0, keepdim=True)
                    else:
                        supp_desc = query_features[b, n_idx:n_idx+1]

                    # --- Tracker vis: use real base visibility at current frame ---
                    if base_visibility is not None:
                        vis_val = base_visibility[b, n_idx, t:t+1].float().to(device).unsqueeze(0)
                    else:
                        vis_val = torch.ones(1, 1, device=device, dtype=torch.float32)

                    # --- Run head ---
                    with torch.no_grad():
                        out = head(search_fmap, supp_desc, vis_val)
                    pred_xy_local = out["pred_xy"][0]

                    # --- Local -> Global: exact crop geometry ---
                    crop_frac_x = search_crop_size / float(W_img)
                    crop_frac_y = search_crop_size / float(H_img)
                    pred_global_x = cx / W_img + (pred_xy_local[0].item() - 0.5) * crop_frac_x
                    pred_global_y = cy / H_img + (pred_xy_local[1].item() - 0.5) * crop_frac_y
                    pred_global = torch.tensor(
                        [pred_global_x, pred_global_y], device=device, dtype=torch.float32
                    ).clamp(0.0, 1.0)

                    step_out[b, n_idx, t] = pred_global - current_pos
                    heatmap = out.get("heatmap", None)
                    if heatmap is not None:
                        conf_out[b, n_idx, t] = torch.sigmoid(heatmap).max().item()
                    else:
                        conf_out[b, n_idx, t] = 0.5

        return step_out, conf_out

    @staticmethod
    def _extract_crop_torch(
        image: torch.Tensor, center_xy: torch.Tensor, crop_size: int,
        H: int, W: int,
    ) -> torch.Tensor:
        """Extract a square crop using grid_sample (GPU-friendly)."""
        cx, cy = center_xy[0], center_xy[1]
        half = crop_size / 2.0

        gy = torch.linspace(cy.item() - half, cy.item() + half, crop_size, device=image.device)
        gx = torch.linspace(cx.item() - half, cx.item() + half, crop_size, device=image.device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")

        grid_x = 2.0 * grid_x / (W - 1) - 1.0
        grid_y = 2.0 * grid_y / (H - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        crop = F.grid_sample(
            image.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        return crop

    def _get_framewise_feature_map(
        self,
        *,
        cotracker_fmaps: Optional[torch.Tensor],
        geo_backbone_features: Optional[torch.Tensor],
        geo_pyramid: Optional[Dict[str, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        if not self.framewise_init_enabled:
            return None
        if self.framewise_init_feature_source == "cotracker":
            return cotracker_fmaps if isinstance(cotracker_fmaps, torch.Tensor) else None

        level = str(self.framewise_init_feature_level).lower().strip()
        if level in ("c2", "c3", "c4", "c5") and isinstance(geo_pyramid, dict):
            fmap = geo_pyramid.get(level, None)
            if isinstance(fmap, torch.Tensor) and fmap.dim() == 5:
                return fmap
        if isinstance(geo_backbone_features, torch.Tensor) and geo_backbone_features.dim() == 5:
            return geo_backbone_features
        return None

    def _build_focus_correction_mask(
        self,
        *,
        tracks: torch.Tensor,
        query_t: Optional[torch.Tensor],
        relocal_mask_nt: Optional[torch.Tensor],
        corr_conf_nt: Optional[torch.Tensor],
        relocal_conf_nt: Optional[torch.Tensor],
        base_visibility_nt: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Build a boolean (B,N,T) mask indicating where full-strength corrections are allowed.

        Outside this mask, corrections are attenuated by `focus_corrections_non_focus_scale`.
        This targets aggressive updates to hard/uncertain regions and protects easy regions.
        """
        if not self.focus_corrections_enabled:
            return None
        if not isinstance(tracks, torch.Tensor) or tracks.dim() != 4:
            return None

        mask: Optional[torch.Tensor] = None
        target_shape = tracks[..., 0].shape  # (B,N,T)

        def _accumulate_mask(src: Optional[torch.Tensor]) -> None:
            nonlocal mask
            if not isinstance(src, torch.Tensor):
                return
            if src.shape != target_shape:
                return
            src_bool = src.to(device=tracks.device).bool()
            mask = src_bool if mask is None else (mask | src_bool)

        if self.focus_corrections_use_relocal_mask:
            _accumulate_mask(relocal_mask_nt)

        if self.focus_corrections_use_corr_conf and isinstance(corr_conf_nt, torch.Tensor):
            thr = float(self.focus_corrections_corr_threshold)
            _accumulate_mask(
                (corr_conf_nt.to(device=tracks.device, dtype=tracks.dtype) >= thr)
            )

        if self.focus_corrections_use_relocal_conf and isinstance(relocal_conf_nt, torch.Tensor):
            thr = float(self.focus_corrections_relocal_threshold)
            _accumulate_mask(
                (relocal_conf_nt.to(device=tracks.device, dtype=tracks.dtype) >= thr)
            )

        if self.focus_corrections_include_occluded and isinstance(base_visibility_nt, torch.Tensor):
            thr = float(self.focus_corrections_occluded_threshold)
            _accumulate_mask(
                (base_visibility_nt.to(device=tracks.device, dtype=tracks.dtype) < thr)
            )

        if self.focus_corrections_use_long_occlusion_query and isinstance(base_visibility_nt, torch.Tensor):
            # Query-level mask: if the base visibility indicates a long occlusion
            # run (>= long_occlusion_len), allow full corrections for the entire
            # query trajectory.
            vis_thr = float(self.focus_corrections_long_occlusion_vis_threshold)
            occ = base_visibility_nt.to(device=tracks.device, dtype=tracks.dtype) < vis_thr
            if occ.shape == target_shape:
                B, N, T = occ.shape
                t_idx = torch.arange(T, device=tracks.device).view(1, 1, T)
                # Ignore occlusion before/at the query frame (matches TAP-Vid long-occ subset logic).
                if isinstance(query_t, torch.Tensor) and query_t.shape == (B, N):
                    occ = occ & (t_idx > query_t.to(device=tracks.device).unsqueeze(-1))

                # Optional: refine the occlusion mask using low correlation confidence.
                # This filters out false "occluded" spans where the tracker remains confident.
                if (
                    self.focus_corrections_long_occlusion_use_corr_conf_for_occ
                    and isinstance(corr_conf_nt, torch.Tensor)
                    and corr_conf_nt.shape == target_shape
                ):
                    thr = float(self.focus_corrections_long_occlusion_occ_corr_threshold)
                    low_conf = corr_conf_nt.to(device=tracks.device, dtype=tracks.dtype) < thr
                    occ = occ & low_conf

                # Optional smoothing: fill short "visible" gaps inside occlusions.
                # This addresses flickery base visibility that would otherwise
                # break long occlusion runs into shorter segments.
                fill = int(self.focus_corrections_long_occlusion_fill_gaps)
                if fill > 0 and T >= 3:
                    for _ in range(fill):
                        occ_filled = occ.clone()
                        occ_filled[..., 1:-1] |= occ[..., :-2] & occ[..., 2:]
                        occ = occ_filled

                window_before = int(self.focus_corrections_long_occlusion_apply_window_before_occlusion)
                mask_before = (
                    torch.zeros((B, N, T), device=tracks.device, dtype=torch.bool) if window_before > 0 else None
                )

                if self.focus_corrections_long_occlusion_mode == "reappearance":
                    # Re-appearance detection for query-level gating:
                    # - base visibility can fail to flip back to visible after GT re-appearance
                    # - corr/relocal confidence can provide a more direct cue that the point was re-acquired
                    reappear_src = str(self.focus_corrections_long_occlusion_reappear_source)
                    reappear_thr_raw = float(self.focus_corrections_long_occlusion_reappear_threshold)
                    base_vis_thr = float(self.focus_corrections_long_occlusion_vis_threshold)
                    if reappear_src == "visibility" and math.isfinite(reappear_thr_raw):
                        base_vis_thr = float(reappear_thr_raw)
                    base_reappear_signal = (
                        base_visibility_nt.to(device=tracks.device, dtype=tracks.dtype) >= base_vis_thr
                    )
                    reappear_signal: Optional[torch.Tensor] = None
                    if (
                        reappear_src == "corr_conf"
                        and isinstance(corr_conf_nt, torch.Tensor)
                        and corr_conf_nt.shape == target_shape
                    ):
                        thr = (
                            float(reappear_thr_raw)
                            if math.isfinite(reappear_thr_raw)
                            else float(self.focus_corrections_corr_threshold)
                        )
                        reappear_signal = (
                            corr_conf_nt.to(device=tracks.device, dtype=tracks.dtype) >= thr
                        )
                    elif (
                        reappear_src == "relocal_conf"
                        and isinstance(relocal_conf_nt, torch.Tensor)
                        and relocal_conf_nt.shape == target_shape
                    ):
                        thr = (
                            float(reappear_thr_raw)
                            if math.isfinite(reappear_thr_raw)
                            else float(self.focus_corrections_relocal_threshold)
                        )
                        reappear_signal = (
                            relocal_conf_nt.to(device=tracks.device, dtype=tracks.dtype) >= thr
                        )
                    else:
                        reappear_signal = base_reappear_signal

                    # Robustness: keep classic base-visibility as an OR fallback.
                    if (
                        reappear_src in ("corr_conf", "relocal_conf")
                        and isinstance(reappear_signal, torch.Tensor)
                        and reappear_signal.shape == base_reappear_signal.shape
                    ):
                        reappear_signal = reappear_signal | base_reappear_signal

                    # Mask to after-query to match TAP-Vid subset selection.
                    if isinstance(query_t, torch.Tensor) and query_t.shape == (B, N):
                        reappear_signal = reappear_signal & (
                            t_idx > query_t.to(device=tracks.device).unsqueeze(-1)
                        )

                    # Count the maximum occlusion run that is followed by a visible frame
                    # (i.e., a true re-appearance). If occluded until end-of-video, we do
                    # NOT count that tail run.
                    run = torch.zeros((B, N), device=tracks.device, dtype=torch.int32)
                    best = torch.zeros((B, N), device=tracks.device, dtype=torch.int32)
                    for t in range(T):
                        reappear = reappear_signal[..., t].bool()
                        best = torch.where(reappear, torch.maximum(best, run), best)
                        is_occ = occ[..., t] & (~reappear)
                        if mask_before is not None and window_before > 0 and int(self.focus_corrections_long_occlusion_len) > 0:
                            long_len = int(self.focus_corrections_long_occlusion_len)
                            reached = is_occ & (run == (long_len - 1))
                            if bool(reached.any()):
                                start = t - long_len + 1
                                for k in range(1, window_before + 1):
                                    idx = start - k
                                    if idx < 0:
                                        break
                                    mask_before[..., idx] |= reached
                        run = torch.where(is_occ, run + 1, torch.zeros_like(run))
                    max_run = best
                else:
                    # Max run length anywhere after the query frame (tail run counts).
                    run = torch.zeros((B, N), device=tracks.device, dtype=torch.int32)
                    max_run = torch.zeros((B, N), device=tracks.device, dtype=torch.int32)
                    for t in range(T):
                        is_occ = occ[..., t]
                        if mask_before is not None and window_before > 0 and int(self.focus_corrections_long_occlusion_len) > 0:
                            long_len = int(self.focus_corrections_long_occlusion_len)
                            reached = is_occ & (run == (long_len - 1))
                            if bool(reached.any()):
                                start = t - long_len + 1
                                for k in range(1, window_before + 1):
                                    idx = start - k
                                    if idx < 0:
                                        break
                                    mask_before[..., idx] |= reached
                        run = torch.where(is_occ, run + 1, torch.zeros_like(run))
                        max_run = torch.maximum(max_run, run)

                long_occ_query = max_run >= int(self.focus_corrections_long_occlusion_len)
                apply_mode = str(self.focus_corrections_long_occlusion_apply_mode)
                window = int(self.focus_corrections_long_occlusion_apply_window_after_reappear)

                if apply_mode == "query":
                    _accumulate_mask(long_occ_query.unsqueeze(-1).expand(-1, -1, T))
                else:
                    mask_long = torch.zeros((B, N, T), device=tracks.device, dtype=torch.bool)

                    if apply_mode in ("occluded", "occluded_or_after_reappear"):
                        # Apply only on frames where the *base* is occluded.
                        # This avoids perturbing easy/visible frames while still
                        # covering hard regions where CoTracker typically fails.
                        mask_long |= long_occ_query.unsqueeze(-1) & occ

                    if apply_mode in ("after_reappear", "occluded_or_after_reappear"):
                        # Apply after re-appearance following a long occlusion run.
                        # NOTE: This requires base_visibility to actually flip back
                        # to visible; if the base remains "occluded" even after
                        # GT re-appearance, the occluded mask above is the fallback.
                        if window == 0:
                            # If window is 0, interpret as "to end" when using after-reappear modes.
                            window = -1
                        # Select the re-appearance signal. We default to base visibility,
                        # but allow corr/relocal confidence to trigger re-appearance even
                        # when base visibility stays low.
                        reappear_src = str(self.focus_corrections_long_occlusion_reappear_source)
                        reappear_thr_raw = float(self.focus_corrections_long_occlusion_reappear_threshold)
                        reappear_signal: Optional[torch.Tensor] = None  # (B,N,T) bool
                        base_reappear_signal: Optional[torch.Tensor] = None  # (B,N,T) bool
                        if isinstance(base_visibility_nt, torch.Tensor) and base_visibility_nt.shape == target_shape:
                            # We always compute a base-visibility reappearance signal so we can
                            # (optionally) fall back to it when corr/relocal confidence is low.
                            vis_thr = (
                                reappear_thr_raw
                                if (reappear_src == "visibility" and math.isfinite(reappear_thr_raw))
                                else float(self.focus_corrections_long_occlusion_vis_threshold)
                            )
                            base_reappear_signal = (
                                base_visibility_nt.to(device=tracks.device, dtype=tracks.dtype) >= float(vis_thr)
                            )
                        if (
                            reappear_src == "corr_conf"
                            and isinstance(corr_conf_nt, torch.Tensor)
                            and corr_conf_nt.shape == target_shape
                        ):
                            thr = (
                                reappear_thr_raw
                                if math.isfinite(reappear_thr_raw)
                                else float(self.focus_corrections_corr_threshold)
                            )
                            reappear_signal = (
                                corr_conf_nt.to(device=tracks.device, dtype=tracks.dtype) >= float(thr)
                            )
                        elif (
                            reappear_src == "relocal_conf"
                            and isinstance(relocal_conf_nt, torch.Tensor)
                            and relocal_conf_nt.shape == target_shape
                        ):
                            thr = (
                                reappear_thr_raw
                                if math.isfinite(reappear_thr_raw)
                                else float(self.focus_corrections_relocal_threshold)
                            )
                            reappear_signal = (
                                relocal_conf_nt.to(device=tracks.device, dtype=tracks.dtype) >= float(thr)
                            )
                        elif reappear_src == "visibility":
                            reappear_signal = base_reappear_signal

                        if reappear_signal is None:
                            reappear_signal = base_reappear_signal

                        # Robustness: when using corr/relocal confidence, keep the classic
                        # visibility-based trigger as an OR fallback. This prevents empty
                        # masks on videos where base visibility is already reliable.
                        if (
                            reappear_src in ("corr_conf", "relocal_conf")
                            and isinstance(base_reappear_signal, torch.Tensor)
                            and isinstance(reappear_signal, torch.Tensor)
                            and base_reappear_signal.shape == reappear_signal.shape
                        ):
                            reappear_signal = reappear_signal | base_reappear_signal

                        if reappear_signal is None:
                            # No usable re-appearance signal; leave the after-reappear mask empty.
                            reappear_signal = torch.zeros(
                                (B, N, T), device=tracks.device, dtype=torch.bool
                            )
                        # Ensure we only react to re-appearance after the query frame.
                        if isinstance(query_t, torch.Tensor) and query_t.shape == (B, N):
                            t_idx = torch.arange(T, device=tracks.device).view(1, 1, T)
                            reappear_signal = reappear_signal & (
                                t_idx > query_t.to(device=tracks.device).unsqueeze(-1)
                            )
                        mask_after = torch.zeros((B, N, T), device=tracks.device, dtype=torch.bool)
                        long_len = int(self.focus_corrections_long_occlusion_len)
                        run = torch.zeros((B, N), device=tracks.device, dtype=torch.int32)
                        for t in range(T):
                            is_occ = occ[..., t]
                            reappear = reappear_signal[..., t]
                            long_end = reappear & (run >= long_len)
                            if long_occ_query is not None:
                                long_end = long_end & long_occ_query
                            if bool(long_end.any()):
                                if window < 0:
                                    mask_after[..., t:] |= long_end.unsqueeze(-1)
                                else:
                                    t_end = min(T, t + window)
                                    mask_after[..., t:t_end] |= long_end.unsqueeze(-1)
                            run = torch.where(is_occ, run + 1, torch.zeros_like(run))
                        mask_long |= mask_after

                    if (
                        mask_before is not None
                        and window_before > 0
                        and isinstance(query_t, torch.Tensor)
                        and query_t.shape == (B, N)
                    ):
                        # Ensure we only react to frames strictly after the query frame.
                        t_idx2 = torch.arange(T, device=tracks.device).view(1, 1, T)
                        mask_before = mask_before & (
                            t_idx2 > query_t.to(device=tracks.device).unsqueeze(-1)
                        )

                    if mask_before is not None and window_before > 0:
                        mask_long |= long_occ_query.unsqueeze(-1) & mask_before

                    _accumulate_mask(mask_long)

        # Important: if focusing is enabled but no criterion fired, we still want to
        # apply attenuation everywhere (i.e., "nothing is focused"). Returning None
        # would disable focus_corrections entirely and can unintentionally apply
        # full-strength corrections on videos/queries without triggers.
        if mask is None:
            return torch.zeros(target_shape, device=tracks.device, dtype=torch.bool)

        return mask

    def _compute_relocalization_residual_step(
        self,
        *,
        temporal_features: torch.Tensor,  # (B,T,N,C)
        relocal_mask_nt: Optional[torch.Tensor],  # (B,N,T) bool
        relocal_conf_nt: Optional[torch.Tensor],  # (B,N,T) float
    ) -> Optional[torch.Tensor]:
        """
        Predict an additional residual correction dedicated to long-occlusion frames.

        Returns:
            step_nt: (B,N,T,2) normalized [dy,dx], or None when disabled.
        """
        if not self.relocalization_residual_enabled:
            return None
        if (
            not isinstance(temporal_features, torch.Tensor)
            or temporal_features.dim() != 4
            or not isinstance(self.relocalization_residual_delta_head, nn.Module)
            or not isinstance(self.relocalization_residual_gate_head, nn.Module)
        ):
            return None

        B, T, N, _ = temporal_features.shape
        delta_bt = self.relocalization_residual_delta_head(temporal_features)  # (B,T,N,2)
        if self.relocalization_residual_tanh:
            delta_bt = torch.tanh(delta_bt)
        gate_bt = torch.sigmoid(self.relocalization_residual_gate_head(temporal_features).squeeze(-1))  # (B,T,N)
        step_bt = delta_bt * gate_bt.unsqueeze(-1) * float(self.relocalization_residual_scale)
        step_nt = step_bt.permute(0, 2, 1, 3).contiguous()  # (B,N,T,2)

        if isinstance(relocal_mask_nt, torch.Tensor) and relocal_mask_nt.shape == (B, N, T):
            mask = relocal_mask_nt.to(device=step_nt.device, dtype=step_nt.dtype).unsqueeze(-1)
            step_nt = step_nt * mask
        elif self.relocalization_residual_require_mask:
            return torch.zeros_like(step_nt)

        if (
            self.relocalization_residual_conf_threshold > 0
            and isinstance(relocal_conf_nt, torch.Tensor)
            and relocal_conf_nt.shape == (B, N, T)
        ):
            conf_gate = _confidence_gate(
                relocal_conf_nt.to(device=step_nt.device, dtype=step_nt.dtype),
                threshold=float(self.relocalization_residual_conf_threshold),
                power=float(self.relocalization_residual_conf_power),
            )
            step_nt = step_nt * conf_gate.unsqueeze(-1)

        if self.relocalization_residual_max_step > 0:
            step_nt = torch.clamp(
                step_nt,
                -float(self.relocalization_residual_max_step),
                float(self.relocalization_residual_max_step),
            )
        return step_nt

    def _compute_global_relocalization_step(
        self,
        *,
        feature_map: torch.Tensor,  # (B,T,H,W,C)
        query_features: torch.Tensor,  # (B,N,C) or (B,N,K,C)
        tracks: torch.Tensor,  # (B,N,T,2)
        mask_nt: torch.Tensor,  # (B,N,T) bool
        downsample: Optional[int] = None,
        feature_source: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Global matching step for re-localization after long occlusion.

        This computes a full-frame correlation between per-point query features and
        a feature map at each time step, but *only applies* (and therefore only
        computes) results for masked points/frames.
        """
        if (
            self.local_correlation is None
            or not isinstance(feature_map, torch.Tensor)
            or not isinstance(query_features, torch.Tensor)
            or not isinstance(tracks, torch.Tensor)
            or not isinstance(mask_nt, torch.Tensor)
        ):
            raise ValueError("Invalid inputs to _compute_global_relocalization_step.")

        if feature_map.dim() != 5:
            raise ValueError(f"feature_map must be (B,T,H,W,C), got {tuple(feature_map.shape)}")
        if tracks.dim() != 4 or tracks.shape[-1] != 2:
            raise ValueError(f"tracks must be (B,N,T,2), got {tuple(tracks.shape)}")
        if mask_nt.shape[:3] != tracks.shape[:3]:
            raise ValueError(
                f"mask_nt must be (B,N,T), got {tuple(mask_nt.shape)} vs tracks {tuple(tracks.shape)}"
            )

        B, N, T, _ = tracks.shape
        if query_features.dim() not in (3, 4):
            raise ValueError(
                "query_features must be (B,N,C) or (B,N,K,C), got "
                f"{tuple(query_features.shape)}"
            )
        if int(query_features.shape[0]) != int(B) or int(query_features.shape[1]) != int(N):
            raise ValueError(
                f"query_features leading dims must match (B,N)=({B},{N}), got {tuple(query_features.shape[:2])}"
            )
        if query_features.dim() == 4 and int(query_features.shape[2]) < 1:
            raise ValueError(f"query_features bank size must be >=1, got {int(query_features.shape[2])}")

        if not mask_nt.any():
            step = torch.zeros_like(tracks)
            conf = torch.zeros_like(tracks[..., 0])
            return step, conf

        corr_mod = self.local_correlation
        source = str(feature_source or "").lower().strip()
        use_dino_cosine = source == "dino"

        if use_dino_cosine:
            if int(feature_map.shape[-1]) != int(query_features.shape[-1]):
                step = torch.zeros_like(tracks)
                conf = torch.zeros_like(tracks[..., 0])
                return step, conf
        else:
            if int(feature_map.shape[-1]) != int(getattr(corr_mod, "in_dim", feature_map.shape[-1])):
                step = torch.zeros_like(tracks)
                conf = torch.zeros_like(tracks[..., 0])
                return step, conf

        H, W = int(feature_map.shape[2]), int(feature_map.shape[3])
        fmap = feature_map.reshape(B * T, H, W, feature_map.shape[-1]).permute(0, 3, 1, 2)  # (B*T,C,H,W)
        if use_dino_cosine:
            fmap = F.normalize(fmap.float(), dim=1)
        else:
            fmap = corr_mod.map_proj(fmap)  # (B*T,corr_dim,H,W)
            if corr_mod.normalize:
                fmap = F.normalize(fmap, dim=1)

        ds = int(self.relocalization_downsample if downsample is None else downsample)
        if ds > 1 and fmap.shape[-2] >= ds and fmap.shape[-1] >= ds:
            fmap = F.avg_pool2d(fmap, kernel_size=ds, stride=ds)
        Hf, Wf = int(fmap.shape[-2]), int(fmap.shape[-1])

        if use_dino_cosine:
            q = F.normalize(query_features.float(), dim=-1)
        else:
            q = corr_mod.query_proj(query_features)  # (B,N,corr_dim) or (B,N,K,corr_dim)
            if corr_mod.normalize:
                q = F.normalize(q, dim=-1)

        # Precompute (y,x) coords for soft-argmax.
        # Use float32 for the global softmax/soft-argmax for stability (HW can be large).
        coords_y = torch.arange(Hf, device=fmap.device, dtype=torch.float32)
        coords_x = torch.arange(Wf, device=fmap.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords_y, coords_x, indexing="ij")
        coords = torch.stack([yy, xx], dim=-1).reshape(Hf * Wf, 2)  # (HW,2)
        denom = torch.tensor([float(Hf), float(Wf)], device=fmap.device, dtype=torch.float32).view(1, 2)

        step_out = torch.zeros_like(tracks)  # (B,N,T,2)
        pred_abs_out = torch.zeros_like(tracks)  # (B,N,T,2) predicted abs position (y,x) in [0,1]
        conf_out = torch.zeros_like(tracks[..., 0])  # (B,N,T)
        margin_out = torch.zeros_like(tracks[..., 0])  # (B,N,T)
        shift_out = torch.zeros_like(tracks[..., 0])  # (B,N,T)

        mask_bt = mask_nt.permute(0, 2, 1).contiguous()  # (B,T,N)
        temp = max(float(self.relocalization_temperature), 1.0e-6)
        corr_scale = 1.0
        if (not use_dino_cosine) and (not corr_mod.normalize):
            corr_scale = 1.0 / math.sqrt(float(getattr(corr_mod, "corr_dim", q.shape[-1])))
        topk_relocal = max(1, int(self.relocalization_topk))
        temporal_sigma = float(self.relocalization_temporal_sigma)
        temporal_power = float(self.relocalization_temporal_power)
        temporal_min_prev_conf = float(self.relocalization_temporal_min_prev_conf)
        export_debug = getattr(self, "_export_candidates_debug", False)
        if export_debug:
            # Clear stale state first; this helper can be called multiple times per forward.
            self._relocal_debug_topk_positions = None
            self._relocal_debug_topk_scores = None
            self._relocal_debug_topk_valid = None
            self._relocal_debug_topk_kind = None

        debug_topk_positions = None
        debug_topk_scores = None
        debug_topk_valid = None
        debug_topk_kind = None
        if export_debug:
            if q.dim() == 4:
                # In template-bank mode, each bank element produces one candidate.
                debug_topk_k = int(q.shape[2])
                debug_topk_kind = "template_bank"
            else:
                # In single-descriptor mode, export the spatial top-k peaks when enabled.
                debug_topk_k = min(topk_relocal, int(Hf * Wf))
                debug_topk_kind = "spatial_topk" if debug_topk_k > 1 else "top1"

            debug_topk_positions = torch.full(
                (B, N, T, debug_topk_k, 2),
                float("nan"),
                device=step_out.device,
                dtype=step_out.dtype,
            )
            debug_topk_scores = torch.full(
                (B, N, T, debug_topk_k),
                float("nan"),
                device=conf_out.device,
                dtype=conf_out.dtype,
            )
            debug_topk_valid = torch.zeros(
                (B, N, T, debug_topk_k),
                device=step_out.device,
                dtype=torch.bool,
            )

        for b in range(B):
            q_b = q[b]  # (N,corr_dim) or (N,K,corr_dim)
            prev_pred_b = torch.zeros((N, 2), device=step_out.device, dtype=step_out.dtype)
            prev_conf_b = torch.zeros((N,), device=step_out.device, dtype=step_out.dtype)
            prev_valid_b = torch.zeros((N,), device=step_out.device, dtype=torch.bool)
            for t in range(T):
                idx = torch.nonzero(mask_bt[b, t], as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    prev_valid_b.zero_()
                    continue
                fmap_bt = fmap[b * T + t].reshape(fmap.shape[1], Hf * Wf)  # (corr_dim,HW)
                if q_b.dim() == 2:
                    q_sel = q_b[idx]  # (M,corr_dim)
                    corr = torch.matmul(q_sel, fmap_bt) * float(corr_scale)  # (M,HW)
                    corr = (corr / temp).float()
                    prob = torch.softmax(corr, dim=-1)  # (M,HW) float32

                    top2 = torch.topk(prob, k=2, dim=-1).values  # (M,2)
                    max_prob = top2[:, 0]
                    second_prob = top2[:, 1]

                    # How much better the selected match is compared to the current estimate.
                    tracks_bt = tracks[b, idx, t, :].to(device=prob.device, dtype=torch.float32)  # (M,2) y,x in [0,1]
                    y_idx = (tracks_bt[:, 0] * float(Hf)).clamp(0.0, float(Hf - 1)).to(dtype=torch.long)
                    x_idx = (tracks_bt[:, 1] * float(Wf)).clamp(0.0, float(Wf - 1)).to(dtype=torch.long)
                    center_lin = y_idx * int(Wf) + x_idx
                    center_prob = prob[torch.arange(idx.numel(), device=prob.device), center_lin]

                    M = int(idx.numel())
                    use_topk = min(topk_relocal, int(prob.shape[-1])) > 1
                    if use_topk:
                        k_sel = min(topk_relocal, int(prob.shape[-1]))
                        cand_prob, cand_idx = torch.topk(prob, k=k_sel, dim=-1)  # (M,k)
                        cand_coords = coords[cand_idx]  # (M,k,2)
                        cand_norm = (cand_coords / denom.view(1, 1, 2)).to(device=step_out.device, dtype=step_out.dtype)
                        score = cand_prob

                        if temporal_sigma > 0 and prev_valid_b.any():
                            prev_valid = prev_valid_b[idx]
                            if temporal_min_prev_conf > 0:
                                prev_valid = prev_valid & (prev_conf_b[idx] >= temporal_min_prev_conf)
                            if prev_valid.any():
                                prev_points = prev_pred_b[idx].to(device=step_out.device, dtype=step_out.dtype)  # (M,2)
                                dist = torch.norm(cand_norm - prev_points.unsqueeze(1), dim=-1)  # (M,k)
                                temporal_gate = torch.exp(-torch.square(dist / float(temporal_sigma))).to(
                                    device=score.device, dtype=score.dtype
                                )
                                if temporal_power != 1.0:
                                    temporal_gate = temporal_gate ** float(temporal_power)
                                score = torch.where(
                                    prev_valid.unsqueeze(-1),
                                    score * temporal_gate,
                                    score,
                                )

                        if self.relocalization_candidate_fusion_enabled and k_sel > 1:
                            fusion_temp = float(self.relocalization_candidate_fusion_temperature)
                            weights_raw = torch.softmax(score / fusion_temp, dim=-1)
                            weights = weights_raw
                            min_weight = float(self.relocalization_candidate_fusion_min_weight)
                            if min_weight > 0:
                                weights_masked = torch.where(
                                    weights_raw >= min_weight,
                                    weights_raw,
                                    torch.zeros_like(weights_raw),
                                )
                                weights_sum = weights_masked.sum(dim=-1, keepdim=True)
                                weights = torch.where(
                                    weights_sum > 0,
                                    weights_masked / weights_sum.clamp_min(1.0e-6),
                                    weights_raw,
                                )
                            pred_norm = (cand_norm * weights.unsqueeze(-1)).sum(dim=1)
                            conf_selected = (cand_prob * weights).sum(dim=-1)
                            top2_score = torch.topk(score, k=min(2, k_sel), dim=-1).values
                            margin = (top2_score[:, 0] - top2_score[:, 1]).clamp(min=0.0) if int(top2_score.shape[-1]) > 1 else top2_score[:, 0]
                            shift = (conf_selected - center_prob).clamp(min=0.0)
                        else:
                            best_j = score.argmax(dim=-1)  # (M,)
                            ar = torch.arange(M, device=best_j.device)
                            best_prob = cand_prob[ar, best_j]  # (M,)
                            best_pred = cand_norm[ar, best_j, :]  # (M,2) normalized y/x
                            margin = (best_prob - second_prob).clamp(min=0.0)
                            shift = (best_prob - center_prob).clamp(min=0.0)
                            pred_norm = best_pred
                            conf_selected = best_prob
                    else:
                        if self.relocalization_step_mode == "argmax":
                            max_idx = prob.argmax(dim=-1)  # (M,)
                            expected = coords[max_idx]  # (M,2) y,x
                        else:
                            expected = torch.matmul(prob, coords)  # (M,2) y,x
                        pred_norm = expected / denom  # (M,2) normalized y/x
                        pred_norm = pred_norm.to(device=step_out.device, dtype=step_out.dtype)
                        margin = (max_prob - second_prob).clamp(min=0.0)
                        shift = (max_prob - center_prob).clamp(min=0.0)
                        conf_selected = max_prob

                    if export_debug and debug_topk_positions is not None and debug_topk_scores is not None and debug_topk_valid is not None:
                        if use_topk:
                            cand_pos_debug = cand_norm
                            cand_score_debug = cand_prob.to(device=debug_topk_scores.device, dtype=debug_topk_scores.dtype)
                        else:
                            cand_pos_debug = pred_norm.unsqueeze(1)
                            cand_score_debug = conf_selected.unsqueeze(-1).to(
                                device=debug_topk_scores.device,
                                dtype=debug_topk_scores.dtype,
                            )
                        k_dbg = int(cand_pos_debug.shape[1])
                        debug_topk_positions[b, idx, t, :k_dbg, :] = cand_pos_debug
                        debug_topk_scores[b, idx, t, :k_dbg] = cand_score_debug
                        debug_topk_valid[b, idx, t, :k_dbg] = True

                    conf_out[b, idx, t] = conf_selected.to(device=conf_out.device, dtype=conf_out.dtype)
                    margin_out[b, idx, t] = margin.to(device=margin_out.device, dtype=margin_out.dtype)
                    shift_out[b, idx, t] = shift.to(device=shift_out.device, dtype=shift_out.dtype)
                    pred_abs_out[b, idx, t, :] = pred_norm
                    step_out[b, idx, t, :] = pred_norm - tracks[b, idx, t, :]
                else:
                    q_sel = q_b[idx]  # (M,K,corr_dim)
                    M, K, corr_dim = int(q_sel.shape[0]), int(q_sel.shape[1]), int(q_sel.shape[2])
                    q_flat = q_sel.reshape(M * K, corr_dim)  # (M*K,corr_dim)
                    corr = torch.matmul(q_flat, fmap_bt) * float(corr_scale)  # (M*K,HW)
                    corr = (corr / temp).float()
                    prob = torch.softmax(corr, dim=-1)  # (M*K,HW)

                    top2 = torch.topk(prob, k=2, dim=-1).values  # (M*K,2)
                    max_prob = top2[:, 0].view(M, K)
                    second_prob = top2[:, 1].view(M, K)
                    margin = (max_prob - second_prob).clamp(min=0.0)  # (M,K)

                    tracks_bt = tracks[b, idx, t, :].to(device=prob.device, dtype=torch.float32)  # (M,2)
                    tracks_rep = tracks_bt.repeat_interleave(K, dim=0)  # (M*K,2)
                    y_idx = (tracks_rep[:, 0] * float(Hf)).clamp(0.0, float(Hf - 1)).to(dtype=torch.long)
                    x_idx = (tracks_rep[:, 1] * float(Wf)).clamp(0.0, float(Wf - 1)).to(dtype=torch.long)
                    center_lin = y_idx * int(Wf) + x_idx  # (M*K,)
                    center_prob = prob[torch.arange(M * K, device=prob.device), center_lin]
                    shift = (top2[:, 0] - center_prob).clamp(min=0.0).view(M, K)  # (M,K)

                    if self.relocalization_step_mode == "argmax":
                        max_idx = prob.argmax(dim=-1)  # (M*K,)
                        expected = coords[max_idx]  # (M*K,2)
                    else:
                        expected = torch.matmul(prob, coords)  # (M*K,2)
                    pred_norm_all = (expected / denom).to(device=step_out.device, dtype=step_out.dtype).view(M, K, 2)

                    # Pick the best template per point (prefer confident + unambiguous + beats current).
                    score = max_prob
                    if self.relocalization_margin_threshold > 0:
                        score = score * _confidence_gate(
                            margin,
                            threshold=float(self.relocalization_margin_threshold),
                            power=float(self.relocalization_margin_power),
                        ).to(device=score.device, dtype=score.dtype)
                    if self.relocalization_shift_threshold > 0:
                        score = score * _confidence_gate(
                            shift,
                            threshold=float(self.relocalization_shift_threshold),
                            power=float(self.relocalization_shift_power),
                        ).to(device=score.device, dtype=score.dtype)

                    if temporal_sigma > 0 and prev_valid_b.any():
                        prev_valid = prev_valid_b[idx]
                        if temporal_min_prev_conf > 0:
                            prev_valid = prev_valid & (prev_conf_b[idx] >= temporal_min_prev_conf)
                        if prev_valid.any():
                            prev_points = prev_pred_b[idx].to(
                                device=pred_norm_all.device, dtype=pred_norm_all.dtype
                            )  # (M,2)
                            dist = torch.norm(pred_norm_all - prev_points.unsqueeze(1), dim=-1)  # (M,K)
                            temporal_gate = torch.exp(-torch.square(dist / float(temporal_sigma))).to(
                                device=score.device, dtype=score.dtype
                            )
                            if temporal_power != 1.0:
                                temporal_gate = temporal_gate ** float(temporal_power)
                            score = torch.where(
                                prev_valid.unsqueeze(-1),
                                score * temporal_gate,
                                score,
                            )

                    if self.relocalization_candidate_fusion_enabled and K > 1:
                        fusion_temp = float(self.relocalization_candidate_fusion_temperature)
                        weights_raw = torch.softmax(score / fusion_temp, dim=-1)
                        weights = weights_raw
                        min_weight = float(self.relocalization_candidate_fusion_min_weight)
                        if min_weight > 0:
                            weights_masked = torch.where(
                                weights_raw >= min_weight,
                                weights_raw,
                                torch.zeros_like(weights_raw),
                            )
                            weights_sum = weights_masked.sum(dim=-1, keepdim=True)
                            weights = torch.where(
                                weights_sum > 0,
                                weights_masked / weights_sum.clamp_min(1.0e-6),
                                weights_raw,
                            )
                        best_prob = (max_prob * weights).sum(dim=-1)
                        best_margin = (margin * weights).sum(dim=-1)
                        best_shift = (shift * weights).sum(dim=-1)
                        best_pred = (pred_norm_all * weights.unsqueeze(-1)).sum(dim=1)
                    else:
                        best_k = score.argmax(dim=-1)  # (M,)
                        ar = torch.arange(M, device=best_k.device)
                        best_prob = max_prob[ar, best_k]
                        best_margin = margin[ar, best_k]
                        best_shift = shift[ar, best_k]
                        best_pred = pred_norm_all[ar, best_k, :]  # (M,2)

                    if export_debug and debug_topk_positions is not None and debug_topk_scores is not None and debug_topk_valid is not None:
                        debug_topk_positions[b, idx, t, :K, :] = pred_norm_all
                        debug_topk_scores[b, idx, t, :K] = max_prob.to(
                            device=debug_topk_scores.device,
                            dtype=debug_topk_scores.dtype,
                        )
                        debug_topk_valid[b, idx, t, :K] = True

                    conf_out[b, idx, t] = best_prob.to(device=conf_out.device, dtype=conf_out.dtype)
                    margin_out[b, idx, t] = best_margin.to(device=margin_out.device, dtype=margin_out.dtype)
                    shift_out[b, idx, t] = best_shift.to(device=shift_out.device, dtype=shift_out.dtype)
                    pred_abs_out[b, idx, t, :] = best_pred
                    step_out[b, idx, t, :] = best_pred - tracks[b, idx, t, :]

                # Keep only immediate previous-frame predictions for temporal consensus.
                next_prev_valid = torch.zeros_like(prev_valid_b)
                next_prev_valid[idx] = True
                prev_valid_b = next_prev_valid
                prev_pred_b[idx] = pred_abs_out[b, idx, t, :].to(
                    device=prev_pred_b.device, dtype=prev_pred_b.dtype
                )
                prev_conf_b[idx] = conf_out[b, idx, t].to(device=prev_conf_b.device, dtype=prev_conf_b.dtype)

        # Apply confidence gate (ramp) and optional per-step clamp.
        gate = _confidence_gate(
            conf_out,
            threshold=float(self.relocalization_conf_threshold),
            power=float(self.relocalization_gate_power),
        ).to(device=step_out.device, dtype=step_out.dtype)
        if self.relocalization_margin_threshold > 0:
            gate = gate * _confidence_gate(
                margin_out,
                threshold=float(self.relocalization_margin_threshold),
                power=float(self.relocalization_margin_power),
            ).to(device=step_out.device, dtype=step_out.dtype)
        if self.relocalization_shift_threshold > 0:
            gate = gate * _confidence_gate(
                shift_out,
                threshold=float(self.relocalization_shift_threshold),
                power=float(self.relocalization_shift_power),
            ).to(device=step_out.device, dtype=step_out.dtype)

        # Optional: require two-frame consistency before allowing a global jump.
        if self.relocalization_confirm_enabled and float(self.relocalization_confirm_dist_threshold) > 0 and T > 1:
            thr = float(self.relocalization_confirm_dist_threshold)
            # Compare the *correction step* across consecutive frames. This is
            # more tolerant to actual object motion than comparing absolute
            # positions, while still filtering spurious one-off peaks.
            step_raw = step_out.to(device=gate.device, dtype=torch.float32)  # (B,N,T,2) ungated
            dist = torch.norm(step_raw[:, :, 1:, :] - step_raw[:, :, :-1, :], dim=-1)  # (B,N,T-1)
            pair_mask = mask_nt[:, :, 1:] & mask_nt[:, :, :-1]
            consistent = (dist <= thr) & pair_mask
            confirm = torch.zeros_like(mask_nt)
            confirm[:, :, 1:] |= consistent
            confirm[:, :, :-1] |= consistent
            if self.relocalization_confirm_allow_single_frame:
                single = (mask_nt.sum(dim=-1) == 1).unsqueeze(-1)
                confirm = confirm | (single & mask_nt)
            gate = gate * confirm.to(device=gate.device, dtype=gate.dtype)
        # Capture raw step before gate for debug export
        step_raw_before_gate = step_out.detach().clone()

        step_out = step_out * gate.unsqueeze(-1)
        step_out = step_out * mask_nt.to(device=step_out.device, dtype=step_out.dtype).unsqueeze(-1)

        # Store debug data on instance for later retrieval
        if export_debug:
            self._relocal_debug_topk_positions = (
                debug_topk_positions.detach().cpu() if isinstance(debug_topk_positions, torch.Tensor) else None
            )
            self._relocal_debug_topk_scores = (
                debug_topk_scores.detach().cpu() if isinstance(debug_topk_scores, torch.Tensor) else None
            )
            self._relocal_debug_topk_valid = (
                debug_topk_valid.detach().cpu() if isinstance(debug_topk_valid, torch.Tensor) else None
            )
            self._relocal_debug_topk_kind = debug_topk_kind
            self._relocal_debug_data = {
                "relocal_step_raw": step_raw_before_gate,
                "relocal_step_gated": step_out.detach(),
                "relocal_gate_value": gate.detach(),
                "relocal_conf_out": conf_out.detach(),
                "relocal_margin_out": margin_out.detach(),
                "relocal_pred_abs": pred_abs_out.detach(),
                "relocal_topk_positions": self._relocal_debug_topk_positions,
                "relocal_topk_scores": self._relocal_debug_topk_scores,
                "relocal_topk_valid": self._relocal_debug_topk_valid,
                "relocal_topk_kind": self._relocal_debug_topk_kind,
            }

        if self.relocalization_max_step > 0:
            step_out = torch.clamp(step_out, -float(self.relocalization_max_step), float(self.relocalization_max_step))

        return step_out, conf_out

    def train(self, mode: bool = True):
        super().train(mode)
        # Base tracker is always frozen/inference-only.
        self.base_tracker.eval()
        for param in self.base_tracker.parameters():
            param.requires_grad = False

        # Keep explicitly-frozen submodules in eval to avoid BN/stat updates.
        for name in list(self._frozen_submodules):
            module = getattr(self, name, None)
            if isinstance(module, nn.Module):
                module.eval()
        return self

    def _apply_trainability_from_config(self, refiner_cfg) -> None:
        """Optionally freeze parts of the refiner via config.model.refiner.*."""
        if refiner_cfg is None:
            return

        trainable_raw = getattr(refiner_cfg, "trainable_modules", None)
        freeze_raw = getattr(refiner_cfg, "freeze_modules", None)

        def _as_name_list(value) -> Optional[List[str]]:
            if value is None:
                return None
            if isinstance(value, str):
                return [value]
            try:
                return [str(v) for v in list(value)]
            except Exception:
                return None

        trainable_list = _as_name_list(trainable_raw)
        freeze_list = _as_name_list(freeze_raw)

        candidates: Dict[str, Optional[nn.Module]] = {
            "geo_backbone": self.geo_backbone,
            "semantic_encoder": self.semantic_encoder,
            "freq_semantic": self.freq_semantic,
            "temporal_transformer": self.temporal_transformer,
            "residual_decoder": self.residual_decoder,
            "relocalization_residual": self.relocalization_residual_module,
            "relocal_acceptor": self.relocal_acceptor_head,
            "verifier": self.relocal_acceptor_head,
            "policy_gate": self.policy_gate_head,
            "occlusion_predictor": self.occlusion_predictor,
        }

        # Some entries intentionally alias the same module object
        # (e.g. relocal_acceptor and verifier). Apply trainability once per
        # underlying module so aliases do not override each other.
        grouped_candidates: List[Tuple[str, nn.Module, List[str]]] = []
        seen_module_ids: Set[int] = set()
        for name, module in candidates.items():
            if module is None:
                continue
            module_id = id(module)
            if module_id in seen_module_ids:
                continue
            alias_names = [alias_name for alias_name, alias_module in candidates.items() if alias_module is module]
            grouped_candidates.append((name, module, alias_names))
            seen_module_ids.add(module_id)

        def _normalize_names(names: List[str]) -> Set[str]:
            cleaned = set()
            for n in names:
                if n is None:
                    continue
                n = str(n).strip()
                if not n:
                    continue
                cleaned.add(n)
            return cleaned

        if trainable_list is not None and len(trainable_list) > 0:
            trainable = _normalize_names(trainable_list)
            unknown = sorted(list(trainable.difference(candidates.keys())))
            if unknown:
                logger.warning(f"Unknown refiner.trainable_modules entries ignored: {unknown}")
            for canonical_name, module, alias_names in grouped_candidates:
                self._set_submodule_trainable(
                    canonical_name,
                    module,
                    trainable=any(alias_name in trainable for alias_name in alias_names),
                )
        elif freeze_list is not None and len(freeze_list) > 0:
            frozen = _normalize_names(freeze_list)
            unknown = sorted(list(frozen.difference(candidates.keys())))
            if unknown:
                logger.warning(f"Unknown refiner.freeze_modules entries ignored: {unknown}")
            for canonical_name, module, alias_names in grouped_candidates:
                if not any(alias_name in frozen for alias_name in alias_names):
                    continue
                self._set_submodule_trainable(canonical_name, module, trainable=False)

        if self._frozen_submodules:
            logger.info(f"Frozen refiner submodules: {sorted(self._frozen_submodules)}")

    def _set_submodule_trainable(self, name: str, module: nn.Module, trainable: bool) -> None:
        if not isinstance(module, nn.Module):
            return

        if name == "semantic_encoder" and isinstance(module, SemanticEncoder):
            # Respect module.freeze: when clip is frozen, keep CLIP weights frozen even if projection trains.
            for p in module.projection.parameters():
                p.requires_grad = bool(trainable)
            if getattr(module, "freeze", True):
                for p in module.clip_model.parameters():
                    p.requires_grad = False
            else:
                for p in module.clip_model.parameters():
                    p.requires_grad = bool(trainable)
        else:
            for p in module.parameters():
                p.requires_grad = bool(trainable)

        if trainable:
            self._frozen_submodules.discard(name)
        else:
            self._frozen_submodules.add(name)
            module.eval()

    def set_trainable_modules(self, trainable_modules: Optional[List[str]]) -> None:
        """Freeze all candidates except those listed in trainable_modules."""
        if trainable_modules is None:
            return
        trainable = {str(n).strip() for n in trainable_modules if str(n).strip()}
        candidates: Dict[str, Optional[nn.Module]] = {
            "geo_backbone": self.geo_backbone,
            "semantic_encoder": self.semantic_encoder,
            "freq_semantic": self.freq_semantic,
            "temporal_transformer": self.temporal_transformer,
            "residual_decoder": self.residual_decoder,
            "relocalization_residual": self.relocalization_residual_module,
            "relocal_acceptor": self.relocal_acceptor_head,
            "verifier": self.relocal_acceptor_head,
            "policy_gate": self.policy_gate_head,
            "occlusion_predictor": self.occlusion_predictor,
        }

        grouped_candidates: List[Tuple[str, nn.Module, List[str]]] = []
        seen_module_ids: Set[int] = set()
        for name, module in candidates.items():
            if module is None:
                continue
            module_id = id(module)
            if module_id in seen_module_ids:
                continue
            alias_names = [alias_name for alias_name, alias_module in candidates.items() if alias_module is module]
            grouped_candidates.append((name, module, alias_names))
            seen_module_ids.add(module_id)

        unknown = sorted(list(trainable.difference(candidates.keys())))
        if unknown:
            logger.warning(f"Unknown trainable_modules ignored: {unknown}")
        for canonical_name, module, alias_names in grouped_candidates:
            self._set_submodule_trainable(
                canonical_name,
                module,
                trainable=any(alias_name in trainable for alias_name in alias_names),
            )

    def apply_refiner_stage(self, epoch: int) -> None:
        """
        Apply staged freeze/unfreeze schedule defined in config.refiner.stages.

        Each stage is a dict with:
          - start_epoch (default 0)
          - end_epoch (optional, exclusive)
          - until_epoch (optional alias for end_epoch-1, inclusive)
          - trainable_modules: list[str]
        """
        refiner_cfg = getattr(self.config, "refiner", None)
        stages = getattr(refiner_cfg, "stages", None) if refiner_cfg is not None else None
        if stages is None:
            return
        try:
            stages_list = list(stages)
        except Exception:
            return
        if len(stages_list) == 0:
            return

        active_idx: Optional[int] = None
        for idx, stage in enumerate(stages_list):
            if stage is None:
                continue
            try:
                start_epoch = int(stage.get("start_epoch", 0))
            except Exception:
                start_epoch = 0
            end_epoch = stage.get("end_epoch", None)
            until_epoch = stage.get("until_epoch", None)
            if end_epoch is None and until_epoch is not None:
                try:
                    end_epoch = int(until_epoch) + 1
                except Exception:
                    end_epoch = None
            try:
                end_epoch_i = int(end_epoch) if end_epoch is not None else None
            except Exception:
                end_epoch_i = None

            if epoch < start_epoch:
                continue
            if end_epoch_i is not None and epoch >= end_epoch_i:
                continue
            # Prefer the latest matching stage (highest start_epoch)
            if active_idx is None:
                active_idx = idx
            else:
                try:
                    prev_start = int(stages_list[active_idx].get("start_epoch", 0))
                except Exception:
                    prev_start = 0
                if start_epoch >= prev_start:
                    active_idx = idx

        if active_idx is None:
            return
        if self._active_stage_idx is not None and int(self._active_stage_idx) == int(active_idx):
            return

        stage = stages_list[active_idx]
        trainable_modules = None
        if hasattr(stage, "get"):
            try:
                trainable_modules = stage.get("trainable_modules", None)
            except Exception:
                trainable_modules = None
        if trainable_modules is None:
            try:
                stage_dict = dict(stage)
                trainable_modules = stage_dict.get("trainable_modules", None)
            except Exception:
                trainable_modules = None
        if trainable_modules is not None:
            try:
                trainable_list = [str(v) for v in list(trainable_modules)]
            except Exception:
                trainable_list = None
            self.set_trainable_modules(trainable_list)
            logger.info(f"Applied refiner stage {active_idx} at epoch {epoch}: {trainable_list}")
        self._active_stage_idx = int(active_idx)

    @staticmethod
    def _strip_base_tracker_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Do not load/save CoTracker base weights inside FSPT checkpoints."""
        if not isinstance(state_dict, dict):
            return state_dict
        prefixes = ("base_tracker.", "base_tracker._predictor.")

        stripped_state = {}
        for key, value in state_dict.items():
            normalized = key
            # DDP/DataParallel may call state_dict with prefix="module.".
            # Strip repeated "module." for robust filtering.
            while normalized.startswith("module."):
                normalized = normalized[len("module.") :]
            if normalized.startswith(prefixes):
                continue
            stripped_state[key] = value
        return stripped_state

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        state = super().state_dict(*args, **kwargs)
        return self._strip_base_tracker_keys(state)

    def load_state_dict(self, state_dict, strict: bool = False):  # type: ignore[override]
        # train.py/load_checkpoint calls model.load_state_dict(...) without strict=False.
        # We default to non-strict to support checkpoints that intentionally omit CoTracker weights.
        filtered = self._strip_base_tracker_keys(state_dict)
        return super().load_state_dict(filtered, strict=bool(strict))

    def _init_relocalization_residual_identity(self) -> None:
        """Initialize long-occlusion residual branch to near-identity behavior."""
        if not self.relocalization_residual_enabled:
            return
        if isinstance(self.relocalization_residual_delta_head, nn.Sequential):
            delta_last = self.relocalization_residual_delta_head[-1]
            if isinstance(delta_last, nn.Linear):
                nn.init.constant_(delta_last.weight, 0.0)
                if delta_last.bias is not None:
                    nn.init.constant_(delta_last.bias, 0.0)
        if isinstance(self.relocalization_residual_gate_head, nn.Sequential):
            gate_last = self.relocalization_residual_gate_head[-1]
            if isinstance(gate_last, nn.Linear):
                nn.init.constant_(gate_last.weight, 0.0)
                if gate_last.bias is not None:
                    nn.init.constant_(gate_last.bias, float(self.relocalization_residual_gate_init_bias))

    def _init_weights(self):
        for name, m in self.named_modules():
            if not name:
                continue
            # Do not destroy pretrained weights.
            if name.startswith("geo_backbone.backbone."):
                continue
            if name.startswith("semantic_encoder.clip_model."):
                continue
            if name.startswith("base_tracker."):
                continue

            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _sample_features(
        self,
        features: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, T, H, W, C)
            points: (B, N, 2) or (B, T, N, 2) in [y,x] normalized to [0,1]
        Returns:
            point_features: (B, T, N, C)
        """
        B, T, H, W, C = features.shape
        features_flat = features.reshape(B * T, H, W, C).permute(0, 3, 1, 2)  # (B*T, C, H, W)

        if points.dim() == 3:
            N = points.shape[1]
            points_bt = points.unsqueeze(1).expand(B, T, N, 2)
        else:
            N = points.shape[2]
            points_bt = points
        grid = _normalized_yx_to_grid_xy(points_bt, H, W)
        grid = grid.reshape(B * T, N, 1, 2)

        sampled = F.grid_sample(features_flat, grid, mode="bilinear", align_corners=True)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (B*T, N, C)
        return sampled.reshape(B, T, N, C)

    def _align_tracks_to_query(
        self,
        tracks: torch.Tensor,
        init_positions: torch.Tensor,
        query_t: torch.Tensor,
    ) -> torch.Tensor:
        B, N, T, _ = tracks.shape
        delta = tracks - init_positions.unsqueeze(2)
        batch_idx = torch.arange(B, device=tracks.device).view(B, 1).expand(B, N)
        point_idx = torch.arange(N, device=tracks.device).view(1, N).expand(B, N)
        delta_at_query = delta[batch_idx, point_idx, query_t]
        aligned = init_positions.unsqueeze(2) + (delta - delta_at_query.unsqueeze(2))
        return torch.clamp(aligned, 0.0, 1.0)

    def _get_semantic_features(
        self,
        video: torch.Tensor,
        geo_features: torch.Tensor,
        video_names: Optional[List[str]] = None,
    ) -> torch.Tensor:
        if not self.use_semantic or self.semantic_encoder is None:
            return geo_features

        use_semantic_cache = (
            self.semantic_cache is not None
            and video_names is not None
            and (not self.training or not self.semantic_cache_only_eval)
        )
        semantic_cache = self.semantic_cache if use_semantic_cache else None
        semantic_cache_key = video_names if use_semantic_cache else None
        semantic_cache_mode = self.semantic_cache_mode if use_semantic_cache else None

        semantic_features = None
        if semantic_cache_mode != "global" and self._semantic_spatial_available:
            try:
                semantic_features = self.semantic_encoder(
                    video,
                    mode="spatial",
                    cache=semantic_cache,
                    cache_key=semantic_cache_key,
                )
            except Exception as exc:
                self._semantic_spatial_available = False
                if not self._warned_semantic_fallback:
                    logger.warning(f"Spatial semantic extraction failed, fallback to global: {exc}")
                    self._warned_semantic_fallback = True
        if semantic_features is None:
            global_features = self.semantic_encoder(
                video,
                mode="global",
                cache=semantic_cache,
                cache_key=semantic_cache_key,
            )  # (B,T,dim)
            Hs, Ws = geo_features.shape[2], geo_features.shape[3]
            semantic_features = global_features[:, :, None, None, :].expand(-1, -1, Hs, Ws, -1)
        return semantic_features

    @torch.no_grad()
    def _maybe_apply_retracking(
        self,
        *,
        video: torch.Tensor,  # (B,T,3,H,W)
        tracks: torch.Tensor,  # (B,N,T,2)
        query_t: torch.Tensor,  # (B,N)
        relocal_mask_nt: Optional[torch.Tensor],  # (B,N,T)
        relocal_anchor_tracks: Optional[torch.Tensor],  # (B,N,T,2)
        relocal_conf_nt: Optional[torch.Tensor],  # (B,N,T)
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Restart the base tracker at a relocalized re-appearance frame and splice the tail.

        This is an eval-friendly, low-risk structural change: we still rely on the
        strong base tracker for long tails, but with a better initialization.
        """
        if not self.retracking_enabled:
            return tracks, None
        if self.training and (not self.retracking_apply_in_train):
            return tracks, None
        if (
            relocal_mask_nt is None
            or not isinstance(relocal_mask_nt, torch.Tensor)
            or not isinstance(relocal_anchor_tracks, torch.Tensor)
        ):
            return tracks, None
        if relocal_mask_nt.shape != tracks[..., 0].shape:
            return tracks, None
        if relocal_anchor_tracks.shape != tracks.shape:
            return tracks, None

        B, N, T, _ = tracks.shape
        device = tracks.device

        any_mask = relocal_mask_nt.any(dim=-1)  # (B,N)
        if not bool(any_mask.any()):
            return tracks, {"retrack_mask_bn": any_mask.to(device=device)}

        t_grid = torch.arange(T, device=device).view(1, 1, T).expand(B, N, T)
        if self.retracking_selection == "max_conf" and isinstance(relocal_conf_nt, torch.Tensor):
            if relocal_conf_nt.shape == relocal_mask_nt.shape:
                conf = relocal_conf_nt.to(device=device, dtype=torch.float32)
                masked = torch.where(relocal_mask_nt, conf, torch.full_like(conf, -1.0e9))
                t0 = masked.argmax(dim=-1)  # (B,N)
            else:
                t0 = torch.where(
                    relocal_mask_nt, t_grid, torch.full_like(t_grid, T)
                ).min(dim=-1).values
        else:
            t0 = torch.where(
                relocal_mask_nt, t_grid, torch.full_like(t_grid, T)
            ).min(dim=-1).values  # (B,N)

        valid = any_mask & (t0 < T)
        if not bool(valid.any()):
            return tracks, {"retrack_mask_bn": valid.to(device=device), "retrack_t0": t0.to(device=device)}

        t0_idx = t0.clamp(0, T - 1).to(dtype=torch.long)

        # Optional confidence gate at t0.
        if self.retracking_min_conf is not None and isinstance(relocal_conf_nt, torch.Tensor):
            if relocal_conf_nt.shape == relocal_mask_nt.shape:
                conf_t = torch.gather(relocal_conf_nt, dim=-1, index=t0_idx.unsqueeze(-1)).squeeze(-1)
                valid = valid & (conf_t.to(device=device, dtype=torch.float32) >= float(self.retracking_min_conf))

        if not bool(valid.any()):
            return tracks, {"retrack_mask_bn": valid.to(device=device), "retrack_t0": t0_idx.to(device=device)}

        gather_idx = t0_idx.view(B, N, 1, 1).expand(B, N, 1, 2)
        anchor_yx = torch.gather(relocal_anchor_tracks, dim=2, index=gather_idx).squeeze(2)  # (B,N,2)
        anchor_yx = anchor_yx.to(device=device, dtype=tracks.dtype).clamp(0.0, 1.0)

        tracks_out = tracks.clone()
        for b in range(B):
            idx = torch.nonzero(valid[b], as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue

            t_sel = t0_idx[b, idx].to(device=device, dtype=video.dtype)
            yx_sel = anchor_yx[b, idx].to(device=device, dtype=video.dtype)
            q = torch.stack([t_sel, yx_sel[:, 0], yx_sel[:, 1]], dim=-1).unsqueeze(0)  # (1,M,3)

            retracked, _vis = self.base_tracker(video[b : b + 1], q)  # (1,M,T,2)
            retracked = retracked.to(device=device, dtype=tracks_out.dtype).clamp(0.0, 1.0)

            # Force exact init at the restart frame.
            for m in range(int(idx.numel())):
                n = int(idx[m].item())
                t_start = int(t0_idx[b, n].item())
                if 0 <= t_start < T:
                    retracked[0, m, t_start, :] = anchor_yx[b, n]

            # Splice tail.
            for m in range(int(idx.numel())):
                n = int(idx[m].item())
                t_start = int(t0_idx[b, n].item())
                if not (0 <= t_start < T):
                    continue
                if self.retracking_overwrite_mode == "window" and self.retracking_overwrite_window > 0:
                    t_end = min(T, t_start + int(self.retracking_overwrite_window))
                else:
                    t_end = T
                tracks_out[b, n, t_start:t_end, :] = retracked[0, m, t_start:t_end, :]

        info = {"retrack_mask_bn": valid.to(device=device), "retrack_t0": t0_idx.to(device=device)}
        return tracks_out, info

    def forward(
        self,
        video: torch.Tensor,
        query_points: torch.Tensor,
        meta: Optional[Dict] = None,
        return_info: bool = False,
        return_iter_tracks: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict]]:
        B, T, C, H, W = video.shape
        N = query_points.shape[1]

        # Query alignment targets
        init_positions = query_points[:, :, 1:3]
        query_t = query_points[:, :, 0].round().long().clamp(0, T - 1)

        video_names = None
        if isinstance(meta, dict):
            raw_name = meta.get("video_name", None)
            if raw_name is not None:
                if isinstance(raw_name, (list, tuple)) and len(raw_name) == B:
                    video_names = [str(v) for v in raw_name]
                elif isinstance(raw_name, torch.Tensor):
                    try:
                        if raw_name.ndim == 0 and B == 1:
                            video_names = [str(raw_name.item())]
                        elif raw_name.ndim == 1 and int(raw_name.shape[0]) == B:
                            video_names = [str(v.item()) for v in raw_name]
                    except Exception:
                        video_names = None
                elif B == 1:
                    video_names = [str(raw_name)]

        # 0) Base tracking (CoTracker) - optionally use precomputed base tracks.
        base_tracks = None
        _base_visibility = None
        if isinstance(meta, dict):
            base_tracks = meta.get("base_tracks", None)
            _base_visibility = meta.get("base_visibility", None)

        def _normalize_base_inputs(tracks, vis):
            if tracks is None or vis is None:
                return None, None
            if isinstance(tracks, (list, tuple)):
                tracks = tracks[0] if tracks else None
            if isinstance(vis, (list, tuple)):
                vis = vis[0] if vis else None
            if not isinstance(tracks, torch.Tensor) or not isinstance(vis, torch.Tensor):
                return None, None
            if tracks.dim() == 3:
                tracks = tracks.unsqueeze(0)
            if vis.dim() == 2:
                vis = vis.unsqueeze(0)
            if tracks.dim() != 4 or tracks.shape[-1] != 2:
                return None, None
            if vis.dim() != 3:
                return None, None
            if tracks.shape[0] != B or tracks.shape[1] != N or tracks.shape[2] != T:
                return None, None
            if vis.shape[0] != B or vis.shape[1] != N or vis.shape[2] != T:
                return None, None
            tracks = tracks.to(device=video.device, dtype=video.dtype)
            vis = vis.to(device=video.device)
            if vis.dtype != torch.float32 and vis.dtype != torch.float16 and vis.dtype != torch.bfloat16:
                vis = vis.float()
            return tracks, vis

        base_tracks, _base_visibility = _normalize_base_inputs(base_tracks, _base_visibility)
        cotracker_fmaps = None
        need_cotracker_features = bool(
            self.refiner_feature_source == "cotracker"
            or (self.local_correlation is not None and self.local_corr_feature_source == "cotracker")
            or (self.relocalization_enabled and self.relocalization_feature_source == "cotracker")
            or (self.framewise_init_enabled and self.framewise_init_feature_source == "cotracker")
        )
        if need_cotracker_features:
            base_tracks, _base_visibility, base_info = self.base_tracker(
                video, query_points, return_features=True
            )
            if isinstance(base_info, dict):
                cotracker_fmaps = base_info.get("cotracker_fmaps", None)
        elif base_tracks is None or _base_visibility is None:
            base_tracks, _base_visibility = self.base_tracker(video, query_points)  # (B,N,T,2), (B,N,T)

        base_tracks, _base_visibility = self._maybe_apply_base_occlusion_drift(
            base_tracks, _base_visibility, query_t=query_t, height=H, width=W
        )
        tracks = base_tracks
        tracks = self._maybe_apply_base_jitter(tracks, query_t=query_t, height=H, width=W)
        tracks = self._align_tracks_to_query(tracks, init_positions, query_t)

        # 1) Feature maps for refinement
        geo_pyramid = None
        geo_backbone_features = None
        use_cotracker_features = bool(
            self.refiner_feature_source == "cotracker" and isinstance(cotracker_fmaps, torch.Tensor)
        )
        need_geo_backbone = bool(
            (not use_cotracker_features)
            or (
                self.local_correlation is not None
                and self.local_corr_feature_source == "geo"
            )
            or (
                self.local_correlation_fine is not None
                and self.local_corr_fine_feature_source == "geo"
            )
            or (self.relocalization_enabled and self.relocalization_feature_source == "geo")
            or (self.framewise_init_enabled and self.framewise_init_feature_source == "geo")
        )
        need_geo_pyramid = False
        if need_geo_backbone:
            if (
                self.local_correlation is not None
                and self.local_corr_feature_source == "geo"
                and self.local_corr_feature_level in ("c2", "c3", "c4", "c5")
            ):
                need_geo_pyramid = True
            if (
                self.local_correlation_fine is not None
                and self.local_corr_fine_feature_source == "geo"
                and self.local_corr_fine_feature_level in ("c2", "c3", "c4", "c5")
            ):
                need_geo_pyramid = True
            if (
                self.relocalization_enabled
                and self.relocalization_feature_source == "geo"
                and self.relocalization_feature_level in ("c2", "c3", "c4", "c5")
            ):
                need_geo_pyramid = True
            if (
                self.framewise_init_enabled
                and self.framewise_init_feature_source == "geo"
                and self.framewise_init_feature_level in ("c2", "c3", "c4", "c5")
            ):
                need_geo_pyramid = True
        if need_geo_backbone:
            if need_geo_pyramid:
                geo_backbone_features, geo_pyramid = self.geo_backbone(video, return_pyramid=True)  # (B,T,H',W',dim), dict
            else:
                geo_backbone_features = self.geo_backbone(video)  # (B,T,H',W',dim)

        if use_cotracker_features:
            geo_features = cotracker_fmaps.to(device=video.device, dtype=video.dtype)
        else:
            geo_features = geo_backbone_features
        semantic_features = self._get_semantic_features(video, geo_features, video_names=video_names)

        # Optional: detect long-occlusion reappearance frames (using base visibility)
        # and prepare query/template features for a global re-localization step.
        relocal_mask_nt: Optional[torch.Tensor] = None
        relocal_feature_map: Optional[torch.Tensor] = None
        relocal_query_features: Optional[torch.Tensor] = None
        relocal_anchor_tracks: Optional[torch.Tensor] = None
        relocal_anchor_conf_nt: Optional[torch.Tensor] = None
        if self.relocalization_enabled and isinstance(_base_visibility, torch.Tensor):
            try:
                relocal_mask_nt = _compute_reappearance_mask(
                    _base_visibility,
                    query_t,
                    min_occlusion_len=int(self.relocalization_min_occlusion_len),
                    frames_after=int(self.relocalization_frames_after),
                    vis_threshold=float(self.relocalization_vis_threshold),
                )
            except Exception as exc:
                logger.warning(f"Failed to compute reappearance mask; disabling relocalization: {exc}")
                relocal_mask_nt = None

            if relocal_mask_nt is not None and relocal_mask_nt.any():
                relocal_feature_map = self._get_relocalization_feature_map(
                    cotracker_fmaps=cotracker_fmaps,
                    geo_backbone_features=geo_backbone_features,
                    geo_pyramid=geo_pyramid,
                    video=video,
                )
                if isinstance(relocal_feature_map, torch.Tensor) and relocal_feature_map.dim() == 5:
                    relocal_feature_map = relocal_feature_map.to(device=video.device, dtype=video.dtype)
                    template_mode = str(self.relocalization_template_mode).lower().strip()
                    if (
                        template_mode in ("last_visible", "visible_bank")
                        and isinstance(base_tracks, torch.Tensor)
                        and base_tracks.shape == tracks.shape
                    ):
                        try:
                            relocal_mask_cpu = relocal_mask_nt.detach().cpu()
                            base_vis_cpu = _base_visibility.detach().cpu()
                            query_t_cpu = query_t.detach().cpu()
                            base_tracks_bt = base_tracks.permute(0, 2, 1, 3).contiguous()
                            template_point_features = self._sample_features(relocal_feature_map, base_tracks_bt)  # (B,T,N,C)
                            bank_size = max(1, int(self.relocalization_template_bank_size))
                            bank_stride = max(1, int(self.relocalization_template_bank_stride))
                            template_ts = torch.zeros((B, N, bank_size), device=query_t_cpu.device, dtype=query_t_cpu.dtype)

                            for b in range(B):
                                for n in range(N):
                                    m = relocal_mask_cpu[b, n]
                                    tq = int(query_t_cpu[b, n].item())
                                    if not bool(m.any()):
                                        template_ts[b, n, :] = tq
                                        continue
                                    first = int(torch.nonzero(m, as_tuple=False)[0].item())
                                    if first <= tq:
                                        template_ts[b, n, :] = tq
                                        continue

                                    visible = base_vis_cpu[b, n] >= float(self.relocalization_vis_threshold)
                                    chosen: List[int] = []
                                    if template_mode == "last_visible":
                                        for tt in range(first - 1, tq - 1, -bank_stride):
                                            if bool(visible[tt]):
                                                chosen.append(int(tt))
                                                if len(chosen) >= bank_size:
                                                    break
                                    else:
                                        visible_idx = [
                                            int(tt) for tt in range(tq, first) if bool(visible[tt])
                                        ]
                                        if len(visible_idx) > 0:
                                            subsampled = visible_idx[::bank_stride]
                                            if len(subsampled) == 0:
                                                subsampled = [int(visible_idx[-1])]
                                            if subsampled[-1] != visible_idx[-1]:
                                                subsampled.append(int(visible_idx[-1]))
                                            if tq not in subsampled:
                                                subsampled = [tq] + subsampled
                                            deduped: List[int] = []
                                            seen = set()
                                            for tt in subsampled:
                                                if tt not in seen:
                                                    deduped.append(int(tt))
                                                    seen.add(int(tt))
                                            chosen = deduped
                                            if len(chosen) > bank_size:
                                                if bank_size == 1:
                                                    chosen = [int(chosen[-1])]
                                                else:
                                                    sel: List[int] = []
                                                    denom = float(bank_size - 1)
                                                    span = float(len(chosen) - 1)
                                                    for k in range(bank_size):
                                                        idx_sel = int(round((float(k) * span) / denom))
                                                        sel.append(int(chosen[idx_sel]))
                                                    chosen = sel

                                    if len(chosen) == 0:
                                        chosen = [tq]
                                    while len(chosen) < bank_size:
                                        chosen.append(int(chosen[-1]))
                                    template_ts[b, n, :] = torch.tensor(
                                        chosen[:bank_size],
                                        device=query_t_cpu.device,
                                        dtype=query_t_cpu.dtype,
                                    )

                            template_ts = template_ts.to(device=video.device, dtype=query_t.dtype)
                            if bank_size <= 1:
                                gather_idx = template_ts[..., :1].view(B, 1, N, 1).expand(
                                    B, 1, N, template_point_features.shape[-1]
                                )
                                relocal_query_features = torch.gather(
                                    template_point_features, dim=1, index=gather_idx
                                ).squeeze(1)  # (B,N,C)
                            else:
                                gather_idx = template_ts.permute(0, 2, 1).contiguous().view(
                                    B, bank_size, N, 1
                                ).expand(B, bank_size, N, template_point_features.shape[-1])
                                templates = torch.gather(template_point_features, dim=1, index=gather_idx)  # (B,K,N,C)
                                relocal_query_features = templates.permute(0, 2, 1, 3).contiguous()  # (B,N,K,C)
                        except Exception as exc:
                            logger.warning(
                                f"Failed to compute relocal templates (mode={template_mode}); falling back to query template: {exc}"
                            )
                            template_mode = "query"

                    if template_mode == "query":
                        template_tracks_bt = init_positions[:, None, :, :].expand(B, T, N, 2).contiguous()
                        template_point_features = self._sample_features(relocal_feature_map, template_tracks_bt)  # (B,T,N,C)
                        gather_idx = query_t.view(B, 1, N, 1).expand(
                            B, 1, N, template_point_features.shape[-1]
                        )
                        relocal_query_features = torch.gather(
                            template_point_features, dim=1, index=gather_idx
                        ).squeeze(1)  # (B,N,C)
                else:
                    relocal_mask_nt = None
                    relocal_feature_map = None

        # Optional: frame-wise global init (TAPIR-style).
        framewise_feature_map: Optional[torch.Tensor] = None
        framewise_query_features: Optional[torch.Tensor] = None
        framewise_mask_nt: Optional[torch.Tensor] = None
        if self.framewise_init_enabled:
            framewise_feature_map = self._get_framewise_feature_map(
                cotracker_fmaps=cotracker_fmaps,
                geo_backbone_features=geo_backbone_features,
                geo_pyramid=geo_pyramid,
            )
            if isinstance(framewise_feature_map, torch.Tensor) and framewise_feature_map.dim() == 5:
                framewise_feature_map = framewise_feature_map.to(device=video.device, dtype=video.dtype)
                template_mode = str(self.framewise_init_template_mode).lower().strip()
                if template_mode not in ("query", "last_visible"):
                    template_mode = "query"
                if template_mode == "last_visible" and not isinstance(base_tracks, torch.Tensor):
                    template_mode = "query"
                if template_mode == "last_visible" and not isinstance(_base_visibility, torch.Tensor):
                    template_mode = "query"

                if template_mode == "last_visible":
                    try:
                        base_vis_cpu = _base_visibility.detach().cpu()
                        query_t_cpu = query_t.detach().cpu()
                        base_tracks_bt = base_tracks.permute(0, 2, 1, 3).contiguous()
                        template_point_features = self._sample_features(framewise_feature_map, base_tracks_bt)  # (B,T,N,C)

                        last_idx = torch.zeros((B, N), dtype=torch.long, device=query_t_cpu.device)
                        for b in range(B):
                            for n in range(N):
                                vis = base_vis_cpu[b, n]
                                if bool(vis.any()):
                                    last = int(torch.nonzero(vis, as_tuple=False)[-1].item())
                                else:
                                    last = int(query_t_cpu[b, n].item())
                                last_idx[b, n] = last
                        last_idx = last_idx.to(device=video.device, dtype=torch.long)
                        gather_idx = last_idx.view(B, 1, N, 1).expand(
                            B, 1, N, template_point_features.shape[-1]
                        )
                        framewise_query_features = torch.gather(
                            template_point_features, dim=1, index=gather_idx
                        ).squeeze(1)  # (B,N,C)
                    except Exception as exc:
                        logger.warning(f"Framewise init last_visible failed; falling back to query: {exc}")
                        template_mode = "query"

                if template_mode == "query":
                    template_tracks_bt = init_positions[:, None, :, :].expand(B, T, N, 2).contiguous()
                    template_point_features = self._sample_features(framewise_feature_map, template_tracks_bt)  # (B,T,N,C)
                    gather_idx = query_t.view(B, 1, N, 1).expand(
                        B, 1, N, template_point_features.shape[-1]
                    )
                    framewise_query_features = torch.gather(
                        template_point_features, dim=1, index=gather_idx
                    ).squeeze(1)  # (B,N,C)

                t_grid = torch.arange(T, device=video.device).view(1, 1, T).expand(B, N, T)
                framewise_mask_nt = torch.ones((B, N, T), device=video.device, dtype=torch.bool)
                if self.framewise_init_after_query:
                    framewise_mask_nt = framewise_mask_nt & (t_grid > query_t.unsqueeze(-1))
                if self.framewise_init_exclude_query_frame:
                    framewise_mask_nt = framewise_mask_nt & (t_grid != query_t.unsqueeze(-1))
                if self.framewise_init_stride > 1:
                    diff = (t_grid - query_t.unsqueeze(-1)).clamp(min=0)
                    framewise_mask_nt = framewise_mask_nt & ((diff % int(self.framewise_init_stride)) == 0)
            else:
                framewise_feature_map = None

        framewise_init_tracks: Optional[torch.Tensor] = None
        if (
            self.framewise_init_enabled
            and isinstance(framewise_feature_map, torch.Tensor)
            and isinstance(framewise_query_features, torch.Tensor)
            and isinstance(framewise_mask_nt, torch.Tensor)
            and framewise_mask_nt.any()
        ):
            apply_prob = float(self.framewise_init_apply_prob)
            if apply_prob >= 1.0 or random.random() <= apply_prob:
                try:
                    step_fw, conf_fw = self._compute_global_relocalization_step(
                        feature_map=framewise_feature_map,
                        query_features=framewise_query_features,
                        tracks=tracks,
                        mask_nt=framewise_mask_nt,
                        downsample=int(self.framewise_init_downsample),
                        feature_source=self.framewise_init_feature_source,
                    )
                    if isinstance(conf_fw, torch.Tensor) and self.framewise_init_conf_threshold > 0:
                        gate = _confidence_gate(
                            conf_fw.to(device=step_fw.device, dtype=step_fw.dtype),
                            threshold=float(self.framewise_init_conf_threshold),
                            power=float(self.framewise_init_gate_power),
                        )
                        step_fw = step_fw * gate.unsqueeze(-1)
                    tracks = torch.clamp(tracks + step_fw.to(device=tracks.device, dtype=tracks.dtype), 0.0, 1.0)
                    tracks = self._align_tracks_to_query(tracks, init_positions, query_t)
                    framewise_init_tracks = tracks
                except Exception as exc:
                    logger.warning(f"Framewise init failed; skipping: {exc}")

        iter_tracks: Optional[List[torch.Tensor]] = [tracks] if return_iter_tracks else None
        last_freq_info: Dict = {}
        last_semantic_points: Optional[torch.Tensor] = None
        confidence: Optional[torch.Tensor] = None
        visibility: Optional[torch.Tensor] = None
        last_corr_logits: Optional[torch.Tensor] = None
        last_corr_center_tracks_bt: Optional[torch.Tensor] = None
        last_corr_feature_hw: Optional[torch.Tensor] = None
        last_corr_window_size: Optional[int] = None
        last_temporal_features: Optional[torch.Tensor] = None
        last_corr_conf: Optional[torch.Tensor] = None
        last_relocal_conf_nt: Optional[torch.Tensor] = None
        tracks_pre_retracking: Optional[torch.Tensor] = None

        for iter_idx in range(self.num_refinement_iters):
            relocal_conf_nt: Optional[torch.Tensor] = None
            if (
                relocal_mask_nt is not None
                and relocal_query_features is not None
                and relocal_feature_map is not None
                and self.local_correlation is not None
                and (iter_idx == 0 or (not self.relocalization_apply_only_first_iter))
            ):
                try:
                    # Check if we should use the learned head instead of raw cosine
                    use_learned = (
                        getattr(self, "_use_learned_head", False)
                        and self.relocalization_feature_source == "dino"
                        and self._ensure_learned_head(video.device)
                    )

                    if use_learned:
                        relocal_step, relocal_conf_nt = self._apply_learned_recovery_head(
                            feature_map=relocal_feature_map,
                            query_features=relocal_query_features,
                            tracks=tracks,
                            mask_nt=relocal_mask_nt,
                            video=video,
                            base_visibility=_base_visibility,
                        )
                    else:
                        relocal_step, relocal_conf_nt = self._compute_global_relocalization_step(
                            feature_map=relocal_feature_map,
                            query_features=relocal_query_features,
                            tracks=tracks,
                            mask_nt=relocal_mask_nt,
                            feature_source=self.relocalization_feature_source,
                        )
                    tracks = torch.clamp(tracks + relocal_step.to(device=tracks.device, dtype=tracks.dtype), 0.0, 1.0)
                    tracks = self._align_tracks_to_query(tracks, init_positions, query_t)
                    # Apply the trust-region clamp immediately so subsequent local
                    # refinement modules operate on the same bounded tracks that
                    # will be evaluated/supervised.
                    if (
                        self.delta_max_total is not None
                        and self.delta_max_total > 0
                        and isinstance(base_tracks, torch.Tensor)
                        and base_tracks.shape == tracks.shape
                    ):
                        base_max_total = float(self.delta_max_total)
                        max_total_map = torch.full_like(tracks[..., :1], base_max_total)
                        if (
                            self.relocalization_enabled
                            and self.relocalization_max_total > 0
                            and isinstance(relocal_mask_nt, torch.Tensor)
                            and relocal_mask_nt.shape == tracks[..., 0].shape
                        ):
                            relocal_max_total = float(self.relocalization_max_total)
                            if relocal_max_total > base_max_total:
                                mask = relocal_mask_nt.to(device=tracks.device)
                                max_total_map = torch.where(
                                    mask.unsqueeze(-1),
                                    torch.full_like(max_total_map, relocal_max_total),
                                    max_total_map,
                                )

                        delta_total = tracks - base_tracks
                        delta_total = torch.max(torch.min(delta_total, max_total_map), -max_total_map)
                        tracks = torch.clamp(base_tracks + delta_total, 0.0, 1.0)
                        tracks = self._align_tracks_to_query(tracks, init_positions, query_t)
                    if (
                        relocal_anchor_tracks is None
                        and iter_idx == 0
                        and self.retracking_enabled
                        and (self.retracking_apply_in_train or (not self.training))
                    ):
                        relocal_anchor_tracks = tracks.detach()
                        if isinstance(relocal_conf_nt, torch.Tensor):
                            relocal_anchor_conf_nt = relocal_conf_nt.detach()
                except Exception as exc:
                    logger.warning(f"Relocalization step failed (iter={iter_idx}); skipping: {exc}")
                    relocal_conf_nt = None

            tracks_for_sample = tracks.permute(0, 2, 1, 3).contiguous()  # (B,T,N,2)

            geo_point_features = self._sample_features(geo_features, tracks_for_sample)
            if self.use_semantic:
                semantic_point_features = self._sample_features(semantic_features, tracks_for_sample)
            else:
                semantic_point_features = geo_point_features

            last_semantic_points = semantic_point_features

            # 2) Frequency/semantic fusion
            if self.use_frequency and self.freq_semantic is not None:
                fused, freq_info = self.freq_semantic(geo_point_features, semantic_point_features)
            else:
                fused, freq_info = geo_point_features, {}
            last_freq_info = freq_info

            freq_hf_ratio: Optional[torch.Tensor] = None
            if self.freq_guided_fusion_enabled:
                src = str(self.freq_guided_fusion_source).lower().strip()
                eps = float(self.freq_guided_fusion_eps)

                if (
                    src == "track_velocity"
                    and isinstance(base_tracks, torch.Tensor)
                    and base_tracks.shape == tracks.shape
                ):
                    # Motion "high-frequency" proxy: base track velocity magnitude.
                    # velocity is in normalized coords (roughly px / W or px / H).
                    vel = torch.zeros_like(base_tracks[..., 0])  # (B,N,T)
                    if base_tracks.shape[2] > 1:
                        delta = base_tracks[:, :, 1:, :] - base_tracks[:, :, :-1, :]
                        speed = torch.norm(delta, dim=-1)  # (B,N,T-1)
                        vel[:, :, 1:] = speed
                    th = float(self.freq_guided_fusion_velocity_threshold)
                    th = 1.0e-8 if th <= 0 else th
                    freq_hf_ratio = (vel / th).clamp(0.0, 1.0)

                elif src == "feature_diff":
                    # Feature temporal-derivative energy as a parameter-free HF proxy.
                    # geo_point_features: (B,T,N,C)
                    low_e = geo_point_features.to(device=fused.device, dtype=fused.dtype).pow(2).mean(dim=-1)  # (B,T,N)
                    high_e = torch.zeros_like(low_e)
                    if geo_point_features.shape[1] > 1:
                        diff = geo_point_features[:, 1:, :, :] - geo_point_features[:, :-1, :, :]
                        high_e[:, 1:, :] = diff.to(device=fused.device, dtype=fused.dtype).pow(2).mean(dim=-1)
                    ratio_bt = high_e / (high_e + low_e + eps)
                    freq_hf_ratio = ratio_bt.permute(0, 2, 1).contiguous().clamp(0.0, 1.0)

                else:
                    # Default: use LFD band energy (requires the frequency module).
                    if self.use_frequency and isinstance(freq_info, dict):
                        band_feats = freq_info.get("band_features", None)
                        if isinstance(band_feats, dict):
                            low_idx = int(self.freq_guided_fusion_low_band)
                            high_idx = int(self.freq_guided_fusion_high_band)

                            available: List[int] = []
                            for key in band_feats.keys():
                                if not isinstance(key, str) or not key.startswith("band_"):
                                    continue
                                try:
                                    available.append(int(key.split("_", 1)[1]))
                                except Exception:
                                    continue
                            available = sorted(set(available))
                            if available:
                                if low_idx not in available:
                                    low_idx = available[0]
                                if high_idx < 0 or high_idx not in available:
                                    high_idx = available[-1]

                                low = band_feats.get(f"band_{low_idx}", None)
                                high = band_feats.get(f"band_{high_idx}", None)
                                if (
                                    isinstance(low, torch.Tensor)
                                    and isinstance(high, torch.Tensor)
                                    and low.shape == high.shape
                                    and low.dim() == 4
                                ):
                                    low_e = low.to(device=fused.device, dtype=fused.dtype).pow(2).mean(dim=-1)
                                    high_e = high.to(device=fused.device, dtype=fused.dtype).pow(2).mean(dim=-1)
                                    ratio_bt = high_e / (high_e + low_e + eps)
                                    freq_hf_ratio = ratio_bt.permute(0, 2, 1).contiguous().clamp(0.0, 1.0)

                if freq_hf_ratio is not None:
                    if self.freq_guided_fusion_detach:
                        freq_hf_ratio = freq_hf_ratio.detach()
                    power = float(self.freq_guided_fusion_power)
                    if abs(power - 1.0) > 1e-6:
                        freq_hf_ratio = freq_hf_ratio ** power

            corr_step = None
            corr_conf = None
            if self.local_correlation is not None or self.local_correlation_fine is not None:
                corr_emb_total = None

                step_candidates: List[torch.Tensor] = []
                conf_candidates: List[Optional[torch.Tensor]] = []
                apply_step_before_residual = False

                corr_logits_for_loss: Optional[torch.Tensor] = None
                corr_feature_hw_for_loss: Optional[torch.Tensor] = None
                corr_window_size_for_loss: Optional[int] = None

                def _gate_step(
                    step: torch.Tensor,
                    conf: Optional[torch.Tensor],
                    gate_mode: str,
                    gate_threshold: float,
                    gate_power: float,
                    gate_train: bool,
                ) -> torch.Tensor:
                    if (
                        gate_mode in ("confidence", "conf", "maxprob", "max_prob")
                        and conf is not None
                        and ((not self.training) or gate_train)
                    ):
                        conf_t = conf.to(device=tracks.device, dtype=tracks.dtype)
                        gate = _confidence_gate(
                            conf_t,
                            threshold=gate_threshold,
                            power=gate_power,
                        )
                        return step * gate.unsqueeze(-1)
                    return step

                # --- Coarse correlation (typically CoTracker fnet features) ---
                if self.local_correlation is not None:
                    corr_feature_map = geo_features
                    corr_point_features = geo_point_features
                    if self.local_corr_feature_source == "cotracker" and isinstance(cotracker_fmaps, torch.Tensor):
                        # Reuse already-sampled point features when the refiner backbone is CoTracker features.
                        if use_cotracker_features:
                            corr_feature_map = geo_features
                            corr_point_features = geo_point_features
                        else:
                            corr_feature_map = cotracker_fmaps.to(device=video.device, dtype=video.dtype)
                            corr_point_features = self._sample_features(corr_feature_map, tracks_for_sample)
                    elif (
                        self.local_corr_feature_source == "geo"
                        and self.local_corr_feature_level in ("c2", "c3", "c4", "c5")
                        and isinstance(geo_pyramid, dict)
                    ):
                        level_map = geo_pyramid.get(self.local_corr_feature_level, None)
                        if isinstance(level_map, torch.Tensor) and level_map.dim() == 5:
                            corr_feature_map = level_map.to(device=video.device, dtype=video.dtype)
                            corr_point_features = self._sample_features(corr_feature_map, tracks_for_sample)

                    corr_step_coarse = None
                    corr_conf_coarse = None
                    corr_logits_coarse = None
                    if self.local_corr_use_step:
                        want_logits = bool(return_info or self.training or self.local_corr_map_update_enabled)
                        corr_out = self.local_correlation(
                            corr_feature_map,
                            corr_point_features,
                            tracks_for_sample,
                            query_t,
                            return_step=True,
                            return_logits=want_logits,
                        )
                        if want_logits and isinstance(corr_out, (list, tuple)) and len(corr_out) == 4:
                            corr_emb, corr_step_coarse, corr_conf_coarse, corr_logits_coarse = corr_out
                        else:
                            corr_emb, corr_step_coarse, corr_conf_coarse = corr_out
                        if (
                            self.local_corr_map_update_enabled
                            and self.local_corr_map_update is not None
                            and isinstance(corr_logits_coarse, torch.Tensor)
                        ):
                            step_cells, step_gate = self.local_corr_map_update(corr_logits_coarse)  # (B,T,N,2), (B,T,N)
                            denom = torch.tensor(
                                [float(corr_feature_map.shape[2]), float(corr_feature_map.shape[3])],
                                device=step_cells.device,
                                dtype=step_cells.dtype,
                            ).view(1, 1, 1, 2)
                            step_bt = step_cells / denom  # (B,T,N,2) normalized y/x delta
                            step = step_bt.permute(0, 2, 1, 3).contiguous()  # (B,N,T,2)
                            gate = step_gate.permute(0, 2, 1).contiguous()  # (B,N,T)
                            corr_step_coarse = step * gate.unsqueeze(-1) * float(self.local_corr_map_update_step_scale)
                            if corr_conf_coarse is None:
                                corr_conf_coarse = gate
                            else:
                                corr_conf_coarse = corr_conf_coarse * gate
                    else:
                        corr_emb = self.local_correlation(
                            corr_feature_map,
                            corr_point_features,
                            tracks_for_sample,
                            query_t,
                        )

                    corr_emb_total = corr_emb if corr_emb_total is None else (corr_emb_total + corr_emb)

                    if corr_step_coarse is not None:
                        step = corr_step_coarse.to(device=tracks.device, dtype=tracks.dtype) * float(
                            self.local_corr_step_scale
                        )
                        step = _gate_step(
                            step,
                            corr_conf_coarse,
                            gate_mode=self.local_corr_step_gate,
                            gate_threshold=self.local_corr_gate_threshold,
                            gate_power=self.local_corr_gate_power,
                            gate_train=self.local_corr_gate_step_train,
                        )
                        step_candidates.append(step)
                        conf_candidates.append(corr_conf_coarse)
                        apply_step_before_residual = apply_step_before_residual or bool(
                            self.local_corr_apply_step_before_residual
                        )

                    # Prefer fine correlation for supervision if it exists; otherwise use coarse.
                    if (
                        corr_logits_for_loss is None
                        and isinstance(corr_logits_coarse, torch.Tensor)
                        and corr_logits_coarse.dim() == 4
                    ):
                        corr_logits_for_loss = corr_logits_coarse
                        corr_feature_hw_for_loss = torch.tensor(
                            [corr_feature_map.shape[2], corr_feature_map.shape[3]],
                            device=corr_feature_map.device,
                            dtype=torch.int32,
                        )
                        corr_window_size_for_loss = getattr(self.local_correlation, "window_size", None)

                # --- Fine correlation (typically geo pyramid c2 features) ---
                if self.local_correlation_fine is not None:
                    corr_feature_map_fine = None
                    if self.local_corr_fine_feature_source == "cotracker" and isinstance(cotracker_fmaps, torch.Tensor):
                        corr_feature_map_fine = geo_features if use_cotracker_features else cotracker_fmaps.to(
                            device=video.device, dtype=video.dtype
                        )
                    else:
                        if (
                            self.local_corr_fine_feature_level in ("c2", "c3", "c4", "c5")
                            and isinstance(geo_pyramid, dict)
                        ):
                            corr_feature_map_fine = geo_pyramid.get(self.local_corr_fine_feature_level, None)
                        if corr_feature_map_fine is None:
                            corr_feature_map_fine = geo_backbone_features

                    corr_step_fine = None
                    corr_conf_fine = None
                    corr_logits_fine = None
                    if isinstance(corr_feature_map_fine, torch.Tensor) and corr_feature_map_fine.dim() == 5:
                        expected_dim = int(getattr(self.local_correlation_fine, "in_dim", corr_feature_map_fine.shape[-1]))
                        if int(corr_feature_map_fine.shape[-1]) != expected_dim:
                            if not self._warned_local_corr_fine_channel_mismatch:
                                logger.warning(
                                    "local_correlation_fine skipped due to channel mismatch: "
                                    f"feature_map C={corr_feature_map_fine.shape[-1]} vs expected in_dim={expected_dim} "
                                    f"(level={self.local_corr_fine_feature_level})."
                                )
                                self._warned_local_corr_fine_channel_mismatch = True
                        else:
                            corr_feature_map_fine = corr_feature_map_fine.to(device=video.device, dtype=video.dtype)
                            corr_point_features_fine = self._sample_features(corr_feature_map_fine, tracks_for_sample)
                            if self.local_corr_fine_use_step:
                                want_logits = bool(return_info or self.training or self.local_corr_fine_map_update_enabled)
                                corr_out = self.local_correlation_fine(
                                    corr_feature_map_fine,
                                    corr_point_features_fine,
                                    tracks_for_sample,
                                    query_t,
                                    return_step=True,
                                    return_logits=want_logits,
                                )
                                if want_logits and isinstance(corr_out, (list, tuple)) and len(corr_out) == 4:
                                    corr_emb_fine, corr_step_fine, corr_conf_fine, corr_logits_fine = corr_out
                                else:
                                    corr_emb_fine, corr_step_fine, corr_conf_fine = corr_out
                                if (
                                    self.local_corr_fine_map_update_enabled
                                    and self.local_corr_fine_map_update is not None
                                    and isinstance(corr_logits_fine, torch.Tensor)
                                ):
                                    step_cells, step_gate = self.local_corr_fine_map_update(corr_logits_fine)  # (B,T,N,2), (B,T,N)
                                    denom = torch.tensor(
                                        [float(corr_feature_map_fine.shape[2]), float(corr_feature_map_fine.shape[3])],
                                        device=step_cells.device,
                                        dtype=step_cells.dtype,
                                    ).view(1, 1, 1, 2)
                                    step_bt = step_cells / denom  # (B,T,N,2) normalized y/x delta
                                    step = step_bt.permute(0, 2, 1, 3).contiguous()  # (B,N,T,2)
                                    gate = step_gate.permute(0, 2, 1).contiguous()  # (B,N,T)
                                    corr_step_fine = step * gate.unsqueeze(-1) * float(
                                        self.local_corr_fine_map_update_step_scale
                                    )
                                    if corr_conf_fine is None:
                                        corr_conf_fine = gate
                                    else:
                                        corr_conf_fine = corr_conf_fine * gate
                            else:
                                corr_emb_fine = self.local_correlation_fine(
                                    corr_feature_map_fine,
                                    corr_point_features_fine,
                                    tracks_for_sample,
                                    query_t,
                                )
                            corr_emb_total = (
                                corr_emb_fine
                                if corr_emb_total is None
                                else (corr_emb_total + corr_emb_fine)
                            )

                    if corr_step_fine is not None:
                        step = corr_step_fine.to(device=tracks.device, dtype=tracks.dtype) * float(
                            self.local_corr_fine_step_scale
                        )
                        step = _gate_step(
                            step,
                            corr_conf_fine,
                            gate_mode=self.local_corr_fine_step_gate,
                            gate_threshold=self.local_corr_fine_gate_threshold,
                            gate_power=self.local_corr_fine_gate_power,
                            gate_train=self.local_corr_fine_gate_step_train,
                        )
                        step_candidates.append(step)
                        conf_candidates.append(corr_conf_fine)
                        apply_step_before_residual = apply_step_before_residual or bool(
                            self.local_corr_fine_apply_step_before_residual
                        )

                    # Prefer fine logits for supervision (overrides coarse).
                    if isinstance(corr_logits_fine, torch.Tensor) and corr_logits_fine.dim() == 4:
                        corr_logits_for_loss = corr_logits_fine
                        corr_feature_hw_for_loss = torch.tensor(
                            [corr_feature_map_fine.shape[2], corr_feature_map_fine.shape[3]],
                            device=corr_feature_map_fine.device,
                            dtype=torch.int32,
                        )
                        corr_window_size_for_loss = getattr(self.local_correlation_fine, "window_size", None)

                if corr_emb_total is not None:
                    fused = fused + corr_emb_total

                if len(step_candidates) == 1:
                    corr_step = step_candidates[0]
                    corr_conf = conf_candidates[0]
                elif len(step_candidates) >= 2:
                    step1, step2 = step_candidates[0], step_candidates[1]
                    conf1, conf2 = conf_candidates[0], conf_candidates[1]

                    if self.local_corr_fusion_enabled:
                        power = float(self.local_corr_fusion_power)
                        eps = float(self.local_corr_fusion_eps)
                        if conf1 is None:
                            w1 = torch.ones(step1.shape[:-1], device=step1.device, dtype=step1.dtype)
                        else:
                            w1 = conf1.to(device=step1.device, dtype=step1.dtype).clamp(min=0.0) ** power
                        if conf2 is None:
                            w2 = torch.ones(step2.shape[:-1], device=step2.device, dtype=step2.dtype)
                        else:
                            w2 = conf2.to(device=step2.device, dtype=step2.dtype).clamp(min=0.0) ** power

                        # Frequency-guided bias: (coarse, fine) = (low, high) frequency.
                        if freq_hf_ratio is not None:
                            hf = freq_hf_ratio.to(device=step1.device, dtype=step1.dtype)
                            if self.freq_guided_fusion_invert:
                                hf = 1.0 - hf
                            strength = float(self.freq_guided_fusion_strength)
                            bias_eps = eps
                            bias_coarse = ((1.0 - strength) + strength * (1.0 - hf)).clamp(min=bias_eps)
                            bias_fine = ((1.0 - strength) + strength * hf).clamp(min=bias_eps)
                            w1 = w1 * bias_coarse
                            w2 = w2 * bias_fine
                        denom = (w1 + w2).clamp(min=eps)
                        corr_step = (step1 * w1.unsqueeze(-1) + step2 * w2.unsqueeze(-1)) / denom.unsqueeze(-1)
                    else:
                        corr_step = step1 + step2

                    if conf1 is not None and conf2 is not None:
                        corr_conf = torch.maximum(conf1, conf2)
                    else:
                        corr_conf = conf1 if conf2 is None else conf2

                if corr_step is not None and apply_step_before_residual:
                    tracks = torch.clamp(tracks + corr_step, 0.0, 1.0)
                    tracks = self._align_tracks_to_query(tracks, init_positions, query_t)
                    # Avoid applying the same step again after the residual update.
                    corr_step = None
                if isinstance(corr_conf, torch.Tensor):
                    last_corr_conf = corr_conf

                # Correlation supervision should focus on correcting the *base* estimate.
                # Store logits only for the first refinement iteration so the target offset
                # reflects (GT - base) rather than (GT - already-refined).
                if last_corr_logits is None and corr_logits_for_loss is not None:
                    last_corr_logits = corr_logits_for_loss
                    last_corr_center_tracks_bt = tracks_for_sample
                    last_corr_feature_hw = corr_feature_hw_for_loss
                    last_corr_window_size = corr_window_size_for_loss

            # 3) Temporal modeling
            if (
                self.prior_features_enabled
                and self.prior_features_mlp is not None
                and isinstance(base_tracks, torch.Tensor)
                and base_tracks.shape == tracks.shape
            ):
                prior_parts: List[torch.Tensor] = []
                if (
                    self.prior_features_use_base_visibility
                    and isinstance(_base_visibility, torch.Tensor)
                    and _base_visibility.shape == tracks[..., 0].shape
                ):
                    base_vis = _base_visibility.to(device=fused.device, dtype=fused.dtype)
                    prior_parts.append(base_vis.permute(0, 2, 1).unsqueeze(-1).contiguous())
                if self.prior_features_use_delta:
                    delta = tracks - base_tracks
                    if self.prior_features_detach_delta:
                        delta = delta.detach()
                    delta_bt = delta.permute(0, 2, 1, 3).contiguous()
                    prior_parts.append(delta_bt.to(device=fused.device, dtype=fused.dtype))
                    if self.prior_features_use_delta_norm:
                        prior_parts.append(torch.norm(delta_bt, dim=-1, keepdim=True))
                if prior_parts:
                    prior_in = torch.cat(prior_parts, dim=-1)
                    prior_emb = self.prior_features_mlp(prior_in)
                    fused = fused + prior_emb * float(self.prior_features_scale)

            temporal_features = self.temporal_transformer(fused)
            last_temporal_features = temporal_features

            # 4) Residual update
            tracks_before_residual = tracks
            tracks, visibility_pred = self.residual_decoder(temporal_features, tracks)
            tracks = self._align_tracks_to_query(tracks, init_positions, query_t)
            visibility = visibility_pred
            if (
                self.local_correlation is not None
                and self.local_corr_gate_residual
                and ((not self.training) or self.local_corr_gate_residual_train)
                and corr_conf is not None
                and isinstance(tracks_before_residual, torch.Tensor)
                and tracks_before_residual.shape == tracks.shape
            ):
                conf = corr_conf.to(device=tracks.device, dtype=tracks.dtype)  # (B,N,T)
                gate = _confidence_gate(
                    conf,
                    threshold=self.local_corr_residual_gate_threshold,
                    power=self.local_corr_residual_gate_power,
                )
                step = (tracks - tracks_before_residual) * gate.unsqueeze(-1)
                tracks = torch.clamp(tracks_before_residual + step, 0.0, 1.0)
                tracks = self._align_tracks_to_query(tracks, init_positions, query_t)
            if corr_step is not None:
                # `corr_step` is already scaled/gated (and may be fused across multiple
                # correlation sources). Apply it as-is.
                step = corr_step.to(device=tracks.device, dtype=tracks.dtype)
                tracks = torch.clamp(tracks + step, 0.0, 1.0)
                tracks = self._align_tracks_to_query(tracks, init_positions, query_t)

            if (
                self.relocalization_residual_enabled
                and (iter_idx == 0 or (not self.relocalization_residual_apply_only_first_iter))
            ):
                relocal_residual_step = self._compute_relocalization_residual_step(
                    temporal_features=temporal_features,
                    relocal_mask_nt=relocal_mask_nt,
                    relocal_conf_nt=relocal_conf_nt,
                )
                if isinstance(relocal_residual_step, torch.Tensor) and relocal_residual_step.shape == tracks.shape:
                    step = relocal_residual_step.to(device=tracks.device, dtype=tracks.dtype)
                    tracks = torch.clamp(tracks + step, 0.0, 1.0)
                    tracks = self._align_tracks_to_query(tracks, init_positions, query_t)
            if isinstance(relocal_conf_nt, torch.Tensor):
                last_relocal_conf_nt = relocal_conf_nt

            # 5) Occlusion-aware correction
            if self.use_occlusion and _has_valid_band_features(freq_info.get("band_features")):
                if self.use_semantic:
                    semantic_at_tracks = self._sample_features(
                        semantic_features, tracks.permute(0, 2, 1, 3).contiguous()
                    )
                else:
                    semantic_at_tracks = semantic_point_features

                occ_prob, pred_positions, confidence_pred = self.occlusion_predictor(
                    freq_info["band_features"],
                    semantic_at_tracks,
                    tracks.permute(0, 2, 1, 3).contiguous(),  # (B,T,N,2)
                )
                confidence = confidence_pred.permute(0, 2, 1).contiguous()
                apply_track_fusion = (
                    self.occlusion_track_fusion_train if self.training else self.occlusion_track_fusion_eval
                )
                if apply_track_fusion:
                    occ_mask = occ_prob.unsqueeze(-1).permute(0, 2, 1, 3)  # (B,N,T,1)
                    pred_positions_transposed = pred_positions.permute(0, 2, 1, 3).contiguous()
                    tracks = (1.0 - occ_mask) * tracks + occ_mask * pred_positions_transposed
                    tracks = torch.clamp(tracks, 0.0, 1.0)
                    tracks = self._align_tracks_to_query(tracks, init_positions, query_t)
                visibility = 1.0 - occ_prob.permute(0, 2, 1).float().contiguous()
            elif self.use_occlusion and not self._warned_missing_band_features:
                logger.warning("Occlusion predictor skipped: missing or empty band_features.")
                self._warned_missing_band_features = True

            if (
                self.focus_corrections_enabled
                and (self.focus_corrections_apply_in_train or (not self.training))
                and isinstance(base_tracks, torch.Tensor)
                and base_tracks.shape == tracks.shape
            ):
                focus_mask_nt = self._build_focus_correction_mask(
                    tracks=tracks,
                    query_t=query_t,
                    relocal_mask_nt=relocal_mask_nt,
                    corr_conf_nt=corr_conf if isinstance(corr_conf, torch.Tensor) else None,
                    relocal_conf_nt=relocal_conf_nt if isinstance(relocal_conf_nt, torch.Tensor) else None,
                    base_visibility_nt=_base_visibility if isinstance(_base_visibility, torch.Tensor) else None,
                )
                if isinstance(focus_mask_nt, torch.Tensor):
                    non_focus_scale = float(self.focus_corrections_non_focus_scale)
                    focus_scale = torch.full_like(tracks[..., 0], non_focus_scale)
                    focus_scale = torch.where(
                        focus_mask_nt.to(device=tracks.device),
                        torch.ones_like(focus_scale),
                        focus_scale,
                    )
                    delta_focus = (tracks - base_tracks) * focus_scale.unsqueeze(-1)
                    tracks = torch.clamp(base_tracks + delta_focus, 0.0, 1.0)
                    tracks = self._align_tracks_to_query(tracks, init_positions, query_t)

            if (
                self.delta_max_total is not None
                and self.delta_max_total > 0
                and isinstance(base_tracks, torch.Tensor)
                and base_tracks.shape == tracks.shape
            ):
                delta_total = tracks - base_tracks
                base_max_total = float(self.delta_max_total)

                # Optionally allow a larger trust region for explicit re-localization frames.
                # This makes the "global search at re-appearance" actually useful: if we
                # keep the same clamp as normal refinement, large corrections are impossible.
                max_total_map = torch.full_like(delta_total[..., :1], base_max_total)
                if (
                    self.relocalization_enabled
                    and self.relocalization_max_total > 0
                    and isinstance(relocal_mask_nt, torch.Tensor)
                    and relocal_mask_nt.shape == tracks[..., 0].shape
                ):
                    relocal_max_total = float(self.relocalization_max_total)
                    if relocal_max_total > base_max_total:
                        mask = relocal_mask_nt.to(device=tracks.device)
                        max_total_map = torch.where(
                            mask.unsqueeze(-1),
                            torch.full_like(max_total_map, relocal_max_total),
                            max_total_map,
                        )

                # Optional: adapt clamp based on correlation confidence (including relocal confidence).
                conf_for_clamp = None
                if self.delta_max_total_adaptive_enabled:
                    if (
                        corr_conf is not None
                        and isinstance(corr_conf, torch.Tensor)
                        and corr_conf.shape == tracks[..., 0].shape
                    ):
                        conf_for_clamp = corr_conf.detach()
                    if (
                        relocal_conf_nt is not None
                        and isinstance(relocal_conf_nt, torch.Tensor)
                        and relocal_conf_nt.shape == tracks[..., 0].shape
                    ):
                        conf_for_clamp = (
                            relocal_conf_nt.detach()
                            if conf_for_clamp is None
                            else torch.maximum(conf_for_clamp, relocal_conf_nt.detach())
                        )

                if conf_for_clamp is not None:
                    conf = conf_for_clamp.to(device=tracks.device, dtype=tracks.dtype)
                    gate = _confidence_gate(
                        conf,
                        threshold=float(self.delta_max_total_adaptive_threshold),
                        power=float(self.delta_max_total_adaptive_power),
                    )
                    min_scale = float(self.delta_max_total_adaptive_min_scale)
                    max_scale = float(self.delta_max_total_adaptive_max_scale)
                    scale = min_scale + (max_scale - min_scale) * gate
                    max_total_map = max_total_map * scale.unsqueeze(-1)

                delta_total = torch.max(torch.min(delta_total, max_total_map), -max_total_map)
                tracks = torch.clamp(base_tracks + delta_total, 0.0, 1.0)
                tracks = self._align_tracks_to_query(tracks, init_positions, query_t)

            if return_iter_tracks and iter_tracks is not None:
                iter_tracks.append(tracks)

        retracking_info: Optional[Dict[str, torch.Tensor]] = None
        tracks_pre_accept: Optional[torch.Tensor] = None
        relocal_acceptor: Optional[torch.Tensor] = None
        relocal_accept_mask: Optional[torch.Tensor] = None
        verifier_decisions: Optional[torch.Tensor] = None
        if (
            self.relocal_acceptor_enabled
            and isinstance(self.relocal_acceptor_head, nn.Module)
            and isinstance(base_tracks, torch.Tensor)
            and base_tracks.shape == tracks.shape
            and isinstance(last_temporal_features, torch.Tensor)
            and isinstance(relocal_mask_nt, torch.Tensor)
            and relocal_mask_nt.shape == tracks[..., 0].shape
        ):
            accept_inputs: List[torch.Tensor] = [last_temporal_features]
            B, T, N, _ = last_temporal_features.shape

            def _zeros_bt_accept() -> torch.Tensor:
                return torch.zeros(
                    (B, T, N, 1),
                    device=last_temporal_features.device,
                    dtype=last_temporal_features.dtype,
                )

            if "delta_norm" in self.relocal_acceptor_extra:
                delta_norm = torch.norm((tracks - base_tracks), dim=-1, keepdim=True)
                accept_inputs.append(delta_norm.permute(0, 2, 1, 3).contiguous())
            if "corr_conf" in self.relocal_acceptor_extra:
                if isinstance(last_corr_conf, torch.Tensor) and last_corr_conf.shape == tracks[..., 0].shape:
                    accept_inputs.append(last_corr_conf.permute(0, 2, 1).unsqueeze(-1).contiguous())
                else:
                    accept_inputs.append(_zeros_bt_accept())
            if "relocal_conf" in self.relocal_acceptor_extra:
                if isinstance(last_relocal_conf_nt, torch.Tensor) and last_relocal_conf_nt.shape == tracks[..., 0].shape:
                    accept_inputs.append(last_relocal_conf_nt.permute(0, 2, 1).unsqueeze(-1).contiguous())
                else:
                    accept_inputs.append(_zeros_bt_accept())
            if "base_vis" in self.relocal_acceptor_extra:
                if isinstance(_base_visibility, torch.Tensor) and _base_visibility.shape == tracks[..., 0].shape:
                    accept_inputs.append(_base_visibility.permute(0, 2, 1).unsqueeze(-1).contiguous())
                else:
                    accept_inputs.append(_zeros_bt_accept())
            if "relocal_mask" in self.relocal_acceptor_extra:
                accept_inputs.append(
                    relocal_mask_nt.to(device=last_temporal_features.device, dtype=last_temporal_features.dtype)
                    .permute(0, 2, 1)
                    .unsqueeze(-1)
                    .contiguous()
                )

            accept_in = accept_inputs[0] if len(accept_inputs) == 1 else torch.cat(accept_inputs, dim=-1)
            accept_logits = self.relocal_acceptor_head(accept_in)
            accept_bt = torch.sigmoid(accept_logits).clamp(0.0, 1.0)
            accept_nt = accept_bt.permute(0, 2, 1, 3).contiguous()
            tracks_pre_accept = tracks
            relocal_accept_mask = relocal_mask_nt.to(device=tracks.device)
            verifier_decisions = (
                accept_nt.squeeze(-1) >= float(self.relocal_acceptor_threshold)
            ) & relocal_accept_mask
            if self.relocal_acceptor_apply_mode == "hard":
                accepted_tracks = torch.where(
                    verifier_decisions.unsqueeze(-1),
                    tracks,
                    base_tracks,
                )
            else:
                accepted_tracks = torch.clamp(base_tracks + accept_nt * (tracks - base_tracks), 0.0, 1.0)
            tracks = torch.where(relocal_accept_mask.unsqueeze(-1), accepted_tracks, tracks)
            tracks = self._align_tracks_to_query(tracks, init_positions, query_t)
            relocal_acceptor = accept_nt.squeeze(-1)

        tracks_pre_gate: Optional[torch.Tensor] = None
        policy_gate: Optional[torch.Tensor] = None
        if (
            self.policy_gate_enabled
            and isinstance(self.policy_gate_head, nn.Module)
            and isinstance(base_tracks, torch.Tensor)
            and base_tracks.shape == tracks.shape
            and isinstance(last_temporal_features, torch.Tensor)
        ):
            gate_inputs: List[torch.Tensor] = [last_temporal_features]
            B, T, N, _ = last_temporal_features.shape

            def _zeros_bt() -> torch.Tensor:
                return torch.zeros(
                    (B, T, N, 1),
                    device=last_temporal_features.device,
                    dtype=last_temporal_features.dtype,
                )

            if "delta_norm" in self.policy_gate_extra:
                delta_norm = torch.norm((tracks - base_tracks), dim=-1, keepdim=True)  # (B,N,T,1)
                gate_inputs.append(delta_norm.permute(0, 2, 1, 3).contiguous())
            if "corr_conf" in self.policy_gate_extra:
                if isinstance(last_corr_conf, torch.Tensor) and last_corr_conf.shape == tracks[..., 0].shape:
                    gate_inputs.append(last_corr_conf.permute(0, 2, 1).unsqueeze(-1).contiguous())
                else:
                    gate_inputs.append(_zeros_bt())
            if "relocal_conf" in self.policy_gate_extra:
                if (
                    isinstance(last_relocal_conf_nt, torch.Tensor)
                    and last_relocal_conf_nt.shape == tracks[..., 0].shape
                ):
                    gate_inputs.append(last_relocal_conf_nt.permute(0, 2, 1).unsqueeze(-1).contiguous())
                else:
                    gate_inputs.append(_zeros_bt())
            if "base_vis" in self.policy_gate_extra:
                if isinstance(_base_visibility, torch.Tensor) and _base_visibility.shape == tracks[..., 0].shape:
                    gate_inputs.append(_base_visibility.permute(0, 2, 1).unsqueeze(-1).contiguous())
                else:
                    gate_inputs.append(_zeros_bt())

            gate_in = gate_inputs[0] if len(gate_inputs) == 1 else torch.cat(gate_inputs, dim=-1)
            gate_logits = self.policy_gate_head(gate_in)
            gate_bt = torch.sigmoid(gate_logits).clamp(0.0, 1.0)
            gate_nt = gate_bt.permute(0, 2, 1, 3).contiguous()
            tracks_pre_gate = tracks
            tracks = torch.clamp(base_tracks + gate_nt * (tracks - base_tracks), 0.0, 1.0)
            tracks = self._align_tracks_to_query(tracks, init_positions, query_t)
            policy_gate = gate_nt.squeeze(-1)

        if (
            self.retracking_enabled
            and isinstance(base_tracks, torch.Tensor)
            and base_tracks.shape == tracks.shape
            and isinstance(relocal_anchor_tracks, torch.Tensor)
        ):
            try:
                tracks_pre_retracking = tracks.detach().clone()
                tracks, retracking_info = self._maybe_apply_retracking(
                    video=video,
                    tracks=tracks,
                    query_t=query_t,
                    relocal_mask_nt=relocal_mask_nt,
                    relocal_anchor_tracks=relocal_anchor_tracks,
                    relocal_conf_nt=last_relocal_conf_nt,
                )
                tracks = self._align_tracks_to_query(tracks, init_positions, query_t)
            except Exception as exc:
                logger.warning(f"Retracking splice failed; keeping pre-retracking tracks: {exc}")
                retracking_info = None

        if visibility is None:
            visibility = _base_visibility

        # Optional: fuse base tracker visibility as a prior (paper-friendly ablation knob).
        mode = self.visibility_prior_mode
        if (
            mode not in ("none", "", "off", "disabled")
            and isinstance(visibility, torch.Tensor)
            and isinstance(_base_visibility, torch.Tensor)
            and visibility.shape == _base_visibility.shape
        ):
            base_vis = _base_visibility.to(device=visibility.device, dtype=visibility.dtype)
            vis = visibility
            if mode in ("blend", "lerp", "mix"):
                alpha = float(self.visibility_prior_alpha)
                alpha = 0.0 if alpha < 0 else (1.0 if alpha > 1 else alpha)
                vis = (1.0 - alpha) * vis + alpha * base_vis
            elif mode in ("multiply", "mul", "gate"):
                vis = vis * base_vis
            elif mode in ("min", "minimum"):
                vis = torch.minimum(vis, base_vis)
            elif mode in ("max", "maximum"):
                vis = torch.maximum(vis, base_vis)
            elif mode in ("logit_add", "logit"):
                strength = float(self.visibility_prior_strength)
                eps = float(self.visibility_prior_eps)
                eps = 1e-6 if eps <= 0 else (0.1 if eps > 0.1 else eps)
                vis_logit = torch.logit(vis.clamp(eps, 1.0 - eps))
                base_logit = torch.logit(base_vis.clamp(eps, 1.0 - eps))
                vis = torch.sigmoid(vis_logit + strength * base_logit)
            else:
                logger.warning(
                    f"Unknown refiner.visibility_prior.mode={mode!r}; ignoring."
                )
            visibility = vis.clamp(0.0, 1.0)

        if return_info:
            info: Dict[str, object] = {"freq_info": last_freq_info}
            if last_semantic_points is not None and self.use_semantic:
                info["semantic_features"] = last_semantic_points
            if return_iter_tracks and iter_tracks is not None:
                info["iter_tracks"] = iter_tracks
            if confidence is not None:
                info["confidence"] = confidence
            if last_corr_logits is not None:
                info["corr_logits"] = last_corr_logits
            if last_corr_center_tracks_bt is not None:
                info["corr_center_tracks_bt"] = last_corr_center_tracks_bt
            if last_corr_feature_hw is not None:
                info["corr_feature_hw"] = last_corr_feature_hw
            if last_corr_window_size is not None:
                info["corr_window_size"] = last_corr_window_size
            info["base_tracks"] = base_tracks
            info["base_visibility"] = _base_visibility
            if relocal_mask_nt is not None:
                info["relocal_mask"] = relocal_mask_nt
                info["relocalization_mask"] = relocal_mask_nt
            if isinstance(last_relocal_conf_nt, torch.Tensor):
                info["relocal_conf"] = last_relocal_conf_nt
                info["relocalization_conf"] = last_relocal_conf_nt
            if isinstance(relocal_anchor_tracks, torch.Tensor):
                info["relocal_anchor_tracks"] = relocal_anchor_tracks
            if isinstance(relocal_anchor_conf_nt, torch.Tensor):
                info["relocal_anchor_conf"] = relocal_anchor_conf_nt
            if tracks_pre_accept is not None:
                info["pre_accept_tracks"] = tracks_pre_accept
            if isinstance(tracks_pre_retracking, torch.Tensor):
                info["pre_retracking_tracks"] = tracks_pre_retracking
            if relocal_acceptor is not None:
                info["relocal_acceptor"] = relocal_acceptor
                info["verifier_scores"] = relocal_acceptor
            if relocal_accept_mask is not None:
                info["relocal_accept_mask"] = relocal_accept_mask
                info["verifier_mask"] = relocal_accept_mask
            if verifier_decisions is not None:
                info["verifier_decisions"] = verifier_decisions
            info["verifier_threshold"] = float(self.relocal_acceptor_threshold)
            info["verifier_apply_mode"] = str(self.relocal_acceptor_apply_mode)
            info["online_recovery_requested"] = bool(self.online_recovery_requested)
            info["online_recovery_mode"] = str(self.online_recovery_mode)
            info["online_recovery_feature_source"] = str(self.relocalization_feature_source)
            info["online_recovery_template_mode"] = str(self.relocalization_template_mode)
            info["online_recovery_min_occlusion_len"] = int(self.relocalization_min_occlusion_len)
            info["online_recovery_frames_after"] = int(self.relocalization_frames_after)
            info["online_recovery_vis_threshold"] = float(self.relocalization_vis_threshold)
            info["online_recovery_template_bank_size"] = int(self.relocalization_template_bank_size)
            info["online_recovery_summary"] = dict(self.online_recovery_summary)
            if retracking_info is not None:
                info["retracking"] = retracking_info
                for key, value in retracking_info.items():
                    info[f"retracking_{key}"] = value
            if tracks_pre_gate is not None:
                info["pre_gate_tracks"] = tracks_pre_gate
            if policy_gate is not None:
                info["policy_gate"] = policy_gate
            if self.framewise_init_enabled and isinstance(framewise_feature_map, torch.Tensor):
                if "framewise_init_tracks" in locals():
                    info["framewise_init_tracks"] = framewise_init_tracks
            return tracks, visibility, info

        return tracks, visibility


def create_cotracker_refiner(config) -> CoTrackerFSPTRefiner:
    return CoTrackerFSPTRefiner(config)
