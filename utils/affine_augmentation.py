from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _cfg_get(cfg: Optional[object], key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def normalize_affine_cfg(cfg: Optional[object]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    raw = cfg
    if raw is None:
        raw = {}
    if isinstance(raw, dict):
        out.update(raw)
    else:
        try:
            out.update(dict(raw))
        except Exception:
            out = {}

    out.setdefault("flip", True)
    out.setdefault("flip_prob", 0.5)
    out.setdefault("rotation_deg", 0.0)
    out.setdefault("scale", (1.0, 1.0))
    out.setdefault("translate", (0.0, 0.0))
    out.setdefault("align_corners", False)
    out.setdefault("padding_mode", "zeros")
    out.setdefault("mode", "bilinear")

    scale = out.get("scale", (1.0, 1.0))
    if isinstance(scale, (int, float)):
        scale = (float(scale), float(scale))
    elif isinstance(scale, list):
        scale = tuple(scale)
    if not isinstance(scale, tuple) or len(scale) != 2:
        scale = (1.0, 1.0)
    out["scale"] = (float(scale[0]), float(scale[1]))

    translate = out.get("translate", (0.0, 0.0))
    if isinstance(translate, (int, float)):
        translate = (float(translate), float(translate))
    elif isinstance(translate, list):
        translate = tuple(translate)
    if not isinstance(translate, tuple) or len(translate) != 2:
        translate = (0.0, 0.0)
    out["translate"] = (float(translate[0]), float(translate[1]))

    out["flip_prob"] = float(out.get("flip_prob", 0.5) or 0.0)
    out["rotation_deg"] = float(out.get("rotation_deg", 0.0) or 0.0)
    out["align_corners"] = bool(out.get("align_corners", False))
    out["flip"] = bool(out.get("flip", True))
    out["padding_mode"] = str(out.get("padding_mode", "zeros")).lower().strip()
    out["mode"] = str(out.get("mode", "bilinear")).lower().strip()
    return out


def invert_affine_matrices(theta: torch.Tensor) -> torch.Tensor:
    """
    Invert a batch of 2x3 affine matrices in normalized (-1..1) coordinates.

    Args:
        theta: (B, 2, 3)
    Returns:
        theta_inv: (B, 2, 3)
    """
    if not isinstance(theta, torch.Tensor) or theta.dim() != 3 or theta.shape[1:] != (2, 3):
        raise ValueError(f"Expected theta with shape (B,2,3), got {getattr(theta, 'shape', None)}")

    linear = theta[:, :2, :2]
    trans = theta[:, :2, 2]
    linear_inv = torch.linalg.inv(linear)
    trans_inv = -(linear_inv @ trans.unsqueeze(-1)).squeeze(-1)
    return torch.cat([linear_inv, trans_inv.unsqueeze(-1)], dim=-1)


def sample_affine_matrices(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    cfg: Optional[object] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample per-sample affine transforms for spatial augmentation.

    Conventions:
    - Matrices operate on normalized coordinates in [-1, 1] using (x, y).
    - theta_orig2aug maps original -> augmented.
    - theta_aug2orig maps augmented -> original (inverse).

    Args:
        batch_size: number of samples.
        device: output device.
        dtype: output dtype for theta.
        cfg: dict-like config with keys:
             - flip (bool), flip_prob (float)
             - rotation_deg (float) : max abs rotation
             - scale (min,max) : uniform scaling range
             - translate (ty,tx) : max translation fraction in [0,1] coords
    """
    cfg_n = normalize_affine_cfg(cfg)
    B = int(batch_size)
    if B <= 0:
        raise ValueError("batch_size must be positive")

    scale_min, scale_max = cfg_n["scale"]
    scale_min, scale_max = float(scale_min), float(scale_max)
    if scale_min <= 0 or scale_max <= 0:
        scale_min, scale_max = 1.0, 1.0
    if scale_max < scale_min:
        scale_min, scale_max = scale_max, scale_min

    rot_deg = float(cfg_n["rotation_deg"])
    rot_rad = rot_deg * math.pi / 180.0

    max_ty, max_tx = cfg_n["translate"]
    max_ty = max(0.0, float(max_ty))
    max_tx = max(0.0, float(max_tx))

    # Sample params.
    rand = torch.rand((B,), device=device, dtype=torch.float32, generator=generator)
    do_flip = cfg_n["flip"] and (rand < float(cfg_n["flip_prob"]))

    scale = torch.rand((B,), device=device, dtype=torch.float32, generator=generator)
    scale = scale_min + (scale_max - scale_min) * scale

    angle = torch.rand((B,), device=device, dtype=torch.float32, generator=generator)
    angle = (angle * 2.0 - 1.0) * rot_rad

    trans = torch.rand((B, 2), device=device, dtype=torch.float32, generator=generator)
    trans = trans * 2.0 - 1.0  # [-1, 1]
    # translate in fraction of [0,1] coords -> normalized [-1,1] is *2
    trans[:, 0] = trans[:, 0] * (max_ty * 2.0)
    trans[:, 1] = trans[:, 1] * (max_tx * 2.0)

    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    flip_sign = torch.where(do_flip, -torch.ones_like(scale), torch.ones_like(scale))
    sx = scale * flip_sign
    sy = scale

    # Linear part: R @ S
    a00 = cos_a * sx
    a01 = -sin_a * sy
    a10 = sin_a * sx
    a11 = cos_a * sy

    theta_orig2aug = torch.stack(
        [
            torch.stack([a00, a01, trans[:, 1]], dim=-1),
            torch.stack([a10, a11, trans[:, 0]], dim=-1),
        ],
        dim=1,
    ).to(dtype=dtype)

    theta_aug2orig = invert_affine_matrices(theta_orig2aug)
    return theta_orig2aug, theta_aug2orig


def warp_video_affine(
    video: torch.Tensor,
    theta_aug2orig: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Warp a (B,T,C,H,W) video tensor using per-sample affine transforms.

    Args:
        video: (B, T, C, H, W)
        theta_aug2orig: (B, 2, 3) mapping augmented->original (output->input for grid_sample)
    Returns:
        warped video: (B, T, C, H, W)
    """
    if video.dim() != 5:
        raise ValueError(f"Expected video of shape (B,T,C,H,W), got {video.shape}")
    B, T, C, H, W = video.shape
    if theta_aug2orig.shape != (B, 2, 3):
        raise ValueError(f"Expected theta_aug2orig shape {(B,2,3)}, got {theta_aug2orig.shape}")

    video_flat = video.reshape(B * T, C, H, W)
    theta_rep = theta_aug2orig.to(device=video.device, dtype=torch.float32).repeat_interleave(T, dim=0)
    grid = F.affine_grid(theta_rep, video_flat.shape, align_corners=align_corners)
    warped = F.grid_sample(
        video_flat,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return warped.reshape(B, T, C, H, W)


def transform_points_yx(
    points_yx: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform points in [y,x] normalized coordinates (0..1) with a per-sample affine matrix.

    Args:
        points_yx: (B, ..., 2) in [0,1] order [y,x]
        theta: (B, 2, 3) operating in normalized [-1,1] (x,y)
    Returns:
        points_out_yx: same shape as input, clamped to [0,1]
        in_bounds: (B, ...) boolean mask in augmented coords before clamping
    """
    if points_yx.dim() < 2 or points_yx.shape[-1] != 2:
        raise ValueError(f"Expected points_yx with last dim=2, got {points_yx.shape}")
    if theta.dim() != 3 or theta.shape[1:] != (2, 3):
        raise ValueError(f"Expected theta with shape (B,2,3), got {theta.shape}")

    B = theta.shape[0]
    if points_yx.shape[0] != B:
        raise ValueError(f"Batch mismatch: points B={points_yx.shape[0]} theta B={B}")

    xy = points_yx[..., [1, 0]] * 2.0 - 1.0
    ones = torch.ones_like(xy[..., :1])
    xy1 = torch.cat([xy, ones], dim=-1)  # (B,...,3)
    theta_f = theta.to(device=points_yx.device, dtype=xy1.dtype)
    out_xy = torch.einsum("bij,b...j->b...i", theta_f, xy1)

    in_bounds = (out_xy[..., 0] >= -1.0 - eps) & (out_xy[..., 0] <= 1.0 + eps)
    in_bounds = in_bounds & (out_xy[..., 1] >= -1.0 - eps) & (out_xy[..., 1] <= 1.0 + eps)

    out_y = (out_xy[..., 1] + 1.0) * 0.5
    out_x = (out_xy[..., 0] + 1.0) * 0.5
    out = torch.stack([out_y, out_x], dim=-1).clamp(0.0, 1.0)
    return out, in_bounds

