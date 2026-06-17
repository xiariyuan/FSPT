from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import PointTrackingAugmentation, TemporalAugmentation

logger = logging.getLogger(__name__)

try:
    from omegaconf import OmegaConf

    OMEGACONF_AVAILABLE = True
except Exception:
    OmegaConf = None
    OMEGACONF_AVAILABLE = False


def _normalize_aug_config(augmentation: Optional[object]) -> Dict[str, Any]:
    if augmentation is None:
        return {"enabled": False}
    if isinstance(augmentation, bool):
        return {"enabled": bool(augmentation)}
    if OMEGACONF_AVAILABLE and OmegaConf is not None:
        try:
            if OmegaConf.is_config(augmentation):
                cfg = OmegaConf.to_container(augmentation, resolve=True)
            else:
                cfg = augmentation
        except Exception:
            cfg = augmentation
    else:
        cfg = augmentation
    if isinstance(cfg, dict):
        cfg = dict(cfg)
    else:
        try:
            cfg = dict(cfg)
        except Exception:
            cfg = {"enabled": True}
    cfg.setdefault("enabled", True)
    return cfg


def _ensure_query_visible_indices(
    query_points: torch.Tensor,
    occluded: torch.Tensor,
    num_points: Optional[int],
    rng: Optional[np.random.RandomState] = None,
) -> Optional[torch.Tensor]:
    """
    Replicates TAP-Vid datasets' _ensure_query_visible selection, but returns indices.
    """
    if not isinstance(query_points, torch.Tensor) or not isinstance(occluded, torch.Tensor):
        return None
    if query_points.numel() == 0 or occluded.numel() == 0:
        return None
    if occluded.dim() != 2:
        return None
    if query_points.dim() != 2 or query_points.shape[-1] != 3:
        return None
    N = int(occluded.shape[0])
    T = int(occluded.shape[1])
    if N == 0 or T == 0 or int(query_points.shape[0]) != N:
        return None

    query_t = query_points[:, 0].round().long().clamp(0, T - 1)
    point_idx = torch.arange(N, device=occluded.device)
    visible_mask = ~occluded[point_idx, query_t]
    if bool(visible_mask.all().item()):
        return None
    valid_idx = torch.nonzero(visible_mask, as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        return None

    if num_points is not None and int(num_points) > 0:
        k = int(num_points)
        if valid_idx.numel() >= k:
            if rng is not None:
                local = rng.choice(int(valid_idx.numel()), k, replace=False)
                perm = torch.as_tensor(local, device=valid_idx.device, dtype=torch.long)
            else:
                perm = torch.randperm(valid_idx.numel(), device=valid_idx.device)[:k]
            chosen = valid_idx[perm]
        else:
            if rng is not None:
                local = rng.randint(0, int(valid_idx.numel()), size=(k,))
                rand = torch.as_tensor(local, device=valid_idx.device, dtype=torch.long)
            else:
                rand = torch.randint(0, valid_idx.numel(), (k,), device=valid_idx.device)
            chosen = valid_idx[rand]
        return chosen
    return valid_idx


def _apply_temporal_params_to_visibility(
    visibility: torch.Tensor,
    params: Dict[str, Any],
) -> torch.Tensor:
    if not isinstance(visibility, torch.Tensor) or visibility.numel() == 0:
        return visibility
    if visibility.dim() != 2:
        return visibility
    T = int(visibility.shape[1])
    if T <= 1:
        return visibility

    out = visibility
    if bool(params.get("reverse", False)):
        out = torch.flip(out, dims=[1])

    speed = float(params.get("speed", 1.0) or 1.0)
    if abs(speed - 1.0) > 1e-3:
        device = out.device
        time = torch.linspace(0, T - 1, T, device=device)
        center = (T - 1) / 2.0
        src_times = (time - center) * speed + center
        src_times = src_times.clamp(0, T - 1)

        t0 = torch.floor(src_times).long()
        t1 = torch.clamp(t0 + 1, max=T - 1)
        alpha = (src_times - t0.float()).view(1, T)

        out = (1.0 - alpha) * out[:, t0] + alpha * out[:, t1]

    offset = int(params.get("offset", 0) or 0)
    if offset != 0 and T > 1:
        out = torch.roll(out, shifts=-offset, dims=1)

    return out


class AugmentationAfterBaseTracksWrapper(Dataset):
    """
    Apply point-tracking augmentation *after* base tracks injection.

    Motivation:
    - Base tracks are precomputed on deterministic query_points.
    - Dataset-level random augmentation would make cached base tracks inconsistent.
    - This wrapper augments {video, query_points, target_points, occluded} and also
      {base_tracks, base_visibility} in a consistent way.
    """

    def __init__(self, dataset: Dataset, augmentation: Optional[object]) -> None:
        self.dataset = dataset
        self.aug_cfg = _normalize_aug_config(augmentation)
        self.enabled = bool(self.aug_cfg.get("enabled", False))

        self.num_points = getattr(dataset, "num_points", None)
        self.spatial_aug: Optional[PointTrackingAugmentation] = None
        self.temporal_aug: Optional[TemporalAugmentation] = None

        if not self.enabled:
            return

        cfg = self.aug_cfg
        random_scale = cfg.get("random_scale", None)
        if isinstance(random_scale, list):
            random_scale = tuple(random_scale)
        resolution = getattr(dataset, "resolution", None)
        crop_size = cfg.get("crop_size", resolution)
        if isinstance(crop_size, list):
            crop_size = tuple(crop_size)
        if crop_size is None:
            crop_size = (256, 256)
        random_rotation = float(cfg.get("random_rotation", 0.0) or 0.0)

        self.spatial_aug = PointTrackingAugmentation(
            random_crop=bool(cfg.get("random_crop", True)),
            random_flip=bool(cfg.get("random_flip", True)),
            color_jitter=float(cfg.get("color_jitter", 0.4) or 0.0),
            random_scale=random_scale,
            random_rotation=random_rotation,
            crop_size=crop_size,
        )

        temporal_cfg = cfg.get("temporal", {})
        if not isinstance(temporal_cfg, dict):
            temporal_cfg = {}
        temporal_enabled = bool(temporal_cfg.get("enabled", False))
        if any(k in cfg for k in ["random_reverse", "random_speed", "random_start"]):
            temporal_enabled = True
        if temporal_enabled:
            random_speed = temporal_cfg.get("random_speed", cfg.get("random_speed", (0.5, 2.0)))
            if isinstance(random_speed, list):
                random_speed = tuple(random_speed)
            self.temporal_aug = TemporalAugmentation(
                random_reverse=bool(temporal_cfg.get("random_reverse", cfg.get("random_reverse", True))),
                random_speed=random_speed,
                random_start=bool(temporal_cfg.get("random_start", cfg.get("random_start", True))),
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]
        if not self.enabled or not isinstance(sample, dict):
            return sample

        video = sample.get("video", None)
        query_points = sample.get("query_points", None)
        target_points = sample.get("target_points", None)
        occluded = sample.get("occluded", None)
        if not all(isinstance(v, torch.Tensor) for v in [video, query_points, target_points, occluded]):
            return sample

        base_tracks = sample.get("base_tracks", None)
        base_visibility = sample.get("base_visibility", None)
        has_base = isinstance(base_tracks, torch.Tensor) and isinstance(base_visibility, torch.Tensor)

        # Spatial augmentation (shared for target + base tracks by concatenating along N).
        if self.spatial_aug is not None:
            if has_base:
                N = int(target_points.shape[0])
                if base_tracks.shape[:2] != target_points.shape[:2]:
                    # Unsafe to augment cached base tracks if shapes do not align.
                    logger.warning(
                        "base_tracks shape mismatch; dropping cached base tracks for this sample. "
                        f"target={tuple(target_points.shape)} base={tuple(base_tracks.shape)}"
                    )
                    has_base = False
                else:
                    base_occ0 = torch.zeros_like(occluded, dtype=torch.bool)
                    all_points = torch.cat([target_points, base_tracks], dim=0)
                    all_occluded = torch.cat([occluded, base_occ0], dim=0)
                    all_query = torch.cat([query_points, query_points], dim=0)
                    out = self.spatial_aug(video, all_points, all_occluded, all_query)
                    if len(out) == 4:
                        video, all_points, all_occluded, all_query = out
                    else:
                        video, all_points, all_occluded = out
                        all_query = all_query
                    target_points = all_points[:N]
                    base_tracks = all_points[N:]
                    occluded = all_occluded[:N]
                    base_oob = all_occluded[N:]
                    query_points = all_query[:N]

                    # base_visibility keeps original values but zeros out of bounds.
                    if base_visibility.dtype == torch.bool:
                        base_visibility = base_visibility & (~base_oob)
                    else:
                        base_visibility = base_visibility.float() * (~base_oob).float()
            if not has_base:
                out = self.spatial_aug(video, target_points, occluded, query_points)
                if len(out) == 4:
                    video, target_points, occluded, query_points = out
                else:
                    video, target_points, occluded = out

        # Temporal augmentation (apply once; propagate params to base_visibility).
        if self.temporal_aug is not None:
            # Sample temporal params ourselves to keep base_visibility consistent.
            T = int(video.shape[0])
            reverse = bool(self.temporal_aug.random_reverse and np.random.rand() > 0.5)
            speed = 1.0
            if self.temporal_aug.random_speed is not None:
                try:
                    speed = float(np.random.uniform(*self.temporal_aug.random_speed))
                except Exception:
                    speed = 1.0
            offset = 0
            if bool(self.temporal_aug.random_start) and T > 1:
                offset = int(np.random.randint(0, T))
            params = {"reverse": reverse, "speed": speed, "offset": offset}

            # Apply to video/points/occluded/query_points using the same code path.
            # (We re-use TemporalAugmentation internals by temporarily configuring its randomness.)
            # NOTE: we do not call self.temporal_aug directly because it samples its own params.
            if reverse:
                video = torch.flip(video, dims=[0])
                target_points = torch.flip(target_points, dims=[1])
                occluded = torch.flip(occluded, dims=[1])
                query_points = query_points.clone()
                query_points[:, 0] = T - 1 - query_points[:, 0]
                if has_base:
                    base_tracks = torch.flip(base_tracks, dims=[1])

            if abs(speed - 1.0) > 1e-3 and T > 1:
                device = video.device
                time = torch.linspace(0, T - 1, T, device=device)
                center = (T - 1) / 2.0
                src_times = (time - center) * speed + center
                src_times = src_times.clamp(0, T - 1)

                t0 = torch.floor(src_times).long()
                t1 = torch.clamp(t0 + 1, max=T - 1)
                alpha = (src_times - t0.float()).view(T, 1, 1, 1)

                video = (1.0 - alpha) * video[t0] + alpha * video[t1]
                alpha_pts = alpha.view(1, T, 1)
                target_points = (1.0 - alpha_pts) * target_points[:, t0, :] + alpha_pts * target_points[:, t1, :]
                if has_base:
                    base_tracks = (1.0 - alpha_pts) * base_tracks[:, t0, :] + alpha_pts * base_tracks[:, t1, :]

                choose_t1 = (src_times - t0.float()) > 0.5
                occluded = torch.where(choose_t1.view(1, T), occluded[:, t1], occluded[:, t0])

                query_points = query_points.clone()
                q_t = query_points[:, 0].view(-1, 1)
                new_t = torch.abs(src_times.view(1, T) - q_t).argmin(dim=1).float()
                query_points[:, 0] = new_t
                new_t_idx = new_t.long().clamp(0, T - 1)
                point_idx = torch.arange(target_points.shape[0], device=target_points.device)
                query_points[:, 1] = target_points[point_idx, new_t_idx, 0]
                query_points[:, 2] = target_points[point_idx, new_t_idx, 1]

            if offset != 0 and T > 1:
                video = torch.roll(video, shifts=-offset, dims=0)
                target_points = torch.roll(target_points, shifts=-offset, dims=1)
                occluded = torch.roll(occluded, shifts=-offset, dims=1)
                query_points = query_points.clone()
                query_points[:, 0] = (query_points[:, 0] - offset) % T
                if has_base:
                    base_tracks = torch.roll(base_tracks, shifts=-offset, dims=1)

            # Ensure query_points' (y,x) always matches target_points at the query frame after
            # any temporal transforms (reverse/speed/offset). This keeps model alignment stable.
            query_points = query_points.clone()
            q_t_idx = query_points[:, 0].round().long().clamp(0, T - 1)
            point_idx = torch.arange(target_points.shape[0], device=target_points.device)
            query_points[:, 1] = target_points[point_idx, q_t_idx, 0]
            query_points[:, 2] = target_points[point_idx, q_t_idx, 1]

            if has_base:
                base_visibility = _apply_temporal_params_to_visibility(
                    base_visibility.float() if base_visibility.dtype != torch.bool else base_visibility.float(),
                    params,
                )

        # Final clamping (datasets do this after augmentation).
        query_points = query_points.clone()
        query_points[:, 0] = torch.clamp(query_points[:, 0], 0, video.shape[0] - 1)
        query_points[:, 1:3] = torch.clamp(query_points[:, 1:3], 0.0, 1.0)

        # Ensure query frame is visible; apply the same selection to cached base tracks.
        chosen = _ensure_query_visible_indices(
            query_points=query_points,
            occluded=occluded,
            num_points=self.num_points,
            rng=None,
        )
        if chosen is not None:
            query_points = query_points[chosen]
            target_points = target_points[chosen]
            occluded = occluded[chosen]
            if has_base:
                base_tracks = base_tracks[chosen]
                base_visibility = base_visibility[chosen]

        # Write back.
        sample["video"] = video
        sample["query_points"] = query_points
        sample["target_points"] = target_points
        sample["occluded"] = occluded
        if has_base:
            sample["base_tracks"] = base_tracks
            sample["base_visibility"] = base_visibility
        return sample
