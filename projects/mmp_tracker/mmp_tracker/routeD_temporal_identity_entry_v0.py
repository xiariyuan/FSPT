"""Causal full-population entry features for Route-D Gate 3C1F0."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np
import torch

ENTRY_FEATURE_SCHEMA_VERSION: Final = (
    "routeD_temporal_identity_full_population_entry_features_gate3c1f0_v0"
)


@dataclass(frozen=True)
class EntryFeatureConfig:
    trajectory_frames: int = 8
    trajectory_channels: int = 9
    memory_levels: int = 4
    feature_dim: int = 130

    def __post_init__(self) -> None:
        if (self.trajectory_frames, self.trajectory_channels) != (8, 9):
            raise ValueError("Gate 3C1F0 trajectory contract drift")
        if self.memory_levels != 4 or self.feature_dim != 130:
            raise ValueError("Gate 3C1F0 entry feature contract drift")


def _numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _memory_summary(value: torch.Tensor | np.ndarray) -> np.ndarray:
    array = _numpy(value).astype(np.float32, copy=False)
    if array.ndim < 2:
        raise ValueError("entry memory tensor must retain a row axis")
    flat = array.reshape(array.shape[0], -1)
    scale = np.sqrt(float(flat.shape[1]))
    return np.column_stack(
        [
            flat.mean(axis=1),
            flat.std(axis=1),
            flat.min(axis=1),
            flat.max(axis=1),
            np.linalg.norm(flat, axis=1) / scale,
        ]
    ).astype(np.float32)


def build_entry_features(
    *,
    trajectory_features: torch.Tensor | np.ndarray,
    visibility_probability: torch.Tensor | np.ndarray,
    confidence_probability: torch.Tensor | np.ndarray,
    native_track_features: Sequence[torch.Tensor | np.ndarray],
    native_track_supports: Sequence[torch.Tensor | np.ndarray],
    config: EntryFeatureConfig = EntryFeatureConfig(),
) -> np.ndarray:
    """Build 130-D observed-history entry features with no teacher/future input."""
    trajectory = _numpy(trajectory_features).astype(np.float32, copy=False)
    visibility = _numpy(visibility_probability).astype(np.float32, copy=False).reshape(-1)
    confidence = _numpy(confidence_probability).astype(np.float32, copy=False).reshape(-1)
    if trajectory.ndim != 3 or trajectory.shape[1:] != (
        config.trajectory_frames,
        config.trajectory_channels,
    ):
        raise ValueError("entry trajectory shape drift")
    rows = trajectory.shape[0]
    if visibility.shape != (rows,) or confidence.shape != (rows,):
        raise ValueError("entry probability shape drift")
    if len(native_track_features) != config.memory_levels or len(native_track_supports) != config.memory_levels:
        raise ValueError("entry memory-level count drift")
    if not (
        np.isfinite(trajectory).all()
        and np.isfinite(visibility).all()
        and np.isfinite(confidence).all()
    ):
        raise ValueError("entry input is non-finite")
    if np.any(visibility < 0.0) or np.any(visibility > 1.0) or np.any(confidence < 0.0) or np.any(confidence > 1.0):
        raise ValueError("entry probability range drift")

    xy = trajectory[..., :2]
    delta_xy = trajectory[..., 2:4]
    speed = np.linalg.norm(delta_xy, axis=2)
    historical_joint = trajectory[..., 4] * trajectory[..., 5]
    compact = np.column_stack(
        [
            speed.mean(axis=1),
            speed.std(axis=1),
            speed.max(axis=1),
            np.linalg.norm(xy[:, -1] - xy[:, 0], axis=1),
            trajectory[..., 4].min(axis=1),
            trajectory[..., 4].mean(axis=1),
            trajectory[..., 4].std(axis=1),
            trajectory[..., 5].min(axis=1),
            trajectory[..., 5].mean(axis=1),
            trajectory[..., 5].std(axis=1),
            historical_joint.min(axis=1),
            historical_joint.mean(axis=1),
            historical_joint[:, -1],
            trajectory[..., -1].mean(axis=1),
            trajectory[..., -1].max(axis=1),
            trajectory[:, -1, -1],
        ]
    ).astype(np.float32)
    parts = [trajectory.reshape(rows, -1), visibility[:, None], confidence[:, None], compact]
    for value in (*native_track_features, *native_track_supports):
        summary = _memory_summary(value)
        if summary.shape[0] != rows:
            raise ValueError("entry memory row drift")
        parts.append(summary)
    output = np.concatenate(parts, axis=1).astype(np.float32)
    if output.shape != (rows, config.feature_dim) or not np.isfinite(output).all():
        raise ValueError("entry feature output drift")
    return output


def entry_action_mask(
    *,
    entry_probability: np.ndarray,
    native_joint_probability: np.ndarray,
    probability_min: float = 0.93,
    joint_probability_max: float = 0.02,
) -> np.ndarray:
    probability = np.asarray(entry_probability, dtype=np.float64)
    joint = np.asarray(native_joint_probability, dtype=np.float64)
    if probability.ndim != 1 or joint.shape != probability.shape:
        raise ValueError("entry action probability shape drift")
    if not np.isfinite(probability).all() or not np.isfinite(joint).all():
        raise ValueError("entry action probability is non-finite")
    if np.any(probability < 0.0) or np.any(probability > 1.0) or np.any(joint < 0.0) or np.any(joint > 1.0):
        raise ValueError("entry action probability range drift")
    return (probability >= float(probability_min)) & (joint <= float(joint_probability_max))
