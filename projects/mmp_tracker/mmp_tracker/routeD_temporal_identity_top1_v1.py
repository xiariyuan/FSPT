"""Two-stage causal top-1 policy features for Route-D Gate 3C1D v1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch

ROW_POLICY_FEATURE_SCHEMA_VERSION: Final = (
    "routeD_temporal_identity_top1_row_policy_features_gate3c1d_v1"
)


@dataclass(frozen=True)
class RowPolicyFeatureConfig:
    shortlist_size: int = 9
    candidate_raw_feature_dim: int = 49

    def __post_init__(self) -> None:
        if self.shortlist_size != 9:
            raise ValueError("Gate 3C1D v1 shortlist size must remain nine")
        if self.candidate_raw_feature_dim != 49:
            raise ValueError("Gate 3C1D v1 raw candidate feature dimension drift")

    @property
    def output_dim(self) -> int:
        # 17 prediction summaries + selected/native/difference raw causal fields.
        return 17 + 3 * self.candidate_raw_feature_dim


def select_minimum_expected_distance(expected_distance_px: np.ndarray) -> np.ndarray:
    """Choose the stable minimum predicted-distance shortlist slot."""
    value = np.asarray(expected_distance_px)
    if value.ndim != 2 or value.shape[1] != 9:
        raise ValueError("expected distance must have shape [rows,9]")
    if not np.isfinite(value).all() or np.any(value < 0.0):
        raise ValueError("expected distance is invalid")
    # np.argmin is stable and therefore uses the lower frozen shortlist slot on ties.
    return value.argmin(axis=1).astype(np.int64)


def build_row_policy_features(
    *,
    candidate_features: torch.Tensor | np.ndarray,
    support_probability: np.ndarray,
    expected_distance_px: np.ndarray,
    selected_slot: np.ndarray,
    config: RowPolicyFeatureConfig = RowPolicyFeatureConfig(),
) -> np.ndarray:
    """Build label-free row features from candidate predictions and causal fields.

    Candidate predictions supplied here must be out-of-fold when used to train
    the row-level value/harm models. Teacher distance is intentionally absent.
    """
    features = (
        candidate_features.detach().cpu().numpy()
        if isinstance(candidate_features, torch.Tensor)
        else np.asarray(candidate_features)
    )
    support = np.asarray(support_probability, dtype=np.float64)
    expected = np.asarray(expected_distance_px, dtype=np.float64)
    slot = np.asarray(selected_slot, dtype=np.int64)
    if features.ndim != 3 or features.shape[1:] != (config.shortlist_size, 102):
        raise ValueError("candidate features must have shape [rows,9,102]")
    rows = features.shape[0]
    if support.shape != (rows, config.shortlist_size):
        raise ValueError("support probability shape drift")
    if expected.shape != support.shape or slot.shape != (rows,):
        raise ValueError("row prediction shape drift")
    if np.any(slot < 0) or np.any(slot >= config.shortlist_size):
        raise ValueError("selected slot is outside shortlist")
    if not (
        np.isfinite(features).all()
        and np.isfinite(support).all()
        and np.isfinite(expected).all()
    ):
        raise ValueError("row policy input is non-finite")
    if np.any(support < 0.0) or np.any(support > 1.0) or np.any(expected < 0.0):
        raise ValueError("row policy prediction range drift")

    row = np.arange(rows)
    selected_support = support[row, slot]
    native_support = support[:, 0]
    selected_expected = expected[row, slot]
    native_expected = expected[:, 0]
    support_sorted = np.sort(support, axis=1)
    expected_sorted = np.sort(expected, axis=1)
    entropy = -(
        support * np.log(np.clip(support, 1.0e-8, 1.0))
        + (1.0 - support) * np.log(np.clip(1.0 - support, 1.0e-8, 1.0))
    ).mean(axis=1)
    summary = np.column_stack(
        [
            selected_support,
            native_support,
            selected_support - native_support,
            support.max(axis=1),
            support_sorted[:, -1] - support_sorted[:, -2],
            support.mean(axis=1),
            support.std(axis=1),
            entropy,
            selected_expected,
            native_expected,
            native_expected - selected_expected,
            expected.min(axis=1),
            expected_sorted[:, 1] - expected_sorted[:, 0],
            expected.mean(axis=1),
            expected.std(axis=1),
            slot.astype(np.float64) / float(config.shortlist_size - 1),
            (slot == 0).astype(np.float64),
        ]
    )
    raw = features[..., : config.candidate_raw_feature_dim].astype(np.float64)
    selected_raw = raw[row, slot]
    native_raw = raw[:, 0]
    output = np.concatenate(
        [summary, selected_raw, native_raw, selected_raw - native_raw], axis=1
    ).astype(np.float32)
    if output.shape != (rows, config.output_dim) or not np.isfinite(output).all():
        raise ValueError("row policy feature contract drift")
    return output
