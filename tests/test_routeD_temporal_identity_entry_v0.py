import numpy as np
import torch

from projects.mmp_tracker.mmp_tracker.routeD_temporal_identity_entry_v0 import (
    EntryFeatureConfig,
    build_entry_features,
    entry_action_mask,
)


def test_entry_features_have_frozen_shape_and_are_label_free():
    rows = 5
    trajectory = torch.linspace(0.0, 1.0, rows * 8 * 9).reshape(rows, 8, 9)
    visibility = torch.linspace(0.05, 0.95, rows)
    confidence = torch.linspace(0.1, 0.9, rows)
    features = [torch.randn(rows, 1, 128) for _ in range(4)]
    supports = [torch.randn(rows, 49, 128) for _ in range(4)]
    output = build_entry_features(
        trajectory_features=trajectory,
        visibility_probability=visibility,
        confidence_probability=confidence,
        native_track_features=features,
        native_track_supports=supports,
    )
    assert output.shape == (rows, EntryFeatureConfig().feature_dim) == (5, 130)
    assert np.isfinite(output).all()


def test_entry_action_requires_high_failure_probability_and_low_native_joint():
    mask = entry_action_mask(
        entry_probability=np.array([0.929, 0.93, 0.99, 0.99]),
        native_joint_probability=np.array([0.001, 0.021, 0.02, 0.001]),
    )
    assert mask.tolist() == [False, False, True, True]
