from __future__ import annotations

import math

import torch

from projects.mmp_tracker.mmp_tracker.routeD_redetection_metrics import (
    compute_official_aj_rd,
    count_consecutive_invisibility,
    eligible_redetection_events,
    recovery_latency_and_safety,
)


def test_record_breaking_events_and_durations():
    visible = torch.tensor([[[1], [0], [1], [1], [0], [0], [1], [0], [1]]], dtype=torch.bool)
    durations = count_consecutive_invisibility(visible)
    assert durations[0, 2, 0].item() == 1
    assert durations[0, 6, 0].item() == 2
    assert durations[0, 8, 0].item() == 1
    events = eligible_redetection_events(visible)
    assert [(e.frame_index, e.invisibility_duration) for e in events] == [(2, 1), (6, 2)]


def test_perfect_redetection_is_one_for_available_buckets():
    gt = torch.zeros(1, 8, 1, 2)
    visible = torch.tensor([[[1], [1], [0], [0], [1], [1], [1], [1]]], dtype=torch.bool)
    result = compute_official_aj_rd(gt, visible, gt, visible)
    assert result["AJ_RD"] == 1.0
    assert result["AJ_RD_dmin1"] == 1.0
    assert math.isnan(result["AJ_RD_dmin4"])


def test_visibility_false_positive_reduces_ajrd():
    gt = torch.zeros(1, 8, 1, 2)
    visible = torch.tensor([[[1], [0], [0], [1], [1], [0], [0], [0]]], dtype=torch.bool)
    pred_visible = torch.ones_like(visible)
    result = compute_official_aj_rd(gt, pred_visible, gt, visible)
    assert 0.0 < result["AJ_RD"] < 1.0


def test_recovery_latency_and_harmful_intervention():
    gt = torch.zeros(1, 7, 1, 2)
    visible = torch.tensor([[[1], [0], [0], [1], [1], [1], [1]]], dtype=torch.bool)
    pred = gt.clone()
    pred[:, 3, :, 0] = 10.0
    pred_visible = visible.clone()
    selected = torch.zeros_like(visible)
    selected[:, 3] = True
    native = gt.clone()
    stats = recovery_latency_and_safety(
        pred,
        pred_visible,
        gt,
        visible,
        selected_non_native=selected,
        native_tracks=native,
    )
    assert stats["eligible_events"] == 1
    assert stats["recovered_events"] == 1
    assert stats["mean_recovery_latency_frames"] == 1.0
    assert stats["harmful_intervention_rate"] == 1.0
