"""Frame-level keep/drop features for ReEntry-VisCalibrator V2.3.

V2.3 is a second-stage risk decoder.  It does not propose new recovery frames.
It only decides whether to keep V1-proposed recovery frames.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from utils.reentry_viscalibrator_features import FEATURE_NAMES


THRESHOLDS_256: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)

EXTRA_FRAME_KEEP_FEATURE_NAMES: List[str] = [
    "v1_prob",
    "segment_prob_mean",
    "segment_prob_std",
    "segment_prob_min",
    "segment_prob_max",
    "prob_minus_segment_mean",
    "prob_div_segment_max",
    "segment_len_norm",
    "position_in_segment_norm",
    "dist_to_segment_start_norm",
    "dist_to_segment_end_norm",
    "is_segment_start",
    "is_segment_end",
    "raw_recovery_count_norm",
    "raw_recovery_rate_candidate",
    "candidate_len_norm",
    "event_gate_prob",
    "trigger_t_norm",
    "query_t_norm",
    "frame_minus_trigger_norm",
    "frame_minus_query_norm",
]

FRAME_KEEP_FEATURE_NAMES: List[str] = [f"frame_{n}" for n in FEATURE_NAMES] + EXTRA_FRAME_KEEP_FEATURE_NAMES


def candidate_mask_from_example(ex: Dict[str, Any], *, require_override_visible: bool = True) -> np.ndarray:
    mask = np.asarray(ex["loss_mask"], dtype=bool).copy()
    if require_override_visible:
        mask &= np.asarray(ex["override_visibility"], dtype=np.float32) > 0.5
    mask &= np.asarray(ex["valid_mask"], dtype=bool)
    return mask


def raw_recovery_mask_from_prob(
    ex: Dict[str, Any],
    prob: np.ndarray,
    *,
    threshold: float,
    require_override_visible: bool = True,
) -> np.ndarray:
    cand = candidate_mask_from_example(ex, require_override_visible=require_override_visible)
    prob = np.asarray(prob, dtype=np.float32)
    return cand & (prob >= float(threshold))


def contiguous_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    segs: List[Tuple[int, int]] = []
    i = 0
    n = int(mask.shape[0])
    while i < n:
        if not bool(mask[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(mask[j + 1]):
            j += 1
        segs.append((i, j))
        i = j + 1
    return segs


def frame_keep_utility(
    *,
    gt_visible: bool,
    base_error_256: float,
    thresholds: Sequence[float] = THRESHOLDS_256,
) -> float:
    """Approximate TAP-style local utility of keeping one added visible frame."""
    if not bool(gt_visible):
        return -1.0
    vals = [1.0 if float(base_error_256) < float(thr) else -1.0 for thr in thresholds]
    return float(np.mean(vals))


def build_frame_keep_samples(
    ex: Dict[str, Any],
    prob: np.ndarray,
    *,
    threshold: float,
    require_override_visible: bool = True,
    gate_prob: float | None = None,
    include_base_visible: bool = False,
    thresholds: Sequence[float] = THRESHOLDS_256,
) -> Dict[str, Any]:
    """Build per-frame keep/drop samples for V1-proposed recovery frames.

    By default, samples are built only for frames whose visibility would actually
    change relative to base, i.e. raw recovery frames with base_visible=False.
    """
    x = np.asarray(ex["x"], dtype=np.float32)
    prob = np.asarray(prob, dtype=np.float32)
    raw = raw_recovery_mask_from_prob(ex, prob, threshold=threshold, require_override_visible=require_override_visible)
    base_vis = np.asarray(ex["base_visibility"], dtype=np.float32) > 0.5
    cand = candidate_mask_from_example(ex, require_override_visible=require_override_visible)
    sample_mask = raw.copy()
    if not include_base_visible:
        sample_mask &= ~base_vis

    frames = np.asarray(ex["frame_indices"], dtype=np.int32)
    gt_vis = np.asarray(ex["gt_visibility"], dtype=bool)
    err = np.asarray(ex["base_error_256"], dtype=np.float32)
    trigger_t = int(ex.get("trigger_t", 0))
    query_t = int(ex.get("query_t", 0))
    gate_val = 0.5 if gate_prob is None else float(gate_prob)
    raw_count = int(raw.sum())
    cand_count = int(cand.sum())

    rows: List[np.ndarray] = []
    y_gt_visible: List[float] = []
    y_safe16: List[float] = []
    y_utility: List[float] = []
    y_safe8: List[float] = []
    y_safe4: List[float] = []
    weight: List[float] = []
    util_list: List[float] = []
    meta: List[Dict[str, Any]] = []

    for lo, hi in contiguous_segments(raw):
        seg_idx = np.arange(lo, hi + 1)
        seg_prob = prob[seg_idx]
        seg_len = int(hi - lo + 1)
        seg_mean = float(seg_prob.mean()) if seg_len else 0.0
        seg_std = float(seg_prob.std()) if seg_len else 0.0
        seg_min = float(seg_prob.min()) if seg_len else 0.0
        seg_max = float(seg_prob.max()) if seg_len else 0.0
        for local_idx in seg_idx:
            if not bool(sample_mask[local_idx]):
                continue
            u = frame_keep_utility(
                gt_visible=bool(gt_vis[local_idx]),
                base_error_256=float(err[local_idx]),
                thresholds=thresholds,
            )
            pos = int(local_idx - lo)
            extra = np.asarray([
                float(prob[local_idx]),
                seg_mean,
                seg_std,
                seg_min,
                seg_max,
                float(prob[local_idx]) - seg_mean,
                float(prob[local_idx]) / max(seg_max, 1e-6),
                float(seg_len) / 32.0,
                float(pos) / max(float(seg_len - 1), 1.0),
                float(local_idx - lo) / 16.0,
                float(hi - local_idx) / 16.0,
                1.0 if local_idx == lo else 0.0,
                1.0 if local_idx == hi else 0.0,
                float(raw_count) / 32.0,
                float(raw_count) / max(float(cand_count), 1.0),
                float(cand_count) / 32.0,
                gate_val,
                float(trigger_t) / 256.0,
                float(query_t) / 256.0,
                float(frames[local_idx] - trigger_t) / 16.0,
                float(frames[local_idx] - query_t) / 256.0,
            ], dtype=np.float32)
            rows.append(np.concatenate([x[local_idx].astype(np.float32), extra], axis=0))
            visible = bool(gt_vis[local_idx])
            error = float(err[local_idx])
            y_gt_visible.append(1.0 if visible else 0.0)
            y_safe16.append(1.0 if (visible and error < 16.0) else 0.0)
            y_utility.append(1.0 if u > 0.0 else 0.0)
            y_safe8.append(1.0 if (visible and error < 8.0) else 0.0)
            y_safe4.append(1.0 if (visible and error < 4.0) else 0.0)
            weight.append(float(max(0.25, min(4.0, abs(u) * 2.0))))
            util_list.append(float(u))
            meta.append({
                "video_id": str(ex.get("video_id", "")),
                "query_idx": int(ex.get("query_idx", -1)),
                "trigger_t": int(trigger_t),
                "query_t": int(query_t),
                "frame_t": int(frames[local_idx]),
                "local_idx": int(local_idx),
                "segment_lo_t": int(frames[lo]),
                "segment_hi_t": int(frames[hi]),
                "segment_len": int(seg_len),
                "v1_prob": float(prob[local_idx]),
                "gate_prob": gate_val,
                "base_error_256": error,
                "gt_visible": visible,
                "utility": float(u),
            })

    if rows:
        X = np.stack(rows, axis=0).astype(np.float32)
    else:
        X = np.zeros((0, len(FRAME_KEEP_FEATURE_NAMES)), dtype=np.float32)
    return {
        "X": X,
        "y_gt_visible": np.asarray(y_gt_visible, dtype=np.float32),
        "y_safe16": np.asarray(y_safe16, dtype=np.float32),
        "y_utility": np.asarray(y_utility, dtype=np.float32),
        "y_safe8": np.asarray(y_safe8, dtype=np.float32),
        "y_safe4": np.asarray(y_safe4, dtype=np.float32),
        "weight": np.asarray(weight, dtype=np.float32),
        "utility": np.asarray(util_list, dtype=np.float32),
        "meta": meta,
        "raw_recovery_frames": int(raw.sum()),
        "changed_recovery_frames": int((raw & (~base_vis)).sum()),
        "candidate_frames": int(cand.sum()),
    }
