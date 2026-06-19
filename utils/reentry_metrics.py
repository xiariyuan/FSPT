"""Shared helpers for re-entry and re-detection metrics."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .coords import (
    find_first_reentry, find_reentry_events,
    pixel_l2_error, yx_norm_to_xy_256, yx_norm_to_xy_pixel,
)

DEFAULT_PROXY_THRESHOLDS: tuple[int, ...] = (1, 2, 4, 8, 16)
DEFAULT_AJ_THRESHOLDS: tuple[int, ...] = (1, 2, 4, 8, 16)
DEFAULT_AJRD_D_MINS: tuple[int, ...] = (1, 4, 16, 64, 256)


def eligible_reentry_events(
    gt_visibility: np.ndarray,
    query_t: int,
    min_occ_length: Optional[int] = None,
) -> List[Dict[str, int]]:
    """Return record-breaking reappearance events for a query track.

    An event is eligible if its undetectability duration is strictly greater
    than all previous reappearance events on the same track.
    """
    events = find_reentry_events(gt_visibility, query_t)
    eligible: List[Dict[str, int]] = []
    best_occ = 0
    for evt in events:
        occ_len = int(evt["occ_length"])
        if occ_len > best_occ:
            best_occ = occ_len
            if min_occ_length is None or occ_len >= int(min_occ_length):
                eligible.append(dict(evt))
    return eligible


def compute_first_reentry_proxy(
    pred_tracks: np.ndarray,
    gt_tracks: np.ndarray,
    pred_visibility: np.ndarray,
    gt_visibility: np.ndarray,
    query_t: int,
    height: int,
    width: int,
    thresholds: Sequence[int] = DEFAULT_PROXY_THRESHOLDS,
) -> Optional[Dict[str, Any]]:
    """Compute the historical first-reentry-frame diagnostic proxy."""
    event = find_first_reentry(gt_visibility, int(query_t))
    if event is None:
        return None

    event = dict(event)
    event["query_t"] = int(query_t)
    return compute_reentry_frame_proxy(
        pred_tracks=pred_tracks,
        gt_tracks=gt_tracks,
        pred_visibility=pred_visibility,
        gt_visibility=gt_visibility,
        event=event,
        height=height,
        width=width,
        thresholds=thresholds,
    )


def compute_reentry_frame_proxy(
    pred_tracks: np.ndarray,
    gt_tracks: np.ndarray,
    pred_visibility: np.ndarray,
    gt_visibility: np.ndarray,
    event: Dict[str, int],
    height: int,
    width: int,
    thresholds: Sequence[int] = DEFAULT_PROXY_THRESHOLDS,
) -> Optional[Dict[str, Any]]:
    """Compute the diagnostic proxy for a specific reappearance event."""
    t_re = int(event["reentry_frame"])
    if t_re < 0 or t_re >= int(pred_tracks.shape[0]):
        return None

    pred_yx = np.asarray(pred_tracks[t_re], dtype=np.float32)
    gt_yx = np.asarray(gt_tracks[t_re], dtype=np.float32)
    err = float(
        pixel_l2_error(
            pred_yx[None, :],
            gt_yx[None, :],
            height,
            width,
            pred_fmt="yx_norm",
            gt_fmt="yx_norm",
        )[0]
    )

    pred_visible_at_re = bool(pred_visibility[t_re])
    gt_visible_at_re = bool(gt_visibility[t_re])

    jaccards: Dict[str, float] = {}
    aj_terms: List[float] = []
    for thr in thresholds:
        within_dist = err < float(thr)
        true_positive = 1.0 if (pred_visible_at_re and gt_visible_at_re and within_dist) else 0.0
        gt_positive = 1.0 if gt_visible_at_re else 0.0
        false_positive = 1.0 if (pred_visible_at_re and ((not gt_visible_at_re) or (not within_dist))) else 0.0
        j = true_positive / (gt_positive + false_positive) if (gt_positive + false_positive) > 0 else 0.0
        jaccards[f"jaccard_{int(thr)}"] = float(j)
        aj_terms.append(float(j))

    return {
        "query_t": int(event.get("query_t", -1)),
        "reentry_t": t_re,
        "occ_length": int(event["occ_length"]),
        "last_visible_t": int(event["last_visible_t"]),
        "error_px": round(err, 3),
        "pred_visible": pred_visible_at_re,
        "gt_visible": gt_visible_at_re,
        "aj_proxy": round(float(np.mean(aj_terms)) if aj_terms else 0.0, 4),
        **{k: round(v, 4) for k, v in jaccards.items()},
    }


def compute_reappearance_segment_aj(
    pred_tracks: np.ndarray,
    gt_tracks: np.ndarray,
    pred_visibility: np.ndarray,
    gt_visibility: np.ndarray,
    event: Dict[str, int],
    height: int,
    width: int,
    thresholds: Sequence[int] = DEFAULT_AJ_THRESHOLDS,
    use_256_space: bool = False,
) -> Optional[Dict[str, Any]]:
    """Compute AJ on the post-reappearance segment for one eligible event.

    Args:
        ...
        use_256_space: If True, convert to 256×256 space (TAPNext++ convention)
                       instead of original resolution. When True, thresholds
                       are interpreted in 256-space pixels.
    """
    reentry_t = int(event["reentry_frame"])
    if reentry_t < 0 or reentry_t >= int(pred_tracks.shape[0]):
        return None

    pred_seg = np.asarray(pred_tracks[reentry_t:], dtype=np.float32)
    gt_seg = np.asarray(gt_tracks[reentry_t:], dtype=np.float32)
    pred_vis_seg = np.asarray(pred_visibility[reentry_t:], dtype=bool)
    gt_vis_seg = np.asarray(gt_visibility[reentry_t:], dtype=bool)

    if pred_seg.shape[0] == 0:
        return None

    if use_256_space:
        # Convert to 256×256 space for TAPNext++ comparable AJ_RD
        pred_px = yx_norm_to_xy_256(pred_seg)
        gt_px = yx_norm_to_xy_256(gt_seg)
    else:
        pred_px = yx_norm_to_xy_pixel(pred_seg, height, width)
        gt_px = yx_norm_to_xy_pixel(gt_seg, height, width)

    sq_dist = np.sum((pred_px - gt_px) ** 2, axis=-1)
    jaccards: Dict[str, float] = {}
    aj_terms: List[float] = []

    for thr in thresholds:
        within_dist = sq_dist < float(thr) ** 2
        is_visible = gt_vis_seg
        pred_visible = pred_vis_seg
        true_positives = float(np.sum(within_dist & is_visible & pred_visible))
        gt_positives = float(np.sum(is_visible))
        false_positives = float(np.sum((~is_visible & pred_visible) | ((~within_dist) & pred_visible)))
        denom = gt_positives + false_positives
        j = true_positives / denom if denom > 0 else 0.0
        jaccards[f"jaccard_{int(thr)}"] = float(j)
        aj_terms.append(float(j))

    return {
        "query_t": int(event.get("query_t", -1)),
        "reentry_t": reentry_t,
        "occ_length": int(event["occ_length"]),
        "last_visible_t": int(event["last_visible_t"]),
        "n_eval_frames": int(pred_seg.shape[0]),
        "n_visible_frames": int(np.sum(gt_vis_seg)),
        "aj_segment": round(float(np.mean(aj_terms)) if aj_terms else 0.0, 4),
        **{k: round(v, 4) for k, v in jaccards.items()},
    }


def summarize_reappearance_ajrd(
    segment_rows: Sequence[Dict[str, Any]],
    d_mins: Sequence[int] = DEFAULT_AJRD_D_MINS,
) -> Dict[str, Any]:
    """Summarize eligible reappearance events for one query/sample.

    For each dmin, we average AJ over eligible reappearance events, then take
    the mean across the available dmin values for this sample. Only dmin values
    with at least one eligible event contribute to the mean; dmin values with
    zero eligible events are silently excluded (N/A), not counted as 0.

    Note for DAVIS: d_min=64 and d_min=256 have no eligible events (max occlusion
    on DAVIS is 99 frames, but no occ run in [64,99] survived the eligibility
    filter). Therefore on DAVIS this function effectively averages over
    {1, 4, 16} rather than the full {1, 4, 16, 64, 256} set.
    This is not equivalent to the TAPNext++ AJ_RD definition which uses
    the full fixed d_min set; see compute_reappearance_segment_aj with
    use_256_space=True for the TAPNext++-comparable path.
    """
    by_dmin: Dict[str, Dict[str, Any]] = {}
    dmin_means: List[float] = []

    for d in d_mins:
        vals = [float(row["aj_segment"]) for row in segment_rows if int(row["occ_length"]) >= int(d)]
        mean_val = float(np.mean(vals)) if vals else None
        by_dmin[str(int(d))] = {
            "n": int(len(vals)),
            "mean": round(mean_val, 4) if mean_val is not None else None,
        }
        if mean_val is not None:
            dmin_means.append(mean_val)

    aj_rd = round(float(np.mean(dmin_means)), 4) if dmin_means else None
    return {
        "aj_rd": aj_rd,
        "ajrd_by_dmin": by_dmin,
        "by_dmin": by_dmin,
        "n_segment_events": int(len(segment_rows)),
        "n_valid_dmins": int(len(dmin_means)),
    }


def aggregate_reappearance_ajrd(
    sample_rows: Sequence[Dict[str, Any]],
    d_mins: Sequence[int] = DEFAULT_AJRD_D_MINS,
) -> Dict[str, Any]:
    """Aggregate per-sample AJRD scores into a dataset-level AJRD."""
    sample_vals: List[float] = []
    by_dmin: Dict[str, Dict[str, Any]] = {}

    for row in sample_rows:
        row_score = row.get("aj_rd")
        if row_score is None and isinstance(row.get("ajrd_summary"), dict):
            row_score = row["ajrd_summary"].get("aj_rd")
        if row_score is not None:
            sample_vals.append(float(row_score))

    for d in d_mins:
        vals: List[float] = []
        for row in sample_rows:
            by_dmin_row = row.get("ajrd_by_dmin")
            if by_dmin_row is None and isinstance(row.get("ajrd_summary"), dict):
                by_dmin_row = row["ajrd_summary"].get("ajrd_by_dmin")
            if by_dmin_row is None:
                by_dmin_row = row.get("by_dmin")
            stats = by_dmin_row.get(str(int(d))) if isinstance(by_dmin_row, dict) else None
            if isinstance(stats, dict) and stats.get("mean") is not None:
                vals.append(float(stats["mean"]))
        by_dmin[str(int(d))] = {
            "n": int(len(vals)),
            "mean": round(float(np.mean(vals)), 4) if vals else None,
        }

    return {
        "aj_rd": round(float(np.mean(sample_vals)), 4) if sample_vals else None,
        "by_dmin": by_dmin,
        "n_samples": int(len(sample_rows)),
        "n_valid_samples": int(len(sample_vals)),
    }
