"""Event-level features and labels for ReEntry no-action gating.

The V2.1 gate decides whether a candidate re-entry window should be allowed to
modify base visibility.  It is deliberately an event-level module placed before
frame-level recovery:

    candidate window -> event allow/no-op decision -> frame-level recovery

No GT is used to build inference features.  GT is used only to compute training
utility labels.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from utils.reentry_viscalibrator_features import FEATURE_NAMES


THRESHOLDS_256: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)

EVENT_FEATURE_NAMES: List[str] = []
for _name in FEATURE_NAMES:
    EVENT_FEATURE_NAMES.extend([
        f"cand_mean_{_name}",
        f"cand_std_{_name}",
        f"cand_min_{_name}",
        f"cand_max_{_name}",
    ])
EVENT_FEATURE_NAMES.extend([
    "candidate_len_norm",
    "candidate_valid_rate",
    "trigger_t_norm",
    "query_t_norm",
    "trigger_minus_query_norm",
    "base_vis_rate_candidate",
    "override_vis_rate_candidate",
    "disagree_rate_candidate",
    "proposal_recover_count_norm",
    "proposal_recover_rate",
    "proposal_prob_mean_candidate",
    "proposal_prob_std_candidate",
    "proposal_prob_max_candidate",
    "proposal_prob_mean_recover",
    "proposal_first_offset_norm",
    "proposal_last_offset_norm",
    "proposal_span_norm",
])


def candidate_mask_from_example(ex: Dict[str, Any], *, require_override_visible: bool = True) -> np.ndarray:
    mask = np.asarray(ex["loss_mask"], dtype=bool).copy()
    if require_override_visible:
        mask &= np.asarray(ex["override_visibility"], dtype=np.float32) > 0.5
    mask &= np.asarray(ex["valid_mask"], dtype=bool)
    return mask


def proposal_mask_from_prob(
    ex: Dict[str, Any],
    prob: np.ndarray,
    *,
    threshold: float,
    require_override_visible: bool = True,
) -> np.ndarray:
    cand = candidate_mask_from_example(ex, require_override_visible=require_override_visible)
    prob = np.asarray(prob, dtype=np.float32)
    return cand & (prob >= float(threshold))


def _safe_stats(vals: np.ndarray) -> Tuple[float, float, float, float]:
    vals = np.asarray(vals, dtype=np.float32)
    if vals.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(vals.mean()), float(vals.std()), float(vals.min()), float(vals.max())


def build_event_features(
    ex: Dict[str, Any],
    prob: np.ndarray,
    *,
    threshold: float,
    require_override_visible: bool = True,
) -> np.ndarray:
    """Build event-level inference features from a candidate example and V1 probs."""
    x = np.asarray(ex["x"], dtype=np.float32)
    prob = np.asarray(prob, dtype=np.float32)
    cand = candidate_mask_from_example(ex, require_override_visible=require_override_visible)
    rec = proposal_mask_from_prob(ex, prob, threshold=threshold, require_override_visible=require_override_visible)
    valid = np.asarray(ex["valid_mask"], dtype=bool)
    frames = np.asarray(ex["frame_indices"], dtype=np.int32)
    t_len_norm = max(float(np.max(frames[valid]) - np.min(frames[valid]) + 1) if np.any(valid) else 1.0, 1.0)

    feats: List[float] = []
    if np.any(cand):
        xc = x[cand]
        for j in range(x.shape[1]):
            feats.extend(_safe_stats(xc[:, j]))
    else:
        feats.extend([0.0] * (4 * x.shape[1]))

    cand_count = int(cand.sum())
    valid_count = int(valid.sum())
    rec_count = int(rec.sum())
    base_vis = np.asarray(ex["base_visibility"], dtype=np.float32) > 0.5
    override_vis = np.asarray(ex["override_visibility"], dtype=np.float32) > 0.5
    disagree = base_vis != override_vis
    trigger_t = int(ex.get("trigger_t", 0))
    query_t = int(ex.get("query_t", 0))

    if np.any(cand):
        prob_c = prob[cand]
        pmean, pstd, _pmin, pmax = _safe_stats(prob_c)
        base_rate = float(base_vis[cand].mean())
        override_rate = float(override_vis[cand].mean())
        disagree_rate = float(disagree[cand].mean())
    else:
        pmean = pstd = pmax = base_rate = override_rate = disagree_rate = 0.0

    if rec_count > 0:
        prob_r = prob[rec]
        rmean = float(prob_r.mean())
        rec_frames = frames[rec]
        first_off = float(rec_frames.min() - trigger_t) / 16.0
        last_off = float(rec_frames.max() - trigger_t) / 16.0
        span = float(rec_frames.max() - rec_frames.min() + 1) / 32.0
    else:
        rmean = 0.0
        first_off = 0.0
        last_off = 0.0
        span = 0.0

    feats.extend([
        float(cand_count) / 32.0,
        float(cand_count) / max(float(valid_count), 1.0),
        float(trigger_t) / 256.0,
        float(query_t) / 256.0,
        float(trigger_t - query_t) / 256.0,
        base_rate,
        override_rate,
        disagree_rate,
        float(rec_count) / 32.0,
        float(rec_count) / max(float(cand_count), 1.0),
        pmean,
        pstd,
        pmax,
        rmean,
        first_off,
        last_off,
        span,
    ])
    return np.asarray(feats, dtype=np.float32)


def compute_recovery_utility(
    ex: Dict[str, Any],
    prob: np.ndarray,
    *,
    threshold: float,
    require_override_visible: bool = True,
    thresholds: Sequence[float] = THRESHOLDS_256,
) -> Dict[str, Any]:
    """Approximate local AJ utility of allowing the V1 recovery proposal.

    Only frames whose visibility would change relative to base contribute.
    For each threshold, a recovered frame contributes +1 if it converts a missed
    visible frame into a correct visible point; otherwise it contributes -1.
    The per-frame contribution is averaged over thresholds.
    """
    rec = proposal_mask_from_prob(ex, prob, threshold=threshold, require_override_visible=require_override_visible)
    base_vis = np.asarray(ex["base_visibility"], dtype=np.float32) > 0.5
    changed = rec & (~base_vis)
    gt_vis = np.asarray(ex["gt_visibility"], dtype=bool)
    err = np.asarray(ex["base_error_256"], dtype=np.float32)
    util = 0.0
    good = 0
    bad = 0
    contribs: List[float] = []
    thrs = tuple(float(t) for t in thresholds)
    for idx in np.where(changed)[0]:
        if bool(gt_vis[idx]):
            vals = [1.0 if float(err[idx]) < t else -1.0 for t in thrs]
        else:
            vals = [-1.0 for _ in thrs]
        c = float(np.mean(vals))
        contribs.append(c)
        util += c
        if c > 0:
            good += 1
        else:
            bad += 1
    return {
        "utility": float(util),
        "label_allow": bool(util > 0.0),
        "weight": float(min(10.0, max(0.1, abs(util)))),
        "proposal_frames": int(rec.sum()),
        "changed_frames": int(changed.sum()),
        "good_changed_frames": int(good),
        "bad_changed_frames": int(bad),
        "mean_changed_contribution": float(np.mean(contribs)) if contribs else 0.0,
    }
