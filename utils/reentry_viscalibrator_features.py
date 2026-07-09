"""Feature extraction for the learned ReEntry visibility calibrator.

The learned module is deliberately scoped to visibility recovery only:
coordinates are kept from the base/offline cache.  This utility builds the
same candidate windows and per-frame features for dataset construction and
inference, avoiding train/eval feature drift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from utils.coords import yx_norm_to_xy_256


FEATURE_NAMES: List[str] = [
    "base_visible",
    "override_visible",
    "visibility_disagree",
    "candidate_window_mask",
    "relative_time_to_trigger",
    "time_since_query_norm",
    "is_after_query",
    "base_x_norm",
    "base_y_norm",
    "override_x_norm",
    "override_y_norm",
    "base_override_dist_norm",
    "base_speed_norm",
    "override_speed_norm",
    "speed_diff_norm",
    "base_acc_norm",
    "override_acc_norm",
    "base_invisible_run_norm",
    "base_visible_run_norm",
    "override_visible_run_norm",
    "override_invisible_run_norm",
    "base_border_dist_norm",
    "override_border_dist_norm",
    "override_future4_vis_rate",
    "override_future8_vis_rate",
    "base_past8_invis_rate",
    "base_override_xdiff_norm",
    "base_override_ydiff_norm",
]


@dataclass(frozen=True)
class CandidateConfig:
    context_before: int = 16
    context_after: int = 16
    candidate_pre: int = 1
    candidate_post: int = 16
    trigger_k: int = 1
    trigger_persist: int = 2
    min_trigger_t_after_query: int = 1


@dataclass(frozen=True)
class LabelConfig:
    coord_thresholds: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0)
    coord_values: Tuple[float, ...] = (1.0, 0.9, 0.75, 0.55, 0.25)


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr


def check_alignment(base: Dict[str, Any], override: Dict[str, Any], name: str = "override") -> None:
    if len(base.get("records", [])) != len(override.get("records", [])):
        raise ValueError(f"{name}: record count mismatch")
    for i, (br, orr) in enumerate(zip(base["records"], override["records"])):
        if str(br["video_id"]) != str(orr["video_id"]):
            raise ValueError(f"{name}: video_id mismatch at record={i}: {br['video_id']} vs {orr['video_id']}")
        for key in ("query_points", "gt_tracks", "gt_visibility", "original_size"):
            if key == "gt_visibility":
                ok = np.array_equal(npy(br[key], bool), npy(orr[key], bool))
            else:
                ok = np.allclose(npy(br[key]), npy(orr[key]), atol=1e-6, rtol=1e-6)
            if not ok:
                raise ValueError(f"{name}: alignment mismatch at record={i}, key={key}")


def run_lengths_ending_at(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros(mask.shape[0], dtype=np.float32)
    cur = 0
    for i, v in enumerate(mask):
        cur = cur + 1 if bool(v) else 0
        out[i] = float(cur)
    return out


def invisible_run_before(visibility: np.ndarray, t: int) -> int:
    count = 0
    j = int(t) - 1
    while j >= 0 and not bool(visibility[j]):
        count += 1
        j -= 1
    return count


def trigger_ok(base_visibility: np.ndarray, override_visibility: np.ndarray, t: int, *, k: int, persist: int) -> bool:
    if invisible_run_before(base_visibility, int(t)) < int(k):
        return False
    if int(t) + int(persist) > int(override_visibility.shape[0]):
        return False
    return bool(np.all(override_visibility[int(t) : int(t) + int(persist)]))


def find_candidate_triggers(
    base_visibility: np.ndarray,
    override_visibility: np.ndarray,
    query_t: int,
    cfg: CandidateConfig,
) -> List[int]:
    """Find prediction-only high-recall candidate re-entry triggers."""
    t_len = int(base_visibility.shape[0])
    triggers: List[int] = []
    t = max(int(query_t) + int(cfg.min_trigger_t_after_query), 1)
    while t < t_len:
        if trigger_ok(base_visibility, override_visibility, t, k=cfg.trigger_k, persist=cfg.trigger_persist):
            triggers.append(int(t))
            # Skip the candidate post-window to avoid many near-duplicate windows.
            t = min(t_len, t + int(cfg.candidate_post) + 1)
        else:
            t += 1
    return triggers


def _speed_and_acc(xy_256: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if xy_256.shape[0] == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    vel = np.zeros_like(xy_256, dtype=np.float32)
    vel[1:] = xy_256[1:] - xy_256[:-1]
    speed = np.linalg.norm(vel, axis=-1).astype(np.float32)
    acc_vec = np.zeros_like(xy_256, dtype=np.float32)
    acc_vec[1:] = vel[1:] - vel[:-1]
    acc = np.linalg.norm(acc_vec, axis=-1).astype(np.float32)
    return speed, acc


def _future_rate(mask: np.ndarray, horizon: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32)
    n = mask.shape[0]
    out = np.zeros(n, dtype=np.float32)
    csum = np.concatenate([[0.0], np.cumsum(mask)])
    for i in range(n):
        hi = min(n, i + int(horizon))
        out[i] = float((csum[hi] - csum[i]) / max(hi - i, 1))
    return out


def _past_rate(mask: np.ndarray, horizon: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32)
    n = mask.shape[0]
    out = np.zeros(n, dtype=np.float32)
    csum = np.concatenate([[0.0], np.cumsum(mask)])
    for i in range(n):
        lo = max(0, i - int(horizon) + 1)
        out[i] = float((csum[i + 1] - csum[lo]) / max(i + 1 - lo, 1))
    return out


def safe_visible_soft_label(error_256: np.ndarray, gt_visible: np.ndarray, cfg: LabelConfig) -> np.ndarray:
    error_256 = np.asarray(error_256, dtype=np.float32)
    gt_visible = np.asarray(gt_visible, dtype=bool)
    label = np.zeros(error_256.shape, dtype=np.float32)
    # Assign the first matching threshold value. Thresholds must be ascending.
    assigned = np.zeros(error_256.shape, dtype=bool)
    for thr, val in zip(cfg.coord_thresholds, cfg.coord_values):
        m = gt_visible & (~assigned) & (error_256 < float(thr))
        label[m] = float(val)
        assigned[m] = True
    return label


def build_candidate_example(
    base_record: Dict[str, Any],
    override_record: Dict[str, Any],
    query_idx: int,
    trigger_t: int,
    candidate_cfg: CandidateConfig,
    label_cfg: LabelConfig | None = None,
) -> Dict[str, Any]:
    """Build one fixed-length candidate-window example.

    Returns feature matrix (L,F), labels, masks, and metadata. Labels require GT
    and are used for training only; inference can ignore them.
    """
    label_cfg = label_cfg or LabelConfig()
    qi = int(query_idx)
    trigger_t = int(trigger_t)

    base_tracks = npy(base_record["pred_tracks"], np.float32)[qi]
    override_tracks = npy(override_record["pred_tracks"], np.float32)[qi]
    base_vis = npy(base_record["pred_visibility"], bool)[qi]
    override_vis = npy(override_record["pred_visibility"], bool)[qi]
    gt_tracks = npy(base_record["gt_tracks"], np.float32)[qi]
    gt_vis = npy(base_record["gt_visibility"], bool)[qi]
    qpts = npy(base_record["query_points"], np.float32)
    query_t = max(0, min(base_vis.shape[0] - 1, int(round(float(qpts[qi, 0])))))

    t_len = int(base_vis.shape[0])
    start = trigger_t - int(candidate_cfg.context_before)
    end = trigger_t + int(candidate_cfg.context_after) + 1
    frame_indices = np.arange(start, end, dtype=np.int32)
    valid_mask = (frame_indices >= 0) & (frame_indices < t_len)
    clipped = np.clip(frame_indices, 0, t_len - 1)

    base_xy = yx_norm_to_xy_256(base_tracks)
    override_xy = yx_norm_to_xy_256(override_tracks)
    gt_xy = yx_norm_to_xy_256(gt_tracks)

    base_speed, base_acc = _speed_and_acc(base_xy)
    override_speed, override_acc = _speed_and_acc(override_xy)

    base_invis_run = run_lengths_ending_at(~base_vis)
    base_vis_run = run_lengths_ending_at(base_vis)
    override_vis_run = run_lengths_ending_at(override_vis)
    override_invis_run = run_lengths_ending_at(~override_vis)

    override_future4 = _future_rate(override_vis, 4)
    override_future8 = _future_rate(override_vis, 8)
    base_past8_invis = _past_rate(~base_vis, 8)

    bxy = base_xy[clipped]
    oxy = override_xy[clipped]
    bvis = base_vis[clipped].astype(np.float32)
    ovis = override_vis[clipped].astype(np.float32)
    dist = np.linalg.norm(bxy - oxy, axis=-1).astype(np.float32)
    speed_diff = np.abs(base_speed[clipped] - override_speed[clipped]).astype(np.float32)

    base_border = np.minimum.reduce([bxy[:, 0], bxy[:, 1], 255.0 - bxy[:, 0], 255.0 - bxy[:, 1]]).astype(np.float32)
    override_border = np.minimum.reduce([oxy[:, 0], oxy[:, 1], 255.0 - oxy[:, 0], 255.0 - oxy[:, 1]]).astype(np.float32)

    candidate_lo = trigger_t - int(candidate_cfg.candidate_pre)
    candidate_hi = trigger_t + int(candidate_cfg.candidate_post)
    candidate_window = (frame_indices >= candidate_lo) & (frame_indices <= candidate_hi) & valid_mask
    after_query = (frame_indices > query_t) & valid_mask
    loss_mask = candidate_window & after_query

    features = np.stack(
        [
            bvis,
            ovis,
            (bvis != ovis).astype(np.float32),
            candidate_window.astype(np.float32),
            (frame_indices.astype(np.float32) - float(trigger_t)) / max(float(candidate_cfg.context_after), 1.0),
            (frame_indices.astype(np.float32) - float(query_t)) / max(float(t_len), 1.0),
            after_query.astype(np.float32),
            np.clip(bxy[:, 0] / 255.0, -1.0, 2.0),
            np.clip(bxy[:, 1] / 255.0, -1.0, 2.0),
            np.clip(oxy[:, 0] / 255.0, -1.0, 2.0),
            np.clip(oxy[:, 1] / 255.0, -1.0, 2.0),
            np.clip(dist / 64.0, 0.0, 8.0),
            np.clip(base_speed[clipped] / 32.0, 0.0, 8.0),
            np.clip(override_speed[clipped] / 32.0, 0.0, 8.0),
            np.clip(speed_diff / 32.0, 0.0, 8.0),
            np.clip(base_acc[clipped] / 32.0, 0.0, 8.0),
            np.clip(override_acc[clipped] / 32.0, 0.0, 8.0),
            np.clip(base_invis_run[clipped] / 32.0, 0.0, 8.0),
            np.clip(base_vis_run[clipped] / 32.0, 0.0, 8.0),
            np.clip(override_vis_run[clipped] / 32.0, 0.0, 8.0),
            np.clip(override_invis_run[clipped] / 32.0, 0.0, 8.0),
            np.clip(base_border / 128.0, -2.0, 2.0),
            np.clip(override_border / 128.0, -2.0, 2.0),
            override_future4[clipped],
            override_future8[clipped],
            base_past8_invis[clipped],
            np.clip((oxy[:, 0] - bxy[:, 0]) / 64.0, -8.0, 8.0),
            np.clip((oxy[:, 1] - bxy[:, 1]) / 64.0, -8.0, 8.0),
        ],
        axis=-1,
    ).astype(np.float32)

    # Zero out padded rows for easier normalization/training.
    features[~valid_mask] = 0.0

    base_error = np.linalg.norm(base_xy[clipped] - gt_xy[clipped], axis=-1).astype(np.float32)
    gtv = gt_vis[clipped] & valid_mask
    y_soft = safe_visible_soft_label(base_error, gtv, label_cfg)
    y_hard = (gtv & (base_error < 8.0)).astype(np.float32)
    y_soft[~valid_mask] = 0.0
    y_hard[~valid_mask] = 0.0

    return {
        "x": features,
        "y_soft": y_soft.astype(np.float32),
        "y_hard": y_hard.astype(np.float32),
        "valid_mask": valid_mask.astype(bool),
        "loss_mask": loss_mask.astype(bool),
        "frame_indices": frame_indices.astype(np.int32),
        "trigger_t": int(trigger_t),
        "query_t": int(query_t),
        "query_idx": int(qi),
        "video_id": str(base_record.get("video_id", "")),
        "base_error_256": base_error.astype(np.float32),
        "gt_visibility": gtv.astype(bool),
        "base_visibility": bvis.astype(np.float32),
        "override_visibility": ovis.astype(np.float32),
    }


def iter_candidate_examples(
    base_record: Dict[str, Any],
    override_record: Dict[str, Any],
    candidate_cfg: CandidateConfig,
    label_cfg: LabelConfig | None = None,
) -> Iterable[Dict[str, Any]]:
    base_vis = npy(base_record["pred_visibility"], bool)
    override_vis = npy(override_record["pred_visibility"], bool)
    qpts = npy(base_record["query_points"], np.float32)
    n_tracks, t_len = base_vis.shape
    for qi in range(n_tracks):
        query_t = max(0, min(t_len - 1, int(round(float(qpts[qi, 0])))))
        triggers = find_candidate_triggers(base_vis[qi], override_vis[qi], query_t, candidate_cfg)
        for t in triggers:
            yield build_candidate_example(base_record, override_record, qi, t, candidate_cfg, label_cfg)
