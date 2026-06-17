#!/usr/bin/env python3
"""
Selective evaluation for point tracking.

This script turns the current verifier / gate outputs into a paper-facing
evaluation protocol:

  - selective AJ / OA / pixel error at different coverage levels
  - risk-coverage summaries
  - calibration summaries (ECE / Brier)
  - oracle-gap-closed under selective thresholds
  - long-occlusion breakdowns

The goal is not to change the tracker here. The goal is to measure whether the
existing reliability signal is strong enough to justify the new paper story:

  calibrated selective point tracking
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.tapvid_official_eval import compute_tapvid_metrics_official
from scripts.eval_long_occlusion_subset import (
    _as_bool,
    _build_val_loader_from_config,
    _ensure_batch_dim,
    _load_model,
    _max_reappearance_occlusion_run,
    _resolve_resolution_from_batch,
    _setup_logging,
)

logger = logging.getLogger(__name__)


def _resolve_query_mode(config, override: str) -> Optional[str]:
    query_mode = override.strip()
    if not query_mode and hasattr(config, "evaluation") and hasattr(config.evaluation, "query_mode"):
        query_mode = str(getattr(config.evaluation, "query_mode") or "").strip()
    if not query_mode or query_mode.lower() in ("none", "null", ""):
        return None
    return query_mode


def _to_pixel_tracks(
    tracks: torch.Tensor,
    resolution: Tuple[int, int],
) -> torch.Tensor:
    """
    Convert [y, x] tracks to pixel [x, y] coordinates used by TAP-Vid metrics.
    """
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"Expected tracks shape (N,T,2), got {tuple(tracks.shape)}")
    height, width = float(resolution[0]), float(resolution[1])
    scale_w = max(width - 1.0, 1.0)
    scale_h = max(height - 1.0, 1.0)
    xy_scale = torch.tensor([scale_w, scale_h], device=tracks.device, dtype=tracks.dtype)

    tracks_xy = tracks[..., [1, 0]].to(dtype=tracks.dtype)
    if max(float(tracks_xy.abs().max()), 0.0) <= 1.5:
        return tracks_xy * xy_scale
    return tracks_xy


def _to_pixel_query_points(
    query_points: torch.Tensor,
    resolution: Tuple[int, int],
) -> torch.Tensor:
    """
    Convert query points [t, y, x] to pixel space for the official metric.
    """
    if query_points.ndim != 2 or query_points.shape[-1] < 3:
        raise ValueError(f"Expected query_points shape (N,3+), got {tuple(query_points.shape)}")
    height, width = float(resolution[0]), float(resolution[1])
    scale_w = max(width - 1.0, 1.0)
    scale_h = max(height - 1.0, 1.0)

    q = query_points[:, :3].to(dtype=torch.float32).clone()
    if q[:, 1:3].numel() > 0 and float(q[:, 1:3].abs().max()) <= 1.5:
        q[:, 1] *= scale_h
        q[:, 2] *= scale_w
    return q


def _evaluation_mask(query_t: torch.Tensor, num_frames: int, query_mode: Optional[str]) -> torch.Tensor:
    device = query_t.device
    frame_idx = torch.arange(num_frames, device=device).view(1, num_frames)
    mode = "strided" if query_mode is None else str(query_mode).strip().lower()
    if mode == "first":
        return frame_idx > query_t.view(-1, 1)
    if mode == "strided":
        mask = torch.ones((query_t.shape[0], num_frames), device=device, dtype=torch.bool)
        mask[torch.arange(query_t.shape[0], device=device), query_t] = False
        return mask
    raise ValueError(f"Unknown query_mode: {query_mode}")


def _safe_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            value = value.detach().cpu().item()
        return float(value)
    except Exception:
        return None


def _mean_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys: Iterable[str] = rows[0].keys()
    out: Dict[str, float] = {}
    for k in keys:
        vals: List[float] = []
        for row in rows:
            v = _safe_float(row.get(k))
            if v is None or not np.isfinite(v):
                continue
            vals.append(v)
        if vals:
            out[k] = float(np.mean(vals))
    return out


def _trackwise_official_metrics(
    pred_tracks_px: torch.Tensor,
    gt_tracks_px: torch.Tensor,
    pred_visible: torch.Tensor,
    gt_visible: torch.Tensor,
    query_points_px: torch.Tensor,
    query_mode: Optional[str],
) -> Dict[str, np.ndarray]:
    off = compute_tapvid_metrics_official(
        query_points=query_points_px.detach().cpu().numpy()[None, ...],
        gt_occluded=(~gt_visible).detach().cpu().numpy()[None, ...],
        gt_tracks=gt_tracks_px.detach().cpu().numpy()[None, ...],
        pred_occluded=(~pred_visible).detach().cpu().numpy()[None, ...],
        pred_tracks=pred_tracks_px.detach().cpu().numpy()[None, ...],
        query_mode="strided" if query_mode is None else str(query_mode).strip().lower(),
        thresholds=(1, 2, 4, 8, 16),
        get_trackwise_metrics=True,
    )

    out: Dict[str, np.ndarray] = {}
    for key, value in off.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)
        out[key] = arr
    return out


def _per_query_error_stats(
    pred_tracks_px: torch.Tensor,
    gt_tracks_px: torch.Tensor,
    gt_visible: torch.Tensor,
    query_points_px: torch.Tensor,
    query_mode: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if pred_tracks_px.shape != gt_tracks_px.shape:
        raise ValueError(
            f"Pred/GT track shape mismatch: pred={tuple(pred_tracks_px.shape)} gt={tuple(gt_tracks_px.shape)}"
        )
    n_queries, num_frames = pred_tracks_px.shape[:2]
    query_t = query_points_px[:, 0].round().long().clamp(0, num_frames - 1)
    eval_mask = _evaluation_mask(query_t, num_frames, query_mode)
    visible_eval = gt_visible & eval_mask

    error = torch.sqrt(torch.clamp_min(((pred_tracks_px - gt_tracks_px) ** 2).sum(dim=-1), 0.0))
    avg = torch.zeros((n_queries,), device=pred_tracks_px.device, dtype=torch.float32)
    med = torch.zeros((n_queries,), device=pred_tracks_px.device, dtype=torch.float32)
    for i in range(n_queries):
        mask = visible_eval[i]
        if mask.any():
            vals = error[i][mask]
            avg[i] = vals.mean()
            med[i] = vals.median()
    return avg, med


def _extract_score_tensor(info: Dict[str, Any], source: str = "auto") -> Tuple[Optional[torch.Tensor], str]:
    aliases = {
        "auto": (
            "verifier_scores",
            "relocal_acceptor",
            "policy_gate",
            "confidence",
            "relocal_conf",
            "relocalization_conf",
        ),
        "verifier": ("verifier_scores", "relocal_acceptor"),
        "verifier_scores": ("verifier_scores",),
        "relocal_acceptor": ("relocal_acceptor", "verifier_scores"),
        "policy_gate": ("policy_gate",),
        "confidence": ("confidence",),
        "relocal_conf": ("relocal_conf", "relocalization_conf"),
        "relocalization_conf": ("relocalization_conf", "relocal_conf"),
        "pred_visibility": (),
    }
    src = str(source or "auto").strip().lower()
    keys = aliases.get(src, (src,))
    for key in keys:
        value = info.get(key, None)
        if isinstance(value, torch.Tensor):
            return value, key
    return None, src


def _find_mask_tensor(info: Dict[str, Any]) -> Optional[torch.Tensor]:
    for key in (
        "verifier_mask",
        "relocal_accept_mask",
        "relocal_mask",
        "relocalization_mask",
    ):
        value = info.get(key, None)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _find_decision_tensor(info: Dict[str, Any]) -> Optional[torch.Tensor]:
    value = info.get("verifier_decisions", None)
    if isinstance(value, torch.Tensor):
        return value
    return None


def _score_name(info: Dict[str, Any]) -> str:
    for key in ("verifier_scores", "relocal_acceptor", "policy_gate", "confidence", "relocal_conf"):
        if key in info and isinstance(info[key], torch.Tensor):
            return key
    return "unknown"


def _select_by_coverage(scores: np.ndarray, coverage: float) -> np.ndarray:
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    n = int(scores.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=bool)
    k = int(math.ceil(float(coverage) * n))
    k = max(1, min(k, n))
    order = np.argsort(-scores, kind="mergesort")
    sel = np.zeros((n,), dtype=bool)
    sel[order[:k]] = True
    return sel


def _coverage_grid(num_points: int, num_levels: int = 11) -> List[float]:
    if num_points <= 0:
        return []
    levels = np.linspace(0.1, 1.0, num_levels)
    return [float(x) for x in levels]


def _ece(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    if scores.size == 0:
        return float("nan")
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(scores.shape[0])
    out = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi < 1.0:
            mask = (scores >= lo) & (scores < hi)
        else:
            mask = (scores >= lo) & (scores <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(scores[mask]))
        acc = float(np.mean(labels[mask]))
        out += (float(np.sum(mask)) / total) * abs(acc - conf)
    return float(out)


def _brier(scores: np.ndarray, labels: np.ndarray) -> float:
    if scores.size == 0:
        return float("nan")
    scores = np.asarray(scores).reshape(-1).astype(np.float32)
    labels = np.asarray(labels).reshape(-1).astype(np.float32)
    return float(np.mean((scores - labels) ** 2))


def _aurc(risks: Sequence[float], coverages: Sequence[float]) -> float:
    if not risks or not coverages:
        return float("nan")
    paired = sorted(zip(coverages, risks))
    if len(paired) < 2:
        return float("nan")
    xs = np.asarray([p[0] for p in paired], dtype=np.float32)
    ys = np.asarray([p[1] for p in paired], dtype=np.float32)
    return float(np.trapz(ys, xs))


def _risk_coverage_curve(losses: np.ndarray, scores: np.ndarray, coverages: Sequence[float]) -> Dict[str, Any]:
    losses = np.asarray(losses).reshape(-1).astype(np.float32)
    scores = np.asarray(scores).reshape(-1).astype(np.float32)
    if losses.size == 0:
        return {"coverage": [], "risk": [], "aurc": float("nan"), "coverage_at_90_risk": float("nan")}
    out_cov: List[float] = []
    out_risk: List[float] = []
    for cov in coverages:
        sel = _select_by_coverage(scores, cov)
        if not np.any(sel):
            continue
        out_cov.append(float(np.mean(sel)))
        out_risk.append(float(np.mean(losses[sel])))
    aurc = _aurc(out_risk, out_cov)
    coverage_at_90_risk = float("nan")
    for cov, risk in zip(out_cov, out_risk):
        if risk <= 0.10:
            coverage_at_90_risk = cov
            break
    return {
        "coverage": out_cov,
        "risk": out_risk,
        "aurc": aurc,
        "coverage_at_90_risk": coverage_at_90_risk,
    }


def _mean_std(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr).reshape(-1).astype(np.float32)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def _query_metric_row(
    *,
    aj: float,
    oa: float,
    p1: float,
    p2: float,
    p4: float,
    p8: float,
    p16: float,
    avg_pts: float,
    avg_err: float,
    med_err: float,
) -> Dict[str, float]:
    return {
        "AJ": float(aj),
        "OA": float(oa),
        "<1px": float(p1),
        "<2px": float(p2),
        "<4px": float(p4),
        "<8px": float(p8),
        "<16px": float(p16),
        "<avg": float(avg_pts),
        "avg_error_px": float(avg_err),
        "median_error_px": float(med_err),
    }


def _reduce_query_signal(
    signal: Optional[torch.Tensor],
    *,
    num_queries: int,
    mask: Optional[torch.Tensor] = None,
) -> Optional[np.ndarray]:
    """
    Reduce a per-frame/per-query score tensor to one scalar per query.

    Expected shapes are usually (N, T), (N, T, 1), or (N,).
    """
    if not isinstance(signal, torch.Tensor):
        return None

    x = signal.detach().float()
    while x.ndim > 1 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    if x.ndim == 0:
        return np.asarray([float(x.item())], dtype=np.float32)

    if x.ndim == 1:
        if x.shape[0] != num_queries:
            return None
        return x.cpu().numpy().reshape(-1).astype(np.float32)

    if x.shape[0] != num_queries:
        query_axes = [i for i, dim in enumerate(x.shape) if int(dim) == int(num_queries)]
        if not query_axes:
            return None
        x = torch.movedim(x, query_axes[0], 0)

    if x.shape[0] != num_queries:
        return None

    if isinstance(mask, torch.Tensor):
        m = mask.detach().bool()
        while m.ndim > 1 and m.shape[-1] == 1:
            m = m.squeeze(-1)
        if m.ndim == 1 and m.shape[0] == num_queries:
            return x.reshape(num_queries, -1).mean(dim=1).cpu().numpy().astype(np.float32)
        if m.shape == x.shape:
            out: List[float] = []
            for i in range(num_queries):
                vals = x[i][m[i]]
                if vals.numel() == 0:
                    vals = x[i].reshape(-1)
                out.append(float(vals.mean().item()) if vals.numel() else 0.0)
            return np.asarray(out, dtype=np.float32)

    return x.reshape(num_queries, -1).mean(dim=1).cpu().numpy().astype(np.float32)


def _reduce_query_bool(
    signal: Optional[torch.Tensor],
    *,
    num_queries: int,
    mask: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
) -> Optional[np.ndarray]:
    values = _reduce_query_signal(signal, num_queries=num_queries, mask=mask)
    if values is None:
        return None
    return values >= float(threshold)


def _default_score_threshold(config: Any, model: torch.nn.Module) -> float:
    """
    Resolve the default operating threshold.

    Prefer explicit eval thresholds in the config, then model attributes, then 0.5.
    """
    candidates: List[Any] = []
    loss_cfg = getattr(config, "loss", None)
    if loss_cfg is not None:
        for group_name in ("verifier", "relocal_acceptor"):
            group = getattr(loss_cfg, group_name, None)
            if group is None:
                continue
            for key in ("eval_threshold", "threshold"):
                value = getattr(group, key, None)
                if value is not None:
                    candidates.append(value)
    for attr in ("relocal_acceptor_threshold", "verifier_threshold", "verifier_eval_threshold"):
        value = getattr(model, attr, None)
        if value is not None:
            candidates.append(value)
    for value in candidates:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            continue
    return 0.5


def _select_topk(scores: np.ndarray, coverage: float) -> np.ndarray:
    scores = np.asarray(scores).reshape(-1).astype(np.float32)
    n = int(scores.size)
    if n <= 0:
        return np.zeros((0,), dtype=bool)
    k = int(math.ceil(float(coverage) * n))
    k = max(1, min(k, n))
    order = np.argsort(-scores, kind="mergesort")
    sel = np.zeros((n,), dtype=bool)
    sel[order[:k]] = True
    return sel


def _coverage_grid(num_points: int, num_levels: int = 11) -> List[float]:
    if num_points <= 0:
        return []
    return [float(x) for x in np.linspace(0.1, 1.0, num_levels)]


def _ece(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    scores = np.asarray(scores).reshape(-1).astype(np.float32)
    labels = np.asarray(labels).reshape(-1).astype(np.float32)
    if scores.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(scores.size)
    out = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi < 1.0:
            mask = (scores >= lo) & (scores < hi)
        else:
            mask = (scores >= lo) & (scores <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(scores[mask]))
        acc = float(np.mean(labels[mask]))
        out += (float(np.sum(mask)) / total) * abs(acc - conf)
    return float(out)


def _brier(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores).reshape(-1).astype(np.float32)
    labels = np.asarray(labels).reshape(-1).astype(np.float32)
    if scores.size == 0:
        return float("nan")
    return float(np.mean((scores - labels) ** 2))


def _mean_std(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr).reshape(-1).astype(np.float32)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def _curve_from_selection(
    *,
    base_err: np.ndarray,
    pred_err: np.ndarray,
    base_aj: np.ndarray,
    pred_aj: np.ndarray,
    base_oa: np.ndarray,
    pred_oa: np.ndarray,
    scores: np.ndarray,
    coverages: Sequence[float],
) -> Dict[str, Any]:
    base_err = np.asarray(base_err).reshape(-1).astype(np.float32)
    pred_err = np.asarray(pred_err).reshape(-1).astype(np.float32)
    base_aj = np.asarray(base_aj).reshape(-1).astype(np.float32)
    pred_aj = np.asarray(pred_aj).reshape(-1).astype(np.float32)
    base_oa = np.asarray(base_oa).reshape(-1).astype(np.float32)
    pred_oa = np.asarray(pred_oa).reshape(-1).astype(np.float32)
    scores = np.asarray(scores).reshape(-1).astype(np.float32)

    coverage_vals: List[float] = []
    mixed_err: List[float] = []
    mixed_aj: List[float] = []
    mixed_oa: List[float] = []
    for cov in coverages:
        sel = _select_topk(scores, cov)
        if not np.any(sel):
            continue
        coverage_vals.append(float(np.mean(sel)))
        mixed_err.append(float(np.mean(np.where(sel, pred_err, base_err))))
        mixed_aj.append(float(np.mean(np.where(sel, pred_aj, base_aj))))
        mixed_oa.append(float(np.mean(np.where(sel, pred_oa, base_oa))))

    aurc = float(np.trapz(mixed_err, coverage_vals)) if len(coverage_vals) >= 2 else float("nan")
    return {
        "coverage": coverage_vals,
        "avg_error_px": mixed_err,
        "AJ": mixed_aj,
        "OA": mixed_oa,
        "aurc": aurc,
    }


@dataclass
class SelectiveAccumulator:
    threshold: int
    default_tau: float
    tau_grid: List[float] = field(default_factory=list)
    num_videos: int = 0
    num_queries: int = 0
    base_rows: List[Dict[str, float]] = field(default_factory=list)
    pred_rows: List[Dict[str, float]] = field(default_factory=list)
    selected_rows: List[Dict[str, float]] = field(default_factory=list)
    oracle_rows: List[Dict[str, float]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    gain_err_labels: List[float] = field(default_factory=list)
    gain_aj_labels: List[float] = field(default_factory=list)
    base_errs: List[float] = field(default_factory=list)
    pred_errs: List[float] = field(default_factory=list)
    base_ajs: List[float] = field(default_factory=list)
    pred_ajs: List[float] = field(default_factory=list)
    base_oas: List[float] = field(default_factory=list)
    pred_oas: List[float] = field(default_factory=list)
    selected_mask: List[float] = field(default_factory=list)
    oracle_mask: List[float] = field(default_factory=list)

    def add_video(
        self,
        *,
        base_rows: List[Dict[str, float]],
        pred_rows: List[Dict[str, float]],
        scores: np.ndarray,
        selected_mask: np.ndarray,
        oracle_mask: np.ndarray,
        gain_err_labels: np.ndarray,
        gain_aj_labels: np.ndarray,
    ) -> None:
        if len(base_rows) == 0:
            return
        self.num_videos += 1
        self.num_queries += int(len(base_rows))
        self.base_rows.extend(base_rows)
        self.pred_rows.extend(pred_rows)
        self.scores.extend([float(x) for x in np.asarray(scores).reshape(-1).tolist()])
        self.gain_err_labels.extend([float(x) for x in np.asarray(gain_err_labels).reshape(-1).tolist()])
        self.gain_aj_labels.extend([float(x) for x in np.asarray(gain_aj_labels).reshape(-1).tolist()])
        self.selected_mask.extend([float(x) for x in np.asarray(selected_mask).reshape(-1).tolist()])
        self.oracle_mask.extend([float(x) for x in np.asarray(oracle_mask).reshape(-1).tolist()])

        for i, sel in enumerate(np.asarray(selected_mask).reshape(-1).astype(bool)):
            self.selected_rows.append(pred_rows[i] if sel else base_rows[i])
        for i, sel in enumerate(np.asarray(oracle_mask).reshape(-1).astype(bool)):
            self.oracle_rows.append(pred_rows[i] if sel else base_rows[i])

        for row in base_rows:
            self.base_errs.append(float(row["avg_error_px"]))
            self.base_ajs.append(float(row["AJ"]))
            self.base_oas.append(float(row["OA"]))
        for row in pred_rows:
            self.pred_errs.append(float(row["avg_error_px"]))
            self.pred_ajs.append(float(row["AJ"]))
            self.pred_oas.append(float(row["OA"]))

    def _summarize_mask(self, selected_mask: np.ndarray) -> Dict[str, Any]:
        selected_mask = np.asarray(selected_mask).reshape(-1).astype(bool)
        if selected_mask.size != len(self.base_rows):
            raise ValueError(
                "Selected mask length mismatch: "
                f"mask={selected_mask.size} rows={len(self.base_rows)}"
            )

        base = _mean_metrics(self.base_rows)
        pred = _mean_metrics(self.pred_rows)
        selected_rows = [self.pred_rows[i] if sel else self.base_rows[i] for i, sel in enumerate(selected_mask)]
        oracle_mask = np.asarray(self.oracle_mask, dtype=np.float32).astype(bool)
        oracle_rows = [self.pred_rows[i] if sel else self.base_rows[i] for i, sel in enumerate(oracle_mask)]
        selected = _mean_metrics(selected_rows)
        oracle = _mean_metrics(oracle_rows)
        delta_pred = {k: pred[k] - base[k] for k in pred.keys() if k in base}
        delta_selected = {k: selected[k] - base[k] for k in selected.keys() if k in base}
        delta_oracle = {k: oracle[k] - base[k] for k in oracle.keys() if k in base}

        scores = np.asarray(self.scores, dtype=np.float32)
        gain_err = np.asarray(self.gain_err_labels, dtype=np.float32)
        gain_aj = np.asarray(self.gain_aj_labels, dtype=np.float32)
        base_err = np.asarray(self.base_errs, dtype=np.float32)
        pred_err = np.asarray(self.pred_errs, dtype=np.float32)
        base_aj = np.asarray(self.base_ajs, dtype=np.float32)
        pred_aj = np.asarray(self.pred_ajs, dtype=np.float32)
        base_oa = np.asarray(self.base_oas, dtype=np.float32)
        pred_oa = np.asarray(self.pred_oas, dtype=np.float32)

        selected_rate = float(np.mean(selected_mask)) if selected_mask.size else float("nan")
        oracle_rate = float(np.mean(oracle_mask)) if oracle_mask.size else float("nan")
        gain_err_rate = float(np.mean(gain_err)) if gain_err.size else float("nan")
        gain_aj_rate = float(np.mean(gain_aj)) if gain_aj.size else float("nan")

        selected_tp = float(np.sum(selected_mask & (gain_err > 0.5)))
        selected_pred_pos = float(np.sum(selected_mask))
        selected_target = float(np.sum(gain_err > 0.5))
        precision = float(selected_tp / max(selected_pred_pos, 1.0))
        recall = float(selected_tp / max(selected_target, 1.0))
        accuracy = float(np.mean(selected_mask == (gain_err > 0.5))) if gain_err.size else float("nan")

        selected_base_err = float(np.mean(base_err[selected_mask])) if np.any(selected_mask) else float("nan")
        selected_pred_err = float(np.mean(pred_err[selected_mask])) if np.any(selected_mask) else float("nan")
        oracle_err = float(np.mean(np.minimum(base_err, pred_err))) if base_err.size else float("nan")
        selected_gap_closed = float(
            (selected_base_err - selected_pred_err)
            / max((float(np.mean(base_err)) - oracle_err), 1.0e-6)
        ) if base_err.size and pred_err.size else float("nan")

        default_curve_mask = scores >= float(self.default_tau)
        default_curve = _curve_from_selection(
            base_err=base_err,
            pred_err=pred_err,
            base_aj=base_aj,
            pred_aj=pred_aj,
            base_oa=base_oa,
            pred_oa=pred_oa,
            scores=scores,
            coverages=_coverage_grid(len(scores)),
        )

        coverages = _coverage_grid(len(scores))
        selected_curve = _curve_from_selection(
            base_err=base_err,
            pred_err=pred_err,
            base_aj=base_aj,
            pred_aj=pred_aj,
            base_oa=base_oa,
            pred_oa=pred_oa,
            scores=scores,
            coverages=coverages,
        )

        # Selected/oracle rows are already aggregated at the default operating point.
        selected_error_mean = float(selected.get("avg_error_px", float("nan")))
        oracle_error_mean = float(oracle.get("avg_error_px", float("nan")))
        base_error_mean = float(base.get("avg_error_px", float("nan")))
        oracle_gap_closed = float(
            (base_error_mean - selected_error_mean) / max(base_error_mean - oracle_error_mean, 1.0e-6)
        ) if np.isfinite(base_error_mean) and np.isfinite(selected_error_mean) and np.isfinite(oracle_error_mean) else float("nan")

        base_aj_mean = float(base.get("AJ", float("nan")))
        selected_aj_mean = float(selected.get("AJ", float("nan")))
        oracle_aj_mean = float(oracle.get("AJ", float("nan")))
        oracle_gap_closed_aj = float(
            (selected_aj_mean - base_aj_mean) / max(oracle_aj_mean - base_aj_mean, 1.0e-6)
        ) if np.isfinite(base_aj_mean) and np.isfinite(selected_aj_mean) and np.isfinite(oracle_aj_mean) else float("nan")

        return {
            "threshold": self.threshold,
            "num_videos": self.num_videos,
            "num_queries": self.num_queries,
            "default_tau": float(self.default_tau),
            "base": base,
            "pred": pred,
            "selected": selected,
            "oracle": oracle,
            "delta_pred": delta_pred,
            "delta_selected": delta_selected,
            "delta_oracle": delta_oracle,
            "score_stats": {
                "mean": float(np.mean(scores)) if scores.size else float("nan"),
                "std": float(np.std(scores)) if scores.size else float("nan"),
                "min": float(np.min(scores)) if scores.size else float("nan"),
                "max": float(np.max(scores)) if scores.size else float("nan"),
            },
            "gain_err_rate": gain_err_rate,
            "gain_aj_rate": gain_aj_rate,
            "selected_rate": selected_rate,
            "oracle_rate": oracle_rate,
            "selected_precision": precision,
            "selected_recall": recall,
            "selected_accuracy": accuracy,
            "selected_mean_score": float(np.mean(scores[selected_mask])) if np.any(selected_mask) else float("nan"),
            "selected_mean_error": selected_error_mean,
            "oracle_mean_error": oracle_error_mean,
            "oracle_gap_closed": oracle_gap_closed,
            "oracle_gap_closed_aj": oracle_gap_closed_aj,
            "ece": _ece(scores, gain_err),
            "brier": _brier(scores, gain_err),
            "risk_coverage": default_curve,
            "selective_curve": selected_curve,
        }

    def summarize(self) -> Dict[str, Any]:
        default_mask = np.asarray(self.selected_mask, dtype=np.float32).astype(bool)
        summary = self._summarize_mask(default_mask)
        summary.update(
            {
                "threshold": self.threshold,
                "num_videos": self.num_videos,
                "num_queries": self.num_queries,
                "default_tau": float(self.default_tau),
            }
        )
        if self.tau_grid:
            summary["tau_sweep"] = []
            scores = np.asarray(self.scores, dtype=np.float32)
            for tau in self.tau_grid:
                tau_mask = scores >= float(tau)
                tau_summary = self._summarize_mask(tau_mask)
                summary["tau_sweep"].append(
                    {
                        "tau": float(tau),
                        "selected_rate": tau_summary["selected_rate"],
                        "selected_precision": tau_summary["selected_precision"],
                        "selected_recall": tau_summary["selected_recall"],
                        "selected_accuracy": tau_summary["selected_accuracy"],
                        "selected_mean_error": tau_summary["selected_mean_error"],
                        "oracle_gap_closed": tau_summary["oracle_gap_closed"],
                        "oracle_gap_closed_aj": tau_summary["oracle_gap_closed_aj"],
                        "ece": tau_summary["ece"],
                        "brier": tau_summary["brier"],
                        "delta_selected": tau_summary["delta_selected"],
                    }
                )
        return summary


def _fmt(v: Optional[float], precision: int = 6, width: int = 10) -> str:
    if v is None:
        return "-".rjust(width)
    try:
        fv = float(v)
    except Exception:
        return "-".rjust(width)
    if not np.isfinite(fv):
        return "nan".rjust(width)
    return f"{fv:{width}.{precision}f}"


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="Selective evaluation for calibrated point tracking.")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--thresholds", type=str, default="10,20,30", help="Comma-separated long-occlusion thresholds.")
    parser.add_argument(
        "--metric-resolution-mode",
        type=str,
        default="original",
        choices=("original", "input"),
        help="Compute metrics in original resolution (paper) or input resolution (debug).",
    )
    parser.add_argument("--query-mode", type=str, default="", help="Override query_mode (first|strided).")
    parser.add_argument(
        "--score-source",
        type=str,
        default="auto",
        help="Score source to evaluate: auto|verifier_scores|relocal_acceptor|relocal_conf|policy_gate|confidence|pred_visibility.",
    )
    parser.add_argument(
        "--default-tau",
        type=float,
        default=None,
        help="Override the default operating threshold for selected outputs.",
    )
    parser.add_argument(
        "--tau-grid",
        type=str,
        default="",
        help="Optional comma-separated tau grid for sweep results, e.g. 0.1,0.2,0.3.",
    )
    parser.add_argument("--use-ema", action="store_true", help="Force using EMA weights if present.")
    parser.add_argument("--no-ema", action="store_true", help="Force not using EMA weights.")
    parser.add_argument("--max-videos", type=int, default=0, help="Limit evaluation to first N videos (0=all).")
    parser.add_argument("--output-json", type=str, default="", help="Optional path to save JSON summary.")
    args = parser.parse_args()

    if args.use_ema and args.no_ema:
        raise ValueError("Conflicting flags: --use-ema and --no-ema")
    use_ema: Optional[bool] = None
    if args.use_ema:
        use_ema = True
    if args.no_ema:
        use_ema = False

    thresholds = [int(x) for x in str(args.thresholds).split(",") if str(x).strip()]
    thresholds = sorted(set(int(x) for x in thresholds if int(x) > 0))
    if not thresholds:
        raise ValueError("No valid thresholds provided.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)

    model, config = _load_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        use_ema=use_ema,
        device=device,
    )
    dataloader = _build_val_loader_from_config(config)
    query_mode = _resolve_query_mode(config, args.query_mode)
    default_tau = float(args.default_tau) if args.default_tau is not None else _default_score_threshold(config, model)
    tau_grid = [float(x) for x in str(args.tau_grid).split(",") if str(x).strip()]
    tau_grid = sorted(set(float(x) for x in tau_grid if np.isfinite(float(x))))

    accumulators = {
        thr: SelectiveAccumulator(threshold=thr, default_tau=default_tau, tau_grid=tau_grid) for thr in thresholds
    }

    score_key = "unknown"
    score_source_arg = str(args.score_source or "auto").strip().lower()
    iterator = tqdm(dataloader, desc="Selective eval", disable=False)
    for idx, batch in enumerate(iterator):
        if args.max_videos and idx >= int(args.max_videos):
            break
        if batch is None or not isinstance(batch, dict):
            continue
        if "video" not in batch or "query_points" not in batch or "target_points" not in batch or "occluded" not in batch:
            continue

        video = batch["video"].to(device)
        query_points = batch["query_points"].to(device)
        target_points = batch["target_points"].to(device)
        occluded = _as_bool(batch["occluded"].to(device))

        video = _ensure_batch_dim(video, 4)
        query_points = _ensure_batch_dim(query_points, 2)
        target_points = _ensure_batch_dim(target_points, 3)
        occluded = _ensure_batch_dim(occluded, 2)

        bsz = int(video.shape[0])
        input_resolution = tuple(video.shape[-2:])

        meta = {
            "video_name": batch.get("video_name", None),
            "base_tracks": batch.get("base_tracks", None),
            "base_visibility": batch.get("base_visibility", None),
        }

        with torch.no_grad():
            try:
                outputs = model(video, query_points, meta=meta, return_info=True)
            except TypeError:
                try:
                    outputs = model(video, query_points, return_info=True)
                except TypeError:
                    outputs = model(video, query_points)

        if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
            raise ValueError(f"Unexpected model outputs: {type(outputs)}")

        pred_tracks = outputs[0]
        pred_visibility = _as_bool(outputs[1])
        info: Dict[str, Any] = {}
        if len(outputs) >= 3 and isinstance(outputs[2], dict):
            info = outputs[2]

        base_tracks = info.get("base_tracks", batch.get("base_tracks", None))
        base_visibility = info.get("base_visibility", batch.get("base_visibility", None))
        if not isinstance(base_tracks, torch.Tensor) or not isinstance(base_visibility, torch.Tensor):
            raise ValueError("Selective evaluation requires base_tracks/base_visibility.")
        base_tracks = base_tracks.to(device)
        base_visibility = _as_bool(base_visibility.to(device))

        score_tensor, info_score_key = _extract_score_tensor(info, source=score_source_arg)
        mask_tensor = _find_mask_tensor(info)
        decision_tensor = _find_decision_tensor(info)
        if info_score_key != "unknown":
            score_key = info_score_key

        for b in range(bsz):
            if args.metric_resolution_mode == "input":
                resolution = input_resolution
            else:
                resolution = _resolve_resolution_from_batch(batch, b, input_resolution)

            q_t = query_points[b, :, 0].round().long().clamp(0, pred_tracks.shape[2] - 1)
            max_occ = _max_reappearance_occlusion_run(occluded[b], q_t)

            q_px = _to_pixel_query_points(query_points[b], resolution)
            pred_px = _to_pixel_tracks(pred_tracks[b], resolution)
            base_px = _to_pixel_tracks(base_tracks[b], resolution)
            gt_px = _to_pixel_tracks(target_points[b], resolution)

            pred_metrics = _trackwise_official_metrics(
                pred_px,
                gt_px,
                pred_visibility[b],
                ~occluded[b],
                q_px,
                query_mode,
            )
            base_metrics = _trackwise_official_metrics(
                base_px,
                gt_px,
                base_visibility[b],
                ~occluded[b],
                q_px,
                query_mode,
            )

            pred_avg_err, pred_med_err = _per_query_error_stats(
                pred_px,
                gt_px,
                ~occluded[b],
                q_px,
                query_mode,
            )
            base_avg_err, base_med_err = _per_query_error_stats(
                base_px,
                gt_px,
                ~occluded[b],
                q_px,
                query_mode,
            )

            num_queries = int(pred_tracks.shape[1])
            score_q = _reduce_query_signal(
                score_tensor[b] if isinstance(score_tensor, torch.Tensor) and score_tensor.shape[0] == pred_tracks.shape[0] else None,
                num_queries=num_queries,
                mask=mask_tensor[b] if isinstance(mask_tensor, torch.Tensor) and mask_tensor.shape[0] == pred_tracks.shape[0] else None,
            )
            if score_q is None:
                score_q = _reduce_query_signal(
                    pred_visibility[b].float(),
                    num_queries=num_queries,
                    mask=None,
                )
                if score_source_arg == "pred_visibility":
                    score_key = "pred_visibility"
                if score_key == "unknown":
                    score_key = "pred_visibility_fallback"
            if score_q is None:
                score_q = np.zeros((num_queries,), dtype=np.float32)
            score_q = np.clip(np.asarray(score_q, dtype=np.float32).reshape(-1), 0.0, 1.0)

            decision_q: Optional[np.ndarray] = None
            use_model_decision = score_source_arg in ("auto", "verifier", "verifier_scores", "relocal_acceptor")
            if use_model_decision and isinstance(decision_tensor, torch.Tensor):
                decision_q = _reduce_query_bool(
                    decision_tensor[b],
                    num_queries=num_queries,
                    mask=mask_tensor[b] if isinstance(mask_tensor, torch.Tensor) and mask_tensor.shape[0] == pred_tracks.shape[0] else None,
                    threshold=0.5,
                )
            if decision_q is None:
                decision_q = score_q >= float(default_tau)

            gain_err = (pred_avg_err.detach().cpu().numpy().reshape(-1) < base_avg_err.detach().cpu().numpy().reshape(-1)).astype(np.float32)
            gain_aj = (np.asarray(pred_metrics["average_jaccard"]).reshape(-1) > np.asarray(base_metrics["average_jaccard"]).reshape(-1)).astype(np.float32)

            pred_err_q = pred_avg_err.detach().cpu().numpy().reshape(-1).astype(np.float32)
            base_err_q = base_avg_err.detach().cpu().numpy().reshape(-1).astype(np.float32)
            pred_aj_q = np.asarray(pred_metrics["average_jaccard"]).reshape(-1).astype(np.float32)
            base_aj_q = np.asarray(base_metrics["average_jaccard"]).reshape(-1).astype(np.float32)
            pred_oa_q = np.asarray(pred_metrics["occlusion_accuracy"]).reshape(-1).astype(np.float32)
            base_oa_q = np.asarray(base_metrics["occlusion_accuracy"]).reshape(-1).astype(np.float32)
            pred_p1_q = np.asarray(pred_metrics["pts_within_1"]).reshape(-1).astype(np.float32)
            base_p1_q = np.asarray(base_metrics["pts_within_1"]).reshape(-1).astype(np.float32)
            pred_p2_q = np.asarray(pred_metrics["pts_within_2"]).reshape(-1).astype(np.float32)
            base_p2_q = np.asarray(base_metrics["pts_within_2"]).reshape(-1).astype(np.float32)
            pred_p4_q = np.asarray(pred_metrics["pts_within_4"]).reshape(-1).astype(np.float32)
            base_p4_q = np.asarray(base_metrics["pts_within_4"]).reshape(-1).astype(np.float32)
            pred_p8_q = np.asarray(pred_metrics["pts_within_8"]).reshape(-1).astype(np.float32)
            base_p8_q = np.asarray(base_metrics["pts_within_8"]).reshape(-1).astype(np.float32)
            pred_p16_q = np.asarray(pred_metrics["pts_within_16"]).reshape(-1).astype(np.float32)
            base_p16_q = np.asarray(base_metrics["pts_within_16"]).reshape(-1).astype(np.float32)
            pred_ap_q = np.asarray(pred_metrics["average_pts_within_thresh"]).reshape(-1).astype(np.float32)
            base_ap_q = np.asarray(base_metrics["average_pts_within_thresh"]).reshape(-1).astype(np.float32)
            pred_med_q = pred_med_err.detach().cpu().numpy().reshape(-1).astype(np.float32)
            base_med_q = base_med_err.detach().cpu().numpy().reshape(-1).astype(np.float32)

            if not (len(score_q) == len(base_err_q) == len(pred_err_q) == len(gain_err)):
                raise ValueError(
                    "Per-query arrays are misaligned: "
                    f"score={len(score_q)} base_err={len(base_err_q)} pred_err={len(pred_err_q)} gain={len(gain_err)}"
                )

            base_rows = [
                _query_metric_row(
                    aj=float(base_aj_q[i]),
                    oa=float(base_oa_q[i]),
                    p1=float(base_p1_q[i]),
                    p2=float(base_p2_q[i]),
                    p4=float(base_p4_q[i]),
                    p8=float(base_p8_q[i]),
                    p16=float(base_p16_q[i]),
                    avg_pts=float(base_ap_q[i]),
                    avg_err=float(base_err_q[i]),
                    med_err=float(base_med_q[i]),
                )
                for i in range(len(score_q))
            ]
            pred_rows = [
                _query_metric_row(
                    aj=float(pred_aj_q[i]),
                    oa=float(pred_oa_q[i]),
                    p1=float(pred_p1_q[i]),
                    p2=float(pred_p2_q[i]),
                    p4=float(pred_p4_q[i]),
                    p8=float(pred_p8_q[i]),
                    p16=float(pred_p16_q[i]),
                    avg_pts=float(pred_ap_q[i]),
                    avg_err=float(pred_err_q[i]),
                    med_err=float(pred_med_q[i]),
                )
                for i in range(len(score_q))
            ]

            selected_mask_q = np.asarray(decision_q, dtype=bool).reshape(-1)
            oracle_mask_q = np.asarray(gain_err, dtype=bool).reshape(-1)

            for thr, acc in accumulators.items():
                subset = np.asarray(max_occ.detach().cpu().numpy().reshape(-1) >= int(thr), dtype=bool)
                if not np.any(subset):
                    continue

                acc.add_video(
                    base_rows=[base_rows[i] for i in np.where(subset)[0].tolist()],
                    pred_rows=[pred_rows[i] for i in np.where(subset)[0].tolist()],
                    scores=score_q[subset],
                    selected_mask=selected_mask_q[subset],
                    oracle_mask=oracle_mask_q[subset],
                    gain_err_labels=gain_err[subset],
                    gain_aj_labels=gain_aj[subset],
                )

    summaries = [accumulators[thr].summarize() for thr in thresholds]

    print()
    print("=== Selective tracking evaluation ===")
    print(
        f"thresholds={thresholds} metric_resolution_mode={args.metric_resolution_mode} "
        f"query_mode={query_mode} score_source={score_key} default_tau={default_tau:.3f} "
        f"tau_grid={tau_grid if tau_grid else '[]'}"
    )
    print()

    header = (
        f"{'thr':>4} | {'vids':>5} {'qs':>6} |"
        f"{'AJ_b':>10} {'AJ_p':>10} {'AJ_s':>10} {'AJ_o':>10} |"
        f"{'OA_b':>10} {'OA_p':>10} {'OA_s':>10} |"
        f"{'cov':>8} {'prec':>8} {'rec':>8} {'ECE':>8} {'AURC':>8}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        base = s["base"]
        pred = s["pred"]
        selected = s["selected"]
        oracle = s["oracle"]
        curve = s["risk_coverage"]
        print(
            f"{int(s['threshold']):4d} |"
            f"{int(s['num_videos']):5d} {int(s['num_queries']):6d} |"
            f"{_fmt(base.get('AJ'))} {_fmt(pred.get('AJ'))} {_fmt(selected.get('AJ'))} {_fmt(oracle.get('AJ'))} |"
            f"{_fmt(base.get('OA'))} {_fmt(pred.get('OA'))} {_fmt(selected.get('OA'))} |"
            f"{_fmt(s.get('selected_rate'))} {_fmt(s.get('selected_precision'))} {_fmt(s.get('selected_recall'))} "
            f"{_fmt(s.get('ece'))} {_fmt(curve.get('aurc'))}"
        )

    payload = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "metric_resolution_mode": args.metric_resolution_mode,
        "query_mode": query_mode,
        "score_source": score_key,
        "default_tau": default_tau,
        "thresholds": thresholds,
        "results": summaries,
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print()
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
