#!/usr/bin/env python3
"""
Evaluate long-occlusion oracle gap on TAP-Vid.

This script reuses the long-occlusion subset definition from
`scripts/eval_long_occlusion_subset.py`, but instead of only comparing mean
base/refined metrics, it computes per-query metrics and forms two oracle
upper-bounds:

  - oracle_by_error: choose base/refined per query by lower GT average error
  - oracle_by_aj:    choose base/refined per query by higher per-query AJ

The primary decision signal is the error-based oracle gap on the official TAP-Vid
metrics. If that upper bound is still small, the current relocalization line is
unlikely to justify further investment.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Make the repository root importable so we can reuse helper functions from
# sibling scripts when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the model / dataloader / subset helpers from the existing script.
from scripts.eval_long_occlusion_subset import (
    _as_bool,
    _build_val_loader_from_config,
    _ensure_batch_dim,
    _load_model,
    _max_reappearance_occlusion_run,
    _resolve_resolution_from_batch,
    _setup_logging,
)

from datasets.tapvid_official_eval import compute_tapvid_metrics_official

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
    Convert [y, x] tracks to pixel [x, y] space used by the official TAP-Vid metric.
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
    """
    Return a (N,T) mask of evaluation frames under the given query protocol.
    """
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


def _per_query_error_stats(
    pred_tracks_px: torch.Tensor,
    gt_tracks_px: torch.Tensor,
    gt_visible: torch.Tensor,
    query_points_px: torch.Tensor,
    query_mode: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-query average and median pixel error on visible evaluation frames.
    """
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


def _trackwise_official_metrics(
    pred_tracks_px: torch.Tensor,
    gt_tracks_px: torch.Tensor,
    pred_visible: torch.Tensor,
    gt_visible: torch.Tensor,
    query_points_px: torch.Tensor,
    query_mode: Optional[str],
) -> Dict[str, np.ndarray]:
    """
    Compute per-query official TAP-Vid metrics.
    """
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


def _safe_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    return float(value)


def _mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    out: Dict[str, float] = {}
    for k in keys:
        vals = []
        for row in rows:
            try:
                v = float(row.get(k, float("nan")))
            except Exception:
                continue
            if np.isfinite(v):
                vals.append(v)
        if vals:
            out[k] = float(np.mean(vals))
    return out


def _metric_row_from_arrays(
    *,
    aj: float,
    occ_acc: float,
    pts1: float,
    pts2: float,
    pts4: float,
    pts8: float,
    pts16: float,
    avg_pts: float,
    avg_err: float,
    med_err: float,
) -> Dict[str, float]:
    return {
        "AJ": float(aj),
        "OA": float(occ_acc),
        "<1px": float(pts1),
        "<2px": float(pts2),
        "<4px": float(pts4),
        "<8px": float(pts8),
        "<16px": float(pts16),
        "<avg": float(avg_pts),
        "avg_error_px": float(avg_err),
        "median_error_px": float(med_err),
    }


@dataclass
class OracleAccumulator:
    threshold: int
    num_videos: int = 0
    num_queries: int = 0
    base_rows: List[Dict[str, float]] = field(default_factory=list)
    refined_rows: List[Dict[str, float]] = field(default_factory=list)
    oracle_error_rows: List[Dict[str, float]] = field(default_factory=list)
    oracle_aj_rows: List[Dict[str, float]] = field(default_factory=list)
    refined_wins_error: int = 0
    refined_wins_aj: int = 0

    def add(
        self,
        *,
        base_rows: List[Dict[str, float]],
        refined_rows: List[Dict[str, float]],
        oracle_error_rows: List[Dict[str, float]],
        oracle_aj_rows: List[Dict[str, float]],
        refined_wins_error: int,
        refined_wins_aj: int,
    ) -> None:
        self.num_videos += 1
        self.num_queries += int(len(base_rows))
        self.base_rows.extend(base_rows)
        self.refined_rows.extend(refined_rows)
        self.oracle_error_rows.extend(oracle_error_rows)
        self.oracle_aj_rows.extend(oracle_aj_rows)
        self.refined_wins_error += int(refined_wins_error)
        self.refined_wins_aj += int(refined_wins_aj)

    def summarize(self) -> Dict[str, Any]:
        base = _mean_metrics(self.base_rows)
        refined = _mean_metrics(self.refined_rows)
        oracle_error = _mean_metrics(self.oracle_error_rows)
        oracle_aj = _mean_metrics(self.oracle_aj_rows)
        return {
            "threshold": self.threshold,
            "num_videos": self.num_videos,
            "num_queries": self.num_queries,
            "base": base,
            "refined": refined,
            "oracle_error": oracle_error,
            "oracle_aj": oracle_aj,
            "refined_wins_error_rate": float(self.refined_wins_error / max(self.num_queries, 1)),
            "refined_wins_aj_rate": float(self.refined_wins_aj / max(self.num_queries, 1)),
        }


def _fmt(v: Optional[float], precision: int = 6, width: int = 10) -> str:
    if v is None:
        return "-".rjust(width)
    return f"{float(v):{width}.{precision}f}"


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="Evaluate oracle gap on the long-occlusion subset.")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--thresholds", type=str, default="10,20,30", help="Comma-separated thresholds.")
    parser.add_argument(
        "--metric-resolution-mode",
        type=str,
        default="original",
        choices=("original", "input"),
        help="Compute metrics in original resolution (paper) or input resolution (debug).",
    )
    parser.add_argument("--query-mode", type=str, default="", help="Override query_mode (first|strided).")
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

    accumulators = {thr: OracleAccumulator(threshold=thr) for thr in thresholds}

    from tqdm import tqdm

    iterator = tqdm(dataloader, desc="Long-occ oracle eval", disable=False)
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

        base_tracks = None
        base_visibility = None
        if len(outputs) >= 3 and isinstance(outputs[2], dict):
            info = outputs[2]
            base_tracks = info.get("base_tracks", None)
            base_visibility = info.get("base_visibility", None)
        if base_tracks is None:
            base_tracks = batch.get("base_tracks", None)
        if base_visibility is None:
            base_visibility = batch.get("base_visibility", None)
        if not isinstance(base_tracks, torch.Tensor) or not isinstance(base_visibility, torch.Tensor):
            raise ValueError("Oracle gap evaluation requires base_tracks/base_visibility.")

        base_tracks = base_tracks.to(device)
        base_visibility = _as_bool(base_visibility.to(device))

        for b in range(bsz):
            resolution = input_resolution if args.metric_resolution_mode == "input" else _resolve_resolution_from_batch(
                batch, b, input_resolution
            )

            t_q = query_points[b, :, 0].round().long().clamp(0, pred_tracks.shape[2] - 1)
            max_occ = _max_reappearance_occlusion_run(occluded[b], t_q)

            for thr, acc in accumulators.items():
                sel = max_occ >= int(thr)
                num_sel = int(sel.long().sum().item())
                if num_sel <= 0:
                    continue

                pred_tracks_sel = pred_tracks[b][sel]
                pred_visibility_sel = pred_visibility[b][sel]
                base_tracks_sel = base_tracks[b][sel]
                base_visibility_sel = base_visibility[b][sel]
                gt_tracks_sel = target_points[b][sel]
                gt_visible_sel = ~occluded[b][sel]
                query_sel = query_points[b][sel]

                pred_tracks_px = _to_pixel_tracks(pred_tracks_sel, resolution)
                base_tracks_px = _to_pixel_tracks(base_tracks_sel, resolution)
                gt_tracks_px = _to_pixel_tracks(gt_tracks_sel, resolution)
                query_points_px = _to_pixel_query_points(query_sel, resolution)

                refined_off = _trackwise_official_metrics(
                    pred_tracks_px,
                    gt_tracks_px,
                    pred_visibility_sel,
                    gt_visible_sel,
                    query_points_px,
                    query_mode,
                )
                base_off = _trackwise_official_metrics(
                    base_tracks_px,
                    gt_tracks_px,
                    base_visibility_sel,
                    gt_visible_sel,
                    query_points_px,
                    query_mode,
                )

                refined_avg_err, refined_med_err = _per_query_error_stats(
                    pred_tracks_px,
                    gt_tracks_px,
                    gt_visible_sel,
                    query_points_px,
                    query_mode,
                )
                base_avg_err, base_med_err = _per_query_error_stats(
                    base_tracks_px,
                    gt_tracks_px,
                    gt_visible_sel,
                    query_points_px,
                    query_mode,
                )

                refined_rows: List[Dict[str, float]] = []
                base_rows: List[Dict[str, float]] = []
                oracle_error_rows: List[Dict[str, float]] = []
                oracle_aj_rows: List[Dict[str, float]] = []
                refined_wins_error = 0
                refined_wins_aj = 0

                aj_refined = np.asarray(refined_off["average_jaccard"]).reshape(-1)
                aj_base = np.asarray(base_off["average_jaccard"]).reshape(-1)
                occ_refined = np.asarray(refined_off["occlusion_accuracy"]).reshape(-1)
                occ_base = np.asarray(base_off["occlusion_accuracy"]).reshape(-1)
                p1_refined = np.asarray(refined_off["pts_within_1"]).reshape(-1)
                p1_base = np.asarray(base_off["pts_within_1"]).reshape(-1)
                p2_refined = np.asarray(refined_off["pts_within_2"]).reshape(-1)
                p2_base = np.asarray(base_off["pts_within_2"]).reshape(-1)
                p4_refined = np.asarray(refined_off["pts_within_4"]).reshape(-1)
                p4_base = np.asarray(base_off["pts_within_4"]).reshape(-1)
                p8_refined = np.asarray(refined_off["pts_within_8"]).reshape(-1)
                p8_base = np.asarray(base_off["pts_within_8"]).reshape(-1)
                p16_refined = np.asarray(refined_off["pts_within_16"]).reshape(-1)
                p16_base = np.asarray(base_off["pts_within_16"]).reshape(-1)
                ap_refined = np.asarray(refined_off["average_pts_within_thresh"]).reshape(-1)
                ap_base = np.asarray(base_off["average_pts_within_thresh"]).reshape(-1)

                for i in range(num_sel):
                    base_row = _metric_row_from_arrays(
                        aj=_safe_float(aj_base[i]),
                        occ_acc=_safe_float(occ_base[i]),
                        pts1=_safe_float(p1_base[i]),
                        pts2=_safe_float(p2_base[i]),
                        pts4=_safe_float(p4_base[i]),
                        pts8=_safe_float(p8_base[i]),
                        pts16=_safe_float(p16_base[i]),
                        avg_pts=_safe_float(ap_base[i]),
                        avg_err=_safe_float(base_avg_err[i]),
                        med_err=_safe_float(base_med_err[i]),
                    )
                    refined_row = _metric_row_from_arrays(
                        aj=_safe_float(aj_refined[i]),
                        occ_acc=_safe_float(occ_refined[i]),
                        pts1=_safe_float(p1_refined[i]),
                        pts2=_safe_float(p2_refined[i]),
                        pts4=_safe_float(p4_refined[i]),
                        pts8=_safe_float(p8_refined[i]),
                        pts16=_safe_float(p16_refined[i]),
                        avg_pts=_safe_float(ap_refined[i]),
                        avg_err=_safe_float(refined_avg_err[i]),
                        med_err=_safe_float(refined_med_err[i]),
                    )

                    base_rows.append(base_row)
                    refined_rows.append(refined_row)

                    choose_refined_error = refined_row["avg_error_px"] <= base_row["avg_error_px"]
                    choose_refined_aj = refined_row["AJ"] >= base_row["AJ"]
                    if choose_refined_error:
                        refined_wins_error += 1
                        oracle_error_rows.append(refined_row)
                    else:
                        oracle_error_rows.append(base_row)
                    if choose_refined_aj:
                        refined_wins_aj += 1
                        oracle_aj_rows.append(refined_row)
                    else:
                        oracle_aj_rows.append(base_row)

                acc.add(
                    base_rows=base_rows,
                    refined_rows=refined_rows,
                    oracle_error_rows=oracle_error_rows,
                    oracle_aj_rows=oracle_aj_rows,
                    refined_wins_error=refined_wins_error,
                    refined_wins_aj=refined_wins_aj,
                )

    summaries = [accumulators[thr].summarize() for thr in thresholds]

    print()
    print("=== Long-occlusion oracle gap ===")
    print(f"thresholds={thresholds} metric_resolution_mode={args.metric_resolution_mode} query_mode={query_mode}")
    print()

    header = (
        f"{'thr':>4} | {'vids':>5} {'qs':>6} |"
        f"{'AJ_b':>10} {'AJ_r':>10} {'AJ_oE':>10} {'AJ_oA':>10} |"
        f"{'err_b':>10} {'err_r':>10} {'err_oE':>10} |"
        f"{'refE%':>8} {'refA%':>8}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        base = s["base"]
        refined = s["refined"]
        oracle_error = s["oracle_error"]
        oracle_aj = s["oracle_aj"]
        print(
            f"{int(s['threshold']):4d} |"
            f"{int(s['num_videos']):5d} {int(s['num_queries']):6d} |"
            f"{_fmt(base.get('AJ'))} {_fmt(refined.get('AJ'))} {_fmt(oracle_error.get('AJ'))} {_fmt(oracle_aj.get('AJ'))} |"
            f"{_fmt(base.get('avg_error_px'))} {_fmt(refined.get('avg_error_px'))} {_fmt(oracle_error.get('avg_error_px'))} |"
            f"{100.0 * float(s['refined_wins_error_rate']):8.2f} {100.0 * float(s['refined_wins_aj_rate']):8.2f}"
        )

    if args.output_json:
        out_path = Path(args.output_json)
        payload = {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "metric_resolution_mode": args.metric_resolution_mode,
            "query_mode": query_mode,
            "thresholds": thresholds,
            "results": summaries,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print()
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
