#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils import data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACKON_ROOT = PROJECT_ROOT / "baselines" / "track_on"
for path in (PROJECT_ROOT, TRACKON_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from dataset.tapvid import TAPVid  # type: ignore
from model.trackon_predictor import Predictor as TrackOnPredictor  # type: ignore
from utils.coord_utils import get_points_on_a_grid  # type: ignore
from utils.eval_utils import compute_tapvid_metrics  # type: ignore
from utils.train_utils import load_args_from_yaml  # type: ignore


VARIANTS = ("baseline", "oracle_mask", "anchor_only", "oracle_mask_anchor")


def _compute_delta_avg(metrics: Dict[str, float]) -> Optional[float]:
    keys = ("pts_within_1", "pts_within_2", "pts_within_4", "pts_within_8", "pts_within_16")
    values = [metrics.get(k) for k in keys if metrics.get(k) is not None]
    if not values:
        return None
    return float(np.mean(values) * 100.0)


def _compute_summary_from_official(metrics: Dict[str, np.ndarray]) -> Dict[str, float]:
    out = {
        "AJ": float(np.asarray(metrics["average_jaccard"]).reshape(-1)[0] * 100.0),
        "OA": float(np.asarray(metrics["occlusion_accuracy"]).reshape(-1)[0] * 100.0),
        "<4px": float(np.asarray(metrics["pts_within_4"]).reshape(-1)[0] * 100.0),
    }
    delta_avg = _compute_delta_avg(
        {
            "pts_within_1": float(np.asarray(metrics["pts_within_1"]).reshape(-1)[0]),
            "pts_within_2": float(np.asarray(metrics["pts_within_2"]).reshape(-1)[0]),
            "pts_within_4": float(np.asarray(metrics["pts_within_4"]).reshape(-1)[0]),
            "pts_within_8": float(np.asarray(metrics["pts_within_8"]).reshape(-1)[0]),
            "pts_within_16": float(np.asarray(metrics["pts_within_16"]).reshape(-1)[0]),
        }
    )
    if delta_avg is not None:
        out["delta_avg"] = delta_avg
    return out


def _mean_dict(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for key in keys:
        values: List[float] = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            value_f = float(value)
            if np.isfinite(value_f):
                values.append(value_f)
        out[key] = float(np.mean(values)) if values else None
    return out


def _max_reappearance_occlusion_run(gt_visibility: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    n_queries, num_frames = gt_visibility.shape
    out = np.zeros((n_queries,), dtype=np.int64)
    gt_occluded = ~gt_visibility
    for q in range(n_queries):
        start = int(query_t[q]) + 1
        if start >= num_frames:
            continue
        best = 0
        run = 0
        for val in gt_occluded[q, start:].tolist():
            if val:
                run += 1
                continue
            if run > best:
                best = run
            run = 0
        out[q] = best
    return out


def _reentry_frame_from_longest_run(gt_visibility: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    n_queries, num_frames = gt_visibility.shape
    out = np.full((n_queries,), -1, dtype=np.int64)
    gt_occluded = ~gt_visibility
    for q in range(n_queries):
        start = int(query_t[q]) + 1
        if start >= num_frames:
            continue
        best = 0
        best_reentry = -1
        run = 0
        for frame_idx in range(start, num_frames):
            if gt_occluded[q, frame_idx]:
                run += 1
                continue
            if run > best:
                best = run
                best_reentry = frame_idx
            run = 0
        out[q] = best_reentry
    return out


def _subset_official_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    mask: np.ndarray,
    query_mode: str,
) -> Optional[Dict[str, float]]:
    if mask.sum() == 0:
        return None
    metrics = compute_tapvid_metrics(
        query_points[:, mask],
        gt_occluded[:, mask],
        gt_tracks[:, mask],
        pred_occluded[:, mask],
        pred_tracks[:, mask],
        query_mode,
    )
    return _compute_summary_from_official(metrics)


def _reentry_metrics(
    pred_tracks_nt: np.ndarray,
    gt_tracks_nt: np.ndarray,
    gt_visibility_nt: np.ndarray,
    query_t: np.ndarray,
    long_occ_min_run: int,
) -> Dict[str, Optional[float]]:
    reentry_frame = _reentry_frame_from_longest_run(gt_visibility_nt, query_t)
    long_run = _max_reappearance_occlusion_run(gt_visibility_nt, query_t)
    errors: List[float] = []
    for idx in range(pred_tracks_nt.shape[0]):
        if long_run[idx] < long_occ_min_run:
            continue
        frame_idx = int(reentry_frame[idx])
        if frame_idx < 0:
            continue
        err = float(np.linalg.norm(pred_tracks_nt[idx, frame_idx] - gt_tracks_nt[idx, frame_idx]))
        if np.isfinite(err):
            errors.append(err)
    if not errors:
        return {
            "count": 0,
            "mean_error_px": None,
            "median_error_px": None,
            "<4px": None,
            "<8px": None,
        }
    errors_np = np.asarray(errors, dtype=np.float32)
    return {
        "count": int(errors_np.size),
        "mean_error_px": float(errors_np.mean()),
        "median_error_px": float(np.median(errors_np)),
        "<4px": float((errors_np < 4.0).mean() * 100.0),
        "<8px": float((errors_np < 8.0).mean() * 100.0),
    }


def _bucket_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
    long_runs: np.ndarray,
    buckets: Sequence[Tuple[str, int, Optional[int]]],
) -> Dict[str, Optional[Dict[str, float]]]:
    out: Dict[str, Optional[Dict[str, float]]] = {}
    for name, lo, hi in buckets:
        if hi is None:
            mask = long_runs >= lo
        else:
            mask = (long_runs >= lo) & (long_runs <= hi)
        out[name] = _subset_official_metrics(
            query_points=query_points,
            gt_occluded=gt_occluded,
            gt_tracks=gt_tracks,
            pred_occluded=pred_occluded,
            pred_tracks=pred_tracks,
            mask=mask,
            query_mode=query_mode,
        )
    return out


@dataclass
class VariantState:
    predictor: TrackOnPredictor
    tracking_to_original: List[int]
    q_init_prev_n: int = 0


def _add_queries(
    state: VariantState,
    frame_features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    new_queries: Optional[torch.Tensor],
    new_indices: Sequence[int],
    height: int,
    width: int,
    keep_anchor: bool,
) -> None:
    if new_queries is None or new_queries.shape[0] == 0:
        return
    prev_n = state.predictor.N
    state.predictor.init_queries((frame_features[-1], frame_features[-1].device), new_queries, height, width)
    state.tracking_to_original.extend(int(i) for i in new_indices)
    if keep_anchor:
        start = prev_n
        end = state.predictor.N
        state.predictor.point_memory[start:end, 0] = state.predictor.q_init[start:end]
        state.predictor.temporal_mask[start:end, 0] = False


def _append_memory_rows(
    point_memory: torch.Tensor,
    temporal_mask: torch.Tensor,
    q_new: torch.Tensor,
    active_mask: torch.Tensor,
    visible_mask: torch.Tensor,
    keep_anchor: bool,
    use_oracle_mask: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    point_memory = point_memory.clone()
    temporal_mask = temporal_mask.clone()
    rows = active_mask.nonzero(as_tuple=True)[0]
    if rows.numel() == 0:
        return point_memory, temporal_mask

    start_col = 1 if keep_anchor else 0
    if point_memory.shape[1] - start_col <= 0:
        return point_memory, temporal_mask

    tail_memory = point_memory[rows, start_col:]
    tail_mask = temporal_mask[rows, start_col:]
    tail_memory = torch.roll(tail_memory, shifts=-1, dims=1)
    tail_mask = torch.roll(tail_mask, shifts=-1, dims=1)
    tail_memory[:, -1] = q_new[rows]
    if use_oracle_mask:
        tail_mask[:, -1] = ~visible_mask[rows]
    else:
        tail_mask[:, -1] = False
    point_memory[rows, start_col:] = tail_memory
    temporal_mask[rows, start_col:] = tail_mask
    return point_memory, temporal_mask


def _run_variant(
    variant: str,
    predictor: TrackOnPredictor,
    video: torch.Tensor,
    queries_tx_yx: torch.Tensor,
    visibility_btn: torch.Tensor,
    support_grid_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = video.device
    _, num_frames, _, height, width = video.shape
    queries = queries_tx_yx.squeeze(0)
    query_times = queries[:, 0].long()
    query_coords = queries[:, 1:]
    num_original = int(queries.shape[0])

    if support_grid_size > 0:
        extra = get_points_on_a_grid(support_grid_size, (height, width), device).squeeze(0)
        extra_queries = torch.cat([torch.zeros(extra.shape[0], 1, device=device), extra], dim=1)
        queries_all = torch.cat([queries, extra_queries], dim=0)
        query_times_all = queries_all[:, 0].long()
        query_coords_all = queries_all[:, 1:]
    else:
        queries_all = queries
        query_times_all = query_times
        query_coords_all = query_coords

    predictor.reset()
    predictor.initial_capacity = int(queries_all.shape[0])
    tracking_to_original: List[int] = []
    pred_trajectory = torch.zeros(1, num_frames, num_original, 2, device="cpu", dtype=torch.float32)
    pred_visibility = torch.zeros(1, num_frames, num_original, device="cpu", dtype=torch.bool)

    keep_anchor = "anchor" in variant
    use_oracle_mask = "oracle_mask" in variant

    for frame_idx in range(num_frames):
        frame = video[0, frame_idx].unsqueeze(0)
        frame_features = predictor.model.extract_frame_features(frame)

        new_mask = query_times_all == frame_idx
        new_queries = query_coords_all[new_mask] if new_mask.any() else None
        new_indices = new_mask.nonzero(as_tuple=True)[0].tolist() if new_mask.any() else []
        _add_queries(
            VariantState(predictor=predictor, tracking_to_original=tracking_to_original),
            frame_features,
            new_queries,
            new_indices,
            height,
            width,
            keep_anchor=keep_anchor,
        )

        if predictor.N == 0:
            continue

        p_t, v_logit, q_new = predictor.model.track_frame(
            predictor.q_init[:predictor.N],
            predictor.temporal_mask[:predictor.N],
            predictor.point_memory[:predictor.N],
            frame_features,
            height,
            width,
        )
        v_t = (v_logit.sigmoid() >= predictor.delta_v)

        tracking_to_original_tensor = torch.tensor(tracking_to_original, dtype=torch.long, device="cpu")
        orig_mask = tracking_to_original_tensor < num_original
        if orig_mask.any():
            orig_indices = tracking_to_original_tensor[orig_mask]
            pred_trajectory[0, frame_idx, orig_indices] = p_t[orig_mask].cpu()
            pred_visibility[0, frame_idx, orig_indices] = v_t[orig_mask].cpu()

        active_mask = torch.ones(predictor.N, dtype=torch.bool, device=device)
        visible_mask = torch.ones(predictor.N, dtype=torch.bool, device=device)
        for row_idx, orig_idx in enumerate(tracking_to_original):
            if orig_idx < num_original:
                visible_mask[row_idx] = bool(visibility_btn[0, frame_idx, orig_idx].item())
        predictor.point_memory[:predictor.N], predictor.temporal_mask[:predictor.N] = _append_memory_rows(
            predictor.point_memory[:predictor.N],
            predictor.temporal_mask[:predictor.N],
            q_new,
            active_mask,
            visible_mask,
            keep_anchor=keep_anchor,
            use_oracle_mask=use_oracle_mask,
        )

    return pred_trajectory, pred_visibility


def _build_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Track-On2 Memory Hygiene Diagnostic",
        "",
        f"- dataset: `{summary['dataset_name']}`",
        f"- query_mode: `{summary['query_mode']}`",
        f"- max_videos: `{summary['num_videos']}`",
        f"- long_occ_min_run: `{summary['long_occ_min_run']}`",
        f"- memory_size_eval: `{summary['memory_size_eval']}`",
        f"- support_grid_size: `{summary['support_grid_size']}`",
        "",
        "| Variant | AJ | delta_avg | OA | <4px | long_occ_AJ | long_occ_delta_avg | reentry_mean_px |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in summary["variants"]:
        overall = summary["variants"][variant]["overall_mean"]
        long_occ = summary["variants"][variant]["long_occ_mean"]
        reentry = summary["variants"][variant]["reentry_mean"]
        lines.append(
            "| {variant} | {AJ:.2f} | {delta_avg:.2f} | {OA:.2f} | {lt4:.2f} | {long_AJ:.2f} | {long_delta:.2f} | {reentry_mean:.2f} |".format(
                variant=variant,
                AJ=overall.get("AJ") or float("nan"),
                delta_avg=overall.get("delta_avg") or float("nan"),
                OA=overall.get("OA") or float("nan"),
                lt4=overall.get("<4px") or float("nan"),
                long_AJ=long_occ.get("AJ") or float("nan"),
                long_delta=long_occ.get("delta_avg") or float("nan"),
                reentry_mean=reentry.get("mean_error_px") or float("nan"),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Track-On2 memory hygiene diagnostic on TAP-Vid DAVIS.")
    parser.add_argument("--dataset-name", type=str, default="davis", choices=("davis",))
    parser.add_argument("--dataset-path", type=str, default="/gemini/code/datasets/tapvid_davis/tapvid_davis.pkl")
    parser.add_argument("--config", type=str, default="baselines/track_on/config/test.yaml")
    parser.add_argument("--checkpoint", type=str, default="baselines/track_on/checkpoints_trackon2_dinov3.pt")
    parser.add_argument(
        "--dinov3-local-dir",
        type=str,
        default="/gemini/code/FSPT/third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m",
        help="Local DINOv3 directory for gated-weight-free loading.",
    )
    parser.add_argument("--variants", type=str, default="baseline,oracle_mask,anchor_only,oracle_mask_anchor")
    parser.add_argument("--query-mode", type=str, default="first", choices=("first",))
    parser.add_argument("--max-videos", type=int, default=2)
    parser.add_argument("--long-occ-min-run", type=int, default=20)
    parser.add_argument("--support-grid-size", type=int, default=20)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    variants = tuple(x.strip() for x in args.variants.split(",") if x.strip())
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")

    dinov3_local_dir = Path(args.dinov3_local_dir).expanduser().resolve()
    if not dinov3_local_dir.is_dir():
        raise FileNotFoundError(f"DINOv3 local dir not found: {dinov3_local_dir}")
    os.environ["DINOV3_LOCAL_DIR"] = str(dinov3_local_dir)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args = load_args_from_yaml(str(PROJECT_ROOT / args.config))
    model_args.M_i = 24
    model_args.memory_update_policy = "unconditional"

    dataset = TAPVid(None, data_root=str(Path(args.dataset_path)), dataset_type=args.dataset_name)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

    summary: Dict[str, Any] = {
        "dataset_name": args.dataset_name,
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "config": str((PROJECT_ROOT / args.config).resolve()),
        "checkpoint": str((PROJECT_ROOT / args.checkpoint).resolve()),
        "dinov3_local_dir": str(dinov3_local_dir),
        "query_mode": args.query_mode,
        "num_videos": 0,
        "long_occ_min_run": int(args.long_occ_min_run),
        "support_grid_size": int(args.support_grid_size),
        "memory_size_eval": int(model_args.M_i),
        "variants": {},
    }

    per_video: List[Dict[str, Any]] = []

    for variant in variants:
        summary["variants"][variant] = {
            "overall_rows": [],
            "long_occ_rows": [],
            "reentry_rows": [],
            "bucket_rows": [],
        }

    for video_idx, batch in enumerate(dataloader):
        if args.max_videos > 0 and video_idx >= args.max_videos:
            break
        video, trajectory, visibility, query_points_i = batch
        video = video.cuda(non_blocking=True)
        trajectory = trajectory.cuda(non_blocking=True)
        visibility = visibility.cuda(non_blocking=True)
        query_points_i = query_points_i.cuda(non_blocking=True)

        queries = torch.stack(
            [query_points_i[:, :, 0], query_points_i[:, :, 2], query_points_i[:, :, 1]],
            dim=2,
        ).float()

        gt_tracks = trajectory.permute(0, 2, 1, 3).cpu().numpy()
        gt_visibility = visibility.permute(0, 2, 1).cpu().numpy().astype(np.bool_)
        gt_occluded = (~gt_visibility).astype(np.bool_)
        query_points_np = query_points_i.cpu().numpy()
        query_t = np.round(query_points_np[0, :, 0]).astype(np.int64)
        long_runs = _max_reappearance_occlusion_run(gt_visibility[0], query_t)
        long_mask = long_runs >= int(args.long_occ_min_run)

        video_entry: Dict[str, Any] = {
            "video_index": video_idx,
            "num_queries": int(query_points_np.shape[1]),
            "num_long_occ_queries": int(long_mask.sum()),
            "variants": {},
        }

        for variant in variants:
            predictor = TrackOnPredictor(
                model_args=model_args,
                checkpoint_path=str(PROJECT_ROOT / args.checkpoint),
                support_grid_size=args.support_grid_size,
            ).to(device)
            predictor.model.memory_update_policy = "unconditional"
            predictor.eval()

            pred_trajectory, pred_visibility = _run_variant(
                variant=variant,
                predictor=predictor,
                video=video,
                queries_tx_yx=queries,
                visibility_btn=visibility,
                support_grid_size=args.support_grid_size,
            )
            pred_tracks = pred_trajectory.permute(0, 2, 1, 3).cpu().numpy()
            pred_visibility_np = pred_visibility.permute(0, 2, 1).cpu().numpy().astype(np.bool_)
            pred_occluded = (~pred_visibility_np).astype(np.bool_)

            overall_official = compute_tapvid_metrics(
                query_points_np,
                gt_occluded,
                gt_tracks,
                pred_occluded,
                pred_tracks,
                args.query_mode,
            )
            overall = _compute_summary_from_official(overall_official)
            long_occ = _subset_official_metrics(
                query_points=query_points_np,
                gt_occluded=gt_occluded,
                gt_tracks=gt_tracks,
                pred_occluded=pred_occluded,
                pred_tracks=pred_tracks,
                mask=long_mask,
                query_mode=args.query_mode,
            )
            reentry = _reentry_metrics(
                pred_tracks_nt=pred_tracks[0],
                gt_tracks_nt=gt_tracks[0],
                gt_visibility_nt=gt_visibility[0],
                query_t=query_t,
                long_occ_min_run=int(args.long_occ_min_run),
            )
            buckets = _bucket_metrics(
                query_points=query_points_np,
                gt_occluded=gt_occluded,
                gt_tracks=gt_tracks,
                pred_occluded=pred_occluded,
                pred_tracks=pred_tracks,
                query_mode=args.query_mode,
                long_runs=long_runs,
                buckets=(("20-39", 20, 39), ("40-71", 40, 71), ("72+", 72, None)),
            )

            summary["variants"][variant]["overall_rows"].append(overall)
            if long_occ is not None:
                summary["variants"][variant]["long_occ_rows"].append(long_occ)
            summary["variants"][variant]["reentry_rows"].append(reentry)
            summary["variants"][variant]["bucket_rows"].append(buckets)

            video_entry["variants"][variant] = {
                "overall": overall,
                "long_occ": long_occ,
                "reentry": reentry,
                "buckets": buckets,
            }

            del predictor
            torch.cuda.empty_cache()

        per_video.append(video_entry)
        summary["num_videos"] = len(per_video)

    for variant in variants:
        overall_rows = summary["variants"][variant].pop("overall_rows")
        long_occ_rows = summary["variants"][variant].pop("long_occ_rows")
        reentry_rows = summary["variants"][variant].pop("reentry_rows")
        bucket_rows = summary["variants"][variant].pop("bucket_rows")

        summary["variants"][variant]["overall_mean"] = _mean_dict(
            overall_rows, ("AJ", "OA", "delta_avg", "<4px")
        )
        summary["variants"][variant]["long_occ_mean"] = _mean_dict(
            long_occ_rows, ("AJ", "OA", "delta_avg", "<4px")
        )
        summary["variants"][variant]["reentry_mean"] = _mean_dict(
            reentry_rows, ("count", "mean_error_px", "median_error_px", "<4px", "<8px")
        )

        bucket_summary: Dict[str, Dict[str, Optional[float]]] = {}
        for bucket_name in ("20-39", "40-71", "72+"):
            bucket_metrics = [row[bucket_name] for row in bucket_rows if row.get(bucket_name) is not None]
            bucket_summary[bucket_name] = _mean_dict(bucket_metrics, ("AJ", "OA", "delta_avg", "<4px"))
        summary["variants"][variant]["bucket_mean"] = bucket_summary

    manifest = {
        "command": " ".join(sys.argv),
        "dataset_name": args.dataset_name,
        "dataset_path": summary["dataset_path"],
        "config": summary["config"],
        "checkpoint": summary["checkpoint"],
        "dinov3_local_dir": summary["dinov3_local_dir"],
        "variants": list(variants),
        "query_mode": args.query_mode,
        "max_videos": int(args.max_videos),
        "long_occ_min_run": int(args.long_occ_min_run),
        "support_grid_size": int(args.support_grid_size),
        "memory_size_eval": int(model_args.M_i),
        "device": str(device),
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    with open(out_dir / "per_video_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_video, f, ensure_ascii=True, indent=2)
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(_build_markdown(summary))

    print(json.dumps(summary["variants"], ensure_ascii=True, indent=2))
    print(f"[ok] wrote outputs to {out_dir}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
