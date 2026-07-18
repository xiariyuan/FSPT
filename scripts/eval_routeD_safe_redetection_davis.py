#!/usr/bin/env python3
"""Official TAP-Vid-DAVIS evaluation for Route-D safe re-detection."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_cotracker3_safe_redetection_runtime import (
    compute_video_fmaps,
    run_one_query_safe_redetection,
)
from projects.mmp_tracker.mmp_tracker.routeD_redetection_metrics import compute_official_aj_rd
from projects.mmp_tracker.mmp_tracker.routeD_safe_redetection import SafeRedetectionConfig, SafeRedetectionModel
from projects.mmp_tracker.mmp_tracker.routeD_safe_redetection_cache import file_sha256


def deterministic(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)


def mean_metric(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-source-root", required=True)
    ap.add_argument("--checkpoint", default="/gemini/code/FSPT/weights/scaled_online.pth")
    ap.add_argument("--davis", default="/gemini/code/FSPT/datasets/tapvid_davis/tapvid_davis.pkl")
    ap.add_argument("--model-checkpoint", default="")
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/routeD_safe_redetection_v0.yaml"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--max-queries-per-video", type=int, default=0)
    args = ap.parse_args()
    deterministic(0)

    source = str(Path(args.official_source_root).resolve())
    sys.path = [row for row in sys.path if "baselines/cotracker" not in row]
    sys.path.insert(0, source)
    from cotracker.datasets.tap_vid_datasets import TapVidDataset
    from cotracker.evaluation.core.eval_utils import compute_tapvid_metrics
    from cotracker.predictor import CoTrackerOnlinePredictor

    config = yaml.safe_load(Path(args.config).read_text())
    model_cfg = dict(config["model"]); model_cfg.update(input_height=256, input_width=256)
    model = SafeRedetectionModel(SafeRedetectionConfig(**model_cfg)).to(args.device).eval()
    model_state_sha = None
    model_checkpoint_sha = None
    if args.model_checkpoint:
        bundle = torch.load(args.model_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(bundle["model_state"], strict=True)
        model_state_sha = bundle["model_state_sha256"]
        model_checkpoint_sha = file_sha256(args.model_checkpoint)

    dataset = TapVidDataset(
        data_root=str(Path(args.davis).resolve()),
        dataset_type="davis",
        resize_to=[256, 256],
        queried_first=True,
    )
    video_count = min(len(dataset), args.max_videos) if args.max_videos > 0 else len(dataset)
    rows = []
    started = time.time()
    for video_index in range(video_count):
        sample = dataset[video_index]
        video = sample.video.unsqueeze(0).to(args.device)
        query_tyx = sample.query_points.unsqueeze(0).float().to(args.device)
        if args.max_queries_per_video > 0:
            query_tyx = query_tyx[:, : args.max_queries_per_video]
        query_txy = torch.stack((query_tyx[..., 0], query_tyx[..., 2], query_tyx[..., 1]), dim=-1)
        predictor = CoTrackerOnlinePredictor(
            checkpoint=args.checkpoint, offline=False, v2=False, window_len=16
        ).to(args.device).eval()
        fmaps = compute_video_fmaps(predictor, video)
        track_rows = []
        visibility_rows = []
        intervention_rows = []
        recovery_rows = []
        write_rows = []
        diagnostics = []
        for query_index in range(query_txy.shape[1]):
            result = run_one_query_safe_redetection(
                predictor,
                model,
                video,
                query_txy[:, query_index : query_index + 1],
                precomputed_fmaps=fmaps,
            )
            track_rows.append(result.tracks_xy)
            visibility_rows.append(result.visibility)
            intervention_rows.append(result.selected_non_native)
            recovery_rows.append(result.confirmed_recovery)
            write_rows.append(result.writeback_applied)
            diagnostics.append(result.diagnostics)
        pred_tracks_btn = torch.cat(track_rows, dim=2)
        pred_visible_btn = torch.cat(visibility_rows, dim=2)
        gt_tracks_btn = sample.trajectory[:, : pred_tracks_btn.shape[2]].unsqueeze(0).to(args.device)
        gt_visible_btn = sample.visibility[:, : pred_tracks_btn.shape[2]].unsqueeze(0).to(args.device)
        metric = compute_tapvid_metrics(
            query_tyx.cpu().numpy(),
            (~gt_visible_btn.permute(0, 2, 1)).cpu().numpy(),
            gt_tracks_btn.permute(0, 2, 1, 3).cpu().numpy(),
            (~pred_visible_btn.permute(0, 2, 1)).cpu().numpy(),
            pred_tracks_btn.permute(0, 2, 1, 3).cpu().numpy(),
            query_mode="first",
        )
        metric = {key: float(np.mean(value)) for key, value in metric.items()}
        ajrd = compute_official_aj_rd(
            pred_tracks_btn,
            pred_visible_btn,
            gt_tracks_btn,
            gt_visible_btn,
        )
        interventions = torch.cat(intervention_rows, dim=2)
        recoveries = torch.cat(recovery_rows, dim=2)
        writes = torch.cat(write_rows, dim=2)
        row = {
            "video_index": video_index,
            "video_name": sample.seq_name,
            "queries": int(query_txy.shape[1]),
            "metrics": metric,
            "AJ_RD": ajrd,
            "interventions": int(interventions.sum().item()),
            "confirmed_recoveries": int(recoveries.sum().item()),
            "writes": int(writes.sum().item()),
            "diagnostics": diagnostics,
        }
        rows.append(row)
        print(json.dumps({
            "video": video_index,
            "name": sample.seq_name,
            "queries": row["queries"],
            "AJ": metric["average_jaccard"],
            "OA": metric["occlusion_accuracy"],
            "delta": metric["average_pts_within_thresh"],
            "AJ_RD": ajrd["AJ_RD"],
            "interventions": row["interventions"],
            "writes": row["writes"],
        }), flush=True)
        del predictor, fmaps, video
        torch.cuda.empty_cache()

    avg_metrics = {
        key: mean_metric([row["metrics"] for row in rows], key)
        for key in rows[0]["metrics"]
    }
    ajrd_keys = [key for key, value in rows[0]["AJ_RD"].items() if isinstance(value, (int, float)) and key.startswith("AJ_RD")]
    avg_ajrd = {}
    for key in ajrd_keys:
        values = [float(row["AJ_RD"][key]) for row in rows]
        values = [value for value in values if not np.isnan(value)]
        avg_ajrd[key] = float(np.mean(values)) if values else float("nan")
    official_reference = config["official_alignment_gate"]
    output_payload = {
        "schema_version": "routeD_safe_redetection_davis_eval_v0",
        "date": "2026-07-18",
        "scope": "Official TAP-Vid-DAVIS first-query, one evaluation query per forward stream",
        "official_source_commit": config["official_alignment_gate"]["official_source_commit"],
        "official_checkpoint_sha256": file_sha256(args.checkpoint),
        "dataset_sha256": file_sha256(args.davis),
        "model_checkpoint": str(Path(args.model_checkpoint).resolve()) if args.model_checkpoint else None,
        "model_checkpoint_sha256": model_checkpoint_sha,
        "model_state_sha256": model_state_sha,
        "videos": len(rows),
        "max_queries_per_video": args.max_queries_per_video,
        "average_metrics": avg_metrics,
        "average_AJ_RD": avg_ajrd,
        "official_full_call_reference": {
            "AJ": official_reference["frozen_baseline_AJ"],
            "OA": official_reference["frozen_baseline_OA"],
            "delta_average": official_reference["frozen_baseline_delta_average"],
            "streaming_minus_full_call": {
                "AJ": avg_metrics["average_jaccard"] - official_reference["frozen_baseline_AJ"],
                "OA": avg_metrics["occlusion_accuracy"] - official_reference["frozen_baseline_OA"],
                "delta_average": avg_metrics["average_pts_within_thresh"] - official_reference["frozen_baseline_delta_average"],
            },
        },
        "behavior": {
            "interventions": sum(row["interventions"] for row in rows),
            "confirmed_recoveries": sum(row["confirmed_recoveries"] for row in rows),
            "writes": sum(row["writes"] for row in rows),
        },
        "runtime_seconds": time.time() - started,
        "per_video": rows,
        "locked_data_read": {
            "pointodyssey_internal_holdout": False,
            "pointodyssey_test": False,
            "kinetics_1144": False,
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2, allow_nan=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "average_metrics": avg_metrics,
        "average_AJ_RD": avg_ajrd,
        "behavior": output_payload["behavior"],
        "runtime_seconds": output_payload["runtime_seconds"],
    }, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
