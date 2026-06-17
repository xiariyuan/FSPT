#!/usr/bin/env python3
"""
Evaluate Track-On2 directly on PRT (Persistent Re-entry Tracking) queries.

This script mirrors scripts/eval_cotracker3_prt.py, but uses the local
Track-On predictor under baselines/track_on as an external online-memory
baseline.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "baselines" / "track_on"))

from scripts.eval_world_state_stage0 import (
    discover_sequences,
    find_reentry_queries,
    load_sequence,
    project_3d_to_2d,
)
from scripts.build_prt_splits import load_image_size


def load_trackon(
    device: torch.device,
    checkpoint_path: str,
    support_grid_size: int = 0,
    config_path: Optional[str] = None,
):
    from model.trackon_predictor import Predictor
    from utils.train_utils import load_args_from_yaml

    model_args = None
    if config_path is not None:
        model_args = load_args_from_yaml(config_path)
        # Inference-only path: disable gradient checkpointing to avoid
        # compatibility issues in forward() with current dependencies.
        model_args.grad_checkpoint = False

    model = Predictor(
        model_args=model_args,
        checkpoint_path=checkpoint_path,
        support_grid_size=support_grid_size,
    )
    model = model.to(device).eval()
    return model


def load_video_frames(
    seq_path: Path,
    target_hw: Tuple[int, int],
    max_frames: int = 0,
) -> Optional[torch.Tensor]:
    """Load RGB frames as (T, 3, H, W) float tensor in [0, 255]."""
    import cv2

    rgb_dir = seq_path / "rgbs"
    if not rgb_dir.is_dir():
        return None

    frame_files = sorted(rgb_dir.glob("rgb_*.jpg"))
    if not frame_files:
        return None
    if max_frames > 0:
        frame_files = frame_files[:max_frames]

    target_h, target_w = target_hw
    frames = []
    for f in frame_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        frames.append(torch.from_numpy(img).permute(2, 0, 1))

    if not frames:
        return None
    return torch.stack(frames, dim=0).float()


def discover_checkpoint(checkpoint_arg: str) -> str:
    candidates = []
    if checkpoint_arg:
        candidates.append(Path(checkpoint_arg))

    candidates.extend(
        [
            PROJECT_ROOT / "baselines" / "track_on" / "checkpoints" / "trackon2_dinov3_checkpoint.pt",
            PROJECT_ROOT / "baselines" / "track_on" / "checkpoints" / "track_on2.pt",
            PROJECT_ROOT / "baselines" / "track_on" / "checkpoints" / "offline.pth",
        ]
    )

    for path in candidates:
        if path.exists():
            return str(path.resolve())

    raise FileNotFoundError(
        "Track-On checkpoint not found. Pass --checkpoint or place a checkpoint under "
        "baselines/track_on/checkpoints/."
    )


def main():
    parser = argparse.ArgumentParser(description="Track-On2 on PRT")
    parser.add_argument("--data-root", type=str, default="/gemini/code/FSPT/datasets/pointodyssey")
    parser.add_argument("--splits", type=str, default="val")
    parser.add_argument("--min-occ-length", type=int, default=20)
    parser.add_argument("--min-camera-motion", type=float, default=0.30)
    parser.add_argument("--max-sequences", type=int, default=3)
    parser.add_argument("--max-queries-per-seq", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional Track-On yaml config. If empty, Predictor default args are used.",
    )
    parser.add_argument(
        "--support-grid-size",
        type=int,
        default=0,
        help="Extra support grid queries used by Track-On Predictor.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="/gemini/code/FSPT/outputs/trackon_prt_eval.json",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = discover_checkpoint(args.checkpoint)
    config_path = args.config or None

    print("Loading Track-On...", flush=True)
    model = load_trackon(
        device=device,
        checkpoint_path=checkpoint_path,
        support_grid_size=args.support_grid_size,
        config_path=config_path,
    )
    model_h = int(model.model.input_size[0])
    model_w = int(model.model.input_size[1])
    print(f"Track-On loaded. input_size=({model_h}, {model_w})", flush=True)

    data_root = Path(args.data_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    sequences = discover_sequences(data_root, splits)
    if args.max_sequences > 0:
        sequences = sequences[:args.max_sequences]
    print(f"Evaluating {len(sequences)} sequences.", flush=True)

    all_results: List[Dict] = []

    for si, seq_path in enumerate(sequences):
        print(f"\n[{si+1}/{len(sequences)}] {seq_path.name}", flush=True)
        seq = load_sequence(seq_path)
        trajs_2d = seq["trajs_2d"]
        trajs_3d = seq["trajs_3d"]
        visibs = seq["visibs"]
        intrinsics = seq["intrinsics"]
        extrinsics = seq["extrinsics"]

        queries = find_reentry_queries(seq, min_occ_length=args.min_occ_length)
        extr_inv = np.linalg.inv(extrinsics)
        filtered = []
        for q in queries:
            e_rel = extrinsics[q.reentry_frame] @ extr_inv[q.query_frame]
            cam_motion = float(np.linalg.norm(e_rel[:3, :3] - np.eye(3)))
            if cam_motion >= args.min_camera_motion:
                filtered.append((q, cam_motion))

        if not filtered:
            print("  No PRT queries passed filter.", flush=True)
            continue

        rng = np.random.default_rng(42)
        if len(filtered) > args.max_queries_per_seq:
            indices = rng.choice(len(filtered), args.max_queries_per_seq, replace=False)
            filtered = [filtered[i] for i in sorted(indices)]
        print(f"  {len(filtered)} PRT queries.", flush=True)

        video = load_video_frames(seq_path, target_hw=(model_h, model_w))
        if video is None:
            print("  Could not load video frames.", flush=True)
            continue
        video = video.unsqueeze(0).to(device)

        img_h, img_w = load_image_size(seq_path)
        scale_y = model_h / float(img_h)
        scale_x = model_w / float(img_w)

        for qi, (q, cam_motion) in enumerate(filtered):
            if qi % 10 == 0:
                print(f"  query {qi}/{len(filtered)}...", flush=True)

            t_q = q.query_frame
            t_re = q.reentry_frame
            i = q.point_idx

            query_y_orig = float(trajs_2d[t_q, i, 0])
            query_x_orig = float(trajs_2d[t_q, i, 1])
            query_y = query_y_orig * scale_y
            query_x = query_x_orig * scale_x

            if not np.isfinite(query_y) or not np.isfinite(query_x):
                continue

            queries_tensor = torch.tensor(
                [[[float(t_q), query_x, query_y]]],
                device=device,
                dtype=torch.float32,
            )

            with torch.no_grad():
                try:
                    pred_tracks, pred_visibility = model(video, queries_tensor)
                except Exception as e:
                    print(f"  Track-On inference failed: {e}", flush=True)
                    continue

            track_xy = pred_tracks[0, :, 0, :].cpu().numpy()
            vis_pred = pred_visibility[0, :, 0].cpu().numpy()

            gt_y_orig = float(trajs_2d[t_re, i, 0])
            gt_x_orig = float(trajs_2d[t_re, i, 1])
            gt_y = gt_y_orig * scale_y
            gt_x = gt_x_orig * scale_x

            pred_x_re = float(track_xy[t_re, 0])
            pred_y_re = float(track_xy[t_re, 1])
            trackon_err = float(np.sqrt((pred_y_re - gt_y) ** 2 + (pred_x_re - gt_x) ** 2))

            hold_err = float(np.sqrt((query_y - gt_y) ** 2 + (query_x - gt_x) ** 2))

            hold_3d = trajs_3d[t_q, i].astype(np.float32)
            pt_cam = extrinsics[t_q][:3, :3] @ hold_3d + extrinsics[t_q][:3, 3]
            z_depth = float(pt_cam[2])
            if np.isfinite(z_depth) and z_depth > 1e-6:
                k_inv = np.linalg.inv(intrinsics[t_q])
                pixels_h = np.array([trajs_2d[t_q, i, 0], trajs_2d[t_q, i, 1], 1.0], dtype=np.float32)
                pt_cam_gt = (k_inv @ pixels_h) * z_depth
                e_inv = np.linalg.inv(extrinsics[t_q])
                world_3d = (e_inv[:3, :3] @ pt_cam_gt) + e_inv[:3, 3]
                reproj_3d = project_3d_to_2d(world_3d, intrinsics[t_re], extrinsics[t_re])
                reproj_3d_scaled = np.array([reproj_3d[0] * scale_y, reproj_3d[1] * scale_x])
                gt_re_scaled = np.array([gt_y, gt_x])
                hold_3d_err = float(np.linalg.norm(reproj_3d_scaled - gt_re_scaled))
            else:
                hold_3d_err = float("nan")

            all_results.append(
                {
                    "seq": seq_path.name,
                    "query_frame": int(t_q),
                    "reentry_frame": int(t_re),
                    "point_idx": int(i),
                    "occ_length": int(q.occ_length),
                    "camera_motion": cam_motion,
                    "trackon_err_px": trackon_err,
                    "hold_2d_err_px": hold_err,
                    "hold_3d_err_px": hold_3d_err,
                    "trackon_visible_at_reentry": bool(vis_pred[t_re]),
                }
            )

    print(f"\nTotal PRT queries evaluated: {len(all_results)}", flush=True)
    if not all_results:
        print("No results.", flush=True)
        return

    trackon_err = np.array([r["trackon_err_px"] for r in all_results])
    hold_err = np.array([r["hold_2d_err_px"] for r in all_results])
    hold3d_err = np.array([r["hold_3d_err_px"] for r in all_results if np.isfinite(r["hold_3d_err_px"])])

    summary = {
        "n": len(all_results),
        "trackon_median_px": float(np.median(trackon_err)),
        "hold_2d_median_px": float(np.median(hold_err)),
        "hold_3d_median_px": float(np.median(hold3d_err)) if len(hold3d_err) > 0 else None,
        "trackon_lt4px": float(np.mean(trackon_err < 4)),
        "hold_2d_lt4px": float(np.mean(hold_err < 4)),
        "hold_3d_lt4px": float(np.mean(hold3d_err < 4)) if len(hold3d_err) > 0 else None,
        "trackon_visible_frac": float(np.mean([r["trackon_visible_at_reentry"] for r in all_results])),
        "checkpoint": checkpoint_path,
        "input_size": [model_h, model_w],
        "support_grid_size": int(args.support_grid_size),
    }

    print(f"\n{'=' * 60}")
    print("Track-On on PRT Results")
    print(f"{'=' * 60}")
    print(f"{'Method':<25} {'Median px':>10} {'<4px':>8}")
    print(f"{'-' * 43}")
    print(f"{'2D hold':<25} {summary['hold_2d_median_px']:>10.2f} {summary['hold_2d_lt4px']:>7.1%}")
    if summary["hold_3d_median_px"] is not None:
        print(f"{'3D hold (GT depth)':<25} {summary['hold_3d_median_px']:>10.2f} {summary['hold_3d_lt4px']:>7.1%}")
    print(f"{'Track-On':<25} {summary['trackon_median_px']:>10.2f} {summary['trackon_lt4px']:>7.1%}")
    print(f"{'=' * 60}")
    print(f"Track-On visible at reentry: {summary['trackon_visible_frac']:.1%}")

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "config": vars(args),
                "summary": summary,
                "results": all_results[:200],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
