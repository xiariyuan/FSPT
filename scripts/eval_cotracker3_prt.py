#!/usr/bin/env python3
"""
Evaluate CoTracker3 directly on PRT (Persistent Re-entry Tracking) queries.

This script answers: "How well does a state-of-the-art tracker handle
long-occlusion re-entry without any special relocalization module?"

It loads PointOdyssey sequences, extracts PRT queries, runs CoTracker3 inference,
and computes per-query error at the reentry frame.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "baselines" / "cotracker"))

from scripts.eval_world_state_stage0 import (
    discover_sequences,
    find_reentry_queries,
    load_sequence,
    project_3d_to_2d,
)
from scripts.build_prt_splits import load_image_size


def load_cotracker(device: torch.device, checkpoint_path: str, online: bool = False):
    from cotracker.predictor import CoTrackerOnlinePredictor, CoTrackerPredictor
    if online:
        model = CoTrackerOnlinePredictor(checkpoint=checkpoint_path)
    else:
        model = CoTrackerPredictor(checkpoint=checkpoint_path)
    model = model.to(device).eval()
    return model


def load_video_frames(seq_path: Path, max_frames: int = 0, resize: int = 256) -> torch.Tensor:
    """Load RGB frames as a (T, 3, H, W) tensor in [0, 255], resized to resize×resize."""
    import cv2
    rgb_dir = seq_path / "rgbs"
    if not rgb_dir.is_dir():
        return None
    frame_files = sorted(rgb_dir.glob("rgb_*.jpg"))
    if not frame_files:
        return None
    if max_frames > 0:
        frame_files = frame_files[:max_frames]

    frames = []
    for f in frame_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize > 0:
            img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_LINEAR)
        frames.append(torch.from_numpy(img).permute(2, 0, 1))  # (3, H, W)

    if not frames:
        return None
    return torch.stack(frames, dim=0).float()  # (T, 3, H, W)


def main():
    parser = argparse.ArgumentParser(description="CoTracker3 on PRT")
    parser.add_argument("--data-root", type=str, default="/gemini/code/FSPT/datasets/pointodyssey")
    parser.add_argument("--splits", type=str, default="val")
    parser.add_argument("--min-occ-length", type=int, default=20)
    parser.add_argument("--min-camera-motion", type=float, default=0.30)
    parser.add_argument("--max-sequences", type=int, default=3)
    parser.add_argument("--max-queries-per-seq", type=int, default=50)
    parser.add_argument("--checkpoint", type=str,
                        default="/gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth")
    parser.add_argument("--online", action="store_true",
                        help="Use CoTracker3 online predictor API instead of offline predictor.")
    parser.add_argument("--output-json", type=str,
                        default="/gemini/code/FSPT/outputs/cotracker3_prt_eval.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CoTracker3
    mode_name = "online" if args.online else "offline"
    print(f"Loading CoTracker3 ({mode_name})...", flush=True)
    model = load_cotracker(device, args.checkpoint, online=args.online)
    print(f"CoTracker3 ({mode_name}) loaded.", flush=True)

    # Discover sequences
    data_root = Path(args.data_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    sequences = discover_sequences(data_root, splits)
    if args.max_sequences > 0:
        sequences = sequences[:args.max_sequences]
    print(f"Evaluating {len(sequences)} sequences.", flush=True)

    all_results = []

    for si, seq_path in enumerate(sequences):
        print(f"\n[{si+1}/{len(sequences)}] {seq_path.name}", flush=True)
        seq = load_sequence(seq_path)
        trajs_2d = seq["trajs_2d"]
        trajs_3d = seq["trajs_3d"]
        visibs = seq["visibs"]
        intrinsics = seq["intrinsics"]
        extrinsics = seq["extrinsics"]
        T, N = visibs.shape

        # Find PRT queries (same criteria as PRT benchmark)
        queries = find_reentry_queries(seq, min_occ_length=args.min_occ_length)
        # Filter by camera motion
        extr_inv = np.linalg.inv(extrinsics)
        filtered = []
        for q in queries:
            e_rel = extrinsics[q.reentry_frame] @ extr_inv[q.query_frame]
            cam_motion = float(np.linalg.norm(e_rel[:3, :3] - np.eye(3)))
            if cam_motion >= args.min_camera_motion:
                filtered.append((q, cam_motion))

        if not filtered:
            print(f"  No PRT queries passed filter.", flush=True)
            continue

        # Sample queries
        rng = np.random.default_rng(42)
        if len(filtered) > args.max_queries_per_seq:
            indices = rng.choice(len(filtered), args.max_queries_per_seq, replace=False)
            filtered = [filtered[i] for i in sorted(indices)]
        print(f"  {len(filtered)} PRT queries.", flush=True)

        # Load video (resized to 256×256)
        resize = 256
        video = load_video_frames(seq_path, resize=resize)
        if video is None:
            print(f"  Could not load video frames.", flush=True)
            continue
        video = video.unsqueeze(0).to(device)  # (1, T, 3, H, W)

        # Get original image size for coordinate scaling
        img_size = load_image_size(seq_path)  # (height, width) of original
        scale_y = resize / float(img_size[0])
        scale_x = resize / float(img_size[1])

        # For each query, run CoTracker3 with the query point
        for qi, (q, cam_motion) in enumerate(filtered):
            if qi % 10 == 0:
                print(f"  query {qi}/{len(filtered)}...", flush=True)

            t_q = q.query_frame
            t_re = q.reentry_frame
            i = q.point_idx

            # Query point: scale from original to 256×256
            query_y_orig, query_x_orig = float(trajs_2d[t_q, i, 0]), float(trajs_2d[t_q, i, 1])
            query_y = query_y_orig * scale_y
            query_x = query_x_orig * scale_x

            if not np.isfinite(query_y) or not np.isfinite(query_x):
                continue

            # CoTracker3 expects queries as (1, N, 3) with (t, x, y)
            queries_tensor = torch.tensor([[[float(t_q), query_x, query_y]]], device=device, dtype=torch.float32)

            # Run inference
            with torch.no_grad():
                try:
                    if args.online:
                        # Initialize with the first chunk that contains the query frame.
                        model(video_chunk=video[:, : model.step * 2], is_first_step=True, queries=queries_tensor)
                        preds_track = []
                        preds_vis = []
                        for ind in range(0, video.shape[1] - model.step, model.step):
                            chunk = video[:, ind : ind + model.step * 2]
                            pred_tracks, pred_visibility = model(video_chunk=chunk)
                            preds_track.append(pred_tracks)
                            preds_vis.append(pred_visibility)
                        pred_tracks = preds_track[-1]
                        pred_visibility = preds_vis[-1]
                    else:
                        pred_tracks, pred_visibility = model(
                            video,
                            queries=queries_tensor,
                            grid_size=1,  # not used when queries are provided
                        )
                except Exception as e:
                    print(f"  CoTracker3 inference failed: {e}", flush=True)
                    continue

            # pred_tracks: (1, T, 1, 2) in (x, y) pixel coords
            # pred_visibility: (1, T, 1)
            track_xy = pred_tracks[0, :, 0, :].cpu().numpy()  # (T, 2) in (x, y)
            vis_pred = pred_visibility[0, :, 0].cpu().numpy()  # (T,)

            # GT at reentry (scaled to 256×256)
            gt_y_orig, gt_x_orig = float(trajs_2d[t_re, i, 0]), float(trajs_2d[t_re, i, 1])
            gt_y = gt_y_orig * scale_y
            gt_x = gt_x_orig * scale_x

            # CoTracker3 prediction at reentry (already in 256×256 space)
            pred_x_re = float(track_xy[t_re, 0])
            pred_y_re = float(track_xy[t_re, 1])

            # Error in 256×256 space
            cotracker_err = float(np.sqrt((pred_y_re - gt_y)**2 + (pred_x_re - gt_x)**2))

            # 2D hold baseline (same space)
            hold_err = float(np.sqrt((query_y - gt_y)**2 + (query_x - gt_x)**2))

            # 3D hold (in original resolution, then scale to 256×256)
            hold_3d = trajs_3d[t_q, i].astype(np.float32)
            pt_cam = extrinsics[t_q][:3, :3] @ hold_3d + extrinsics[t_q][:3, 3]
            z_depth = float(pt_cam[2])
            if np.isfinite(z_depth) and z_depth > 1e-6:
                k_inv = np.linalg.inv(intrinsics[t_q])
                pixels_h = np.array([trajs_2d[t_q, i, 0], trajs_2d[t_q, i, 1], 1.0], dtype=np.float32)
                pt_cam_noisy = (k_inv @ pixels_h) * z_depth  # GT depth (no noise for reference)
                e_inv = np.linalg.inv(extrinsics[t_q])
                world_3d = (e_inv[:3, :3] @ pt_cam_noisy) + e_inv[:3, 3]
                reproj_3d = project_3d_to_2d(world_3d, intrinsics[t_re], extrinsics[t_re])
                # Scale reproj from original to 256×256
                reproj_3d_scaled = np.array([reproj_3d[0] * scale_y, reproj_3d[1] * scale_x])
                gt_re_scaled = np.array([gt_y, gt_x])
                hold_3d_err = float(np.linalg.norm(reproj_3d_scaled - gt_re_scaled))
            else:
                hold_3d_err = float('nan')

            all_results.append({
                "seq": seq_path.name,
                "query_frame": int(t_q),
                "reentry_frame": int(t_re),
                "point_idx": int(i),
                "occ_length": int(q.occ_length),
                "camera_motion": cam_motion,
                "cotracker_err_px": cotracker_err,
                "hold_2d_err_px": hold_err,
                "hold_3d_err_px": hold_3d_err,
                "cotracker_visible_at_reentry": bool(vis_pred[t_re] > 0.5),
            })

    # Summary
    print(f"\nTotal PRT queries evaluated: {len(all_results)}", flush=True)
    if not all_results:
        print("No results.", flush=True)
        return

    ct_err = np.array([r["cotracker_err_px"] for r in all_results])
    hold_err = np.array([r["hold_2d_err_px"] for r in all_results])
    hold3d_err = np.array([r["hold_3d_err_px"] for r in all_results if np.isfinite(r["hold_3d_err_px"])])

    summary = {
        "n": len(all_results),
        "cotracker_median_px": float(np.median(ct_err)),
        "hold_2d_median_px": float(np.median(hold_err)),
        "hold_3d_median_px": float(np.median(hold3d_err)) if len(hold3d_err) > 0 else None,
        "cotracker_lt4px": float(np.mean(ct_err < 4)),
        "hold_2d_lt4px": float(np.mean(hold_err < 4)),
        "hold_3d_lt4px": float(np.mean(hold3d_err < 4)) if len(hold3d_err) > 0 else None,
        "cotracker_visible_frac": float(np.mean([r["cotracker_visible_at_reentry"] for r in all_results])),
        "mode": mode_name,
    }

    print(f"\n{'='*60}")
    print("CoTracker3 on PRT Results")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'Median px':>10} {'<4px':>8}")
    print(f"{'-'*43}")
    print(f"{'2D hold':<25} {summary['hold_2d_median_px']:>10.2f} {summary['hold_2d_lt4px']:>7.1%}")
    if summary['hold_3d_median_px'] is not None:
        print(f"{'3D hold (GT depth)':<25} {summary['hold_3d_median_px']:>10.2f} {summary['hold_3d_lt4px']:>7.1%}")
    print(f"{'CoTracker3':<25} {summary['cotracker_median_px']:>10.2f} {summary['cotracker_lt4px']:>7.1%}")
    print(f"{'='*60}")
    print(f"CoTracker3 visible at reentry: {summary['cotracker_visible_frac']:.1%}")

    # Save
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": vars(args),
        "summary": summary,
        "results": all_results[:200],
    }, indent=2) + "\n")
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
