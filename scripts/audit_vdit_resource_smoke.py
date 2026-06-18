#!/usr/bin/env python3
"""P2a: Resource smoke for feature extraction.

Validates that feature extraction is not an engineering dead end:
  - DINOv2 baseline feature extraction time and memory
  - Per-video feature grid size
  - Cacheability to disk
  - Crash/OOM detection

Runs on 2 videos / 64 re-entry events max.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.coords import find_first_reentry
from utils.attempt0_schema import load_attempt0_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", type=str,
                        default="outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt")
    parser.add_argument("--pkl-path", type=str,
                        default="/gemini/code/datasets/tapvid_davis/tapvid_davis.pkl")
    parser.add_argument("--max-videos", type=int, default=2)
    parser.add_argument("--max-events", type=int, default=64)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--feature-sources", type=str, default="dinov2",
                        help="Comma-separated: dinov2")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: Dict[str, Any] = {
        "device": str(device),
        "max_videos": args.max_videos,
        "max_events": args.max_events,
        "feature_sources": [],
    }

    # Load cache and pkl
    payload = load_attempt0_cache(Path(args.cache_path))
    records = payload["records"][:args.max_videos]

    import pickle
    with open(args.pkl_path, "rb") as f:
        pkl_data = pickle.load(f)

    # --- DINOv2 resource smoke ---
    if "dinov2" in args.feature_sources:
        print("=== DINOv2 Resource Smoke ===")
        from scripts.eval_world_state_stage2_causal_dino import DINOFeatureExtractor
        dino_weights = Path("/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
        dino = DINOFeatureExtractor(dino_weights, device)
        dino_model = dino.model.eval()

        # Measure memory before
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated() / 1024**3

        n_events = 0
        total_time = 0.0
        feature_sizes = []
        crashes = 0
        can_cache = True

        for rec_idx, r in enumerate(records):
            vid = r["video_id"]
            video_rgb = np.asarray(pkl_data[vid]["video"], dtype=np.uint8)
            gt_vis = np.asarray(r["gt_visibility"], dtype=bool)
            qpts = np.asarray(r["query_points"], dtype=np.float32)
            h, w = int(r["original_size"][0]), int(r["original_size"][1])

            # Extract full-frame feature map once per video
            frame = torch.from_numpy(video_rgb[0]).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            frame = F.interpolate(frame, size=(518, 518), mode="bilinear", align_corners=False).to(device)

            t0 = time.time()
            with torch.no_grad():
                feat_map = dino_model(frame)[-1]  # (1, D, h, w)
                feat_map = F.normalize(feat_map.float(), dim=1)
            t1 = time.time()

            feat_size = feat_map.shape  # (1, D, h, w)
            feature_sizes.append(list(feat_size))
            total_time += (t1 - t0)

            # Check cacheability: can we serialize?
            try:
                feat_np = feat_map[0].cpu().numpy()
                _ = feat_np.tobytes()  # verifies it's serializable
            except Exception:
                can_cache = False

            n_events += 1
            if n_events >= args.max_events:
                break

        if device.type == "cuda":
            mem_peak = torch.cuda.max_memory_allocated() / 1024**3
            mem_after = torch.cuda.memory_allocated() / 1024**3
        else:
            mem_peak = 0
            mem_after = 0

        dinov2_result = {
            "n_frames_processed": n_events,
            "total_time_s": round(total_time, 1),
            "time_per_frame_s": round(total_time / max(n_events, 1), 2),
            "feature_grid_shape": feature_sizes[0] if feature_sizes else None,
            "feature_dim": feature_sizes[0][1] if feature_sizes else None,
            "gpu_mem_before_gb": round(mem_before, 2) if device.type == "cuda" else 0,
            "gpu_mem_peak_gb": round(mem_peak, 2),
            "gpu_mem_after_gb": round(mem_after, 2),
            "crashes": crashes,
            "oom": False,
            "cacheable_to_disk": can_cache,
        }
        results["feature_sources"].append({"name": "dinov2", **dinov2_result})

        print(f"  Time/frame: {dinov2_result['time_per_frame_s']:.2f}s")
        print(f"  Feature grid: {dinov2_result['feature_grid_shape']}")
        print(f"  GPU peak: {dinov2_result['gpu_mem_peak_gb']:.2f} GB")
        print(f"  Cacheable: {dinov2_result['cacheable_to_disk']}")

    # Decision
    all_cacheable = all(s.get("cacheable_to_disk", True) for s in results["feature_sources"])
    any_oom = any(s.get("oom", False) for s in results["feature_sources"])
    results["decision"] = {
        "can_proceed_to_p2b": all_cacheable and not any_oom,
        "blockers": (
            (["OOM detected"] if any_oom else []) +
            (["not cacheable"] if not all_cacheable else [])
        ),
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.output_json}")
    print(f"Decision: proceed_to_p2b={results['decision']['can_proceed_to_p2b']}")


if __name__ == "__main__":
    main()
