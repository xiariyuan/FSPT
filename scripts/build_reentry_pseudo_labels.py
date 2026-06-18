#!/usr/bin/env python3
"""Build re-entry oracle labels from CT-offline centered local search.

P3A is oracle-only diagnostic code. It is not a deployable pseudo-label
generation pipeline and must not be used as a training-data source on DAVIS.

For each re-entry query:
  1. CoTracker3 offline prediction → coarse center
  2. Local DINOv2 template matching within 32px radius → top-5 candidates
  3. Select best candidate as pseudo-label
  4. Attach quality flags: FB consistency (placeholder), CT-offline agreement

Output: JSONL with schema:
  {video_id, track_id, frame, xy, coord_format, visible, teacher, teacher_conf,
   fb_error, flow_consistency_error, occ_run_len, is_reentry_frame,
   source_bucket, quality_flags}
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.coords import feat_yx_to_xy_pixel, find_first_reentry, yx_norm_to_xy_pixel
from utils.attempt0_schema import load_attempt0_cache


def build_pseudo_labels(
    cache_path: str,
    ct_offline_cache: str,
    pkl_path: str,
    dino_weights: str,
    max_videos: int = 0,
    max_queries: int = 0,
    search_radius_px: int = 32,
    query_crop_size: int = 112,
) -> List[Dict[str, Any]]:
    """Build pseudo-label dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from scripts.eval_world_state_stage2_causal_dino import (
        DINOFeatureExtractor, extract_crop, extract_template, template_score_map,
    )
    dino = DINOFeatureExtractor(Path(dino_weights), device)

    payload = load_attempt0_cache(Path(cache_path))
    ct_off_payload = load_attempt0_cache(Path(ct_offline_cache))
    records = payload["records"]
    # Build full ct_off dict by video_id (fix D: avoid positional mismatch)
    ct_off_by_video = {str(r["video_id"]): r for r in ct_off_payload["records"]}
    if max_videos > 0:
        records = records[:max_videos]
    # Verify all records have ct_off data
    for r in records:
        vid = str(r["video_id"])
        if vid not in ct_off_by_video:
            raise ValueError(f"ct-offline cache missing video: {vid}")

    import pickle
    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)

    frame_feat_cache: Dict[tuple, torch.Tensor] = {}

    def get_frame_feat(video_id: str, t: int, frame_rgb: np.ndarray) -> torch.Tensor:
        key = (video_id, t)
        if key not in frame_feat_cache:
            frame_feat_cache[key] = dino.feature_map(frame_rgb)
        return frame_feat_cache[key]

        from utils.fb_consistency import compute_fb_error

    labels: List[Dict[str, Any]] = []
    total = 0
    stop = False

    for rec_idx, r in enumerate(records):
        if stop:
            break
        vid = r["video_id"]
        video_rgb = np.asarray(pkl_data[vid]["video"], dtype=np.uint8)
        h, w = int(r["original_size"][0]), int(r["original_size"][1])
        gt_vis = np.asarray(r["gt_visibility"], dtype=bool)
        gt_tracks = np.asarray(r["gt_tracks"], dtype=np.float32)
        qpts = np.asarray(r["query_points"], dtype=np.float32)
        if vid not in ct_off_by_video:
            raise ValueError(f"ct-offline cache missing video_id={vid}")
        ct_pred = np.asarray(ct_off_by_video[vid]["pred_tracks"], dtype=np.float32)

        for qi in range(qpts.shape[0]):
            if max_queries > 0 and total >= max_queries:
                stop = True
                break

            qt = int(round(float(qpts[qi, 0])))
            re = find_first_reentry(gt_vis[qi], qt)
            if re is None:
                continue
            t_re = re["reentry_frame"]
            occ_len = re["occ_length"]
            last_vis_t = re["last_visible_t"]

            gt_yx = gt_tracks[qi, t_re]
            gt_xy_px = yx_norm_to_xy_pixel(gt_yx, h, w)
            ct_off_yx = ct_pred[qi, t_re]
            ct_off_xy_px = yx_norm_to_xy_pixel(ct_off_yx, h, w)
            ct_off_error = float(np.linalg.norm(ct_off_xy_px - gt_xy_px))

            # Last-visible anchor + local search around CT-offline
            last_yx = gt_tracks[qi, last_vis_t]
            last_xy_px = yx_norm_to_xy_pixel(last_yx, h, w)
            last_frame = video_rgb[last_vis_t]
            last_crop = extract_crop(last_frame, last_xy_px, query_crop_size)
            last_feat = dino.feature_map(last_crop)
            qc = np.array([query_crop_size / 2.0] * 2, dtype=np.float32)
            last_templates = [extract_template(last_feat, qc, query_crop_size, r) for r in (1, 2)]

            reentry_feat = get_frame_feat(vid, t_re, video_rgb[t_re])
            score_map = torch.stack(
                [template_score_map(t, reentry_feat) for t in last_templates], dim=0
            ).mean(dim=0)

            # Mask to local region around CT-offline
            h_feat, w_feat = reentry_feat.shape[-2:]
            ct_off_xy_px[0] = float(np.clip(ct_off_xy_px[0], 0.0, max(w - 1, 0)))
            ct_off_xy_px[1] = float(np.clip(ct_off_xy_px[1], 0.0, max(h - 1, 0)))
            cy = ct_off_xy_px[1] * h_feat / float(h)
            cx = ct_off_xy_px[0] * w_feat / float(w)
            r_y = search_radius_px * h_feat / float(h)
            r_x = search_radius_px * w_feat / float(w)
            y0, y1 = max(0, int(cy - r_y)), min(h_feat, int(cy + r_y) + 1)
            x0, x1 = max(0, int(cx - r_x)), min(w_feat, int(cx + r_x) + 1)
            local_mask = torch.zeros_like(score_map, dtype=torch.bool)
            local_mask[y0:y1, x0:x1] = True
            if not bool(local_mask.any()):
                continue
            local_score = score_map.masked_fill(~local_mask, float("-inf"))

            # Top-1 candidate (pseudo-label)
            h_s, w_s = local_score.shape
            best_idx = int(torch.argmax(local_score.reshape(-1)).item())
            best_y = best_idx // w_s
            best_x = best_idx % w_s
            pl_xy = feat_yx_to_xy_pixel(
                np.asarray([[best_y, best_x]], dtype=np.float32),
                h_s,
                w_s,
                h,
                w,
            )[0]
            pl_x = float(pl_xy[0])
            pl_y = float(pl_xy[1])
            pl_error = float(np.linalg.norm(np.array([pl_x, pl_y]) - gt_xy_px))

            # Quality flags
            quality_flags = []
            if pl_error < ct_off_error:
                quality_flags.append("better_than_ct_offline")
            if pl_error < 8:
                quality_flags.append("within_8px")
            if pl_error < 16:
                quality_flags.append("within_16px")

            # FB consistency: forward-backward tracking via DINO
            if t_re + 1 < len(video_rgb):
                from utils.fb_consistency import compute_fb_error
                fb_error = compute_fb_error(
                    video_rgb[t_re], video_rgb[t_re + 1],
                    float(pl_x), float(pl_y),
                    dino.model, device,
                    patch_size=48, search_radius=12,
                )
            else:
                fb_error = -1.0
            flow_consistency_error = -1.0  # TBD: requires external flow

            labels.append({
                "video_id": vid,
                "track_id": qi,
                "frame": t_re,
                "xy": [round(pl_x, 2), round(pl_y, 2)],
                "coord_format": "xy_pixel",
                "visible": True,
                "teacher": "cotracker3_offline_dinov2_local",
                "teacher_conf": round(float(score_map[best_y, best_x].item()), 4),
                "fb_error": fb_error,
                "flow_consistency_error": flow_consistency_error,
                "occ_run_len": occ_len,
                "is_reentry_frame": True,
                "source_bucket": "ct_offline",
                "quality_flags": quality_flags,
                "gt_xy_px": gt_xy_px.tolist(),
                "pseudo_label_error_px": round(pl_error, 2),
                "ct_offline_error_px": round(ct_off_error, 2),
            })
            total += 1

    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", type=str,
                        default="caches/trackon2_strided_original.pt")
    parser.add_argument("--ct-offline-cache", type=str,
                        default="outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt")
    parser.add_argument("--pkl-path", type=str,
                        default="/gemini/code/datasets/tapvid_davis/tapvid_davis.pkl")
    parser.add_argument("--dino-weights", type=str,
                        default="/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--output-jsonl", type=str, required=True)
    args = parser.parse_args()

    labels = build_pseudo_labels(
        cache_path=args.cache_path,
        ct_offline_cache=args.ct_offline_cache,
        pkl_path=args.pkl_path,
        dino_weights=args.dino_weights,
        max_videos=args.max_videos,
        max_queries=args.max_queries,
    )

    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_jsonl, "w") as f:
        for label in labels:
            f.write(json.dumps(label, ensure_ascii=False) + "\n")

    errors = np.array([l["pseudo_label_error_px"] for l in labels])
    print(f"Built {len(labels)} pseudo-labels")
    print(f"  median error: {np.median(errors):.1f}px")
    print(f"  <4px: {np.mean(errors < 4)*100:.1f}%")
    print(f"  <8px: {np.mean(errors < 8)*100:.1f}%")
    print(f"  <16px: {np.mean(errors < 16)*100:.1f}%")
    print(f"  better than CT-offline: {np.mean([l['pseudo_label_error_px'] < l['ct_offline_error_px'] for l in labels])*100:.1f}%")
    print(f"Wrote {args.output_jsonl}")


if __name__ == "__main__":
    main()
