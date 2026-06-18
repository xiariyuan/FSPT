#!/usr/bin/env python3
"""P2b: Feature-based top-k recall audit for re-entry candidate generation.

Tests: DINOv2 whole-frame template matching (baseline) + CoTracker3 offline
coarse prior guided local search.

Metrics: top1/top5/top10 @ 4/8/16px, long-occ subset, top1 miss but top5 hit.
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

from utils.coords import find_first_reentry, yx_norm_to_xy_pixel
from utils.attempt0_schema import load_attempt0_cache


def _topk_coords(score_map: torch.Tensor, topk: int, W: int, H: int) -> np.ndarray:
    """Extract top-k (x,y) pixel coords from score map."""
    h, w = score_map.shape
    vals, idx = torch.topk(score_map.reshape(-1), k=min(topk, score_map.numel()))
    coords = []
    for i in idx.tolist():
        y = i // w
        x = i % w
        coords.append([
            (x + 0.5) * W / float(w),
            (y + 0.5) * H / float(h),
        ])
    return np.asarray(coords, dtype=np.float32)


def audit_feature_recall(
    cache_path: str,
    ct_offline_cache: str,
    pkl_path: str,
    dino_weights: str,
    max_videos: int = 5,
    max_queries: int = 128,
    topk: int = 10,
    query_crop_size: int = 112,
    search_radius_px: int = 32,
) -> Dict[str, Any]:
    """Audit feature-based top-k recall for re-entry queries."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DINO extractor
    from scripts.eval_world_state_stage2_causal_dino import (
        DINOFeatureExtractor, extract_crop, extract_template, template_score_map,
    )
    dino = DINOFeatureExtractor(Path(dino_weights), device)

    # Load caches
    payload = load_attempt0_cache(Path(cache_path))
    ct_off_payload = load_attempt0_cache(Path(ct_offline_cache))
    records = payload["records"][:max_videos]
    ct_off_records = ct_off_payload["records"][:max_videos]

    import pickle
    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)

    frame_feat_cache: Dict[tuple, torch.Tensor] = {}

    def get_frame_feat(video_id: str, t: int, frame_rgb: np.ndarray) -> torch.Tensor:
        key = (video_id, t)
        if key not in frame_feat_cache:
            feat = dino.feature_map(frame_rgb)
            frame_feat_cache[key] = feat
        return frame_feat_cache[key]

    all_samples = []
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
        ct_pred = np.asarray(ct_off_records[rec_idx]["pred_tracks"], dtype=np.float32)

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

            # --- V0: DINOv2 query-frame anchor, whole-frame retrieval ---
            query_yx = gt_tracks[qi, qt]
            query_xy_px = yx_norm_to_xy_pixel(query_yx, h, w)
            query_frame = video_rgb[qt]
            query_crop = extract_crop(query_frame, query_xy_px, query_crop_size)
            query_feat = dino.feature_map(query_crop)
            qc = np.array([query_crop_size / 2.0, query_crop_size / 2.0], dtype=np.float32)
            templates = [extract_template(query_feat, qc, query_crop_size, r) for r in (1, 2)]
            reentry_feat = get_frame_feat(vid, t_re, video_rgb[t_re])
            score_map = torch.stack([template_score_map(t, reentry_feat) for t in templates], dim=0).mean(dim=0)

            # V0: whole-frame top-k
            wf_coords = _topk_coords(score_map, topk, w, h)
            wf_dists = np.linalg.norm(wf_coords - gt_xy_px[None, :], axis=1)
            wf_top1 = float(wf_dists[0])
            wf_top5 = float(wf_dists[:5].min())
            wf_top10 = float(wf_dists.min())

            # --- V1: Last-visible anchor + whole-frame ---
            last_yx = gt_tracks[qi, last_vis_t]
            last_xy_px = yx_norm_to_xy_pixel(last_yx, h, w)
            last_frame = video_rgb[last_vis_t]
            last_crop = extract_crop(last_frame, last_xy_px, query_crop_size)
            last_feat = dino.feature_map(last_crop)
            last_templates = [extract_template(last_feat, qc, query_crop_size, r) for r in (1, 2)]
            last_score_map = torch.stack([template_score_map(t, reentry_feat) for t in last_templates], dim=0).mean(dim=0)
            lv_coords = _topk_coords(last_score_map, topk, w, h)
            lv_dists = np.linalg.norm(lv_coords - gt_xy_px[None, :], axis=1)
            lv_top1 = float(lv_dists[0])
            lv_top5 = float(lv_dists[:5].min())
            lv_top10 = float(lv_dists.min())

            # --- V2: CT-offline coarse prior + local search ---
            ct_off_yx = ct_pred[qi, t_re]
            ct_off_xy_px = yx_norm_to_xy_pixel(ct_off_yx, h, w)

            # Mask score map to local region
            h_feat, w_feat = reentry_feat.shape[-2:]
            ct_off_xy_px[0] = float(np.clip(ct_off_xy_px[0], 0.0, max(w - 1, 0)))
            ct_off_xy_px[1] = float(np.clip(ct_off_xy_px[1], 0.0, max(h - 1, 0)))
            cy = ct_off_xy_px[1] * h_feat / float(h)
            cx = ct_off_xy_px[0] * w_feat / float(w)
            r_y = search_radius_px * h_feat / float(h)
            r_x = search_radius_px * w_feat / float(w)
            y0, y1 = max(0, int(cy - r_y)), min(h_feat, int(cy + r_y) + 1)
            x0, x1 = max(0, int(cx - r_x)), min(w_feat, int(cx + r_x) + 1)
            local_mask = torch.zeros_like(last_score_map, dtype=torch.bool)
            local_mask[y0:y1, x0:x1] = True
            if not bool(local_mask.any()):
                local_score = last_score_map
            else:
                local_score = last_score_map.masked_fill(~local_mask, float("-inf"))
            ct_local_coords = _topk_coords(local_score, topk, w, h)
            ct_local_dists = np.linalg.norm(ct_local_coords - gt_xy_px[None, :], axis=1)
            ct_local_top1 = float(ct_local_dists[0])
            ct_local_top5 = float(ct_local_dists[:5].min())
            ct_local_top10 = float(ct_local_dists.min())

            all_samples.append({
                "video_id": vid,
                "query_idx": qi,
                "reentry_t": t_re,
                "occ_length": occ_len,
                "gt_xy_px": gt_xy_px.tolist(),
                "ct_offline_xy_px": ct_off_xy_px.tolist(),
                "ct_offline_error_px": float(np.linalg.norm(ct_off_xy_px - gt_xy_px)),
                "v0_dinov2_query_anchor": {
                    "top1_px": wf_top1, "top5_px": wf_top5, "top10_px": wf_top10,
                },
                "v1_last_visible_anchor": {
                    "top1_px": lv_top1, "top5_px": lv_top5, "top10_px": lv_top10,
                },
                "v2_ct_offline_local": {
                    "top1_px": ct_local_top1, "top5_px": ct_local_top5, "top10_px": ct_local_top10,
                },
            })
            total += 1

    # Summarize
    def summarize(samples, variant_key, sub_key):
        vals = np.array([s[variant_key][sub_key] for s in samples])
        return {
            "n": len(vals),
            "median_px": round(float(np.median(vals)), 2),
            "lt4px": round(float(np.mean(vals < 4)), 4),
            "lt8px": round(float(np.mean(vals < 8)), 4),
            "lt16px": round(float(np.mean(vals < 16)), 4),
        }

    n = len(all_samples)
    long_idx = [i for i, s in enumerate(all_samples) if s["occ_length"] >= 20]

    variants = {
        "v0_dinov2_query_anchor": "v0_dinov2_query_anchor",
        "v1_last_visible_anchor": "v1_last_visible_anchor",
        "v2_ct_offline_local": "v2_ct_offline_local",
        "ct_offline_raw": None,  # special: uses ct_offline_error_px directly
    }

    summary = {}
    for vname, vkey in variants.items():
        if vkey is None:
            # CT-offline raw
            raw_errs = np.array([s["ct_offline_error_px"] for s in all_samples])
            overall = {
                "n": len(raw_errs),
                "median_px": round(float(np.median(raw_errs)), 2),
                "lt4px": round(float(np.mean(raw_errs < 4)), 4),
                "lt8px": round(float(np.mean(raw_errs < 8)), 4),
                "lt16px": round(float(np.mean(raw_errs < 16)), 4),
            }
            long_errs = raw_errs[long_idx] if long_idx else np.array([])
            lo = {
                "n": len(long_errs),
                "median_px": round(float(np.median(long_errs)), 2) if len(long_errs) else 0,
                "lt4px": round(float(np.mean(long_errs < 4)), 4) if len(long_errs) else 0,
            }
        else:
            for sub in ["top1_px", "top5_px", "top10_px"]:
                overall = summarize(all_samples, vkey, sub)
                lo = summarize([all_samples[i] for i in long_idx], vkey, sub) if long_idx else {"n": 0}
                summary[f"{vname}_{sub}"] = {"overall": overall, "long_occ_ge20": lo}

    # Top1 miss but top5 hit
    top1_v0 = np.array([s["v0_dinov2_query_anchor"]["top1_px"] for s in all_samples])
    top5_v0 = np.array([s["v0_dinov2_query_anchor"]["top5_px"] for s in all_samples])
    top1_miss_top5_hit_16 = float(np.mean((top1_v0 > 16) & (top5_v0 <= 16)))

    # FB consistency pass rate (placeholder — actual implementation requires flow)
    fb_pass_rate = None  # TBD in P3

    return {
        "n_videos": len(records),
        "n_queries": n,
        "n_long_occ": len(long_idx),
        "summary": summary,
        "top1_miss_top5_hit_16px": top1_miss_top5_hit_16,
        "fb_consistency_pass_rate": fb_pass_rate,
        "config": {
            "max_videos": max_videos,
            "max_queries": max_queries,
            "topk": topk,
            "query_crop_size": query_crop_size,
            "search_radius_px": search_radius_px,
        },
        "samples": all_samples,
    }


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
    parser.add_argument("--max-videos", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=128)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--search-radius-px", type=int, default=32)
    parser.add_argument("--output-json", type=str, required=True)
    args = parser.parse_args()

    results = audit_feature_recall(
        cache_path=args.cache_path,
        ct_offline_cache=args.ct_offline_cache,
        pkl_path=args.pkl_path,
        dino_weights=args.dino_weights,
        max_videos=args.max_videos,
        max_queries=args.max_queries,
        topk=args.topk,
        search_radius_px=args.search_radius_px,
    )

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    # Print key metrics
    for vname in ["v0_dinov2_query_anchor", "v1_last_visible_anchor", "v2_ct_offline_local"]:
        for sub in ["top5_px"]:
            if f"{vname}_{sub}" in results["summary"]:
                s = results["summary"][f"{vname}_{sub}"]
                o = s["overall"]
                lo = s.get("long_occ_ge20", {})
                print(f"{vname}.{sub}: n={o['n']}, median={o['median_px']:.1f}px, "
                      f"<4={o['lt4px']*100:.1f}%, <16={o['lt16px']*100:.1f}%, "
                      f"long_occ<16={lo.get('lt16px', 0)*100:.1f}%")

    print(f"\ntop1 miss but top5 hit @16px: {results['top1_miss_top5_hit_16px']*100:.1f}%")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
