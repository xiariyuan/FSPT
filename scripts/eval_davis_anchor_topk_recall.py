#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_world_state_stage2_causal_dino import (
    DINOFeatureExtractor,
    extract_crop,
    extract_template,
    template_score_map,
)
from utils.attempt0_schema import load_attempt0_cache


def _find_first_reentry(gt_visibility: np.ndarray, query_t: int) -> Optional[Tuple[int, int]]:
    in_occlusion = False
    occ_len = 0
    for t in range(int(query_t) + 1, int(gt_visibility.shape[0])):
        visible = bool(gt_visibility[t])
        if not visible:
            in_occlusion = True
            occ_len += 1
            continue
        if in_occlusion:
            return t, occ_len
    return None


def _yx_norm_to_xy_px(yx: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.array(
        [
            float(yx[1]) * max(float(width) - 1.0, 1.0),
            float(yx[0]) * max(float(height) - 1.0, 1.0),
        ],
        dtype=np.float32,
    )


def _topk_coords(score_map: torch.Tensor, topk: int, image_width: int, image_height: int) -> np.ndarray:
    h, w = score_map.shape
    vals, idx = torch.topk(score_map.reshape(-1), k=min(topk, score_map.numel()))
    coords = []
    for i in idx.tolist():
        y = i // w
        x = i % w
        coords.append([(x + 0.5) * image_width / float(w), (y + 0.5) * image_height / float(h)])
    return np.asarray(coords, dtype=np.float32)


def _summarize_samples(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {
            "n": 0,
            "top1_median_px": None,
            "top5_best_median_px": None,
            "top1_hit_4px": None,
            "top1_hit_8px": None,
            "top1_hit_16px": None,
            "top5_hit_4px": None,
            "top5_hit_8px": None,
            "top5_hit_16px": None,
            "top1_miss_top5_hit_8px": None,
            "top1_miss_top5_hit_16px": None,
        }
    top1 = np.asarray([s["top1_dist_px"] for s in samples], dtype=np.float32)
    top5 = np.asarray([s["top5_best_dist_px"] for s in samples], dtype=np.float32)
    top1_hit_8 = top1 <= 8.0
    top5_hit_8 = top5 <= 8.0
    top1_hit_16 = top1 <= 16.0
    top5_hit_16 = top5 <= 16.0
    return {
        "n": int(len(samples)),
        "top1_median_px": float(np.median(top1)),
        "top5_best_median_px": float(np.median(top5)),
        "top1_hit_4px": float(np.mean(top1 <= 4.0)),
        "top1_hit_8px": float(np.mean(top1_hit_8)),
        "top1_hit_16px": float(np.mean(top1_hit_16)),
        "top5_hit_4px": float(np.mean(top5 <= 4.0)),
        "top5_hit_8px": float(np.mean(top5_hit_8)),
        "top5_hit_16px": float(np.mean(top5_hit_16)),
        "top1_miss_top5_hit_8px": float(np.mean((~top1_hit_8) & top5_hit_8)),
        "top1_miss_top5_hit_16px": float(np.mean((~top1_hit_16) & top5_hit_16)),
    }


def _bucket_name(occ_length: int) -> str:
    if occ_length < 20:
        return "<20"
    if occ_length < 50:
        return "20-49"
    if occ_length < 100:
        return "50-99"
    return "100+"


def _make_markdown(
    *,
    config: Dict[str, Any],
    overall: Dict[str, Any],
    long_occ: Dict[str, Any],
    buckets: Dict[str, Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("# Phase 2 — DAVIS Anchor Top-K Recall Smoke")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- Dataset: `tapvid_davis`")
    lines.append(f"- Query protocol: `{config['protocol']}`")
    lines.append(f"- Anchor source: `query-frame GT patch`")
    lines.append(f"- Retrieval target: `first re-entry frame`")
    lines.append(f"- Feature extractor: `DINOv2 ViT-S/14`")
    lines.append(f"- Top-K: `{config['topk']}`")
    lines.append(f"- Query crop size: `{config['query_crop_size']}`")
    lines.append(f"- Sample cap: `max_videos={config['max_videos']}`, `max_queries={config['max_queries']}`")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| n | top1 median px | top5 best median px | top1@8 | top5@8 | top1@16 | top5@16 | top1 miss but top5 hit @16 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {overall['n']} | {overall['top1_median_px']:.2f} | {overall['top5_best_median_px']:.2f} | "
        f"{100.0*overall['top1_hit_8px']:.1f}% | {100.0*overall['top5_hit_8px']:.1f}% | "
        f"{100.0*overall['top1_hit_16px']:.1f}% | {100.0*overall['top5_hit_16px']:.1f}% | "
        f"{100.0*overall['top1_miss_top5_hit_16px']:.1f}% |"
    )
    lines.append("")
    lines.append("## Long-Occlusion (occ >= 20)")
    lines.append("")
    lines.append("| n | top1 median px | top5 best median px | top1@8 | top5@8 | top1@16 | top5@16 | top1 miss but top5 hit @16 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    if long_occ["n"]:
        lines.append(
            f"| {long_occ['n']} | {long_occ['top1_median_px']:.2f} | {long_occ['top5_best_median_px']:.2f} | "
            f"{100.0*long_occ['top1_hit_8px']:.1f}% | {100.0*long_occ['top5_hit_8px']:.1f}% | "
            f"{100.0*long_occ['top1_hit_16px']:.1f}% | {100.0*long_occ['top5_hit_16px']:.1f}% | "
            f"{100.0*long_occ['top1_miss_top5_hit_16px']:.1f}% |"
        )
    else:
        lines.append("| 0 | - | - | - | - | - | - | - |")
    lines.append("")
    lines.append("## Buckets")
    lines.append("")
    lines.append("| Occ bucket | n | top1@8 | top5@8 | top1@16 | top5@16 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for bucket_name in ("<20", "20-49", "50-99", "100+"):
        bucket = buckets[bucket_name]
        if not bucket["n"]:
            lines.append(f"| {bucket_name} | 0 | - | - | - | - |")
            continue
        lines.append(
            f"| {bucket_name} | {bucket['n']} | {100.0*bucket['top1_hit_8px']:.1f}% | "
            f"{100.0*bucket['top5_hit_8px']:.1f}% | {100.0*bucket['top1_hit_16px']:.1f}% | "
            f"{100.0*bucket['top5_hit_16px']:.1f}% |"
        )
    lines.append("")
    lines.append("## Decision Hints")
    lines.append("")
    lines.append("- `Phase 2 pass` requires strong Top-5 recall. The pre-registered ladder threshold was `Top-5 >= 70%`.")
    lines.append("- `Phase 3 value` is indicated by `top1 miss but top5 hit`: the anchor is present in the candidate set, but ranking is wrong.")
    return "\n".join(lines).strip() + "\n"


@dataclass
class QuerySample:
    video_id: str
    query_index: int
    query_t: int
    reentry_t: int
    occ_length: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DAVIS anchor Top-K recall on strided+original re-entry queries.")
    parser.add_argument("--cache-path", type=str, default="caches/trackon2_strided_original.pt")
    parser.add_argument("--pkl-path", type=str, default="/gemini/code/datasets/tapvid_davis/tapvid_davis.pkl")
    parser.add_argument("--weights", type=str, default="/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    parser.add_argument("--query-crop-size", type=int, default=112)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-videos", type=int, default=5, help="0 means all")
    parser.add_argument("--max-queries", type=int, default=128, help="0 means all")
    parser.add_argument("--min-occ-length", type=int, default=20)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DINOFeatureExtractor(Path(args.weights), device)

    payload = load_attempt0_cache(Path(args.cache_path))
    records = payload["records"]
    if args.max_videos > 0:
        records = records[: args.max_videos]

    with open(args.pkl_path, "rb") as f:
        pkl_data = pickle.load(f)

    frame_feature_cache: Dict[Tuple[str, int], torch.Tensor] = {}

    def get_frame_feature(video_id: str, frame_idx: int, frame_rgb: np.ndarray) -> torch.Tensor:
        key = (video_id, int(frame_idx))
        cached = frame_feature_cache.get(key)
        if cached is not None:
            return cached
        feat = extractor.feature_map(frame_rgb)
        frame_feature_cache[key] = feat
        return feat

    samples: List[Dict[str, Any]] = []
    total_queries = 0
    stop = False
    for record in records:
        video_id = str(record["video_id"])
        video_entry = pkl_data[video_id]
        video_rgb = np.asarray(video_entry["video"], dtype=np.uint8)
        height, width = video_rgb.shape[1], video_rgb.shape[2]

        query_points = np.asarray(record["query_points"], dtype=np.float32)
        gt_tracks = np.asarray(record["gt_tracks"], dtype=np.float32)
        gt_visibility = np.asarray(record["gt_visibility"], dtype=bool)

        for query_index in range(query_points.shape[0]):
            if args.max_queries > 0 and total_queries >= args.max_queries:
                stop = True
                break

            query_t = int(np.clip(round(float(query_points[query_index, 0])), 0, gt_visibility.shape[1] - 1))
            reentry = _find_first_reentry(gt_visibility[query_index], query_t)
            if reentry is None:
                continue
            reentry_t, occ_length = reentry

            query_yx = gt_tracks[query_index, query_t]
            reentry_yx = gt_tracks[query_index, reentry_t]
            query_xy_px = _yx_norm_to_xy_px(query_yx, height, width)
            gt_reentry_xy_px = _yx_norm_to_xy_px(reentry_yx, height, width)

            query_frame = video_rgb[query_t]
            reentry_frame = video_rgb[reentry_t]

            query_crop = extract_crop(query_frame, query_xy_px, args.query_crop_size)
            query_feat = extractor.feature_map(query_crop)
            reentry_feat = get_frame_feature(video_id, reentry_t, reentry_frame)

            q_center_xy = np.array([args.query_crop_size / 2.0, args.query_crop_size / 2.0], dtype=np.float32)
            templates = [
                extract_template(query_feat, q_center_xy, args.query_crop_size, radius)
                for radius in (1, 2)
            ]
            score_maps = [template_score_map(template, reentry_feat) for template in templates]
            score_map = torch.stack(score_maps, dim=0).mean(dim=0)
            coords = _topk_coords(score_map, args.topk, width, height)
            dists = np.linalg.norm(coords - gt_reentry_xy_px[None, :], axis=1)

            samples.append(
                {
                    "video_id": video_id,
                    "query_index": int(query_index),
                    "query_t": int(query_t),
                    "reentry_t": int(reentry_t),
                    "occ_length": int(occ_length),
                    "gt_reentry_xy_px": gt_reentry_xy_px.tolist(),
                    "topk_coords_xy_px": coords.tolist(),
                    "topk_dists_px": dists.astype(np.float32).tolist(),
                    "top1_dist_px": float(dists[0]),
                    "top5_best_dist_px": float(dists.min()),
                }
            )
            total_queries += 1
        if stop:
            break

    long_occ_samples = [s for s in samples if int(s["occ_length"]) >= args.min_occ_length]
    bucket_samples: Dict[str, List[Dict[str, Any]]] = {k: [] for k in ("<20", "20-49", "50-99", "100+")}
    for sample in samples:
        bucket_samples[_bucket_name(int(sample["occ_length"]))].append(sample)

    overall = _summarize_samples(samples)
    long_occ = _summarize_samples(long_occ_samples)
    buckets = {k: _summarize_samples(v) for k, v in bucket_samples.items()}

    output = {
        "config": {
            "protocol": str(payload.get("protocol", "")),
            "cache_path": args.cache_path,
            "pkl_path": args.pkl_path,
            "topk": int(args.topk),
            "query_crop_size": int(args.query_crop_size),
            "max_videos": int(args.max_videos),
            "max_queries": int(args.max_queries),
            "min_occ_length": int(args.min_occ_length),
        },
        "summary": {
            "overall": overall,
            "long_occ": long_occ,
            "buckets": buckets,
        },
        "samples": samples,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    out_md.write_text(
        _make_markdown(config=output["config"], overall=overall, long_occ=long_occ, buckets=buckets),
        encoding="utf-8",
    )

    print(json.dumps(output["summary"], indent=2, ensure_ascii=True))
    print(f"[ok] wrote {out_json}")
    print(f"[ok] wrote {out_md}")


if __name__ == "__main__":
    main()
