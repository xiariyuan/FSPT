#!/usr/bin/env python3
"""Inspect RGB frame access and patch extraction for re-entry appearance work.

This script verifies that we can map cache `video_id` values to RGB frames and
extract patches at normalized yx coordinates.  It is intentionally lightweight:
no model inference, no training, and no metric evaluation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.tapvid_rgb_stacking import TAPVidRGBStackingDataset  # noqa: E402
from utils.reentry_viscalibrator_features import npy  # noqa: E402


DEFAULT_NATURAL_CACHE = "outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/offline_rgb_stacking_fresh20_49.pt"
DEFAULT_TRANSLATE_CACHE = "outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_49/translate_L16/predictions/cotracker3_offline_translate_L16_fresh20_49.pt"
DEFAULT_OCCLUDER_CACHE = "outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_49/occluder_L16/predictions/cotracker3_offline_occluder_L16_fresh20_49.pt"
DEFAULT_TRANSLATE_STRESS = "outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_49/translate_L16/stress_dataset.pt"
DEFAULT_OCCLUDER_STRESS = "outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_49/occluder_L16/stress_dataset.pt"


def load_cache(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def to_uint8_video(video: Any) -> np.ndarray:
    """Convert video to uint8 THWC."""
    if isinstance(video, torch.Tensor):
        arr = video.detach().cpu().numpy()
    else:
        arr = np.asarray(video)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D video, got shape {arr.shape}")
    # TCHW -> THWC
    if arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.dtype != np.uint8:
        if float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected 3-channel video, got shape {arr.shape}")
    return arr


def parse_rgb_index(video_id: str) -> int:
    m = re.search(r"rgb_stacking_(\d{6})", str(video_id))
    if not m:
        raise ValueError(f"Cannot parse RGB stacking index from video_id={video_id}")
    return int(m.group(1))


def load_natural_video(video_id: str, *, root: str, pkl_name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    idx = parse_rgb_index(video_id)
    ds = TAPVidRGBStackingDataset(root=root, pkl_name=pkl_name, start_index=idx, num_videos=1)
    sample = ds[0]
    video = to_uint8_video(sample["video"])
    return video, {
        "resolver": "TAPVidRGBStackingDataset",
        "dataset_video_name": str(sample["video_name"]),
        "sequence_index": int(sample["sequence_index"]),
        "original_size": [int(x) for x in sample["original_size"].tolist()],
    }


def load_stress_index(path: str | Path) -> Dict[str, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload.get("records", [])
    return {str(r["video_id"]): r for r in records}


def load_stress_video(video_id: str, translate_index: Dict[str, Dict[str, Any]], occluder_index: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    vid = str(video_id)
    if vid in translate_index:
        r = translate_index[vid]
        return to_uint8_video(r["video"]), {
            "resolver": "stress_dataset_translate",
            "source_video_id": str(r.get("source_video_id", "")),
            "sequence_index": int(r.get("sequence_index", -1)),
            "original_size": [int(x) for x in npy(r["original_size"], np.int32).reshape(-1).tolist()],
            "stress_type": str(r.get("stress_type", "")),
            "stress_params": r.get("stress_params", {}),
        }
    if vid in occluder_index:
        r = occluder_index[vid]
        return to_uint8_video(r["video"]), {
            "resolver": "stress_dataset_occluder",
            "source_video_id": str(r.get("source_video_id", "")),
            "sequence_index": int(r.get("sequence_index", -1)),
            "original_size": [int(x) for x in npy(r["original_size"], np.int32).reshape(-1).tolist()],
            "stress_type": str(r.get("stress_type", "")),
            "stress_params": r.get("stress_params", {}),
        }
    raise KeyError(f"video_id not found in stress datasets: {video_id}")


def resolve_video(video_id: str, *, root: str, pkl_name: str, translate_index: Dict[str, Dict[str, Any]], occluder_index: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    vid = str(video_id)
    if "translate_L16" in vid or "occluder_L16" in vid:
        return load_stress_video(vid, translate_index, occluder_index)
    return load_natural_video(vid, root=root, pkl_name=pkl_name)


def norm_yx_to_pixel(yx: np.ndarray, h: int, w: int) -> Tuple[float, float]:
    y = float(yx[0]) * max(h - 1, 1)
    x = float(yx[1]) * max(w - 1, 1)
    return y, x


def extract_patch(img: np.ndarray, y: float, x: float, size: int) -> Tuple[np.ndarray, float]:
    assert img.ndim == 3 and img.shape[-1] == 3
    half = int(size) // 2
    cy = int(round(float(y)))
    cx = int(round(float(x)))
    h, w = img.shape[:2]
    y0, y1 = cy - half, cy + half + 1
    x0, x1 = cx - half, cx + half + 1
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    sy0, sy1 = max(0, y0), min(h, y1)
    sx0, sx1 = max(0, x0), min(w, x1)
    if sy1 <= sy0 or sx1 <= sx0:
        return patch, 0.0
    py0, px0 = sy0 - y0, sx0 - x0
    patch[py0 : py0 + (sy1 - sy0), px0 : px0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
    valid = float((sy1 - sy0) * (sx1 - sx0)) / float(size * size)
    return patch, valid


def patch_stats(patch: np.ndarray, valid_ratio: float) -> Dict[str, Any]:
    arr = patch.astype(np.float32) / 255.0
    return {
        "valid_ratio": round(float(valid_ratio), 4),
        "mean_rgb": [round(float(x), 4) for x in arr.reshape(-1, 3).mean(axis=0).tolist()],
        "std_rgb": [round(float(x), 4) for x in arr.reshape(-1, 3).std(axis=0).tolist()],
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
    }


def choose_query(record: Dict[str, Any]) -> int:
    gt_vis = npy(record["gt_visibility"], bool)
    base_vis = npy(record["pred_visibility"], bool)
    # Prefer a query with both visible and invisible frames for patch sanity.
    scores = (gt_vis.sum(axis=1) > 0).astype(np.int32) + ((~gt_vis).sum(axis=1) > 0).astype(np.int32) + (base_vis.sum(axis=1) > 0).astype(np.int32)
    return int(np.argmax(scores))


def inspect_record(record: Dict[str, Any], video: np.ndarray, resolver_info: Dict[str, Any], *, max_patch_sizes: List[int]) -> Dict[str, Any]:
    vid = str(record["video_id"])
    h, w = int(video.shape[1]), int(video.shape[2])
    rec_size = [int(x) for x in npy(record["original_size"], np.int32).reshape(-1).tolist()]
    qi = choose_query(record)
    qpts = npy(record["query_points"], np.float32)
    pred_tracks = npy(record["pred_tracks"], np.float32)
    gt_tracks = npy(record["gt_tracks"], np.float32)
    gt_vis = npy(record["gt_visibility"], bool)
    base_vis = npy(record["pred_visibility"], bool)
    t_len = int(video.shape[0])
    query_t = int(round(float(qpts[qi, 0])))
    query_t = max(0, min(t_len - 1, query_t))
    # Pick a later frame that is in-bounds and, if possible, gt-visible.
    visible_frames = np.where(gt_vis[qi])[0]
    later_visible = visible_frames[visible_frames >= query_t]
    cand_t = int(later_visible[min(len(later_visible) - 1, max(0, len(later_visible) // 2))]) if later_visible.size else query_t
    cand_t = max(0, min(t_len - 1, cand_t))

    q_yx = np.asarray([qpts[qi, 1], qpts[qi, 2]], dtype=np.float32)
    gt_yx = gt_tracks[qi, cand_t]
    base_yx = pred_tracks[qi, cand_t]
    q_py, q_px = norm_yx_to_pixel(q_yx, h, w)
    gt_py, gt_px = norm_yx_to_pixel(gt_yx, h, w)
    base_py, base_px = norm_yx_to_pixel(base_yx, h, w)

    patch_report = []
    for size in max_patch_sizes:
        q_patch, q_valid = extract_patch(video[query_t], q_py, q_px, int(size))
        b_patch, b_valid = extract_patch(video[cand_t], base_py, base_px, int(size))
        g_patch, g_valid = extract_patch(video[cand_t], gt_py, gt_px, int(size))
        patch_report.append({
            "size": int(size),
            "query_patch": patch_stats(q_patch, q_valid),
            "base_candidate_patch": patch_stats(b_patch, b_valid),
            "gt_candidate_patch": patch_stats(g_patch, g_valid),
        })

    return {
        "video_id": vid,
        "resolver_info": resolver_info,
        "video_shape_THWC": [int(x) for x in video.shape],
        "record_original_size": rec_size,
        "size_matches_record": bool([h, w] == rec_size),
        "frame_count_matches_record": bool(t_len == int(record.get("frame_count", t_len))),
        "selected_query_idx": int(qi),
        "query_t": int(query_t),
        "candidate_t": int(cand_t),
        "query_yx_norm": [round(float(x), 6) for x in q_yx.tolist()],
        "base_candidate_yx_norm": [round(float(x), 6) for x in base_yx.tolist()],
        "gt_candidate_yx_norm": [round(float(x), 6) for x in gt_yx.tolist()],
        "query_pixel_yx": [round(float(q_py), 3), round(float(q_px), 3)],
        "base_candidate_pixel_yx": [round(float(base_py), 3), round(float(base_px), 3)],
        "gt_candidate_pixel_yx": [round(float(gt_py), 3), round(float(gt_px), 3)],
        "base_visible_at_candidate": bool(base_vis[qi, cand_t]),
        "gt_visible_at_candidate": bool(gt_vis[qi, cand_t]),
        "patch_report": patch_report,
    }


def inspect_cache(cache_path: str, setting: str, *, root: str, pkl_name: str, translate_index: Dict[str, Dict[str, Any]], occluder_index: Dict[str, Dict[str, Any]], max_records: int, patch_sizes: List[int]) -> Dict[str, Any]:
    cache = load_cache(cache_path)
    records = cache.get("records", [])[: int(max_records)]
    out_records = []
    for r in records:
        video, info = resolve_video(str(r["video_id"]), root=root, pkl_name=pkl_name, translate_index=translate_index, occluder_index=occluder_index)
        out_records.append(inspect_record(r, video, info, max_patch_sizes=patch_sizes))
    return {
        "setting": setting,
        "cache_path": cache_path,
        "n_records_checked": len(out_records),
        "records": out_records,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect RGB frame access and patch extraction")
    ap.add_argument("--rgb-root", default="/gemini/code/datasets/tapvid_rgb_stacking")
    ap.add_argument("--rgb-pkl", default="tapvid_rgb_stacking.pkl")
    ap.add_argument("--natural-cache", default=DEFAULT_NATURAL_CACHE)
    ap.add_argument("--translate-cache", default=DEFAULT_TRANSLATE_CACHE)
    ap.add_argument("--occluder-cache", default=DEFAULT_OCCLUDER_CACHE)
    ap.add_argument("--translate-stress", default=DEFAULT_TRANSLATE_STRESS)
    ap.add_argument("--occluder-stress", default=DEFAULT_OCCLUDER_STRESS)
    ap.add_argument("--max-records", type=int, default=2)
    ap.add_argument("--patch-sizes", type=int, nargs="*", default=[9, 17, 33])
    ap.add_argument("--out-json", default="outputs/paper_discovery_2026-06-27/reentry_appearance/inspect_rgb_frame_access.json")
    ap.add_argument("--out-md", default="docs/reentry_appearance_frame_access_inspection_2026-07-03.md")
    args = ap.parse_args()

    translate_index = load_stress_index(args.translate_stress)
    occluder_index = load_stress_index(args.occluder_stress)
    payload = {
        "rgb_root": args.rgb_root,
        "rgb_pkl": args.rgb_pkl,
        "translate_stress": args.translate_stress,
        "occluder_stress": args.occluder_stress,
        "patch_sizes": [int(x) for x in args.patch_sizes],
        "settings": [
            inspect_cache(args.natural_cache, "natural", root=args.rgb_root, pkl_name=args.rgb_pkl, translate_index=translate_index, occluder_index=occluder_index, max_records=args.max_records, patch_sizes=args.patch_sizes),
            inspect_cache(args.translate_cache, "translate_L16", root=args.rgb_root, pkl_name=args.rgb_pkl, translate_index=translate_index, occluder_index=occluder_index, max_records=args.max_records, patch_sizes=args.patch_sizes),
            inspect_cache(args.occluder_cache, "occluder_L16", root=args.rgb_root, pkl_name=args.rgb_pkl, translate_index=translate_index, occluder_index=occluder_index, max_records=args.max_records, patch_sizes=args.patch_sizes),
        ],
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# ReEntry Appearance Frame Access Inspection", ""]
    lines.append(f"RGB root: `{args.rgb_root}`")
    lines.append(f"Patch sizes: `{args.patch_sizes}`")
    lines.append("")
    for setting in payload["settings"]:
        lines.append(f"## {setting['setting']}")
        lines.append("")
        for rec in setting["records"]:
            lines.append(f"### {rec['video_id']}")
            lines.append("")
            lines.append(f"Resolver: `{rec['resolver_info']['resolver']}`")
            lines.append(f"Video shape THWC: `{rec['video_shape_THWC']}`")
            lines.append(f"Record original size: `{rec['record_original_size']}`")
            lines.append(f"Size matches record: `{rec['size_matches_record']}`; frame count matches: `{rec['frame_count_matches_record']}`")
            lines.append(f"Query idx/t: `{rec['selected_query_idx']}` / `{rec['query_t']}`; candidate t: `{rec['candidate_t']}`")
            lines.append(f"Query pixel yx: `{rec['query_pixel_yx']}`; base candidate pixel yx: `{rec['base_candidate_pixel_yx']}`; gt candidate pixel yx: `{rec['gt_candidate_pixel_yx']}`")
            lines.append(f"Base visible at candidate: `{rec['base_visible_at_candidate']}`; GT visible: `{rec['gt_visible_at_candidate']}`")
            lines.append("")
            lines.append("| patch | query valid | base valid | gt valid | query mean | base mean | gt mean |")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|")
            for pr in rec["patch_report"]:
                lines.append(
                    f"| {pr['size']} | {pr['query_patch']['valid_ratio']} | {pr['base_candidate_patch']['valid_ratio']} | {pr['gt_candidate_patch']['valid_ratio']} | "
                    f"{pr['query_patch']['mean']} | {pr['base_candidate_patch']['mean']} | {pr['gt_candidate_patch']['mean']} |"
                )
            lines.append("")
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
