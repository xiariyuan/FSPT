#!/usr/bin/env python3
"""
Build PRT (Persistent Re-entry Tracking) benchmark splits.

Stage 1 target:
  - PRT-Synth from PointOdyssey
  - query extraction
  - off-screen vs in-frame occlusion categorization
  - stratified summary JSON
  - preview JSON with sample queries

This script intentionally focuses on split construction rather than model eval.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_world_state_stage0 import discover_sequences, load_sequence


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PRTQuery:
    seq_name: str
    split: str
    point_idx: int
    query_frame: int
    reentry_frame: int
    occ_length: int
    camera_motion: float
    camera_translation: float
    reentry_type: str
    offscreen_frac: float
    inframe_occ_frac: float
    image_width: int
    image_height: int


def load_image_size(seq_path: Path) -> Tuple[int, int]:
    rgb_dir = seq_path / "rgbs"
    files = sorted(rgb_dir.glob("rgb_*.jpg"))
    if not files:
        raise FileNotFoundError(f"No RGB frames found in {rgb_dir}")
    img = cv2.imread(str(files[0]))
    if img is None:
        raise RuntimeError(f"Failed to read {files[0]}")
    h, w = img.shape[:2]
    return h, w


def frame_in_image(points_xy: np.ndarray, width: int, height: int) -> np.ndarray:
    x = points_xy[..., 0]
    y = points_xy[..., 1]
    finite = np.isfinite(x) & np.isfinite(y)
    return finite & (x >= 0.0) & (x < width) & (y >= 0.0) & (y < height)


def classify_reentry_type(
    trajs_2d_i: np.ndarray,
    visibs_i: np.ndarray,
    valids_i: np.ndarray,
    width: int,
    height: int,
    occ_start: int,
    occ_end: int,
) -> Tuple[str, float, float]:
    """Classify the occlusion interval as in-frame, off-screen, or mixed."""
    if occ_end < occ_start:
        return "mixed", 0.0, 0.0

    occ_xy = trajs_2d_i[occ_start : occ_end + 1]
    occ_vis = visibs_i[occ_start : occ_end + 1].astype(bool)
    occ_valid = valids_i[occ_start : occ_end + 1].astype(bool)
    occ_in = frame_in_image(occ_xy, width=width, height=height)
    occ_not_vis = ~occ_vis

    denom = max(len(occ_xy), 1)
    offscreen_mask = occ_not_vis & ~occ_in
    inframe_occ_mask = occ_not_vis & occ_valid & occ_in

    offscreen_frac = float(offscreen_mask.sum() / denom)
    inframe_occ_frac = float(inframe_occ_mask.sum() / denom)

    if offscreen_frac >= 0.5:
        label = "offscreen_return"
    elif inframe_occ_frac >= 0.5:
        label = "in_frame_occlusion"
    else:
        label = "mixed"
    return label, offscreen_frac, inframe_occ_frac


def extract_prt_queries_for_sequence(
    seq_path: Path,
    split_name: str,
    min_occ_length: int,
    min_camera_motion: float,
) -> List[PRTQuery]:
    seq = load_sequence(seq_path)
    trajs_2d = seq["trajs_2d"]
    visibs = seq["visibs"]
    valids = seq["valids"] if "valids" in seq else np.ones_like(visibs, dtype=bool)
    extrinsics = seq["extrinsics"]

    height, width = load_image_size(seq_path)
    t_total, n_points = visibs.shape
    queries: List[PRTQuery] = []

    # Pre-compute inverse extrinsics for all frames (vectorized)
    extr_inv = np.linalg.inv(extrinsics)  # (T, 4, 4)

    for i in range(n_points):
        vis = visibs[:, i].astype(bool)
        val = valids[:, i].astype(bool)
        xy = trajs_2d[:, i]
        if not vis.any():
            continue

        run_start = -1
        last_vis_before = -1
        for t in range(t_total):
            if vis[t]:
                if run_start >= 0 and last_vis_before >= 0:
                    occ_length = t - run_start
                    if occ_length >= min_occ_length:
                        e_rel = extrinsics[t] @ extr_inv[last_vis_before]
                        cam_motion = float(np.linalg.norm(e_rel[:3, :3] - np.eye(3)))
                        cam_translation = float(np.linalg.norm(e_rel[:3, 3]))
                        if cam_motion >= min_camera_motion:
                            reentry_type, offscreen_frac, inframe_occ_frac = classify_reentry_type(
                                trajs_2d_i=xy,
                                visibs_i=vis,
                                valids_i=val,
                                width=width,
                                height=height,
                                occ_start=run_start,
                                occ_end=t - 1,
                            )
                            queries.append(
                                PRTQuery(
                                    seq_name=seq_path.name,
                                    split=split_name,
                                    point_idx=int(i),
                                    query_frame=int(last_vis_before),
                                    reentry_frame=int(t),
                                    occ_length=int(occ_length),
                                    camera_motion=cam_motion,
                                    camera_translation=cam_translation,
                                    reentry_type=reentry_type,
                                    offscreen_frac=offscreen_frac,
                                    inframe_occ_frac=inframe_occ_frac,
                                    image_width=int(width),
                                    image_height=int(height),
                                )
                            )
                last_vis_before = t
                run_start = -1
            else:
                if run_start < 0:
                    run_start = t

    return queries


def occ_band(occ_length: int) -> str:
    if occ_length >= 50:
        return "occ_50p"
    if occ_length >= 30:
        return "occ_30_49"
    if occ_length >= 20:
        return "occ_20_29"
    if occ_length >= 10:
        return "occ_10_19"
    return "occ_lt_10"


def camera_motion_band(camera_motion: float) -> str:
    if camera_motion >= 0.5:
        return "cam_0.50p"
    if camera_motion >= 0.3:
        return "cam_0.30_0.49"
    if camera_motion >= 0.1:
        return "cam_0.10_0.29"
    return "cam_lt_0.10"


def hardest_bucket_key(q: PRTQuery) -> str:
    occ_tag = "occ20p" if q.occ_length >= 20 else "occ10p"
    cam_tag = "cam0.30p" if q.camera_motion >= 0.3 else "cam0.10p" if q.camera_motion >= 0.1 else "cam0.00p"
    return f"{occ_tag}_{cam_tag}_{q.reentry_type}"


def summarize_queries(queries: Iterable[PRTQuery]) -> Dict[str, Any]:
    queries = list(queries)
    if not queries:
        return {
            "count": 0,
            "reentry_type_counts": {},
            "occ_bucket_counts": {},
            "cam_bucket_counts": {},
            "joint_bucket_counts": {},
        }

    reentry_counter = Counter(q.reentry_type for q in queries)
    occ_band_counter = Counter(occ_band(q.occ_length) for q in queries)
    cam_band_counter = Counter(camera_motion_band(q.camera_motion) for q in queries)
    joint_counter = Counter(
        f"{occ_band(q.occ_length)}__{camera_motion_band(q.camera_motion)}__{q.reentry_type}"
        for q in queries
    )
    occ_threshold_counts = {
        "occ_gte_10": int(sum(q.occ_length >= 10 for q in queries)),
        "occ_gte_20": int(sum(q.occ_length >= 20 for q in queries)),
        "occ_gte_30": int(sum(q.occ_length >= 30 for q in queries)),
        "occ_gte_50": int(sum(q.occ_length >= 50 for q in queries)),
    }
    cam_threshold_counts = {
        "cam_gte_0.10": int(sum(q.camera_motion >= 0.10 for q in queries)),
        "cam_gte_0.30": int(sum(q.camera_motion >= 0.30 for q in queries)),
        "cam_gte_0.50": int(sum(q.camera_motion >= 0.50 for q in queries)),
    }

    camera_motion_vals = np.array([q.camera_motion for q in queries], dtype=np.float32)
    occ_vals = np.array([q.occ_length for q in queries], dtype=np.float32)
    offscreen_vals = np.array([q.offscreen_frac for q in queries], dtype=np.float32)

    return {
        "count": len(queries),
        "reentry_type_counts": dict(sorted(reentry_counter.items())),
        "occ_band_counts": dict(sorted(occ_band_counter.items())),
        "cam_band_counts": dict(sorted(cam_band_counter.items())),
        "occ_threshold_counts": occ_threshold_counts,
        "cam_threshold_counts": cam_threshold_counts,
        "joint_bucket_counts": dict(sorted(joint_counter.items())),
        "occ_length_mean": float(occ_vals.mean()),
        "occ_length_median": float(np.median(occ_vals)),
        "camera_motion_mean": float(camera_motion_vals.mean()),
        "camera_motion_median": float(np.median(camera_motion_vals)),
        "offscreen_frac_mean": float(offscreen_vals.mean()),
    }


def build_preview(
    queries: List[PRTQuery],
    per_type: int = 10,
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[PRTQuery]] = defaultdict(list)
    for q in queries:
        grouped[q.reentry_type].append(q)

    preview: Dict[str, List[Dict[str, Any]]] = {}
    for reentry_type, items in grouped.items():
        items_sorted = sorted(
            items,
            key=lambda x: (x.occ_length, x.camera_motion),
            reverse=True,
        )
        preview[reentry_type] = [asdict(x) for x in items_sorted[:per_type]]
    return preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--min-occ-length", type=int, default=10)
    parser.add_argument("--min-camera-motion", type=float, default=0.0)
    parser.add_argument("--max-sequences-per-split", type=int, default=0)
    parser.add_argument("--stats-output", type=str, required=True)
    parser.add_argument("--preview-output", type=str, required=True)
    parser.add_argument("--preview-per-type", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    all_queries: List[PRTQuery] = []
    split_summaries: Dict[str, Any] = {}

    for split in splits:
        sequence_paths = discover_sequences(data_root, [split])
        if args.max_sequences_per_split > 0:
            sequence_paths = sequence_paths[: args.max_sequences_per_split]
        logger.info("Split %s: %d sequences", split, len(sequence_paths))

        split_queries: List[PRTQuery] = []
        for seq_path in sequence_paths:
            seq_queries = extract_prt_queries_for_sequence(
                seq_path=seq_path,
                split_name=split,
                min_occ_length=args.min_occ_length,
                min_camera_motion=args.min_camera_motion,
            )
            split_queries.extend(seq_queries)
        split_summaries[split] = summarize_queries(split_queries)
        all_queries.extend(split_queries)
        logger.info("Split %s: %d queries", split, len(split_queries))

    hardest_counter = Counter(hardest_bucket_key(q) for q in all_queries)

    stats = {
        "config": {
            "data_root": str(data_root),
            "splits": splits,
            "min_occ_length": int(args.min_occ_length),
            "min_camera_motion": float(args.min_camera_motion),
            "max_sequences_per_split": int(args.max_sequences_per_split),
        },
        "overall": summarize_queries(all_queries),
        "per_split": split_summaries,
        "hardest_bucket_counts": dict(sorted(hardest_counter.items())),
    }

    preview = {
        "config": stats["config"],
        "overall_count": len(all_queries),
        "preview": build_preview(all_queries, per_type=args.preview_per_type),
    }

    stats_output = Path(args.stats_output)
    preview_output = Path(args.preview_output)
    stats_output.parent.mkdir(parents=True, exist_ok=True)
    preview_output.parent.mkdir(parents=True, exist_ok=True)

    stats_output.write_text(json.dumps(stats, indent=2))
    preview_output.write_text(json.dumps(preview, indent=2))

    logger.info("Wrote stats to %s", stats_output)
    logger.info("Wrote preview to %s", preview_output)


if __name__ == "__main__":
    main()
