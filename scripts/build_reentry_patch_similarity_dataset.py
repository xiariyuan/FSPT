#!/usr/bin/env python3
"""Build lightweight RGB patch-similarity features for V2.3 proposed recovery frames.

This diagnostic builder consumes an existing V2.3 frame-keep NPZ and the matching
base/offline cache.  For each V1-proposed recovery frame sample, it extracts:

  - query-frame patch at query point
  - last reliable base-visible patch before trigger_t
  - candidate patch at base coordinate at frame_t

Then it computes simple RGB similarity features.  No GT is used for feature
construction; GT labels are copied from the input NPZ for analysis only.
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


PATCH_FEATURE_NAMES: List[str] = []
for size in [9, 17, 33]:
    for ref_name in ["query", "last_visible"]:
        prefix = f"{ref_name}_base_s{size}"
        PATCH_FEATURE_NAMES.extend([
            f"{prefix}_valid_min",
            f"{prefix}_mad",
            f"{prefix}_mse",
            f"{prefix}_cosine",
            f"{prefix}_ncc",
            f"{prefix}_mean_color_l2",
            f"{prefix}_std_color_l2",
            f"{prefix}_grad_mad",
        ])
PATCH_FEATURE_NAMES.extend([
    "last_visible_age_norm",
    "has_last_visible_ref",
    "query_candidate_age_norm",
])


def load_npz(path: str | Path) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def load_cache(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def parse_rgb_index(video_id: str) -> int:
    m = re.search(r"rgb_stacking_(\d{6})", str(video_id))
    if not m:
        raise ValueError(f"Cannot parse RGB stacking index from video_id={video_id}")
    return int(m.group(1))


def to_uint8_video(video: Any) -> np.ndarray:
    if isinstance(video, torch.Tensor):
        arr = video.detach().cpu().numpy()
    else:
        arr = np.asarray(video)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D video, got shape {arr.shape}")
    if arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.dtype != np.uint8:
        if float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB video, got shape {arr.shape}")
    return arr


def load_natural_video(video_id: str, *, root: str, pkl_name: str) -> np.ndarray:
    idx = parse_rgb_index(video_id)
    ds = TAPVidRGBStackingDataset(root=root, pkl_name=pkl_name, start_index=idx, num_videos=1)
    sample = ds[0]
    return to_uint8_video(sample["video"])


def norm_yx_to_pixel(yx: np.ndarray, h: int, w: int) -> Tuple[float, float]:
    return float(yx[0]) * max(h - 1, 1), float(yx[1]) * max(w - 1, 1)


def extract_patch(img: np.ndarray, y: float, x: float, size: int) -> Tuple[np.ndarray, float]:
    half = int(size) // 2
    cy = int(round(float(y)))
    cx = int(round(float(x)))
    h, w = img.shape[:2]
    y0, y1 = cy - half, cy + half + 1
    x0, x1 = cx - half, cx + half + 1
    patch = np.zeros((size, size, 3), dtype=np.float32)
    sy0, sy1 = max(0, y0), min(h, y1)
    sx0, sx1 = max(0, x0), min(w, x1)
    if sy1 <= sy0 or sx1 <= sx0:
        return patch, 0.0
    py0, px0 = sy0 - y0, sx0 - x0
    patch[py0 : py0 + (sy1 - sy0), px0 : px0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0
    valid = float((sy1 - sy0) * (sx1 - sx0)) / float(size * size)
    return patch, valid


def grad_mag(patch: np.ndarray) -> np.ndarray:
    gray = patch.mean(axis=-1)
    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)
    gy[1:] = gray[1:] - gray[:-1]
    gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    return np.sqrt(gx * gx + gy * gy)


def patch_pair_features(a: np.ndarray, va: float, b: np.ndarray, vb: float) -> List[float]:
    af = a.reshape(-1, 3).astype(np.float32)
    bf = b.reshape(-1, 3).astype(np.float32)
    diff = af - bf
    mad = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    avec = af.reshape(-1)
    bvec = bf.reshape(-1)
    an = avec - float(avec.mean())
    bn = bvec - float(bvec.mean())
    cosine = float(np.dot(avec, bvec) / max(float(np.linalg.norm(avec) * np.linalg.norm(bvec)), 1e-8))
    ncc = float(np.dot(an, bn) / max(float(np.linalg.norm(an) * np.linalg.norm(bn)), 1e-8))
    mean_l2 = float(np.linalg.norm(af.mean(axis=0) - bf.mean(axis=0)))
    std_l2 = float(np.linalg.norm(af.std(axis=0) - bf.std(axis=0)))
    gmad = float(np.mean(np.abs(grad_mag(a) - grad_mag(b))))
    return [float(min(va, vb)), mad, mse, cosine, ncc, mean_l2, std_l2, gmad]


def choose_last_visible_frame(base_vis: np.ndarray, trigger_t: int) -> int | None:
    idx = np.where(base_vis[: max(0, int(trigger_t))])[0]
    if idx.size == 0:
        return None
    return int(idx[-1])


def build_features_for_sample(meta: Dict[str, Any], record: Dict[str, Any], video: np.ndarray, patch_sizes: List[int]) -> Tuple[np.ndarray, Dict[str, Any]]:
    qi = int(meta["query_idx"])
    frame_t = int(meta["frame_t"])
    trigger_t = int(meta["trigger_t"])
    qpts = npy(record["query_points"], np.float32)
    pred_tracks = npy(record["pred_tracks"], np.float32)
    base_vis = npy(record["pred_visibility"], bool)[qi]
    h, w = int(video.shape[1]), int(video.shape[2])
    t_len = int(video.shape[0])
    frame_t = max(0, min(t_len - 1, frame_t))
    query_t = int(round(float(qpts[qi, 0])))
    query_t = max(0, min(t_len - 1, query_t))
    last_t = choose_last_visible_frame(base_vis, trigger_t)
    has_last = last_t is not None
    if last_t is None:
        last_t = query_t
    last_t = max(0, min(t_len - 1, int(last_t)))

    q_yx = np.asarray([qpts[qi, 1], qpts[qi, 2]], dtype=np.float32)
    last_yx = pred_tracks[qi, last_t]
    cand_yx = pred_tracks[qi, frame_t]
    q_py, q_px = norm_yx_to_pixel(q_yx, h, w)
    last_py, last_px = norm_yx_to_pixel(last_yx, h, w)
    cand_py, cand_px = norm_yx_to_pixel(cand_yx, h, w)

    feats: List[float] = []
    for size in patch_sizes:
        q_patch, q_valid = extract_patch(video[query_t], q_py, q_px, size)
        last_patch, last_valid = extract_patch(video[last_t], last_py, last_px, size)
        cand_patch, cand_valid = extract_patch(video[frame_t], cand_py, cand_px, size)
        feats.extend(patch_pair_features(q_patch, q_valid, cand_patch, cand_valid))
        feats.extend(patch_pair_features(last_patch, last_valid, cand_patch, cand_valid))
    feats.extend([
        float(frame_t - last_t) / 256.0,
        1.0 if has_last else 0.0,
        float(frame_t - query_t) / 256.0,
    ])
    info = {
        "video_id": str(meta["video_id"]),
        "query_idx": qi,
        "frame_t": frame_t,
        "trigger_t": trigger_t,
        "query_t": query_t,
        "last_visible_t": int(last_t),
        "has_last_visible_ref": bool(has_last),
        "candidate_yx_norm": [float(cand_yx[0]), float(cand_yx[1])],
    }
    return np.asarray(feats, dtype=np.float32), info


def main() -> None:
    ap = argparse.ArgumentParser(description="Build RGB patch-similarity features for frame-keep samples")
    ap.add_argument("--frame-keep-npz", required=True)
    ap.add_argument("--base-cache", required=True)
    ap.add_argument("--rgb-root", default="/gemini/code/datasets/tapvid_rgb_stacking")
    ap.add_argument("--rgb-pkl", default="tapvid_rgb_stacking.pkl")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--patch-sizes", type=int, nargs="*", default=[9, 17, 33])
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    data = load_npz(args.frame_keep_npz)
    cache = load_cache(args.base_cache)
    records = {str(r["video_id"]): r for r in cache["records"]}
    meta = [json.loads(str(x)) for x in data["meta_json"].tolist()]
    if args.max_samples <= 0 or int(args.max_samples) >= len(meta):
        n = len(meta)
        meta = meta[:n]
    else:
        n = int(args.max_samples)
        # Uniformly sample across the whole NPZ rather than taking the prefix; the
        # prefix can be dominated by one video and gives misleading AUC estimates.
        idx = np.linspace(0, len(meta) - 1, n, dtype=np.int64)
        meta = [meta[int(i)] for i in idx]

    video_cache: Dict[str, np.ndarray] = {}
    X_rows: List[np.ndarray] = []
    out_meta: List[str] = []
    for i, m in enumerate(meta):
        vid = str(m["video_id"])
        if vid not in records:
            raise KeyError(f"video_id {vid} not found in base cache")
        if vid not in video_cache:
            video_cache[vid] = load_natural_video(vid, root=args.rgb_root, pkl_name=args.rgb_pkl)
        feat, info = build_features_for_sample(m, records[vid], video_cache[vid], [int(x) for x in args.patch_sizes])
        X_rows.append(feat)
        info.update({"source_meta": m})
        out_meta.append(json.dumps(info, ensure_ascii=False))
        if (i + 1) % 10000 == 0:
            print({"processed": i + 1, "total": n, "videos_loaded": len(video_cache)}, flush=True)

    X_patch = np.stack(X_rows, axis=0).astype(np.float32) if X_rows else np.zeros((0, len(PATCH_FEATURE_NAMES)), dtype=np.float32)
    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X_patch=X_patch,
        patch_feature_names=np.asarray(PATCH_FEATURE_NAMES, dtype=object),
        y_gt_visible=np.asarray(data["y_gt_visible"][:n], dtype=np.float32),
        y_safe16=np.asarray(data["y_safe16"][:n], dtype=np.float32),
        y_safe8=np.asarray(data["y_safe8"][:n], dtype=np.float32),
        y_utility=np.asarray(data["y_utility"][:n], dtype=np.float32),
        y_safe4=np.asarray(data["y_safe4"][:n], dtype=np.float32),
        numeric_X=np.asarray(data["X"][:n], dtype=np.float32),
        numeric_feature_names=np.asarray(data["feature_names"], dtype=object),
        meta_json=np.asarray(out_meta, dtype=object),
        source_frame_keep_npz=np.asarray(str(args.frame_keep_npz), dtype=object),
        source_base_cache=np.asarray(str(args.base_cache), dtype=object),
    )
    print(json.dumps({
        "out_npz": str(out),
        "n_samples": int(X_patch.shape[0]),
        "patch_feature_dim": int(X_patch.shape[1]),
        "videos_loaded": len(video_cache),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
