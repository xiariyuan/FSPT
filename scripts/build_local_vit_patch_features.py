#!/usr/bin/env python3
"""Build local ViT/DINO patch-embedding similarity features for V2.3 samples.

This diagnostic uses a project-local Hugging Face vision model directory, e.g.
third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m.

For each V1-proposed recovery frame sample it crops:
  - query frame at query point
  - last reliable base-visible frame before trigger_t at base coordinate
  - candidate frame at frame_t at base coordinate

Then it computes cosine/L2 similarities between normalized CLS embeddings.  No
GT is used for feature construction; labels are copied from the frame-keep NPZ.
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
from PIL import Image
from transformers import AutoModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.tapvid_rgb_stacking import TAPVidRGBStackingDataset  # noqa: E402
from utils.reentry_viscalibrator_features import npy  # noqa: E402


VIT_FEATURE_NAMES = [
    "query_candidate_cosine",
    "last_candidate_cosine",
    "best_ref_candidate_cosine",
    "worst_ref_candidate_cosine",
    "query_candidate_l2",
    "last_candidate_l2",
    "best_ref_candidate_l2",
    "worst_ref_candidate_l2",
    "query_last_cosine",
    "query_last_l2",
    "last_visible_age_norm",
    "has_last_visible_ref",
    "query_candidate_age_norm",
]


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
    arr = video.detach().cpu().numpy() if isinstance(video, torch.Tensor) else np.asarray(video)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D video, got {arr.shape}")
    if arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.dtype != np.uint8:
        if float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB video, got {arr.shape}")
    return arr


def load_natural_video(video_id: str, *, root: str, pkl_name: str) -> np.ndarray:
    idx = parse_rgb_index(video_id)
    ds = TAPVidRGBStackingDataset(root=root, pkl_name=pkl_name, start_index=idx, num_videos=1)
    return to_uint8_video(ds[0]["video"])


def load_stress_video_index(path: str | Path | None) -> Dict[str, Any]:
    if path is None or str(path) == "":
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {str(r["video_id"]): r for r in payload.get("records", [])}


def resolve_video(
    video_id: str,
    *,
    root: str,
    pkl_name: str,
    stress_index: Dict[str, Any],
    video_source: str,
) -> np.ndarray:
    vid = str(video_id)
    if video_source == "stress":
        if vid not in stress_index:
            raise KeyError(f"video_id {vid} not found in stress dataset")
        return to_uint8_video(stress_index[vid]["video"])
    if video_source == "auto" and vid in stress_index:
        return to_uint8_video(stress_index[vid]["video"])
    return load_natural_video(vid, root=root, pkl_name=pkl_name)


def norm_yx_to_pixel(yx: np.ndarray, h: int, w: int) -> Tuple[float, float]:
    return float(yx[0]) * max(h - 1, 1), float(yx[1]) * max(w - 1, 1)


def crop_square(img: np.ndarray, y: float, x: float, size: int) -> Image.Image:
    half = int(size) // 2
    cy = int(round(float(y)))
    cx = int(round(float(x)))
    h, w = img.shape[:2]
    y0, y1 = cy - half, cy + half + 1
    x0, x1 = cx - half, cx + half + 1
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    sy0, sy1 = max(0, y0), min(h, y1)
    sx0, sx1 = max(0, x0), min(w, x1)
    if sy1 > sy0 and sx1 > sx0:
        py0, px0 = sy0 - y0, sx0 - x0
        patch[py0 : py0 + (sy1 - sy0), px0 : px0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
    return Image.fromarray(patch, mode="RGB")


def manual_preprocess(images: List[Image.Image], image_size: int) -> torch.Tensor:
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arrs = []
    for img in images:
        img = img.convert("RGB").resize((int(image_size), int(image_size)), Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = (arr - mean[None, None, :]) / std[None, None, :]
        arr = np.transpose(arr, (2, 0, 1))
        arrs.append(arr)
    return torch.from_numpy(np.stack(arrs, axis=0)).float()


def choose_last_visible_frame(base_vis: np.ndarray, trigger_t: int) -> int | None:
    idx = np.where(base_vis[: max(0, int(trigger_t))])[0]
    return int(idx[-1]) if idx.size else None


def prepare_triplet(meta: Dict[str, Any], record: Dict[str, Any], video: np.ndarray, crop_size: int) -> Tuple[List[Image.Image], Dict[str, Any]]:
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
    imgs = [
        crop_square(video[query_t], q_py, q_px, crop_size),
        crop_square(video[last_t], last_py, last_px, crop_size),
        crop_square(video[frame_t], cand_py, cand_px, crop_size),
    ]
    info = {
        "video_id": str(meta["video_id"]),
        "query_idx": qi,
        "query_t": query_t,
        "last_visible_t": int(last_t),
        "frame_t": frame_t,
        "trigger_t": trigger_t,
        "has_last_visible_ref": bool(has_last),
        "last_visible_age_norm": float(frame_t - last_t) / 256.0,
        "query_candidate_age_norm": float(frame_t - query_t) / 256.0,
    }
    return imgs, info


def embed_images(images: List[Image.Image], model: torch.nn.Module, device: str, image_size: int, batch_size: int) -> np.ndarray:
    outs: List[np.ndarray] = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        pixel_values = manual_preprocess(batch, image_size=image_size).to(device)
        with torch.no_grad():
            out = model(pixel_values=pixel_values)
        emb = out.last_hidden_state[:, 0].float()
        emb = emb / torch.clamp(torch.linalg.vector_norm(emb, dim=1, keepdim=True), min=1e-8)
        outs.append(emb.detach().cpu().numpy().astype(np.float32))
        del pixel_values, out, emb
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 384), dtype=np.float32)


def feature_from_triplet(q: np.ndarray, last: np.ndarray, cand: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
    q_c = float(np.dot(q, cand))
    l_c = float(np.dot(last, cand))
    q_l = float(np.dot(q, last))
    q_l2 = float(np.linalg.norm(q - cand))
    l_l2 = float(np.linalg.norm(last - cand))
    qlast_l2 = float(np.linalg.norm(q - last))
    return np.asarray([
        q_c,
        l_c,
        max(q_c, l_c),
        min(q_c, l_c),
        q_l2,
        l_l2,
        min(q_l2, l_l2),
        max(q_l2, l_l2),
        q_l,
        qlast_l2,
        float(info["last_visible_age_norm"]),
        1.0 if info["has_last_visible_ref"] else 0.0,
        float(info["query_candidate_age_norm"]),
    ], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-keep-npz", required=True)
    ap.add_argument("--base-cache", required=True)
    ap.add_argument("--model-dir", default="third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m")
    ap.add_argument("--rgb-root", default="/gemini/code/datasets/tapvid_rgb_stacking")
    ap.add_argument("--rgb-pkl", default="tapvid_rgb_stacking.pkl")
    ap.add_argument("--stress-dataset", default="", help="Optional stress_dataset.pt for translate/occluder videos")
    ap.add_argument("--video-source", choices=["auto", "natural", "stress"], default="auto", help="Where to resolve video frames from")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--sample-mode", choices=["prefix", "uniform", "slice"], default="uniform")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--end-index", type=int, default=0)
    ap.add_argument("--crop-size", type=int, default=33)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--embed-batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    data = load_npz(args.frame_keep_npz)
    cache = load_cache(args.base_cache)
    records = {str(r["video_id"]): r for r in cache["records"]}
    stress_index = load_stress_video_index(args.stress_dataset)
    all_meta = [json.loads(str(x)) for x in data["meta_json"].tolist()]
    if args.sample_mode == "slice" or int(args.end_index) > 0:
        start = max(0, int(args.start_index))
        end = len(all_meta) if int(args.end_index) <= 0 else min(len(all_meta), int(args.end_index))
        if end < start:
            raise ValueError(f"end-index {end} < start-index {start}")
        indices = np.arange(start, end, dtype=np.int64)
    elif args.max_samples <= 0 or int(args.max_samples) >= len(all_meta):
        indices = np.arange(len(all_meta), dtype=np.int64)
    elif args.sample_mode == "prefix":
        indices = np.arange(int(args.max_samples), dtype=np.int64)
    else:
        indices = np.linspace(0, len(all_meta) - 1, int(args.max_samples), dtype=np.int64)
    meta = [all_meta[int(i)] for i in indices]

    model = AutoModel.from_pretrained(str(args.model_dir), local_files_only=True).to(args.device).eval()
    video_cache: Dict[str, np.ndarray] = {}
    triplet_images: List[Image.Image] = []
    infos: List[Dict[str, Any]] = []
    for i, m in enumerate(meta):
        vid = str(m["video_id"])
        if vid not in records:
            raise KeyError(f"video_id {vid} not found in base cache")
        if vid not in video_cache:
            video_cache[vid] = resolve_video(
                vid,
                root=args.rgb_root,
                pkl_name=args.rgb_pkl,
                stress_index=stress_index,
                video_source=args.video_source,
            )
        imgs, info = prepare_triplet(m, records[vid], video_cache[vid], int(args.crop_size))
        triplet_images.extend(imgs)
        info["source_meta"] = m
        infos.append(info)
        if (i + 1) % 5000 == 0:
            print({"prepared": i + 1, "total": len(meta), "videos_loaded": len(video_cache)}, flush=True)

    emb = embed_images(triplet_images, model, args.device, int(args.image_size), int(args.embed_batch_size))
    if emb.shape[0] != len(meta) * 3:
        raise RuntimeError(f"Embedding count mismatch: {emb.shape[0]} vs {len(meta)*3}")
    X_rows: List[np.ndarray] = []
    for i, info in enumerate(infos):
        q, last, cand = emb[3 * i], emb[3 * i + 1], emb[3 * i + 2]
        X_rows.append(feature_from_triplet(q, last, cand, info))
    X_vit = np.stack(X_rows, axis=0).astype(np.float32) if X_rows else np.zeros((0, len(VIT_FEATURE_NAMES)), dtype=np.float32)
    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X_patch=X_vit,
        patch_feature_names=np.asarray(VIT_FEATURE_NAMES, dtype=object),
        y_gt_visible=np.asarray(data["y_gt_visible"])[indices].astype(np.float32),
        y_safe16=np.asarray(data["y_safe16"])[indices].astype(np.float32),
        y_safe8=np.asarray(data["y_safe8"])[indices].astype(np.float32),
        y_utility=np.asarray(data["y_utility"])[indices].astype(np.float32),
        y_safe4=np.asarray(data["y_safe4"])[indices].astype(np.float32),
        numeric_X=np.asarray(data["X"])[indices].astype(np.float32),
        numeric_feature_names=np.asarray(data["feature_names"], dtype=object),
        meta_json=np.asarray([json.dumps(x, ensure_ascii=False) for x in infos], dtype=object),
        source_frame_keep_npz=np.asarray(str(args.frame_keep_npz), dtype=object),
        source_base_cache=np.asarray(str(args.base_cache), dtype=object),
        stress_dataset=np.asarray(str(args.stress_dataset), dtype=object),
        video_source=np.asarray(str(args.video_source), dtype=object),
        model_dir=np.asarray(str(args.model_dir), dtype=object),
        crop_size=np.asarray(int(args.crop_size), dtype=np.int32),
        image_size=np.asarray(int(args.image_size), dtype=np.int32),
        sample_mode=np.asarray(str(args.sample_mode), dtype=object),
        sample_indices=indices.astype(np.int64),
    )
    print(json.dumps({
        "out_npz": str(out),
        "n_samples": int(X_vit.shape[0]),
        "index_start": int(indices[0]) if len(indices) else None,
        "index_end_exclusive": int(indices[-1] + 1) if len(indices) else None,
        "feature_dim": int(X_vit.shape[1]),
        "videos_loaded": len(video_cache),
        "model_dir": str(args.model_dir),
        "stress_dataset": str(args.stress_dataset),
        "video_source": str(args.video_source),
        "crop_size": int(args.crop_size),
        "image_size": int(args.image_size),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
