#!/usr/bin/env python3
"""Build DINOScore decisions for a target cache without full 3-crop feature NPZ.

This standalone version intentionally avoids importing project-local modules to
reduce runtime I/O fragility. It only relies on numpy/torch/PIL/transformers.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel


_RGB_PICKLE_CACHE: Dict[Tuple[str, str], Any] = {}


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


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
    arr = npy(video)
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
    key = (str(root), str(pkl_name))
    if key not in _RGB_PICKLE_CACHE:
        pkl_path = Path(root) / pkl_name
        last_exc = None
        for attempt in range(1, 6):
            try:
                with pkl_path.open("rb") as f:
                    _RGB_PICKLE_CACHE[key] = pickle.load(f)
                break
            except OSError as exc:
                last_exc = exc
                print({"rgb_pickle_load_retry": attempt, "path": str(pkl_path), "error": repr(exc)}, flush=True)
                time.sleep(float(attempt))
        if key not in _RGB_PICKLE_CACHE:
            raise RuntimeError(f"Failed to load RGB pickle after retries: {pkl_path}") from last_exc
    data = _RGB_PICKLE_CACHE[key]
    if idx < 0 or idx >= len(data):
        raise IndexError(f"RGB stacking index {idx} out of range for {len(data)} videos")
    return to_uint8_video(data[idx]["video"])


def load_stress_index(path: str | Path | None) -> Dict[str, Any]:
    if path is None or str(path) == "":
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {str(r["video_id"]): r for r in payload.get("records", [])}


def resolve_video(video_id: str, *, root: str, pkl_name: str, stress_index: Dict[str, Any], video_source: str) -> np.ndarray:
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
        arrs.append(np.transpose(arr, (2, 0, 1)))
    return torch.from_numpy(np.stack(arrs, axis=0)).float()


def choose_last_visible_frame(base_vis: np.ndarray, trigger_t: int) -> int | None:
    idx = np.where(base_vis[: max(0, int(trigger_t))])[0]
    return int(idx[-1]) if idx.size else None


def prepare_pair(meta: Dict[str, Any], base_record: Dict[str, Any], video: np.ndarray, crop_size: int) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
    qi = int(meta["query_idx"])
    frame_t = int(meta["frame_t"])
    trigger_t = int(meta["trigger_t"])
    qpts = npy(base_record["query_points"], np.float32)
    pred_tracks = npy(base_record["pred_tracks"], np.float32)
    base_vis = npy(base_record["pred_visibility"], bool)[qi]
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
    last_yx = pred_tracks[qi, last_t]
    cand_yx = pred_tracks[qi, frame_t]
    last_py, last_px = norm_yx_to_pixel(last_yx, h, w)
    cand_py, cand_px = norm_yx_to_pixel(cand_yx, h, w)
    ref_img = crop_square(video[last_t], last_py, last_px, crop_size)
    cand_img = crop_square(video[frame_t], cand_py, cand_px, crop_size)
    info = {
        "video_id": str(meta["video_id"]),
        "query_idx": qi,
        "frame_t": frame_t,
        "trigger_t": trigger_t,
        "query_t": query_t,
        "last_visible_t": int(last_t),
        "has_last_visible_ref": bool(has_last),
    }
    return ref_img, cand_img, info


def embed(images: List[Image.Image], model: torch.nn.Module, device: str, image_size: int, batch_size: int) -> np.ndarray:
    out_chunks: List[np.ndarray] = []
    for i in range(0, len(images), batch_size):
        pixel_values = manual_preprocess(images[i : i + batch_size], image_size).to(device)
        with torch.no_grad():
            out = model(pixel_values=pixel_values)
        emb = out.last_hidden_state[:, 0].float()
        emb = emb / torch.clamp(torch.linalg.vector_norm(emb, dim=1, keepdim=True), min=1e-8)
        out_chunks.append(emb.detach().cpu().numpy().astype(np.float32))
        del pixel_values, out, emb
    return np.concatenate(out_chunks, axis=0) if out_chunks else np.zeros((0, 384), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-keep-npz", required=True)
    ap.add_argument("--base-cache", required=True)
    ap.add_argument("--target-cache", required=True)
    ap.add_argument("--model-dir", default="third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m")
    ap.add_argument("--rgb-root", default="/gemini/code/datasets/tapvid_rgb_stacking")
    ap.add_argument("--rgb-pkl", default="tapvid_rgb_stacking.pkl")
    ap.add_argument("--stress-dataset", default="")
    ap.add_argument("--video-source", choices=["auto", "natural", "stress"], default="auto")
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--end-index", type=int, default=0)
    ap.add_argument("--crop-size", type=int, default=33)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--embed-batch-size", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=0.395)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    data = load_npz(args.frame_keep_npz)
    base = load_cache(args.base_cache)
    target = load_cache(args.target_cache)
    base_records = {str(r["video_id"]): r for r in base["records"]}
    target_records = {str(r["video_id"]): r for r in target["records"]}
    stress_index = load_stress_index(args.stress_dataset)
    all_meta = [json.loads(str(x)) for x in data["meta_json"].tolist()]
    start = max(0, int(args.start_index))
    end = len(all_meta) if int(args.end_index) <= 0 else min(len(all_meta), int(args.end_index))
    row_indices = np.arange(start, end, dtype=np.int64)

    selected_rows: List[int] = []
    images: List[Image.Image] = []
    infos: List[str] = []
    video_cache: Dict[str, np.ndarray] = {}
    missing = 0
    already_invisible = 0
    for row_idx in row_indices:
        m = all_meta[int(row_idx)]
        vid = str(m["video_id"])
        qi = int(m["query_idx"])
        t = int(m["frame_t"])
        if vid not in base_records or vid not in target_records:
            missing += 1
            continue
        target_vis = npy(target_records[vid]["pred_visibility"], bool)
        if qi < 0 or qi >= target_vis.shape[0] or t < 0 or t >= target_vis.shape[1] or not bool(target_vis[qi, t]):
            already_invisible += 1
            continue
        if vid not in video_cache:
            video_cache[vid] = resolve_video(vid, root=args.rgb_root, pkl_name=args.rgb_pkl, stress_index=stress_index, video_source=args.video_source)
        ref_img, cand_img, info = prepare_pair(m, base_records[vid], video_cache[vid], int(args.crop_size))
        images.extend([ref_img, cand_img])
        selected_rows.append(int(row_idx))
        infos.append(json.dumps(info, ensure_ascii=False))
        if len(selected_rows) % 5000 == 0:
            print({"selected_visible": len(selected_rows), "scanned": int(row_idx) - start + 1, "videos_loaded": len(video_cache)}, flush=True)

    model = AutoModel.from_pretrained(str(args.model_dir), local_files_only=True).to(args.device).eval()
    emb = embed(images, model, args.device, int(args.image_size), int(args.embed_batch_size))
    if emb.shape[0] != len(selected_rows) * 2:
        raise RuntimeError(f"embedding count mismatch {emb.shape[0]} vs {len(selected_rows)*2}")
    scores = np.asarray([float(np.dot(emb[2 * i], emb[2 * i + 1])) for i in range(len(selected_rows))], dtype=np.float32)
    fire_np = scores < float(args.threshold)

    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        row_indices=np.asarray(selected_rows, dtype=np.int64),
        scores=scores,
        fires=fire_np.astype(bool),
        info_json=np.asarray(infos, dtype=object),
        threshold=np.asarray(float(args.threshold), dtype=np.float32),
        start_index=np.asarray(start, dtype=np.int64),
        end_index=np.asarray(end, dtype=np.int64),
        scanned_rows=np.asarray(len(row_indices), dtype=np.int64),
        selected_visible_rows=np.asarray(len(selected_rows), dtype=np.int64),
        already_invisible_rows=np.asarray(already_invisible, dtype=np.int64),
        missing_rows=np.asarray(missing, dtype=np.int64),
        frame_keep_npz=np.asarray(str(args.frame_keep_npz), dtype=object),
        base_cache=np.asarray(str(args.base_cache), dtype=object),
        target_cache=np.asarray(str(args.target_cache), dtype=object),
        stress_dataset=np.asarray(str(args.stress_dataset), dtype=object),
        video_source=np.asarray(str(args.video_source), dtype=object),
    )
    print(json.dumps({
        "out_npz": str(out),
        "start": start,
        "end": end,
        "scanned_rows": int(len(row_indices)),
        "selected_visible_rows": int(len(selected_rows)),
        "fires": int(fire_np.sum()),
        "already_invisible_rows": int(already_invisible),
        "missing_rows": int(missing),
        "videos_loaded": int(len(video_cache)),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
