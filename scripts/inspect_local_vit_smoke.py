#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel


def manual_preprocess(img: Image.Image, size: int = 224) -> torch.Tensor:
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean[None, None, :]) / std[None, None, :]
    arr = np.transpose(arr, (2, 0, 1))[None]
    return torch.from_numpy(arr).float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="baselines/hf/facebook_dinov2_small")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-json", default="outputs/paper_discovery_2026-06-27/reentry_appearance/local_vit_smoke.json")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(str(model_dir))

    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(args.device).eval()
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8), mode="RGB")
    pixel_values = manual_preprocess(img, size=int(args.image_size)).to(args.device)
    with torch.no_grad():
        out = model(pixel_values=pixel_values)
    last = out.last_hidden_state.detach().cpu()
    pooled = last[:, 0].float()
    payload = {
        "model_dir": str(model_dir),
        "device": args.device,
        "manual_preprocess": True,
        "image_size": int(args.image_size),
        "model_class": model.__class__.__name__,
        "last_hidden_state_shape": list(last.shape),
        "cls_embedding_shape": list(pooled.shape),
        "cls_norm": float(torch.linalg.vector_norm(pooled, dim=1)[0]),
        "dtype": str(last.dtype),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
