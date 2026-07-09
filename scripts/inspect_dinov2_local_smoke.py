#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, AutoModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="baselines/hf/facebook_dinov2_small")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-json", default="outputs/paper_discovery_2026-06-27/reentry_appearance/dinov2_local_smoke.json")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(str(model_dir))

    processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(args.device).eval()

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8), mode="RGB")
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(args.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    last = out.last_hidden_state.detach().cpu()
    pooled = last[:, 0].float()

    payload = {
        "model_dir": str(model_dir),
        "device": args.device,
        "processor_class": processor.__class__.__name__,
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
