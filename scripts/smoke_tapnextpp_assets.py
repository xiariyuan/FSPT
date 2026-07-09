#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path("/gemini/code/FSPT")
REPO = ROOT / "external/tapnextpp/repo"
CKPT = ROOT / "checkpoints/tapnextpp/tapnextpp_ckpt.pt"
OUT = ROOT / "outputs/paper_discovery_2026-07-05/tapnextpp_smoke"
OUT.mkdir(parents=True, exist_ok=True)

summary = {
    "repo": str(REPO),
    "checkpoint": str(CKPT),
    "repo_exists": REPO.exists(),
    "checkpoint_exists": CKPT.exists(),
    "checkpoint_size_bytes": CKPT.stat().st_size if CKPT.exists() else None,
    "python": sys.executable,
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "repo_inventory": {},
    "checkpoint_probe": {},
    "import_probe": {},
}

# 1. Inventory
try:
    files = []
    for p in REPO.rglob("*"):
        if p.is_file():
            rel = p.relative_to(REPO).as_posix()
            if any(x in rel.lower() for x in [
                "tapnext", "tap_next", "tapnet", "model", "checkpoint",
                "inference", "demo", "eval", "config", "readme"
            ]):
                files.append(rel)
    summary["repo_inventory"]["num_relevant_files"] = len(files)
    summary["repo_inventory"]["relevant_files_first_200"] = files[:200]

    top_dirs = sorted([p.name for p in REPO.iterdir() if p.is_dir()]) if REPO.exists() else []
    summary["repo_inventory"]["top_dirs"] = top_dirs
except Exception as e:
    summary["repo_inventory"]["error"] = repr(e)
    summary["repo_inventory"]["traceback"] = traceback.format_exc()

# 2. Checkpoint load smoke
try:
    ckpt = torch.load(CKPT, map_location="cpu")
    summary["checkpoint_probe"]["type"] = type(ckpt).__name__

    if isinstance(ckpt, dict):
        summary["checkpoint_probe"]["num_top_entries"] = len(ckpt)
        summary["checkpoint_probe"]["top_keys_first_80"] = list(ckpt.keys())[:80]

        tensor_count = 0
        tensor_numel = 0
        tensor_examples = []

        def walk(prefix, obj):
            nonlocal_vars = None
            global tensor_count, tensor_numel
            if torch.is_tensor(obj):
                tensor_examples.append({
                    "key": prefix,
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "numel": int(obj.numel()),
                })
                return int(obj.numel()), 1
            if isinstance(obj, dict):
                total, count = 0, 0
                for k, v in obj.items():
                    t, c = walk(f"{prefix}.{k}" if prefix else str(k), v)
                    total += t
                    count += c
                return total, count
            if isinstance(obj, (list, tuple)):
                total, count = 0, 0
                for i, v in enumerate(obj):
                    t, c = walk(f"{prefix}[{i}]", v)
                    total += t
                    count += c
                return total, count
            return 0, 0

        tensor_numel, tensor_count = walk("", ckpt)
        summary["checkpoint_probe"]["tensor_count"] = tensor_count
        summary["checkpoint_probe"]["tensor_numel"] = tensor_numel
        summary["checkpoint_probe"]["tensor_examples_first_80"] = tensor_examples[:80]
    else:
        summary["checkpoint_probe"]["repr"] = repr(ckpt)[:2000]

    summary["checkpoint_probe"]["load_ok"] = True
except Exception as e:
    summary["checkpoint_probe"]["load_ok"] = False
    summary["checkpoint_probe"]["error"] = repr(e)
    summary["checkpoint_probe"]["traceback"] = traceback.format_exc()

# 3. Import smoke
try:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "tapnet"))

    candidates = [
        "tapnet.tapnextpp",
        "tapnextpp",
        "tapnet.tapnextpp.models",
        "tapnet.tapnextpp.model",
        "tapnet.tapnextpp.inference",
        "tapnet.tapnextpp.demo",
    ]

    import_results = []
    for name in candidates:
        try:
            __import__(name)
            import_results.append({"module": name, "ok": True})
        except Exception as e:
            import_results.append({
                "module": name,
                "ok": False,
                "error": repr(e),
            })

    summary["import_probe"]["candidates"] = import_results
    summary["import_probe"]["any_import_ok"] = any(x["ok"] for x in import_results)
except Exception as e:
    summary["import_probe"]["error"] = repr(e)
    summary["import_probe"]["traceback"] = traceback.format_exc()

# 4. Write summary
out_json = OUT / "tapnextpp_asset_smoke_summary.json"
out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"\nWROTE {out_json}")
