#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

PACKAGES = ["torch", "torchvision", "timm", "transformers", "PIL", "cv2", "sklearn", "numpy", "scipy"]
KEYWORDS = ["dino", "dinov2", "dinov3", "vit", "vision_transformer", "clip", "swin", "tapir", "tapnext", "tapnet", "locotrack", "trackon", "track_on", "track-on", "cotracker"]
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".pkl", ".npz", ".onnx"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", "wandb"}


def inspect_package(name: str) -> Dict[str, Any]:
    out = {"name": name, "available": False, "version": None, "file": None, "error": None}
    try:
        m = importlib.import_module(name)
        out["available"] = True
        out["version"] = getattr(m, "__version__", None)
        out["file"] = getattr(m, "__file__", None)
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def torch_details() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        import torch
        out["torch_version"] = getattr(torch, "__version__", None)
        out["cuda_available"] = bool(torch.cuda.is_available())
        out["cuda_version"] = getattr(torch.version, "cuda", None)
        out["device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            out["device_name_0"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def iter_project_files(roots: Iterable[Path], max_files_per_root: int = 25000, max_depth: int = 8):
    for root in roots:
        if not root.exists():
            continue
        root = root.resolve()
        count = 0
        for dirpath, dirnames, filenames in os.walk(root):
            p = Path(dirpath)
            rel = p.relative_to(root).parts if p != root else ()
            if len(rel) > max_depth:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                count += 1
                if count > max_files_per_root:
                    break
                yield p / fn
            if count > max_files_per_root:
                break


def hits(path_or_text: str) -> List[str]:
    s = path_or_text.lower()
    return [k for k in KEYWORDS if k.lower() in s]


def size_mb(path: Path):
    try:
        return round(path.stat().st_size / 1024 / 1024, 3)
    except Exception:
        return None


def find_checkpoints(project_root: Path) -> List[Dict[str, Any]]:
    roots = [project_root / "baselines", project_root / "checkpoints", project_root / "models", project_root / "outputs"]
    rows = []
    for path in iter_project_files(roots):
        if path.suffix.lower() not in CHECKPOINT_SUFFIXES:
            continue
        h = hits(str(path))
        if h or path.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}:
            rows.append({"path": str(path), "suffix": path.suffix.lower(), "size_mb": size_mb(path), "keyword_hits": h})
    rows.sort(key=lambda r: (len(r["keyword_hits"]), r["size_mb"] or 0.0), reverse=True)
    return rows[:200]


def find_code_refs(project_root: Path) -> List[Dict[str, Any]]:
    roots = [project_root / "baselines", project_root / "models", project_root / "scripts", project_root / "datasets", project_root / "utils"]
    pattern = re.compile("|".join(re.escape(k) for k in KEYWORDS), re.IGNORECASE)
    rows = []
    for path in iter_project_files(roots):
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        path_hits = hits(str(path))
        line_hits = []
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if pattern.search(line):
                        line_hits.append({"line": i, "text": line.strip()[:220]})
                    if len(line_hits) >= 5:
                        break
        except Exception:
            continue
        if path_hits or line_hits:
            rows.append({"path": str(path), "keyword_hits_in_path": path_hits, "line_hits": line_hits})
        if len(rows) >= 300:
            break
    return rows


def recommend(packages: Dict[str, Dict[str, Any]], checkpoints: List[Dict[str, Any]], refs: List[Dict[str, Any]]) -> Dict[str, Any]:
    available = {k: bool(v.get("available")) for k, v in packages.items()}
    ckpt_text = "\n".join(r["path"].lower() for r in checkpoints)
    ref_text = "\n".join(r["path"].lower() for r in refs)
    all_text = ckpt_text + "\n" + ref_text
    has_dino_ckpt = any(k in ckpt_text for k in ["dino", "dinov2", "dinov3"])
    has_vit_tooling = bool(available.get("timm") or available.get("transformers") or available.get("torchvision"))
    has_teacher = any(k in all_text for k in ["tapir", "tapnext", "tapnet", "locotrack", "trackon", "track_on", "track-on"])
    has_cotracker = "cotracker" in all_text
    if has_dino_ckpt and has_vit_tooling:
        route = "dino_patch_features"
        reason = "Project contains DINO-like checkpoint references and ViT tooling is available."
    elif has_teacher:
        route = "external_tracker_teacher_agreement"
        reason = "Project contains external teacher/tracker references beyond CoTracker."
    elif has_vit_tooling:
        route = "vit_patch_features_possible_but_checkpoint_needed"
        reason = "ViT tooling is available, but no project-local DINO checkpoint was detected."
    elif has_cotracker:
        route = "cotracker_family_ensemble"
        reason = "Only CoTracker-family resources are clearly available in the project."
    else:
        route = "external_resource_needed"
        reason = "No strong project-local appearance or teacher route was detected."
    return {"best_next_route": route, "reason": reason, "signals": {"has_dino_checkpoint": has_dino_ckpt, "has_vit_tooling": has_vit_tooling, "has_external_teacher_reference": has_teacher, "has_cotracker_reference": has_cotracker}}


def md_table(rows, cols, max_rows=None):
    if max_rows is not None:
        rows = rows[:max_rows]
    lines = ["| " + " | ".join(t for t, _ in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        vals = []
        for _, key in cols:
            v = r.get(key, "")
            if isinstance(v, list):
                v = ", ".join(map(str, v))
            vals.append(str(v).replace("\n", " ")[:240])
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_md(path: Path, payload: Dict[str, Any]) -> None:
    rec = payload["recommendation"]
    lines = ["# ReEntry Appearance / Teacher Availability Inspection", "", "## Recommendation", "", f"Best next route: `{rec['best_next_route']}`", "", rec["reason"], "", "Signals:"]
    for k, v in rec["signals"].items():
        lines.append(f"- `{k}`: `{v}`")
    lines += ["", "## Environment", "", f"Python: `{payload['environment']['python']}`", f"Platform: `{payload['environment']['platform']}`", "", "Torch details:", "", "```json", json.dumps(payload["torch_details"], indent=2, ensure_ascii=False), "```", ""]
    lines += ["## Python packages", "", md_table(list(payload["packages"].values()), [("package", "name"), ("available", "available"), ("version", "version"), ("file", "file"), ("error", "error")]), ""]
    lines += ["## Top checkpoint candidates", "", md_table(payload["checkpoint_candidates"], [("path", "path"), ("suffix", "suffix"), ("size MB", "size_mb"), ("hits", "keyword_hits")], max_rows=80), ""]
    lines += ["## Code references", ""]
    for ref in payload["code_references"][:100]:
        lines.append(f"### `{ref['path']}`")
        if ref["keyword_hits_in_path"]:
            lines.append(f"Path hits: `{', '.join(ref['keyword_hits_in_path'])}`")
        for h in ref["line_hits"][:5]:
            lines.append(f"- L{h['line']}: `{h['text']}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--out-json", default="outputs/paper_discovery_2026-06-27/reentry_appearance/teacher_availability.json")
    ap.add_argument("--out-md", default="docs/reentry_appearance_teacher_availability_2026-07-03.md")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    packages = {n: inspect_package(n) for n in PACKAGES}
    checkpoints = find_checkpoints(root)
    refs = find_code_refs(root)
    rec = recommend(packages, checkpoints, refs)
    payload = {"project_root": str(root), "environment": {"python": sys.version, "platform": platform.platform(), "cwd": os.getcwd()}, "torch_details": torch_details(), "packages": packages, "checkpoint_candidates": checkpoints, "code_references": refs, "recommendation": rec}
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md = Path(args.out_md)
    write_md(out_md, payload)
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), "best_next_route": rec["best_next_route"], "reason": rec["reason"], "n_checkpoint_candidates": len(checkpoints), "n_code_references": len(refs), "packages_available": {k: v["available"] for k, v in packages.items()}}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
