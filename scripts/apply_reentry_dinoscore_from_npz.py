#!/usr/bin/env python3
"""Apply a conservative DINOScore visibility micro-filter to an existing cache.

This script uses a full DINOv3 feature NPZ built from the V2.3 frame-keep sample
set.  It maps each NPZ row back to (video_id, query_idx, frame_t) and, if the
input target cache currently marks that frame visible, flips it back to invisible
when the selected DINOScore rule fires.

It does not alter coordinates.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.metrics import compute_tapvid_metrics  # noqa: E402


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def load_npz(path: str | Path) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def feature_index(names: List[str], name: str) -> int:
    if name not in names:
        raise KeyError(f"feature {name} not found; available={names}")
    return names.index(name)


def should_drop(score: float, *, direction: str, threshold: float) -> bool:
    if direction == "low_is_bad":
        return score < threshold
    if direction == "high_is_bad":
        return score > threshold
    raise ValueError(f"Unknown direction {direction}")


def eval_std(cache: Dict[str, Any]) -> Dict[str, Any]:
    aj: List[float] = []
    oa: List[float] = []
    da: List[float] = []
    n = 0
    for r in cache["records"]:
        pred = torch.from_numpy(npy(r["pred_tracks"], np.float32))
        gt = torch.from_numpy(npy(r["gt_tracks"], np.float32))
        pv = torch.from_numpy(npy(r["pred_visibility"], bool))
        gv = torch.from_numpy(npy(r["gt_visibility"], bool))
        q = torch.from_numpy(npy(r["query_points"], np.float32))
        n += int(q.shape[0])
        m = compute_tapvid_metrics(pred, gt, pv, gv, q, resolution=256, query_mode="strided")
        aj.append(float(m.get("AJ", 0.0)))
        oa.append(float(m.get("OA", 0.0)))
        da.append(float(m.get("average_pts_within_thresh", 0.0)))
    return {
        "AJ_256": round(float(np.mean(aj)) * 100.0, 4),
        "OA_256": round(float(np.mean(oa)) * 100.0, 4),
        "delta_avg_256": round(float(np.mean(da)) * 100.0, 4),
        "n_records": len(cache["records"]),
        "n_queries": n,
    }


def eval_ajrd(cache: Dict[str, Any], name: str) -> Dict[str, Any]:
    tmp = Path(tempfile.gettempdir()) / f"{name}.pt"
    out = Path(tempfile.gettempdir()) / f"{name}_ajrd.json"
    torch.save(cache, tmp)
    subprocess.run(
        [sys.executable, "scripts/eval_aj_rd_from_cache.py", "--cache-path", str(tmp), "--output-json", str(out)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.load(open(out))


def build_record_index(cache: Dict[str, Any]) -> Dict[str, int]:
    return {str(r["video_id"]): i for i, r in enumerate(cache["records"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-cache", required=True)
    ap.add_argument("--vit-npz", required=True)
    ap.add_argument("--feature", default="last_candidate_cosine")
    ap.add_argument("--direction", choices=["low_is_bad", "high_is_bad"], default="low_is_bad")
    ap.add_argument("--threshold", type=float, default=0.395)
    ap.add_argument("--out-cache", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--name", default="reentry_v24_dinoscore_micro")
    args = ap.parse_args()

    target = torch.load(args.target_cache, map_location="cpu", weights_only=False)
    output = dict(target)
    # Shallow copy records and deep-copy mutable arrays we may edit.
    output_records = []
    for r in target["records"]:
        nr = dict(r)
        nr["pred_visibility"] = npy(r["pred_visibility"], bool).copy()
        nr["pred_tracks"] = npy(r["pred_tracks"], np.float32).copy()
        output_records.append(nr)
    output["records"] = output_records

    vit = load_npz(args.vit_npz)
    X = np.asarray(vit["X_patch"], dtype=np.float32)
    names = [str(x) for x in vit["patch_feature_names"].tolist()]
    j = feature_index(names, args.feature)
    meta = [json.loads(str(x)) for x in vit["meta_json"].tolist()]
    row_scores = X[:, j]
    record_idx = build_record_index(output)

    considered = 0
    fired = 0
    target_visible = 0
    flipped = 0
    already_invisible = 0
    missing_record = 0
    label_counts: Dict[str, Dict[str, int]] = {}
    for label in ["y_safe16", "y_gt_visible", "y_safe8", "y_utility", "y_safe4"]:
        if label in vit:
            label_counts[label] = {"fired_positive": 0, "fired_negative": 0, "flipped_positive": 0, "flipped_negative": 0}

    per_video: Dict[str, Dict[str, int]] = {}
    for i, m in enumerate(meta):
        vid = str(m["video_id"])
        qi = int(m["query_idx"])
        t = int(m["frame_t"])
        considered += 1
        if vid not in record_idx:
            missing_record += 1
            continue
        if not should_drop(float(row_scores[i]), direction=args.direction, threshold=float(args.threshold)):
            continue
        fired += 1
        rec = output["records"][record_idx[vid]]
        pv = rec["pred_visibility"]
        if qi < 0 or qi >= pv.shape[0] or t < 0 or t >= pv.shape[1]:
            continue
        if vid not in per_video:
            per_video[vid] = {"fired": 0, "target_visible": 0, "flipped": 0, "already_invisible": 0}
        per_video[vid]["fired"] += 1
        for label, counts in label_counts.items():
            y = bool(np.asarray(vit[label])[i] > 0.5)
            counts["fired_positive" if y else "fired_negative"] += 1
        if bool(pv[qi, t]):
            target_visible += 1
            per_video[vid]["target_visible"] += 1
            pv[qi, t] = False
            flipped += 1
            per_video[vid]["flipped"] += 1
            for label, counts in label_counts.items():
                y = bool(np.asarray(vit[label])[i] > 0.5)
                counts["flipped_positive" if y else "flipped_negative"] += 1
        else:
            already_invisible += 1
            per_video[vid]["already_invisible"] += 1

    out_cache = Path(args.out_cache)
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    output["model_name"] = args.name
    output["dinoscore_policy"] = {
        "feature": args.feature,
        "direction": args.direction,
        "threshold": float(args.threshold),
        "vit_npz": str(args.vit_npz),
        "target_cache": str(args.target_cache),
    }
    torch.save(output, out_cache)

    std = eval_std(output)
    ajrd = eval_ajrd(output, args.name.replace("/", "_"))
    manifest = {
        "method": "ReEntry-V2.4-DINOScore-Micro",
        "cache": str(out_cache),
        "target_cache": str(args.target_cache),
        "vit_npz": str(args.vit_npz),
        "feature": args.feature,
        "direction": args.direction,
        "threshold": float(args.threshold),
        "uses_gt_at_inference": False,
        "considered_rows": int(considered),
        "fired_rows": int(fired),
        "target_visible_rows": int(target_visible),
        "flipped_rows": int(flipped),
        "already_invisible_rows": int(already_invisible),
        "missing_record_rows": int(missing_record),
        "fired_over_considered": float(fired / max(considered, 1)),
        "flipped_over_considered": float(flipped / max(considered, 1)),
        "flipped_over_fired": float(flipped / max(fired, 1)),
        "label_counts": label_counts,
        "per_video": per_video,
        "true_AJ_RD_256": ajrd.get("true_AJ_RD_256"),
        "true_AJ_RD": ajrd.get("true_AJ_RD"),
        **std,
    }
    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
