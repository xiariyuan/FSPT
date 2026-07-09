#!/usr/bin/env python3
"""Apply precomputed streaming DINOScore decision chunks to a target cache."""
from __future__ import annotations

import argparse
import json
import re
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


def load_decision_npz(path: Path) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def parse_range(path: Path) -> Tuple[int, int]:
    m = re.search(r"_(\d{5,6})_(\d{5,6})\.npz$", path.name)
    if not m:
        return (0, 0)
    return int(m.group(1)), int(m.group(2))


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-cache", required=True)
    ap.add_argument("--chunk-dir", required=True)
    ap.add_argument("--glob", default="*.npz")
    ap.add_argument("--out-cache", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--name", default="v24_dinoscore_stream")
    args = ap.parse_args()

    target = torch.load(args.target_cache, map_location="cpu", weights_only=False)
    output = dict(target)
    output_records = []
    for r in target["records"]:
        nr = dict(r)
        nr["pred_visibility"] = npy(r["pred_visibility"], bool).copy()
        nr["pred_tracks"] = npy(r["pred_tracks"], np.float32).copy()
        output_records.append(nr)
    output["records"] = output_records
    record_idx = {str(r["video_id"]): i for i, r in enumerate(output_records)}

    paths = sorted(Path(args.chunk_dir).glob(args.glob), key=parse_range)
    if not paths:
        raise FileNotFoundError(f"No decision chunks in {args.chunk_dir} matching {args.glob}")

    scanned_rows = 0
    selected_visible_rows = 0
    fires = 0
    flipped = 0
    already_invisible_at_apply = 0
    per_video: Dict[str, Dict[str, int]] = {}
    chunks_summary = []
    for path in paths:
        d = load_decision_npz(path)
        row_indices = np.asarray(d["row_indices"], dtype=np.int64)
        fire = np.asarray(d["fires"], dtype=bool)
        infos = [json.loads(str(x)) for x in d["info_json"].tolist()]
        scanned = int(np.asarray(d.get("scanned_rows", np.asarray(0))).item()) if "scanned_rows" in d else int(np.asarray(d["end_index"]).item() - np.asarray(d["start_index"]).item())
        scanned_rows += scanned
        selected_visible_rows += int(row_indices.shape[0])
        fires += int(fire.sum())
        chunk_flipped = 0
        for is_fire, info in zip(fire, infos):
            if not bool(is_fire):
                continue
            vid = str(info["video_id"])
            qi = int(info["query_idx"])
            t = int(info["frame_t"])
            if vid not in record_idx:
                continue
            rec = output_records[record_idx[vid]]
            pv = rec["pred_visibility"]
            if not bool(pv[qi, t]):
                already_invisible_at_apply += 1
                continue
            pv[qi, t] = False
            flipped += 1
            chunk_flipped += 1
            if vid not in per_video:
                per_video[vid] = {"fires": 0, "flipped": 0}
            per_video[vid]["fires"] += 1
            per_video[vid]["flipped"] += 1
        chunks_summary.append({"path": str(path), "selected_visible_rows": int(row_indices.shape[0]), "fires": int(fire.sum()), "flipped": int(chunk_flipped)})

    out_cache = Path(args.out_cache)
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    output["model_name"] = args.name
    output["dinoscore_decision_chunks"] = {"chunk_dir": str(args.chunk_dir), "glob": args.glob, "target_cache": str(args.target_cache)}
    torch.save(output, out_cache)

    std = eval_std(output)
    ajrd = eval_ajrd(output, args.name.replace("/", "_"))
    manifest = {
        "method": "ReEntry-V2.4-DINOScore-Stream",
        "cache": str(out_cache),
        "target_cache": str(args.target_cache),
        "chunk_dir": str(args.chunk_dir),
        "glob": args.glob,
        "n_chunks": len(paths),
        "scanned_rows": int(scanned_rows),
        "selected_visible_rows": int(selected_visible_rows),
        "fires": int(fires),
        "flipped": int(flipped),
        "already_invisible_at_apply": int(already_invisible_at_apply),
        "selected_visible_over_scanned": float(selected_visible_rows / max(scanned_rows, 1)),
        "fires_over_selected_visible": float(fires / max(selected_visible_rows, 1)),
        "flipped_over_scanned": float(flipped / max(scanned_rows, 1)),
        "per_video": per_video,
        "chunks": chunks_summary,
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
