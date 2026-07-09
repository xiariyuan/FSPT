#!/usr/bin/env python3
"""Analyze local ViT/DINO features on hard subsets of V2.3 samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def load_npz(path: str | Path) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.shape[0], dtype=np.float64)
    i = 0
    while i < x.shape[0]:
        j = i
        while j + 1 < x.shape[0] and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def roc_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    yb = np.asarray(y, dtype=np.float32) > 0.5
    pos = int(yb.sum())
    neg = int(yb.shape[0] - pos)
    if pos == 0 or neg == 0:
        return None
    ranks = average_ranks(np.asarray(score, dtype=np.float64))
    sum_pos = float(ranks[yb].sum())
    return float((sum_pos - pos * (pos + 1) / 2.0) / float(pos * neg))


def top_auc(X: np.ndarray, y: np.ndarray, names: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for j, name in enumerate(names):
        auc = roc_auc(y, X[:, j])
        if auc is None:
            continue
        rows.append({
            "feature": name,
            "auc_raw": float(auc),
            "auc_best": float(max(auc, 1.0 - auc)),
            "direction": "+" if auc >= 0.5 else "-",
        })
    rows.sort(key=lambda r: r["auc_best"], reverse=True)
    return rows


def idx(names: List[str], name: str) -> int:
    if name not in names:
        raise KeyError(name)
    return names.index(name)


def make_subsets(Xn: np.ndarray, nnames: List[str]) -> Dict[str, np.ndarray]:
    event = Xn[:, idx(nnames, "event_gate_prob")]
    dist = Xn[:, idx(nnames, "frame_base_override_dist_norm")]
    v1 = Xn[:, idx(nnames, "v1_prob")]
    segmax = Xn[:, idx(nnames, "segment_prob_max")]
    return {
        "all": np.ones(Xn.shape[0], dtype=bool),
        "numeric_thinks_safe_event_ge_002_dist_lt_025": (event >= 0.02) & (dist < 0.25),
        "distance_ambiguous_025_050": (dist >= 0.25) & (dist < 0.50),
        "low_v1_prob_010_030": (v1 >= 0.10) & (v1 < 0.30),
        "high_event_low_dist_low_v1": (event >= 0.02) & (dist < 0.25) & (v1 < 0.30),
        "high_event_low_dist_high_v1": (event >= 0.02) & (dist < 0.25) & (v1 >= 0.50),
        "low_segment_max_lt_050": segmax < 0.50,
    }


def md_table(rows: List[Dict[str, Any]], cols: List[Tuple[str, str]], max_rows: int = 20) -> str:
    rows = rows[:max_rows]
    out = ["| " + " | ".join(t for t, _ in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        vals = []
        for _, k in cols:
            v = r.get(k, "")
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vit-npz", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()
    d = load_npz(args.vit_npz)
    Xv = np.asarray(d["X_patch"], dtype=np.float32)
    Xn = np.asarray(d["numeric_X"], dtype=np.float32)
    vnames = [str(x) for x in d["patch_feature_names"].tolist()]
    nnames = [str(x) for x in d["numeric_feature_names"].tolist()]
    labels = [k for k in ["y_safe16", "y_gt_visible", "y_safe8", "y_utility", "y_safe4"] if k in d]
    subsets = make_subsets(Xn, nnames)
    payload: Dict[str, Any] = {"vit_npz": str(args.vit_npz), "n_samples": int(Xv.shape[0]), "subsets": {}}
    for subset_name, mask in subsets.items():
        entry: Dict[str, Any] = {"n": int(mask.sum()), "labels": {}}
        if int(mask.sum()) < 50:
            payload["subsets"][subset_name] = entry
            continue
        for label in labels:
            y = np.asarray(d[label], dtype=np.float32)[mask]
            entry["labels"][label] = {
                "positive_rate": float(y.mean()) if y.size else 0.0,
                "vit_top": top_auc(Xv[mask], y, vnames)[:10],
                "numeric_top": top_auc(Xn[mask], y, nnames)[:10],
            }
        payload["subsets"][subset_name] = entry
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# DINO/ViT Hard-Subset Analysis", "", f"Samples: `{payload['n_samples']}`", ""]
    for subset_name, entry in payload["subsets"].items():
        lines.append(f"## {subset_name}")
        lines.append("")
        lines.append(f"n = `{entry['n']}`")
        lines.append("")
        for label, res in entry.get("labels", {}).items():
            lines.append(f"### {label}")
            lines.append("")
            lines.append(f"positive_rate = `{res['positive_rate']:.4f}`")
            lines.append("")
            lines.append("DINO/ViT top:")
            lines.append(md_table(res["vit_top"], [("feature", "feature"), ("AUC", "auc_best"), ("dir", "direction")], max_rows=5))
            lines.append("")
            lines.append("Numeric top:")
            lines.append(md_table(res["numeric_top"], [("feature", "feature"), ("AUC", "auc_best"), ("dir", "direction")], max_rows=5))
            lines.append("")
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out_json": str(out_json), "out_md": str(args.out_md), "subsets": {k: v["n"] for k, v in payload["subsets"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
