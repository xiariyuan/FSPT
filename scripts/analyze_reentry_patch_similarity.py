#!/usr/bin/env python3
"""Analyze RGB patch-similarity features for V2.3 frame-keep samples."""
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


def average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    yb = np.asarray(y, dtype=np.float32) > 0.5
    n_pos = int(yb.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    ys = yb[order]
    tp = np.cumsum(ys, dtype=np.float64)
    rank = np.arange(1, ys.shape[0] + 1, dtype=np.float64)
    prec = tp / rank
    return float(prec[ys].sum() / float(n_pos))


def feature_stats(X: np.ndarray, y: np.ndarray, names: List[str]) -> List[Dict[str, Any]]:
    rows = []
    yb = np.asarray(y, dtype=np.float32) > 0.5
    for j, name in enumerate(names):
        score = np.asarray(X[:, j], dtype=np.float64)
        auc = roc_auc(yb, score)
        if auc is None:
            continue
        direction = "+" if auc >= 0.5 else "-"
        oriented = score if direction == "+" else -score
        ap = average_precision(yb, oriented)
        pos = score[yb]
        neg = score[~yb]
        rows.append({
            "feature": name,
            "auc_raw": float(auc),
            "auc_best": float(max(auc, 1.0 - auc)),
            "direction": direction,
            "ap_oriented": float(ap) if ap is not None else None,
            "pos_mean": float(pos.mean()) if pos.size else 0.0,
            "neg_mean": float(neg.mean()) if neg.size else 0.0,
        })
    rows.sort(key=lambda r: (r["auc_best"], r["ap_oriented"] or 0.0), reverse=True)
    return rows


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
    ap.add_argument("--patch-npz", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()
    d = load_npz(args.patch_npz)
    Xp = np.asarray(d["X_patch"], dtype=np.float32)
    Xn = np.asarray(d["numeric_X"], dtype=np.float32)
    pnames = [str(x) for x in d["patch_feature_names"].tolist()]
    nnames = [str(x) for x in d["numeric_feature_names"].tolist()]
    labels = [k for k in ["y_safe16", "y_gt_visible", "y_safe8", "y_utility", "y_safe4"] if k in d]
    payload = {
        "patch_npz": str(args.patch_npz),
        "n_samples": int(Xp.shape[0]),
        "patch_feature_dim": int(Xp.shape[1]),
        "numeric_feature_dim": int(Xn.shape[1]),
        "labels": labels,
        "patch_feature_auc": {},
        "numeric_feature_auc": {},
    }
    for label in labels:
        y = np.asarray(d[label], dtype=np.float32)
        payload["patch_feature_auc"][label] = feature_stats(Xp, y, pnames)
        payload["numeric_feature_auc"][label] = feature_stats(Xn, y, nnames)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# ReEntry Patch Similarity Feature Analysis", ""]
    lines.append(f"Samples: `{payload['n_samples']}`")
    lines.append(f"Patch feature dim: `{payload['patch_feature_dim']}`; numeric dim: `{payload['numeric_feature_dim']}`")
    lines.append("")
    for label in labels:
        lines.append(f"## `{label}` patch features")
        lines.append("")
        lines.append(md_table(payload["patch_feature_auc"][label], [
            ("feature", "feature"), ("AUC", "auc_best"), ("dir", "direction"), ("AP", "ap_oriented"), ("pos mean", "pos_mean"), ("neg mean", "neg_mean")
        ]))
        lines.append("")
        lines.append(f"## `{label}` numeric reference features")
        lines.append("")
        lines.append(md_table(payload["numeric_feature_auc"][label], [
            ("feature", "feature"), ("AUC", "auc_best"), ("dir", "direction"), ("AP", "ap_oriented"), ("pos mean", "pos_mean"), ("neg mean", "neg_mean")
        ], max_rows=10))
        lines.append("")
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "out_json": str(out_json),
        "out_md": str(args.out_md),
        "n_samples": int(Xp.shape[0]),
        "top_patch_safe16": payload["patch_feature_auc"].get("y_safe16", [])[:5],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
