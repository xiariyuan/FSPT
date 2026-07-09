#!/usr/bin/env python3
"""Threshold-sweep diagnostic for V2.4-DINOScore.

This script does not run downstream cache evaluation.  It only asks whether a
small number of DINOv3 patch-similarity scores can selectively drop risky V1
proposed recovery frames in the V2.3 frame-keep dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


LABELS = ["y_safe16", "y_gt_visible", "y_safe8", "y_utility", "y_safe4"]
COSINE_FEATURES = ["best_ref_candidate_cosine", "last_candidate_cosine", "query_candidate_cosine"]
L2_FEATURES = ["best_ref_candidate_l2", "last_candidate_l2", "query_candidate_l2"]


def load_npz(path: str | Path) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def idx(names: List[str], name: str) -> int:
    if name not in names:
        raise KeyError(name)
    return names.index(name)


def make_subsets(Xn: np.ndarray, nnames: List[str]) -> Dict[str, np.ndarray]:
    event = Xn[:, idx(nnames, "event_gate_prob")]
    dist = Xn[:, idx(nnames, "frame_base_override_dist_norm")]
    v1 = Xn[:, idx(names := nnames, "v1_prob")]
    segmax = Xn[:, idx(nnames, "segment_prob_max")]
    return {
        "all": np.ones(Xn.shape[0], dtype=bool),
        "numeric_thinks_safe_event_ge_002_dist_lt_025": (event >= 0.02) & (dist < 0.25),
        "low_v1_prob_010_030": (v1 >= 0.10) & (v1 < 0.30),
        "low_segment_max_lt_050": segmax < 0.50,
        "union_ambiguous": ((event >= 0.02) & (dist < 0.25)) | ((v1 >= 0.10) & (v1 < 0.30)) | (segmax < 0.50),
        "distance_ambiguous_025_050": (dist >= 0.25) & (dist < 0.50),
    }


def safe_rate(y: np.ndarray, mask: np.ndarray) -> float | None:
    if int(mask.sum()) == 0:
        return None
    return float(np.asarray(y, dtype=np.float32)[mask].mean())


def sweep_one(
    scores: np.ndarray,
    *,
    feature: str,
    direction: str,
    region_name: str,
    region: np.ndarray,
    labels: Dict[str, np.ndarray],
    thresholds: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n_total = int(scores.shape[0])
    n_region = int(region.sum())
    if n_region == 0:
        return rows
    for thr in thresholds:
        if direction == "low_is_bad":
            drop = region & (scores < float(thr))
        else:
            drop = region & (scores > float(thr))
        keep = ~drop
        n_drop = int(drop.sum())
        row: Dict[str, Any] = {
            "feature": feature,
            "direction": direction,
            "region": region_name,
            "threshold": float(thr),
            "n_total": n_total,
            "n_region": n_region,
            "n_drop": n_drop,
            "drop_rate_total": float(n_drop / max(n_total, 1)),
            "drop_rate_region": float(n_drop / max(n_region, 1)),
        }
        for label, y in labels.items():
            yb = np.asarray(y, dtype=np.float32) > 0.5
            neg = ~yb
            n_neg = int(neg.sum())
            n_pos = int(yb.sum())
            dropped_neg = int((drop & neg).sum())
            dropped_pos = int((drop & yb).sum())
            row[f"{label}_drop_positive_rate"] = safe_rate(y, drop)
            row[f"{label}_drop_negative_precision"] = float(dropped_neg / max(n_drop, 1)) if n_drop else None
            row[f"{label}_negative_recall"] = float(dropped_neg / max(n_neg, 1))
            row[f"{label}_positive_loss_rate"] = float(dropped_pos / max(n_pos, 1))
            row[f"{label}_keep_positive_rate"] = safe_rate(y, keep)
        rows.append(row)
    return rows


def pick_thresholds(scores: np.ndarray, region: np.ndarray, feature: str, direction: str) -> np.ndarray:
    vals = np.asarray(scores[region], dtype=np.float64)
    if vals.size == 0:
        return np.asarray([], dtype=np.float64)
    if direction == "low_is_bad":
        # Drop low-similarity tail.
        qs = [0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    else:
        # Drop high-distance tail.
        qs = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.98, 0.99]
    return np.unique(np.quantile(vals, qs)).astype(np.float64)


def summarize_best(rows: List[Dict[str, Any]], primary_label: str = "y_utility") -> List[Dict[str, Any]]:
    scored = []
    for r in rows:
        n_drop = int(r["n_drop"])
        if n_drop < 20:
            continue
        neg_prec = r.get(f"{primary_label}_drop_negative_precision")
        neg_recall = r.get(f"{primary_label}_negative_recall")
        safe16_loss = r.get("y_safe16_positive_loss_rate")
        safe16_drop_pos_rate = r.get("y_safe16_drop_positive_rate")
        if neg_prec is None:
            continue
        # Diagnostic score: prefer dropping mostly utility-negative frames, but penalize safe16-positive loss.
        score = float(neg_prec) + 0.5 * float(neg_recall) - 0.75 * float(safe16_loss or 0.0)
        row = dict(r)
        row["diagnostic_score"] = score
        row["safe16_drop_positive_rate"] = safe16_drop_pos_rate
        scored.append(row)
    scored.sort(key=lambda x: (x["diagnostic_score"], x.get(f"{primary_label}_drop_negative_precision") or 0.0), reverse=True)
    return scored[:50]


def md_float(x: Any) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = ["# V2.4-DINOScore Threshold Sweep", "", f"NPZ: `{payload['vit_npz']}`", f"Samples: `{payload['n_samples']}`", ""]
    lines.append("## Dataset label rates")
    lines.append("")
    lines.append("| label | positive rate |")
    lines.append("|---|---:|")
    for label, rate in payload["label_rates"].items():
        lines.append(f"| {label} | {rate:.4f} |")
    lines.append("")
    lines.append("## Regions")
    lines.append("")
    lines.append("| region | n | rate |")
    lines.append("|---|---:|---:|")
    for region, n in payload["region_counts"].items():
        lines.append(f"| {region} | {n} | {n / max(payload['n_samples'], 1):.4f} |")
    lines.append("")
    lines.append("## Top diagnostic thresholds, utility-negative precision objective")
    lines.append("")
    lines.append("| feature | region | direction | threshold | n_drop | drop total | util neg precision | util neg recall | safe16 pos loss | safe16 drop pos rate | score |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in payload["best_rows"][:40]:
        lines.append(
            f"| {r['feature']} | {r['region']} | {r['direction']} | {r['threshold']:.5f} | {r['n_drop']} | {r['drop_rate_total']:.4f} | "
            f"{md_float(r.get('y_utility_drop_negative_precision'))} | {md_float(r.get('y_utility_negative_recall'))} | "
            f"{md_float(r.get('y_safe16_positive_loss_rate'))} | {md_float(r.get('y_safe16_drop_positive_rate'))} | {r['diagnostic_score']:.4f} |"
        )
    lines.append("")
    lines.append("## Interpretation note")
    lines.append("")
    lines.append("This is label-level threshold diagnostics only. It does not yet evaluate AJ_RD/AJ/OA. Promising rows should be converted into a full evaluator only if they drop enough frames with high utility-negative precision and low safe16-positive loss.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


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
    labels = {label: np.asarray(d[label], dtype=np.float32) for label in LABELS if label in d}
    subsets = make_subsets(Xn, nnames)

    all_rows: List[Dict[str, Any]] = []
    for region_name, region in subsets.items():
        if int(region.sum()) == 0:
            continue
        for feature in COSINE_FEATURES:
            scores = Xv[:, idx(vnames, feature)]
            thresholds = pick_thresholds(scores, region, feature, "low_is_bad")
            all_rows.extend(sweep_one(scores, feature=feature, direction="low_is_bad", region_name=region_name, region=region, labels=labels, thresholds=thresholds))
        for feature in L2_FEATURES:
            scores = Xv[:, idx(vnames, feature)]
            thresholds = pick_thresholds(scores, region, feature, "high_is_bad")
            all_rows.extend(sweep_one(scores, feature=feature, direction="high_is_bad", region_name=region_name, region=region, labels=labels, thresholds=thresholds))

    payload = {
        "vit_npz": str(args.vit_npz),
        "n_samples": int(Xv.shape[0]),
        "label_rates": {label: float(y.mean()) for label, y in labels.items()},
        "region_counts": {name: int(mask.sum()) for name, mask in subsets.items()},
        "rows": all_rows,
        "best_rows": summarize_best(all_rows, primary_label="y_utility"),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(Path(args.out_md), payload)
    print(json.dumps({"out_json": str(out_json), "out_md": str(args.out_md), "n_rows": len(all_rows), "top": payload["best_rows"][:5]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
