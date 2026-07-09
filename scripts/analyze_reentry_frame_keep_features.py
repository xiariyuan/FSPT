#!/usr/bin/env python3
"""Analyze V2.3 frame-keep feature separability and train/val shift.

This is a diagnostic script, not a model-training script.  It answers whether the
current numeric features can separate useful from harmful V1 proposed recovery
frames.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


DEFAULT_LABELS = ["y_safe16", "y_gt_visible", "y_safe8", "y_utility", "y_safe4"]


def load_npz(path: str | Path) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def average_ranks(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.shape[0], dtype=np.float64)
    i = 0
    n = x.shape[0]
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def roc_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    pos = int(y.sum())
    neg = int(y.shape[0] - pos)
    if pos == 0 or neg == 0:
        return None
    ranks = average_ranks(score)
    sum_pos = float(ranks[y].sum())
    auc = (sum_pos - pos * (pos + 1) / 2.0) / float(pos * neg)
    return float(auc)


def average_precision(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted, dtype=np.float64)
    ranks = np.arange(1, y_sorted.shape[0] + 1, dtype=np.float64)
    precision = tp / ranks
    ap = float(precision[y_sorted].sum() / float(n_pos))
    return ap


def effect_size(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.size == 0 or neg.size == 0:
        return 0.0
    pooled = np.sqrt(0.5 * (float(pos.var()) + float(neg.var())) + 1e-12)
    return float((float(pos.mean()) - float(neg.mean())) / pooled)


def summarize_dataset(name: str, data: Dict[str, Any], labels: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "name": name,
        "n_samples": int(np.asarray(data["X"]).shape[0]),
        "feature_dim": int(np.asarray(data["X"]).shape[1]),
    }
    for label in labels:
        if label in data:
            y = np.asarray(data[label], dtype=np.float32)
            out[f"positive_rate_{label}"] = float(y.mean()) if y.size else 0.0
            out[f"positive_count_{label}"] = int((y > 0.5).sum())
    if "utility" in data:
        u = np.asarray(data["utility"], dtype=np.float32)
        out["mean_utility"] = float(u.mean()) if u.size else 0.0
    return out


def feature_label_stats(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    yb = np.asarray(y, dtype=np.float32) > 0.5
    for j, name in enumerate(feature_names):
        s = np.asarray(X[:, j], dtype=np.float64)
        auc_raw = roc_auc(yb, s)
        if auc_raw is None:
            continue
        direction = "+" if auc_raw >= 0.5 else "-"
        best_auc = float(max(auc_raw, 1.0 - auc_raw))
        oriented_score = s if direction == "+" else -s
        ap = average_precision(yb, oriented_score)
        pos = s[yb]
        neg = s[~yb]
        rows.append({
            "feature": name,
            "auc_raw": float(auc_raw),
            "auc_best": best_auc,
            "direction": direction,
            "ap_oriented": float(ap) if ap is not None else None,
            "positive_mean": float(pos.mean()) if pos.size else 0.0,
            "negative_mean": float(neg.mean()) if neg.size else 0.0,
            "positive_std": float(pos.std()) if pos.size else 0.0,
            "negative_std": float(neg.std()) if neg.size else 0.0,
            "effect_size": effect_size(pos, neg),
        })
    rows.sort(key=lambda r: (r["auc_best"], r["ap_oriented"] if r["ap_oriented"] is not None else 0.0), reverse=True)
    return rows


def feature_shift(train_X: np.ndarray, val_X: np.ndarray, feature_names: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for j, name in enumerate(feature_names):
        tr = np.asarray(train_X[:, j], dtype=np.float64)
        va = np.asarray(val_X[:, j], dtype=np.float64)
        tr_mean = float(tr.mean())
        va_mean = float(va.mean())
        tr_std = float(tr.std())
        va_std = float(va.std())
        gap = float((va_mean - tr_mean) / max(tr_std, 1e-6))
        rows.append({
            "feature": name,
            "train_mean": tr_mean,
            "val_mean": va_mean,
            "train_std": tr_std,
            "val_std": va_std,
            "standardized_mean_gap": gap,
            "abs_standardized_mean_gap": abs(gap),
        })
    rows.sort(key=lambda r: r["abs_standardized_mean_gap"], reverse=True)
    return rows


def find_feature(feature_names: List[str], name: str) -> int | None:
    try:
        return feature_names.index(name)
    except ValueError:
        return None


def bin_rows(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    feature: str,
    bins: List[Tuple[str, float, float]],
) -> List[Dict[str, Any]]:
    idx = find_feature(feature_names, feature)
    if idx is None:
        return []
    values = np.asarray(X[:, idx], dtype=np.float64)
    yb = np.asarray(y, dtype=np.float32) > 0.5
    rows: List[Dict[str, Any]] = []
    for label, lo, hi in bins:
        if np.isneginf(lo):
            m = values < hi
        elif np.isposinf(hi):
            m = values >= lo
        else:
            m = (values >= lo) & (values < hi)
        count = int(m.sum())
        rows.append({
            "feature": feature,
            "bin": label,
            "count": count,
            "rate": float(count / max(values.shape[0], 1)),
            "positive_rate": float(yb[m].mean()) if count else 0.0,
            "value_mean": float(values[m].mean()) if count else 0.0,
        })
    return rows


def subgroup_analysis(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    specs = {
        "v1_prob": [
            ("[0.10,0.15)", 0.10, 0.15),
            ("[0.15,0.30)", 0.15, 0.30),
            ("[0.30,0.50)", 0.30, 0.50),
            ("[0.50,+inf)", 0.50, np.inf),
        ],
        "event_gate_prob": [
            ("[-inf,0.005)", -np.inf, 0.005),
            ("[0.005,0.01)", 0.005, 0.01),
            ("[0.01,0.02)", 0.01, 0.02),
            ("[0.02,+inf)", 0.02, np.inf),
        ],
        "segment_len_norm": [
            ("[0,0.0625) len<2", 0.0, 0.0625),
            ("[0.0625,0.125) len2-3", 0.0625, 0.125),
            ("[0.125,0.28125) len4-8", 0.125, 0.28125),
            ("[0.28125,+inf) len9+", 0.28125, np.inf),
        ],
        "position_in_segment_norm": [
            ("start-ish", -np.inf, 0.10),
            ("early-mid", 0.10, 0.45),
            ("late-mid", 0.45, 0.90),
            ("end-ish", 0.90, np.inf),
        ],
        "dist_to_segment_start_norm": [
            ("0", 0.0, 0.000001),
            ("(0,0.125)", 0.000001, 0.125),
            ("[0.125,0.5)", 0.125, 0.5),
            ("[0.5,+inf)", 0.5, np.inf),
        ],
        "dist_to_segment_end_norm": [
            ("0", 0.0, 0.000001),
            ("(0,0.125)", 0.000001, 0.125),
            ("[0.125,0.5)", 0.125, 0.5),
            ("[0.5,+inf)", 0.5, np.inf),
        ],
        "frame_base_override_dist_norm": [
            ("[0,0.25)", 0.0, 0.25),
            ("[0.25,0.5)", 0.25, 0.5),
            ("[0.5,1.0)", 0.5, 1.0),
            ("[1.0,+inf)", 1.0, np.inf),
        ],
    }
    out: Dict[str, List[Dict[str, Any]]] = {}
    for feature, bins in specs.items():
        rows = bin_rows(X, y, feature_names, feature, bins)
        if rows:
            out[feature] = rows
    return out


def fmt_float(x: Any, ndigits: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x)


def md_table(rows: List[Dict[str, Any]], cols: List[Tuple[str, str]], max_rows: int | None = None) -> str:
    if max_rows is not None:
        rows = rows[:max_rows]
    header = "| " + " | ".join(title for title, _ in cols) + " |"
    sep = "|" + "|".join(["---" for _ in cols]) + "|"
    body = []
    for r in rows:
        vals = []
        for _, key in cols:
            v = r.get(key, "")
            if isinstance(v, float):
                vals.append(fmt_float(v))
            else:
                vals.append(str(v))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + body)


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    labels = payload["labels"]
    lines: List[str] = []
    lines.append("# ReEntry Frame-Keep V2.3 Feature Separability Analysis")
    lines.append("")
    lines.append("## Dataset summary")
    lines.append("")
    lines.append(md_table(payload["dataset_summary"], [
        ("split", "name"),
        ("samples", "n_samples"),
        ("feature_dim", "feature_dim"),
        ("safe16+", "positive_rate_y_safe16"),
        ("gt_visible+", "positive_rate_y_gt_visible"),
        ("safe8+", "positive_rate_y_safe8"),
        ("utility+", "positive_rate_y_utility"),
        ("mean utility", "mean_utility"),
    ]))
    lines.append("")
    for label in labels:
        if label not in payload["feature_auc"]:
            continue
        lines.append(f"## Top feature separability for `{label}` on validation")
        lines.append("")
        lines.append(md_table(payload["feature_auc"][label], [
            ("feature", "feature"),
            ("best AUC", "auc_best"),
            ("dir", "direction"),
            ("AP", "ap_oriented"),
            ("pos mean", "positive_mean"),
            ("neg mean", "negative_mean"),
            ("effect", "effect_size"),
        ], max_rows=20))
        top = payload["feature_auc"][label][0] if payload["feature_auc"][label] else None
        if top:
            auc = top["auc_best"]
            lines.append("")
            if auc >= 0.70:
                lines.append(f"Interpretation: strongest single feature AUC is {auc:.3f}, so numeric features contain a meaningful signal for `{label}`.")
            elif auc >= 0.62:
                lines.append(f"Interpretation: strongest single feature AUC is {auc:.3f}, so the signal is moderate but not clean.")
            else:
                lines.append(f"Interpretation: strongest single feature AUC is only {auc:.3f}, suggesting weak separability from current numeric features.")
            lines.append("")
    lines.append("## Largest train/validation feature shifts")
    lines.append("")
    lines.append(md_table(payload["feature_shift"], [
        ("feature", "feature"),
        ("train mean", "train_mean"),
        ("val mean", "val_mean"),
        ("train std", "train_std"),
        ("val std", "val_std"),
        ("std gap", "standardized_mean_gap"),
    ], max_rows=25))
    lines.append("")
    lines.append("## Validation subgroup analysis for `y_safe16`")
    lines.append("")
    for feature, rows in payload["subgroups_y_safe16"].items():
        lines.append(f"### {feature}")
        lines.append("")
        lines.append(md_table(rows, [
            ("bin", "bin"),
            ("count", "count"),
            ("rate", "rate"),
            ("positive rate", "positive_rate"),
            ("value mean", "value_mean"),
        ]))
        lines.append("")
    lines.append("## Decision note")
    lines.append("")
    safe16_top = payload["feature_auc"].get("y_safe16", [{}])[0].get("auc_best", 0.0)
    if safe16_top >= 0.70:
        lines.append("The current numeric features show useful safe16 separability. Next step should be improving training/selection: class-balanced or calibrated model, checkpoint selection by downstream AJ_RD/AJ, and possibly a temporal model over recovery segments.")
    elif safe16_top >= 0.62:
        lines.append("The current numeric features show moderate safe16 separability. Next step can try better training and calibration, but appearance or external-teacher features may still be needed for the hard cases.")
    else:
        lines.append("The current numeric features show weak safe16 separability. Further MLP tuning is unlikely to close the oracle gap; prioritize appearance verifier or external tracker teacher features.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze V2.3 frame-keep features")
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--val-npz", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    args = ap.parse_args()

    train = load_npz(args.train_npz)
    val = load_npz(args.val_npz)
    feature_names = [str(x) for x in train["feature_names"].tolist()]
    train_X = np.asarray(train["X"], dtype=np.float32)
    val_X = np.asarray(val["X"], dtype=np.float32)
    labels = [l for l in args.labels if l in train and l in val]

    payload: Dict[str, Any] = {
        "train_npz": str(args.train_npz),
        "val_npz": str(args.val_npz),
        "labels": labels,
        "dataset_summary": [
            summarize_dataset("train", train, labels),
            summarize_dataset("val", val, labels),
        ],
        "feature_auc": {},
        "feature_shift": feature_shift(train_X, val_X, feature_names),
        "subgroups_y_safe16": {},
    }

    for label in labels:
        y_val = np.asarray(val[label], dtype=np.float32)
        payload["feature_auc"][label] = feature_label_stats(val_X, y_val, feature_names)

    if "y_safe16" in val:
        payload["subgroups_y_safe16"] = subgroup_analysis(val_X, np.asarray(val["y_safe16"], dtype=np.float32), feature_names)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(Path(args.out_md), payload)
    print(json.dumps({
        "out_json": str(out_json),
        "out_md": str(args.out_md),
        "labels": labels,
        "top_safe16": payload["feature_auc"].get("y_safe16", [])[:5],
        "top_shift": payload["feature_shift"][:5],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
