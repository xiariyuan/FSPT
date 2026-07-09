#!/usr/bin/env python3
"""Probe whether DINO/ViT patch features add learnable signal beyond numeric features.

Train lightweight logistic regression probes on dev0-6 and evaluate on dev7-9.
This is diagnostic only; it does not produce a method cache.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LABELS = ["y_safe16", "y_gt_visible", "y_safe8", "y_utility", "y_safe4"]


def load_npz(path: str | Path) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def safe_auc_ap(y: np.ndarray, p: np.ndarray) -> Dict[str, float | None]:
    y = np.asarray(y, dtype=np.float32)
    p = np.asarray(p, dtype=np.float64)
    if int((y > 0.5).sum()) == 0 or int((y <= 0.5).sum()) == 0:
        return {"auc": None, "ap": None}
    return {
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
    }


def idx(names: List[str], name: str) -> int:
    if name not in names:
        raise KeyError(name)
    return names.index(name)


def make_subsets(Xn: np.ndarray, names: List[str]) -> Dict[str, np.ndarray]:
    event = Xn[:, idx(names, "event_gate_prob")]
    dist = Xn[:, idx(names, "frame_base_override_dist_norm")]
    v1 = Xn[:, idx(names, "v1_prob")]
    segmax = Xn[:, idx(names, "segment_prob_max")]
    return {
        "all": np.ones(Xn.shape[0], dtype=bool),
        "numeric_thinks_safe_event_ge_002_dist_lt_025": (event >= 0.02) & (dist < 0.25),
        "distance_ambiguous_025_050": (dist >= 0.25) & (dist < 0.50),
        "low_v1_prob_010_030": (v1 >= 0.10) & (v1 < 0.30),
        "high_event_low_dist_low_v1": (event >= 0.02) & (dist < 0.25) & (v1 < 0.30),
        "low_segment_max_lt_050": segmax < 0.50,
    }


def fit_predict(xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, *, class_weight: str | None) -> np.ndarray:
    # liblinear is stable on small diagnostic sets; max_iter high enough for convergence.
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            C=1.0,
            class_weight=class_weight,
            random_state=0,
        ),
    )
    clf.fit(xtr, ytr.astype(int))
    return clf.predict_proba(xva)[:, 1].astype(np.float64)


def evaluate_feature_set(
    name: str,
    xtr: np.ndarray,
    xva: np.ndarray,
    train: Dict[str, Any],
    val: Dict[str, Any],
    subsets: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for label in LABELS:
        if label not in train or label not in val:
            continue
        ytr = np.asarray(train[label], dtype=np.float32)
        yva = np.asarray(val[label], dtype=np.float32)
        label_out: Dict[str, Any] = {
            "train_positive_rate": float(ytr.mean()) if ytr.size else 0.0,
            "val_positive_rate": float(yva.mean()) if yva.size else 0.0,
            "classifiers": {},
        }
        for cw_name, cw in [("plain", None), ("balanced", "balanced")]:
            try:
                p = fit_predict(xtr, ytr, xva, class_weight=cw)
            except Exception as exc:  # noqa: BLE001
                label_out["classifiers"][cw_name] = {"error": repr(exc)}
                continue
            cls_out: Dict[str, Any] = {"all": safe_auc_ap(yva, p)}
            for subset_name, mask in subsets.items():
                if subset_name == "all":
                    continue
                if int(mask.sum()) < 30:
                    cls_out[subset_name] = {"n": int(mask.sum()), "auc": None, "ap": None, "positive_rate": None}
                    continue
                m = mask.astype(bool)
                metrics = safe_auc_ap(yva[m], p[m])
                cls_out[subset_name] = {
                    "n": int(m.sum()),
                    "positive_rate": float(yva[m].mean()) if int(m.sum()) else None,
                    **metrics,
                }
            label_out["classifiers"][cw_name] = cls_out
        out[label] = label_out
    return out


def md_float(x: Any) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = [
        "# DINO/ViT + Numeric Probe",
        "",
        f"Train NPZ: `{payload['train_npz']}`",
        f"Val NPZ: `{payload['val_npz']}`",
        "",
        "## Summary, all validation samples",
        "",
    ]
    rows = []
    for fs_name, fs in payload["feature_sets"].items():
        for label, label_out in fs.items():
            for clf_name, cls_out in label_out.get("classifiers", {}).items():
                if "error" in cls_out:
                    continue
                allm = cls_out.get("all", {})
                rows.append({
                    "features": fs_name,
                    "label": label,
                    "classifier": clf_name,
                    "auc": allm.get("auc"),
                    "ap": allm.get("ap"),
                    "val_pos": label_out.get("val_positive_rate"),
                })
    lines.append("| features | label | classifier | AUC | AP | val positive rate |")
    lines.append("|---|---|---|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['features']} | {r['label']} | {r['classifier']} | {md_float(r['auc'])} | {md_float(r['ap'])} | {md_float(r['val_pos'])} |"
        )
    lines.append("")
    lines.append("## Hard subsets, balanced classifier")
    lines.append("")
    for label in LABELS:
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| subset | features | n | pos rate | AUC | AP |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for subset_name in payload["subsets"]:
            if subset_name == "all":
                continue
            for fs_name, fs in payload["feature_sets"].items():
                if label not in fs:
                    continue
                cls = fs[label].get("classifiers", {}).get("balanced", {})
                sm = cls.get(subset_name, {})
                lines.append(
                    f"| {subset_name} | {fs_name} | {sm.get('n','')} | {md_float(sm.get('positive_rate'))} | {md_float(sm.get('auc'))} | {md_float(sm.get('ap'))} |"
                )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--val-npz", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    train = load_npz(args.train_npz)
    val = load_npz(args.val_npz)
    Xn_tr = np.asarray(train["numeric_X"], dtype=np.float32)
    Xn_va = np.asarray(val["numeric_X"], dtype=np.float32)
    Xv_tr = np.asarray(train["X_patch"], dtype=np.float32)
    Xv_va = np.asarray(val["X_patch"], dtype=np.float32)
    nnames = [str(x) for x in val["numeric_feature_names"].tolist()]
    subsets = make_subsets(Xn_va, nnames)
    feature_inputs = {
        "numeric": (Xn_tr, Xn_va),
        "dinov3": (Xv_tr, Xv_va),
        "numeric_plus_dinov3": (np.concatenate([Xn_tr, Xv_tr], axis=1), np.concatenate([Xn_va, Xv_va], axis=1)),
    }
    payload: Dict[str, Any] = {
        "train_npz": str(args.train_npz),
        "val_npz": str(args.val_npz),
        "n_train": int(Xn_tr.shape[0]),
        "n_val": int(Xn_va.shape[0]),
        "numeric_dim": int(Xn_tr.shape[1]),
        "dinov3_dim": int(Xv_tr.shape[1]),
        "subsets": {k: int(v.sum()) for k, v in subsets.items()},
        "feature_sets": {},
    }
    for fs_name, (xtr, xva) in feature_inputs.items():
        payload["feature_sets"][fs_name] = evaluate_feature_set(fs_name, xtr, xva, train, val, subsets)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(Path(args.out_md), payload)
    print(json.dumps({
        "out_json": str(out_json),
        "out_md": str(args.out_md),
        "n_train": payload["n_train"],
        "n_val": payload["n_val"],
        "subsets": payload["subsets"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
