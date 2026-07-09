#!/usr/bin/env python3
from __future__ import annotations
import json, math, argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch

TEACHERS = ["cotracker3_online", "cotracker3_offline", "trackon2"]
CACHE_PATHS = {
    "cotracker3_online": "outputs/redetection_ladder_2026-06-17/caches/cotracker3_online_strided_original.pt",
    "cotracker3_offline": "outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt",
    "trackon2": "caches/trackon2_strided_original.pt",
}


def load_cache(path: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return {str(r["video_id"]): r for r in obj["records"]}


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def px_scale(record):
    s = to_np(record["original_size"]).astype(float).reshape(-1)
    h, w = float(s[0]), float(s[1])
    return np.array([h - 1.0, w - 1.0], dtype=np.float32)  # y,x scale


def point_px(record, q, t):
    tr = to_np(record["pred_tracks"])[q, t].astype(np.float32)
    return tr * px_scale(record)


def vis_val(record, q, t):
    v = to_np(record["pred_visibility"])[q, t]
    return float(v)


def safe_norm(v):
    return float(np.sqrt(np.sum(np.asarray(v, dtype=np.float32) ** 2)))


def build_features(label_rows, caches):
    rows = []
    for e in label_rows:
        vid = str(e["video_id"])
        q = int(e["query_idx"])
        t = int(e["reentry_t"])
        if any(vid not in caches[name] for name in TEACHERS):
            continue
        rec0 = caches[TEACHERS[0]][vid]
        T = int(to_np(rec0["pred_tracks"]).shape[1])
        if q < 0 or q >= int(to_np(rec0["pred_tracks"]).shape[0]) or t < 0 or t >= T:
            continue

        feat = {
            "video_id": vid,
            "query_idx": q,
            "reentry_t": t,
            "occ_length": float(e.get("occ_length") or 0),
            "t_norm": float(t / max(T - 1, 1)),
            "label": str(e["best_teacher"]),
        }
        pts = {}
        for name in TEACHERS:
            rec = caches[name][vid]
            tr = to_np(rec["pred_tracks"])
            vis = to_np(rec["pred_visibility"])
            scale = px_scale(rec)
            p = tr[q, t].astype(np.float32) * scale
            pts[name] = p
            prev = tr[q, max(t - 1, 0)].astype(np.float32) * scale
            nxt = tr[q, min(t + 1, T - 1)].astype(np.float32) * scale
            feat[f"{name}_vis_t"] = float(vis[q, t])
            lo, hi = max(0, t - 2), min(T, t + 3)
            feat[f"{name}_vis_win"] = float(np.mean(vis[q, lo:hi].astype(np.float32)))
            feat[f"{name}_jump_prev"] = safe_norm(p - prev)
            feat[f"{name}_jump_next"] = safe_norm(nxt - p)
            feat[f"{name}_speed_local"] = safe_norm(nxt - prev) / max(min(t + 1, T - 1) - max(t - 1, 0), 1)
            y_norm, x_norm = tr[q, t]
            feat[f"{name}_in_bounds"] = float(0.0 <= y_norm <= 1.0 and 0.0 <= x_norm <= 1.0)
        for name in TEACHERS:
            dists = [safe_norm(pts[name] - pts[other]) for other in TEACHERS if other != name]
            feat[f"{name}_disagree_mean"] = float(np.mean(dists))
            feat[f"{name}_disagree_max"] = float(np.max(dists))
        # Pairwise named distances.
        for i,a in enumerate(TEACHERS):
            for b in TEACHERS[i+1:]:
                feat[f"dist_{a}_vs_{b}"] = safe_norm(pts[a] - pts[b])
        # Evaluation-only fields for reporting, not used as features.
        terr = e.get("teacher_errors", {}) or {}
        for name in TEACHERS:
            feat[f"error_{name}"] = float(terr.get(name, math.nan))
        feat["oracle_error"] = float(e.get("best_error", math.nan))
        rows.append(feat)
    return rows


def matrix(rows):
    feature_keys = [k for k in rows[0].keys() if k not in {"video_id","query_idx","reentry_t","label","error_cotracker3_online","error_cotracker3_offline","error_trackon2","oracle_error"}]
    X = np.array([[float(r[k]) for k in feature_keys] for r in rows], dtype=np.float32)
    y = np.array([TEACHERS.index(r["label"]) for r in rows], dtype=np.int64)
    groups = np.array([r["video_id"] for r in rows])
    return X, y, groups, feature_keys


def evaluate(rows, out_dir: Path):
    X, y, groups, feature_keys = matrix(rows)
    n = len(rows)
    label_counts = Counter([r["label"] for r in rows])
    result = {"n_events": n, "label_counts": dict(label_counts), "feature_keys": feature_keys}

    # Baseline errors.
    def mean_err_for_choice(choices):
        errs=[]
        for r, c in zip(rows, choices):
            errs.append(float(r[f"error_{c}"]))
        return float(np.nanmean(errs)), float(np.nanmedian(errs))

    baselines = {}
    for t in TEACHERS:
        choices = [t] * n
        baselines[t] = dict(zip(["mean_error","median_error"], mean_err_for_choice(choices)))
    oracle_choices = [r["label"] for r in rows]
    baselines["oracle"] = dict(zip(["mean_error","median_error"], mean_err_for_choice(oracle_choices)))
    result["baselines_error_px"] = baselines

    # Try sklearn models.
    try:
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        models = {
            "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", multi_class="auto")),
            "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", random_state=42),
            "hgb": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42),
        }
        logo = LeaveOneGroupOut()
        cv = {}
        for name, model in models.items():
            pred = np.zeros_like(y)
            for train_idx, test_idx in logo.split(X, y, groups):
                model.fit(X[train_idx], y[train_idx])
                pred[test_idx] = model.predict(X[test_idx])
            choices = [TEACHERS[int(i)] for i in pred]
            mean_err, med_err = mean_err_for_choice(choices)
            cv[name] = {
                "accuracy": float(accuracy_score(y, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "mean_error": mean_err,
                "median_error": med_err,
                "choice_counts": dict(Counter(choices)),
            }
        result["leave_video_out_cv"] = cv
    except Exception as exc:
        result["sklearn_error"] = repr(exc)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "teacher_selector_a0_features.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "teacher_selector_a0_result.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="outputs/route_sota_recovery_2026-06-27/teacher_selector_labels_from_oracle.jsonl")
    ap.add_argument("--out-dir", default="outputs/route_sota_recovery_2026-06-27")
    args = ap.parse_args()
    label_rows = [json.loads(l) for l in Path(args.labels).read_text().splitlines() if l.strip()]
    caches = {name: load_cache(path) for name, path in CACHE_PATHS.items()}
    rows = build_features(label_rows, caches)
    evaluate(rows, Path(args.out_dir))

if __name__ == "__main__":
    main()
