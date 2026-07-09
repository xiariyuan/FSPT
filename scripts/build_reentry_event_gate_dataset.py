#!/usr/bin/env python3
"""Build an event-level no-action gate dataset for ReEntry-VisCalibrator V2.1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.reentry_viscalibrator import build_model_from_checkpoint_payload  # noqa: E402
from utils.reentry_event_gate_features import (  # noqa: E402
    EVENT_FEATURE_NAMES,
    build_event_features,
    compute_recovery_utility,
)
from utils.reentry_viscalibrator_features import (  # noqa: E402
    CandidateConfig,
    check_alignment,
    find_candidate_triggers,
    build_candidate_example,
    npy,
)


def load_cache(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ReEntry event-gate dataset")
    ap.add_argument("--base-cache", required=True)
    ap.add_argument("--override-cache", required=True)
    ap.add_argument("--v1-model", required=True)
    ap.add_argument("--v1-threshold", type=float, default=0.10)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--setting", default="unknown")
    ap.add_argument("--require-override-visible", action="store_true", default=True)
    ap.add_argument("--no-require-override-visible", dest="require_override_visible", action="store_false")
    ap.add_argument("--context-before", type=int, default=16)
    ap.add_argument("--context-after", type=int, default=16)
    ap.add_argument("--candidate-pre", type=int, default=1)
    ap.add_argument("--candidate-post", type=int, default=16)
    ap.add_argument("--trigger-k", type=int, default=1)
    ap.add_argument("--trigger-persist", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    base = load_cache(args.base_cache)
    override = load_cache(args.override_cache)
    check_alignment(base, override, "override")
    ckpt = torch.load(args.v1_model, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint_payload(ckpt).to(args.device).eval()
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    cfg = CandidateConfig(
        context_before=args.context_before,
        context_after=args.context_after,
        candidate_pre=args.candidate_pre,
        candidate_post=args.candidate_post,
        trigger_k=args.trigger_k,
        trigger_persist=args.trigger_persist,
    )

    xs: List[np.ndarray] = []
    ys: List[int] = []
    weights: List[float] = []
    utilities: List[float] = []
    meta: List[Dict[str, Any]] = []

    for br, orr in zip(base["records"], override["records"]):
        base_vis = npy(br["pred_visibility"], bool)
        override_vis = npy(orr["pred_visibility"], bool)
        qpts = npy(br["query_points"], np.float32)
        n_tracks, t_len = base_vis.shape
        for qi in range(n_tracks):
            query_t = max(0, min(t_len - 1, int(round(float(qpts[qi, 0])))))
            triggers = find_candidate_triggers(base_vis[qi], override_vis[qi], query_t, cfg)
            for trigger_t in triggers:
                ex = build_candidate_example(br, orr, qi, int(trigger_t), cfg)
                xf = (ex["x"].astype(np.float32) - mean[None, :]) / std[None, :]
                xf[~ex["valid_mask"]] = 0.0
                xb = torch.from_numpy(xf[None]).to(args.device).float()
                valid = torch.from_numpy(ex["valid_mask"][None]).to(args.device).bool()
                with torch.no_grad():
                    prob = torch.sigmoid(model(xb, valid_mask=valid))[0].detach().cpu().numpy()
                evx = build_event_features(
                    ex,
                    prob,
                    threshold=args.v1_threshold,
                    require_override_visible=args.require_override_visible,
                )
                u = compute_recovery_utility(
                    ex,
                    prob,
                    threshold=args.v1_threshold,
                    require_override_visible=args.require_override_visible,
                )
                xs.append(evx)
                ys.append(1 if u["label_allow"] else 0)
                weights.append(float(u["weight"]))
                utilities.append(float(u["utility"]))
                meta.append({
                    "video_id": str(br["video_id"]),
                    "query_idx": int(qi),
                    "query_t": int(query_t),
                    "trigger_t": int(trigger_t),
                    **{k: v for k, v in u.items() if k != "label_allow"},
                })

    X = np.stack(xs, axis=0).astype(np.float32) if xs else np.zeros((0, len(EVENT_FEATURE_NAMES)), dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32)
    util = np.asarray(utilities, dtype=np.float32)
    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=X,
        y=y,
        weight=w,
        utility=util,
        feature_names=np.asarray(EVENT_FEATURE_NAMES, dtype=object),
        meta_json=np.asarray([json.dumps(m, ensure_ascii=False) for m in meta], dtype=object),
        setting=np.asarray(args.setting, dtype=object),
        base_cache=np.asarray(args.base_cache, dtype=object),
        override_cache=np.asarray(args.override_cache, dtype=object),
        v1_model=np.asarray(args.v1_model, dtype=object),
        v1_threshold=np.asarray(float(args.v1_threshold), dtype=np.float32),
        require_override_visible=np.asarray(bool(args.require_override_visible), dtype=bool),
    )
    summary = {
        "out_npz": str(out),
        "setting": args.setting,
        "n_examples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "positive_rate": float(y.mean()) if y.size else 0.0,
        "mean_utility": float(util.mean()) if util.size else 0.0,
        "sum_positive_utility": float(util[util > 0].sum()) if util.size else 0.0,
        "sum_negative_utility": float(util[util < 0].sum()) if util.size else 0.0,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
