#!/usr/bin/env python3
"""Build V2.3 frame-keep/risk-decoder dataset.

Samples are built only on frames that V1 proposes to recover.  The model learns
whether a proposed recovery frame should be kept or dropped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.reentry_viscalibrator import build_model_from_checkpoint_payload  # noqa: E402
from utils.reentry_event_gate_features import build_event_features  # noqa: E402
from utils.reentry_frame_keep_features import (  # noqa: E402
    FRAME_KEEP_FEATURE_NAMES,
    build_frame_keep_samples,
)
from utils.reentry_viscalibrator_features import (  # noqa: E402
    CandidateConfig,
    build_candidate_example,
    check_alignment,
    find_candidate_triggers,
    npy,
)


class EventGateMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_cache(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_gate_from_ckpt(ckpt: Dict[str, Any], device: torch.device) -> EventGateMLP:
    cfg = ckpt.get("model_config", {})
    model = EventGateMLP(int(cfg.get("in_dim", len(ckpt["feature_mean"]))), int(cfg.get("hidden", 128)), float(cfg.get("dropout", 0.1)))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build V2.3 frame-keep dataset")
    ap.add_argument("--base-cache", required=True)
    ap.add_argument("--override-cache", required=True)
    ap.add_argument("--v1-model", required=True)
    ap.add_argument("--gate-model", default="")
    ap.add_argument("--v1-threshold", type=float, default=0.10)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--setting", default="unknown")
    ap.add_argument("--include-base-visible", action="store_true")
    ap.add_argument("--no-require-override-visible", action="store_true")
    ap.add_argument("--context-before", type=int, default=16)
    ap.add_argument("--context-after", type=int, default=16)
    ap.add_argument("--candidate-pre", type=int, default=1)
    ap.add_argument("--candidate-post", type=int, default=16)
    ap.add_argument("--trigger-k", type=int, default=1)
    ap.add_argument("--trigger-persist", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    base = load_cache(args.base_cache)
    override = load_cache(args.override_cache)
    check_alignment(base, override, "override")
    v1_ckpt = torch.load(args.v1_model, map_location="cpu", weights_only=False)
    v1_model = build_model_from_checkpoint_payload(v1_ckpt).to(device).eval()
    v1_mean = np.asarray(v1_ckpt["feature_mean"], dtype=np.float32)
    v1_std = np.asarray(v1_ckpt["feature_std"], dtype=np.float32)

    gate_model = None
    gate_mean = gate_std = None
    if args.gate_model:
        gate_ckpt = torch.load(args.gate_model, map_location="cpu", weights_only=False)
        gate_model = build_gate_from_ckpt(gate_ckpt, device)
        gate_mean = np.asarray(gate_ckpt["feature_mean"], dtype=np.float32)
        gate_std = np.asarray(gate_ckpt["feature_std"], dtype=np.float32)

    cfg = CandidateConfig(
        context_before=args.context_before,
        context_after=args.context_after,
        candidate_pre=args.candidate_pre,
        candidate_post=args.candidate_post,
        trigger_k=args.trigger_k,
        trigger_persist=args.trigger_persist,
    )
    require_override_visible = not args.no_require_override_visible

    Xs: List[np.ndarray] = []
    y_gt_visible: List[np.ndarray] = []
    y_safe16: List[np.ndarray] = []
    y_util: List[np.ndarray] = []
    y_safe8: List[np.ndarray] = []
    y_safe4: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    utilities: List[np.ndarray] = []
    meta_json: List[str] = []
    total_candidate_windows = 0
    total_raw_recovery_frames = 0
    total_changed_recovery_frames = 0
    total_candidate_frames = 0

    for br, orr in zip(base["records"], override["records"]):
        base_vis = npy(br["pred_visibility"], bool)
        override_vis = npy(orr["pred_visibility"], bool)
        qpts = npy(br["query_points"], np.float32)
        n_tracks, t_len = base_vis.shape
        for qi in range(n_tracks):
            query_t = max(0, min(t_len - 1, int(round(float(qpts[qi, 0])))))
            triggers = find_candidate_triggers(base_vis[qi], override_vis[qi], query_t, cfg)
            for trigger_t in triggers:
                total_candidate_windows += 1
                ex = build_candidate_example(br, orr, qi, int(trigger_t), cfg)
                xf = (ex["x"].astype(np.float32) - v1_mean[None, :]) / v1_std[None, :]
                xf[~ex["valid_mask"]] = 0.0
                xb = torch.from_numpy(xf[None]).to(device).float()
                valid = torch.from_numpy(ex["valid_mask"][None]).to(device).bool()
                with torch.no_grad():
                    prob = torch.sigmoid(v1_model(xb, valid_mask=valid))[0].detach().cpu().numpy()

                gate_prob = None
                if gate_model is not None:
                    evx = build_event_features(ex, prob, threshold=args.v1_threshold, require_override_visible=require_override_visible)
                    evxn = (evx.astype(np.float32) - gate_mean) / gate_std
                    with torch.no_grad():
                        gate_prob = float(torch.sigmoid(gate_model(torch.from_numpy(evxn[None]).to(device).float()))[0].detach().cpu())

                pack = build_frame_keep_samples(
                    ex,
                    prob,
                    threshold=float(args.v1_threshold),
                    require_override_visible=require_override_visible,
                    gate_prob=gate_prob,
                    include_base_visible=bool(args.include_base_visible),
                )
                total_raw_recovery_frames += int(pack["raw_recovery_frames"])
                total_changed_recovery_frames += int(pack["changed_recovery_frames"])
                total_candidate_frames += int(pack["candidate_frames"])
                if pack["X"].shape[0] == 0:
                    continue
                Xs.append(pack["X"])
                y_gt_visible.append(pack["y_gt_visible"])
                y_safe16.append(pack["y_safe16"])
                y_util.append(pack["y_utility"])
                y_safe8.append(pack["y_safe8"])
                y_safe4.append(pack["y_safe4"])
                weights.append(pack["weight"])
                utilities.append(pack["utility"])
                for m in pack["meta"]:
                    meta_json.append(json.dumps(m, ensure_ascii=False))

    X = np.concatenate(Xs, axis=0).astype(np.float32) if Xs else np.zeros((0, len(FRAME_KEEP_FEATURE_NAMES)), dtype=np.float32)
    ygv = np.concatenate(y_gt_visible, axis=0).astype(np.float32) if y_gt_visible else np.zeros(0, dtype=np.float32)
    y16 = np.concatenate(y_safe16, axis=0).astype(np.float32) if y_safe16 else np.zeros(0, dtype=np.float32)
    yu = np.concatenate(y_util, axis=0).astype(np.float32) if y_util else np.zeros(0, dtype=np.float32)
    ys = np.concatenate(y_safe8, axis=0).astype(np.float32) if y_safe8 else np.zeros(0, dtype=np.float32)
    y4 = np.concatenate(y_safe4, axis=0).astype(np.float32) if y_safe4 else np.zeros(0, dtype=np.float32)
    w = np.concatenate(weights, axis=0).astype(np.float32) if weights else np.zeros(0, dtype=np.float32)
    util = np.concatenate(utilities, axis=0).astype(np.float32) if utilities else np.zeros(0, dtype=np.float32)

    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=X,
        y_gt_visible=ygv,
        y_safe16=y16,
        y_utility=yu,
        y_safe8=ys,
        y_safe4=y4,
        weight=w,
        utility=util,
        feature_names=np.asarray(FRAME_KEEP_FEATURE_NAMES, dtype=object),
        meta_json=np.asarray(meta_json, dtype=object),
        setting=np.asarray(args.setting, dtype=object),
        base_cache=np.asarray(args.base_cache, dtype=object),
        override_cache=np.asarray(args.override_cache, dtype=object),
        v1_model=np.asarray(args.v1_model, dtype=object),
        gate_model=np.asarray(args.gate_model, dtype=object),
        v1_threshold=np.asarray(float(args.v1_threshold), dtype=np.float32),
        require_override_visible=np.asarray(bool(require_override_visible), dtype=bool),
        include_base_visible=np.asarray(bool(args.include_base_visible), dtype=bool),
    )
    summary = {
        "out_npz": str(out),
        "setting": args.setting,
        "n_samples": int(X.shape[0]),
        "feature_dim": int(X.shape[1]),
        "positive_rate_gt_visible": float(ygv.mean()) if ygv.size else 0.0,
        "positive_rate_safe16": float(y16.mean()) if y16.size else 0.0,
        "positive_rate_utility": float(yu.mean()) if yu.size else 0.0,
        "positive_rate_safe8": float(ys.mean()) if ys.size else 0.0,
        "positive_rate_safe4": float(y4.mean()) if y4.size else 0.0,
        "mean_utility": float(util.mean()) if util.size else 0.0,
        "candidate_windows": int(total_candidate_windows),
        "candidate_frames": int(total_candidate_frames),
        "raw_recovery_frames": int(total_raw_recovery_frames),
        "changed_recovery_frames": int(total_changed_recovery_frames),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
