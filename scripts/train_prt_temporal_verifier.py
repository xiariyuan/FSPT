#!/usr/bin/env python3
"""
Train a minimal temporal verifier on cached PRT candidate data.

This is intentionally lightweight. It does not use raw video again; instead it
consumes the cached candidate dataset built by `scripts/build_prt_candidate_dataset.py`
and tests the core hypothesis:

  Does short-window temporal evidence help choose between
  baseline / candidate / abstain better than single-frame selection?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _track_stats(track_xy: np.ndarray, track_score: np.ndarray) -> np.ndarray:
    """Causal trajectory statistics that do not use GT-derived errors."""
    steps = track_xy[1:] - track_xy[:-1]
    step_l2 = np.linalg.norm(steps, axis=1) if len(steps) > 0 else np.zeros(1, dtype=np.float32)
    total_disp = float(np.linalg.norm(track_xy[-1] - track_xy[0])) if len(track_xy) > 1 else 0.0
    path_len = float(step_l2.sum())
    path_ratio = float(path_len / (total_disp + 1e-6))

    if len(track_xy) >= 3:
        accel = (track_xy[2:] - track_xy[1:-1]) - (track_xy[1:-1] - track_xy[:-2])
        accel_l2 = np.linalg.norm(accel, axis=1)
    else:
        accel_l2 = np.zeros(1, dtype=np.float32)

    return np.array(
        [
            float(np.mean(step_l2)),
            float(np.max(step_l2)),
            float(np.std(step_l2)),
            path_len,
            total_disp,
            path_ratio,
            float(np.mean(accel_l2)),
            float(np.max(accel_l2)),
            float(np.mean(track_score)),
            float(np.min(track_score)),
            float(np.std(track_score)),
        ],
        dtype=np.float32,
    )


def build_feature_matrix(cache: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    cand_score = cache["cand_score"]
    cand_ncc = cache["cand_ncc"]
    cand_dist = cache["cand_dist_norm"]
    label_index = cache["label_index"]
    occ_length = cache["occ_length"]
    camera_motion = cache["camera_motion"]
    reentry_type_id = cache["reentry_type_id"]
    baseline_xy = cache["baseline_xy"]
    cand_xy = cache["cand_xy"]
    baseline_track_xy = cache["baseline_track_xy"]
    cand_track_xy = cache["cand_track_xy"]
    baseline_track_score = cache["baseline_track_score"]
    cand_track_score = cache["cand_track_score"]

    n, k = cand_xy.shape[:2]
    feat_list: List[np.ndarray] = []
    label_list: List[int] = []

    for idx in range(n):
        b_stats = _track_stats(baseline_track_xy[idx], baseline_track_score[idx])
        for cand_idx in range(k):
            c_stats = _track_stats(cand_track_xy[idx, cand_idx], cand_track_score[idx, cand_idx])
            init_gap = float(np.linalg.norm(cand_xy[idx, cand_idx] - baseline_xy[idx]))
            track_div = np.linalg.norm(
                cand_track_xy[idx, cand_idx] - baseline_track_xy[idx],
                axis=1,
            )

            feat = np.array(
                [
                    init_gap / 128.0,
                    float(cand_score[idx, cand_idx]),
                    float(cand_ncc[idx, cand_idx]),
                    float(cand_dist[idx, cand_idx]),
                    b_stats[0] / 128.0,
                    b_stats[1] / 128.0,
                    b_stats[2] / 128.0,
                    b_stats[3] / 128.0,
                    b_stats[4] / 128.0,
                    b_stats[5],
                    b_stats[6] / 128.0,
                    b_stats[7] / 128.0,
                    b_stats[8],
                    b_stats[9],
                    b_stats[10],
                    c_stats[0] / 128.0,
                    c_stats[1] / 128.0,
                    c_stats[2] / 128.0,
                    c_stats[3] / 128.0,
                    c_stats[4] / 128.0,
                    c_stats[5],
                    c_stats[6] / 128.0,
                    c_stats[7] / 128.0,
                    c_stats[8],
                    c_stats[9],
                    c_stats[10],
                    (c_stats[0] - b_stats[0]) / 128.0,
                    (c_stats[1] - b_stats[1]) / 128.0,
                    (c_stats[3] - b_stats[3]) / 128.0,
                    (c_stats[4] - b_stats[4]) / 128.0,
                    float(c_stats[8] - b_stats[8]),
                    float(np.mean(track_div) / 128.0),
                    float(np.max(track_div) / 128.0),
                    float(occ_length[idx] / 300.0),
                    float(camera_motion[idx]),
                    float(reentry_type_id[idx] / 2.0),
                ],
                dtype=np.float32,
            )
            feat_list.append(feat)

            # Supervision can use GT-derived labels; features above remain causal.
            label = 1 if label_index[idx] == (cand_idx + 1) else 0
            label_list.append(label)

    return np.stack(feat_list, axis=0), np.asarray(label_list, dtype=np.int64)


class TemporalAcceptor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def evaluate_acceptor(
    model: TemporalAcceptor,
    feats_by_sample: np.ndarray,
    labels_by_sample: np.ndarray,
    cache: Dict[str, np.ndarray],
    device: torch.device,
    threshold: float,
) -> Dict[str, float]:
    model.eval()
    n, k, d = feats_by_sample.shape
    pred_errs = []
    base_errs = []
    oracle_errs = []
    accept_rates = []
    better_fracs = []

    with torch.no_grad():
        flat_feats = torch.from_numpy(feats_by_sample.reshape(n * k, d)).to(device)
        probs = torch.sigmoid(model(flat_feats)).reshape(n, k).cpu().numpy()

    baseline_err = cache["baseline_err"]
    cand_err = cache["cand_err"]
    label_index = cache["label_index"]

    for i in range(n):
        probs_i = probs[i]
        cand_err_i = cand_err[i]
        base_err_i = float(baseline_err[i])
        accept_mask = probs_i >= threshold
        if accept_mask.any():
            chosen = int(np.argmax(np.where(accept_mask, probs_i, -1e9)))
            pred_err = float(cand_err_i[chosen])
            accept_rates.append(1.0)
        else:
            pred_err = base_err_i
            accept_rates.append(0.0)
        oracle_err = min(base_err_i, float(cand_err_i.min()))
        pred_errs.append(pred_err)
        base_errs.append(base_err_i)
        oracle_errs.append(oracle_err)
        better_fracs.append(float(pred_err < base_err_i))

    pred = np.asarray(pred_errs, dtype=np.float32)
    base = np.asarray(base_errs, dtype=np.float32)
    oracle = np.asarray(oracle_errs, dtype=np.float32)
    return {
        "pred_median_px": float(np.median(pred)),
        "baseline_median_px": float(np.median(base)),
        "oracle_median_px": float(np.median(oracle)),
        "pred_lt4px": float(np.mean(pred < 4.0)),
        "baseline_lt4px": float(np.mean(base < 4.0)),
        "oracle_lt4px": float(np.mean(oracle < 4.0)),
        "better_frac": float(np.mean(pred < base)),
        "accept_rate": float(np.mean(accept_rates)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train minimal PRT temporal verifier")
    parser.add_argument("--train-cache", type=str, required=True)
    parser.add_argument("--val-cache", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cache = dict(np.load(args.train_cache))
    val_cache = dict(np.load(args.val_cache))

    train_x_flat, train_y_flat = build_feature_matrix(train_cache)
    val_x_flat, val_y_flat = build_feature_matrix(val_cache)

    k = train_cache["cand_err"].shape[1]
    d = train_x_flat.shape[1]
    train_n = train_cache["baseline_err"].shape[0]
    val_n = val_cache["baseline_err"].shape[0]

    train_x = train_x_flat.reshape(train_n, k, d)
    train_y = train_y_flat.reshape(train_n, k)
    val_x = val_x_flat.reshape(val_n, k, d)
    val_y = val_y_flat.reshape(val_n, k)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalAcceptor(in_dim=d).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history: List[Dict[str, float]] = []
    best_metric = float("inf")
    bad_epochs = 0

    flat_train_x = train_x.reshape(train_n * k, d)
    flat_train_y = train_y.reshape(train_n * k).astype(np.float32)

    pos_weight = torch.tensor([max(1.0, float((flat_train_y == 0).sum() / max((flat_train_y == 1).sum(), 1)))], device=device)

    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(flat_train_x))
        losses = []
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            x = torch.from_numpy(flat_train_x[idx]).to(device)
            y = torch.from_numpy(flat_train_y[idx]).to(device)
            logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics = evaluate_acceptor(
            model=model,
            feats_by_sample=val_x,
            labels_by_sample=val_y,
            cache=val_cache,
            device=device,
            threshold=args.threshold,
        )
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = float(np.mean(losses))
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=True), flush=True)

        if metrics["pred_median_px"] < best_metric:
            best_metric = metrics["pred_median_px"]
            bad_epochs = 0
            torch.save(model.state_dict(), out_dir / "temporal_acceptor_best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    (out_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
    best = min(history, key=lambda x: x["pred_median_px"])
    (out_dir / "best.json").write_text(json.dumps(best, indent=2) + "\n")
    print(f"Best epoch: {json.dumps(best, ensure_ascii=True)}", flush=True)


if __name__ == "__main__":
    main()
