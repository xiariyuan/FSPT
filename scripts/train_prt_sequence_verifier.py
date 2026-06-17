#!/usr/bin/env python3
"""
Train a sequence-level verifier over candidate tracklets for PRT.

This script is the next step after the lightweight temporal-statistics acceptor.
It consumes cached PRT candidate datasets and learns from:

  - short-window candidate/baseline tracklet sequences
  - static re-entry patches (query / baseline / candidate)
  - causal candidate metadata (score, NCC, distance, occlusion, camera motion)

Important:
  - GT-derived errors are used only for supervision/evaluation.
  - No GT-derived error is used as model input.
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


CAUSAL_FEATURE_AUDIT = {
    "sequence_token_features": [
        "baseline_rel_xy",
        "candidate_rel_xy",
        "baseline_step_xy",
        "candidate_step_xy",
        "candidate_minus_baseline_xy",
        "baseline_track_score",
        "candidate_track_score",
    ],
    "global_features": [
        "candidate_singleframe_dino_score",
        "candidate_singleframe_ncc",
        "candidate_singleframe_dist_norm",
        "candidate_initial_gap_from_baseline",
        "occlusion_length",
        "camera_motion",
        "reentry_type_onehot",
    ],
    "static_patch_inputs": [
        "query_patch",
        "baseline_patch",
        "candidate_patch",
    ],
    "excluded_gt_features": [
        "baseline_err",
        "cand_err",
        "baseline_track_err",
        "cand_track_err",
        "gt_reentry_xy",
        "gt_window_xy",
        "gt_window_vis",
    ],
}


def build_sequence_token_features(
    baseline_track_xy: np.ndarray,
    candidate_track_xy: np.ndarray,
    baseline_track_score: np.ndarray,
    candidate_track_score: np.ndarray,
    coord_scale: float = 128.0,
) -> np.ndarray:
    """Build causal per-timestep token features for one candidate."""
    b_rel = (baseline_track_xy - baseline_track_xy[0:1]) / coord_scale
    c_rel = (candidate_track_xy - candidate_track_xy[0:1]) / coord_scale

    b_step = np.zeros_like(b_rel)
    c_step = np.zeros_like(c_rel)
    if len(b_rel) > 1:
        b_step[1:] = b_rel[1:] - b_rel[:-1]
        c_step[1:] = c_rel[1:] - c_rel[:-1]

    diff = (candidate_track_xy - baseline_track_xy) / coord_scale
    b_score = baseline_track_score[:, None]
    c_score = candidate_track_score[:, None]
    return np.concatenate([b_rel, c_rel, b_step, c_step, diff, b_score, c_score], axis=-1).astype(np.float32)


def build_candidate_arrays(cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    query_patch = cache["query_patch"]
    baseline_patch = cache["baseline_patch"]
    cand_patches = cache["cand_patches"]
    baseline_xy = cache["baseline_xy"]
    cand_xy = cache["cand_xy"]
    cand_score = cache["cand_score"]
    cand_ncc = cache["cand_ncc"]
    cand_dist = cache["cand_dist_norm"]
    occ_length = cache["occ_length"]
    camera_motion = cache["camera_motion"]
    reentry_type_id = cache["reentry_type_id"]
    baseline_track_xy = cache["baseline_track_xy"]
    cand_track_xy = cache["cand_track_xy"]
    baseline_track_score = cache["baseline_track_score"]
    cand_track_score = cache["cand_track_score"]
    label_index = cache["label_index"]

    n, k = cand_xy.shape[:2]
    seq_tokens: List[np.ndarray] = []
    global_feats: List[np.ndarray] = []
    query_patches: List[np.ndarray] = []
    baseline_patches: List[np.ndarray] = []
    candidate_patches: List[np.ndarray] = []
    labels: List[int] = []
    sample_ids: List[int] = []
    candidate_ids: List[int] = []

    for i in range(n):
        type_onehot = np.zeros(3, dtype=np.float32)
        type_onehot[int(np.clip(reentry_type_id[i], 0, 2))] = 1.0
        for j in range(k):
            seq_tokens.append(
                build_sequence_token_features(
                    baseline_track_xy=baseline_track_xy[i],
                    candidate_track_xy=cand_track_xy[i, j],
                    baseline_track_score=baseline_track_score[i],
                    candidate_track_score=cand_track_score[i, j],
                )
            )
            init_gap = float(np.linalg.norm(cand_xy[i, j] - baseline_xy[i])) / 128.0
            global_feats.append(
                np.concatenate(
                    [
                        np.array(
                            [
                                float(cand_score[i, j]),
                                float(cand_ncc[i, j]),
                                float(cand_dist[i, j]),
                                init_gap,
                                float(occ_length[i] / 300.0),
                                float(camera_motion[i]),
                            ],
                            dtype=np.float32,
                        ),
                        type_onehot,
                    ],
                    axis=0,
                )
            )
            query_patches.append(query_patch[i].astype(np.float32))
            baseline_patches.append(baseline_patch[i].astype(np.float32))
            candidate_patches.append(cand_patches[i, j].astype(np.float32))
            labels.append(1 if label_index[i] == (j + 1) else 0)
            sample_ids.append(i)
            candidate_ids.append(j)

    return {
        "seq_tokens": np.stack(seq_tokens, axis=0),
        "global_feats": np.stack(global_feats, axis=0),
        "query_patch": np.stack(query_patches, axis=0),
        "baseline_patch": np.stack(baseline_patches, axis=0),
        "candidate_patch": np.stack(candidate_patches, axis=0),
        "label": np.asarray(labels, dtype=np.float32),
        "sample_id": np.asarray(sample_ids, dtype=np.int32),
        "candidate_id": np.asarray(candidate_ids, dtype=np.int32),
    }


class PatchEncoder(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x).flatten(1))


class SequenceTrackletVerifier(nn.Module):
    def __init__(
        self,
        token_dim: int,
        global_dim: int,
        patch_dim: int = 64,
        hidden_dim: int = 128,
        gru_dim: int = 128,
    ) -> None:
        super().__init__()
        self.patch_encoder = PatchEncoder(out_dim=patch_dim)
        self.token_proj = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(hidden_dim, gru_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(gru_dim + global_dim + patch_dim * 3 + 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        seq_tokens: torch.Tensor,
        global_feats: torch.Tensor,
        query_patch: torch.Tensor,
        baseline_patch: torch.Tensor,
        candidate_patch: torch.Tensor,
    ) -> torch.Tensor:
        q = F.normalize(self.patch_encoder(query_patch), dim=-1)
        b = F.normalize(self.patch_encoder(baseline_patch), dim=-1)
        c = F.normalize(self.patch_encoder(candidate_patch), dim=-1)
        patch_sims = torch.stack(
            [
                torch.sum(q * c, dim=-1),
                torch.sum(q * b, dim=-1),
                torch.sum(c * b, dim=-1),
            ],
            dim=-1,
        )

        x = self.token_proj(seq_tokens)
        _, h = self.gru(x)
        seq_embed = h[-1]

        fused = torch.cat([seq_embed, global_feats, q, b, c, patch_sims], dim=-1)
        return self.head(fused).squeeze(-1)


def evaluate_model(
    model: SequenceTrackletVerifier,
    packed: Dict[str, np.ndarray],
    raw_cache: Dict[str, np.ndarray],
    device: torch.device,
    threshold: float,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        x_seq = torch.from_numpy(packed["seq_tokens"]).to(device)
        x_global = torch.from_numpy(packed["global_feats"]).to(device)
        x_q = torch.from_numpy(packed["query_patch"]).to(device)
        x_b = torch.from_numpy(packed["baseline_patch"]).to(device)
        x_c = torch.from_numpy(packed["candidate_patch"]).to(device)
        probs = torch.sigmoid(model(x_seq, x_global, x_q, x_b, x_c)).cpu().numpy()

    sample_ids = packed["sample_id"]
    candidate_ids = packed["candidate_id"]
    n = raw_cache["baseline_err"].shape[0]
    k = raw_cache["cand_err"].shape[1]
    prob_matrix = np.full((n, k), -1e9, dtype=np.float32)
    for p, sid, cid in zip(probs, sample_ids, candidate_ids):
        prob_matrix[int(sid), int(cid)] = float(p)

    baseline_err = raw_cache["baseline_err"]
    cand_err = raw_cache["cand_err"]

    pred_errs = []
    accept_rates = []
    for i in range(n):
        probs_i = prob_matrix[i]
        accept_mask = probs_i >= threshold
        if accept_mask.any():
            j = int(np.argmax(np.where(accept_mask, probs_i, -1e9)))
            pred_errs.append(float(cand_err[i, j]))
            accept_rates.append(1.0)
        else:
            pred_errs.append(float(baseline_err[i]))
            accept_rates.append(0.0)

    pred = np.asarray(pred_errs, dtype=np.float32)
    base = np.asarray(baseline_err, dtype=np.float32)
    oracle = np.minimum(base, cand_err.min(axis=1))
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
    parser = argparse.ArgumentParser(description="Train PRT sequence-level verifier")
    parser.add_argument("--train-cache", type=str, required=True)
    parser.add_argument("--val-cache", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_audit.json").write_text(json.dumps(CAUSAL_FEATURE_AUDIT, indent=2) + "\n")
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    train_raw = dict(np.load(args.train_cache))
    val_raw = dict(np.load(args.val_cache))
    train_pack = build_candidate_arrays(train_raw)
    val_pack = build_candidate_arrays(val_raw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_dim = train_pack["seq_tokens"].shape[-1]
    global_dim = train_pack["global_feats"].shape[-1]
    model = SequenceTrackletVerifier(token_dim=token_dim, global_dim=global_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_y = train_pack["label"]
    pos_count = max(float((train_y == 1).sum()), 1.0)
    neg_count = max(float((train_y == 0).sum()), 1.0)
    pos_weight = torch.tensor([neg_count / pos_count], device=device)

    history: List[Dict[str, float]] = []
    best_metric = float("inf")
    bad_epochs = 0
    n_train = len(train_y)

    for epoch in range(args.epochs):
        model.train()
        order = np_rng.permutation(n_train)
        losses = []
        for start in range(0, n_train, args.batch_size):
            idx = order[start : start + args.batch_size]
            x_seq = torch.from_numpy(train_pack["seq_tokens"][idx]).to(device)
            x_global = torch.from_numpy(train_pack["global_feats"][idx]).to(device)
            x_q = torch.from_numpy(train_pack["query_patch"][idx]).to(device)
            x_b = torch.from_numpy(train_pack["baseline_patch"][idx]).to(device)
            x_c = torch.from_numpy(train_pack["candidate_patch"][idx]).to(device)
            y = torch.from_numpy(train_pack["label"][idx]).to(device)

            logit = model(x_seq, x_global, x_q, x_b, x_c)
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics = evaluate_model(
            model=model,
            packed=val_pack,
            raw_cache=val_raw,
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
            torch.save(model.state_dict(), out_dir / "sequence_verifier_best.pt")
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
