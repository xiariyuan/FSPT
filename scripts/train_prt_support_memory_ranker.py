#!/usr/bin/env python3
"""
Train a minimal support-memory ranker for PRT causal re-entry.

Inputs:
  - support patches from pre-occlusion visible frames
  - baseline patch at re-entry
  - top-k candidate patches at re-entry
  - lightweight metadata (score / NCC / distance / occ_len / cam_motion / type)

Output:
  - multiclass prediction over {baseline, cand1..K}
  - optional abstain is handled later via score-margin calibration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_world_state_stage2_causal_dino import DINOFeatureExtractor


def load_cache(path: Path) -> Dict[str, np.ndarray]:
    cache = np.load(path, allow_pickle=False)
    return {k: cache[k] for k in cache.files}


def encode_patch_batch(
    extractor: DINOFeatureExtractor,
    patches_np: np.ndarray,
    batch_size: int = 32,
) -> torch.Tensor:
    out = []
    for start in range(0, len(patches_np), batch_size):
        batch = torch.from_numpy(patches_np[start:start + batch_size]).float()
        batch = F.interpolate(batch, size=(518, 518), mode="bilinear", align_corners=False).to(extractor.device)
        with torch.no_grad():
            feat = extractor.model(batch)[-1].mean(dim=[-2, -1]).float()
            feat = F.normalize(feat, dim=-1)
        out.append(feat.cpu())
    return torch.cat(out, dim=0)


def build_feature_table(cache: Dict[str, np.ndarray], extractor: DINOFeatureExtractor) -> Dict[str, np.ndarray]:
    n = cache["query_patch"].shape[0]
    topk = cache["cand_patches"].shape[1]

    query_feat = encode_patch_batch(extractor, cache["query_patch"]).numpy()
    baseline_feat = encode_patch_batch(extractor, cache["baseline_patch"]).numpy()

    cand_flat = cache["cand_patches"].reshape(n * topk, *cache["cand_patches"].shape[2:])
    cand_feat = encode_patch_batch(extractor, cand_flat).numpy().reshape(n, topk, -1)

    # Support features: use dedicated support_patches if available,
    # otherwise fall back to query_patch as single-frame support memory
    if "support_patches" in cache and "support_count" in cache:
        support_count = cache["support_patches"].shape[1]
        support_flat = cache["support_patches"].reshape(n * support_count, *cache["support_patches"].shape[2:])
        support_feat = encode_patch_batch(extractor, support_flat).numpy().reshape(n, support_count, -1)
        support_mask = (
            np.arange(support_count, dtype=np.int32)[None, :] < cache["support_count"][:, None]
        ).astype(np.float32)
    else:
        # Fallback: use query_patch as single support view
        support_feat = query_feat[:, None, :]  # (n, 1, D)
        support_mask = np.ones((n, 1), dtype=np.float32)

    return {
        "query_feat": query_feat.astype(np.float32),
        "baseline_feat": baseline_feat.astype(np.float32),
        "cand_feat": cand_feat.astype(np.float32),
        "support_feat": support_feat.astype(np.float32),
        "support_mask": support_mask.astype(np.float32),
    }


class SupportMemoryRanker(nn.Module):
    def __init__(self, feat_dim: int = 384, meta_dim: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.support_proj = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 4 + meta_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def pool_support(self, support_feat: torch.Tensor, support_mask: torch.Tensor, query_feat: torch.Tensor) -> torch.Tensor:
        q = query_feat.unsqueeze(1).expand_as(support_feat)
        logits = self.support_proj(torch.cat([support_feat, q], dim=-1))
        logits = (logits * support_feat).sum(dim=-1)
        logits = logits.masked_fill(support_mask < 0.5, -1e4)
        attn = torch.softmax(logits, dim=-1).unsqueeze(-1)
        pooled = (attn * support_feat).sum(dim=1)
        return F.normalize(pooled, dim=-1)

    def forward(
        self,
        query_feat: torch.Tensor,
        baseline_feat: torch.Tensor,
        support_feat: torch.Tensor,
        support_mask: torch.Tensor,
        candidate_feat: torch.Tensor,
        candidate_meta: torch.Tensor,
    ) -> torch.Tensor:
        pooled_support = self.pool_support(support_feat, support_mask, query_feat)
        q = query_feat.unsqueeze(1).expand_as(candidate_feat)
        b = baseline_feat.unsqueeze(1).expand_as(candidate_feat)
        s = pooled_support.unsqueeze(1).expand_as(candidate_feat)
        x = torch.cat([q, b, s, candidate_feat, candidate_meta], dim=-1)
        return self.head(x).squeeze(-1)


def make_candidate_tensor(cache: Dict[str, np.ndarray], feat_table: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    n = cache["query_patch"].shape[0]
    topk = cache["cand_patches"].shape[1]
    d = feat_table["query_feat"].shape[1]

    candidate_feat = np.zeros((n, topk + 1, d), dtype=np.float32)
    candidate_feat[:, 0, :] = feat_table["baseline_feat"]
    candidate_feat[:, 1:, :] = feat_table["cand_feat"]

    support_feat = feat_table["support_feat"]
    support_mask = feat_table["support_mask"]
    support_mask_sum = np.clip(support_mask.sum(axis=1, keepdims=True), 1.0, None)
    pooled_support = (support_feat * support_mask[:, :, None]).sum(axis=1) / support_mask_sum
    pooled_support /= np.linalg.norm(pooled_support, axis=1, keepdims=True).clip(1e-8, None)

    baseline_support_sim = np.sum(feat_table["baseline_feat"] * pooled_support, axis=1).astype(np.float32)
    cand_support_sim = np.sum(feat_table["cand_feat"] * pooled_support[:, None, :], axis=2).astype(np.float32)
    cand_support_margin = cand_support_sim - baseline_support_sim[:, None]

    meta_dim = 9
    candidate_meta = np.zeros((n, topk + 1, meta_dim), dtype=np.float32)
    # Baseline slot (index 0): is_baseline flag + support similarity + causal metadata.
    candidate_meta[:, 0, 0] = 1.0
    candidate_meta[:, 0, 6] = baseline_support_sim
    candidate_meta[:, 0, 7] = baseline_support_sim
    candidate_meta[:, 0, 8] = 0.0
    candidate_meta[:, 0, 4] = cache["occ_length"] / 300.0
    candidate_meta[:, 0, 5] = cache["camera_motion"]

    # Candidate slots (index 1..K): DINO score, NCC, dist, occ, cam_motion,
    # plus explicit support-memory similarity features.
    candidate_meta[:, 1:, 1] = cache["cand_score"]
    candidate_meta[:, 1:, 2] = cache["cand_ncc"]
    candidate_meta[:, 1:, 3] = cache["cand_dist_norm"]
    candidate_meta[:, 1:, 4] = cache["occ_length"][:, None] / 300.0
    candidate_meta[:, 1:, 5] = cache["camera_motion"][:, None]
    candidate_meta[:, 1:, 6] = cand_support_sim
    candidate_meta[:, 1:, 7] = baseline_support_sim[:, None]
    candidate_meta[:, 1:, 8] = cand_support_margin

    target = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        baseline_err = float(cache["baseline_err"][i])
        cand_err = cache["cand_err"][i]
        best_idx = int(np.argmin(cand_err))
        if float(cand_err[best_idx]) < baseline_err:
            target[i] = best_idx + 1
        else:
            target[i] = 0

    oracle_err = np.minimum(cache["baseline_err"], cache["cand_err"].min(axis=1))
    return {
        "candidate_feat": candidate_feat,
        "candidate_meta": candidate_meta,
        "target": target,
        "oracle_err": oracle_err.astype(np.float32),
    }


def evaluate(
    model: SupportMemoryRanker,
    tensors: Dict[str, torch.Tensor],
    raw_cache: Dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    pred_errs = []
    base_errs = []
    oracle_errs = []
    chosen = []
    with torch.no_grad():
        n = tensors["query_feat"].shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            logits = model(
                tensors["query_feat"][start:end].to(device),
                tensors["baseline_feat"][start:end].to(device),
                tensors["support_feat"][start:end].to(device),
                tensors["support_mask"][start:end].to(device),
                tensors["candidate_feat"][start:end].to(device),
                tensors["candidate_meta"][start:end].to(device),
            )
            idx = torch.argmax(logits, dim=1).cpu().numpy()
            chosen.append(idx)
            for j, choice in enumerate(idx.tolist(), start=start):
                if choice == 0:
                    pred_errs.append(float(raw_cache["baseline_err"][j]))
                else:
                    pred_errs.append(float(raw_cache["cand_err"][j, choice - 1]))
                base_errs.append(float(raw_cache["baseline_err"][j]))
                oracle_errs.append(float(np.minimum(raw_cache["baseline_err"][j], raw_cache["cand_err"][j].min())))
    pred = np.asarray(pred_errs, dtype=np.float32)
    base = np.asarray(base_errs, dtype=np.float32)
    oracle = np.asarray(oracle_errs, dtype=np.float32)
    chosen_np = np.concatenate(chosen, axis=0)
    return {
        "pred_median_px": float(np.median(pred)),
        "baseline_median_px": float(np.median(base)),
        "oracle_median_px": float(np.median(oracle)),
        "pred_lt4px": float(np.mean(pred < 4.0)),
        "baseline_lt4px": float(np.mean(base < 4.0)),
        "oracle_lt4px": float(np.mean(oracle < 4.0)),
        "better_frac": float(np.mean(pred < base)),
        "choose_baseline_frac": float(np.mean(chosen_np == 0)),
    }


def collect_logits_and_errors(
    model: SupportMemoryRanker,
    tensors: Dict[str, torch.Tensor],
    raw_cache: Dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    logits_all = []
    with torch.no_grad():
        n = tensors["query_feat"].shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            logits = model(
                tensors["query_feat"][start:end].to(device),
                tensors["baseline_feat"][start:end].to(device),
                tensors["support_feat"][start:end].to(device),
                tensors["support_mask"][start:end].to(device),
                tensors["candidate_feat"][start:end].to(device),
                tensors["candidate_meta"][start:end].to(device),
            )
            logits_all.append(logits.cpu())
    logits_np = torch.cat(logits_all, dim=0).numpy()
    return {
        "logits": logits_np.astype(np.float32),
        "baseline_err": raw_cache["baseline_err"].astype(np.float32),
        "cand_err": raw_cache["cand_err"].astype(np.float32),
    }


def selective_metrics_from_logits(
    logits_np: np.ndarray,
    baseline_err: np.ndarray,
    cand_err: np.ndarray,
    margin_thr: float,
) -> Dict[str, float]:
    pred_idx = np.argmax(logits_np, axis=1)
    top1 = logits_np[np.arange(len(logits_np)), pred_idx]
    baseline_logit = logits_np[:, 0]
    margin = top1 - baseline_logit
    accept = (pred_idx != 0) & (margin >= margin_thr)

    final_err = baseline_err.copy()
    accepted_err = []
    for i in range(len(final_err)):
        if accept[i]:
            final_err[i] = cand_err[i, pred_idx[i] - 1]
            accepted_err.append(final_err[i])
    accepted_err = np.asarray(accepted_err, dtype=np.float32)

    return {
        "margin_thr": float(margin_thr),
        "coverage": float(np.mean(accept)),
        "median_px": float(np.median(final_err)),
        "lt4px": float(np.mean(final_err < 4.0)),
        "better_frac": float(np.mean(final_err < baseline_err)),
        "accept_only_median_px": (
            float(np.median(accepted_err)) if accepted_err.size > 0 else None
        ),
        "choose_baseline_frac": float(np.mean(~accept)),
    }


def risk_coverage_curve_from_logits(
    logits_np: np.ndarray,
    baseline_err: np.ndarray,
    cand_err: np.ndarray,
    num_points: int = 11,
) -> List[Dict[str, float]]:
    pred_idx = np.argmax(logits_np, axis=1)
    top1 = logits_np[np.arange(len(logits_np)), pred_idx]
    baseline_logit = logits_np[:, 0]
    margins = top1 - baseline_logit
    thresholds = np.quantile(margins, np.linspace(0.0, 1.0, num_points))
    rows = []
    seen = set()
    for thr in thresholds.tolist():
        thr = round(float(thr), 6)
        if thr in seen:
            continue
        seen.add(thr)
        rows.append(selective_metrics_from_logits(logits_np, baseline_err, cand_err, margin_thr=thr))
    return rows


def tensors_from_numpy(data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in data.items():
        if v.dtype.kind in ("i", "u"):
            out[k] = torch.from_numpy(v).long()
        else:
            out[k] = torch.from_numpy(v).float()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train support-memory ranker for PRT")
    parser.add_argument("--train-cache", type=str, required=True)
    parser.add_argument("--val-cache", type=str, required=True)
    parser.add_argument("--weights", type=str, default="/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DINOFeatureExtractor(Path(args.weights), device)

    train_cache = load_cache(Path(args.train_cache))
    val_cache = load_cache(Path(args.val_cache))

    print(f"Encoding train cache: {train_cache['query_patch'].shape[0]} samples", flush=True)
    train_feat = build_feature_table(train_cache, extractor)
    print(f"Encoding val cache: {val_cache['query_patch'].shape[0]} samples", flush=True)
    val_feat = build_feature_table(val_cache, extractor)

    train_np = {
        "query_feat": train_feat["query_feat"],
        "baseline_feat": train_feat["baseline_feat"],
        "support_feat": train_feat["support_feat"],
        "support_mask": train_feat["support_mask"],
        "baseline_err": train_cache["baseline_err"].astype(np.float32),
        "cand_err": train_cache["cand_err"].astype(np.float32),
        **make_candidate_tensor(train_cache, train_feat),
    }
    val_np = {
        "query_feat": val_feat["query_feat"],
        "baseline_feat": val_feat["baseline_feat"],
        "support_feat": val_feat["support_feat"],
        "support_mask": val_feat["support_mask"],
        "baseline_err": val_cache["baseline_err"].astype(np.float32),
        "cand_err": val_cache["cand_err"].astype(np.float32),
        **make_candidate_tensor(val_cache, val_feat),
    }

    train_tensors = tensors_from_numpy(train_np)
    val_tensors = tensors_from_numpy(val_np)

    model = SupportMemoryRanker(
        feat_dim=train_np["query_feat"].shape[1],
        meta_dim=train_np["candidate_meta"].shape[-1],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    n_train = train_np["query_feat"].shape[0]
    history: List[Dict[str, float]] = []
    best_metric = float("inf")
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(n_train)
        losses = []
        for start in range(0, n_train, args.batch_size):
            idx = order[start:start + args.batch_size]
            idx_t = torch.from_numpy(idx).long()

            logits = model(
                train_tensors["query_feat"][idx_t].to(device),
                train_tensors["baseline_feat"][idx_t].to(device),
                train_tensors["support_feat"][idx_t].to(device),
                train_tensors["support_mask"][idx_t].to(device),
                train_tensors["candidate_feat"][idx_t].to(device),
                train_tensors["candidate_meta"][idx_t].to(device),
            )
            target = train_tensors["target"][idx_t].to(device)
            loss_ce = F.cross_entropy(logits, target)

            probs = torch.softmax(logits, dim=1)
            choice_err = torch.cat(
                [
                    train_tensors["baseline_err"][idx_t].unsqueeze(1),
                    train_tensors["cand_err"][idx_t],
                ],
                dim=1,
            ).to(device)
            norm_err = choice_err / choice_err.max(dim=1, keepdim=True).values.clamp(min=1.0)
            loss_cost = (probs * norm_err).sum(dim=1).mean()
            loss = loss_ce + 0.5 * loss_cost

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        metrics = evaluate(model, val_tensors, val_cache, args.batch_size, device)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = float(np.mean(losses))
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=True), flush=True)

        if metrics["pred_median_px"] < best_metric:
            best_metric = metrics["pred_median_px"]
            no_improve = 0
            torch.save(model.state_dict(), output_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    (output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
    best = min(history, key=lambda x: x["pred_median_px"])
    (output_dir / "best.json").write_text(json.dumps(best, indent=2) + "\n")

    model.load_state_dict(torch.load(output_dir / "best.pt", map_location=device, weights_only=True))
    collected = collect_logits_and_errors(model, val_tensors, val_cache, args.batch_size, device)
    selective_rows = []
    for thr in [-1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0]:
        selective_rows.append(
            selective_metrics_from_logits(
                collected["logits"],
                collected["baseline_err"],
                collected["cand_err"],
                margin_thr=thr,
            )
        )
    selective = {
        "fixed_thresholds": selective_rows,
        "risk_coverage": risk_coverage_curve_from_logits(
            collected["logits"],
            collected["baseline_err"],
            collected["cand_err"],
            num_points=11,
        ),
    }
    (output_dir / "selective.json").write_text(json.dumps(selective, indent=2) + "\n")
    print(f"Best epoch: {json.dumps(best, ensure_ascii=True)}", flush=True)


if __name__ == "__main__":
    main()
