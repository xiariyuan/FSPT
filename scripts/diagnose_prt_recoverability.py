#!/usr/bin/env python3
"""
Cheap recoverability diagnostic for PRT support-memory models.

Goal:
  Before investing in a heavier learned recovery module, quickly test whether
  the current signal is learnable at all under different model capacities and
  loss choices.

Runs a grid over:
  - model: linear / mlp / full
  - loss: ce / rank / hybrid

Each configuration predicts over {baseline, cand1..K}.
Outputs metrics comparable to the existing hand-crafted selector:
  - pred median
  - baseline median
  - oracle median
  - better frac
  - choose baseline frac
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_world_state_stage2_causal_dino import DINOFeatureExtractor
from scripts.train_prt_support_memory_ranker import (
    build_feature_table,
    load_cache,
    make_candidate_tensor,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensors_from_numpy(data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in data.items():
        if v.dtype.kind in ("i", "u"):
            out[k] = torch.from_numpy(v).long()
        else:
            out[k] = torch.from_numpy(v).float()
    return out


@dataclass
class PreparedData:
    raw_cache: Dict[str, np.ndarray]
    np_data: Dict[str, np.ndarray]
    tensors: Dict[str, torch.Tensor]


def subset_cache(cache: Dict[str, np.ndarray], keep_mask: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in cache.items():
        if not isinstance(v, np.ndarray):
            out[k] = v
            continue
        if v.shape[0] == keep_mask.shape[0]:
            out[k] = v[keep_mask]
        else:
            out[k] = v
    return out


def build_filter_mask(
    cache: Dict[str, np.ndarray],
    reentry_type: str,
    occ_min: int | None,
    occ_max: int | None,
) -> np.ndarray:
    n = cache["query_patch"].shape[0]
    mask = np.ones(n, dtype=bool)

    if reentry_type != "all":
        type_id = cache["reentry_type_id"]
        mapping = {"in_frame": 0, "offscreen": 1, "mixed": 2}
        if reentry_type not in mapping:
            raise ValueError(f"Unknown reentry_type filter: {reentry_type}")
        mask &= (type_id == mapping[reentry_type])

    occ = cache["occ_length"]
    if occ_min is not None:
        mask &= (occ >= occ_min)
    if occ_max is not None:
        mask &= (occ < occ_max)
    return mask


def prepare_dataset(cache: Dict[str, np.ndarray], extractor: DINOFeatureExtractor) -> PreparedData:
    feat = build_feature_table(cache, extractor)
    np_data = {
        "query_feat": feat["query_feat"],
        "baseline_feat": feat["baseline_feat"],
        "support_feat": feat["support_feat"],
        "support_mask": feat["support_mask"],
        "baseline_err": cache["baseline_err"].astype(np.float32),
        "cand_err": cache["cand_err"].astype(np.float32),
        **make_candidate_tensor(cache, feat),
    }
    return PreparedData(
        raw_cache=cache,
        np_data=np_data,
        tensors=tensors_from_numpy(np_data),
    )


def build_pair_features(np_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    query = np_data["query_feat"]
    baseline = np_data["baseline_feat"]
    support = np_data["support_feat"]
    support_mask = np_data["support_mask"]
    candidate_feat = np_data["candidate_feat"]
    candidate_meta = np_data["candidate_meta"]

    support_mask_sum = np.clip(support_mask.sum(axis=1, keepdims=True), 1.0, None)
    pooled_support = (support * support_mask[:, :, None]).sum(axis=1) / support_mask_sum
    pooled_support /= np.linalg.norm(pooled_support, axis=1, keepdims=True).clip(1e-8, None)

    support_expand = np.repeat(pooled_support[:, None, :], candidate_feat.shape[1], axis=1)
    query_expand = np.repeat(query[:, None, :], candidate_feat.shape[1], axis=1)
    baseline_expand = np.repeat(baseline[:, None, :], candidate_feat.shape[1], axis=1)

    pair_feat = np.concatenate(
        [query_expand, baseline_expand, support_expand, candidate_feat, candidate_meta],
        axis=-1,
    ).astype(np.float32)
    return pair_feat, pooled_support.astype(np.float32)


class LinearRanker(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.head = nn.Linear(input_dim, 1)

    def forward(self, pair_feat: torch.Tensor) -> torch.Tensor:
        return self.head(pair_feat).squeeze(-1)


class MLPRanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pair_feat: torch.Tensor) -> torch.Tensor:
        return self.head(pair_feat).squeeze(-1)


class FullAttentionRanker(nn.Module):
    def __init__(self, feat_dim: int, meta_dim: int, hidden_dim: int = 256):
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

    def pool_support(
        self,
        support_feat: torch.Tensor,
        support_mask: torch.Tensor,
        query_feat: torch.Tensor,
    ) -> torch.Tensor:
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


def make_model(kind: str, prepared: PreparedData) -> nn.Module:
    if kind == "linear":
        pair_feat, _ = build_pair_features(prepared.np_data)
        return LinearRanker(pair_feat.shape[-1])
    if kind == "mlp":
        pair_feat, _ = build_pair_features(prepared.np_data)
        return MLPRanker(pair_feat.shape[-1])
    if kind == "full":
        return FullAttentionRanker(
            feat_dim=prepared.np_data["query_feat"].shape[1],
            meta_dim=prepared.np_data["candidate_meta"].shape[-1],
        )
    raise ValueError(f"Unknown model kind: {kind}")


def model_forward(kind: str, model: nn.Module, batch: Dict[str, torch.Tensor], idx: torch.Tensor) -> torch.Tensor:
    if kind in {"linear", "mlp"}:
        pair_feat = batch["pair_feat"][idx]
        return model(pair_feat)
    return model(
        batch["query_feat"][idx],
        batch["baseline_feat"][idx],
        batch["support_feat"][idx],
        batch["support_mask"][idx],
        batch["candidate_feat"][idx],
        batch["candidate_meta"][idx],
    )


def build_training_batch(prepared: PreparedData) -> Dict[str, torch.Tensor]:
    tensors = dict(prepared.tensors)
    pair_feat, _ = build_pair_features(prepared.np_data)
    tensors["pair_feat"] = torch.from_numpy(pair_feat).float()
    return tensors


def ranking_loss(logits: torch.Tensor, choice_err: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    baseline_logit = logits[:, 0:1]
    cand_logits = logits[:, 1:]
    baseline_err = choice_err[:, 0:1]
    cand_err = choice_err[:, 1:]

    better_mask = (cand_err + 2.0 < baseline_err).float()
    worse_mask = (cand_err >= baseline_err).float()

    loss_better = F.relu(margin - (cand_logits - baseline_logit)) * better_mask
    loss_worse = F.relu(margin - (baseline_logit - cand_logits)) * worse_mask

    denom = better_mask.sum() + worse_mask.sum()
    if float(denom.item()) < 1.0:
        return logits.new_tensor(0.0)
    return (loss_better.sum() + loss_worse.sum()) / denom


def balanced_train_indices(
    baseline_err: torch.Tensor,
    cand_err: torch.Tensor,
    target_size: int,
) -> torch.Tensor:
    if target_size <= 0:
        raise ValueError("target_size must be positive")

    best_cand = cand_err.min(dim=1).values
    baseline_best = torch.where(best_cand >= baseline_err)[0]
    candidate_better = torch.where(best_cand + 2.0 < baseline_err)[0]
    hard_negative = torch.where((best_cand < baseline_err) & (best_cand + 2.0 >= baseline_err))[0]

    pools = [baseline_best, candidate_better, hard_negative]
    per_group = max(1, target_size // len(pools))
    picks = []
    for pool in pools:
        if len(pool) == 0:
            continue
        if len(pool) >= per_group:
            perm = pool[torch.randperm(len(pool), device=pool.device)[:per_group]]
            picks.append(perm)
        else:
            extra = pool[torch.randint(0, len(pool), (per_group,), device=pool.device)]
            picks.append(extra)
    if not picks:
        return torch.randperm(len(baseline_err), device=baseline_err.device)[:target_size]

    merged = torch.cat(picks, dim=0)
    if len(merged) < target_size:
        extra_n = target_size - len(merged)
        universe = torch.arange(len(baseline_err), device=baseline_err.device)
        extra = universe[torch.randint(0, len(universe), (extra_n,), device=universe.device)]
        merged = torch.cat([merged, extra], dim=0)
    merged = merged[torch.randperm(len(merged), device=merged.device)]
    return merged[:target_size]


def compute_loss(
    loss_kind: str,
    logits: torch.Tensor,
    target: torch.Tensor,
    choice_err: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_ce = F.cross_entropy(logits, target)
    loss_rank = ranking_loss(logits, choice_err)
    probs = torch.softmax(logits, dim=1)
    norm_err = choice_err / choice_err.max(dim=1, keepdim=True).values.clamp(min=1.0)
    loss_cost = (probs * norm_err).sum(dim=1).mean()

    if loss_kind == "ce":
        loss = loss_ce
    elif loss_kind == "rank":
        loss = loss_rank
    elif loss_kind == "hybrid":
        loss = loss_ce + 0.5 * loss_rank + 0.5 * loss_cost
    else:
        raise ValueError(f"Unknown loss kind: {loss_kind}")

    return loss, {
        "loss_ce": float(loss_ce.detach().cpu()),
        "loss_rank": float(loss_rank.detach().cpu()),
        "loss_cost": float(loss_cost.detach().cpu()),
    }


def evaluate_predictions(logits_np: np.ndarray, raw_cache: Dict[str, np.ndarray]) -> Dict[str, float]:
    idx = np.argmax(logits_np, axis=1)
    pred = []
    base = raw_cache["baseline_err"].astype(np.float32)
    oracle = np.minimum(base, raw_cache["cand_err"].min(axis=1)).astype(np.float32)
    for i, choice in enumerate(idx.tolist()):
        if choice == 0:
            pred.append(float(base[i]))
        else:
            pred.append(float(raw_cache["cand_err"][i, choice - 1]))
    pred = np.asarray(pred, dtype=np.float32)
    return {
        "pred_median_px": float(np.median(pred)),
        "baseline_median_px": float(np.median(base)),
        "oracle_median_px": float(np.median(oracle)),
        "pred_lt4px": float(np.mean(pred < 4.0)),
        "baseline_lt4px": float(np.mean(base < 4.0)),
        "oracle_lt4px": float(np.mean(oracle < 4.0)),
        "better_frac": float(np.mean(pred < base)),
        "choose_baseline_frac": float(np.mean(idx == 0)),
    }


def collect_logits(
    kind: str,
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    all_logits = []
    n = batch["query_feat"].shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = torch.arange(start, end, dtype=torch.long)
            logits = model_forward(kind, model, batch, idx.to(batch["query_feat"].device))
            all_logits.append(logits.detach().cpu())
    return torch.cat(all_logits, dim=0).numpy().astype(np.float32)


def train_one_config(
    model_kind: str,
    loss_kind: str,
    train_prepared: PreparedData,
    val_prepared: PreparedData,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    balanced_sampling: bool,
) -> Dict[str, object]:
    set_seed(seed)
    train_batch = build_training_batch(train_prepared)
    val_batch = build_training_batch(val_prepared)

    for k in train_batch:
        train_batch[k] = train_batch[k].to(device)
    for k in val_batch:
        val_batch[k] = val_batch[k].to(device)

    model = make_model(model_kind, train_prepared).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n_train = train_batch["query_feat"].shape[0]
    best = None
    history = []

    for epoch in range(epochs):
        model.train()
        if balanced_sampling:
            order = balanced_train_indices(
                train_batch["baseline_err"],
                train_batch["cand_err"],
                n_train,
            )
        else:
            order = torch.randperm(n_train, device=device)
        train_losses = []
        last_terms = {}
        for start in range(0, n_train, batch_size):
            idx = order[start:start + batch_size]
            logits = model_forward(model_kind, model, train_batch, idx)
            target = train_batch["target"][idx]
            choice_err = torch.cat(
                [train_batch["baseline_err"][idx].unsqueeze(1), train_batch["cand_err"][idx]],
                dim=1,
            )
            loss, last_terms = compute_loss(loss_kind, logits, target, choice_err)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        val_logits = collect_logits(model_kind, model, val_batch, batch_size, device)
        val_metrics = evaluate_predictions(val_logits, val_prepared.raw_cache)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)),
            **last_terms,
            **val_metrics,
        }
        history.append(row)
        if best is None or row["pred_median_px"] < best["pred_median_px"]:
            best = dict(row)

    return {
        "model": model_kind,
        "loss": loss_kind,
        "best": best,
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose learnability of PRT recovery signals")
    parser.add_argument("--train-cache", type=str, required=True)
    parser.add_argument("--val-cache", type=str, required=True)
    parser.add_argument("--weights", type=str, default="/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--models", type=str, default="linear,mlp,full")
    parser.add_argument("--losses", type=str, default="ce,rank,hybrid")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-reentry-type", type=str, default="all")
    parser.add_argument("--val-reentry-type", type=str, default="all")
    parser.add_argument("--train-occ-min", type=int, default=None)
    parser.add_argument("--train-occ-max", type=int, default=None)
    parser.add_argument("--val-occ-min", type=int, default=None)
    parser.add_argument("--val-occ-max", type=int, default=None)
    parser.add_argument("--balanced-sampling", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DINOFeatureExtractor(Path(args.weights), device)

    train_cache = load_cache(Path(args.train_cache))
    val_cache = load_cache(Path(args.val_cache))
    train_mask = build_filter_mask(
        train_cache,
        reentry_type=args.train_reentry_type,
        occ_min=args.train_occ_min,
        occ_max=args.train_occ_max,
    )
    val_mask = build_filter_mask(
        val_cache,
        reentry_type=args.val_reentry_type,
        occ_min=args.val_occ_min,
        occ_max=args.val_occ_max,
    )
    train_cache = subset_cache(train_cache, train_mask)
    val_cache = subset_cache(val_cache, val_mask)

    if train_cache["query_patch"].shape[0] == 0 or val_cache["query_patch"].shape[0] == 0:
        raise RuntimeError("Filtered train/val cache is empty. Relax filter settings.")

    print(f"Encoding train cache: {train_cache['query_patch'].shape[0]} samples", flush=True)
    train_prepared = prepare_dataset(train_cache, extractor)
    print(f"Encoding val cache: {val_cache['query_patch'].shape[0]} samples", flush=True)
    val_prepared = prepare_dataset(val_cache, extractor)

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    losses = [x.strip() for x in args.losses.split(",") if x.strip()]

    results = []
    for model_kind in models:
        for loss_kind in losses:
            print(f"Running model={model_kind} loss={loss_kind}", flush=True)
            result = train_one_config(
                model_kind=model_kind,
                loss_kind=loss_kind,
                train_prepared=train_prepared,
                val_prepared=val_prepared,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                balanced_sampling=args.balanced_sampling,
            )
            results.append(result)
            print(json.dumps(result["best"], ensure_ascii=True), flush=True)

    summary = []
    for result in results:
        row = dict(result["best"])
        row["model"] = result["model"]
        row["loss"] = result["loss"]
        summary.append(row)
    summary.sort(key=lambda x: x["pred_median_px"])

    payload = {
        "train_cache": args.train_cache,
        "val_cache": args.val_cache,
        "models": models,
        "losses": losses,
        "filters": {
            "train_reentry_type": args.train_reentry_type,
            "val_reentry_type": args.val_reentry_type,
            "train_occ_min": args.train_occ_min,
            "train_occ_max": args.train_occ_max,
            "val_occ_min": args.val_occ_min,
            "val_occ_max": args.val_occ_max,
            "balanced_sampling": args.balanced_sampling,
        },
        "summary": summary,
        "results": results,
    }
    (output_dir / "diagnostic_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved diagnostic results to {output_dir / 'diagnostic_results.json'}", flush=True)


if __name__ == "__main__":
    main()
