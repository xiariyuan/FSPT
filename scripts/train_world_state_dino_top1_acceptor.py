#!/usr/bin/env python3
"""
Frozen DINO top-1 acceptor.

The DINO local search already gives a top-1 candidate.
The main failure mode is catastrophic false accepts, not ranking among top-k.
So this script learns a binary decision: accept top-1 candidate or fallback to baseline.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_world_state_stage2_causal_dino import DINOFeatureExtractor
from scripts.train_world_state_ranker import collect_candidates_with_patches
from scripts.eval_world_state_stage0 import discover_sequences, find_reentry_queries, load_sequence


class Top1Acceptor(nn.Module):
    def __init__(self, feat_dim: int = 384, geom_dim: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 3 + geom_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        query_feat: torch.Tensor,
        baseline_feat: torch.Tensor,
        cand_feat: torch.Tensor,
        geom_feat: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([query_feat, baseline_feat, cand_feat, geom_feat], dim=-1)
        return self.head(x).squeeze(-1)


def encode_patch_batch_dino(extractor: DINOFeatureExtractor, patches: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        x = F.interpolate(patches, size=(518, 518), mode="bilinear", align_corners=False).to(extractor.device)
        feats = extractor.model(x)[-1].mean(dim=[-2, -1]).float()
        feats = F.normalize(feats, dim=-1)
    return feats


def prepare_dataset(samples: List[dict], extractor: DINOFeatureExtractor, device: torch.device, positive_margin_px: float):
    prepared = []
    for s in samples:
        if len(s["cand_errors"]) == 0:
            continue
        q = torch.from_numpy(s["query_patch"]).float().unsqueeze(0).to(device)
        b = torch.from_numpy(s["baseline_patch"]).float().unsqueeze(0).to(device)
        c0 = torch.from_numpy(s["cand_patches"][0]).float().unsqueeze(0).to(device)

        q_feat = encode_patch_batch_dino(extractor, q)[0].cpu()
        b_feat = encode_patch_batch_dino(extractor, b)[0].cpu()
        c_feat = encode_patch_batch_dino(extractor, c0)[0].cpu()

        baseline_err = float(s["baseline_err"])
        cand_err = float(s["cand_errors"][0])
        label = 1.0 if cand_err < baseline_err - positive_margin_px else 0.0
        q_b_cos = float(torch.dot(q_feat, b_feat))
        q_c_cos = float(torch.dot(q_feat, c_feat))
        c_b_cos = float(torch.dot(c_feat, b_feat))
        geom = torch.tensor(
            [[
                float(s["cand_dino_scores"][0]),
                q_b_cos,
                q_c_cos,
                c_b_cos,
                float(s["occ_length"]) / 300.0,
                float(s["camera_motion"]),
            ]],
            dtype=torch.float32,
        )[0]

        prepared.append(
            {
                "query_feat": q_feat,
                "baseline_feat": b_feat,
                "cand_feat": c_feat,
                "geom_feat": geom,
                "baseline_err": baseline_err,
                "cand_err": cand_err,
                "label": label,
            }
        )
    return prepared


def evaluate(model: Top1Acceptor, dataset, device: torch.device, threshold: float = 0.5):
    model.eval()
    pred_errs, base_errs, cand_errs, accepts = [], [], [], []
    with torch.no_grad():
        for s in dataset:
            q = s["query_feat"].unsqueeze(0).to(device)
            b = s["baseline_feat"].unsqueeze(0).to(device)
            c = s["cand_feat"].unsqueeze(0).to(device)
            g = s["geom_feat"].unsqueeze(0).to(device)
            logit = model(q, b, c, g)
            prob = float(torch.sigmoid(logit).item())
            accept = prob >= threshold
            pred_errs.append(s["cand_err"] if accept else s["baseline_err"])
            base_errs.append(s["baseline_err"])
            cand_errs.append(s["cand_err"])
            accepts.append(float(accept))

    pred = np.asarray(pred_errs)
    base = np.asarray(base_errs)
    cand = np.asarray(cand_errs)
    accepts = np.asarray(accepts)
    return {
        "top1_acceptor_pred_median": float(np.median(pred)),
        "top1_acceptor_baseline_median": float(np.median(base)),
        "top1_acceptor_cand0_median": float(np.median(cand)),
        "top1_acceptor_better_frac": float(np.mean(pred < base)),
        "top1_acceptor_pred_lt4px": float(np.mean(pred < 4.0)),
        "top1_acceptor_baseline_lt4px": float(np.mean(base < 4.0)),
        "top1_acceptor_cand0_lt4px": float(np.mean(cand < 4.0)),
        "top1_acceptor_accept_rate": float(np.mean(accepts)),
        "top1_cand0_better_frac": float(np.mean(cand < base)),
    }


def main():
    parser = argparse.ArgumentParser(description="Frozen DINO top-1 acceptor")
    parser.add_argument("--data-root", type=str, default="/gemini/code/FSPT/datasets/pointodyssey")
    parser.add_argument("--train-splits", type=str, default="train")
    parser.add_argument("--val-splits", type=str, default="val")
    parser.add_argument("--min-occ-length", type=int, default=20)
    parser.add_argument("--min-camera-motion", type=float, default=0.30)
    parser.add_argument("--train-max-sequences", type=int, default=1)
    parser.add_argument("--val-max-sequences", type=int, default=1)
    parser.add_argument("--train-max-samples", type=int, default=200)
    parser.add_argument("--val-max-samples", type=int, default=100)
    parser.add_argument("--depth-noise-sigma", type=float, default=0.10)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--query-crop-size", type=int, default=112)
    parser.add_argument("--search-crop-size", type=int, default=224)
    parser.add_argument("--positive-margin-px", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weights", type=str, default="/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = DINOFeatureExtractor(Path(args.weights), device)

    print("Collecting train samples...", flush=True)
    train_seqs = discover_sequences(Path(args.data_root), args.train_splits.split(","))
    if args.train_max_sequences > 0:
        train_seqs = train_seqs[:args.train_max_sequences]
    train_samples = []
    for sp in train_seqs:
        seq = load_sequence(sp)
        queries = find_reentry_queries(seq, min_occ_length=args.min_occ_length)
        cands = collect_candidates_with_patches(
            extractor, sp, queries, seq, args.depth_noise_sigma,
            np.random.default_rng(42), topk=args.topk,
            query_crop_size=args.query_crop_size, search_crop_size=args.search_crop_size,
            patch_size=args.patch_size, min_cam_motion=args.min_camera_motion,
            max_samples=args.train_max_samples - len(train_samples),
        )
        train_samples.extend(cands)
        if len(train_samples) >= args.train_max_samples:
            break
    print(f"  Train: {len(train_samples)}", flush=True)

    print("Collecting val samples...", flush=True)
    val_seqs = discover_sequences(Path(args.data_root), args.val_splits.split(","))
    if args.val_max_sequences > 0:
        val_seqs = val_seqs[:args.val_max_sequences]
    val_samples = []
    for sp in val_seqs:
        seq = load_sequence(sp)
        queries = find_reentry_queries(seq, min_occ_length=args.min_occ_length)
        cands = collect_candidates_with_patches(
            extractor, sp, queries, seq, args.depth_noise_sigma,
            np.random.default_rng(999), topk=args.topk,
            query_crop_size=args.query_crop_size, search_crop_size=args.search_crop_size,
            patch_size=args.patch_size, min_cam_motion=args.min_camera_motion,
            max_samples=args.val_max_samples - len(val_samples),
        )
        val_samples.extend(cands)
        if len(val_samples) >= args.val_max_samples:
            break
    print(f"  Val: {len(val_samples)}", flush=True)

    train_set = prepare_dataset(train_samples, extractor, device, args.positive_margin_px)
    val_set = prepare_dataset(val_samples, extractor, device, args.positive_margin_px)
    print(f"Prepared: train={len(train_set)}, val={len(val_set)}", flush=True)

    model = Top1Acceptor().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_metric = float("inf")
    no_improve = 0
    for epoch in range(args.epochs):
        model.train()
        order = np.random.permutation(len(train_set))
        losses = []
        for idx in order:
            s = train_set[int(idx)]
            q = s["query_feat"].unsqueeze(0).to(device)
            b = s["baseline_feat"].unsqueeze(0).to(device)
            c = s["cand_feat"].unsqueeze(0).to(device)
            g = s["geom_feat"].unsqueeze(0).to(device)
            y = torch.tensor([s["label"]], dtype=torch.float32, device=device)
            logit = model(q, b, c, g)
            pos_weight = torch.tensor([3.0], device=device)
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics = evaluate(model, val_set, device)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = float(np.mean(losses))
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=True), flush=True)

        if metrics["top1_acceptor_pred_median"] < best_metric:
            best_metric = metrics["top1_acceptor_pred_median"]
            no_improve = 0
            torch.save(model.state_dict(), output_dir / "top1_acceptor_best.pt")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    (output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
    best = min(history, key=lambda h: h["top1_acceptor_pred_median"])
    print(f"Best epoch: {json.dumps(best)}", flush=True)


if __name__ == "__main__":
    main()
