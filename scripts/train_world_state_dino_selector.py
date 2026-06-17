#!/usr/bin/env python3
"""
Frozen DINO patch-embedding selector over baseline + K candidates.

Reuses pre-collected patch candidate samples from train_world_state_ranker.py.
"""

import argparse
import json
import ast
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_world_state_stage2_causal_dino import DINOFeatureExtractor


def load_samples(path: Path) -> List[dict]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected list at {path}")
    return data


def to_np_f32(x):
    if isinstance(x, str):
        x = ast.literal_eval(x)
    return np.asarray(x, dtype=np.float32)


def build_selector_rows(sample: dict) -> Tuple[np.ndarray, int, np.ndarray]:
    baseline_err = float(sample["baseline_err"])
    baseline_xy = to_np_f32(sample["baseline_xy"])
    gt_xy = to_np_f32(sample["gt_xy"])
    cand_errors = to_np_f32(sample["cand_errors"])
    cand_scores = to_np_f32(sample["cand_dino_scores"])

    errors = np.concatenate([[baseline_err], cand_errors], axis=0)
    target = int(np.argmin(errors))

    rows = [[
        0.0,
        0.0,
        0.0,
        baseline_err / 256.0,
        float(sample["occ_length"]) / 300.0,
        float(sample["camera_motion"]),
    ]]

    for i, pred_xy in enumerate(sample["baseline_xy"] for _ in []):
        pass

    K = len(cand_errors)
    for i in range(K):
        pred_xy = to_np_f32(sample["baseline_xy"])
        # Stored candidate xy may be unavailable in compact exports, so fallback to scalar-only geometry.
        rows.append([
            1.0,
            float(cand_scores[i]),
            float(i) / max(K - 1, 1),
            baseline_err / 256.0,
            float(sample["occ_length"]) / 300.0,
            float(sample["camera_motion"]),
        ])

    return np.asarray(rows, dtype=np.float32), target, errors


class DINOSelector(nn.Module):
    def __init__(self, feat_dim: int = 384, geom_dim: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
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
        cand_feats: torch.Tensor,
        geom_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        query_feat: (B, D)
        baseline_feat: (B, D)
        cand_feats: (B, N, D) including baseline candidate at index 0
        geom_feats: (B, N, G)
        """
        B, N, D = cand_feats.shape
        q = query_feat.unsqueeze(1).expand(B, N, D)
        b = baseline_feat.unsqueeze(1).expand(B, N, D)
        x = torch.cat([q, cand_feats, b, geom_feats], dim=-1)
        return self.proj(x).squeeze(-1)


def encode_patch_batch(extractor: DINOFeatureExtractor, patches_np: np.ndarray) -> torch.Tensor:
    """
    patches_np: (N, 3, H, W) in [0,1]
    returns: (N, 384)
    """
    out = []
    bs = 32
    for s in range(0, len(patches_np), bs):
        batch = torch.from_numpy(patches_np[s:s + bs]).float()
        batch = F.interpolate(batch, size=(518, 518), mode="bilinear", align_corners=False).to(extractor.device)
        with torch.no_grad():
            feat = extractor.model(batch)[-1].mean(dim=[-2, -1]).float()
            feat = F.normalize(feat, dim=-1)
        out.append(feat.cpu())
    return torch.cat(out, dim=0)


def prepare_dataset(samples: List[dict], extractor: DINOFeatureExtractor):
    prepared = []
    for s in samples:
        query_patch = to_np_f32(s["query_patch"])
        baseline_patch = to_np_f32(s["baseline_patch"])
        cand_patches = to_np_f32(s["cand_patches"])

        q_feat = encode_patch_batch(extractor, query_patch[None])[0].numpy()
        b_feat = encode_patch_batch(extractor, baseline_patch[None])[0].numpy()
        c_feat = encode_patch_batch(extractor, cand_patches).numpy()

        geom_rows, target, errors = build_selector_rows(s)
        cand_feats = np.concatenate([b_feat[None], c_feat], axis=0)
        prepared.append(
            {
                "query_feat": q_feat,
                "baseline_feat": b_feat,
                "cand_feats": cand_feats,
                "geom_feats": geom_rows,
                "target": target,
                "errors": errors,
            }
        )
    return prepared


def evaluate(model: DINOSelector, dataset, device: torch.device):
    model.eval()
    pred_errs, base_errs, oracle_errs, chosen = [], [], [], []
    with torch.no_grad():
        for s in dataset:
            q = torch.from_numpy(s["query_feat"]).float().to(device)[None]
            b = torch.from_numpy(s["baseline_feat"]).float().to(device)[None]
            c = torch.from_numpy(s["cand_feats"]).float().to(device)[None]
            g = torch.from_numpy(s["geom_feats"]).float().to(device)[None]
            logits = model(q, b, c, g)[0]
            idx = int(torch.argmax(logits).item())
            errs = s["errors"]
            pred_errs.append(float(errs[idx]))
            base_errs.append(float(errs[0]))
            oracle_errs.append(float(np.min(errs)))
            chosen.append(idx)
    pred = np.asarray(pred_errs)
    base = np.asarray(base_errs)
    oracle = np.asarray(oracle_errs)
    chosen = np.asarray(chosen)
    return {
        "selector_pred_median": float(np.median(pred)),
        "selector_baseline_median": float(np.median(base)),
        "selector_oracle_median": float(np.median(oracle)),
        "selector_better_frac": float(np.mean(pred < base)),
        "selector_pred_lt4px": float(np.mean(pred < 4.0)),
        "selector_baseline_lt4px": float(np.mean(base < 4.0)),
        "selector_oracle_lt4px": float(np.mean(oracle < 4.0)),
        "selector_choose_baseline_frac": float(np.mean(chosen == 0)),
    }


def main():
    parser = argparse.ArgumentParser(description="Frozen DINO selector on pre-collected candidate patches")
    parser.add_argument("--train-samples", type=str, required=True)
    parser.add_argument("--val-samples", type=str, required=True)
    parser.add_argument("--weights", type=str, default="/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DINOFeatureExtractor(Path(args.weights), device)

    train_raw = load_samples(Path(args.train_samples))
    val_raw = load_samples(Path(args.val_samples))
    print(f"Preparing train features from {len(train_raw)} samples...", flush=True)
    train_set = prepare_dataset(train_raw, extractor)
    print(f"Preparing val features from {len(val_raw)} samples...", flush=True)
    val_set = prepare_dataset(val_raw, extractor)

    feat_dim = train_set[0]["cand_feats"].shape[1]
    geom_dim = train_set[0]["geom_feats"].shape[1]
    model = DINOSelector(feat_dim=feat_dim, geom_dim=geom_dim).to(device)
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
            q = torch.from_numpy(s["query_feat"]).float().to(device)[None]
            b = torch.from_numpy(s["baseline_feat"]).float().to(device)[None]
            c = torch.from_numpy(s["cand_feats"]).float().to(device)[None]
            g = torch.from_numpy(s["geom_feats"]).float().to(device)[None]
            y = torch.tensor([s["target"]], dtype=torch.long, device=device)
            errs = torch.from_numpy(s["errors"]).float().to(device)
            logits = model(q, b, c, g)
            loss_ce = F.cross_entropy(logits, y)
            probs = torch.softmax(logits[0], dim=0)
            norm_err = errs / errs.max().clamp(min=1.0)
            loss_cost = (probs * norm_err).sum()
            loss = loss_ce + 0.5 * loss_cost
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

        if metrics["selector_pred_median"] < best_metric:
            best_metric = metrics["selector_pred_median"]
            no_improve = 0
            torch.save(model.state_dict(), output_dir / "selector_best.pt")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    (output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
    best = min(history, key=lambda h: h["selector_pred_median"])
    print(f"Best epoch: {json.dumps(best)}", flush=True)


if __name__ == "__main__":
    main()
