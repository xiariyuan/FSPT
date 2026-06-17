#!/usr/bin/env python3
"""
Offline baseline evaluator for recovery anchor prediction.

Compares three methods on the recovery-anchor dataset:
  1. Base tracker (noisy depth reprojection)
  2. Raw DINO cosine readout
  3. Oracle (upper bound)

Usage:
  python scripts/eval_offline_recovery_anchor_baselines.py \
    --dataset-dir outputs/recovery_anchor_dataset_smoke \
    --weights /gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth \
    --output outputs/recovery_anchor_baselines_smoke.json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_world_state_stage2_causal_dino import DINOFeatureExtractor
from scripts.eval_prt_hybrid_selector import encode_patch_batch


def extract_patch_feature(extractor, patch_rgb: np.ndarray) -> torch.Tensor:
    """Extract DINOv2 feature vector from a single patch (H,W,3 uint8)."""
    t = torch.from_numpy(patch_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    t = F.interpolate(t, size=(518, 518), mode="bilinear", align_corners=False).to(extractor.device)
    with torch.no_grad():
        feat = extractor.model(t)[-1].mean(dim=[-2, -1]).float()
        feat = F.normalize(feat, dim=-1)
    return feat[0].cpu()


def dino_cosine_readout(
    extractor,
    support_patches: np.ndarray,  # (K, H, W, 3) uint8
    search_crop: np.ndarray,      # (H, W, 3) uint8
    base_xy: np.ndarray,          # (2,) in original image coords
    search_crop_size: int,
) -> tuple:
    """Raw DINO cosine similarity readout: find the position in the search crop
    whose DINO feature is most similar to the pooled support descriptor.

    Returns: (pred_xy, sim_score) in original image coordinates.
    """
    K = support_patches.shape[0]
    # Encode support patches
    support_t = torch.from_numpy(support_patches).float()  # (K,H,W,3)
    support_t = support_t.permute(0, 3, 1, 2) / 255.0
    support_t = F.interpolate(support_t, size=(518, 518), mode="bilinear", align_corners=False).to(extractor.device)
    with torch.no_grad():
        sfeat = extractor.model(support_t)[-1].mean(dim=[-2, -1]).float()
        sfeat = F.normalize(sfeat, dim=-1)
    # Pool support features (mean)
    support_pool = sfeat.mean(dim=0, keepdim=True)  # (1, D)
    support_pool = F.normalize(support_pool, dim=-1)

    # Encode search crop as feature map
    search_t = torch.from_numpy(search_crop).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    search_t = F.interpolate(search_t, size=(518, 518), mode="bilinear", align_corners=False).to(extractor.device)
    with torch.no_grad():
        smap = extractor.model(search_t)[-1]  # (1, D, h, w)
        smap = F.normalize(smap.float(), dim=1)

    # Cosine similarity map
    D, h, w = smap.shape[1], smap.shape[2], smap.shape[3]
    sim = torch.matmul(support_pool, smap.reshape(D, h * w)).reshape(h, w)
    sim = sim.cpu().numpy()

    # Find peak
    peak_idx = np.unravel_index(sim.argmax(), sim.shape)
    peak_sim = float(sim[peak_idx])

    # Convert feature-map coords back to search crop coords
    scale = search_crop_size / float(h)
    pred_in_crop = np.array([(peak_idx[1] + 0.5) * scale, (peak_idx[0] + 0.5) * scale], dtype=np.float32)

    # Convert to original image coords (search crop is centered on base_xy)
    center_offset = np.array([search_crop_size / 2.0, search_crop_size / 2.0], dtype=np.float32)
    pred_xy = base_xy - center_offset + pred_in_crop

    return pred_xy, peak_sim


def load_samples(dataset_dir: Path, split: str) -> List[Dict]:
    """Load sample index for a split. Try v2 format (index_all.json) first,
    then v1 format (per-sequence index files)."""
    # v2: single index file with all metadata inline
    index_path = dataset_dir / f"index_{split}.json"
    if index_path.exists():
        with open(index_path) as f:
            entries = json.load(f)
        # Check if entries already have gt_xy (v2 format)
        if entries and "gt_xy" in entries[0]:
            return entries
        # v1: need to enrich from per-sequence index
        seq_meta_cache = {}
        enriched = []
        for entry in entries:
            seq_name = entry["seq_name"]
            if seq_name not in seq_meta_cache:
                seq_idx_path = dataset_dir / seq_name / "index.json"
                if seq_idx_path.exists():
                    with open(seq_idx_path) as f:
                        seq_meta_cache[seq_name] = {s["sample_id"]: s for s in json.load(f)}
                else:
                    seq_meta_cache[seq_name] = {}
            meta = seq_meta_cache[seq_name].get(entry["sample_id"], {})
            enriched.append({**entry, **meta})
        return enriched
    return []


def main():
    parser = argparse.ArgumentParser(description="Evaluate offline recovery baselines")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--weights", type=str,
                        default="/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DINOFeatureExtractor(Path(args.weights), device)

    dataset_dir = Path(args.dataset_dir)
    samples = load_samples(dataset_dir, args.split)
    print(f"Loaded {len(samples)} samples from {args.split} split")

    if len(samples) == 0:
        print("No samples found!")
        return

    results = []
    for idx, sample in enumerate(samples):
        # Load tensors
        npz_path = dataset_dir / sample["file"]
        data = np.load(str(npz_path))

        gt_xy = np.array(sample["gt_xy"], dtype=np.float32)
        base_xy = np.array(sample["base_xy"], dtype=np.float32)
        base_err = float(np.linalg.norm(base_xy - gt_xy))

        # Method 1: Base tracker (already computed)
        # base_err is the metric

        # Method 2: Raw DINO cosine readout
        search_crop = data["search_crop"]  # (H,W,3) uint8
        support_patches = data["support_patches"]  # (K,H,W,3) uint8

        pred_xy, sim_score = dino_cosine_readout(
            extractor, support_patches, search_crop,
            base_xy, sample["search_crop_size"],
        )
        dino_err = float(np.linalg.norm(pred_xy - gt_xy))

        entry = {
            "seq_name": sample.get("seq_name", sample.get("video_name", "unknown")),
            "occ_length": sample["occ_length"],
            "base_error_px": base_err,
            "dino_cosine_error_px": dino_err,
            "dino_better": dino_err < base_err,
            "sim_score": sim_score,
        }
        results.append(entry)

        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(samples)} processed")

    # Compute statistics
    base_errs = np.array([r["base_error_px"] for r in results])
    dino_errs = np.array([r["dino_cosine_error_px"] for r in results])

    def stats(errs):
        return {
            "count": int(len(errs)),
            "mean": round(float(np.mean(errs)), 2),
            "median": round(float(np.median(errs)), 2),
            "p90": round(float(np.percentile(errs, 90)), 2),
            "p95": round(float(np.percentile(errs, 95)), 2),
        }

    occ = np.array([r["occ_length"] for r in results])
    better_frac = float(np.mean([r["dino_better"] for r in results]))

    # Stratified by occ length
    stratified = {}
    for lo, hi, label in [(20, 100, "20-100"), (100, 200, "100-200"), (200, 500, "200-500"), (500, 10000, "500+")]:
        mask = (occ >= lo) & (occ < hi)
        if mask.sum() == 0:
            continue
        stratified[label] = {
            "n": int(mask.sum()),
            "base": stats(base_errs[mask]),
            "dino_cosine": stats(dino_errs[mask]),
            "better_frac": round(float(np.mean(dino_errs[mask] < base_errs[mask])), 3),
            "delta_mean": round(float(np.mean(base_errs[mask]) - np.mean(dino_errs[mask])), 2),
            "delta_median": round(float(np.median(base_errs[mask]) - np.median(dino_errs[mask])), 2),
        }

    # Sequence-grouped bootstrap for mean diff
    rng = np.random.default_rng(42)
    unique_seqs = list(set(r["seq_name"] for r in results))
    n_boot = 2000
    boot_diffs = []
    for _ in range(n_boot):
        chosen = rng.choice(unique_seqs, size=len(unique_seqs), replace=True)
        idx = []
        for s in chosen:
            idx.extend([i for i, r in enumerate(results) if r["seq_name"] == s])
        idx = np.array(idx)
        boot_diffs.append(float(np.mean(base_errs[idx] - dino_errs[idx])))

    summary = {
        "split": args.split,
        "n_samples": len(results),
        "n_sequences": len(unique_seqs),
        "overall": {
            "base": stats(base_errs),
            "dino_cosine": stats(dino_errs),
            "better_frac": better_frac,
            "delta_mean": round(float(np.mean(base_errs) - np.mean(dino_errs)), 2),
            "delta_median": round(float(np.median(base_errs) - np.median(dino_errs)), 2),
            "mean_diff_bootstrap_ci": [
                round(float(np.percentile(boot_diffs, 5)), 2),
                round(float(np.percentile(boot_diffs, 95)), 2),
            ],
        },
        "stratified_by_occ": stratified,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Baseline Evaluation ({args.split}, n={len(results)})")
    print(f"{'='*60}")
    print(f"  {'Method':<20s} {'Mean':>8s} {'Median':>8s} {'P90':>8s} {'P95':>8s}")
    print(f"  {'Base tracker':<20s} {np.mean(base_errs):>8.2f} {np.median(base_errs):>8.2f} {np.percentile(base_errs,90):>8.2f} {np.percentile(base_errs,95):>8.2f}")
    print(f"  {'DINO cosine':<20s} {np.mean(dino_errs):>8.2f} {np.median(dino_errs):>8.2f} {np.percentile(dino_errs,90):>8.2f} {np.percentile(dino_errs,95):>8.2f}")
    print(f"\n  better_frac: {better_frac:.3f}")
    print(f'  mean diff CI: [{summary["overall"]["mean_diff_bootstrap_ci"][0]}, {summary["overall"]["mean_diff_bootstrap_ci"][1]}]')
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
