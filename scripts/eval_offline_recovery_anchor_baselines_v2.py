#!/usr/bin/env python3
"""
Offline baseline evaluator v2 for recovery-anchor datasets.

Three baselines:
  1. Base tracker (from dataset metadata)
  2. Matched raw DINO: per-support-patch DINO features, template search in search crop
     (closest to online pipeline)
  3. Alt global mean: global mean support descriptor + whole-search argmax

Also reports:
  - Per-occ-length stratified results
  - Grouped bootstrap CI by video
  - Base-bad subset (base_error > threshold)

Usage:
  python scripts/eval_offline_recovery_anchor_baselines_v2.py \
    --dataset-dir outputs/recovery_anchor_dataset_v3_full \
    --split val \
    --output outputs/recovery_anchor_baselines_v3_val.json
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
from utils.crop_utils import extract_crop


def load_samples(dataset_dir: Path, split: str) -> List[Dict]:
    index_path = dataset_dir / f"index_{split}.json"
    if not index_path.exists():
        return []
    with open(index_path) as f:
        return json.load(f)


def matched_dino_baseline(
    extractor,
    sample: Dict,
    dataset_dir: Path,
    device: torch.device,
) -> float:
    """Matched raw DINO baseline: use precomputed support_descriptor + search_feature_map,
    do cosine similarity, find argmax, convert to pixel coords.

    This is closest to what the learned head will consume.
    """
    npz_path = dataset_dir / sample["file"]
    data = np.load(str(npz_path))

    if "support_descriptor" not in data or "search_feature_map" not in data:
        return float("nan")

    supp_desc = torch.from_numpy(data["support_descriptor"]).to(device)  # (D,)
    search_fmap = torch.from_numpy(data["search_feature_map"]).to(device)  # (D, h, w)

    # Cosine similarity map
    D, h, w = search_fmap.shape
    sim = torch.matmul(supp_desc.unsqueeze(0), search_fmap.reshape(D, h * w)).reshape(h, w)
    sim = sim.cpu().numpy()

    # Argmax -> feature map coords -> pixel coords
    peak_idx = np.unravel_index(sim.argmax(), sim.shape)
    search_size = sample["search_crop_size"]
    scale = search_size / float(h)
    pred_in_crop = np.array([(peak_idx[1] + 0.5) * scale, (peak_idx[0] + 0.5) * scale], dtype=np.float32)
    base_xy = np.array(sample["base_xy"], dtype=np.float32)
    center_offset = np.array([search_size / 2.0, search_size / 2.0], dtype=np.float32)
    pred_xy = base_xy - center_offset + pred_in_crop

    gt_xy = np.array(sample["gt_xy"], dtype=np.float32)
    return float(np.linalg.norm(pred_xy - gt_xy))


def compute_metrics(errors: np.ndarray, baseline_errors: np.ndarray) -> Dict:
    return {
        "count": int(len(errors)),
        "mean": round(float(np.mean(errors)), 2),
        "median": round(float(np.median(errors)), 2),
        "p90": round(float(np.percentile(errors, 90)), 2),
        "p95": round(float(np.percentile(errors, 95)), 2),
        "lt4px": round(float(np.mean(errors < 4.0)), 4),
        "better_frac": round(float(np.mean(errors < baseline_errors)), 3) if baseline_errors is not None else None,
    }


def grouped_bootstrap_ci(
    errors: np.ndarray,
    video_names: List[str],
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict:
    """Sequence-grouped bootstrap for mean error."""
    rng = np.random.default_rng(seed)
    unique_vids = list(set(video_names))
    boot_means = []
    for _ in range(n_boot):
        chosen = rng.choice(unique_vids, size=len(unique_vids), replace=True)
        idx = []
        for v in chosen:
            idx.extend([i for i, vn in enumerate(video_names) if vn == v])
        boot_means.append(float(np.mean(errors[np.array(idx)])))
    return {
        "mean": round(float(np.mean(boot_means)), 2),
        "ci_90": [round(float(np.percentile(boot_means, 5)), 2),
                  round(float(np.percentile(boot_means, 95)), 2)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--weights", type=str,
                        default="/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DINOFeatureExtractor(Path(args.weights), device)
    dataset_dir = Path(args.dataset_dir)

    samples = load_samples(dataset_dir, args.split)
    print(f"Loaded {len(samples)} {args.split} samples")
    if not samples:
        return

    video_names = [s["video_name"] for s in samples]
    unique_vids = sorted(set(video_names))
    print(f"Videos: {unique_vids}")

    # Compute errors
    base_errors = np.array([s["base_error_px"] for s in samples])
    matched_errors = []
    for s in samples:
        err = matched_dino_baseline(extractor, s, dataset_dir, device)
        matched_errors.append(err)
    matched_errors = np.array(matched_errors)
    valid = ~np.isnan(matched_errors)

    # Overall metrics
    print(f"\n{'='*60}")
    print(f"Overall (n={len(samples)}, {len(unique_vids)} videos)")
    print(f"{'='*60}")
    print(f"  {'Method':<25s} {'Mean':>8s} {'Median':>8s} {'P90':>8s} {'P95':>8s} {'<4px':>6s}")
    print(f"  {'Base tracker':<25s} {np.mean(base_errors):>8.2f} {np.median(base_errors):>8.2f} {np.percentile(base_errors,90):>8.2f} {np.percentile(base_errors,95):>8.2f} {np.mean(base_errors<4):>6.3f}")
    if valid.any():
        me = matched_errors[valid]
        print(f"  {'Matched DINO':<25s} {np.mean(me):>8.2f} {np.median(me):>8.2f} {np.percentile(me,90):>8.2f} {np.percentile(me,95):>8.2f} {np.mean(me<4):>6.3f}")

    # Stratified by occ length
    occ = np.array([s["occ_length"] for s in samples])
    print(f"\n--- By occlusion length ---")
    stratified = {}
    for lo, hi, label in [(20, 100, "20-100"), (100, 200, "100-200"), (200, 500, "200-500"), (500, 10000, "500+")]:
        m = (occ >= lo) & (occ < hi)
        if m.sum() == 0: continue
        be = base_errors[m]
        me = matched_errors[m]
        v = valid[m]
        row = {"n": int(m.sum())}
        row["base"] = compute_metrics(be, None)
        if v.any():
            row["matched_dino"] = compute_metrics(me[v], be[v])
        stratified[label] = row
        print(f"  {label:>10s} n={m.sum():>4d}: base_med={np.median(be):.1f}", end="")
        if v.any():
            print(f", matched_med={np.median(me[v]):.1f}, better={np.mean(me[v]<be[v]):.3f}", end="")
        print()

    # Grouped bootstrap
    print(f"\n--- Grouped bootstrap (video-level) ---")
    base_boot = grouped_bootstrap_ci(base_errors, video_names)
    print(f"  Base: mean={base_boot['mean']}, CI90={base_boot['ci_90']}")
    if valid.any():
        me_valid = matched_errors[valid]
        vids_valid = [video_names[i] for i in range(len(video_names)) if valid[i]]
        matched_boot = grouped_bootstrap_ci(me_valid, vids_valid)
        print(f"  Matched: mean={matched_boot['mean']}, CI90={matched_boot['ci_90']}")

    # Base-bad subset
    for thr in [16, 32]:
        bad = base_errors > thr
        if bad.sum() == 0: continue
        print(f"\n--- Base error > {thr}px (n={bad.sum()}) ---")
        print(f"  Base: mean={np.mean(base_errors[bad]):.1f}, median={np.median(base_errors[bad]):.1f}")
        if valid[bad].any():
            print(f"  Matched: mean={np.mean(matched_errors[bad][valid[bad]]):.1f}, median={np.median(matched_errors[bad][valid[bad]]):.1f}")

    # Save
    result = {
        "split": args.split,
        "n_samples": len(samples),
        "n_videos": len(unique_vids),
        "videos": unique_vids,
        "overall": {
            "base": compute_metrics(base_errors, None),
            "matched_dino": compute_metrics(matched_errors[valid], base_errors[valid]) if valid.any() else None,
        },
        "stratified_by_occ": stratified,
        "bootstrap": {
            "base": base_boot,
            "matched_dino": matched_boot if valid.any() else None,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
