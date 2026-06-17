#!/usr/bin/env python3
"""
DiReCT Quick-Screen: verify if cross-attention between support memory
and current local feature patch has minimal localization ability.

Tests 3 variants:
  A. Random linear projections (no training)
  B. Tiny supervised tuning (100-300 steps)
  C. Base-centered Gaussian baseline

Uses existing anchor dataset: support_descriptor + search_feature_map.

Usage:
  python scripts/audit_direct_recovery_attention.py \
    --dataset-dir outputs/recovery_anchor_dataset_v3_full \
    --index-json outputs/recovery_anchor_dataset_v3_full/index_val.json \
    --output-dir outputs/direct_recovery_attention_audit \
    --tune-steps 300
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sf(v, d=0.0):
    try: return float(v)
    except: return float(d)


def load_sample(ds_dir, entry):
    """Load support_descriptor and search_feature_map for one entry."""
    npz_path = ds_dir / entry["file"]
    if not npz_path.exists():
        return None
    data = np.load(str(npz_path))
    sd = data.get("support_descriptor")  # (384,) or (1, 384)
    sfm = data.get("search_feature_map")  # (384, Hf, Wf)
    if sd is None or sfm is None:
        return None
    if sd.ndim == 2:
        sd = sd[0]  # (384,)
    # Check non-zero
    if np.count_nonzero(sd) == 0 or np.count_nonzero(sfm) == 0:
        return None
    return {
        "support_descriptor": sd.astype(np.float32),
        "search_feature_map": sfm.astype(np.float32),
        "base_xy": np.array(entry["base_xy"], dtype=np.float32),
        "gt_xy": np.array(entry["gt_xy"], dtype=np.float32),
        "base_error_px": _sf(entry.get("base_error_px", 0)),
        "search_crop_size": int(entry.get("search_crop_size", 224)),
        "image_H": int(entry.get("image_H", 480)),
        "image_W": int(entry.get("image_W", 910)),
    }


def gt_position_in_feature_map(sample):
    """Map GT position to feature map coordinates relative to search crop center."""
    sfm = sample["search_feature_map"]
    _, Hf, Wf = sfm.shape
    crop_size = sample["search_crop_size"]
    base_xy = sample["base_xy"]
    gt_xy = sample["gt_xy"]

    # GT offset from base in pixel space
    offset_px = gt_xy - base_xy  # (dx, dy) in pixels

    # Map to feature grid coords
    # Feature map is centered on base position
    # Each feature covers crop_size/Hf pixels
    feat_scale_y = Hf / float(crop_size)
    feat_scale_x = Wf / float(crop_size)

    gt_fy = Hf / 2.0 + offset_px[1] * feat_scale_y  # y offset
    gt_fx = Wf / 2.0 + offset_px[0] * feat_scale_x  # x offset

    # Clamp to valid range
    gt_fy = np.clip(gt_fy, 0, Hf - 1)
    gt_fx = np.clip(gt_fx, 0, Wf - 1)

    return int(round(gt_fy)), int(round(gt_fx))


def compute_attention_map(support_desc, search_fmap, q_proj=None, k_proj=None):
    """Compute cross-attention: Q=support_desc, K=search_fmap.

    Args:
        support_desc: (384,) support descriptor
        search_fmap: (384, Hf, Wf) search feature map
        q_proj: optional linear projection for Q
        k_proj: optional linear projection for K

    Returns:
        attention_map: (Hf, Wf) attention weights
    """
    D, Hf, Wf = search_fmap.shape

    # Flatten search features: (Hf*Wf, D)
    k = torch.from_numpy(search_fmap).reshape(D, Hf * Wf).t()  # (N, D)
    q = torch.from_numpy(support_desc).unsqueeze(0)  # (1, D)

    if q_proj is not None:
        # Ensure same device
        dev = next(iter(q_proj.parameters())).device
        q = q.to(dev)
        k = k.to(dev)
        q = q_proj(q)
        k = k_proj(k)

    # Cosine similarity attention
    q_norm = F.normalize(q, dim=-1)
    k_norm = F.normalize(k, dim=-1)
    scores = torch.matmul(q_norm, k_norm.t()).squeeze(0)  # (N,)
    attn = F.softmax(scores, dim=0)
    attn_map = attn.reshape(Hf, Wf)

    return attn_map.numpy(), scores.numpy()


def gaussian_baseline(Hf, Wf, sigma_frac=0.15):
    """Gaussian centered at feature map center (base position)."""
    cy, cx = Hf / 2.0, Wf / 2.0
    sigma = max(Hf, Wf) * sigma_frac
    yy, xx = np.mgrid[0:Hf, 0:Wf]
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def gt_rank_in_map(attn_map, gt_fy, gt_fx):
    """Get GT's rank in the attention map (1 = highest attention)."""
    Hf, Wf = attn_map.shape
    flat = attn_map.flatten()
    gt_idx = gt_fy * Wf + gt_fx
    gt_score = flat[gt_idx]
    rank = int((flat >= gt_score).sum())
    return rank, float(gt_score)


def peak_position(attn_map):
    """Get peak position in feature map."""
    idx = np.argmax(attn_map)
    Hf, Wf = attn_map.shape
    return idx // Wf, idx % Wf


def distance_px(fy1, fx1, fy2, fx2, crop_size, Hf, Wf):
    """Distance in pixels between two feature map positions."""
    py1 = fy1 * crop_size / Hf
    px1 = fx1 * crop_size / Wf
    py2 = fy2 * crop_size / Hf
    px2 = fx2 * crop_size / Wf
    return float(np.sqrt((py1 - py2) ** 2 + (px1 - px2) ** 2))


class TinyQKProjection(nn.Module):
    """Minimal Q/K projection for cross-attention tuning."""
    def __init__(self, dim=384, proj_dim=64):
        super().__init__()
        self.q_proj = nn.Linear(dim, proj_dim, bias=False)
        self.k_proj = nn.Linear(dim, proj_dim, bias=False)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, q, k):
        """q: (1, D), k: (N, D) -> scores (N,)"""
        q = F.normalize(self.q_proj(q), dim=-1)
        k = F.normalize(self.k_proj(k), dim=-1)
        return (q @ k.t()).squeeze(0) * self.temperature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--index-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tune-steps", type=int, default=300)
    parser.add_argument("--tune-lr", type=float, default=1e-2)
    args = parser.parse_args()

    ds_dir = Path(args.dataset_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load index
    index = json.load(open(args.index_json))
    print(f"Index: {len(index)} entries")

    # Load all samples
    samples = []
    for entry in index:
        s = load_sample(ds_dir, entry)
        if s is not None:
            s["sample_id"] = entry.get("sample_id")
            s["video_name"] = entry.get("video_name")
            s["point_idx"] = entry.get("point_idx")
            s["group_base16"] = _sf(entry.get("base_error_px", 0)) > 16
            s["group_base32"] = _sf(entry.get("base_error_px", 0)) > 32
            samples.append(s)
    print(f"Loaded: {len(samples)} valid samples")

    if not samples:
        print("No valid samples!")
        return

    # Get feature map dimensions from first sample
    _, Hf, Wf = samples[0]["search_feature_map"].shape
    N = Hf * Wf
    D = 384
    print(f"Feature map: {Hf}x{Wf} = {N} positions, D={D}")

    # ====================================================================
    # Variant C: Gaussian baseline (compute first, no model needed)
    # ====================================================================
    print("\n--- Variant C: Gaussian baseline ---")
    gauss = gaussian_baseline(Hf, Wf)

    gauss_results = []
    for s in samples:
        gt_fy, gt_fx = gt_position_in_feature_map(s)
        rank, score = gt_rank_in_map(gauss, gt_fy, gt_fx)
        peak_fy, peak_fx = peak_position(gauss)
        peak_to_gt = distance_px(peak_fy, peak_fx, gt_fy, gt_fx, s["search_crop_size"], Hf, Wf)
        peak_to_base = distance_px(peak_fy, peak_fx, Hf//2, Wf//2, s["search_crop_size"], Hf, Wf)
        gauss_results.append({
            "gt_rank": rank, "gt_rank_pct": rank / N,
            "peak_to_gt_px": peak_to_gt, "peak_to_base_px": peak_to_base,
            "gt_top1pct": rank <= max(1, N // 100),
            "gt_top5pct": rank <= max(1, N // 20),
        })

    print(f"  GT top-1% hit: {sum(r['gt_top1pct'] for r in gauss_results)}/{len(gauss_results)}")
    print(f"  GT top-5% hit: {sum(r['gt_top5pct'] for r in gauss_results)}/{len(gauss_results)}")
    print(f"  Peak-to-GT median: {np.median([r['peak_to_gt_px'] for r in gauss_results]):.1f}px")

    # ====================================================================
    # Variant A: Random linear projections
    # ====================================================================
    print("\n--- Variant A: Random projections ---")
    torch.manual_seed(42)
    q_proj_r = nn.Linear(D, 64, bias=False)  # CPU
    k_proj_r = nn.Linear(D, 64, bias=False)  # CPU

    random_results = []
    for s in samples:
        with torch.no_grad():
            attn_map, scores = compute_attention_map(
                s["support_descriptor"], s["search_feature_map"],
                q_proj_r, k_proj_r
            )

        gt_fy, gt_fx = gt_position_in_feature_map(s)
        rank, _ = gt_rank_in_map(attn_map, gt_fy, gt_fx)
        peak_fy, peak_fx = peak_position(attn_map)
        peak_to_gt = distance_px(peak_fy, peak_fx, gt_fy, gt_fx, s["search_crop_size"], Hf, Wf)
        random_results.append({
            "gt_rank": rank, "gt_rank_pct": rank / N,
            "peak_to_gt_px": peak_to_gt,
            "gt_top1pct": rank <= max(1, N // 100),
            "gt_top5pct": rank <= max(1, N // 20),
        })

    print(f"  GT top-1% hit: {sum(r['gt_top1pct'] for r in random_results)}/{len(random_results)}")
    print(f"  GT top-5% hit: {sum(r['gt_top5pct'] for r in random_results)}/{len(random_results)}")
    print(f"  Peak-to-GT median: {np.median([r['peak_to_gt_px'] for r in random_results]):.1f}px")

    # ====================================================================
    # Variant B: Tiny supervised tuning
    # ====================================================================
    print(f"\n--- Variant B: Tiny supervised tuning ({args.tune_steps} steps) ---")
    torch.manual_seed(123)
    model = TinyQKProjection(D, 64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.tune_lr)

    # Flatten all training data
    all_q = torch.from_numpy(np.stack([s["support_descriptor"] for s in samples])).to(device)  # (M, D)
    all_k = torch.from_numpy(np.stack([s["search_feature_map"].reshape(D, N).T for s in samples])).to(device)  # (M, N, D)
    all_gt = []
    for s in samples:
        gt_fy, gt_fx = gt_position_in_feature_map(s)
        all_gt.append(gt_fy * Wf + gt_fx)
    all_gt = torch.tensor(all_gt, dtype=torch.long, device=device)  # (M,)

    for step in range(args.tune_steps):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        for i in range(len(samples)):
            scores = model(all_q[i:i+1], all_k[i])  # (N,)
            # Cross-entropy: push GT position to top
            loss = F.cross_entropy(scores.unsqueeze(0), all_gt[i:i+1])
            total_loss = total_loss + loss
        total_loss = total_loss / len(samples)
        total_loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}: loss={total_loss.item():.4f}")

    # Evaluate tuned model
    tuned_results = []
    model.eval()
    with torch.no_grad():
        for i, s in enumerate(samples):
            scores = model(all_q[i:i+1], all_k[i]).cpu().numpy()
            attn_map = scores.reshape(Hf, Wf)
            # Softmax for attention map
            attn_map_sm = np.exp(attn_map - attn_map.max())
            attn_map_sm = attn_map_sm / attn_map_sm.sum()

            gt_fy, gt_fx = gt_position_in_feature_map(s)
            rank_raw, _ = gt_rank_in_map(attn_map, gt_fy, gt_fx)
            rank_sm, _ = gt_rank_in_map(attn_map_sm, gt_fy, gt_fx)
            peak_fy, peak_fx = peak_position(attn_map)
            peak_to_gt = distance_px(peak_fy, peak_fx, gt_fy, gt_fx, s["search_crop_size"], Hf, Wf)
            peak_to_base = distance_px(peak_fy, peak_fx, Hf//2, Wf//2, s["search_crop_size"], Hf, Wf)

            tuned_results.append({
                "gt_rank": rank_raw, "gt_rank_pct": rank_raw / N,
                "gt_rank_softmax": rank_sm,
                "peak_to_gt_px": peak_to_gt, "peak_to_base_px": peak_to_base,
                "gt_top1pct": rank_raw <= max(1, N // 100),
                "gt_top5pct": rank_raw <= max(1, N // 20),
                "gt_top1pct_sm": rank_sm <= max(1, N // 100),
                "gt_top5pct_sm": rank_sm <= max(1, N // 20),
            })

    print(f"  GT top-1% hit (raw): {sum(r['gt_top1pct'] for r in tuned_results)}/{len(tuned_results)}")
    print(f"  GT top-5% hit (raw): {sum(r['gt_top5pct'] for r in tuned_results)}/{len(tuned_results)}")
    print(f"  GT top-5% hit (softmax): {sum(r['gt_top5pct_sm'] for r in tuned_results)}/{len(tuned_results)}")
    print(f"  Peak-to-GT median: {np.median([r['peak_to_gt_px'] for r in tuned_results]):.1f}px")
    print(f"  Peak-to-base median: {np.median([r['peak_to_base_px'] for r in tuned_results]):.1f}px")

    # ====================================================================
    # Summary + Pass/Fail
    # ====================================================================
    def summarize(results, name, extra_keys=None):
        gt_ranks = [r["gt_rank_pct"] for r in results]
        peak_to_gts = [r["peak_to_gt_px"] for r in results]
        top1pct = sum(r.get("gt_top1pct", r.get("gt_top1pct_sm", False)) for r in results)
        top5pct = sum(r.get("gt_top5pct", r.get("gt_top5pct_sm", False)) for r in results)
        return {
            "variant": name,
            "n": len(results),
            "gt_rank_pct_median": round(float(np.median(gt_ranks)), 4),
            "gt_rank_pct_mean": round(float(np.mean(gt_ranks)), 4),
            "peak_to_gt_px_median": round(float(np.median(peak_to_gts)), 2),
            "peak_to_gt_px_mean": round(float(np.mean(peak_to_gts)), 2),
            "gt_top1pct_hit": int(top1pct),
            "gt_top1pct_hit_rate": round(top1pct / max(1, len(results)), 3),
            "gt_top5pct_hit": int(top5pct),
            "gt_top5pct_hit_rate": round(top5pct / max(1, len(results)), 3),
        }

    summary_random = summarize(random_results, "random_projection")
    summary_gauss = summarize(gauss_results, "gaussian_baseline")
    summary_tuned = summarize(tuned_results, "tiny_tuned")

    # Hard subset
    hard_idx = [i for i, s in enumerate(samples) if s["group_base16"]]
    tuned_hard = [tuned_results[i] for i in hard_idx] if hard_idx else []
    gauss_hard = [gauss_results[i] for i in hard_idx] if hard_idx else []
    summary_tuned_hard = summarize(tuned_hard, "tiny_tuned_base16") if tuned_hard else {}
    summary_gauss_hard = summarize(gauss_hard, "gaussian_baseline_base16") if gauss_hard else {}

    # Pass/Fail
    # Criterion 1: tuned GT top-5% hit rate > gaussian baseline
    c1 = summary_tuned["gt_top5pct_hit_rate"] > summary_gauss["gt_top5pct_hit_rate"]
    # Criterion 2: tuned peak-to-GT < gaussian baseline
    c2 = summary_tuned["peak_to_gt_px_median"] < summary_gauss["peak_to_gt_px_median"]
    # Criterion 3: hard subset has signal
    c3 = False
    if summary_tuned_hard and summary_gauss_hard:
        c3 = summary_tuned_hard["gt_top5pct_hit_rate"] > summary_gauss_hard["gt_top5pct_hit_rate"]
        # Or peak-to-GT on hard subset is better
        c3 = c3 or (summary_tuned_hard["peak_to_gt_px_median"] < summary_gauss_hard["peak_to_gt_px_median"])

    passed = sum([c1, c2, c3])
    # Check failure conditions
    f1 = summary_tuned["gt_top5pct_hit_rate"] <= 0.05  # near random
    f2 = summary_tuned["peak_to_gt_px_median"] >= summary_gauss["peak_to_gt_px_median"]  # not better than gaussian
    f3 = summary_random["gt_top5pct_hit_rate"] >= summary_tuned["gt_top5pct_hit_rate"]  # random ≈ tuned

    if f1 or (f2 and f3):
        verdict = "FAIL"
    elif passed >= 2:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    summary = {
        "n_samples": len(samples),
        "feature_map_size": f"{Hf}x{Wf}",
        "tune_steps": args.tune_steps,
        "variants": {
            "gaussian_baseline": summary_gauss,
            "random_projection": summary_random,
            "tiny_tuned": summary_tuned,
        },
        "hard_subset": {
            "tiny_tuned_base16": summary_tuned_hard,
            "gaussian_baseline_base16": summary_gauss_hard,
        },
        "criteria": {
            "tuned_top5pct_better_than_gaussian": c1,
            "tuned_peak_closer_than_gaussian": c2,
            "hard_subset_signal": c3,
            "criteria_passed": passed,
        },
        "failure_checks": {
            "tuned_near_random": f1,
            "tuned_not_better_than_gaussian": f2,
            "random_approx_tuned": f3,
        },
        "verdict": verdict,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Per-sample
    per_sample = []
    for i, s in enumerate(samples):
        per_sample.append({
            "sample_id": s.get("sample_id"),
            "video_name": s.get("video_name"),
            "point_idx": s.get("point_idx"),
            "base_error_px": s["base_error_px"],
            "group_base16": s["group_base16"],
            "group_base32": s["group_base32"],
            "gt_rank_random": random_results[i]["gt_rank"],
            "gt_rank_tuned": tuned_results[i]["gt_rank"],
            "gt_top1pct_random": random_results[i]["gt_top1pct"],
            "gt_top5pct_random": random_results[i]["gt_top5pct"],
            "gt_top1pct_tuned": tuned_results[i]["gt_top1pct"],
            "gt_top5pct_tuned": tuned_results[i]["gt_top5pct"],
            "peak_to_gt_px_random": random_results[i]["peak_to_gt_px"],
            "peak_to_gt_px_tuned": tuned_results[i]["peak_to_gt_px"],
            "peak_to_gt_px_gaussian": gauss_results[i]["peak_to_gt_px"],
            "peak_to_base_px_tuned": tuned_results[i].get("peak_to_base_px", 0),
        })

    with open(out_dir / "per_sample_attention_stats.jsonl", "w") as f:
        for p in per_sample:
            f.write(json.dumps(p) + "\n")

    # Heatmap summary
    with open(out_dir / "heatmap_summary.json", "w") as f:
        json.dump({
            "gaussian_baseline": summary_gauss,
            "random_projection": summary_random,
            "tiny_tuned": summary_tuned,
            "hard_subset_tuned": summary_tuned_hard,
            "hard_subset_gaussian": summary_gauss_hard,
        }, f, indent=2)

    # Print
    print(f"\n{'='*60}")
    print(f"DiReCT Quick-Screen Results")
    print(f"{'='*60}")
    for name, s in [("Gaussian baseline", summary_gauss), ("Random projection", summary_random), ("Tiny tuned", summary_tuned)]:
        print(f"  {name:25s}: top1%={s['gt_top1pct_hit_rate']:.3f}, top5%={s['gt_top5pct_hit_rate']:.3f}, "
              f"peak_to_gt_med={s['peak_to_gt_px_median']:.1f}px")
    if summary_tuned_hard:
        print(f"  {'Tiny tuned (base>16)':25s}: top5%={summary_tuned_hard['gt_top5pct_hit_rate']:.3f}, "
              f"peak_to_gt_med={summary_tuned_hard['peak_to_gt_px_median']:.1f}px")
    if summary_gauss_hard:
        print(f"  {'Gaussian (base>16)':25s}: top5%={summary_gauss_hard['gt_top5pct_hit_rate']:.3f}, "
              f"peak_to_gt_med={summary_gauss_hard['peak_to_gt_px_median']:.1f}px")
    print(f"\n  Criteria: top5% better={c1}, peak closer={c2}, hard signal={c3}")
    print(f"  Verdict: {verdict} ({passed}/3 criteria)")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
