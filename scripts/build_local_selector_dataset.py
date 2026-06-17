#!/usr/bin/env python3
"""
Build local selector dataset from base-centered DINO local search.

Each sample is a recovery event with K candidates. Label = oracle_index (which
candidate is best) or abstain (K) if none beats base by >2px.

Stage-2b notes:
  - supports explicit split selection (`train` / `val`)
  - writes split-specific JSONL files
  - adds stronger event/candidate statistics for gate + ranker training

Usage:
  python scripts/build_local_selector_dataset.py \
    --config configs/fspt_online_recovery_real_eval256.yaml \
    --checkpoint checkpoints/...best.pth \
    --output-dir outputs/local_selector_dataset_v2 \
    --split train \
    --max-batches 30 --topk 5 --crop-radius 64
"""

from __future__ import annotations
import argparse, json, sys, math, signal
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse from audit script
from scripts.audit_base_centered_local_recovery import (
    load_config, _TimeoutError, _timeout_handler,
    find_t_last_visible, precompute_frame_dino_features, local_search,
)


def build_loader_from_config(config, split: str):
    """Create train/val dataloader using the same filtering logic as eval scripts."""
    from datasets import get_dataloader

    split = str(split).strip().lower()
    if split not in ("train", "val"):
        raise ValueError(f"Unsupported split={split!r}; expected train|val")

    if not hasattr(config, "data") or not hasattr(config.data, split):
        raise ValueError(f"Config missing data.{split}.")

    ds_cfg = getattr(config.data, split)
    ds_name = str(getattr(ds_cfg, "dataset", "")).lower()
    extra_args = {}

    for key in (
        "backend",
        "allow_synthetic_tracks",
        "annotation_file",
        "use_tfds",
        "tfds_name",
        "shuffle_buffer",
        "shuffle_files",
        "tfds_low_memory",
        "tfds_read_buffer_size",
        "deterministic_sampling",
        "deterministic_seed",
    ):
        if hasattr(ds_cfg, key):
            value = getattr(ds_cfg, key)
            if value is not None:
                extra_args[key] = value

    if hasattr(ds_cfg, "sampling") and ds_cfg.sampling is not None:
        extra_args["sampling"] = ds_cfg.sampling
    if hasattr(ds_cfg, "resolution") and ds_cfg.resolution is not None:
        extra_args["resolution"] = tuple(ds_cfg.resolution)
    # Dataset building must be deterministic and GT-aligned.
    extra_args["augmentation"] = {"enabled": False}
    if hasattr(ds_cfg, "num_points"):
        extra_args["num_points"] = ds_cfg.num_points

    if "davis" in ds_name or "kinetics" in ds_name:
        query_mode = getattr(ds_cfg, "query_mode", None) if hasattr(ds_cfg, "query_mode") else None
        if query_mode is None and hasattr(config, "evaluation") and hasattr(config.evaluation, "query_mode"):
            query_mode = getattr(config.evaluation, "query_mode")
        if query_mode is not None and str(query_mode).strip().lower() not in ("", "none", "null"):
            extra_args["query_mode"] = str(query_mode).strip()
        if hasattr(ds_cfg, "query_stride") and getattr(ds_cfg, "query_stride") is not None:
            extra_args["query_stride"] = max(1, int(ds_cfg.query_stride))
        if hasattr(ds_cfg, "points_order") and getattr(ds_cfg, "points_order") is not None:
            extra_args["points_order"] = str(ds_cfg.points_order)

    if "kinetics" in ds_name:
        if hasattr(ds_cfg, "max_frames") and ds_cfg.max_frames is not None:
            extra_args["max_frames"] = int(ds_cfg.max_frames)
        elif hasattr(ds_cfg, "num_frames") and ds_cfg.num_frames is not None and int(ds_cfg.num_frames) > 0:
            extra_args["max_frames"] = int(ds_cfg.num_frames)
    elif "davis" not in ds_name:
        if hasattr(ds_cfg, "num_frames") and ds_cfg.num_frames is not None:
            extra_args["num_frames"] = ds_cfg.num_frames

    if hasattr(ds_cfg, "base_tracks_dir") and ds_cfg.base_tracks_dir is not None:
        extra_args["base_tracks_dir"] = ds_cfg.base_tracks_dir
        if hasattr(ds_cfg, "base_tracks_strict"):
            extra_args["base_tracks_strict"] = bool(ds_cfg.base_tracks_strict)
        if hasattr(ds_cfg, "base_tracks_query_tol") and ds_cfg.base_tracks_query_tol is not None:
            extra_args["base_tracks_query_tol"] = float(ds_cfg.base_tracks_query_tol)

    num_workers = 0
    if split == "val":
        num_workers = int(getattr(getattr(config, "evaluation", None), "num_workers", None) or 0)
    if num_workers <= 0:
        num_workers = int(getattr(getattr(config, "training", None), "num_workers", None) or 4)

    return get_dataloader(
        name=ds_cfg.dataset,
        root=ds_cfg.root,
        batch_size=1,
        split=split,
        num_workers=num_workers,
        pin_memory=False,
        seed=int(getattr(getattr(config, "experiment", None), "seed", 0) or 0),
        **extra_args,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--max-batches", type=int, default=30)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--crop-radius", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    # Load model
    from models.cotracker_refiner import CoTrackerFSPTRefiner
    model = CoTrackerFSPTRefiner(cfg.get("model", {}))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    model._export_candidates_debug = True
    model = model.to(device).eval()

    # DINO
    from models.recovery_features import DINORecoveryExtractor
    dino_extractor = DINORecoveryExtractor()
    dino_extractor._ensure_loaded(device)

    dataloader = build_loader_from_config(cfg, args.split)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import time as _time
    samples = []
    sample_id = 0

    print(
        f"Building local selector dataset: split={args.split}, "
        f"max_batches={args.max_batches}, topk={args.topk}, r={args.crop_radius}"
    )

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.max_batches:
            break
        if not isinstance(batch, dict):
            continue

        video = batch.get("video")
        query_points = batch.get("query_points")
        target_points = batch.get("target_points", batch.get("tracks"))
        occluded = batch.get("occluded")
        video_name = batch.get("video_name", ["unknown"])
        if video is None or query_points is None or target_points is None:
            continue

        video_dev = video.to(device)
        query_dev = query_points.to(device)
        if video_dev.dim() == 4: video_dev = video_dev.unsqueeze(0)
        if query_dev.dim() == 2: query_dev = query_dev.unsqueeze(0)
        if target_points.dim() == 3: target_points = target_points.unsqueeze(0)
        occ_np = None
        if occluded is not None:
            if occluded.dim() == 2: occluded = occluded.unsqueeze(0)
            occ_np = occluded.cpu().numpy()

        B, T, C, H, W = video_dev.shape
        N_pts = query_dev.shape[1]
        meta = {"video_name": video_name, "base_tracks": batch.get("base_tracks"), "base_visibility": batch.get("base_visibility")}

        with torch.no_grad():
            try:
                old_h = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(90)
                outputs = model(video_dev, query_dev, meta=meta, return_info=True)
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_h)
            except _TimeoutError:
                signal.alarm(0)
                print(f"  Batch {batch_idx}: TIMEOUT")
                continue
            except Exception as e:
                signal.alarm(0)
                print(f"  Batch {batch_idx}: ERROR {e}")
                continue

        info = outputs[2] if len(outputs) > 2 and isinstance(outputs[2], dict) else {}
        relocal_mask = info.get("relocal_mask")
        if not isinstance(relocal_mask, torch.Tensor):
            continue

        base_tracks = info.get("base_tracks", batch.get("base_tracks"))
        if base_tracks is not None: base_tracks = base_tracks.cpu().numpy()
        else: base_tracks = outputs[0].cpu().numpy()
        gt_tracks = target_points.cpu().numpy()

        orig_size = batch.get("original_size")
        if isinstance(orig_size, torch.Tensor):
            orig_h, orig_w = int(orig_size[0, 0].item()), int(orig_size[0, 1].item())
        else:
            orig_h, orig_w = H, W

        vid_name = str(video_name[0] if isinstance(video_name, list) else video_name)

        # Precompute DINO
        video_np = (video_dev[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8).transpose(0, 2, 3, 1)
        dino_cache = {}

        def get_dino(t):
            if t not in dino_cache:
                dino_cache[t] = precompute_frame_dino_features(dino_extractor, video_np[t], device)
            return dino_cache[t]

        for b in range(B):
            for n_idx in range(N_pts):
                rm = relocal_mask[b, n_idx].cpu().numpy()
                triggered = np.where(rm)[0]
                if len(triggered) == 0:
                    continue

                for t0 in triggered:
                    t0 = int(t0)
                    if occ_np is not None and occ_np[b, n_idx, t0]:
                        continue

                    t_last_vis = find_t_last_visible(occ_np[b, n_idx], t0) if occ_np is not None else max(0, t0 - 1)
                    occ_length = t0 - t_last_vis - 1

                    gt_norm = gt_tracks[b, n_idx, t0]
                    if np.linalg.norm(gt_norm) < 1e-6:
                        continue

                    base_norm = base_tracks[b, n_idx, t0]
                    base_px = np.array([base_norm[0] * orig_w, base_norm[1] * orig_h])
                    gt_px = np.array([gt_norm[0] * orig_w, gt_norm[1] * orig_h])
                    base_err = float(np.linalg.norm(base_px - gt_px))

                    # GT must be in crop
                    r = args.crop_radius
                    if np.linalg.norm(gt_px - base_px) >= r:
                        continue

                    re_feat = get_dino(t0)
                    sf_feat = get_dino(t_last_vis)
                    query_at_last = base_tracks[b, n_idx, t_last_vis]

                    try:
                        results = local_search(re_feat, sf_feat, query_at_last, base_norm, orig_w, orig_h, r, topk=args.topk)
                    except Exception:
                        continue

                    if len(results) < 2:
                        continue

                    K = len(results)

                    # Compute per-candidate features
                    top1_norm = np.array(results[0]["pos_norm"])
                    top1_px = np.array([top1_norm[0] * orig_w, top1_norm[1] * orig_h])
                    top1_score = results[0]["score"]
                    top1_err = float(np.linalg.norm(top1_px - gt_px))

                    candidates = []
                    oracle_idx = 0
                    oracle_err = top1_err

                    cand_errors = []
                    for j, tk in enumerate(results):
                        cand_norm = np.array(tk["pos_norm"])
                        cand_px = np.array([cand_norm[0] * orig_w, cand_norm[1] * orig_h])
                        cand_err = float(np.linalg.norm(cand_px - gt_px))
                        cand_errors.append(cand_err)
                        cand_score = tk["score"]
                        cand_shift_px = float(np.linalg.norm(cand_px - base_px))
                        cand_shift_norm = float(np.linalg.norm(cand_norm - base_norm))
                        cand_dist_to_top1_px = float(np.linalg.norm(cand_px - top1_px))

                        # Margin: this candidate's score minus score of rank-2
                        if j == 0:
                            margin = top1_score - results[1]["score"] if K > 1 else top1_score
                        elif j == 1:
                            margin = results[0]["score"] - cand_score
                        else:
                            margin = results[0]["score"] - cand_score

                        candidates.append({
                            "rank_by_dino": j,
                            "cand_xy_norm": cand_norm.tolist(),
                            "cand_error_px": round(cand_err, 2),
                            "cand_score": round(cand_score, 4),
                            "cand_margin_to_top2": round(margin, 4),
                            "cand_shift_px_from_base": round(cand_shift_px, 2),
                            "cand_shift_norm_from_base": round(cand_shift_norm, 4),
                            "cand_dist_to_base_px": round(cand_shift_px, 2),
                            "cand_dist_to_top1_px": round(cand_dist_to_top1_px, 2),
                            "cand_is_oracle": False,  # set below
                            "cand_better_than_base": bool(cand_err < base_err),
                            "cand_better_by_2px": bool(cand_err < base_err - 2),
                            "cand_worse_than_base_by_2px": bool(cand_err > base_err + 2),
                            "cand_crop_radius_px": r,
                        })

                        if cand_err < oracle_err:
                            oracle_err = cand_err
                            oracle_idx = j

                    # Mark oracle
                    candidates[oracle_idx]["cand_is_oracle"] = True

                    # Aggregate features
                    scores = [c["cand_score"] for c in candidates]
                    shifts = [c["cand_shift_px_from_base"] for c in candidates]
                    score_probs = np.exp(np.asarray(scores, dtype=np.float64) - float(np.max(scores)))
                    score_probs = score_probs / max(1e-8, float(score_probs.sum()))
                    score_entropy = float(-(score_probs * np.log(score_probs + 1e-8)).sum())
                    score_ranks = np.argsort(np.argsort(scores))[::-1]  # rank 0 = highest score
                    shift_ranks = np.argsort(np.argsort(shifts))

                    for j, c in enumerate(candidates):
                        c["score_rank_normalized"] = round(float(score_ranks[j]) / max(1, K - 1), 3)
                        c["shift_rank_normalized"] = round(float(shift_ranks[j]) / max(1, K - 1), 3)
                        c["candidate_score_minus_top1"] = round(float(c["cand_score"] - scores[0]), 4)
                        c["candidate_score_minus_mean"] = round(float(c["cand_score"] - np.mean(scores)), 4)
                        c["candidate_error_minus_base"] = round(float(c["cand_error_px"] - base_err), 2)

                    # Label: oracle_index or abstain
                    oracle_better_by_2px = candidates[oracle_idx]["cand_better_by_2px"]
                    label = oracle_idx if oracle_better_by_2px else K  # K = abstain

                    sample = {
                        "sample_id": sample_id,
                        "video_name": vid_name,
                        "point_idx": int(n_idx),
                        "t_query": int(t0),
                        "t_last_visible": int(t_last_vis),
                        "t_reentry": int(t0),
                        "occ_length": int(occ_length),
                        "base_xy_norm": base_norm.tolist(),
                        "gt_xy_norm": gt_norm.tolist(),
                        "base_error_px": round(base_err, 2),
                        "group_base16": bool(base_err > 16),
                        "group_base32": bool(base_err > 32),
                        "crop_radius_px": r,
                        "gt_in_crop": True,
                        "num_candidates": K,
                        "oracle_index": oracle_idx,
                        "oracle_error_px": round(oracle_err, 2),
                        "top1_index_by_dino": 0,
                        "top1_error_px": round(top1_err, 2),
                        "oracle_better_by_2px": bool(oracle_better_by_2px),
                        "top1_better_by_2px": bool(top1_err < base_err - 2),
                        "label": label,
                        "is_abstain": label == K,
                        "has_positive_candidate": bool(oracle_better_by_2px),
                        # Event-level features for selector
                        "top1_score": round(top1_score, 4),
                        "top1_margin": round(results[0]["score"] - results[1]["score"] if K > 1 else top1_score, 4),
                        "top1_shift_px": round(float(np.linalg.norm(top1_px - base_px)), 2),
                        "score_std_topk": round(float(np.std(scores)), 4),
                        "score_range_topk": round(float(max(scores) - min(scores)), 4),
                        "score_entropy_topk": round(score_entropy, 4),
                        "peak_sharpness_topk": round(float((scores[0] - np.mean(scores[1:])) if K > 1 else scores[0]), 4),
                        "shift_std_topk": round(float(np.std(shifts)), 2),
                        "shift_range_topk": round(float(max(shifts) - min(shifts)), 2),
                        "cand_error_min_topk": round(float(np.min(cand_errors)), 2),
                        "candidates": candidates,
                    }
                    samples.append(sample)
                    sample_id += 1

        n_so_far = len(samples)
        print(f"  Batch {batch_idx+1}: {n_so_far} samples, dino_cache={len(dino_cache)}")
        sys.stdout.flush()

    # Save
    n = len(samples)
    print(f"\nTotal: {n} samples")

    split_jsonl = out_dir / f"{args.split}.jsonl"
    with open(split_jsonl, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    # Summary stats
    K = args.topk
    labels = [s["label"] for s in samples]
    abstain_count = labels.count(K)
    label_dist = {i: labels.count(i) for i in range(K + 1)}

    base16_samples = [s for s in samples if s["group_base16"]]
    base32_samples = [s for s in samples if s["group_base32"]]

    summary = {
        "split": args.split,
        "n_samples": n,
        "n_sequences": len(set(s["video_name"] for s in samples)),
        "topk": K,
        "crop_radius": args.crop_radius,
        "label_distribution": {str(k): v for k, v in label_dist.items()},
        "abstain_ratio": round(abstain_count / max(1, n), 3),
        "oracle_better_by_2px_frac": round(sum(1 for s in samples if s["oracle_better_by_2px"]) / max(1, n), 3),
        "top1_better_by_2px_frac": round(sum(1 for s in samples if s["top1_better_by_2px"]) / max(1, n), 3),
        "base_median": round(float(np.median([s["base_error_px"] for s in samples])), 2),
        "oracle_median": round(float(np.median([s["oracle_error_px"] for s in samples])), 2),
        "top1_median": round(float(np.median([s["top1_error_px"] for s in samples])), 2),
        "base16": {
            "n": len(base16_samples),
            "abstain_ratio": round(sum(1 for s in base16_samples if s["is_abstain"]) / max(1, len(base16_samples)), 3),
            "oracle_better_by_2px_frac": round(sum(1 for s in base16_samples if s["oracle_better_by_2px"]) / max(1, len(base16_samples)), 3),
            "base_median": round(float(np.median([s["base_error_px"] for s in base16_samples])), 2) if base16_samples else None,
            "oracle_median": round(float(np.median([s["oracle_error_px"] for s in base16_samples])), 2) if base16_samples else None,
        },
        "base32": {
            "n": len(base32_samples),
            "abstain_ratio": round(sum(1 for s in base32_samples if s["is_abstain"]) / max(1, len(base32_samples)), 3),
            "oracle_better_by_2px_frac": round(sum(1 for s in base32_samples if s["oracle_better_by_2px"]) / max(1, len(base32_samples)), 3),
        },
        "per_sequence": {},
    }

    # Per-sequence
    for vid in sorted(set(s["video_name"] for s in samples)):
        vs = [s for s in samples if s["video_name"] == vid]
        summary["per_sequence"][vid] = {
            "n": len(vs),
            "abstain_ratio": round(sum(1 for s in vs if s["is_abstain"]) / len(vs), 3),
            "oracle_better_frac": round(sum(1 for s in vs if s["oracle_better_by_2px"]) / len(vs), 3),
            "base_median": round(float(np.median([s["base_error_px"] for s in vs])), 2),
        }

    summary_path = out_dir / "summary.json"
    merged = {}
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                merged = json.load(f)
        except Exception:
            merged = {}
    merged[args.split] = summary
    with open(summary_path, "w") as f:
        json.dump(merged, f, indent=2)

    # Print
    print(f"\n{'='*60}")
    print(f"Local Selector Dataset")
    print(f"{'='*60}")
    print(f"  n={n}, sequences={summary['n_sequences']}, K={K}, r={args.crop_radius}")
    print(f"  label distribution: {label_dist}")
    print(f"  abstain ratio: {summary['abstain_ratio']:.3f}")
    print(f"  oracle_better_2px: {summary['oracle_better_by_2px_frac']:.3f}")
    print(f"  top1_better_2px: {summary['top1_better_by_2px_frac']:.3f}")
    print(f"  base_median={summary['base_median']:.1f}, oracle_median={summary['oracle_median']:.1f}, top1_median={summary['top1_median']:.1f}")
    if base16_samples:
        print(f"  base16 (n={len(base16_samples)}): abstain={summary['base16']['abstain_ratio']:.3f}, oracle_better_2px={summary['base16']['oracle_better_by_2px_frac']:.3f}")
    print(f"\nSaved to {split_jsonl}")


if __name__ == "__main__":
    main()
