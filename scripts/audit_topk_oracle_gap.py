#!/usr/bin/env python3
"""
Top-k oracle gap audit: does the relocalization search produce any usable candidates?

Answers three questions:
  1. Does top-1 ever leave base? (Type A/B diagnosis)
  2. Does top-k oracle contain better-than-base candidates?
  3. Are top-k candidates diverse or just repeats of the same peak?

Usage:
  python scripts/audit_topk_oracle_gap.py \
    --config configs/fspt_dino_nogate_ablation.yaml \
    --checkpoint checkpoints/...best.pth \
    --output outputs/topk_oracle_audit_dino.json \
    --max-batches 10
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str):
    from omegaconf import OmegaConf
    import yaml
    def _load_and_resolve(path, _seen=None):
        if _seen is None: _seen = set()
        path = str(Path(path).resolve())
        if path in _seen: return OmegaConf.create({})
        _seen.add(path)
        with open(path) as f: raw = yaml.safe_load(f)
        if raw is None: return OmegaConf.create({})
        defaults = raw.pop("defaults", []) or []
        base = OmegaConf.create({})
        for entry in defaults:
            if isinstance(entry, str) and entry != "_self_":
                bp = Path(path).parent / f"{entry}.yaml"
                if not bp.exists(): bp = Path(path).parent / entry
                if bp.exists(): base = OmegaConf.merge(base, _load_and_resolve(str(bp), _seen))
            elif isinstance(entry, dict):
                for k, v in entry.items():
                    if k != "_self_":
                        bp = Path(path).parent / f"{v}.yaml"
                        if not bp.exists(): bp = Path(path).parent / v
                        if bp.exists(): base = OmegaConf.merge(base, _load_and_resolve(str(bp), _seen))
        current = OmegaConf.create(raw)
        return OmegaConf.merge(base, current)
    return _load_and_resolve(config_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)

    from models.cotracker_refiner import CoTrackerFSPTRefiner
    model = CoTrackerFSPTRefiner(cfg.get("model", {}))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    model._export_candidates_debug = True
    model = model.to(device).eval()

    from scripts.eval_long_occlusion_subset import _build_val_loader_from_config
    dataloader = _build_val_loader_from_config(cfg)

    samples = []

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
            occ_np = occluded.cpu().numpy()  # (B, N, T)

        B, T, C, H, W = video_dev.shape
        N_pts = query_dev.shape[1]
        meta = {"video_name": video_name, "base_tracks": batch.get("base_tracks"), "base_visibility": batch.get("base_visibility")}

        with torch.no_grad():
            try:
                outputs = model(video_dev, query_dev, meta=meta, return_info=True)
            except Exception as e:
                print(f"  Batch {batch_idx}: ERROR - {e}")
                continue

        info = outputs[2] if len(outputs) > 2 and isinstance(outputs[2], dict) else {}
        relocal_mask = info.get("relocal_mask")
        if not isinstance(relocal_mask, torch.Tensor):
            continue

        base_tracks = info.get("base_tracks", batch.get("base_tracks"))
        if base_tracks is not None: base_tracks = base_tracks.cpu()
        gt_tracks = target_points.cpu()

        orig_size = batch.get("original_size")
        if isinstance(orig_size, torch.Tensor):
            orig_h, orig_w = int(orig_size[0, 0].item()), int(orig_size[0, 1].item())
        else:
            orig_h, orig_w = H, W

        # Get top-k debug data. Newer exports are aligned to (B,N,T,K,...).
        debug = getattr(model, '_relocal_debug_data', {})
        topk_positions = debug.get("relocal_topk_positions")
        topk_scores = debug.get("relocal_topk_scores")
        topk_valid = debug.get("relocal_topk_valid")
        topk_kind = debug.get("relocal_topk_kind")

        vid_name = str(video_name[0] if isinstance(video_name, list) else video_name)

        for b in range(B):
            for n_idx in range(N_pts):
                rm = relocal_mask[b, n_idx].cpu().numpy()
                triggered = np.where(rm)[0]
                if len(triggered) == 0:
                    continue

                for t0 in triggered:
                    t0 = int(t0)
                    # Skip if point is occluded at this time (GT is invalid)
                    if occ_np is not None and occ_np[b, n_idx, t0]:
                        continue
                    if base_tracks is not None:
                        base_norm = base_tracks[b, n_idx, t0].numpy()
                    else:
                        base_norm = outputs[0][b, n_idx, t0].cpu().numpy()
                    base_px = np.array([base_norm[0] * orig_w, base_norm[1] * orig_h])
                    gt_norm = gt_tracks[b, n_idx, t0].numpy()
                    gt_px = np.array([gt_norm[0] * orig_w, gt_norm[1] * orig_h])
                    base_err = float(np.linalg.norm(base_px - gt_px))

                    entry = {
                        "video_name": vid_name,
                        "point_idx": int(n_idx),
                        "t0": t0,
                        "base_error_px": round(base_err, 2),
                        "base_bad": bool(base_err > 16),
                        "base_very_bad": bool(base_err > 32),
                    }

                    # Top-k analysis
                    if topk_positions is not None and topk_scores is not None:
                        try:
                            entry["topk_kind"] = topk_kind

                            tk_pos_bt = topk_positions[b, n_idx, t0]
                            tk_score_bt = topk_scores[b, n_idx, t0]
                            if topk_valid is not None:
                                tk_valid_bt = topk_valid[b, n_idx, t0].numpy().astype(bool)
                            else:
                                tk_valid_bt = np.isfinite(tk_score_bt.numpy())

                            tk_pos = tk_pos_bt.numpy()[tk_valid_bt]
                            tk_score_arr = tk_score_bt.numpy()[tk_valid_bt]
                            K = int(tk_pos.shape[0])
                            if K == 0:
                                entry["topk_missing"] = True
                                entry["valid_topk"] = False
                            else:
                                entry["valid_topk"] = True
                                tk_px = tk_pos * np.array([orig_w, orig_h])
                                gt_px_k = np.tile(gt_px[None, :], (K, 1))
                                tk_errors = np.linalg.norm(tk_px - gt_px_k, axis=-1)

                                entry["topk_count"] = K
                                entry["topk_errors_px"] = [round(float(e), 2) for e in tk_errors]
                                entry["topk_scores"] = [round(float(s), 4) for s in tk_score_arr]
                                entry["oracle_topk_error_px"] = round(float(tk_errors.min()), 2)
                                entry["topk_better_than_base"] = int((tk_errors < base_err).sum())
                                entry["topk_better_by_2px"] = int((tk_errors < base_err - 2).sum())

                            # Candidate diversity: pairwise distance between top-k
                            if K > 1:
                                pairwise_dists = []
                                for i in range(K):
                                    for j in range(i+1, K):
                                        d = float(np.linalg.norm(tk_px[i] - tk_px[j]))
                                        pairwise_dists.append(d)
                                entry["topk_pairwise_dist_mean"] = round(float(np.mean(pairwise_dists)), 2)
                                entry["topk_pairwise_dist_min"] = round(float(np.min(pairwise_dists)), 2)
                                # Unique modes: count candidates > 5px apart
                                n_unique = 1
                                for i in range(1, K):
                                    min_dist = min(float(np.linalg.norm(tk_px[i] - tk_px[j])) for j in range(i))
                                    if min_dist > 5.0:
                                        n_unique += 1
                                entry["topk_unique_modes"] = n_unique
                        except Exception:
                            pass

                    samples.append(entry)

        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx+1}: {len(samples)} triggered samples")

    n = len(samples)
    if n == 0:
        print("No triggered samples!")
        return

    # Summary
    base_errs = np.array([s["base_error_px"] for s in samples])
    base_bad = np.array([s["base_bad"] for s in samples])
    base_vbad = np.array([s["base_very_bad"] for s in samples])

    oracle_errs = np.array([s.get("oracle_topk_error_px", np.nan) for s in samples], dtype=np.float32)
    better_counts = np.array([s.get("topk_better_than_base", 0) for s in samples], dtype=np.float32)
    better_2px = np.array([s.get("topk_better_by_2px", 0) for s in samples], dtype=np.float32)
    topk_count = np.array([s.get("topk_count", 0) for s in samples], dtype=np.int32)

    def stats(mask, label):
        idx = np.asarray(mask, dtype=bool)
        n_group = int(idx.sum())
        if n_group == 0:
            return None
        base_g = base_errs[idx]
        oracle_g = oracle_errs[idx]
        valid_g = np.isfinite(oracle_g)
        better_g = better_counts[idx]
        better2_g = better_2px[idx]
        topk_g = topk_count[idx]
        return {
            "group": label,
            "n": n_group,
            "valid_topk_frac": round(float(valid_g.mean()), 3),
            "topk_count_mean": round(float(topk_g.mean()), 2),
            "base_median": round(float(np.median(base_g)), 2),
            "oracle_median": round(float(np.nanmedian(oracle_g)) if valid_g.any() else np.nan, 2),
            "oracle_better_frac": round(float(np.nanmean(oracle_g < base_g)) if valid_g.any() else np.nan, 3),
            "oracle_better_by_2px_frac": round(float(np.nanmean(oracle_g < base_g - 2)) if valid_g.any() else np.nan, 3),
            "topk_better_count_mean": round(float(better_g.mean()), 2),
            "topk_better_2px_count_mean": round(float(better2_g.mean()), 2),
        }

    # Per-sequence aggregation
    video_names_arr = np.array([s["video_name"] for s in samples])
    unique_vids = sorted(set(video_names_arr.tolist()))
    per_sequence = []
    for vid in unique_vids:
        vm = video_names_arr == vid
        vm_samples = [samples[i] for i in range(n) if vm[i]]
        vb = np.array([s["base_error_px"] for s in vm_samples])
        vb_bad = vb > 16
        vb_vbad = vb > 32
        vo = np.array([s.get("oracle_topk_error_px", 999) for s in vm_samples])
        seq_entry = {
            "video_name": vid,
            "n": int(vm.sum()),
            "base_bad_n": int(vb_bad.sum()),
            "base_very_bad_n": int(vb_vbad.sum()),
        }
        if vb_bad.sum() > 0:
            seq_entry["base_bad_oracle_better_by_2px_frac"] = round(float((vo[vb_bad] < vb[vb_bad] - 2).mean()), 3)
            seq_entry["positive_base_bad_count"] = int((vo[vb_bad] < vb[vb_bad] - 2).sum())
        if vb_vbad.sum() > 0:
            seq_entry["base_very_bad_oracle_better_by_2px_frac"] = round(float((vo[vb_vbad] < vb[vb_vbad] - 2).mean()), 3)
        pds = [s.get("topk_pairwise_dist_mean") for s in vm_samples if s.get("topk_pairwise_dist_mean") is not None]
        if pds:
            seq_entry["topk_pairwise_dist_mean"] = round(float(np.mean(pds)), 2)
        per_sequence.append(seq_entry)

    # valid_topk_frac
    valid_g = np.array([s.get("valid_topk", False) for s in samples]).astype(bool)

    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "max_batches": int(args.max_batches),
        "n_triggered": n,
        "topk_kind": topk_kind,
        "valid_topk_frac": round(float(valid_g.mean()), 3) if len(valid_g) > 0 else None,
        "overall": stats(np.ones_like(base_bad, dtype=bool), "all"),
        "base_bad": stats(base_bad, "base>16px"),
        "base_very_bad": stats(base_vbad, "base>32px"),
        "diversity": {
            "topk_pairwise_dist_mean": round(float(np.mean([s["topk_pairwise_dist_mean"] for s in samples if "topk_pairwise_dist_mean" in s])), 2)
            if any("topk_pairwise_dist_mean" in s for s in samples) else None,
            "topk_unique_modes_mean": round(float(np.mean([s["topk_unique_modes"] for s in samples if "topk_unique_modes" in s])), 2)
            if any("topk_unique_modes" in s for s in samples) else None,
        },
        "per_sequence": per_sequence,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n{'='*60}")
    print(f"Top-k Oracle Gap Audit (n={n})")
    print(f"{'='*60}")
    for key in ["overall", "base_bad", "base_very_bad"]:
        s = summary.get(key)
        if s is None or s["n"] == 0:
            continue
        print(f"  {s['group']:20s} (n={s['n']:>4d}): base_med={s['base_median']:.1f}, oracle_med={s['oracle_median']:.1f}, "
              f"oracle_better={s['oracle_better_frac']:.3f}, oracle_better_2px={s['oracle_better_by_2px_frac']:.3f}")
    print(f"  Diversity: pairwise_dist_mean={summary['diversity']['topk_pairwise_dist_mean']:.2f}, "
          f"unique_modes_mean={summary['diversity']['topk_unique_modes_mean']:.2f}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
