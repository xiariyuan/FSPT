#!/usr/bin/env python3
"""
Candidate generation audit: compare raw vs gated relocalization steps.

Answers:
  - Does raw top-1 leave the base position?
  - Does gate compress it back?
  - Is there oracle gap in top-k?

Usage:
  python scripts/audit_relocal_candidates.py \
    --config configs/fspt_online_recovery_real_eval256.yaml \
    --checkpoint checkpoints/fspt_routeA_stage3_relocal_accept_visiblebank_l30_eval256_from_kinetics_guardrail/best.pth \
    --output outputs/relocal_candidate_audit_cotracker.json \
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

    # Enable debug export
    from models.cotracker_refiner import CoTrackerFSPTRefiner
    model_cfg = cfg.get("model", {})
    # Inject debug flag into refiner config
    if not hasattr(model_cfg, 'refiner'):
        from omegaconf import OmegaConf
        model_cfg = OmegaConf.merge(model_cfg, OmegaConf.create({"refiner": {}}))
    model_cfg.refiner["relocalization_export_candidates_debug"] = True

    model = CoTrackerFSPTRefiner(model_cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    model._export_candidates_debug = True
    model = model.to(device).eval()

    from scripts.eval_long_occlusion_subset import _build_val_loader_from_config
    dataloader = _build_val_loader_from_config(cfg)

    all_results = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.max_batches:
            break
        if not isinstance(batch, dict):
            continue

        video = batch.get("video")
        query_points = batch.get("query_points")
        target_points = batch.get("target_points", batch.get("tracks"))
        occluded = batch.get("occluded", batch.get("visibility"))
        video_name = batch.get("video_name", ["unknown"])

        if video is None or query_points is None:
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

        meta = {
            "video_name": video_name,
            "base_tracks": batch.get("base_tracks"),
            "base_visibility": batch.get("base_visibility"),
        }

        with torch.no_grad():
            try:
                outputs = model(video_dev, query_dev, meta=meta, return_info=True)
            except Exception as e:
                print(f"  Batch {batch_idx}: ERROR - {e}")
                continue

        pred_tracks = outputs[0].cpu()
        info = outputs[2] if len(outputs) > 2 and isinstance(outputs[2], dict) else {}

        relocal_mask = info.get("relocal_mask")
        if not isinstance(relocal_mask, torch.Tensor):
            continue

        # Get debug fields from model instance
        debug_data = getattr(model, '_relocal_debug_data', {})
        step_raw = debug_data.get("relocal_step_raw")
        step_gated = debug_data.get("relocal_step_gated")
        gate_val = debug_data.get("relocal_gate_value")
        conf_out = debug_data.get("relocal_conf_out")
        margin_out = debug_data.get("relocal_margin_out")
        pred_abs = debug_data.get("relocal_pred_abs")
        base_tracks = info.get("base_tracks", batch.get("base_tracks"))

        if base_tracks is not None:
            base_tracks = base_tracks.cpu()
        gt_tracks = target_points.cpu()

        orig_size = batch.get("original_size")
        if isinstance(orig_size, torch.Tensor):
            orig_h, orig_w = int(orig_size[0, 0].item()), int(orig_size[0, 1].item())
        else:
            orig_h, orig_w = H, W

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

                    # Base position
                    if base_tracks is not None:
                        base_norm = base_tracks[b, n_idx, t0].numpy()
                    else:
                        base_norm = pred_tracks[b, n_idx, t0].numpy()
                    base_px = np.array([base_norm[0] * orig_w, base_norm[1] * orig_h])

                    # GT
                    gt_norm = gt_tracks[b, n_idx, t0].numpy()
                    gt_px = np.array([gt_norm[0] * orig_w, gt_norm[1] * orig_h])

                    base_err = float(np.linalg.norm(base_px - gt_px))

                    entry = {
                        "video_name": vid_name,
                        "point_idx": int(n_idx),
                        "t0": t0,
                        "base_xy_norm": base_norm.tolist(),
                        "gt_xy_norm": gt_norm.tolist(),
                        "base_error_px": round(base_err, 2),
                    }

                    # Raw step (before gate)
                    if step_raw is not None:
                        raw_step = step_raw[b, n_idx, t0].cpu().numpy()
                        raw_pred_norm = base_norm + raw_step
                        raw_pred_px = np.array([raw_pred_norm[0] * orig_w, raw_pred_norm[1] * orig_h])
                        entry["raw_step_norm"] = raw_step.tolist()
                        entry["raw_step_px"] = [raw_step[0] * orig_w, raw_step[1] * orig_h]
                        entry["raw_pred_xy_norm"] = raw_pred_norm.tolist()
                        entry["raw_pred_error_px"] = round(float(np.linalg.norm(raw_pred_px - gt_px)), 2)
                        entry["raw_delta_px"] = round(float(np.linalg.norm(raw_pred_px - base_px)), 2)

                    # Gated step (after gate)
                    if step_gated is not None:
                        gated_step = step_gated[b, n_idx, t0].cpu().numpy()
                        gated_pred_norm = base_norm + gated_step
                        entry["gated_step_norm"] = gated_step.tolist()
                        entry["gated_delta_px"] = round(float(np.linalg.norm(gated_step) * orig_w), 2)

                    # Gate value
                    if gate_val is not None:
                        gv = gate_val[b, n_idx, t0].cpu().item()
                        entry["gate_value"] = round(float(gv), 6)

                    # Conf / margin
                    if conf_out is not None:
                        entry["relocal_conf"] = round(float(conf_out[b, n_idx, t0].cpu().item()), 6)
                    if margin_out is not None:
                        entry["relocal_margin"] = round(float(margin_out[b, n_idx, t0].cpu().item()), 6)

                    all_results.append(entry)

        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx+1}: {len(all_results)} triggered samples")

    # Summary statistics
    n_samples = len(all_results)
    if n_samples == 0:
        print("No triggered samples found!")
        return

    raw_deltas = np.array([r.get("raw_delta_px", 0) for r in all_results])
    gated_deltas = np.array([r.get("gated_delta_px", 0) for r in all_results])
    gate_values = np.array([r.get("gate_value", 0) for r in all_results])
    raw_errors = np.array([r.get("raw_pred_error_px", 0) for r in all_results])
    base_errors = np.array([r["base_error_px"] for r in all_results])
    relocal_confs = np.array([r.get("relocal_conf", 0) for r in all_results])

    # Failure type classification
    raw_leaves_base = raw_deltas > 2.0  # raw step moves > 2px from base
    gate_collapses = (raw_deltas > 2.0) & (gated_deltas < 1.0)  # raw moves but gate kills it

    summary = {
        "n_triggered": n_samples,
        "raw_delta_px": {
            "mean": round(float(np.mean(raw_deltas)), 2),
            "median": round(float(np.median(raw_deltas)), 2),
            "p90": round(float(np.percentile(raw_deltas, 90)), 2),
            "p95": round(float(np.percentile(raw_deltas, 95)), 2),
        },
        "gated_delta_px": {
            "mean": round(float(np.mean(gated_deltas)), 2),
            "median": round(float(np.median(gated_deltas)), 2),
            "p90": round(float(np.percentile(gated_deltas, 90)), 2),
        },
        "gate_value": {
            "mean": round(float(np.mean(gate_values)), 4),
            "median": round(float(np.median(gate_values)), 4),
            "min": round(float(np.min(gate_values)), 4),
            "max": round(float(np.max(gate_values)), 4),
        },
        "relocal_conf": {
            "mean": round(float(np.mean(relocal_confs)), 6),
            "median": round(float(np.median(relocal_confs)), 6),
        },
        "failure_analysis": {
            "raw_leaves_base_count": int(raw_leaves_base.sum()),
            "raw_leaves_base_frac": round(float(raw_leaves_base.mean()), 3),
            "gate_collapses_count": int(gate_collapses.sum()),
            "gate_collapses_frac": round(float(gate_collapses.mean()), 3),
            "raw_never_leaves": int((~raw_leaves_base).sum()),
            "raw_never_leaves_frac": round(float((~raw_leaves_base).mean()), 3),
        },
        "failure_type": "Type A: raw top-1 never leaves base" if raw_leaves_base.mean() < 0.10
            else ("Type B: raw moves but gate collapses" if gate_collapses.mean() > 0.30
            else "Type C: mixed"),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n{'='*60}")
    print(f"Relocal Candidate Audit (n={n_samples})")
    print(f"{'='*60}")
    print(f"  Raw step delta:   median={summary['raw_delta_px']['median']:.2f}, p90={summary['raw_delta_px']['p90']:.2f}")
    print(f"  Gated step delta: median={summary['gated_delta_px']['median']:.2f}, p90={summary['gated_delta_px']['p90']:.2f}")
    print(f"  Gate value:       median={summary['gate_value']['median']:.4f}")
    print(f"  Relocal conf:     median={summary['relocal_conf']['median']:.6f}")
    print(f"")
    print(f"  Raw leaves base (>2px): {summary['failure_analysis']['raw_leaves_base_count']}/{n_samples} ({summary['failure_analysis']['raw_leaves_base_frac']:.1%})")
    print(f"  Gate collapses:         {summary['failure_analysis']['gate_collapses_count']}/{n_samples} ({summary['failure_analysis']['gate_collapses_frac']:.1%})")
    print(f"  Raw never leaves:       {summary['failure_analysis']['raw_never_leaves']}/{n_samples} ({summary['failure_analysis']['raw_never_leaves_frac']:.1%})")
    print(f"")
    print(f"  FAILURE TYPE: {summary['failure_type']}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
