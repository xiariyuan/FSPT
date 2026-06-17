#!/usr/bin/env python3
"""
Geometry-consistency rerank audit: use escort neighbor motion to rerank
existing base-centered local DINO candidates.

Does NOT train any model. Uses fixed geometric rules only.

Variants:
  A. stage2c_reference — reproduce Stage-2c baseline
  B. geo_translation_nearest — pick candidate closest to translation center
  C. geo_affine_nearest — pick candidate closest to affine center
  D. geo_affine_nearest_dino_tiebreak — C + DINO rank tiebreak (<=2px)
  E. geo_affine_top2_dino — exploratory: top-2 by affine dist, pick higher DINO rank

Usage:
  python scripts/audit_local_geometry_rerank.py \
    --config configs/fspt_online_recovery_real_eval256.yaml \
    --checkpoint checkpoints/...best.pth \
    --per-sample-jsonl outputs/local_selector_stage2b_from_anchor_sweep/per_sample_gate_0.50.jsonl \
    --selector-jsonl outputs/local_selector_dataset_v3_from_anchor_smoke/val.jsonl \
    --output-dir outputs/local_geometry_rerank_audit \
    --max-batches 30
"""

from __future__ import annotations
import argparse, json, sys, signal
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_local_motion_prior_recovery import (
    load_config, _TimeoutError, _timeout_handler,
    find_t_last_visible, select_escort_points,
    compute_translation_center, compute_affine_center,
)


def _sf(v, d=0.0):
    try: return float(v)
    except: return float(d)


def evaluate_variant(name, entries, orig_w, orig_h):
    """Evaluate one variant across all entries."""
    base_errs = np.array([e["base_error_px"] for e in entries])
    oracle_cand_errs = np.array([e.get("oracle_candidate_error_px", 999) for e in entries])
    oracle_sel_errs = np.array([e.get("oracle_selective_error_px", e.get("base_error_px", 0)) for e in entries])
    stage2c_final = np.array([e.get("stage2c_final_error_px", e.get("base_error_px", 0)) for e in entries])

    final_errs = np.array([e.get("final_error_px", e.get("stage2c_final_error_px", e.get("base_error_px", 0))) for e in entries])
    pos_mask = np.array([e.get("has_positive_candidate", False) for e in entries], dtype=bool)
    b16_mask = np.array([e.get("group_base16", False) for e in entries], dtype=bool)
    b32_mask = np.array([e.get("group_base32", False) for e in entries], dtype=bool)

    changed = sum(1 for e in entries if abs(e["final_error_px"] - e.get("stage2c_final_error_px", e["final_error_px"])) > 0.01)

    def gstats(mask, label):
        if mask.sum() == 0: return {"group": label, "n": 0}
        be, fe, ose, s2c = base_errs[mask], final_errs[mask], oracle_sel_errs[mask], stage2c_final[mask]
        return {
            "group": label, "n": int(mask.sum()),
            "base_median": round(float(np.median(be)), 2),
            "stage2c_final_median": round(float(np.median(s2c)), 2),
            "geometry_final_median": round(float(np.median(fe)), 2),
            "oracle_selective_median": round(float(np.median(ose)), 2),
            "final_better_2px_frac": round(float((fe < be - 2).mean()), 3),
            "oracle_better_2px_frac": round(float((oracle_cand_errs[mask] < be - 2).mean()), 3) if label != "all" else 0,
            "overall_diff_vs_base": round(float(np.median(fe) - np.median(be)), 2),
            "changed_from_stage2c": int((np.abs(fe - s2c) > 0.01).sum()),
        }

    # Positive-specific metrics
    pos_final = final_errs[pos_mask] if pos_mask.sum() else np.array([])
    pos_oracle = oracle_cand_errs[pos_mask] if pos_mask.sum() else np.array([])

    # Exact oracle accuracy on positive
    oracle_acc = 0.0
    if pos_mask.sum() > 0:
        oracle_acc = sum(1 for e in entries if e.get("has_positive_candidate") and
                         abs(e["final_error_px"] - e.get("oracle_candidate_error_px", 999)) < 0.5) / max(1, pos_mask.sum())

    # Positive improved/worsened vs Stage-2c
    pos_improved = 0
    pos_worsened = 0
    for e in entries:
        if not e.get("has_positive_candidate"): continue
        diff = e["final_error_px"] - e.get("stage2c_final_error_px", e["final_error_px"])
        if diff < -0.5: pos_improved += 1
        elif diff > 0.5: pos_worsened += 1

    return {
        "variant": name,
        "overall": gstats(np.ones(len(entries), dtype=bool), "overall"),
        "positive": gstats(pos_mask, "positive"),
        "base16": gstats(b16_mask, "base16"),
        "base32": gstats(b32_mask, "base32"),
        "exact_oracle_accuracy_positive": round(oracle_acc, 3),
        "positive_final_median": round(float(np.median(pos_final)), 2) if len(pos_final) else None,
        "base16_final_median": round(float(np.median(final_errs[b16_mask])), 2) if b16_mask.sum() else None,
        "total_changed": changed,
        "positive_improved_vs_stage2c": pos_improved,
        "positive_worsened_vs_stage2c": pos_worsened,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--per-sample-jsonl", type=str, required=True)
    parser.add_argument("--selector-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=30)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    per_sample = [json.loads(l) for l in open(args.per_sample_jsonl) if l.strip()]
    selector_ds = [json.loads(l) for l in open(args.selector_jsonl) if l.strip()]
    print(f"Per-sample: {len(per_sample)}, Selector dataset: {len(selector_ds)}")

    # Align by sample_id
    sel_by_id = {s["sample_id"]: s for s in selector_ds}
    aligned = []
    for ps in per_sample:
        sid = ps.get("sample_id", -1)
        sel = sel_by_id.get(sid)
        if sel is None:
            print(f"  WARNING: sample_id={sid} not in selector dataset, skipping")
            continue
        aligned.append({"ps": ps, "sel": sel})
    print(f"Aligned: {len(aligned)}")

    # Load model for tracks
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

    # Pre-load video data and compute tracks for all val videos
    video_data = {}  # video_name -> {tracks, occ, orig_w, orig_h, T}
    print("Loading tracks...")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.max_batches:
            break
        if not isinstance(batch, dict):
            continue

        video_name = batch.get("video_name", ["unknown"])
        vid = str(video_name[0] if isinstance(video_name, list) else video_name)

        # Only process videos that have aligned samples
        if not any(a["ps"]["video_name"] == vid for a in aligned):
            continue

        video = batch.get("video")
        query_points = batch.get("query_points")
        occluded = batch.get("occluded")
        if video is None or query_points is None:
            continue

        video_dev = video.to(device)
        query_dev = query_points.to(device)
        if video_dev.dim() == 4: video_dev = video_dev.unsqueeze(0)
        if query_dev.dim() == 2: query_dev = query_dev.unsqueeze(0)
        occ_np = None
        if occluded is not None:
            if occluded.dim() == 2: occluded = occluded.unsqueeze(0)
            occ_np = occluded.cpu().numpy()

        B, T, C, H, W = video_dev.shape
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

        orig_size = batch.get("original_size")
        if isinstance(orig_size, torch.Tensor):
            orig_h, orig_w = int(orig_size[0, 0].item()), int(orig_size[0, 1].item())
        else:
            orig_h, orig_w = H, W

        tracks_all = outputs[0].cpu().numpy()  # (B, N, T, 2)
        info = outputs[2] if len(outputs) > 2 and isinstance(outputs[2], dict) else {}
        base_tracks = info.get("base_tracks", batch.get("base_tracks"))
        if base_tracks is not None: base_tracks = base_tracks.cpu().numpy()
        else: base_tracks = tracks_all

        video_data[vid] = {
            "tracks": tracks_all[0],  # (N, T, 2)
            "base_tracks": base_tracks[0],
            "occ": occ_np[0] if occ_np is not None else np.zeros((tracks_all.shape[1], T), dtype=bool),
            "orig_w": orig_w, "orig_h": orig_h, "T": T,
        }
        print(f"  Loaded {vid}: T={T}, N={tracks_all.shape[1]}")

    print(f"Loaded tracks for {len(video_data)} videos")

    # Compute geometry and rerank for each aligned sample
    entries = []  # Per-sample output entries
    for pair in aligned:
        ps = pair["ps"]
        sel = pair["sel"]
        vid = ps["video_name"]
        n_idx = ps["point_idx"]
        t0 = ps["t_reentry"]

        base_err = _sf(ps["base_error_px"])
        gate_accept = ps.get("gate_accept", False)
        stage2c_rank = ps.get("chosen_candidate_rank_by_dino", -1)
        stage2c_final = _sf(ps.get("final_error_px", base_err))
        oracle_cand = _sf(ps.get("oracle_candidate_error_px", 999))
        oracle_sel = _sf(ps.get("oracle_selective_error_px", base_err))

        entry = {
            "sample_id": ps.get("sample_id"),
            "video_name": vid,
            "point_idx": n_idx,
            "t_last_visible": ps.get("t_last_visible"),
            "t_reentry": t0,
            "gate_accept": gate_accept,
            "has_positive_candidate": ps.get("has_positive_candidate", False),
            "base_error_px": round(base_err, 2),
            "oracle_candidate_error_px": round(oracle_cand, 2),
            "oracle_selective_error_px": round(oracle_sel, 2),
            "stage2c_choice_rank": stage2c_rank,
            "stage2c_final_error_px": round(stage2c_final, 2),
            "group_base16": ps.get("group_base16", False),
            "group_base32": ps.get("group_base32", False),
            "occ_length": ps.get("occ_length", 0),
        }

        # Default: copy Stage-2c choice
        for variant in ["geo_translation_nearest", "geo_affine_nearest",
                         "geo_affine_nearest_dino_tiebreak", "geo_affine_top2_dino"]:
            entry[f"{variant}_final_error_px"] = stage2c_final
            entry[f"{variant}_chosen_rank"] = stage2c_rank
            entry[f"{variant}_valid"] = False

        entry["geometry_valid"] = False
        entry["n_valid_escorts"] = 0
        entry["translation_center_error_px"] = -1
        entry["affine_center_error_px"] = -1
        entry["affine_residual"] = -1

        candidates = sel.get("candidates", [])
        if not candidates or vid not in video_data:
            entries.append(entry)
            continue

        vd = video_data[vid]
        tracks_b = vd["tracks"]
        occ_b = vd["occ"]
        orig_w, orig_h = vd["orig_w"], vd["orig_h"]
        T = vd["T"]

        if n_idx >= tracks_b.shape[0] or t0 >= T:
            entries.append(entry)
            continue

        # Find t_last_visible
        t_last_vis = find_t_last_visible(occ_b[n_idx], t0)

        # Select escorts
        escort_data = select_escort_points(tracks_b, occ_b, n_idx, t_last_vis, t0, K=16)
        entry["n_valid_escorts"] = escort_data["n_valid"]

        # Compute geometry centers
        trans_center, trans_valid = compute_translation_center(tracks_b, escort_data, n_idx, t_last_vis, t0)
        aff_center, aff_valid, aff_fallback, aff_residual = compute_affine_center(tracks_b, escort_data, n_idx, t_last_vis, t0)

        trans_px = np.array([trans_center[0] * orig_w, trans_center[1] * orig_h])
        aff_px = np.array([aff_center[0] * orig_w, aff_center[1] * orig_h])
        gt_norm = sel.get("gt_xy_norm", [0, 0])
        gt_px = np.array([gt_norm[0] * orig_w, gt_norm[1] * orig_h])

        entry["translation_center_error_px"] = round(float(np.linalg.norm(trans_px - gt_px)), 2)
        entry["affine_center_error_px"] = round(float(np.linalg.norm(aff_px - gt_px)), 2)
        entry["affine_residual"] = round(float(aff_residual), 4) if aff_residual >= 0 else -1
        entry["affine_valid"] = bool(aff_valid and not aff_fallback)
        entry["translation_valid"] = bool(trans_valid)

        # Geometry rerank: only when gate accepted
        if gate_accept and (trans_valid or aff_valid):
            entry["geometry_valid"] = True

            cand_debug = []
            for j, c in enumerate(candidates):
                cn = np.array(c["cand_xy_norm"])
                cp = np.array([cn[0] * orig_w, cn[1] * orig_h])
                d_trans = float(np.linalg.norm(cp - trans_px))
                d_aff = float(np.linalg.norm(cp - aff_px))
                d_base = _sf(c.get("cand_dist_to_base_px", 0))
                cand_debug.append({
                    "rank_by_dino": c.get("rank_by_dino", j),
                    "cand_error_px": round(_sf(c.get("cand_error_px", 0)), 2),
                    "dist_to_translation_center_px": round(d_trans, 2),
                    "dist_to_affine_center_px": round(d_aff, 2),
                })
            entry["candidate_debug"] = cand_debug

            K = len(candidates)

            # Variant B: translation nearest
            if trans_valid:
                dists_trans = [cd["dist_to_translation_center_px"] for cd in cand_debug]
                best_trans = int(np.argmin(dists_trans))
                entry["geo_translation_nearest_chosen_rank"] = best_trans
                entry["geo_translation_nearest_final_error_px"] = round(_sf(candidates[best_trans].get("cand_error_px", base_err)), 2)
                entry["geo_translation_nearest_valid"] = True

            # Variant C: affine nearest
            if aff_valid:
                dists_aff = [cd["dist_to_affine_center_px"] for cd in cand_debug]
                best_aff = int(np.argmin(dists_aff))
                entry["geo_affine_nearest_chosen_rank"] = best_aff
                entry["geo_affine_nearest_final_error_px"] = round(_sf(candidates[best_aff].get("cand_error_px", base_err)), 2)
                entry["geo_affine_nearest_valid"] = True

                # Variant D: affine nearest + DINO tiebreak (within 2px)
                close_to_best = [j for j in range(K) if abs(dists_aff[j] - dists_aff[best_aff]) <= 2.0]
                if len(close_to_best) > 1:
                    # Among close candidates, pick higher DINO rank (lower rank number)
                    best_dino = min(close_to_best, key=lambda j: candidates[j].get("rank_by_dino", K))
                    entry["geo_affine_nearest_dino_tiebreak_chosen_rank"] = best_dino
                    entry["geo_affine_nearest_dino_tiebreak_final_error_px"] = round(_sf(candidates[best_dino].get("cand_error_px", base_err)), 2)
                else:
                    entry["geo_affine_nearest_dino_tiebreak_chosen_rank"] = best_aff
                    entry["geo_affine_nearest_dino_tiebreak_final_error_px"] = entry["geo_affine_nearest_final_error_px"]
                entry["geo_affine_nearest_dino_tiebreak_valid"] = True

                # Variant E (exploratory): top-2 by affine dist, pick higher DINO
                sorted_by_aff = sorted(range(K), key=lambda j: dists_aff[j])
                top2 = sorted_by_aff[:2]
                best_e = min(top2, key=lambda j: candidates[j].get("rank_by_dino", K))
                entry["geo_affine_top2_dino_chosen_rank"] = best_e
                entry["geo_affine_top2_dino_final_error_px"] = round(_sf(candidates[best_e].get("cand_error_px", base_err)), 2)
                entry["geo_affine_top2_dino_valid"] = True

        # If not gate_accept or geometry invalid, all variants keep Stage-2c choice
        entries.append(entry)

    # Build per-variant evaluation
    variant_names = [
        "stage2c_reference",
        "geo_translation_nearest",
        "geo_affine_nearest",
        "geo_affine_nearest_dino_tiebreak",
        "geo_affine_top2_dino",
    ]

    variant_results = {}
    for vn in variant_names:
        if vn == "stage2c_reference":
            variant_entries = []
            for e in entries:
                ve = dict(e)
                ve["final_error_px"] = e["stage2c_final_error_px"]
                variant_entries.append(ve)
        else:
            variant_entries = []
            for e in entries:
                ve = dict(e)
                key = f"{vn}_final_error_px"
                if key in e and e.get(f"{vn}_valid", False):
                    ve["final_error_px"] = e[key]
                else:
                    ve["final_error_px"] = e["stage2c_final_error_px"]
                variant_entries.append(ve)

        result = evaluate_variant(vn, variant_entries, 0, 0)
        variant_results[vn] = result

    # Save results
    output = {
        "n_samples": len(entries),
        "n_videos": len(set(e["video_name"] for e in entries)),
        "positive_n": sum(1 for e in entries if e.get("has_positive_candidate")),
        "base16_n": sum(1 for e in entries if e.get("group_base16")),
        "base32_n": sum(1 for e in entries if e.get("group_base32")),
        "gate_threshold": 0.50,
        "variant_results": variant_results,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)

    # Variant summary
    summary_rows = []
    for vn in variant_names:
        r = variant_results[vn]
        summary_rows.append({
            "variant": vn,
            "overall_diff": r["overall"]["overall_diff_vs_base"],
            "positive_final_median": r.get("positive_final_median"),
            "base16_final_median": r.get("base16_final_median"),
            "base16_better_2px": r["base16"]["final_better_2px_frac"],
            "oracle_acc_positive": r.get("exact_oracle_accuracy_positive"),
            "total_changed": r.get("total_changed"),
            "positive_improved": r.get("positive_improved_vs_stage2c"),
            "positive_worsened": r.get("positive_worsened_vs_stage2c"),
        })
    with open(out_dir / "variant_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    # Per-sample predictions
    with open(out_dir / "per_sample_predictions.jsonl", "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    # Changed cases
    changed_cases = [e for e in entries if any(
        abs(e.get(f"{vn}_final_error_px", e["stage2c_final_error_px"]) - e["stage2c_final_error_px"]) > 0.01
        for vn in ["geo_affine_nearest", "geo_affine_nearest_dino_tiebreak"]
    )]
    with open(out_dir / "debug_changed_cases.json", "w") as f:
        json.dump(changed_cases, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Geometry Rerank Audit (n={len(entries)})")
    print(f"{'='*70}")
    print(f"{'Variant':35s} {'overall':>8s} {'pos_med':>8s} {'b16_med':>8s} {'b16_b2px':>8s} {'oracle_acc':>10s} {'changed':>8s}")
    for vn in variant_names:
        r = variant_results[vn]
        print(f"  {vn:33s} {r['overall']['overall_diff_vs_base']:>+7.2f} {r.get('positive_final_median', 0) or 0:>8.2f} "
              f"{r.get('base16_final_median', 0) or 0:>8.2f} {r['base16']['final_better_2px_frac']:>8.3f} "
              f"{r.get('exact_oracle_accuracy_positive', 0):>10.3f} {r.get('total_changed', 0):>8d}")

    # PASS/FAIL criteria
    ref = variant_results["stage2c_reference"]
    best_vn = None
    best_pos_med = ref.get("positive_final_median", 999)

    for vn in variant_names[1:]:
        r = variant_results[vn]
        pos_med = r.get("positive_final_median") or 999
        if pos_med < best_pos_med:
            best_pos_med = pos_med
            best_vn = vn

    if best_vn:
        best = variant_results[best_vn]
        c1 = best["base16"]["final_better_2px_frac"] >= 0.526
        c2 = (best.get("positive_final_median") or 999) <= 18.0
        c3 = best["overall"]["overall_diff_vs_base"] <= 1.0
        c4 = (best.get("exact_oracle_accuracy_positive") or 0) > 0.25
        passed = sum([c1, c2, c3, c4])

        print(f"\nBest variant: {best_vn}")
        print(f"  Criteria: b16_better2px>={0.526} {'PASS' if c1 else 'FAIL'}, pos_med<={18.0} {'PASS' if c2 else 'FAIL'}, "
              f"overall_diff<={1.0} {'PASS' if c3 else 'FAIL'}, oracle_acc>{0.25} {'PASS' if c4 else 'FAIL'}")
        verdict = "PASS" if passed >= 4 else "FAIL"
        print(f"  Verdict: {verdict} ({passed}/4 criteria)")
    else:
        print(f"\nNo geometry variant improved over Stage-2c")
        verdict = "FAIL"

    output["verdict"] = verdict
    output["best_variant"] = best_vn
    with open(out_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
