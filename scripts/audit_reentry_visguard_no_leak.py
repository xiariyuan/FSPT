#!/usr/bin/env python3
"""No-leak / metric sanity audit for ReEntry-VisGuard.

The strongest diagnosis in the paper is that re-entry coordinates are often
already correct while visibility lags. This script audits that claim and creates
sanity baselines that should fail if the metric is meaningful:

  - coordinate threshold rates over GT-visible re-entry segments
  - first re-entry coordinate and predicted-visibility rates
  - pred-vs-GT allclose / exact-copy checks
  - optional shuffled-visibility baseline
  - optional coordinate-perturbation baseline

GT is used here only for auditing/evaluation, not for the ReEntry-VisGuard
inference method.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_candidate_pool_oracle import eval_one, npy
from utils.coords import find_reentry_events, yx_norm_to_xy_256


DEFAULT_CACHE = "outputs/paper_discovery_2026-06-27/reentry_visguard_sweep/rgb_fresh20_49_natural/visguard_w8_p2.pt"
DEFAULT_OUT = "outputs/paper_discovery_2026-06-27/reentry_visguard_no_leak_sanity/rgb_fresh20_49_natural"


def load_payload(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return None


def reentry_segment_coord_visibility_stats(
    payload: Dict[str, Any],
    thresholds: Tuple[int, ...] = (1, 2, 4, 8, 16),
) -> Dict[str, Any]:
    """Audit coordinate vs visibility over GT-visible re-entry segments."""
    coord_hits = {int(t): [] for t in thresholds}
    first_coord_hits = {int(t): [] for t in thresholds}
    sq_errors: List[float] = []
    first_errors: List[float] = []
    predvis: List[bool] = []
    first_predvis: List[bool] = []
    joint_hits = {int(t): [] for t in thresholds}
    first_joint_hits = {int(t): [] for t in thresholds}
    event_rows: List[Dict[str, Any]] = []

    total_queries = 0
    reentry_queries = 0
    total_events = 0
    gt_visible_reentry_frames = 0

    # Copy checks across all coordinates.
    all_abs_diffs: List[float] = []
    exact_close_frames = 0
    all_coord_frames = 0

    for record in payload["records"]:
        video_id = str(record["video_id"])
        pred_tracks = npy(record["pred_tracks"], np.float32)
        gt_tracks = npy(record["gt_tracks"], np.float32)
        pred_visibility = npy(record["pred_visibility"], bool)
        gt_visibility = npy(record["gt_visibility"], bool)
        query_points = npy(record["query_points"], np.float32)
        n, t_len = pred_visibility.shape
        total_queries += int(n)

        diff = np.abs(pred_tracks - gt_tracks).reshape(-1)
        if diff.size:
            all_abs_diffs.extend(diff.astype(np.float64).tolist())
        close = np.all(np.isclose(pred_tracks, gt_tracks, atol=1e-8, rtol=1e-8), axis=-1)
        exact_close_frames += int(np.sum(close))
        all_coord_frames += int(close.size)

        pred_xy = yx_norm_to_xy_256(pred_tracks)
        gt_xy = yx_norm_to_xy_256(gt_tracks)
        error = np.sqrt(np.sum((pred_xy - gt_xy) ** 2, axis=-1))  # (N,T)

        for qi in range(n):
            qt = max(0, min(t_len - 1, int(round(float(query_points[qi, 0])))))
            events = find_reentry_events(gt_visibility[qi], qt)
            if events:
                reentry_queries += 1
            for evt in events:
                total_events += 1
                rt = int(evt["reentry_frame"])
                visible_mask = gt_visibility[qi, rt:]
                if visible_mask.size == 0:
                    continue
                err_seg = error[qi, rt:][visible_mask]
                pv_seg = pred_visibility[qi, rt:][visible_mask]
                gt_visible_reentry_frames += int(err_seg.size)
                if err_seg.size:
                    sq_errors.extend(err_seg.astype(np.float64).tolist())
                    predvis.extend(pv_seg.astype(bool).tolist())
                    for thr in thresholds:
                        hit = err_seg < float(thr)
                        coord_hits[int(thr)].extend(hit.astype(bool).tolist())
                        joint_hits[int(thr)].extend((hit & pv_seg).astype(bool).tolist())
                first_err = float(error[qi, rt])
                first_errors.append(first_err)
                first_pv = bool(pred_visibility[qi, rt])
                first_predvis.append(first_pv)
                for thr in thresholds:
                    fh = first_err < float(thr)
                    first_coord_hits[int(thr)].append(bool(fh))
                    first_joint_hits[int(thr)].append(bool(fh and first_pv))
                if len(event_rows) < 200:
                    event_rows.append(
                        {
                            "video_id": video_id,
                            "query_idx": int(qi),
                            "query_t": int(qt),
                            "reentry_t": int(rt),
                            "occ_length": int(evt.get("occ_length", -1)),
                            "first_error_256_px": round(first_err, 4),
                            "first_pred_visible": first_pv,
                        }
                    )

    def rate(vals: List[bool]) -> float | None:
        return round(float(np.mean(np.asarray(vals, dtype=bool))), 6) if vals else None

    def dist(vals: List[float]) -> Dict[str, Any]:
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            return {"n": 0}
        return {
            "n": int(arr.size),
            "mean": round(float(np.mean(arr)), 6),
            "median": round(float(np.median(arr)), 6),
            "p90": round(float(np.percentile(arr, 90)), 6),
            "p95": round(float(np.percentile(arr, 95)), 6),
            "max": round(float(np.max(arr)), 6),
        }

    all_diff_arr = np.asarray(all_abs_diffs, dtype=np.float64)
    copy_check = {
        "all_coord_frames": int(all_coord_frames),
        "exact_close_frames_atol1e-8": int(exact_close_frames),
        "exact_close_frame_rate_atol1e-8": round(float(exact_close_frames) / max(int(all_coord_frames), 1), 8),
        "abs_diff_distribution": dist(all_abs_diffs),
        "warning": "If exact_close_frame_rate is near 1.0, inspect for possible GT leakage. Normal trackers should not be exact copies of GT.",
    }
    if all_diff_arr.size:
        copy_check["abs_diff_nonzero_rate_gt1e-8"] = round(float(np.mean(all_diff_arr > 1e-8)), 8)

    return {
        "total_queries": int(total_queries),
        "reentry_queries": int(reentry_queries),
        "total_reentry_events": int(total_events),
        "gt_visible_reentry_frames": int(gt_visible_reentry_frames),
        "coord_threshold_rates_256": {str(t): rate(coord_hits[int(t)]) for t in thresholds},
        "first_coord_threshold_rates_256": {str(t): rate(first_coord_hits[int(t)]) for t in thresholds},
        "joint_coord_and_predvis_rates_256": {str(t): rate(joint_hits[int(t)]) for t in thresholds},
        "first_joint_coord_and_predvis_rates_256": {str(t): rate(first_joint_hits[int(t)]) for t in thresholds},
        "predvis_recall_on_gt_visible_reentry_frames": rate(predvis),
        "first_predvis_rate": rate(first_predvis),
        "reentry_segment_error_256_px": dist(sq_errors),
        "first_reentry_error_256_px": dist(first_errors),
        "pred_vs_gt_copy_check": copy_check,
        "sample_event_rows_first200": event_rows,
    }


def shuffle_visibility_baseline(payload: Dict[str, Any], out_path: Path, seed: int = 20260702) -> Path:
    rng = np.random.default_rng(seed)
    records = []
    for r in payload["records"]:
        nr = dict(r)
        pv = npy(r["pred_visibility"], bool).copy()
        # Shuffle visibility tracks across query dimension within a video.
        if pv.shape[0] > 1:
            perm = rng.permutation(pv.shape[0])
            pv = pv[perm]
        nr["pred_visibility"] = pv.astype(bool)
        nr["sanity_baseline"] = "query-shuffled visibility within each video"
        records.append(nr)
    out = dict(payload)
    out["records"] = records
    out["model_name"] = "sanity_visibility_shuffled"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    return out_path


def coordinate_perturbation_baseline(payload: Dict[str, Any], out_path: Path, sigma_px: float = 8.0, seed: int = 20260702) -> Path:
    rng = np.random.default_rng(seed)
    records = []
    # Coordinates are yx_norm. For AJ_RD_256 sanity, sigma_px / 255 is the right scale.
    sigma_norm = float(sigma_px) / 255.0
    for r in payload["records"]:
        nr = dict(r)
        pred = npy(r["pred_tracks"], np.float32).copy()
        noise = rng.normal(loc=0.0, scale=sigma_norm, size=pred.shape).astype(np.float32)
        pred = np.clip(pred + noise, 0.0, 1.0)
        nr["pred_tracks"] = pred.astype(np.float32)
        nr["sanity_baseline"] = f"coordinate Gaussian perturbation sigma={sigma_px}px in 256-space"
        records.append(nr)
    out = dict(payload)
    out["records"] = records
    out["model_name"] = f"sanity_coord_perturb_sigma{sigma_px:g}px"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    return out_path


def run_ajrd(cache_path: Path, output_json: Path) -> Dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            "scripts/eval_aj_rd_from_cache.py",
            "--cache-path",
            str(cache_path),
            "--output-json",
            str(output_json),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    with output_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_markdown(summary: Dict[str, Any], path: Path) -> None:
    main = summary["main_audit"]
    base = summary.get("baseline_evals", {})
    lines = [
        "# ReEntry-VisGuard No-Leak / Metric Sanity Audit",
        "",
        f"Cache: `{summary['cache_path']}`",
        "",
        "## Main coordinate-vs-visibility audit",
        "",
        f"- Re-entry queries: `{main['reentry_queries']}`",
        f"- Re-entry events: `{main['total_reentry_events']}`",
        f"- GT-visible re-entry frames: `{main['gt_visible_reentry_frames']}`",
        f"- Pred-visible recall on GT-visible re-entry frames: `{main['predvis_recall_on_gt_visible_reentry_frames']}`",
        f"- First re-entry pred-visible rate: `{main['first_predvis_rate']}`",
        "",
        "### Coordinate threshold rates in 256-space",
        "",
        "| threshold px | segment coord rate | first re-entry coord rate | segment joint coord+predvis | first joint coord+predvis |",
        "|---:|---:|---:|---:|---:|",
    ]
    for thr in ["1", "2", "4", "8", "16"]:
        lines.append(
            f"| {thr} | {main['coord_threshold_rates_256'].get(thr)} | {main['first_coord_threshold_rates_256'].get(thr)} | "
            f"{main['joint_coord_and_predvis_rates_256'].get(thr)} | {main['first_joint_coord_and_predvis_rates_256'].get(thr)} |"
        )
    lines += [
        "",
        "## Pred-vs-GT copy check",
        "",
        f"- Exact-close frame rate @1e-8: `{main['pred_vs_gt_copy_check']['exact_close_frame_rate_atol1e-8']}`",
        f"- Abs diff distribution: `{main['pred_vs_gt_copy_check']['abs_diff_distribution']}`",
        "",
        "Interpretation: exact-close rate should be far from 1.0. If it is near 1.0, inspect for GT leakage.",
        "",
    ]
    if base:
        lines += ["## Optional sanity baselines", "", "| baseline | AJ_RD_256 | AJ_256 | OA_256 |", "|---|---:|---:|---:|"]
        for name, row in base.items():
            lines.append(f"| {name} | {row.get('AJ_RD_256')} | {row.get('AJ_256')} | {row.get('OA_256')} |")
        lines += [
            "",
            "Expected behavior: shuffled visibility and coordinate perturbation should reduce AJ_RD / standard metrics. If they do not, the metric path needs inspection.",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit no-leak and coordinate/visibility sanity for a ReEntry-VisGuard cache.")
    ap.add_argument("--cache-path", default=DEFAULT_CACHE)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--make-baselines", action="store_true", help="Also build/evaluate visibility-shuffle and coordinate-perturbation sanity baselines.")
    ap.add_argument("--coord-perturb-sigma-px", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=20260702)
    args = ap.parse_args()

    cache_path = Path(args.cache_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = load_payload(cache_path)
    main_audit = reentry_segment_coord_visibility_stats(payload)

    baseline_evals: Dict[str, Any] = {}
    if args.make_baselines:
        shuffled = shuffle_visibility_baseline(payload, out_dir / "sanity_visibility_shuffled.pt", seed=args.seed)
        perturbed = coordinate_perturbation_baseline(
            payload,
            out_dir / f"sanity_coord_perturb_sigma{args.coord_perturb_sigma_px:g}px.pt",
            sigma_px=args.coord_perturb_sigma_px,
            seed=args.seed,
        )
        # eval_one writes AJ_RD json sidecars and computes standard metrics.
        baseline_evals["original"] = eval_one("original", cache_path)
        baseline_evals["visibility_shuffled"] = eval_one("visibility_shuffled", shuffled)
        baseline_evals[f"coord_perturb_sigma{args.coord_perturb_sigma_px:g}px"] = eval_one(
            f"coord_perturb_sigma{args.coord_perturb_sigma_px:g}px",
            perturbed,
        )

    summary = {
        "cache_path": str(cache_path),
        "audit_name": "reentry_visguard_no_leak_sanity",
        "main_audit": main_audit,
        "baseline_evals": baseline_evals,
        "notes": [
            "GT is used only for audit/evaluation, not method inference.",
            "coord8_rate saturation should be supported by multi-threshold errors and copy checks.",
            "If shuffled visibility or perturbed coordinates do not hurt, inspect metric implementation.",
        ],
    }
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(summary, summary_md)

    compact = {
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "reentry_queries": main_audit["reentry_queries"],
        "coord_threshold_rates_256": main_audit["coord_threshold_rates_256"],
        "first_coord_threshold_rates_256": main_audit["first_coord_threshold_rates_256"],
        "predvis_recall_on_gt_visible_reentry_frames": main_audit["predvis_recall_on_gt_visible_reentry_frames"],
        "first_predvis_rate": main_audit["first_predvis_rate"],
        "exact_close_frame_rate_atol1e-8": main_audit["pred_vs_gt_copy_check"]["exact_close_frame_rate_atol1e-8"],
        "baseline_evals": {k: {m: v.get(m) for m in ["AJ_RD_256", "AJ_256", "OA_256"]} for k, v in baseline_evals.items()},
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
