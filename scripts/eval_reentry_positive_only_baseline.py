#!/usr/bin/env python3
"""Evaluate positive-only ReEntry visibility baselines.

This is an important control for the learned ReEntry-VisCalibrator.  The rule
baseline differs from the original VisGuard/full local override:

  original VisGuard rule: pred_visibility[window] = override_visibility[window]
  positive-only rule:    pred_visibility[window] = base_visibility OR override_visibility

So it can only recover visible frames; it never turns a base-visible frame into
invisible.  If this simple rule matches the learned model, the learned model's
contribution should be interpreted more conservatively.
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

from datasets.metrics import compute_tapvid_metrics  # noqa: E402
from scripts.run_reentry_visguard_w8p2_eval import check_alignment, predicted_reentry_mask, npy  # noqa: E402

DEFAULT_BASE = "outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/offline_rgb_stacking_fresh20_49.pt"
DEFAULT_OVERRIDE = "outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/online_rgb_stacking_fresh20_49.pt"
DEFAULT_OUT = "outputs/paper_discovery_2026-06-27/reentry_positive_only/rgb_fresh20_49_natural_w8p2"


def load_cache(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_positive_only_records(
    base: Dict[str, Any],
    override: Dict[str, Any],
    *,
    window: int,
    persist: int,
    pre: int,
    k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "total_trigger_events": 0,
        "tracks_with_trigger": 0,
        "candidate_window_frames": 0,
        "override_visible_candidate_frames": 0,
        "newly_recovered_frames": 0,
        "already_base_visible_frames_in_candidate": 0,
        "per_video": [],
    }

    for br, orr in zip(base["records"], override["records"]):
        base_tracks = npy(br["pred_tracks"], np.float32)
        base_vis = npy(br["pred_visibility"], bool)
        override_vis = npy(orr["pred_visibility"], bool)
        qpts = npy(br["query_points"], np.float32)
        n_tracks, t_len = base_vis.shape
        pred_vis = base_vis.copy()
        vstats = {
            "video_id": str(br["video_id"]),
            "tracks": int(n_tracks),
            "total_trigger_events": 0,
            "tracks_with_trigger": 0,
            "candidate_window_frames": 0,
            "override_visible_candidate_frames": 0,
            "newly_recovered_frames": 0,
            "already_base_visible_frames_in_candidate": 0,
        }
        for qi in range(n_tracks):
            qt = max(0, min(t_len - 1, int(round(float(qpts[qi, 0])))))
            mask, triggers = predicted_reentry_mask(
                base_vis[qi],
                override_vis[qi],
                qt,
                window=window,
                persist=persist,
                pre=pre,
                k=k,
            )
            if not triggers:
                continue
            cand = mask.astype(bool)
            ov = override_vis[qi] & cand
            newly = ov & (~base_vis[qi])
            already = cand & base_vis[qi]
            pred_vis[qi, newly] = True

            stats["total_trigger_events"] += len(triggers)
            stats["tracks_with_trigger"] += 1
            stats["candidate_window_frames"] += int(cand.sum())
            stats["override_visible_candidate_frames"] += int(ov.sum())
            stats["newly_recovered_frames"] += int(newly.sum())
            stats["already_base_visible_frames_in_candidate"] += int(already.sum())

            vstats["total_trigger_events"] += len(triggers)
            vstats["tracks_with_trigger"] += 1
            vstats["candidate_window_frames"] += int(cand.sum())
            vstats["override_visible_candidate_frames"] += int(ov.sum())
            vstats["newly_recovered_frames"] += int(newly.sum())
            vstats["already_base_visible_frames_in_candidate"] += int(already.sum())

        nr = dict(br)
        nr["pred_tracks"] = base_tracks.copy().astype(np.float32)
        nr["pred_visibility"] = pred_vis.astype(bool)
        nr["reentry_positive_only"] = {
            "method": f"ReEntry-PositiveOnly-W{window}P{persist}",
            "coordinates": "base/offline coordinates",
            "visibility": "base visibility OR override-visible inside predicted re-entry windows",
            "window": int(window),
            "persist": int(persist),
            "pre": int(pre),
            "k": int(k),
            "uses_gt_at_inference": False,
        }
        records.append(nr)
        stats["per_video"].append(vstats)

    stats["newly_recovered_rate_over_candidate_frames"] = round(
        float(stats["newly_recovered_frames"]) / max(int(stats["candidate_window_frames"]), 1), 6
    )
    stats["override_visible_rate_over_candidate_frames"] = round(
        float(stats["override_visible_candidate_frames"]) / max(int(stats["candidate_window_frames"]), 1), 6
    )
    return records, stats


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
    return json.loads(output_json.read_text(encoding="utf-8"))


def eval_standard(cache: Dict[str, Any]) -> Dict[str, Any]:
    aj, oa, da = [], [], []
    q_total = 0
    for r in cache["records"]:
        pred = torch.from_numpy(npy(r["pred_tracks"], np.float32))
        gt = torch.from_numpy(npy(r["gt_tracks"], np.float32))
        pv = torch.from_numpy(npy(r["pred_visibility"], bool))
        gv = torch.from_numpy(npy(r["gt_visibility"], bool))
        q = torch.from_numpy(npy(r["query_points"], np.float32))
        q_total += int(q.shape[0])
        m = compute_tapvid_metrics(pred, gt, pv, gv, q, resolution=256, query_mode="strided")
        aj.append(float(m.get("AJ", 0.0)))
        oa.append(float(m.get("OA", 0.0)))
        da.append(float(m.get("average_pts_within_thresh", 0.0)))
    return {
        "AJ_256": round(float(np.mean(aj)) * 100.0, 4) if aj else None,
        "OA_256": round(float(np.mean(oa)) * 100.0, 4) if oa else None,
        "delta_avg_256": round(float(np.mean(da)) * 100.0, 4) if da else None,
        "n_records": len(cache["records"]),
        "n_queries": int(q_total),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate positive-only ReEntry visibility rule.")
    ap.add_argument("--base-cache", default=DEFAULT_BASE)
    ap.add_argument("--override-cache", default=DEFAULT_OVERRIDE)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--name", default="reentry_positive_only_w8p2")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--persist", type=int, default=2)
    ap.add_argument("--pre", type=int, default=1)
    ap.add_argument("--k", type=int, default=1)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = load_cache(args.base_cache)
    override = load_cache(args.override_cache)
    check_alignment(base, override, "override")

    records, stats = build_positive_only_records(
        base,
        override,
        window=args.window,
        persist=args.persist,
        pre=args.pre,
        k=args.k,
    )
    payload = dict(base)
    payload["records"] = records
    payload["model_name"] = args.name
    payload["reentry_positive_only"] = {
        "method": f"ReEntry-PositiveOnly-W{args.window}P{args.persist}",
        "base_cache": str(args.base_cache),
        "override_cache": str(args.override_cache),
        "coordinates": "base/offline coordinates",
        "visibility": "base visibility OR override-visible inside predicted re-entry windows",
        "window": int(args.window),
        "persist": int(args.persist),
        "pre": int(args.pre),
        "k": int(args.k),
        "uses_gt_at_inference": False,
    }

    cache_path = out_dir / f"{args.name}.pt"
    ajrd_json = out_dir / f"{args.name}_ajrd.json"
    standard_json = out_dir / f"{args.name}_standard.json"
    manifest_json = out_dir / "manifest.json"
    torch.save(payload, cache_path)
    ajrd = run_ajrd(cache_path, ajrd_json)
    std = eval_standard(payload)
    standard_json.write_text(json.dumps({**std, **stats}, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "method": f"ReEntry-PositiveOnly-W{args.window}P{args.persist}",
        "cache": str(cache_path),
        "base_cache": str(args.base_cache),
        "override_cache": str(args.override_cache),
        "ajrd_json": str(ajrd_json),
        "standard_json": str(standard_json),
        "uses_gt_at_inference": False,
        "true_AJ_RD_256": ajrd.get("true_AJ_RD_256"),
        "true_AJ_RD": ajrd.get("true_AJ_RD"),
        **std,
        **stats,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_video"}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
