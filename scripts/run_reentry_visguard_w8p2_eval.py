#!/usr/bin/env python3
"""Clean ReEntry-VisGuard-W8P2 evaluation runner.

This is the reproducible entrypoint for the final improvement-paper main method:

    coordinates = base/offline coordinates
    visibility  = override visibility inside predicted re-entry windows

The inference trigger is prediction-only. GT visibility/tracks are used only for
post-hoc evaluation and trigger diagnostics.
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

from datasets.metrics import compute_tapvid_metrics
from scripts.eval_aj_rd_from_cache import compute_reentry_metrics
from utils.coords import find_reentry_events


DEFAULT_BASE = "outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/offline_rgb_stacking_fresh20_49.pt"
DEFAULT_OVERRIDE = "outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/online_rgb_stacking_fresh20_49.pt"
DEFAULT_OUT = "outputs/paper_discovery_2026-06-27/reentry_visguard_w8p2_repro/rgb_fresh20_49_natural"


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr


def load_payload(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def check_alignment(base: Dict[str, Any], override: Dict[str, Any], name: str = "override") -> None:
    if len(base.get("records", [])) != len(override.get("records", [])):
        raise ValueError(f"{name}: record count mismatch: {len(base.get('records', []))} vs {len(override.get('records', []))}")
    for i, (br, orr) in enumerate(zip(base["records"], override["records"])):
        if str(br["video_id"]) != str(orr["video_id"]):
            raise ValueError(f"{name}: video_id mismatch at record {i}: {br['video_id']} vs {orr['video_id']}")
        for key in ("query_points", "gt_tracks", "gt_visibility", "original_size"):
            if key == "gt_visibility":
                ok = np.array_equal(npy(br[key], bool), npy(orr[key], bool))
            else:
                ok = np.allclose(npy(br[key]), npy(orr[key]), atol=1e-6, rtol=1e-6)
            if not ok:
                raise ValueError(f"{name}: alignment mismatch at record {i}, key={key}")


def invisible_run_before(visibility: np.ndarray, t: int) -> int:
    count = 0
    j = int(t) - 1
    while j >= 0 and not bool(visibility[j]):
        count += 1
        j -= 1
    return count


def trigger_ok(
    base_visibility: np.ndarray,
    override_visibility: np.ndarray,
    t: int,
    *,
    k: int = 1,
    persist: int = 2,
) -> bool:
    """Prediction-only trigger for a candidate re-entry window."""
    if invisible_run_before(base_visibility, int(t)) < int(k):
        return False
    if int(t) + int(persist) > int(override_visibility.shape[0]):
        return False
    return bool(np.all(override_visibility[int(t) : int(t) + int(persist)]))


def predicted_reentry_mask(
    base_visibility: np.ndarray,
    override_visibility: np.ndarray,
    query_t: int,
    *,
    window: int,
    persist: int,
    pre: int,
    k: int,
) -> Tuple[np.ndarray, List[int]]:
    t_len = int(base_visibility.shape[0])
    mask = np.zeros(t_len, dtype=bool)
    triggers: List[int] = []
    t = max(1, int(query_t) + 1)
    while t < t_len:
        if trigger_ok(base_visibility, override_visibility, t, k=k, persist=persist):
            lo = max(0, t - int(pre))
            hi = min(t_len, t + int(window) + 1)
            mask[lo:hi] = True
            triggers.append(int(t))
            # Skip to the end of the accepted window to avoid double-counting overlapping triggers.
            t = hi
        else:
            t += 1
    return mask, triggers


def build_visguard_records(
    base: Dict[str, Any],
    override: Dict[str, Any],
    *,
    window: int = 8,
    persist: int = 2,
    pre: int = 1,
    k: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build ReEntry-VisGuard records.

    Coordinates are always copied from base. Visibility is copied from override
    only inside predicted re-entry windows.
    """
    records: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "total_trigger_events": 0,
        "tracks_with_trigger": 0,
        "gt_reentry_tracks": 0,
        "triggered_reentry_tracks": 0,
        "triggered_nonreentry_tracks": 0,
        "missed_reentry_tracks": 0,
        "per_video_trigger_stats": [],
    }

    for br, orr in zip(base["records"], override["records"]):
        base_tracks = npy(br["pred_tracks"], np.float32)
        base_vis = npy(br["pred_visibility"], bool)
        override_vis = npy(orr["pred_visibility"], bool)
        gt_vis = npy(br["gt_visibility"], bool)
        qpts = npy(br["query_points"], np.float32)

        pred_tracks = base_tracks.copy()
        pred_vis = base_vis.copy()
        n_tracks, t_len = pred_vis.shape

        video_stats = {
            "video_id": str(br["video_id"]),
            "tracks": int(n_tracks),
            "total_trigger_events": 0,
            "tracks_with_trigger": 0,
            "gt_reentry_tracks": 0,
            "triggered_reentry_tracks": 0,
            "triggered_nonreentry_tracks": 0,
            "missed_reentry_tracks": 0,
        }

        for qi in range(n_tracks):
            qt = max(0, min(t_len - 1, int(round(float(qpts[qi, 0])))))
            has_reentry = bool(find_reentry_events(gt_vis[qi], qt))
            if has_reentry:
                stats["gt_reentry_tracks"] += 1
                video_stats["gt_reentry_tracks"] += 1

            mask, triggers = predicted_reentry_mask(
                base_vis[qi],
                override_vis[qi],
                qt,
                window=window,
                persist=persist,
                pre=pre,
                k=k,
            )
            if triggers:
                stats["total_trigger_events"] += len(triggers)
                stats["tracks_with_trigger"] += 1
                video_stats["total_trigger_events"] += len(triggers)
                video_stats["tracks_with_trigger"] += 1
                if has_reentry:
                    stats["triggered_reentry_tracks"] += 1
                    video_stats["triggered_reentry_tracks"] += 1
                else:
                    stats["triggered_nonreentry_tracks"] += 1
                    video_stats["triggered_nonreentry_tracks"] += 1
                # Visibility-only intervention: coordinates stay base.
                pred_vis[qi, mask] = override_vis[qi, mask]
            elif has_reentry:
                stats["missed_reentry_tracks"] += 1
                video_stats["missed_reentry_tracks"] += 1

        nr = dict(br)
        nr["pred_tracks"] = pred_tracks.astype(np.float32)
        nr["pred_visibility"] = pred_vis.astype(bool)
        nr["reentry_visguard"] = {
            "method": "ReEntry-VisGuard-W8P2" if window == 8 and persist == 2 else "ReEntry-VisGuard",
            "coordinates": "base/offline coordinates",
            "visibility": "override visibility inside predicted re-entry windows",
            "trigger": "base invisible run >= k and override visible for P consecutive frames",
            "window": int(window),
            "persist": int(persist),
            "pre": int(pre),
            "k": int(k),
            "uses_gt_at_inference": False,
        }
        records.append(nr)
        stats["per_video_trigger_stats"].append(video_stats)

    stats["trigger_precision_track"] = round(
        float(stats["triggered_reentry_tracks"]) / max(int(stats["tracks_with_trigger"]), 1), 6
    )
    stats["trigger_recall_track"] = round(
        float(stats["triggered_reentry_tracks"]) / max(int(stats["gt_reentry_tracks"]), 1), 6
    )
    return records, stats


def eval_standard(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    aj, oa, delta = [], [], []
    q_total = 0
    for r in records:
        pred = torch.from_numpy(npy(r["pred_tracks"], np.float32))
        gt = torch.from_numpy(npy(r["gt_tracks"], np.float32))
        pv = torch.from_numpy(npy(r["pred_visibility"], bool))
        gv = torch.from_numpy(npy(r["gt_visibility"], bool))
        q = torch.from_numpy(npy(r["query_points"], np.float32))
        q_total += int(q.shape[0])
        m = compute_tapvid_metrics(pred, gt, pv, gv, q, resolution=256, query_mode="strided")
        aj.append(float(m.get("AJ", 0.0)))
        oa.append(float(m.get("OA", 0.0)))
        delta.append(float(m.get("average_pts_within_thresh", 0.0)))
    return {
        "AJ_256": round(float(np.mean(aj)) * 100.0, 4) if aj else None,
        "OA_256": round(float(np.mean(oa)) * 100.0, 4) if oa else None,
        "delta_avg_256": round(float(np.mean(delta)) * 100.0, 4) if delta else None,
        "n_records": int(len(records)),
        "n_queries": int(q_total),
    }


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


def per_video_metrics(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in records:
        h, w = int(r["original_size"][0]), int(r["original_size"][1])
        pred = npy(r["pred_tracks"], np.float32)
        gt = npy(r["gt_tracks"], np.float32)
        pv = npy(r["pred_visibility"], bool)
        gv = npy(r["gt_visibility"], bool)
        q = npy(r["query_points"], np.float32)
        m = compute_tapvid_metrics(
            torch.from_numpy(pred),
            torch.from_numpy(gt),
            torch.from_numpy(pv),
            torch.from_numpy(gv),
            torch.from_numpy(q),
            resolution=256,
            query_mode="strided",
        )
        rm = compute_reentry_metrics(pred, gt, pv, gv, q, h, w)
        out.append(
            {
                "video_id": str(r["video_id"]),
                "AJ_RD_256": rm.get("true_AJ_RD_256"),
                "AJ_256": round(float(m.get("AJ", 0.0)) * 100.0, 4),
                "OA_256": round(float(m.get("OA", 0.0)) * 100.0, 4),
                "delta_avg_256": round(float(m.get("average_pts_within_thresh", 0.0)) * 100.0, 4),
                "n_reentry_queries": int(rm.get("n_reentry_queries", 0)),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run clean ReEntry-VisGuard-W8P2 evaluation.")
    ap.add_argument("--base-cache", default=DEFAULT_BASE)
    ap.add_argument("--override-cache", default=DEFAULT_OVERRIDE)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--name", default="reentry_visguard_w8p2")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--persist", type=int, default=2)
    ap.add_argument("--pre", type=int, default=1)
    ap.add_argument("--k", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_payload(args.base_cache)
    override = load_payload(args.override_cache)
    check_alignment(base, override, "override")

    records, trigger_stats = build_visguard_records(
        base,
        override,
        window=args.window,
        persist=args.persist,
        pre=args.pre,
        k=args.k,
    )

    payload = dict(base)
    payload["model_name"] = args.name
    payload["reentry_visguard"] = {
        "method": "ReEntry-VisGuard-W8P2" if args.window == 8 and args.persist == 2 else "ReEntry-VisGuard",
        "coordinates": "base/offline coordinates",
        "visibility": "override visibility inside predicted re-entry windows",
        "base_cache": str(args.base_cache),
        "override_cache": str(args.override_cache),
        "window": int(args.window),
        "persist": int(args.persist),
        "pre": int(args.pre),
        "k": int(args.k),
        "uses_gt_at_inference": False,
    }
    payload["records"] = records

    cache_path = out_dir / f"{args.name}.pt"
    ajrd_json = out_dir / f"{args.name}_ajrd.json"
    standard_json = out_dir / f"{args.name}_standard.json"
    per_video_json = out_dir / f"{args.name}_per_video.json"
    manifest_path = out_dir / "manifest.json"

    torch.save(payload, cache_path)
    ajrd = run_ajrd(cache_path, ajrd_json)
    standard = eval_standard(records)
    per_video = per_video_metrics(records)
    standard_json.write_text(json.dumps({**standard, **trigger_stats}, indent=2, ensure_ascii=False), encoding="utf-8")
    per_video_json.write_text(json.dumps(per_video, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "method": payload["reentry_visguard"]["method"],
        "name": args.name,
        "cache": str(cache_path),
        "base_cache": str(args.base_cache),
        "override_cache": str(args.override_cache),
        "ajrd_json": str(ajrd_json),
        "standard_json": str(standard_json),
        "per_video_json": str(per_video_json),
        "coordinates": "base/offline coordinates",
        "visibility": "override visibility inside predicted re-entry windows",
        "trigger": "base invisible run >= k and override visible for P consecutive frames",
        "window": int(args.window),
        "persist": int(args.persist),
        "pre": int(args.pre),
        "k": int(args.k),
        "uses_gt_at_inference": False,
        "true_AJ_RD_256": ajrd.get("true_AJ_RD_256"),
        "true_AJ_RD": ajrd.get("true_AJ_RD"),
        **standard,
        **trigger_stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
