#!/usr/bin/env python3
"""Channel-wise factorial intervention for ReEntry-VisGuard.

For a base cache and an override cache, build/evaluate:

  1. base_coord + base_vis
  2. override_coord + override_vis
  3. override_coord + base_vis
  4. base_coord + override_vis
  5. base_coord + local_override_vis = ReEntry-VisGuard

This is the key improvement-paper experiment proving that the gain comes from
visibility-channel recovery rather than arbitrary full trajectory replacement.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_candidate_pool_oracle import check_alignment, eval_one, npy
from scripts.run_reentry_visguard_w8p2_eval import build_visguard_records


DEFAULT_BASE = "outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/offline_rgb_stacking_fresh20_49.pt"
DEFAULT_OVERRIDE = "outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/online_rgb_stacking_fresh20_49.pt"
DEFAULT_OUT = "outputs/paper_discovery_2026-06-27/reentry_channel_factorial/rgb_fresh20_49_natural"


def load_payload(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_channel_mix(
    name: str,
    coord_cache: Dict[str, Any],
    vis_cache: Dict[str, Any],
    out_path: Path,
    *,
    coordinate_source: str,
    visibility_source: str,
) -> Path:
    check_alignment(coord_cache, vis_cache, name)
    records: List[Dict[str, Any]] = []
    for cr, vr in zip(coord_cache["records"], vis_cache["records"]):
        nr = dict(cr)
        nr["pred_tracks"] = npy(cr["pred_tracks"], np.float32).copy()
        nr["pred_visibility"] = npy(vr["pred_visibility"], bool).copy()
        nr["channel_factorial"] = {
            "name": name,
            "coordinate_source": coordinate_source,
            "visibility_source": visibility_source,
            "local_window": False,
            "uses_gt_at_inference": False,
        }
        records.append(nr)
    payload = dict(coord_cache)
    payload["records"] = records
    payload["model_name"] = name
    payload["channel_factorial"] = {
        "name": name,
        "coordinate_source": coordinate_source,
        "visibility_source": visibility_source,
        "local_window": False,
        "uses_gt_at_inference": False,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return out_path


def build_local_visguard(
    name: str,
    base_cache: Dict[str, Any],
    override_cache: Dict[str, Any],
    out_path: Path,
    *,
    window: int,
    persist: int,
    pre: int,
    k: int,
) -> Path:
    records, trigger_stats = build_visguard_records(
        base_cache,
        override_cache,
        window=window,
        persist=persist,
        pre=pre,
        k=k,
    )
    payload = dict(base_cache)
    payload["records"] = records
    payload["model_name"] = name
    payload["channel_factorial"] = {
        "name": name,
        "coordinate_source": "base",
        "visibility_source": "override_local_window",
        "local_window": True,
        "window": int(window),
        "persist": int(persist),
        "pre": int(pre),
        "k": int(k),
        "uses_gt_at_inference": False,
        "trigger_stats": trigger_stats,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return out_path


def gain(row: Dict[str, Any], ref: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "AJ_RD_256": round(float(row["AJ_RD_256"]) - float(ref["AJ_RD_256"]), 6),
        "AJ_256": round(float(row["AJ_256"]) - float(ref["AJ_256"]), 6),
        "OA_256": round(float(row["OA_256"]) - float(ref["OA_256"]), 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate channel-wise coordinate/visibility factorial combinations.")
    ap.add_argument("--base-cache", default=DEFAULT_BASE)
    ap.add_argument("--override-cache", default=DEFAULT_OVERRIDE)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--setting", default="rgb_fresh20_49_natural")
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

    method_paths: Dict[str, Path] = {
        "base_coord_base_vis": Path(args.base_cache),
        "override_coord_override_vis": Path(args.override_cache),
    }

    method_paths["override_coord_base_vis"] = build_channel_mix(
        f"{args.setting}_override_coord_base_vis",
        override,
        base,
        out_dir / "override_coord_base_vis.pt",
        coordinate_source="override",
        visibility_source="base",
    )
    method_paths["base_coord_override_vis_global"] = build_channel_mix(
        f"{args.setting}_base_coord_override_vis_global",
        base,
        override,
        out_dir / "base_coord_override_vis_global.pt",
        coordinate_source="base",
        visibility_source="override_global",
    )
    method_paths["reentry_visguard_w8p2"] = build_local_visguard(
        f"{args.setting}_reentry_visguard_w{args.window}p{args.persist}",
        base,
        override,
        out_dir / f"reentry_visguard_w{args.window}p{args.persist}.pt",
        window=args.window,
        persist=args.persist,
        pre=args.pre,
        k=args.k,
    )

    rows = [eval_one(name, path) for name, path in method_paths.items()]
    by = {r["name"]: r for r in rows}
    ref = by["base_coord_base_vis"]
    gains = {f"{name}_vs_base": gain(row, ref) for name, row in by.items() if name != "base_coord_base_vis"}

    # A compact mechanism readout for the paper.
    mechanism = {
        "does_global_override_visibility_help_AJRD": gains.get("base_coord_override_vis_global_vs_base", {}).get("AJ_RD_256"),
        "does_global_override_visibility_preserve_AJ": gains.get("base_coord_override_vis_global_vs_base", {}).get("AJ_256"),
        "does_local_override_visibility_help_AJRD": gains.get("reentry_visguard_w8p2_vs_base", {}).get("AJ_RD_256"),
        "does_local_override_visibility_preserve_AJ": gains.get("reentry_visguard_w8p2_vs_base", {}).get("AJ_256"),
        "full_override_AJRD_gain": gains.get("override_coord_override_vis_vs_base", {}).get("AJ_RD_256"),
        "full_override_AJ_change": gains.get("override_coord_override_vis_vs_base", {}).get("AJ_256"),
    }

    summary = {
        "setting": args.setting,
        "base_cache": str(args.base_cache),
        "override_cache": str(args.override_cache),
        "protocol": "Channel-wise factorial: mix coordinate and visibility channels from base and override. GT is used only for evaluation.",
        "window": int(args.window),
        "persist": int(args.persist),
        "pre": int(args.pre),
        "k": int(args.k),
        "method_paths": {k: str(v) for k, v in method_paths.items()},
        "methods": rows,
        "gains": gains,
        "mechanism_readout": mechanism,
    }
    out_json = out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    compact = {
        "summary_json": str(out_json),
        "setting": args.setting,
        "methods": [{k: r[k] for k in ["name", "AJ_RD_256", "AJ_256", "OA_256"]} for r in rows],
        "gains": gains,
        "mechanism_readout": mechanism,
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
