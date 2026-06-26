#!/usr/bin/env python3
"""Build a uniform ensemble cache from unified teacher prediction caches.

The cache schema follows the strided+original unified cache format used by
scripts/eval_aj_rd_from_cache.py.  Tracks are averaged in the cache coordinate
space; visibility defaults to majority vote.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def _parse_cache_arg(entry: str) -> Tuple[str, Path]:
    if "=" not in entry:
        path = Path(entry)
        return path.stem, path
    name, path = entry.split("=", 1)
    return name, Path(path)


def _load_cache(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError(f"{path} is not a unified cache dict with records")
    return payload


def _arrays_close(a: Any, b: Any, name: str, idx: int) -> None:
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    if arr_a.shape != arr_b.shape or not np.allclose(arr_a, arr_b, atol=1e-6, rtol=1e-6):
        raise ValueError(f"record {idx}: {name} differs across caches")


def _validate_aligned(records_by_teacher: Dict[str, List[Dict[str, Any]]]) -> None:
    names = list(records_by_teacher)
    ref_name = names[0]
    ref_records = records_by_teacher[ref_name]
    for name in names[1:]:
        records = records_by_teacher[name]
        if len(records) != len(ref_records):
            raise ValueError(f"{name} has {len(records)} records, expected {len(ref_records)}")

    for idx, ref in enumerate(ref_records):
        for name in names[1:]:
            cur = records_by_teacher[name][idx]
            for key in ("video_id", "sequence_index", "frame_count"):
                if cur.get(key) != ref.get(key):
                    raise ValueError(f"record {idx}: {key} differs for {name}: {cur.get(key)} vs {ref.get(key)}")
            for key in ("query_points", "gt_tracks", "gt_visibility", "original_size", "model_input_size"):
                _arrays_close(cur[key], ref[key], key, idx)
            if np.asarray(cur["pred_tracks"]).shape != np.asarray(ref["pred_tracks"]).shape:
                raise ValueError(f"record {idx}: pred_tracks shape differs for {name}")
            if np.asarray(cur["pred_visibility"]).shape != np.asarray(ref["pred_visibility"]).shape:
                raise ValueError(f"record {idx}: pred_visibility shape differs for {name}")


def _ensemble_visibility(vis_stack: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "majority":
        return np.sum(vis_stack.astype(np.int32), axis=0) >= int(np.ceil(vis_stack.shape[0] / 2.0))
    if strategy == "union":
        return np.any(vis_stack, axis=0)
    if strategy == "intersection":
        return np.all(vis_stack, axis=0)
    raise ValueError(f"unknown visibility strategy: {strategy}")


def build_ensemble(
    cache_map: Dict[str, Path],
    visibility_strategy: str = "majority",
    model_name: str = "uniform_teacher_ensemble",
) -> Dict[str, Any]:
    payloads = {name: _load_cache(path) for name, path in cache_map.items()}
    records_by_teacher = {name: payload["records"] for name, payload in payloads.items()}
    _validate_aligned(records_by_teacher)

    names = list(cache_map)
    ref_payload = payloads[names[0]]
    ref_records = records_by_teacher[names[0]]
    ensemble_records: List[Dict[str, Any]] = []

    for idx, ref in enumerate(ref_records):
        pred_stack = np.stack(
            [np.asarray(records_by_teacher[name][idx]["pred_tracks"], dtype=np.float32) for name in names],
            axis=0,
        )
        vis_stack = np.stack(
            [np.asarray(records_by_teacher[name][idx]["pred_visibility"], dtype=bool) for name in names],
            axis=0,
        )

        record = dict(ref)
        record["pred_tracks"] = np.mean(pred_stack, axis=0).astype(np.float32)
        record["pred_visibility"] = _ensemble_visibility(vis_stack, visibility_strategy).astype(bool)
        record["model_name"] = model_name
        record["ensemble_members"] = list(names)
        record["ensemble_visibility_strategy"] = visibility_strategy
        record["ensemble_note"] = "uniform average of pred_tracks in unified cache coordinate space"
        ensemble_records.append(record)

    output = dict(ref_payload)
    output["schema_version"] = int(ref_payload.get("schema_version", 1))
    output["model_name"] = model_name
    output["checkpoint_path"] = ""
    output["ensemble_members"] = list(names)
    output["ensemble_member_paths"] = {name: str(path) for name, path in cache_map.items()}
    output["ensemble_weights"] = {name: round(1.0 / len(names), 6) for name in names}
    output["ensemble_visibility_strategy"] = visibility_strategy
    output["records"] = ensemble_records
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a uniform ensemble unified cache from teacher caches")
    parser.add_argument("--teacher-caches", nargs="+", required=True, help="name=path entries")
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--model-name", default="uniform_three_teacher")
    parser.add_argument("--visibility-strategy", choices=("majority", "union", "intersection"), default="majority")
    args = parser.parse_args()

    cache_map = dict(_parse_cache_arg(entry) for entry in args.teacher_caches)
    if len(cache_map) < 2:
        raise ValueError("at least two teacher caches are required")

    output = build_ensemble(cache_map, args.visibility_strategy, args.model_name)
    out_path = Path(args.output_cache)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)

    print(f"Wrote {out_path}")
    print(f"members={list(cache_map)}")
    print(f"visibility_strategy={args.visibility_strategy}")
    print(f"records={len(output['records'])}")


if __name__ == "__main__":
    main()
