#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load_npz(path: Path) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def parse_range(path: Path):
    m = re.search(r"_(\d{5})_(\d{5})\.npz$", path.name)
    if not m:
        raise ValueError(f"Cannot parse range from {path.name}")
    return int(m.group(1)), int(m.group(2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-dir", required=True)
    ap.add_argument("--glob", default="*.npz")
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()
    paths = sorted(Path(args.chunk_dir).glob(args.glob), key=parse_range)
    if not paths:
        raise FileNotFoundError(f"No chunks in {args.chunk_dir} matching {args.glob}")
    ranges = [parse_range(p) for p in paths]
    expected = ranges[0][0]
    for start, end in ranges:
        if start != expected:
            raise ValueError(f"Non-contiguous chunks: expected start {expected}, got {start}")
        expected = end
    chunks = [load_npz(p) for p in paths]
    concat_keys = [
        "X_patch",
        "y_gt_visible",
        "y_safe16",
        "y_safe8",
        "y_utility",
        "y_safe4",
        "numeric_X",
        "meta_json",
        "sample_indices",
    ]
    out: Dict[str, Any] = {}
    for key in concat_keys:
        out[key] = np.concatenate([c[key] for c in chunks], axis=0)
    first = chunks[0]
    for key in [
        "patch_feature_names",
        "numeric_feature_names",
        "source_frame_keep_npz",
        "source_base_cache",
        "model_dir",
        "crop_size",
        "image_size",
        "sample_mode",
    ]:
        if key in first:
            out[key] = first[key]
    out["chunk_paths"] = np.asarray([str(p) for p in paths], dtype=object)
    out["chunk_ranges_json"] = np.asarray(json.dumps(ranges), dtype=object)
    sample_indices = out["sample_indices"]
    if sample_indices.shape[0] != int(sample_indices[-1]) + 1:
        raise ValueError("Sample indices are not dense from zero")
    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(json.dumps({
        "out_npz": str(out_path),
        "n_chunks": len(paths),
        "n_samples": int(out["X_patch"].shape[0]),
        "feature_dim": int(out["X_patch"].shape[1]),
        "first_index": int(sample_indices[0]),
        "last_index": int(sample_indices[-1]),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
