#!/usr/bin/env python3
"""Audit project TAP-Vid metrics against a pinned official source snapshot."""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.metrics import compute_tapvid_metrics
from datasets.tapvid_official_eval import compute_tapvid_metrics_official


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_function_from_source(path: Path, function_name: str) -> Callable[..., Any]:
    source = path.read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: Dict[str, Any] = {"np": np, "Mapping": Dict}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function_name]


def max_metric_difference(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    keys = set(left) | set(right)
    if set(left) != set(right):
        raise AssertionError(f"Metric key mismatch: {set(left) ^ set(right)}")
    differences = []
    for key in sorted(keys):
        a = np.asarray(left[key], dtype=np.float64)
        b = np.asarray(right[key], dtype=np.float64)
        if a.shape != b.shape:
            raise AssertionError(f"Shape mismatch for {key}: {a.shape} vs {b.shape}")
        diff = np.abs(a - b)
        finite = np.isfinite(diff)
        if np.any(finite):
            differences.append(float(np.max(diff[finite])))
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            raise AssertionError(f"NaN mask mismatch for {key}")
    return max(differences, default=0.0)


def randomized_direct_parity(official_fn: Callable[..., Any], seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    max_diff = 0.0
    cases = 0
    for mode in ("first", "strided"):
        for trackwise in (False, True):
            for _ in range(25):
                b, n, t = 3, 11, 19
                gt = rng.uniform(-8.0, 264.0, size=(b, n, t, 2)).astype(np.float32)
                pred = gt + rng.normal(0.0, 7.0, size=gt.shape).astype(np.float32)
                gt_occ = rng.random((b, n, t)) < 0.25
                pred_occ = rng.random((b, n, t)) < 0.30
                query = np.zeros((b, n, 3), dtype=np.float32)
                query[..., 0] = rng.integers(0, t - 1, size=(b, n))
                # Keep denominators finite in every batch/track.
                gt_occ[..., -1] = False
                expected = official_fn(
                    query, gt_occ, gt, pred_occ, pred, mode, trackwise
                )
                actual = compute_tapvid_metrics_official(
                    query,
                    gt_occ,
                    gt,
                    pred_occ,
                    pred,
                    mode,
                    get_trackwise_metrics=trackwise,
                )
                max_diff = max(max_diff, max_metric_difference(expected, actual))
                cases += 1
    return {"cases": cases, "max_abs_difference": max_diff, "pass": max_diff == 0.0}


def randomized_wrapper_parity(official_fn: Callable[..., Any], seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed + 1)
    max_diff = 0.0
    cases = 0
    for mode in ("first", "strided"):
        for height, width in ((256, 256), (192, 320)):
            for _ in range(25):
                n, t = 13, 17
                gt_yx = rng.uniform(-0.03, 1.03, size=(n, t, 2)).astype(np.float32)
                pred_yx = gt_yx + rng.normal(0.0, 0.018, size=(n, t, 2)).astype(np.float32)
                gt_vis = rng.random((n, t)) > 0.2
                pred_vis = rng.random((n, t)) > 0.3
                gt_vis[:, -1] = True
                query_frames = rng.integers(0, t - 1, size=n)
                query = np.zeros((n, 3), dtype=np.float32)
                query[:, 0] = query_frames
                query[:, 1:] = gt_yx[np.arange(n), query_frames]

                scale_xy = np.asarray([width, height], dtype=np.float32)
                query_px = query.copy()
                query_px[:, 1] *= height
                query_px[:, 2] *= width
                expected = official_fn(
                    query_px[None],
                    (~gt_vis)[None],
                    (gt_yx[..., [1, 0]] * scale_xy)[None],
                    (~pred_vis)[None],
                    (pred_yx[..., [1, 0]] * scale_xy)[None],
                    mode,
                    False,
                )
                wrapped = compute_tapvid_metrics(
                    torch.from_numpy(pred_yx),
                    torch.from_numpy(gt_yx),
                    torch.from_numpy(pred_vis),
                    torch.from_numpy(gt_vis),
                    torch.from_numpy(query),
                    resolution=(height, width),
                    query_mode=mode,
                )
                wrapped_official_keys = {key: wrapped[key] for key in expected}
                expected_scalars = {
                    key: float(np.asarray(value).reshape(-1)[0])
                    for key, value in expected.items()
                }
                max_diff = max(
                    max_diff,
                    max_metric_difference(wrapped_official_keys, expected_scalars),
                )
                cases += 1
    return {"cases": cases, "max_abs_difference": max_diff, "pass": max_diff <= 1.0e-7}


def real_kinetics_parity(
    official_fn: Callable[..., Any], shard_path: Path
) -> Dict[str, Any]:
    with shard_path.open("rb") as handle:
        sample = pickle.load(handle)[0]
    with Image.open(io.BytesIO(bytes(sample["video"][0]))) as image:
        source_width, source_height = image.size

    stored_xy = np.asarray(sample["points"], dtype=np.float32)
    gt_occ = np.asarray(sample["occluded"], dtype=bool)
    valid = np.sum(~gt_occ, axis=1) > 0
    stored_xy = stored_xy[valid]
    gt_occ = gt_occ[valid]
    first_visible = np.asarray([np.flatnonzero(~row)[0] for row in gt_occ], dtype=np.int64)
    query = np.zeros((stored_xy.shape[0], 3), dtype=np.float32)
    query[:, 0] = first_visible
    query[:, 1] = stored_xy[np.arange(stored_xy.shape[0]), first_visible, 1]
    query[:, 2] = stored_xy[np.arange(stored_xy.shape[0]), first_visible, 0]

    # Deterministic perturbations include values on both sides of official pixel
    # thresholds, making scale mistakes observable rather than relying on chance.
    offsets_px = np.asarray([0.5, 0.999, 1.001, 1.999, 2.001, 3.999, 4.001], dtype=np.float32)
    pred_xy = stored_xy.copy()
    for track_index in range(pred_xy.shape[0]):
        pred_xy[track_index, :, 0] += offsets_px[track_index % len(offsets_px)] / 256.0
    pred_occ = gt_occ.copy()
    pred_occ[:, ::11] = ~pred_occ[:, ::11]

    expected = official_fn(
        query[None],
        gt_occ[None],
        (stored_xy * 256.0)[None],
        pred_occ[None],
        (pred_xy * 256.0)[None],
        "first",
        False,
    )
    wrapped = compute_tapvid_metrics(
        torch.from_numpy(pred_xy[..., [1, 0]]),
        torch.from_numpy(stored_xy[..., [1, 0]]),
        torch.from_numpy(~pred_occ),
        torch.from_numpy(~gt_occ),
        torch.from_numpy(query),
        resolution=(256, 256),
        query_mode="first",
    )
    wrapped_official_keys = {key: wrapped[key] for key in expected}
    expected_scalars = {
        key: float(np.asarray(value).reshape(-1)[0]) for key, value in expected.items()
    }
    max_diff = max_metric_difference(wrapped_official_keys, expected_scalars)
    return {
        "source_frame_size": [source_height, source_width],
        "tracks": int(stored_xy.shape[0]),
        "frames": int(stored_xy.shape[1]),
        "max_abs_difference": max_diff,
        "pass": max_diff <= 1.0e-7,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source", required=True)
    parser.add_argument("--official-commit", required=True)
    parser.add_argument("--kinetics-shard", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    source_path = Path(args.official_source).resolve()
    shard_path = Path(args.kinetics_shard).resolve()
    official_fn = load_function_from_source(source_path, "compute_tapvid_metrics")

    direct = randomized_direct_parity(official_fn, args.seed)
    wrapper = randomized_wrapper_parity(official_fn, args.seed)
    kinetics = real_kinetics_parity(official_fn, shard_path)
    result = {
        "official_repository": "google-deepmind/tapnet",
        "official_commit": args.official_commit,
        "official_source": str(source_path),
        "official_source_sha256": sha256(source_path),
        "vendored_source": str(Path("datasets/tapvid_official_eval.py").resolve()),
        "vendored_source_sha256": sha256(Path("datasets/tapvid_official_eval.py").resolve()),
        "kinetics_shard": str(shard_path),
        "kinetics_shard_sha256": sha256(shard_path),
        "randomized_direct_parity": direct,
        "randomized_wrapper_parity": wrapper,
        "real_kinetics_annotation_parity": kinetics,
        "pass": bool(direct["pass"] and wrapper["pass"] and kinetics["pass"]),
        "coordinate_contract": (
            "Normalized TAP-Vid coordinates are converted to official raster "
            "coordinates by multiplying x by width and y by height."
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
