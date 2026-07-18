#!/usr/bin/env python3
"""Fit-only multi-scene oracle verification of TAPNext++ mid-layer query state.

The scene list and all gates come from a frozen manifest written before model
execution.  This script loads the frozen model once and retains every selected
scene, including failures.  Teacher-state replacement is an oracle diagnostic,
never a deployable or learned result.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audit_tapnextpp_query_state_bridge_static import (
    clone_tracking_state,
    extract_persistent_query_state,
    file_sha256,
    inject_persistent_query_state,
    load_model,
    model_step,
    prepare_frame,
    record_prediction,
    scale_query,
    summarize,
)
from audit_tapnextpp_query_state_bridge_dynamic import (
    choose_dynamic_query,
    load_rgb,
    target_yx,
)
from mmp_tracker.tapnextpp_identity_state_bridge import (
    compose_persistent_query_state,
    persistent_state_replaced_fraction,
)

DEFAULT_REPO = Path("/gemini/code/FSPT/external/tapnextpp/repo")
DEFAULT_CKPT = Path("/gemini/code/FSPT/checkpoints/tapnextpp/tapnextpp_ckpt.pt")
DEFAULT_DATA_ROOT = Path("/gemini/code/FSPT/datasets/pointodyssey/train")
DEFAULT_MANIFEST = Path(
    "/gemini/code/FSPT_bridgetrack_stage/docs/generated/"
    "BRIDGETRACK_MID4_MULTISCENE_MANIFEST_2026-07-18.json"
)


def bootstrap_mean_ci(
    values: list[float], *, seed: int, samples: int = 10_000
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be a finite non-empty vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "samples": int(samples),
        "seed": int(seed),
    }


def evaluate_multiscene_gate(
    scene_rows: list[dict[str, Any]], *, expected_scenes: int
) -> dict[str, Any]:
    complete = [row for row in scene_rows if row.get("status") == "complete"]
    all_complete = len(complete) == expected_scenes
    gains = [float(row["delta"]["mid4_gain_vs_native"]) for row in complete]
    retained = [
        float(row["delta"]["mid4_retained_full_query_gain"]) for row in complete
        if row["delta"]["mid4_retained_full_query_gain"] is not None
    ]
    improved_frames = [
        float(row["delta"]["mid4_improved_frame_fraction"]) for row in complete
    ]
    ci = (
        bootstrap_mean_ci(gains, seed=17018, samples=10_000)
        if gains
        else {"mean": None, "lower": None, "upper": None, "samples": 0, "seed": 17018}
    )
    positive_scenes = sum(gain > 0.0 for gain in gains)
    median_gain = float(np.median(gains)) if gains else None
    median_retained = float(np.median(retained)) if len(retained) == expected_scenes else None
    median_frame_fraction = (
        float(np.median(improved_frames)) if improved_frames else None
    )
    worst_gain = float(min(gains)) if gains else None
    checks = {
        "all_six_scenes_complete": all_complete,
        "mid4_improves_at_least_5_of_6_scenes": all_complete
        and positive_scenes >= 5,
        "median_mid4_gain_ge_2px": all_complete
        and median_gain is not None
        and median_gain >= 2.0,
        "scene_bootstrap_ci_lower_positive": all_complete
        and ci["lower"] is not None
        and float(ci["lower"]) > 0.0,
        "median_retained_full_query_gain_ge_0p80": all_complete
        and median_retained is not None
        and median_retained >= 0.80,
        "median_improved_frame_fraction_ge_0p75": all_complete
        and median_frame_fraction is not None
        and median_frame_fraction >= 0.75,
        "no_scene_regression_worse_than_1px": all_complete
        and worst_gain is not None
        and worst_gain >= -1.0,
    }
    passed = all(checks.values())
    return {
        "expected_scenes": expected_scenes,
        "complete_scenes": len(complete),
        "positive_scenes": positive_scenes,
        "median_mid4_gain_px_256": median_gain,
        "mean_gain_scene_bootstrap_ci": ci,
        "median_retained_full_query_gain": median_retained,
        "median_improved_frame_fraction": median_frame_fraction,
        "worst_scene_gain_px_256": worst_gain,
        "checks": checks,
        "pass": passed,
        "decision": (
            "ALLOW_FIT_ONLY_MID4_RECONSTRUCTOR_DESIGN"
            if passed
            else "CLOSE_OR_REDESIGN_MID4_QUERY_STATE_ROUTE"
        ),
        "training_allowed": False,
        "model_validation_allowed": False,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def run_scene(
    model,
    *,
    scene: Path,
    protocol: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    start = int(protocol["start_frame"])
    pre = int(protocol["pre_frames"])
    gap = int(protocol["gap_frames"])
    post = int(protocol["post_frames"])
    margin = float(protocol["margin_px"])
    quantile = float(protocol["motion_quantile"])
    total = pre + gap + post
    bridge_step = pre + gap

    point_index, coords_xy, selection = choose_dynamic_query(
        scene,
        start_frame=start,
        pre_frames=pre,
        gap_frames=gap,
        post_frames=post,
        margin_px=margin,
        motion_quantile=quantile,
    )
    if int(selection["eligible_points"]) < int(protocol["minimum_eligible_points"]):
        raise RuntimeError("scene no longer meets frozen minimum eligible-point count")

    images = [load_rgb(scene, start + rel)[0] for rel in range(total)]
    shape = images[0].shape[:2]
    if any(image.shape[:2] != shape for image in images):
        raise RuntimeError("inconsistent image shapes")
    frames = [prepare_frame(image, device=device) for image in images]
    fixed_mean = np.broadcast_to(
        images[0].reshape(-1, 3).mean(axis=0).round().astype(np.uint8),
        images[0].shape,
    ).copy()
    blackout = prepare_frame(fixed_mean, device=device)
    query = scale_query(coords_xy[0], shape).to(device)

    teacher_state = None
    student_state = None
    for rel in range(bridge_step):
        _, _, teacher_state = model_step(
            model,
            frames[rel],
            query=query if teacher_state is None else None,
            state=teacher_state,
        )
        student_input = frames[rel] if rel < pre else blackout
        _, _, student_state = model_step(
            model,
            student_input,
            query=query if student_state is None else None,
            state=student_state,
        )
    if teacher_state is None or student_state is None:
        raise RuntimeError("failed to construct causal states")

    current_query = extract_persistent_query_state(student_state)
    teacher_query = extract_persistent_query_state(teacher_state)
    if len(current_query.layers) != 12:
        raise RuntimeError(f"expected 12 recurrent layers, got {len(current_query.layers)}")
    mid_mask = [4 <= index < 8 for index in range(12)]
    mid_query = compose_persistent_query_state(
        current_query,
        teacher_query,
        layer_mask=mid_mask,
        use_rg_lru=True,
        use_conv1d=True,
    )
    replaced_fraction = persistent_state_replaced_fraction(current_query, mid_query)

    branches = {
        "native_student": clone_tracking_state(student_state),
        "teacher_mid4_joint_oracle": inject_persistent_query_state(
            clone_tracking_state(student_state), mid_query
        ),
        "teacher_full_query_oracle": inject_persistent_query_state(
            clone_tracking_state(student_state), teacher_query
        ),
        "teacher_full_state_oracle": clone_tracking_state(teacher_state),
    }
    rows = {name: [] for name in branches}
    for rel in range(bridge_step, total):
        target = target_yx(coords_xy[rel], shape, device)
        for name in tuple(branches):
            tracks, vis, branches[name] = model_step(
                model, frames[rel], state=branches[name]
            )
            rows[name].append(record_prediction(tracks, vis, target))

    summary = {name: summarize(values) for name, values in rows.items()}
    native_error = summary["native_student"]["mean_error_px_256"]
    mid_error = summary["teacher_mid4_joint_oracle"]["mean_error_px_256"]
    query_error = summary["teacher_full_query_oracle"]["mean_error_px_256"]
    full_error = summary["teacher_full_state_oracle"]["mean_error_px_256"]
    mid_gain = native_error - mid_error
    full_query_gain = native_error - query_error
    full_state_gain = native_error - full_error
    retained = mid_gain / full_query_gain if full_query_gain > 0.0 else None
    frame_fraction = float(np.mean([
        mid["error_px_256"] < native["error_px_256"]
        for mid, native in zip(
            rows["teacher_mid4_joint_oracle"], rows["native_student"]
        )
    ]))
    return {
        "scene": scene.name,
        "status": "complete",
        "point_index": point_index,
        "selection": selection,
        "mid4_replaced_fraction": replaced_fraction,
        "post_reappearance": summary,
        "delta": {
            "mid4_gain_vs_native": mid_gain,
            "full_query_gain_vs_native": full_query_gain,
            "full_state_gain_vs_native": full_state_gain,
            "mid4_retained_full_query_gain": retained,
            "mid4_improved_frame_fraction": frame_fraction,
        },
        "frame_rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-checkpoint-sha256", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    selected = list(manifest["selected_scenes"])
    if len(selected) != 6 or len(set(selected)) != 6:
        raise ValueError("frozen manifest must contain six unique scenes")
    if "ani13_new_f" in selected:
        raise ValueError("discovery scene must be excluded")
    if any("test" in (args.data_root / scene).parts for scene in selected):
        raise ValueError("test data is forbidden")
    clip = manifest["clip"]
    protocol = {
        "start_frame": int(clip["start_frame"]),
        "pre_frames": int(clip["pre_frames"]),
        "gap_frames": int(clip["gap_frames"]),
        "post_frames": int(clip["post_frames"]),
        "margin_px": float(clip["margin_px"]),
        "motion_quantile": float(clip["motion_quantile"]),
        "minimum_eligible_points": int(manifest["minimum_eligible_points_per_scene"]),
    }

    source_file = args.repo / "tapnet/tapnext/tapnext_torch.py"
    output: dict[str, Any] = {
        "schema_version": "bridgetrack_mid4_multiscene_oracle_v0",
        "audit_status": "fit-only same-time teacher oracle; not learned or deployable",
        "source": {
            "official_repository": "https://github.com/google-deepmind/tapnet",
            "official_source_commit": "989a1fd62f7b2a3cf7f1c339bbde38e086e3a0fc",
            "mirror_repo_commit": "4f3c01d15a5d3ed14639971d79bffef7748fc96e",
            "implementation_source": str(source_file.resolve()),
            "implementation_source_sha256": file_sha256(source_file),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_size_bytes": args.checkpoint.stat().st_size,
            "checkpoint_sha256": (
                None if args.skip_checkpoint_sha256 else file_sha256(args.checkpoint)
            ),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": file_sha256(args.manifest),
            "locked_data_read": {
                "pointodyssey_model_validation": False,
                "pointodyssey_internal_holdout": False,
                "pointodyssey_test": False,
                "davis_method_eval": False,
                "kinetics_1144": False,
            },
        },
        "protocol": protocol,
        "selected_scenes": selected,
        "scene_results": [],
        "gate": None,
    }
    atomic_json(args.output, output)
    model = load_model(args.repo, args.checkpoint, device=args.device)

    for index, scene_name in enumerate(selected):
        scene = args.data_root / scene_name
        print(json.dumps({"scene": scene_name, "index": index, "status": "start"}), flush=True)
        try:
            result = run_scene(model, scene=scene, protocol=protocol, device=args.device)
        except Exception as error:  # retain every frozen scene and fail closed
            result = {
                "scene": scene_name,
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        output["scene_results"].append(result)
        output["gate"] = evaluate_multiscene_gate(
            output["scene_results"], expected_scenes=len(selected)
        )
        atomic_json(args.output, output)
        print(json.dumps({
            "scene": scene_name,
            "status": result["status"],
            "delta": result.get("delta"),
            "partial_gate": output["gate"],
        }), flush=True)
        torch.cuda.empty_cache()

    output["gate"] = evaluate_multiscene_gate(
        output["scene_results"], expected_scenes=len(selected)
    )
    atomic_json(args.output, output)
    print(json.dumps({"output": str(args.output), "gate": output["gate"]}, indent=2))


if __name__ == "__main__":
    main()
