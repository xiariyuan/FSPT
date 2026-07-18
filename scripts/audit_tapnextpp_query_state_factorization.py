#!/usr/bin/env python3
"""Causal factorization oracle for TAPNext++ query-state reconstruction.

Uses the frozen dynamic fit-only stress protocol and replaces only selected
query-cache components/layers with the same-time uncorrupted teacher state.
The purpose is to identify a compact reconstruction target before any model is
trained. Oracle replacement is never a deployable result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
DEFAULT_SCENE = Path("/gemini/code/FSPT/datasets/pointodyssey/train/ani13_new_f")


def layer_mask(count: int, start: int, end: int) -> list[bool]:
    return [start <= index < end for index in range(count)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--official-source-commit", default="989a1fd62f7b2a3cf7f1c339bbde38e086e3a0fc")
    ap.add_argument("--mirror-repo-commit", default="4f3c01d15a5d3ed14639971d79bffef7748fc96e")
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--pre-frames", type=int, default=8)
    ap.add_argument("--gap-frames", type=int, default=32)
    ap.add_argument("--post-frames", type=int, default=16)
    ap.add_argument("--margin-px", type=float, default=32.0)
    ap.add_argument("--motion-quantile", type=float, default=0.90)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-checkpoint-sha256", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if "test" in args.scene.parts:
        raise ValueError("PointOdyssey test scenes are forbidden")

    total = args.pre_frames + args.gap_frames + args.post_frames
    bridge_step = args.pre_frames + args.gap_frames
    point_index, coords_xy, selection = choose_dynamic_query(
        args.scene,
        start_frame=args.start_frame,
        pre_frames=args.pre_frames,
        gap_frames=args.gap_frames,
        post_frames=args.post_frames,
        margin_px=args.margin_px,
        motion_quantile=args.motion_quantile,
    )
    images = [load_rgb(args.scene, args.start_frame + rel)[0] for rel in range(total)]
    shape = images[0].shape[:2]
    frames = [prepare_frame(image, device=args.device) for image in images]
    fixed_mean = np.broadcast_to(
        images[0].reshape(-1, 3).mean(axis=0).round().astype(np.uint8),
        images[0].shape,
    ).copy()
    blackout = prepare_frame(fixed_mean, device=args.device)
    query = scale_query(coords_xy[0], shape).to(args.device)

    source_file = args.repo / "tapnet/tapnext/tapnext_torch.py"
    source_sha = file_sha256(source_file)
    checkpoint_sha = None if args.skip_checkpoint_sha256 else file_sha256(args.checkpoint)
    model = load_model(args.repo, args.checkpoint, device=args.device)

    teacher_state = None
    student_state = None
    for rel in range(bridge_step):
        _, _, teacher_state = model_step(
            model,
            frames[rel],
            query=query if teacher_state is None else None,
            state=teacher_state,
        )
        student_input = frames[rel] if rel < args.pre_frames else blackout
        _, _, student_state = model_step(
            model,
            student_input,
            query=query if student_state is None else None,
            state=student_state,
        )
    if teacher_state is None or student_state is None:
        raise RuntimeError("failed to build factorization states")

    current_query = extract_persistent_query_state(student_state)
    donor_query = extract_persistent_query_state(teacher_state)
    layers = len(current_query.layers)
    if layers != 12:
        raise RuntimeError(f"expected 12 TAPNext++ layers, found {layers}")

    definitions = {
        "teacher_query_all": (None, True, True),
        "teacher_rg_all": (None, True, False),
        "teacher_conv_all": (None, False, True),
        "teacher_early4_all": (layer_mask(layers, 0, 4), True, True),
        "teacher_mid4_all": (layer_mask(layers, 4, 8), True, True),
        "teacher_late4_all": (layer_mask(layers, 8, 12), True, True),
        "teacher_early6_all": (layer_mask(layers, 0, 6), True, True),
        "teacher_late6_all": (layer_mask(layers, 6, 12), True, True),
        "teacher_late6_rg": (layer_mask(layers, 6, 12), True, False),
        "teacher_late6_conv": (layer_mask(layers, 6, 12), False, True),
    }
    persistent_variants = {
        name: compose_persistent_query_state(
            current_query,
            donor_query,
            layer_mask=mask,
            use_rg_lru=use_rg,
            use_conv1d=use_conv,
        )
        for name, (mask, use_rg, use_conv) in definitions.items()
    }
    replaced_fraction = {
        name: persistent_state_replaced_fraction(current_query, value)
        for name, value in persistent_variants.items()
    }
    branches = {"native_student": clone_tracking_state(student_state)}
    branches.update({
        name: inject_persistent_query_state(clone_tracking_state(student_state), value)
        for name, value in persistent_variants.items()
    })
    branches["teacher_full_oracle"] = clone_tracking_state(teacher_state)

    rows = {name: [] for name in branches}
    for rel in range(bridge_step, total):
        target = target_yx(coords_xy[rel], shape, args.device)
        for name in tuple(branches):
            tracks, vis, branches[name] = model_step(model, frames[rel], state=branches[name])
            rows[name].append(record_prediction(tracks, vis, target))
    summary = {name: summarize(value) for name, value in rows.items()}
    native_error = summary["native_student"]["mean_error_px_256"]
    full_query_gain = native_error - summary["teacher_query_all"]["mean_error_px_256"]
    if full_query_gain <= 0:
        raise RuntimeError("full teacher query state has no positive gain in factorization audit")

    variants = {}
    for name in definitions:
        gain = native_error - summary[name]["mean_error_px_256"]
        variants[name] = {
            "replaced_fraction": replaced_fraction[name],
            "mean_error_px_256": summary[name]["mean_error_px_256"],
            "gain_vs_native": gain,
            "retained_full_query_gain": gain / full_query_gain,
            "improved_frame_fraction": float(np.mean([
                row["error_px_256"] < native["error_px_256"]
                for row, native in zip(rows[name], rows["native_student"])
            ])),
        }
    compact = {
        name: value for name, value in variants.items()
        if name != "teacher_query_all" and value["replaced_fraction"] <= 0.5 + 1e-12
    }
    best_compact_name = max(compact, key=lambda key: compact[key]["retained_full_query_gain"])
    component_names = ("teacher_rg_all", "teacher_conv_all")
    best_component_name = max(
        component_names, key=lambda key: variants[key]["retained_full_query_gain"]
    )
    gate = {
        "compact_le_50pct_retains_ge_80pct_query_gain": (
            compact[best_compact_name]["retained_full_query_gain"] >= 0.80
        ),
        "single_component_retains_ge_70pct_query_gain": (
            variants[best_component_name]["retained_full_query_gain"] >= 0.70
        ),
        "best_compact_improves_ge_75pct_frames": (
            compact[best_compact_name]["improved_frame_fraction"] >= 0.75
        ),
    }
    gate["pass"] = all(gate.values())
    gate["decision"] = (
        "ALLOW_MULTI_SCENE_QUERY_STATE_FACTORIZATION"
        if gate["pass"]
        else "STOP_OR_REDESIGN_QUERY_STATE_FACTORIZATION"
    )
    output = {
        "schema_version": "tapnextpp_query_state_factorization_oracle_v0",
        "audit_status": "fit-only oracle architecture audit; no learned claim",
        "source": {
            "official_repository": "https://github.com/google-deepmind/tapnet",
            "official_source_commit": args.official_source_commit,
            "mirror_repo_commit": args.mirror_repo_commit,
            "implementation_source_sha256": source_sha,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_size_bytes": args.checkpoint.stat().st_size,
            "checkpoint_sha256": checkpoint_sha,
            "scene": str(args.scene.resolve()),
            "point_index": point_index,
            "selection": selection,
            "locked_data_read": {
                "pointodyssey_internal_holdout": False,
                "pointodyssey_test": False,
                "davis_method_eval": False,
                "kinetics_1144": False,
            },
        },
        "protocol": {
            "pre_frames": args.pre_frames,
            "gap_frames": args.gap_frames,
            "post_frames": args.post_frames,
            "factorization_variants": definitions,
        },
        "post_reappearance": summary,
        "full_query_gain_vs_native": full_query_gain,
        "variants": variants,
        "best_compact_variant": best_compact_name,
        "best_single_component": best_component_name,
        "gate": gate,
        "frame_rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "best_compact_variant": {best_compact_name: compact[best_compact_name]},
        "best_single_component": {best_component_name: variants[best_component_name]},
        "gate": gate,
    }, indent=2))


if __name__ == "__main__":
    main()
