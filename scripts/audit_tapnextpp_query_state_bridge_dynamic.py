#!/usr/bin/env python3
"""Dynamic fit-scene redesign of the TAPNext++ query-state oracle gate.

This audit is run only because the preregistered static Gate A had a full-state
oracle ceiling below its 2 px threshold.  It keeps the threshold unchanged and
replaces the repeated image with a real moving PointOdyssey fit clip.

Selection is frozen before execution: among points visible, valid, finite, and
inside the spatial margin for every protocol frame, choose the point at the
90th percentile of motion accumulated across the synthetic gap.  GT is used
only for fit-scene stress selection and scoring; all state branches are causal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from audit_tapnextpp_query_state_bridge_static import (
    clone_tracking_state,
    extract_persistent_query_state,
    file_sha256,
    inject_persistent_query_state,
    load_model,
    model_step,
    prepare_frame,
    query_state_distance,
    record_prediction,
    scale_query,
    summarize,
)

DEFAULT_REPO = Path("/gemini/code/FSPT/external/tapnextpp/repo")
DEFAULT_CKPT = Path("/gemini/code/FSPT/checkpoints/tapnextpp/tapnextpp_ckpt.pt")
DEFAULT_SCENE = Path("/gemini/code/FSPT/datasets/pointodyssey/train/ani13_new_f")


def load_rgb(scene: Path, frame_index: int) -> tuple[np.ndarray, Path]:
    path = scene / "rgbs" / f"rgb_{frame_index:05d}.jpg"
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8), path


def choose_dynamic_query(
    scene: Path,
    *,
    start_frame: int,
    pre_frames: int,
    gap_frames: int,
    post_frames: int,
    margin_px: float,
    motion_quantile: float,
) -> tuple[int, np.ndarray, dict[str, float | int]]:
    total = pre_frames + gap_frames + post_frames
    end = start_frame + total
    with np.load(scene / "anno.npz", allow_pickle=False) as data:
        coords = np.asarray(data["trajs_2d"][start_frame:end], dtype=np.float32)
        visible = np.asarray(
            data["visibs"][start_frame:end] & data["valids"][start_frame:end],
            dtype=bool,
        )
    image, _ = load_rgb(scene, start_frame)
    height, width = image.shape[:2]
    finite = np.isfinite(coords).all(axis=-1)
    inside = (
        (coords[..., 0] >= margin_px)
        & (coords[..., 0] < width - margin_px)
        & (coords[..., 1] >= margin_px)
        & (coords[..., 1] < height - margin_px)
    )
    candidates = np.flatnonzero((visible & finite & inside).all(axis=0))
    if candidates.size == 0:
        raise RuntimeError("no point remains valid and inside for the full protocol")
    gap_start = pre_frames - 1
    gap_end = pre_frames + gap_frames - 1
    gap_motion = np.linalg.norm(
        coords[gap_end, candidates] - coords[gap_start, candidates], axis=-1
    )
    path_length = np.linalg.norm(
        np.diff(coords[:, candidates], axis=0), axis=-1
    ).sum(axis=0)
    # Stable sort by motion, then point index. Select a fixed quantile, not max.
    order = np.lexsort((candidates, gap_motion))
    rank = int(round(motion_quantile * (len(order) - 1)))
    selected_local = int(order[rank])
    point_index = int(candidates[selected_local])
    metadata = {
        "eligible_points": int(candidates.size),
        "motion_quantile": float(motion_quantile),
        "selected_rank_ascending": rank,
        "selected_gap_motion_px_original": float(gap_motion[selected_local]),
        "selected_path_length_px_original": float(path_length[selected_local]),
        "selected_total_displacement_px_original": float(
            np.linalg.norm(coords[-1, point_index] - coords[0, point_index])
        ),
    }
    return point_index, coords[:, point_index], metadata


def target_yx(coord_xy: np.ndarray, shape: tuple[int, int], device: str) -> torch.Tensor:
    height, width = shape
    return torch.tensor(
        [
            float(coord_xy[1]) * 256.0 / float(height),
            float(coord_xy[0]) * 256.0 / float(width),
        ],
        dtype=torch.float32,
        device=device,
    )


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
    if min(args.pre_frames, args.gap_frames, args.post_frames) <= 0:
        raise ValueError("pre/gap/post must be positive")
    if not 0.0 <= args.motion_quantile <= 1.0:
        raise ValueError("motion quantile must lie in [0,1]")
    if "test" in args.scene.parts:
        raise ValueError("PointOdyssey test scenes are forbidden")

    total = args.pre_frames + args.gap_frames + args.post_frames
    point_index, coords_xy, selection = choose_dynamic_query(
        args.scene,
        start_frame=args.start_frame,
        pre_frames=args.pre_frames,
        gap_frames=args.gap_frames,
        post_frames=args.post_frames,
        margin_px=args.margin_px,
        motion_quantile=args.motion_quantile,
    )
    images: list[np.ndarray] = []
    frame_paths: list[Path] = []
    for rel in range(total):
        image, path = load_rgb(args.scene, args.start_frame + rel)
        images.append(image)
        frame_paths.append(path)
    shape = images[0].shape[:2]
    if any(image.shape[:2] != shape for image in images):
        raise RuntimeError("protocol frames have inconsistent shapes")
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
    pre_gap_snapshot = None
    teacher_before = []
    student_gap = []
    bridge_step = args.pre_frames + args.gap_frames
    for rel in range(bridge_step):
        teacher_tracks, teacher_vis, teacher_state = model_step(
            model,
            frames[rel],
            query=query if teacher_state is None else None,
            state=teacher_state,
        )
        student_input = frames[rel] if rel < args.pre_frames else blackout
        student_tracks, student_vis, student_state = model_step(
            model,
            student_input,
            query=query if student_state is None else None,
            state=student_state,
        )
        target = target_yx(coords_xy[rel], shape, args.device)
        teacher_before.append(record_prediction(teacher_tracks, teacher_vis, target))
        student_gap.append(record_prediction(student_tracks, student_vis, target))
        if rel == args.pre_frames - 1:
            pre_gap_snapshot = extract_persistent_query_state(student_state)

    if pre_gap_snapshot is None or teacher_state is None or student_state is None:
        raise RuntimeError("failed to construct bridge states")
    teacher_query = extract_persistent_query_state(teacher_state)
    student_query = extract_persistent_query_state(student_state)
    distance = {
        key: float(value[0, 0].item())
        for key, value in query_state_distance(student_query, teacher_query).items()
    }
    branches = {
        "native_student": clone_tracking_state(student_state),
        "pre_gap_query": inject_persistent_query_state(
            clone_tracking_state(student_state), pre_gap_snapshot
        ),
        "teacher_query_oracle": inject_persistent_query_state(
            clone_tracking_state(student_state), teacher_query
        ),
        "teacher_full_oracle": clone_tracking_state(teacher_state),
    }
    rows = {key: [] for key in branches}
    for rel in range(bridge_step, total):
        target = target_yx(coords_xy[rel], shape, args.device)
        for name in tuple(branches):
            tracks, vis, branches[name] = model_step(
                model, frames[rel], state=branches[name]
            )
            rows[name].append(record_prediction(tracks, vis, target))

    summary = {name: summarize(values) for name, values in rows.items()}
    native = summary["native_student"]["mean_error_px_256"]
    pre = summary["pre_gap_query"]["mean_error_px_256"]
    query_error = summary["teacher_query_oracle"]["mean_error_px_256"]
    full = summary["teacher_full_oracle"]["mean_error_px_256"]
    query_gain = native - query_error
    full_gain = native - full
    native_rows = rows["native_student"]
    query_rows = rows["teacher_query_oracle"]
    improved_fraction = float(np.mean([
        q["error_px_256"] < n["error_px_256"]
        for n, q in zip(native_rows, query_rows)
    ]))
    gate = {
        "teacher_query_improves_mean_error_by_ge_2px": query_gain >= 2.0,
        "pre_gap_query_improves_mean_error": pre < native,
        "query_state_retains_ge_half_full_oracle_improvement": full_gain > 0.0
        and query_gain >= 0.5 * full_gain,
        "teacher_query_improves_ge_75pct_post_frames": improved_fraction >= 0.75,
    }
    gate["pass"] = all(gate.values())
    gate["decision"] = (
        "ALLOW_DYNAMIC_QUERY_STATE_GATE_EXPANSION"
        if gate["pass"]
        else "STOP_OR_REDESIGN_DYNAMIC_QUERY_STATE_BRIDGE"
    )
    output = {
        "schema_version": "tapnextpp_query_state_dynamic_oracle_v0",
        "audit_status": "post_static_gate_redesign; fit-only oracle diagnostic",
        "source": {
            "official_repository": "https://github.com/google-deepmind/tapnet",
            "official_source_commit": args.official_source_commit,
            "mirror_repository": str(args.repo.resolve()),
            "mirror_repo_commit": args.mirror_repo_commit,
            "implementation_source": str(source_file.resolve()),
            "implementation_source_sha256": source_sha,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_size_bytes": args.checkpoint.stat().st_size,
            "checkpoint_sha256": checkpoint_sha,
            "scene": str(args.scene.resolve()),
            "split": "fit/train",
            "start_frame": args.start_frame,
            "end_frame_exclusive": args.start_frame + total,
            "first_frame": str(frame_paths[0].resolve()),
            "last_frame": str(frame_paths[-1].resolve()),
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
            "student_gap_input": "fixed mean RGB from first frame",
            "teacher_input": "original moving fit clip",
            "student_post_input": "same original frames as teacher",
            "point_selection": "90th percentile gap motion among full-interval valid interior points",
            "state_boundary": "query-token RG-LRU and Conv1D only for query branches",
        },
        "pre_teacher": summarize(teacher_before[: args.pre_frames]),
        "gap_student": summarize(student_gap[args.pre_frames :]),
        "state_distance_at_bridge": distance,
        "post_reappearance": summary,
        "delta": {
            "pre_gap_query_mean_error_improvement": native - pre,
            "teacher_query_mean_error_improvement": query_gain,
            "teacher_full_mean_error_improvement": full_gain,
            "query_fraction_of_full_improvement": query_gain / full_gain if full_gain > 0 else 0.0,
            "teacher_query_improved_frame_fraction": improved_fraction,
        },
        "gate": gate,
        "frame_rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "selection": selection, "delta": output["delta"], "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
