#!/usr/bin/env python3
"""Static-sequence oracle audit for persistent TAPNext++ query state.

A fit-scene image is repeated to create a controlled static sequence.  The
student observes a deterministic mean-color blackout interval while a teacher
continues to observe the original image.  At the end of the blackout, four
causal branches are rolled forward on identical visible frames:

  native_student       : no state intervention;
  pre_gap_query         : restore only query-token state saved before blackout;
  teacher_query_oracle  : replace only query-token state from the uncorrupted
                          teacher at the same causal step;
  teacher_full_oracle   : replace the complete recurrent state (upper bound).

Ground truth is used only to choose one valid fit-scene point and score the
static coordinate.  This is an oracle feasibility audit, not a learned result.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from mmp_tracker.tapnextpp_identity_state_bridge import (
    extract_persistent_query_state,
    inject_persistent_query_state,
    query_state_distance,
)


DEFAULT_REPO = Path("/gemini/code/FSPT/external/tapnextpp/repo")
DEFAULT_CKPT = Path("/gemini/code/FSPT/checkpoints/tapnextpp/tapnextpp_ckpt.pt")
DEFAULT_SCENE = Path("/gemini/code/FSPT/datasets/pointodyssey/train/ani13_new_f")


def file_sha256(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def clone_tracking_state(state: Any) -> Any:
    hidden = []
    for cache in state.hidden_state:
        if hasattr(cache, "_replace"):
            hidden.append(
                cache._replace(
                    rg_lru_state=cache.rg_lru_state.detach().clone(),
                    conv1d_state=cache.conv1d_state.detach().clone(),
                )
            )
        else:
            hidden.append(
                type(cache)(
                    rg_lru_state=cache.rg_lru_state.detach().clone(),
                    conv1d_state=cache.conv1d_state.detach().clone(),
                )
            )
    return type(state)(
        step=int(state.step),
        query_points=state.query_points.detach().clone(),
        hidden_state=hidden,
    )


def load_frame(scene: Path, frame_index: int) -> tuple[np.ndarray, Path]:
    frame_path = scene / "rgbs" / f"rgb_{frame_index:05d}.jpg"
    if not frame_path.is_file():
        raise FileNotFoundError(frame_path)
    image = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.uint8)
    return image, frame_path


def choose_static_query(
    scene: Path,
    frame_index: int,
    *,
    margin_px: float,
) -> tuple[int, np.ndarray]:
    with np.load(scene / "anno.npz", allow_pickle=False) as data:
        coords = np.asarray(data["trajs_2d"][frame_index], dtype=np.float32)
        visible = np.asarray(
            data["visibs"][frame_index] & data["valids"][frame_index], dtype=bool
        )
    image, _ = load_frame(scene, frame_index)
    height, width = image.shape[:2]
    finite = np.isfinite(coords).all(axis=-1)
    inside = (
        (coords[:, 0] >= margin_px)
        & (coords[:, 0] < width - margin_px)
        & (coords[:, 1] >= margin_px)
        & (coords[:, 1] < height - margin_px)
    )
    candidates = np.flatnonzero(visible & finite & inside)
    if candidates.size == 0:
        raise RuntimeError("no valid static query point with requested margin")
    # Deterministic interior point, avoiding dependence on later frames.
    center = np.asarray([width / 2.0, height / 2.0], dtype=np.float32)
    order = np.argsort(np.linalg.norm(coords[candidates] - center[None], axis=-1))
    point_index = int(candidates[order[len(order) // 3]])
    return point_index, coords[point_index]


def prepare_frame(image: np.ndarray, *, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(image.copy()).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1)[None]
    tensor = torch.nn.functional.interpolate(
        tensor, size=(256, 256), mode="bilinear", align_corners=False
    )
    tensor = tensor.permute(0, 2, 3, 1) / 127.5 - 1.0
    return tensor[:, None]


def scale_query(coord_xy: np.ndarray, image_shape: tuple[int, int]) -> torch.Tensor:
    height, width = image_shape
    x = float(coord_xy[0]) * 256.0 / float(width)
    y = float(coord_xy[1]) * 256.0 / float(height)
    # TAPNext query convention is [time, y, x].
    return torch.tensor([[[0.0, y, x]]], dtype=torch.float32)


def load_model(repo: Path, checkpoint: Path, *, device: str):
    sys.path.insert(0, str(repo))
    from tapnet.tapnext.tapnext_torch import TAPNext  # pylint: disable=import-outside-toplevel

    model = TAPNext(image_size=(256, 256))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(
        {key.replace("tapnext.", ""): value for key, value in payload["state_dict"].items()}
    )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def model_step(model, frame: torch.Tensor, *, query=None, state=None):
    with torch.inference_mode():
        tracks, _, visibility_logits, next_state = model(
            video=frame,
            query_points=query,
            state=state,
        )
    return tracks, visibility_logits, next_state


def record_prediction(
    tracks: torch.Tensor,
    visibility_logits: torch.Tensor,
    target_yx: torch.Tensor,
) -> dict[str, float]:
    predicted = tracks[0, -1, 0].float()
    target = target_yx.to(predicted.device)
    error = torch.linalg.norm(predicted - target)
    logit = visibility_logits[0, -1, 0, 0].float()
    return {
        "pred_y": float(predicted[0].item()),
        "pred_x": float(predicted[1].item()),
        "error_px_256": float(error.item()),
        "visibility_logit": float(logit.item()),
        "visibility_probability": float(torch.sigmoid(logit).item()),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    error = np.asarray([row["error_px_256"] for row in rows], dtype=np.float64)
    probability = np.asarray(
        [row["visibility_probability"] for row in rows], dtype=np.float64
    )
    return {
        "frames": int(error.size),
        "mean_error_px_256": float(error.mean()),
        "median_error_px_256": float(np.median(error)),
        "max_error_px_256": float(error.max()),
        "hit_1px": float((error <= 1.0).mean()),
        "hit_2px": float((error <= 2.0).mean()),
        "hit_4px": float((error <= 4.0).mean()),
        "hit_8px": float((error <= 8.0).mean()),
        "mean_visibility_probability": float(probability.mean()),
        "visible_rate_at_0p5": float((probability >= 0.5).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--official-source-commit", default="989a1fd62f7b2a3cf7f1c339bbde38e086e3a0fc")
    parser.add_argument("--mirror-repo-commit", default="4f3c01d15a5d3ed14639971d79bffef7748fc96e")
    parser.add_argument("--skip-checkpoint-sha256", action="store_true")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--pre-frames", type=int, default=8)
    parser.add_argument("--gap-frames", type=int, default=32)
    parser.add_argument("--post-frames", type=int, default=16)
    parser.add_argument("--margin-px", type=float, default=64.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.pre_frames, args.gap_frames, args.post_frames) <= 0:
        raise ValueError("pre/gap/post frame counts must all be positive")
    if "test" in args.scene.parts:
        raise ValueError("PointOdyssey test scenes are forbidden for feasibility audits")

    image, frame_path = load_frame(args.scene, args.frame_index)
    point_index, coord_xy = choose_static_query(
        args.scene, args.frame_index, margin_px=args.margin_px
    )
    target_yx = torch.tensor(
        [
            float(coord_xy[1]) * 256.0 / image.shape[0],
            float(coord_xy[0]) * 256.0 / image.shape[1],
        ],
        dtype=torch.float32,
        device=args.device,
    )
    query = scale_query(coord_xy, image.shape[:2]).to(args.device)
    visible_frame = prepare_frame(image, device=args.device)
    mean_rgb = np.broadcast_to(
        image.reshape(-1, 3).mean(axis=0).round().astype(np.uint8), image.shape
    ).copy()
    blackout_frame = prepare_frame(mean_rgb, device=args.device)
    source_file = args.repo / "tapnet/tapnext/tapnext_torch.py"
    source_sha256 = file_sha256(source_file)
    checkpoint_sha256 = None if args.skip_checkpoint_sha256 else file_sha256(args.checkpoint)
    model = load_model(args.repo, args.checkpoint, device=args.device)

    teacher_state = None
    student_state = None
    pre_gap_snapshot = None
    teacher_rows = []
    student_gap_rows = []
    total_before_post = args.pre_frames + args.gap_frames
    for step in range(total_before_post):
        teacher_tracks, teacher_vis, teacher_state = model_step(
            model,
            visible_frame,
            query=query if teacher_state is None else None,
            state=teacher_state,
        )
        student_frame = visible_frame if step < args.pre_frames else blackout_frame
        student_tracks, student_vis, student_state = model_step(
            model,
            student_frame,
            query=query if student_state is None else None,
            state=student_state,
        )
        teacher_rows.append(record_prediction(teacher_tracks, teacher_vis, target_yx))
        student_gap_rows.append(record_prediction(student_tracks, student_vis, target_yx))
        if step == args.pre_frames - 1:
            pre_gap_snapshot = extract_persistent_query_state(student_state)

    if pre_gap_snapshot is None or teacher_state is None or student_state is None:
        raise RuntimeError("failed to construct bridge states")
    teacher_query = extract_persistent_query_state(teacher_state)
    student_query = extract_persistent_query_state(student_state)
    state_distance = {
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
    rows: dict[str, list[dict[str, float]]] = {key: [] for key in branches}
    for _ in range(args.post_frames):
        for name in tuple(branches):
            tracks, vis, branches[name] = model_step(
                model, visible_frame, state=branches[name]
            )
            rows[name].append(record_prediction(tracks, vis, target_yx))

    summary = {name: summarize(value) for name, value in rows.items()}
    native_error = summary["native_student"]["mean_error_px_256"]
    query_error = summary["teacher_query_oracle"]["mean_error_px_256"]
    full_error = summary["teacher_full_oracle"]["mean_error_px_256"]
    pre_error = summary["pre_gap_query"]["mean_error_px_256"]
    full_improvement = native_error - full_error
    query_improvement = native_error - query_error
    gate = {
        "teacher_query_improves_mean_error_by_ge_2px": query_improvement >= 2.0,
        "pre_gap_query_improves_mean_error": pre_error < native_error,
        "query_state_retains_ge_half_full_oracle_improvement": full_improvement > 0.0
        and query_improvement >= 0.5 * full_improvement,
    }
    gate["pass"] = all(gate.values())
    gate["decision"] = (
        "ALLOW_PERSISTENT_QUERY_STATE_MODEL"
        if gate["pass"]
        else "STOP_OR_REDESIGN_QUERY_STATE_BRIDGE"
    )

    output = {
        "schema_version": "tapnextpp_query_state_static_oracle_v0",
        "source": {
            "repo": str(args.repo.resolve()),
            "official_repository": "https://github.com/google-deepmind/tapnet",
            "official_source_commit": args.official_source_commit,
            "mirror_repository": str(args.repo.resolve()),
            "mirror_repo_commit": args.mirror_repo_commit,
            "implementation_source": str(source_file.resolve()),
            "implementation_source_sha256": source_sha256,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_size_bytes": args.checkpoint.stat().st_size,
            "checkpoint_sha256": checkpoint_sha256,
            "scene": str(args.scene.resolve()),
            "split": "fit/train",
            "frame": str(frame_path.resolve()),
            "point_index": point_index,
            "query_coord_xy_original": coord_xy.tolist(),
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
            "corruption": "whole-frame deterministic mean RGB during gap",
            "teacher": "same static source image on every frame",
            "student": "same image except gap corruption",
            "state_boundary": (
                "query-token RG-LRU and Conv1D caches only; image tokens and causal "
                "step preserved for query-only branches"
            ),
        },
        "pre_gap_teacher": summarize(teacher_rows[: args.pre_frames]),
        "gap_student": summarize(student_gap_rows[args.pre_frames :]),
        "student_teacher_query_state_distance_at_bridge": state_distance,
        "post_reappearance": summary,
        "delta": {
            "pre_gap_query_mean_error_improvement": native_error - pre_error,
            "teacher_query_mean_error_improvement": query_improvement,
            "teacher_full_mean_error_improvement": full_improvement,
            "query_fraction_of_full_improvement": (
                query_improvement / full_improvement if full_improvement > 0 else 0.0
            ),
        },
        "gate": gate,
        "frame_rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "delta": output["delta"], "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
