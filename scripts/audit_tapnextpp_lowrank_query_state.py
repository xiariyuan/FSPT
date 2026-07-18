#!/usr/bin/env python3
"""Fit-only low-rank oracle for TAPNext++ query-state repair.

A PCA/SVD repair basis is fit only on already exposed PointOdyssey fit clips.
Previously unused qualified fit scenes are used only for causal future-rollout
verification.  Oracle projection coefficients use the hidden teacher state and
therefore test state-delta compressibility, not deployable predictability.
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
    add_persistent_query_state_delta,
    flatten_persistent_query_state,
)

DEFAULT_REPO = Path("/gemini/code/FSPT/external/tapnextpp/repo")
DEFAULT_CKPT = Path("/gemini/code/FSPT/checkpoints/tapnextpp/tapnextpp_ckpt.pt")
DEFAULT_DATA_ROOT = Path("/gemini/code/FSPT/datasets/pointodyssey/train")
DEFAULT_MANIFEST = Path(
    "/gemini/code/FSPT_bridgetrack_stage/docs/generated/"
    "BRIDGETRACK_LOWRANK_QUERY_STATE_MANIFEST_2026-07-18.json"
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def bootstrap_mean_ci(
    values: list[float], *, seed: int = 17018, samples: int = 10_000
) -> dict[str, float | int | None]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.size == 0:
        return {"mean": None, "lower": None, "upper": None, "samples": 0, "seed": seed}
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("bootstrap values must be a finite vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, vector.size, size=(samples, vector.size))
    means = vector[indices].mean(axis=1)
    return {
        "mean": float(vector.mean()),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "samples": int(samples),
        "seed": int(seed),
    }


def expanded_block_scales(
    block_rms: torch.Tensor, block_sizes: tuple[int, ...]
) -> torch.Tensor:
    if block_rms.ndim != 1 or block_rms.numel() != len(block_sizes):
        raise ValueError("block RMS shape differs from block layout")
    return torch.repeat_interleave(
        block_rms,
        torch.tensor(block_sizes, device=block_rms.device),
    )


def fit_block_normalized_pca(
    delta_matrix: torch.Tensor,
    *,
    block_sizes: tuple[int, ...],
    max_rank: int,
    epsilon: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Fit PCA to N,D deltas after scalar RMS normalization per state block."""
    if delta_matrix.ndim != 2 or delta_matrix.shape[0] < 2:
        raise ValueError("delta_matrix must have shape N,D with N>=2")
    if int(sum(block_sizes)) != int(delta_matrix.shape[1]):
        raise ValueError("block sizes do not sum to state dimension")
    rms_rows = []
    start = 0
    for size in block_sizes:
        block = delta_matrix[:, start : start + size]
        rms_rows.append(block.square().mean().sqrt().clamp_min(epsilon))
        start += size
    block_rms = torch.stack(rms_rows)
    scales = expanded_block_scales(block_rms, block_sizes)
    normalized = delta_matrix / scales[None]
    mean = normalized.mean(dim=0)
    centered = normalized - mean[None]
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    rank = min(max_rank, int(vh.shape[0]))
    components = vh[:rank].contiguous()
    total_energy = singular_values.square().sum().clamp_min(1e-12)
    cumulative = singular_values.square().cumsum(0) / total_energy
    return {
        "block_rms": block_rms,
        "mean_normalized": mean,
        "components": components,
        "singular_values": singular_values,
        "cumulative_energy": cumulative,
    }


def project_delta(
    raw_delta: torch.Tensor,
    *,
    block_rms: torch.Tensor,
    block_sizes: tuple[int, ...],
    mean_normalized: torch.Tensor,
    components: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Oracle-project one D-vector through the basis using hidden coefficients."""
    if raw_delta.ndim != 1:
        raise ValueError("raw_delta must be one-dimensional")
    scales = expanded_block_scales(block_rms, block_sizes)
    normalized = raw_delta / scales
    centered = normalized - mean_normalized
    if rank < 0 or rank > components.shape[0]:
        raise ValueError("rank is outside the fitted component range")
    if rank == 0:
        reconstructed = mean_normalized
    else:
        basis = components[:rank]
        coefficients = basis @ centered
        reconstructed = mean_normalized + coefficients @ basis
    return reconstructed * scales


def evaluate_gate(scene_rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    complete = [row for row in scene_rows if row.get("status") == "complete"]
    all_complete = len(complete) == expected
    full_gains = [float(row["delta"]["full_query_gain_vs_native"]) for row in complete]
    rank8_gains = [float(row["delta"]["rank8_gain_vs_native"]) for row in complete]
    retained = [
        float(row["delta"]["rank8_retained_full_query_gain"])
        for row in complete
        if row["delta"]["rank8_retained_full_query_gain"] is not None
    ]
    frame_fraction = [
        float(row["delta"]["rank8_improved_frame_fraction"]) for row in complete
    ]
    ci = bootstrap_mean_ci(rank8_gains)
    median_gain = float(np.median(rank8_gains)) if rank8_gains else None
    median_retained = float(np.median(retained)) if len(retained) == expected else None
    median_frames = float(np.median(frame_fraction)) if frame_fraction else None
    worst = float(min(rank8_gains)) if rank8_gains else None
    checks = {
        "all_10_scenes_complete": all_complete,
        "full_query_positive_at_least_8_of_10": all_complete
        and sum(value > 0.0 for value in full_gains) >= 8,
        "rank8_positive_at_least_8_of_10": all_complete
        and sum(value > 0.0 for value in rank8_gains) >= 8,
        "rank8_median_gain_ge_2px": all_complete
        and median_gain is not None
        and median_gain >= 2.0,
        "rank8_scene_bootstrap_ci_lower_positive": all_complete
        and ci["lower"] is not None
        and float(ci["lower"]) > 0.0,
        "rank8_median_retained_full_query_gain_ge_0p75": all_complete
        and median_retained is not None
        and median_retained >= 0.75,
        "rank8_median_improved_frame_fraction_ge_0p75": all_complete
        and median_frames is not None
        and median_frames >= 0.75,
        "rank8_no_scene_regression_worse_than_1px": all_complete
        and worst is not None
        and worst >= -1.0,
    }
    passed = all(checks.values())
    return {
        "expected_scenes": expected,
        "complete_scenes": len(complete),
        "full_query_positive_scenes": sum(value > 0.0 for value in full_gains),
        "rank8_positive_scenes": sum(value > 0.0 for value in rank8_gains),
        "rank8_median_gain_px_256": median_gain,
        "rank8_mean_gain_scene_bootstrap_ci": ci,
        "rank8_median_retained_full_query_gain": median_retained,
        "rank8_median_improved_frame_fraction": median_frames,
        "rank8_worst_scene_gain_px_256": worst,
        "checks": checks,
        "pass": passed,
        "decision": (
            "ALLOW_FIT_ONLY_LOWRANK_COEFFICIENT_PREDICTOR_DESIGN"
            if passed
            else "CLOSE_BRIDGETRACK_QUERY_STATE_REPAIR"
        ),
        "training_allowed": False,
        "model_validation_allowed": False,
    }


def prepare_clip(
    scene: Path,
    *,
    start_frame: int,
    protocol: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    pre = int(protocol["pre_frames"])
    gap = int(protocol["gap_frames"])
    post = int(protocol["post_frames"])
    total = pre + gap + post
    point_index, coords_xy, selection = choose_dynamic_query(
        scene,
        start_frame=start_frame,
        pre_frames=pre,
        gap_frames=gap,
        post_frames=post,
        margin_px=float(protocol["margin_px"]),
        motion_quantile=float(protocol["motion_quantile"]),
    )
    if int(selection["eligible_points"]) < int(protocol["minimum_eligible_points"]):
        raise RuntimeError("clip violates frozen minimum eligible-point count")
    images = [load_rgb(scene, start_frame + rel)[0] for rel in range(total)]
    shape = images[0].shape[:2]
    if any(image.shape[:2] != shape for image in images):
        raise RuntimeError("inconsistent image shape")
    frames = [prepare_frame(image, device=device) for image in images]
    fixed_mean = np.broadcast_to(
        images[0].reshape(-1, 3).mean(axis=0).round().astype(np.uint8),
        images[0].shape,
    ).copy()
    blackout = prepare_frame(fixed_mean, device=device)
    query = scale_query(coords_xy[0], shape).to(device)
    return {
        "point_index": point_index,
        "coords_xy": coords_xy,
        "selection": selection,
        "shape": shape,
        "frames": frames,
        "blackout": blackout,
        "query": query,
    }


def run_to_bridge(model, clip: dict[str, Any], protocol: dict[str, Any]):
    pre = int(protocol["pre_frames"])
    bridge_step = pre + int(protocol["gap_frames"])
    teacher_state = None
    student_state = None
    for rel in range(bridge_step):
        _, _, teacher_state = model_step(
            model,
            clip["frames"][rel],
            query=clip["query"] if teacher_state is None else None,
            state=teacher_state,
        )
        student_input = clip["frames"][rel] if rel < pre else clip["blackout"]
        _, _, student_state = model_step(
            model,
            student_input,
            query=clip["query"] if student_state is None else None,
            state=student_state,
        )
    if teacher_state is None or student_state is None:
        raise RuntimeError("failed to construct bridge state")
    return teacher_state, student_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-checkpoint-sha256", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--basis-output", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    basis_clips = list(manifest["basis_clips"])
    verification_clips = list(manifest["verification_clips"])
    if len(basis_clips) != 21 or len(verification_clips) != 10:
        raise ValueError("frozen manifest requires 21 basis and 10 verification clips")
    basis_scenes = set(manifest["basis_scenes"])
    verification_scenes = set(manifest["verification_scenes"])
    if basis_scenes & verification_scenes:
        raise ValueError("basis and verification scenes overlap")
    clip_protocol = manifest["clip_protocol"]
    protocol = {
        "pre_frames": int(clip_protocol["pre_frames"]),
        "gap_frames": int(clip_protocol["gap_frames"]),
        "post_frames": int(clip_protocol["post_frames"]),
        "margin_px": float(clip_protocol["margin_px"]),
        "minimum_eligible_points": int(clip_protocol["minimum_eligible_points"]),
        "motion_quantile": float(clip_protocol["motion_quantile"]),
    }
    ranks = tuple(int(value) for value in manifest["state_representation"]["reported_ranks"])
    primary_rank = int(manifest["state_representation"]["primary_rank"])
    if ranks != (0, 1, 2, 4, 8) or primary_rank != 8:
        raise ValueError("rank protocol differs from frozen manifest")

    source_file = args.repo / "tapnet/tapnext/tapnext_torch.py"
    output: dict[str, Any] = {
        "schema_version": "bridgetrack_lowrank_query_state_oracle_v0",
        "audit_status": "fit-only oracle compressibility audit; not learned or deployable",
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
            "locked_data_read": manifest["locked_data_read"],
        },
        "protocol": manifest["state_representation"] | protocol,
        "basis_progress": [],
        "basis_summary": None,
        "verification_results": [],
        "gate": None,
    }
    atomic_json(args.output, output)
    model = load_model(args.repo, args.checkpoint, device=args.device)

    delta_rows: list[torch.Tensor] = []
    layout = None
    for index, item in enumerate(basis_clips):
        scene = args.data_root / item["scene"]
        start_frame = int(item["start_frame"])
        print(json.dumps({"phase": "basis", "index": index, **item, "status": "start"}), flush=True)
        clip = prepare_clip(
            scene, start_frame=start_frame, protocol=protocol, device=args.device
        )
        teacher_state, student_state = run_to_bridge(model, clip, protocol)
        teacher_query = extract_persistent_query_state(teacher_state)
        student_query = extract_persistent_query_state(student_state)
        teacher_vector, teacher_layout = flatten_persistent_query_state(teacher_query)
        student_vector, student_layout = flatten_persistent_query_state(student_query)
        if teacher_layout != student_layout:
            raise RuntimeError("teacher/student state layouts differ")
        if layout is None:
            layout = teacher_layout
        elif layout != teacher_layout:
            raise RuntimeError("basis clip state layouts differ")
        delta = (teacher_vector - student_vector)[0, 0].detach().float().cpu()
        delta_rows.append(delta)
        output["basis_progress"].append({
            "scene": item["scene"],
            "start_frame": start_frame,
            "point_index": clip["point_index"],
            "selection": clip["selection"],
            "delta_l2": float(torch.linalg.norm(delta).item()),
        })
        atomic_json(args.output, output)
        del clip, teacher_state, student_state, teacher_query, student_query
        torch.cuda.empty_cache()

    if layout is None or len(delta_rows) != 21:
        raise RuntimeError("basis collection incomplete")
    matrix = torch.stack(delta_rows, dim=0)
    fit = fit_block_normalized_pca(
        matrix,
        block_sizes=layout.block_sizes,
        max_rank=primary_rank,
    )
    args.basis_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.basis_output,
        mean_normalized=fit["mean_normalized"].numpy(),
        components=fit["components"].numpy(),
        block_rms=fit["block_rms"].numpy(),
        singular_values=fit["singular_values"].numpy(),
        cumulative_energy=fit["cumulative_energy"].numpy(),
        block_names=np.asarray(layout.block_names),
        block_sizes=np.asarray(layout.block_sizes, dtype=np.int64),
        block_shapes=np.asarray([json.dumps(shape) for shape in layout.block_shapes]),
    )
    output["basis_summary"] = {
        "samples": int(matrix.shape[0]),
        "state_dimension": int(matrix.shape[1]),
        "block_names": list(layout.block_names),
        "block_sizes": list(layout.block_sizes),
        "singular_values": fit["singular_values"].tolist(),
        "cumulative_energy": fit["cumulative_energy"].tolist(),
        "rank8_cumulative_energy": float(fit["cumulative_energy"][min(7, len(fit["cumulative_energy"])-1)].item()),
        "basis_output": str(args.basis_output.resolve()),
        "basis_output_sha256": file_sha256(args.basis_output),
    }
    atomic_json(args.output, output)

    for index, item in enumerate(verification_clips):
        scene_name = item["scene"]
        print(json.dumps({"phase": "verification", "index": index, **item, "status": "start"}), flush=True)
        try:
            clip = prepare_clip(
                args.data_root / scene_name,
                start_frame=int(item["start_frame"]),
                protocol=protocol,
                device=args.device,
            )
            teacher_state, student_state = run_to_bridge(model, clip, protocol)
            teacher_query = extract_persistent_query_state(teacher_state)
            student_query = extract_persistent_query_state(student_state)
            teacher_vector, current_layout = flatten_persistent_query_state(teacher_query)
            student_vector, student_layout = flatten_persistent_query_state(student_query)
            if current_layout != layout or student_layout != layout:
                raise RuntimeError("verification state layout differs from basis")
            raw_delta = (teacher_vector - student_vector)[0, 0].detach().float().cpu()
            branches = {
                "native_student": clone_tracking_state(student_state),
                "teacher_full_query_oracle": inject_persistent_query_state(
                    clone_tracking_state(student_state), teacher_query
                ),
                "teacher_full_state_oracle": clone_tracking_state(teacher_state),
            }
            for rank in ranks:
                projected = project_delta(
                    raw_delta,
                    block_rms=fit["block_rms"],
                    block_sizes=layout.block_sizes,
                    mean_normalized=fit["mean_normalized"],
                    components=fit["components"],
                    rank=rank,
                )
                repaired_query = add_persistent_query_state_delta(
                    student_query,
                    projected.to(args.device)[None, None],
                    source_step=teacher_query.source_step,
                )
                branches[f"rank{rank}_oracle"] = inject_persistent_query_state(
                    clone_tracking_state(student_state), repaired_query
                )
            rows = {name: [] for name in branches}
            bridge_step = protocol["pre_frames"] + protocol["gap_frames"]
            total = bridge_step + protocol["post_frames"]
            for rel in range(bridge_step, total):
                target = target_yx(clip["coords_xy"][rel], clip["shape"], args.device)
                for name in tuple(branches):
                    tracks, vis, branches[name] = model_step(
                        model, clip["frames"][rel], state=branches[name]
                    )
                    rows[name].append(record_prediction(tracks, vis, target))
            summaries = {name: summarize(values) for name, values in rows.items()}
            native_error = summaries["native_student"]["mean_error_px_256"]
            full_query_gain = native_error - summaries["teacher_full_query_oracle"]["mean_error_px_256"]
            rank_results = {}
            for rank in ranks:
                key = f"rank{rank}_oracle"
                gain = native_error - summaries[key]["mean_error_px_256"]
                rank_results[str(rank)] = {
                    "gain_vs_native": gain,
                    "retained_full_query_gain": (
                        gain / full_query_gain if full_query_gain > 0.0 else None
                    ),
                    "improved_frame_fraction": float(np.mean([
                        candidate["error_px_256"] < native["error_px_256"]
                        for candidate, native in zip(rows[key], rows["native_student"])
                    ])),
                    "summary": summaries[key],
                }
            rank8 = rank_results[str(primary_rank)]
            result = {
                "scene": scene_name,
                "status": "complete",
                "point_index": clip["point_index"],
                "selection": clip["selection"],
                "post_reappearance": {
                    "native_student": summaries["native_student"],
                    "teacher_full_query_oracle": summaries["teacher_full_query_oracle"],
                    "teacher_full_state_oracle": summaries["teacher_full_state_oracle"],
                },
                "rank_results": rank_results,
                "delta": {
                    "full_query_gain_vs_native": full_query_gain,
                    "full_state_gain_vs_native": native_error - summaries["teacher_full_state_oracle"]["mean_error_px_256"],
                    "rank8_gain_vs_native": rank8["gain_vs_native"],
                    "rank8_retained_full_query_gain": rank8["retained_full_query_gain"],
                    "rank8_improved_frame_fraction": rank8["improved_frame_fraction"],
                },
                "frame_rows": rows,
            }
        except Exception as error:  # fail closed and retain all frozen scenes
            result = {
                "scene": scene_name,
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        output["verification_results"].append(result)
        output["gate"] = evaluate_gate(
            output["verification_results"], expected=len(verification_clips)
        )
        atomic_json(args.output, output)
        print(json.dumps({
            "phase": "verification",
            "scene": scene_name,
            "status": result["status"],
            "delta": result.get("delta"),
            "partial_gate": output["gate"],
        }), flush=True)
        torch.cuda.empty_cache()

    output["gate"] = evaluate_gate(
        output["verification_results"], expected=len(verification_clips)
    )
    atomic_json(args.output, output)
    print(json.dumps({
        "output": str(args.output),
        "basis_output": str(args.basis_output),
        "basis_summary": output["basis_summary"],
        "gate": output["gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
