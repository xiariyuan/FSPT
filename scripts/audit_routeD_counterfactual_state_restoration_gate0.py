#!/usr/bin/env python3
"""Fit-only Gate 0 audit for counterfactual CoTracker3 state restoration."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = Path("/gemini/code/FSPT")
COTRACKER_ROOT = EXTERNAL_ROOT / "baselines/cotracker"
for path in (COTRACKER_ROOT, EXTERNAL_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from cotracker.predictor import CoTrackerOnlinePredictor

from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restoration import (
    GATE0_SCHEMA_VERSION,
    CoTrackerOnlineStateSnapshot,
    composite_corrupt_snapshot,
    future_difference,
    restore_coordinates_only,
    restore_cotracker_online_state,
    snapshot_cotracker_online_state,
    snapshot_tensor_hashes,
    snapshots_exact,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256
from scripts.audit_routeD_cotracker3_interface import (
    load_manifest_sample,
    prepare_sample,
    set_deterministic,
)

DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_counterfactual_state_restoration_gate0_v0.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/routeD_counterfactual_state_restoration_gate0_20260719/primary.json"


def _snapshot_artifact(snapshot: CoTrackerOnlineStateSnapshot) -> dict[str, Any]:
    return {
        "predictor_n": int(snapshot.predictor_n),
        "predictor_queries": snapshot.predictor_queries.detach().cpu(),
        "online_ind": int(snapshot.online_ind),
        "online_track_feat": [
            None if value is None else value.detach().cpu()
            for value in snapshot.online_track_feat
        ],
        "online_track_support": [
            None if value is None else value.detach().cpu()
            for value in snapshot.online_track_support
        ],
        "online_coords_predicted": snapshot.online_coords_predicted.detach().cpu(),
        "online_vis_predicted": snapshot.online_vis_predicted.detach().cpu(),
        "online_conf_predicted": snapshot.online_conf_predicted.detach().cpu(),
    }


def _build_queries(prepared: dict[str, Any], video: torch.Tensor) -> torch.Tensor:
    _, _, _, height, width = video.shape
    queries_tyx = prepared["query_points_tyx"].to(video.device)
    queries = torch.zeros(
        1, queries_tyx.shape[0], 3, device=video.device, dtype=torch.float32
    )
    queries[0, :, 0] = queries_tyx[:, 0]
    queries[0, :, 1] = queries_tyx[:, 2] * float(width - 1)
    queries[0, :, 2] = queries_tyx[:, 1] * float(height - 1)
    return queries


def _continue_one_window(
    predictor: CoTrackerOnlinePredictor,
    video: torch.Tensor,
    snapshot: CoTrackerOnlineStateSnapshot,
    *,
    chunk_start: int,
) -> CoTrackerOnlineStateSnapshot:
    restore_cotracker_online_state(predictor, snapshot)
    chunk = video[:, chunk_start : chunk_start + predictor.step * 2]
    predictor(
        video_chunk=chunk,
        is_first_step=False,
        add_support_grid=False,
        grid_size=0,
    )
    return snapshot_cotracker_online_state(predictor)


def _exact_future(
    clean: CoTrackerOnlineStateSnapshot, other: CoTrackerOnlineStateSnapshot
) -> bool:
    return bool(
        torch.equal(clean.online_coords_predicted, other.online_coords_predicted)
        and torch.equal(clean.online_vis_predicted, other.online_vis_predicted)
        and torch.equal(clean.online_conf_predicted, other.online_conf_predicted)
        and snapshots_exact(clean, other)
    )


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if config.get("schema_version") != GATE0_SCHEMA_VERSION:
        raise ValueError("unexpected state-restoration Gate 0 schema")
    if file_sha256(config["backbone"]["checkpoint"]) != config["backbone"]["checkpoint_sha256"]:
        raise ValueError("CoTracker checkpoint hash drift")
    if file_sha256(config["fit_sample"]["manifest"]) != config["fit_sample"]["manifest_sha256"]:
        raise ValueError("fit manifest hash drift")
    for key in (
        "model_validation_read",
        "calibration_read",
        "final_holdout_read",
        "tapvid_davis_read",
        "tapvid_kinetics_read",
        "official_kinetics_1144_rerun",
    ):
        if config["integrity"][key] is not False:
            raise ValueError(f"locked-data flag must remain false: {key}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    sidecar = output.with_suffix(".pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path)
    sample_cfg = config["fit_sample"]
    sample, sample_metadata = load_manifest_sample(
        Path(sample_cfg["manifest"]), int(sample_cfg["source_index"])
    )
    prepared = prepare_sample(sample, 256)
    if str(prepared["video_name"]) != str(sample_cfg["video_name"]):
        raise ValueError("fit sample video-name drift")
    if int(prepared["video"].shape[1]) != int(sample_cfg["expected_frames"]):
        raise ValueError("fit sample frame-count drift")
    if int(prepared["gt_tracks_yx"].shape[0]) != int(sample_cfg["expected_points"]):
        raise ValueError("fit sample point-count drift")

    set_deterministic(int(config["corruption"]["seed"]))
    predictor = CoTrackerOnlinePredictor(
        checkpoint=str(config["backbone"]["checkpoint"])
    ).to(args.device).eval()
    video = prepared["video"].to(args.device)
    queries = _build_queries(prepared, video)
    predictor(
        video_chunk=video,
        is_first_step=True,
        queries=queries,
        add_support_grid=False,
        grid_size=0,
    )
    first_start = int(config["commit_interface"]["first_window_chunk_start"])
    first_chunk = video[:, first_start : first_start + predictor.step * 2]
    predictor(
        video_chunk=first_chunk,
        is_first_step=False,
        add_support_grid=False,
        grid_size=0,
    )
    initial = snapshot_cotracker_online_state(predictor)
    if int(initial.online_ind) != int(
        config["commit_interface"]["snapshot_online_ind_expected"]
    ):
        raise RuntimeError("unexpected CoTracker commit index")
    restore_cotracker_online_state(predictor, initial)
    snapshot_roundtrip = snapshot_cotracker_online_state(predictor)
    snapshot_roundtrip_exact = snapshots_exact(initial, snapshot_roundtrip)

    next_start = int(config["commit_interface"]["next_window_chunk_start"])
    clean = _continue_one_window(predictor, video, initial, chunk_start=next_start)
    noop = _continue_one_window(predictor, video, initial, chunk_start=next_start)

    corruption_cfg = config["corruption"]
    overlap = config["commit_interface"]["overlap_frames_in_global_time"]
    corrupted_initial = composite_corrupt_snapshot(
        initial,
        input_height=int(video.shape[-2]),
        input_width=int(video.shape[-1]),
        interp_height=int(predictor.interp_shape[0]),
        interp_width=int(predictor.interp_shape[1]),
        overlap_start=int(overlap[0]),
        overlap_end_inclusive=int(overlap[1]),
        active_before_frame=16,
        coordinate_shift_input_xy_px=tuple(
            float(value)
            for value in corruption_cfg["coordinate"]["input_raster_shift_xy_px"]
        ),
        visibility_logit_delta=float(corruption_cfg["visibility"]["logit_delta"]),
        confidence_logit_delta=float(corruption_cfg["confidence"]["logit_delta"]),
        support_channel_keep_fraction=float(
            corruption_cfg["track_support"]["deterministic_channel_keep_fraction"]
        ),
        seed=int(corruption_cfg["seed"]),
    )
    composite = _continue_one_window(
        predictor, video, corrupted_initial, chunk_start=next_start
    )
    coordinate_only_initial = restore_coordinates_only(corrupted_initial, initial)
    coordinate_only = _continue_one_window(
        predictor, video, coordinate_only_initial, chunk_start=next_start
    )
    # Deliberately restore a corrupted live state first, then the complete clean
    # state. This tests the actual restoration operation, not merely reuse.
    restore_cotracker_online_state(predictor, corrupted_initial)
    restore_cotracker_online_state(predictor, initial)
    full_restore = _continue_one_window(
        predictor,
        video,
        snapshot_cotracker_online_state(predictor),
        chunk_start=next_start,
    )

    future = config["commit_interface"]["future_measurement_frames_in_global_time"]
    variant_snapshots = {
        "clean": clean,
        "exact_noop_restore": noop,
        "composite_corrupt": composite,
        "coordinate_only_restore_after_composite_corruption": coordinate_only,
        "full_state_restore_after_composite_corruption": full_restore,
    }
    differences = {
        name: future_difference(
            clean,
            value,
            future_start=int(future[0]),
            future_end_inclusive=int(future[1]),
            input_height=int(video.shape[-2]),
            input_width=int(video.shape[-1]),
            interp_height=int(predictor.interp_shape[0]),
            interp_width=int(predictor.interp_shape[1]),
        )
        for name, value in variant_snapshots.items()
        if name != "clean"
    }
    local_checks = {
        "snapshot_roundtrip_state_exact": snapshot_roundtrip_exact,
        "exact_noop_restore_future_exact": _exact_future(clean, noop),
        "full_state_restore_future_exact": _exact_future(clean, full_restore),
        "composite_corruption_future_mean_l2_px_min": (
            differences["composite_corrupt"]["coordinate_mean_l2_px"]
            >= float(config["gates"]["composite_corruption_future_mean_l2_px_min"])
        ),
        "composite_corruption_future_fraction_above_1px_min": (
            differences["composite_corrupt"]["coordinate_fraction_above_1px"]
            >= float(
                config["gates"][
                    "composite_corruption_future_fraction_above_1px_min"
                ]
            )
        ),
        "coordinate_only_restore_future_mean_l2_px_min": (
            differences[
                "coordinate_only_restore_after_composite_corruption"
            ]["coordinate_mean_l2_px"]
            >= float(
                config["gates"]["coordinate_only_restore_future_mean_l2_px_min"]
            )
        ),
        "coordinate_only_restore_future_fraction_above_1px_min": (
            differences[
                "coordinate_only_restore_after_composite_corruption"
            ]["coordinate_fraction_above_1px"]
            >= float(
                config["gates"][
                    "coordinate_only_restore_future_fraction_above_1px_min"
                ]
            )
        ),
        "full_restore_better_than_coordinate_only": (
            differences["full_state_restore_after_composite_corruption"][
                "coordinate_mean_l2_px"
            ]
            < differences[
                "coordinate_only_restore_after_composite_corruption"
            ]["coordinate_mean_l2_px"]
        ),
        "full_restore_better_than_composite_corruption": (
            differences["full_state_restore_after_composite_corruption"][
                "coordinate_mean_l2_px"
            ]
            < differences["composite_corrupt"]["coordinate_mean_l2_px"]
        ),
    }
    local_pass = all(local_checks.values())

    artifact = {
        "schema_version": GATE0_SCHEMA_VERSION,
        "initial": _snapshot_artifact(initial),
        "snapshot_roundtrip": _snapshot_artifact(snapshot_roundtrip),
        "corrupted_initial": _snapshot_artifact(corrupted_initial),
        "coordinate_only_initial": _snapshot_artifact(coordinate_only_initial),
        "variants": {
            name: _snapshot_artifact(value) for name, value in variant_snapshots.items()
        },
    }
    torch.save(artifact, sidecar)
    report = {
        "schema_version": GATE0_SCHEMA_VERSION,
        "status": "completed_awaiting_independent_replay",
        "decision": (
            "AWAIT_INDEPENDENT_REPLAY"
            if local_pass
            else config["formal_decisions"]["fail"]
        ),
        "local_pass": local_pass,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "backbone_checkpoint_sha256": config["backbone"]["checkpoint_sha256"],
        "sample": {
            "source_index": int(sample_cfg["source_index"]),
            "video_name": str(prepared["video_name"]),
            "frames": int(video.shape[1]),
            "points": int(prepared["gt_tracks_yx"].shape[0]),
            **sample_metadata,
        },
        "interp_shape": [int(value) for value in predictor.interp_shape],
        "state_shapes": {
            "track_feat": [
                None if value is None else list(value.shape)
                for value in initial.online_track_feat
            ],
            "track_support": [
                None if value is None else list(value.shape)
                for value in initial.online_track_support
            ],
            "coordinates": list(initial.online_coords_predicted.shape),
            "visibility": list(initial.online_vis_predicted.shape),
            "confidence": list(initial.online_conf_predicted.shape),
        },
        "initial_state_hashes": snapshot_tensor_hashes(initial),
        "corrupted_initial_state_hashes": snapshot_tensor_hashes(corrupted_initial),
        "variant_state_hashes": {
            name: snapshot_tensor_hashes(value)
            for name, value in variant_snapshots.items()
        },
        "future_differences_vs_clean": differences,
        "local_checks": local_checks,
        "sidecar": str(sidecar),
        "sidecar_sha256": file_sha256(sidecar),
        "integrity": config["integrity"],
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
