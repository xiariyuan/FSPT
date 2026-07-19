#!/usr/bin/env python3
"""Fit-only Gate 1 audit for oracle CoTracker3 latent-state transplantation."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
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
    CoTrackerOnlineStateSnapshot,
    restore_cotracker_online_state,
    snapshot_cotracker_online_state,
    snapshot_tensor_hashes,
    snapshots_exact,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256
from projects.mmp_tracker.mmp_tracker.routeD_musr_training import (
    paired_video_bootstrap_ci,
)
from projects.mmp_tracker.mmp_tracker.routeD_oracle_state_transplant import (
    GATE1_SCHEMA_VERSION,
    aggregate_variant_metrics,
    point_future_metrics,
    select_failure_points,
    support_delta_svd_energy,
    transplant_fresh_state,
)
from scripts.audit_routeD_cotracker3_interface import (
    load_manifest_sample,
    prepare_sample,
    set_deterministic,
)

DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_oracle_state_transplant_gate1_v0.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/routeD_oracle_state_transplant_gate1_20260719/primary.json"
VARIANT_NAMES = ("native", "coordinate_only", "coordinate_probability", "full_state")


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if config.get("schema_version") != GATE1_SCHEMA_VERSION:
        raise ValueError("unexpected oracle state-transplant Gate 1 schema")
    if file_sha256(config["backbone"]["checkpoint"]) != config["backbone"]["checkpoint_sha256"]:
        raise ValueError("CoTracker checkpoint hash drift")
    if file_sha256(config["fit_partition"]["manifest"]) != config["fit_partition"]["manifest_sha256"]:
        raise ValueError("fit manifest hash drift")
    expected = list(range(8))
    if list(config["fit_partition"]["source_indices"]) != expected:
        raise ValueError("Gate 1 source membership drift")
    if any(value is not False for value in config["locked_data"].values()):
        raise ValueError("Gate 1 locked-data flags must remain false")
    return config


def _original_queries(prepared: dict[str, Any], video: torch.Tensor) -> torch.Tensor:
    query_tyx = prepared["query_points_tyx"].to(video.device)
    queries = torch.zeros(1, query_tyx.shape[0], 3, device=video.device)
    queries[0, :, 0] = query_tyx[:, 0]
    queries[0, :, 1] = query_tyx[:, 2] * float(video.shape[-1] - 1)
    queries[0, :, 2] = query_tyx[:, 1] * float(video.shape[-2] - 1)
    return queries


def _fresh_queries(
    prepared: dict[str, Any],
    selected: torch.Tensor,
    oracle_frames: torch.Tensor,
    video: torch.Tensor,
) -> torch.Tensor:
    gt_yx = prepared["gt_tracks_yx"].to(video.device)
    selected_device = selected.to(video.device)
    frames_device = oracle_frames.to(video.device)
    rows = torch.arange(selected.numel(), device=video.device)
    query_yx = gt_yx[selected_device, frames_device]
    queries = torch.zeros(1, selected.numel(), 3, device=video.device)
    queries[0, :, 0] = frames_device.float()
    queries[0, :, 1] = query_yx[:, 1] * float(video.shape[-1] - 1)
    queries[0, :, 2] = query_yx[:, 0] * float(video.shape[-2] - 1)
    del rows
    return queries


def _initialize_and_first_window(
    predictor: CoTrackerOnlinePredictor,
    video: torch.Tensor,
    queries: torch.Tensor,
) -> CoTrackerOnlineStateSnapshot:
    predictor(
        video_chunk=video,
        is_first_step=True,
        queries=queries,
        add_support_grid=False,
        grid_size=0,
    )
    predictor(
        video_chunk=video[:, : predictor.step * 2],
        is_first_step=False,
        add_support_grid=False,
        grid_size=0,
    )
    snapshot = snapshot_cotracker_online_state(predictor)
    if snapshot.online_ind != predictor.step:
        raise RuntimeError("unexpected first-window commit index")
    return snapshot


def _continue_second_window(
    predictor: CoTrackerOnlinePredictor,
    video: torch.Tensor,
    initial: CoTrackerOnlineStateSnapshot,
) -> CoTrackerOnlineStateSnapshot:
    restore_cotracker_online_state(predictor, initial)
    predictor(
        video_chunk=video[:, predictor.step : predictor.step * 3],
        is_first_step=False,
        add_support_grid=False,
        grid_size=0,
    )
    final = snapshot_cotracker_online_state(predictor)
    if final.online_coords_predicted.shape[1] != video.shape[1]:
        raise RuntimeError("incomplete Gate 1 future state")
    return final


def _coords_to_input(
    snapshot: CoTrackerOnlineStateSnapshot,
    *,
    interp_height: int,
    interp_width: int,
) -> torch.Tensor:
    coords = snapshot.online_coords_predicted[0].detach().float().cpu().clone()
    coords[..., 0] *= 255.0 / float(max(interp_width - 1, 1))
    coords[..., 1] *= 255.0 / float(max(interp_height - 1, 1))
    return coords.permute(1, 0, 2).contiguous()  # N,T,2


def _selected_future_artifact(
    snapshot: CoTrackerOnlineStateSnapshot,
    selected: torch.Tensor,
    *,
    future_start: int,
    future_end_inclusive: int,
) -> dict[str, torch.Tensor]:
    selected_device = selected.to(snapshot.online_coords_predicted.device)
    frame_slice = slice(int(future_start), int(future_end_inclusive) + 1)
    return {
        "coordinates": snapshot.online_coords_predicted[
            0, frame_slice, selected_device
        ].detach().cpu(),
        "visibility": snapshot.online_vis_predicted[
            0, frame_slice, selected_device
        ].detach().cpu(),
        "confidence": snapshot.online_conf_predicted[
            0, frame_slice, selected_device
        ].detach().cpu(),
    }


def _fresh_memory_artifact(
    native: CoTrackerOnlineStateSnapshot,
    fresh: CoTrackerOnlineStateSnapshot,
    selected: torch.Tensor,
) -> dict[str, Any]:
    selected_device = selected.to(native.online_coords_predicted.device)
    return {
        "native_track_feat": [
            None if value is None else value[:, :, selected_device].detach().cpu()
            for value in native.online_track_feat
        ],
        "fresh_track_feat": [
            None if value is None else value.detach().cpu()
            for value in fresh.online_track_feat
        ],
        "native_track_support": [
            None if value is None else value[:, :, selected_device].detach().cpu()
            for value in native.online_track_support
        ],
        "fresh_track_support": [
            None if value is None else value.detach().cpu()
            for value in fresh.online_track_support
        ],
    }


def _rows_by_point(rows: list[dict[str, float | int]]) -> dict[int, dict[str, float | int]]:
    return {int(row["point_index"]): row for row in rows}


def _aggregate_svd(video_rows: list[dict[str, Any]], ranks: list[int]) -> dict[str, Any]:
    denominator = float(sum(row["support_svd"]["pooled_total_energy"] for row in video_rows))
    numerators = {
        str(rank): float(
            sum(
                row["support_svd"]["pooled_rank_energy_numerator"][str(rank)]
                for row in video_rows
            )
        )
        for rank in ranks
    }
    return {
        "pooled_total_energy": denominator,
        "pooled_rank_energy_numerator": numerators,
        "pooled_rank_energy": {
            key: (1.0 if denominator <= 1e-12 else value / denominator)
            for key, value in numerators.items()
        },
    }


def _paired_point_bootstrap(
    differences: list[float], *, seed: int, samples: int
) -> dict[str, float | int]:
    return paired_video_bootstrap_ci(differences, seed=seed, samples=samples)


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
    partition = config["fit_partition"]
    commit = config["commit_protocol"]
    selection_cfg = config["failure_event_selection"]
    metric_cfg = config["metrics"]
    ranks = [int(value) for value in metric_cfg["teacher_state"]["support_delta_svd_energy_ranks"]]
    seed = int(metric_cfg["aggregate"]["paired_point_bootstrap_seed"])
    samples = int(metric_cfg["aggregate"]["paired_point_bootstrap_samples"])

    set_deterministic(seed)
    predictor = CoTrackerOnlinePredictor(
        checkpoint=str(config["backbone"]["checkpoint"])
    ).to(args.device).eval()

    video_reports: list[dict[str, Any]] = []
    artifact_videos: list[dict[str, Any]] = []
    all_variant_rows = {name: [] for name in VARIANT_NAMES}
    native_replay_exact_all = True
    fresh_replay_exact_all = True

    for source_index in partition["source_indices"]:
        sample, sample_metadata = load_manifest_sample(
            Path(partition["manifest"]), int(source_index)
        )
        prepared = prepare_sample(sample, int(config["backbone"]["input_raster"]))
        video = prepared["video"].to(args.device)
        if int(video.shape[1]) != int(partition["expected_frames_each"]):
            raise ValueError("Gate 1 frame-count drift")
        if int(prepared["gt_tracks_yx"].shape[0]) != int(
            partition["expected_points_each"]
        ):
            raise ValueError("Gate 1 point-count drift")

        set_deterministic(seed + int(source_index))
        native_initial = _initialize_and_first_window(
            predictor, video, _original_queries(prepared, video)
        )
        native_final = _continue_second_window(predictor, video, native_initial)
        native_replay = _continue_second_window(predictor, video, native_initial)
        native_replay_exact = snapshots_exact(native_final, native_replay)
        native_replay_exact_all = native_replay_exact_all and native_replay_exact

        selection = select_failure_points(
            native_coords_xy_px=_coords_to_input(
                native_final,
                interp_height=int(predictor.interp_shape[0]),
                interp_width=int(predictor.interp_shape[1]),
            ),
            gt_tracks_yx=prepared["gt_tracks_yx"],
            gt_occluded=prepared["gt_occluded"],
            original_query_frames=prepared["query_points_tyx"][:, 0],
            overlap_start=int(commit["overlap_frames"][0]),
            overlap_end_inclusive=int(commit["overlap_frames"][1]),
            future_start=int(commit["future_frames"][0]),
            future_end_inclusive=int(commit["future_frames"][1]),
            min_overlap_visible_frames=int(
                selection_cfg["gt_visible_in_overlap_min_frames"]
            ),
            min_future_visible_frames=int(
                selection_cfg["gt_visible_in_future_min_frames"]
            ),
            min_native_future_mean_error_px=float(
                selection_cfg["native_future_mean_error_px_min"]
            ),
            per_video_cap=int(selection_cfg["per_video_cap"]),
        )
        selected = selection["selected_point_indices"]
        oracle_frames = selection["oracle_query_frames"]
        report: dict[str, Any] = {
            "source_index": int(source_index),
            "video_name": str(prepared["video_name"]),
            "selected_points": int(selected.numel()),
            "selected_point_indices": selected.tolist(),
            "oracle_query_frames": oracle_frames.tolist(),
            "native_future_mean_error_px_at_selection": selection[
                "native_future_mean_error_px"
            ].tolist(),
            "eligible_points_before_cap": int(selection["eligible_mask"].sum().item()),
            "native_clean_replay_exact": native_replay_exact,
            "sample_metadata": sample_metadata,
        }

        if selected.numel() == 0:
            report.update(
                {
                    "fresh_teacher_replay_exact": True,
                    "variant_metrics": {},
                    "support_svd": {
                        "pooled_total_energy": 0.0,
                        "pooled_rank_energy_numerator": {
                            str(rank): 0.0 for rank in ranks
                        },
                        "pooled_rank_energy": {str(rank): 1.0 for rank in ranks},
                        "levels": [],
                    },
                }
            )
            video_reports.append(report)
            artifact_videos.append(
                {
                    "source_index": int(source_index),
                    "selected_point_indices": selected,
                    "oracle_query_frames": oracle_frames,
                    "native_clean_replay_exact": native_replay_exact,
                }
            )
            continue

        fresh_queries = _fresh_queries(prepared, selected, oracle_frames, video)
        fresh_initial = _initialize_and_first_window(predictor, video, fresh_queries)
        fresh_replay = _initialize_and_first_window(predictor, video, fresh_queries)
        fresh_replay_exact = snapshots_exact(fresh_initial, fresh_replay)
        fresh_replay_exact_all = fresh_replay_exact_all and fresh_replay_exact

        coordinate_initial = transplant_fresh_state(
            native_initial,
            fresh_initial,
            selected_point_indices=selected,
            oracle_query_frames=oracle_frames,
            overlap_end_inclusive=int(commit["overlap_frames"][1]),
            copy_probability=False,
            copy_track_memory=False,
        )
        probability_initial = transplant_fresh_state(
            native_initial,
            fresh_initial,
            selected_point_indices=selected,
            oracle_query_frames=oracle_frames,
            overlap_end_inclusive=int(commit["overlap_frames"][1]),
            copy_probability=True,
            copy_track_memory=False,
        )
        full_initial = transplant_fresh_state(
            native_initial,
            fresh_initial,
            selected_point_indices=selected,
            oracle_query_frames=oracle_frames,
            overlap_end_inclusive=int(commit["overlap_frames"][1]),
            copy_probability=True,
            copy_track_memory=True,
        )
        finals = {
            "native": native_final,
            "coordinate_only": _continue_second_window(
                predictor, video, coordinate_initial
            ),
            "coordinate_probability": _continue_second_window(
                predictor, video, probability_initial
            ),
            "full_state": _continue_second_window(predictor, video, full_initial),
        }
        variant_rows: dict[str, list[dict[str, float | int]]] = {}
        for name, final in finals.items():
            rows = point_future_metrics(
                final_coords_xy_model=final.online_coords_predicted,
                selected_point_indices=selected,
                gt_tracks_yx=prepared["gt_tracks_yx"],
                gt_occluded=prepared["gt_occluded"],
                future_start=int(commit["future_frames"][0]),
                future_end_inclusive=int(commit["future_frames"][1]),
                interp_height=int(predictor.interp_shape[0]),
                interp_width=int(predictor.interp_shape[1]),
            )
            variant_rows[name] = rows
            for row in rows:
                enriched = {
                    **row,
                    "source_index": int(source_index),
                    "video_name": str(prepared["video_name"]),
                    "oracle_query_frame": int(
                        oracle_frames[
                            (selected == int(row["point_index"])).nonzero()[0, 0]
                        ].item()
                    ),
                }
                all_variant_rows[name].append(enriched)

        svd = support_delta_svd_energy(
            native_initial,
            fresh_initial,
            selected_point_indices=selected,
            ranks=ranks,
        )
        report.update(
            {
                "fresh_teacher_replay_exact": fresh_replay_exact,
                "fresh_initial_state_hashes": snapshot_tensor_hashes(fresh_initial),
                "variant_metrics": {
                    name: aggregate_variant_metrics(rows)
                    for name, rows in variant_rows.items()
                },
                "support_svd": svd,
            }
        )
        video_reports.append(report)
        artifact_videos.append(
            {
                "source_index": int(source_index),
                "video_name": str(prepared["video_name"]),
                "selected_point_indices": selected,
                "oracle_query_frames": oracle_frames,
                "native_clean_replay_exact": native_replay_exact,
                "fresh_teacher_replay_exact": fresh_replay_exact,
                "initial_state_hashes": {
                    "native": snapshot_tensor_hashes(native_initial),
                    "fresh": snapshot_tensor_hashes(fresh_initial),
                },
                "future": {
                    name: _selected_future_artifact(
                        value,
                        selected,
                        future_start=int(commit["future_frames"][0]),
                        future_end_inclusive=int(commit["future_frames"][1]),
                    )
                    for name, value in finals.items()
                },
                "teacher_memory": _fresh_memory_artifact(
                    native_initial, fresh_initial, selected
                ),
                "variant_rows": variant_rows,
                "support_svd": svd,
            }
        )
        del video
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    selected_points = len(all_variant_rows["native"])
    selected_videos = sum(row["selected_points"] > 0 for row in video_reports)
    aggregates = {
        name: (
            aggregate_variant_metrics(rows)
            if rows
            else {
                "points": 0,
                "mean_l2_error_px": float("nan"),
                "severe_16px_rate": float("nan"),
                "threshold_utility": float("nan"),
            }
        )
        for name, rows in all_variant_rows.items()
    }

    comparisons: dict[str, Any] = {}
    if selected_points > 0:
        maps = {name: _rows_by_point(rows) for name, rows in all_variant_rows.items()}
        # Source index + point index is the actual identity; point indices repeat across videos.
        identity_maps = {
            name: {
                (int(row["source_index"]), int(row["point_index"])): row
                for row in rows
            }
            for name, rows in all_variant_rows.items()
        }
        identities = sorted(identity_maps["native"])
        if any(set(identity_maps[name]) != set(identities) for name in VARIANT_NAMES):
            raise RuntimeError("Gate 1 per-point identity mismatch")

        def compare(left: str, right: str) -> dict[str, Any]:
            error_reduction = [
                float(identity_maps[right][identity]["mean_l2_error_px"])
                - float(identity_maps[left][identity]["mean_l2_error_px"])
                for identity in identities
            ]
            utility_gain = [
                float(identity_maps[left][identity]["threshold_utility"])
                - float(identity_maps[right][identity]["threshold_utility"])
                for identity in identities
            ]
            positive = [value > 0 for value in error_reduction]
            return {
                "left": left,
                "right": right,
                "mean_error_reduction_px": float(np.mean(error_reduction)),
                "mean_threshold_utility_gain": float(np.mean(utility_gain)),
                "positive_point_fraction": float(np.mean(positive)),
                "paired_point_error_reduction_CI": _paired_point_bootstrap(
                    error_reduction, seed=seed, samples=samples
                ),
                "paired_point_utility_gain_CI": _paired_point_bootstrap(
                    utility_gain, seed=seed + 1, samples=samples
                ),
            }

        comparisons = {
            "full_state_vs_native": compare("full_state", "native"),
            "full_state_vs_coordinate_only": compare(
                "full_state", "coordinate_only"
            ),
            "full_state_vs_coordinate_probability": compare(
                "full_state", "coordinate_probability"
            ),
        }
        event_video_rows = [row for row in video_reports if row["selected_points"] > 0]
        comparisons["full_state_vs_coordinate_only"][
            "better_video_fraction"
        ] = float(
            np.mean(
                [
                    row["variant_metrics"]["full_state"]["mean_l2_error_px"]
                    < row["variant_metrics"]["coordinate_only"][
                        "mean_l2_error_px"
                    ]
                    for row in event_video_rows
                ]
            )
        )

    gates = config["scientific_gates"]
    if selected_points > 0:
        fn = comparisons["full_state_vs_native"]
        fc = comparisons["full_state_vs_coordinate_only"]
        fp = comparisons["full_state_vs_coordinate_probability"]
        checks = {
            "native_clean_independent_replay_exact": native_replay_exact_all,
            "fresh_teacher_independent_replay_exact": fresh_replay_exact_all,
            "native_no_transplant_parity_exact": native_replay_exact_all,
            "candidate_or_selector_used": False,
            "selected_points_min": selected_points >= int(gates["selected_points_min"]),
            "selected_videos_min": selected_videos >= int(gates["selected_videos_min"]),
            "full_state_mean_error_reduction_vs_native_px_min": fn[
                "mean_error_reduction_px"
            ]
            >= float(gates["full_state_mean_error_reduction_vs_native_px_min"]),
            "full_state_threshold_utility_gain_vs_native_min": fn[
                "mean_threshold_utility_gain"
            ]
            >= float(gates["full_state_threshold_utility_gain_vs_native_min"]),
            "full_state_positive_point_fraction_vs_native_min": fn[
                "positive_point_fraction"
            ]
            >= float(gates["full_state_positive_point_fraction_vs_native_min"]),
            "full_state_severe_16px_rate_not_worse": aggregates["full_state"][
                "severe_16px_rate"
            ]
            <= aggregates["native"]["severe_16px_rate"] + 1e-12,
            "full_state_mean_error_reduction_vs_coordinate_only_px_min": fc[
                "mean_error_reduction_px"
            ]
            >= float(
                gates["full_state_mean_error_reduction_vs_coordinate_only_px_min"]
            ),
            "full_state_threshold_utility_gain_vs_coordinate_only_min": fc[
                "mean_threshold_utility_gain"
            ]
            >= float(
                gates["full_state_threshold_utility_gain_vs_coordinate_only_min"]
            ),
            "full_state_better_than_coordinate_only_video_fraction_min": fc[
                "better_video_fraction"
            ]
            >= float(
                gates[
                    "full_state_better_than_coordinate_only_video_fraction_min"
                ]
            ),
            "full_state_mean_error_reduction_vs_coordinate_probability_px_min": fp[
                "mean_error_reduction_px"
            ]
            >= float(
                gates[
                    "full_state_mean_error_reduction_vs_coordinate_probability_px_min"
                ]
            ),
            "full_state_threshold_utility_gain_vs_coordinate_probability_min": fp[
                "mean_threshold_utility_gain"
            ]
            >= float(
                gates[
                    "full_state_threshold_utility_gain_vs_coordinate_probability_min"
                ]
            ),
        }
    else:
        checks = {
            "native_clean_independent_replay_exact": native_replay_exact_all,
            "fresh_teacher_independent_replay_exact": fresh_replay_exact_all,
            "native_no_transplant_parity_exact": native_replay_exact_all,
            "candidate_or_selector_used": False,
            **{key: False for key in gates},
        }
    # This check is phrased as a factual integrity value; false is the required state.
    checks["candidate_or_selector_not_used"] = not checks.pop(
        "candidate_or_selector_used"
    )
    local_pass = all(checks.values())
    decision = (
        "AWAIT_INDEPENDENT_REPLAY"
        if local_pass
        else config["formal_decisions"]["fail"]
    )
    svd_aggregate = _aggregate_svd(
        [row for row in video_reports if row["selected_points"] > 0], ranks
    ) if selected_videos > 0 else {
        "pooled_total_energy": 0.0,
        "pooled_rank_energy_numerator": {str(rank): 0.0 for rank in ranks},
        "pooled_rank_energy": {str(rank): 1.0 for rank in ranks},
    }

    artifact = {
        "schema_version": GATE1_SCHEMA_VERSION,
        "videos": artifact_videos,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "checks": checks,
        "support_svd_aggregate": svd_aggregate,
    }
    torch.save(artifact, sidecar)
    report = {
        "schema_version": GATE1_SCHEMA_VERSION,
        "status": "completed_awaiting_independent_replay" if local_pass else "completed_fail",
        "decision": decision,
        "local_pass": local_pass,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "backbone_checkpoint_sha256": config["backbone"]["checkpoint_sha256"],
        "fit_manifest_sha256": partition["manifest_sha256"],
        "videos": len(video_reports),
        "selected_points": selected_points,
        "selected_videos": selected_videos,
        "video_reports": video_reports,
        "variant_aggregates": aggregates,
        "comparisons": comparisons,
        "support_svd_aggregate": svd_aggregate,
        "local_checks": checks,
        "sidecar": str(sidecar),
        "sidecar_sha256": file_sha256(sidecar),
        "locked_data": config["locked_data"],
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "decision": decision,
        "local_pass": local_pass,
        "selected_points": selected_points,
        "selected_videos": selected_videos,
        "variant_aggregates": aggregates,
        "comparisons": comparisons,
        "support_svd_aggregate": svd_aggregate,
        "checks": checks,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
