#!/usr/bin/env python3
"""Evaluate a frozen Route-D selector in open-loop or true closed-loop mode."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import compute_tapvid_metrics
from projects.mmp_tracker.mmp_tracker import MMPTracker
from projects.mmp_tracker.mmp_tracker.hypothesis_scorer import build_hypothesis_features
from projects.mmp_tracker.mmp_tracker.routeD_selector import load_routeD_selector
from projects.mmp_tracker.train_mmp import (
    config_from_dict,
    resolve_dataset,
    resolve_eval_query_mode,
    resolve_eval_resolution,
)


def finite_mean(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    return sum(values) / len(values) if values else float("nan")


def normalize_metrics(metrics):
    return {
        "AJ": float(metrics.get("AJ", metrics.get("average_jaccard", 0.0))),
        "OA": float(metrics.get("OA", metrics.get("occlusion_accuracy", 0.0))),
        "delta_avg": float(
            metrics.get("<avg", metrics.get("average_pts_within_thresh", 0.0))
        ),
    }


def load_base_model(config, checkpoint_state, device):
    model = MMPTracker(config_from_dict(config))
    model.load_state_dict(checkpoint_state, strict=True)
    return model.to(device).eval()


def active_after_query(query_points, time):
    frame_index = torch.arange(time, device=query_points.device).view(1, 1, time)
    return frame_index > query_points[..., 0].round().long().unsqueeze(-1)


def masked_mean(values, mask):
    denominator = mask.sum().clamp_min(1)
    return float((values * mask.to(values.dtype)).sum() / denominator)


def resolve_video_name(batch, sample_index):
    value = batch.get("video_name")
    if value is None:
        return str(sample_index)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else sample_index
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selector-bundle", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--fusion-strength", type=float, default=None)
    parser.add_argument("--max-switch-distance-px", type=float, default=None)
    parser.add_argument("--profile-p1-tolerance", type=float, default=None)
    parser.add_argument("--profile-min-coarse-gain", type=float, default=0.0)
    parser.add_argument("--profile-min-total-gain", type=float, default=0.0)
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--query-mode", choices=("first", "strided"), default=None)
    parser.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--metric-resolution", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baseline-device", default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = yaml.safe_load(Path(args.config).read_text())
    config.setdefault("model", {}).setdefault("tracking", {})[
        "enable_multi_hypothesis_diagnostics"
    ] = True
    data_config = config.setdefault("data", {}).setdefault(args.split, {})
    if args.dataset:
        data_config["dataset"] = args.dataset
    data_config["root"] = args.dataset_root
    data_config["annotation_file"] = args.annotation_file
    data_config["split"] = args.dataset_split
    data_config["subset"] = args.limit
    if args.query_mode is not None:
        data_config["query_mode"] = args.query_mode
    if args.resolution is not None:
        data_config["resolution"] = [int(args.resolution[0]), int(args.resolution[1])]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_state = checkpoint["model"]

    # The baseline model never receives a Route-D selector. The Route-D model is
    # a separate instance with the same checkpoint, so comparisons are causal
    # and do not accidentally compare a model against itself.
    baseline_device = torch.device(args.baseline_device if args.baseline_device else str(device))
    if baseline_device.type == "cuda" and not torch.cuda.is_available():
        baseline_device = device
    baseline_model = load_base_model(config, checkpoint_state, baseline_device)
    route_model = None
    open_loop_selector = load_routeD_selector(
        args.selector_bundle, device=device, threshold=args.threshold
    )
    if args.closed_loop:
        route_model = load_base_model(config, checkpoint_state, device)
        route_model.attach_routeD_selector(
            args.selector_bundle,
            device=device,
            p1_tolerance=float(args.profile_p1_tolerance or 0.0),
            min_coarse_gain=args.profile_min_coarse_gain,
            min_total_gain=args.profile_min_total_gain,
            threshold=args.threshold,
            update_memory=True,
        )

    dataset, _ = resolve_dataset(config, args.split, False)
    # Model construction and selector loading consume random numbers. Reset here
    # so stochastic query sampling is identical across open/closed evaluations.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    prediction_names = (
        ("baseline", "local", "routeD_open", "routeD_closed")
        if args.closed_loop
        else ("baseline", "local", "routeD")
    )
    metric_accumulator = {
        name: {"AJ": [], "OA": [], "delta_avg": []}
        for name in prediction_names
    }
    diagnostic_keys = (
        "global_selection_rate",
        "trajectory_changed_rate",
        "mean_trajectory_diff_px",
        "max_trajectory_diff_px",
        "mean_prior_diff_px",
        "mean_state_diff_px",
        "memory_write_disagreement_rate",
        "mean_memory_write_position_diff_px",
        "open_global_selection_rate",
        "open_trajectory_changed_rate",
        "open_mean_trajectory_diff_px",
        "mean_closed_vs_open_diff_px",
    )
    diagnostic_accumulator = {key: [] for key in diagnostic_keys}
    rows = []

    with torch.no_grad():
        for sample_index, batch in enumerate(loader):
            if sample_index >= args.limit:
                break
            video = batch["video"].to(device)
            query = batch["query_points"].to(device)
            target = batch["target_points"].to(device)
            occluded = batch["occluded"].to(device).bool()

            if next(baseline_model.parameters()).device != video.device:
                baseline_video = video.to(next(baseline_model.parameters()).device)
                baseline_query = query.to(next(baseline_model.parameters()).device)
            else:
                baseline_video = video
                baseline_query = query
            baseline_tracks_cpu, baseline_visibility_cpu, baseline_info_cpu = baseline_model(
                baseline_video, baseline_query, return_info=True
            )
            baseline_tracks = baseline_tracks_cpu.to(device)
            baseline_visibility = baseline_visibility_cpu.to(device)
            baseline_info = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in baseline_info_cpu.items()}
            del baseline_tracks_cpu, baseline_visibility_cpu, baseline_info_cpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            candidate_points = baseline_info["hypothesis_candidate_points"]
            local_tracks = candidate_points[..., 0, :]
            active = active_after_query(query, baseline_tracks.shape[-2])
            local_tracks = torch.where(
                active.unsqueeze(-1), local_tracks, baseline_tracks
            )

            scale = torch.tensor(
                [video.shape[-2] - 1, video.shape[-1] - 1],
                device=device,
                dtype=baseline_tracks.dtype,
            )

            if args.closed_loop:
                candidate_quality = baseline_info["hypothesis_candidate_quality"]
                candidate_entropy = baseline_info["hypothesis_candidate_entropy"]
                previous_points = baseline_info["hypothesis_previous_points"]
                previous_confidence = baseline_info[
                    "hypothesis_previous_confidence"
                ]
                features = build_hypothesis_features(
                    candidate_points,
                    candidate_quality,
                    candidate_points[..., 0, :],
                    previous_points,
                    previous_confidence,
                    candidate_entropy,
                )
                valid = torch.isfinite(candidate_points).all(dim=-1)
                open_output = open_loop_selector.select(
                    features,
                    candidate_points,
                    valid,
                    fallback_points=baseline_tracks,
                    max_switch_distance_px=args.max_switch_distance_px,
                    pixel_scale=(
                        scale if args.max_switch_distance_px is not None else None
                    ),
                    fusion_strength=args.fusion_strength,
                    profile_p1_tolerance=args.profile_p1_tolerance,
                    profile_min_coarse_gain=args.profile_min_coarse_gain,
                    profile_min_total_gain=args.profile_min_total_gain,
                )
                open_tracks = torch.where(
                    active.unsqueeze(-1), open_output["points"], baseline_tracks
                )
                open_selected_global = (open_output["index"] > 0) & active
                open_diff_px = torch.norm(
                    (open_tracks - baseline_tracks) * scale, dim=-1
                )

                route_tracks, route_visibility, route_info = route_model(
                    video, query, return_info=True
                )
                route_selected_global = route_info["routeD_selected_global_mask"] & active
                trajectory_diff_px = torch.norm(
                    (route_tracks - baseline_tracks) * scale, dim=-1
                )
                closed_vs_open_diff_px = torch.norm(
                    (route_tracks - open_tracks) * scale, dim=-1
                )
                prior_diff_px = torch.norm(
                    (route_info["prior_points"] - baseline_info["prior_points"])
                    * scale,
                    dim=-1,
                )
                state_diff_px = torch.norm(
                    (route_info["state_points"] - baseline_info["state_points"])
                    * scale,
                    dim=-1,
                )
                route_write = route_info["memory_write_mask"]
                baseline_write = baseline_info["memory_write_mask"]
                memory_disagreement = (route_write != baseline_write) & active
                write_union = (route_write | baseline_write) & active
                memory_position_diff_px = torch.norm(
                    (route_info["state_points"] - baseline_info["state_points"])
                    * scale,
                    dim=-1,
                )
                row_diagnostics = {
                    "global_selection_rate": masked_mean(
                        route_selected_global.float(), active
                    ),
                    "trajectory_changed_rate": masked_mean(
                        (trajectory_diff_px > 1.0e-4).float(), active
                    ),
                    "mean_trajectory_diff_px": masked_mean(
                        trajectory_diff_px, active
                    ),
                    "max_trajectory_diff_px": float(
                        trajectory_diff_px.masked_fill(~active, 0.0).max()
                    ),
                    "mean_prior_diff_px": masked_mean(prior_diff_px, active),
                    "mean_state_diff_px": masked_mean(state_diff_px, active),
                    "memory_write_disagreement_rate": masked_mean(
                        memory_disagreement.float(), active
                    ),
                    "mean_memory_write_position_diff_px": masked_mean(
                        memory_position_diff_px, write_union
                    ),
                    "open_global_selection_rate": masked_mean(
                        open_selected_global.float(), active
                    ),
                    "open_trajectory_changed_rate": masked_mean(
                        (open_diff_px > 1.0e-4).float(), active
                    ),
                    "open_mean_trajectory_diff_px": masked_mean(
                        open_diff_px, active
                    ),
                    "mean_closed_vs_open_diff_px": masked_mean(
                        closed_vs_open_diff_px, active
                    ),
                }
            else:
                candidate_quality = baseline_info["hypothesis_candidate_quality"]
                candidate_entropy = baseline_info["hypothesis_candidate_entropy"]
                previous_points = baseline_info["hypothesis_previous_points"]
                previous_confidence = baseline_info[
                    "hypothesis_previous_confidence"
                ]
                features = build_hypothesis_features(
                    candidate_points,
                    candidate_quality,
                    candidate_points[..., 0, :],
                    previous_points,
                    previous_confidence,
                    candidate_entropy,
                )
                valid = torch.isfinite(candidate_points).all(dim=-1)
                output = open_loop_selector.select(
                    features,
                    candidate_points,
                    valid,
                    fallback_points=baseline_tracks,
                    max_switch_distance_px=args.max_switch_distance_px,
                    pixel_scale=(
                        scale if args.max_switch_distance_px is not None else None
                    ),
                    fusion_strength=args.fusion_strength,
                    profile_p1_tolerance=args.profile_p1_tolerance,
                    profile_min_coarse_gain=args.profile_min_coarse_gain,
                    profile_min_total_gain=args.profile_min_total_gain,
                )
                route_tracks = torch.where(
                    active.unsqueeze(-1), output["points"], baseline_tracks
                )
                route_visibility = baseline_visibility
                route_selected_global = (output["index"] > 0) & active
                trajectory_diff_px = torch.norm(
                    (route_tracks - baseline_tracks) * scale, dim=-1
                )
                row_diagnostics = {
                    "global_selection_rate": masked_mean(
                        route_selected_global.float(), active
                    ),
                    "trajectory_changed_rate": masked_mean(
                        (trajectory_diff_px > 1.0e-4).float(), active
                    ),
                    "mean_trajectory_diff_px": masked_mean(
                        trajectory_diff_px, active
                    ),
                    "max_trajectory_diff_px": float(
                        trajectory_diff_px.masked_fill(~active, 0.0).max()
                    ),
                    "mean_prior_diff_px": 0.0,
                    "mean_state_diff_px": 0.0,
                    "memory_write_disagreement_rate": 0.0,
                    "mean_memory_write_position_diff_px": 0.0,
                    "open_global_selection_rate": masked_mean(
                        route_selected_global.float(), active
                    ),
                    "open_trajectory_changed_rate": masked_mean(
                        (trajectory_diff_px > 1.0e-4).float(), active
                    ),
                    "open_mean_trajectory_diff_px": masked_mean(
                        trajectory_diff_px, active
                    ),
                    "mean_closed_vs_open_diff_px": 0.0,
                }

            query_mode = resolve_eval_query_mode(batch, dataset)
            resolution = (
                (int(args.metric_resolution[0]), int(args.metric_resolution[1]))
                if args.metric_resolution is not None
                else resolve_eval_resolution(batch, video, 0)
            )
            row = {
                "sample": sample_index,
                "video_name": resolve_video_name(batch, sample_index),
                **row_diagnostics,
            }
            if args.closed_loop:
                predictions = (
                    ("baseline", baseline_tracks, baseline_visibility),
                    ("local", local_tracks, baseline_visibility),
                    ("routeD_open", open_tracks, baseline_visibility),
                    ("routeD_closed", route_tracks, route_visibility),
                )
            else:
                predictions = (
                    ("baseline", baseline_tracks, baseline_visibility),
                    ("local", local_tracks, baseline_visibility),
                    ("routeD", route_tracks, route_visibility),
                )
            for name, tracks, visibility in predictions:
                metrics = normalize_metrics(
                    compute_tapvid_metrics(
                        tracks[0],
                        target[0],
                        visibility[0],
                        ~occluded[0],
                        query[0],
                        resolution=resolution,
                        exclude_query_frame=True,
                        query_mode=query_mode,
                    )
                )
                row[name] = metrics
                for key, value in metrics.items():
                    metric_accumulator[name][key].append(value)
            for key in diagnostic_keys:
                diagnostic_accumulator[key].append(row_diagnostics[key])
            rows.append(row)
            print(sample_index, row)

    aggregate = {
        name: {key: finite_mean(values) for key, values in metrics.items()}
        for name, metrics in metric_accumulator.items()
    }
    if args.closed_loop:
        delta = {
            "closed_vs_baseline": {
                key: aggregate["routeD_closed"][key] - aggregate["baseline"][key]
                for key in aggregate["baseline"]
            },
            "open_vs_baseline": {
                key: aggregate["routeD_open"][key] - aggregate["baseline"][key]
                for key in aggregate["baseline"]
            },
            "closed_vs_open": {
                key: aggregate["routeD_closed"][key] - aggregate["routeD_open"][key]
                for key in aggregate["baseline"]
            },
        }
    else:
        delta = {
            key: aggregate["routeD"][key] - aggregate["baseline"][key]
            for key in aggregate["baseline"]
        }
    aggregate_diagnostics = {
        key: finite_mean(values) for key, values in diagnostic_accumulator.items()
    }
    result = {
        "evidence_tier": "development_diagnostic_only",
        "paper_claim_eligible": False,
        "evaluation_mode": (
            "closed_loop_routeD" if args.closed_loop else "open_loop_posthoc_selector"
        ),
        "closed_loop_state_updated": bool(args.closed_loop),
        "independent_baseline_model": True,
        "baseline_has_routeD_selector": False,
        "threshold": args.threshold,
        "seed": args.seed,
        "dataset": args.dataset or data_config.get("dataset"),
        "dataset_root": str(args.dataset_root),
        "query_mode": getattr(dataset, "query_mode", args.query_mode),
        "input_resolution": list(video.shape[-2:]) if rows else (list(args.resolution) if args.resolution else None),
        "metric_resolution": list(args.metric_resolution) if args.metric_resolution is not None else None,
        "fusion_strength": args.fusion_strength,
        "max_switch_distance_px": args.max_switch_distance_px,
        "profile_p1_tolerance": args.profile_p1_tolerance,
        "profile_min_coarse_gain": args.profile_min_coarse_gain,
        "profile_min_total_gain": args.profile_min_total_gain,
        "samples": len(rows),
        "aggregate": aggregate,
        "aggregate_diagnostics": aggregate_diagnostics,
        "delta_routeD_vs_baseline": delta,
        "per_sample": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "per_sample"}, indent=2))


if __name__ == "__main__":
    main()
