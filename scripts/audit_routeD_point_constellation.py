#!/usr/bin/env python3
"""Audit whether local feature constellations improve point re-identification.

The audit is deliberately non-learned.  Candidate generation is the exact
zero-step multi-anchor proposal from safe re-detection v0.  Ground truth is
used only for fit-role weight selection and metric computation; no GT quantity
enters any candidate score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from mmp_tracker.routeD_point_constellation import (
    POINT_CONSTELLATION_SCHEMA_VERSION,
    PointConstellationConfig,
    candidate_distance_utility,
    combined_constellation_score,
    direct_correspondence_similarity,
    sample_point_constellations,
    structural_self_similarity,
)
from mmp_tracker.routeD_safe_redetection import (
    RedetectionStepState,
    SafeRedetectionConfig,
    SafeRedetectionModel,
    initialize_appearance_memory,
    initialize_confirmation_state,
    sample_batched_map,
)
from mmp_tracker.routeD_safe_redetection_cache import verify_event_cache


WEIGHT_GRID: tuple[tuple[float, float], ...] = tuple(
    (direct, structure)
    for direct in (0.0, 0.25, 0.5, 1.0)
    for structure in (0.0, 0.25, 0.5, 1.0)
    if direct > 0.0 or structure > 0.0
)
THRESHOLDS = (1.0, 2.0, 4.0, 8.0, 16.0)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_index(path: str | Path, role: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("role") != role or not payload.get("complete"):
        raise ValueError(f"index is not a complete {role} cache")
    if int(payload.get("completed_count", -1)) != len(payload.get("events", [])):
        raise ValueError("index completed_count does not match events")
    return payload


def load_event(
    row: Mapping[str, Any],
    *,
    protocol_sha256: str,
    role: str,
    verify_hashes: bool,
) -> dict[str, Any]:
    if verify_hashes:
        return verify_event_cache(
            row["sidecar"],
            expected_protocol_sha256=protocol_sha256,
            expected_role=role,
        )
    payload = torch.load(row["sidecar"], map_location="cpu", weights_only=False)
    if payload.get("provenance", {}).get("protocol_sha256") != protocol_sha256:
        raise ValueError("event protocol hash mismatch")
    if payload.get("provenance", {}).get("role") != role:
        raise ValueError("event role mismatch")
    return payload


def initialize_state(
    model: SafeRedetectionModel,
    tensors: Mapping[str, torch.Tensor],
    *,
    device: str,
) -> tuple[RedetectionStepState, int, torch.Tensor]:
    frame_rel = tensors["frame_indices_rel"].long()
    query_rel = int(tensors["query_frame_rel"].item())
    match = torch.where(frame_rel == query_rel)[0]
    if match.numel() != 1:
        raise ValueError("query frame must appear exactly once")
    row = int(match.item())
    fmap = tensors["feature_maps_f16"][row : row + 1].to(device=device, dtype=torch.float32)
    coord = tensors["query_coord_xy_px"].to(device=device, dtype=torch.float32).view(1, 1, 2)
    query_feature = sample_batched_map(fmap, coord, model.config)[:, 0]
    state = RedetectionStepState(
        initialize_appearance_memory(
            query_feature,
            torch.tensor([query_rel], device=device),
            model.config,
        ),
        initialize_confirmation_state(1, device=device),
    )
    return state, row, fmap


def event_rows(
    payload: Mapping[str, Any],
    model: SafeRedetectionModel,
    constellation_config: PointConstellationConfig,
    *,
    device: str,
) -> list[dict[str, Any]]:
    tensors = payload["tensors"]
    state, _, query_map = initialize_state(model, tensors, device=device)
    query_coord = tensors["query_coord_xy_px"].to(device=device, dtype=torch.float32).view(1, 1, 2)
    reference = sample_point_constellations(query_map, query_coord, constellation_config)[:, 0]
    frame_rel = tensors["frame_indices_rel"].long()
    frame_abs = tensors["frame_indices_abs"].long()
    query_rel = int(tensors["query_frame_rel"].item())
    reentry_abs = int(payload["provenance"]["event"]["reappearance_frame"])
    outputs: list[dict[str, Any]] = []
    previous_rel: int | None = None
    with torch.no_grad():
        for row_id in range(frame_rel.numel()):
            current_rel = int(frame_rel[row_id].item())
            if current_rel < query_rel:
                continue
            if previous_rel is not None and current_rel != previous_rel + 1:
                state = RedetectionStepState(
                    state.memory,
                    initialize_confirmation_state(1, device=device),
                )
            fmap = tensors["feature_maps_f16"][row_id : row_id + 1].to(
                device=device, dtype=torch.float32
            )
            native = tensors["native_coords_xy_px"][row_id : row_id + 1].to(
                device=device, dtype=torch.float32
            )
            result, state = model.step(
                frame_index=current_rel,
                feature_map=fmap,
                native_coord_xy_px=native,
                native_visibility_probability=tensors["native_visibility_probability"][
                    row_id : row_id + 1
                ].to(device=device, dtype=torch.float32),
                native_confidence_probability=tensors["native_confidence_probability"][
                    row_id : row_id + 1
                ].to(device=device, dtype=torch.float32),
                occlusion_age=tensors["occlusion_age"][row_id : row_id + 1].to(device),
                state=state,
            )
            current_abs = int(frame_abs[row_id].item())
            visible = bool(tensors["gt_visible"][row_id].item())
            if current_abs >= reentry_abs and visible:
                coords = result["candidate_coords_xy_px"].float()
                valid = result["candidate_valid_mask"].bool()
                center = result["candidate_scores"].float().masked_fill(~valid, float("-inf"))
                constellations = sample_point_constellations(fmap, coords, constellation_config)
                direct = direct_correspondence_similarity(
                    reference,
                    constellations,
                    exclude_center=constellation_config.exclude_center_from_direct,
                ).masked_fill(~valid, float("-inf"))
                structure = structural_self_similarity(reference, constellations).masked_fill(
                    ~valid, float("-inf")
                )
                gt = tensors["gt_coords_xy_px"][row_id : row_id + 1].to(
                    device=device, dtype=torch.float32
                )
                utility = candidate_distance_utility(coords, gt)[0]
                distance = torch.linalg.norm(coords[0] - gt[0], dim=-1)
                outputs.append(
                    {
                        "frame_abs": current_abs,
                        "center": center[0].cpu(),
                        "direct": direct[0].cpu(),
                        "structure": structure[0].cpu(),
                        "utility": utility.cpu(),
                        "distance": distance.cpu(),
                        "valid": valid[0].cpu(),
                    }
                )
            previous_rel = current_rel
    if not outputs:
        raise RuntimeError("event has no visible post-reappearance audit rows")
    return outputs


def candidate_score(row: Mapping[str, torch.Tensor], weight: tuple[float, float] | None) -> torch.Tensor:
    if weight is None:
        score = row["center"].clone()
    else:
        score = combined_constellation_score(
            row["center"][None],
            row["direct"][None],
            row["structure"][None],
            direct_weight=weight[0],
            structure_weight=weight[1],
        )[0]
    return score.masked_fill(~row["valid"], float("-inf"))


def select_index(
    row: Mapping[str, torch.Tensor],
    weight: tuple[float, float] | None,
    *,
    allow_native: bool,
) -> int:
    score = candidate_score(row, weight)
    if not allow_native:
        score = score.clone()
        score[0] = float("-inf")
        if not torch.isfinite(score).any():
            return 0
    return int(score.argmax().item())


def summarize_event(
    rows: list[dict[str, Any]],
    weight: tuple[float, float] | None,
) -> dict[str, float]:
    utilities: list[float] = []
    distances: list[float] = []
    oracle: list[float] = []
    recoverable_rows = 0
    recoverable_nonnative_utility_sum = 0.0
    recoverable_oracle_nonnative_utility_sum = 0.0
    beneficial_nonnative_rows = 0
    intervention_rows = 0
    harmful_intervention_rows = 0
    for row in rows:
        index = select_index(row, weight, allow_native=True)
        nonnative_index = select_index(row, weight, allow_native=False)
        native_utility = float(row["utility"][0].item())
        selected_utility = float(row["utility"][index].item())
        nonnative_utility = float(row["utility"][nonnative_index].item())
        valid_non_native = row["valid"].clone()
        valid_non_native[0] = False
        if bool(valid_non_native.any()):
            oracle_non_native = float(
                row["utility"].masked_fill(~valid_non_native, -1.0).max().item()
            )
        else:
            oracle_non_native = native_utility
        recoverable = oracle_non_native > native_utility + 1e-6
        if recoverable:
            recoverable_rows += 1
            recoverable_nonnative_utility_sum += nonnative_utility
            recoverable_oracle_nonnative_utility_sum += oracle_non_native
            beneficial_nonnative_rows += int(nonnative_utility > native_utility + 1e-6)
        intervention = index > 0
        intervention_rows += int(intervention)
        harmful_intervention_rows += int(
            intervention and selected_utility + 1e-6 < native_utility
        )
        utilities.append(selected_utility)
        distances.append(float(row["distance"][index].item()))
        oracle.append(float(row["utility"].masked_fill(~row["valid"], -1.0).max().item()))
    result = {
        "rows": float(len(rows)),
        "utility": float(np.mean(utilities)),
        "oracle_utility": float(np.mean(oracle)),
        "mean_distance_px": float(np.mean(distances)),
        "recoverable_rows": float(recoverable_rows),
        "recoverable_nonnative_utility_sum": float(recoverable_nonnative_utility_sum),
        "recoverable_oracle_nonnative_utility_sum": float(
            recoverable_oracle_nonnative_utility_sum
        ),
        "beneficial_nonnative_rows": float(beneficial_nonnative_rows),
        "intervention_rows": float(intervention_rows),
        "harmful_intervention_rows": float(harmful_intervention_rows),
    }
    result["recoverable_nonnative_utility"] = (
        recoverable_nonnative_utility_sum / recoverable_rows if recoverable_rows else 0.0
    )
    result["beneficial_nonnative_recall"] = (
        beneficial_nonnative_rows / recoverable_rows if recoverable_rows else 0.0
    )
    result["harmful_intervention_rate"] = (
        harmful_intervention_rows / intervention_rows if intervention_rows else 0.0
    )
    for threshold in THRESHOLDS:
        result[f"hit_{int(threshold)}px"] = float(np.mean(np.asarray(distances) <= threshold))
    return result



def summarize_native_event(rows: list[dict[str, Any]]) -> dict[str, float]:
    utilities: list[float] = []
    distances: list[float] = []
    recoverable_rows = 0
    recoverable_native_utility_sum = 0.0
    for row in rows:
        native_utility = float(row["utility"][0].item())
        valid_non_native = row["valid"].clone()
        valid_non_native[0] = False
        oracle_non_native = (
            float(row["utility"].masked_fill(~valid_non_native, -1.0).max().item())
            if bool(valid_non_native.any())
            else native_utility
        )
        if oracle_non_native > native_utility + 1e-6:
            recoverable_rows += 1
            recoverable_native_utility_sum += native_utility
        utilities.append(native_utility)
        distances.append(float(row["distance"][0].item()))
    result = {
        "rows": float(len(rows)),
        "utility": float(np.mean(utilities)),
        "oracle_utility": float(np.mean(utilities)),
        "mean_distance_px": float(np.mean(distances)),
        "recoverable_rows": float(recoverable_rows),
        "recoverable_nonnative_utility_sum": float(recoverable_native_utility_sum),
        "recoverable_oracle_nonnative_utility_sum": float(recoverable_native_utility_sum),
        "beneficial_nonnative_rows": 0.0,
        "intervention_rows": 0.0,
        "harmful_intervention_rows": 0.0,
        "recoverable_nonnative_utility": (
            recoverable_native_utility_sum / recoverable_rows if recoverable_rows else 0.0
        ),
        "beneficial_nonnative_recall": 0.0,
        "harmful_intervention_rate": 0.0,
    }
    for threshold in THRESHOLDS:
        result[f"hit_{int(threshold)}px"] = float(
            np.mean(np.asarray(distances) <= threshold)
        )
    return result

def prepare_role(
    index: Mapping[str, Any],
    *,
    role: str,
    model: SafeRedetectionModel,
    constellation_config: PointConstellationConfig,
    device: str,
    verify_hashes: bool,
    max_events: int,
) -> list[dict[str, Any]]:
    output = []
    all_events = list(index["events"])
    if max_events > 0:
        # Deterministic two-axis smoke sampling: rotate across duration buckets,
        # and within every bucket rotate across scenes before taking a second
        # event from the same scene.
        grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for event_row in all_events:
            grouped[str(event_row["duration_bucket"])][str(event_row["scene"])].append(
                event_row
            )
        bucket_order = sorted(grouped)
        scene_order = {bucket: sorted(grouped[bucket]) for bucket in bucket_order}
        scene_cursor = {bucket: 0 for bucket in bucket_order}
        row_cursor: dict[tuple[str, str], int] = defaultdict(int)
        events = []
        while len(events) < min(max_events, len(all_events)):
            progressed = False
            for bucket in bucket_order:
                scenes = scene_order[bucket]
                for _ in range(len(scenes)):
                    scene = scenes[scene_cursor[bucket] % len(scenes)]
                    scene_cursor[bucket] += 1
                    offset = row_cursor[(bucket, scene)]
                    if offset < len(grouped[bucket][scene]):
                        events.append(grouped[bucket][scene][offset])
                        row_cursor[(bucket, scene)] += 1
                        progressed = True
                        break
                if len(events) >= max_events:
                    break
            if not progressed:
                break
    else:
        events = all_events
    for event_id, row in enumerate(events):
        payload = load_event(
            row,
            protocol_sha256=index["protocol_sha256"],
            role=role,
            verify_hashes=verify_hashes,
        )
        audit_rows = event_rows(
            payload,
            model,
            constellation_config,
            device=device,
        )
        event = payload["provenance"]["event"]
        methods: dict[str, dict[str, float]] = {
            "center": summarize_event(audit_rows, None),
        }
        for weight in WEIGHT_GRID:
            methods[f"d{weight[0]:g}_s{weight[1]:g}"] = summarize_event(audit_rows, weight)
        # Candidate zero is the exact native tracker.
        methods["native"] = summarize_native_event(audit_rows)
        output.append(
            {
                "event_id": event_id,
                "event_identity_sha256": event["event_identity_sha256"],
                "scene": event["scene"],
                "duration_bucket": event["duration_bucket"],
                "invisibility_duration": int(event["invisibility_duration"]),
                "rows": len(audit_rows),
                "methods": methods,
            }
        )
        print(
            json.dumps(
                {
                    "role": role,
                    "event": event_id,
                    "scene": event["scene"],
                    "bucket": event["duration_bucket"],
                    "rows": len(audit_rows),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return output


def weight_key(weight: tuple[float, float]) -> str:
    return f"d{weight[0]:g}_s{weight[1]:g}"


def _group_recoverable_gain(
    events: list[Mapping[str, Any]],
    method: str,
    group_key: str,
) -> dict[str, float]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event[group_key])].append(event)
    output: dict[str, float] = {}
    for key, rows in sorted(grouped.items()):
        selected_sum = sum(
            float(row["methods"][method]["recoverable_nonnative_utility_sum"])
            for row in rows
        )
        center_sum = sum(
            float(row["methods"]["center"]["recoverable_nonnative_utility_sum"])
            for row in rows
        )
        count = sum(float(row["methods"][method]["recoverable_rows"]) for row in rows)
        if count > 0:
            output[key] = (selected_sum - center_sum) / count
    return output


def aggregate(events: Iterable[Mapping[str, Any]], method: str) -> dict[str, Any]:
    rows = list(events)
    utility = np.asarray([row["methods"][method]["utility"] for row in rows], dtype=np.float64)
    center = np.asarray([row["methods"]["center"]["utility"] for row in rows], dtype=np.float64)
    native = np.asarray([row["methods"]["native"]["utility"] for row in rows], dtype=np.float64)
    gains = utility - center
    recoverable_rows = sum(float(row["methods"][method]["recoverable_rows"]) for row in rows)
    selected_recoverable_sum = sum(
        float(row["methods"][method]["recoverable_nonnative_utility_sum"])
        for row in rows
    )
    center_recoverable_sum = sum(
        float(row["methods"]["center"]["recoverable_nonnative_utility_sum"])
        for row in rows
    )
    beneficial_rows = sum(
        float(row["methods"][method]["beneficial_nonnative_rows"]) for row in rows
    )
    center_beneficial_rows = sum(
        float(row["methods"]["center"]["beneficial_nonnative_rows"]) for row in rows
    )
    interventions = sum(float(row["methods"][method]["intervention_rows"]) for row in rows)
    harmful = sum(float(row["methods"][method]["harmful_intervention_rows"]) for row in rows)
    return {
        "events": len(rows),
        "utility": float(utility.mean()),
        "gain_vs_center": float(gains.mean()),
        "gain_vs_native": float((utility - native).mean()),
        "positive_events_vs_center": int((gains > 0).sum()),
        "positive_event_fraction_vs_center": float((gains > 0).mean()),
        "recoverable_rows": int(recoverable_rows),
        "recoverable_nonnative_utility": (
            selected_recoverable_sum / recoverable_rows if recoverable_rows else 0.0
        ),
        "recoverable_nonnative_utility_gain_vs_center": (
            (selected_recoverable_sum - center_recoverable_sum) / recoverable_rows
            if recoverable_rows
            else 0.0
        ),
        "beneficial_nonnative_recall": (
            beneficial_rows / recoverable_rows if recoverable_rows else 0.0
        ),
        "beneficial_nonnative_recall_gain_vs_center": (
            (beneficial_rows - center_beneficial_rows) / recoverable_rows
            if recoverable_rows
            else 0.0
        ),
        "intervention_rows": int(interventions),
        "harmful_intervention_rows": int(harmful),
        "harmful_intervention_rate": harmful / interventions if interventions else 0.0,
        "scene_recoverable_gain": _group_recoverable_gain(rows, method, "scene"),
        "bucket_recoverable_gain": _group_recoverable_gain(
            rows, method, "duration_bucket"
        ),
        "mean_distance_px": float(
            np.mean([row["methods"][method]["mean_distance_px"] for row in rows])
        ),
        **{
            f"hit_{int(threshold)}px": float(
                np.mean([row["methods"][method][f"hit_{int(threshold)}px"] for row in rows])
            )
            for threshold in THRESHOLDS
        },
    }


def choose_weight(events: list[dict[str, Any]]) -> tuple[float, float]:
    """Select only on fit-role recoverable rows; lower-complexity ties win."""
    candidates = []
    for weight in WEIGHT_GRID:
        summary = aggregate(events, weight_key(weight))
        scene_values = np.asarray(
            list(summary["scene_recoverable_gain"].values()), dtype=np.float64
        )
        bucket_values = np.asarray(
            list(summary["bucket_recoverable_gain"].values()), dtype=np.float64
        )
        if scene_values.size == 0 or bucket_values.size == 0:
            criterion = float("-inf")
        else:
            criterion = (
                float(scene_values.mean())
                - 0.5 * float(scene_values.std(ddof=0))
                + min(0.0, float(bucket_values.min()))
                + 0.1 * float(summary["beneficial_nonnative_recall_gain_vs_center"])
            )
        candidates.append((criterion, -(weight[0] + weight[1]), -weight[1], weight))
    return max(candidates)[-1]

def fit_scene_loso(events: list[dict[str, Any]]) -> dict[str, Any]:
    scenes = sorted({row["scene"] for row in events})
    heldout = []
    for scene in scenes:
        train = [row for row in events if row["scene"] != scene]
        test = [row for row in events if row["scene"] == scene]
        weight = choose_weight(train)
        summary = aggregate(test, weight_key(weight))
        heldout.append(
            {
                "scene": scene,
                "weight": list(weight),
                "events": len(test),
                "gain_vs_center": summary["gain_vs_center"],
                "recoverable_gain_vs_center": summary[
                    "recoverable_nonnative_utility_gain_vs_center"
                ],
            }
        )
    gains = np.asarray(
        [row["recoverable_gain_vs_center"] for row in heldout], dtype=np.float64
    )
    return {
        "folds": heldout,
        "mean_scene_recoverable_gain_vs_center": float(gains.mean()),
        "positive_scenes": int((gains > 0).sum()),
        "scene_count": len(heldout),
    }


def bootstrap_ci(values: np.ndarray, *, samples: int, seed: int) -> dict[str, float | int]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap values must be non-empty and one-dimensional")
    generator = np.random.default_rng(seed)
    draw = generator.integers(0, values.size, size=(samples, values.size))
    means = values[draw].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "samples": int(samples),
        "seed": int(seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-index", required=True)
    parser.add_argument("--validation-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-fit-events", type=int, default=0)
    parser.add_argument("--max-validation-events", type=int, default=0)
    parser.add_argument("--skip-sidecar-hash-verification", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    fit_index = load_index(args.fit_index, "fit")
    validation_index = load_index(args.validation_index, "model_validation")
    if fit_index["protocol_sha256"] != validation_index["protocol_sha256"]:
        raise ValueError("fit and validation protocol hashes differ")

    model_config = SafeRedetectionConfig()
    constellation_config = PointConstellationConfig(
        input_height=model_config.input_height,
        input_width=model_config.input_width,
    )
    model = SafeRedetectionModel(model_config).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    fit_events = prepare_role(
        fit_index,
        role="fit",
        model=model,
        constellation_config=constellation_config,
        device=args.device,
        verify_hashes=not args.skip_sidecar_hash_verification,
        max_events=args.max_fit_events,
    )
    selected_weight = choose_weight(fit_events)
    fit_loso = fit_scene_loso(fit_events) if len({row["scene"] for row in fit_events}) > 1 else None

    validation_events = prepare_role(
        validation_index,
        role="model_validation",
        model=model,
        constellation_config=constellation_config,
        device=args.device,
        verify_hashes=not args.skip_sidecar_hash_verification,
        max_events=args.max_validation_events,
    )
    method = weight_key(selected_weight)
    validation_summary = aggregate(validation_events, method)
    center_summary = aggregate(validation_events, "center")
    native_summary = aggregate(validation_events, "native")
    recoverable_event_gains = []
    for row in validation_events:
        selected = row["methods"][method]
        center = row["methods"]["center"]
        count = float(selected["recoverable_rows"])
        if count > 0:
            recoverable_event_gains.append(
                (
                    float(selected["recoverable_nonnative_utility_sum"])
                    - float(center["recoverable_nonnative_utility_sum"])
                )
                / count
            )
    if not recoverable_event_gains:
        raise RuntimeError("model-validation contains no recoverable event rows")
    ci = bootstrap_ci(
        np.asarray(recoverable_event_gains, dtype=np.float64),
        samples=args.bootstrap_samples,
        seed=args.seed + 999,
    )
    full_scale = args.max_fit_events == 0 and args.max_validation_events == 0
    long_bucket_keys = ("d16_63", "d64_255", "d256_plus")
    long_bucket = validation_summary["bucket_recoverable_gain"]
    long_values = [long_bucket[key] for key in long_bucket_keys if key in long_bucket]
    gate = {
        "full_scale": full_scale,
        "recoverable_utility_gain_vs_center_ge_0p03": validation_summary[
            "recoverable_nonnative_utility_gain_vs_center"
        ] >= 0.03,
        "beneficial_recall_gain_vs_center_ge_0p10": validation_summary[
            "beneficial_nonnative_recall_gain_vs_center"
        ] >= 0.10,
        "paired_recoverable_event_CI_lower_positive": ci["lower"] > 0.0,
        "no_long_duration_bucket_regression": len(long_values) == 3
        and min(long_values) >= 0.0,
    }
    gate["pass"] = bool(full_scale and all(value for key, value in gate.items() if key != "pass"))
    gate["decision"] = (
        "ALLOW_TRACKLET_CONSTELLATION_STAGE"
        if gate["pass"]
        else "STOP_OR_REDESIGN_CONSTELLATION_BEFORE_TRACKLET_STAGE"
    )

    output = {
        "schema_version": "routeD_point_constellation_feasibility_v0",
        "constellation_schema_version": POINT_CONSTELLATION_SCHEMA_VERSION,
        "protocol_sha256": fit_index["protocol_sha256"],
        "source": {
            "fit_index": str(Path(args.fit_index).resolve()),
            "fit_index_sha256": file_sha256(args.fit_index),
            "validation_index": str(Path(args.validation_index).resolve()),
            "validation_index_sha256": file_sha256(args.validation_index),
            "sidecar_hash_verification": not args.skip_sidecar_hash_verification,
        },
        "ground_truth_boundary": {
            "candidate_score": "GT-free; fixed zero-step proposal plus query-frame local constellation",
            "fit_use": "fit-role GT selects two non-negative evidence weights from a preregistered grid",
            "validation_use": "metrics only; selected weights are frozen before model-validation evaluation",
            "locked_data_read": {
                "internal_holdout": False,
                "pointodyssey_test": False,
                "kinetics_1144": False,
            },
        },
        "configuration": {
            "seed": args.seed,
            "device": args.device,
            "offsets_xy_px": [list(row) for row in constellation_config.offsets_xy_px],
            "weight_grid": [list(row) for row in WEIGHT_GRID],
            "selected_weight": {
                "direct": selected_weight[0],
                "structure": selected_weight[1],
                "method_key": method,
            },
        },
        "fit": {
            "events": len(fit_events),
            "selected_method": aggregate(fit_events, method),
            "center": aggregate(fit_events, "center"),
            "native": aggregate(fit_events, "native"),
            "scene_loso": fit_loso,
        },
        "model_validation": {
            "events": len(validation_events),
            "selected_method": validation_summary,
            "center": center_summary,
            "native": native_summary,
            "paired_recoverable_event_gain_CI": ci,
        },
        "gate": gate,
        "event_rows": {
            "fit": fit_events,
            "model_validation": validation_events,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_path), "gate": gate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
