#!/usr/bin/env python3
"""Post-gate candidate-capacity diagnosis for completed Route-D Gate 3A v1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer_training import (
    load_csrr_cache_video,
)
from projects.mmp_tracker.mmp_tracker.routeD_discrete_candidates import (
    extract_discrete_candidates,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    canonical_json_sha256,
    file_sha256,
)
from scripts.audit_routeD_cotracker3_interface import (
    load_manifest_sample,
    prepare_sample,
)


REPRESENTATIONS = (
    "M0_pooled_native_gate2",
    "M1_geometry_native",
    "M2_geometry_immutable_query",
)


def _teacher_xy(config: dict, source_index: int, point_indices: torch.Tensor) -> torch.Tensor:
    sample, _ = load_manifest_sample(
        Path(config["partition"]["manifest"]), source_index
    )
    prepared = prepare_sample(sample, 256)
    yx = prepared["gt_tracks_yx"][point_indices, 15]
    return torch.stack([yx[:, 1], yx[:, 0]], dim=-1).float() * 255.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/routeD_geometry_representation_audit_gate3a_v1.yaml",
    )
    parser.add_argument(
        "--candidate-index",
        default="outputs/routeD_geometry_representation_cache_gate3a_v1_20260719/cache_index.json",
    )
    parser.add_argument(
        "--gate2-index",
        default="outputs/routeD_counterfactual_state_restorer_cache_20260719/fit_internal_validation/cache_index.json",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    candidate_index_path = Path(args.candidate_index).resolve()
    gate2_index_path = Path(args.gate2_index).resolve()
    config = yaml.safe_load(config_path.read_text())
    candidate_index = json.loads(candidate_index_path.read_text())
    gate2_index = json.loads(gate2_index_path.read_text())
    gate2_rows = {int(row["source_index"]): row for row in gate2_index["rows"]}
    top_k_values = (8, 16, 32, 64)
    distances = {
        representation: {top_k: [] for top_k in top_k_values}
        for representation in REPRESENTATIONS
    }
    map_max_abs = []
    map_cosine = []
    support_max_abs = []
    support_cosine = []
    for candidate_row in candidate_index["rows"]:
        source_index = int(candidate_row["source_index"])
        frozen = torch.load(
            candidate_row["sidecar"], map_location="cpu", weights_only=False
        )
        artifact = load_csrr_cache_video(
            gate2_rows[source_index],
            expected_partition="fit_internal_validation",
            expected_config_sha256=str(config["frozen_parent"]["gate2_config_sha256"]),
        )
        tensors = artifact["model_tensors"]
        failure_local = torch.where(tensors["apply_target"].float() > 0.5)[0]
        point_indices = tensors["point_indices"][failure_local].long()
        teacher = _teacher_xy(config, source_index, point_indices)
        native = frozen["candidates"][REPRESENTATIONS[0]][
            "candidate_coordinates_xy"
        ][:, 0].float()
        m1 = frozen["score_maps"][REPRESENTATIONS[1]].float()
        m2 = frozen["score_maps"][REPRESENTATIONS[2]].float()
        map_max_abs.append(float((m1 - m2).abs().max()))
        map_cosine.append(float(F.cosine_similarity(m1.flatten(1), m2.flatten(1)).mean()))
        native_support = [
            value[failure_local].float() for value in tensors["native_track_support"]
        ]
        query_support = [
            value.float() for value in frozen["immutable_query_track_support"]
        ]
        for native_value, query_value in zip(native_support, query_support):
            support_max_abs.append(float((native_value - query_value).abs().max()))
            support_cosine.extend(
                F.cosine_similarity(
                    native_value.flatten(1), query_value.flatten(1), dim=1
                ).tolist()
            )
        for representation in REPRESENTATIONS:
            score_map = frozen["score_maps"][representation].float()
            for top_k in top_k_values:
                candidate = extract_discrete_candidates(
                    score_map,
                    native,
                    top_k=top_k,
                    nms_radius_grid_cells=3,
                    local_refinement_window_grid_cells=5,
                    local_softmax_temperature=0.05,
                    deduplicate_radius_input_px=4.0,
                    input_height=256,
                    input_width=256,
                )
                coordinates = candidate["candidate_coordinates_xy"].float()
                valid = candidate["candidate_valid_mask"]
                distance = torch.linalg.vector_norm(
                    coordinates - teacher[:, None], dim=-1
                )
                distance[~valid] = float("inf")
                distances[representation][top_k].extend(
                    distance.min(dim=1).values.tolist()
                )
    curves = {}
    for representation in REPRESENTATIONS:
        curves[representation] = {}
        for top_k in top_k_values:
            values = np.asarray(distances[representation][top_k], dtype=np.float64)
            curves[representation][str(top_k)] = {
                "nonnative_top_k": top_k,
                "points": int(values.size),
                "recall_within_4px": float(np.mean(values <= 4.0)),
                "recall_within_8px": float(np.mean(values <= 8.0)),
                "recall_within_12px": float(np.mean(values <= 12.0)),
                "median_nearest_candidate_error_px": float(np.median(values)),
                "mean_nearest_candidate_error_px": float(np.mean(values)),
            }
    output = {
        "schema_version": "routeD_geometry_candidate_capacity_diagnostic_gate3a_v1",
        "date": "2026-07-19",
        "status": "completed_post_gate_diagnostic",
        "claim_scope": "fit-only post-gate diagnosis; not a preregistered pass result",
        "config_sha256": file_sha256(config_path),
        "candidate_index_sha256": file_sha256(candidate_index_path),
        "gate2_index_sha256": file_sha256(gate2_index_path),
        "candidate_capacity_curves": curves,
        "M1_M2_static_equivalence": {
            "score_map_max_abs_mean": float(np.mean(map_max_abs)),
            "score_map_max_abs_max": float(np.max(map_max_abs)),
            "score_map_cosine_mean": float(np.mean(map_cosine)),
            "native_vs_query_support_max_abs_mean": float(np.mean(support_max_abs)),
            "native_vs_query_support_max_abs_max": float(np.max(support_max_abs)),
            "native_vs_query_support_cosine_mean": float(np.mean(support_cosine)),
            "native_vs_query_support_cosine_min": float(np.min(support_cosine)),
        },
        "locked_data": config["locked_data"],
    }
    output["summary_payload_sha256"] = canonical_json_sha256(output)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
