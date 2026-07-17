#!/usr/bin/env python3
"""Audit exact zero-step P0h equality for the P0i late metric adapter."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_feature_cache import load_complete_feature_index
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_late_metric_adapter import (
    LMRA_SCHEMA_VERSION,
    LMRA_TRAINABLE_PARAMETERS,
    LateMetricResidualAdapter,
    LMRAConfig,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_cache import (
    inject_dynamic_summary,
    load_complete_pairwise_token_index,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_safety import (
    CMCPLocalPairwiseSafetyComparator,
    CMCPLocalSafetyConfig,
    build_cmcp_local_candidate_tokens,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_training import (
    StaticTokenNormalization,
    _update_summary,
    load_pairwise_token_video,
    normalize_static_tokens,
    predict_pairwise_video,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_training import load_cmcp_video
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256
from projects.mmp_tracker.mmp_tracker.routeD_multi_memory_proposal import (
    CMCPConfig,
    CausalMultiMemoryProposalGenerator,
    build_causal_multi_memory_correlations,
    extract_proposal_candidates,
)
from projects.mmp_tracker.mmp_tracker.routeD_musr_training import state_dict_sha256

DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_cmcp_lmra_v0.yaml"
DEFAULT_FEATURE_INDEX = REPO_ROOT / "outputs/routeD_cmcp_feature_cache_20260717/fit/cache_index.json"
DEFAULT_TOKEN_INDEX = REPO_ROOT / "outputs/routeD_cmcp_pairwise_token_cache_20260717/fit/cache_index.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/routeD_cmcp_lmra_20260717/interface_smoke.json"


def _deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)


def _load_models(config: dict[str, Any], device: str):
    cmcp_path = (REPO_ROOT / config["initial_checkpoints"]["cmcp"]).resolve()
    cmcp_bundle = torch.load(cmcp_path, map_location="cpu", weights_only=False)
    cmcp = CausalMultiMemoryProposalGenerator(CMCPConfig(**cmcp_bundle["model_config"]))
    cmcp.load_state_dict(cmcp_bundle["model_state"], strict=True)
    cmcp.to(device).eval()
    if state_dict_sha256(cmcp.state_dict()) != config["initial_checkpoints"]["cmcp_model_state_sha256"]:
        raise RuntimeError("CMCP checkpoint state mismatch")

    comparator_path = (REPO_ROOT / config["initial_checkpoints"]["comparator"]).resolve()
    comparator_bundle = torch.load(comparator_path, map_location="cpu", weights_only=False)
    comparator = CMCPLocalPairwiseSafetyComparator(
        CMCPLocalSafetyConfig(**comparator_bundle["model_config"])
    )
    comparator.load_state_dict(comparator_bundle["model_state"], strict=True)
    comparator.to(device).eval()
    if state_dict_sha256(comparator.state_dict()) != config["initial_checkpoints"]["comparator_model_state_sha256"]:
        raise RuntimeError("comparator checkpoint state mismatch")
    normalization = StaticTokenNormalization(
        torch.tensor(comparator_bundle["normalization"]["mean"], dtype=torch.float32),
        torch.tensor(comparator_bundle["normalization"]["std"], dtype=torch.float32),
    )
    adapter_cfg = dict(config["adapter"])
    for key in ("class", "trainable_parameters", "insertion", "up_projection_zero_initialized"):
        adapter_cfg.pop(key, None)
    adapter = LateMetricResidualAdapter(LMRAConfig(**adapter_cfg)).to(device).eval()
    return (
        cmcp,
        cmcp_bundle,
        cmcp_path,
        comparator,
        comparator_bundle,
        comparator_path,
        normalization,
        adapter,
    )


def _run_live(
    *,
    cmcp: CausalMultiMemoryProposalGenerator,
    comparator: CMCPLocalPairwiseSafetyComparator,
    adapter: LateMetricResidualAdapter,
    normalization: StaticTokenNormalization,
    feature_video,
    token_video,
    point_chunk: int,
    device: str,
) -> dict[str, torch.Tensor]:
    tensors = feature_video.tensors
    frozen_maps = feature_video.feature_maps.float().to(device)
    adapted_maps = adapter(frozen_maps)
    if not torch.equal(adapted_maps, frozen_maps):
        raise RuntimeError("zero-step LMRA feature identity failed")
    native_all = tensors["native_coords_xy_px"]
    query_all = tensors["query_points_tyx"]
    collected: dict[str, list[torch.Tensor]] = {
        "candidate_tokens": [],
        "candidate_coords_xy_px": [],
        "candidate_scores": [],
        "candidate_valid_mask": [],
        "frame_valid": [],
        "selected_candidate_index": [],
        "selected_coord_xy_px": [],
    }
    for start in range(0, native_all.shape[0], point_chunk):
        end = min(native_all.shape[0], start + point_chunk)
        native = native_all[start:end].to(device)
        queries = query_all[start:end].to(device)
        correlation, motion, frame_valid = build_causal_multi_memory_correlations(
            adapted_maps,
            native,
            queries,
            input_height=cmcp.config.input_height,
            input_width=cmcp.config.input_width,
            ema_alpha=cmcp.config.ema_alpha,
            motion_sigma_cells=cmcp.config.motion_sigma_cells,
        )
        state = None
        summary = torch.zeros(end - start, 4, device=device)
        chunk = {key: [] for key in collected}
        with torch.no_grad():
            for frame in range(native.shape[1]):
                dense, state = cmcp.step(
                    correlation[:, frame],
                    motion[:, frame],
                    state,
                    frame_valid=frame_valid[:, frame],
                )
                proposal = extract_proposal_candidates(
                    dense["proposal_score"],
                    native[:, frame],
                    dense["native_logit"],
                    cmcp.config,
                )
                candidate_valid = proposal["candidate_valid_mask"].clone()
                candidate_valid[:, 1:] &= frame_valid[:, frame, None]
                zero_summary = torch.zeros_like(summary)
                raw_token = build_cmcp_local_candidate_tokens(
                    hidden_map=dense["hidden_map"],
                    recurrent_input=dense["recurrent_input"],
                    utility_logit=dense["utility_logit"],
                    risk_logit=dense["risk_logit"],
                    proposal_score=dense["proposal_score"],
                    candidate_coords_xy_px=proposal["candidate_coords_xy_px"],
                    candidate_scores=proposal["candidate_scores"],
                    candidate_valid_mask=candidate_valid,
                    native_visibility_probability=tensors["native_visibility_probability"][start:end, frame].to(device),
                    native_confidence_probability=tensors["native_confidence_probability"][start:end, frame].to(device),
                    native_joint_probability=tensors["native_joint_probability"][start:end, frame].to(device),
                    previous_decision_summary=zero_summary,
                    input_height=cmcp.config.input_height,
                    input_width=cmcp.config.input_width,
                )
                normalized = normalize_static_tokens(raw_token, candidate_valid, normalization)
                dynamic = inject_dynamic_summary(normalized, summary)
                compared = comparator(dynamic, candidate_valid, proposal["candidate_coords_xy_px"])
                summary = _update_summary(
                    compared,
                    proposal["candidate_coords_xy_px"],
                    frame_valid[:, frame],
                    summary,
                )
                chunk["candidate_tokens"].append(raw_token.cpu())
                chunk["candidate_coords_xy_px"].append(proposal["candidate_coords_xy_px"].cpu())
                chunk["candidate_scores"].append(proposal["candidate_scores"].cpu())
                chunk["candidate_valid_mask"].append(candidate_valid.cpu())
                chunk["frame_valid"].append(frame_valid[:, frame].cpu())
                chunk["selected_candidate_index"].append(compared["selected_candidate_index"].cpu())
                chunk["selected_coord_xy_px"].append(compared["selected_coord_xy_px"].cpu())
        for key in collected:
            collected[key].append(torch.stack(chunk[key], dim=1))
    output = {key: torch.cat(value, dim=0) for key, value in collected.items()}
    output["frozen_feature_maps"] = frozen_maps.cpu()
    output["adapted_feature_maps"] = adapted_maps.cpu()
    cache_checks = {
        "candidate_tokens": torch.equal(output["candidate_tokens"], token_video.candidate_tokens),
        "candidate_coords_xy_px": torch.equal(output["candidate_coords_xy_px"], token_video.candidate_coords_xy_px),
        "candidate_scores": torch.equal(output["candidate_scores"], token_video.candidate_scores),
        "candidate_valid_mask": torch.equal(output["candidate_valid_mask"], token_video.candidate_valid_mask),
        "frame_valid": torch.equal(output["frame_valid"], token_video.frame_valid),
    }
    if not all(cache_checks.values()):
        raise RuntimeError(f"zero-step live/cache mismatch: {cache_checks}")
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--feature-index", default=str(DEFAULT_FEATURE_INDEX))
    ap.add_argument("--token-index", default=str(DEFAULT_TOKEN_INDEX))
    ap.add_argument("--source-index", type=int, default=0)
    ap.add_argument("--point-chunk", type=int, default=4)
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--seed", type=int, default=17000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    _deterministic(args.seed)
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    feature_index = load_complete_feature_index(args.feature_index, expected_partition="fit")
    token_index = load_complete_pairwise_token_index(args.token_index, expected_partition="fit")
    feature_rows = {int(row["source_index"]): row for row in feature_index["videos"]}
    token_rows = {int(row["source_index"]): row for row in token_index["videos"]}
    feature_row = feature_rows[args.source_index]
    token_row = token_rows[args.source_index]
    if feature_row["base_sidecar_sha256"] != token_row["base_sidecar_sha256"]:
        raise RuntimeError("feature/token cache base identity mismatch")
    feature_video = load_cmcp_video(feature_row)
    token_video = load_pairwise_token_video(token_row)
    (
        cmcp,
        cmcp_bundle,
        cmcp_path,
        comparator,
        comparator_bundle,
        comparator_path,
        normalization,
        adapter,
    ) = _load_models(config, args.device)
    baseline, _ = predict_pairwise_video(
        comparator,
        token_video,
        normalization,
        device=args.device,
        point_batch_size=args.point_chunk,
    )
    first = _run_live(
        cmcp=cmcp,
        comparator=comparator,
        adapter=adapter,
        normalization=normalization,
        feature_video=feature_video,
        token_video=token_video,
        point_chunk=args.point_chunk,
        device=args.device,
    )
    second = _run_live(
        cmcp=cmcp,
        comparator=comparator,
        adapter=adapter,
        normalization=normalization,
        feature_video=feature_video,
        token_video=token_video,
        point_chunk=args.point_chunk,
        device=args.device,
    )
    replay_checks = {key: torch.equal(first[key], second[key]) for key in first}
    if not all(replay_checks.values()):
        raise RuntimeError(f"LMRA in-process replay mismatch: {replay_checks}")
    formal_p0h_checks = {
        "selected_candidate_index": torch.equal(
            first["selected_candidate_index"], baseline["selected_candidate_index"]
        ),
        "selected_coord_xy_px": torch.equal(
            first["selected_coord_xy_px"], baseline["selected_coords_xy_px"]
        ),
        "candidate_coords_xy_px": torch.equal(
            first["candidate_coords_xy_px"], token_video.candidate_coords_xy_px
        ),
        "candidate_tokens": torch.equal(first["candidate_tokens"], token_video.candidate_tokens),
        "frozen_feature_identity": torch.equal(
            first["adapted_feature_maps"], first["frozen_feature_maps"]
        ),
    }
    if not all(formal_p0h_checks.values()):
        raise RuntimeError(f"formal P0h zero-step mismatch: {formal_p0h_checks}")
    if not torch.equal(first["candidate_coords_xy_px"][..., 0, :], token_video.native_coords_xy_px):
        raise RuntimeError("candidate-0/native drift")
    tensor_hashes = {key: tensor_sha256(value) for key, value in first.items()}
    trainable_parameters = {
        "lmra": sum(p.numel() for p in adapter.parameters() if p.requires_grad),
        "cmcp": sum(p.numel() for p in cmcp.parameters() if p.requires_grad),
        "comparator": sum(p.numel() for p in comparator.parameters() if p.requires_grad),
    }
    if trainable_parameters["lmra"] != LMRA_TRAINABLE_PARAMETERS:
        raise RuntimeError("LMRA trainable parameter count mismatch")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_suffix(".pt")
    artifact = {
        "schema_version": LMRA_SCHEMA_VERSION,
        "adapter_state": {key: value.cpu() for key, value in adapter.state_dict().items()},
        "adapter_state_sha256": state_dict_sha256(adapter.state_dict()),
        "tensors": {
            key: value
            for key, value in first.items()
            if key not in ("frozen_feature_maps", "adapted_feature_maps")
        },
        "tensor_hashes": tensor_hashes,
    }
    torch.save(artifact, sidecar)
    report = {
        "schema_version": "routeD_cmcp_lmra_interface_report_v0",
        "adapter_schema_version": LMRA_SCHEMA_VERSION,
        "source_partition": "fit",
        "source_index": args.source_index,
        "video_name": feature_video.video_name,
        "points": int(token_video.native_coords_xy_px.shape[0]),
        "frames": int(token_video.native_coords_xy_px.shape[1]),
        "candidate_count": int(token_video.candidate_coords_xy_px.shape[2]),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "feature_index_sha256": feature_index["_index_sha256"],
        "token_index_sha256": token_index["_index_sha256"],
        "base_sidecar_sha256": feature_row["base_sidecar_sha256"],
        "cmcp_checkpoint": str(cmcp_path),
        "cmcp_checkpoint_sha256": file_sha256(cmcp_path),
        "cmcp_model_state_sha256": cmcp_bundle["model_state_sha256"],
        "comparator_checkpoint": str(comparator_path),
        "comparator_checkpoint_sha256": file_sha256(comparator_path),
        "comparator_model_state_sha256": comparator_bundle["model_state_sha256"],
        "adapter_state_sha256": artifact["adapter_state_sha256"],
        "trainable_parameters": trainable_parameters,
        "total_trainable_parameters": sum(trainable_parameters.values()),
        "zero_step_formal_p0h_checks": formal_p0h_checks,
        "zero_step_formal_p0h_exact": all(formal_p0h_checks.values()),
        "in_process_replay_checks": replay_checks,
        "in_process_replay_exact": all(replay_checks.values()),
        "tensor_hashes": tensor_hashes,
        "sidecar": str(sidecar),
        "sidecar_sha256": file_sha256(sidecar),
        "integrity": {
            "native_trajectory_modified": False,
            "candidate_zero_modified": False,
            "ground_truth_used_for_adapter_or_selection": False,
            "validation_read": False,
            "calibration_read": False,
            "final_holdout_read": False,
            "tapvid_davis_read": False,
            "tapvid_kinetics_read": False,
        },
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
