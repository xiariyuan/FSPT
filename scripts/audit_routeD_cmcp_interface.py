#!/usr/bin/env python3
"""One-video native-safe and deterministic interface audit for Route-D CMCP."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

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

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    file_sha256,
    load_protocol,
    resolve_manifest_path,
    verify_protocol_files,
)
from projects.mmp_tracker.mmp_tracker.routeD_multi_memory_proposal import (
    CMCPConfig,
    CMCP_SCHEMA_VERSION,
    CausalMultiMemoryProposalGenerator,
    build_causal_multi_memory_correlations,
    extract_proposal_candidates,
)
from scripts.audit_routeD_cotracker3_interface import (
    compute_fmaps,
    load_manifest_sample,
    prepare_sample,
    run_true_streaming,
)
from scripts.build_routeD_cotracker3_raw_v1_cache import (
    _load_base_artifact,
    _load_base_index,
    _resolve_checkpoint,
    _verify_backbone_native_state,
)

DEFAULT_PROTOCOL = REPO_ROOT / "configs/routeD_musr_kubric_cache_protocol_v0.json"
DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_cmcp_cotracker3_stage0.yaml"
DEFAULT_BASE_INDEX = (
    REPO_ROOT / "outputs/routeD_musr_kubric_cache_20260717/fit/cache_index.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs/routeD_cmcp_20260717/interface_smoke"


def set_deterministic(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def _build_config(path: Path) -> CMCPConfig:
    payload = yaml.safe_load(path.read_text())
    model = payload["model"]
    memory = payload["memory"]
    return CMCPConfig(
        feature_dim=int(model["feature_dim"]),
        hidden_channels=int(model["hidden_channels"]),
        proposal_topk=int(model["proposal_topk"]),
        nms_radius_cells=int(model["nms_radius_cells"]),
        ema_alpha=float(memory["ema_alpha"]),
        motion_sigma_cells=float(model["motion_sigma_cells"]),
        risk_weight=float(model["risk_weight"]),
        native_logit_bias=float(model["native_logit_bias"]),
        input_height=256,
        input_width=256,
    )


def _process_chunk(
    model: CausalMultiMemoryProposalGenerator,
    fmaps: torch.Tensor,
    native: torch.Tensor,
    queries: torch.Tensor,
    config: CMCPConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    correlations, motion, valid = build_causal_multi_memory_correlations(
        fmaps,
        native,
        queries,
        input_height=config.input_height,
        input_width=config.input_width,
        ema_alpha=config.ema_alpha,
        motion_sigma_cells=config.motion_sigma_cells,
    )
    output = model.forward_sequence(correlations, motion, valid)
    candidate_coords = []
    candidate_scores = []
    candidate_valid = []
    selected_index = []
    selected_coords = []
    for frame_index in range(native.shape[1]):
        proposal = extract_proposal_candidates(
            output["proposal_score"][:, frame_index],
            native[:, frame_index],
            output["native_logit"][:, frame_index],
            config,
        )
        candidate_coords.append(proposal["candidate_coords_xy_px"])
        candidate_scores.append(proposal["candidate_scores"])
        candidate_valid.append(proposal["candidate_valid_mask"])
        selected_index.append(proposal["selected_candidate_index"])
        selected_coords.append(proposal["selected_coord_xy_px"])
    tensors = {
        "candidate_coords_xy_px": torch.stack(candidate_coords, dim=1),
        "candidate_scores": torch.stack(candidate_scores, dim=1),
        "candidate_valid_mask": torch.stack(candidate_valid, dim=1),
        "selected_candidate_index": torch.stack(selected_index, dim=1),
        "selected_coord_xy_px": torch.stack(selected_coords, dim=1),
        "frame_valid": valid,
    }
    internals = {
        "correlation_maps": correlations,
        "motion_prior": motion,
        "proposal_score": output["proposal_score"],
        "native_logit": output["native_logit"],
    }
    return tensors, internals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--base-index", default=str(DEFAULT_BASE_INDEX))
    ap.add_argument("--source-index", type=int, default=0)
    ap.add_argument("--point-chunk", type=int, default=4)
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--seed", type=int, default=17000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.point_chunk <= 0:
        raise ValueError("point-chunk must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    set_deterministic(args.seed)
    protocol_path = Path(args.protocol).resolve()
    config_path = Path(args.config).resolve()
    base_index_path = Path(args.base_index).resolve()
    protocol = load_protocol(protocol_path)
    verify_protocol_files(protocol)
    config = _build_config(config_path)
    base_index = _load_base_index(base_index_path, partition="fit")
    row_by_index = {int(row["source_index"]): row for row in base_index["videos"]}
    if args.source_index not in row_by_index:
        raise KeyError(f"source index {args.source_index} not in fit cache")
    base_row = row_by_index[args.source_index]
    base_path, base_artifact = _load_base_artifact(base_row)
    manifest_alias = protocol["partitions"]["fit"]["source"]
    manifest_path = resolve_manifest_path(protocol, manifest_alias)
    sample, _ = load_manifest_sample(manifest_path, args.source_index)
    prepared = prepare_sample(sample, config.input_height)
    if str(prepared["video_name"]) != str(base_row["sample_identity"]["video_name"]):
        raise ValueError("sample identity drift")

    checkpoint = _resolve_checkpoint(protocol)
    predictor = CoTrackerOnlinePredictor(checkpoint=str(checkpoint)).to(args.device).eval()
    video = prepared["video"].to(args.device)
    queries = prepared["query_points_tyx"].to(args.device)
    run_true_streaming(predictor, video, queries)
    base_tensors = base_artifact["tensors"]
    point_count, frame_count = base_tensors["native_visibility"].shape
    native_hashes = _verify_backbone_native_state(
        predictor,
        base_tensors,
        frame_count=frame_count,
        point_count=point_count,
        input_raster=config.input_height,
    )
    fmaps = compute_fmaps(predictor, video)
    if fmaps.shape[1] != config.feature_dim:
        raise ValueError(f"feature dimension drift: {tuple(fmaps.shape)}")
    model = CausalMultiMemoryProposalGenerator(config).to(args.device).eval()

    all_tensors: dict[str, list[torch.Tensor]] = {}
    replay: dict[str, Any] | None = None
    with torch.no_grad():
        for start in range(0, point_count, args.point_chunk):
            end = min(point_count, start + args.point_chunk)
            native_chunk = base_tensors["native_coords_xy_px"][start:end].to(args.device)
            query_chunk = base_tensors["query_points_tyx"][start:end].to(args.device)
            tensors, internals = _process_chunk(
                model, fmaps, native_chunk, query_chunk, config
            )
            if start == 0:
                second_tensors, second_internals = _process_chunk(
                    model, fmaps, native_chunk, query_chunk, config
                )
                replay_checks = {
                    **{
                        key: torch.equal(internals[key], second_internals[key])
                        for key in internals
                    },
                    **{
                        key: torch.equal(tensors[key], second_tensors[key])
                        for key in tensors
                    },
                }
                replay = {
                    "checks": replay_checks,
                    "exact": all(replay_checks.values()),
                    "correlation_sha256": tensor_sha256(
                        internals["correlation_maps"].cpu()
                    ),
                    "proposal_score_sha256": tensor_sha256(
                        internals["proposal_score"].cpu()
                    ),
                }
                if not replay["exact"]:
                    raise RuntimeError(f"CMCP replay mismatch: {replay_checks}")
            for key, value in tensors.items():
                all_tensors.setdefault(key, []).append(value.detach().cpu())

    combined = {key: torch.cat(values, dim=0) for key, values in all_tensors.items()}
    native = base_tensors["native_coords_xy_px"]
    if not torch.equal(combined["candidate_coords_xy_px"][..., 0, :], native):
        raise RuntimeError("candidate-0 coordinate parity failed")
    if not combined["candidate_valid_mask"][..., 0].all():
        raise RuntimeError("candidate-0 validity failed")
    active = combined["frame_valid"]
    if not torch.equal(
        combined["selected_candidate_index"][active],
        torch.zeros_like(combined["selected_candidate_index"][active]),
    ):
        raise RuntimeError("zero-step CMCP selected non-native proposal")
    if not torch.equal(combined["selected_coord_xy_px"][active], native[active]):
        raise RuntimeError("zero-step selected coordinate differs from native")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_suffix(".pt")
    report_path = output.with_suffix(".json")
    artifact = {
        "schema_version": "routeD_cmcp_interface_smoke_v1",
        "config": asdict(config),
        "tensors": combined,
        "tensor_hashes": {key: tensor_sha256(value) for key, value in combined.items()},
    }
    torch.save(artifact, sidecar)
    report = {
        "schema_version": "routeD_cmcp_interface_smoke_report_v1",
        "proposal_schema_version": CMCP_SCHEMA_VERSION,
        "source_partition": "fit",
        "source_index": args.source_index,
        "video_name": prepared["video_name"],
        "points": point_count,
        "frames": frame_count,
        "feature_map_shape": list(fmaps.shape),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol["_protocol_sha256"],
        "base_cache_index_sha256": file_sha256(base_index_path),
        "base_sidecar": str(base_path),
        "base_sidecar_sha256": file_sha256(base_path),
        "checkpoint_sha256": file_sha256(checkpoint),
        "native_state_hashes": native_hashes,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "candidate_zero_native_parity": True,
        "zero_step_selected_native_all_active": True,
        "deterministic_first_chunk_replay": replay,
        "tensor_hashes": artifact["tensor_hashes"],
        "sidecar": str(sidecar),
        "sidecar_sha256": file_sha256(sidecar),
        "integrity": {
            "candidate_generation_ground_truth_free": True,
            "calibration_read": False,
            "final_holdout_read": False,
            "tapvid_davis_read": False,
            "tapvid_kinetics_read": False,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
