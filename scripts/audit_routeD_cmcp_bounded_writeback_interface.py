#!/usr/bin/env python3
"""P0k fit-video interface audit for variant-C bounded coordinate writeback.

The audit first compares the deployable commit-time stream with the formal P0j-C
cache path.  The write hook is executed only if every output-only parity gate is
exact.  No model-validation or locked data is read.
"""
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
EXTERNAL_ROOT = Path("/gemini/code/FSPT")
COTRACKER_ROOT = REPO_ROOT / "baselines/cotracker"
for path in (EXTERNAL_ROOT, COTRACKER_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cotracker.predictor import CoTrackerOnlinePredictor

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_bounded_writeback import (
    BoundedWritebackConfig,
    bounded_coordinate_writeback,
    initialize_variant_c_online_state,
    normalize_adapted_feature_maps,
    overlap_write_eligible,
    variant_c_online_step,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_feature_cache import (
    load_complete_feature_index,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_late_metric_adapter import (
    LMRAConfig,
    LateMetricResidualAdapter,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_lmra_training import (
    predict_lmra_video,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_safety import (
    CMCPLocalPairwiseSafetyComparator,
    CMCPLocalSafetyConfig,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_training import (
    StaticTokenNormalization,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_training import load_cmcp_video
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    file_sha256,
    load_protocol,
    resolve_manifest_path,
    verify_protocol_files,
)
from projects.mmp_tracker.mmp_tracker.routeD_multi_memory_proposal import (
    CMCPConfig,
    CausalMultiMemoryProposalGenerator,
)
from projects.mmp_tracker.mmp_tracker.routeD_musr_training import state_dict_sha256
from scripts.audit_routeD_cotracker3_interface import (
    load_manifest_sample,
    prepare_sample,
)

DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_cmcp_bounded_writeback_v0.yaml"
DEFAULT_PROTOCOL = REPO_ROOT / "configs/routeD_musr_kubric_cache_protocol_v0.json"
DEFAULT_FEATURE_INDEX = REPO_ROOT / "outputs/routeD_cmcp_feature_cache_20260717/fit/cache_index.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/routeD_cmcp_bounded_writeback_20260719/interface_fit0.json"


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


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value.resolve()
    return (REPO_ROOT / value).resolve()


def _load_variant_c(config: dict[str, Any], device: str):
    checkpoint = _resolve(config["base_model"]["checkpoint"])
    if file_sha256(checkpoint) != config["base_model"]["checkpoint_sha256"]:
        raise RuntimeError("variant-C checkpoint SHA mismatch")
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    adapter = LateMetricResidualAdapter(LMRAConfig(**bundle["adapter_config"]))
    cmcp = CausalMultiMemoryProposalGenerator(CMCPConfig(**bundle["cmcp_config"]))
    comparator = CMCPLocalPairwiseSafetyComparator(
        CMCPLocalSafetyConfig(**bundle["comparator_config"])
    )
    adapter.load_state_dict(bundle["adapter_state"], strict=True)
    cmcp.load_state_dict(bundle["cmcp_state"], strict=True)
    comparator.load_state_dict(bundle["comparator_state"], strict=True)
    combined = {}
    combined.update({f"adapter.{k}": v for k, v in adapter.state_dict().items()})
    combined.update({f"cmcp.{k}": v for k, v in cmcp.state_dict().items()})
    combined.update({f"comparator.{k}": v for k, v in comparator.state_dict().items()})
    if state_dict_sha256(combined) != config["base_model"]["combined_model_state_sha256"]:
        raise RuntimeError("variant-C combined model-state SHA mismatch")
    normalization = StaticTokenNormalization(
        torch.tensor(bundle["normalization"]["mean"], dtype=torch.float32),
        torch.tensor(bundle["normalization"]["std"], dtype=torch.float32),
    )
    return (
        adapter.to(device).eval(),
        cmcp.to(device).eval(),
        comparator.to(device).eval(),
        normalization,
        checkpoint,
    )


def _point_batches(point_count: int, batch_size: int) -> list[tuple[int, int]]:
    return [
        (start, min(point_count, start + batch_size))
        for start in range(0, point_count, batch_size)
    ]


def _predict_from_fixed_trajectory(
    *,
    adapted_maps: torch.Tensor,
    native: torch.Tensor,
    vis: torch.Tensor,
    conf: torch.Tensor,
    queries: torch.Tensor,
    cmcp,
    comparator,
    normalization,
    point_batch_size: int,
) -> dict[str, torch.Tensor]:
    points, frames = native.shape[:2]
    candidates = []
    valid = []
    selected_index = []
    selected_coord = []
    summary = []
    with torch.no_grad():
        for start, end in _point_batches(points, point_batch_size):
            state = initialize_variant_c_online_state(
                adapted_maps,
                queries[start:end].to(adapted_maps.device),
                input_height=cmcp.config.input_height,
                input_width=cmcp.config.input_width,
            )
            rows = {key: [] for key in ("candidate", "valid", "index", "coord", "summary")}
            for frame in range(frames):
                output, state = variant_c_online_step(
                    frame_index=frame,
                    normalized_feature_map=adapted_maps[frame],
                    native_coord_xy_px=native[start:end, frame].to(adapted_maps.device),
                    native_visibility_probability=vis[start:end, frame].to(adapted_maps.device),
                    native_confidence_probability=conf[start:end, frame].to(adapted_maps.device),
                    state=state,
                    cmcp=cmcp,
                    comparator=comparator,
                    normalization=normalization,
                )
                rows["candidate"].append(output["candidate_coords_xy_px"].cpu())
                rows["valid"].append(output["candidate_valid_mask"].cpu())
                rows["index"].append(output["selected_candidate_index"].cpu())
                rows["coord"].append(output["selected_coord_xy_px"].cpu())
                rows["summary"].append(output["decision_summary"].cpu())
            candidates.append(torch.stack(rows["candidate"], dim=1))
            valid.append(torch.stack(rows["valid"], dim=1))
            selected_index.append(torch.stack(rows["index"], dim=1))
            selected_coord.append(torch.stack(rows["coord"], dim=1))
            summary.append(torch.stack(rows["summary"], dim=1))
    return {
        "candidate_coords_xy_px": torch.cat(candidates),
        "candidate_valid_mask": torch.cat(valid),
        "selected_candidate_index": torch.cat(selected_index),
        "selected_coords_xy_px": torch.cat(selected_coord),
        "decision_summary": torch.cat(summary),
    }


def _stream_commit_time(
    *,
    predictor,
    video: torch.Tensor,
    queries: torch.Tensor,
    adapted_maps: torch.Tensor,
    cmcp,
    comparator,
    normalization,
    point_batch_size: int,
    write_config: BoundedWritebackConfig,
    enable_writeback: bool,
) -> dict[str, torch.Tensor]:
    _, frames, _, height, width = video.shape
    points = queries.shape[0]
    interp_height, interp_width = predictor.interp_shape
    query_input = torch.zeros(1, points, 3, device=video.device)
    query_input[0, :, 0] = queries[:, 0].to(video.device)
    query_input[0, :, 1] = queries[:, 2].to(video.device) * float(width - 1)
    query_input[0, :, 2] = queries[:, 1].to(video.device) * float(height - 1)
    predictor(
        video_chunk=video,
        is_first_step=True,
        queries=query_input,
        add_support_grid=False,
        grid_size=0,
    )
    batches = _point_batches(points, point_batch_size)
    states = [
        initialize_variant_c_online_state(
            adapted_maps,
            queries[start:end].to(adapted_maps.device),
            input_height=cmcp.config.input_height,
            input_width=cmcp.config.input_width,
        )
        for start, end in batches
    ]
    candidate = torch.zeros(points, frames, 1 + cmcp.config.proposal_topk, 2)
    candidate_valid = torch.zeros(
        points, frames, 1 + cmcp.config.proposal_topk, dtype=torch.bool
    )
    selected_index = torch.zeros(points, frames, dtype=torch.long)
    selected_coord = torch.zeros(points, frames, 2)
    summary = torch.zeros(points, frames, 4)
    decision_native = torch.zeros(points, frames, 2)
    decision_vis = torch.zeros(points, frames)
    decision_conf = torch.zeros(points, frames)
    first_seen_chunk = torch.full((frames,), -1, dtype=torch.long)
    write_applied = torch.zeros(points, frames, dtype=torch.bool)
    write_coord = torch.zeros(points, frames, 2)
    write_norm = torch.zeros(points, frames)
    vis_hook_unchanged = torch.ones(points, frames, dtype=torch.bool)
    conf_hook_unchanged = torch.ones(points, frames, dtype=torch.bool)
    processed_until = -1

    with torch.no_grad():
        for chunk_start in range(0, frames - predictor.step, predictor.step):
            chunk = video[:, chunk_start : chunk_start + predictor.step * 2]
            predictor(
                video_chunk=chunk,
                is_first_step=False,
                add_support_grid=False,
                grid_size=0,
            )
            current_frames = min(
                int(predictor.model.online_coords_predicted.shape[1]), frames
            )
            first_new = max(processed_until + 1, 0)
            for frame in range(first_new, current_frames):
                first_seen_chunk[frame] = chunk_start
                raw = predictor.model.online_coords_predicted[0, frame, :points].float()
                native = raw.clone()
                native[:, 0] *= 255.0 / float(max(interp_width - 1, 1))
                native[:, 1] *= 255.0 / float(max(interp_height - 1, 1))
                vis = torch.sigmoid(
                    predictor.model.online_vis_predicted[0, frame, :points].float()
                )
                conf = torch.sigmoid(
                    predictor.model.online_conf_predicted[0, frame, :points].float()
                )
                decision_native[:, frame] = native.cpu()
                decision_vis[:, frame] = vis.cpu()
                decision_conf[:, frame] = conf.cpu()
                for batch_id, (start, end) in enumerate(batches):
                    output, states[batch_id] = variant_c_online_step(
                        frame_index=frame,
                        normalized_feature_map=adapted_maps[frame],
                        native_coord_xy_px=native[start:end].to(adapted_maps.device),
                        native_visibility_probability=vis[start:end].to(adapted_maps.device),
                        native_confidence_probability=conf[start:end].to(adapted_maps.device),
                        state=states[batch_id],
                        cmcp=cmcp,
                        comparator=comparator,
                        normalization=normalization,
                    )
                    candidate[start:end, frame] = output["candidate_coords_xy_px"].cpu()
                    candidate_valid[start:end, frame] = output["candidate_valid_mask"].cpu()
                    selected_index[start:end, frame] = output["selected_candidate_index"].cpu()
                    selected_coord[start:end, frame] = output["selected_coord_xy_px"].cpu()
                    summary[start:end, frame] = output["decision_summary"].cpu()

                if enable_writeback:
                    eligible = overlap_write_eligible(
                        frame,
                        chunk_start=chunk_start,
                        step=predictor.step,
                        point_count=points,
                        device=video.device,
                    )
                    write, applied, norm = bounded_coordinate_writeback(
                        native,
                        selected_coord[:, frame].to(video.device),
                        selected_index[:, frame].to(video.device),
                        eligible,
                        write_config,
                    )
                    before_vis = predictor.model.online_vis_predicted[
                        0, frame, :points
                    ].clone()
                    before_conf = predictor.model.online_conf_predicted[
                        0, frame, :points
                    ].clone()
                    write_interp = write.clone()
                    write_interp[:, 0] *= float(interp_width - 1) / 255.0
                    write_interp[:, 1] *= float(interp_height - 1) / 255.0
                    predictor.model.online_coords_predicted[0, frame, :points] = torch.where(
                        applied[:, None], write_interp, raw
                    )
                    vis_hook_unchanged[:, frame] = torch.eq(
                        before_vis,
                        predictor.model.online_vis_predicted[0, frame, :points],
                    ).cpu()
                    conf_hook_unchanged[:, frame] = torch.eq(
                        before_conf,
                        predictor.model.online_conf_predicted[0, frame, :points],
                    ).cpu()
                    write_applied[:, frame] = applied.cpu()
                    write_coord[:, frame] = write.cpu()
                    write_norm[:, frame] = norm.cpu()
            processed_until = max(processed_until, current_frames - 1)

    if processed_until != frames - 1:
        raise RuntimeError(f"incomplete stream {processed_until + 1}/{frames}")
    final_raw = predictor.model.online_coords_predicted[0, :frames, :points].float()
    final_native = final_raw.clone()
    final_native[..., 0] *= 255.0 / float(max(interp_width - 1, 1))
    final_native[..., 1] *= 255.0 / float(max(interp_height - 1, 1))
    final_native = final_native.permute(1, 0, 2).cpu()
    final_vis = torch.sigmoid(
        predictor.model.online_vis_predicted[0, :frames, :points].float()
    ).permute(1, 0).cpu()
    final_conf = torch.sigmoid(
        predictor.model.online_conf_predicted[0, :frames, :points].float()
    ).permute(1, 0).cpu()
    return {
        "candidate_coords_xy_px": candidate,
        "candidate_valid_mask": candidate_valid,
        "selected_candidate_index": selected_index,
        "selected_coords_xy_px": selected_coord,
        "decision_summary": summary,
        "decision_native_coords_xy_px": decision_native,
        "decision_visibility_probability": decision_vis,
        "decision_confidence_probability": decision_conf,
        "first_seen_chunk_start": first_seen_chunk,
        "write_applied": write_applied,
        "write_coords_xy_px": write_coord,
        "write_norm_px": write_norm,
        "visibility_hook_unchanged": vis_hook_unchanged,
        "confidence_hook_unchanged": conf_hook_unchanged,
        "final_native_coords_xy_px": final_native,
        "final_visibility_probability": final_vis,
        "final_confidence_probability": final_conf,
    }


def _compare(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    exact = torch.equal(left, right)
    max_abs = None
    if left.shape == right.shape and left.dtype.is_floating_point:
        max_abs = float((left.float() - right.float()).abs().max().item())
    return {
        "exact": exact,
        "max_abs": max_abs,
        "left_sha256": tensor_sha256(left),
        "right_sha256": tensor_sha256(right),
    }


def _core_hashes(payload: dict[str, torch.Tensor]) -> dict[str, str]:
    keys = (
        "candidate_coords_xy_px",
        "candidate_valid_mask",
        "selected_candidate_index",
        "selected_coords_xy_px",
        "decision_summary",
        "decision_native_coords_xy_px",
        "final_native_coords_xy_px",
        "final_visibility_probability",
        "final_confidence_probability",
    )
    return {key: tensor_sha256(payload[key]) for key in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    ap.add_argument("--feature-index", default=str(DEFAULT_FEATURE_INDEX))
    ap.add_argument("--source-index", type=int, default=0)
    ap.add_argument("--point-batch-size", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    if args.source_index != 0:
        raise ValueError("P0k interface is frozen to fit source index 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    protocol = load_protocol(args.protocol)
    verify_protocol_files(protocol)
    if file_sha256(Path(config["backbone"]["checkpoint"])) != config["backbone"]["checkpoint_sha256"]:
        raise RuntimeError("backbone checkpoint SHA mismatch")
    index = load_complete_feature_index(args.feature_index, expected_partition="fit")
    rows = {int(row["source_index"]): row for row in index["videos"]}
    feature_row = rows[args.source_index]
    bundle = load_cmcp_video(feature_row)
    manifest = resolve_manifest_path(protocol, protocol["partitions"]["fit"]["source"])
    sample, sample_meta = load_manifest_sample(manifest, args.source_index)
    prepared = prepare_sample(sample, int(config["backbone"]["input_raster"]))
    if prepared["video_name"] != bundle.video_name:
        raise RuntimeError("sample identity mismatch")

    _deterministic(17)
    adapter, cmcp, comparator, normalization, c_checkpoint = _load_variant_c(config, args.device)
    frozen_maps = bundle.feature_maps.to(args.device, dtype=torch.float32)
    with torch.no_grad():
        adapted_maps = normalize_adapted_feature_maps(adapter, frozen_maps)
    formal = _predict_from_fixed_trajectory(
        adapted_maps=adapted_maps,
        native=bundle.tensors["native_coords_xy_px"].float(),
        vis=bundle.tensors["native_visibility_probability"].float(),
        conf=bundle.tensors["native_confidence_probability"].float(),
        queries=bundle.tensors["query_points_tyx"].float(),
        cmcp=cmcp,
        comparator=comparator,
        normalization=normalization,
        point_batch_size=args.point_batch_size,
    )
    cached_prediction, _ = predict_lmra_video(
        adapter,
        cmcp,
        comparator,
        bundle,
        normalization,
        device=args.device,
        point_batch_size=args.point_batch_size,
    )
    formal_self_check = {
        "candidate_coordinates": _compare(
            formal["candidate_coords_xy_px"], cached_prediction["candidate_coords_xy_px"]
        ),
        "candidate_valid_mask": _compare(
            formal["candidate_valid_mask"], cached_prediction["candidate_valid_mask"]
        ),
        "selected_indices": _compare(
            formal["selected_candidate_index"], cached_prediction["selected_candidate_index"]
        ),
        "selected_coordinates": _compare(
            formal["selected_coords_xy_px"], cached_prediction["selected_coords_xy_px"]
        ),
    }
    if not all(row["exact"] for row in formal_self_check.values()):
        raise RuntimeError(f"framewise formal implementation mismatch: {formal_self_check}")

    video = prepared["video"].to(args.device)
    queries = prepared["query_points_tyx"].float()
    no_write_runs = []
    for replay in range(2):
        _deterministic(17000)
        predictor = CoTrackerOnlinePredictor(
            checkpoint=config["backbone"]["checkpoint"]
        ).to(args.device).eval()
        no_write_runs.append(
            _stream_commit_time(
                predictor=predictor,
                video=video,
                queries=queries,
                adapted_maps=adapted_maps,
                cmcp=cmcp,
                comparator=comparator,
                normalization=normalization,
                point_batch_size=args.point_batch_size,
                write_config=BoundedWritebackConfig(
                    max_step_px=float(config["intervention"]["max_step_px"])
                ),
                enable_writeback=False,
            )
        )
        del predictor
        torch.cuda.empty_cache() if args.device.startswith("cuda") else None
    replay_exact = _core_hashes(no_write_runs[0]) == _core_hashes(no_write_runs[1])
    no_write = no_write_runs[0]

    base_native = bundle.tensors["native_coords_xy_px"].float()
    base_vis = bundle.tensors["native_visibility_probability"].float()
    base_conf = bundle.tensors["native_confidence_probability"].float()
    final_native_parity = {
        "native_coordinates": _compare(no_write["final_native_coords_xy_px"], base_native),
        "visibility_probability": _compare(no_write["final_visibility_probability"], base_vis),
        "confidence_probability": _compare(no_write["final_confidence_probability"], base_conf),
    }
    commit_vs_formal = {
        "candidate_coordinates": _compare(
            no_write["candidate_coords_xy_px"], formal["candidate_coords_xy_px"]
        ),
        "candidate_valid_mask": _compare(
            no_write["candidate_valid_mask"], formal["candidate_valid_mask"]
        ),
        "selected_indices": _compare(
            no_write["selected_candidate_index"], formal["selected_candidate_index"]
        ),
        "selected_coordinates": _compare(
            no_write["selected_coords_xy_px"], formal["selected_coords_xy_px"]
        ),
        "dynamic_summary": _compare(
            no_write["decision_summary"], formal["decision_summary"]
        ),
    }
    formal_parity = all(row["exact"] for row in commit_vs_formal.values())
    native_final_exact = all(row["exact"] for row in final_native_parity.values())
    interface_pass = replay_exact and native_final_exact and formal_parity

    decision_native_delta = torch.linalg.vector_norm(
        no_write["decision_native_coords_xy_px"] - base_native, dim=-1
    )
    first_seen = no_write["first_seen_chunk_start"]
    eligible_frames = torch.zeros(base_native.shape[1], dtype=torch.bool)
    for frame in range(base_native.shape[1]):
        start = int(first_seen[frame].item())
        if start >= 0:
            eligible_frames[frame] = start + 8 <= frame < start + 16
    eligible_delta = decision_native_delta[:, eligible_frames]

    bounded = None
    if interface_pass:
        _deterministic(17000)
        predictor = CoTrackerOnlinePredictor(
            checkpoint=config["backbone"]["checkpoint"]
        ).to(args.device).eval()
        bounded = _stream_commit_time(
            predictor=predictor,
            video=video,
            queries=queries,
            adapted_maps=adapted_maps,
            cmcp=cmcp,
            comparator=comparator,
            normalization=normalization,
            point_batch_size=args.point_batch_size,
            write_config=BoundedWritebackConfig(
                max_step_px=float(config["intervention"]["max_step_px"])
            ),
            enable_writeback=True,
        )
        del predictor

    decision = (
        "ALLOW_P0K_COMPLETE_MODEL_VALIDATION"
        if interface_pass
        else "STOP_P0K_BEFORE_MODEL_VALIDATION_COMMIT_STATE_MISMATCH"
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_suffix(".pt")
    artifact = {
        "schema_version": "routeD_cmcp_bounded_writeback_interface_v0",
        "formal": formal,
        "commit_no_write": no_write,
        "bounded_write": bounded,
    }
    torch.save(artifact, sidecar)
    report = {
        "schema_version": "routeD_cmcp_bounded_writeback_interface_report_v0",
        "decision": decision,
        "interface_pass": interface_pass,
        "source_index": args.source_index,
        "video_name": bundle.video_name,
        "frames": int(base_native.shape[1]),
        "points": int(base_native.shape[0]),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "protocol": str(Path(args.protocol).resolve()),
        "protocol_sha256": protocol["_protocol_sha256"],
        "variant_C_checkpoint": str(c_checkpoint),
        "variant_C_checkpoint_sha256": file_sha256(c_checkpoint),
        "feature_index": str(Path(args.feature_index).resolve()),
        "feature_index_sha256": file_sha256(Path(args.feature_index).resolve()),
        "feature_sidecar": bundle.feature_sidecar,
        "sample_manifest": sample_meta,
        "formal_self_check": formal_self_check,
        "no_write_replay_exact": replay_exact,
        "no_write_replay_hashes": [
            _core_hashes(no_write_runs[0]),
            _core_hashes(no_write_runs[1]),
        ],
        "final_native_parity": final_native_parity,
        "commit_time_vs_formal_C": commit_vs_formal,
        "commit_native_vs_final_native": {
            "all_rows_max_abs_px": float(decision_native_delta.max().item()),
            "all_rows_nonzero_fraction": float((decision_native_delta > 0).float().mean().item()),
            "eligible_rows_max_abs_px": float(eligible_delta.max().item()) if eligible_delta.numel() else 0.0,
            "eligible_rows_nonzero_fraction": float((eligible_delta > 0).float().mean().item()) if eligible_delta.numel() else 0.0,
        },
        "bounded_write_executed": bounded is not None,
        "sidecar": str(sidecar),
        "sidecar_sha256": file_sha256(sidecar),
        "locked_data_read": {
            "calibration": False,
            "final_holdout": False,
            "tapvid_davis": False,
            "tapvid_kinetics": False,
        },
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
