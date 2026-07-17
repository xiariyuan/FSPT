#!/usr/bin/env python3
"""Audit zero-step P0h comparator on one authorized fit video."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_feature_cache import (
    load_complete_feature_index,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_safety import (
    CMCP_PAIRWISE_LOCAL_TOKEN_DIM,
    CMCP_PAIRWISE_SAFETY_SCHEMA_VERSION,
    CMCPLocalPairwiseSafetyComparator,
    CMCPLocalSafetyConfig,
    build_cmcp_local_candidate_tokens,
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

DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_cmcp_local_pairwise_safety_v0.yaml"
DEFAULT_FEATURE_INDEX = REPO_ROOT / "outputs/routeD_cmcp_feature_cache_20260717/fit/cache_index.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/routeD_cmcp_pairwise_safety_20260717/interface_smoke.json"


def _set_deterministic(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _generator(checkpoint_path: Path, device: str) -> tuple[CausalMultiMemoryProposalGenerator, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = CausalMultiMemoryProposalGenerator(CMCPConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if state_dict_sha256(model.state_dict()) != checkpoint["model_state_sha256"]:
        raise RuntimeError("frozen CMCP state hash mismatch")
    return model, checkpoint


def _run(
    generator: CausalMultiMemoryProposalGenerator,
    comparator: CMCPLocalPairwiseSafetyComparator,
    bundle,
    *,
    point_chunk: int,
    device: str,
) -> dict[str, torch.Tensor]:
    tensors = bundle.tensors
    fmaps = bundle.feature_maps.float().to(device)
    native_all = tensors["native_coords_xy_px"]
    query_all = tensors["query_points_tyx"]
    outputs: dict[str, list[torch.Tensor]] = {
        "candidate_coords_xy_px": [],
        "candidate_scores": [],
        "candidate_valid_mask": [],
        "selected_candidate_index": [],
        "selected_coord_xy_px": [],
        "frame_valid": [],
    }
    points = native_all.shape[0]
    for start in range(0, points, point_chunk):
        end = min(points, start + point_chunk)
        native = native_all[start:end].to(device)
        queries = query_all[start:end].to(device)
        corr, motion, valid = build_causal_multi_memory_correlations(
            fmaps,
            native,
            queries,
            input_height=generator.config.input_height,
            input_width=generator.config.input_width,
            ema_alpha=generator.config.ema_alpha,
            motion_sigma_cells=generator.config.motion_sigma_cells,
        )
        state = None
        previous_summary = torch.zeros(end-start, 4, device=device)
        chunk = {key: [] for key in outputs}
        with torch.no_grad():
            for frame in range(native.shape[1]):
                dense, state = generator.step(
                    corr[:, frame], motion[:, frame], state,
                    frame_valid=valid[:, frame],
                )
                proposal = extract_proposal_candidates(
                    dense["proposal_score"], native[:, frame], dense["native_logit"], generator.config
                )
                candidate_valid = proposal["candidate_valid_mask"].clone()
                candidate_valid[:, 1:] &= valid[:, frame, None]
                token = build_cmcp_local_candidate_tokens(
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
                    previous_decision_summary=previous_summary,
                    input_height=generator.config.input_height,
                    input_width=generator.config.input_width,
                )
                compared = comparator(token, candidate_valid, proposal["candidate_coords_xy_px"])
                selected = compared["selected_candidate_index"]
                selected_coord = compared["selected_coord_xy_px"]
                score = compared["candidate_score"]
                selected_score = score.gather(1, selected[:, None]).squeeze(1)
                native_score = score[:, 0]
                displacement = torch.linalg.vector_norm(selected_coord-native[:,frame],dim=-1) / 255.0
                previous_summary = torch.stack([
                    (selected>0).float(),
                    selected.float()/float(max(generator.config.proposal_topk,1)),
                    selected_score-native_score,
                    displacement,
                ],dim=-1)
                chunk["candidate_coords_xy_px"].append(proposal["candidate_coords_xy_px"].cpu())
                chunk["candidate_scores"].append(proposal["candidate_scores"].cpu())
                chunk["candidate_valid_mask"].append(candidate_valid.cpu())
                chunk["selected_candidate_index"].append(selected.cpu())
                chunk["selected_coord_xy_px"].append(selected_coord.cpu())
                chunk["frame_valid"].append(valid[:,frame].cpu())
        for key in outputs:
            outputs[key].append(torch.stack(chunk[key], dim=1))
    return {key: torch.cat(value, dim=0) for key, value in outputs.items()}


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--feature-index", default=str(DEFAULT_FEATURE_INDEX))
    ap.add_argument("--source-index", type=int, default=0)
    ap.add_argument("--point-chunk", type=int, default=4)
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--seed", type=int, default=17000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args=ap.parse_args()
    _set_deterministic(args.seed)
    config_path=Path(args.config).resolve(); config=_load_config(config_path)
    index=load_complete_feature_index(args.feature_index, expected_partition="fit")
    rows={int(row["source_index"]): row for row in index["videos"]}
    row=rows[args.source_index]
    bundle=load_cmcp_video(row)
    checkpoint_path=(REPO_ROOT / config["frozen_generator"]["checkpoint"]).resolve()
    generator, checkpoint=_generator(checkpoint_path,args.device)
    if checkpoint["model_state_sha256"] != config["frozen_generator"]["model_state_sha256"]:
        raise RuntimeError("configured generator state hash mismatch")
    model_cfg=dict(config["model"])
    for key in ("class","native_safe_initialization"):
        model_cfg.pop(key,None)
    comparator=CMCPLocalPairwiseSafetyComparator(CMCPLocalSafetyConfig(**model_cfg)).to(args.device).eval()
    first=_run(generator, comparator, bundle, point_chunk=args.point_chunk, device=args.device)
    second=_run(generator, comparator, bundle, point_chunk=args.point_chunk, device=args.device)
    checks={key:torch.equal(first[key],second[key]) for key in first}
    if not all(checks.values()): raise RuntimeError(f"in-process P0h replay mismatch: {checks}")
    native=bundle.tensors["native_coords_xy_px"]
    if not torch.equal(first["candidate_coords_xy_px"][...,0,:], native):
        raise RuntimeError("candidate-0/native coordinate drift")
    active=first["frame_valid"]
    if not torch.equal(first["selected_candidate_index"][active], torch.zeros_like(first["selected_candidate_index"][active])):
        raise RuntimeError("zero-step comparator did not select native")
    if not torch.equal(first["selected_coord_xy_px"], native):
        raise RuntimeError("zero-step selected coordinate differs from native")
    comparator_parameters=sum(p.numel() for p in comparator.parameters() if p.requires_grad)
    tensor_hashes={key:tensor_sha256(value) for key,value in first.items()}
    artifact={
        "schema_version":CMCP_PAIRWISE_SAFETY_SCHEMA_VERSION,
        "frozen_generator_model_state_sha256":checkpoint["model_state_sha256"],
        "comparator_state":{k:v.cpu() for k,v in comparator.state_dict().items()},
        "tensors":first,
        "tensor_hashes":tensor_hashes,
    }
    output=Path(args.output).resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    sidecar=output.with_suffix(".pt"); torch.save(artifact,sidecar)
    report={
        "schema_version":"routeD_cmcp_pairwise_safety_interface_report_v0",
        "comparator_schema_version":CMCP_PAIRWISE_SAFETY_SCHEMA_VERSION,
        "source_partition":"fit",
        "source_index":args.source_index,
        "video_name":bundle.video_name,
        "points":int(native.shape[0]),
        "frames":int(native.shape[1]),
        "candidate_count":int(first["candidate_coords_xy_px"].shape[2]),
        "local_token_dim":CMCP_PAIRWISE_LOCAL_TOKEN_DIM,
        "comparator_trainable_parameters":comparator_parameters,
        "config_path":str(config_path),
        "config_sha256":file_sha256(config_path),
        "frozen_generator_checkpoint":str(checkpoint_path),
        "frozen_generator_checkpoint_sha256":file_sha256(checkpoint_path),
        "frozen_generator_model_state_sha256":checkpoint["model_state_sha256"],
        "feature_index_sha256":index["_index_sha256"],
        "base_sidecar_sha256":row["base_sidecar_sha256"],
        "candidate_zero_native_parity":True,
        "zero_step_selected_native_all_active":True,
        "generator_parameters_frozen":all(not p.requires_grad for p in generator.parameters()),
        "in_process_replay_checks":checks,
        "in_process_replay_exact":all(checks.values()),
        "tensor_hashes":tensor_hashes,
        "sidecar":str(sidecar),
        "sidecar_sha256":file_sha256(sidecar),
        "integrity":{
            "ground_truth_used_for_tokens_or_selection":False,
            "candidate_coordinates_modified":False,
            "calibration_read":False,
            "final_holdout_read":False,
            "tapvid_davis_read":False,
            "tapvid_kinetics_read":False,
        },
    }
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
