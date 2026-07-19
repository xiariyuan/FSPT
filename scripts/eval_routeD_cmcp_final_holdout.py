#!/usr/bin/env python3
"""Evaluate frozen P0j-C once on a complete sealed P0l holdout cache."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_cmcp_final_holdout import (
    FINAL_HOLDOUT_RESULT_SCHEMA,
    load_final_holdout_config,
    partition_indices,
    verify_frozen_files,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_late_metric_adapter import (
    LMRAConfig,
    LateMetricResidualAdapter,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_lmra_training import evaluate_lmra_index
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_safety import (
    CMCPLocalPairwiseSafetyComparator,
    CMCPLocalSafetyConfig,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_training import StaticTokenNormalization
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_training import load_cmcp_video
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256
from projects.mmp_tracker.mmp_tracker.routeD_multi_memory_proposal import (
    CMCPConfig,
    CausalMultiMemoryProposalGenerator,
)
from projects.mmp_tracker.mmp_tracker.routeD_musr_training import state_dict_sha256

DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_cmcp_final_holdout_v0.yaml"
DEFAULT_INDEX = REPO_ROOT / "outputs/routeD_cmcp_final_holdout_20260719/final_holdout/cache_index.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/routeD_cmcp_final_holdout_20260719/evaluation/primary.json"


def _deterministic(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)


def _load_model(config, device: str):
    verified = verify_frozen_files(config)
    checkpoint = Path(verified["model"]["path"])
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    adapter = LateMetricResidualAdapter(LMRAConfig(**bundle["adapter_config"]))
    cmcp = CausalMultiMemoryProposalGenerator(CMCPConfig(**bundle["cmcp_config"]))
    comparator = CMCPLocalPairwiseSafetyComparator(CMCPLocalSafetyConfig(**bundle["comparator_config"]))
    adapter.load_state_dict(bundle["adapter_state"], strict=True)
    cmcp.load_state_dict(bundle["cmcp_state"], strict=True)
    comparator.load_state_dict(bundle["comparator_state"], strict=True)
    combined = {}
    combined.update({f"adapter.{k}":v for k,v in adapter.state_dict().items()})
    combined.update({f"cmcp.{k}":v for k,v in cmcp.state_dict().items()})
    combined.update({f"comparator.{k}":v for k,v in comparator.state_dict().items()})
    combined_sha = state_dict_sha256(combined)
    if combined_sha != config["frozen_model"]["combined_model_state_sha256"]:
        raise RuntimeError("frozen variant-C state mismatch")
    normalization = StaticTokenNormalization(
        torch.tensor(bundle["normalization"]["mean"], dtype=torch.float32),
        torch.tensor(bundle["normalization"]["std"], dtype=torch.float32),
    )
    return adapter.to(device).eval(), cmcp.to(device).eval(), comparator.to(device).eval(), normalization, bundle, combined_sha


def _verify_sealed_index(index_path: Path, config) -> dict:
    payload = json.loads(index_path.read_text())
    expected = list(partition_indices(config, "final_holdout"))
    if payload.get("partition") != "final_holdout":
        raise RuntimeError("evaluator accepts final_holdout only")
    if payload.get("protocol_sha256") != config["_config_sha256"]:
        raise RuntimeError("sealed cache config mismatch")
    if payload.get("expected_source_indices") != expected or payload.get("completed_source_indices") != expected:
        raise RuntimeError("sealed cache membership mismatch")
    if not payload.get("complete") or int(payload.get("completed_count", -1)) != 16:
        raise RuntimeError("sealed cache is incomplete")
    integrity = payload.get("integrity", {})
    if integrity.get("performance_metrics_computed") is not False:
        raise RuntimeError("cache stage exposed performance")
    if integrity.get("calibration_read") is not False:
        raise RuntimeError("calibration was read")
    for row in payload["videos"]:
        if row["integrity"]["performance_metrics_computed"] is not False:
            raise RuntimeError("video cache exposed performance")
        load_cmcp_video(row)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    config = load_final_holdout_config(args.config)
    index_path = Path(args.index).resolve()
    index = _verify_sealed_index(index_path, config)
    seed = int(config["formal_evaluation"]["seed"])
    _deterministic(seed)
    adapter, cmcp, comparator, normalization, bundle, combined_sha = _load_model(config, args.device)
    metrics = evaluate_lmra_index(
        adapter, cmcp, comparator, index_path, normalization,
        expected_partition="final_holdout", device=args.device,
        point_batch_size=int(config["formal_evaluation"]["point_batch_size"]),
        bootstrap_samples=int(config["formal_evaluation"]["bootstrap_samples"]),
        bootstrap_seed=int(config["formal_evaluation"]["bootstrap_seed"]),
    )
    if int(metrics["videos"]) != 16:
        raise RuntimeError("formal evaluation did not cover 16 videos")
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": FINAL_HOLDOUT_RESULT_SCHEMA,
        "seed": seed,
        "config_path": config["_config_path"],
        "config_sha256": config["_config_sha256"],
        "cache_index": str(index_path),
        "cache_index_sha256": file_sha256(index_path),
        "cache_index_payload_sha256": index["cache_index_payload_sha256"],
        "checkpoint": str(Path(config["frozen_model"]["checkpoint"]).resolve() if Path(config["frozen_model"]["checkpoint"]).is_absolute() else (REPO_ROOT/Path(config["frozen_model"]["checkpoint"])).resolve()),
        "checkpoint_sha256": config["frozen_model"]["checkpoint_sha256"],
        "combined_model_state_sha256": combined_sha,
        "adapter_state_sha256": state_dict_sha256(adapter.state_dict()),
        "cmcp_state_sha256": state_dict_sha256(cmcp.state_dict()),
        "comparator_state_sha256": state_dict_sha256(comparator.state_dict()),
        "normalization": normalization.to_json(),
        "metrics": metrics,
        "model_selection_on_final_holdout": False,
        "calibration_read": False,
        "external_data_read": {"tapvid_davis":False,"tapvid_kinetics":False},
    }
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "output":str(output),"sha256":file_sha256(output),"videos":metrics["videos"],
        "AJ_gain_points":metrics["selected_gain_points"]["AJ"],
        "AJ_CI":metrics["paired_video_selected_AJ_gain_CI"],
        "harmful_non_native_rate":metrics["behavior"]["harmful_non_native_rate"],
        "performance_read_once_after_complete_cache":True,
    },indent=2))

if __name__ == "__main__": main()
