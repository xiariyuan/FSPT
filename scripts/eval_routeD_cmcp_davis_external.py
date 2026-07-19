#!/usr/bin/env python3
"""Evaluate frozen output-only P0j-C on a complete sealed DAVIS cache."""
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

from projects.mmp_tracker.mmp_tracker.routeD_cmcp_davis_external import (
    DAVIS_EXTERNAL_RESULT_SCHEMA,
    evaluate_davis_lmra_index,
    load_davis_external_config,
    verify_davis_external_files,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_late_metric_adapter import (
    LMRAConfig,
    LateMetricResidualAdapter,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_safety import (
    CMCPLocalPairwiseSafetyComparator,
    CMCPLocalSafetyConfig,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_training import (
    StaticTokenNormalization,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256
from projects.mmp_tracker.mmp_tracker.routeD_multi_memory_proposal import (
    CMCPConfig,
    CausalMultiMemoryProposalGenerator,
)
from projects.mmp_tracker.mmp_tracker.routeD_musr_training import state_dict_sha256

DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_cmcp_davis_external_v0.yaml"
DEFAULT_INDEX = REPO_ROOT / "outputs/routeD_cmcp_davis_external_20260719/davis_external/cache_index.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/routeD_cmcp_davis_external_20260719/evaluation/primary.json"


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


def load_variant_c_external(config: dict, device: str):
    verified = verify_davis_external_files(config)
    checkpoint = Path(verified["model"]["path"])
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
    combined.update({f"adapter.{key}": value for key, value in adapter.state_dict().items()})
    combined.update({f"cmcp.{key}": value for key, value in cmcp.state_dict().items()})
    combined.update(
        {f"comparator.{key}": value for key, value in comparator.state_dict().items()}
    )
    combined_sha = state_dict_sha256(combined)
    if combined_sha != config["frozen_model"]["combined_model_state_sha256"]:
        raise RuntimeError("frozen variant-C state mismatch")
    normalization = StaticTokenNormalization(
        torch.tensor(bundle["normalization"]["mean"], dtype=torch.float32),
        torch.tensor(bundle["normalization"]["std"], dtype=torch.float32),
    )
    return (
        adapter.to(device).eval(),
        cmcp.to(device).eval(),
        comparator.to(device).eval(),
        normalization,
        combined_sha,
        checkpoint,
    )


def verify_sealed_davis_index(index_path: Path, config: dict) -> dict:
    payload = json.loads(index_path.read_text())
    if payload.get("partition") != "davis_external":
        raise RuntimeError("P0m evaluator accepts davis_external only")
    if payload.get("protocol_sha256") != config["_config_sha256"]:
        raise RuntimeError("DAVIS cache config mismatch")
    expected = list(range(30))
    if (
        payload.get("expected_source_indices") != expected
        or payload.get("completed_source_indices") != expected
        or int(payload.get("completed_count", -1)) != 30
        or not payload.get("complete")
    ):
        raise RuntimeError("DAVIS cache is incomplete")
    if payload.get("dataset_sha256") != config["external_dataset"]["sha256"]:
        raise RuntimeError("DAVIS cache dataset mismatch")
    integrity = payload.get("integrity", {})
    if integrity.get("performance_metrics_computed") is not False:
        raise RuntimeError("DAVIS cache exposed performance")
    if integrity.get("model_variant_C_executed") is not False:
        raise RuntimeError("DAVIS cache ran variant C before completion")
    if integrity.get("partial_performance_exposed") is not False:
        raise RuntimeError("DAVIS cache exposed partial performance")
    for expected_index, row in enumerate(payload["videos"]):
        if int(row.get("source_index", -1)) != expected_index:
            raise RuntimeError("DAVIS cache row-order mismatch")
        if row["integrity"]["performance_metrics_computed"] is not False:
            raise RuntimeError("DAVIS video cache exposed performance")
        if not isinstance(row.get("native_state_hashes"), dict):
            raise RuntimeError("DAVIS cache row lacks native-state provenance")
    # Full feature/base hashes and native-state provenance are verified exactly
    # once by load_cmcp_video during the frozen evaluation loop. Avoiding a
    # redundant pre-pass changes no model or metric behavior.
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    config = load_davis_external_config(args.config)
    index_path = Path(args.index).resolve()
    index = verify_sealed_davis_index(index_path, config)
    seed = int(config["formal_evaluation"]["seed"])
    _deterministic(seed)
    adapter, cmcp, comparator, normalization, combined_sha, checkpoint = (
        load_variant_c_external(config, args.device)
    )
    metrics = evaluate_davis_lmra_index(
        adapter,
        cmcp,
        comparator,
        index_path,
        normalization,
        device=args.device,
        point_batch_size=int(config["formal_evaluation"]["point_batch_size"]),
        bootstrap_samples=int(config["formal_evaluation"]["bootstrap_samples"]),
        bootstrap_seed=int(config["formal_evaluation"]["bootstrap_seed"]),
    )
    if int(metrics["videos"]) != 30:
        raise RuntimeError("P0m evaluation did not cover 30 videos")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": DAVIS_EXTERNAL_RESULT_SCHEMA,
        "seed": seed,
        "config_path": config["_config_path"],
        "config_sha256": config["_config_sha256"],
        "cache_index": str(index_path),
        "cache_index_sha256": file_sha256(index_path),
        "cache_index_payload_sha256": index["cache_index_payload_sha256"],
        "dataset_sha256": config["external_dataset"]["sha256"],
        "video_order_sha256": index["video_order_sha256"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": config["frozen_model"]["checkpoint_sha256"],
        "combined_model_state_sha256": combined_sha,
        "adapter_state_sha256": state_dict_sha256(adapter.state_dict()),
        "cmcp_state_sha256": state_dict_sha256(cmcp.state_dict()),
        "comparator_state_sha256": state_dict_sha256(comparator.state_dict()),
        "normalization": normalization.to_json(),
        "metrics": metrics,
        "model_selection_on_DAVIS": False,
        "threshold_selection_on_DAVIS": False,
        "calibration_read": False,
        "kinetics_read_or_rerun": False,
        "claim_boundary": config["claim_boundary"],
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": file_sha256(output),
                "videos": metrics["videos"],
                "AJ_gain_points": metrics["selected_gain_points"]["AJ"],
                "AJ_CI": metrics["paired_video_selected_AJ_gain_CI"],
                "harmful_non_native_rate": metrics["behavior"][
                    "harmful_non_native_rate"
                ],
                "complete_cache_read_once": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
