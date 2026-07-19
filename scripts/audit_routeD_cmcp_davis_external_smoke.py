#!/usr/bin/env python3
"""Kubric fit-0 exact model/schema smoke for P0m; does not read DAVIS samples."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_davis_external import (
    load_davis_external_config,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_feature_cache import (
    load_complete_feature_index,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_lmra_training import predict_lmra_video
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_training import load_cmcp_video
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256
from scripts.eval_routeD_cmcp_davis_external import load_variant_c_external
from scripts.eval_routeD_cmcp_final_holdout import _load_model as load_variant_c_p0l

DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_cmcp_davis_external_v0.yaml"
DEFAULT_INDEX = REPO_ROOT / "outputs/routeD_cmcp_feature_cache_20260717/fit/cache_index.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs/generated/ROUTED_STRONG_BACKBONE_DAVIS_EXTERNAL_SMOKE_SUMMARY_2026-07-19.json"


def _compare(left: torch.Tensor, right: torch.Tensor) -> dict:
    exact = torch.equal(left, right)
    max_abs = (
        float((left.float() - right.float()).abs().max().item())
        if left.dtype.is_floating_point and left.shape == right.shape
        else None
    )
    return {
        "exact": exact,
        "max_abs": max_abs,
        "left_sha256": tensor_sha256(left),
        "right_sha256": tensor_sha256(right),
    }


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
    index = load_complete_feature_index(args.index, expected_partition="fit")
    row = next(item for item in index["videos"] if int(item["source_index"]) == 0)
    bundle = load_cmcp_video(row)
    external = load_variant_c_external(config, args.device)
    p0l_config = REPO_ROOT / "configs/routeD_cmcp_final_holdout_v0.yaml"
    p0l = load_variant_c_p0l(
        __import__(
            "projects.mmp_tracker.mmp_tracker.routeD_cmcp_final_holdout",
            fromlist=["load_final_holdout_config"],
        ).load_final_holdout_config(p0l_config),
        args.device,
    )
    adapter_e, cmcp_e, comparator_e, norm_e, combined_e, _ = external
    adapter_l, cmcp_l, comparator_l, norm_l, _, combined_l = p0l
    with torch.no_grad():
        pred_e, behavior_e = predict_lmra_video(
            adapter_e,
            cmcp_e,
            comparator_e,
            bundle,
            norm_e,
            device=args.device,
            point_batch_size=4,
        )
        pred_l, behavior_l = predict_lmra_video(
            adapter_l,
            cmcp_l,
            comparator_l,
            bundle,
            norm_l,
            device=args.device,
            point_batch_size=4,
        )
    keys = (
        "selected_coords_xy_px",
        "selected_candidate_index",
        "candidate_coords_xy_px",
        "candidate_valid_mask",
        "oracle_coords_xy_px",
        "oracle_candidate_index",
    )
    checks = {key: _compare(pred_e[key], pred_l[key]) for key in keys}
    exact = (
        all(value["exact"] for value in checks.values())
        and behavior_e == behavior_l
        and norm_e.to_json() == norm_l.to_json()
        and combined_e == combined_l
    )
    if not exact:
        raise RuntimeError("P0m Kubric smoke mismatch")
    summary = {
        "schema_version": "routeD_cmcp_davis_external_smoke_summary_v0",
        "date": "2026-07-19",
        "status": "completed_pass",
        "decision": "ALLOW_P0M_COMPLETE_DAVIS_CACHE_BUILD",
        "source": "Kubric fit index 0 only",
        "config_sha256": config["_config_sha256"],
        "combined_model_state_sha256": combined_e,
        "normalization_exact": True,
        "behavior_exact": True,
        "checks": checks,
        "DAVIS_sample_read": False,
        "DAVIS_performance_computed": False,
        "kinetics_read_or_rerun": False,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": file_sha256(output),
                "decision": summary["decision"],
                "exact": exact,
                "DAVIS_sample_read": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
