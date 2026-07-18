#!/usr/bin/env python3
"""Formal P0j component ablations for the P0i strong-backbone route."""
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
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_cmcp_late_metric_adapter import (
    LateMetricResidualAdapter,
    LMRAConfig,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_lmra_training import (
    LMRAJointLossConfig,
    evaluate_lmra_index,
    safety_feasible_lmra,
    train_lmra_epoch,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_safety import (
    CMCPLocalPairwiseSafetyComparator,
    CMCPLocalSafetyConfig,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_training import (
    PairwiseSafetyLossConfig,
    StaticTokenNormalization,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_training import CMCPLossConfig
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256
from projects.mmp_tracker.mmp_tracker.routeD_multi_memory_proposal import (
    CMCPConfig,
    CausalMultiMemoryProposalGenerator,
)
from projects.mmp_tracker.mmp_tracker.routeD_musr_training import state_dict_sha256

DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_cmcp_lmra_v0.yaml"
DEFAULT_FIT = REPO_ROOT / "outputs/routeD_cmcp_feature_cache_20260717/fit/cache_index.json"
DEFAULT_VALIDATION = REPO_ROOT / "outputs/routeD_cmcp_feature_cache_20260717/model_validation/cache_index.json"
DEFAULT_P0H_METRICS = REPO_ROOT / "outputs/routeD_cmcp_pairwise_safety_training_20260717/seed17/metrics.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/routeD_cmcp_lmra_ablation_20260717"


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


def _clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _combined_state(
    adapter_state: dict[str, torch.Tensor],
    cmcp_state: dict[str, torch.Tensor],
    comparator_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    output = {}
    output.update({f"adapter.{key}": value for key, value in adapter_state.items()})
    output.update({f"cmcp.{key}": value for key, value in cmcp_state.items()})
    output.update({f"comparator.{key}": value for key, value in comparator_state.items()})
    return output


def _p0h_equivalence_payload(validation: dict, p0h: dict) -> tuple[dict, dict]:
    keys = (
        "native_metrics",
        "selected_metrics",
        "oracle_metrics",
        "selected_gain_points",
        "oracle_gain_points",
        "paired_video_selected_AJ_gain_CI",
        "paired_video_oracle_AJ_gain_CI",
        "paired_video_selected_delta_gain_CI",
        "severe_16px_rate",
        "behavior",
    )
    current = {key: validation[key] for key in keys}
    expected = {key: p0h[key] for key in keys}
    expected_rows = []
    for row in p0h["per_video"]:
        expected_rows.append(row)
    current_rows = []
    for row in validation["per_video"]:
        current_rows.append({key: row[key] for key in expected_rows[len(current_rows)]})
    current["per_video"] = current_rows
    expected["per_video"] = expected_rows
    return current, expected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--fit-index", default=str(DEFAULT_FIT))
    ap.add_argument("--validation-index", default=str(DEFAULT_VALIDATION))
    ap.add_argument("--p0h-metrics", default=str(DEFAULT_P0H_METRICS))
    ap.add_argument("--variant", choices=("B", "C"), required=True)
    ap.add_argument("--output", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--point-batch-size", type=int, default=0)
    ap.add_argument("--eval-point-batch-size", type=int, default=0)
    ap.add_argument("--bootstrap-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max-train-videos", type=int, default=0)
    ap.add_argument("--max-validation-videos", type=int, default=0)
    args = ap.parse_args()
    _deterministic(args.seed)
    variant = args.variant
    train_adapter = variant == "C"
    train_cmcp = variant == "B"
    train_comparator = True
    output_arg = args.output or str(DEFAULT_OUTPUT_ROOT / f"{variant}_seed{args.seed}")
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    training = config["training"]
    joint = config["joint_loss"]

    cmcp_path = (REPO_ROOT / config["initial_checkpoints"]["cmcp"]).resolve()
    cmcp_bundle = torch.load(cmcp_path, map_location="cpu", weights_only=False)
    cmcp = CausalMultiMemoryProposalGenerator(CMCPConfig(**cmcp_bundle["model_config"]))
    cmcp.load_state_dict(cmcp_bundle["model_state"], strict=True)
    cmcp.to(args.device)
    if state_dict_sha256(cmcp.state_dict()) != config["initial_checkpoints"]["cmcp_model_state_sha256"]:
        raise RuntimeError("initial CMCP state mismatch")

    comparator_path = (REPO_ROOT / config["initial_checkpoints"]["comparator"]).resolve()
    comparator_bundle = torch.load(comparator_path, map_location="cpu", weights_only=False)
    comparator = CMCPLocalPairwiseSafetyComparator(
        CMCPLocalSafetyConfig(**comparator_bundle["model_config"])
    )
    comparator.load_state_dict(comparator_bundle["model_state"], strict=True)
    comparator.to(args.device)
    if state_dict_sha256(comparator.state_dict()) != config["initial_checkpoints"]["comparator_model_state_sha256"]:
        raise RuntimeError("initial comparator state mismatch")
    normalization = StaticTokenNormalization(
        torch.tensor(comparator_bundle["normalization"]["mean"], dtype=torch.float32),
        torch.tensor(comparator_bundle["normalization"]["std"], dtype=torch.float32),
    )

    adapter_cfg = dict(config["adapter"])
    for key in ("class", "trainable_parameters", "insertion", "up_projection_zero_initialized"):
        adapter_cfg.pop(key, None)
    adapter = LateMetricResidualAdapter(LMRAConfig(**adapter_cfg)).to(args.device)
    for parameter in adapter.parameters():
        parameter.requires_grad_(train_adapter)
    for parameter in cmcp.parameters():
        parameter.requires_grad_(train_cmcp)
    for parameter in comparator.parameters():
        parameter.requires_grad_(train_comparator)
    initial_adapter_state_sha256 = state_dict_sha256(adapter.state_dict())
    initial_cmcp_state_sha256 = state_dict_sha256(cmcp.state_dict())
    initial_comparator_state_sha256 = state_dict_sha256(comparator.state_dict())

    cmcp_loss = CMCPLossConfig(**cmcp_bundle["loss_config"])
    comparator_loss = PairwiseSafetyLossConfig(**comparator_bundle["loss_config"])
    joint_loss = LMRAJointLossConfig(
        dense_cmcp_weight=joint["dense_cmcp_weight"],
        local_comparator_weight=joint["local_comparator_weight"],
        feature_distortion_weight=joint["feature_distortion_weight"],
        grad_clip_norm=training["grad_clip_norm"],
    )
    epochs = args.epochs or training["epochs"]
    patience = args.patience or training["patience"]
    point_batch = args.point_batch_size or training["point_batch_size"]
    eval_batch = args.eval_point_batch_size or training["eval_point_batch_size"]
    bootstrap = args.bootstrap_samples or training["bootstrap_samples"]

    initial = evaluate_lmra_index(
        adapter,
        cmcp,
        comparator,
        args.validation_index,
        normalization,
        expected_partition="model_validation",
        device=args.device,
        point_batch_size=eval_batch,
        bootstrap_samples=bootstrap,
        bootstrap_seed=17999,
        max_videos=args.max_validation_videos,
    )
    full_scale = args.max_train_videos == 0 and args.max_validation_videos == 0
    p0h_metrics = json.loads(Path(args.p0h_metrics).read_text())
    zero_step_formal_p0h_exact = False
    if full_scale:
        current, expected = _p0h_equivalence_payload(initial, p0h_metrics["final_validation"])
        zero_step_formal_p0h_exact = (
            current == expected
            and initial["candidate_coordinate_combined_sha256"]
            == p0h_metrics["validation_candidate_coordinate_combined_sha256"]
        )
        if not zero_step_formal_p0h_exact:
            raise RuntimeError("zero-step LMRA does not exactly reproduce formal P0h")

    best_adapter = _clone_state(adapter)
    best_cmcp = _clone_state(cmcp)
    best_comparator = _clone_state(comparator)
    best_validation = initial
    best_epoch = -1
    best_safe_aj = float(initial["selected_gain_points"]["AJ"])
    stale = 0
    history = []
    parameter_groups = []
    if train_adapter:
        parameter_groups.append({"params": adapter.parameters(), "lr": training["lmra_learning_rate"]})
    if train_cmcp:
        parameter_groups.append({"params": cmcp.parameters(), "lr": training["cmcp_learning_rate"]})
    if train_comparator:
        parameter_groups.append({"params": comparator.parameters(), "lr": training["comparator_learning_rate"]})
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=training["weight_decay"],
    )
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(epochs):
        train_metrics = train_lmra_epoch(
            adapter,
            cmcp,
            comparator,
            args.fit_index,
            normalization,
            cmcp_loss,
            comparator_loss,
            joint_loss,
            optimizer,
            device=args.device,
            point_batch_size=point_batch,
            generator=generator,
            max_videos=args.max_train_videos,
            train_adapter=train_adapter,
            train_cmcp=train_cmcp,
            train_comparator=train_comparator,
        )
        validation = evaluate_lmra_index(
            adapter,
            cmcp,
            comparator,
            args.validation_index,
            normalization,
            expected_partition="model_validation",
            device=args.device,
            point_batch_size=eval_batch,
            bootstrap_samples=bootstrap,
            bootstrap_seed=17000 + epoch,
            max_videos=args.max_validation_videos,
        )
        safe = safety_feasible_lmra(validation)
        aj = float(validation["selected_gain_points"]["AJ"])
        improved = safe and aj > best_safe_aj + 1e-12
        if improved:
            best_adapter = _clone_state(adapter)
            best_cmcp = _clone_state(cmcp)
            best_comparator = _clone_state(comparator)
            best_validation = validation
            best_epoch = epoch
            best_safe_aj = aj
            stale = 0
        else:
            stale += 1
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation,
            "safety_feasible": safe,
            "checkpoint_improved": improved,
        })
        print(json.dumps({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "AJ_gain_points": aj,
            "oracle_AJ_gain_points": validation["oracle_gain_points"]["AJ"],
            "AJ_CI": validation["paired_video_selected_AJ_gain_CI"],
            "harmful_non_native_rate": validation["behavior"]["harmful_non_native_rate"],
            "beneficial_candidate_recall": validation["behavior"]["beneficial_candidate_recall"],
            "severe_16px_delta": validation["severe_16px_rate"]["selected_delta"],
            "safety_feasible": safe,
            "feature_distortion": train_metrics["feature_distortion"],
        }), flush=True)
        if stale >= patience:
            break

    adapter.load_state_dict(best_adapter, strict=True)
    cmcp.load_state_dict(best_cmcp, strict=True)
    comparator.load_state_dict(best_comparator, strict=True)
    final = evaluate_lmra_index(
        adapter,
        cmcp,
        comparator,
        args.validation_index,
        normalization,
        expected_partition="model_validation",
        device=args.device,
        point_batch_size=eval_batch,
        bootstrap_samples=bootstrap,
        bootstrap_seed=17999,
        max_videos=args.max_validation_videos,
    )
    gate = {
        "zero_step_formal_p0h_equality": zero_step_formal_p0h_exact,
        "native_candidate_parity_all": final["native_candidate_parity_all"],
        "adapter_rank_and_layer_identity": adapter.config.rank == 32 and adapter.config.feature_dim == 128,
        "candidate_oracle_AJ_gain_at_least_3": final["oracle_gain_points"]["AJ"] >= 3.0,
        "direct_AJ_gain_at_least_0_5": final["selected_gain_points"]["AJ"] >= 0.5,
        "paired_AJ_CI_lower_positive": final["paired_video_selected_AJ_gain_CI"]["lower"] > 0,
        "delta_gain_positive": final["selected_gain_points"]["delta_average"] > 0,
        "severe_16px_not_worse": final["severe_16px_rate"]["selected_delta"] <= 0,
        "harmful_non_native_rate_at_most_0_01": final["behavior"]["harmful_non_native_rate"] <= 0.01,
    }
    gate["pass"] = all(gate.values()) and full_scale
    gate["decision"] = "COMPLETE_P0J_COMPONENT_ABLATION"
    combined = _combined_state(best_adapter, best_cmcp, best_comparator)
    output = Path(output_arg).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best.pt"
    metrics_path = output / "metrics.json"
    bundle = {
        "schema_version": "routeD_cmcp_lmra_component_ablation_bundle_v0",
        "ablation_variant": variant,
        "trainable_components": {
            "lmra": train_adapter,
            "cmcp": train_cmcp,
            "comparator": train_comparator,
            "native_tracker": False,
        },
        "seed": args.seed,
        "best_epoch": best_epoch,
        "adapter_config": adapter.config.__dict__,
        "cmcp_config": cmcp.config.__dict__,
        "comparator_config": comparator.config.__dict__,
        "cmcp_loss_config": cmcp_loss.__dict__,
        "comparator_loss_config": comparator_loss.__dict__,
        "joint_loss_config": joint_loss.__dict__,
        "adapter_state": best_adapter,
        "cmcp_state": best_cmcp,
        "comparator_state": best_comparator,
        "combined_model_state_sha256": state_dict_sha256(combined),
        "adapter_state_sha256": state_dict_sha256(best_adapter),
        "cmcp_state_sha256": state_dict_sha256(best_cmcp),
        "comparator_state_sha256": state_dict_sha256(best_comparator),
        "normalization": normalization.to_json(),
        "optimizer": {
            "type": "AdamW",
            "lmra_learning_rate": training["lmra_learning_rate"] if train_adapter else 0.0,
            "cmcp_learning_rate": training["cmcp_learning_rate"] if train_cmcp else 0.0,
            "comparator_learning_rate": training["comparator_learning_rate"],
            "weight_decay": training["weight_decay"],
            "epochs_requested": epochs,
            "patience": patience,
            "point_batch_size": point_batch,
            "eval_point_batch_size": eval_batch,
            "checkpoint_rule": training["checkpoint_rule"],
        },
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "fit_feature_index_sha256": file_sha256(Path(args.fit_index).resolve()),
        "validation_feature_index_sha256": file_sha256(Path(args.validation_index).resolve()),
        "initial_cmcp_checkpoint_sha256": file_sha256(cmcp_path),
        "initial_cmcp_model_state_sha256": cmcp_bundle["model_state_sha256"],
        "initial_comparator_checkpoint_sha256": file_sha256(comparator_path),
        "initial_comparator_model_state_sha256": comparator_bundle["model_state_sha256"],
        "initial_component_state_sha256": {
            "adapter": initial_adapter_state_sha256,
            "cmcp": initial_cmcp_state_sha256,
            "comparator": initial_comparator_state_sha256,
        },
        "frozen_component_state_exact": {
            "adapter": (not train_adapter) and state_dict_sha256(best_adapter) == initial_adapter_state_sha256,
            "cmcp": (not train_cmcp) and state_dict_sha256(best_cmcp) == initial_cmcp_state_sha256,
            "comparator": False,
        },
        "zero_step_formal_p0h_exact": zero_step_formal_p0h_exact,
        "initialization_validation": initial,
        "final_validation": final,
        "gate": gate,
        "history": history,
        "external_data_read": {
            "calibration": False,
            "final_holdout": False,
            "tapvid_davis": False,
            "tapvid_kinetics": False,
        },
        "smoke_limits": {
            "max_train_videos": args.max_train_videos,
            "max_validation_videos": args.max_validation_videos,
        },
    }
    torch.save(bundle, checkpoint_path)
    serial = {
        key: value
        for key, value in bundle.items()
        if key not in ("adapter_state", "cmcp_state", "comparator_state")
    }
    serial["checkpoint_path"] = str(checkpoint_path)
    serial["checkpoint_sha256"] = file_sha256(checkpoint_path)
    metrics_path.write_text(json.dumps(serial, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "ablation_variant": variant,
        "trainable_components": bundle["trainable_components"],
        "frozen_component_state_exact": bundle["frozen_component_state_exact"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": serial["checkpoint_sha256"],
        "combined_model_state_sha256": bundle["combined_model_state_sha256"],
        "best_epoch": best_epoch,
        "selected_gain_points": final["selected_gain_points"],
        "oracle_gain_points": final["oracle_gain_points"],
        "selected_AJ_CI": final["paired_video_selected_AJ_gain_CI"],
        "behavior": final["behavior"],
        "severe_16px_rate": final["severe_16px_rate"],
        "gate": gate,
    }, indent=2))


if __name__ == "__main__":
    main()
