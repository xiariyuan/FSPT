#!/usr/bin/env python3
"""Train CMCP on fit only and gate proposal-only performance on model-validation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_cmcp_feature_cache import load_complete_feature_index
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_training import (
    CMCPConfig,
    CMCPLossConfig,
    evaluate_cmcp_index,
    train_cmcp_epoch,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256
from projects.mmp_tracker.mmp_tracker.routeD_multi_memory_proposal import CausalMultiMemoryProposalGenerator
from projects.mmp_tracker.mmp_tracker.routeD_musr_training import clone_state_dict_cpu, state_dict_sha256

DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_cmcp_cotracker3_stage0.yaml"
DEFAULT_FIT = REPO_ROOT / "outputs/routeD_cmcp_feature_cache_20260717/fit/cache_index.json"
DEFAULT_VALIDATION = REPO_ROOT / "outputs/routeD_cmcp_feature_cache_20260717/model_validation/cache_index.json"
DEFAULT_QUALIFICATION = REPO_ROOT / "docs/generated/ROUTED_MUSR_KUBRIC_QUALIFICATION_SUMMARY_2026-07-17.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/routeD_cmcp_training_20260717/seed17"


def set_deterministic(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)


def build_configs(path: Path) -> tuple[CMCPConfig, CMCPLossConfig, dict]:
    raw = yaml.safe_load(path.read_text())
    model = raw["model"]
    memory = raw["memory"]
    supervision = raw["supervision"]
    loss = raw["loss_weights"]
    config = CMCPConfig(
        feature_dim=int(model["feature_dim"]),
        hidden_channels=int(model["hidden_channels"]),
        proposal_topk=int(model["proposal_topk"]),
        nms_radius_cells=int(model["nms_radius_cells"]),
        ema_alpha=float(memory["ema_alpha"]),
        motion_sigma_cells=float(model["motion_sigma_cells"]),
        risk_weight=float(model["risk_weight"]),
        native_logit_bias=float(model["native_logit_bias"]),
        input_height=int(raw["protocol"]["input_raster"]),
        input_width=int(raw["protocol"]["input_raster"]),
    )
    loss_config = CMCPLossConfig(
        focal_weight=float(loss["focal"]),
        utility_weight=float(loss["utility"]),
        risk_weight=float(loss["catastrophic_risk"]),
        native_fallback_weight=float(loss["native_fallback"]),
        no_harm_weight=float(loss["no_harm"]),
        temporal_consistency_weight=float(loss["temporal_consistency"]),
        focal_gamma=float(loss["focal_gamma"]),
        focal_negative_beta=float(loss["focal_negative_beta"]),
        ranking_margin=float(loss["ranking_margin"]),
        gaussian_sigma_px=float(supervision["gaussian_sigma_px"]),
        catastrophe_threshold_px=float(supervision["catastrophe_threshold_px"]),
        grad_clip_norm=float(raw["training"]["grad_clip_norm"]),
    )
    if int(model.get("recurrent_input_channels", -1)) != 9:
        raise ValueError("CMCP preregistered recurrent input must contain nine channels")
    return config, loss_config, raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--fit-index", default=str(DEFAULT_FIT))
    ap.add_argument("--validation-index", default=str(DEFAULT_VALIDATION))
    ap.add_argument("--qualification-summary", default=str(DEFAULT_QUALIFICATION))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--point-batch-size", type=int, default=0)
    ap.add_argument("--eval-point-batch-size", type=int, default=0)
    ap.add_argument("--learning-rate", type=float, default=0.0)
    ap.add_argument("--weight-decay", type=float, default=-1.0)
    ap.add_argument("--bootstrap-samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train-videos", type=int, default=0)
    ap.add_argument("--max-validation-videos", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    config, loss_config, raw = build_configs(config_path)
    training = raw["training"]
    seed = int(args.seed or training["seed"])
    epochs = int(args.epochs or training["epochs"])
    patience = int(args.patience or training["patience"])
    point_batch_size = int(args.point_batch_size or training["point_batch_size"])
    eval_point_batch_size = int(args.eval_point_batch_size or training["eval_point_batch_size"])
    learning_rate = float(args.learning_rate or training["learning_rate"])
    weight_decay = float(training["weight_decay"] if args.weight_decay < 0 else args.weight_decay)
    bootstrap_samples = int(args.bootstrap_samples or training["bootstrap_samples"])
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    qualification_path = Path(args.qualification_summary).resolve()
    qualification = json.loads(qualification_path.read_text())
    if not qualification["qualification"]["training_authorized"]:
        raise RuntimeError("Kubric qualification did not authorize fit-only training")
    if qualification["integrity"]["final_holdout_read"]:
        raise RuntimeError("qualification indicates final-holdout contamination")

    fit_index = load_complete_feature_index(args.fit_index, expected_partition="fit")
    validation_index = load_complete_feature_index(args.validation_index, expected_partition="model_validation")
    if fit_index["protocol_sha256"] != validation_index["protocol_sha256"]:
        raise RuntimeError("fit/model-validation protocol mismatch")
    if fit_index["protocol_sha256"] != qualification["protocol"]["sha256"]:
        raise RuntimeError("feature caches do not match qualified protocol")

    set_deterministic(seed)
    model = CausalMultiMemoryProposalGenerator(config).to(args.device)
    if model.input_projection[0].in_channels != 9:
        raise RuntimeError("CMCP implementation omitted preregistered pairwise differences")

    initialization = evaluate_cmcp_index(
        model,
        args.validation_index,
        config,
        expected_partition="model_validation",
        device=args.device,
        point_batch_size=eval_point_batch_size,
        bootstrap_samples=min(bootstrap_samples, 1000),
        bootstrap_seed=seed * 1000 - 1,
        max_videos=args.max_validation_videos,
    )
    if (
        abs(initialization["selected_gain_points"]["AJ"]) > 1e-10
        or abs(initialization["selected_gain_points"]["delta_average"]) > 1e-10
        or abs(initialization["severe_16px_rate"]["selected_delta"]) > 1e-12
        or initialization["behavior"]["selected_non_native_rate"] != 0.0
    ):
        raise RuntimeError("CMCP native-safe initialization failed exact top-1 parity")

    # The exact native-safe zero-step model is a legitimate checkpoint baseline.
    # A trained epoch must strictly improve complete model-validation direct AJ
    # or the formal result falls back to native rather than selecting harm.
    best_state = clone_state_dict_cpu(model)
    best_epoch = -1
    best_validation_AJ = float(initialization["selected_metrics"]["AJ"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    history = []
    stale = 0
    for epoch in range(epochs):
        train_metrics = train_cmcp_epoch(
            model,
            args.fit_index,
            config,
            loss_config,
            optimizer,
            device=args.device,
            point_batch_size=point_batch_size,
            generator=generator,
            max_videos=args.max_train_videos,
        )
        validation = evaluate_cmcp_index(
            model,
            args.validation_index,
            config,
            expected_partition="model_validation",
            device=args.device,
            point_batch_size=eval_point_batch_size,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=seed * 1000 + epoch,
            max_videos=args.max_validation_videos,
        )
        aj = float(validation["selected_metrics"]["AJ"])
        row = {"epoch": epoch, "train": train_metrics, "validation": validation}
        history.append(row)
        print(json.dumps({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "selected_AJ_gain_points": validation["selected_gain_points"]["AJ"],
            "oracle_AJ_gain_points": validation["oracle_gain_points"]["AJ"],
            "selected_AJ_CI": validation["paired_video_selected_AJ_gain_CI"],
            "harmful_non_native_rate": validation["behavior"]["harmful_non_native_rate"],
            "severe_16px_delta": validation["severe_16px_rate"]["selected_delta"],
        }, ensure_ascii=False), flush=True)
        if aj > best_validation_AJ + 1e-8:
            best_validation_AJ = aj
            best_epoch = epoch
            best_state = clone_state_dict_cpu(model)
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state, strict=True)
    final = evaluate_cmcp_index(
        model,
        args.validation_index,
        config,
        expected_partition="model_validation",
        device=args.device,
        point_batch_size=eval_point_batch_size,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=seed * 1000 + 999,
        max_videos=args.max_validation_videos,
    )
    gate = {
        "candidate_oracle_AJ_gain_at_least_3": final["oracle_gain_points"]["AJ"] >= 3.0,
        "direct_top1_AJ_gain_at_least_0_5": final["selected_gain_points"]["AJ"] >= 0.5,
        "paired_top1_AJ_CI_lower_positive": final["paired_video_selected_AJ_gain_CI"]["lower"] > 0.0,
        "delta_gain_positive": final["selected_gain_points"]["delta_average"] > 0.0,
        "severe_16px_not_worse": final["severe_16px_rate"]["selected_delta"] <= 0.0,
        "harmful_non_native_rate_at_most_0_01": final["behavior"]["harmful_non_native_rate"] <= 0.01,
    }
    gate["pass"] = all(gate.values())
    gate["decision"] = "ALLOW_CMCP_AS_TRAINABLE_CANDIDATE_GENERATOR" if gate["pass"] else "STOP_CMCP_AND_CONSIDER_LATE_BACKBONE_FINETUNING"
    bundle = {
        "schema_version": "routeD_cmcp_proposal_training_bundle_v0",
        "seed": seed,
        "best_epoch": best_epoch,
        "model_config": asdict(config),
        "loss_config": asdict(loss_config),
        "model_state": best_state,
        "model_state_sha256": state_dict_sha256(best_state),
        "optimizer": {
            "type": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs_requested": epochs,
            "patience": patience,
            "point_batch_size": point_batch_size,
            "eval_point_batch_size": eval_point_batch_size,
        },
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "protocol_sha256": fit_index["protocol_sha256"],
        "fit_cache_index_sha256": fit_index["_index_sha256"],
        "validation_cache_index_sha256": validation_index["_index_sha256"],
        "qualification_summary_sha256": file_sha256(qualification_path),
        "initialization_validation": initialization,
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
            "max_train_videos": int(args.max_train_videos),
            "max_validation_videos": int(args.max_validation_videos),
        },
    }
    checkpoint = output / "best.pt"
    torch.save(bundle, checkpoint)
    metrics = {key: value for key, value in bundle.items() if key != "model_state"}
    metrics["checkpoint_path"] = str(checkpoint)
    metrics["checkpoint_sha256"] = file_sha256(checkpoint)
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "model_state_sha256": bundle["model_state_sha256"],
        "best_epoch": best_epoch,
        "selected_gain_points": final["selected_gain_points"],
        "oracle_gain_points": final["oracle_gain_points"],
        "selected_AJ_CI": final["paired_video_selected_AJ_gain_CI"],
        "behavior": final["behavior"],
        "severe_16px_rate": final["severe_16px_rate"],
        "gate": gate,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
