#!/usr/bin/env python3
"""Train the preregistered Route-D MUSR model on frozen Kubric caches."""
from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_musr_training import (
    clone_state_dict_cpu,
    compute_feature_normalization,
    evaluate_model_on_cache_index,
    load_training_rows,
    state_dict_sha256,
    train_one_epoch,
)
from projects.mmp_tracker.mmp_tracker.routeD_recovery_network import (
    MultiHypothesisStateRecoveryNetwork,
    RecoveryLossConfig,
    RecoveryNetworkConfig,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256

DEFAULT_CACHE_ROOT = REPO_ROOT / "outputs/routeD_musr_kubric_cache_20260717"
DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_musr_cotracker3_stage0.yaml"
DEFAULT_QUALIFICATION = (
    REPO_ROOT
    / "docs/generated/ROUTED_MUSR_KUBRIC_QUALIFICATION_SUMMARY_2026-07-17.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs/routeD_musr_training_20260717/seed17"


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
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


def build_configs(config_path: Path) -> tuple[RecoveryNetworkConfig, RecoveryLossConfig, dict]:
    payload = yaml.safe_load(config_path.read_text())
    model_cfg = payload["model"]
    loss_cfg = payload["loss"]
    network = RecoveryNetworkConfig(
        candidate_feature_dim=int(model_cfg["candidate_feature_dim"]),
        state_feature_dim=int(model_cfg["state_feature_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_attention_heads=int(model_cfg["num_attention_heads"]),
        num_attention_layers=int(model_cfg["num_attention_layers"]),
        feedforward_multiplier=int(model_cfg["feedforward_multiplier"]),
        num_candidate_sources=int(model_cfg["num_candidate_sources"]),
        thresholds_px=tuple(float(v) for v in model_cfg["thresholds_px"]),
        threshold_utility_weights=tuple(
            float(v) for v in model_cfg["threshold_utility_weights"]
        ),
        catastrophe_risk_weight=float(model_cfg["catastrophe_risk_weight"]),
        max_contextual_selector_bias=float(model_cfg["max_contextual_selector_bias"]),
        coordinate_scale_px=float(model_cfg["coordinate_scale_px"]),
        max_coordinate_update_px=float(model_cfg["max_coordinate_update_px"]),
        selection_temperature=float(model_cfg["selection_temperature"]),
        dropout=float(model_cfg["dropout"]),
        state_fields=tuple(str(v) for v in model_cfg["state_fields"]),
        native_safe_initialization=bool(model_cfg.get("native_safe_initialization", True)),
        initial_abstention_bias=float(model_cfg.get("initial_abstention_bias", 2.0)),
        initial_coordinate_write_bias=float(
            model_cfg.get("initial_coordinate_write_bias", -2.0)
        ),
        initial_other_write_bias=float(model_cfg.get("initial_other_write_bias", -4.0)),
    )
    loss = RecoveryLossConfig(**{key: float(value) for key, value in loss_cfg.items()})
    return network, loss, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--fit-index", default=str(DEFAULT_CACHE_ROOT / "fit/cache_index.json"))
    parser.add_argument(
        "--validation-index",
        default=str(DEFAULT_CACHE_ROOT / "model_validation/cache_index.json"),
    )
    parser.add_argument("--qualification-summary", default=str(DEFAULT_QUALIFICATION))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    qualification_path = Path(args.qualification_summary).resolve()
    qualification = json.loads(qualification_path.read_text())
    if not qualification["qualification"]["training_authorized"]:
        raise RuntimeError("candidate qualification did not authorize training")
    if qualification["integrity"]["final_holdout_read"]:
        raise RuntimeError("qualification summary indicates final holdout contamination")

    set_deterministic(args.seed)
    config_path = Path(args.config).resolve()
    network_config, loss_config, raw_config = build_configs(config_path)
    fit_rows, fit_index = load_training_rows(args.fit_index, expected_partition="fit")
    _, validation_index = load_training_rows(
        args.validation_index, expected_partition="model_validation"
    )
    if fit_index["protocol_sha256"] != validation_index["protocol_sha256"]:
        raise RuntimeError("fit/validation protocol mismatch")
    if fit_index["protocol_sha256"] != qualification["protocol"]["sha256"]:
        raise RuntimeError("training caches do not match qualification protocol")

    normalization = compute_feature_normalization(fit_rows)
    model = MultiHypothesisStateRecoveryNetwork(network_config).to(args.device)
    initialization_validation = evaluate_model_on_cache_index(
        model,
        args.validation_index,
        normalization,
        expected_partition="model_validation",
        device=args.device,
        batch_size=args.eval_batch_size,
        bootstrap_samples=min(args.bootstrap_samples, 1000),
        bootstrap_seed=args.seed * 1000 - 1,
    )
    if (
        abs(initialization_validation["gain_points"]["AJ"]) > 1.0e-10
        or abs(initialization_validation["gain_points"]["delta_average"]) > 1.0e-10
        or abs(initialization_validation["severe_16px_rate"]["delta"]) > 1.0e-12
    ):
        raise RuntimeError("native-safe initialization failed exact metric parity")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    history = []
    best_state = None
    best_epoch = -1
    best_validation_AJ = float("-inf")
    stale = 0
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model,
            fit_rows,
            normalization,
            optimizer,
            network_config,
            loss_config,
            device=args.device,
            batch_size=args.batch_size,
            generator=generator,
        )
        validation = evaluate_model_on_cache_index(
            model,
            args.validation_index,
            normalization,
            expected_partition="model_validation",
            device=args.device,
            batch_size=args.eval_batch_size,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.seed * 1000 + epoch,
        )
        validation_AJ = float(validation["learned_metrics"]["AJ"])
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation,
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "validation_AJ": validation_AJ,
                    "validation_AJ_gain_points": validation["gain_points"]["AJ"],
                    "validation_AJ_CI": validation["paired_video_AJ_gain_CI"],
                    "severe_16px_delta": validation["severe_16px_rate"]["delta"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if validation_AJ > best_validation_AJ + 1.0e-8:
            best_validation_AJ = validation_AJ
            best_epoch = epoch
            best_state = clone_state_dict_cpu(model)
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state, strict=True)
    final_validation = evaluate_model_on_cache_index(
        model,
        args.validation_index,
        normalization,
        expected_partition="model_validation",
        device=args.device,
        batch_size=args.eval_batch_size,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed * 1000 + 999,
    )
    gate = {
        "AJ_gain_at_least_1": final_validation["gain_points"]["AJ"] >= 1.0,
        "paired_AJ_CI_lower_positive": final_validation["paired_video_AJ_gain_CI"][
            "lower"
        ]
        > 0.0,
        "delta_gain_positive": final_validation["gain_points"]["delta_average"] > 0.0,
        "severe_16px_not_worse": final_validation["severe_16px_rate"]["delta"] <= 0.0,
    }
    gate["pass"] = all(gate.values())
    gate["decision"] = (
        "ALLOW_CALIBRATION_PARTITION_READ"
        if gate["pass"]
        else "STOP_BEFORE_CALIBRATION_AND_REDESIGN_MUSR"
    )

    checkpoint = {
        "schema_version": "routeD_musr_training_bundle_v1",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "network_config": asdict(network_config),
        "loss_config": asdict(loss_config),
        "model_state": best_state,
        "normalization": normalization.to_serializable(),
        "optimizer": {
            "type": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "patience": args.patience,
        },
        "protocol_sha256": fit_index["protocol_sha256"],
        "fit_cache_index_sha256": fit_index["_index_sha256"],
        "validation_cache_index_sha256": validation_index["_index_sha256"],
        "qualification_summary_sha256": file_sha256(qualification_path),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "initialization_validation": initialization_validation,
        "final_validation": final_validation,
        "gate": gate,
        "history": history,
        "external_data_read": {
            "calibration": False,
            "final_holdout": False,
            "tapvid_davis": False,
            "tapvid_kinetics": False,
        },
    }
    checkpoint["model_state_sha256"] = state_dict_sha256(best_state)
    checkpoint_path = output / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    metrics = {key: value for key, value in checkpoint.items() if key != "model_state"}
    metrics["checkpoint_path"] = str(checkpoint_path)
    metrics["checkpoint_sha256"] = file_sha256(checkpoint_path)
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": metrics["checkpoint_sha256"],
                "best_epoch": best_epoch,
                "model_state_sha256": checkpoint["model_state_sha256"],
                "initialization_gain_points": initialization_validation["gain_points"],
                "validation_gain_points": final_validation["gain_points"],
                "validation_AJ_CI": final_validation["paired_video_AJ_gain_CI"],
                "severe_16px_rate": final_validation["severe_16px_rate"],
                "gate": gate,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
