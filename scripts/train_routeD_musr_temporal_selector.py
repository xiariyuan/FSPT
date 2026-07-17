#!/usr/bin/env python3
"""Formal P0f training for the causal MUSR temporal selector."""
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

from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256
from projects.mmp_tracker.mmp_tracker.routeD_musr_training import (
    SelectorLossConfig,
    clone_state_dict_cpu,
    load_cache_index,
    state_dict_sha256,
)
from projects.mmp_tracker.mmp_tracker.routeD_recovery_network import RecoveryNetworkConfig
from projects.mmp_tracker.mmp_tracker.routeD_temporal_selector import (
    CausalSetEvidenceTemporalSelector,
    TemporalSelectorConfig,
)
from projects.mmp_tracker.mmp_tracker.routeD_temporal_training import (
    evaluate_temporal_selector,
    streaming_feature_normalization,
    train_temporal_selector_one_epoch,
)
from scripts.train_routeD_musr import build_configs

CACHE_ROOT = REPO_ROOT / "outputs/routeD_musr_raw_v1_cache_20260717"
DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_musr_cotracker3_temporal_v0.yaml"
DEFAULT_QUALIFICATION = (
    REPO_ROOT / "docs/generated/ROUTED_MUSR_KUBRIC_QUALIFICATION_SUMMARY_2026-07-17.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "outputs/routeD_musr_temporal_v0_20260717/seed17"


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


def build_temporal_configs(
    config_path: Path,
) -> tuple[RecoveryNetworkConfig, TemporalSelectorConfig, dict]:
    network, _, payload = build_configs(config_path)
    temporal_payload = payload["temporal"]
    temporal = TemporalSelectorConfig(
        window_frames=int(temporal_payload["window_frames"]),
        temporal_layers=int(temporal_payload["temporal_layers"]),
        temporal_heads=int(temporal_payload["temporal_heads"]),
        temporal_feedforward_multiplier=int(
            temporal_payload["temporal_feedforward_multiplier"]
        ),
        dropout=float(temporal_payload["dropout"]),
    )
    return network, temporal, payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--fit-index", default=str(CACHE_ROOT / "fit/cache_index.json"))
    ap.add_argument(
        "--validation-index",
        default=str(CACHE_ROOT / "model_validation/cache_index.json"),
    )
    ap.add_argument("--qualification-summary", default=str(DEFAULT_QUALIFICATION))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--learning-rate", type=float, default=3.0e-4)
    ap.add_argument("--weight-decay", type=float, default=1.0e-4)
    ap.add_argument("--bootstrap-samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, patience, and batch sizes must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    qualification_path = Path(args.qualification_summary).resolve()
    qualification = json.loads(qualification_path.read_text())
    if not qualification["qualification"]["training_authorized"]:
        raise RuntimeError("candidate qualification does not authorize clean training")
    if qualification["integrity"]["final_holdout_read"]:
        raise RuntimeError("qualification reports final-holdout contamination")

    config_path = Path(args.config).resolve()
    network_config, temporal_config, raw_config = build_temporal_configs(config_path)
    normalization, fit_index = streaming_feature_normalization(
        args.fit_index, expected_partition="fit"
    )
    validation_index = load_cache_index(
        args.validation_index, expected_partition="model_validation"
    )
    for name, index in (("fit", fit_index), ("model_validation", validation_index)):
        if index["protocol_sha256"] != qualification["protocol"]["sha256"]:
            raise RuntimeError(f"{name} protocol mismatch")
        if index.get("representation_schema_version") != (
            "routeD_musr_cotracker3_raw_representation_v1"
        ):
            raise RuntimeError(f"{name} is not the frozen raw-v1 representation")
        if int(index["candidate_feature_dim"]) != network_config.candidate_feature_dim:
            raise RuntimeError(f"{name} candidate feature dimension mismatch")
        if int(index["state_feature_dim"]) != network_config.state_feature_dim:
            raise RuntimeError(f"{name} state feature dimension mismatch")
        integrity = index.get("integrity", {})
        if integrity.get("candidate_coordinates_changed"):
            raise RuntimeError(f"{name} reports candidate-coordinate drift")
        if any(
            bool(integrity.get(key, False))
            for key in (
                "calibration_read",
                "final_holdout_read",
                "tapvid_davis_read",
                "tapvid_kinetics_read",
            )
        ):
            raise RuntimeError(f"{name} reports locked-data contamination")
    if fit_index["protocol_sha256"] != validation_index["protocol_sha256"]:
        raise RuntimeError("fit/model-validation protocol mismatch")

    set_deterministic(args.seed)
    model = CausalSetEvidenceTemporalSelector(network_config, temporal_config).to(
        args.device
    )
    initialization = evaluate_temporal_selector(
        model,
        args.validation_index,
        normalization,
        device=args.device,
        batch_size=args.eval_batch_size,
        bootstrap_samples=min(args.bootstrap_samples, 1000),
        bootstrap_seed=args.seed * 1000 - 1,
    )
    if (
        abs(float(initialization["gain_points"]["AJ"])) > 1.0e-10
        or abs(float(initialization["gain_points"]["delta_average"])) > 1.0e-10
        or abs(float(initialization["severe_16px_rate"]["delta"])) > 1.0e-12
        or float(initialization["behavior"]["selected_non_native_rate"]) != 0.0
    ):
        raise RuntimeError("temporal selector native-safe initialization failed")

    selector_config = SelectorLossConfig()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    history = []
    best_state = None
    best_epoch = -1
    best_validation_aj = float("-inf")
    stale = 0
    for epoch in range(args.epochs):
        train = train_temporal_selector_one_epoch(
            model,
            args.fit_index,
            normalization,
            optimizer,
            network_config,
            selector_config,
            device=args.device,
            batch_size=args.batch_size,
            generator=generator,
        )
        validation = evaluate_temporal_selector(
            model,
            args.validation_index,
            normalization,
            device=args.device,
            batch_size=args.eval_batch_size,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.seed * 1000 + epoch,
        )
        history.append({"epoch": epoch, "train": train, "validation": validation})
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train["loss"],
                    "selector_accuracy": train["selector_accuracy"],
                    "train_regret": train["mean_utility_regret"],
                    "validation_AJ_gain_points": validation["gain_points"]["AJ"],
                    "validation_AJ_CI": validation["paired_video_AJ_gain_CI"],
                    "behavior": validation["behavior"],
                    "severe_16px_delta": validation["severe_16px_rate"]["delta"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        score = float(validation["selected_metrics"]["AJ"])
        if score > best_validation_aj + 1.0e-8:
            best_validation_aj = score
            best_epoch = epoch
            best_state = clone_state_dict_cpu(model)
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("temporal training produced no checkpoint")

    model.load_state_dict(best_state, strict=True)
    final = evaluate_temporal_selector(
        model,
        args.validation_index,
        normalization,
        device=args.device,
        batch_size=args.eval_batch_size,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed * 1000 + 999,
    )
    gate = {
        "AJ_gain_at_least_0_5": final["gain_points"]["AJ"] >= 0.5,
        "paired_AJ_CI_lower_positive": final["paired_video_AJ_gain_CI"]["lower"] > 0.0,
        "delta_gain_positive": final["gain_points"]["delta_average"] > 0.0,
        "severe_16px_not_worse": final["severe_16px_rate"]["delta"] <= 0.0,
        "harmful_global_selection_rate_at_most_0_01": final["behavior"][
            "harmful_global_selection_rate"
        ]
        <= 0.01,
    }
    gate["pass"] = all(gate.values())
    gate["decision"] = (
        "ALLOW_STAGE_B_STATE_WRITE_TRAINING"
        if gate["pass"]
        else "STOP_TEMPORAL_SELECTOR_AND_REDESIGN_CANDIDATE_GENERATION"
    )

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema_version": "routeD_musr_temporal_selector_bundle_v0",
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "network_config": asdict(network_config),
        "temporal_config": asdict(temporal_config),
        "selector_loss_config": asdict(selector_config),
        "model_state": best_state,
        "model_state_sha256": state_dict_sha256(best_state),
        "normalization": normalization.to_serializable(),
        "optimizer": {
            "type": "AdamW",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "epochs_requested": int(args.epochs),
            "patience": int(args.patience),
        },
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "protocol_sha256": fit_index["protocol_sha256"],
        "representation_schema_version": fit_index["representation_schema_version"],
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
    }
    checkpoint = output / "best.pt"
    torch.save(bundle, checkpoint)
    metrics = {key: value for key, value in bundle.items() if key != "model_state"}
    metrics["checkpoint_path"] = str(checkpoint)
    metrics["checkpoint_sha256"] = file_sha256(checkpoint)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": metrics["checkpoint_sha256"],
                "model_state_sha256": bundle["model_state_sha256"],
                "best_epoch": best_epoch,
                "validation_gain_points": final["gain_points"],
                "validation_AJ_CI": final["paired_video_AJ_gain_CI"],
                "behavior": final["behavior"],
                "severe_16px_rate": final["severe_16px_rate"],
                "gate": gate,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
