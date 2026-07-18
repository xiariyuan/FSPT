#!/usr/bin/env python3
"""Formal fit-only training for Route-D safe long-occlusion re-detection."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_musr_training import state_dict_sha256
from projects.mmp_tracker.mmp_tracker.routeD_safe_redetection import (
    SafeRedetectionConfig,
    SafeRedetectionLossConfig,
    SafeRedetectionModel,
)
from projects.mmp_tracker.mmp_tracker.routeD_safe_redetection_cache import file_sha256
from projects.mmp_tracker.mmp_tracker.routeD_safe_redetection_training import (
    SafeRedetectionTrainingConfig,
    clone_model_state,
    evaluate_safe_redetection_index,
    safety_feasible,
    train_safe_redetection_epoch,
)

DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_safe_redetection_v0.yaml"
DEFAULT_FIT = REPO_ROOT / "outputs/routeD_safe_redetection_cache_20260718/fit/cache_index.json"
DEFAULT_VALIDATION = REPO_ROOT / "outputs/routeD_safe_redetection_cache_20260718/model_validation/cache_index.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/routeD_safe_redetection_training_20260718/seed17"


def deterministic(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--fit-index", default=str(DEFAULT_FIT))
    ap.add_argument("--validation-index", default=str(DEFAULT_VALIDATION))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--bootstrap-samples", type=int, default=0)
    ap.add_argument("--max-train-events", type=int, default=0)
    ap.add_argument("--max-validation-events", type=int, default=0)
    args = ap.parse_args()
    deterministic(args.seed)

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    alignment = Path(config["official_alignment_gate"]["summary"])
    if not alignment.is_absolute(): alignment = REPO_ROOT / alignment
    alignment_payload = json.loads(alignment.read_text())
    if not alignment_payload["gate"]["pass"]:
        raise RuntimeError("official CoTracker3 alignment gate is not passed")
    parity = Path(config["official_redetection_metric"]["parity_artifact"])
    if not parity.is_absolute(): parity = REPO_ROOT / parity
    parity_payload = json.loads(parity.read_text())
    if not parity_payload["pass"] or parity_payload["maximum_absolute_scalar_difference"] != 0.0:
        raise RuntimeError("official AJ_RD parity gate is not exact")

    model_cfg = dict(config["model"])
    model_cfg.update(input_height=256, input_width=256)
    model = SafeRedetectionModel(SafeRedetectionConfig(**model_cfg)).to(args.device)
    loss_keys = (
        "utility_weight", "risk_weight", "preference_weight", "visibility_weight",
        "reappearance_weight", "abstention_weight", "writeback_weight",
        "false_reacquisition_weight",
    )
    loss_cfg = SafeRedetectionLossConfig(**{key: config["loss"][key] for key in loss_keys})
    training_cfg = SafeRedetectionTrainingConfig(
        dense_proposal_weight=config["loss"]["dense_proposal_weight"],
        local_decision_weight=1.0,
        gradient_clip_norm=config["optimizer"]["gradient_clip_norm"],
        bootstrap_samples=args.bootstrap_samples or 5000,
    )
    epochs = args.epochs or config["optimizer"]["epochs"]
    patience = args.patience or config["optimizer"]["patience"]
    bootstrap = args.bootstrap_samples or training_cfg.bootstrap_samples
    full_scale = args.max_train_events == 0 and args.max_validation_events == 0

    initial = evaluate_safe_redetection_index(
        model,
        args.validation_index,
        expected_role="model_validation",
        device=args.device,
        bootstrap_samples=bootstrap,
        bootstrap_seed=17999,
        max_events=args.max_validation_events,
    )
    zero_parity = bool(
        initial["zero_step_native_parity"]["coordinate_exact_all"]
        and initial["zero_step_native_parity"]["visibility_exact_all"]
        and abs(initial["gain"]["AJ_RD"]) <= 1e-12
    )
    if not zero_parity:
        raise RuntimeError("zero-step model does not exactly reproduce native coordinate/visibility")
    oracle_gate = initial["gain"]["oracle_AJ_RD"] >= config["formal_gates"]["model_validation_candidate_oracle_AJ_RD_gain_min"]
    if full_scale and not oracle_gate:
        raise RuntimeError(
            f"candidate oracle AJ_RD gate failed before training: {initial['gain']['oracle_AJ_RD']}"
        )

    baseline = initial
    best_state = clone_model_state(model)
    best_validation = initial
    best_epoch = -1
    best_gain = float(initial["gain"]["AJ_RD"])
    stale = 0
    history = []
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["learning_rate"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(epochs):
        train_metrics = train_safe_redetection_epoch(
            model,
            args.fit_index,
            optimizer,
            loss_cfg,
            training_cfg,
            device=args.device,
            generator=generator,
            max_events=args.max_train_events,
        )
        validation = evaluate_safe_redetection_index(
            model,
            args.validation_index,
            expected_role="model_validation",
            device=args.device,
            bootstrap_samples=bootstrap,
            bootstrap_seed=17000 + epoch,
            max_events=args.max_validation_events,
        )
        safe = safety_feasible(validation, baseline)
        gain = float(validation["gain"]["AJ_RD"])
        improved = safe and gain > best_gain + 1e-12
        if improved:
            best_state = clone_model_state(model)
            best_validation = validation
            best_epoch = epoch
            best_gain = gain
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
            "AJ_RD_gain": validation["gain"]["AJ_RD"],
            "oracle_AJ_RD_gain": validation["gain"]["oracle_AJ_RD"],
            "paired_CI": validation["paired_event_AJ_gain_CI"],
            "harmful_rate": validation["behavior"]["harmful_intervention_rate"],
            "false_reacquisition_rate": validation["behavior"]["false_reacquisition_rate"],
            "severe_16px_delta": validation["severe_16px_rate"]["delta"],
            "visibility_accuracy_gain": validation["event_standard_metrics"]["visibility_accuracy_gain"],
            "safe": safe,
            "improved": improved,
        }), flush=True)
        if stale >= patience:
            break

    model.load_state_dict(best_state, strict=True)
    final = evaluate_safe_redetection_index(
        model,
        args.validation_index,
        expected_role="model_validation",
        device=args.device,
        bootstrap_samples=bootstrap,
        bootstrap_seed=17999,
        max_events=args.max_validation_events,
    )
    gates = config["formal_gates"]
    gate = {
        "official_baseline_alignment": True,
        "official_AJ_RD_parity_exact": True,
        "zero_step_coordinate_visibility_parity": zero_parity,
        "candidate_oracle_AJ_RD_gain": final["gain"]["oracle_AJ_RD"] >= gates["model_validation_candidate_oracle_AJ_RD_gain_min"],
        "direct_AJ_RD_gain": final["gain"]["AJ_RD"] >= gates["model_validation_direct_AJ_RD_gain_min"],
        "paired_event_AJ_gain_CI_lower_positive": final["paired_event_AJ_gain_CI"]["lower"] > 0,
        "event_standard_AJ_not_regressed": final["event_standard_metrics"]["AJ_gain"] >= gates["model_validation_standard_AJ_regression_min"],
        "visibility_accuracy_not_regressed": final["event_standard_metrics"]["visibility_accuracy_gain"] >= gates["model_validation_OA_regression_min"],
        "harmful_intervention_rate": final["behavior"]["harmful_intervention_rate"] <= gates["harmful_intervention_rate_max"],
        "false_reacquisition_rate": final["behavior"]["false_reacquisition_rate"] <= baseline["behavior"]["false_reacquisition_rate"] + gates["false_reacquisition_rate_increase_max"],
        "severe_16px_not_worse": final["severe_16px_rate"]["delta"] <= 0.0,
        "full_scale": full_scale,
    }
    gate["pass"] = all(gate.values())
    gate["decision"] = (
        "ALLOW_LOCKED_POINTODYSSEY_HOLDOUT_AND_FULL_DAVIS_DEVELOPMENT_EVALUATION"
        if gate["pass"]
        else "STOP_SAFE_REDETECTION_BEFORE_LOCKED_DATA"
    )

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best.pt"
    metrics_path = output / "metrics.json"
    bundle = {
        "schema_version": "routeD_safe_redetection_training_bundle_v0",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "model_config": model.config.__dict__,
        "loss_config": loss_cfg.__dict__,
        "training_config": training_cfg.__dict__,
        "model_state": best_state,
        "model_state_sha256": state_dict_sha256(best_state),
        "optimizer": {
            "type": "AdamW",
            "learning_rate": config["optimizer"]["learning_rate"],
            "weight_decay": config["optimizer"]["weight_decay"],
            "epochs_requested": epochs,
            "patience": patience,
            "checkpoint_rule": config["optimizer"]["checkpoint_rule"],
        },
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "fit_index": str(Path(args.fit_index).resolve()),
        "fit_index_sha256": file_sha256(args.fit_index),
        "validation_index": str(Path(args.validation_index).resolve()),
        "validation_index_sha256": file_sha256(args.validation_index),
        "official_alignment_summary_sha256": file_sha256(alignment),
        "official_AJ_RD_parity_sha256": file_sha256(parity),
        "zero_step_validation": initial,
        "baseline_validation": baseline,
        "final_validation": final,
        "history": history,
        "gate": gate,
        "external_data_read": {
            "locked_internal_holdout": False,
            "locked_pointodyssey_test": False,
            "tapvid_kinetics_official_1144": False,
        },
        "smoke_limits": {
            "max_train_events": args.max_train_events,
            "max_validation_events": args.max_validation_events,
        },
    }
    torch.save(bundle, checkpoint_path)
    serial = {key: value for key, value in bundle.items() if key != "model_state"}
    serial["checkpoint_path"] = str(checkpoint_path)
    serial["checkpoint_sha256"] = file_sha256(checkpoint_path)
    metrics_path.write_text(json.dumps(serial, indent=2, allow_nan=True) + "\n")
    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": serial["checkpoint_sha256"],
        "model_state_sha256": bundle["model_state_sha256"],
        "best_epoch": best_epoch,
        "AJ_RD_gain": final["gain"]["AJ_RD"],
        "oracle_AJ_RD_gain": final["gain"]["oracle_AJ_RD"],
        "paired_CI": final["paired_event_AJ_gain_CI"],
        "behavior": final["behavior"],
        "gate": gate,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
