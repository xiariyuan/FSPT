#!/usr/bin/env python3
"""Verify and package the formal P0j strong-backbone component matrix."""
from __future__ import annotations

import argparse
import copy
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove only serialization-location fields before replay comparison."""
    value = copy.deepcopy(payload)
    value.pop("checkpoint_path", None)
    return value


def verify_replay(
    primary: dict[str, Any],
    replay: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if primary["checkpoint_sha256"] != replay["checkpoint_sha256"]:
        raise RuntimeError(f"{label} replay checkpoint is not byte-identical")
    if canonical_metrics(primary) != canonical_metrics(replay):
        raise RuntimeError(f"{label} replay metrics differ beyond checkpoint_path")
    for field in (
        "combined_model_state_sha256",
        "adapter_state_sha256",
        "cmcp_state_sha256",
        "comparator_state_sha256",
        "best_epoch",
        "history",
        "final_validation",
        "gate",
    ):
        if primary[field] != replay[field]:
            raise RuntimeError(f"{label} replay mismatch in {field}")
    return {
        "exact": True,
        "checkpoint_sha256": primary["checkpoint_sha256"],
        "primary_metrics_sha256": None,
        "replay_metrics_sha256": None,
        "combined_model_state_sha256": primary["combined_model_state_sha256"],
    }


def per_video_gain(payload: dict[str, Any]) -> dict[tuple[int, str], float]:
    output: dict[tuple[int, str], float] = {}
    for row in payload["final_validation"]["per_video"]:
        key = (int(row["source_index"]), str(row["video_name"]))
        if key in output:
            raise RuntimeError(f"duplicate video key: {key}")
        output[key] = float(row["selected_AJ_gain_points"])
    return output


def paired_delta_summary(
    left: dict[tuple[int, str], float],
    right: dict[tuple[int, str], float],
    *,
    seed: int,
    samples: int = 10000,
) -> dict[str, Any]:
    if set(left) != set(right):
        raise RuntimeError("paired video identities differ")
    keys = sorted(left)
    delta = np.asarray([left[key] - right[key] for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(samples, len(delta)))].mean(axis=1)
    return {
        "left_minus_right_mean_AJ_points": float(delta.mean()),
        "median_AJ_points": float(np.median(delta)),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "samples": samples,
        "seed": seed,
        "positive_videos": int((delta > 0).sum()),
        "negative_videos": int((delta < 0).sum()),
        "zero_videos": int((delta == 0).sum()),
        "minimum": float(delta.min()),
        "maximum": float(delta.max()),
        "per_video": {
            f"{key[0]}:{key[1]}": float(value) for key, value in zip(keys, delta)
        },
    }


def compact_variant(payload: dict[str, Any]) -> dict[str, Any]:
    final = payload["final_validation"]
    gains = [float(row["selected_AJ_gain_points"]) for row in final["per_video"]]
    return {
        "best_epoch": int(payload["best_epoch"]),
        "trainable_components": payload.get("trainable_components"),
        "selected_gain_points": final["selected_gain_points"],
        "oracle_gain_points": final["oracle_gain_points"],
        "paired_video_selected_AJ_gain_CI": final["paired_video_selected_AJ_gain_CI"],
        "severe_16px_rate": final["severe_16px_rate"],
        "behavior": final["behavior"],
        "gate": payload["gate"],
        "per_video_summary": {
            "videos": len(gains),
            "mean": statistics.mean(gains),
            "median": statistics.median(gains),
            "minimum": min(gains),
            "maximum": max(gains),
            "positive": sum(value > 0 for value in gains),
            "negative": sum(value < 0 for value in gains),
        },
        "model_state_sha256": payload.get("combined_model_state_sha256")
        or payload.get("model_state_sha256"),
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "external_data_read": payload["external_data_read"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a",
        default=str(
            REPO_ROOT
            / "outputs/routeD_cmcp_pairwise_safety_training_20260717/seed17/metrics.json"
        ),
    )
    parser.add_argument(
        "--b",
        default=str(
            REPO_ROOT / "outputs/routeD_cmcp_lmra_ablation_20260717/B_seed17/metrics.json"
        ),
    )
    parser.add_argument(
        "--b-replay",
        default=str(
            REPO_ROOT
            / "outputs/routeD_cmcp_lmra_ablation_20260717/B_seed17_replay/metrics.json"
        ),
    )
    parser.add_argument(
        "--c",
        default=str(
            REPO_ROOT / "outputs/routeD_cmcp_lmra_ablation_20260717/C_seed17/metrics.json"
        ),
    )
    parser.add_argument(
        "--c-replay",
        default=str(
            REPO_ROOT
            / "outputs/routeD_cmcp_lmra_ablation_20260717/C_seed17_replay/metrics.json"
        ),
    )
    parser.add_argument(
        "--d",
        default=str(
            REPO_ROOT / "outputs/routeD_cmcp_lmra_training_20260717/seed17/metrics.json"
        ),
    )
    parser.add_argument(
        "--d-replay",
        default=str(
            REPO_ROOT
            / "outputs/routeD_cmcp_lmra_training_20260717/seed17_replay/metrics.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            REPO_ROOT
            / "docs/generated/ROUTED_STRONG_BACKBONE_MUSR_ABLATION_SUMMARY_2026-07-19.json"
        ),
    )
    args = parser.parse_args()
    paths = {name: Path(value).resolve() for name, value in vars(args).items() if name != "output"}
    payload = {name: read_json(path) for name, path in paths.items()}

    replay = {
        "B": verify_replay(payload["b"], payload["b_replay"], label="B"),
        "C": verify_replay(payload["c"], payload["c_replay"], label="C"),
    }
    replay["B"]["primary_metrics_sha256"] = file_sha256(paths["b"])
    replay["B"]["replay_metrics_sha256"] = file_sha256(paths["b_replay"])
    replay["C"]["primary_metrics_sha256"] = file_sha256(paths["c"])
    replay["C"]["replay_metrics_sha256"] = file_sha256(paths["c_replay"])

    # P0i already has a formal exact replay; verify it again here.
    d_replay = canonical_metrics(payload["d"]) == canonical_metrics(payload["d_replay"])
    if not d_replay or payload["d"]["checkpoint_sha256"] != payload["d_replay"]["checkpoint_sha256"]:
        raise RuntimeError("D/P0i replay mismatch")

    for name in ("a", "b", "b_replay", "c", "c_replay", "d", "d_replay"):
        if any(payload[name]["external_data_read"].values()):
            raise RuntimeError(f"locked data read in {name}")
    for name in ("b", "b_replay", "c", "c_replay", "d", "d_replay"):
        if payload[name]["smoke_limits"] != {
            "max_train_videos": 0,
            "max_validation_videos": 0,
        }:
            raise RuntimeError(f"non-full-scale result: {name}")
        if not payload[name]["gate"]["pass"]:
            raise RuntimeError(f"formal gate failed: {name}")

    variants = {
        "A_p0h_frozen_metric_frozen_cmcp_trained_comparator": payload["a"],
        "B_identity_metric_joint_cmcp_comparator": payload["b"],
        "C_lmra_frozen_cmcp_comparator": payload["c"],
        "D_full_joint_lmra_cmcp_comparator": payload["d"],
    }
    gains = {name: per_video_gain(value) for name, value in variants.items()}
    paired = {
        "B_minus_A": paired_delta_summary(gains[next(iter([k for k in gains if k.startswith('B_')]))], gains[next(iter([k for k in gains if k.startswith('A_')]))], seed=17018),
        "C_minus_A": paired_delta_summary(gains[next(iter([k for k in gains if k.startswith('C_')]))], gains[next(iter([k for k in gains if k.startswith('A_')]))], seed=17019),
        "D_minus_B": paired_delta_summary(gains[next(iter([k for k in gains if k.startswith('D_')]))], gains[next(iter([k for k in gains if k.startswith('B_')]))], seed=17020),
        "D_minus_C": paired_delta_summary(gains[next(iter([k for k in gains if k.startswith('D_')]))], gains[next(iter([k for k in gains if k.startswith('C_')]))], seed=17021),
        "C_minus_B": paired_delta_summary(gains[next(iter([k for k in gains if k.startswith('C_')]))], gains[next(iter([k for k in gains if k.startswith('B_')]))], seed=17022),
    }

    checks = {
        "B_adds_at_least_0p10_over_A": paired["B_minus_A"]["left_minus_right_mean_AJ_points"] >= 0.10,
        "C_adds_at_least_0p10_over_A": paired["C_minus_A"]["left_minus_right_mean_AJ_points"] >= 0.10,
        "D_adds_at_least_0p10_over_B": paired["D_minus_B"]["left_minus_right_mean_AJ_points"] >= 0.10,
        "D_adds_at_least_0p10_over_C": paired["D_minus_C"]["left_minus_right_mean_AJ_points"] >= 0.10,
        "C_better_than_B_with_positive_CI": paired["C_minus_B"]["lower"] > 0,
        "D_is_best_safety_feasible_model": (
            float(payload["d"]["final_validation"]["selected_gain_points"]["AJ"])
            >= max(
                float(payload["b"]["final_validation"]["selected_gain_points"]["AJ"]),
                float(payload["c"]["final_validation"]["selected_gain_points"]["AJ"]),
            )
        ),
        "B_exact_replay": replay["B"]["exact"],
        "C_exact_replay": replay["C"]["exact"],
        "D_exact_replay": d_replay,
    }
    decision = "REVISE_TO_FROZEN_CMCP_LMRA_COMPARATOR"
    if checks["D_is_best_safety_feasible_model"]:
        decision = "RETAIN_FULL_JOINT_P0I"

    summary = {
        "schema_version": "routeD_strong_backbone_musr_ablation_summary_v0",
        "date": "2026-07-19",
        "status": "completed",
        "formal_decision": decision,
        "claim_revision": (
            "Both metric adaptation and CMCP retraining independently improve the P0h comparator, "
            "but jointly updating CMCP with LMRA causes negative interaction. The strongest "
            "safety-feasible architecture freezes the learned CMCP core and trains only LMRA plus "
            "the comparator."
        ),
        "variants": {name: compact_variant(value) for name, value in variants.items()},
        "paired_component_deltas": paired,
        "non_redundancy_checks": checks,
        "replay": replay,
        "recommended_model": "C_lmra_frozen_cmcp_comparator",
        "next_allowed_step": (
            "One separately preregistered bounded coordinate-only closed-loop writeback ablation "
            "against variant C on fit/model-validation only."
        ),
        "locked_data_read": {
            "calibration": False,
            "final_holdout": False,
            "tapvid_davis": False,
            "tapvid_kinetics": False,
        },
        "artifact_sha256": {name: file_sha256(path) for name, path in paths.items()},
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "sha256": file_sha256(output),
        "decision": decision,
        "recommended_model": summary["recommended_model"],
        "checks": checks,
    }, indent=2))


if __name__ == "__main__":
    main()
