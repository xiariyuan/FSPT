#!/usr/bin/env python3
"""Train and replay the causal full-population entry model for Gate 3C1F0."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import sklearn
import torch
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer_cache import (
    verify_csrr_cache_artifact,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    canonical_json_sha256,
    file_sha256,
)
from projects.mmp_tracker.mmp_tracker.routeD_temporal_identity_entry_v0 import (
    ENTRY_FEATURE_SCHEMA_VERSION,
    build_entry_features,
    entry_action_mask,
)

SCHEMA = "routeD_temporal_identity_entry_gate3c1f0_v0"
RESULT_SCHEMA = "routeD_temporal_identity_entry_result_gate3c1f0_v0"
BUNDLE_SCHEMA = "routeD_temporal_identity_entry_bundle_gate3c1f0_v0"
DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_temporal_identity_entry_gate3c1f0_v0.yaml"


def _atomic_json_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _result_payload_sha256(result: Mapping[str, Any]) -> str:
    value = dict(result)
    value.pop("result_payload_sha256", None)
    return canonical_json_sha256(value)


def _validate_parent(config: Mapping[str, Any]) -> dict[str, Any]:
    parent = config["authorized_parent"]
    for key, hash_key in (("result", "result_sha256"), ("replay", "replay_sha256")):
        if file_sha256(parent[key]) != parent[hash_key]:
            raise ValueError(f"Gate 3C1F0 parent hash drift: {key}")
    replay = json.loads(Path(parent["replay"]).read_text())
    if replay.get("result_payload_sha256") != _result_payload_sha256(replay):
        raise ValueError("Gate 3C1F0 parent replay payload drift")
    if (
        not bool(replay.get("exact_replay"))
        or not bool(replay.get("gate", {}).get("pass"))
        or replay.get("gate", {}).get("decision") != parent["required_decision"]
    ):
        raise ValueError("Gate 3C1E did not authorize Gate 3C1F0")
    return replay


def _validate_implementation(config: Mapping[str, Any]) -> None:
    for name, authority in config["implementation"].items():
        if file_sha256(authority["path"]) != authority["sha256"]:
            raise ValueError(f"Gate 3C1F0 implementation hash drift: {name}")


def _validate_index(authority: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(authority["path"])
    if file_sha256(path) != authority["file_sha256"]:
        raise ValueError("Gate 3C1F0 cache index file drift")
    index = json.loads(path.read_text())
    without_hash = dict(index)
    payload = without_hash.pop("cache_index_payload_sha256", None)
    expected_sources = list(range(int(authority["source_indices"][0]), int(authority["source_indices"][1]) + 1))
    if (
        payload != authority["payload_sha256"]
        or payload != canonical_json_sha256(without_hash)
        or not bool(index.get("complete"))
        or index.get("partition") != authority["partition"]
        or index.get("completed_source_indices") != expected_sources
        or int(index.get("total_failure_rows", -1)) != int(authority["failure_rows"])
        or int(index.get("total_clean_rows", -1)) != int(authority["clean_rows"])
        or any(bool(value) for value in index.get("integrity", {}).values())
    ):
        raise ValueError("Gate 3C1F0 cache index payload drift")
    return index


def _load_partition(authority: Mapping[str, Any]) -> dict[str, Any]:
    index = _validate_index(authority)
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    joint: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    point_records: list[dict[str, Any]] = []
    for row in index["rows"]:
        if file_sha256(row["sidecar"]) != row["sidecar_sha256"]:
            raise ValueError("Gate 3C1F0 sidecar file drift")
        artifact = verify_csrr_cache_artifact(
            row["sidecar"],
            expected_partition=authority["partition"],
            expected_source_index=int(row["source_index"]),
            expected_config_sha256=index["config_sha256"],
        )
        tensors = artifact["model_tensors"]
        value = build_entry_features(
            trajectory_features=tensors["trajectory_features"],
            visibility_probability=tensors["native_visibility_probability"],
            confidence_probability=tensors["native_confidence_probability"],
            native_track_features=tensors["native_track_feat"],
            native_track_supports=tensors["native_track_support"],
        )
        target = tensors["apply_target"].long().numpy()
        joint_probability = (
            tensors["native_visibility_probability"].float()
            * tensors["native_confidence_probability"].float()
        ).numpy()
        source_index = int(row["source_index"])
        features.append(value)
        labels.append(target)
        joint.append(joint_probability)
        groups.append(np.full(len(target), source_index, dtype=np.int64))
        for point_index, label, joint_value in zip(
            tensors["point_indices"].tolist(), target.tolist(), joint_probability.tolist()
        ):
            point_records.append(
                {
                    "source_index": source_index,
                    "video_name": str(artifact["video_name"]),
                    "point_index": int(point_index),
                    "failure_label": int(label),
                    "native_joint_probability": float(joint_value),
                }
            )
    output = {
        "features": np.concatenate(features).astype(np.float32),
        "labels": np.concatenate(labels).astype(np.int64),
        "joint": np.concatenate(joint).astype(np.float64),
        "groups": np.concatenate(groups).astype(np.int64),
        "point_records": point_records,
    }
    expected_rows = int(authority["failure_rows"]) + int(authority["clean_rows"])
    if len(output["labels"]) != expected_rows:
        raise ValueError("Gate 3C1F0 partition row drift")
    return output


def _model(config: Mapping[str, Any]) -> HistGradientBoostingClassifier:
    model = config["model"]
    return HistGradientBoostingClassifier(
        max_iter=int(model["max_iter"]),
        learning_rate=float(model["learning_rate"]),
        max_leaf_nodes=int(model["max_leaf_nodes"]),
        l2_regularization=float(model["l2_regularization"]),
        min_samples_leaf=int(model["min_samples_leaf"]),
        random_state=int(model["random_state"]),
        early_stopping=False,
    )


def _metrics(
    labels: np.ndarray,
    probability: np.ndarray,
    joint: np.ndarray,
    operating: Mapping[str, Any],
) -> dict[str, Any]:
    action = entry_action_mask(
        entry_probability=probability,
        native_joint_probability=joint,
        probability_min=float(operating["entry_probability_min"]),
        joint_probability_max=float(operating["native_joint_probability_max"]),
    )
    positive = labels == 1
    negative = ~positive
    true_positive = int((positive & action).sum())
    false_positive = int((negative & action).sum())
    return {
        "rows": int(len(labels)),
        "failure_rows": int(positive.sum()),
        "clean_rows": int(negative.sum()),
        "AUC": float(roc_auc_score(labels, probability)),
        "AP": float(average_precision_score(labels, probability)),
        "action_rows": int(action.sum()),
        "action_coverage": float(action.mean()),
        "failure_recall": float(true_positive / max(int(positive.sum()), 1)),
        "clean_false_apply_rate": float(false_positive / max(int(negative.sum()), 1)),
        "action_precision": float(true_positive / max(int(action.sum()), 1)),
    }


def _checks(metrics: Mapping[str, Any], gates: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "AUC": float(metrics["AUC"]) >= float(gates["AUC_min"]),
        "AP": float(metrics["AP"]) >= float(gates["AP_min"]),
        "failure_recall": float(metrics["failure_recall"]) >= float(gates["failure_recall_min"]),
        "clean_false_apply": float(metrics["clean_false_apply_rate"]) <= float(gates["clean_false_apply_rate_max"]),
        "action_precision": float(metrics["action_precision"]) >= float(gates["action_precision_min"]),
    }


def run(
    *, config_path: Path, output_path: Path, reference_path: Path | None
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected Gate 3C1F0 config")
    if any(value is not False for value in config["locked_data"].values()):
        raise ValueError("Gate 3C1F0 locked-data flags must remain false")
    _validate_parent(config)
    _validate_implementation(config)
    train = _load_partition(config["data"]["train"])
    validation = _load_partition(config["data"]["validation"])

    x = train["features"]
    y = train["labels"]
    groups = train["groups"]
    oof = np.zeros(len(y), dtype=np.float64)
    splitter = GroupKFold(int(config["oof"]["folds"]))
    for train_index, valid_index in splitter.split(x, y, groups):
        model = _model(config).fit(x[train_index], y[train_index])
        oof[valid_index] = model.predict_proba(x[valid_index])[:, 1]
    final_model = _model(config).fit(x, y)
    validation_probability = final_model.predict_proba(validation["features"])[:, 1]

    bundle = {
        "schema_version": BUNDLE_SCHEMA,
        "feature_schema_version": ENTRY_FEATURE_SCHEMA_VERSION,
        "model": final_model,
        "operating_point": config["operating_point"],
        "training_digests": {
            "train_features": tensor_sha256(torch.from_numpy(train["features"])),
            "train_labels": tensor_sha256(torch.from_numpy(train["labels"])),
            "train_joint": tensor_sha256(torch.from_numpy(train["joint"])),
            "train_groups": tensor_sha256(torch.from_numpy(train["groups"])),
            "oof_probability": tensor_sha256(torch.from_numpy(oof)),
            "validation_features": tensor_sha256(torch.from_numpy(validation["features"])),
            "validation_labels": tensor_sha256(torch.from_numpy(validation["labels"])),
            "validation_joint": tensor_sha256(torch.from_numpy(validation["joint"])),
            "validation_probability": tensor_sha256(torch.from_numpy(validation_probability)),
        },
    }
    artifact_key = "primary_bundle" if reference_path is None else "replay_bundle"
    bundle_path = Path(config["artifacts"][artifact_key])
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path)
    bundle_artifact = {"path": str(bundle_path), "file_sha256": file_sha256(bundle_path)}

    oof_metrics = _metrics(y, oof, train["joint"], config["operating_point"])
    validation_metrics = _metrics(
        validation["labels"],
        validation_probability,
        validation["joint"],
        config["operating_point"],
    )
    scientific = {
        "oof_metrics": oof_metrics,
        "validation_metrics": validation_metrics,
        "training_digests": bundle["training_digests"],
        "train_point_records_digest": canonical_json_sha256(train["point_records"]),
        "validation_point_records_digest": canonical_json_sha256(validation["point_records"]),
        "operating_point": config["operating_point"],
    }
    scientific["scientific_payload_sha256"] = canonical_json_sha256(scientific)
    comparison = None
    exact_replay = False
    reference = None
    if reference_path is not None:
        reference = json.loads(reference_path.read_text())
        keys = (
            "oof_metrics",
            "validation_metrics",
            "training_digests",
            "train_point_records_digest",
            "validation_point_records_digest",
            "operating_point",
            "scientific_payload_sha256",
        )
        comparison = {key: scientific[key] == reference["scientific"][key] for key in keys}
        exact_replay = all(comparison.values())

    checks = {
        "oof": _checks(oof_metrics, config["gates"]["oof"]),
        "validation": _checks(validation_metrics, config["gates"]["validation"]),
        "exact_replay": exact_replay,
    }
    if reference_path is None:
        passed = False
        decision = "PRIMARY_COMPLETE_AWAIT_EXACT_REPLAY"
        frozen_bundle = bundle_artifact
    else:
        passed = all(checks["oof"].values()) and all(checks["validation"].values()) and exact_replay
        decision = config["formal_decisions"]["pass" if passed else "fail"]
        frozen_bundle = reference["frozen_bundle_artifact"]
        if file_sha256(frozen_bundle["path"]) != frozen_bundle["file_sha256"]:
            raise ValueError("Gate 3C1F0 primary bundle drift during replay")
    result = {
        "schema_version": RESULT_SCHEMA,
        "date": "2026-07-20",
        "status": "completed",
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "bundle_artifact": bundle_artifact,
        "frozen_bundle_artifact": frozen_bundle,
        "scientific": scientific,
        "reference": None if reference_path is None else str(reference_path),
        "replay_comparison": comparison,
        "exact_replay": exact_replay,
        "gate": {"checks": checks, "pass": passed, "decision": decision},
        "runtime_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "locked_data": config["locked_data"],
        "claim_boundary": config["claim_scope"],
    }
    result["result_payload_sha256"] = _result_payload_sha256(result)
    _atomic_json_save(result, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", default=None)
    args = parser.parse_args()
    result = run(
        config_path=Path(args.config).resolve(),
        output_path=Path(args.output).resolve(),
        reference_path=None if args.reference is None else Path(args.reference).resolve(),
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "scientific": result["scientific"],
                "gate": result["gate"],
                "exact_replay": result["exact_replay"],
                "result_payload_sha256": result["result_payload_sha256"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
