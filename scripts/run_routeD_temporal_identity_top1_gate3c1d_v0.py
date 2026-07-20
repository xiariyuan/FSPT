#!/usr/bin/env python3
"""Train and confirm the causal Gate 3C1D shortlist top-1 selector."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import torch
import yaml
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import (
    canonical_json_sha256,
    file_sha256,
)
from projects.mmp_tracker.mmp_tracker.routeD_temporal_identity_selector import (
    QueryClosureIdentitySelectorConfig,
)
from projects.mmp_tracker.mmp_tracker.routeD_temporal_identity_top1 import (
    Top1FeatureConfig,
    build_shortlist_top1_features,
)
from scripts.build_routeD_temporal_identity_train_cache_gate3c1a_v0 import (
    verify_temporal_identity_cache_payload,
)

SCHEMA = "routeD_temporal_identity_top1_gate3c1d_v0"
RESULT_SCHEMA = "routeD_temporal_identity_top1_result_gate3c1d_v0"
DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_temporal_identity_top1_gate3c1d_v0.yaml"


def _atomic_json_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _result_payload_sha256(result: Mapping[str, Any]) -> str:
    value = dict(result)
    value.pop("result_payload_sha256", None)
    return canonical_json_sha256(value)


def _validate_parent(config: Mapping[str, Any]) -> None:
    parent = config["authorized_parent"]
    for path_key, hash_key in (
        ("result", "result_sha256"),
        ("summary", "summary_sha256"),
        ("replay", "replay_sha256"),
    ):
        if file_sha256(parent[path_key]) != parent[hash_key]:
            raise ValueError(f"Gate 3C1D parent hash drift: {path_key}")
    summary = json.loads(Path(parent["summary"]).read_text())
    if summary.get("summary_payload_sha256") != parent["summary_payload_sha256"]:
        raise ValueError("Gate 3C1D parent summary payload drift")
    replay = json.loads(Path(parent["replay"]).read_text())
    if (
        not bool(replay.get("exact_replay"))
        or not bool(replay.get("gate", {}).get("pass"))
        or replay.get("gate", {}).get("decision") != parent["required_decision"]
    ):
        raise ValueError("Gate 3C1C did not authorize Gate 3C1D")


def _validate_implementation(config: Mapping[str, Any]) -> None:
    for name, authority in config["implementation"].items():
        path = Path(authority["path"])
        if file_sha256(path) != authority["sha256"]:
            raise ValueError(f"Gate 3C1D implementation hash drift: {name}")


def _selector_config() -> QueryClosureIdentitySelectorConfig:
    return QueryClosureIdentitySelectorConfig(
        cycle_weight=2.0,
        query_frame_identity_weight=1.0,
        mean_identity_weight=1.0,
        minimum_identity_weight=1.0,
        retained_nonnative=8,
        zscore_epsilon=1.0e-4,
    )


def _feature_config(config: Mapping[str, Any]) -> Top1FeatureConfig:
    payload = config["feature_contract"]
    return Top1FeatureConfig(
        shortlist_size=int(payload["shortlist_size"]),
        temporal_summary_channels=tuple(
            int(value) for value in payload["temporal_summary_channels"]
        ),
        zscore_epsilon=float(payload["candidate_relative_zscore_epsilon"]),
    )


def _load_index(config: Mapping[str, Any], partition: str) -> dict[str, Any]:
    authority = config["cache_indices"][partition]
    path = Path(authority["path"])
    if file_sha256(path) != authority["file_sha256"]:
        raise ValueError(f"Gate 3C1D {partition} cache-index file hash drift")
    index = json.loads(path.read_text())
    expected = list(
        range(int(authority["source_indices"][0]), int(authority["source_indices"][1]) + 1)
    )
    if index.get("index_payload_sha256") != authority["payload_sha256"]:
        raise ValueError(f"Gate 3C1D {partition} cache-index payload drift")
    if index.get("config_sha256") != authority["config_sha256"]:
        raise ValueError(f"Gate 3C1D {partition} cache config drift")
    if index.get("partition") != partition:
        raise ValueError(f"Gate 3C1D {partition} partition drift")
    if index.get("completed_source_indices") != expected:
        raise ValueError(f"Gate 3C1D {partition} membership drift")
    if int(index.get("failure_rows", -1)) != int(authority["failure_rows"]):
        raise ValueError(f"Gate 3C1D {partition} failure-row drift")
    if not bool(index.get("checks", {}).get("all_sidecars_hash_verified")):
        raise ValueError(f"Gate 3C1D {partition} sidecar authority failed")
    return index


def _load_dataset(
    config: Mapping[str, Any], partition: str
) -> dict[str, torch.Tensor | list[str]]:
    index = _load_index(config, partition)
    selector_config = _selector_config()
    feature_config = _feature_config(config)
    features = []
    distances = []
    shortlist_indices = []
    groups = []
    point_indices = []
    video_names: list[str] = []
    feature_hashes = []
    selected_hashes = []
    actual_state = {
        key: bool(value) for key, value in index.get("read_state", {}).items()
    }
    if not actual_state:
        actual_state = {
            "checkpoint_selection_read": False,
            "fit_only_internal_audit_read": False,
            "original_model_validation_read": False,
            "external_read": False,
        }
    for index_row in index["rows"]:
        source_index = int(index_row["source_index"])
        sidecar = Path(index_row["sidecar"])
        if file_sha256(sidecar) != index_row["sidecar_sha256"]:
            raise ValueError(f"Gate 3C1D {partition} sidecar hash drift")
        payload = torch.load(sidecar, map_location="cpu", weights_only=False)
        verify_temporal_identity_cache_payload(
            payload,
            expected_partition=partition,
            expected_source_index=source_index,
            expected_config_sha256=index["config_sha256"],
            expected_read_state=actual_state,
        )
        tensors = payload["tensors"]
        rows = int(tensors["point_indices"].numel())
        if rows == 0:
            continue
        causal = build_shortlist_top1_features(
            temporal_features=tensors["temporal_features"].float(),
            static_features=tensors["static_features"].float(),
            query_frames=tensors["query_frames"].long(),
            valid_mask=tensors["candidate_valid_mask"].bool(),
            selector_config=selector_config,
            feature_config=feature_config,
        )
        candidate_features = causal["candidate_features"].cpu().contiguous()
        selected = causal["selected_indices"].cpu().contiguous()
        # Teacher distances are accessed only after causal features and shortlist indices freeze.
        teacher_distance = tensors["teacher_candidate_distance_px"].float()
        shortlist_distance = teacher_distance.gather(1, selected).cpu().contiguous()
        if torch.any(shortlist_distance < 0.0):
            raise ValueError("Gate 3C1D shortlist contains invalid teacher label")
        features.append(candidate_features)
        distances.append(shortlist_distance)
        shortlist_indices.append(selected)
        groups.append(torch.full((rows,), source_index, dtype=torch.long))
        point_indices.append(tensors["point_indices"].long().cpu())
        video_names.extend([str(payload["video_name"])] * rows)
        feature_hashes.append(tensor_sha256(candidate_features))
        selected_hashes.append(tensor_sha256(selected))
    output = {
        "features": torch.cat(features, dim=0),
        "distances": torch.cat(distances, dim=0),
        "shortlist_indices": torch.cat(shortlist_indices, dim=0),
        "groups": torch.cat(groups, dim=0),
        "point_indices": torch.cat(point_indices, dim=0),
        "video_names": video_names,
        "feature_digest": canonical_json_sha256(feature_hashes),
        "shortlist_digest": canonical_json_sha256(selected_hashes),
    }
    expected_rows = int(config["cache_indices"][partition]["failure_rows"])
    if int(output["features"].shape[0]) != expected_rows:
        raise ValueError(f"Gate 3C1D {partition} assembled row drift")
    return output


def _build_model(config: Mapping[str, Any]) -> HistGradientBoostingClassifier:
    model = config["model"]
    return HistGradientBoostingClassifier(
        max_iter=int(model["max_iter"]),
        learning_rate=float(model["learning_rate"]),
        max_leaf_nodes=int(model["max_leaf_nodes"]),
        l2_regularization=float(model["l2_regularization"]),
        min_samples_leaf=int(model["min_samples_leaf"]),
        random_state=int(model["random_state"]),
    )


def _fit_model(
    config: Mapping[str, Any], dataset: Mapping[str, Any]
) -> HistGradientBoostingClassifier:
    features = dataset["features"].numpy().reshape(-1, 102)
    labels = (dataset["distances"].numpy().reshape(-1) <= 12.0).astype(np.int64)
    model = _build_model(config)
    model.fit(features, labels)
    return model


def _predict(model: HistGradientBoostingClassifier, dataset: Mapping[str, Any]) -> np.ndarray:
    features = dataset["features"].numpy().reshape(-1, 102)
    probability = model.predict_proba(features)[:, 1].reshape(-1, 9)
    if not np.isfinite(probability).all():
        raise ValueError("Gate 3C1D non-finite probability")
    return probability


def _video_cluster_ci(
    groups: np.ndarray, values: np.ndarray, *, seed: int, samples: int
) -> dict[str, float | int]:
    unique = np.unique(groups)
    per_video = np.array([values[groups == value].mean() for value in unique], dtype=np.float64)
    generator = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        bootstrap[index] = generator.choice(per_video, len(per_video), replace=True).mean()
    return {
        "mean": float(per_video.mean()),
        "lower": float(np.quantile(bootstrap, 0.025)),
        "upper": float(np.quantile(bootstrap, 0.975)),
        "videos": int(len(unique)),
        "samples": int(samples),
        "seed": int(seed),
    }


def _policy_metrics(
    *,
    dataset: Mapping[str, Any],
    probability: np.ndarray,
    threshold: float,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    distance = dataset["distances"].numpy()
    groups = dataset["groups"].numpy()
    shortlist = dataset["shortlist_indices"].numpy()
    points = dataset["point_indices"].numpy()
    selected_slot = probability.argmax(axis=1)
    row = np.arange(len(selected_slot))
    selected_probability = probability[row, selected_slot]
    selected_distance = distance[row, selected_slot]
    action = (selected_slot != 0) & (selected_probability >= float(threshold))
    output_slot = np.where(action, selected_slot, 0)
    output_distance = distance[row, output_slot]
    native_distance = distance[:, 0]
    error_reduction = native_distance - output_distance
    candidate_labels = (distance <= 12.0).astype(np.int64).reshape(-1)
    candidate_probability = probability.reshape(-1)
    auc = float(roc_auc_score(candidate_labels, candidate_probability))
    ap = float(average_precision_score(candidate_labels, candidate_probability))
    action_count = int(action.sum())
    action_precision = (
        0.0 if action_count == 0 else float((selected_distance[action] <= 12.0).mean())
    )
    harmful_all = float((output_distance > native_distance + 4.0).mean())
    harmful_action = (
        0.0
        if action_count == 0
        else float((selected_distance[action] > native_distance[action] + 4.0).mean())
    )
    video_ci = _video_cluster_ci(
        groups,
        error_reduction,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
    )
    unique = np.unique(groups)
    nonnegative_video_fraction = float(
        np.mean([error_reduction[groups == value].mean() >= 0.0 for value in unique])
    )
    metrics = {
        "rows": int(len(output_distance)),
        "candidate_AUC": auc,
        "candidate_AP": ap,
        "raw_top1_within_12px": float((selected_distance <= 12.0).mean()),
        "action_threshold": float(threshold),
        "action_rows": action_count,
        "action_coverage": float(action.mean()),
        "action_precision_within_12px": action_precision,
        "overall_within_12px": float((output_distance <= 12.0).mean()),
        "native_within_12px": float((native_distance <= 12.0).mean()),
        "mean_native_commit_error_px": float(native_distance.mean()),
        "mean_policy_commit_error_px": float(output_distance.mean()),
        "median_policy_commit_error_px": float(np.median(output_distance)),
        "mean_commit_error_reduction_px": float(error_reduction.mean()),
        "video_cluster_error_reduction_CI": video_ci,
        "nonnegative_video_fraction": nonnegative_video_fraction,
        "harmful_all_rows_gt_native_plus_4px": harmful_all,
        "harmful_action_fraction": harmful_action,
        "selected_probability_mean": float(selected_probability.mean()),
        "selected_probability_action_mean": (
            0.0 if action_count == 0 else float(selected_probability[action].mean())
        ),
    }
    records = []
    video_names = dataset["video_names"]
    for index in range(len(output_distance)):
        records.append(
            {
                "source_index": int(groups[index]),
                "video_name": str(video_names[index]),
                "point_index": int(points[index]),
                "selected_shortlist_slot": int(selected_slot[index]),
                "selected_candidate_index": int(shortlist[index, selected_slot[index]]),
                "selected_probability": float(selected_probability[index]),
                "action": bool(action[index]),
                "output_candidate_index": int(shortlist[index, output_slot[index]]),
                "native_distance_px": float(native_distance[index]),
                "selected_distance_px": float(selected_distance[index]),
                "output_distance_px": float(output_distance[index]),
                "commit_error_reduction_px": float(error_reduction[index]),
            }
        )
    return metrics, records, output_slot


def _checkpoint_conditions(metrics: Mapping[str, Any], gates: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "action_coverage": float(metrics["action_coverage"])
        >= float(gates["action_coverage_min"]),
        "action_precision": float(metrics["action_precision_within_12px"])
        >= float(gates["action_precision_within_12px_min"]),
        "mean_error_reduction": float(metrics["mean_commit_error_reduction_px"])
        >= float(gates["mean_commit_error_reduction_px_min"]),
        "error_CI_lower": float(metrics["video_cluster_error_reduction_CI"]["lower"])
        >= float(gates["video_cluster_error_reduction_CI_lower_px_min"]),
        "harmful_all_rows": float(metrics["harmful_all_rows_gt_native_plus_4px"])
        <= float(gates["harmful_all_rows_gt_native_plus_4px_max"]),
    }


def select_checkpoint_threshold(
    *,
    config: Mapping[str, Any],
    dataset: Mapping[str, Any],
    probability: np.ndarray,
) -> tuple[float | None, list[dict[str, Any]]]:
    selection = config["threshold_selection"]
    records = []
    chosen = None
    for threshold in selection["fixed_grid"]:
        metrics, _, _ = _policy_metrics(
            dataset=dataset,
            probability=probability,
            threshold=float(threshold),
            bootstrap_seed=int(config["bootstrap"]["seed"]),
            bootstrap_samples=int(config["bootstrap"]["samples"]),
        )
        checks = _checkpoint_conditions(metrics, selection["gates"])
        records.append({"threshold": float(threshold), "metrics": metrics, "checks": checks})
        if chosen is None and all(checks.values()):
            chosen = float(threshold)
    return chosen, records


def _confirmation_checks(
    metrics: Mapping[str, Any], gates: Mapping[str, Any], exact_replay: bool
) -> dict[str, bool]:
    return {
        "action_coverage": float(metrics["action_coverage"])
        >= float(gates["action_coverage_min"]),
        "action_precision": float(metrics["action_precision_within_12px"])
        >= float(gates["action_precision_within_12px_min"]),
        "mean_error_reduction": float(metrics["mean_commit_error_reduction_px"])
        >= float(gates["mean_commit_error_reduction_px_min"]),
        "error_CI_lower": float(metrics["video_cluster_error_reduction_CI"]["lower"])
        >= float(gates["video_cluster_error_reduction_CI_lower_px_min"]),
        "harmful_all_rows": float(metrics["harmful_all_rows_gt_native_plus_4px"])
        <= float(gates["harmful_all_rows_gt_native_plus_4px_max"]),
        "candidate_AUC": float(metrics["candidate_AUC"])
        >= float(gates["candidate_AUC_min"]),
        "candidate_AP": float(metrics["candidate_AP"])
        >= float(gates["candidate_AP_min"]),
        "exact_replay": bool(exact_replay),
    }


def _validate_authorization(
    config: Mapping[str, Any], partition: str
) -> dict[str, Any]:
    authority = config["authorization_chain"][partition]
    path = Path(authority["required_result"])
    if not path.is_file():
        raise ValueError(f"Gate 3C1D {partition} authorization result is absent")
    result = json.loads(path.read_text())
    if result.get("result_payload_sha256") != _result_payload_sha256(result):
        raise ValueError(f"Gate 3C1D {partition} authorization payload drift")
    if result.get("config_sha256") != file_sha256(config["_config_path"]):
        raise ValueError(f"Gate 3C1D {partition} authorization config drift")
    if result.get("partition") != authority["required_partition"]:
        raise ValueError(f"Gate 3C1D {partition} authorization partition drift")
    if not bool(result.get("exact_replay")) or not bool(result.get("gate", {}).get("pass")):
        raise ValueError(f"Gate 3C1D {partition} authorization replay failed")
    if result.get("gate", {}).get("decision") != authority["required_decision"]:
        raise ValueError(f"Gate 3C1D {partition} authorization decision drift")
    return result


def _model_file_for_run(config: Mapping[str, Any], reference: Path | None) -> Path:
    return Path(
        config["artifacts"]["primary_model" if reference is None else "replay_model"]
    )


def run(
    *,
    config_path: Path,
    partition: str,
    output_path: Path,
    reference_path: Path | None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text())
    config["_config_path"] = str(config_path)
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected Gate 3C1D config")
    if any(value is not False for value in config["locked_data"].values()):
        raise ValueError("Gate 3C1D locked-data flags must remain false")
    _validate_parent(config)
    _validate_implementation(config)
    if partition not in (
        "checkpoint_selection",
        "fit_only_internal_audit",
        "original_model_validation",
    ):
        raise ValueError("Gate 3C1D partition is not authorized")

    reference = None if reference_path is None else json.loads(reference_path.read_text())
    frozen_model_artifact = None
    if partition == "checkpoint_selection":
        train = _load_dataset(config, "gradient_train")
        target = _load_dataset(config, partition)
        model = _fit_model(config, train)
        model_path = _model_file_for_run(config, reference_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        probability = _predict(model, target)
        threshold, threshold_records = select_checkpoint_threshold(
            config=config, dataset=target, probability=probability
        )
        if threshold is None:
            threshold = float(config["threshold_selection"]["fixed_grid"][-1])
        metrics, records, output_slot = _policy_metrics(
            dataset=target,
            probability=probability,
            threshold=threshold,
            bootstrap_seed=int(config["bootstrap"]["seed"]),
            bootstrap_samples=int(config["bootstrap"]["samples"]),
        )
        threshold_checks = _checkpoint_conditions(
            metrics, config["threshold_selection"]["gates"]
        )
        model_artifact = {
            "path": str(model_path),
            "file_sha256": file_sha256(model_path),
        }
        frozen_model_artifact = model_artifact if reference is None else reference[
            "frozen_model_artifact"
        ]
        frozen_threshold = float(threshold)
    else:
        authorization = _validate_authorization(config, partition)
        frozen_model_artifact = authorization["frozen_model_artifact"]
        model_path = Path(frozen_model_artifact["path"])
        if file_sha256(model_path) != frozen_model_artifact["file_sha256"]:
            raise ValueError("Gate 3C1D frozen model hash drift")
        model = joblib.load(model_path)
        target = _load_dataset(config, partition)
        probability = _predict(model, target)
        frozen_threshold = float(authorization["frozen_threshold"])
        metrics, records, output_slot = _policy_metrics(
            dataset=target,
            probability=probability,
            threshold=frozen_threshold,
            bootstrap_seed=int(config["bootstrap"]["seed"]),
            bootstrap_samples=int(config["bootstrap"]["samples"]),
        )
        threshold_records = None
        threshold_checks = None
        model_artifact = frozen_model_artifact

    probability_tensor = torch.from_numpy(probability.astype(np.float64))
    scientific = {
        "partition": partition,
        "feature_digest": target["feature_digest"],
        "shortlist_digest": target["shortlist_digest"],
        "probability_digest": tensor_sha256(probability_tensor),
        "output_slot_digest": tensor_sha256(torch.from_numpy(output_slot.astype(np.int64))),
        "frozen_threshold": frozen_threshold,
        "metrics": metrics,
        "threshold_grid_digest": (
            None if threshold_records is None else canonical_json_sha256(threshold_records)
        ),
        "records_digest": canonical_json_sha256(records),
    }
    scientific["scientific_payload_sha256"] = canonical_json_sha256(scientific)

    replay_comparison = None
    exact_replay = False
    if reference is not None:
        reference_scientific = reference["scientific"]
        replay_comparison = {
            "feature_digest": scientific["feature_digest"]
            == reference_scientific["feature_digest"],
            "shortlist_digest": scientific["shortlist_digest"]
            == reference_scientific["shortlist_digest"],
            "probability_digest": scientific["probability_digest"]
            == reference_scientific["probability_digest"],
            "output_slot_digest": scientific["output_slot_digest"]
            == reference_scientific["output_slot_digest"],
            "frozen_threshold": scientific["frozen_threshold"]
            == reference_scientific["frozen_threshold"],
            "threshold_grid_digest": scientific["threshold_grid_digest"]
            == reference_scientific["threshold_grid_digest"],
            "records_digest": scientific["records_digest"]
            == reference_scientific["records_digest"],
            "scientific_payload_sha256": scientific["scientific_payload_sha256"]
            == reference_scientific["scientific_payload_sha256"],
        }
        exact_replay = all(replay_comparison.values())

    if partition == "checkpoint_selection":
        checks = dict(threshold_checks)
        checks["threshold_found"] = any(
            all(record["checks"].values()) for record in threshold_records
        )
        checks["exact_replay"] = exact_replay
        if reference is None:
            passed = False
            decision = "PRIMARY_COMPLETE_AWAIT_EXACT_REPLAY"
        else:
            passed = all(checks.values())
            decision = (
                "AUTHORIZE_GATE3C1D_FIT_ONLY_AUDIT"
                if passed
                else "STOP_GATE3C1D_TOP1_SELECTOR"
            )
    else:
        gates = config["confirmation_gates"][partition]
        checks = _confirmation_checks(metrics, gates, exact_replay)
        if reference is None:
            passed = False
            decision = "PRIMARY_COMPLETE_AWAIT_EXACT_REPLAY"
        else:
            passed = all(checks.values())
            decision = gates["decision_pass" if passed else "decision_fail"]

    result = {
        "schema_version": RESULT_SCHEMA,
        "date": "2026-07-20",
        "status": "completed",
        "partition": partition,
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "model_artifact": model_artifact,
        "frozen_model_artifact": frozen_model_artifact,
        "frozen_threshold": frozen_threshold,
        "threshold_grid_records": threshold_records,
        "scientific": scientific,
        "records": records,
        "reference": None if reference_path is None else str(reference_path),
        "replay_comparison": replay_comparison,
        "exact_replay": exact_replay,
        "gate": {"checks": checks, "pass": passed, "decision": decision},
        "locked_data": config["locked_data"],
        "claim_boundary": config["claim_scope"],
        "runtime_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    result["result_payload_sha256"] = _result_payload_sha256(result)
    _atomic_json_save(result, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--partition",
        required=True,
        choices=[
            "checkpoint_selection",
            "fit_only_internal_audit",
            "original_model_validation",
        ],
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", default=None)
    args = parser.parse_args()
    result = run(
        config_path=Path(args.config).resolve(),
        partition=str(args.partition),
        output_path=Path(args.output).resolve(),
        reference_path=(None if args.reference is None else Path(args.reference).resolve()),
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "partition": result["partition"],
                "frozen_threshold": result["frozen_threshold"],
                "metrics": result["scientific"]["metrics"],
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
