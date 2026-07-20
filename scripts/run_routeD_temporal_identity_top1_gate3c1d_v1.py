#!/usr/bin/env python3
"""Train and confirm the two-stage causal Gate 3C1D v1 top-1 policy."""
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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import canonical_json_sha256, file_sha256
from projects.mmp_tracker.mmp_tracker.routeD_temporal_identity_selector import QueryClosureIdentitySelectorConfig
from projects.mmp_tracker.mmp_tracker.routeD_temporal_identity_top1 import Top1FeatureConfig, build_shortlist_top1_features
from projects.mmp_tracker.mmp_tracker.routeD_temporal_identity_top1_v1 import (
    RowPolicyFeatureConfig,
    build_row_policy_features,
    select_minimum_expected_distance,
)
from scripts.build_routeD_temporal_identity_train_cache_gate3c1a_v0 import verify_temporal_identity_cache_payload

SCHEMA = "routeD_temporal_identity_top1_gate3c1d_v1"
RESULT_SCHEMA = "routeD_temporal_identity_top1_result_gate3c1d_v1"
CACHE_INDEX_SCHEMA = "routeD_temporal_identity_eval_cache_index_gate3c1d_v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_temporal_identity_top1_gate3c1d_v1.yaml"
PARTITIONS = ("checkpoint_selection_v1", "fit_only_internal_audit_v1", "model_validation_v1")


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
    for key, hash_key in (("result", "result_sha256"), ("summary", "summary_sha256")):
        if file_sha256(parent[key]) != parent[hash_key]:
            raise ValueError(f"Gate 3C1D v1 parent hash drift: {key}")
    summary = json.loads(Path(parent["summary"]).read_text())
    without_hash = dict(summary)
    payload_hash = without_hash.pop("summary_payload_sha256", None)
    if (
        payload_hash != parent["summary_payload_sha256"]
        or payload_hash != canonical_json_sha256(without_hash)
        or not bool(summary.get("pass"))
        or summary.get("formal_decision") != parent["required_decision"]
    ):
        raise ValueError("Gate 3C2 did not authorize Gate 3C1D v1")


def _validate_implementation(config: Mapping[str, Any]) -> None:
    for name, authority in config["implementation"].items():
        if file_sha256(authority["path"]) != authority["sha256"]:
            raise ValueError(f"Gate 3C1D v1 implementation hash drift: {name}")


def _validate_runtime(config: Mapping[str, Any]) -> None:
    actual = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    expected = {key: str(value) for key, value in config["runtime_contract"].items()}
    if actual != expected:
        raise ValueError(f"Gate 3C1D v1 runtime version drift: {actual} != {expected}")


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
        temporal_summary_channels=tuple(int(v) for v in payload["temporal_summary_channels"]),
        zscore_epsilon=float(payload["candidate_relative_zscore_epsilon"]),
    )


def _old_index(config: Mapping[str, Any], partition: str) -> dict[str, Any]:
    authority = config["development_cache_indices"][partition]
    path = Path(authority["path"])
    if file_sha256(path) != authority["file_sha256"]:
        raise ValueError(f"Gate 3C1D v1 old {partition} index file drift")
    index = json.loads(path.read_text())
    expected = list(range(int(authority["source_indices"][0]), int(authority["source_indices"][1]) + 1))
    if (
        index.get("index_payload_sha256") != authority["payload_sha256"]
        or index.get("config_sha256") != authority["config_sha256"]
        or index.get("partition") != partition
        or index.get("completed_source_indices") != expected
        or int(index.get("failure_rows", -1)) != int(authority["failure_rows"])
    ):
        raise ValueError(f"Gate 3C1D v1 old {partition} authority drift")
    return index


def _renewed_index(config: Mapping[str, Any], partition: str) -> tuple[dict[str, Any], str]:
    authority = config["renewed_cache_indices"][partition]
    cache_config = Path(authority["config"])
    if file_sha256(cache_config) != authority["config_sha256"]:
        raise ValueError(f"Gate 3C1D v1 {partition} cache config drift")
    path = Path(authority["path"])
    if not path.is_file():
        raise ValueError(f"Gate 3C1D v1 {partition} cache index is absent")
    index_file_sha = file_sha256(path)
    index = json.loads(path.read_text())
    expected = list(range(int(authority["source_indices"][0]), int(authority["source_indices"][1]) + 1))
    without_hash = dict(index)
    payload_hash = without_hash.pop("index_payload_sha256", None)
    if (
        index.get("schema_version") != CACHE_INDEX_SCHEMA
        or not bool(index.get("pass"))
        or index.get("config_sha256") != authority["config_sha256"]
        or index.get("partition") != partition
        or index.get("completed_source_indices") != expected
        or int(index.get("videos", -1)) != len(expected)
        or payload_hash != canonical_json_sha256(without_hash)
        or not all(index.get("checks", {}).values())
    ):
        raise ValueError(f"Gate 3C1D v1 {partition} cache index drift")
    return index, index_file_sha


def _assemble_dataset(
    *, config: Mapping[str, Any], partition: str, index: Mapping[str, Any]
) -> dict[str, Any]:
    selector_config = _selector_config()
    feature_config = _feature_config(config)
    features: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    shortlist_indices: list[torch.Tensor] = []
    groups: list[torch.Tensor] = []
    point_indices: list[torch.Tensor] = []
    video_names: list[str] = []
    feature_hashes: list[str] = []
    shortlist_hashes: list[str] = []
    read_state = {key: bool(value) for key, value in index.get("read_state", {}).items()}
    for index_row in index["rows"]:
        source_index = int(index_row["source_index"])
        sidecar = Path(index_row["sidecar"])
        if file_sha256(sidecar) != index_row["sidecar_sha256"]:
            raise ValueError(f"Gate 3C1D v1 {partition} sidecar hash drift")
        payload = torch.load(sidecar, map_location="cpu", weights_only=False)
        verify_temporal_identity_cache_payload(
            payload,
            expected_partition=partition,
            expected_source_index=source_index,
            expected_config_sha256=index["config_sha256"],
            expected_read_state=read_state,
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
        teacher_distance = tensors["teacher_candidate_distance_px"].float()
        shortlist_distance = teacher_distance.gather(1, selected).cpu().contiguous()
        if torch.any(shortlist_distance < 0.0):
            raise ValueError("Gate 3C1D v1 shortlist contains invalid label")
        features.append(candidate_features)
        distances.append(shortlist_distance)
        shortlist_indices.append(selected)
        groups.append(torch.full((rows,), source_index, dtype=torch.long))
        point_indices.append(tensors["point_indices"].long().cpu())
        video_names.extend([str(payload["video_name"])] * rows)
        feature_hashes.append(tensor_sha256(candidate_features))
        shortlist_hashes.append(tensor_sha256(selected))
    if not features:
        raise ValueError(f"Gate 3C1D v1 {partition} has no failure rows")
    output = {
        "features": torch.cat(features),
        "distances": torch.cat(distances),
        "shortlist_indices": torch.cat(shortlist_indices),
        "groups": torch.cat(groups),
        "point_indices": torch.cat(point_indices),
        "video_names": video_names,
        "feature_digest": canonical_json_sha256(feature_hashes),
        "shortlist_digest": canonical_json_sha256(shortlist_hashes),
    }
    if int(output["features"].shape[0]) != int(index["failure_rows"]):
        raise ValueError(f"Gate 3C1D v1 {partition} assembled row drift")
    return output


def _load_development(config: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for partition in config["development_order"]:
        rows.append(_assemble_dataset(config=config, partition=partition, index=_old_index(config, partition)))
    output = {
        key: torch.cat([row[key] for row in rows])
        for key in ("features", "distances", "shortlist_indices", "groups", "point_indices")
    }
    output["video_names"] = sum((row["video_names"] for row in rows), [])
    output["feature_digest"] = canonical_json_sha256([row["feature_digest"] for row in rows])
    output["shortlist_digest"] = canonical_json_sha256([row["shortlist_digest"] for row in rows])
    if int(output["features"].shape[0]) != int(config["development_expected_rows"]):
        raise ValueError("Gate 3C1D v1 development row drift")
    return output


def _candidate_classifier(config: Mapping[str, Any]) -> HistGradientBoostingClassifier:
    p = config["models"]["candidate_support"]
    return HistGradientBoostingClassifier(
        max_iter=int(p["max_iter"]), learning_rate=float(p["learning_rate"]),
        max_leaf_nodes=int(p["max_leaf_nodes"]), l2_regularization=float(p["l2_regularization"]),
        min_samples_leaf=int(p["min_samples_leaf"]), random_state=int(p["random_state"]),
        early_stopping=False,
    )


def _candidate_regressor(config: Mapping[str, Any]) -> HistGradientBoostingRegressor:
    p = config["models"]["candidate_distance"]
    return HistGradientBoostingRegressor(
        max_iter=int(p["max_iter"]), learning_rate=float(p["learning_rate"]),
        max_leaf_nodes=int(p["max_leaf_nodes"]), l2_regularization=float(p["l2_regularization"]),
        min_samples_leaf=int(p["min_samples_leaf"]), random_state=int(p["random_state"]),
        early_stopping=False, loss="squared_error",
    )


def _value_regressor(config: Mapping[str, Any]) -> HistGradientBoostingRegressor:
    p = config["models"]["row_value"]
    return HistGradientBoostingRegressor(
        max_iter=int(p["max_iter"]), learning_rate=float(p["learning_rate"]),
        max_leaf_nodes=int(p["max_leaf_nodes"]), l2_regularization=float(p["l2_regularization"]),
        min_samples_leaf=int(p["min_samples_leaf"]), random_state=int(p["random_state"]),
        early_stopping=False, loss="squared_error",
    )


def _harm_classifier(config: Mapping[str, Any]) -> HistGradientBoostingClassifier:
    p = config["models"]["row_harm"]
    return HistGradientBoostingClassifier(
        max_iter=int(p["max_iter"]), learning_rate=float(p["learning_rate"]),
        max_leaf_nodes=int(p["max_leaf_nodes"]), l2_regularization=float(p["l2_regularization"]),
        min_samples_leaf=int(p["min_samples_leaf"]), random_state=int(p["random_state"]),
        early_stopping=False,
    )


def _fit_candidate(config: Mapping[str, Any], features: np.ndarray, distance: np.ndarray):
    x = features.reshape(-1, features.shape[-1])
    d = distance.reshape(-1)
    support = _candidate_classifier(config).fit(x, (d <= 12.0).astype(np.int64))
    expected = _candidate_regressor(config).fit(x, np.log1p(np.clip(d, 0.0, 64.0)))
    return support, expected


def _predict_candidate(models, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = features.reshape(-1, features.shape[-1])
    support = models[0].predict_proba(x)[:, 1].reshape(features.shape[:2])
    expected = np.expm1(models[1].predict(x)).reshape(features.shape[:2]).clip(0.0, 64.0)
    if not np.isfinite(support).all() or not np.isfinite(expected).all():
        raise ValueError("Gate 3C1D v1 candidate prediction is non-finite")
    return support, expected


def _train_bundle(config: Mapping[str, Any], development: Mapping[str, Any]) -> dict[str, Any]:
    x = development["features"].numpy().astype(np.float32)
    d = development["distances"].numpy().astype(np.float32)
    groups = development["groups"].numpy()
    folds = int(config["oof_contract"]["folds"])
    oof_support = np.zeros(d.shape, dtype=np.float64)
    oof_expected = np.zeros(d.shape, dtype=np.float64)
    splitter = GroupKFold(folds)
    for train, valid in splitter.split(np.zeros(len(groups)), groups=groups):
        models = _fit_candidate(config, x[train], d[train])
        oof_support[valid], oof_expected[valid] = _predict_candidate(models, x[valid])
    selected = select_minimum_expected_distance(oof_expected)
    row_features = build_row_policy_features(
        candidate_features=x,
        support_probability=oof_support,
        expected_distance_px=oof_expected,
        selected_slot=selected,
    )
    row = np.arange(len(selected))
    value_target = np.clip(d[:, 0] - d[row, selected], -64.0, 64.0)
    harm_target = (d[row, selected] > d[:, 0] + 4.0).astype(np.int64)
    value_model = _value_regressor(config).fit(row_features, value_target)
    harm_model = _harm_classifier(config).fit(row_features, harm_target)
    final_support, final_expected = _fit_candidate(config, x, d)
    return {
        "schema_version": "routeD_temporal_identity_top1_bundle_gate3c1d_v1",
        "candidate_support": final_support,
        "candidate_distance": final_expected,
        "row_value": value_model,
        "row_harm": harm_model,
        "oof_feature_digest": tensor_sha256(torch.from_numpy(row_features)),
        "oof_support_digest": tensor_sha256(torch.from_numpy(oof_support)),
        "oof_expected_distance_digest": tensor_sha256(torch.from_numpy(oof_expected)),
        "oof_selected_slot_digest": tensor_sha256(torch.from_numpy(selected)),
        "development_feature_digest": development["feature_digest"],
        "development_shortlist_digest": development["shortlist_digest"],
    }


def _predict_bundle(bundle: Mapping[str, Any], dataset: Mapping[str, Any]) -> dict[str, np.ndarray]:
    x = dataset["features"].numpy().astype(np.float32)
    support, expected = _predict_candidate(
        (bundle["candidate_support"], bundle["candidate_distance"]), x
    )
    selected = select_minimum_expected_distance(expected)
    row_features = build_row_policy_features(
        candidate_features=x, support_probability=support,
        expected_distance_px=expected, selected_slot=selected,
    )
    value = bundle["row_value"].predict(row_features)
    harm = bundle["row_harm"].predict_proba(row_features)[:, 1]
    if not np.isfinite(value).all() or not np.isfinite(harm).all():
        raise ValueError("Gate 3C1D v1 row prediction is non-finite")
    return {"support": support, "expected": expected, "selected": selected, "row_features": row_features, "value": value, "harm": harm}


def _video_ci(groups: np.ndarray, values: np.ndarray, *, seed: int, samples: int) -> dict[str, Any]:
    unique = np.unique(groups)
    per_video = np.array([values[groups == group].mean() for group in unique], dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        bootstrap[index] = rng.choice(per_video, len(per_video), replace=True).mean()
    return {"mean": float(per_video.mean()), "lower": float(np.quantile(bootstrap, .025)), "upper": float(np.quantile(bootstrap, .975)), "videos": int(len(unique)), "samples": int(samples), "seed": int(seed)}


def _metrics(
    *, dataset: Mapping[str, Any], predictions: Mapping[str, np.ndarray], policy: Mapping[str, float],
    bootstrap_seed: int, bootstrap_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    d = dataset["distances"].numpy(); groups = dataset["groups"].numpy(); shortlist = dataset["shortlist_indices"].numpy(); points = dataset["point_indices"].numpy()
    slot = predictions["selected"]; row = np.arange(len(slot)); selected_d = d[row, slot]; native_d = d[:, 0]
    selected_support = predictions["support"][row, slot]
    action = (slot != 0) & (selected_support >= float(policy["support_min"])) & (predictions["value"] >= float(policy["value_min_px"])) & (predictions["harm"] <= float(policy["harm_max"]))
    output_slot = np.where(action, slot, 0); output_d = d[row, output_slot]; reduction = native_d - output_d; ac = int(action.sum())
    labels = (d <= 12.0).reshape(-1).astype(np.int64); prob = predictions["support"].reshape(-1)
    ci = _video_ci(groups, reduction, seed=bootstrap_seed, samples=bootstrap_samples)
    unique = np.unique(groups)
    metrics = {
        "rows": int(len(slot)), "candidate_AUC": float(roc_auc_score(labels, prob)), "candidate_AP": float(average_precision_score(labels, prob)),
        "raw_top1_within_12px": float((selected_d <= 12.0).mean()), "raw_top1_mean_error_px": float(selected_d.mean()), "raw_top1_median_error_px": float(np.median(selected_d)),
        "action_rows": ac, "action_coverage": float(action.mean()), "action_precision_within_12px": 0.0 if ac == 0 else float((selected_d[action] <= 12.0).mean()),
        "overall_within_12px": float((output_d <= 12.0).mean()), "native_within_12px": float((native_d <= 12.0).mean()),
        "mean_native_commit_error_px": float(native_d.mean()), "mean_policy_commit_error_px": float(output_d.mean()), "median_policy_commit_error_px": float(np.median(output_d)),
        "mean_commit_error_reduction_px": float(reduction.mean()), "video_cluster_error_reduction_CI": ci,
        "nonnegative_video_fraction": float(np.mean([reduction[groups == group].mean() >= 0.0 for group in unique])),
        "harmful_all_rows_gt_native_plus_4px": float((output_d > native_d + 4.0).mean()),
        "harmful_action_fraction": 0.0 if ac == 0 else float((selected_d[action] > native_d[action] + 4.0).mean()),
        "selected_support_mean": float(selected_support.mean()), "selected_support_action_mean": 0.0 if ac == 0 else float(selected_support[action].mean()),
        "predicted_value_mean": float(predictions["value"].mean()), "predicted_harm_mean": float(predictions["harm"].mean()),
    }
    records = []
    for index in range(len(slot)):
        records.append({
            "source_index": int(groups[index]), "video_name": str(dataset["video_names"][index]), "point_index": int(points[index]),
            "selected_shortlist_slot": int(slot[index]), "selected_candidate_index": int(shortlist[index, slot[index]]),
            "selected_support_probability": float(selected_support[index]), "predicted_value_px": float(predictions["value"][index]), "predicted_harm_probability": float(predictions["harm"][index]),
            "action": bool(action[index]), "output_candidate_index": int(shortlist[index, output_slot[index]]),
            "native_distance_px": float(native_d[index]), "selected_distance_px": float(selected_d[index]), "output_distance_px": float(output_d[index]), "commit_error_reduction_px": float(reduction[index]),
        })
    return metrics, records, output_slot


def _checks(metrics: Mapping[str, Any], gates: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "candidate_AUC": float(metrics["candidate_AUC"]) >= float(gates["candidate_AUC_min"]),
        "candidate_AP": float(metrics["candidate_AP"]) >= float(gates["candidate_AP_min"]),
        "raw_top1": float(metrics["raw_top1_within_12px"]) >= float(gates["raw_top1_within_12px_min"]),
        "action_coverage": float(metrics["action_coverage"]) >= float(gates["action_coverage_min"]),
        "action_precision": float(metrics["action_precision_within_12px"]) >= float(gates["action_precision_within_12px_min"]),
        "mean_error_reduction": float(metrics["mean_commit_error_reduction_px"]) >= float(gates["mean_commit_error_reduction_px_min"]),
        "error_CI_lower": float(metrics["video_cluster_error_reduction_CI"]["lower"]) >= float(gates["video_cluster_error_reduction_CI_lower_px_min"]),
        "harmful_all_rows": float(metrics["harmful_all_rows_gt_native_plus_4px"]) <= float(gates["harmful_all_rows_gt_native_plus_4px_max"]),
        "harmful_action": float(metrics["harmful_action_fraction"]) <= float(gates["harmful_action_fraction_max"]),
        "nonnegative_video_fraction": float(metrics["nonnegative_video_fraction"]) >= float(gates["nonnegative_video_fraction_min"]),
    }


def _select_policy(config: Mapping[str, Any], dataset: Mapping[str, Any], predictions: Mapping[str, np.ndarray]) -> tuple[dict[str, float] | None, list[dict[str, Any]]]:
    selection = config["policy_selection"]
    records = []
    for support in selection["support_min_grid"]:
        for value in selection["value_min_px_grid"]:
            for harm in selection["harm_max_grid"]:
                policy = {"support_min": float(support), "value_min_px": float(value), "harm_max": float(harm)}
                metrics, _, _ = _metrics(dataset=dataset, predictions=predictions, policy=policy, bootstrap_seed=int(config["bootstrap"]["seed"]), bootstrap_samples=int(config["bootstrap"]["selection_samples"]))
                checks = _checks(metrics, selection["gates"])
                records.append({"policy": policy, "metrics": metrics, "checks": checks})
    passing = [record for record in records if all(record["checks"].values())]
    if not passing:
        return None, records
    passing.sort(key=lambda record: (-record["metrics"]["action_coverage"], record["policy"]["harm_max"], -record["policy"]["support_min"], -record["policy"]["value_min_px"]))
    return passing[0]["policy"], records


def _validate_authorization(config: Mapping[str, Any], partition: str) -> dict[str, Any]:
    authority = config["authorization_chain"][partition]
    path = Path(authority["required_result"])
    if not path.is_file():
        raise ValueError(f"Gate 3C1D v1 {partition} authorization absent")
    result = json.loads(path.read_text())
    if result.get("result_payload_sha256") != _result_payload_sha256(result):
        raise ValueError(f"Gate 3C1D v1 {partition} authorization payload drift")
    if result.get("config_sha256") != file_sha256(config["_config_path"]):
        raise ValueError(f"Gate 3C1D v1 {partition} authorization config drift")
    if result.get("partition") != authority["required_partition"] or not bool(result.get("exact_replay")) or not bool(result.get("gate", {}).get("pass")) or result.get("gate", {}).get("decision") != authority["required_decision"]:
        raise ValueError(f"Gate 3C1D v1 {partition} authorization failed")
    return result


def run(*, config_path: Path, partition: str, output_path: Path, reference_path: Path | None) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text()); config["_config_path"] = str(config_path)
    if config.get("schema_version") != SCHEMA or partition not in PARTITIONS:
        raise ValueError("unexpected Gate 3C1D v1 run")
    if any(value is not False for value in config["locked_data"].values()):
        raise ValueError("Gate 3C1D v1 locked-data flags must remain false")
    _validate_parent(config); _validate_implementation(config); _validate_runtime(config)
    target_index, target_index_sha = _renewed_index(config, partition)
    target = _assemble_dataset(config=config, partition=partition, index=target_index)
    reference = None if reference_path is None else json.loads(reference_path.read_text())

    if partition == "checkpoint_selection_v1":
        development = _load_development(config)
        bundle = _train_bundle(config, development)
        bundle_path = Path(config["artifacts"]["primary_bundle" if reference is None else "replay_bundle"])
        bundle_path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(bundle, bundle_path)
        bundle_artifact = {"path": str(bundle_path), "file_sha256": file_sha256(bundle_path)}
        frozen_bundle = bundle_artifact if reference is None else reference["frozen_bundle_artifact"]
        predictions = _predict_bundle(bundle, target)
        policy, grid = _select_policy(config, target, predictions)
        policy_found = policy is not None
        if policy is None:
            policy = {"support_min": float(config["policy_selection"]["support_min_grid"][-1]), "value_min_px": float(config["policy_selection"]["value_min_px_grid"][-1]), "harm_max": float(config["policy_selection"]["harm_max_grid"][0])}
        frozen_policy = policy
    else:
        authorization = _validate_authorization(config, partition)
        frozen_bundle = authorization["frozen_bundle_artifact"]
        bundle_path = Path(frozen_bundle["path"])
        if file_sha256(bundle_path) != frozen_bundle["file_sha256"]:
            raise ValueError("Gate 3C1D v1 frozen bundle hash drift")
        bundle = joblib.load(bundle_path); bundle_artifact = frozen_bundle
        frozen_policy = authorization["frozen_policy"]
        predictions = _predict_bundle(bundle, target); grid = None; policy_found = True

    metrics, records, output_slot = _metrics(dataset=target, predictions=predictions, policy=frozen_policy, bootstrap_seed=int(config["bootstrap"]["seed"]), bootstrap_samples=int(config["bootstrap"]["final_samples"]))
    scientific = {
        "partition": partition, "target_cache_index_file_sha256": target_index_sha, "target_cache_index_payload_sha256": target_index["index_payload_sha256"],
        "feature_digest": target["feature_digest"], "shortlist_digest": target["shortlist_digest"],
        "support_probability_digest": tensor_sha256(torch.from_numpy(predictions["support"])), "expected_distance_digest": tensor_sha256(torch.from_numpy(predictions["expected"])),
        "selected_slot_digest": tensor_sha256(torch.from_numpy(predictions["selected"])), "row_feature_digest": tensor_sha256(torch.from_numpy(predictions["row_features"])),
        "value_prediction_digest": tensor_sha256(torch.from_numpy(predictions["value"])), "harm_prediction_digest": tensor_sha256(torch.from_numpy(predictions["harm"])),
        "output_slot_digest": tensor_sha256(torch.from_numpy(output_slot)), "frozen_policy": frozen_policy, "metrics": metrics,
        "bundle_training_digests": {
            key: bundle[key]
            for key in (
                "oof_feature_digest",
                "oof_support_digest",
                "oof_expected_distance_digest",
                "oof_selected_slot_digest",
                "development_feature_digest",
                "development_shortlist_digest",
            )
        },
        "policy_grid_digest": None if grid is None else canonical_json_sha256(grid), "records_digest": canonical_json_sha256(records),
    }
    scientific["scientific_payload_sha256"] = canonical_json_sha256(scientific)
    comparison = None; exact_replay = False
    if reference is not None:
        rs = reference["scientific"]
        keys = ["target_cache_index_file_sha256", "target_cache_index_payload_sha256", "feature_digest", "shortlist_digest", "support_probability_digest", "expected_distance_digest", "selected_slot_digest", "row_feature_digest", "value_prediction_digest", "harm_prediction_digest", "output_slot_digest", "frozen_policy", "bundle_training_digests", "policy_grid_digest", "records_digest", "scientific_payload_sha256"]
        comparison = {key: scientific[key] == rs[key] for key in keys}
        exact_replay = all(comparison.values())

    gates = config["gates"][partition]
    checks = _checks(metrics, gates); checks["policy_found"] = bool(policy_found); checks["exact_replay"] = bool(exact_replay)
    if reference is None:
        passed = False; decision = "PRIMARY_COMPLETE_AWAIT_EXACT_REPLAY"
    else:
        passed = all(checks.values()); decision = gates["decision_pass" if passed else "decision_fail"]
    result = {
        "schema_version": RESULT_SCHEMA, "date": "2026-07-20", "status": "completed", "partition": partition,
        "config": str(config_path), "config_sha256": file_sha256(config_path), "bundle_artifact": bundle_artifact, "frozen_bundle_artifact": frozen_bundle,
        "frozen_policy": frozen_policy, "policy_grid_records": grid, "scientific": scientific, "records": records,
        "reference": None if reference_path is None else str(reference_path), "replay_comparison": comparison, "exact_replay": exact_replay,
        "gate": {"checks": checks, "pass": passed, "decision": decision}, "locked_data": config["locked_data"], "claim_boundary": config["claim_scope"],
        "runtime_versions": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
    }
    result["result_payload_sha256"] = _result_payload_sha256(result); _atomic_json_save(result, output_path); return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=str(DEFAULT_CONFIG)); parser.add_argument("--partition", required=True, choices=PARTITIONS); parser.add_argument("--output", required=True); parser.add_argument("--reference", default=None); args = parser.parse_args()
    result = run(config_path=Path(args.config).resolve(), partition=args.partition, output_path=Path(args.output).resolve(), reference_path=None if args.reference is None else Path(args.reference).resolve())
    print(json.dumps({"output": str(Path(args.output).resolve()), "partition": result["partition"], "frozen_policy": result["frozen_policy"], "metrics": result["scientific"]["metrics"], "gate": result["gate"], "exact_replay": result["exact_replay"], "result_payload_sha256": result["result_payload_sha256"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
