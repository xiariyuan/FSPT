#!/usr/bin/env python3
"""Nested source-video OOF visibility-coupling model selection for Gate 3C1G1."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import sklearn
import torch
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.metrics import compute_tapvid_metrics
from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import canonical_json_sha256, file_sha256
from projects.mmp_tracker.mmp_tracker.routeD_musr_training import _tracks_from_xy
from scripts.build_routeD_visibility_coupling_cache_gate3c1g0_v0 import verify_cache_payload

DEFAULT_CONFIG = ROOT / "configs/routeD_visibility_coupling_gate3c1g1_v0.yaml"
RESULT_SCHEMA = "routeD_visibility_coupling_gate3c1g1_v0_result"
BUNDLE_SCHEMA = "routeD_visibility_coupling_gate3c1g1_v0_bundle"


@dataclass(frozen=True)
class Candidate:
    name: str
    target: str
    kind: str
    parameter: float


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _runtime_contract(config: Mapping[str, Any]) -> None:
    actual = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    expected = {key: str(value) for key, value in config["runtime_contract"].items()}
    if actual != expected:
        raise ValueError(f"Gate 3C1G1 runtime drift: {actual} != {expected}")


def _validate_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(path.read_text())
    if config.get("schema_version") != "routeD_visibility_coupling_gate3c1g1_v0":
        raise ValueError("Gate 3C1G1 config schema drift")
    _runtime_contract(config)
    parent = config["authorized_parent"]
    summary_path = Path(parent["summary_path"])
    if file_sha256(summary_path) != parent["summary_file_sha256"]:
        raise ValueError("Gate 3C1G1 parent summary file drift")
    summary = json.loads(summary_path.read_text())
    without_hash = dict(summary)
    embedded = without_hash.pop("summary_payload_sha256")
    if embedded != canonical_json_sha256(without_hash) or embedded != parent["summary_payload_sha256"]:
        raise ValueError("Gate 3C1G1 parent summary payload drift")
    if summary.get("formal_decision") != "AUTHORIZE_GATE3C1G1_NESTED_VIDEO_OOF_PREREGISTRATION":
        raise ValueError("Gate 3C1G1 parent authorization missing")
    index_path = Path(parent["cache_index_path"])
    if file_sha256(index_path) != parent["cache_index_file_sha256"]:
        raise ValueError("Gate 3C1G1 cache index file drift")
    index = json.loads(index_path.read_text())
    without_index_hash = dict(index)
    index_hash = without_index_hash.pop("index_payload_sha256")
    if index_hash != canonical_json_sha256(without_index_hash) or index_hash != parent["cache_index_payload_sha256"]:
        raise ValueError("Gate 3C1G1 cache index payload drift")
    if (index.get("videos"), index.get("actions"), index.get("frame_rows")) != (44, 89, 801):
        raise ValueError("Gate 3C1G1 cache support drift")
    for authority in config["implementation"].values():
        if file_sha256(Path(authority["path"])) != authority["sha256"]:
            raise ValueError(f"Gate 3C1G1 implementation drift: {authority['path']}")
    if not all(value is False for value in config["locked_data"].values()):
        raise ValueError("Gate 3C1G1 locked-data contract drift")
    return config, index


def _metric_dict(value: Mapping[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    aliases = {"AJ": "AJ", "<avg": "delta_avg", "OA": "OA"}
    for source, target in aliases.items():
        item = np.asarray(value[source])
        if item.size != 1:
            raise ValueError("Gate 3C1G1 metric scalar drift")
        output[target] = float(item.reshape(-1)[0])
    return output


def _load_dataset(index: Mapping[str, Any]) -> dict[str, Any]:
    features, groups, points, frames = [], [], [], []
    gt_visible, utility16, soft_utility = [], [], []
    videos: dict[int, dict[str, Any]] = {}
    feature_names = None
    rows = sorted(index["rows"], key=lambda item: int(item["source_index"]))
    for row in rows:
        payload = torch.load(row["sidecar"], map_location="cpu", weights_only=False)
        verify_cache_payload(payload, source_index=int(row["source_index"]))
        if file_sha256(Path(row["sidecar"])) != row["sidecar_sha256"]:
            raise ValueError("Gate 3C1G1 sidecar file drift")
        names = tuple(payload["feature_channels"])
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Gate 3C1G1 feature-channel drift")
        tensors = payload["tensors"]
        source = int(row["source_index"])
        features.append(tensors["features"].float().numpy())
        groups.append(tensors["source_indices"].long().numpy())
        points.append(tensors["point_indices"].long().numpy())
        frames.append(tensors["frame_indices"].long().numpy())
        visible = tensors["gt_visible_action_frame"].reshape(-1).long().numpy()
        hits = tensors["modified_threshold_hits_action_frame"].reshape(-1, 5).long().numpy()
        gt_visible.append(visible)
        utility16.append(hits[:, -1])
        soft_utility.append(hits.mean(axis=1).astype(np.float32))
        videos[source] = {
            "source_index": source,
            "video_name": payload["video_name"],
            "modified_coordinates_xy": tensors["modified_coordinates_xy"].float(),
            "native_coordinates_xy": tensors["native_coordinates_xy"].float(),
            "modified_visibility": tensors["modified_visibility"].bool(),
            "native_visibility": tensors["native_visibility"].bool(),
            "gt_tracks_yx": tensors["gt_tracks_yx"].float(),
            "gt_visibility": (~tensors["gt_occluded"].bool()),
            "query_points_tyx": tensors["query_points_tyx"].float(),
        }
    result = {
        "features": np.concatenate(features).astype(np.float32),
        "groups": np.concatenate(groups).astype(np.int64),
        "point_indices": np.concatenate(points).astype(np.int64),
        "frame_indices": np.concatenate(frames).astype(np.int64),
        "gt_visible": np.concatenate(gt_visible).astype(np.int64),
        "utility16": np.concatenate(utility16).astype(np.int64),
        "soft_utility": np.concatenate(soft_utility).astype(np.float32),
        "feature_names": feature_names,
        "videos": videos,
    }
    if result["features"].shape != (801, 66) or len(np.unique(result["groups"])) != 44:
        raise ValueError("Gate 3C1G1 dataset shape drift")
    if not np.isfinite(result["features"]).all():
        raise ValueError("Gate 3C1G1 non-finite features")
    return result


def _candidates(config: Mapping[str, Any]) -> list[Candidate]:
    return [Candidate(str(item["name"]), str(item["target"]), str(item["kind"]), float(item["parameter"])) for item in config["model_grid"]]


def _sample_weight(groups: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(groups, return_counts=True)
    mapping = {int(group): float(len(groups)) / float(len(unique) * count) for group, count in zip(unique, counts)}
    return np.array([mapping[int(group)] for group in groups], dtype=np.float64)


def _target(dataset: Mapping[str, Any], candidate: Candidate) -> np.ndarray:
    return np.asarray(dataset[candidate.target])


def _fit(candidate: Candidate, x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> Any:
    weight = _sample_weight(groups)
    if candidate.kind == "logistic":
        model = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=candidate.parameter, max_iter=3000, solver="lbfgs", random_state=seed)),
        ])
        model.fit(x, y.astype(np.int64), model__sample_weight=weight)
        return model
    if candidate.kind == "hgb_classifier":
        model = HistGradientBoostingClassifier(
            max_iter=240, learning_rate=0.04, max_leaf_nodes=15,
            l2_regularization=candidate.parameter, min_samples_leaf=15,
            random_state=seed, early_stopping=False,
        )
        model.fit(x, y.astype(np.int64), sample_weight=weight)
        return model
    if candidate.kind == "hgb_regressor":
        model = HistGradientBoostingRegressor(
            max_iter=240, learning_rate=0.04, max_leaf_nodes=15,
            l2_regularization=candidate.parameter, min_samples_leaf=15,
            random_state=seed, early_stopping=False, loss="squared_error",
        )
        model.fit(x, y.astype(np.float64), sample_weight=weight)
        return model
    raise ValueError(f"unknown candidate kind {candidate.kind}")


def _predict(candidate: Candidate, model: Any, x: np.ndarray) -> np.ndarray:
    if candidate.kind in {"logistic", "hgb_classifier"}:
        score = model.predict_proba(x)[:, 1]
    else:
        score = model.predict(x)
    score = np.asarray(score, dtype=np.float64)
    if not np.isfinite(score).all():
        raise ValueError("Gate 3C1G1 non-finite prediction")
    return np.clip(score, 0.0, 1.0)


def _view_metrics(dataset: Mapping[str, Any], row_mask: np.ndarray | None, sources: np.ndarray, view: str) -> dict[str, Any]:
    source_set = {int(value) for value in np.asarray(sources).tolist()}
    per_video: dict[int, dict[str, float]] = {}
    for source in sorted(source_set):
        video = dataset["videos"][source]
        if view == "native":
            coordinates = video["native_coordinates_xy"]
            visibility = video["native_visibility"]
        else:
            coordinates = video["modified_coordinates_xy"]
            visibility = video["modified_visibility"].clone()
            if view == "calibrated":
                if row_mask is None:
                    raise ValueError("calibrated view requires row mask")
                rows = np.where(dataset["groups"] == source)[0]
                visibility[torch.from_numpy(dataset["point_indices"][rows]), torch.from_numpy(dataset["frame_indices"][rows])] = torch.from_numpy(row_mask[rows].astype(np.bool_))
        metrics = compute_tapvid_metrics(
            _tracks_from_xy(coordinates, 256),
            video["gt_tracks_yx"], visibility, video["gt_visibility"],
            video["query_points_tyx"], resolution=256, query_mode="first",
        )
        per_video[source] = _metric_dict(metrics)
    mean = {key: float(np.mean([value[key] for value in per_video.values()])) for key in ("AJ", "delta_avg", "OA")}
    return {"mean": mean, "per_video": per_video}


def _classification(y: np.ndarray, utility: np.ndarray, score: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    visible = y.astype(bool); occluded = ~visible; predicted = mask.astype(bool)
    return {
        "AUC": float(roc_auc_score(y, score)),
        "AP": float(average_precision_score(y, score)),
        "predicted_visible_rows": int(predicted.sum()),
        "GT_visible_recall": float((predicted & visible).sum() / max(1, visible.sum())),
        "GT_occluded_false_positive_rate": float((predicted & occluded).sum() / max(1, occluded.sum())),
        "predicted_visible_precision": float((predicted & visible).sum() / max(1, predicted.sum())),
        "utility16_precision": float((predicted & utility.astype(bool)).sum() / max(1, predicted.sum())),
    }


def _evaluate_threshold(dataset: Mapping[str, Any], scores: np.ndarray, threshold: float, sources: np.ndarray) -> dict[str, Any]:
    source_set = {int(value) for value in np.asarray(sources).tolist()}
    row_select = np.array([int(group) in source_set for group in dataset["groups"]], dtype=bool)
    mask = np.zeros(len(dataset["groups"]), dtype=bool)
    mask[row_select] = scores[row_select] >= float(threshold)
    actual = _view_metrics(dataset, None, sources, "actual")
    calibrated = _view_metrics(dataset, mask, sources, "calibrated")
    native = _view_metrics(dataset, None, sources, "native")
    cls = _classification(dataset["gt_visible"][row_select], dataset["utility16"][row_select], scores[row_select], mask[row_select])
    gains = {
        "AJ_over_actual": calibrated["mean"]["AJ"] - actual["mean"]["AJ"],
        "AJ_over_native": calibrated["mean"]["AJ"] - native["mean"]["AJ"],
        "OA_over_actual": calibrated["mean"]["OA"] - actual["mean"]["OA"],
        "OA_over_native": calibrated["mean"]["OA"] - native["mean"]["OA"],
    }
    return {"threshold": float(threshold), "gains": gains, "classification": cls, "calibrated": calibrated["mean"], "actual": actual["mean"], "native": native["mean"]}


def _inner_oof(dataset: Mapping[str, Any], train_rows: np.ndarray, candidate: Candidate, folds: int, seed: int) -> np.ndarray:
    x = dataset["features"]; groups = dataset["groups"]; y = _target(dataset, candidate)
    local_groups = groups[train_rows]
    output = np.full(len(groups), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=folds)
    for fold, (fit_local, val_local) in enumerate(splitter.split(x[train_rows], y[train_rows], local_groups)):
        fit_rows = train_rows[fit_local]; val_rows = train_rows[val_local]
        model = _fit(candidate, x[fit_rows], y[fit_rows], groups[fit_rows], seed + fold)
        output[val_rows] = _predict(candidate, model, x[val_rows])
    if not np.isfinite(output[train_rows]).all():
        raise ValueError("Gate 3C1G1 inner OOF incomplete")
    return output


def _rank(record: Mapping[str, Any], config: Mapping[str, Any], candidate_order: int, threshold_order: int) -> tuple[Any, ...]:
    cls = record["classification"]; gains = record["gains"]; constraints = config["inner_selection_constraints"]
    feasible = (
        gains["OA_over_actual"] >= float(constraints["OA_over_actual_min"])
        and cls["GT_visible_recall"] >= float(constraints["GT_visible_recall_min"])
        and cls["GT_occluded_false_positive_rate"] <= float(constraints["GT_occluded_false_positive_rate_max"])
    )
    return (
        int(feasible), gains["AJ_over_actual"], gains["OA_over_actual"],
        -cls["GT_occluded_false_positive_rate"], cls["GT_visible_recall"],
        -cls["predicted_visible_rows"], -candidate_order, -threshold_order,
    )


def _select(dataset: Mapping[str, Any], train_rows: np.ndarray, config: Mapping[str, Any], seed: int) -> tuple[Candidate, float, dict[str, Any], list[dict[str, Any]]]:
    sources = np.unique(dataset["groups"][train_rows])
    records: list[dict[str, Any]] = []
    candidates = _candidates(config)
    thresholds = [float(value) for value in config["threshold_grid"]]
    best_key = None; best = None; best_candidate = None
    for candidate_order, candidate in enumerate(candidates):
        scores = _inner_oof(dataset, train_rows, candidate, int(config["folds"]["inner"]), seed + 100 * candidate_order)
        for threshold_order, threshold in enumerate(thresholds):
            record = _evaluate_threshold(dataset, scores, threshold, sources)
            record.update({"candidate": candidate.name, "target": candidate.target, "kind": candidate.kind, "parameter": candidate.parameter})
            key = _rank(record, config, candidate_order, threshold_order)
            records.append(record)
            if best_key is None or key > best_key:
                best_key = key; best = record; best_candidate = candidate
    assert best is not None and best_candidate is not None
    best = dict(best); best["inner_feasible"] = bool(best_key[0])
    return best_candidate, float(best["threshold"]), best, records


def _bootstrap(per_video_left: Mapping[int, Mapping[str, float]], per_video_right: Mapping[int, Mapping[str, float]], key: str, seed: int, samples: int) -> dict[str, Any]:
    sources = sorted(set(per_video_left) & set(per_video_right))
    values = np.array([per_video_left[source][key] - per_video_right[source][key] for source in sources], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draws[index] = rng.choice(values, len(values), replace=True).mean()
    return {"mean": float(values.mean()), "lower": float(np.quantile(draws, .025)), "upper": float(np.quantile(draws, .975)), "videos": len(sources), "samples": samples, "seed": seed}


def _run(config: Mapping[str, Any], index: Mapping[str, Any], bundle_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = _load_dataset(index)
    x = dataset["features"]; groups = dataset["groups"]
    outer = GroupKFold(n_splits=int(config["folds"]["outer"]))
    outer_score = np.full(len(groups), np.nan, dtype=np.float64)
    outer_mask = np.zeros(len(groups), dtype=bool)
    outer_records: list[dict[str, Any]] = []
    base_target = dataset["gt_visible"]
    for fold, (train_rows, test_rows) in enumerate(outer.split(x, base_target, groups)):
        candidate, threshold, selected, _ = _select(dataset, train_rows, config, int(config["determinism"]["seed"]) + 10000 * fold)
        model = _fit(candidate, x[train_rows], _target(dataset, candidate)[train_rows], groups[train_rows], int(config["determinism"]["seed"]) + fold)
        score = _predict(candidate, model, x[test_rows])
        outer_score[test_rows] = score; outer_mask[test_rows] = score >= threshold
        outer_records.append({
            "fold": fold, "train_videos": sorted(np.unique(groups[train_rows]).astype(int).tolist()),
            "test_videos": sorted(np.unique(groups[test_rows]).astype(int).tolist()),
            "candidate": candidate.__dict__, "threshold": threshold, "inner_selection": selected,
            "test_score_digest": tensor_sha256(torch.from_numpy(score.astype(np.float32))),
            "test_mask_digest": tensor_sha256(torch.from_numpy((score >= threshold))),
        })
    if not np.isfinite(outer_score).all():
        raise ValueError("Gate 3C1G1 outer OOF incomplete")
    sources = np.unique(groups)
    native = _view_metrics(dataset, None, sources, "native")
    actual = _view_metrics(dataset, None, sources, "actual")
    calibrated = _view_metrics(dataset, outer_mask, sources, "calibrated")
    cls = _classification(dataset["gt_visible"], dataset["utility16"], outer_score, outer_mask)
    samples = int(config["bootstrap"]["samples"]); seed = int(config["bootstrap"]["seed"])
    paired = {
        "calibrated_minus_actual_AJ": _bootstrap(calibrated["per_video"], actual["per_video"], "AJ", seed, samples),
        "calibrated_minus_native_AJ": _bootstrap(calibrated["per_video"], native["per_video"], "AJ", seed + 1, samples),
        "calibrated_minus_actual_OA": _bootstrap(calibrated["per_video"], actual["per_video"], "OA", seed + 2, samples),
        "calibrated_minus_native_OA": _bootstrap(calibrated["per_video"], native["per_video"], "OA", seed + 3, samples),
    }
    all_rows = np.arange(len(groups))
    final_candidate, final_threshold, final_selection, final_grid = _select(dataset, all_rows, config, int(config["determinism"]["seed"]) + 50000)
    final_model = _fit(final_candidate, x, _target(dataset, final_candidate), groups, int(config["determinism"]["seed"]) + 60000)
    bundle = {
        "schema_version": BUNDLE_SCHEMA, "feature_names": list(dataset["feature_names"]),
        "candidate": final_candidate.__dict__, "threshold": final_threshold, "model": final_model,
        "cache_index_payload_sha256": index["index_payload_sha256"],
    }
    bundle_path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(bundle, bundle_path)
    scientific = {
        "support": {"videos": 44, "actions": 89, "frame_rows": 801, "feature_dim": 66},
        "native_mean": native["mean"], "actual_modified_mean": actual["mean"], "nested_OOF_calibrated_mean": calibrated["mean"],
        "nested_OOF_gains": {
            "AJ_over_actual": calibrated["mean"]["AJ"] - actual["mean"]["AJ"],
            "AJ_over_native": calibrated["mean"]["AJ"] - native["mean"]["AJ"],
            "OA_over_actual": calibrated["mean"]["OA"] - actual["mean"]["OA"],
            "OA_over_native": calibrated["mean"]["OA"] - native["mean"]["OA"],
            "delta_avg_over_actual": calibrated["mean"]["delta_avg"] - actual["mean"]["delta_avg"],
        },
        "nested_OOF_classification": cls, "paired_video_CI": paired,
        "outer_folds": outer_records, "outer_candidate_counts": dict(Counter(record["candidate"]["name"] for record in outer_records)),
        "final_selection": final_selection,
        "final_candidate": final_candidate.__dict__, "final_threshold": final_threshold,
        "digests": {
            "features": tensor_sha256(torch.from_numpy(dataset["features"])),
            "gt_visible": tensor_sha256(torch.from_numpy(dataset["gt_visible"])),
            "utility16": tensor_sha256(torch.from_numpy(dataset["utility16"])),
            "soft_utility": tensor_sha256(torch.from_numpy(dataset["soft_utility"])),
            "groups": tensor_sha256(torch.from_numpy(dataset["groups"])),
            "outer_score": tensor_sha256(torch.from_numpy(outer_score.astype(np.float32))),
            "outer_mask": tensor_sha256(torch.from_numpy(outer_mask)),
            "final_grid": canonical_json_sha256(final_grid),
        },
    }
    scientific["scientific_payload_sha256"] = canonical_json_sha256(scientific)
    return scientific, bundle


def _gate(scientific: Mapping[str, Any], config: Mapping[str, Any], exact_replay: bool) -> dict[str, Any]:
    thresholds = config["formal_gates"]; gains = scientific["nested_OOF_gains"]; cls = scientific["nested_OOF_classification"]; ci = scientific["paired_video_CI"]
    checks = {
        "support": scientific["support"] == {"videos": 44, "actions": 89, "frame_rows": 801, "feature_dim": 66},
        "AUC": cls["AUC"] >= float(thresholds["AUC_min"]),
        "AP": cls["AP"] >= float(thresholds["AP_min"]),
        "GT_visible_recall": cls["GT_visible_recall"] >= float(thresholds["GT_visible_recall_min"]),
        "GT_occluded_false_positive_rate": cls["GT_occluded_false_positive_rate"] <= float(thresholds["GT_occluded_false_positive_rate_max"]),
        "AJ_over_actual": gains["AJ_over_actual"] >= float(thresholds["AJ_over_actual_min"]),
        "AJ_over_actual_CI": ci["calibrated_minus_actual_AJ"]["lower"] > float(thresholds["AJ_over_actual_CI_lower_min"]),
        "AJ_over_native": gains["AJ_over_native"] >= float(thresholds["AJ_over_native_min"]),
        "OA_over_actual": gains["OA_over_actual"] >= float(thresholds["OA_over_actual_min"]),
        "OA_over_native": gains["OA_over_native"] >= float(thresholds["OA_over_native_min"]),
        "delta_avg_exact": abs(gains["delta_avg_over_actual"]) <= 1.0e-12,
        "exact_replay": bool(exact_replay),
    }
    passed = all(checks.values())
    return {"checks": checks, "pass": passed, "decision": "AUTHORIZE_GATE3C1G2_RAW_DISJOINT_VISIBILITY_CONFIRMATION_DATA" if passed else ("STOP_GATE3C1G1_VISIBILITY_MODEL" if exact_replay else "PRIMARY_COMPLETE_AWAIT_EXACT_REPLAY")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--reference", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bundle-output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve(); config, index = _validate_config(config_path)
    scientific, _ = _run(config, index, Path(args.bundle_output).resolve())
    exact = False; comparison = None
    if args.reference:
        reference_path = Path(args.reference).resolve(); reference = json.loads(reference_path.read_text())
        reference_scientific = reference["scientific"]
        comparison = {
            "scientific_payload_sha256": scientific["scientific_payload_sha256"] == reference_scientific["scientific_payload_sha256"],
            "outer_score": scientific["digests"]["outer_score"] == reference_scientific["digests"]["outer_score"],
            "outer_mask": scientific["digests"]["outer_mask"] == reference_scientific["digests"]["outer_mask"],
            "final_selection": scientific["final_selection"] == reference_scientific["final_selection"],
        }
        exact = all(comparison.values())
    result = {
        "schema_version": RESULT_SCHEMA, "date": "2026-07-20", "status": "completed",
        "config": str(config_path), "config_sha256": file_sha256(config_path),
        "reference": None if not args.reference else str(Path(args.reference).resolve()),
        "bundle_output": str(Path(args.bundle_output).resolve()),
        "scientific": scientific, "exact_replay": exact, "replay_comparison": comparison,
    }
    result["gate"] = _gate(scientific, config, exact)
    result["result_payload_sha256"] = canonical_json_sha256(result)
    _atomic_json(result, Path(args.output).resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
