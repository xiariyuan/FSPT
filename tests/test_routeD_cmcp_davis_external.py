from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import projects.mmp_tracker.mmp_tracker.routeD_cmcp_davis_external as davis_external_module
from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_davis_external import (
    davis_video_order_sha256,
    evaluate_davis_external_gate,
    load_davis_external_config,
    load_davis_sample_preserving_points,
    mean_davis_metric_dicts,
    prepare_davis_sample,
    validated_davis_native_state_hashes,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs/routeD_cmcp_davis_external_v0.yaml"


class MutatingDataset:
    def __init__(self):
        self.video_names = ["v"]
        self.points_dataset = {"v": {"points": np.array([[[0.25, 0.5]]], dtype=np.float32)}}

    def __getitem__(self, index):
        self.points_dataset["v"]["points"] *= np.array([255.0, 255.0], dtype=np.float32)
        return self.points_dataset["v"]["points"].copy()


def _sample():
    video = torch.zeros(3, 3, 256, 256)
    trajectory = torch.tensor(
        [
            [[10.0, 20.0], [30.0, 40.0]],
            [[11.0, 21.0], [31.0, 41.0]],
            [[12.0, 22.0], [32.0, 42.0]],
        ]
    )
    visibility = torch.tensor([[True, False], [True, True], [True, True]])
    query = torch.tensor([[0.0, 20.0, 10.0], [1.0, 41.0, 31.0]])
    return SimpleNamespace(
        video=video,
        trajectory=trajectory,
        visibility=visibility,
        query_points=query,
        seq_name="demo",
    )


def _metrics():
    return {
        "videos": 30,
        "native_candidate_parity_all": True,
        "oracle_gain_points": {"AJ": 4.0},
        "selected_gain_points": {"AJ": 0.4, "delta_average": 0.5},
        "paired_video_selected_AJ_gain_CI": {"lower": 0.1},
        "severe_16px_rate": {"selected_delta": -0.001},
        "behavior": {"harmful_non_native_rate": 0.009},
        "per_video": [{"selected_AJ_gain_points": 0.1} for _ in range(30)],
    }


def test_protocol_boundary_and_complete_membership():
    config = load_davis_external_config(CONFIG)
    assert config["external_dataset"]["expected_videos"] == 30
    assert config["external_dataset"]["subset_allowed"] is False
    assert config["claim_boundary"]["untouched_final_test_claim"] == "forbidden"
    assert config["kinetics_lock"]["rerun"] == "forbidden"


def test_loader_mutation_is_restored_exactly():
    dataset = MutatingDataset()
    before = dataset.points_dataset["v"]["points"].copy()
    sample = load_davis_sample_preserving_points(dataset, 0)
    assert np.allclose(sample, before * 255.0)
    assert np.array_equal(dataset.points_dataset["v"]["points"], before)


def test_prepare_davis_sample_contract():
    row = prepare_davis_sample(_sample())
    assert row["video"].shape == (1, 3, 3, 256, 256)
    assert row["query_points_tyx"].shape == (2, 3)
    assert row["gt_tracks_yx"].shape == (2, 3, 2)
    assert row["gt_occluded"].shape == (2, 3)
    assert torch.allclose(row["query_points_tyx"][0], torch.tensor([0.0, 20.0 / 255.0, 10.0 / 255.0]))
    assert torch.allclose(row["gt_tracks_yx"][0, 0], torch.tensor([20.0 / 255.0, 10.0 / 255.0]))


def test_external_gate_requires_all_frozen_checks():
    config = load_davis_external_config(CONFIG)
    gate = evaluate_davis_external_gate(_metrics(), config, exact_replay=True)
    assert gate["pass"]
    failed = copy.deepcopy(_metrics())
    for row in failed["per_video"][:13]:
        row["selected_AJ_gain_points"] = -0.1
    assert not evaluate_davis_external_gate(failed, config, exact_replay=True)["pass"]
    assert not evaluate_davis_external_gate(_metrics(), config, exact_replay=False)["pass"]


def test_video_order_hash_is_order_sensitive():
    assert davis_video_order_sha256(["a", "b"]) != davis_video_order_sha256(["b", "a"])


def test_native_state_hash_contract_is_validated_and_tamper_detected(tmp_path):
    keys = (
        "native_coords_xy_px",
        "native_visibility_probability",
        "native_confidence_probability",
        "native_joint_probability",
        "native_visibility",
    )
    tensors = {
        "native_coords_xy_px": torch.zeros(2, 3, 2),
        "native_visibility_probability": torch.full((2, 3), 0.8),
        "native_confidence_probability": torch.full((2, 3), 0.7),
        "native_joint_probability": torch.full((2, 3), 0.56),
        "native_visibility": torch.zeros(2, 3, dtype=torch.bool),
    }
    base = tmp_path / "base.pt"
    feature = tmp_path / "feature.pt"
    torch.save({"tensors": tensors}, base)
    hashes = {key: tensor_sha256(tensors[key]) for key in keys}
    torch.save({"native_state_hashes": hashes}, feature)
    assert validated_davis_native_state_hashes(base, feature) == hashes
    bad = dict(hashes)
    bad["native_coords_xy_px"] = "0" * 64
    torch.save({"native_state_hashes": bad}, feature)
    try:
        validated_davis_native_state_hashes(base, feature)
    except ValueError as error:
        assert "do not match" in str(error)
    else:
        raise AssertionError("tampered DAVIS native-state hash must fail")


def test_davis_metric_aggregation_is_equal_video_mean():
    rows = [
        {"AJ": 0.2, "OA": 0.8, "<avg": 0.5},
        {"AJ": 0.6, "OA": 1.0, "<avg": 0.7},
    ]
    assert mean_davis_metric_dicts(rows) == {"AJ": 0.4, "OA": 0.9, "<avg": 0.6}


def test_davis_metric_aggregation_rejects_nonfinite_or_key_drift():
    try:
        mean_davis_metric_dicts([{"AJ": 0.2}, {"AJ": float("nan")}])
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("non-finite DAVIS metrics must fail")
    try:
        mean_davis_metric_dicts([{"AJ": 0.2}, {"OA": 0.8}])
    except ValueError as error:
        assert "metric-key drift" in str(error)
    else:
        raise AssertionError("DAVIS metric-key drift must fail")


def test_davis_evaluator_accepts_variable_frame_lengths(monkeypatch):
    def make_bundle(source_index: int, frames: int):
        gt = torch.full((1, frames, 2), 0.1)
        native = torch.full((1, frames, 2), 25.5)
        tensors = {
            "native_coords_xy_px": native,
            "native_visibility": torch.ones(1, frames, dtype=torch.bool),
            "query_points_tyx": torch.tensor([[0.0, 0.1, 0.1]]),
            "gt_tracks_yx": gt,
            "gt_occluded": torch.zeros(1, frames, dtype=torch.bool),
        }
        return SimpleNamespace(
            source_index=source_index,
            video_name=f"v{source_index}",
            tensors=tensors,
        )

    bundles = {0: make_bundle(0, 2), 1: make_bundle(1, 3)}
    monkeypatch.setattr(
        davis_external_module,
        "load_complete_feature_index",
        lambda *_args, **_kwargs: {
            "videos": [{"source_index": 0}, {"source_index": 1}],
            "_index_sha256": "index",
        },
    )
    monkeypatch.setattr(
        davis_external_module,
        "load_cmcp_video",
        lambda row: bundles[int(row["source_index"])],
    )

    def fake_predict(_adapter, _cmcp, _comparator, bundle, _normalization, **_kwargs):
        native = bundle.tensors["native_coords_xy_px"]
        candidates = native.unsqueeze(2)
        frames = int(native.shape[1])
        prediction = {
            "selected_coords_xy_px": native.clone(),
            "selected_candidate_index": torch.zeros(1, frames, dtype=torch.long),
            "candidate_coords_xy_px": candidates,
            "candidate_valid_mask": torch.ones(1, frames, 1, dtype=torch.bool),
            "oracle_coords_xy_px": native.clone(),
            "oracle_candidate_index": torch.zeros(1, frames, dtype=torch.long),
            "adapted_feature_sha256": f"feature-{bundle.source_index}",
        }
        behavior = {
            "rows": frames - 1,
            "selected_non_native_rate": 0.0,
            "harmful_non_native_rate": 0.0,
            "beneficial_candidate_available_rate": 0.0,
            "beneficial_candidate_recall": 0.0,
        }
        return prediction, behavior

    monkeypatch.setattr(davis_external_module, "predict_lmra_video", fake_predict)
    result = davis_external_module.evaluate_davis_lmra_index(
        None, None, None, "unused.json", None, device="cpu",
        point_batch_size=1, bootstrap_samples=20, bootstrap_seed=7,
    )
    assert result["videos"] == 2
    assert result["aggregation"] == "official_equal_video_mean"
    assert result["native_candidate_parity_all"]
    assert result["selected_gain_points"]["AJ"] == 0.0
    assert [row["frames"] for row in result["per_video"]] == [2, 3]
