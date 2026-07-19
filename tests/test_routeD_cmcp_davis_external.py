from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from projects.mmp_tracker.mmp_tracker.routeD_cmcp_davis_external import (
    davis_video_order_sha256,
    evaluate_davis_external_gate,
    load_davis_external_config,
    load_davis_sample_preserving_points,
    prepare_davis_sample,
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
