from pathlib import Path
import json

import torch

from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import canonical_json_sha256

from projects.mmp_tracker.mmp_tracker.routeD_visibility_coupling_v0 import (
    VISIBILITY_FEATURE_CHANNELS,
    VISIBILITY_FEATURE_SCHEMA_VERSION,
    assemble_visibility_features,
    flatten_visibility_rows,
)
from scripts.build_routeD_visibility_coupling_cache_gate3c1g0_v0 import (
    _validate_config,
)

CONFIG = Path("configs/routeD_visibility_coupling_cache_gate3c1g0_v0.yaml")


def test_visibility_feature_schema_is_frozen_at_66_channels():
    assert VISIBILITY_FEATURE_SCHEMA_VERSION == (
        "routeD_post_writeback_visibility_features_gate3c1g0_v0"
    )
    assert len(VISIBILITY_FEATURE_CHANNELS) == 66
    assert len(set(VISIBILITY_FEATURE_CHANNELS)) == 66
    channels = {
        name: torch.full((2, 9), float(index))
        for index, name in enumerate(VISIBILITY_FEATURE_CHANNELS)
    }
    features = assemble_visibility_features(channels, rows=2, frames=9)
    assert features.shape == (2, 9, 66)
    assert torch.equal(features[..., 0], torch.zeros(2, 9))
    assert torch.equal(features[..., -1], torch.full((2, 9), 65.0))


def test_visibility_feature_flattening_preserves_identities():
    features = torch.arange(2 * 3 * 66, dtype=torch.float32).reshape(2, 3, 66)
    flat = flatten_visibility_rows(
        features,
        source_index=7,
        point_indices=torch.tensor([10, 20]),
        frame_indices=torch.tensor([15, 16, 17]),
    )
    assert flat["features"].shape == (6, 66)
    assert flat["source_indices"].tolist() == [7] * 6
    assert flat["point_indices"].tolist() == [10, 10, 10, 20, 20, 20]
    assert flat["frame_indices"].tolist() == [15, 16, 17, 15, 16, 17]


def test_gate3c1g0_cache_config_and_materialized_result_are_exact():
    config, _, _, reference = _validate_config(CONFIG.resolve())
    assert len(config["source_indices"]) == 44
    assert sum(reference[index]["scientific"]["top1_action_rows"] for index in config["source_indices"]) == 89
    assert config["expected_support"]["frame_rows"] == 801
    assert all(value is False for value in config["locked_data"].values())
    root = Path(config["output_root"])
    assert root.is_dir()
    index = json.loads((root / "cache_index.json").read_text())
    assert index["videos"] == 44
    assert index["actions"] == 89
    assert index["frame_rows"] == 801
    assert index["index_payload_sha256"] == "a990040bcd8a86808451c8e77c6bed3b38ec2032522b9ef3bd1372f08dc1ea72"
    summary_path = Path("docs/generated/ROUTED_VISIBILITY_COUPLING_CACHE_GATE3C1G0_V0_SUMMARY_2026-07-20.json")
    summary = json.loads(summary_path.read_text())
    without_hash = dict(summary)
    embedded = without_hash.pop("summary_payload_sha256")
    assert embedded == canonical_json_sha256(without_hash)
    assert summary["formal_decision"] == "AUTHORIZE_GATE3C1G1_NESTED_VIDEO_OOF_PREREGISTRATION"


def test_low_dim_probe_is_frozen_negative_result():
    path = Path(
        "docs/generated/ROUTED_VISIBILITY_COUPLING_GATE3C1G0_LOW_DIM_PROBE_2026-07-20.json"
    )
    payload = json.loads(path.read_text())
    without_hash = dict(payload)
    embedded = without_hash.pop("summary_payload_sha256")
    assert embedded == canonical_json_sha256(without_hash)
    assert payload["formal_interpretation"] == (
        "REJECT_LOW_DIM_VISIBILITY_CALIBRATION_AS_PRIMARY_REDESIGN"
    )
    assert payload["baseline_modified_visibility"]["pooled_affected_frame_AJ"] == (
        0.09244669620974144
    )
    best = payload["models"]["gt_visible"]["log"]["best_grid_record"]
    assert best["pooled_affected_frame_AJ"] == 0.09816258895786352
    assert best["GT_occluded_false_positive_rate"] > 0.68
