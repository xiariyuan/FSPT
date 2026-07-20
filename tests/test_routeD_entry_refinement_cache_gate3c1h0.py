import json
from pathlib import Path

import torch

from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import canonical_json_sha256
from scripts.build_routeD_entry_refinement_cache_gate3c1h0_v0 import _validate_config, verify_payload

CONFIG = Path("configs/routeD_entry_refinement_cache_gate3c1h0_v0.yaml")


def test_gate3c1h0_config_authority_and_output_are_frozen():
    config, replay, references = _validate_config(CONFIG.resolve())
    assert replay["gate"]["decision"] == "STOP_BEFORE_OFFICIAL_TAPVID"
    assert len(references) == 128
    assert config["expected_support"] == {
        "videos": 128,
        "rows": 5596,
        "feature_dim": 130,
        "current_entry_rows": 537,
    }
    assert all(value is False for value in config["locked_data"].values())
    assert not Path(config["output_root"]).exists()


def test_gate3c1h0_source0_smoke_is_exact():
    path = Path("/tmp/routeD_gate3c1h0_smoke/video_00000.pt")
    assert path.is_file()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    verify_payload(payload, source_index=0)
    assert payload["rows"] == 53
    assert payload["feature_dim"] == 130
    assert payload["current_entry_rows"] == 5
    assert payload["category_counts"] == {
        "failure": 13,
        "clean": 10,
        "ambiguous": 26,
        "other": 4,
    }


def test_activation_oracle_upper_bound_rejects_activation_only_mainline():
    path = Path(
        "docs/generated/ROUTED_VISIBILITY_ACTIVATION_ORACLE_GATE3C1G1_POST_FAILURE_2026-07-20.json"
    )
    payload = json.loads(path.read_text())
    without_hash = dict(payload)
    embedded = without_hash.pop("summary_payload_sha256")
    assert embedded == canonical_json_sha256(without_hash)
    assert payload["formal_interpretation"] == (
        "REJECT_ACTIVATION_ONLY_AS_SUFFICIENT_MAINLINE"
    )
    assert payload["support"]["activation_rows"] == 454
    assert payload["maximum_observed_oracle_AJ_gain_fraction"] < (
        payload["required_action_video_AJ_gain_fraction"]
    )
