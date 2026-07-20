import json
from copy import deepcopy
from pathlib import Path

import yaml

from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import canonical_json_sha256
from scripts.run_routeD_temporal_identity_full_population_gate3c1f2_v0 import (
    RESULT_SCHEMA,
    VIDEO_SCHEMA,
    _aggregate,
    _validate_config,
)

CONFIG = Path("configs/routeD_temporal_identity_full_population_gate3c1f2_v0.yaml")


def _video_record(index: int, *, future_reduction: float = 10.0):
    scientific = {
        "source_index": index,
        "video_name": str(index),
        "eligible_rows": 50,
        "entry_trigger_rows": 2,
        "top1_action_rows": 1,
        "entry_category_counts": {"failure": 2},
        "action_category_counts": {"failure": 1},
        "native_metrics": {"AJ": 0.25, "<avg": 0.37, "OA": 0.84},
        "modified_metrics": {"AJ": 0.251, "<avg": 0.374, "OA": 0.845},
        "gains": {"AJ": 0.001, "delta_avg": 0.004, "OA": 0.005},
        "evaluable_action_rows": 1,
        "no_future_visible_action_rows": 0,
        "action_future_reductions_px": [future_reduction],
        "digests": {"synthetic": str(index)},
    }
    scientific["scientific_payload_sha256"] = canonical_json_sha256(scientific)
    record = {
        "schema_version": VIDEO_SCHEMA,
        "date": "2026-07-20",
        "status": "completed",
        "config": str(CONFIG.resolve()),
        "config_sha256": "synthetic",
        "sample_metadata": {},
        "scientific": scientific,
        "action_records": [],
    }
    record["result_payload_sha256"] = canonical_json_sha256(record)
    return record


def test_gate3c1f2_config_authorities_and_locked_outputs_are_exact():
    config, _, parents = _validate_config(CONFIG.resolve())
    assert parents["data"]["formal_decision"] == (
        "AUTHORIZE_GATE3C1F2_FULL_POPULATION_CONFIRMATION_PREREGISTRATION"
    )
    assert all(value is False for value in config["locked_data"].values())
    assert not Path(config["determinism"]["primary_work_root"]).exists()
    assert not Path(config["determinism"]["replay_work_root"]).exists()
    assert not Path(
        "docs/generated/ROUTED_TEMPORAL_IDENTITY_FULL_POPULATION_GATE3C1F2_V0_PRIMARY_2026-07-20.json"
    ).exists()
    assert not Path(
        "docs/generated/ROUTED_TEMPORAL_IDENTITY_FULL_POPULATION_GATE3C1F2_V0_REPLAY_2026-07-20.json"
    ).exists()


def test_gate3c1f2_synthetic_primary_and_exact_replay_pass(tmp_path):
    config = yaml.safe_load(CONFIG.read_text())
    records = [_video_record(index) for index in range(128)]
    primary = _aggregate(
        config=config,
        config_path=CONFIG.resolve(),
        video_records=records,
        reference_path=None,
    )
    assert primary["schema_version"] == RESULT_SCHEMA
    assert primary["gate"]["decision"] == "PRIMARY_COMPLETE_AWAIT_EXACT_REPLAY"
    assert primary["gate"]["checks"]["AJ_CI_lower"] is True
    reference = tmp_path / "primary.json"
    reference.write_text(json.dumps(primary))
    replay = _aggregate(
        config=config,
        config_path=CONFIG.resolve(),
        video_records=deepcopy(records),
        reference_path=reference,
    )
    assert replay["exact_replay"] is True
    assert replay["gate"]["pass"] is True
    assert replay["gate"]["decision"] == (
        "AUTHORIZE_GATE3C1F3_OFFICIAL_TAPVID_PREREGISTRATION"
    )


def test_gate3c1f2_harm_gate_stops_even_with_positive_complete_metrics(tmp_path):
    config = yaml.safe_load(CONFIG.read_text())
    records = [_video_record(index, future_reduction=-5.0) for index in range(128)]
    primary = _aggregate(
        config=config,
        config_path=CONFIG.resolve(),
        video_records=records,
        reference_path=None,
    )
    reference = tmp_path / "primary.json"
    reference.write_text(json.dumps(primary))
    replay = _aggregate(
        config=config,
        config_path=CONFIG.resolve(),
        video_records=records,
        reference_path=reference,
    )
    assert replay["exact_replay"] is True
    assert replay["gate"]["checks"]["action_future_harm"] is False
    assert replay["gate"]["pass"] is False
    assert replay["gate"]["decision"] == "STOP_BEFORE_OFFICIAL_TAPVID"
