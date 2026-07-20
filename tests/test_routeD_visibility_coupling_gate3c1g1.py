import json
from pathlib import Path

from scripts.run_routeD_visibility_coupling_gate3c1g1_v0 import _gate, _load_dataset, _validate_config

CONFIG = Path("configs/routeD_visibility_coupling_gate3c1g1_v0.yaml")
PRIMARY = Path("docs/generated/ROUTED_VISIBILITY_COUPLING_GATE3C1G1_V0_PRIMARY_2026-07-20.json")
REPLAY = Path("docs/generated/ROUTED_VISIBILITY_COUPLING_GATE3C1G1_V0_REPLAY_2026-07-20.json")


def _scientific(*, aj=0.002, ci=0.0005, fpr=0.3, exact_delta=0.0):
    return {
        "support": {"videos": 44, "actions": 89, "frame_rows": 801, "feature_dim": 66},
        "nested_OOF_gains": {
            "AJ_over_actual": aj,
            "AJ_over_native": aj,
            "OA_over_actual": 0.0,
            "OA_over_native": 0.001,
            "delta_avg_over_actual": exact_delta,
        },
        "nested_OOF_classification": {
            "AUC": 0.8,
            "AP": 0.75,
            "GT_visible_recall": 0.8,
            "GT_occluded_false_positive_rate": fpr,
        },
        "paired_video_CI": {
            "calibrated_minus_actual_AJ": {"lower": ci},
        },
    }


def test_gate3c1g1_authority_cache_and_outputs_are_frozen():
    config, index = _validate_config(CONFIG.resolve())
    dataset = _load_dataset(index)
    assert dataset["features"].shape == (801, 66)
    assert len(dataset["videos"]) == 44
    assert len(config["model_grid"]) == 10
    assert len(config["threshold_grid"]) == 37
    assert all(value is False for value in config["locked_data"].values())
    assert PRIMARY.is_file()
    assert REPLAY.is_file()
    primary = json.loads(PRIMARY.read_text())
    replay = json.loads(REPLAY.read_text())
    assert replay["exact_replay"] is True
    assert all(replay["replay_comparison"].values())
    assert replay["gate"]["decision"] == "STOP_GATE3C1G1_VISIBILITY_MODEL"
    assert replay["scientific"]["scientific_payload_sha256"] == primary["scientific"]["scientific_payload_sha256"]
    assert Path(config["determinism"]["primary_bundle"]).is_file()
    assert Path(config["determinism"]["replay_bundle"]).is_file()


def test_gate3c1g1_synthetic_primary_waits_for_replay():
    config, _ = _validate_config(CONFIG.resolve())
    gate = _gate(_scientific(), config, exact_replay=False)
    assert gate["pass"] is False
    assert gate["decision"] == "PRIMARY_COMPLETE_AWAIT_EXACT_REPLAY"
    assert all(value for key, value in gate["checks"].items() if key != "exact_replay")


def test_gate3c1g1_synthetic_exact_replay_passes():
    config, _ = _validate_config(CONFIG.resolve())
    gate = _gate(_scientific(), config, exact_replay=True)
    assert gate["pass"] is True
    assert gate["decision"] == "AUTHORIZE_GATE3C1G2_RAW_DISJOINT_VISIBILITY_CONFIRMATION_DATA"


def test_gate3c1g1_false_positive_or_aj_failure_stops():
    config, _ = _validate_config(CONFIG.resolve())
    assert _gate(_scientific(fpr=0.5), config, exact_replay=True)["decision"] == "STOP_GATE3C1G1_VISIBILITY_MODEL"
    assert _gate(_scientific(aj=0.0001, ci=-0.0001), config, exact_replay=True)["decision"] == "STOP_GATE3C1G1_VISIBILITY_MODEL"
