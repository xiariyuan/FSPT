import json

from scripts.build_routeD_temporal_identity_eval_cache_gate3c1b_v0 import (
    _expected_sources_for_partition,
    _validate_runtime_authorization,
)


def test_expected_sources_include_original_model_validation():
    assert _expected_sources_for_partition("checkpoint_selection") == list(
        range(384, 448)
    )
    assert _expected_sources_for_partition("fit_only_internal_audit") == list(
        range(448, 512)
    )
    assert _expected_sources_for_partition("original_model_validation") == list(
        range(48, 64)
    )


def test_original_model_validation_requires_future_rollout_authorization(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "partition": "fit_only_internal_audit",
                "exact_replay": True,
                "gate": {
                    "pass": True,
                    "decision": "AUTHORIZE_GATE3C1C_ORIGINAL_MODEL_VALIDATION_PREREGISTRATION",
                },
            }
        )
    )
    config = {
        "runtime_authorization": {
            "required_future_rollout_replay_result": str(result),
            "required_partition": "fit_only_internal_audit",
            "required_exact_replay": True,
            "required_gate_pass": True,
            "required_decision": "AUTHORIZE_GATE3C1C_ORIGINAL_MODEL_VALIDATION_PREREGISTRATION",
        }
    }
    _validate_runtime_authorization(config, "original_model_validation")


def test_original_model_validation_rejects_failed_authorization(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "partition": "fit_only_internal_audit",
                "exact_replay": True,
                "gate": {"pass": False, "decision": "STOP"},
            }
        )
    )
    config = {
        "runtime_authorization": {
            "required_future_rollout_replay_result": str(result),
            "required_partition": "fit_only_internal_audit",
            "required_exact_replay": True,
            "required_gate_pass": True,
            "required_decision": "AUTHORIZE_GATE3C1C_ORIGINAL_MODEL_VALIDATION_PREREGISTRATION",
        }
    }
    try:
        _validate_runtime_authorization(config, "original_model_validation")
    except ValueError as error:
        assert "authorization gate failed" in str(error)
    else:
        raise AssertionError("failed future-rollout authorization was accepted")
