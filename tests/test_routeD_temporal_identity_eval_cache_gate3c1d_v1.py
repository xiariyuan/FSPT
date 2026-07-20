import hashlib
from pathlib import Path

import json

import pytest
import yaml

from scripts.build_routeD_temporal_identity_eval_cache_gate3c1d_v1 import (
    PARTITIONS,
    _validate_parent,
    _validate_runtime_authorization,
    canonical_json_sha256,
    file_sha256,
)


def test_checkpoint_parent_and_membership_are_frozen():
    config = yaml.safe_load(
        Path("configs/routeD_temporal_identity_top1v1_checkpoint_cache_gate3c1d_v1.yaml").read_text()
    )
    _validate_parent(config)
    _validate_runtime_authorization(config, "checkpoint_selection_v1")
    assert PARTITIONS["checkpoint_selection_v1"] == list(range(256))
    assert config["partition"]["read_state"]["renewed_checkpoint_selection_v1_read"] is True
    assert config["partition"]["read_state"]["renewed_fit_only_internal_audit_v1_read"] is False


def test_cache_implementation_hashes_are_exact():
    for path in Path("configs").glob("routeD_temporal_identity_top1v1_*_cache_gate3c1d_v1.yaml"):
        config = yaml.safe_load(path.read_text())
        for authority in config["implementation"].values():
            actual = hashlib.sha256(Path(authority["path"]).read_bytes()).hexdigest()
            assert actual == authority["sha256"]


def test_later_cache_authorization_pins_current_protocol_config(tmp_path):
    config = yaml.safe_load(
        Path("configs/routeD_temporal_identity_top1v1_audit_cache_gate3c1d_v1.yaml").read_text()
    )
    protocol = Path(config["runtime_authorization"]["required_protocol_config"]).resolve()
    result = {
        "partition": "checkpoint_selection_v1",
        "config": str(protocol),
        "config_sha256": file_sha256(protocol),
        "exact_replay": True,
        "gate": {
            "pass": True,
            "decision": "AUTHORIZE_GATE3C1D_V1_FIT_ONLY_AUDIT",
        },
    }
    result["result_payload_sha256"] = canonical_json_sha256(result)
    path = tmp_path / "checkpoint_replay.json"
    path.write_text(json.dumps(result))
    config["runtime_authorization"]["required_replay_result"] = str(path)
    _validate_runtime_authorization(config, "fit_only_internal_audit_v1")

    result["config_sha256"] = "0" * 64
    result.pop("result_payload_sha256")
    result["result_payload_sha256"] = canonical_json_sha256(result)
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="protocol hash drift"):
        _validate_runtime_authorization(config, "fit_only_internal_audit_v1")
