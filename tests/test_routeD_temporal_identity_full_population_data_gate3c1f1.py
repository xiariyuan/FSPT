import hashlib
import json
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate3c1f1_hashes_membership_and_materialized_result_are_frozen():
    path = Path("configs/routeD_temporal_identity_full_population_data_gate3c1f1_v0.yaml")
    config = yaml.safe_load(path.read_text())
    assert config["expected_selection"]["excluded_raw_records"] == 1024
    assert config["materialization"]["max_samples"] == 128
    assert config["expected_selection"]["selected_source_files"] == 14
    assert config["confirmation_contract"]["source_indices"] == [0, 127]
    assert all(value is False for value in config["locked_data"].values())
    for authority in config["implementation"].values():
        assert _sha256(Path(authority["path"])) == authority["sha256"]
    for exclusion in config["exclusions"]:
        assert _sha256(Path(exclusion["manifest"])) == exclusion["manifest_sha256"]

    output_dir = Path(config["materialization"]["output_dir"])
    manifest = output_dir / "train.index.json"
    assert output_dir.is_dir()
    assert manifest.is_file()
    payload = json.loads(manifest.read_text())
    assert payload["num_samples"] == 128
    assert len(payload["shards"]) == 8
    assert payload["selected_identity_digest"] == config["expected_selection"]["selected_identity_digest"]


def test_gate3c1f1_qualification_summary_passes_all_checks():
    path = Path(
        "docs/generated/ROUTED_TEMPORAL_IDENTITY_FULL_POPULATION_DATA_GATE3C1F1_V0_SUMMARY_2026-07-20.json"
    )
    payload = json.loads(path.read_text())
    assert payload["status"] == "completed_pass"
    assert payload["pass"] is True
    assert payload["formal_decision"] == (
        "AUTHORIZE_GATE3C1F2_FULL_POPULATION_CONFIRMATION_PREREGISTRATION"
    )
    assert all(payload["checks"].values())
    assert payload["selected_identities"] == 128
    assert payload["excluded_identities"] == 1024
    assert payload["selected_identity_digest"] == (
        "85193d0aa6381c7d78442e85bf87750ce72a497995a92016001ca0cf51832f28"
    )
