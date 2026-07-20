import hashlib
from pathlib import Path

import yaml


def test_gate3c1f1_preregistration_hashes_and_membership_are_frozen():
    path = Path("configs/routeD_temporal_identity_full_population_data_gate3c1f1_v0.yaml")
    config = yaml.safe_load(path.read_text())
    assert config["expected_selection"]["excluded_raw_records"] == 1024
    assert config["materialization"]["max_samples"] == 128
    assert config["expected_selection"]["selected_source_files"] == 14
    assert config["confirmation_contract"]["source_indices"] == [0, 127]
    assert all(value is False for value in config["locked_data"].values())
    for authority in config["implementation"].values():
        assert hashlib.sha256(Path(authority["path"]).read_bytes()).hexdigest() == authority["sha256"]
    for exclusion in config["exclusions"]:
        assert hashlib.sha256(Path(exclusion["manifest"]).read_bytes()).hexdigest() == exclusion["manifest_sha256"]
    assert not Path(config["materialization"]["output_dir"]).exists()
