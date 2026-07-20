import hashlib
from pathlib import Path

import numpy as np
import yaml

from scripts.train_routeD_temporal_identity_entry_gate3c1f0_v0 import (
    _checks,
    _load_partition,
    _metrics,
)

CONFIG = Path("configs/routeD_temporal_identity_entry_gate3c1f0_v0.yaml")


def test_entry_preregistration_hashes_and_locked_data_are_exact():
    config = yaml.safe_load(CONFIG.read_text())
    assert all(value is False for value in config["locked_data"].values())
    for authority in config["implementation"].values():
        assert hashlib.sha256(Path(authority["path"]).read_bytes()).hexdigest() == authority["sha256"]
    for authority in (config["data"]["train"], config["data"]["validation"]):
        assert hashlib.sha256(Path(authority["path"]).read_bytes()).hexdigest() == authority["file_sha256"]


def test_entry_cache_partitions_reconstruct_expected_rows():
    config = yaml.safe_load(CONFIG.read_text())
    train = _load_partition(config["data"]["train"])
    validation = _load_partition(config["data"]["validation"])
    assert train["features"].shape == (376, 130)
    assert validation["features"].shape == (160, 130)
    assert int(train["labels"].sum()) == 164
    assert int(validation["labels"].sum()) == 80


def test_entry_metrics_enforce_probability_and_native_joint_intersection():
    labels = np.array([1, 1, 0, 0])
    probability = np.array([0.95, 0.95, 0.99, 0.2])
    joint = np.array([0.01, 0.03, 0.01, 0.01])
    operating = {"entry_probability_min": 0.93, "native_joint_probability_max": 0.02}
    metrics = _metrics(labels, probability, joint, operating)
    assert metrics["action_rows"] == 2
    assert metrics["failure_recall"] == 0.5
    assert metrics["clean_false_apply_rate"] == 0.5
    gates = {
        "AUC_min": 0.0,
        "AP_min": 0.0,
        "failure_recall_min": 0.4,
        "clean_false_apply_rate_max": 0.1,
        "action_precision_min": 0.9,
    }
    checks = _checks(metrics, gates)
    assert checks["failure_recall"] is True
    assert checks["clean_false_apply"] is False
