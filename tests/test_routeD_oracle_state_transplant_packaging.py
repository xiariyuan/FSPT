import importlib.util
from pathlib import Path

import torch

MODULE_PATH = Path("scripts/package_routeD_oracle_state_transplant_gate1.py")
spec = importlib.util.spec_from_file_location("gate1_packaging", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_nested_exact_handles_tensors_and_detects_change():
    left = {"x": torch.tensor([1, 2]), "rows": [None, {"y": 3.0}]}
    right = {"x": torch.tensor([1, 2]), "rows": [None, {"y": 3.0}]}
    assert module.nested_exact(left, right)
    right["x"][0] = 9
    assert not module.nested_exact(left, right)


def test_report_view_ignores_only_sidecar_identity():
    report = {
        "decision": "x",
        "sidecar": "/tmp/a.pt",
        "sidecar_sha256": "abc",
        "selected_points": 4,
    }
    assert module.report_view(report) == {"decision": "x", "selected_points": 4}
