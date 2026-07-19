import importlib.util
from pathlib import Path

import torch

MODULE_PATH = Path("scripts/package_routeD_counterfactual_state_restoration_gate0.py")
spec = importlib.util.spec_from_file_location("gate0_packaging", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_nested_exact_handles_tensors_and_lists():
    left = {"x": torch.tensor([1.0, 2.0]), "rows": [None, torch.tensor([3])], "n": 2}
    right = {"x": torch.tensor([1.0, 2.0]), "rows": [None, torch.tensor([3])], "n": 2}
    assert module._nested_exact(left, right)
    right["x"][0] = 4
    assert not module._nested_exact(left, right)


def test_report_view_ignores_only_artifact_paths():
    row = {"decision": "x", "sidecar": "/a.pt", "sidecar_sha256": "abc", "value": 1}
    assert module._report_view(row) == {"decision": "x", "value": 1}
