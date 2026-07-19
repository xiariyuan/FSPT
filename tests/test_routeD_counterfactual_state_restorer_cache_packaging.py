import subprocess
import sys
from pathlib import Path

import torch

from scripts.package_routeD_counterfactual_state_restorer_cache_smoke import nested_exact


def test_cache_smoke_nested_exact():
    left = {"a": [torch.tensor([1, 2]), {"b": torch.tensor(3.0)}]}
    right = {"a": [torch.tensor([1, 2]), {"b": torch.tensor(3.0)}]}
    wrong = {"a": [torch.tensor([1, 3]), {"b": torch.tensor(3.0)}]}
    assert nested_exact(left, right)
    assert not nested_exact(left, wrong)


def test_cache_smoke_packager_direct_help():
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo / "scripts/package_routeD_counterfactual_state_restorer_cache_smoke.py"), "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--primary-index" in result.stdout
