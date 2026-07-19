import subprocess
import sys
from pathlib import Path

import torch

from scripts.package_routeD_counterfactual_state_restorer_training import nested_exact


def test_training_package_nested_exact():
    left = {"state": {"x": torch.tensor([1.0, 2.0])}, "epoch": 2}
    right = {"state": {"x": torch.tensor([1.0, 2.0])}, "epoch": 2}
    wrong = {"state": {"x": torch.tensor([1.0, 3.0])}, "epoch": 2}
    assert nested_exact(left, right)
    assert not nested_exact(left, wrong)


def test_training_packager_direct_help():
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo / "scripts/package_routeD_counterfactual_state_restorer_training.py"), "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--primary" in result.stdout
    assert "--replay" in result.stdout
