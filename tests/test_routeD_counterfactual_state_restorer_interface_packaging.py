from scripts.package_routeD_counterfactual_state_restorer_interface import _exact

import torch


def test_nested_tensor_exact_comparison():
    left = {"x": [torch.tensor([1.0, 2.0]), {"y": 3}]}
    right = {"x": [torch.tensor([1.0, 2.0]), {"y": 3}]}
    wrong = {"x": [torch.tensor([1.0, 2.1]), {"y": 3}]}
    assert _exact(left, right)
    assert not _exact(left, wrong)


def test_packager_direct_entrypoint_help():
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo / "scripts/package_routeD_counterfactual_state_restorer_interface.py"), "--help"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--primary" in result.stdout
