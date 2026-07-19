import subprocess
import sys
from pathlib import Path


def test_complete_cache_packager_direct_help():
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo / "scripts/package_routeD_counterfactual_state_restorer_cache.py"), "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--train-index" in result.stdout
    assert "--validation-index" in result.stdout
