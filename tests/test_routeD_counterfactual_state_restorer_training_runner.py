import subprocess
import sys
from pathlib import Path

from scripts.train_routeD_counterfactual_state_restorer_gate2 import _candidate_better


def test_checkpoint_tie_break_order():
    best = {
        "checkpoint_score": 0.1,
        "forced_failure": {"variants": {"full_state": {"mean_l2_error_px": 10.0}}},
        "learned_gate": {"clean_false_apply_rate": 0.1},
    }
    better_score = {
        "checkpoint_score": 0.2,
        "forced_failure": {"variants": {"full_state": {"mean_l2_error_px": 20.0}}},
        "learned_gate": {"clean_false_apply_rate": 0.2},
    }
    tie_better_error = {
        "checkpoint_score": 0.1,
        "forced_failure": {"variants": {"full_state": {"mean_l2_error_px": 9.0}}},
        "learned_gate": {"clean_false_apply_rate": 0.2},
    }
    assert _candidate_better(better_score, best)
    assert _candidate_better(tie_better_error, best)


def test_training_runner_direct_help():
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo / "scripts/train_routeD_counterfactual_state_restorer_gate2.py"), "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--interface-smoke" in result.stdout
