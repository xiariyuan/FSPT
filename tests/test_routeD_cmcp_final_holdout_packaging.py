from __future__ import annotations

import copy
import pytest

from scripts.package_routeD_cmcp_final_holdout_result import verify_eval_replay


def _row():
    return {
        "schema_version":"x","seed":17,"config_sha256":"c","cache_index_sha256":"i","cache_index_payload_sha256":"p",
        "checkpoint_sha256":"k","combined_model_state_sha256":"m","adapter_state_sha256":"a","cmcp_state_sha256":"g","comparator_state_sha256":"q",
        "normalization":{"x":1},"metrics":{"AJ":1},"model_selection_on_final_holdout":False,"calibration_read":False,"external_data_read":{"d":False},
    }


def test_eval_replay_exact_except_paths_not_compared():
    primary=_row(); replay=copy.deepcopy(primary); primary["output"]="a"; replay["output"]="b"
    assert all(verify_eval_replay(primary,replay).values())


def test_eval_replay_rejects_metric_drift():
    primary=_row(); replay=copy.deepcopy(primary); replay["metrics"]["AJ"]=2
    with pytest.raises(RuntimeError,match="replay mismatch"):
        verify_eval_replay(primary,replay)
