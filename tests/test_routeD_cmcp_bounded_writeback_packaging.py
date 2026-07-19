from __future__ import annotations

import copy

import pytest
import torch

from scripts.package_routeD_cmcp_bounded_writeback_interface import (
    structural_diagnostics,
    verify_tensor_tree_exact,
)


def _artifact():
    formal_index = torch.tensor([[0, 1], [0, 0]])
    commit_index = torch.tensor([[0, 0], [0, 0]])
    formal_coord = torch.tensor(
        [[[1.0, 1.0], [4.0, 4.0]], [[2.0, 2.0], [3.0, 3.0]]]
    )
    commit_coord = torch.tensor(
        [[[1.0, 1.0], [2.0, 2.0]], [[2.0, 2.0], [3.0, 3.0]]]
    )
    candidate_formal = torch.stack([formal_coord, formal_coord + 1], dim=2)
    candidate_commit = torch.stack([commit_coord, commit_coord + 1], dim=2)
    return {
        "formal": {
            "selected_candidate_index": formal_index,
            "selected_coords_xy_px": formal_coord,
            "candidate_coords_xy_px": candidate_formal,
        },
        "commit_no_write": {
            "selected_candidate_index": commit_index,
            "selected_coords_xy_px": commit_coord,
            "candidate_coords_xy_px": candidate_commit,
            "first_seen_chunk_start": torch.tensor([0, -7]),
            "decision_native_coords_xy_px": commit_coord,
            "final_native_coords_xy_px": formal_coord,
        },
        "bounded_write": None,
    }


def test_tensor_tree_exact_accepts_exact_copy_and_rejects_difference():
    value = {"a": torch.tensor([1.0]), "b": [True, {"x": 2}]}
    assert verify_tensor_tree_exact(value, copy.deepcopy(value)) == 1
    changed = copy.deepcopy(value)
    changed["a"][0] = 3.0
    with pytest.raises(RuntimeError, match="tensor mismatch"):
        verify_tensor_tree_exact(value, changed)


def test_structural_diagnostics_detects_commit_mismatch():
    row = structural_diagnostics(_artifact())
    assert row["rows"] == 4
    assert row["selected_index_mismatch_rows"] == 1
    assert row["selected_coordinate_mismatch_rows"] == 1
    assert row["formal_non_native_rows"] == 1
    assert row["commit_non_native_rows"] == 0
    assert row["eligible_rows"] == 0
    assert row["eligible_commit_native_vs_final_max_difference_px"] == 0.0


def test_structural_diagnostics_rejects_executed_write():
    artifact = _artifact()
    artifact["bounded_write"] = {"x": torch.tensor(1)}
    with pytest.raises(RuntimeError, match="must not execute"):
        structural_diagnostics(artifact)
