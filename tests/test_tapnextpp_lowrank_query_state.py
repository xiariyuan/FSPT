import torch

from scripts.audit_tapnextpp_lowrank_query_state import (
    evaluate_gate,
    fit_block_normalized_pca,
    project_delta,
)


def test_pca_projection_reconstructs_training_subspace():
    matrix = torch.tensor([
        [1.0, 0.0, 2.0, 0.0],
        [2.0, 0.0, 4.0, 0.0],
        [3.0, 0.0, 6.0, 0.0],
    ])
    fit = fit_block_normalized_pca(matrix, block_sizes=(2, 2), max_rank=2)
    reconstructed = project_delta(
        matrix[1],
        block_rms=fit["block_rms"],
        block_sizes=(2, 2),
        mean_normalized=fit["mean_normalized"],
        components=fit["components"],
        rank=1,
    )
    assert torch.allclose(reconstructed, matrix[1], atol=1e-5)


def test_rank_zero_returns_basis_mean_delta():
    matrix = torch.tensor([[1.0, 3.0], [3.0, 5.0]])
    fit = fit_block_normalized_pca(matrix, block_sizes=(1, 1), max_rank=1)
    projected = project_delta(
        torch.tensor([9.0, 9.0]),
        block_rms=fit["block_rms"],
        block_sizes=(1, 1),
        mean_normalized=fit["mean_normalized"],
        components=fit["components"],
        rank=0,
    )
    assert torch.allclose(projected, matrix.mean(0), atol=1e-5)


def row(scene, full, rank8, retained=0.8, frames=0.8):
    return {
        "scene": scene,
        "status": "complete",
        "delta": {
            "full_query_gain_vs_native": full,
            "rank8_gain_vs_native": rank8,
            "rank8_retained_full_query_gain": retained,
            "rank8_improved_frame_fraction": frames,
        },
    }


def test_lowrank_gate_passes_strong_ten_scene_signal():
    rows = [row(str(i), 3.0, 2.5) for i in range(10)]
    gate = evaluate_gate(rows, expected=10)
    assert gate["pass"]
    assert not gate["training_allowed"]


def test_lowrank_gate_fails_on_incomplete_or_harmful_scene():
    incomplete = [row(str(i), 3.0, 2.5) for i in range(9)]
    assert not evaluate_gate(incomplete, expected=10)["pass"]
    harmful = [row(str(i), 3.0, 2.5) for i in range(9)] + [row("bad", 3.0, -1.1)]
    gate = evaluate_gate(harmful, expected=10)
    assert not gate["pass"]
    assert not gate["checks"]["rank8_no_scene_regression_worse_than_1px"]
