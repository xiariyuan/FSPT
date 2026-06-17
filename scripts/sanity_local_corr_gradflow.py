#!/usr/bin/env python3
"""
Sanity check: local correlation + confidence gate must allow gradients.

This script is intentionally self-contained and does NOT depend on:
  - CoTracker installation
  - datasets
  - GPU

It verifies:
  1) LocalTrackCorrelation forward works on random tensors.
  2) _confidence_gate produces non-zero gates for conf>0.
  3) Backprop through (step * gate) yields non-zero grads on corr parameters.

Run:
  python scripts/sanity_local_corr_gradflow.py
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    try:
        import torch
    except Exception as exc:
        print(f"torch not available: {exc}")
        return 0

    from models.cotracker_refiner import LocalTrackCorrelation, _confidence_gate

    torch.manual_seed(0)

    # Small shapes: fast on CPU.
    B, T, H, W, C = 1, 4, 16, 16, 64
    N = 8

    feature_map = torch.randn(B, T, H, W, C)
    center_point_features = torch.randn(B, T, N, C)
    tracks_bt = torch.rand(B, T, N, 2)
    query_t = torch.zeros(B, N, dtype=torch.long)

    module = LocalTrackCorrelation(
        in_dim=C,
        out_dim=C,
        window_size=7,
        corr_dim=32,
        hidden_dim=64,
        scale=1.0,
        normalize=True,
        temperature=1.0,
    )
    module.train()

    emb, step, conf = module(
        feature_map,
        center_point_features,
        tracks_bt,
        query_t,
        return_step=True,
    )

    # Gate with a fairly strict threshold; should still be >0 for conf>0.
    gate = _confidence_gate(conf, threshold=0.15, power=2.0)
    step_gated = step * gate.unsqueeze(-1)

    loss = emb.pow(2).mean() + step_gated.pow(2).mean()
    loss.backward()

    grad_sum = 0.0
    for p in module.parameters():
        if p.grad is not None:
            grad_sum += float(p.grad.abs().sum().item())

    print(f"conf[min,max]=({conf.min().item():.6f},{conf.max().item():.6f})")
    print(f"gate[min,max]=({gate.min().item():.6f},{gate.max().item():.6f})")
    print(f"grad_sum={grad_sum:.6f}")

    if grad_sum <= 0:
        raise AssertionError("No gradients through local correlation + gate. This indicates a dead zone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
