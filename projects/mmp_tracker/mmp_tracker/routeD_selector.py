"""Inference-only Route-D risk-aware selector.

This adapter keeps MMP backbone unchanged. It applies a frozen hypothesis scorer
and calibrated gate to candidate tensors produced by MMP.
"""
from __future__ import annotations

import torch

from .hypothesis_scorer import HypothesisScorer
from .routeD_scorer_training import normalize_hypothesis_features, select_routeD_candidate


class RouteDRiskSelector:
    def __init__(self, bundle_path, device="cpu", threshold=0.4):
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        hidden_dim = int(bundle.get("config", {}).get("hidden_dim", 0))
        if hidden_dim <= 0:
            hidden_dim = int(bundle["model_state"]["network.0.weight"].shape[0])
        self.model = HypothesisScorer(feature_dim=int(bundle["feature_dim"]), hidden_dim=hidden_dim)
        self.model.load_state_dict(bundle["model_state"], strict=True)
        self.model.to(device).eval()
        self.mean = bundle.get("feature_mean")
        self.std = bundle.get("feature_std")
        self.device = device
        self.threshold = float(threshold)

    @torch.no_grad()
    def select(self, features, candidate_points, valid_mask=None):
        features = features.to(self.device, dtype=torch.float32)
        if self.mean is not None:
            features = normalize_hypothesis_features(features, self.mean, self.std)
        logits = self.model(features)
        if valid_mask is None:
            valid_mask = torch.ones_like(logits, dtype=torch.bool)
        index, probability, global_index = select_routeD_candidate(
            logits,
            valid_mask.to(self.device),
            selection_mode="risk_gate",
            gate_threshold=self.threshold,
        )
        selected = candidate_points.to(self.device).gather(
            -2,
            index.unsqueeze(-1).unsqueeze(-1).expand(*index.shape, 1, 2),
        ).squeeze(-2)
        return {
            "points": selected,
            "index": index,
            "gate_probability": probability,
            "global_index": global_index,
            "logits": logits,
        }
