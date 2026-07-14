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
    def select(
        self,
        features,
        candidate_points,
        valid_mask=None,
        *,
        fallback_points=None,
        max_switch_distance_px=None,
        pixel_scale=None,
        fusion_strength=None,
        fusion_power=1.0,
    ):
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
        candidate_points = candidate_points.to(self.device)
        selected = candidate_points.gather(
            -2,
            index.unsqueeze(-1).unsqueeze(-1).expand(*index.shape, 1, 2),
        ).squeeze(-2)
        raw_index = index
        switch_distance_px = torch.zeros_like(probability)
        guard_pass = torch.ones_like(index, dtype=torch.bool)
        fusion_alpha = torch.zeros_like(probability)
        if fallback_points is not None:
            fallback_points = fallback_points.to(self.device, dtype=selected.dtype)
            if fallback_points.shape != selected.shape:
                raise ValueError("fallback_points must match selected point shape")
            if max_switch_distance_px is not None:
                if pixel_scale is None:
                    raise ValueError("pixel_scale is required for a pixel-distance guard")
                scale = torch.as_tensor(
                    pixel_scale, device=self.device, dtype=selected.dtype
                )
                if tuple(scale.shape) != (2,):
                    raise ValueError("pixel_scale must contain [height-1, width-1]")
                switch_distance_px = torch.norm(
                    (selected - fallback_points) * scale, dim=-1
                )
                guard_pass = switch_distance_px <= float(max_switch_distance_px)
            effective_global = (raw_index > 0) & guard_pass
            if fusion_strength is None:
                fusion_alpha = effective_global.to(selected.dtype)
            else:
                strength = float(fusion_strength)
                if not 0.0 <= strength <= 1.0:
                    raise ValueError("fusion_strength must be in [0, 1]")
                power = float(fusion_power)
                if power <= 0.0:
                    raise ValueError("fusion_power must be positive")
                denominator = max(1.0 - self.threshold, 1.0e-6)
                calibrated_probability = (
                    (probability - self.threshold) / denominator
                ).clamp(0.0, 1.0)
                fusion_alpha = (
                    strength
                    * calibrated_probability.pow(power)
                    * effective_global.to(selected.dtype)
                )
            selected = (
                fallback_points
                + fusion_alpha.unsqueeze(-1) * (selected - fallback_points)
            )
            index = torch.where(
                fusion_alpha > 0.0, raw_index, torch.zeros_like(raw_index)
            )
        return {
            "points": selected,
            "index": index,
            "raw_index": raw_index,
            "gate_probability": probability,
            "global_index": global_index,
            "switch_distance_px": switch_distance_px,
            "guard_pass": guard_pass,
            "fusion_alpha": fusion_alpha,
            "logits": logits,
        }
