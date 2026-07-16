"""Inference adapter for a frozen Kubric-trained Route-D tree controller."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from .hypothesis_scorer import MultiThresholdHypothesisScorer
from .routeD_scorer_training import normalize_hypothesis_features


class RouteDTreeSelector:
    """Select local/global candidates with a frozen scorer + ExtraTrees gate.

    The tree controller is fitted and calibrated offline. This class performs no
    fitting and never accesses ground truth.
    """

    def __init__(self, controller_path, device="cpu", threshold=None):
        del threshold
        controller_path = Path(controller_path)
        with open(controller_path, "rb") as handle:
            controller = pickle.load(handle)
        if controller.get("kind") != "routeD_kubric_extratrees_controller":
            raise ValueError(
                f"Unsupported Route-D tree controller kind: {controller.get('kind')}"
            )
        bundle = controller.get("scorer_bundle_payload")
        scorer_bundle_path = controller.get("scorer_bundle")
        if bundle is None:
            if not scorer_bundle_path:
                raise ValueError("Tree controller does not record its scorer bundle")
            bundle = torch.load(
                scorer_bundle_path, map_location="cpu", weights_only=False
            )
        if bundle.get("kind") != "routeD_multithreshold_utility_scorer":
            raise ValueError("Tree controller requires a multi-threshold scorer")
        hidden_dim = int(bundle.get("config", {}).get("hidden_dim", 0))
        if hidden_dim <= 0:
            hidden_dim = int(bundle["model_state"]["network.0.weight"].shape[0])
        threshold_count = int(
            bundle.get("threshold_count", len(bundle.get("thresholds_px", ())))
        )
        self.model = MultiThresholdHypothesisScorer(
            feature_dim=int(bundle["feature_dim"]),
            hidden_dim=hidden_dim,
            threshold_count=threshold_count,
        )
        self.model.load_state_dict(bundle["model_state"], strict=True)
        self.model.to(device).eval()
        self.mean = bundle.get("feature_mean")
        self.std = bundle.get("feature_std")
        self.device = torch.device(device)
        self.thresholds_px = tuple(float(v) for v in bundle["thresholds_px"])
        self.tree = controller["tree"]
        self.isotonic = controller["isotonic"]
        self.policy = dict(controller["policy"])
        self.controller_path = str(controller_path)
        self.scorer_bundle_path = (
            str(scorer_bundle_path) if scorer_bundle_path is not None else None
        )

    @staticmethod
    def _gather_candidate(values: torch.Tensor, index: torch.Tensor):
        return values.gather(
            -2,
            index.unsqueeze(-1).unsqueeze(-1).expand(*index.shape, 1, values.shape[-1]),
        ).squeeze(-2)

    @staticmethod
    def _build_action_features(
        raw_features: torch.Tensor,
        threshold_probabilities: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        local_profile = threshold_probabilities[..., 0, :]
        global_profiles = threshold_probabilities[..., 1:, :]
        global_valid = valid_mask[..., 1:]
        global_coarse = global_profiles[..., 1:].mean(dim=-1)
        global_coarse = global_coarse.masked_fill(~global_valid, -1.0e9)
        best_global_relative = global_coarse.argmax(dim=-1)
        best_global_index = best_global_relative + 1
        has_global = global_valid.any(dim=-1)
        best_global_index = torch.where(
            has_global, best_global_index, torch.zeros_like(best_global_index)
        )

        best_global_profile = RouteDTreeSelector._gather_candidate(
            threshold_probabilities, best_global_index
        )
        best_global_raw = RouteDTreeSelector._gather_candidate(
            raw_features, best_global_index
        )
        local_raw = raw_features[..., 0, :]
        p1_margin = best_global_profile[..., 0] - local_profile[..., 0]
        coarse_gain = (
            best_global_profile[..., 1:].mean(dim=-1)
            - local_profile[..., 1:].mean(dim=-1)
        )
        total_gain = best_global_profile.mean(dim=-1) - local_profile.mean(dim=-1)
        action_features = torch.cat(
            [
                local_profile,
                best_global_profile,
                best_global_profile - local_profile,
                local_raw,
                best_global_raw,
                best_global_raw - local_raw,
                p1_margin.unsqueeze(-1),
                coarse_gain.unsqueeze(-1),
                total_gain.unsqueeze(-1),
            ],
            dim=-1,
        )
        return {
            "features": action_features,
            "best_global_index": best_global_index,
            "has_global": has_global,
            "p1_margin": p1_margin,
            "coarse_gain": coarse_gain,
            "total_gain": total_gain,
        }

    def _tree_probability(self, action_features: torch.Tensor):
        original_shape = action_features.shape[:-1]
        values = (
            action_features.detach()
            .float()
            .cpu()
            .reshape(-1, action_features.shape[-1])
            .numpy()
            .astype(np.float32, copy=False)
        )
        values = np.nan_to_num(values, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        raw = self.tree.predict_proba(values)
        if raw.shape[1] != 2:
            raise RuntimeError("Tree controller did not observe both target classes")
        calibrated = self.isotonic.predict(raw[:, 1])
        return torch.from_numpy(np.asarray(calibrated, dtype=np.float32)).reshape(
            *original_shape
        ).to(self.device)

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
        profile_p1_tolerance=None,
        profile_min_coarse_gain=0.0,
        profile_min_total_gain=0.0,
    ):
        del profile_p1_tolerance, profile_min_coarse_gain, profile_min_total_gain
        raw_features = features.to(self.device, dtype=torch.float32)
        normalized = raw_features
        if self.mean is not None:
            normalized = normalize_hypothesis_features(
                normalized, self.mean, self.std
            )
        threshold_logits = self.model(normalized)
        threshold_probabilities = torch.sigmoid(threshold_logits)
        if valid_mask is None:
            valid_mask = torch.ones(
                raw_features.shape[:-1], device=self.device, dtype=torch.bool
            )
        else:
            valid_mask = valid_mask.to(self.device, dtype=torch.bool)
        action = self._build_action_features(
            raw_features, threshold_probabilities, valid_mask
        )
        probability = self._tree_probability(action["features"])
        policy = self.policy
        eligible = action["has_global"]
        eligible &= probability >= float(policy["probability_threshold"])
        eligible &= action["p1_margin"] >= float(policy["min_p1_margin"])
        eligible &= action["coarse_gain"] >= float(policy["min_coarse_gain"])
        eligible &= action["total_gain"] >= float(policy["min_total_gain"])
        raw_index = torch.where(
            eligible,
            action["best_global_index"],
            torch.zeros_like(action["best_global_index"]),
        )

        candidate_points = candidate_points.to(self.device)
        selected = self._gather_candidate(candidate_points, raw_index)
        index = raw_index
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
                threshold = float(policy["probability_threshold"])
                denominator = max(1.0 - threshold, 1.0e-6)
                calibrated_probability = (
                    (probability - threshold) / denominator
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
            "global_index": action["best_global_index"],
            "switch_distance_px": switch_distance_px,
            "guard_pass": guard_pass,
            "fusion_alpha": fusion_alpha,
            "logits": probability,
            "threshold_probabilities": threshold_probabilities,
            "profile_diagnostics": {
                "p1_margin": action["p1_margin"],
                "coarse_gain": action["coarse_gain"],
                "total_gain": action["total_gain"],
                "best_global_index": action["best_global_index"],
                "eligible": eligible,
            },
        }
