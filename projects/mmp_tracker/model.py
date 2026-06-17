from dataclasses import asdict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    SimpleFeatureEncoder,
    sample_patch_features,
    sample_patch_features_per_point,
    sample_point_features,
    sample_point_features_per_point,
)
from .config import MMPTrackerConfig
from .global_relocator import GlobalRelocator
from .local_matcher import LocalMatcher, LocalMatchOutput, PatchMatcher
from .memory_bank import append_memory, initialize_memory_bank
from .posterior_fusion import PosteriorFusionHead


class MMPTracker(nn.Module):
    def __init__(self, config: Optional[MMPTrackerConfig] = None):
        super().__init__()
        self.config = config or MMPTrackerConfig()
        enc_cfg = self.config.encoder
        self.encoder = SimpleFeatureEncoder(
            in_channels=enc_cfg.in_channels,
            base_dim=enc_cfg.base_dim,
            out_dim=enc_cfg.out_dim,
        )
        self.local_matcher = LocalMatcher(**asdict(self.config.local_matcher))
        refine_template_mode = str(
            getattr(self.config.tracking, "global_refine_template_mode", "point") or "point"
        ).strip().lower()
        if refine_template_mode in {"patch", "tokens"}:
            self.global_refine_matcher = PatchMatcher(
                radius=self.config.tracking.global_refine_radius,
                temperature=self.config.local_matcher.temperature,
            )
        else:
            self.global_refine_matcher = LocalMatcher(
                radius=self.config.tracking.global_refine_radius,
                temperature=self.config.local_matcher.temperature,
            )
        self.global_relocator = GlobalRelocator(
            topk=self.config.global_relocator.topk,
            temperature=self.config.global_relocator.temperature,
        )
        self.posterior_fusion = PosteriorFusionHead(**asdict(self.config.posterior_fusion))
        hidden_dim = int(self.config.posterior_fusion.hidden_dim)
        self.visibility_predictor = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.global_selector = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.candidate_scorer = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.candidate_gate = nn.Sequential(
            nn.Linear(13, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.candidate_ranker = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.commit_selector = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.variant = str(getattr(self.config, "variant", "posterior") or "posterior").strip().lower()

    @staticmethod
    def _gather_by_index(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        if values.ndim < 3:
            raise ValueError("values must have a candidate dimension.")
        gather_shape = list(index.shape) + [1] * (values.ndim - index.ndim)
        gather_index = index.view(*gather_shape)
        expand_shape = list(index.shape) + list(values.shape[index.ndim:])
        gather_index = gather_index.expand(*expand_shape)
        return values.gather(2, gather_index)

    def _refine_global_candidates(
        self,
        frame_feat: torch.Tensor,
        template_feat: torch.Tensor,
        coarse_points: torch.Tensor,
    ):
        batch, num_points, num_candidates, _ = coarse_points.shape
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive for refinement.")
        if template_feat.ndim == 3:
            flat_template = (
                template_feat.unsqueeze(2)
                .expand(batch, num_points, num_candidates, template_feat.shape[-1])
                .reshape(batch, num_points * num_candidates, template_feat.shape[-1])
            )
        elif template_feat.ndim == 4:
            flat_template = (
                template_feat.unsqueeze(2)
                .expand(batch, num_points, num_candidates, template_feat.shape[-2], template_feat.shape[-1])
                .reshape(batch, num_points * num_candidates, template_feat.shape[-2], template_feat.shape[-1])
            )
        else:
            raise ValueError("template_feat must have shape (B, N, C) or (B, N, T, C).")
        flat_centers = coarse_points.reshape(batch, num_points * num_candidates, 2)
        refined = self.global_refine_matcher(frame_feat, flat_template, flat_centers)
        kernel_h, kernel_w = refined.heatmap.shape[-2:]
        refined_points = refined.points.reshape(batch, num_points, num_candidates, 2)
        refined_conf = refined.confidence.reshape(batch, num_points, num_candidates)
        refined_entropy = refined.entropy.reshape(batch, num_points, num_candidates)
        refined_heatmap = refined.heatmap.reshape(batch, num_points, num_candidates, kernel_h, kernel_w)
        return refined_points, refined_conf, refined_entropy, refined_heatmap

    def forward(self, video: torch.Tensor, query_points: torch.Tensor, return_info: bool = False):
        if query_points.ndim != 3 or query_points.shape[-1] != 3:
            raise ValueError("query_points must be shaped as (B, N, 3) with [t, y, x].")
        query_t = query_points[..., 0].round().long()

        features = self.encoder(video)
        batch, time, channels, _, _ = features.shape
        num_points = query_points.shape[1]
        query_t = query_t.clamp(0, time - 1)

        init_points = query_points[..., 1:3].to(video.dtype)
        batch_index = torch.arange(batch, device=video.device).unsqueeze(1).expand(batch, num_points)
        query_frame_feat = features[batch_index, query_t]
        anchor_feat = sample_point_features_per_point(query_frame_feat, init_points)
        template_feat = anchor_feat.clone()
        memory = initialize_memory_bank(anchor_feat, init_points, capacity=self.config.global_relocator.memory_size)
        use_patch_refine = (
            str(getattr(self.config.tracking, "global_refine_template_mode", "point") or "point").strip().lower()
            in {"patch", "tokens"}
        )
        patch_radius = max(int(getattr(self.config.tracking, "global_refine_template_radius", 1)), 0)
        anchor_patch = None
        template_patch = None
        if use_patch_refine:
            anchor_patch = sample_patch_features_per_point(query_frame_feat, init_points, radius=patch_radius)
            template_patch = anchor_patch.clone()

        tracks = torch.zeros(batch, num_points, time, 2, device=video.device, dtype=video.dtype)
        visibility = torch.zeros(batch, num_points, time, device=video.device, dtype=video.dtype)
        confidence = torch.zeros(batch, num_points, time, device=video.device, dtype=video.dtype)
        prior_points = init_points
        prior_confidence = torch.ones(batch, num_points, device=video.device, dtype=video.dtype)
        pending_points = init_points.clone()
        pending_confidence = torch.zeros(batch, num_points, device=video.device, dtype=video.dtype)
        pending_valid = torch.zeros(batch, num_points, device=video.device, dtype=torch.bool)
        debug: Dict[str, list] = {
            "local_confidence": [],
            "global_confidence": [],
            "posterior_confidence": [],
            "selector_probability": [],
            "candidate_logits": [],
            "candidate_probabilities": [],
            "candidate_selected_index": [],
            "candidate_gate_probability": [],
            "candidate_gate_logit": [],
            "candidate_global_rank_logits": [],
            "candidate_global_rank_probabilities": [],
            "candidate_global_selected_index": [],
            "candidate_points": [],
            "global_candidate_points": [],
            "global_candidate_coarse_points": [],
            "local_entropy": [],
            "global_entropy": [],
            "global_heatmap": [],
            "global_points": [],
            "global_coarse_points": [],
            "global_expected_points": [],
            "global_refine_heatmap": [],
            "local_heatmap": [],
            "local_centers": [],
            "local_points": [],
            "prior_points": [],
            "active_mask": [],
            "write_safe_mask": [],
            "commit_mask": [],
            "selected_global_mask": [],
            "commit_probability": [],
            "commit_quality_pass": [],
            "commit_consistency_pass": [],
            "global_local_agreement_px": [],
            "coarse_refine_consistency_px": [],
            "pending_input_mask": [],
            "pending_stage_mask": [],
            "pending_confirm_mask": [],
            "pending_verify_pass": [],
            "pending_reconfirm_pass": [],
            "pending_motion_consistency_px": [],
            "pending_global_reconfirm_px": [],
        }

        for t in range(time):
            frame_feat = features[:, t]
            active_mask = query_t <= t
            just_activated = query_t == t
            prev_prior_points = prior_points
            prev_pending_points = pending_points
            prev_pending_confidence = pending_confidence
            prev_pending_valid = pending_valid
            mixed_template = F.normalize(
                template_feat * (1.0 - self.config.tracking.anchor_mix) + anchor_feat * self.config.tracking.anchor_mix,
                dim=-1,
            )
            global_template = mixed_template
            if use_patch_refine:
                global_template = F.normalize(
                    template_patch * (1.0 - self.config.tracking.anchor_mix) + anchor_patch * self.config.tracking.anchor_mix,
                    dim=-1,
                )
            local_out = self.local_matcher(frame_feat, mixed_template, prev_prior_points)
            global_out = self.global_relocator(frame_feat, memory)
            local_quality = local_out.confidence * (1.0 - local_out.entropy)
            global_quality = global_out.confidence * (1.0 - global_out.entropy)
            px_scale = torch.tensor(
                [max(video.shape[-2] - 1, 1), max(video.shape[-1] - 1, 1)],
                device=video.device,
                dtype=video.dtype,
            )
            global_branch_conf = global_out.confidence
            global_branch_entropy = global_out.entropy
            global_top1 = global_out.points[:, :, 0, :] if global_out.points.shape[2] > 0 else local_out.points
            selected_global = torch.zeros_like(local_quality, dtype=torch.bool)
            write_safe_mask = active_mask.clone()
            commit_probability = torch.zeros_like(local_quality)
            commit_quality_pass = torch.zeros_like(local_quality, dtype=torch.bool)
            commit_consistency_pass = torch.zeros_like(local_quality, dtype=torch.bool)
            local_global_agreement_px = torch.zeros_like(local_quality)
            coarse_refine_consistency_px = torch.zeros_like(local_quality)
            pending_input_mask = torch.zeros_like(local_quality, dtype=torch.bool)
            pending_stage_mask = torch.zeros_like(local_quality, dtype=torch.bool)
            pending_confirm_mask = torch.zeros_like(local_quality, dtype=torch.bool)
            pending_verify_pass = torch.zeros_like(local_quality, dtype=torch.bool)
            pending_reconfirm_pass = torch.zeros_like(local_quality, dtype=torch.bool)
            pending_motion_consistency_px = torch.zeros_like(local_quality)
            pending_global_reconfirm_px = torch.zeros_like(local_quality)
            next_pending_points = prev_pending_points
            next_pending_confidence = prev_pending_confidence
            next_pending_valid = torch.zeros_like(prev_pending_valid)
            candidate_logits = local_quality.unsqueeze(2)
            candidate_probabilities = torch.ones_like(local_quality).unsqueeze(2)
            candidate_selected_index = torch.zeros(batch, num_points, device=video.device, dtype=torch.long)
            candidate_gate_probability = torch.zeros_like(local_quality)
            candidate_gate_logit = torch.zeros_like(local_quality)
            candidate_global_rank_logits = torch.zeros(batch, num_points, 0, device=video.device, dtype=video.dtype)
            candidate_global_rank_probabilities = torch.zeros(batch, num_points, 0, device=video.device, dtype=video.dtype)
            candidate_global_selected_index = torch.zeros(batch, num_points, device=video.device, dtype=torch.long)
            candidate_points = local_out.points.unsqueeze(2)
            global_candidate_points = global_out.points
            global_candidate_coarse_points = global_out.points
            if self.variant == "local":
                current_points = local_out.points
                current_confidence = local_quality
                selector_prob = torch.zeros_like(local_quality)
                global_refine_out = local_out
                global_points = local_out.points
                commit_mask = active_mask.clone()
                commit_points = current_points
                fused = None
            elif self.variant in {"localglobal_topk", "candidate_topk", "topk_abstain"}:
                if global_out.points.shape[2] > 0:
                    num_candidates = global_out.points.shape[2]
                    global_margin = torch.zeros_like(local_quality)
                    if num_candidates > 1:
                        global_margin = global_out.scores[:, :, 0] - global_out.scores[:, :, 1]
                    elif num_candidates == 1:
                        global_margin = global_out.scores[:, :, 0]

                    refined_points, refined_confidence, refined_entropy, refined_heatmap = self._refine_global_candidates(
                        frame_feat,
                        global_template,
                        global_out.points,
                    )
                    global_candidate_coarse_points = global_out.points
                    global_candidate_points = refined_points
                    global_refine_quality = refined_confidence * (1.0 - refined_entropy)
                    global_candidate_quality = 0.5 * (global_out.scores + global_refine_quality)

                    local_dist_to_prior = torch.norm(local_out.points - prev_prior_points, dim=-1)
                    global_dist_to_prior = torch.norm(refined_points - prev_prior_points.unsqueeze(2), dim=-1)
                    global_dist_to_local = torch.norm(refined_points - local_out.points.unsqueeze(2), dim=-1)
                    coarse_refine_disp = torch.norm(refined_points - global_out.points, dim=-1)

                    local_features = torch.stack(
                        [
                            local_quality,
                            local_out.entropy,
                            local_dist_to_prior,
                            torch.zeros_like(local_quality),
                            torch.zeros_like(local_quality),
                            torch.zeros_like(local_quality),
                            global_margin,
                            local_quality,
                            prior_confidence,
                            torch.zeros_like(local_quality),
                        ],
                        dim=-1,
                    ).unsqueeze(2)
                    global_features = torch.stack(
                        [
                            global_candidate_quality,
                            refined_entropy,
                            global_dist_to_prior,
                            global_dist_to_local,
                            coarse_refine_disp,
                            global_out.scores,
                            global_margin.unsqueeze(2).expand(-1, -1, num_candidates),
                            local_quality.unsqueeze(2).expand(-1, -1, num_candidates),
                            prior_confidence.unsqueeze(2).expand(-1, -1, num_candidates),
                            torch.ones_like(global_out.scores),
                        ],
                        dim=-1,
                    )
                    candidate_points = torch.cat([local_out.points.unsqueeze(2), refined_points], dim=2)
                    candidate_quality = torch.cat([local_quality.unsqueeze(2), global_candidate_quality], dim=2)
                    routing_mode = str(
                        getattr(self.config.tracking, "candidate_routing_mode", "flat") or "flat"
                    ).strip().lower()
                    if routing_mode in {"two_stage", "gated_rank", "gate_rank", "gate_then_rank"}:
                        candidate_global_rank_logits = self.candidate_ranker(global_features).squeeze(-1)
                        candidate_global_rank_probabilities = torch.softmax(candidate_global_rank_logits, dim=2)
                        candidate_global_selected_index = candidate_global_rank_probabilities.argmax(dim=2)
                        global_rank_hard = F.one_hot(
                            candidate_global_selected_index,
                            num_classes=num_candidates,
                        ).to(video.dtype)
                        global_rank_gate = (
                            global_rank_hard
                            + candidate_global_rank_probabilities
                            - candidate_global_rank_probabilities.detach()
                        )
                        selected_global_points = (global_rank_gate.unsqueeze(-1) * refined_points).sum(dim=2)
                        selected_global_quality = (global_rank_gate * global_candidate_quality).sum(dim=2)
                        selected_global_entropy = (global_rank_gate * refined_entropy).sum(dim=2)
                        selected_global_coarse = (global_rank_gate.unsqueeze(-1) * global_out.points).sum(dim=2)
                        selected_global_confidence = (global_rank_gate * refined_confidence).sum(dim=2)
                        selected_global_heatmap = self._gather_by_index(refined_heatmap, candidate_global_selected_index).squeeze(2)
                        rank_margin = torch.zeros_like(local_quality)
                        if num_candidates > 1:
                            rank_top2 = torch.topk(candidate_global_rank_logits, k=2, dim=2).values
                            rank_margin = rank_top2[:, :, 0] - rank_top2[:, :, 1]
                        else:
                            rank_margin = candidate_global_rank_logits[:, :, 0]
                        selected_global_rank_probability = (
                            global_rank_gate * candidate_global_rank_probabilities
                        ).sum(dim=2)
                        selected_quality_gap = selected_global_quality - local_quality
                        selected_entropy_gap = local_out.entropy - selected_global_entropy
                        gate_features = torch.stack(
                            [
                                local_quality,
                                local_out.entropy,
                                prior_confidence,
                                selected_global_quality,
                                selected_global_entropy,
                                torch.norm(selected_global_points - prev_prior_points, dim=-1),
                                torch.norm(selected_global_points - local_out.points, dim=-1),
                                torch.norm(selected_global_points - selected_global_coarse, dim=-1),
                                rank_margin,
                                global_margin,
                                selected_global_rank_probability,
                                selected_quality_gap,
                                selected_entropy_gap,
                            ],
                            dim=-1,
                        )
                        candidate_gate_logit = self.candidate_gate(gate_features).squeeze(-1)
                        candidate_gate_probability = torch.sigmoid(candidate_gate_logit)
                        gate_hard = (candidate_gate_probability > self.config.tracking.candidate_gate_threshold).to(video.dtype)
                        gate_st = gate_hard + candidate_gate_probability - candidate_gate_probability.detach()
                        current_points = (
                            gate_st.unsqueeze(-1) * selected_global_points
                            + (1.0 - gate_st).unsqueeze(-1) * local_out.points
                        )
                        current_confidence = gate_st * selected_global_quality + (1.0 - gate_st) * local_quality
                        selected_global = gate_hard > 0.5
                        selector_prob = candidate_gate_probability
                        candidate_selected_index = torch.where(
                            selected_global,
                            candidate_global_selected_index + 1,
                            torch.zeros_like(candidate_global_selected_index),
                        )
                        rank_log_prob = torch.log_softmax(candidate_global_rank_logits, dim=2)
                        combined_log_prob = torch.cat(
                            [
                                torch.log((1.0 - candidate_gate_probability).clamp_min(1.0e-8)).unsqueeze(2),
                                torch.log(candidate_gate_probability.clamp_min(1.0e-8)).unsqueeze(2) + rank_log_prob,
                            ],
                            dim=2,
                        )
                        candidate_logits = combined_log_prob
                        candidate_probabilities = combined_log_prob.exp()
                        global_points = selected_global_points
                        global_top1 = selected_global_coarse
                        global_branch_conf = selected_global_quality
                        global_branch_entropy = selected_global_entropy
                        global_refine_out = LocalMatchOutput(
                            points=selected_global_points,
                            confidence=selected_global_confidence,
                            entropy=selected_global_entropy,
                            heatmap=selected_global_heatmap,
                        )
                        local_global_agreement_px = torch.norm((global_points - local_out.points) * px_scale, dim=-1)
                        coarse_refine_consistency_px = torch.norm((global_points - global_top1) * px_scale, dim=-1)
                    else:
                        candidate_features = torch.cat([local_features, global_features], dim=2)
                        candidate_logits = self.candidate_scorer(candidate_features).squeeze(-1)
                        candidate_prob_soft = torch.softmax(candidate_logits, dim=2)
                        candidate_selected_index = candidate_prob_soft.argmax(dim=2)
                        candidate_onehot_hard = F.one_hot(candidate_selected_index, num_classes=1 + num_candidates).to(video.dtype)
                        candidate_gate = candidate_onehot_hard + candidate_prob_soft - candidate_prob_soft.detach()
                        current_points = (candidate_gate.unsqueeze(-1) * candidate_points).sum(dim=2)
                        current_confidence = (candidate_gate * candidate_quality).sum(dim=2)
                        candidate_probabilities = candidate_prob_soft
                        selected_global = candidate_selected_index > 0
                        selector_prob = candidate_prob_soft[:, :, 1:].sum(dim=2)
                        global_candidate_logits = candidate_logits[:, :, 1:]
                        global_candidate_probs = torch.softmax(global_candidate_logits, dim=2)
                        global_points = (global_candidate_probs.unsqueeze(-1) * refined_points).sum(dim=2)
                        global_top1 = global_out.points[:, :, 0, :]
                        global_branch_conf = (global_candidate_probs * global_candidate_quality).sum(dim=2)
                        global_branch_entropy = (global_candidate_probs * refined_entropy).sum(dim=2)
                        global_refine_out = LocalMatchOutput(
                            points=refined_points[:, :, 0, :],
                            confidence=refined_confidence[:, :, 0],
                            entropy=refined_entropy[:, :, 0],
                            heatmap=refined_heatmap[:, :, 0, :, :],
                        )
                        local_global_agreement_px = torch.norm((global_points - local_out.points) * px_scale, dim=-1)
                        coarse_refine_consistency_px = (
                            global_candidate_probs
                            * torch.norm((refined_points - global_out.points) * px_scale.unsqueeze(0).unsqueeze(0), dim=-1)
                        ).sum(dim=2)
                    commit_mask = torch.zeros_like(selected_global)
                    commit_points = current_points
                    write_safe_mask = active_mask.clone()
                else:
                    current_points = local_out.points
                    current_confidence = local_quality
                    selector_prob = torch.zeros_like(local_quality)
                    global_refine_out = local_out
                    global_points = local_out.points
                    commit_mask = torch.zeros_like(local_quality, dtype=torch.bool)
                    commit_points = current_points
                    write_safe_mask = active_mask.clone()
                fused = None
            elif self.variant == "localglobal":
                if global_out.points.shape[2] > 0:
                    global_top1 = global_out.points[:, :, 0, :]
                    refine_center_mode = str(getattr(self.config.tracking, "global_refine_center", "expected") or "expected").strip().lower()
                    global_center = global_top1 if refine_center_mode == "top1" else global_out.expected_points
                    global_refine_out = self.global_refine_matcher(frame_feat, global_template, global_center)
                    global_refine_quality = global_refine_out.confidence * (1.0 - global_refine_out.entropy)
                    global_points = global_refine_out.points
                    global_conf = 0.5 * (global_quality + global_refine_quality)
                    global_branch_conf = global_conf
                    global_branch_entropy = global_refine_out.entropy
                    selector_features = torch.stack(
                        [
                            prior_confidence,
                            local_out.confidence,
                            local_out.entropy,
                            global_conf,
                            global_refine_out.entropy,
                            torch.norm(global_points - prev_prior_points, dim=-1),
                            torch.norm(global_points - local_out.points, dim=-1),
                        ],
                        dim=-1,
                    )
                    selector_prob = torch.sigmoid(self.global_selector(selector_features)).squeeze(-1)
                    selector_hard = (selector_prob > self.config.tracking.selector_threshold).to(local_out.points.dtype)
                    selector_gate = selector_hard + selector_prob - selector_prob.detach()
                    selected_global = selector_hard > 0.5
                    current_points = (
                        selector_gate.unsqueeze(-1) * global_points
                        + (1.0 - selector_gate).unsqueeze(-1) * local_out.points
                    )
                    current_confidence = selector_gate * global_conf + (1.0 - selector_gate) * local_quality
                    local_global_agreement_px = torch.norm((global_points - local_out.points) * px_scale, dim=-1)
                    coarse_refine_consistency_px = torch.norm((global_points - global_center) * px_scale, dim=-1)
                    commit_quality_pass = global_conf >= self.config.tracking.global_commit_min_quality
                    commit_consistency_pass = (
                        coarse_refine_consistency_px <= self.config.tracking.global_commit_consistency_px
                    )
                    commit_features = torch.stack(
                        [
                            prior_confidence,
                            local_out.confidence,
                            local_out.entropy,
                            global_conf,
                            global_refine_out.entropy,
                            torch.norm(global_points - prev_prior_points, dim=-1),
                            torch.norm(global_points - local_out.points, dim=-1),
                            torch.norm(global_points - global_center, dim=-1),
                        ],
                        dim=-1,
                    )
                    commit_probability = torch.sigmoid(self.commit_selector(commit_features)).squeeze(-1)
                    commit_mode = str(getattr(self.config.tracking, "commit_mode", "heuristic") or "heuristic").strip().lower()
                    commit_points = current_points
                    if commit_mode == "selector":
                        commit_hard = commit_probability > self.config.tracking.commit_threshold
                        commit_mask = active_mask & selected_global & commit_hard
                    elif commit_mode in {"none", "disabled", "nocommit", "no_commit"}:
                        commit_mask = torch.zeros_like(selected_global)
                    elif commit_mode == "always_selected":
                        commit_mask = active_mask & selected_global
                    elif commit_mode == "deferred":
                        pending_input_mask = active_mask & prev_pending_valid
                        pending_refine_out = self.global_refine_matcher(frame_feat, global_template, prev_pending_points)
                        pending_refine_quality = pending_refine_out.confidence * (1.0 - pending_refine_out.entropy)
                        pending_motion_consistency_px = torch.norm(
                            (pending_refine_out.points - prev_pending_points) * px_scale,
                            dim=-1,
                        )
                        pending_global_reconfirm_px = torch.norm(
                            (global_points - prev_pending_points) * px_scale,
                            dim=-1,
                        )
                        pending_verify_quality_pass = (
                            pending_refine_quality >= self.config.tracking.deferred_commit_min_quality
                        )
                        pending_global_quality_pass = (
                            global_conf >= self.config.tracking.deferred_commit_min_quality
                        )
                        pending_motion_pass = (
                            pending_motion_consistency_px <= self.config.tracking.deferred_commit_confirm_px
                        )
                        pending_verify_pass = pending_verify_quality_pass & pending_motion_pass
                        pending_reconfirm_pass = pending_global_quality_pass & (
                            pending_global_reconfirm_px <= self.config.tracking.deferred_commit_reconfirm_px
                        )
                        pending_confirm_mask = pending_input_mask & (
                            pending_verify_pass | pending_reconfirm_pass
                        )
                        pending_commit_uses_refine = pending_verify_pass
                        pending_commit_points = torch.where(
                            pending_commit_uses_refine.unsqueeze(-1),
                            pending_refine_out.points,
                            global_points,
                        )
                        pending_commit_conf = torch.where(
                            pending_commit_uses_refine,
                            pending_refine_quality,
                            global_conf,
                        )
                        current_points = torch.where(
                            pending_confirm_mask.unsqueeze(-1),
                            pending_commit_points,
                            current_points,
                        )
                        current_confidence = torch.where(
                            pending_confirm_mask,
                            pending_commit_conf,
                            current_confidence,
                        )
                        commit_points = pending_commit_points
                        commit_mask = pending_confirm_mask
                        stage_interest_mask = selected_global | (
                            local_global_agreement_px >= self.config.tracking.deferred_stage_min_disagreement_px
                        )
                        pending_stage_mask = (
                            active_mask
                            & commit_quality_pass
                            & commit_consistency_pass
                            & (~pending_confirm_mask)
                            & stage_interest_mask
                        )
                        next_pending_points = torch.where(
                            pending_stage_mask.unsqueeze(-1),
                            global_points,
                            prev_pending_points,
                        )
                        next_pending_confidence = torch.where(
                            pending_stage_mask,
                            global_conf,
                            torch.zeros_like(global_conf),
                        )
                        next_pending_valid = pending_stage_mask
                        write_hold_mask = (pending_input_mask & (~pending_confirm_mask)) | pending_stage_mask
                        write_safe_mask = active_mask & ((~write_hold_mask) | commit_mask)
                    else:
                        commit_mask = active_mask & selected_global & commit_quality_pass & commit_consistency_pass
                    if commit_mode != "deferred":
                        write_safe_mask = active_mask & (
                            (~selected_global)
                            | commit_mask
                            | (local_global_agreement_px <= self.config.tracking.safe_write_agreement_px)
                        )
                else:
                    current_points = local_out.points
                    current_confidence = local_quality
                    selector_prob = torch.zeros_like(local_quality)
                    global_refine_out = local_out
                    global_points = local_out.points
                    global_center = local_out.points
                    commit_mask = torch.zeros_like(local_quality, dtype=torch.bool)
                    commit_points = current_points
                    write_safe_mask = active_mask.clone()
                fused = None
            else:
                fused = self.posterior_fusion(
                    prior_points=prev_prior_points,
                    prior_conf=prior_confidence,
                    local_points=local_out.points,
                    local_conf=local_quality,
                    local_entropy=local_out.entropy,
                    global_points=global_out.points,
                    global_scores=global_out.scores,
                    global_entropy=global_out.entropy,
                )
                current_points = fused.points
                current_confidence = fused.confidence
                selector_prob = torch.zeros_like(current_confidence)
                global_refine_out = local_out
                global_points = local_out.points
                commit_mask = active_mask.clone()
                commit_points = current_points
                write_safe_mask = active_mask.clone()

            visibility_features = torch.stack(
                [
                    prior_confidence,
                    local_out.confidence,
                    local_out.entropy,
                    global_branch_conf,
                    global_branch_entropy,
                    current_confidence,
                ],
                dim=-1,
            )
            predicted_visibility = torch.sigmoid(self.visibility_predictor(visibility_features)).squeeze(-1)
            if fused is not None:
                current_visibility = 0.5 * fused.visibility + 0.5 * predicted_visibility
            else:
                current_visibility = predicted_visibility

            current_points = torch.where(just_activated.unsqueeze(-1), init_points, current_points)
            current_visibility = torch.where(just_activated, torch.ones_like(current_visibility), current_visibility)
            current_confidence = torch.where(just_activated, torch.ones_like(current_confidence), current_confidence)
            current_points = torch.where(active_mask.unsqueeze(-1), current_points, prev_prior_points)
            current_visibility = torch.where(active_mask, current_visibility, torch.zeros_like(current_visibility))
            current_confidence = torch.where(active_mask, current_confidence, torch.zeros_like(current_confidence))
            selector_prob = torch.where(active_mask, selector_prob, torch.zeros_like(selector_prob))
            selected_global = torch.where(active_mask, selected_global, torch.zeros_like(selected_global))
            commit_mask = torch.where(active_mask, commit_mask, torch.zeros_like(commit_mask))
            write_safe_mask = torch.where(active_mask, write_safe_mask, torch.zeros_like(write_safe_mask))
            commit_probability = torch.where(active_mask, commit_probability, torch.zeros_like(commit_probability))
            commit_quality_pass = torch.where(active_mask, commit_quality_pass, torch.zeros_like(commit_quality_pass))
            commit_consistency_pass = torch.where(active_mask, commit_consistency_pass, torch.zeros_like(commit_consistency_pass))
            local_global_agreement_px = torch.where(active_mask, local_global_agreement_px, torch.zeros_like(local_global_agreement_px))
            coarse_refine_consistency_px = torch.where(active_mask, coarse_refine_consistency_px, torch.zeros_like(coarse_refine_consistency_px))
            pending_input_mask = torch.where(active_mask, pending_input_mask, torch.zeros_like(pending_input_mask))
            pending_stage_mask = torch.where(active_mask, pending_stage_mask, torch.zeros_like(pending_stage_mask))
            pending_confirm_mask = torch.where(active_mask, pending_confirm_mask, torch.zeros_like(pending_confirm_mask))
            pending_verify_pass = torch.where(active_mask, pending_verify_pass, torch.zeros_like(pending_verify_pass))
            pending_reconfirm_pass = torch.where(active_mask, pending_reconfirm_pass, torch.zeros_like(pending_reconfirm_pass))
            pending_motion_consistency_px = torch.where(
                active_mask,
                pending_motion_consistency_px,
                torch.zeros_like(pending_motion_consistency_px),
            )
            pending_global_reconfirm_px = torch.where(
                active_mask,
                pending_global_reconfirm_px,
                torch.zeros_like(pending_global_reconfirm_px),
            )

            tracks[:, :, t] = current_points
            visibility[:, :, t] = current_visibility
            confidence[:, :, t] = current_confidence

            state_points = torch.where(commit_mask.unsqueeze(-1), commit_points, local_out.points)
            state_confidence = torch.where(commit_mask, current_confidence, local_quality)
            current_desc = sample_point_features(frame_feat, state_points)
            update_mask = (
                (current_visibility > self.config.tracking.visibility_update_threshold)
                & (state_confidence > self.config.tracking.visibility_update_threshold)
                & write_safe_mask
                & active_mask
            ).unsqueeze(-1)
            updated_template = F.normalize(
                template_feat * self.config.tracking.template_momentum
                + current_desc * (1.0 - self.config.tracking.template_momentum),
                dim=-1,
            )
            template_feat = torch.where(update_mask, updated_template, template_feat)
            if use_patch_refine:
                current_patch = sample_patch_features(frame_feat, state_points, radius=patch_radius)
                updated_patch = F.normalize(
                    template_patch * self.config.tracking.template_momentum
                    + current_patch * (1.0 - self.config.tracking.template_momentum),
                    dim=-1,
                )
                template_patch = torch.where(update_mask.unsqueeze(-1), updated_patch, template_patch)
            memory = append_memory(
                memory,
                descriptors=current_desc,
                positions=state_points,
                visibility=(
                    active_mask
                    & write_safe_mask
                    & (current_visibility > self.config.tracking.visibility_update_threshold)
                    & (state_confidence > self.config.tracking.visibility_update_threshold)
                ).to(video.dtype),
                capacity=self.config.global_relocator.memory_size,
            )
            prior_points = torch.where(active_mask.unsqueeze(-1), state_points, prev_prior_points)
            prior_confidence = torch.where(active_mask, state_confidence, prior_confidence)
            pending_points = torch.where(next_pending_valid.unsqueeze(-1), next_pending_points, state_points)
            pending_confidence = torch.where(next_pending_valid, next_pending_confidence, torch.zeros_like(next_pending_confidence))
            pending_valid = next_pending_valid

            debug["local_confidence"].append(local_out.confidence)
            debug["global_confidence"].append(global_branch_conf)
            debug["local_entropy"].append(local_out.entropy)
            debug["global_entropy"].append(global_branch_entropy)
            debug["selector_probability"].append(selector_prob)
            debug["candidate_logits"].append(candidate_logits)
            debug["candidate_probabilities"].append(candidate_probabilities)
            debug["candidate_selected_index"].append(candidate_selected_index)
            debug["candidate_gate_probability"].append(candidate_gate_probability)
            debug["candidate_gate_logit"].append(candidate_gate_logit)
            debug["candidate_global_rank_logits"].append(candidate_global_rank_logits)
            debug["candidate_global_rank_probabilities"].append(candidate_global_rank_probabilities)
            debug["candidate_global_selected_index"].append(candidate_global_selected_index)
            debug["candidate_points"].append(candidate_points)
            debug["global_candidate_points"].append(global_candidate_points)
            debug["global_candidate_coarse_points"].append(global_candidate_coarse_points)
            debug["global_heatmap"].append(global_out.heatmap)
            debug["global_points"].append(global_points)
            debug["global_coarse_points"].append(global_top1 if global_out.points.shape[2] > 0 else local_out.points)
            debug["global_expected_points"].append(global_out.expected_points)
            debug["global_refine_heatmap"].append(global_refine_out.heatmap)
            debug["local_heatmap"].append(local_out.heatmap)
            debug["local_centers"].append(prev_prior_points)
            debug["local_points"].append(local_out.points)
            debug["prior_points"].append(prev_prior_points)
            debug["active_mask"].append(active_mask)
            debug["write_safe_mask"].append(write_safe_mask)
            debug["commit_mask"].append(commit_mask)
            debug["selected_global_mask"].append(selected_global)
            debug["commit_probability"].append(commit_probability)
            debug["commit_quality_pass"].append(commit_quality_pass)
            debug["commit_consistency_pass"].append(commit_consistency_pass)
            debug["global_local_agreement_px"].append(local_global_agreement_px)
            debug["coarse_refine_consistency_px"].append(coarse_refine_consistency_px)
            debug["pending_input_mask"].append(pending_input_mask)
            debug["pending_stage_mask"].append(pending_stage_mask)
            debug["pending_confirm_mask"].append(pending_confirm_mask)
            debug["pending_verify_pass"].append(pending_verify_pass)
            debug["pending_reconfirm_pass"].append(pending_reconfirm_pass)
            debug["pending_motion_consistency_px"].append(pending_motion_consistency_px)
            debug["pending_global_reconfirm_px"].append(pending_global_reconfirm_px)
            if fused is not None:
                debug["posterior_confidence"].append(fused.confidence)
            else:
                debug["posterior_confidence"].append(current_confidence)

        if not return_info:
            return tracks, visibility

        info = {
            "confidence": confidence,
            "debug": {key: torch.stack(value, dim=2) for key, value in debug.items()},
            "local_heatmap": torch.stack(debug["local_heatmap"], dim=2),
            "local_centers": torch.stack(debug["local_centers"], dim=2),
            "local_points": torch.stack(debug["local_points"], dim=2),
            "prior_points": torch.stack(debug["prior_points"], dim=2),
            "global_heatmap": torch.stack(debug["global_heatmap"], dim=2),
            "global_points": torch.stack(debug["global_points"], dim=2),
            "global_coarse_points": torch.stack(debug["global_coarse_points"], dim=2),
            "global_expected_points": torch.stack(debug["global_expected_points"], dim=2),
            "global_refine_heatmap": torch.stack(debug["global_refine_heatmap"], dim=2),
            "selector_probability": torch.stack(debug["selector_probability"], dim=2),
            "candidate_logits": torch.stack(debug["candidate_logits"], dim=2),
            "candidate_probabilities": torch.stack(debug["candidate_probabilities"], dim=2),
            "candidate_selected_index": torch.stack(debug["candidate_selected_index"], dim=2),
            "candidate_gate_probability": torch.stack(debug["candidate_gate_probability"], dim=2),
            "candidate_gate_logit": torch.stack(debug["candidate_gate_logit"], dim=2),
            "candidate_global_rank_logits": torch.stack(debug["candidate_global_rank_logits"], dim=2),
            "candidate_global_rank_probabilities": torch.stack(debug["candidate_global_rank_probabilities"], dim=2),
            "candidate_global_selected_index": torch.stack(debug["candidate_global_selected_index"], dim=2),
            "candidate_points": torch.stack(debug["candidate_points"], dim=2),
            "global_candidate_points": torch.stack(debug["global_candidate_points"], dim=2),
            "global_candidate_coarse_points": torch.stack(debug["global_candidate_coarse_points"], dim=2),
            "active_mask": torch.stack(debug["active_mask"], dim=2),
            "write_safe_mask": torch.stack(debug["write_safe_mask"], dim=2),
            "commit_mask": torch.stack(debug["commit_mask"], dim=2),
            "selected_global_mask": torch.stack(debug["selected_global_mask"], dim=2),
            "commit_probability": torch.stack(debug["commit_probability"], dim=2),
            "commit_quality_pass": torch.stack(debug["commit_quality_pass"], dim=2),
            "commit_consistency_pass": torch.stack(debug["commit_consistency_pass"], dim=2),
            "global_local_agreement_px": torch.stack(debug["global_local_agreement_px"], dim=2),
            "coarse_refine_consistency_px": torch.stack(debug["coarse_refine_consistency_px"], dim=2),
            "pending_input_mask": torch.stack(debug["pending_input_mask"], dim=2),
            "pending_stage_mask": torch.stack(debug["pending_stage_mask"], dim=2),
            "pending_confirm_mask": torch.stack(debug["pending_confirm_mask"], dim=2),
            "pending_verify_pass": torch.stack(debug["pending_verify_pass"], dim=2),
            "pending_reconfirm_pass": torch.stack(debug["pending_reconfirm_pass"], dim=2),
            "pending_motion_consistency_px": torch.stack(debug["pending_motion_consistency_px"], dim=2),
            "pending_global_reconfirm_px": torch.stack(debug["pending_global_reconfirm_px"], dim=2),
            "query_t": query_t,
            "selector_target_margin": self.config.tracking.selector_target_margin,
            "candidate_gate_positive_weight": self.config.tracking.candidate_gate_positive_weight,
            "candidate_gate_focal_gamma": self.config.tracking.candidate_gate_focal_gamma,
            "candidate_gate_margin": self.config.tracking.candidate_gate_margin,
            "candidate_gate_margin_weight": self.config.tracking.candidate_gate_margin_weight,
            "candidate_rank_soft_temperature": self.config.tracking.candidate_rank_soft_temperature,
            "commit_target_margin": self.config.tracking.commit_target_margin,
            "input_height": video.shape[-2],
            "input_width": video.shape[-1],
            "feature_height": features.shape[-2],
            "feature_width": features.shape[-1],
            "local_radius": self.local_matcher.radius,
        }
        return tracks, visibility, info
