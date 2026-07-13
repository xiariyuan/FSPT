from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MMPLossWeights:
    coordinate: float = 1.0
    local_heatmap: float = 0.0
    global_heatmap: float = 0.0
    global_coordinate: float = 0.0
    selector: float = 0.0
    commit: float = 0.0
    visibility: float = 0.2
    posterior_consistency: float = 0.0
    temporal_smoothness: float = 0.0
    no_harm_prior: float = 0.0
    long_occ_focus: float = 0.0


class MMPTrackingLoss(nn.Module):
    def __init__(self, weights: Optional[MMPLossWeights] = None):
        super().__init__()
        self.weights = weights or MMPLossWeights()

    def forward(
        self,
        pred_tracks: torch.Tensor,
        pred_visibility: torch.Tensor,
        gt_tracks: torch.Tensor,
        gt_visibility: torch.Tensor,
        info: Optional[Dict[str, torch.Tensor]] = None,
        prior_tracks: Optional[torch.Tensor] = None,
        focus_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        visible_mask = gt_visibility > 0.5
        visible_mask_f = visible_mask.to(pred_tracks.dtype)
        active_mask = info.get("active_mask", None) if info is not None else None
        selector_margin = 0.01 if info is None else float(info.get("selector_target_margin", 0.01))
        commit_margin = 0.01 if info is None else float(info.get("commit_target_margin", 0.01))
        candidate_gate_positive_weight = 1.0 if info is None else float(info.get("candidate_gate_positive_weight", 1.0))
        candidate_gate_focal_gamma = 0.0 if info is None else float(info.get("candidate_gate_focal_gamma", 0.0))
        candidate_gate_margin = 0.0 if info is None else float(info.get("candidate_gate_margin", 0.0))
        candidate_gate_margin_weight = 0.0 if info is None else float(info.get("candidate_gate_margin_weight", 0.0))
        candidate_rank_soft_temperature = 0.0 if info is None else float(info.get("candidate_rank_soft_temperature", 0.0))
        candidate_routing_mode = "" if info is None else str(info.get("candidate_routing_mode", "")).strip().lower()

        dy = pred_tracks[..., 0] - gt_tracks[..., 0]
        dx = pred_tracks[..., 1] - gt_tracks[..., 1]
        input_height = None if info is None else info.get("input_height", None)
        input_width = None if info is None else info.get("input_width", None)
        if input_height is not None and input_width is not None:
            scale_y = max(int(input_height) - 1, 1)
            scale_x = max(int(input_width) - 1, 1)
            coord_error_px = torch.sqrt((dy * float(scale_y)).square() + (dx * float(scale_x)).square() + 1.0e-8)
            coord_error = coord_error_px / 16.0
        else:
            coord_error = torch.sqrt(dy.square() + dx.square() + 1.0e-8)
        coord_loss = (coord_error * visible_mask_f).sum() / visible_mask_f.sum().clamp_min(1.0)
        losses["coordinate"] = coord_loss * self.weights.coordinate

        global_focus = None
        local_centers = None if info is None else info.get("local_centers", None)
        feature_height = None if info is None else info.get("feature_height", None)
        feature_width = None if info is None else info.get("feature_width", None)
        local_radius = None if info is None else info.get("local_radius", None)
        if isinstance(local_centers, torch.Tensor):
            fy = max(int(feature_height if feature_height is not None else 1) - 1, 1)
            fx = max(int(feature_width if feature_width is not None else 1) - 1, 1)
            local_dy = (gt_tracks[..., 0] - local_centers[..., 0]) * float(fy)
            local_dx = (gt_tracks[..., 1] - local_centers[..., 1]) * float(fx)
            radius = float(local_radius if local_radius is not None else 0)
            global_focus = (local_dy.abs() > radius) | (local_dx.abs() > radius)
            if focus_mask is not None:
                global_focus = global_focus | focus_mask.bool()
            if isinstance(active_mask, torch.Tensor):
                global_focus = global_focus & active_mask.bool()
            global_focus = global_focus & visible_mask

        if self.weights.local_heatmap > 0 and info is not None:
            local_heatmap = info.get("local_heatmap", None)
            if isinstance(local_heatmap, torch.Tensor) and isinstance(local_centers, torch.Tensor):
                kernel = local_heatmap.shape[-1]
                radius = int(local_radius if local_radius is not None else (kernel - 1) // 2)
                heatmap_prob = local_heatmap.reshape(*local_heatmap.shape[:3], kernel * kernel).clamp_min(1.0e-8)
                fy = max(int(feature_height if feature_height is not None else 1) - 1, 1)
                fx = max(int(feature_width if feature_width is not None else 1) - 1, 1)
                dy = (gt_tracks[..., 0] - local_centers[..., 0]) * float(fy)
                dx = (gt_tracks[..., 1] - local_centers[..., 1]) * float(fx)
                target_y = dy.round().long() + radius
                target_x = dx.round().long() + radius
                valid_local = visible_mask & (dy.abs() <= radius + 0.5) & (dx.abs() <= radius + 0.5)
                target_index = (target_y.clamp(0, kernel - 1) * kernel + target_x.clamp(0, kernel - 1)).unsqueeze(-1)
                target_prob = heatmap_prob.gather(-1, target_index).squeeze(-1)
                valid_local_f = valid_local.to(pred_tracks.dtype)
                heatmap_loss = -(target_prob.log() * valid_local_f).sum() / valid_local_f.sum().clamp_min(1.0)
                losses["local_heatmap"] = heatmap_loss * self.weights.local_heatmap

        if self.weights.global_heatmap > 0 and info is not None:
            global_heatmap = info.get("global_heatmap", None)
            if isinstance(global_heatmap, torch.Tensor) and isinstance(local_centers, torch.Tensor):
                heatmap_prob = global_heatmap.reshape(*global_heatmap.shape[:3], -1).clamp_min(1.0e-8)
                fy = max(int(feature_height if feature_height is not None else 1) - 1, 1)
                fx = max(int(feature_width if feature_width is not None else 1) - 1, 1)
                target_y = (gt_tracks[..., 0] * float(fy)).round().long().clamp(0, fy)
                target_x = (gt_tracks[..., 1] * float(fx)).round().long().clamp(0, fx)
                target_index = (target_y * (fx + 1) + target_x).unsqueeze(-1)
                target_prob = heatmap_prob.gather(-1, target_index).squeeze(-1)
                global_focus_f = visible_mask_f if global_focus is None else global_focus.to(pred_tracks.dtype)
                global_loss = -(target_prob.log() * global_focus_f).sum() / global_focus_f.sum().clamp_min(1.0)
                losses["global_heatmap"] = global_loss * self.weights.global_heatmap

        if self.weights.global_coordinate > 0 and info is not None:
            global_points = info.get("global_points", info.get("global_expected_points", None))
            if isinstance(global_points, torch.Tensor):
                global_dy = global_points[..., 0] - gt_tracks[..., 0]
                global_dx = global_points[..., 1] - gt_tracks[..., 1]
                if input_height is not None and input_width is not None:
                    global_coord_error_px = torch.sqrt(
                        (global_dy * float(scale_y)).square() + (global_dx * float(scale_x)).square() + 1.0e-8
                    )
                    global_coord_error = global_coord_error_px / 16.0
                else:
                    global_coord_error = torch.sqrt(global_dy.square() + global_dx.square() + 1.0e-8)
                global_focus_f = visible_mask_f if global_focus is None else global_focus.to(pred_tracks.dtype)
                global_coord_loss = (global_coord_error * global_focus_f).sum() / global_focus_f.sum().clamp_min(1.0)
                losses["global_coordinate"] = global_coord_loss * self.weights.global_coordinate

        if self.weights.selector > 0 and info is not None:
            candidate_logits = info.get("candidate_logits", None)
            candidate_points = info.get("candidate_points", None)
            candidate_gate_probability = info.get("candidate_gate_probability", None)
            candidate_global_rank_logits = info.get("candidate_global_rank_logits", None)
            two_stage_mode = candidate_routing_mode in {
                "two_stage",
                "gated_rank",
                "gate_rank",
                "gate_then_rank",
            }
            flat_mode = candidate_routing_mode in {
                "flat",
                "joint",
                "joint_score",
                "single_stage",
            }
            rank_shape_valid = (
                isinstance(candidate_global_rank_logits, torch.Tensor)
                and isinstance(candidate_points, torch.Tensor)
                and candidate_points.ndim >= 3
                and candidate_global_rank_logits.ndim == candidate_points.ndim - 1
                and candidate_global_rank_logits.shape[:-1] == candidate_points.shape[:-2]
                and candidate_global_rank_logits.shape[-1] == candidate_points.shape[-2] - 1
                and candidate_global_rank_logits.shape[-1] > 0
            )
            gate_shape_valid = (
                isinstance(candidate_gate_probability, torch.Tensor)
                and isinstance(candidate_points, torch.Tensor)
                and candidate_gate_probability.shape == candidate_points.shape[:-2]
            )
            flat_logits_shape_valid = (
                isinstance(candidate_logits, torch.Tensor)
                and isinstance(candidate_points, torch.Tensor)
                and candidate_logits.shape == candidate_points.shape[:-1]
            )
            known_routing_modes = {
                "",
                "flat",
                "joint",
                "joint_score",
                "single_stage",
                "two_stage",
                "gated_rank",
                "gate_rank",
                "gate_then_rank",
            }
            if candidate_routing_mode not in known_routing_modes:
                raise ValueError(
                    f"Unsupported candidate routing mode in loss info: {candidate_routing_mode!r}."
                )
            if two_stage_mode and not (gate_shape_valid and rank_shape_valid):
                gate_shape = (
                    None
                    if not isinstance(candidate_gate_probability, torch.Tensor)
                    else tuple(candidate_gate_probability.shape)
                )
                rank_shape = (
                    None
                    if not isinstance(candidate_global_rank_logits, torch.Tensor)
                    else tuple(candidate_global_rank_logits.shape)
                )
                points_shape = (
                    None
                    if not isinstance(candidate_points, torch.Tensor)
                    else tuple(candidate_points.shape)
                )
                raise ValueError(
                    "two-stage candidate routing requires gate shape candidate_points.shape[:-2] "
                    "and rank-logit shape candidate_points.shape[:-2] + "
                    "(num_global_candidates,); "
                    f"got gate={gate_shape}, rank={rank_shape}, points={points_shape}."
                )
            if flat_mode and isinstance(candidate_points, torch.Tensor) and not flat_logits_shape_valid:
                logits_shape = (
                    None
                    if not isinstance(candidate_logits, torch.Tensor)
                    else tuple(candidate_logits.shape)
                )
                raise ValueError(
                    "flat candidate routing requires "
                    "candidate_logits.shape == candidate_points.shape[:-1]; "
                    f"got logits={logits_shape}, points={tuple(candidate_points.shape)}."
                )
            inferred_two_stage_mode = (
                not candidate_routing_mode
                and gate_shape_valid
                and rank_shape_valid
            )
            use_two_stage_loss = two_stage_mode or inferred_two_stage_mode
            if (
                use_two_stage_loss
                and gate_shape_valid
                and rank_shape_valid
                and isinstance(candidate_points, torch.Tensor)
                and candidate_points.shape[-2] > 1
            ):
                candidate_err = torch.norm(candidate_points - gt_tracks.unsqueeze(-2), dim=-1)
                local_err = candidate_err[..., 0]
                best_global_err, best_global_idx = candidate_err[..., 1:].min(dim=-1)
                use_global = best_global_err + selector_margin < local_err
                selector_valid = visible_mask
                if isinstance(active_mask, torch.Tensor):
                    selector_valid = selector_valid & active_mask.bool()
                selector_weight = selector_valid.to(pred_tracks.dtype)
                if isinstance(global_focus, torch.Tensor):
                    selector_weight = selector_weight * (1.0 + global_focus.to(pred_tracks.dtype))

                gate_target = use_global.to(pred_tracks.dtype)
                gate_prob = candidate_gate_probability.clamp(1.0e-4, 1.0 - 1.0e-4)
                gate_bce = F.binary_cross_entropy(
                    gate_prob,
                    gate_target,
                    reduction="none",
                )
                if candidate_gate_focal_gamma > 0:
                    gate_pt = torch.where(gate_target > 0.5, gate_prob, 1.0 - gate_prob)
                    gate_bce = gate_bce * (1.0 - gate_pt).pow(candidate_gate_focal_gamma)
                positive_mask = selector_valid & use_global
                negative_mask = selector_valid & (~use_global)
                gate_pos = gate_bce * positive_mask.to(pred_tracks.dtype)
                gate_neg = gate_bce * negative_mask.to(pred_tracks.dtype)
                gate_pos_loss = gate_pos.sum() / positive_mask.to(pred_tracks.dtype).sum().clamp_min(1.0)
                gate_neg_loss = gate_neg.sum() / negative_mask.to(pred_tracks.dtype).sum().clamp_min(1.0)
                if positive_mask.any() and negative_mask.any():
                    gate_loss = (
                        candidate_gate_positive_weight * gate_pos_loss + gate_neg_loss
                    ) / (candidate_gate_positive_weight + 1.0)
                elif positive_mask.any():
                    gate_loss = gate_pos_loss
                else:
                    gate_loss = gate_neg_loss
                if candidate_gate_margin_weight > 0:
                    candidate_gate_logit = info.get("candidate_gate_logit", None)
                    if isinstance(candidate_gate_logit, torch.Tensor):
                        gate_sign = gate_target.mul(2.0).sub(1.0)
                        gate_margin_term = F.relu(
                            candidate_gate_margin - gate_sign * candidate_gate_logit
                        )
                        gate_margin_term = (
                            gate_margin_term * selector_weight
                        ).sum() / selector_weight.sum().clamp_min(1.0)
                        gate_loss = gate_loss + candidate_gate_margin_weight * gate_margin_term
                losses["selector_gate_raw"] = gate_loss.detach()

                rank_valid = selector_valid & use_global
                if rank_valid.any():
                    rank_log_prob = F.log_softmax(candidate_global_rank_logits, dim=-1)
                    rank_weight = rank_valid.to(pred_tracks.dtype)
                    if candidate_rank_soft_temperature > 0:
                        rank_target = torch.softmax(
                            -candidate_err[..., 1:] / candidate_rank_soft_temperature,
                            dim=-1,
                        )
                        rank_loss_term = -(rank_target * rank_log_prob).sum(dim=-1)
                    else:
                        rank_target_log_prob = rank_log_prob.gather(-1, best_global_idx.unsqueeze(-1)).squeeze(-1)
                        rank_loss_term = -rank_target_log_prob
                    rank_loss = (rank_loss_term * rank_weight).sum() / rank_weight.sum().clamp_min(1.0)
                    losses["selector_rank_raw"] = rank_loss.detach()
                    selector_loss = 0.5 * (gate_loss + rank_loss)
                else:
                    losses["selector_rank_raw"] = gate_loss.detach().new_zeros(())
                    selector_loss = gate_loss
                losses["selector"] = selector_loss * self.weights.selector
            elif flat_logits_shape_valid and isinstance(candidate_points, torch.Tensor):
                candidate_err = torch.norm(candidate_points - gt_tracks.unsqueeze(-2), dim=-1)
                local_err = candidate_err[..., 0]
                if candidate_err.shape[-1] > 1:
                    best_global_err, best_global_idx = candidate_err[..., 1:].min(dim=-1)
                    use_global = best_global_err + selector_margin < local_err
                    selector_target_idx = torch.where(
                        use_global,
                        best_global_idx + 1,
                        torch.zeros_like(best_global_idx),
                    )
                else:
                    selector_target_idx = torch.zeros_like(local_err, dtype=torch.long)

                selector_valid = visible_mask
                if isinstance(active_mask, torch.Tensor):
                    selector_valid = selector_valid & active_mask.bool()
                selector_weight = selector_valid.to(pred_tracks.dtype)
                if isinstance(global_focus, torch.Tensor):
                    selector_weight = selector_weight * (1.0 + global_focus.to(pred_tracks.dtype))

                log_prob = F.log_softmax(candidate_logits, dim=-1)
                target_log_prob = log_prob.gather(-1, selector_target_idx.unsqueeze(-1)).squeeze(-1)
                selector_loss = -(target_log_prob * selector_weight).sum() / selector_weight.sum().clamp_min(1.0)
                losses["selector"] = selector_loss * self.weights.selector
            else:
                selector_probability = info.get("selector_probability", None)
                local_points = info.get("local_points", None)
                global_points = info.get("global_points", None)
                if (
                    isinstance(selector_probability, torch.Tensor)
                    and isinstance(local_points, torch.Tensor)
                    and isinstance(global_points, torch.Tensor)
                ):
                    local_err = torch.norm(local_points - gt_tracks, dim=-1)
                    global_err = torch.norm(global_points - gt_tracks, dim=-1)
                    selector_target = (global_err + selector_margin < local_err).to(pred_tracks.dtype)
                    selector_valid = global_focus if isinstance(global_focus, torch.Tensor) else visible_mask
                    selector_valid_f = selector_valid.to(pred_tracks.dtype)
                    selector_bce = F.binary_cross_entropy(
                        selector_probability.clamp(1.0e-4, 1.0 - 1.0e-4),
                        selector_target,
                        reduction="none",
                    )
                    selector_loss = (selector_bce * selector_valid_f).sum() / selector_valid_f.sum().clamp_min(1.0)
                    losses["selector"] = selector_loss * self.weights.selector

        if self.weights.commit > 0 and info is not None:
            commit_probability = info.get("commit_probability", None)
            local_points = info.get("local_points", None)
            global_points = info.get("global_points", None)
            if (
                isinstance(commit_probability, torch.Tensor)
                and isinstance(local_points, torch.Tensor)
                and isinstance(global_points, torch.Tensor)
            ):
                local_err = torch.norm(local_points - gt_tracks, dim=-1)
                global_err = torch.norm(global_points - gt_tracks, dim=-1)
                commit_target = (global_err + commit_margin < local_err).to(pred_tracks.dtype)
                commit_valid = global_focus if isinstance(global_focus, torch.Tensor) else visible_mask
                commit_valid_f = commit_valid.to(pred_tracks.dtype)
                commit_bce = F.binary_cross_entropy(
                    commit_probability.clamp(1.0e-4, 1.0 - 1.0e-4),
                    commit_target,
                    reduction="none",
                )
                commit_loss = (commit_bce * commit_valid_f).sum() / commit_valid_f.sum().clamp_min(1.0)
                losses["commit"] = commit_loss * self.weights.commit

        vis_loss = F.binary_cross_entropy(
            pred_visibility.clamp(1.0e-4, 1.0 - 1.0e-4),
            gt_visibility.to(pred_visibility.dtype),
        )
        losses["visibility"] = vis_loss * self.weights.visibility

        if self.weights.temporal_smoothness > 0 and pred_tracks.shape[2] > 1:
            delta = pred_tracks[:, :, 1:, :] - pred_tracks[:, :, :-1, :]
            losses["temporal_smoothness"] = delta.abs().mean() * self.weights.temporal_smoothness

        if self.weights.no_harm_prior > 0 and prior_tracks is not None:
            prior_dy = prior_tracks[..., 0] - gt_tracks[..., 0]
            prior_dx = prior_tracks[..., 1] - gt_tracks[..., 1]
            if input_height is not None and input_width is not None:
                prior_error_px = torch.sqrt(
                    (prior_dy * float(scale_y)).square() + (prior_dx * float(scale_x)).square() + 1.0e-8
                )
                prior_error = prior_error_px / 16.0
            else:
                prior_error = torch.sqrt(prior_dy.square() + prior_dx.square() + 1.0e-8)
            local_reference = None if info is None else info.get("local_points", None)
            if isinstance(local_reference, torch.Tensor):
                local_ref_dy = local_reference[..., 0] - gt_tracks[..., 0]
                local_ref_dx = local_reference[..., 1] - gt_tracks[..., 1]
                if input_height is not None and input_width is not None:
                    local_ref_error_px = torch.sqrt(
                        (local_ref_dy * float(scale_y)).square() + (local_ref_dx * float(scale_x)).square() + 1.0e-8
                    )
                    local_ref_error = local_ref_error_px / 16.0
                else:
                    local_ref_error = torch.sqrt(local_ref_dy.square() + local_ref_dx.square() + 1.0e-8)
                prior_error = torch.minimum(prior_error, local_ref_error)
            pred_error = coord_error
            no_harm = F.relu(pred_error - prior_error) * (gt_visibility > 0.5).to(pred_error.dtype)
            losses["no_harm_prior"] = no_harm.sum() / (gt_visibility > 0.5).to(pred_error.dtype).sum().clamp_min(1.0)
            losses["no_harm_prior"] = losses["no_harm_prior"] * self.weights.no_harm_prior

        if self.weights.posterior_consistency > 0 and info is not None:
            confidence = info.get("confidence", None)
            if isinstance(confidence, torch.Tensor):
                target = torch.exp(-coord_error.detach()).clamp(0.0, 1.0)
                losses["posterior_consistency"] = F.l1_loss(confidence, target) * self.weights.posterior_consistency

        if self.weights.long_occ_focus > 0 and focus_mask is not None:
            long_occ_mask = focus_mask.to(coord_error.dtype)
            long_occ_loss = (coord_error * long_occ_mask).sum() / long_occ_mask.sum().clamp_min(1.0)
            losses["long_occ_focus"] = long_occ_loss * self.weights.long_occ_focus

        total_terms = [value for key, value in losses.items() if not key.endswith("_raw")]
        losses["total"] = torch.stack(total_terms).sum()
        return losses
