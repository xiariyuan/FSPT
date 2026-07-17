"""Joint LMRA + CMCP + local safety training for Route-D P0i."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from datasets.metrics import compute_tapvid_metrics

from .cotracker3_stage0_adapter import tensor_sha256
from .routeD_cmcp_feature_cache import load_complete_feature_index
from .routeD_cmcp_late_metric_adapter import (
    LateMetricResidualAdapter,
    lmra_distortion_loss,
)
from .routeD_cmcp_pairwise_cache import inject_dynamic_summary
from .routeD_cmcp_pairwise_safety import (
    CMCPLocalPairwiseSafetyComparator,
    build_cmcp_local_candidate_tokens,
)
from .routeD_cmcp_pairwise_training import (
    PairwiseSafetyLossConfig,
    StaticTokenNormalization,
    _update_summary,
    normalize_static_tokens,
    pairwise_safety_frame_loss,
)
from .routeD_cmcp_training import (
    CMCPLossConfig,
    CMCPVideoBundle,
    cmcp_proposal_loss,
    load_cmcp_video,
)
from .routeD_kubric_cache import canonical_json_sha256
from .routeD_multi_memory_proposal import (
    CMCPConfig,
    CausalMultiMemoryProposalGenerator,
    build_causal_multi_memory_correlations,
    extract_proposal_candidates,
)
from .routeD_musr_training import (
    _tracks_from_xy,
    _visible_error_stats,
    paired_video_bootstrap_ci,
    visible_post_query_mask,
)


@dataclass(frozen=True)
class LMRAJointLossConfig:
    dense_cmcp_weight: float = 1.0
    local_comparator_weight: float = 1.0
    feature_distortion_weight: float = 0.1
    grad_clip_norm: float = 1.0


def _utility_from_error(error_px: torch.Tensor, comparator) -> torch.Tensor:
    thresholds = torch.tensor(
        comparator.config.thresholds_px,
        device=error_px.device,
        dtype=error_px.dtype,
    )
    weights = torch.tensor(
        comparator.config.utility_weights,
        device=error_px.device,
        dtype=error_px.dtype,
    )
    weights = weights / weights.sum()
    return ((error_px[..., None] <= thresholds) * weights).sum(dim=-1)


def _stack_dense(outputs: list[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "utility_logit": torch.stack([row["utility_logit"] for row in outputs], dim=1),
        "risk_logit": torch.stack([row["risk_logit"] for row in outputs], dim=1),
        "proposal_score": torch.stack([row["proposal_score"] for row in outputs], dim=1),
        "native_logit": torch.stack([row["native_logit"] for row in outputs], dim=1),
    }


def _local_tokens_stop_gradient(
    *,
    dense: Mapping[str, torch.Tensor],
    proposal: Mapping[str, torch.Tensor],
    candidate_valid: torch.Tensor,
    tensors: Mapping[str, torch.Tensor],
    point_index: torch.Tensor,
    frame: int,
    cmcp_config: CMCPConfig,
) -> torch.Tensor:
    batch = int(point_index.numel())
    zero_summary = torch.zeros(batch, 4, device=proposal["candidate_coords_xy_px"].device)
    return build_cmcp_local_candidate_tokens(
        hidden_map=dense["hidden_map"].detach(),
        recurrent_input=dense["recurrent_input"].detach(),
        utility_logit=dense["utility_logit"].detach(),
        risk_logit=dense["risk_logit"].detach(),
        proposal_score=dense["proposal_score"].detach(),
        candidate_coords_xy_px=proposal["candidate_coords_xy_px"].detach(),
        candidate_scores=proposal["candidate_scores"].detach(),
        candidate_valid_mask=candidate_valid,
        native_visibility_probability=tensors["native_visibility_probability"][point_index, frame].to(
            proposal["candidate_coords_xy_px"].device
        ),
        native_confidence_probability=tensors["native_confidence_probability"][point_index, frame].to(
            proposal["candidate_coords_xy_px"].device
        ),
        native_joint_probability=tensors["native_joint_probability"][point_index, frame].to(
            proposal["candidate_coords_xy_px"].device
        ),
        previous_decision_summary=zero_summary,
        input_height=cmcp_config.input_height,
        input_width=cmcp_config.input_width,
    )


def train_lmra_epoch(
    adapter: LateMetricResidualAdapter,
    cmcp: CausalMultiMemoryProposalGenerator,
    comparator: CMCPLocalPairwiseSafetyComparator,
    feature_index_path: str | Path,
    normalization: StaticTokenNormalization,
    cmcp_loss_config: CMCPLossConfig,
    comparator_loss_config: PairwiseSafetyLossConfig,
    joint_loss_config: LMRAJointLossConfig,
    optimizer: torch.optim.Optimizer,
    *,
    device: str,
    point_batch_size: int,
    generator: torch.Generator,
    max_videos: int = 0,
) -> dict[str, float]:
    index = load_complete_feature_index(feature_index_path, expected_partition="fit")
    video_order = torch.randperm(len(index["videos"]), generator=generator).tolist()
    if max_videos > 0:
        video_order = video_order[: int(max_videos)]
    adapter.train()
    cmcp.train()
    comparator.train()
    totals: dict[str, float] = {}
    supervised_rows = 0
    for video_id in video_order:
        bundle = load_cmcp_video(index["videos"][video_id])
        frozen_maps = bundle.feature_maps.to(device=device, dtype=torch.float32)
        tensors = bundle.tensors
        native = tensors["native_coords_xy_px"].float()
        queries = tensors["query_points_tyx"].float()
        gt_xy = tensors["gt_tracks_yx"][..., [1, 0]].float() * 255.0
        visible_mask = visible_post_query_mask(tensors)
        point_order = torch.randperm(native.shape[0], generator=generator)
        point_batches = [
            point_order[start : start + int(point_batch_size)]
            for start in range(0, native.shape[0], int(point_batch_size))
        ]
        video_rows = max(int(visible_mask.sum().item()), 1)
        optimizer.zero_grad(set_to_none=True)
        # The adapter is point-independent. Compute the full adapted feature map
        # once per video, then accumulate deterministic point-batch gradients
        # before a single optimizer step. This preserves the per-visible-row
        # objective while avoiding repeated full-map 1x1 projections.
        adapted_maps = adapter(frozen_maps)
        distortion = lmra_distortion_loss(adapted_maps, frozen_maps)
        for batch_ordinal, point_index in enumerate(point_batches):
            correlation, motion, frame_valid = build_causal_multi_memory_correlations(
                adapted_maps,
                native[point_index].to(device),
                queries[point_index].to(device),
                input_height=cmcp.config.input_height,
                input_width=cmcp.config.input_width,
                ema_alpha=cmcp.config.ema_alpha,
                motion_sigma_cells=cmcp.config.motion_sigma_cells,
                detach_sampled_memories=True,
            )
            state = None
            summary = torch.zeros(len(point_index), 4, device=device)
            dense_rows: list[Mapping[str, torch.Tensor]] = []
            local_losses: list[Mapping[str, torch.Tensor]] = []
            for frame in range(native.shape[1]):
                dense, state = cmcp.step(
                    correlation[:, frame],
                    motion[:, frame],
                    state,
                    frame_valid=frame_valid[:, frame],
                )
                dense_rows.append(dense)
                proposal = extract_proposal_candidates(
                    dense["proposal_score"],
                    native[point_index, frame].to(device),
                    dense["native_logit"],
                    cmcp.config,
                )
                candidate_valid = proposal["candidate_valid_mask"].clone()
                candidate_valid[:, 1:] &= frame_valid[:, frame, None]
                raw_token = _local_tokens_stop_gradient(
                    dense=dense,
                    proposal=proposal,
                    candidate_valid=candidate_valid,
                    tensors=tensors,
                    point_index=point_index,
                    frame=frame,
                    cmcp_config=cmcp.config,
                )
                normalized = normalize_static_tokens(raw_token, candidate_valid, normalization)
                dynamic = inject_dynamic_summary(normalized, summary)
                compared = comparator(
                    dynamic,
                    candidate_valid,
                    proposal["candidate_coords_xy_px"].detach(),
                )
                supervised = (
                    (~tensors["gt_occluded"][point_index, frame].to(device))
                    & (frame > tensors["query_points_tyx"][point_index, 0].round().long().to(device))
                    & frame_valid[:, frame]
                )
                local_losses.append(
                    pairwise_safety_frame_loss(
                        compared,
                        proposal["candidate_coords_xy_px"].detach(),
                        candidate_valid,
                        gt_xy[point_index, frame].to(device),
                        supervised,
                        comparator.config,
                        comparator_loss_config,
                    )
                )
                summary = _update_summary(
                    compared,
                    proposal["candidate_coords_xy_px"].detach(),
                    frame_valid[:, frame],
                    summary,
                )
            dense_sequence = _stack_dense(dense_rows)
            dense_losses = cmcp_proposal_loss(
                dense_sequence,
                native[point_index].to(device),
                gt_xy[point_index].to(device),
                visible_mask[point_index].to(device),
                cmcp.config,
                cmcp_loss_config,
            )
            local = {
                key: torch.stack([row[key] for row in local_losses]).mean()
                for key in local_losses[0]
            }
            total = (
                float(joint_loss_config.dense_cmcp_weight) * dense_losses["loss"]
                + float(joint_loss_config.local_comparator_weight) * local["loss"]
                + float(joint_loss_config.feature_distortion_weight) * distortion
            )
            if not torch.isfinite(total):
                raise FloatingPointError("non-finite LMRA joint loss")
            rows = int(visible_mask[point_index].sum().item())
            row_weight = float(rows) / float(video_rows)
            (total * row_weight).backward(
                retain_graph=batch_ordinal + 1 < len(point_batches)
            )
            supervised_rows += rows
            values = {
                "loss": total,
                "dense_loss": dense_losses["loss"],
                "local_loss": local["loss"],
                "feature_distortion": distortion,
                **{f"dense_{key}": value for key, value in dense_losses.items() if key != "loss"},
                **{f"local_{key}": value for key, value in local.items() if key != "loss"},
            }
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().item()) * rows
        nn.utils.clip_grad_norm_(
            list(adapter.parameters()) + list(cmcp.parameters()) + list(comparator.parameters()),
            float(joint_loss_config.grad_clip_norm),
        )
        optimizer.step()
        del adapted_maps, frozen_maps
    return {key: value / max(supervised_rows, 1) for key, value in totals.items()} | {
        "supervised_rows": float(supervised_rows),
        "videos": float(len(video_order)),
    }


def predict_lmra_video(
    adapter: LateMetricResidualAdapter,
    cmcp: CausalMultiMemoryProposalGenerator,
    comparator: CMCPLocalPairwiseSafetyComparator,
    bundle: CMCPVideoBundle,
    normalization: StaticTokenNormalization,
    *,
    device: str,
    point_batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    tensors = bundle.tensors
    frozen_maps = bundle.feature_maps.to(device=device, dtype=torch.float32)
    native_all = tensors["native_coords_xy_px"].float()
    queries_all = tensors["query_points_tyx"].float()
    selected_all = []
    selected_index_all = []
    candidates_all = []
    valid_all = []
    adapter.eval()
    cmcp.eval()
    comparator.eval()
    with torch.no_grad():
        adapted_maps = adapter(frozen_maps)
        for start in range(0, native_all.shape[0], int(point_batch_size)):
            end = min(native_all.shape[0], start + int(point_batch_size))
            point_index = torch.arange(start, end)
            native = native_all[start:end].to(device)
            correlation, motion, frame_valid = build_causal_multi_memory_correlations(
                adapted_maps,
                native,
                queries_all[start:end].to(device),
                input_height=cmcp.config.input_height,
                input_width=cmcp.config.input_width,
                ema_alpha=cmcp.config.ema_alpha,
                motion_sigma_cells=cmcp.config.motion_sigma_cells,
                detach_sampled_memories=True,
            )
            state = None
            summary = torch.zeros(end - start, 4, device=device)
            selected_frames = []
            index_frames = []
            candidate_frames = []
            valid_frames = []
            for frame in range(native.shape[1]):
                dense, state = cmcp.step(
                    correlation[:, frame], motion[:, frame], state,
                    frame_valid=frame_valid[:, frame],
                )
                proposal = extract_proposal_candidates(
                    dense["proposal_score"], native[:, frame], dense["native_logit"], cmcp.config
                )
                candidate_valid = proposal["candidate_valid_mask"].clone()
                candidate_valid[:, 1:] &= frame_valid[:, frame, None]
                raw_token = _local_tokens_stop_gradient(
                    dense=dense,
                    proposal=proposal,
                    candidate_valid=candidate_valid,
                    tensors=tensors,
                    point_index=point_index,
                    frame=frame,
                    cmcp_config=cmcp.config,
                )
                normalized = normalize_static_tokens(raw_token, candidate_valid, normalization)
                dynamic = inject_dynamic_summary(normalized, summary)
                compared = comparator(dynamic, candidate_valid, proposal["candidate_coords_xy_px"])
                summary = _update_summary(
                    compared, proposal["candidate_coords_xy_px"], frame_valid[:, frame], summary
                )
                selected_frames.append(compared["selected_coord_xy_px"].cpu())
                index_frames.append(compared["selected_candidate_index"].cpu())
                candidate_frames.append(proposal["candidate_coords_xy_px"].cpu())
                valid_frames.append(candidate_valid.cpu())
            selected_all.append(torch.stack(selected_frames, dim=1))
            selected_index_all.append(torch.stack(index_frames, dim=1))
            candidates_all.append(torch.stack(candidate_frames, dim=1))
            valid_all.append(torch.stack(valid_frames, dim=1))
    selected_xy = torch.cat(selected_all)
    selected_index = torch.cat(selected_index_all)
    candidates = torch.cat(candidates_all)
    candidate_valid = torch.cat(valid_all)
    gt_xy = tensors["gt_tracks_yx"][..., [1, 0]].float() * 255.0
    error = torch.linalg.vector_norm(candidates - gt_xy[..., None, :], dim=-1).masked_fill(
        ~candidate_valid, float("inf")
    )
    oracle_index = error.argmin(dim=-1)
    oracle_xy = candidates.gather(2, oracle_index[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
    mask = visible_post_query_mask(tensors)
    native_error = torch.linalg.vector_norm(native_all - gt_xy, dim=-1)
    selected_error = torch.linalg.vector_norm(selected_xy - gt_xy, dim=-1)
    native_utility = _utility_from_error(native_error, comparator)
    selected_utility = _utility_from_error(selected_error, comparator)
    oracle_utility = _utility_from_error(error.min(dim=-1).values, comparator)
    nonnative = selected_index > 0
    harmful = nonnative & (selected_utility + 1e-6 < native_utility)
    available = oracle_utility > native_utility + 1e-6
    beneficial = nonnative & (selected_utility > native_utility + 1e-6)
    behavior = {
        "rows": int(mask.sum().item()),
        "selected_non_native_rate": float(nonnative[mask].float().mean().item()),
        "harmful_non_native_rate": float(harmful[mask].float().mean().item()),
        "beneficial_candidate_available_rate": float(available[mask].float().mean().item()),
        "beneficial_candidate_recall": float(
            beneficial[mask].sum().item() / max(int(available[mask].sum().item()), 1)
        ),
    }
    return {
        "selected_coords_xy_px": selected_xy,
        "selected_candidate_index": selected_index,
        "candidate_coords_xy_px": candidates,
        "candidate_valid_mask": candidate_valid,
        "oracle_coords_xy_px": oracle_xy,
        "oracle_candidate_index": oracle_index,
        "adapted_feature_sha256": tensor_sha256(adapter(frozen_maps).cpu()),
    }, behavior


def evaluate_lmra_index(
    adapter: LateMetricResidualAdapter,
    cmcp: CausalMultiMemoryProposalGenerator,
    comparator: CMCPLocalPairwiseSafetyComparator,
    feature_index_path: str | Path,
    normalization: StaticTokenNormalization,
    *,
    expected_partition: str,
    device: str,
    point_batch_size: int = 4,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 1701,
    max_videos: int = 0,
) -> dict[str, Any]:
    index = load_complete_feature_index(feature_index_path, expected_partition=expected_partition)
    rows = index["videos"][:max_videos] if max_videos > 0 else index["videos"]
    native_tracks = []
    selected_tracks = []
    oracle_tracks = []
    gt_tracks = []
    pred_visibility = []
    gt_visibility = []
    queries = []
    per_video = []
    coordinate_hashes = []
    totals = {"rows": 0, "selected": 0.0, "harmful": 0.0, "available": 0.0, "beneficial": 0.0}
    native_parity_all = True
    for row in rows:
        bundle = load_cmcp_video(row)
        prediction, behavior = predict_lmra_video(
            adapter, cmcp, comparator, bundle, normalization,
            device=device, point_batch_size=point_batch_size,
        )
        tensors = bundle.tensors
        native = tensors["native_coords_xy_px"].float()
        selected = prediction["selected_coords_xy_px"]
        oracle = prediction["oracle_coords_xy_px"]
        candidates = prediction["candidate_coords_xy_px"]
        native_parity = torch.equal(candidates[..., 0, :], native)
        native_parity_all = native_parity_all and native_parity
        coordinate_hash = tensor_sha256(candidates)
        coordinate_hashes.append(coordinate_hash)
        visibility = tensors["native_visibility"]
        gt_vis = ~tensors["gt_occluded"]
        metric_args = (tensors["gt_tracks_yx"], visibility, gt_vis, tensors["query_points_tyx"])
        native_metrics = compute_tapvid_metrics(
            _tracks_from_xy(native, 256), *metric_args, resolution=256, query_mode="first"
        )
        selected_metrics = compute_tapvid_metrics(
            _tracks_from_xy(selected, 256), *metric_args, resolution=256, query_mode="first"
        )
        oracle_metrics = compute_tapvid_metrics(
            _tracks_from_xy(oracle, 256), *metric_args, resolution=256, query_mode="first"
        )
        native_error = _visible_error_stats(native, tensors, raster=256)
        selected_error = _visible_error_stats(selected, tensors, raster=256)
        oracle_error = _visible_error_stats(oracle, tensors, raster=256)
        per_video.append({
            "source_index": bundle.source_index,
            "video_name": bundle.video_name,
            "native_AJ": float(native_metrics["AJ"]),
            "selected_AJ": float(selected_metrics["AJ"]),
            "oracle_AJ": float(oracle_metrics["AJ"]),
            "selected_AJ_gain_points": 100.0 * (float(selected_metrics["AJ"]) - float(native_metrics["AJ"])),
            "oracle_AJ_gain_points": 100.0 * (float(oracle_metrics["AJ"]) - float(native_metrics["AJ"])),
            "native_delta_average": float(native_metrics["<avg"]),
            "selected_delta_average": float(selected_metrics["<avg"]),
            "oracle_delta_average": float(oracle_metrics["<avg"]),
            "selected_delta_gain_points": 100.0 * (float(selected_metrics["<avg"]) - float(native_metrics["<avg"])),
            "oracle_delta_gain_points": 100.0 * (float(oracle_metrics["<avg"]) - float(native_metrics["<avg"])),
            "native_error": native_error,
            "selected_error": selected_error,
            "oracle_error": oracle_error,
            "behavior": behavior,
            "candidate_coordinate_sha256": coordinate_hash,
            "native_candidate_parity": native_parity,
            "adapted_feature_sha256": prediction["adapted_feature_sha256"],
        })
        count = behavior["rows"]
        totals["rows"] += count
        totals["selected"] += behavior["selected_non_native_rate"] * count
        totals["harmful"] += behavior["harmful_non_native_rate"] * count
        totals["available"] += behavior["beneficial_candidate_available_rate"] * count
        totals["beneficial"] += (
            behavior["beneficial_candidate_recall"]
            * behavior["beneficial_candidate_available_rate"]
            * count
        )
        native_tracks.append(_tracks_from_xy(native, 256))
        selected_tracks.append(_tracks_from_xy(selected, 256))
        oracle_tracks.append(_tracks_from_xy(oracle, 256))
        gt_tracks.append(tensors["gt_tracks_yx"])
        pred_visibility.append(visibility)
        gt_visibility.append(gt_vis)
        queries.append(tensors["query_points_tyx"])
    pooled = (
        torch.cat(gt_tracks), torch.cat(pred_visibility), torch.cat(gt_visibility), torch.cat(queries)
    )
    native_metrics = compute_tapvid_metrics(
        torch.cat(native_tracks), *pooled, resolution=256, query_mode="first"
    )
    selected_metrics = compute_tapvid_metrics(
        torch.cat(selected_tracks), *pooled, resolution=256, query_mode="first"
    )
    oracle_metrics = compute_tapvid_metrics(
        torch.cat(oracle_tracks), *pooled, resolution=256, query_mode="first"
    )
    selected_aj = [row["selected_AJ_gain_points"] for row in per_video]
    oracle_aj = [row["oracle_AJ_gain_points"] for row in per_video]
    selected_delta = [row["selected_delta_gain_points"] for row in per_video]
    weights = [row["native_error"]["rows"] for row in per_video]
    native_severe = float(np.average([row["native_error"]["severe_16px_rate"] for row in per_video], weights=weights))
    selected_severe = float(np.average([row["selected_error"]["severe_16px_rate"] for row in per_video], weights=weights))
    oracle_severe = float(np.average([row["oracle_error"]["severe_16px_rate"] for row in per_video], weights=weights))
    count = max(totals["rows"], 1)
    return {
        "partition": expected_partition,
        "videos": len(per_video),
        "native_metrics": native_metrics,
        "selected_metrics": selected_metrics,
        "oracle_metrics": oracle_metrics,
        "selected_gain_points": {
            "AJ": 100.0 * (float(selected_metrics["AJ"]) - float(native_metrics["AJ"])),
            "delta_average": 100.0 * (float(selected_metrics["<avg"]) - float(native_metrics["<avg"])),
            "OA": 100.0 * (float(selected_metrics["OA"]) - float(native_metrics["OA"])),
        },
        "oracle_gain_points": {
            "AJ": 100.0 * (float(oracle_metrics["AJ"]) - float(native_metrics["AJ"])),
            "delta_average": 100.0 * (float(oracle_metrics["<avg"]) - float(native_metrics["<avg"])),
            "OA": 100.0 * (float(oracle_metrics["OA"]) - float(native_metrics["OA"])),
        },
        "paired_video_selected_AJ_gain_CI": paired_video_bootstrap_ci(
            selected_aj, seed=bootstrap_seed, samples=bootstrap_samples
        ),
        "paired_video_oracle_AJ_gain_CI": paired_video_bootstrap_ci(
            oracle_aj, seed=bootstrap_seed + 1, samples=bootstrap_samples
        ),
        "paired_video_selected_delta_gain_CI": paired_video_bootstrap_ci(
            selected_delta, seed=bootstrap_seed + 2, samples=bootstrap_samples
        ),
        "severe_16px_rate": {
            "native": native_severe,
            "selected": selected_severe,
            "oracle": oracle_severe,
            "selected_delta": selected_severe - native_severe,
            "oracle_delta": oracle_severe - native_severe,
        },
        "behavior": {
            "rows": totals["rows"],
            "selected_non_native_rate": totals["selected"] / count,
            "harmful_non_native_rate": totals["harmful"] / count,
            "beneficial_candidate_available_rate": totals["available"] / count,
            "beneficial_candidate_recall": totals["beneficial"] / max(totals["available"], 1.0),
        },
        "candidate_coordinate_combined_sha256": canonical_json_sha256(coordinate_hashes),
        "native_candidate_parity_all": native_parity_all,
        "per_video": per_video,
        "feature_cache_index_sha256": index["_index_sha256"],
    }


def safety_feasible_lmra(validation: Mapping[str, Any]) -> bool:
    return bool(
        validation["native_candidate_parity_all"]
        and validation["behavior"]["harmful_non_native_rate"] <= 0.01
        and validation["severe_16px_rate"]["selected_delta"] <= 1e-12
    )
