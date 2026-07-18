"""Training and event-level evaluation for Route-D safe re-detection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .routeD_redetection_metrics import segment_jaccard
from .routeD_safe_redetection import (
    AppearanceMemoryState,
    ConfirmationState,
    RedetectionStepState,
    SafeRedetectionConfig,
    SafeRedetectionLossConfig,
    SafeRedetectionModel,
    initialize_appearance_memory,
    initialize_confirmation_state,
    safe_redetection_dense_proposal_loss,
    safe_redetection_frame_loss,
    sample_batched_map,
)
from .routeD_safe_redetection_cache import load_complete_event_index, verify_event_cache


@dataclass(frozen=True)
class SafeRedetectionTrainingConfig:
    dense_proposal_weight: float = 1.0
    local_decision_weight: float = 1.0
    gradient_clip_norm: float = 1.0
    bootstrap_samples: int = 5000


def _detach_memory(state: AppearanceMemoryState) -> AppearanceMemoryState:
    return AppearanceMemoryState(
        anchors=state.anchors.detach(),
        valid=state.valid.detach(),
        reliability=state.reliability.detach(),
        frame_index=state.frame_index.detach(),
        next_episodic_slot=state.next_episodic_slot.detach(),
    )


def _detach_confirmation(state: ConfirmationState) -> ConfirmationState:
    return ConfirmationState(
        pending_coord_xy_px=state.pending_coord_xy_px.detach(),
        pending_candidate_index=state.pending_candidate_index.detach(),
        pending_count=state.pending_count.detach(),
        pending_valid=state.pending_valid.detach(),
    )


def detach_step_state(state: RedetectionStepState) -> RedetectionStepState:
    return RedetectionStepState(
        memory=_detach_memory(state.memory),
        confirmation=_detach_confirmation(state.confirmation),
    )


def clone_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _initialize_event_state(
    model: SafeRedetectionModel,
    tensors: Mapping[str, torch.Tensor],
    *,
    device: str,
) -> tuple[RedetectionStepState, int]:
    frame_rel = tensors["frame_indices_rel"].long()
    query_rel = int(tensors["query_frame_rel"].item())
    matches = torch.where(frame_rel == query_rel)[0]
    if matches.numel() != 1:
        raise ValueError("query frame is not represented exactly once in sparse cache")
    query_row = int(matches[0].item())
    fmap = tensors["feature_maps_f16"][query_row : query_row + 1].to(
        device=device, dtype=torch.float32
    )
    coord = tensors["query_coord_xy_px"].to(device=device, dtype=torch.float32).view(1, 1, 2)
    query_feature = sample_batched_map(fmap, coord, model.config)[:, 0]
    memory = initialize_appearance_memory(
        query_feature, torch.tensor([query_rel], device=device), model.config
    )
    confirmation = initialize_confirmation_state(
        1, device=device, dtype=torch.float32
    )
    return RedetectionStepState(memory, confirmation), query_row


def train_safe_redetection_epoch(
    model: SafeRedetectionModel,
    index_path: str | Path,
    optimizer: torch.optim.Optimizer,
    loss_config: SafeRedetectionLossConfig,
    training_config: SafeRedetectionTrainingConfig,
    *,
    device: str,
    generator: torch.Generator,
    max_events: int = 0,
) -> dict[str, float]:
    index = load_complete_event_index(index_path, expected_role="fit")
    order = torch.randperm(len(index["events"]), generator=generator).tolist()
    if max_events > 0:
        order = order[:max_events]
    model.train()
    totals: dict[str, float] = {}
    supervised_frames = 0
    for event_id in order:
        row = index["events"][event_id]
        payload = verify_event_cache(
            row["sidecar"],
            expected_protocol_sha256=index["protocol_sha256"],
            expected_role="fit",
        )
        tensors = payload["tensors"]
        state, _ = _initialize_event_state(model, tensors, device=device)
        frame_abs = tensors["frame_indices_abs"].long()
        frame_rel = tensors["frame_indices_rel"].long()
        reentry_abs = int(payload["provenance"]["event"]["reappearance_frame"])
        query_rel = int(tensors["query_frame_rel"].item())
        optimizer.zero_grad(set_to_none=True)
        frame_losses: list[dict[str, torch.Tensor]] = []
        previous_rel: int | None = None
        for index_row in range(frame_rel.numel()):
            current_rel = int(frame_rel[index_row].item())
            if current_rel < query_rel:
                continue
            if previous_rel is not None and current_rel != previous_rel + 1:
                state = RedetectionStepState(
                    state.memory,
                    initialize_confirmation_state(1, device=device),
                )
            fmap = tensors["feature_maps_f16"][index_row : index_row + 1].to(
                device=device, dtype=torch.float32
            )
            native = tensors["native_coords_xy_px"][index_row : index_row + 1].to(
                device=device, dtype=torch.float32
            )
            native_vis = tensors["native_visibility_probability"][index_row : index_row + 1].to(
                device=device, dtype=torch.float32
            )
            native_conf = tensors["native_confidence_probability"][index_row : index_row + 1].to(
                device=device, dtype=torch.float32
            )
            age = tensors["occlusion_age"][index_row : index_row + 1].to(device)
            output, state = model.step(
                frame_index=current_rel,
                feature_map=fmap,
                native_coord_xy_px=native,
                native_visibility_probability=native_vis,
                native_confidence_probability=native_conf,
                occlusion_age=age,
                state=state,
            )
            gt_coord = tensors["gt_coords_xy_px"][index_row : index_row + 1].to(
                device=device, dtype=torch.float32
            )
            gt_visible = tensors["gt_visible"][index_row : index_row + 1].to(device)
            current_abs = int(frame_abs[index_row].item())
            recovery_target = gt_visible & (
                current_abs >= reentry_abs
            ) & (
                current_abs < reentry_abs + model.config.confirmation_frames
            )
            supervised = torch.ones(1, dtype=torch.bool, device=device)
            dense = safe_redetection_dense_proposal_loss(
                output["proposal_score"],
                output["visibility_evidence_map"],
                gt_coord,
                gt_visible,
                supervised,
                model.config,
            )
            local = safe_redetection_frame_loss(
                output,
                output["candidate_coords_xy_px"].detach(),
                output["candidate_valid_mask"],
                gt_coord,
                gt_visible,
                recovery_target,
                supervised,
                model.config,
                loss_config,
            )
            total = (
                training_config.dense_proposal_weight * dense["loss"]
                + training_config.local_decision_weight * local["loss"]
            )
            if not torch.isfinite(total):
                raise FloatingPointError("non-finite safe redetection loss")
            frame_losses.append(
                {
                    "loss": total,
                    **{f"dense_{key}": value for key, value in dense.items()},
                    **{f"local_{key}": value for key, value in local.items()},
                }
            )
            state = detach_step_state(state)
            previous_rel = current_rel
        if not frame_losses:
            raise RuntimeError("event produced no supervised frames")
        event_loss = torch.stack([row["loss"] for row in frame_losses]).mean()
        event_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip_norm)
        optimizer.step()
        for values in frame_losses:
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().item())
        supervised_frames += len(frame_losses)
    return {
        key: value / max(supervised_frames, 1) for key, value in totals.items()
    } | {
        "events": float(len(order)),
        "supervised_frames": float(supervised_frames),
    }


def _aggregate_event_jaccard(
    rows: Sequence[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    thresholds = (1, 2, 4, 8, 16)
    minimums = (1, 4, 16, 64, 256)
    result: dict[str, Any] = {}
    duration_means = []
    for minimum in minimums:
        threshold_means = []
        eligible = [row for row in rows if row["invisibility_duration"] >= minimum]
        for distance in thresholds:
            values = [row[key][str(distance)] for row in eligible]
            values = [value for value in values if not np.isnan(value)]
            mean = float(np.mean(values)) if values else float("nan")
            result[f"AJ_RD_D{distance}_dmin{minimum}"] = mean
            if not np.isnan(mean):
                threshold_means.append(mean)
        aggregate = float(np.mean(threshold_means)) if threshold_means else float("nan")
        result[f"AJ_RD_dmin{minimum}"] = aggregate
        result[f"events_dmin{minimum}"] = len(eligible)
        if not np.isnan(aggregate):
            duration_means.append(aggregate)
    result["AJ_RD"] = float(np.mean(duration_means)) if duration_means else float("nan")
    return result


def paired_bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan"), "samples": samples, "seed": seed}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "samples": int(samples),
        "seed": int(seed),
    }


def predict_event(
    model: SafeRedetectionModel,
    payload: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    tensors = payload["tensors"]
    state, _ = _initialize_event_state(model, tensors, device=device)
    frame_abs = tensors["frame_indices_abs"].long()
    frame_rel = tensors["frame_indices_rel"].long()
    query_rel = int(tensors["query_frame_rel"].item())
    reentry_abs = int(payload["provenance"]["event"]["reappearance_frame"])
    selected_coord = []
    selected_visible = []
    selected_index = []
    candidates = []
    candidate_valid = []
    previous_rel: int | None = None
    model.eval()
    with torch.no_grad():
        for index_row in range(frame_rel.numel()):
            current_rel = int(frame_rel[index_row].item())
            if current_rel < query_rel:
                selected_coord.append(tensors["native_coords_xy_px"][index_row])
                selected_visible.append(tensors["native_visibility"][index_row])
                selected_index.append(torch.tensor(0))
                candidates.append(tensors["native_coords_xy_px"][index_row].view(1, 2))
                candidate_valid.append(torch.ones(1, dtype=torch.bool))
                continue
            if previous_rel is not None and current_rel != previous_rel + 1:
                state = RedetectionStepState(
                    state.memory,
                    initialize_confirmation_state(1, device=device),
                )
            output, state = model.step(
                frame_index=current_rel,
                feature_map=tensors["feature_maps_f16"][index_row : index_row + 1].to(device=device, dtype=torch.float32),
                native_coord_xy_px=tensors["native_coords_xy_px"][index_row : index_row + 1].to(device=device, dtype=torch.float32),
                native_visibility_probability=tensors["native_visibility_probability"][index_row : index_row + 1].to(device=device, dtype=torch.float32),
                native_confidence_probability=tensors["native_confidence_probability"][index_row : index_row + 1].to(device=device, dtype=torch.float32),
                occlusion_age=tensors["occlusion_age"][index_row : index_row + 1].to(device),
                state=state,
            )
            index_value = output["selected_candidate_index"][0]
            coord = output["selected_coord_xy_px"][0]
            visible = torch.where(
                index_value == 0,
                tensors["native_visibility"][index_row].to(device),
                output["selected_visibility_probability"][0] >= 0.5,
            )
            selected_coord.append(coord.cpu())
            selected_visible.append(visible.cpu())
            selected_index.append(index_value.cpu())
            candidates.append(output["candidate_coords_xy_px"][0].cpu())
            candidate_valid.append(output["candidate_valid_mask"][0].cpu())
            state = detach_step_state(state)
            previous_rel = current_rel
    selected_coord_t = torch.stack(selected_coord)
    selected_visible_t = torch.stack(selected_visible).bool()
    selected_index_t = torch.stack(selected_index).long()
    max_candidates = max(row.shape[0] for row in candidates)
    candidate_tensor = torch.zeros(len(candidates), max_candidates, 2)
    valid_tensor = torch.zeros(len(candidates), max_candidates, dtype=torch.bool)
    for row_id, (coord, valid) in enumerate(zip(candidates, candidate_valid)):
        candidate_tensor[row_id, : coord.shape[0]] = coord
        valid_tensor[row_id, : valid.shape[0]] = valid
    gt = tensors["gt_coords_xy_px"].float()
    error = torch.linalg.norm(candidate_tensor - gt[:, None], dim=-1).masked_fill(~valid_tensor, float("inf"))
    oracle_index = error.argmin(dim=-1)
    oracle_coord = candidate_tensor.gather(1, oracle_index[:, None, None].expand(-1, 1, 2)).squeeze(1)
    post = (frame_abs >= reentry_abs) & (frame_abs < reentry_abs + 16)
    if int(post.sum().item()) < 2:
        raise RuntimeError("event cache lacks consecutive post-reappearance rows")
    native = tensors["native_coords_xy_px"].float()
    native_visible = tensors["native_visibility"].bool()
    gt_visible = tensors["gt_visible"].bool()
    jaccards = {"native": {}, "selected": {}, "oracle": {}}
    for distance in (1, 2, 4, 8, 16):
        jaccards["native"][str(distance)] = float(
            segment_jaccard(native[post], native_visible[post], gt[post], gt_visible[post], distance).item()
        )
        jaccards["selected"][str(distance)] = float(
            segment_jaccard(selected_coord_t[post], selected_visible_t[post], gt[post], gt_visible[post], distance).item()
        )
        jaccards["oracle"][str(distance)] = float(
            segment_jaccard(oracle_coord[post], gt_visible[post], gt[post], gt_visible[post], distance).item()
        )
    visible_post = post & gt_visible
    native_error = torch.linalg.norm(native - gt, dim=-1)
    selected_error = torch.linalg.norm(selected_coord_t - gt, dim=-1)
    intervention = selected_index_t > 0
    harmful = intervention & visible_post & (selected_error > native_error + 1e-6)
    false_reacquisition = selected_visible_t & ~gt_visible
    return {
        "event_identity_sha256": payload["provenance"]["event"]["event_identity_sha256"],
        "scene": payload["provenance"]["event"]["scene"],
        "invisibility_duration": int(payload["provenance"]["event"]["invisibility_duration"]),
        "duration_bucket": payload["provenance"]["event"]["duration_bucket"],
        "jaccard": jaccards,
        "native_event_AJ": float(np.mean(list(jaccards["native"].values()))),
        "selected_event_AJ": float(np.mean(list(jaccards["selected"].values()))),
        "oracle_event_AJ": float(np.mean(list(jaccards["oracle"].values()))),
        "selected_gain": float(np.mean(list(jaccards["selected"].values())) - np.mean(list(jaccards["native"].values()))),
        "oracle_gain": float(np.mean(list(jaccards["oracle"].values())) - np.mean(list(jaccards["native"].values()))),
        "intervention_rows": int(intervention.sum().item()),
        "visible_intervention_rows": int((intervention & visible_post).sum().item()),
        "harmful_rows": int(harmful.sum().item()),
        "false_reacquisition_rows": int(false_reacquisition.sum().item()),
        "invisible_rows": int((~gt_visible).sum().item()),
        "visible_post_rows": int(visible_post.sum().item()),
        "native_severe_16px_rows": int(((native_error > 16) & visible_post).sum().item()),
        "selected_severe_16px_rows": int(((selected_error > 16) & visible_post).sum().item()),
        "native_visibility_correct_rows": int((native_visible == gt_visible).sum().item()),
        "selected_visibility_correct_rows": int((selected_visible_t == gt_visible).sum().item()),
        "visibility_rows": int(gt_visible.numel()),
        "zero_step_native_coordinate_exact": bool(torch.equal(selected_coord_t, native)),
        "zero_step_native_visibility_exact": bool(torch.equal(selected_visible_t, native_visible)),
    }


def evaluate_safe_redetection_index(
    model: SafeRedetectionModel,
    index_path: str | Path,
    *,
    expected_role: str,
    device: str,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 17018,
    max_events: int = 0,
) -> dict[str, Any]:
    index = load_complete_event_index(index_path, expected_role=expected_role)
    source_rows = index["events"][:max_events] if max_events > 0 else index["events"]
    rows = []
    for source in source_rows:
        payload = verify_event_cache(
            source["sidecar"],
            expected_protocol_sha256=index["protocol_sha256"],
            expected_role=expected_role,
        )
        rows.append(predict_event(model, payload, device=device))
    native = _aggregate_event_jaccard(
        [{**row, "metric": row["jaccard"]["native"]} for row in rows], "metric"
    )
    selected = _aggregate_event_jaccard(
        [{**row, "metric": row["jaccard"]["selected"]} for row in rows], "metric"
    )
    oracle = _aggregate_event_jaccard(
        [{**row, "metric": row["jaccard"]["oracle"]} for row in rows], "metric"
    )
    visible_interventions = sum(row["visible_intervention_rows"] for row in rows)
    harmful = sum(row["harmful_rows"] for row in rows)
    invisible = sum(row["invisible_rows"] for row in rows)
    false_reacquisition = sum(row["false_reacquisition_rows"] for row in rows)
    visible_post = sum(row["visible_post_rows"] for row in rows)
    native_severe = sum(row["native_severe_16px_rows"] for row in rows)
    selected_severe = sum(row["selected_severe_16px_rows"] for row in rows)
    visibility_rows = sum(row["visibility_rows"] for row in rows)
    native_visibility_correct = sum(row["native_visibility_correct_rows"] for row in rows)
    selected_visibility_correct = sum(row["selected_visibility_correct_rows"] for row in rows)
    gains = [row["selected_gain"] for row in rows]
    zero_coordinate_exact = all(row["zero_step_native_coordinate_exact"] for row in rows)
    zero_visibility_exact = all(row["zero_step_native_visibility_exact"] for row in rows)
    return {
        "role": expected_role,
        "events": len(rows),
        "native_AJ_RD": native,
        "selected_AJ_RD": selected,
        "oracle_AJ_RD": oracle,
        "gain": {
            "AJ_RD": selected["AJ_RD"] - native["AJ_RD"],
            "oracle_AJ_RD": oracle["AJ_RD"] - native["AJ_RD"],
        },
        "paired_event_AJ_gain_CI": paired_bootstrap_ci(
            gains, seed=bootstrap_seed, samples=bootstrap_samples
        ),
        "event_standard_metrics": {
            "native_AJ": native["AJ_RD_dmin1"],
            "selected_AJ": selected["AJ_RD_dmin1"],
            "AJ_gain": selected["AJ_RD_dmin1"] - native["AJ_RD_dmin1"],
            "native_visibility_accuracy": native_visibility_correct / max(visibility_rows, 1),
            "selected_visibility_accuracy": selected_visibility_correct / max(visibility_rows, 1),
            "visibility_accuracy_gain": (selected_visibility_correct - native_visibility_correct) / max(visibility_rows, 1),
        },
        "zero_step_native_parity": {
            "coordinate_exact_all": zero_coordinate_exact,
            "visibility_exact_all": zero_visibility_exact,
        },
        "behavior": {
            "visible_intervention_rows": visible_interventions,
            "harmful_rows": harmful,
            "harmful_intervention_rate": harmful / max(visible_interventions, 1),
            "invisible_rows": invisible,
            "false_reacquisition_rows": false_reacquisition,
            "false_reacquisition_rate": false_reacquisition / max(invisible, 1),
        },
        "severe_16px_rate": {
            "native": native_severe / max(visible_post, 1),
            "selected": selected_severe / max(visible_post, 1),
            "delta": (selected_severe - native_severe) / max(visible_post, 1),
        },
        "per_event": rows,
        "cache_index_sha256": index["_index_sha256"],
    }


def safety_feasible(validation: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    return bool(
        validation["behavior"]["harmful_intervention_rate"] <= 0.01
        and validation["behavior"]["false_reacquisition_rate"]
        <= baseline["behavior"]["false_reacquisition_rate"] + 0.002
        and validation["severe_16px_rate"]["delta"] <= 0.0
    )
