"""Causal video-context features for Route-D probability calibration."""
from __future__ import annotations

import torch


CAUSAL_VIDEO_CONTEXT_NAMES = (
    "frame_mean_action_logit",
    "prefix_mean_action_logit",
    "frame_mean_p1_margin",
    "prefix_mean_p1_margin",
    "frame_mean_coarse_gain",
    "prefix_mean_coarse_gain",
    "frame_mean_total_gain",
    "frame_mean_global_distance_to_local",
    "frame_mean_candidate_entropy",
    "frame_mean_previous_confidence",
    "normalized_frame_id",
    "log_active_rows",
)


def build_causal_video_context(
    *,
    sample_id: torch.Tensor,
    frame_id: torch.Tensor,
    action_logit: torch.Tensor,
    predicted_p1_margin: torch.Tensor,
    predicted_coarse_gain: torch.Tensor,
    predicted_total_gain: torch.Tensor,
    global_distance_to_local: torch.Tensor,
    candidate_entropy: torch.Tensor,
    previous_confidence: torch.Tensor,
) -> torch.Tensor:
    """Build row-aligned context from current and preceding frames only.

    All tensors must be one-dimensional and row-aligned. Rows that share a
    ``(sample_id, frame_id)`` receive the same context. Prefix means include the
    current frame and never inspect a later frame.
    """
    tensors = {
        "sample_id": sample_id,
        "frame_id": frame_id,
        "action_logit": action_logit,
        "predicted_p1_margin": predicted_p1_margin,
        "predicted_coarse_gain": predicted_coarse_gain,
        "predicted_total_gain": predicted_total_gain,
        "global_distance_to_local": global_distance_to_local,
        "candidate_entropy": candidate_entropy,
        "previous_confidence": previous_confidence,
    }
    lengths = {name: int(value.numel()) for name, value in tensors.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"All context tensors must have equal length: {lengths}")
    rows = next(iter(lengths.values()))
    if rows == 0:
        return torch.empty(
            0,
            len(CAUSAL_VIDEO_CONTEXT_NAMES),
            dtype=torch.float32,
        )
    for name, value in tensors.items():
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")

    device = action_logit.device
    dtype = torch.float32
    sample_id = sample_id.to(device=device, dtype=torch.long)
    frame_id = frame_id.to(device=device, dtype=torch.long)
    signals = torch.stack(
        [
            action_logit,
            predicted_p1_margin,
            predicted_coarse_gain,
            predicted_total_gain,
            global_distance_to_local,
            candidate_entropy,
            previous_confidence,
        ],
        dim=-1,
    ).to(device=device, dtype=dtype)
    if not torch.isfinite(signals).all():
        raise ValueError("Causal context signals must be finite")

    output = torch.empty(
        rows,
        len(CAUSAL_VIDEO_CONTEXT_NAMES),
        device=device,
        dtype=dtype,
    )
    for sample in torch.unique(sample_id, sorted=True):
        sample_mask = sample_id == sample
        frames = torch.unique(frame_id[sample_mask], sorted=True)
        cumulative_sum = torch.zeros(signals.shape[-1], device=device, dtype=dtype)
        cumulative_count = 0
        for frame in frames:
            mask = sample_mask & (frame_id == frame)
            frame_values = signals[mask]
            frame_mean = frame_values.mean(dim=0)
            cumulative_sum = cumulative_sum + frame_values.sum(dim=0)
            cumulative_count += int(frame_values.shape[0])
            prefix_mean = cumulative_sum / float(max(cumulative_count, 1))
            normalized_frame = float(frame.item()) / (float(frame.item()) + 10.0)
            active_rows = torch.log1p(
                torch.tensor(float(frame_values.shape[0]), device=device, dtype=dtype)
            )
            context = torch.stack(
                [
                    frame_mean[0],
                    prefix_mean[0],
                    frame_mean[1],
                    prefix_mean[1],
                    frame_mean[2],
                    prefix_mean[2],
                    frame_mean[3],
                    frame_mean[4],
                    frame_mean[5],
                    frame_mean[6],
                    torch.tensor(normalized_frame, device=device, dtype=dtype),
                    active_rows,
                ]
            )
            output[mask] = context
    return output
