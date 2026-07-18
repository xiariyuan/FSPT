"""Official-compatible CoTracker3 online runtime for Route-D safe re-detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .routeD_safe_redetection import (
    RedetectionStepState,
    SafeRedetectionModel,
    initialize_appearance_memory,
    initialize_confirmation_state,
    sample_batched_map,
)


@dataclass(frozen=True)
class RuntimeResult:
    tracks_xy: torch.Tensor  # 1,T,1,2 in input image pixels
    visibility: torch.Tensor  # 1,T,1 bool
    visibility_probability: torch.Tensor  # 1,T,1
    confidence_probability: torch.Tensor  # 1,T,1
    selected_non_native: torch.Tensor  # 1,T,1 bool
    confirmed_recovery: torch.Tensor  # 1,T,1 bool
    writeback_applied: torch.Tensor  # 1,T,1 bool
    diagnostics: dict[str, Any]


def points_on_grid(
    size: int,
    extent: tuple[float, float],
    center: tuple[float, float] | None,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Exact Meta CoTracker grid formula, returning one row of xy points."""
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]
    if center is None:
        center = (extent[0] / 2, extent[1] / 2)
    margin = extent[1] / 64
    range_y = (
        margin - extent[0] / 2 + center[0],
        extent[0] / 2 + center[0] - margin,
    )
    range_x = (
        margin - extent[1] / 2 + center[1],
        extent[1] / 2 + center[1] - margin,
    )
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack((grid_x, grid_y), dim=-1).reshape(1, -1, 2)


def build_official_single_point_queries(
    target_query_txy: torch.Tensor,
    *,
    input_height: int,
    input_width: int,
    interp_height: int,
    interp_width: int,
    local_grid_size: int = 8,
    global_grid_size: int = 5,
    local_extent: int = 50,
) -> torch.Tensor:
    """Target plus official internal support points in input-image coordinates."""
    if target_query_txy.shape != (1, 1, 3):
        raise ValueError("target query must have shape (1,1,3) in [t,x,y]")
    device = target_query_txy.device
    target = target_query_txy.clone()
    target_x_interp = target[0, 0, 1] * (interp_width - 1) / float(max(input_width - 1, 1))
    target_y_interp = target[0, 0, 2] * (interp_height - 1) / float(max(input_height - 1, 1))
    local_interp = points_on_grid(
        local_grid_size,
        (float(local_extent), float(local_extent)),
        (float(target_y_interp.item()), float(target_x_interp.item())),
        device=device,
    )
    global_interp = points_on_grid(
        global_grid_size,
        (float(interp_height), float(interp_width)),
        None,
        device=device,
    )
    support_interp = torch.cat((local_interp, global_interp), dim=1)
    support_input = support_interp.clone()
    support_input[..., 0] *= (input_width - 1) / float(max(interp_width - 1, 1))
    support_input[..., 1] *= (input_height - 1) / float(max(interp_height - 1, 1))
    support = torch.cat(
        (
            torch.zeros(1, support_input.shape[1], 1, device=device, dtype=target.dtype),
            support_input.to(target.dtype),
        ),
        dim=-1,
    )
    return torch.cat((target, support), dim=1)


def build_official_single_point_queries_interp(
    target_query_txy: torch.Tensor,
    *,
    input_height: int,
    input_width: int,
    interp_height: int,
    interp_width: int,
    local_grid_size: int = 8,
    global_grid_size: int = 5,
    local_extent: int = 50,
) -> torch.Tensor:
    """Exact query tensor passed by official EvaluationPredictor to the model."""
    if target_query_txy.shape != (1, 1, 3):
        raise ValueError("target query must have shape (1,1,3)")
    target = target_query_txy.clone()
    target[..., 1] *= (interp_width - 1) / float(max(input_width - 1, 1))
    target[..., 2] *= (interp_height - 1) / float(max(input_height - 1, 1))
    local = points_on_grid(
        local_grid_size,
        (float(local_extent), float(local_extent)),
        (float(target[0, 0, 2].item()), float(target[0, 0, 1].item())),
        device=target.device,
    )
    global_points = points_on_grid(
        global_grid_size,
        (float(interp_height), float(interp_width)),
        None,
        device=target.device,
    )
    support_xy = torch.cat((local, global_points), dim=1).to(target.dtype)
    support = torch.cat(
        (torch.zeros(1, support_xy.shape[1], 1, device=target.device, dtype=target.dtype), support_xy),
        dim=-1,
    )
    return torch.cat((target, support), dim=1)


def compute_video_fmaps(
    predictor,
    video: torch.Tensor,
    *,
    batch_frames: int = 16,
) -> torch.Tensor:
    """Compute independent per-frame frozen fnet maps, returned on CPU float16."""
    if video.ndim != 5 or video.shape[0] != 1:
        raise ValueError("video must have shape (1,T,3,H,W)")
    interp_height, interp_width = predictor.interp_shape
    outputs = []
    with torch.no_grad():
        for start in range(0, video.shape[1], batch_frames):
            frames = video[0, start : start + batch_frames]
            resized = F.interpolate(
                frames,
                size=(interp_height, interp_width),
                mode="bilinear",
                align_corners=True,
            )
            normalized = 2.0 * (resized / 255.0) - 1.0
            fmap = predictor.model.fnet(normalized)
            outputs.append(F.normalize(fmap.float(), dim=1, eps=1e-12).cpu().half())
    return torch.cat(outputs, dim=0)


def _logit(probability: torch.Tensor) -> torch.Tensor:
    return torch.logit(probability.clamp(1e-6, 1 - 1e-6))


def run_one_query_safe_redetection(
    predictor,
    redetection_model: SafeRedetectionModel,
    video: torch.Tensor,
    target_query_txy: torch.Tensor,
    *,
    precomputed_fmaps: torch.Tensor | None = None,
) -> RuntimeResult:
    """Run one official-independent query with causal output correction/writeback."""
    if video.ndim != 5 or video.shape[0] != 1:
        raise ValueError("video must have shape (1,T,3,H,W)")
    if target_query_txy.shape != (1, 1, 3):
        raise ValueError("query must have shape (1,1,3) in [t,x,y]")
    device = video.device
    _, frames, _, input_height, input_width = video.shape
    interp_height, interp_width = predictor.interp_shape
    support_queries = build_official_single_point_queries_interp(
        target_query_txy,
        input_height=input_height,
        input_width=input_width,
        interp_height=interp_height,
        interp_width=interp_width,
    )
    if precomputed_fmaps is None:
        precomputed_fmaps = compute_video_fmaps(predictor, video)
    if precomputed_fmaps.shape[0] != frames:
        raise ValueError("feature-map frame count mismatch")

    query_frame = int(target_query_txy[0, 0, 0].round().item())
    query_xy_256 = target_query_txy[0, 0, 1:].clone()
    query_xy_256[0] *= 255.0 / float(max(input_width - 1, 1))
    query_xy_256[1] *= 255.0 / float(max(input_height - 1, 1))
    query_fmap = precomputed_fmaps[query_frame : query_frame + 1].to(
        device=device, dtype=torch.float32
    )
    query_feature = sample_batched_map(
        query_fmap,
        query_xy_256.view(1, 1, 2),
        redetection_model.config,
    )[:, 0]
    state = RedetectionStepState(
        initialize_appearance_memory(
            query_feature,
            torch.tensor([query_frame], device=device),
            redetection_model.config,
        ),
        initialize_confirmation_state(1, device=device),
    )

    # Exact official initialization in model-resolution coordinates. Calling
    # CoTrackerOnlinePredictor.is_first_step would rescale already constructed
    # support points through input space and introduce a small round-trip error.
    predictor.model.init_video_online_processing()
    predictor.N = int(support_queries.shape[1])
    predictor.queries = support_queries
    output_tracks = torch.zeros(1, frames, 1, 2, device=device)
    output_vis_prob = torch.zeros(1, frames, 1, device=device)
    output_conf_prob = torch.zeros(1, frames, 1, device=device)
    output_visible = torch.zeros(1, frames, 1, dtype=torch.bool, device=device)
    selected_non_native = torch.zeros_like(output_visible)
    confirmed_recovery = torch.zeros_like(output_visible)
    writeback_applied = torch.zeros_like(output_visible)
    override = torch.zeros(frames, dtype=torch.bool, device=device)
    override_coord = torch.zeros(frames, 2, device=device)
    override_vis_prob = torch.zeros(frames, device=device)
    override_conf_prob = torch.zeros(frames, device=device)
    processed_until = -1
    native_low_confidence_age = 0
    interventions = 0
    writes = 0

    redetection_model.eval()
    with torch.no_grad():
        for chunk_start in range(0, frames - predictor.step, predictor.step):
            chunk = video[:, chunk_start : chunk_start + predictor.step * 2]
            predictor(
                video_chunk=chunk,
                is_first_step=False,
                add_support_grid=False,
                grid_size=0,
            )
            current_frames = min(
                int(predictor.model.online_coords_predicted.shape[1]), frames
            )
            raw_coord = predictor.model.online_coords_predicted[0, :current_frames, 0].float()
            raw_vis = predictor.model.online_vis_predicted[0, :current_frames, 0].float()
            raw_conf = predictor.model.online_conf_predicted[0, :current_frames, 0].float()
            native_coord_input = raw_coord.clone()
            native_coord_input[..., 0] *= (input_width - 1) / float(max(interp_width - 1, 1))
            native_coord_input[..., 1] *= (input_height - 1) / float(max(interp_height - 1, 1))
            native_vis_prob = torch.sigmoid(raw_vis)
            native_conf_prob = torch.sigmoid(raw_conf)
            native_visible = native_vis_prob * native_conf_prob > 0.6

            # Reproduce the official latest-window native output wherever no
            # explicit recovery output has been emitted.
            output_tracks[0, :current_frames, 0] = native_coord_input
            output_vis_prob[0, :current_frames, 0] = native_vis_prob
            output_conf_prob[0, :current_frames, 0] = native_conf_prob
            output_visible[0, :current_frames, 0] = native_visible
            if override.any():
                ids = torch.where(override[:current_frames])[0]
                output_tracks[0, ids, 0] = override_coord[ids]
                output_vis_prob[0, ids, 0] = override_vis_prob[ids]
                output_conf_prob[0, ids, 0] = override_conf_prob[ids]
                output_visible[0, ids, 0] = override_vis_prob[ids] >= 0.5

            first_new = max(processed_until + 1, query_frame)
            for frame in range(first_new, current_frames):
                native_coord_256 = raw_coord[frame : frame + 1].clone()
                native_coord_256[..., 0] *= 255.0 / float(max(interp_width - 1, 1))
                native_coord_256[..., 1] *= 255.0 / float(max(interp_height - 1, 1))
                vis_probability = native_vis_prob[frame : frame + 1]
                conf_probability = native_conf_prob[frame : frame + 1]
                if float((vis_probability * conf_probability).item()) <= 0.6:
                    native_low_confidence_age += 1
                else:
                    native_low_confidence_age = 0
                model_output, state = redetection_model.step(
                    frame_index=frame,
                    feature_map=precomputed_fmaps[frame : frame + 1].to(
                        device=device, dtype=torch.float32
                    ),
                    native_coord_xy_px=native_coord_256,
                    native_visibility_probability=vis_probability,
                    native_confidence_probability=conf_probability,
                    occlusion_age=torch.tensor(
                        [native_low_confidence_age], device=device, dtype=torch.long
                    ),
                    state=state,
                )
                selected_index = int(model_output["selected_candidate_index"][0].item())
                if selected_index > 0:
                    interventions += 1
                    selected_non_native[0, frame, 0] = True
                    confirmed_recovery[0, frame, 0] = bool(
                        model_output["confirmed_recovery"][0].item()
                    )
                    selected_256 = model_output["selected_coord_xy_px"][0]
                    selected_input = selected_256.clone()
                    selected_input[0] *= (input_width - 1) / 255.0
                    selected_input[1] *= (input_height - 1) / 255.0
                    selected_vis = model_output["selected_visibility_probability"][0]
                    selected_conf = torch.maximum(
                        conf_probability[0],
                        model_output["writeback_probability"][0, selected_index],
                    )
                    override[frame] = True
                    override_coord[frame] = selected_input
                    override_vis_prob[frame] = selected_vis
                    override_conf_prob[frame] = selected_conf
                    output_tracks[0, frame, 0] = selected_input
                    output_vis_prob[0, frame, 0] = selected_vis
                    output_conf_prob[0, frame, 0] = selected_conf
                    output_visible[0, frame, 0] = selected_vis >= 0.5

                # The second half of the current window is the state consumed by
                # the next official online window. Edit only after output copy.
                writeback = model_output["future_writeback"]
                if (
                    frame >= chunk_start + predictor.step
                    and bool(writeback.allowed[0].item())
                ):
                    writes += 1
                    writeback_applied[0, frame, 0] = True
                    coord_256 = writeback.coordinate_xy_px[0]
                    coord_interp = coord_256.clone()
                    coord_interp[0] *= (interp_width - 1) / 255.0
                    coord_interp[1] *= (interp_height - 1) / 255.0
                    predictor.model.online_coords_predicted[0, frame, 0] = coord_interp
                    predictor.model.online_vis_predicted[0, frame, 0] = _logit(
                        writeback.visibility_probability[0]
                    )
                    predictor.model.online_conf_predicted[0, frame, 0] = _logit(
                        writeback.confidence_probability[0]
                    )
            processed_until = max(processed_until, current_frames - 1)

    if processed_until < frames - 1:
        raise RuntimeError(
            f"online runtime did not process all frames: {processed_until + 1}/{frames}"
        )
    return RuntimeResult(
        tracks_xy=output_tracks,
        visibility=output_visible,
        visibility_probability=output_vis_prob,
        confidence_probability=output_conf_prob,
        selected_non_native=selected_non_native,
        confirmed_recovery=confirmed_recovery,
        writeback_applied=writeback_applied,
        diagnostics={
            "frames": frames,
            "query_frame": query_frame,
            "internal_support_points": int(support_queries.shape[1] - 1),
            "interventions": interventions,
            "writes": writes,
            "zero_step_override_count": int(override.sum().item()),
        },
    )
