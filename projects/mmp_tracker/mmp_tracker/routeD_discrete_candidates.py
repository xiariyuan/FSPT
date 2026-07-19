"""Deterministic discrete candidate extraction shared by Route-D map audits."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


DISCRETE_CANDIDATE_SCHEMA_VERSION = "routeD_discrete_candidate_extractor_v1"


def _grid_to_input_xy(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    grid_height: int,
    grid_width: int,
    input_height: int,
    input_width: int,
) -> torch.Tensor:
    return torch.stack(
        [
            x * float(input_width - 1) / float(max(grid_width - 1, 1)),
            y * float(input_height - 1) / float(max(grid_height - 1, 1)),
        ],
        dim=-1,
    )


def _sample_map_at_input_xy(
    score_map: torch.Tensor,
    coordinates_xy: torch.Tensor,
    *,
    input_height: int,
    input_width: int,
) -> torch.Tensor:
    """Bilinearly sample one score per row at input-raster coordinates."""
    rows, height, width = score_map.shape
    if coordinates_xy.shape != (rows, 2):
        raise ValueError("native coordinate shape mismatch")
    normalized = coordinates_xy.float().clone()
    normalized[:, 0] = (
        2.0 * normalized[:, 0] / float(max(input_width - 1, 1)) - 1.0
    )
    normalized[:, 1] = (
        2.0 * normalized[:, 1] / float(max(input_height - 1, 1)) - 1.0
    )
    grid = normalized[:, None, None]
    sampled = F.grid_sample(
        score_map[:, None].float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[:, 0, 0, 0]


def _local_softmax_refinement(
    score_map: torch.Tensor,
    *,
    peak_y: int,
    peak_x: int,
    window_cells: int,
    temperature: float,
    input_height: int,
    input_width: int,
) -> torch.Tensor:
    if int(window_cells) <= 0 or int(window_cells) % 2 != 1:
        raise ValueError("local refinement window must be a positive odd integer")
    if float(temperature) <= 0.0:
        raise ValueError("local softmax temperature must be positive")
    height, width = score_map.shape
    half = int(window_cells) // 2
    y0, y1 = max(0, int(peak_y) - half), min(height, int(peak_y) + half + 1)
    x0, x1 = max(0, int(peak_x) - half), min(width, int(peak_x) + half + 1)
    local = score_map[y0:y1, x0:x1].float()
    probability = torch.softmax(local.flatten() / float(temperature), dim=0)
    yy, xx = torch.meshgrid(
        torch.arange(y0, y1, dtype=local.dtype, device=local.device),
        torch.arange(x0, x1, dtype=local.dtype, device=local.device),
        indexing="ij",
    )
    coordinates = _grid_to_input_xy(
        xx.flatten(),
        yy.flatten(),
        grid_height=height,
        grid_width=width,
        input_height=input_height,
        input_width=input_width,
    )
    return probability @ coordinates


def extract_discrete_candidates(
    score_maps: torch.Tensor,
    native_coordinates_xy: torch.Tensor,
    *,
    top_k: int = 8,
    nms_radius_grid_cells: int = 3,
    local_refinement_window_grid_cells: int = 5,
    local_softmax_temperature: float = 0.05,
    deduplicate_radius_input_px: float = 4.0,
    input_height: int = 256,
    input_width: int = 256,
) -> dict[str, Any]:
    """Return native candidate zero plus up to ``top_k`` non-native modes.

    Every examined peak is suppressed with a square Chebyshev-radius NMS mask,
    including peaks later rejected as duplicates of native or an earlier
    refined candidate. Rejected modes are backfilled from the next remaining
    peak. Equal logits follow flattened row-major order, implementing the frozen
    ``descending score, then y, then x`` tie break.
    """
    if score_maps.ndim != 3:
        raise ValueError("score maps must have shape [R,H,W]")
    rows, height, width = score_maps.shape
    if native_coordinates_xy.shape != (rows, 2):
        raise ValueError("native coordinates must have shape [R,2]")
    if int(top_k) <= 0:
        raise ValueError("top_k must be positive")
    if int(nms_radius_grid_cells) < 0:
        raise ValueError("NMS radius must be non-negative")
    if float(deduplicate_radius_input_px) < 0.0:
        raise ValueError("deduplication radius must be non-negative")

    device = score_maps.device
    dtype = score_maps.dtype
    total = int(top_k) + 1
    coordinates = torch.full(
        (rows, total, 2), float("nan"), dtype=torch.float32, device=device
    )
    valid = torch.zeros((rows, total), dtype=torch.bool, device=device)
    scores = torch.full(
        (rows, total), -float("inf"), dtype=torch.float32, device=device
    )
    peak_yx = torch.full((rows, total, 2), -1, dtype=torch.long, device=device)
    native = native_coordinates_xy.float()
    coordinates[:, 0] = native
    valid[:, 0] = True
    native_scores = _sample_map_at_input_xy(
        score_maps,
        native,
        input_height=input_height,
        input_width=input_width,
    )
    scores[:, 0] = native_scores
    counts = torch.ones(rows, dtype=torch.long, device=device)

    for row in range(rows):
        work = score_maps[row].float().clone()
        work[~torch.isfinite(work)] = -float("inf")
        accepted: list[torch.Tensor] = [native[row]]
        examined = 0
        while len(accepted) < total and examined < height * width:
            flat = work.flatten()
            peak_value, flat_index = flat.max(dim=0)
            if not torch.isfinite(peak_value):
                break
            y = int(flat_index.item()) // width
            x = int(flat_index.item()) % width
            refined = _local_softmax_refinement(
                score_maps[row],
                peak_y=y,
                peak_x=x,
                window_cells=int(local_refinement_window_grid_cells),
                temperature=float(local_softmax_temperature),
                input_height=input_height,
                input_width=input_width,
            )
            radius = int(nms_radius_grid_cells)
            work[
                max(0, y - radius) : min(height, y + radius + 1),
                max(0, x - radius) : min(width, x + radius + 1),
            ] = -float("inf")
            examined += 1
            duplicate = any(
                float(torch.linalg.vector_norm(refined - existing))
                <= float(deduplicate_radius_input_px)
                for existing in accepted
            )
            if duplicate:
                continue
            candidate_index = len(accepted)
            accepted.append(refined)
            coordinates[row, candidate_index] = refined
            valid[row, candidate_index] = True
            scores[row, candidate_index] = peak_value
            peak_yx[row, candidate_index] = torch.tensor(
                [y, x], dtype=torch.long, device=device
            )
        counts[row] = len(accepted)

    nonnative_scores = scores[:, 1:]
    native_rank = 1 + (
        (nonnative_scores > native_scores[:, None]) & valid[:, 1:]
    ).sum(dim=1)
    return {
        "candidate_coordinates_xy": coordinates.to(dtype=dtype),
        "candidate_valid_mask": valid,
        "candidate_scores": scores.to(dtype=dtype),
        "candidate_peak_yx": peak_yx,
        "candidate_count": counts,
        "native_map_score": native_scores.to(dtype=dtype),
        "native_candidate_rank": native_rank,
    }
