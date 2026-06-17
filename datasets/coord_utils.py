from __future__ import annotations

import numpy as np


def _coord_denominators(height: int, width: int) -> tuple[float, float]:
    return float(max(int(height) - 1, 1)), float(max(int(width) - 1, 1))


def normalize_points_yx(points: np.ndarray, height: int, width: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.size <= 0:
        return points
    if float(np.nanmax(np.abs(points))) <= 1.5:
        return points
    denom_y, denom_x = _coord_denominators(height, width)
    out = points.copy()
    out[..., 0] /= denom_y
    out[..., 1] /= denom_x
    return out


def normalize_query_points_tyx(query_points: np.ndarray, height: int, width: int) -> np.ndarray:
    query_points = np.asarray(query_points, dtype=np.float32)
    if query_points.size <= 0:
        return query_points
    out = query_points.copy()
    if out.shape[-1] >= 3 and float(np.nanmax(np.abs(out[..., 1:3]))) > 1.5:
        denom_y, denom_x = _coord_denominators(height, width)
        out[..., 1] /= denom_y
        out[..., 2] /= denom_x
    return out

