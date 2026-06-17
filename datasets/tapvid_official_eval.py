"""
Official TAP-Vid evaluation core (vendored).

This file ports the TAP-Vid metric core from:
  google-deepmind/tapnet/tapnet/tapvid/evaluation_datasets.py

Only NumPy-based metric/query-sampling functions are kept here so the training
pipeline can use official formulas without pulling TensorFlow/TFDS runtime deps.
"""

from typing import Mapping, Tuple

import numpy as np


def compute_tapvid_metrics_official(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
    thresholds: Tuple[int, ...] = (1, 2, 4, 8, 16),
    get_trackwise_metrics: bool = False,
) -> Mapping[str, np.ndarray]:
    """Official TAP-Vid metrics (ported from DeepMind tapnet)."""
    summing_axis = (2,) if get_trackwise_metrics else (1, 2)

    metrics = {}

    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)
    if query_mode == "first":
        # evaluate frames after the query frame
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == "strided":
        # evaluate all frames except the query frame
        query_frame_to_eval_frames = 1 - eye
    else:
        raise ValueError("Unknown query mode " + str(query_mode))

    query_frame = np.round(query_points[..., 0]).astype(np.int32)
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0

    # Occlusion accuracy.
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=summing_axis,
    ) / np.sum(evaluation_points, axis=summing_axis)
    metrics["occlusion_accuracy"] = occ_acc

    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []

    for thresh in thresholds:
        # Squared distance for exact official behavior.
        within_dist = (
            np.sum(np.square(pred_tracks - gt_tracks), axis=-1) < np.square(thresh)
        )
        is_correct = np.logical_and(within_dist, visible)

        count_correct = np.sum(is_correct & evaluation_points, axis=summing_axis)
        count_visible_points = np.sum(visible & evaluation_points, axis=summing_axis)
        frac_correct = count_correct / count_visible_points
        metrics["pts_within_" + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=summing_axis
        )

        gt_positives = np.sum(visible & evaluation_points, axis=summing_axis)
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=summing_axis)
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)

    metrics["average_jaccard"] = np.mean(np.stack(all_jaccard, axis=1), axis=1)
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(all_frac_within, axis=1), axis=1
    )
    return metrics


def sample_queries_strided(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
    query_stride: int = 5,
) -> Mapping[str, np.ndarray]:
    """Official TAP-Vid strided query sampler (ported)."""
    tracks = []
    occs = []
    queries = []
    trackgroups = []
    trackgroup = np.arange(target_occluded.shape[0])

    for i in range(0, target_occluded.shape[1], query_stride):
        mask = target_occluded[:, i] == 0
        query = np.stack(
            [
                i * np.ones(target_occluded.shape[0:1]),
                target_points[:, i, 1],
                target_points[:, i, 0],
            ],
            axis=-1,
        )
        queries.append(query[mask])
        tracks.append(target_points[mask])
        occs.append(target_occluded[mask])
        trackgroups.append(trackgroup[mask])

    return {
        "video": frames[np.newaxis, ...],
        "query_points": np.concatenate(queries, axis=0)[np.newaxis, ...],
        "target_points": np.concatenate(tracks, axis=0)[np.newaxis, ...],
        "occluded": np.concatenate(occs, axis=0)[np.newaxis, ...],
        "trackgroup": np.concatenate(trackgroups, axis=0)[np.newaxis, ...],
    }


def sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Official TAP-Vid first-visible query sampler (ported)."""
    valid = np.sum(~target_occluded, axis=1) > 0
    target_points = target_points[valid, :]
    target_occluded = target_occluded[valid, :]

    query_points = []
    for i in range(target_points.shape[0]):
        index = np.where(target_occluded[i] == 0)[0][0]
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(np.array([index, y, x]))  # [t, y, x]
    query_points = np.stack(query_points, axis=0)

    return {
        "video": frames[np.newaxis, ...],
        "query_points": query_points[np.newaxis, ...],
        "target_points": target_points[np.newaxis, ...],
        "occluded": target_occluded[np.newaxis, ...],
    }

