from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, IterableDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import compute_tapvid_metrics, get_dataset
from datasets.tapvid_official_eval import compute_tapvid_metrics_official
from projects.mmp_tracker.mmp_tracker import MMPTracker, MMPTrackerConfig, MMPLossWeights, MMPTrackingLoss
from projects.mmp_tracker.mmp_tracker.config import (
    EncoderConfig,
    GlobalRelocatorConfig,
    LocalMatcherConfig,
    PosteriorFusionConfig,
    TrackingConfig,
)
from projects.mmp_tracker.mmp_tracker.data import wrap_first_frame_query_dataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mmp_train")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_from_dict(cfg: Dict[str, Any]) -> MMPTrackerConfig:
    model_cfg = deepcopy(cfg.get("model", {}))
    return MMPTrackerConfig(
        variant=str(model_cfg.get("variant", "posterior")),
        encoder=EncoderConfig(**model_cfg.get("encoder", {})),
        local_matcher=LocalMatcherConfig(**model_cfg.get("local_matcher", {})),
        global_relocator=GlobalRelocatorConfig(**model_cfg.get("global_relocator", {})),
        posterior_fusion=PosteriorFusionConfig(**model_cfg.get("posterior_fusion", {})),
        tracking=TrackingConfig(**model_cfg.get("tracking", {})),
    )


def loss_from_dict(cfg: Dict[str, Any]) -> MMPTrackingLoss:
    loss_cfg = deepcopy(cfg.get("loss", {}))
    weights = MMPLossWeights(
        coordinate=float(loss_cfg.get("coordinate", {}).get("weight", 1.0)),
        local_heatmap=float(loss_cfg.get("local_heatmap", {}).get("weight", 0.0)),
        global_heatmap=float(loss_cfg.get("global_heatmap", {}).get("weight", 0.0)),
        global_coordinate=float(loss_cfg.get("global_coordinate", {}).get("weight", 0.0)),
        selector=float(loss_cfg.get("selector", {}).get("weight", 0.0)),
        commit=float(loss_cfg.get("commit", {}).get("weight", 0.0)),
        visibility=float(loss_cfg.get("visibility", {}).get("weight", 0.2)),
        posterior_consistency=float(loss_cfg.get("posterior_consistency", {}).get("weight", 0.0)),
        temporal_smoothness=float(loss_cfg.get("temporal_smoothness", {}).get("weight", 0.0)),
        no_harm_prior=float(loss_cfg.get("no_harm_prior", {}).get("weight", 0.0)),
        long_occ_focus=float(loss_cfg.get("long_occ_focus", {}).get("weight", 0.0)),
    )
    return MMPTrackingLoss(weights)


def resolve_dataset(cfg: Dict[str, Any], split_key: str, train: bool):
    data_cfg = deepcopy(cfg.get("data", {}).get(split_key, {}))
    dataset_name = data_cfg.pop("dataset")
    root = data_cfg.pop("root")
    max_points_raw = data_cfg.pop("max_points", None)
    max_points = int(max_points_raw) if max_points_raw is not None else None
    first_frame_query = bool(data_cfg.pop("first_frame_query", True if train else False))
    subset = data_cfg.pop("subset", None)
    explicit_num_points = data_cfg.get("num_points", None)
    if explicit_num_points is not None:
        data_cfg["num_points"] = int(explicit_num_points)
    elif train and max_points is not None:
        data_cfg["num_points"] = int(max_points)
    if train:
        data_cfg.setdefault("split", "train")
        data_cfg.setdefault("backend", "sharded_pkl")
        data_cfg.setdefault("annotation_file", "train.index.json")
        data_cfg.setdefault("query_mode", "first")
    else:
        normalized_dataset_name = (
            str(dataset_name).lower().replace("-", "_").strip()
        )
        if normalized_dataset_name.startswith("tapvid_"):
            normalized_dataset_name = normalized_dataset_name[len("tapvid_") :]
        if normalized_dataset_name in {
            "davis",
            "rgb_stacking",
            "rgbstacking",
            "stacking",
        }:
            data_cfg.pop("split", None)
            data_cfg.pop("backend", None)
        else:
            data_cfg.setdefault("split", "validation")
            data_cfg.setdefault("backend", "sharded_pkl")

    dataset = get_dataset(dataset_name, root, **data_cfg)
    if first_frame_query:
        dataset = wrap_first_frame_query_dataset(dataset, max_points=max_points, random_sample=train)

    max_samples = None
    if subset == "smoke":
        max_samples = 64 if train else 8
    elif isinstance(subset, int):
        max_samples = subset
    return dataset, max_samples


def iterate_limited(loader: Iterable, limit: Optional[int] = None):
    for idx, batch in enumerate(loader):
        if limit is not None and idx >= limit:
            break
        yield idx, batch


def make_loader(dataset, batch_size: int, train: bool):
    shuffle = bool(train and not isinstance(dataset, IterableDataset))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def unwrap_base_dataset(dataset):
    current = dataset
    seen = set()
    while hasattr(current, "base_dataset") and id(current) not in seen:
        seen.add(id(current))
        current = current.base_dataset
    return current


def resolve_eval_query_mode(batch: Dict[str, Any], dataset) -> str:
    batch_query_mode = batch.get("query_mode", None)
    if isinstance(batch_query_mode, (list, tuple)) and len(batch_query_mode) > 0:
        return str(batch_query_mode[0]).strip().lower()
    if isinstance(batch_query_mode, str):
        return str(batch_query_mode).strip().lower()

    base_dataset = unwrap_base_dataset(dataset)
    dataset_query_mode = getattr(base_dataset, "query_mode", None)
    if dataset_query_mode is not None:
        return str(dataset_query_mode).strip().lower()

    return "strided"


def resolve_eval_resolution(batch: Dict[str, Any], video: torch.Tensor, index: int):
    if video.ndim >= 5:
        return (int(video.shape[-2]), int(video.shape[-1]))
    if video.ndim >= 4:
        return (int(video.shape[-2]), int(video.shape[-1]))

    original_size = batch.get("original_size", None)
    if isinstance(original_size, torch.Tensor) and original_size.ndim == 2:
        return (int(original_size[index, 0].item()), int(original_size[index, 1].item()))
    return 256


def gather_candidate_tracks(candidate_tracks: torch.Tensor, best_index: torch.Tensor) -> torch.Tensor:
    gather_index = best_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, candidate_tracks.shape[-1])
    return candidate_tracks.gather(2, gather_index).squeeze(2)


def build_oracle_candidate_tracks(
    local_tracks: torch.Tensor,
    candidate_tracks: torch.Tensor,
    gt_tracks: torch.Tensor,
    visible_mask: torch.Tensor,
    margin: float,
):
    if candidate_tracks.ndim != 4 or candidate_tracks.shape[2] <= 0:
        return local_tracks, torch.zeros_like(visible_mask, dtype=torch.bool)
    local_err = torch.norm(local_tracks - gt_tracks, dim=-1)
    candidate_err = torch.norm(candidate_tracks - gt_tracks.unsqueeze(2), dim=-1)
    best_candidate_err, best_candidate_idx = candidate_err.min(dim=2)
    best_candidate_tracks = gather_candidate_tracks(candidate_tracks, best_candidate_idx)
    use_candidate = visible_mask & (best_candidate_err + float(margin) < local_err)
    oracle_tracks = torch.where(use_candidate.unsqueeze(-1), best_candidate_tracks, local_tracks)
    return oracle_tracks, use_candidate


def accumulate_prefixed_metrics(
    metrics_sum: Dict[str, float],
    longocc_sum: Dict[str, float],
    longocc_count: Dict[str, int],
    prefix: str,
    metrics: Dict[str, float],
    pred_tracks: torch.Tensor,
    target_points: torch.Tensor,
    pred_visibility: torch.Tensor,
    occluded: torch.Tensor,
    query_points: torch.Tensor,
    resolution,
    eval_query_mode: str,
) -> None:
    for key, value in metrics.items():
        metrics_sum[f"{prefix}_{key}"] = metrics_sum.get(f"{prefix}_{key}", 0.0) + float(value)
    for min_occ in (10, 20, 30):
        subset_mask = build_long_occ_subset_mask(occluded, query_points[:, 0], min_occ_frames=min_occ)
        if int(subset_mask.sum().item()) <= 0:
            continue
        subset_metrics = compute_trackwise_tapvid_metrics(
            pred_tracks[subset_mask],
            target_points[subset_mask],
            pred_visibility[subset_mask],
            (~occluded)[subset_mask],
            query_points[subset_mask],
            resolution=resolution,
            exclude_query_frame=True,
            query_mode=eval_query_mode,
        )
        aj_values = subset_metrics.get("average_jaccard", None)
        if aj_values is not None and aj_values.size > 0:
            key = f"{prefix}_AJ_longocc{min_occ}"
            longocc_sum[key] = longocc_sum.get(key, 0.0) + float(np.sum(aj_values))
            longocc_count[key] = longocc_count.get(key, 0) + int(aj_values.size)
        oa_values = subset_metrics.get("occlusion_accuracy", None)
        if oa_values is not None and oa_values.size > 0:
            key = f"{prefix}_OA_longocc{min_occ}"
            longocc_sum[key] = longocc_sum.get(key, 0.0) + float(np.sum(oa_values))
            longocc_count[key] = longocc_count.get(key, 0) + int(oa_values.size)
        reapp_errors = compute_reappearance_errors_px(
            pred_tracks,
            target_points,
            occluded,
            query_points[:, 0],
            resolution=resolution,
            min_occ_frames=min_occ,
        )
        if reapp_errors:
            key = f"{prefix}_reapp_error_longocc{min_occ}_px"
            longocc_sum[key] = longocc_sum.get(key, 0.0) + float(sum(reapp_errors))
            longocc_count[key] = longocc_count.get(key, 0) + len(reapp_errors)


def evaluate(model: MMPTracker, loader: DataLoader, device: torch.device, limit: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    metrics_sum: Dict[str, float] = {}
    count = 0
    longocc_sum: Dict[str, float] = {}
    longocc_count: Dict[str, int] = {}
    routing_sum: Dict[str, float] = {
        "selected_global_rate": 0.0,
        "commit_rate": 0.0,
        "commit_given_selected": 0.0,
        "global_better_rate": 0.0,
        "candidate_global_better_rate": 0.0,
        "selected_when_global_better": 0.0,
        "commit_when_global_better": 0.0,
        "candidate_selected_oracle_rate": 0.0,
        "candidate_gate_when_global_better": 0.0,
        "candidate_rank_oracle_rate": 0.0,
        "selected_precision": 0.0,
        "commit_precision": 0.0,
        "blocked_by_quality_rate": 0.0,
        "blocked_by_consistency_rate": 0.0,
        "pending_input_rate": 0.0,
        "pending_stage_rate": 0.0,
        "pending_confirm_rate": 0.0,
        "confirm_given_pending": 0.0,
        "pending_stage_when_global_better": 0.0,
        "pending_confirm_when_global_better": 0.0,
        "pending_stage_precision": 0.0,
        "pending_confirm_precision": 0.0,
    }
    routing_count: Dict[str, int] = {key: 0 for key in routing_sum}
    with torch.no_grad():
        for _, batch in iterate_limited(loader, limit):
            video = batch["video"].to(device)
            query_points = batch["query_points"].to(device)
            target_points = batch["target_points"].to(device)
            occluded = batch["occluded"].to(device)
            eval_query_mode = resolve_eval_query_mode(batch, loader.dataset)
            pred_tracks, pred_visibility, info = model(video, query_points, return_info=True)
            batch_size = video.shape[0]
            for i in range(batch_size):
                resolution = resolve_eval_resolution(batch, video, i)
                metrics = compute_tapvid_metrics(
                    pred_tracks[i],
                    target_points[i],
                    pred_visibility[i],
                    ~occluded[i],
                    query_points[i],
                    resolution=resolution,
                    exclude_query_frame=True,
                    query_mode=eval_query_mode,
                )
                for key, value in metrics.items():
                    metrics_sum[key] = metrics_sum.get(key, 0.0) + float(value)

                if isinstance(info, dict):
                    active_mask = info.get("active_mask", None)
                    selected_global_mask = info.get("selected_global_mask", None)
                    commit_mask = info.get("commit_mask", None)
                    local_points = info.get("local_points", None)
                    global_points = info.get("global_points", None)
                    candidate_points = info.get("candidate_points", None)
                    global_candidate_points = info.get("global_candidate_points", None)
                    global_candidate_coarse_points = info.get("global_candidate_coarse_points", None)
                    candidate_selected_index = info.get("candidate_selected_index", None)
                    candidate_global_selected_index = info.get("candidate_global_selected_index", None)
                    commit_quality_pass = info.get("commit_quality_pass", None)
                    commit_consistency_pass = info.get("commit_consistency_pass", None)
                    pending_input_mask = info.get("pending_input_mask", None)
                    pending_stage_mask = info.get("pending_stage_mask", None)
                    pending_confirm_mask = info.get("pending_confirm_mask", None)
                    selector_margin = float(info.get("selector_target_margin", 0.01)) if isinstance(info, dict) else 0.01
                    if (
                        isinstance(active_mask, torch.Tensor)
                        and isinstance(selected_global_mask, torch.Tensor)
                        and isinstance(commit_mask, torch.Tensor)
                    ):
                        active_i = active_mask[i].bool()
                        visible_i = (~occluded[i]).bool() & active_i
                        selected_i = selected_global_mask[i].bool() & active_i
                        commit_i = commit_mask[i].bool() & active_i
                        pending_input_i = (
                            pending_input_mask[i].bool() & active_i if isinstance(pending_input_mask, torch.Tensor) else torch.zeros_like(active_i)
                        )
                        pending_stage_i = (
                            pending_stage_mask[i].bool() & active_i if isinstance(pending_stage_mask, torch.Tensor) else torch.zeros_like(active_i)
                        )
                        pending_confirm_i = (
                            pending_confirm_mask[i].bool() & active_i if isinstance(pending_confirm_mask, torch.Tensor) else torch.zeros_like(active_i)
                        )
                        active_count = int(active_i.sum().item())
                        selected_count = int(selected_i.sum().item())
                        commit_count = int(commit_i.sum().item())
                        pending_input_count = int(pending_input_i.sum().item())
                        pending_stage_count = int(pending_stage_i.sum().item())
                        pending_confirm_count = int(pending_confirm_i.sum().item())
                        if active_count > 0:
                            routing_sum["selected_global_rate"] += float(selected_count) / float(active_count)
                            routing_sum["commit_rate"] += float(commit_count) / float(active_count)
                            routing_sum["pending_input_rate"] += float(pending_input_count) / float(active_count)
                            routing_sum["pending_stage_rate"] += float(pending_stage_count) / float(active_count)
                            routing_sum["pending_confirm_rate"] += float(pending_confirm_count) / float(active_count)
                            routing_count["selected_global_rate"] += 1
                            routing_count["commit_rate"] += 1
                            routing_count["pending_input_rate"] += 1
                            routing_count["pending_stage_rate"] += 1
                            routing_count["pending_confirm_rate"] += 1
                        if selected_count > 0:
                            routing_sum["commit_given_selected"] += float((commit_i & selected_i).sum().item()) / float(selected_count)
                            routing_count["commit_given_selected"] += 1
                        if pending_input_count > 0:
                            routing_sum["confirm_given_pending"] += float((pending_confirm_i & pending_input_i).sum().item()) / float(pending_input_count)
                            routing_count["confirm_given_pending"] += 1
                        if isinstance(local_points, torch.Tensor) and isinstance(global_points, torch.Tensor):
                            local_err = torch.norm(local_points[i] - target_points[i], dim=-1)
                            global_err = torch.norm(global_points[i] - target_points[i], dim=-1)
                            global_better_i = (global_err + selector_margin < local_err) & visible_i
                            better_count = int(global_better_i.sum().item())
                            visible_count = int(visible_i.sum().item())
                            if visible_count > 0:
                                routing_sum["global_better_rate"] += float(better_count) / float(visible_count)
                                routing_count["global_better_rate"] += 1
                            if better_count > 0:
                                selected_hit = int((selected_i & global_better_i).sum().item())
                                commit_hit = int((commit_i & global_better_i).sum().item())
                                routing_sum["selected_when_global_better"] += float(selected_hit) / float(better_count)
                                routing_sum["commit_when_global_better"] += float(commit_hit) / float(better_count)
                                pending_stage_hit = int((pending_stage_i & global_better_i).sum().item())
                                pending_confirm_hit = int((pending_confirm_i & global_better_i).sum().item())
                                routing_sum["pending_stage_when_global_better"] += float(pending_stage_hit) / float(better_count)
                                routing_sum["pending_confirm_when_global_better"] += float(pending_confirm_hit) / float(better_count)
                                routing_count["selected_when_global_better"] += 1
                                routing_count["commit_when_global_better"] += 1
                                routing_count["pending_stage_when_global_better"] += 1
                                routing_count["pending_confirm_when_global_better"] += 1
                            if selected_count > 0:
                                routing_sum["selected_precision"] += float((selected_i & global_better_i).sum().item()) / float(selected_count)
                                routing_count["selected_precision"] += 1
                            if commit_count > 0:
                                routing_sum["commit_precision"] += float((commit_i & global_better_i).sum().item()) / float(commit_count)
                                routing_count["commit_precision"] += 1
                            if pending_stage_count > 0:
                                routing_sum["pending_stage_precision"] += float((pending_stage_i & global_better_i).sum().item()) / float(pending_stage_count)
                                routing_count["pending_stage_precision"] += 1
                            if pending_confirm_count > 0:
                                routing_sum["pending_confirm_precision"] += float((pending_confirm_i & global_better_i).sum().item()) / float(pending_confirm_count)
                                routing_count["pending_confirm_precision"] += 1
                        if (
                            isinstance(global_candidate_points, torch.Tensor)
                            and isinstance(local_points, torch.Tensor)
                            and global_candidate_points.ndim >= 4
                            and global_candidate_points.shape[3] > 0
                        ):
                            candidate_local_err = torch.norm(local_points[i] - target_points[i], dim=-1)
                            candidate_global_err = torch.norm(global_candidate_points[i] - target_points[i].unsqueeze(2), dim=-1)
                            best_candidate_err, best_candidate_idx = candidate_global_err.min(dim=2)
                            candidate_better_i = (best_candidate_err + selector_margin < candidate_local_err) & visible_i
                            candidate_better_count = int(candidate_better_i.sum().item())
                            visible_count = int(visible_i.sum().item())
                            if visible_count > 0:
                                routing_sum["candidate_global_better_rate"] += float(candidate_better_count) / float(visible_count)
                                routing_count["candidate_global_better_rate"] += 1
                            if isinstance(candidate_selected_index, torch.Tensor):
                                oracle_selected_idx = torch.where(
                                    candidate_better_i,
                                    best_candidate_idx + 1,
                                    torch.zeros_like(best_candidate_idx),
                                )
                                selected_oracle = ((candidate_selected_index[i] == oracle_selected_idx) & visible_i).sum().item()
                                visible_oracle = int(visible_i.sum().item())
                                if visible_oracle > 0:
                                    routing_sum["candidate_selected_oracle_rate"] += float(selected_oracle) / float(visible_oracle)
                                    routing_count["candidate_selected_oracle_rate"] += 1
                            if candidate_better_count > 0:
                                gate_hit = int((selected_i & candidate_better_i).sum().item())
                                routing_sum["candidate_gate_when_global_better"] += float(gate_hit) / float(candidate_better_count)
                                routing_count["candidate_gate_when_global_better"] += 1
                                if isinstance(candidate_global_selected_index, torch.Tensor):
                                    rank_hit = int(
                                        ((candidate_global_selected_index[i] == best_candidate_idx) & candidate_better_i).sum().item()
                                    )
                                    routing_sum["candidate_rank_oracle_rate"] += float(rank_hit) / float(candidate_better_count)
                                    routing_count["candidate_rank_oracle_rate"] += 1
                        if isinstance(commit_quality_pass, torch.Tensor) and selected_count > 0:
                            blocked_quality = int((selected_i & (~commit_quality_pass[i].bool()) & active_i).sum().item())
                            routing_sum["blocked_by_quality_rate"] += float(blocked_quality) / float(selected_count)
                            routing_count["blocked_by_quality_rate"] += 1
                        if isinstance(commit_consistency_pass, torch.Tensor) and selected_count > 0:
                            blocked_consistency = int((selected_i & (~commit_consistency_pass[i].bool()) & active_i).sum().item())
                            routing_sum["blocked_by_consistency_rate"] += float(blocked_consistency) / float(selected_count)
                            routing_count["blocked_by_consistency_rate"] += 1

                    if (
                        isinstance(local_points, torch.Tensor)
                        and isinstance(global_candidate_coarse_points, torch.Tensor)
                        and global_candidate_coarse_points.ndim >= 4
                        and global_candidate_coarse_points.shape[3] > 0
                    ):
                        oracle_coarse_tracks, _ = build_oracle_candidate_tracks(
                            local_points[i],
                            global_candidate_coarse_points[i],
                            target_points[i],
                            (~occluded[i]).bool(),
                            selector_margin,
                        )
                        oracle_coarse_metrics = compute_tapvid_metrics(
                            oracle_coarse_tracks,
                            target_points[i],
                            pred_visibility[i],
                            ~occluded[i],
                            query_points[i],
                            resolution=resolution,
                            exclude_query_frame=True,
                            query_mode=eval_query_mode,
                        )
                        accumulate_prefixed_metrics(
                            metrics_sum,
                            longocc_sum,
                            longocc_count,
                            prefix="oracle_coarse",
                            metrics=oracle_coarse_metrics,
                            pred_tracks=oracle_coarse_tracks,
                            target_points=target_points[i],
                            pred_visibility=pred_visibility[i],
                            occluded=occluded[i],
                            query_points=query_points[i],
                            resolution=resolution,
                            eval_query_mode=eval_query_mode,
                        )

                    if (
                        isinstance(local_points, torch.Tensor)
                        and isinstance(global_candidate_points, torch.Tensor)
                        and global_candidate_points.ndim >= 4
                        and global_candidate_points.shape[3] > 0
                    ):
                        oracle_rematch_tracks, _ = build_oracle_candidate_tracks(
                            local_points[i],
                            global_candidate_points[i],
                            target_points[i],
                            (~occluded[i]).bool(),
                            selector_margin,
                        )
                        oracle_rematch_metrics = compute_tapvid_metrics(
                            oracle_rematch_tracks,
                            target_points[i],
                            pred_visibility[i],
                            ~occluded[i],
                            query_points[i],
                            resolution=resolution,
                            exclude_query_frame=True,
                            query_mode=eval_query_mode,
                        )
                        accumulate_prefixed_metrics(
                            metrics_sum,
                            longocc_sum,
                            longocc_count,
                            prefix="oracle_rematch",
                            metrics=oracle_rematch_metrics,
                            pred_tracks=oracle_rematch_tracks,
                            target_points=target_points[i],
                            pred_visibility=pred_visibility[i],
                            occluded=occluded[i],
                            query_points=query_points[i],
                            resolution=resolution,
                            eval_query_mode=eval_query_mode,
                        )
                count += 1

                for min_occ in (10, 20, 30):
                    subset_mask = build_long_occ_subset_mask(occluded[i], query_points[i, :, 0], min_occ_frames=min_occ)
                    if int(subset_mask.sum().item()) <= 0:
                        continue
                    subset_metrics = compute_trackwise_tapvid_metrics(
                        pred_tracks[i][subset_mask],
                        target_points[i][subset_mask],
                        pred_visibility[i][subset_mask],
                        (~occluded[i])[subset_mask],
                        query_points[i][subset_mask],
                        resolution=resolution,
                        exclude_query_frame=True,
                        query_mode=eval_query_mode,
                    )
                    aj_values = subset_metrics.get("average_jaccard", None)
                    if aj_values is not None and aj_values.size > 0:
                        key = f"AJ_longocc{min_occ}"
                        longocc_sum[key] = longocc_sum.get(key, 0.0) + float(np.sum(aj_values))
                        longocc_count[key] = longocc_count.get(key, 0) + int(aj_values.size)
                    oa_values = subset_metrics.get("occlusion_accuracy", None)
                    if oa_values is not None and oa_values.size > 0:
                        key = f"OA_longocc{min_occ}"
                        longocc_sum[key] = longocc_sum.get(key, 0.0) + float(np.sum(oa_values))
                        longocc_count[key] = longocc_count.get(key, 0) + int(oa_values.size)
                    reapp_errors = compute_reappearance_errors_px(
                        pred_tracks[i],
                        target_points[i],
                        occluded[i],
                        query_points[i, :, 0],
                        resolution=resolution,
                        min_occ_frames=min_occ,
                    )
                    if reapp_errors:
                        key = f"reapp_error_longocc{min_occ}_px"
                        longocc_sum[key] = longocc_sum.get(key, 0.0) + float(sum(reapp_errors))
                        longocc_count[key] = longocc_count.get(key, 0) + len(reapp_errors)
    if count == 0:
        return {"AJ": 0.0, "OA": 0.0, "avg_error_px": 0.0}
    out = {key: value / count for key, value in metrics_sum.items()}
    for key, value in routing_sum.items():
        out[key] = value / max(routing_count.get(key, 0), 1)
    for key, value in longocc_sum.items():
        denom = max(longocc_count.get(key, 0), 1)
        out[key] = value / denom
    return out


def compute_trackwise_tapvid_metrics(
    pred_tracks: torch.Tensor,
    gt_tracks: torch.Tensor,
    pred_visibility: torch.Tensor,
    gt_visibility: torch.Tensor,
    query_points: torch.Tensor,
    resolution,
    exclude_query_frame: bool = True,
    query_mode: Optional[str] = None,
):
    device = pred_tracks.device
    if isinstance(resolution, (tuple, list)):
        height = float(resolution[0])
        width = float(resolution[1])
    else:
        height = float(resolution)
        width = float(resolution)

    if pred_visibility.dtype != torch.bool:
        pred_visibility = pred_visibility > 0.5
    if gt_visibility.dtype != torch.bool:
        gt_visibility = gt_visibility > 0.5
    pred_occluded = ~pred_visibility
    gt_occluded = ~gt_visibility

    scale_w = max(width - 1.0, 1.0)
    scale_h = max(height - 1.0, 1.0)
    xy_scale = torch.tensor([scale_w, scale_h], device=device, dtype=pred_tracks.dtype)

    pred_tracks_xy = pred_tracks[..., [1, 0]].to(dtype=pred_tracks.dtype)
    gt_tracks_xy = gt_tracks[..., [1, 0]].to(dtype=gt_tracks.dtype)
    if max(float(pred_tracks_xy.abs().max()), float(gt_tracks_xy.abs().max())) <= 1.5:
        pred_tracks_px = pred_tracks_xy * xy_scale
        gt_tracks_px = gt_tracks_xy * xy_scale
    else:
        pred_tracks_px = pred_tracks_xy
        gt_tracks_px = gt_tracks_xy

    query_points_px = query_points[:, :3].to(dtype=pred_tracks.dtype).clone()
    if query_points_px[:, 1:3].numel() > 0 and float(query_points_px[:, 1:3].abs().max()) <= 1.5:
        query_points_px[:, 1] *= scale_h
        query_points_px[:, 2] *= scale_w

    mode = str(query_mode).strip().lower() if query_mode is not None else None
    if mode in ("", "none", "null"):
        mode = None
    if mode is None:
        mode = "strided" if exclude_query_frame else "strided"

    off = compute_tapvid_metrics_official(
        query_points=query_points_px.detach().cpu().numpy()[None, ...],
        gt_occluded=gt_occluded.detach().cpu().numpy()[None, ...],
        gt_tracks=gt_tracks_px.detach().cpu().numpy()[None, ...],
        pred_occluded=pred_occluded.detach().cpu().numpy()[None, ...],
        pred_tracks=pred_tracks_px.detach().cpu().numpy()[None, ...],
        query_mode=mode,
        thresholds=(1, 2, 4, 8, 16),
        get_trackwise_metrics=True,
    )
    return {key: np.asarray(value).reshape(-1) for key, value in off.items()}


def build_long_occ_subset_mask(occluded: torch.Tensor, query_t: torch.Tensor, min_occ_frames: int) -> torch.Tensor:
    num_points, time = occluded.shape
    query_t = query_t.round().long().clamp(0, time - 1)
    subset = torch.zeros(num_points, dtype=torch.bool, device=occluded.device)
    for point_idx in range(num_points):
        run = 0
        started = False
        for t in range(int(query_t[point_idx].item()) + 1, time):
            is_occ = bool(occluded[point_idx, t].item())
            if is_occ:
                run += 1
                started = True
            else:
                if started and run >= int(min_occ_frames):
                    subset[point_idx] = True
                    break
                run = 0
                started = False
    return subset


def compute_reappearance_errors_px(
    pred_tracks: torch.Tensor,
    gt_tracks: torch.Tensor,
    occluded: torch.Tensor,
    query_t: torch.Tensor,
    resolution,
    min_occ_frames: int,
) -> list[float]:
    if isinstance(resolution, (tuple, list)):
        height = float(resolution[0])
        width = float(resolution[1])
    else:
        height = float(resolution)
        width = float(resolution)
    scale = torch.tensor([max(height - 1.0, 1.0), max(width - 1.0, 1.0)], device=pred_tracks.device, dtype=pred_tracks.dtype)
    query_t = query_t.round().long().clamp(0, pred_tracks.shape[1] - 1)
    errors = []
    for point_idx in range(pred_tracks.shape[0]):
        run = 0
        started = False
        for t in range(int(query_t[point_idx].item()) + 1, pred_tracks.shape[1]):
            is_occ = bool(occluded[point_idx, t].item())
            if is_occ:
                run += 1
                started = True
                continue
            if started and run >= int(min_occ_frames):
                delta = (pred_tracks[point_idx, t] - gt_tracks[point_idx, t]) * scale
                errors.append(float(torch.norm(delta, dim=-1).item()))
                break
            run = 0
            started = False
    return errors


def train_one_epoch(
    model: MMPTracker,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MMPTrackingLoss,
    device: torch.device,
    limit: Optional[int] = None,
    accumulation_steps: int = 1,
    long_occ_min_frames: int = 3,
    long_occ_recovery_window: int = 1,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    steps = 0
    component_totals: Dict[str, float] = {}
    accumulation_steps = max(int(accumulation_steps), 1)
    optimizer.zero_grad(set_to_none=True)
    for _, batch in iterate_limited(loader, limit):
        video = batch["video"].to(device)
        query_points = batch["query_points"].to(device)
        target_points = batch["target_points"].to(device)
        occluded = batch["occluded"].to(device)
        pred_tracks, pred_visibility, info = model(video, query_points, return_info=True)
        gt_visibility = (~occluded).to(dtype=pred_visibility.dtype)
        focus_mask = build_long_occ_focus_mask(
            gt_visibility > 0.5,
            min_occ_frames=long_occ_min_frames,
            recovery_window=long_occ_recovery_window,
        )
        losses = criterion(
            pred_tracks=pred_tracks,
            pred_visibility=pred_visibility,
            gt_tracks=target_points,
            gt_visibility=gt_visibility,
            info=info,
            prior_tracks=info.get("prior_points", info.get("local_points", None)),
            focus_mask=focus_mask,
        )
        loss = losses["total"]
        for key, value in losses.items():
            component_totals[key] = component_totals.get(key, 0.0) + float(value.item())
        (loss / accumulation_steps).backward()
        if (steps + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.item())
        steps += 1
    if steps > 0 and steps % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    stats = {"loss": total_loss / max(steps, 1), "steps": steps}
    for key, value in component_totals.items():
        stats[f"train_{key}"] = value / max(steps, 1)
    return stats


def build_long_occ_focus_mask(
    gt_visibility: torch.Tensor,
    min_occ_frames: int = 3,
    recovery_window: int = 1,
) -> torch.Tensor:
    if gt_visibility.ndim != 3:
        raise ValueError("gt_visibility must be shaped as (B, N, T).")
    batch, num_points, time = gt_visibility.shape
    focus_mask = torch.zeros_like(gt_visibility, dtype=torch.bool)
    occ_run = torch.zeros(batch, num_points, dtype=torch.long, device=gt_visibility.device)
    recovery_left = torch.zeros(batch, num_points, dtype=torch.long, device=gt_visibility.device)
    for t in range(time):
        visible_t = gt_visibility[:, :, t].bool()
        reappeared = visible_t & (occ_run >= int(min_occ_frames))
        if int(recovery_window) > 0:
            recovery_left = torch.where(reappeared, torch.full_like(recovery_left, int(recovery_window)), recovery_left)
        focus_mask[:, :, t] = visible_t & (recovery_left > 0)
        recovery_left = torch.where(visible_t & (recovery_left > 0), recovery_left - 1, recovery_left)
        occ_run = torch.where(visible_t, torch.zeros_like(occ_run), occ_run + 1)
    return focus_mask


def compute_longocc_model_score(metrics: Dict[str, float]) -> float:
    aj20 = float(metrics.get("AJ_longocc20", 0.0) or 0.0)
    aj30 = float(metrics.get("AJ_longocc30", 0.0) or 0.0)
    reapp20 = float(metrics.get("reapp_error_longocc20_px", 0.0) or 0.0)
    reapp30 = float(metrics.get("reapp_error_longocc30_px", 0.0) or 0.0)
    return aj20 + aj30 - 0.001 * (reapp20 + reapp30)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MMP-Tracker")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 42))
    set_seed(seed)

    exp_name = cfg.get("experiment", {}).get("name", "mmp_experiment")
    output_dir = Path(args.output or f"outputs/{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config_resolved.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MMPTracker(config_from_dict(cfg)).to(device)
    criterion = loss_from_dict(cfg)
    train_dataset, train_limit = resolve_dataset(cfg, "train", train=True)
    val_dataset, val_limit = resolve_dataset(cfg, "val", train=False)

    training_cfg = cfg.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 1))
    epochs = int(training_cfg.get("epochs", 1))
    lr = float(training_cfg.get("lr", 1.0e-4))
    accumulation_steps = int(training_cfg.get("accumulation_steps", 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_loader = make_loader(train_dataset, batch_size=batch_size, train=True)
    val_loader = make_loader(val_dataset, batch_size=1, train=False)

    best_aj = -1.0
    best_longocc = float("-inf")
    history_path = output_dir / "epoch_metrics.jsonl"
    logger.info("Starting MMP training: %s", exp_name)
    logger.info("Output dir: %s", output_dir)
    logger.info("Seed: %d", seed)
    logger.info("Train limit=%s, Val limit=%s", train_limit, val_limit)

    for epoch in range(epochs):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            limit=train_limit,
            accumulation_steps=accumulation_steps,
            long_occ_min_frames=model.config.tracking.long_occ_min_frames,
            long_occ_recovery_window=model.config.tracking.long_occ_recovery_window,
        )
        val_metrics = evaluate(model, val_loader, device, limit=val_limit)
        record = {"epoch": epoch, **train_stats, **val_metrics}
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Epoch %d | loss=%.4f | AJ=%.4f | OA=%.4f | avg_error_px=%.4f", epoch, train_stats["loss"], val_metrics.get("AJ", 0.0), val_metrics.get("OA", 0.0), val_metrics.get("avg_error_px", 0.0))
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": record}, output_dir / "latest.pth")
        if float(val_metrics.get("AJ", 0.0)) > best_aj:
            best_aj = float(val_metrics.get("AJ", 0.0))
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": record}, output_dir / "best.pth")
        longocc_score = compute_longocc_model_score(val_metrics)
        if longocc_score > best_longocc:
            best_longocc = longocc_score
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "metrics": record, "longocc_score": longocc_score},
                output_dir / "best_longocc.pth",
            )

    logger.info("Training complete. Best AJ=%.4f | Best longocc score=%.4f", best_aj, best_longocc)


if __name__ == "__main__":
    main()
