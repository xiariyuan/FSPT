#!/usr/bin/env python3
"""
FSPT 评估脚本

评估模型在TAP-Vid基准上的性能

使用方法:
    # 评估单个数据集
    python evaluate.py --checkpoint checkpoints/fspt_base/best.pth --dataset davis
    
    # 评估所有数据集
    python evaluate.py --checkpoint checkpoints/fspt_base/best.pth --dataset all
    
    # 生成可视化
    python evaluate.py --checkpoint checkpoints/fspt_base/best.pth --dataset davis --visualize
"""

import os
import sys
import argparse
import logging
import json
import random
import csv
from pathlib import Path
from datetime import datetime
import re
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from tqdm import tqdm

from omegaconf import OmegaConf

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    rank_env = os.environ.get("RANK")
    if rank_env is not None:
        try:
            return int(rank_env) == 0
        except ValueError:
            return True
    return True


def _safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", str(name))


def _extract_video_name(batch, fallback: str) -> str:
    value = batch.get('video_name', fallback)
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else str(fallback)
    return str(value)


def _resolve_resolution_from_batch(batch, index: int, fallback: Tuple[int, int]) -> Tuple[int, int]:
    if not isinstance(batch, dict):
        return fallback
    original_size = batch.get('original_size', None)
    if original_size is None:
        return fallback
    try:
        size = None
        if isinstance(original_size, torch.Tensor):
            if original_size.ndim == 2 and original_size.shape[0] > index:
                size = original_size[index].tolist()
            else:
                size = original_size.tolist()
        elif isinstance(original_size, (list, tuple)):
            # DataLoader collate can turn per-sample `(H, W)` tuples into
            # `(tensor([H...]), tensor([W...]))`. Handle that explicitly.
            if (
                len(original_size) == 2
                and isinstance(original_size[0], torch.Tensor)
                and isinstance(original_size[1], torch.Tensor)
            ):
                h_tensor, w_tensor = original_size[0], original_size[1]
                try:
                    h_val = (
                        h_tensor[index].item()
                        if h_tensor.numel() > index
                        else h_tensor.reshape(-1)[0].item()
                    )
                    w_val = (
                        w_tensor[index].item()
                        if w_tensor.numel() > index
                        else w_tensor.reshape(-1)[0].item()
                    )
                    return (int(h_val), int(w_val))
                except Exception:
                    pass
            if len(original_size) > 0 and isinstance(original_size[0], (list, tuple, torch.Tensor)):
                size = original_size[index] if index < len(original_size) else original_size[0]
                if hasattr(size, 'tolist'):
                    size = size.tolist()
            else:
                size = original_size
        else:
            size = original_size
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            return (int(size[0]), int(size[1]))
    except Exception:
        return fallback
    return fallback


def _max_reappearance_occlusion_run(occluded_nt: torch.Tensor, query_t: torch.Tensor) -> torch.Tensor:
    """
    For each query, compute the longest *continuous* occlusion run AFTER the
    query frame that is followed by at least one visible frame (reappearance).

    Args:
        occluded_nt: (N,T) bool (True=occluded)
        query_t: (N,) long indices in [0, T-1]
    Returns:
        max_run_len: (N,) long
    """
    if occluded_nt.ndim != 2:
        raise ValueError(f"Expected occluded shape (N,T), got {tuple(occluded_nt.shape)}")
    if query_t.ndim != 1:
        raise ValueError(f"Expected query_t shape (N,), got {tuple(query_t.shape)}")
    n_queries, num_frames = occluded_nt.shape
    out = torch.zeros((n_queries,), dtype=torch.long, device=occluded_nt.device)
    for q in range(n_queries):
        start = int(query_t[q].item()) + 1
        if start >= num_frames:
            continue
        mask = occluded_nt[q, start:]
        if mask.numel() == 0:
            continue
        best = 0
        run = 0
        for val in mask.tolist():
            if val:
                run += 1
                continue
            # Run ended and we are visible now => reappearance exists.
            if run > best:
                best = run
            run = 0
        # If `run > 0` at the end, it means occluded until end-of-video (no reappearance).
        out[q] = best
    return out


def _compute_verifier_metrics(
    *,
    candidate_tracks: Optional[torch.Tensor],
    final_tracks: Optional[torch.Tensor],
    base_tracks: Optional[torch.Tensor],
    gt_tracks: torch.Tensor,
    gt_visibility: torch.Tensor,
    verifier_scores: Optional[torch.Tensor],
    verifier_mask: Optional[torch.Tensor],
    verifier_decisions: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
    margin: float = 0.0,
    base_error_threshold: float = 0.0,
) -> Dict[str, float]:
    if (
        not isinstance(candidate_tracks, torch.Tensor)
        or not isinstance(final_tracks, torch.Tensor)
        or not isinstance(base_tracks, torch.Tensor)
        or not isinstance(gt_tracks, torch.Tensor)
        or not isinstance(gt_visibility, torch.Tensor)
        or not isinstance(verifier_scores, torch.Tensor)
        or not isinstance(verifier_mask, torch.Tensor)
    ):
        return {}
    if (
        candidate_tracks.shape != gt_tracks.shape
        or final_tracks.shape != gt_tracks.shape
        or base_tracks.shape != gt_tracks.shape
        or verifier_scores.shape != gt_tracks[..., 0].shape
        or verifier_mask.shape != gt_tracks[..., 0].shape
    ):
        return {}

    err_candidate = torch.norm(candidate_tracks - gt_tracks.to(candidate_tracks.device), dim=-1)
    err_final = torch.norm(final_tracks - gt_tracks.to(final_tracks.device), dim=-1)
    err_base = torch.norm(base_tracks - gt_tracks.to(base_tracks.device), dim=-1)
    valid = gt_visibility.bool().to(device=err_base.device) & verifier_mask.bool().to(device=err_base.device)
    if base_error_threshold > 0:
        valid = valid & (err_base > float(base_error_threshold))
    valid_count = int(valid.long().sum().item())
    if valid_count <= 0:
        return {}

    target = (err_candidate + float(margin) < err_base) & valid
    if isinstance(verifier_decisions, torch.Tensor) and verifier_decisions.shape == valid.shape:
        pred_pos = verifier_decisions.bool().to(device=valid.device) & valid
    else:
        pred_pos = (verifier_scores.to(device=valid.device) >= float(threshold)) & valid

    tp = int((pred_pos & target).long().sum().item())
    tn = int(((~pred_pos) & (~target) & valid).long().sum().item())
    pred_count = int(pred_pos.long().sum().item())
    target_count = int(target.long().sum().item())

    oracle_improvement = (err_base - err_candidate).clamp_min(0.0)
    selected_improvement = (err_base - err_final).clamp_min(0.0)
    oracle_gap = float(oracle_improvement[valid].sum().item())
    selected_gain = float(selected_improvement[valid].sum().item())

    valid_f = valid.float()
    out = {
        "verifier_precision": float(tp / max(pred_count, 1)),
        "verifier_recall": float(tp / max(target_count, 1)),
        "verifier_coverage": float(pred_count / max(valid_count, 1)),
        "verifier_target_rate": float(target_count / max(valid_count, 1)),
        "verifier_accuracy": float((tp + tn) / max(valid_count, 1)),
        "verifier_harm_rate": float(((err_final > err_base) & valid).float().sum().item() / max(valid_count, 1)),
        "verifier_gain_rate": float(((err_final + float(margin) < err_base) & valid).float().sum().item() / max(valid_count, 1)),
        "verifier_oracle_gap_closed": float(selected_gain / max(oracle_gap, 1.0e-6)),
        "verifier_mean_score": float((verifier_scores.to(device=valid.device).float() * valid_f).sum().item() / max(valid_count, 1)),
        "verifier_num_points": float(valid_count),
    }
    return out


def _compute_relocalization_stats(
    *,
    relocal_mask: Optional[torch.Tensor],
    relocal_conf: Optional[torch.Tensor],
    gt_visibility: torch.Tensor,
) -> Dict[str, float]:
    if (
        not isinstance(relocal_mask, torch.Tensor)
        or not isinstance(gt_visibility, torch.Tensor)
        or relocal_mask.shape != gt_visibility.shape
    ):
        return {}

    mask = relocal_mask.bool().to(device=gt_visibility.device)
    valid = gt_visibility.bool().to(device=mask.device)
    active = mask & valid
    valid_count = int(valid.long().sum().item())
    active_count = int(active.long().sum().item())

    out: Dict[str, float] = {
        "relocal_mask_rate": float(mask.float().mean().item()),
        "relocal_mask_valid_rate": float(active_count / max(valid_count, 1)),
        "relocal_mask_num_points": float(active_count),
        "relocal_mask_valid_points": float(valid_count),
    }

    if isinstance(relocal_conf, torch.Tensor) and relocal_conf.shape == relocal_mask.shape:
        conf = relocal_conf.to(device=mask.device, dtype=torch.float32)
        if active_count > 0:
            active_conf = conf[active]
            out["relocal_conf_mean"] = float(active_conf.mean().item())
            out["relocal_conf_std"] = (
                float(active_conf.std(unbiased=False).item()) if active_conf.numel() > 1 else 0.0
            )
        else:
            out["relocal_conf_mean"] = 0.0
            out["relocal_conf_std"] = 0.0
    return out


def parse_args():
    parser = argparse.ArgumentParser(description='FSPT Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (optional, uses checkpoint config if not provided)')
    parser.add_argument('--dataset', type=str, default='davis',
                        choices=['davis', 'kinetics', 'all'],
                        help='Dataset to evaluate on')
    parser.add_argument('--output-dir', type=str, default='outputs/eval',
                        help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization')
    parser.add_argument('--compare-base', action='store_true',
                        help='If model returns base_tracks/base_visibility (e.g., cotracker_refiner), '
                             'also report metrics for the base tracker.')
    parser.add_argument('--base-tracks-dir', type=str, default=None,
                        help='Optional directory of precomputed base tracks (see scripts/precompute_base_tracks.py). '
                             'If provided, the dataloader will inject base_tracks/base_visibility into batches.')
    parser.add_argument(
        '--base-tracks-strict',
        action='store_true',
        help='If set, missing/mismatched base tracks cache files will raise an error '
             '(recommended for paper numbers / debugging).',
    )
    parser.add_argument(
        '--base-tracks-query-tol',
        type=float,
        default=1e-4,
        help='Max abs diff tolerance for cached query_points matching (default: 1e-4).',
    )
    parser.add_argument('--num-vis', type=int, default=10,
                        help='Number of videos to visualize')
    parser.add_argument('--resolution', type=int, nargs=2, default=None,
                        help='Input resolution (H, W)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of dataloader workers')
    parser.add_argument('--include-query-frame', action='store_true',
                        help='Include query frame when computing metrics')
    parser.add_argument(
        '--query-mode',
        type=str,
        default=None,
        choices=['first', 'strided'],
        help="TAP-Vid query protocol override. If unset, follow config.evaluation.query_mode.",
    )
    parser.add_argument(
        '--metric-resolution-mode',
        type=str,
        default=None,
        choices=['original', 'input'],
        help="Metric pixel-space resolution mode: 'original' uses batch.original_size (official/SOTA), "
             "'input' uses model input resolution (training-progress monitoring).",
    )
    parser.add_argument(
        '--save-csv',
        type=str,
        default=None,
        help='Optional path to save a summary CSV table (refined/base/delta). '
             'If a relative path is given, it is saved under --output-dir.',
    )
    parser.add_argument(
        '--save-md',
        type=str,
        default=None,
        help='Optional path to save a Markdown results table (refined/base/delta). '
             'If a relative path is given, it is saved under --output-dir.',
    )
    parser.add_argument(
        '--save-tex',
        type=str,
        default=None,
        help='Optional path to save a LaTeX table (refined/base/delta). '
             'If a relative path is given, it is saved under --output-dir.',
    )
    parser.add_argument('--use-ema', action='store_true', default=None,
                        help='Use EMA weights if available in checkpoint (default: follow config.training.ema.use_for_eval)')
    return parser.parse_args()


def _format_metric(value):
    try:
        if value is None:
            return ""
        if isinstance(value, (int, float, np.floating)):
            return f"{float(value):.4f}"
        if isinstance(value, dict) and "mean" in value:
            mean = value.get("mean", None)
            if isinstance(mean, (int, float, np.floating)):
                return f"{float(mean):.4f}"
    except Exception:
        pass
    return str(value)


def save_results_csv(results: dict, path: Path) -> None:
    metric_keys = sorted({k for v in results.values() if isinstance(v, dict) for k in v.keys()})
    # Expand per-metric statistics when available.
    columns = ["name"]
    for metric in metric_keys:
        columns.append(f"{metric}_mean")
        columns.append(f"{metric}_std")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for name in sorted(results.keys()):
            metrics = results.get(name)
            if not isinstance(metrics, dict):
                continue
            row = [name]
            for metric in metric_keys:
                value = metrics.get(metric, None)
                if isinstance(value, dict) and "mean" in value:
                    row.append(_format_metric(value.get("mean", None)))
                    row.append(_format_metric(value.get("std", None)))
                else:
                    row.append(_format_metric(value))
                    row.append("")
            writer.writerow(row)


def save_results_markdown(results: dict, path: Path) -> None:
    metric_keys = sorted({k for v in results.values() if isinstance(v, dict) for k in v.keys()})
    headers = ["name", *metric_keys]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for name in sorted(results.keys()):
        metrics = results.get(name)
        if not isinstance(metrics, dict):
            continue
        row = [name]
        for metric in metric_keys:
            value = metrics.get(metric, None)
            if isinstance(value, dict) and "mean" in value:
                mean = value.get("mean", None)
                std = value.get("std", None)
                if isinstance(mean, (int, float, np.floating)):
                    if isinstance(std, (int, float, np.floating)):
                        row.append(f"{float(mean):.4f} +/- {float(std):.4f}")
                    else:
                        row.append(f"{float(mean):.4f}")
                else:
                    row.append(_format_metric(mean))
            else:
                row.append(_format_metric(value))
        lines.append("| " + " | ".join(row) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_results_latex(
    results: dict,
    path: Path,
    main_metrics: Optional[list] = None,
) -> None:
    """
    Save a compact LaTeX table.

    Output format:
      Dataset | Variant(refined/base/delta) | metrics...

    Notes:
      - Uses booktabs commands (\\toprule/\\midrule/\\bottomrule).
        Add `\\usepackage{booktabs}` in your paper preamble.
    """
    if main_metrics is None:
        main_metrics = ["AJ", "<4px", "OA"]

    def _escape_latex(text: object) -> str:
        s = str(text)
        repl = {
            "\\": "\\textbackslash{}",
            "&": "\\&",
            "%": "\\%",
            "$": "\\$",
            "#": "\\#",
            "_": "\\_",
            "{": "\\{",
            "}": "\\}",
            "~": "\\textasciitilde{}",
            "^": "\\textasciicircum{}",
            "<": "\\textless{}",
            ">": "\\textgreater{}",
        }
        return "".join(repl.get(ch, ch) for ch in s)

    def _as_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _fmt(stats: dict, signed: bool = False) -> str:
        if not isinstance(stats, dict):
            return "N/A"
        mean = stats.get("mean", None)
        if mean is None:
            return "N/A"
        mean_f = _as_float(mean, 0.0)
        std_f = _as_float(stats.get("std", 0.0), 0.0)
        sign = "+" if signed and mean_f >= 0 else ""
        return f"{sign}{mean_f:.4f} $\\pm$ {std_f:.4f}"

    dataset_keys = []
    for key in results.keys():
        if str(key).endswith("_base") or str(key).endswith("_delta"):
            continue
        dataset_keys.append(key)

    col_spec = "ll" + ("c" * len(main_metrics))
    lines = []
    lines.append("% Requires: \\usepackage{booktabs}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    header = " & ".join(["Dataset", "Variant", *[_escape_latex(m) for m in main_metrics]]) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for dataset_name in dataset_keys:
        refined = results.get(dataset_name, {}) or {}
        base = results.get(f"{dataset_name}_base", None)
        delta = results.get(f"{dataset_name}_delta", None)

        variants = [("refined", refined, False)]
        if isinstance(base, dict):
            variants.append(("base", base, False))
        if isinstance(delta, dict):
            variants.append(("delta", delta, True))

        for variant_name, metrics, signed in variants:
            cells = [_escape_latex(str(dataset_name).upper()), _escape_latex(variant_name)]
            for metric in main_metrics:
                cells.append(_fmt(metrics.get(metric, {}), signed=signed))
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")

    if lines and lines[-1] == "\\midrule":
        lines.pop()
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _align_state_dict_keys(state_dict, model_state):
    if not state_dict or not model_state:
        return state_dict
    sd_keys = list(state_dict.keys())
    ms_keys = list(model_state.keys())
    if not sd_keys or not ms_keys:
        return state_dict
    has_module = sd_keys[0].startswith('module.')
    model_has_module = ms_keys[0].startswith('module.')
    if has_module and not model_has_module:
        return {k[7:]: v for k, v in state_dict.items()}
    if not has_module and model_has_module:
        return {f'module.{k}': v for k, v in state_dict.items()}
    return state_dict


def load_model(checkpoint_path, config=None, use_ema: Optional[bool] = None):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 加载配置
    if config is None:
        if 'config' not in checkpoint:
            raise ValueError("Checkpoint missing config; please provide --config.")
        config = OmegaConf.create(checkpoint['config'])
    
    # 创建模型
    from train import create_model
    model = create_model(config)
    
    # 加载权重（可选EMA）
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if not isinstance(state_dict, dict) or not all(isinstance(v, torch.Tensor) for v in state_dict.values()):
        raise ValueError("Checkpoint missing model_state_dict and is not a raw state dict.")
    if use_ema is None:
        ema_cfg = getattr(getattr(config, 'training', None), 'ema', None)
        use_ema = bool(ema_cfg is not None and getattr(ema_cfg, 'use_for_eval', False))
        if use_ema:
            logger.info("Using EMA weights by default (config.training.ema.use_for_eval=true).")
    if use_ema:
        ema_state = checkpoint.get('ema_state_dict', None)
        if ema_state is not None and 'ema_model' in ema_state:
            state_dict = ema_state['ema_model']
        else:
            logger.warning("EMA weights not found; falling back to model_state_dict")
    state_dict = _align_state_dict_keys(state_dict, model.state_dict())
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(f"State dict mismatch (missing={len(missing)}, unexpected={len(unexpected)})")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    logger.info(f"  Best metrics: {checkpoint.get('metrics', {})}")
    
    return model, config


def get_dataset(name, root, resolution=None, num_workers: int = 4, **kwargs):
    """获取数据集"""
    from datasets import get_dataloader
    dataloader = get_dataloader(
        name=name,
        root=root,
        batch_size=1,
        resolution=tuple(resolution) if resolution else None,
        num_workers=num_workers,
        **kwargs,
    )
    return dataloader


@torch.no_grad()
def evaluate_dataset(
    model,
    dataloader,
    dataset_name,
    exclude_query_frame: bool = True,
    query_mode: Optional[str] = None,
    is_main_process: bool = True,
    compare_base: bool = False,
    metric_resolution_mode: str = 'original',
    long_occ_thresholds: Optional[Sequence[int]] = None,
):
    """
    在数据集上评估模型
    
    Returns:
        results: 包含所有指标的字典
        per_video_results: 每个视频的结果列表
    """
    from datasets.metrics import compute_tapvid_metrics
    
    all_metrics = []
    all_base_metrics = []
    all_delta_metrics = []
    all_relocal_metrics = []
    per_video_results = []

    long_occ_thresholds = list(long_occ_thresholds or [])
    try:
        long_occ_thresholds = sorted({int(v) for v in long_occ_thresholds if int(v) > 0})
    except Exception:
        long_occ_thresholds = []
    all_longocc_metrics = {thr: [] for thr in long_occ_thresholds}
    all_longocc_base_metrics = {thr: [] for thr in long_occ_thresholds}
    all_longocc_delta_metrics = {thr: [] for thr in long_occ_thresholds}
    all_longocc_relocal_metrics = {thr: [] for thr in long_occ_thresholds}
    all_verifier_metrics = []
    all_longocc_verifier_metrics = {thr: [] for thr in long_occ_thresholds}
    longocc_num_queries = {thr: 0 for thr in long_occ_thresholds}
    longocc_num_videos = {thr: 0 for thr in long_occ_thresholds}
    
    device = next(model.parameters()).device
    if dataloader is None:
        logger.warning(f"No samples found for {dataset_name}; skipping.")
        return {}, [], {}, {}
    try:
        if len(dataloader) == 0:
            logger.warning(f"No samples found for {dataset_name}; skipping.")
            return {}, [], {}, {}
    except TypeError:
        pass

    metric_resolution_mode = str(metric_resolution_mode).strip().lower()
    if metric_resolution_mode not in ('original', 'input'):
        logger.warning(
            f"Unknown metric_resolution_mode={metric_resolution_mode}, fallback to 'original'."
        )
        metric_resolution_mode = 'original'

    iterator = tqdm(dataloader, desc=f'Evaluating {dataset_name}', disable=not is_main_process)
    for batch in iterator:
        try:
            if batch is None or not isinstance(batch, dict):
                logger.warning(f"Empty or invalid batch in {dataset_name}, skipping.")
                continue
            required_keys = ('video', 'query_points', 'target_points', 'occluded')
            missing_keys = [k for k in required_keys if k not in batch]
            if missing_keys:
                logger.warning(f"Batch missing keys {missing_keys} in {dataset_name}, skipping.")
                continue
            video = batch['video'].to(device)
            query_points = batch['query_points'].to(device)
            target_points = batch['target_points'].to(device)
            occluded = batch['occluded'].to(device)
            if occluded.dtype != torch.bool:
                occluded = occluded > 0.5
            if video.ndim == 4:
                video = video.unsqueeze(0)
                if query_points.ndim == 2:
                    query_points = query_points.unsqueeze(0)
                if target_points.ndim == 3:
                    target_points = target_points.unsqueeze(0)
                if occluded.ndim == 2:
                    occluded = occluded.unsqueeze(0)

            # 推理 - 统一处理2或3个返回值
            base_tracks = None
            base_visibility = None
            pre_accept_tracks = None
            verifier_scores = None
            verifier_mask = None
            verifier_decisions = None
            relocal_mask = None
            relocal_conf = None
            meta = (
                {
                    'video_name': batch.get('video_name', None),
                    'base_tracks': batch.get('base_tracks', None),
                    'base_visibility': batch.get('base_visibility', None),
                }
                if isinstance(batch, dict)
                else None
            )
            with torch.no_grad():
                if compare_base:
                    try:
                        outputs = model(video, query_points, meta=meta, return_info=True)
                    except TypeError:
                        try:
                            outputs = model(video, query_points, return_info=True)
                        except TypeError:
                            outputs = model(video, query_points)
                else:
                    try:
                        outputs = model(video, query_points, meta=meta)
                    except TypeError:
                        outputs = model(video, query_points)
            if isinstance(outputs, (list, tuple)):
                pred_tracks, pred_visibility = outputs[0], outputs[1]
                if compare_base and len(outputs) >= 3 and isinstance(outputs[2], dict):
                    info = outputs[2]
                    base_tracks = info.get('base_tracks', None)
                    base_visibility = info.get('base_visibility', None)
                    pre_accept_tracks = info.get('pre_accept_tracks', None)
                    verifier_scores = info.get('verifier_scores', info.get('relocal_acceptor', None))
                    verifier_mask = info.get('verifier_mask', info.get('relocal_accept_mask', None))
                    verifier_decisions = info.get('verifier_decisions', None)
                    relocal_mask = info.get('relocal_mask', info.get('relocalization_mask', None))
                    relocal_conf = info.get('relocal_conf', info.get('relocalization_conf', None))
            else:
                raise ValueError(f"Unexpected model output type: {type(outputs)}")

            if pred_tracks.numel() == 0 or target_points.numel() == 0:
                logger.warning(f"Empty predictions/targets in {dataset_name}, skipping.")
                continue

            batch_size = int(video.shape[0])
            video_names = batch.get('video_name', None)
            for i in range(batch_size):
                input_resolution = tuple(video.shape[-2:])
                if metric_resolution_mode == 'input':
                    resolution = input_resolution
                else:
                    resolution = _resolve_resolution_from_batch(batch, i, input_resolution)
                if torch.isnan(pred_tracks[i]).any() or torch.isinf(pred_tracks[i]).any():
                    logger.warning("NaN/Inf detected in predictions, skipping sample.")
                    continue
                if torch.isnan(target_points[i]).any() or torch.isinf(target_points[i]).any():
                    logger.warning("NaN/Inf detected in targets, skipping sample.")
                    continue
                if isinstance(video_names, (list, tuple)):
                    video_name = str(video_names[i]) if i < len(video_names) else f"unknown_{i}"
                elif isinstance(video_names, torch.Tensor):
                    if video_names.ndim == 0:
                        video_name = str(video_names.item())
                    elif i < len(video_names):
                        video_name = str(video_names[i])
                    else:
                        video_name = f"unknown_{i}"
                elif video_names is not None:
                    video_name = str(video_names)
                else:
                    video_name = f"unknown_{i}"
                vis_pred = pred_visibility[i]
                if vis_pred.dtype != torch.bool:
                    vis_pred = vis_pred > 0.5
                metrics = compute_tapvid_metrics(
                    pred_tracks[i],
                    target_points[i],
                    vis_pred,
                    ~occluded[i],
                    query_points[i],
                    resolution=resolution,
                    exclude_query_frame=exclude_query_frame,
                    query_mode=query_mode,
                )
                all_metrics.append(metrics)
                entry = {
                    'video_name': video_name,
                    'metrics': metrics,
                }
                verifier_threshold = float(
                    getattr(model, "relocal_acceptor_eval_threshold", getattr(model, "verifier_eval_threshold", 0.5))
                )
                verifier_margin = float(
                    getattr(model, "relocal_acceptor_eval_margin", getattr(model, "verifier_eval_margin", 0.0))
                )
                verifier_base_error_threshold = float(
                    getattr(
                        model,
                        "relocal_acceptor_eval_base_error_threshold",
                        getattr(model, "verifier_eval_base_error_threshold", 0.0),
                    )
                )

                if base_tracks is not None and base_visibility is not None:
                    try:
                        base_vis = base_visibility[i]
                        if base_vis.dtype != torch.bool:
                            base_vis = base_vis > 0.5
                        base_metrics = compute_tapvid_metrics(
                            base_tracks[i],
                            target_points[i],
                            base_vis,
                            ~occluded[i],
                            query_points[i],
                            resolution=resolution,
                            exclude_query_frame=exclude_query_frame,
                            query_mode=query_mode,
                        )
                        all_base_metrics.append(base_metrics)
                        entry['base_metrics'] = base_metrics

                        try:
                            delta_metrics = {
                                k: float(metrics.get(k, 0.0)) - float(base_metrics.get(k, 0.0))
                                for k in metrics.keys()
                            }
                            all_delta_metrics.append(delta_metrics)
                            entry['delta_metrics'] = delta_metrics
                        except Exception:
                            pass

                        verifier_metrics = _compute_verifier_metrics(
                            candidate_tracks=pre_accept_tracks[i] if isinstance(pre_accept_tracks, torch.Tensor) else None,
                            final_tracks=pred_tracks[i],
                            base_tracks=base_tracks[i],
                            gt_tracks=target_points[i],
                            gt_visibility=(~occluded[i]),
                            verifier_scores=verifier_scores[i] if isinstance(verifier_scores, torch.Tensor) else None,
                            verifier_mask=verifier_mask[i] if isinstance(verifier_mask, torch.Tensor) else None,
                            verifier_decisions=verifier_decisions[i] if isinstance(verifier_decisions, torch.Tensor) else None,
                            threshold=verifier_threshold,
                            margin=verifier_margin,
                            base_error_threshold=verifier_base_error_threshold,
                        )
                        if verifier_metrics:
                            all_verifier_metrics.append(verifier_metrics)
                            entry['verifier_metrics'] = verifier_metrics
                    except Exception as exc:
                        logger.warning(f"Failed to compute base metrics for {video_name}: {exc}")

                relocal_metrics = _compute_relocalization_stats(
                    relocal_mask=relocal_mask[i] if isinstance(relocal_mask, torch.Tensor) else None,
                    relocal_conf=relocal_conf[i] if isinstance(relocal_conf, torch.Tensor) else None,
                    gt_visibility=(~occluded[i]),
                )
                if relocal_metrics:
                    all_relocal_metrics.append(relocal_metrics)
                    entry['relocalization_metrics'] = relocal_metrics

                # Optional: long-occlusion subset evaluation (>=K occluded frames
                # after the query frame and then re-appears).
                if long_occ_thresholds:
                    try:
                        num_frames = int(pred_tracks[i].shape[1])
                        query_t = (
                            query_points[i, :, 0]
                            .round()
                            .long()
                            .clamp(0, max(0, num_frames - 1))
                        )
                        max_occ_run = _max_reappearance_occlusion_run(occluded[i], query_t)
                        for thr in long_occ_thresholds:
                            sel = max_occ_run >= int(thr)
                            num_sel = int(sel.long().sum().item())
                            if num_sel <= 0:
                                continue
                            longocc_num_queries[thr] += num_sel
                            longocc_num_videos[thr] += 1

                            subset_metrics = compute_tapvid_metrics(
                                pred_tracks[i][sel],
                                target_points[i][sel],
                                vis_pred[sel],
                                ~occluded[i][sel],
                                query_points[i][sel],
                                resolution=resolution,
                                exclude_query_frame=exclude_query_frame,
                                query_mode=query_mode,
                            )
                            all_longocc_metrics[thr].append(subset_metrics)

                            if base_tracks is not None and base_visibility is not None:
                                base_vis_subset = base_visibility[i]
                                if base_vis_subset.dtype != torch.bool:
                                    base_vis_subset = base_vis_subset > 0.5
                                base_subset_metrics = compute_tapvid_metrics(
                                    base_tracks[i][sel],
                                    target_points[i][sel],
                                    base_vis_subset[sel],
                                    ~occluded[i][sel],
                                    query_points[i][sel],
                                    resolution=resolution,
                                    exclude_query_frame=exclude_query_frame,
                                    query_mode=query_mode,
                                )
                                all_longocc_base_metrics[thr].append(base_subset_metrics)
                                try:
                                    delta_subset_metrics = {
                                        k: float(subset_metrics.get(k, 0.0))
                                        - float(base_subset_metrics.get(k, 0.0))
                                        for k in subset_metrics.keys()
                                    }
                                    all_longocc_delta_metrics[thr].append(delta_subset_metrics)
                                except Exception:
                                    pass
                                verifier_subset_metrics = _compute_verifier_metrics(
                                    candidate_tracks=pre_accept_tracks[i][sel] if isinstance(pre_accept_tracks, torch.Tensor) else None,
                                    final_tracks=pred_tracks[i][sel],
                                    base_tracks=base_tracks[i][sel],
                                    gt_tracks=target_points[i][sel],
                                    gt_visibility=(~occluded[i][sel]),
                                    verifier_scores=verifier_scores[i][sel] if isinstance(verifier_scores, torch.Tensor) else None,
                                    verifier_mask=verifier_mask[i][sel] if isinstance(verifier_mask, torch.Tensor) else None,
                                    verifier_decisions=verifier_decisions[i][sel] if isinstance(verifier_decisions, torch.Tensor) else None,
                                    threshold=verifier_threshold,
                                    margin=verifier_margin,
                                    base_error_threshold=verifier_base_error_threshold,
                                )
                                if verifier_subset_metrics:
                                    all_longocc_verifier_metrics[thr].append(verifier_subset_metrics)
                                relocal_subset_metrics = _compute_relocalization_stats(
                                    relocal_mask=relocal_mask[i][sel] if isinstance(relocal_mask, torch.Tensor) else None,
                                    relocal_conf=relocal_conf[i][sel] if isinstance(relocal_conf, torch.Tensor) else None,
                                    gt_visibility=(~occluded[i][sel]),
                                )
                                if relocal_subset_metrics:
                                    all_longocc_relocal_metrics[thr].append(relocal_subset_metrics)
                    except Exception as exc:
                        logger.warning(f"Long-occlusion subset evaluation failed for {video_name}: {exc}")

                per_video_results.append(entry)
        except Exception as e:
            logger.warning(f"Error processing sample in {dataset_name}: {e}")
            continue
    
    # 计算平均指标
    if len(all_metrics) == 0:
        logger.warning(f"No valid samples found for {dataset_name}; skipping.")
        return {}, [], {}, {}
    def _safe_float(value):
        try:
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    return None
                value = value.detach().cpu().item()
            return float(value)
        except Exception:
            return None

    def _summarize(metrics_list):
        if not metrics_list:
            return {}
        summary = {}
        for key in metrics_list[0].keys():
            values = []
            for m in metrics_list:
                v = _safe_float(m.get(key))
                if v is None or not np.isfinite(v):
                    continue
                values.append(v)
            if values:
                summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }
            else:
                summary[key] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                }
        return summary

    avg_metrics = _summarize(all_metrics)
    relocal_avg_metrics = _summarize(all_relocal_metrics)
    for key, stats in relocal_avg_metrics.items():
        avg_metrics[key] = stats
    verifier_avg_metrics = _summarize(all_verifier_metrics)
    for key, stats in verifier_avg_metrics.items():
        avg_metrics[key] = stats

    base_avg_metrics = _summarize(all_base_metrics) if compare_base else {}

    delta_avg_metrics = _summarize(all_delta_metrics) if compare_base else {}

    if long_occ_thresholds:
        for thr in long_occ_thresholds:
            suffix = f"_longocc{int(thr)}"

            num_q = float(longocc_num_queries.get(thr, 0))
            num_v = float(longocc_num_videos.get(thr, 0))
            avg_metrics[f"num_queries{suffix}"] = {'mean': num_q, 'std': 0.0, 'min': num_q, 'max': num_q}
            avg_metrics[f"num_videos{suffix}"] = {'mean': num_v, 'std': 0.0, 'min': num_v, 'max': num_v}

            subset_summary = _summarize(all_longocc_metrics.get(thr, []))
            for key, stats in subset_summary.items():
                avg_metrics[f"{key}{suffix}"] = stats

            if compare_base:
                base_subset_summary = _summarize(all_longocc_base_metrics.get(thr, []))
                for key, stats in base_subset_summary.items():
                    base_avg_metrics[f"{key}{suffix}"] = stats

                delta_subset_summary = _summarize(all_longocc_delta_metrics.get(thr, []))
                for key, stats in delta_subset_summary.items():
                    delta_avg_metrics[f"{key}{suffix}"] = stats

                relocal_subset_summary = _summarize(all_longocc_relocal_metrics.get(thr, []))
                for key, stats in relocal_subset_summary.items():
                    avg_metrics[f"{key}{suffix}"] = stats

                verifier_subset_summary = _summarize(all_longocc_verifier_metrics.get(thr, []))
                for key, stats in verifier_subset_summary.items():
                    avg_metrics[f"{key}{suffix}"] = stats

    return avg_metrics, per_video_results, base_avg_metrics, delta_avg_metrics


def visualize_predictions(
    model,
    dataloader,
    output_dir,
    num_videos=10,
):
    """
    可视化预测结果
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        import matplotlib.colors as mcolors
    except Exception as exc:
        logger.warning(f"Visualization skipped (matplotlib unavailable): {exc}")
        return
    
    if dataloader is None:
        logger.warning("No dataloader available for visualization.")
        return
    try:
        if len(dataloader) == 0:
            logger.warning("Empty dataloader; skipping visualization.")
            return
    except TypeError:
        pass

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    device = next(model.parameters()).device
    for i, batch in enumerate(tqdm(dataloader, desc='Visualizing')):
        if i >= num_videos:
            break
        
        video = batch['video'].to(device)
        query_points = batch['query_points'].to(device)
        target_points = batch['target_points'].to(device)
        occluded = batch['occluded'].to(device)
        if occluded.dtype != torch.bool:
            occluded = occluded > 0.5
        if video.ndim == 4:
            video = video.unsqueeze(0)
            if query_points.ndim == 2:
                query_points = query_points.unsqueeze(0)
            if target_points.ndim == 3:
                target_points = target_points.unsqueeze(0)
            if occluded.ndim == 2:
                occluded = occluded.unsqueeze(0)
        video_name = _extract_video_name(batch, f'video_{i}')
        safe_video_name = _safe_name(video_name)
        
        # 推理 - 统一处理2或3个返回值
        with torch.no_grad():
            meta = (
                {
                    'video_name': batch.get('video_name', None),
                    'base_tracks': batch.get('base_tracks', None),
                    'base_visibility': batch.get('base_visibility', None),
                }
                if isinstance(batch, dict)
                else None
            )
            try:
                outputs = model(video, query_points, meta=meta)
            except TypeError:
                outputs = model(video, query_points)
            if isinstance(outputs, (list, tuple)):
                pred_tracks, pred_visibility = outputs[0], outputs[1]
            else:
                raise ValueError(f"Unexpected model output type: {type(outputs)}")
        
        # 转换到CPU和numpy
        video_np = video[0].cpu().permute(0, 2, 3, 1).numpy()  # (T, H, W, 3)
        pred_tracks_np = pred_tracks[0].cpu().numpy()  # (N, T, 2)
        pred_vis_np = (pred_visibility[0] > 0.5).cpu().numpy()  # (N, T)
        gt_tracks_np = target_points[0].cpu().numpy()  # (N, T, 2)
        gt_vis_np = (~occluded[0]).cpu().numpy()  # (N, T)
        query_np = query_points[0].cpu().numpy()  # (N, 3)
        
        T, H, W, _ = video_np.shape
        N = pred_tracks_np.shape[0]
        
        # 创建可视化
        video_dir = output_dir / safe_video_name
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # 选择部分点可视化
        num_points_to_show = min(N, 20)
        point_indices = np.linspace(0, N-1, num_points_to_show, dtype=int)
        
        for t in range(0, T, max(1, T // 20)):  # 采样帧
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # 左图：预测
            axes[0].imshow(video_np[t])
            axes[0].set_title(f'Prediction (Frame {t})')
            
            for j, idx in enumerate(point_indices):
                color = colors[j % len(colors)]
                
                if pred_vis_np[idx, t]:
                    y, x = pred_tracks_np[idx, t]
                    y, x = y * H, x * W
                    circle = Circle((x, y), 5, color=color, fill=True, alpha=0.8)
                    axes[0].add_patch(circle)
                    
                    # 画轨迹
                    for t2 in range(max(0, t-5), t):
                        if pred_vis_np[idx, t2]:
                            y1, x1 = pred_tracks_np[idx, t2]
                            y1, x1 = y1 * H, x1 * W
                            y2, x2 = pred_tracks_np[idx, min(t2+1, t)]
                            y2, x2 = y2 * H, x2 * W
                            axes[0].plot([x1, x2], [y1, y2], color=color, alpha=0.5)
            
            # 右图：Ground Truth
            axes[1].imshow(video_np[t])
            axes[1].set_title(f'Ground Truth (Frame {t})')
            
            for j, idx in enumerate(point_indices):
                color = colors[j % len(colors)]
                
                if gt_vis_np[idx, t]:
                    y, x = gt_tracks_np[idx, t]
                    y, x = y * H, x * W
                    circle = Circle((x, y), 5, color=color, fill=True, alpha=0.8)
                    axes[1].add_patch(circle)
            
            axes[0].axis('off')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig(video_dir / f'frame_{t:04d}.png', dpi=100, bbox_inches='tight')
            plt.close()
        
        # 创建GIF (需要imageio)
        try:
            import imageio
            frames = []
            for t in range(0, T, max(1, T // 20)):
                frame_path = video_dir / f'frame_{t:04d}.png'
                if frame_path.exists():
                    frames.append(imageio.imread(frame_path))
            
            if frames:
                imageio.mimsave(video_dir / 'animation.gif', frames, fps=5)
                logger.info(f"Saved animation to {video_dir / 'animation.gif'}")
        except ImportError:
            pass
    
    logger.info(f"Visualizations saved to {output_dir}")


def print_results_table(results):
    """打印结果表格"""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    for dataset_name, metrics in results.items():
        print(f"\n{dataset_name.upper()}")
        print("-" * 40)
        
        # 主要指标
        main_metrics = ['AJ', '<4px', 'OA']
        for metric in main_metrics:
            if metric in metrics:
                m = metrics[metric]
                print(f"  {metric:10s}: {m['mean']:.4f} ± {m['std']:.4f}")
        
        # 其他指标
        print("\n  All thresholds:")
        for metric, m in metrics.items():
            if metric not in main_metrics:
                print(f"    {metric:10s}: {m['mean']:.4f}")
    
    print("\n" + "=" * 80)


def print_results_table_compared(results):
    """
    Print evaluation results in a single table per dataset.

    When --compare-base is enabled, results contains:
      {dataset}: refined metrics
      {dataset}_base: base metrics
      {dataset}_delta: refined - base
    """
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    def _as_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _fmt(stats: dict, signed: bool = False) -> str:
        if not isinstance(stats, dict):
            return "N/A"
        mean = stats.get("mean", None)
        if mean is None:
            return "N/A"
        mean_f = _as_float(mean, 0.0)
        std_f = _as_float(stats.get("std", 0.0), 0.0)
        if signed:
            return f"{mean_f:+.4f} +/- {std_f:.4f}"
        return f"{mean_f:.4f} +/- {std_f:.4f}"

    dataset_keys = []
    for key in results.keys():
        if str(key).endswith("_base") or str(key).endswith("_delta"):
            continue
        dataset_keys.append(key)

    for dataset_name in dataset_keys:
        refined = results.get(dataset_name, {}) or {}
        base = results.get(f"{dataset_name}_base", None)
        delta = results.get(f"{dataset_name}_delta", None)

        print(f"\n{str(dataset_name).upper()}")
        print("-" * 40)

        main_metrics = ["AJ", "<4px", "OA"]
        for metric in main_metrics:
            if metric not in refined:
                continue
            ref_s = _fmt(refined.get(metric, {}))
            if base is not None or delta is not None:
                base_s = _fmt(base.get(metric, {})) if isinstance(base, dict) else "N/A"
                delta_s = _fmt(delta.get(metric, {}), signed=True) if isinstance(delta, dict) else "N/A"
                print(f"  {metric:10s}: {ref_s} | base {base_s} | d {delta_s}")
            else:
                print(f"  {metric:10s}: {ref_s}")

        print("\n  All thresholds:")
        for metric, stats in refined.items():
            if metric in main_metrics:
                continue
            ref_s = _fmt(stats)
            if base is not None or delta is not None:
                base_s = _fmt(base.get(metric, {})) if isinstance(base, dict) else "N/A"
                delta_s = _fmt(delta.get(metric, {}), signed=True) if isinstance(delta, dict) else "N/A"
                print(f"    {metric:10s}: {ref_s} | base {base_s} | d {delta_s}")
            else:
                print(f"    {metric:10s}: {ref_s}")

    print("\n" + "=" * 80)


def main():
    args = parse_args()
    is_main_process = _is_main_process()
    if not is_main_process:
        logger.info("Non-main process detected; skipping evaluation.")
        return
    
    # 创建输出目录
    project_root = Path(__file__).parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint and not Path(args.checkpoint).is_absolute():
        args.checkpoint = str(project_root / args.checkpoint)
    if args.config and not Path(args.config).is_absolute():
        args.config = str(project_root / args.config)
    if args.checkpoint and not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return
    if args.config and not Path(args.config).exists():
        logger.error(f"Config not found: {args.config}")
        return
    
    # 加载配置
    config = None
    if args.config:
        from train import load_config
        config = load_config(args.config)
    
    # 加载模型
    model, config = load_model(args.checkpoint, config, use_ema=args.use_ema)

    # 设置随机种子，保证评估可复现
    seed = 42
    if config is not None and hasattr(config, 'experiment') and hasattr(config.experiment, 'seed'):
        try:
            seed = int(config.experiment.seed)
        except Exception:
            seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 确定要评估的数据集
    if args.dataset == 'all':
        if config is not None and hasattr(config, 'evaluation') and hasattr(config.evaluation, 'datasets'):
            from datasets import normalize_dataset_name
            datasets = [normalize_dataset_name(d) for d in config.evaluation.datasets]
        else:
            datasets = ['davis', 'kinetics']
    else:
        datasets = [args.dataset]

    # 确定num_workers
    if args.num_workers is not None:
        num_workers = args.num_workers
    elif config is not None and hasattr(config, 'evaluation') and hasattr(config.evaluation, 'num_workers'):
        num_workers = int(config.evaluation.num_workers)
    elif config is not None and hasattr(config, 'training') and hasattr(config.training, 'num_workers'):
        num_workers = int(config.training.num_workers)
    else:
        num_workers = 4
    
    # 评估
    results = {}
    
    from datasets import normalize_dataset_name
    for dataset_name in datasets:
        logger.info(f"\nEvaluating on {dataset_name}...")
        dataset_norm = normalize_dataset_name(dataset_name)
        # 获取数据集路径
        root = None
        selected_cfg = None
        resolution = args.resolution
        extra_args = {}
        if hasattr(config, 'data'):
            val_cfg = getattr(config.data, 'val', None)
            if val_cfg is not None and normalize_dataset_name(val_cfg.dataset) == dataset_norm:
                selected_cfg = val_cfg
                root = val_cfg.root
                if resolution is None and hasattr(val_cfg, 'resolution'):
                    resolution = val_cfg.resolution
                if hasattr(val_cfg, 'augmentation'):
                    extra_args['augmentation'] = val_cfg.augmentation
                    val_aug = val_cfg.augmentation
                    val_aug_enabled = False
                    if isinstance(val_aug, bool):
                        val_aug_enabled = val_aug
                    elif OmegaConf.is_config(val_aug):
                        val_aug_enabled = bool(getattr(val_aug, 'enabled', True))
                    elif isinstance(val_aug, dict):
                        val_aug_enabled = bool(val_aug.get('enabled', True))
                    else:
                        val_aug_enabled = True
                    if val_aug_enabled:
                        logger.warning("Validation augmentation is enabled; metrics may be distorted.")
                if hasattr(val_cfg, 'num_points'):
                    extra_args['num_points'] = val_cfg.num_points
                if hasattr(val_cfg, 'max_samples'):
                    extra_args['max_samples'] = val_cfg.max_samples
                if dataset_norm == 'kinetics':
                    if hasattr(val_cfg, 'max_frames'):
                        extra_args['max_frames'] = val_cfg.max_frames
                    elif hasattr(val_cfg, 'num_frames') and val_cfg.num_frames is not None:
                        if int(val_cfg.num_frames) > 0:
                            extra_args['max_frames'] = int(val_cfg.num_frames)
            elif hasattr(config.data, dataset_name):
                cfg = getattr(config.data, dataset_name)
                selected_cfg = cfg
                root = cfg.root
                if resolution is None and hasattr(cfg, 'resolution'):
                    resolution = cfg.resolution
                if hasattr(cfg, 'augmentation'):
                    extra_args['augmentation'] = cfg.augmentation
                    val_aug = cfg.augmentation
                    val_aug_enabled = False
                    if isinstance(val_aug, bool):
                        val_aug_enabled = val_aug
                    elif OmegaConf.is_config(val_aug):
                        val_aug_enabled = bool(getattr(val_aug, 'enabled', True))
                    elif isinstance(val_aug, dict):
                        val_aug_enabled = bool(val_aug.get('enabled', True))
                    else:
                        val_aug_enabled = True
                    if val_aug_enabled:
                        logger.warning("Validation augmentation is enabled; metrics may be distorted.")
                if hasattr(cfg, 'num_points'):
                    extra_args['num_points'] = cfg.num_points
                if hasattr(cfg, 'max_samples'):
                    extra_args['max_samples'] = cfg.max_samples
                if dataset_norm == 'kinetics':
                    if hasattr(cfg, 'max_frames'):
                        extra_args['max_frames'] = cfg.max_frames
                    elif hasattr(cfg, 'num_frames') and cfg.num_frames is not None:
                        if int(cfg.num_frames) > 0:
                            extra_args['max_frames'] = int(cfg.num_frames)
        if root is None:
            root = str(project_root / 'datasets' / f'tapvid_{dataset_name}')
        root_path = Path(root)
        if not root_path.is_absolute():
            root_path = project_root / root_path
        root = str(root_path)
        if not root_path.exists():
            logger.warning(f"Dataset root does not exist: {root_path}")
            continue

        if dataset_norm in ('davis', 'kinetics'):
            if selected_cfg is not None:
                if hasattr(selected_cfg, 'query_mode') and getattr(selected_cfg, 'query_mode') is not None:
                    mode = str(selected_cfg.query_mode).strip()
                    if mode.lower() not in ('', 'none', 'null'):
                        extra_args['query_mode'] = mode
                if hasattr(selected_cfg, 'query_stride') and getattr(selected_cfg, 'query_stride') is not None:
                    try:
                        extra_args['query_stride'] = max(1, int(selected_cfg.query_stride))
                    except Exception:
                        pass
                if hasattr(selected_cfg, 'points_order') and getattr(selected_cfg, 'points_order') is not None:
                    extra_args['points_order'] = str(selected_cfg.points_order)

            if 'query_mode' not in extra_args and config is not None:
                if hasattr(config, 'evaluation') and hasattr(config.evaluation, 'query_mode'):
                    mode = getattr(config.evaluation, 'query_mode')
                    if mode is not None and str(mode).strip().lower() not in ('', 'none', 'null'):
                        extra_args['query_mode'] = str(mode).strip()
        
        extra_args['seed'] = seed
        if args.base_tracks_dir is not None:
            extra_args['base_tracks_dir'] = args.base_tracks_dir
            extra_args['base_tracks_strict'] = bool(args.base_tracks_strict)
            extra_args['base_tracks_query_tol'] = float(args.base_tracks_query_tol)
        try:
            dataloader = get_dataset(
                dataset_name,
                root,
                resolution=resolution,
                num_workers=num_workers,
                **extra_args,
            )
        except Exception as e:
            logger.warning(f"Could not load {dataset_name}: {e}")
            continue
        
        # 评估
        exclude_query_frame = True
        if args.include_query_frame:
            exclude_query_frame = False
        elif config is not None and hasattr(config, 'evaluation') and hasattr(config.evaluation, 'exclude_query_frame'):
            exclude_query_frame = bool(config.evaluation.exclude_query_frame)
        query_mode = None
        if args.query_mode is not None:
            query_mode = str(args.query_mode).strip()
        elif config is not None and hasattr(config, 'evaluation') and hasattr(config.evaluation, 'query_mode'):
            qm = getattr(config.evaluation, 'query_mode')
            if qm is not None and str(qm).strip().lower() not in ('', 'none', 'null'):
                query_mode = str(qm).strip()

        metric_resolution_mode = 'original'
        if args.metric_resolution_mode is not None:
            metric_resolution_mode = str(args.metric_resolution_mode).strip().lower()
        elif config is not None and hasattr(config, 'evaluation') and hasattr(config.evaluation, 'metric_resolution_mode'):
            mrm = getattr(config.evaluation, 'metric_resolution_mode')
            if mrm is not None and str(mrm).strip().lower() in ('original', 'input'):
                metric_resolution_mode = str(mrm).strip().lower()
        if args.include_query_frame and query_mode is not None:
            logger.warning(
                "--include-query-frame is set; disabling query_mode and using legacy include-query behavior."
            )
            query_mode = None

        long_occ_thresholds = []
        if config is not None and hasattr(config, 'evaluation') and hasattr(config.evaluation, 'long_occlusion_subset'):
            long_occ_cfg = config.evaluation.long_occlusion_subset
            enabled = False
            try:
                enabled = bool(getattr(long_occ_cfg, 'enabled', False))
            except Exception:
                enabled = False
            if enabled:
                raw_thresholds = getattr(long_occ_cfg, 'thresholds', [10, 20, 30])
                try:
                    if isinstance(raw_thresholds, torch.Tensor):
                        raw_thresholds = raw_thresholds.tolist()
                    if raw_thresholds is None:
                        raw_thresholds = [10, 20, 30]
                    if isinstance(raw_thresholds, (int, float, str)):
                        raw_thresholds = [raw_thresholds]
                    thresholds = []
                    for v in list(raw_thresholds):
                        try:
                            thresholds.append(int(v))
                        except Exception:
                            continue
                    long_occ_thresholds = sorted({t for t in thresholds if t > 0})
                except Exception:
                    long_occ_thresholds = [10, 20, 30]
        avg_metrics, per_video_results, base_avg_metrics, delta_avg_metrics = evaluate_dataset(
            model,
            dataloader,
            dataset_name,
            exclude_query_frame=exclude_query_frame,
            query_mode=query_mode,
            is_main_process=is_main_process,
            compare_base=bool(args.compare_base),
            metric_resolution_mode=metric_resolution_mode,
            long_occ_thresholds=long_occ_thresholds,
        )
        results[dataset_name] = avg_metrics
        if args.compare_base and base_avg_metrics:
            results[f"{dataset_name}_base"] = base_avg_metrics
        if args.compare_base and delta_avg_metrics:
            results[f"{dataset_name}_delta"] = delta_avg_metrics
        
        # 保存详细结果
        results_path = output_dir / f'{dataset_name}_results.json'

        def _json_default(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        with open(results_path, 'w') as f:
            json.dump({
                'avg_metrics': avg_metrics,
                'base_avg_metrics': base_avg_metrics,
                'delta_avg_metrics': delta_avg_metrics,
                'per_video': per_video_results,
                'checkpoint': args.checkpoint,
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2, default=_json_default)
        logger.info(f"Saved results to {results_path}")
        
        # 可视化
        if args.visualize:
            vis_dir = output_dir / f'{dataset_name}_vis'
            visualize_predictions(model, dataloader, vis_dir, args.num_vis)
    
    # 打印结果表格
    print_results_table_compared(results)
    
    # 保存汇总结果
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'results': results,
            'checkpoint': args.checkpoint,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")

    if args.save_csv:
        csv_path = Path(args.save_csv)
        if not csv_path.is_absolute():
            csv_path = output_dir / csv_path
        save_results_csv(results, csv_path)
        logger.info(f"Saved CSV summary to {csv_path}")

    if args.save_md:
        md_path = Path(args.save_md)
        if not md_path.is_absolute():
            md_path = output_dir / md_path
        save_results_markdown(results, md_path)
        logger.info(f"Saved Markdown table to {md_path}")

    if args.save_tex:
        tex_path = Path(args.save_tex)
        if not tex_path.is_absolute():
            tex_path = output_dir / tex_path
        save_results_latex(results, tex_path)
        logger.info(f"Saved LaTeX table to {tex_path}")
    
    return results


if __name__ == '__main__':
    main()
