#!/usr/bin/env python3
"""
Test-time adaptation for point tracking on long-occlusion subsets.

This script is the first pivot away from post-hoc relocalization:
  - keep the base tracker frozen
  - adapt only lightweight refiner modules per video
  - optimize on unlabeled test videos using two-view self-consistency
  - anchor the update with the frozen base tracker outputs

The primary use case is a targeted long-occlusion subset evaluation, so we can
quickly tell whether test-time adaptation has any real headroom before scaling
to a full TAP-Vid pass.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Make repository imports work when this file is executed directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.metrics import compute_tapvid_metrics
from scripts.eval_long_occlusion_subset import (
    _as_bool,
    _build_val_loader_from_config,
    _ensure_batch_dim,
    _max_reappearance_occlusion_run,
    _mean_metrics,
    _resolve_resolution_from_batch,
    _setup_logging,
)
from train import PointTrackingLoss, _compute_two_view_consistency_loss, create_model, load_config, unwrap_model
from train import _get_amp_dtype
from utils.affine_augmentation import sample_affine_matrices, transform_points_yx, warp_video_affine

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _align_state_dict_keys(state_dict: Dict[str, torch.Tensor], model_state: Dict[str, torch.Tensor]):
    if not state_dict or not model_state:
        return state_dict
    sd_keys = list(state_dict.keys())
    ms_keys = list(model_state.keys())
    if not sd_keys or not ms_keys:
        return state_dict
    has_module = sd_keys[0].startswith("module.")
    model_has_module = ms_keys[0].startswith("module.")
    if has_module and not model_has_module:
        return {k[7:]: v for k, v in state_dict.items()}
    if not has_module and model_has_module:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict


def _load_model_from_checkpoint(
    *,
    config,
    checkpoint_path: Path,
    device: torch.device,
    use_ema: Optional[bool],
):
    """
    Build the model from `config` and load weights from the checkpoint.

    Unlike the eval-only helper, we intentionally keep the config passed in by
    the caller so TTA-specific losses and trainable-module settings are preserved.
    """
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model = create_model(config).to(device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict) or not all(isinstance(v, torch.Tensor) for v in state_dict.values()):
        raise ValueError("Checkpoint missing model_state_dict and is not a raw state dict.")

    if use_ema is None:
        ema_cfg = getattr(getattr(config, "training", None), "ema", None)
        use_ema = bool(ema_cfg is not None and getattr(ema_cfg, "use_for_eval", False))
        if use_ema:
            logger.info("Using EMA weights by default (config.training.ema.use_for_eval=true).")

    if use_ema:
        ema_state = checkpoint.get("ema_state_dict", None)
        if ema_state is not None and "ema_model" in ema_state:
            state_dict = ema_state["ema_model"]
        else:
            logger.warning("EMA weights not found in checkpoint; falling back to model_state_dict.")

    state_dict = _align_state_dict_keys(state_dict, model.state_dict())
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        def _is_base_tracker_key(k: str) -> bool:
            normalized = str(k)
            while normalized.startswith("module."):
                normalized = normalized[len("module.") :]
            return normalized.startswith(("base_tracker.", "base_tracker._predictor."))

        missing_filtered = [k for k in missing if not _is_base_tracker_key(k)]
        unexpected_filtered = [k for k in unexpected if not _is_base_tracker_key(k)]
        if missing_filtered or unexpected_filtered:
            logger.warning(
                "State dict mismatch (filtered): "
                f"missing={len(missing_filtered)} unexpected={len(unexpected_filtered)} "
                f"(raw missing={len(missing)}, raw unexpected={len(unexpected)})."
            )
        else:
            logger.info(
                "Checkpoint intentionally omits base_tracker weights; "
                f"skipping missing={len(missing)} keys under base_tracker.*"
            )

    model.eval()
    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    return unwrap_model(model), checkpoint


def _module_by_name(model: torch.nn.Module, name: str) -> Optional[torch.nn.Module]:
    aliases = {
        "temporal_transformer": "temporal_transformer",
        "relocalization_residual": "relocalization_residual_module",
        "relocalization_residual_module": "relocalization_residual_module",
        "relocal_acceptor": "relocal_acceptor_head",
        "verifier": "relocal_acceptor_head",
        "policy_gate": "policy_gate_head",
        "occlusion_predictor": "occlusion_predictor",
        "residual_decoder": "residual_decoder",
        "geo_backbone": "geo_backbone",
    }
    attr = aliases.get(str(name), str(name))
    module = getattr(model, attr, None)
    if isinstance(module, torch.nn.Module):
        return module
    return None


def _snapshot_modules(model: torch.nn.Module, module_names: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
    snap: Dict[str, Dict[str, torch.Tensor]] = {}
    for name in module_names:
        module = _module_by_name(model, name)
        if module is None:
            continue
        snap[name] = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    return snap


def _restore_modules(model: torch.nn.Module, snapshot: Dict[str, Dict[str, torch.Tensor]]) -> None:
    for name, state in snapshot.items():
        module = _module_by_name(model, name)
        if module is None:
            continue
        module.load_state_dict(state, strict=True)


def _select_adaptation_subset(
    max_occ: torch.Tensor,
    *,
    threshold: int,
    max_points: int,
) -> torch.Tensor:
    """
    Select hard queries for test-time adaptation.

    Prefer the long-occlusion subset. If that is empty, fall back to the hardest
    available queries so the video still contributes to adaptation.
    """
    if max_occ.ndim != 1:
        raise ValueError(f"Expected max_occ shape (N,), got {tuple(max_occ.shape)}")

    sel = max_occ >= int(threshold)
    if sel.any():
        if max_points > 0 and int(sel.long().sum().item()) > max_points:
            hard_scores = max_occ.float().clone()
            hard_scores[~sel] = -1.0
            idx = torch.topk(hard_scores, k=max_points, largest=True).indices
            out = torch.zeros_like(sel, dtype=torch.bool)
            out[idx] = True
            return out
        return sel

    # Fallback: use the hardest available queries.
    k = min(max_points if max_points > 0 else int(max_occ.numel()), int(max_occ.numel()))
    if k <= 0:
        return torch.zeros_like(sel, dtype=torch.bool)
    idx = torch.topk(max_occ.float(), k=k, largest=True).indices
    out = torch.zeros_like(sel, dtype=torch.bool)
    out[idx] = True
    return out


def _resolve_query_mode(config, override: str) -> Optional[str]:
    query_mode = override.strip()
    if not query_mode and hasattr(config, "evaluation") and hasattr(config.evaluation, "query_mode"):
        query_mode = str(getattr(config.evaluation, "query_mode") or "").strip()
    if not query_mode or query_mode.lower() in ("none", "null", ""):
        return None
    return query_mode


@dataclass
class Accumulator:
    threshold: int
    num_videos: int = 0
    num_queries: int = 0
    base_rows: List[Dict[str, float]] = field(default_factory=list)
    refined_rows: List[Dict[str, float]] = field(default_factory=list)
    adapted_rows: List[Dict[str, float]] = field(default_factory=list)
    oracle_refined_rows: List[Dict[str, float]] = field(default_factory=list)
    oracle_adapted_rows: List[Dict[str, float]] = field(default_factory=list)

    def add(
        self,
        *,
        base_metrics: Dict[str, float],
        refined_metrics: Dict[str, float],
        adapted_metrics: Dict[str, float],
        oracle_refined_metrics: Dict[str, float],
        oracle_adapted_metrics: Dict[str, float],
        num_selected: int,
    ) -> None:
        self.num_videos += 1
        self.num_queries += int(num_selected)
        self.base_rows.append(base_metrics)
        self.refined_rows.append(refined_metrics)
        self.adapted_rows.append(adapted_metrics)
        self.oracle_refined_rows.append(oracle_refined_metrics)
        self.oracle_adapted_rows.append(oracle_adapted_metrics)

    def summarize(self) -> Dict[str, Any]:
        base = _mean_metrics(self.base_rows)
        refined = _mean_metrics(self.refined_rows)
        adapted = _mean_metrics(self.adapted_rows)
        oracle_refined = _mean_metrics(self.oracle_refined_rows)
        oracle_adapted = _mean_metrics(self.oracle_adapted_rows)
        delta_refined = {k: refined[k] - base[k] for k in refined.keys() if k in base}
        delta_adapted = {k: adapted[k] - base[k] for k in adapted.keys() if k in base}
        delta_vs_refined = {k: adapted[k] - refined[k] for k in adapted.keys() if k in refined}
        delta_oracle_vs_refined = {
            k: oracle_adapted[k] - oracle_refined[k]
            for k in oracle_adapted.keys()
            if k in oracle_refined
        }
        return {
            "threshold": self.threshold,
            "num_videos": self.num_videos,
            "num_queries": self.num_queries,
            "base": base,
            "refined": refined,
            "adapted": adapted,
            "oracle_refined": oracle_refined,
            "oracle_adapted": oracle_adapted,
            "delta_refined": delta_refined,
            "delta_adapted": delta_adapted,
            "delta_vs_refined": delta_vs_refined,
            "delta_oracle_vs_refined": delta_oracle_vs_refined,
        }


@dataclass
class FullAccumulator:
    num_videos: int = 0
    num_queries: int = 0
    base_rows: List[Dict[str, float]] = field(default_factory=list)
    refined_rows: List[Dict[str, float]] = field(default_factory=list)
    adapted_rows: List[Dict[str, float]] = field(default_factory=list)
    oracle_refined_rows: List[Dict[str, float]] = field(default_factory=list)
    oracle_adapted_rows: List[Dict[str, float]] = field(default_factory=list)

    def add(
        self,
        *,
        base_metrics: Dict[str, float],
        refined_metrics: Dict[str, float],
        adapted_metrics: Dict[str, float],
        oracle_refined_metrics: Dict[str, float],
        oracle_adapted_metrics: Dict[str, float],
        num_queries: int,
    ) -> None:
        self.num_videos += 1
        self.num_queries += int(num_queries)
        self.base_rows.append(base_metrics)
        self.refined_rows.append(refined_metrics)
        self.adapted_rows.append(adapted_metrics)
        self.oracle_refined_rows.append(oracle_refined_metrics)
        self.oracle_adapted_rows.append(oracle_adapted_metrics)

    def summarize(self) -> Dict[str, Any]:
        base = _mean_metrics(self.base_rows)
        refined = _mean_metrics(self.refined_rows)
        adapted = _mean_metrics(self.adapted_rows)
        oracle_refined = _mean_metrics(self.oracle_refined_rows)
        oracle_adapted = _mean_metrics(self.oracle_adapted_rows)
        delta_refined = {k: refined[k] - base[k] for k in refined.keys() if k in base}
        delta_adapted = {k: adapted[k] - base[k] for k in adapted.keys() if k in base}
        delta_vs_refined = {k: adapted[k] - refined[k] for k in adapted.keys() if k in refined}
        delta_oracle_vs_refined = {
            k: oracle_adapted[k] - oracle_refined[k]
            for k in oracle_adapted.keys()
            if k in oracle_refined
        }
        return {
            "num_videos": self.num_videos,
            "num_queries": self.num_queries,
            "base": base,
            "refined": refined,
            "adapted": adapted,
            "oracle_refined": oracle_refined,
            "oracle_adapted": oracle_adapted,
            "delta_refined": delta_refined,
            "delta_adapted": delta_adapted,
            "delta_vs_refined": delta_vs_refined,
            "delta_oracle_vs_refined": delta_oracle_vs_refined,
        }


def _metric_row(metrics: Dict[str, Any], suffix: str = "") -> Dict[str, float]:
    def g(name: str) -> float:
        value = metrics.get(f"{name}{suffix}", None)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        return float(value)

    return {
        "AJ": g("AJ"),
        "OA": g("OA"),
        "<1px": g("<1px"),
        "<2px": g("<2px"),
        "<4px": g("<4px"),
        "<8px": g("<8px"),
        "<16px": g("<16px"),
        "<avg": g("<avg"),
        "avg_error_px": g("avg_error_px"),
        "median_error_px": g("median_error_px"),
    }


def _format(v: Optional[float], width: int = 10, precision: int = 6) -> str:
    if v is None:
        return "-".rjust(width)
    return f"{float(v):{width}.{precision}f}"


def _evaluate_subset(
    *,
    pred_tracks: torch.Tensor,
    pred_visibility: torch.Tensor,
    base_tracks: torch.Tensor,
    base_visibility: torch.Tensor,
    target_points: torch.Tensor,
    occluded: torch.Tensor,
    query_points: torch.Tensor,
    sel: torch.Tensor,
    resolution: Tuple[int, int],
    query_mode: Optional[str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    refined_metrics = compute_tapvid_metrics(
        pred_tracks[sel],
        target_points[sel],
        pred_visibility[sel],
        ~occluded[sel],
        query_points[sel],
        resolution=resolution,
        exclude_query_frame=True,
        query_mode=query_mode,
    )
    base_metrics = compute_tapvid_metrics(
        base_tracks[sel],
        target_points[sel],
        base_visibility[sel],
        ~occluded[sel],
        query_points[sel],
        resolution=resolution,
        exclude_query_frame=True,
        query_mode=query_mode,
    )
    return _metric_row(base_metrics), _metric_row(refined_metrics)


def _evaluate_subset_oracle_visibility(
    *,
    pred_tracks: torch.Tensor,
    target_points: torch.Tensor,
    occluded: torch.Tensor,
    query_points: torch.Tensor,
    sel: torch.Tensor,
    resolution: Tuple[int, int],
    query_mode: Optional[str],
) -> Dict[str, float]:
    gt_visibility = ~occluded[sel]
    metrics = compute_tapvid_metrics(
        pred_tracks[sel],
        target_points[sel],
        gt_visibility,
        gt_visibility,
        query_points[sel],
        resolution=resolution,
        exclude_query_frame=True,
        query_mode=query_mode,
    )
    return _metric_row(metrics)


def _run_forward(
    model: torch.nn.Module,
    video: torch.Tensor,
    query_points: torch.Tensor,
    *,
    base_tracks: Optional[torch.Tensor] = None,
    base_visibility: Optional[torch.Tensor] = None,
    return_info: bool = True,
    no_grad: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    meta = None
    if isinstance(base_tracks, torch.Tensor) and isinstance(base_visibility, torch.Tensor):
        meta = {"base_tracks": base_tracks, "base_visibility": base_visibility}
    grad_ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with grad_ctx:
        try:
            if meta is not None:
                outputs = model(video, query_points, meta=meta, return_info=return_info)
            else:
                outputs = model(video, query_points, return_info=return_info)
        except TypeError:
            if meta is not None:
                outputs = model(video, query_points, meta=meta)
            else:
                outputs = model(video, query_points)

    if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
        raise ValueError(f"Unexpected model outputs: {type(outputs)}")
    pred_tracks = outputs[0]
    pred_visibility = outputs[1]
    info: Dict[str, Any] = {}
    if return_info and len(outputs) >= 3 and isinstance(outputs[2], dict):
        info = outputs[2]
    return pred_tracks, pred_visibility, info


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="Test-time adaptation for long-occlusion point tracking.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="TTA override config. Defaults should point to the strong base checkpoint config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path to adapt from.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="10,20,30",
        help="Comma-separated long-occlusion thresholds for evaluation.",
    )
    parser.add_argument(
        "--adapt-threshold",
        type=int,
        default=None,
        help="Queries with occlusion run length >= this threshold are used for adaptation.",
    )
    parser.add_argument(
        "--query-mode",
        type=str,
        default="",
        help="Override query_mode (first|strided).",
    )
    parser.add_argument(
        "--metric-resolution-mode",
        type=str,
        default="original",
        choices=("original", "input"),
        help="Compute metrics in original resolution (paper) or input resolution (debug).",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Limit evaluation to first N videos (0 = all).",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Force using EMA weights if present in the checkpoint.",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Force not using EMA weights.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional explicit RNG seed for reproducibility.",
    )
    args = parser.parse_args()

    if args.use_ema and args.no_ema:
        raise ValueError("Conflicting flags: --use-ema and --no-ema")
    use_ema: Optional[bool] = None
    if args.use_ema:
        use_ema = True
    if args.no_ema:
        use_ema = False

    if args.seed is not None:
        _set_seed(int(args.seed))

    config = load_config(args.config)
    if args.seed is not None:
        try:
            setattr(getattr(config, "experiment", config), "seed", int(args.seed))
        except Exception:
            pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(
        getattr(getattr(config, "training", None), "amp", None) is not None
        and getattr(getattr(config.training, "amp", None), "enabled", False)
        and device.type == "cuda"
    )
    amp_dtype = _get_amp_dtype(config) if amp_enabled else torch.float16

    model, checkpoint = _load_model_from_checkpoint(
        config=config,
        checkpoint_path=Path(args.checkpoint),
        device=device,
        use_ema=use_ema,
    )
    model.eval()

    tta_cfg = getattr(config, "tta", None)
    if tta_cfg is None:
        raise ValueError("Config missing tta section.")

    trainable_modules = list(getattr(getattr(config.model, "refiner", None), "trainable_modules", None) or [])
    if not trainable_modules:
        trainable_modules = list(getattr(tta_cfg, "modules", None) or ["temporal_transformer", "relocalization_residual"])

    # Freeze everything except the selected refiner modules.
    if hasattr(model, "set_trainable_modules"):
        model.set_trainable_modules(trainable_modules)
    else:
        raise ValueError("Model does not support set_trainable_modules().")
    # Keep the base tracker frozen, but switch the trainable refiner branches into
    # training mode so checkpointing and any train-time memory optimizations can
    # take effect.
    model.train()

    trainable_params = [p for p in model.parameters() if bool(p.requires_grad)]
    if not trainable_params:
        raise ValueError(f"No trainable parameters found for modules: {trainable_modules}")

    tta_steps = int(getattr(tta_cfg, "steps", 1) or 1)
    tta_steps = max(0, tta_steps)
    tta_lr = float(getattr(tta_cfg, "lr", 1.0e-5) or 1.0e-5)
    tta_weight_decay = float(getattr(tta_cfg, "weight_decay", 0.0) or 0.0)
    tta_grad_clip = float(getattr(tta_cfg, "grad_clip", 1.0) or 1.0)
    tta_max_points = int(getattr(tta_cfg, "max_points", 64) or 64)
    tta_max_points = max(1, tta_max_points)
    adapt_threshold_cfg = getattr(tta_cfg, "adapt_threshold", None)
    adapt_threshold = int(args.adapt_threshold if args.adapt_threshold is not None else adapt_threshold_cfg or 20)
    continual = bool(getattr(tta_cfg, "continual", False))

    criterion = PointTrackingLoss(config)
    optimizer = torch.optim.AdamW(trainable_params, lr=tta_lr, weight_decay=tta_weight_decay)
    scaler = GradScaler(enabled=amp_enabled)

    dataloader = _build_val_loader_from_config(config)
    query_mode = _resolve_query_mode(config, args.query_mode)
    thresholds = [int(x) for x in str(args.thresholds).split(",") if str(x).strip()]
    thresholds = sorted(set(int(x) for x in thresholds if int(x) > 0))
    if not thresholds:
        raise ValueError("No valid thresholds provided.")

    accumulators = {thr: Accumulator(threshold=thr) for thr in thresholds}
    full_accumulator = FullAccumulator()

    module_snapshot = _snapshot_modules(model, trainable_modules)

    iterator = tqdm(dataloader, desc="TTA long-occ eval", disable=False)
    for idx, batch in enumerate(iterator):
        if args.max_videos and idx >= int(args.max_videos):
            break
        if batch is None or not isinstance(batch, dict):
            continue
        if "video" not in batch or "query_points" not in batch or "target_points" not in batch or "occluded" not in batch:
            continue

        video = batch["video"].to(device)
        query_points = batch["query_points"].to(device)
        target_points = batch["target_points"].to(device)
        occluded = _as_bool(batch["occluded"].to(device))

        video = _ensure_batch_dim(video, 4)
        query_points = _ensure_batch_dim(query_points, 2)
        target_points = _ensure_batch_dim(target_points, 3)
        occluded = _ensure_batch_dim(occluded, 2)

        bsz = int(video.shape[0])
        input_resolution = tuple(video.shape[-2:])

        for b in range(bsz):
            resolution = input_resolution if args.metric_resolution_mode == "input" else _resolve_resolution_from_batch(
                batch, b, input_resolution
            )

            # Baseline forward on the full query set.
            with torch.no_grad():
                pred_refined, pred_visibility_refined, info_refined = _run_forward(
                    model,
                    video[b : b + 1],
                    query_points[b : b + 1],
                    return_info=True,
                )

            base_tracks = None
            base_visibility = None
            if isinstance(info_refined, dict):
                base_tracks = info_refined.get("base_tracks", None)
                base_visibility = info_refined.get("base_visibility", None)
            if not isinstance(base_tracks, torch.Tensor) or not isinstance(base_visibility, torch.Tensor):
                if "base_tracks" in batch and "base_visibility" in batch:
                    base_tracks = batch["base_tracks"].to(device)
                    base_visibility = _as_bool(batch["base_visibility"].to(device))
                else:
                    raise ValueError("Missing base tracker outputs needed for TTA.")

            base_visibility = _as_bool(base_visibility)
            pred_visibility_refined = _as_bool(pred_visibility_refined)

            # Adaptation subset from long-occlusion hard cases.
            t_q = query_points[b, :, 0].round().long().clamp(0, pred_refined.shape[2] - 1)
            max_occ = _max_reappearance_occlusion_run(occluded[b], t_q)
            adapt_sel = _select_adaptation_subset(
                max_occ,
                threshold=adapt_threshold,
                max_points=tta_max_points,
            )
            if adapt_sel.sum().item() <= 0:
                # No usable queries for adaptation in this video.
                adapt_sel = torch.zeros_like(adapt_sel, dtype=torch.bool)
                adapt_sel[: min(tta_max_points, adapt_sel.numel())] = True

            query_adapt = query_points[b : b + 1, adapt_sel]
            base_tracks_adapt = base_tracks[:, adapt_sel]
            base_visibility_adapt = base_visibility[:, adapt_sel]
            base_metrics = None
            refined_metrics = None
            adapted_metrics = None
            pred_orig = None
            vis_orig = None
            base_track_loss = None
            base_vis_loss = None
            tv_loss = None
            total_loss = None

            if not continual:
                _restore_modules(model, module_snapshot)
                optimizer = torch.optim.AdamW(trainable_params, lr=tta_lr, weight_decay=tta_weight_decay)
                scaler = GradScaler(enabled=amp_enabled)

            # Small inner-loop adaptation.
            adaptation_oom = False
            for step in range(tta_steps):
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=amp_enabled, dtype=amp_dtype):
                    pred_orig, vis_orig, _ = _run_forward(
                        model,
                        video[b : b + 1],
                        query_adapt,
                        base_tracks=base_tracks_adapt,
                        base_visibility=base_visibility_adapt,
                        return_info=True,
                        no_grad=False,
                    )

                    base_track_loss = criterion._base_track_consistency_loss(
                        pred_orig,
                        base_tracks_adapt,
                        gt_tracks=None,
                        gt_visibility=None,
                        base_visibility=base_visibility_adapt,
                        mask=None,
                    )
                    base_vis_loss = criterion._base_visibility_consistency_loss(
                        vis_orig,
                        base_visibility_adapt,
                        gt_visibility=None,
                        mask=None,
                    )

                    try:
                        tv_loss = _compute_two_view_consistency_loss(
                            model=model,
                            video=video[b : b + 1],
                            query_points=query_adapt,
                            gt_visibility=None,
                            cfg=getattr(config.loss, "two_view_consistency", None),
                            epoch=0,
                            base_tracks=base_tracks_adapt,
                            base_visibility=base_visibility_adapt,
                            total_epochs=1,
                            ema=None,
                            global_step=step,
                            total_steps=tta_steps,
                            optimizer_step=step,
                            total_optimizer_steps=tta_steps,
                        )
                    except RuntimeError as exc:
                        two_view_cfg = getattr(getattr(config, "loss", None), "two_view_consistency", None)
                        oom_safe = bool(getattr(two_view_cfg, "oom_safe", True))
                        msg = str(exc).lower()
                        if oom_safe and ("out of memory" in msg or ("cuda" in msg and "alloc" in msg and "memory" in msg)):
                            if device.type == "cuda":
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass
                            logger.warning(
                                "two_view_consistency hit OOM during TTA; skipping two-view loss for this video "
                                "(loss.two_view_consistency.oom_safe=true)."
                            )
                            tv_loss = None
                        else:
                            raise
                    if tv_loss is None:
                        tv_loss = torch.tensor(0.0, device=device)
                    elif isinstance(tv_loss, dict):
                        tv_loss = tv_loss.get("total", torch.tensor(0.0, device=device))

                    total_loss = tv_loss + base_track_loss + base_vis_loss

                try:
                    scaler.scale(total_loss).backward()
                    if tta_grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_params, tta_grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    if "out of memory" in msg or ("cuda" in msg and "alloc" in msg and "memory" in msg):
                        adaptation_oom = True
                        if device.type == "cuda":
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                        optimizer.zero_grad(set_to_none=True)
                        logger.warning(
                            "TTA backward hit OOM; skipping adaptation update for this video "
                            "(adaptive_step_oom_safe=true)."
                        )
                        break
                    raise

            if adaptation_oom:
                # Leave the model at its pre-update state and fall through to the
                # adapted-forward pass, which now effectively acts as a no-op.
                pass

            # Adapted forward on the full query set.
            try:
                with torch.no_grad():
                    pred_adapted, pred_visibility_adapted, _ = _run_forward(
                        model,
                        video[b : b + 1],
                        query_points[b : b + 1],
                        return_info=True,
                    )
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "out of memory" in msg or ("cuda" in msg and "alloc" in msg and "memory" in msg):
                    if device.type == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    logger.warning(
                        "Adapted forward hit OOM; falling back to refined predictions for this video "
                        "(adaptive_forward_oom_safe=true)."
                    )
                    pred_adapted = pred_refined
                    pred_visibility_adapted = pred_visibility_refined
                else:
                    raise

            gt_visibility_full = ~occluded[b]
            base_full_metrics = _metric_row(
                compute_tapvid_metrics(
                    base_tracks[0],
                    target_points[b],
                    base_visibility[0],
                    gt_visibility_full,
                    query_points[b],
                    resolution=resolution,
                    exclude_query_frame=True,
                    query_mode=query_mode,
                )
            )
            refined_full_metrics = _metric_row(
                compute_tapvid_metrics(
                    pred_refined[0],
                    target_points[b],
                    pred_visibility_refined[0],
                    gt_visibility_full,
                    query_points[b],
                    resolution=resolution,
                    exclude_query_frame=True,
                    query_mode=query_mode,
                )
            )
            adapted_full_metrics = _metric_row(
                compute_tapvid_metrics(
                    pred_adapted[0],
                    target_points[b],
                    _as_bool(pred_visibility_adapted)[0],
                    gt_visibility_full,
                    query_points[b],
                    resolution=resolution,
                    exclude_query_frame=True,
                    query_mode=query_mode,
                )
            )
            oracle_refined_full_metrics = _metric_row(
                compute_tapvid_metrics(
                    pred_refined[0],
                    target_points[b],
                    gt_visibility_full,
                    gt_visibility_full,
                    query_points[b],
                    resolution=resolution,
                    exclude_query_frame=True,
                    query_mode=query_mode,
                )
            )
            oracle_adapted_full_metrics = _metric_row(
                compute_tapvid_metrics(
                    pred_adapted[0],
                    target_points[b],
                    gt_visibility_full,
                    gt_visibility_full,
                    query_points[b],
                    resolution=resolution,
                    exclude_query_frame=True,
                    query_mode=query_mode,
                )
            )
            full_accumulator.add(
                base_metrics=base_full_metrics,
                refined_metrics=refined_full_metrics,
                adapted_metrics=adapted_full_metrics,
                oracle_refined_metrics=oracle_refined_full_metrics,
                oracle_adapted_metrics=oracle_adapted_full_metrics,
                num_queries=int(query_points[b].shape[0]),
            )

            for thr, acc in accumulators.items():
                sel = max_occ >= int(thr)
                num_sel = int(sel.long().sum().item())
                if num_sel <= 0:
                    continue

                base_metrics, refined_metrics = _evaluate_subset(
                    pred_tracks=pred_refined[0],
                    pred_visibility=pred_visibility_refined[0],
                    base_tracks=base_tracks[0],
                    base_visibility=base_visibility[0],
                    target_points=target_points[b],
                    occluded=occluded[b],
                    query_points=query_points[b],
                    sel=sel,
                    resolution=resolution,
                    query_mode=query_mode,
                )
                _, adapted_metrics = _evaluate_subset(
                    pred_tracks=pred_adapted[0],
                    pred_visibility=_as_bool(pred_visibility_adapted)[0],
                    base_tracks=base_tracks[0],
                    base_visibility=base_visibility[0],
                    target_points=target_points[b],
                    occluded=occluded[b],
                    query_points=query_points[b],
                    sel=sel,
                    resolution=resolution,
                    query_mode=query_mode,
                )
                oracle_refined_metrics = _evaluate_subset_oracle_visibility(
                    pred_tracks=pred_refined[0],
                    target_points=target_points[b],
                    occluded=occluded[b],
                    query_points=query_points[b],
                    sel=sel,
                    resolution=resolution,
                    query_mode=query_mode,
                )
                oracle_adapted_metrics = _evaluate_subset_oracle_visibility(
                    pred_tracks=pred_adapted[0],
                    target_points=target_points[b],
                    occluded=occluded[b],
                    query_points=query_points[b],
                    sel=sel,
                    resolution=resolution,
                    query_mode=query_mode,
                )

                acc.add(
                    base_metrics=base_metrics,
                    refined_metrics=refined_metrics,
                    adapted_metrics=adapted_metrics,
                    oracle_refined_metrics=oracle_refined_metrics,
                    oracle_adapted_metrics=oracle_adapted_metrics,
                    num_selected=num_sel,
                )

            # The two-view TTA path allocates a large temporary graph per video.
            # Release those tensors explicitly so the next video starts from a
            # clean CUDA cache footprint instead of accumulating fragmentation.
            del pred_refined, pred_visibility_refined, pred_adapted, pred_visibility_adapted
            del info_refined
            del base_tracks, base_visibility, target_points, occluded, query_points, video
            del base_tracks_adapt, base_visibility_adapt, query_adapt
            if tta_steps > 0:
                del pred_orig, vis_orig, base_track_loss, base_vis_loss, tv_loss, total_loss
            del adapt_sel, max_occ, t_q, base_metrics, refined_metrics, adapted_metrics
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summaries = [accumulators[thr].summarize() for thr in thresholds]
    full_summary = full_accumulator.summarize()

    print()
    print("=== Long-occlusion test-time adaptation ===")
    print(
        f"thresholds={thresholds} adapt_threshold={adapt_threshold} "
        f"tta_steps={tta_steps} tta_lr={tta_lr:g} modules={trainable_modules} "
        f"query_mode={query_mode}"
    )
    print()

    header = (
        f"{'thr':>4} | {'vids':>5} {'qs':>6} |"
        f"{'AJ_b':>10} {'AJ_r':>10} {'AJ_a':>10} |"
        f"{'AJ_a-r':>10} {'AJ_a-b':>10} |"
        f"{'err_b':>10} {'err_r':>10} {'err_a':>10}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        base = s["base"]
        refined = s["refined"]
        adapted = s["adapted"]
        dbr = s["delta_vs_refined"]
        dab = s["delta_adapted"]
        print(
            f"{int(s['threshold']):4d} |"
            f"{int(s['num_videos']):5d} {int(s['num_queries']):6d} |"
            f"{_format(base.get('AJ'))} {_format(refined.get('AJ'))} {_format(adapted.get('AJ'))} |"
            f"{_format(dbr.get('AJ'))} {_format(dab.get('AJ'))} |"
            f"{_format(base.get('avg_error_px'))} {_format(refined.get('avg_error_px'))} {_format(adapted.get('avg_error_px'))}"
        )

    print()
    print("Oracle-visibility decomposition (GT visibility, same tracks):")
    header_oracle = (
        f"{'thr':>4} |"
        f"{'AJ_or_r':>10} {'AJ_or_a':>10} {'dAJ_or':>10} |"
        f"{'err_r':>10} {'err_a':>10}"
    )
    print(header_oracle)
    print("-" * len(header_oracle))
    for s in summaries:
        oracle_refined = s["oracle_refined"]
        oracle_adapted = s["oracle_adapted"]
        d_oracle = s["delta_oracle_vs_refined"]
        print(
            f"{int(s['threshold']):4d} |"
            f"{_format(oracle_refined.get('AJ'))} {_format(oracle_adapted.get('AJ'))} {_format(d_oracle.get('AJ'))} |"
            f"{_format(oracle_refined.get('avg_error_px'))} {_format(oracle_adapted.get('avg_error_px'))}"
        )

    print()
    print("=== Full TAP-Vid evaluation ===")
    print(
        f"videos={int(full_summary.get('num_videos', 0))} queries={int(full_summary.get('num_queries', 0))} "
        f"adapt_threshold={adapt_threshold} tta_steps={tta_steps} tta_lr={tta_lr:g} "
        f"modules={trainable_modules} query_mode={query_mode}"
    )
    full_base = full_summary["base"]
    full_refined = full_summary["refined"]
    full_adapted = full_summary["adapted"]
    full_dbr = full_summary["delta_vs_refined"]
    full_dab = full_summary["delta_adapted"]
    full_oracle_refined = full_summary["oracle_refined"]
    full_oracle_adapted = full_summary["oracle_adapted"]
    full_d_oracle = full_summary["delta_oracle_vs_refined"]
    header_full = (
        f"{'AJ_b':>10} {'AJ_r':>10} {'AJ_a':>10} |"
        f"{'AJ_a-r':>10} {'AJ_a-b':>10} |"
        f"{'OA_b':>10} {'OA_a':>10} {'dOA':>10} |"
        f"{'err_b':>10} {'err_a':>10}"
    )
    print(header_full)
    print("-" * len(header_full))
    print(
        f"{_format(full_base.get('AJ'))} {_format(full_refined.get('AJ'))} {_format(full_adapted.get('AJ'))} |"
        f"{_format(full_dbr.get('AJ'))} {_format(full_dab.get('AJ'))} |"
        f"{_format(full_base.get('OA'))} {_format(full_adapted.get('OA'))} {_format(full_dab.get('OA'))} |"
        f"{_format(full_base.get('avg_error_px'))} {_format(full_adapted.get('avg_error_px'))}"
    )
    print("Oracle-visibility decomposition (full set):")
    full_oracle_delta = full_d_oracle.get("AJ")
    if full_oracle_delta is None:
        full_oracle_delta_str = "-"
    else:
        full_oracle_delta_str = f"{float(full_oracle_delta):+10.6f}"
    print(
        f"  AJ_or_r={_format(full_oracle_refined.get('AJ'))} "
        f"AJ_or_a={_format(full_oracle_adapted.get('AJ'))} "
        f"dAJ_or={full_oracle_delta_str} "
        f"err_r={_format(full_oracle_refined.get('avg_error_px'))} "
        f"err_a={_format(full_oracle_adapted.get('avg_error_px'))}"
    )

    if args.output_json:
        out_path = Path(args.output_json)
        payload = {
            "config": str(Path(args.config)),
            "checkpoint": str(Path(args.checkpoint)),
            "seed": int(args.seed) if args.seed is not None else None,
            "thresholds": thresholds,
            "adapt_threshold": adapt_threshold,
            "tta_steps": tta_steps,
            "tta_lr": tta_lr,
            "modules": trainable_modules,
            "query_mode": query_mode,
            "results": summaries,
            "full": full_summary,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print()
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
