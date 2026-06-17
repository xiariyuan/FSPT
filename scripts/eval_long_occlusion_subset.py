#!/usr/bin/env python3
"""
Evaluate TAP-Vid metrics on a "long occlusion" subset (re-localization focus).

Motivation
----------
When chasing overall SOTA becomes noise-level (AJ_delta ~ 1e-4), a more
publishable direction is to analyze *failure modes* of a strong base tracker and
target improvements there. A very common failure mode is:

  "The point is visible at query time -> gets occluded for a long time ->
   re-appears, but the tracker fails to re-localize."

This script creates a *reproducible subset* based purely on GT occlusion flags:
select queries whose GT trajectory contains a long *continuous occlusion run*
AFTER the query frame, and that occlusion run is followed by at least one
visible frame (i.e. the point actually reappears).

It then evaluates official TAP-Vid metrics (vendored) on only those selected
queries, optionally comparing base vs refined and printing delta.

Usage (server)
--------------
  cd /gemini/code/FSPT
  /root/miniconda3/bin/python scripts/eval_long_occlusion_subset.py \
    --config configs/<your_config>.yaml \
    --checkpoint outputs/<exp>/checkpoints/best_aj.pth \
    --thresholds 10,20,30 \
    --metric-resolution-mode original
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

# Add project root to import path so `datasets/` etc work when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.metrics import compute_tapvid_metrics

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


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


def _load_model(
    *,
    config_path: Path,
    checkpoint_path: Optional[Path],
    use_ema: Optional[bool],
    device: torch.device,
):
    from train import create_model, load_config

    config_from_file = load_config(str(config_path))
    checkpoint: Optional[Dict[str, Any]] = None
    if checkpoint_path is not None:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    # Prefer the checkpoint config when present to avoid config/defaults drift.
    config = config_from_file
    if checkpoint is not None and "config" in checkpoint:
        try:
            config = OmegaConf.create(checkpoint["config"])
            logger.info("Using config embedded in checkpoint (overrides --config for model & dataloader).")
        except Exception as exc:
            logger.warning(f"Failed to load config from checkpoint; falling back to --config. ({exc})")

    model = create_model(config).to(device)

    if checkpoint_path is not None:
        assert checkpoint is not None
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
                logger.warning("EMA weights not found; falling back to model_state_dict.")

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
            # In Route-A we intentionally do not save CoTracker weights in checkpoints;
            # they are loaded from `weights/cotracker/*.pth`. Don't spam scary warnings.
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

        logger.info(f"Loaded checkpoint: {checkpoint_path}")

    model.eval()
    return model, config


def _resolve_resolution_from_batch(batch: Dict[str, Any], index: int, fallback: Tuple[int, int]) -> Tuple[int, int]:
    if not isinstance(batch, dict):
        return fallback
    original_size = batch.get("original_size", None)
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
            if len(original_size) == 2 and isinstance(original_size[0], torch.Tensor) and isinstance(
                original_size[1], torch.Tensor
            ):
                h_tensor, w_tensor = original_size[0], original_size[1]
                try:
                    h_val = h_tensor[index].item() if h_tensor.numel() > index else h_tensor.reshape(-1)[0].item()
                    w_val = w_tensor[index].item() if w_tensor.numel() > index else w_tensor.reshape(-1)[0].item()
                    return (int(h_val), int(w_val))
                except Exception:
                    pass
            if len(original_size) > 0 and isinstance(original_size[0], (list, tuple, torch.Tensor)):
                size = original_size[index] if index < len(original_size) else original_size[0]
                if hasattr(size, "tolist"):
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


def _as_bool(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.bool else x > 0.5


def _ensure_batch_dim(x: torch.Tensor, ndim_without_batch: int) -> torch.Tensor:
    if x.ndim == ndim_without_batch:
        return x.unsqueeze(0)
    return x


def _max_reappearance_occlusion_run(occluded: torch.Tensor, query_t: torch.Tensor) -> torch.Tensor:
    """
    Compute, for each query, the length of the longest *continuous* occlusion run
    AFTER the query frame that is followed by a visible frame (reappearance).

    Args:
        occluded: (N, T) bool
        query_t:  (N,) long in [0, T-1]

    Returns:
        max_run_len: (N,) long
    """
    if occluded.ndim != 2:
        raise ValueError(f"Expected occluded shape (N,T), got {tuple(occluded.shape)}")
    if query_t.ndim != 1:
        raise ValueError(f"Expected query_t shape (N,), got {tuple(query_t.shape)}")

    n_queries, num_frames = occluded.shape
    out = torch.zeros((n_queries,), dtype=torch.long, device=occluded.device)

    # Python loop is fine here: N is typically <= 256 for our training configs.
    for q in range(n_queries):
        start = int(query_t[q].item()) + 1
        if start >= num_frames:
            continue
        mask = occluded[q, start:]
        if mask.numel() == 0:
            continue

        best = 0
        run = 0
        for val in mask.tolist():
            if val:
                run += 1
                continue
            # run ended, and we are now visible -> reappearance exists
            if run > best:
                best = run
            run = 0
        # NOTE: if `run > 0` at the end, it means occluded until end-of-video
        # (no reappearance), so we intentionally ignore it.
        out[q] = best
    return out


def _safe_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            value = value.detach().cpu().item()
        return float(value)
    except Exception:
        return None


def _mean_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys: Iterable[str] = rows[0].keys()
    out: Dict[str, float] = {}
    for k in keys:
        vals: List[float] = []
        for row in rows:
            v = _safe_float(row.get(k))
            if v is None or not np.isfinite(v):
                continue
            vals.append(v)
        if vals:
            out[k] = float(np.mean(vals))
    return out


def _format(v: Optional[float], width: int = 10, precision: int = 6) -> str:
    if v is None:
        return ("-" * (width - 1)).rjust(width)
    s = f"{v:.{precision}f}"
    return s.rjust(width)[:width]


@dataclass
class SubsetAccumulator:
    threshold: int
    num_videos: int = 0
    num_queries: int = 0
    refined: List[Dict[str, float]] = None  # type: ignore[assignment]
    base: List[Dict[str, float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.refined = []
        self.base = []

    def add(
        self,
        *,
        refined_metrics: Dict[str, float],
        base_metrics: Optional[Dict[str, float]],
        num_selected: int,
    ) -> None:
        self.num_videos += 1
        self.num_queries += int(num_selected)
        self.refined.append(refined_metrics)
        if base_metrics is not None:
            self.base.append(base_metrics)

    def summarize(self) -> Dict[str, Any]:
        refined_mean = _mean_metrics(self.refined)
        base_mean = _mean_metrics(self.base) if self.base else {}
        delta: Dict[str, float] = {}
        for key in refined_mean.keys():
            if key in base_mean:
                delta[key] = float(refined_mean[key]) - float(base_mean[key])
        return {
            "threshold": self.threshold,
            "num_videos": self.num_videos,
            "num_queries": self.num_queries,
            "refined": refined_mean,
            "base": base_mean,
            "delta": delta,
        }


def _build_val_loader_from_config(config) -> Any:
    """
    Create the validation dataloader with the same arg filtering as `train.py`.

    This avoids passing unknown config keys into dataset constructors.
    """
    from datasets import get_dataloader

    if not hasattr(config, "data") or not hasattr(config.data, "val"):
        raise ValueError("Config missing data.val.")
    val_config = config.data.val
    val_dataset_name = str(getattr(val_config, "dataset", "")).lower()

    val_extra_args: Dict[str, Any] = {}

    if "kubric" in val_dataset_name:
        for key in (
            "backend",
            "allow_synthetic_tracks",
            "annotation_file",
            "use_tfds",
            "tfds_name",
            "shuffle_buffer",
            "shuffle_files",
            "tfds_low_memory",
            "tfds_read_buffer_size",
            "deterministic_sampling",
            "deterministic_seed",
        ):
            if hasattr(val_config, key):
                value = getattr(val_config, key)
                if value is not None:
                    val_extra_args[key] = value
        if hasattr(val_config, "sampling") and val_config.sampling is not None:
            val_extra_args["sampling"] = val_config.sampling

    if hasattr(val_config, "resolution") and val_config.resolution is not None:
        val_extra_args["resolution"] = tuple(val_config.resolution)
    if hasattr(val_config, "augmentation"):
        val_extra_args["augmentation"] = val_config.augmentation
    if hasattr(val_config, "num_points"):
        val_extra_args["num_points"] = val_config.num_points

    if "davis" in val_dataset_name or "kinetics" in val_dataset_name:
        query_mode = getattr(val_config, "query_mode", None) if hasattr(val_config, "query_mode") else None
        if query_mode is None and hasattr(config, "evaluation") and hasattr(config.evaluation, "query_mode"):
            query_mode = getattr(config.evaluation, "query_mode")
        if query_mode is not None and str(query_mode).strip().lower() not in ("", "none", "null"):
            val_extra_args["query_mode"] = str(query_mode).strip()

        if hasattr(val_config, "query_stride") and getattr(val_config, "query_stride") is not None:
            try:
                val_extra_args["query_stride"] = max(1, int(val_config.query_stride))
            except Exception:
                pass
        if hasattr(val_config, "points_order") and getattr(val_config, "points_order") is not None:
            val_extra_args["points_order"] = str(val_config.points_order)

    if "kinetics" in val_dataset_name:
        if hasattr(val_config, "max_frames"):
            val_extra_args["max_frames"] = val_config.max_frames
        elif hasattr(val_config, "num_frames") and val_config.num_frames is not None:
            if int(val_config.num_frames) > 0:
                val_extra_args["max_frames"] = int(val_config.num_frames)
    elif "davis" not in val_dataset_name:
        if hasattr(val_config, "num_frames") and val_config.num_frames is not None:
            val_extra_args["num_frames"] = val_config.num_frames

    if hasattr(val_config, "base_tracks_dir") and val_config.base_tracks_dir is not None:
        val_extra_args["base_tracks_dir"] = val_config.base_tracks_dir
        if hasattr(val_config, "base_tracks_strict"):
            val_extra_args["base_tracks_strict"] = bool(val_config.base_tracks_strict)
        if hasattr(val_config, "base_tracks_query_tol") and val_config.base_tracks_query_tol is not None:
            try:
                val_extra_args["base_tracks_query_tol"] = float(val_config.base_tracks_query_tol)
            except Exception:
                pass

    val_num_workers = int(getattr(getattr(config, "evaluation", None), "num_workers", None) or 0)
    if val_num_workers <= 0:
        val_num_workers = int(getattr(getattr(config, "training", None), "num_workers", None) or 4)

    return get_dataloader(
        name=val_config.dataset,
        root=val_config.root,
        batch_size=1,
        split="val",
        num_workers=val_num_workers,
        pin_memory=True,
        seed=int(getattr(getattr(config, "experiment", None), "seed", 0) or 0),
        **val_extra_args,
    )


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="Evaluate long-occlusion subset TAP-Vid metrics (base/refined/delta).")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path used to build the model/dataloader.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Checkpoint path. If empty, evaluate random-init weights (debug only).",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="10,20,30",
        help="Comma-separated occlusion-run thresholds in frames (after query). Example: 10,20,30",
    )
    parser.add_argument(
        "--metric-resolution-mode",
        type=str,
        default="original",
        choices=("original", "input"),
        help="Compute metrics in original resolution (paper) or input resolution (debug).",
    )
    parser.add_argument("--query-mode", type=str, default="", help="Override query_mode (first|strided).")
    parser.add_argument(
        "--exclude-query-frame",
        action="store_true",
        default=True,
        help="Exclude query frame from evaluation (official).",
    )
    parser.add_argument("--no-exclude-query-frame", action="store_false", dest="exclude_query_frame")
    parser.add_argument(
        "--compare-base",
        action="store_true",
        default=True,
        help="Request base/refined comparison if model supports return_info base_tracks.",
    )
    parser.add_argument("--no-compare-base", action="store_false", dest="compare_base")
    parser.add_argument("--use-ema", action="store_true", help="Force using EMA weights if present.")
    parser.add_argument("--no-ema", action="store_true", help="Force not using EMA weights.")
    parser.add_argument("--max-videos", type=int, default=0, help="Limit evaluation to first N videos (0=all).")
    parser.add_argument("--output-json", type=str, default="", help="Optional path to save JSON summary.")
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    thresholds = [int(x) for x in str(args.thresholds).split(",") if str(x).strip()]
    thresholds = sorted(set(int(x) for x in thresholds if int(x) > 0))
    if not thresholds:
        raise ValueError("No valid thresholds provided.")

    use_ema: Optional[bool] = None
    if args.use_ema and args.no_ema:
        raise ValueError("Conflicting flags: --use-ema and --no-ema")
    if args.use_ema:
        use_ema = True
    if args.no_ema:
        use_ema = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, config = _load_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        use_ema=use_ema,
        device=device,
    )
    dataloader = _build_val_loader_from_config(config)

    # Use config evaluation defaults if not overridden.
    query_mode = args.query_mode.strip()
    if not query_mode and hasattr(config, "evaluation") and hasattr(config.evaluation, "query_mode"):
        query_mode = str(getattr(config.evaluation, "query_mode") or "").strip()
    if not query_mode or query_mode.lower() in ("none", "null", ""):
        query_mode = None

    accumulators = {thr: SubsetAccumulator(threshold=thr) for thr in thresholds}

    iterator = tqdm(dataloader, desc="Long-occ subset eval", disable=False)
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

        # Ensure batch dim.
        video = _ensure_batch_dim(video, 4)  # (B,T,3,H,W)
        query_points = _ensure_batch_dim(query_points, 2)  # (B,N,3)
        target_points = _ensure_batch_dim(target_points, 3)  # (B,N,T,2)
        occluded = _ensure_batch_dim(occluded, 2)  # (B,N,T)

        bsz = int(video.shape[0])
        input_resolution = tuple(video.shape[-2:])

        meta = {
            "video_name": batch.get("video_name", None),
            "base_tracks": batch.get("base_tracks", None),
            "base_visibility": batch.get("base_visibility", None),
        }

        with torch.no_grad():
            if args.compare_base:
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

        if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
            raise ValueError(f"Unexpected model outputs: {type(outputs)}")

        pred_tracks = outputs[0]
        pred_visibility = outputs[1]

        base_tracks = None
        base_visibility = None
        if args.compare_base and len(outputs) >= 3 and isinstance(outputs[2], dict):
            info = outputs[2]
            base_tracks = info.get("base_tracks", None)
            base_visibility = info.get("base_visibility", None)
        if base_tracks is None:
            base_tracks = batch.get("base_tracks", None)
        if base_visibility is None:
            base_visibility = batch.get("base_visibility", None)

        # Move base tracks to device if present.
        if isinstance(base_tracks, torch.Tensor):
            base_tracks = base_tracks.to(device)
        if isinstance(base_visibility, torch.Tensor):
            base_visibility = base_visibility.to(device)

        pred_visibility = _as_bool(pred_visibility)
        if isinstance(base_visibility, torch.Tensor):
            base_visibility = _as_bool(base_visibility)

        for b in range(bsz):
            # Determine resolution (paper vs debug).
            if args.metric_resolution_mode == "input":
                resolution = input_resolution
            else:
                resolution = _resolve_resolution_from_batch(batch, b, input_resolution)

            # Query times.
            t_q = query_points[b, :, 0].round().long().clamp(0, pred_tracks.shape[2] - 1)
            max_occ = _max_reappearance_occlusion_run(occluded[b], t_q)

            # Evaluate each threshold subset.
            for thr, acc in accumulators.items():
                sel = max_occ >= int(thr)
                num_sel = int(sel.long().sum().item())
                if num_sel <= 0:
                    continue

                refined_metrics = compute_tapvid_metrics(
                    pred_tracks[b][sel],
                    target_points[b][sel],
                    pred_visibility[b][sel],
                    ~occluded[b][sel],
                    query_points[b][sel],
                    resolution=resolution,
                    exclude_query_frame=bool(args.exclude_query_frame),
                    query_mode=query_mode,
                )

                base_metrics = None
                if isinstance(base_tracks, torch.Tensor) and isinstance(base_visibility, torch.Tensor):
                    if base_tracks.ndim == pred_tracks.ndim and base_tracks.shape[:3] == pred_tracks.shape[:3]:
                        base_metrics = compute_tapvid_metrics(
                            base_tracks[b][sel],
                            target_points[b][sel],
                            base_visibility[b][sel],
                            ~occluded[b][sel],
                            query_points[b][sel],
                            resolution=resolution,
                            exclude_query_frame=bool(args.exclude_query_frame),
                            query_mode=query_mode,
                        )

                acc.add(refined_metrics=refined_metrics, base_metrics=base_metrics, num_selected=num_sel)

    summaries = [accumulators[thr].summarize() for thr in thresholds]

    print()
    print("=== Long-occlusion subset (reappearance) ===")
    print(f"thresholds={thresholds} metric_resolution_mode={args.metric_resolution_mode} query_mode={query_mode}")
    print()

    header = (
        f"{'thr':>4} | {'vids':>5} {'qs':>6} |"
        f"{'AJ_b':>10} {'AJ':>10} {'AJ_d':>10} |"
        f"{'<4_b':>10} {'<4':>10} {'<4_d':>10} |"
        f"{'err_b':>10} {'err':>10} {'err_d':>10} |"
        f"{'OA_b':>10} {'OA':>10} {'OA_d':>10}"
    )
    print(header)
    print("-" * len(header))
    for s in summaries:
        refined = s.get("refined", {}) or {}
        base = s.get("base", {}) or {}
        delta = s.get("delta", {}) or {}

        def gf(d: Dict[str, Any], k: str) -> Optional[float]:
            return _safe_float(d.get(k))

        print(
            f"{int(s['threshold']):4d} |"
            f"{int(s['num_videos']):5d} {int(s['num_queries']):6d} |"
            f"{_format(gf(base,'AJ'))} {_format(gf(refined,'AJ'))} {_format(gf(delta,'AJ'))} |"
            f"{_format(gf(base,'<4px'))} {_format(gf(refined,'<4px'))} {_format(gf(delta,'<4px'))} |"
            f"{_format(gf(base,'avg_error_px'))} {_format(gf(refined,'avg_error_px'))} {_format(gf(delta,'avg_error_px'))} |"
            f"{_format(gf(base,'OA'))} {_format(gf(refined,'OA'))} {_format(gf(delta,'OA'))}"
        )

    if args.output_json:
        out_path = Path(args.output_json)
        payload = {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path) if checkpoint_path is not None else "",
            "metric_resolution_mode": args.metric_resolution_mode,
            "query_mode": query_mode,
            "thresholds": thresholds,
            "results": summaries,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print()
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

