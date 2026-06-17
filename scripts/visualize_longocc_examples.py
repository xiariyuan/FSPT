#!/usr/bin/env python3
"""
Find and visualize representative DAVIS examples for the "re-localization after
long occlusion" story (Route A).

What it does
------------
1) Runs the model on TAP-Vid DAVIS val.
2) Selects queries whose GT trajectory has a long *continuous* occlusion run
   (>=K frames) AFTER the query frame, AND later re-appears.
3) Ranks videos by AJ_delta on that subset (refined - base).
4) Saves:
   - ranking.json / ranking.csv
   - side-by-side visualizations for the top-K videos

Usage (server)
--------------
  cd /gemini/code/FSPT
  /root/miniconda3/bin/python scripts/visualize_longocc_examples.py \\
    --config configs/fspt_routeA_stage3_relocal_longocc_m20_v2.yaml \\
    --checkpoint checkpoints/fspt_routeA_stage3_relocal_longocc_m20_v2/best.pth \\
    --threshold 20 \\
    --topk 3 \\
    --out-dir outputs/qual_longocc_stage3v2
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

# Add project root to import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.metrics import compute_tapvid_metrics

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", str(name))


def _as_bool(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.bool:
        return x
    return x > 0.5


def _ensure_batch_dim(x: torch.Tensor, sample_ndim: int) -> torch.Tensor:
    """
    Add a leading batch dim if the tensor is missing it.

    Args:
        x: tensor
        sample_ndim: expected ndim without batch dim
    """
    if x.ndim == sample_ndim:
        return x.unsqueeze(0)
    return x


def _resolve_resolution_from_batch(batch: Dict[str, Any], index: int, fallback: Tuple[int, int]) -> Tuple[int, int]:
    original_size = batch.get("original_size", None)
    if original_size is None:
        return fallback
    try:
        if isinstance(original_size, torch.Tensor):
            if original_size.ndim == 2 and original_size.shape[0] > index:
                size = original_size[index].tolist()
            else:
                size = original_size.tolist()
        elif isinstance(original_size, (list, tuple)):
            if (
                len(original_size) == 2
                and isinstance(original_size[0], torch.Tensor)
                and isinstance(original_size[1], torch.Tensor)
            ):
                h_tensor, w_tensor = original_size[0], original_size[1]
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


def _max_reappearance_occlusion_run(occluded_nt: torch.Tensor, query_t: torch.Tensor) -> torch.Tensor:
    """
    For each query, compute the longest continuous occlusion run AFTER query_t
    that is followed by at least one visible frame (reappearance).
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
            if run > best:
                best = run
            run = 0
        out[q] = best
    return out


def _load_model(
    *,
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
):
    from train import create_model, load_config

    config_from_file = load_config(str(config_path))
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    config = config_from_file
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        try:
            config = OmegaConf.create(checkpoint["config"])
            logger.info("Using config embedded in checkpoint (overrides --config for model & dataloader).")
        except Exception as exc:
            logger.warning(f"Failed to load config from checkpoint; falling back to --config. ({exc})")

    model = create_model(config).to(device)
    model.eval()

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint missing model_state_dict and is not a raw state dict.")
    model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded checkpoint: {checkpoint_path}")
    return model, config


def _build_val_loader_from_config(config):
    from datasets import get_dataloader

    if not hasattr(config, "data") or not hasattr(config.data, "val"):
        raise ValueError("Config missing data.val.")
    val_cfg = config.data.val

    extra: Dict[str, Any] = {}
    if hasattr(val_cfg, "resolution") and val_cfg.resolution is not None:
        extra["resolution"] = tuple(val_cfg.resolution)
    if hasattr(val_cfg, "augmentation"):
        extra["augmentation"] = val_cfg.augmentation
    if hasattr(val_cfg, "num_points"):
        extra["num_points"] = val_cfg.num_points
    if hasattr(val_cfg, "query_mode") and val_cfg.query_mode is not None:
        extra["query_mode"] = str(val_cfg.query_mode)
    if hasattr(val_cfg, "query_stride") and val_cfg.query_stride is not None:
        try:
            extra["query_stride"] = int(val_cfg.query_stride)
        except Exception:
            pass

    # Prefer evaluation.num_workers, fallback to training.num_workers.
    num_workers = int(getattr(getattr(config, "evaluation", None), "num_workers", None) or 0)
    if num_workers <= 0:
        num_workers = int(getattr(getattr(config, "training", None), "num_workers", None) or 4)

    return get_dataloader(
        name=val_cfg.dataset,
        root=val_cfg.root,
        batch_size=1,
        split="val",
        num_workers=num_workers,
        pin_memory=True,
        seed=int(getattr(getattr(config, "experiment", None), "seed", 0) or 0),
        **extra,
    )


def _to_numpy_video(video: torch.Tensor) -> np.ndarray:
    """
    Convert (T,3,H,W) or (1,T,3,H,W) float/uint8 tensor to numpy (T,H,W,3) uint8.
    """
    if video.ndim == 5:
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"Unexpected video shape: {tuple(video.shape)}")
    # assume (T,3,H,W)
    if video.shape[1] != 3 and video.shape[0] == 3:
        # maybe (3,T,H,W)
        video = video.permute(1, 0, 2, 3)
    video = video.detach().cpu()
    if video.dtype != torch.uint8:
        video = video.float().clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
    video = video.permute(0, 2, 3, 1).contiguous()  # (T,H,W,3)
    return video.numpy()


def _visualize_side_by_side(
    *,
    video: np.ndarray,
    base_tracks: np.ndarray,
    refined_tracks: np.ndarray,
    gt_tracks: np.ndarray,
    base_vis: np.ndarray,
    refined_vis: np.ndarray,
    gt_vis: np.ndarray,
    out_dir: Path,
    video_name: str,
    max_points: int = 20,
    fps: int = 6,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Circle

    out_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_name(video_name)

    T, H, W, _ = video.shape
    n_points = int(base_tracks.shape[0])
    if n_points == 0:
        return

    # Auto-detect coords: if max > 1.5, assume pixels
    is_pixel = float(np.max(np.abs(base_tracks))) > 1.5 or float(np.max(np.abs(gt_tracks))) > 1.5

    if n_points > max_points:
        idxs = np.linspace(0, n_points - 1, max_points, dtype=int)
    else:
        idxs = np.arange(n_points, dtype=int)

    colors = list(mcolors.TABLEAU_COLORS.values())
    frames: List[np.ndarray] = []

    for t in range(T):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(video[t])
        axes[0].set_title(f"Base (t={t})")
        axes[1].imshow(video[t])
        axes[1].set_title(f"Refined (t={t})")

        for j, p in enumerate(idxs):
            c = colors[j % len(colors)]

            def _yx_to_px(yx: Sequence[float]) -> Tuple[float, float]:
                y, x = float(yx[0]), float(yx[1])
                if not is_pixel:
                    y, x = y * H, x * W
                return y, x

            # Draw GT (red x) when visible.
            if gt_vis[p, t]:
                gy, gx = _yx_to_px(gt_tracks[p, t])
                for ax in axes:
                    ax.scatter([gx], [gy], c="red", s=18, marker="x", linewidths=2)

            # Draw base point
            if base_vis[p, t]:
                by, bx = _yx_to_px(base_tracks[p, t])
                axes[0].add_patch(Circle((bx, by), 3, color=c, alpha=0.9))
                # short trail
                for t2 in range(max(0, t - 6), t):
                    if base_vis[p, t2]:
                        y1, x1 = _yx_to_px(base_tracks[p, t2])
                        y2, x2 = _yx_to_px(base_tracks[p, min(t2 + 1, t)])
                        axes[0].plot([x1, x2], [y1, y2], color=c, alpha=0.5, linewidth=2)

            # Draw refined point
            if refined_vis[p, t]:
                ry, rx = _yx_to_px(refined_tracks[p, t])
                axes[1].add_patch(Circle((rx, ry), 3, color=c, alpha=0.9))
                for t2 in range(max(0, t - 6), t):
                    if refined_vis[p, t2]:
                        y1, x1 = _yx_to_px(refined_tracks[p, t2])
                        y2, x2 = _yx_to_px(refined_tracks[p, min(t2 + 1, t)])
                        axes[1].plot([x1, x2], [y1, y2], color=c, alpha=0.5, linewidth=2)

        for ax in axes:
            ax.axis("off")

        fig.tight_layout()
        png_path = out_dir / f"{safe}_t{t:03d}.png"
        fig.savefig(png_path, dpi=110, bbox_inches="tight")

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)
        plt.close(fig)

    # Optional GIF
    try:
        import imageio

        gif_path = out_dir / f"{safe}.gif"
        imageio.mimsave(gif_path, frames, fps=fps)
        logger.info(f"Saved GIF: {gif_path}")
    except Exception as exc:
        logger.info(f"GIF skipped ({exc}). Saved PNG frames to: {out_dir}")


@torch.no_grad()
def _rank_videos(
    *,
    model,
    dataloader,
    threshold: int,
    metric_resolution_mode: str,
    query_mode: Optional[str],
    exclude_query_frame: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    device = next(model.parameters()).device

    metric_resolution_mode = str(metric_resolution_mode).strip().lower()
    if metric_resolution_mode not in ("original", "input"):
        metric_resolution_mode = "original"

    for batch in tqdm(dataloader, desc="Ranking videos", leave=False):
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

        meta = {
            "video_name": batch.get("video_name", None),
            "base_tracks": batch.get("base_tracks", None),
            "base_visibility": batch.get("base_visibility", None),
        }

        outputs = model(video, query_points, meta=meta, return_info=True)
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 3 or not isinstance(outputs[2], dict):
            raise ValueError("Model did not return base info (need return_info=True).")

        pred_tracks, pred_vis, info = outputs[0], outputs[1], outputs[2]
        base_tracks = info.get("base_tracks", None)
        base_vis = info.get("base_visibility", None)
        if base_tracks is None or base_vis is None:
            continue

        pred_vis = _as_bool(pred_vis)
        base_vis = _as_bool(base_vis)

        bsz = int(video.shape[0])
        input_resolution = tuple(video.shape[-2:])

        video_names = batch.get("video_name", None)

        for b in range(bsz):
            if metric_resolution_mode == "input":
                resolution = input_resolution
            else:
                resolution = _resolve_resolution_from_batch(batch, b, input_resolution)

            if isinstance(video_names, (list, tuple)):
                name = str(video_names[b]) if b < len(video_names) else f"video_{b}"
            elif video_names is not None:
                name = str(video_names)
            else:
                name = f"video_{b}"

            t_q = query_points[b, :, 0].round().long().clamp(0, pred_tracks.shape[2] - 1)
            max_occ = _max_reappearance_occlusion_run(occluded[b], t_q)
            sel = max_occ >= int(threshold)
            num_sel = int(sel.long().sum().item())
            if num_sel <= 0:
                continue

            refined_metrics = compute_tapvid_metrics(
                pred_tracks[b][sel],
                target_points[b][sel],
                pred_vis[b][sel],
                ~occluded[b][sel],
                query_points[b][sel],
                resolution=resolution,
                exclude_query_frame=exclude_query_frame,
                query_mode=query_mode,
            )
            base_metrics = compute_tapvid_metrics(
                base_tracks[b][sel],
                target_points[b][sel],
                base_vis[b][sel],
                ~occluded[b][sel],
                query_points[b][sel],
                resolution=resolution,
                exclude_query_frame=exclude_query_frame,
                query_mode=query_mode,
            )

            aj_ref = float(refined_metrics.get("AJ", 0.0))
            aj_base = float(base_metrics.get("AJ", 0.0))
            row = {
                "video_name": name,
                "threshold": int(threshold),
                "num_selected": num_sel,
                "AJ_base": aj_base,
                "AJ_refined": aj_ref,
                "AJ_delta": aj_ref - aj_base,
                "avg_error_px_base": float(base_metrics.get("avg_error_px", 0.0)),
                "avg_error_px_refined": float(refined_metrics.get("avg_error_px", 0.0)),
                "avg_error_px_delta": float(refined_metrics.get("avg_error_px", 0.0))
                - float(base_metrics.get("avg_error_px", 0.0)),
            }
            rows.append(row)

    rows.sort(key=lambda r: r.get("AJ_delta", 0.0), reverse=True)
    return rows


@torch.no_grad()
def _visualize_selected(
    *,
    model,
    dataloader,
    selected_video_names: List[str],
    threshold: int,
    metric_resolution_mode: str,
    query_mode: Optional[str],
    exclude_query_frame: bool,
    out_dir: Path,
    max_points: int,
) -> None:
    device = next(model.parameters()).device
    wanted = set(str(n) for n in selected_video_names)
    found: Dict[str, bool] = {n: False for n in wanted}

    metric_resolution_mode = str(metric_resolution_mode).strip().lower()
    if metric_resolution_mode not in ("original", "input"):
        metric_resolution_mode = "original"

    for batch in tqdm(dataloader, desc="Visualizing", leave=False):
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

        video_names = batch.get("video_name", None)
        # Handle only bsz=1 in our loaders, but keep it generic.
        bsz = int(video.shape[0])
        for b in range(bsz):
            if isinstance(video_names, (list, tuple)):
                name = str(video_names[b]) if b < len(video_names) else f"video_{b}"
            elif video_names is not None:
                name = str(video_names)
            else:
                name = f"video_{b}"

            if name not in wanted or found.get(name, False):
                continue

            meta = {
                "video_name": batch.get("video_name", None),
                "base_tracks": batch.get("base_tracks", None),
                "base_visibility": batch.get("base_visibility", None),
            }
            outputs = model(video, query_points, meta=meta, return_info=True)
            pred_tracks, pred_vis, info = outputs[0], outputs[1], outputs[2]
            base_tracks = info.get("base_tracks", None)
            base_vis = info.get("base_visibility", None)
            if base_tracks is None or base_vis is None:
                continue

            pred_vis = _as_bool(pred_vis)
            base_vis = _as_bool(base_vis)

            # Subset selection
            t_q = query_points[b, :, 0].round().long().clamp(0, pred_tracks.shape[2] - 1)
            max_occ = _max_reappearance_occlusion_run(occluded[b], t_q)
            sel = max_occ >= int(threshold)
            num_sel = int(sel.long().sum().item())
            if num_sel <= 0:
                logger.info(f"{name}: no selected long-occ queries for threshold={threshold}")
                found[name] = True
                continue

            # Slice selected points
            base_tracks_sel = base_tracks[b][sel].detach().cpu().numpy()
            pred_tracks_sel = pred_tracks[b][sel].detach().cpu().numpy()
            gt_tracks_sel = target_points[b][sel].detach().cpu().numpy()
            base_vis_sel = base_vis[b][sel].detach().cpu().numpy().astype(bool)
            pred_vis_sel = pred_vis[b][sel].detach().cpu().numpy().astype(bool)
            gt_vis_sel = (~occluded[b][sel]).detach().cpu().numpy().astype(bool)

            # Video to numpy
            video_np = _to_numpy_video(video[b].detach().cpu())

            vis_dir = out_dir / "visuals" / _safe_name(name)
            _visualize_side_by_side(
                video=video_np,
                base_tracks=base_tracks_sel,
                refined_tracks=pred_tracks_sel,
                gt_tracks=gt_tracks_sel,
                base_vis=base_vis_sel,
                refined_vis=pred_vis_sel,
                gt_vis=gt_vis_sel,
                out_dir=vis_dir,
                video_name=name,
                max_points=max_points,
            )

            found[name] = True

        if all(found.values()):
            break

    missing = [n for n, ok in found.items() if not ok]
    if missing:
        logger.warning(f"Did not find videos in dataloader: {missing}")


def _write_ranking(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ranking.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    csv_path = out_dir / "ranking.csv"
    if rows:
        keys = list(rows[0].keys())
    else:
        keys = []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    logger.info(f"Wrote: {json_path}")
    logger.info(f"Wrote: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find/visualize long-occlusion examples (base vs refined).")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
    parser.add_argument("--threshold", type=int, default=20, help="Long occlusion threshold (frames).")
    parser.add_argument("--topk", type=int, default=3, help="How many top videos to visualize.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--metric-resolution-mode",
        type=str,
        default="original",
        choices=("original", "input"),
        help="Metric pixel-space resolution mode.",
    )
    parser.add_argument("--max-points", type=int, default=20, help="Max points to draw per video.")
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)

    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = _load_model(config_path=config_path, checkpoint_path=checkpoint_path, device=device)
    dataloader = _build_val_loader_from_config(config)

    query_mode = None
    if hasattr(config, "evaluation") and hasattr(config.evaluation, "query_mode"):
        qm = getattr(config.evaluation, "query_mode")
        if qm is not None and str(qm).strip().lower() not in ("", "none", "null"):
            query_mode = str(qm).strip()
    exclude_query_frame = True
    if hasattr(config, "evaluation") and hasattr(config.evaluation, "exclude_query_frame"):
        exclude_query_frame = bool(getattr(config.evaluation, "exclude_query_frame"))

    rows = _rank_videos(
        model=model,
        dataloader=dataloader,
        threshold=int(args.threshold),
        metric_resolution_mode=str(args.metric_resolution_mode),
        query_mode=query_mode,
        exclude_query_frame=exclude_query_frame,
    )
    _write_ranking(rows, out_dir)

    if not rows:
        logger.warning("No long-occlusion videos found; nothing to visualize.")
        return

    topk = max(1, int(args.topk))
    selected = [r["video_name"] for r in rows[:topk]]
    logger.info(f"Top-{topk} videos by AJ_delta@longocc{args.threshold}: {selected}")

    # Re-create loader for a fresh iteration.
    dataloader = _build_val_loader_from_config(config)
    _visualize_selected(
        model=model,
        dataloader=dataloader,
        selected_video_names=selected,
        threshold=int(args.threshold),
        metric_resolution_mode=str(args.metric_resolution_mode),
        query_mode=query_mode,
        exclude_query_frame=exclude_query_frame,
        out_dir=out_dir,
        max_points=int(args.max_points),
    )

    logger.info(f"Done. Visuals under: {out_dir / 'visuals'}")


if __name__ == "__main__":
    main()

