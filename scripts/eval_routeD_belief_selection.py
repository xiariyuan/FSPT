#!/usr/bin/env python3
"""Development-only Route-D belief selection evaluation.

Evaluates frozen trajectory-selection policies on top of the existing MMP
candidate set. The model and checkpoint are not modified. DAVIS is treated as a
development/diagnostic set, not as an untouched final test set.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import compute_tapvid_metrics
from projects.mmp_tracker.mmp_tracker import MMPTracker
from projects.mmp_tracker.train_mmp import (
    config_from_dict,
    resolve_dataset,
    resolve_eval_query_mode,
    resolve_eval_resolution,
)


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _gather_candidates(candidates: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather_index = indices.unsqueeze(-1).unsqueeze(-1).expand(
        *indices.shape, 1, candidates.shape[-1]
    )
    return candidates.gather(-2, gather_index).squeeze(-2)


def _error_px(
    tracks: torch.Tensor,
    target_points: torch.Tensor,
    video: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    scale = torch.tensor(
        [max(video.shape[-2] - 1, 1), max(video.shape[-1] - 1, 1)],
        device=tracks.device,
        dtype=tracks.dtype,
    )
    error = torch.norm((tracks - target_points) * scale, dim=-1)
    selected = error[mask]
    return float(selected.mean()) if selected.numel() else float("nan")


def _policy_tracks(
    current_tracks: torch.Tensor,
    candidates: torch.Tensor,
    weights: torch.Tensor,
    policy: str,
    margin: float = 0.0,
) -> torch.Tensor:
    top_weight, top_index = weights.max(dim=-1)
    map_tracks = _gather_candidates(candidates, top_index)
    if policy == "belief_map":
        return map_tracks
    if policy == "local":
        return candidates[..., 0, :]
    if policy == "global_margin":
        local_weight = weights[..., 0]
        use_map = (top_index > 0) & ((top_weight - local_weight) >= float(margin))
        return torch.where(use_map.unsqueeze(-1), map_tracks, current_tracks)
    raise ValueError(f"Unsupported policy: {policy}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import yaml

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg.setdefault("model", {}).setdefault("tracking", {})[
        "enable_multi_hypothesis_diagnostics"
    ] = True
    model = MMPTracker(config_from_dict(cfg))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    dataset, _ = resolve_dataset(cfg, args.split, False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    margins = [0.00, 0.05, 0.10, 0.20, 0.30]
    policy_names = ["current", "local", "belief_map"] + [
        f"belief_global_margin_{margin:.2f}" for margin in margins
    ]
    accumulators: Dict[str, Dict[str, List[float]]] = {
        name: {} for name in policy_names
    }
    per_video: List[Dict[str, object]] = []

    with torch.no_grad():
        for sample_index, batch in enumerate(loader):
            if args.limit is not None and sample_index >= args.limit:
                break
            video = batch["video"].to(device)
            query_points = batch["query_points"].to(device)
            target_points = batch["target_points"].to(device)
            occluded = batch["occluded"].to(device).bool()
            current_tracks, pred_visibility, info = model(
                video, query_points, return_info=True
            )
            candidates = torch.cat(
                [info["local_points"].unsqueeze(-2), info["global_candidate_points"]],
                dim=-2,
            )
            weights = info["belief_weights"]
            if candidates.shape[:-1] != weights.shape + (2,)[:-1]:
                # Explicit shape check below is easier to understand than relying
                # on broadcasting failures later.
                pass
            if weights.shape != candidates.shape[:-1]:
                raise RuntimeError(
                    f"Belief/candidate shape mismatch: {weights.shape} vs {candidates.shape}"
                )

            policy_tracks: Dict[str, torch.Tensor] = {
                "current": current_tracks,
                "local": _policy_tracks(current_tracks, candidates, weights, "local"),
                "belief_map": _policy_tracks(
                    current_tracks, candidates, weights, "belief_map"
                ),
            }
            for margin in margins:
                policy_tracks[f"belief_global_margin_{margin:.2f}"] = _policy_tracks(
                    current_tracks,
                    candidates,
                    weights,
                    "global_margin",
                    margin=margin,
                )

            eval_query_mode = resolve_eval_query_mode(batch, dataset)
            resolution = resolve_eval_resolution(batch, video, 0)
            frame_ids = torch.arange(
                target_points.shape[2], device=device
            ).view(1, 1, -1)
            query_t = query_points[..., 0].round().long().clamp(
                0, target_points.shape[2] - 1
            )
            visible_eval = (frame_ids > query_t.unsqueeze(-1)) & (~occluded)

            video_name = str(batch.get("video_name", [f"sample_{sample_index}"])[0])
            video_row: Dict[str, object] = {"video_name": video_name}
            for name, tracks in policy_tracks.items():
                metrics = compute_tapvid_metrics(
                    tracks[0],
                    target_points[0],
                    pred_visibility[0],
                    ~occluded[0],
                    query_points[0],
                    resolution=resolution,
                    exclude_query_frame=True,
                    query_mode=eval_query_mode,
                )
                normalized = {
                    "AJ": float(metrics.get("AJ", metrics.get("average_jaccard", 0.0))),
                    "OA": float(metrics.get("OA", metrics.get("occlusion_accuracy", 0.0))),
                    "delta_avg": float(
                        metrics.get("<avg", metrics.get("average_pts_within_thresh", 0.0))
                    ),
                    "avg_error_visible_px": _error_px(
                        tracks, target_points, video, visible_eval
                    ),
                }
                video_row[name] = normalized
                for metric_name, value in normalized.items():
                    accumulators[name].setdefault(metric_name, []).append(value)

            map_index = weights.argmax(dim=-1)
            video_row["belief_selects_global_rate"] = float((map_index > 0).float().mean())
            video_row["mean_top1_weight"] = float(weights.max(dim=-1).values.mean())
            per_video.append(video_row)
            print(
                f"[{sample_index + 1}] {video_name}: "
                f"AJ current={video_row['current']['AJ']:.4f}, "
                f"belief_map={video_row['belief_map']['AJ']:.4f}"
            )

    aggregate = {
        policy: {metric: _mean(values) for metric, values in metrics.items()}
        for policy, metrics in accumulators.items()
    }
    current = aggregate["current"]
    deltas = {
        policy: {
            metric: float(values[metric] - current[metric])
            for metric in ["AJ", "OA", "delta_avg", "avg_error_visible_px"]
        }
        for policy, values in aggregate.items()
        if policy != "current"
    }
    result = {
        "evidence_status": "development_diagnostic_only",
        "paper_claim_eligible": False,
        "dataset_note": (
            "DAVIS is reused for development diagnostics; these results cannot be "
            "reported as an untouched final-test comparison."
        ),
        "config": str(config_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "device": str(device),
        "samples_processed": len(per_video),
        "candidate_count": (
            int(candidates.shape[-2]) if per_video else 0
        ),
        "aggregate": aggregate,
        "delta_vs_current": deltas,
        "per_video": per_video,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_video"}, indent=2))


if __name__ == "__main__":
    main()
