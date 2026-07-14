#!/usr/bin/env python3
"""Route-D candidate-oracle and belief diagnostic analysis.

This is an engineering/development diagnostic. It measures whether a frozen MMP
checkpoint already generates useful alternative local/global hypotheses. It does
not modify model weights, trajectory outputs, or visibility outputs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker import MMPTracker
from projects.mmp_tracker.train_mmp import config_from_dict, resolve_dataset


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().flatten()
    y = y.float().flatten()
    mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.square().sum()) * (y.square().sum()))
    if float(denom) <= 0.0:
        return float("nan")
    return float((x * y).sum() / denom)


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    values = values.float().flatten()
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    n = values.numel()
    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().flatten()
    y = y.float().flatten()
    mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.numel() < 2:
        return float("nan")
    return _pearson(_average_ranks(x), _average_ranks(y))


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    return float(selected.mean()) if selected.numel() else float("nan")


def _coordinate_scale(video: torch.Tensor) -> torch.Tensor:
    # Tracker coordinates are normalized and ordered as [y, x].
    return torch.tensor(
        [max(video.shape[-2] - 1, 1), max(video.shape[-1] - 1, 1)],
        device=video.device,
        dtype=video.dtype,
    )


def _active_mask(query_points: torch.Tensor, time: int) -> torch.Tensor:
    query_t = query_points[..., 0].round().long().clamp(0, time - 1)
    frame_ids = torch.arange(time, device=query_points.device).view(1, 1, time)
    return frame_ids >= query_t.unsqueeze(-1)


def _pairwise_diversity_px(candidates: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # candidates: B,N,T,K,2 -> mean off-diagonal pairwise distance per row.
    k = candidates.shape[-2]
    if k <= 1:
        return torch.zeros(candidates.shape[:-2], device=candidates.device, dtype=candidates.dtype)
    delta = (candidates.unsqueeze(-2) - candidates.unsqueeze(-3)) * scale
    distances = torch.norm(delta, dim=-1)
    upper = torch.triu(
        torch.ones(k, k, device=candidates.device, dtype=torch.bool), diagonal=1
    )
    return distances[..., upper].mean(dim=-1)


def _candidate_error_px(
    candidates: torch.Tensor,
    target_points: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return torch.norm((candidates - target_points.unsqueeze(-2)) * scale, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import yaml

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = MMPTracker(config_from_dict(cfg))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    dataset, _ = resolve_dataset(cfg, args.split, False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_entropy: List[torch.Tensor] = []
    all_current_error: List[torch.Tensor] = []
    all_oracle_gain: List[torch.Tensor] = []
    per_video: List[Dict[str, object]] = []
    k_accumulator: Dict[int, Dict[str, List[float]]] = {}
    samples_processed = 0

    with torch.no_grad():
        for index, batch in enumerate(loader):
            if args.limit is not None and index >= args.limit:
                break
            video = batch["video"].to(device)
            query_points = batch["query_points"].to(device)
            target_points = batch["target_points"].to(device)
            occluded = batch["occluded"].to(device).bool()
            tracks, _, info = model(video, query_points, return_info=True)

            local_points = info["local_points"]
            global_points = info["global_candidate_points"]
            candidates = torch.cat([local_points.unsqueeze(-2), global_points], dim=-2)
            scale = _coordinate_scale(video)
            candidate_error = _candidate_error_px(candidates, target_points, scale)
            current_error = torch.norm((tracks - target_points) * scale, dim=-1)
            local_error = candidate_error[..., 0]
            oracle_error, oracle_index = candidate_error.min(dim=-1)
            oracle_gain = current_error - oracle_error
            global_best_error = candidate_error[..., 1:].min(dim=-1).values
            active = _active_mask(query_points, target_points.shape[2])
            visible = active & (~occluded)
            occluded_active = active & occluded
            diversity = _pairwise_diversity_px(candidates, scale)

            entropy = info.get("belief_entropy")
            effective = info.get("belief_effective_hypotheses")
            if isinstance(entropy, torch.Tensor):
                all_entropy.append(entropy[active].detach().cpu())
                all_current_error.append(current_error[active].detach().cpu())
                all_oracle_gain.append(oracle_gain[active].detach().cpu())

            k_values = sorted(set([1, 2, 4, candidates.shape[-2]]))
            for k in k_values:
                k = min(k, candidates.shape[-2])
                error_k = candidate_error[..., :k].min(dim=-1).values
                entry = k_accumulator.setdefault(
                    k,
                    {
                        "visible_oracle_error_px": [],
                        "active_oracle_error_px": [],
                        "visible_gain_vs_current_px": [],
                    },
                )
                entry["visible_oracle_error_px"].append(_masked_mean(error_k, visible))
                entry["active_oracle_error_px"].append(_masked_mean(error_k, active))
                entry["visible_gain_vs_current_px"].append(
                    _masked_mean(current_error - error_k, visible)
                )

            video_name = batch.get("video_name", [f"sample_{index}"])[0]
            better_mask = (global_best_error < local_error) & visible
            video_result: Dict[str, object] = {
                "video_name": str(video_name),
                "points": int(target_points.shape[1]),
                "frames": int(target_points.shape[2]),
                "candidate_count": int(candidates.shape[-2]),
                "visible_rows": int(visible.sum()),
                "active_rows": int(active.sum()),
                "current_error_visible_px": _masked_mean(current_error, visible),
                "local_error_visible_px": _masked_mean(local_error, visible),
                "oracle_error_visible_px": _masked_mean(oracle_error, visible),
                "oracle_gain_visible_px": _masked_mean(oracle_gain, visible),
                "oracle_error_occluded_px": _masked_mean(oracle_error, occluded_active),
                "global_better_than_local_visible_rate": (
                    float(better_mask.sum()) / max(int(visible.sum()), 1)
                ),
                "oracle_selects_global_visible_rate": (
                    float(((oracle_index > 0) & visible).sum()) / max(int(visible.sum()), 1)
                ),
                "candidate_diversity_visible_px": _masked_mean(diversity, visible),
                "belief_entropy_active": (
                    _masked_mean(entropy, active) if isinstance(entropy, torch.Tensor) else float("nan")
                ),
                "effective_hypotheses_active": (
                    _masked_mean(effective, active) if isinstance(effective, torch.Tensor) else float("nan")
                ),
            }
            per_video.append(video_result)
            samples_processed += 1
            print(
                f"[{samples_processed}] {video_name}: "
                f"current={video_result['current_error_visible_px']:.3f}px, "
                f"oracle={video_result['oracle_error_visible_px']:.3f}px, "
                f"gain={video_result['oracle_gain_visible_px']:.3f}px"
            )

    entropy_flat = torch.cat(all_entropy) if all_entropy else torch.empty(0)
    current_error_flat = torch.cat(all_current_error) if all_current_error else torch.empty(0)
    oracle_gain_flat = torch.cat(all_oracle_gain) if all_oracle_gain else torch.empty(0)

    summary = {
        "evidence_status": "development_diagnostic_only",
        "paper_claim_eligible": False,
        "dataset_note": (
            "DAVIS has already been used for development and is not an untouched final test set."
        ),
        "config": str(config_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "device": str(device),
        "samples_processed": samples_processed,
        "available_candidate_count": (
            int(per_video[0]["candidate_count"]) if per_video else 0
        ),
        "aggregate": {
            key: _mean(float(item[key]) for item in per_video)
            for key in [
                "current_error_visible_px",
                "local_error_visible_px",
                "oracle_error_visible_px",
                "oracle_gain_visible_px",
                "oracle_error_occluded_px",
                "global_better_than_local_visible_rate",
                "oracle_selects_global_visible_rate",
                "candidate_diversity_visible_px",
                "belief_entropy_active",
                "effective_hypotheses_active",
            ]
        },
        "correlations_active_rows": {
            "entropy_vs_current_error_pearson": _pearson(entropy_flat, current_error_flat),
            "entropy_vs_current_error_spearman": _spearman(entropy_flat, current_error_flat),
            "entropy_vs_oracle_gain_pearson": _pearson(entropy_flat, oracle_gain_flat),
            "entropy_vs_oracle_gain_spearman": _spearman(entropy_flat, oracle_gain_flat),
        },
        "k_ablation": {
            str(k): {metric: _mean(values) for metric, values in metrics.items()}
            for k, metrics in sorted(k_accumulator.items())
        },
        "per_video": per_video,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_video"}, indent=2))


if __name__ == "__main__":
    main()
