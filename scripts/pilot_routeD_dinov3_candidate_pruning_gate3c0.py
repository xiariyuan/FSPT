#!/usr/bin/env python3
"""Fit-only post-gate DINOv3 pruning pilot for the Gate 3A M1 top-64 bank."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_discrete_candidates import (
    extract_discrete_candidates,
)
from scripts.audit_routeD_cotracker3_interface import (
    load_manifest_sample,
    prepare_sample,
    set_deterministic,
)


def _preprocess(video: torch.Tensor, device: str) -> torch.Tensor:
    value = F.interpolate(
        video.float() / 255.0, size=(224, 224), mode="bilinear", align_corners=False
    )
    mean = value.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = value.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return ((value - mean) / std).to(device)


def _sample(feature: torch.Tensor, coordinates_xy: torch.Tensor) -> torch.Tensor:
    coordinates = coordinates_xy.to(device=feature.device, dtype=feature.dtype)
    grid = coordinates.clone()
    grid[..., 0] = 2.0 * grid[..., 0] / 255.0 - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / 255.0 - 1.0
    value = F.grid_sample(
        feature[None],
        grid[None, :, None],
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0, :, :, 0].T
    return F.normalize(value.float(), dim=-1)


def _row_zscore(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    output = torch.full_like(value, -float("inf"))
    for row in range(value.shape[0]):
        local = value[row, valid[row]]
        output[row, valid[row]] = (local - local.mean()) / local.std().clamp_min(1e-6)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-index",
        default="outputs/routeD_geometry_representation_cache_gate3a_v1_20260719/cache_index.json",
    )
    parser.add_argument(
        "--causal-root",
        default="outputs/routeD_geometry_causal_input_cache_gate3a_v1_20260719",
    )
    parser.add_argument(
        "--manifest",
        default="/gemini/code/FSPT/outputs/beliefcal_mvp1_a2_protocol_run_20260714/kubric_train/train.index.json",
    )
    parser.add_argument(
        "--model",
        default="/gemini/code/FSPT/third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    set_deterministic(193721)
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        args.model, local_files_only=True
    ).to(args.device).eval()
    register_tokens = int(getattr(model.config, "num_register_tokens", 4))
    grid_size = int(model.config.image_size) // int(model.config.patch_size)
    index_path = Path(args.candidate_index).resolve()
    index = json.loads(index_path.read_text())
    methods = {
        "M1_static": [],
        "DINO_query": [],
        "fusion_25static_75dino": [],
        "fusion_50static_50dino": [],
        "fusion_75static_25dino": [],
    }
    coarse_oracle = []
    rows = []
    with torch.no_grad():
        for entry in index["rows"]:
            source_index = int(entry["source_index"])
            frozen = torch.load(entry["sidecar"], map_location="cpu", weights_only=False)
            causal = torch.load(
                Path(args.causal_root) / f"video_{source_index:05d}.pt",
                map_location="cpu",
                weights_only=False,
            )
            video = causal["tensors"]["observed_video_u8"]
            query_frames = frozen["query_frames"].long()
            unique_frames = sorted(set(query_frames.tolist()) | {15})
            encoded = model(pixel_values=_preprocess(video[unique_frames], args.device))
            tokens = encoded.last_hidden_state[:, 1 + register_tokens :]
            feature = tokens.reshape(
                len(unique_frames), grid_size, grid_size, tokens.shape[-1]
            ).permute(0, 3, 1, 2).contiguous()
            feature_by_frame = {
                frame: feature[local] for local, frame in enumerate(unique_frames)
            }
            native = frozen["candidates"]["M1_geometry_native"][
                "candidate_coordinates_xy"
            ][:, 0].float()
            candidate = extract_discrete_candidates(
                frozen["score_maps"]["M1_geometry_native"].float(),
                native,
                top_k=64,
                nms_radius_grid_cells=3,
                local_refinement_window_grid_cells=5,
                local_softmax_temperature=0.05,
                deduplicate_radius_input_px=4.0,
                input_height=256,
                input_width=256,
            )
            coordinates = candidate["candidate_coordinates_xy"].float()
            valid = candidate["candidate_valid_mask"]
            candidate_feature = _sample(
                feature_by_frame[15], coordinates.reshape(-1, 2)
            ).reshape(coordinates.shape[0], coordinates.shape[1], -1)
            query_feature = torch.stack(
                [
                    _sample(
                        feature_by_frame[int(query_frames[row])],
                        frozen["query_coordinates_input_xy"][row : row + 1],
                    )[0]
                    for row in range(query_frames.numel())
                ]
            )
            dino_score = (candidate_feature * query_feature[:, None]).sum(dim=-1).cpu()
            static_score = candidate["candidate_scores"].float().cpu()
            dino_score[~valid] = -float("inf")
            static_score[~valid] = -float("inf")
            z_dino = _row_zscore(dino_score, valid)
            z_static = _row_zscore(static_score, valid)
            scores = {
                "M1_static": static_score,
                "DINO_query": dino_score,
                "fusion_25static_75dino": 0.25 * z_static + 0.75 * z_dino,
                "fusion_50static_50dino": 0.50 * z_static + 0.50 * z_dino,
                "fusion_75static_25dino": 0.75 * z_static + 0.25 * z_dino,
            }
            sample, _ = load_manifest_sample(Path(args.manifest), source_index)
            prepared = prepare_sample(sample, 256)
            point_indices = frozen["point_indices"].long()
            teacher_yx = prepared["gt_tracks_yx"][point_indices, 15]
            teacher = torch.stack([teacher_yx[:, 1], teacher_yx[:, 0]], dim=-1) * 255.0
            distance = torch.linalg.vector_norm(
                coordinates - teacher[:, None], dim=-1
            )
            distance[~valid] = float("inf")
            coarse_oracle.extend(distance.min(dim=1).values.tolist())
            video_row = {"source_index": source_index, "points": int(point_indices.numel())}
            for name, score in scores.items():
                # Match Gate 3A exactly: native candidate zero is mandatory and
                # eight nonnative hypotheses are retained.
                selected = torch.cat(
                    [
                        torch.zeros(
                            (score.shape[0], 1), dtype=torch.long
                        ),
                        score[:, 1:].topk(8, dim=1).indices + 1,
                    ],
                    dim=1,
                )
                nearest = distance.gather(1, selected).min(dim=1).values
                methods[name].extend(nearest.tolist())
                video_row[name] = {
                    "recall_within_12px": float((nearest <= 12.0).float().mean()),
                    "median_error_px": float(nearest.median()),
                }
            rows.append(video_row)
    summary = {}
    for name, values in methods.items():
        value = np.asarray(values, dtype=np.float64)
        summary[name] = {
            "points": int(value.size),
            "recall_within_4px": float(np.mean(value <= 4.0)),
            "recall_within_8px": float(np.mean(value <= 8.0)),
            "recall_within_12px": float(np.mean(value <= 12.0)),
            "median_nearest_error_px": float(np.median(value)),
            "mean_nearest_error_px": float(np.mean(value)),
        }
    coarse = np.asarray(coarse_oracle, dtype=np.float64)
    output = {
        "schema_version": "routeD_dinov3_candidate_pruning_pilot_gate3c0",
        "status": "completed_fit_only_post_gate_pilot",
        "claim_scope": "post-gate feasibility only; fusion weights are descriptive",
        "coarse_M1_top64_oracle": {
            "recall_within_12px": float(np.mean(coarse <= 12.0)),
            "median_nearest_error_px": float(np.median(coarse)),
        },
        "top8_methods": summary,
        "video_rows": rows,
        "locked_data": {
            "model_validation_read": False,
            "external_read": False,
        },
    }
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
