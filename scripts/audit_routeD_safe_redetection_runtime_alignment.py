#!/usr/bin/env python3
"""Audit zero-step Route-D runtime against official EvaluationPredictor."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_cotracker3_safe_redetection_runtime import (
    compute_video_fmaps,
    run_one_query_safe_redetection,
)
from projects.mmp_tracker.mmp_tracker.routeD_safe_redetection import (
    SafeRedetectionConfig,
    SafeRedetectionModel,
)


def tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(
        f"{value.dtype}|{tuple(value.shape)}|".encode() + value.numpy().tobytes()
    ).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-source-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--davis", required=True)
    ap.add_argument("--video-index", type=int, default=0)
    ap.add_argument("--query-index", type=int, default=0)
    ap.add_argument("--query-count", type=int, default=1)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    source = str(Path(args.official_source_root).resolve())
    sys.path = [row for row in sys.path if "baselines/cotracker" not in row]
    sys.path.insert(0, source)
    from cotracker.datasets.tap_vid_datasets import TapVidDataset
    from cotracker.models.build_cotracker import build_cotracker
    from cotracker.models.evaluation_predictor import EvaluationPredictor
    from cotracker.predictor import CoTrackerOnlinePredictor

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)

    dataset = TapVidDataset(
        data_root=str(Path(args.davis).resolve()),
        dataset_type="davis",
        resize_to=[256, 256],
        queried_first=True,
    )
    sample = dataset[args.video_index]
    video = sample.video.unsqueeze(0).cuda()
    query_tyx = sample.query_points[
        args.query_index : args.query_index + args.query_count
    ].unsqueeze(0).float().cuda()
    query_txy = torch.stack(
        (query_tyx[..., 0], query_tyx[..., 2], query_tyx[..., 1]), dim=-1
    )

    official_model = build_cotracker(
        args.checkpoint, offline=False, window_len=16, v2=False
    ).cuda().eval()
    official_predictor = EvaluationPredictor(
        official_model,
        grid_size=5,
        local_grid_size=8,
        single_point=True,
        n_iters=6,
        local_extent=50,
        interp_shape=(384, 512),
    ).cuda().eval()
    with torch.no_grad():
        official_tracks, official_visibility_score = official_predictor(video, query_txy)
    official_target = official_tracks[:, :, : args.query_count]
    official_visible = official_visibility_score[:, :, : args.query_count] > 0.6

    online_predictor = CoTrackerOnlinePredictor(
        checkpoint=args.checkpoint,
        offline=False,
        v2=False,
        window_len=16,
    ).cuda().eval()
    redetection = SafeRedetectionModel(
        SafeRedetectionConfig(input_height=256, input_width=256)
    ).cuda().eval()
    fmaps = compute_video_fmaps(online_predictor, video)
    runtime_rows = []
    runtime_diagnostics = []
    for query_index in range(query_txy.shape[1]):
        runtime_row = run_one_query_safe_redetection(
            online_predictor,
            redetection,
            video,
            query_txy[:, query_index : query_index + 1],
            precomputed_fmaps=fmaps,
        )
        runtime_rows.append(runtime_row)
        runtime_diagnostics.append(runtime_row.diagnostics)
    runtime_tracks = torch.cat([row.tracks_xy for row in runtime_rows], dim=2)
    runtime_visibility = torch.cat([row.visibility for row in runtime_rows], dim=2)
    diff = (runtime_tracks - official_target).abs()
    result = {
        "schema_version": "routeD_safe_redetection_runtime_alignment_v0",
        "video_index": args.video_index,
        "video_name": sample.seq_name,
        "query_index": args.query_index,
        "query_count": args.query_count,
        "query_txy": query_txy.cpu().tolist(),
        "frames": int(video.shape[1]),
        "official_track_sha256": tensor_sha(official_target),
        "runtime_track_sha256": tensor_sha(runtime_tracks),
        "official_visibility_sha256": tensor_sha(official_visible),
        "runtime_visibility_sha256": tensor_sha(runtime_visibility),
        "track_exact": bool(torch.equal(runtime_tracks, official_target)),
        "track_max_abs_px": float(diff.max().item()),
        "track_mean_abs_px": float(diff.mean().item()),
        "visibility_exact": bool(torch.equal(runtime_visibility, official_visible)),
        "visibility_disagreement_rows": int((runtime_visibility != official_visible).sum().item()),
        "runtime_diagnostics": runtime_diagnostics,
    }
    result["numerical_equivalence_tolerance_px"] = 0.02
    result["pass"] = (
        result["track_max_abs_px"] <= result["numerical_equivalence_tolerance_px"]
        and result["visibility_exact"]
        and all(row["interventions"] == 0 and row["writes"] == 0 for row in runtime_diagnostics)
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
