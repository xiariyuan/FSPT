#!/usr/bin/env python3
"""
Stage 0 diagnostic for online recovery:

Measure whether CoTracker3 fnet feature maps contain support-conditioned
re-entry signal before implementing a learned online recovery module.

This script does NOT run full tracking and does NOT train a model. It only
answers:

  "Given a support descriptor from pre-occlusion visible GT frames, does the
   current-frame CoTracker feature map assign higher similarity to the GT
   re-entry location than to random background locations?"

This isolates feature discriminability from trigger quality, tracking drift,
and verifier design.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_world_state_stage0 import discover_sequences, load_sequence
from scripts.build_prt_splits import load_image_size, classify_reentry_type
from models.cotracker_refiner import _ensure_local_cotracker_path


@dataclass
class RecoveryEvent:
    seq_name: str
    split: str
    point_idx: int
    query_frame: int
    reentry_frame: int
    occ_length: int
    camera_motion: float
    reentry_type: str
    image_height: int
    image_width: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_events_for_sequence(
    seq_path: Path,
    split_name: str,
    min_occ_length: int,
    min_camera_motion: float,
    reentry_type_filter: str,
) -> List[RecoveryEvent]:
    seq = load_sequence(seq_path)
    trajs_2d = seq["trajs_2d"]
    visibs = seq["visibs"]
    valids = seq["valids"] if "valids" in seq else np.ones_like(visibs, dtype=bool)
    extrinsics = seq["extrinsics"]
    height, width = load_image_size(seq_path)

    t_total, n_points = visibs.shape
    queries: List[RecoveryEvent] = []
    extr_inv = np.linalg.inv(extrinsics)

    for i in range(n_points):
        vis = visibs[:, i].astype(bool)
        val = valids[:, i].astype(bool)
        xy = trajs_2d[:, i]
        if not vis.any():
            continue

        run_start = -1
        last_vis_before = -1
        for t in range(t_total):
            if vis[t]:
                if run_start >= 0 and (t - run_start) >= min_occ_length and last_vis_before >= 0:
                    e_rel = extrinsics[t] @ extr_inv[last_vis_before]
                    cam_motion = float(np.linalg.norm(e_rel[:3, :3] - np.eye(3)))
                    if cam_motion < min_camera_motion:
                        last_vis_before = t
                        run_start = -1
                        continue
                    label, _, _ = classify_reentry_type(
                        trajs_2d_i=xy,
                        visibs_i=vis,
                        valids_i=val,
                        width=width,
                        height=height,
                        occ_start=run_start,
                        occ_end=t - 1,
                    )
                    if reentry_type_filter != "all" and label != reentry_type_filter:
                        last_vis_before = t
                        run_start = -1
                        continue
                    queries.append(
                        RecoveryEvent(
                            seq_name=seq_path.name,
                            split=split_name,
                            point_idx=i,
                            query_frame=last_vis_before,
                            reentry_frame=t,
                            occ_length=t - run_start,
                            camera_motion=cam_motion,
                            reentry_type=label,
                            image_height=height,
                            image_width=width,
                        )
                    )
                last_vis_before = t
                run_start = -1
            else:
                if run_start < 0:
                    run_start = t
    return queries


def extract_events(
    data_root: Path,
    splits: List[str],
    min_occ_length: int,
    min_camera_motion: float,
    reentry_type_filter: str,
    max_sequences: int,
    max_events: int,
    seed: int,
) -> Tuple[List[Tuple[Path, RecoveryEvent]], Dict]:
    sequences = discover_sequences(data_root, splits)
    if max_sequences > 0:
        sequences = sequences[:max_sequences]

    all_events: List[Tuple[Path, RecoveryEvent]] = []
    per_seq_counts: Dict[str, int] = {}
    for seq_path in sequences:
        split_name = seq_path.parent.name
        events = extract_events_for_sequence(
            seq_path=seq_path,
            split_name=split_name,
            min_occ_length=min_occ_length,
            min_camera_motion=min_camera_motion,
            reentry_type_filter=reentry_type_filter,
        )
        per_seq_counts[seq_path.name] = len(events)
        all_events.extend((seq_path, e) for e in events)

    rng = np.random.default_rng(seed)
    if max_events > 0 and len(all_events) > max_events:
        keep = rng.choice(len(all_events), size=max_events, replace=False)
        keep = sorted(int(i) for i in keep)
        all_events = [all_events[i] for i in keep]

    stats = {
        "n_sequences": len(sequences),
        "n_events_total_before_sample": int(sum(per_seq_counts.values())),
        "n_events_selected": len(all_events),
        "per_sequence_event_counts": per_seq_counts,
    }
    return all_events, stats


def load_selected_video_frames(
    seq_path: Path,
    frame_indices: List[int],
    target_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, Dict[int, int]]:
    rgb_dir = seq_path / "rgbs"
    frame_files = sorted(rgb_dir.glob("rgb_*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"No RGB frames found in {rgb_dir}")

    target_h, target_w = target_hw
    uniq = sorted(set(int(i) for i in frame_indices))
    frames = []
    remap: Dict[int, int] = {}
    for new_idx, old_idx in enumerate(uniq):
        if old_idx < 0 or old_idx >= len(frame_files):
            raise IndexError(f"Frame index {old_idx} out of range for {seq_path.name}")
        f = frame_files[old_idx]
        img = cv2.imread(str(f))
        if img is None:
            raise RuntimeError(f"Failed to read {f}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        frames.append(torch.from_numpy(img).permute(2, 0, 1))
        remap[old_idx] = new_idx
    return torch.stack(frames, dim=0).float() / 255.0, remap


def gather_normalized_descriptors(
    fmaps_bthwc: torch.Tensor,
    frame_idx: int,
    points_yx_norm: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        fmaps_bthwc: (1, T, Hf, Wf, C), normalized on channel dim
        frame_idx: int
        points_yx_norm: (N, 2) in [0,1], y/x order
    Returns:
        desc: (N, C), normalized
    """
    fmap = fmaps_bthwc[0, frame_idx].permute(2, 0, 1).unsqueeze(0)  # (1, C, Hf, Wf)
    _, c, hf, wf = fmap.shape
    y = points_yx_norm[:, 0] * 2.0 - 1.0
    x = points_yx_norm[:, 1] * 2.0 - 1.0
    grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(fmap, grid, mode="bilinear", align_corners=True)
    desc = sampled.squeeze(0).squeeze(-1).transpose(0, 1).contiguous()  # (N, C)
    desc = F.normalize(desc, dim=-1)
    return desc


def xy_to_yx_norm(xy_px: np.ndarray, image_height: int, image_width: int) -> np.ndarray:
    """Convert PointOdyssey pixel coordinates stored as (x, y) into normalized (y, x)."""
    xy_px = np.asarray(xy_px, dtype=np.float32)
    return np.asarray(
        [
            xy_px[1] / float(image_height),
            xy_px[0] / float(image_width),
        ],
        dtype=np.float32,
    )


def occ_bucket(occ_length: int) -> str:
    if occ_length >= 500:
        return "500+"
    if occ_length >= 200:
        return "200-499"
    if occ_length >= 100:
        return "100-199"
    return "20-99"


def summarize_events(events: List[Dict]) -> Dict[str, float]:
    if not events:
        return {
            "n_events": 0,
            "gt_similarity_mean": float("nan"),
            "bg_similarity_mean": float("nan"),
            "bg_similarity_max_mean": float("nan"),
            "gt_minus_bg_max_mean": float("nan"),
            "gt_minus_bg_max_median": float("nan"),
            "gt_minus_bg_max_std": float("nan"),
            "gt_beats_bg_max_frac": float("nan"),
            "rank1_frac": float("nan"),
            "pass_gt_beats_bg_max_frac": False,
            "pass_gt_minus_bg_max_mean_vs_std": False,
            "pass_rank1_frac": False,
        }

    gt_sims = np.asarray([x["gt_similarity"] for x in events], dtype=np.float32)
    bg_means = np.asarray([x["bg_similarity_mean"] for x in events], dtype=np.float32)
    bg_maxs = np.asarray([x["bg_similarity_max"] for x in events], dtype=np.float32)
    margins = np.asarray([x["gt_minus_bg_max"] for x in events], dtype=np.float32)
    rank1 = np.asarray([1.0 if x["gt_dense_rank1"] else 0.0 for x in events], dtype=np.float32)

    margin_std = float(np.std(margins))
    gt_beats_bg_max_frac = float(np.mean(rank1))
    gt_beats_bg = np.asarray([1.0 if x["gt_beats_bg_max"] else 0.0 for x in events], dtype=np.float32)
    gt_minus_bg_max_mean = float(np.mean(margins))
    gt_beats_bg_max_frac = float(np.mean(gt_beats_bg))
    rank1_frac = float(np.mean(rank1))

    return {
        "n_events": len(events),
        "gt_similarity_mean": float(np.mean(gt_sims)),
        "bg_similarity_mean": float(np.mean(bg_means)),
        "bg_similarity_max_mean": float(np.mean(bg_maxs)),
        "gt_minus_bg_max_mean": gt_minus_bg_max_mean,
        "gt_minus_bg_max_median": float(np.median(margins)),
        "gt_minus_bg_max_std": margin_std,
        "gt_beats_bg_max_frac": gt_beats_bg_max_frac,
        "rank1_frac": rank1_frac,
        "pass_gt_beats_bg_max_frac": gt_beats_bg_max_frac >= 0.65,
        "pass_gt_minus_bg_max_mean_vs_std": gt_minus_bg_max_mean >= (0.5 * margin_std),
        "pass_rank1_frac": rank1_frac >= 0.30,
    }


def load_cotracker_fnet(checkpoint: str, device: torch.device) -> Tuple[nn.Module, Tuple[int, int]]:
    _ensure_local_cotracker_path()
    from cotracker.models.build_cotracker import build_cotracker

    model = build_cotracker(checkpoint=checkpoint, offline=True, window_len=60, v2=False)
    model = model.to(device).eval()
    interp_shape = tuple(model.model_resolution)
    fnet = model.fnet
    fnet.eval()
    return fnet, interp_shape


@torch.no_grad()
def extract_fmaps_bthwc(video_b_tchw: torch.Tensor, fnet: nn.Module) -> torch.Tensor:
    """
    Args:
        video_b_tchw: (1, T, 3, H, W) in [0,1]
    Returns:
        fmaps_bthwc: (1, T, Hf, Wf, C), L2-normalized on C
    """
    b, t, c, h, w = video_b_tchw.shape
    feats = fnet(video_b_tchw.reshape(b * t, c, h, w))
    feats = feats.reshape(b, t, feats.shape[1], feats.shape[2], feats.shape[3])
    feats = feats.permute(0, 1, 3, 4, 2).contiguous()
    feats = F.normalize(feats, dim=-1)
    return feats


def random_background_points(
    gt_yx_norm: np.ndarray,
    n_bg: int,
    min_dist_norm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    pts = []
    gt = np.asarray(gt_yx_norm, dtype=np.float32)
    trials = 0
    max_trials = n_bg * 100
    while len(pts) < n_bg and trials < max_trials:
        trials += 1
        cand = np.asarray([rng.random(), rng.random()], dtype=np.float32)
        if np.linalg.norm(cand - gt) >= min_dist_norm:
            pts.append(cand)
    if not pts:
        pts.append(np.asarray([0.0, 0.0], dtype=np.float32))
    while len(pts) < n_bg:
        pts.append(pts[-1].copy())
    return np.stack(pts, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Diagnose CoTracker feature recovery signal")
    parser.add_argument("--data-root", type=str, default="/gemini/code/FSPT/datasets/pointodyssey")
    parser.add_argument("--splits", type=str, default="val")
    parser.add_argument("--min-occ-length", type=int, default=20)
    parser.add_argument("--min-camera-motion", type=float, default=0.30)
    parser.add_argument(
        "--reentry-type",
        type=str,
        default="in_frame_occlusion",
        choices=["all", "in_frame_occlusion", "offscreen_return", "mixed"],
    )
    parser.add_argument("--max-sequences", type=int, default=2)
    parser.add_argument("--max-events", type=int, default=256)
    parser.add_argument("--events-per-seq", type=int, default=32)
    parser.add_argument("--support-bank-size", type=int, default=4)
    parser.add_argument("--n-bg", type=int, default=32)
    parser.add_argument("--bg-min-dist-px", type=float, default=64.0)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--output-json",
        type=str,
        default="/gemini/code/FSPT/outputs/cotracker_feature_recovery_signal_stage0.json",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data_root = Path(args.data_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    all_events, dataset_stats = extract_events(
        data_root=data_root,
        splits=splits,
        min_occ_length=args.min_occ_length,
        min_camera_motion=args.min_camera_motion,
        reentry_type_filter=args.reentry_type,
        max_sequences=args.max_sequences,
        max_events=args.max_events,
        seed=args.seed,
    )

    # Optional per-sequence cap after global sampling.
    if args.events_per_seq > 0:
        buckets: Dict[str, List[Tuple[Path, RecoveryEvent]]] = {}
        for item in all_events:
            buckets.setdefault(item[1].seq_name, []).append(item)
        reduced = []
        for seq_name, items in buckets.items():
            if len(items) > args.events_per_seq:
                keep = rng.choice(len(items), size=args.events_per_seq, replace=False)
                keep = sorted(int(i) for i in keep)
                items = [items[i] for i in keep]
            reduced.extend(items)
        all_events = reduced
        dataset_stats["n_events_selected_after_per_seq_cap"] = len(all_events)

    if not all_events:
        raise RuntimeError("No recovery events selected.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fnet, interp_shape = load_cotracker_fnet(args.checkpoint, device)

    per_event = []
    by_sequence: Dict[str, List[float]] = {}

    grouped: Dict[str, List[RecoveryEvent]] = {}
    for _, event in all_events:
        grouped.setdefault(event.seq_name, []).append(event)

    seq_paths = {}
    for seq_path, event in all_events:
        seq_paths[event.seq_name] = seq_path

    for seq_name, events in grouped.items():
        seq_path = seq_paths[seq_name]
        seq = load_sequence(seq_path)
        trajs_2d = seq["trajs_2d"]
        img_h, img_w = load_image_size(seq_path)
        needed_frames = []
        for e in events:
            needed_frames.append(int(e.reentry_frame))
            t = e.query_frame
            count = 0
            while t >= 0 and count < args.support_bank_size:
                needed_frames.append(int(t))
                t -= 1
                count += 1
        video, frame_remap = load_selected_video_frames(seq_path, needed_frames, target_hw=interp_shape)
        if args.verbose:
            print(f"[Stage0] {seq_name}: events={len(events)} unique_frames={len(frame_remap)}")
        video = video.unsqueeze(0).to(device)
        with torch.no_grad():
            fmaps = extract_fmaps_bthwc(video, fnet)

        for e in events:
            support_frames = []
            t = e.query_frame
            while t >= 0 and len(support_frames) < args.support_bank_size:
                support_frames.append(t)
                t -= 1
            support_frames = sorted(support_frames)

            support_desc_list = []
            for sf in support_frames:
                xy = trajs_2d[sf, e.point_idx].astype(np.float32)
                yx_norm = xy_to_yx_norm(xy, image_height=img_h, image_width=img_w)
                support_desc = gather_normalized_descriptors(
                    fmaps_bthwc=fmaps,
                    frame_idx=frame_remap[sf],
                    points_yx_norm=torch.from_numpy(yx_norm[None]).to(device),
                )[0]
                support_desc_list.append(support_desc)
            if support_desc_list:
                support_desc = torch.stack(support_desc_list, dim=0).mean(dim=0)
                support_desc = F.normalize(support_desc, dim=0)
            else:
                xy = trajs_2d[e.query_frame, e.point_idx].astype(np.float32)
                yx_norm = xy_to_yx_norm(xy, image_height=img_h, image_width=img_w)
                support_desc = gather_normalized_descriptors(
                    fmaps_bthwc=fmaps,
                    frame_idx=frame_remap[e.query_frame],
                    points_yx_norm=torch.from_numpy(yx_norm[None]).to(device),
                )[0]

            gt_xy = trajs_2d[e.reentry_frame, e.point_idx].astype(np.float32)
            gt_yx_norm = xy_to_yx_norm(gt_xy, image_height=img_h, image_width=img_w)
            min_dist_norm = float(args.bg_min_dist_px) / float(max(img_h, img_w))
            bg = random_background_points(
                gt_yx_norm=gt_yx_norm,
                n_bg=args.n_bg,
                min_dist_norm=min_dist_norm,
                rng=rng,
            )

            gt_desc = gather_normalized_descriptors(
                fmaps_bthwc=fmaps,
                frame_idx=frame_remap[e.reentry_frame],
                points_yx_norm=torch.from_numpy(gt_yx_norm[None]).to(device),
            )[0]
            reentry_fmap = fmaps[0, frame_remap[e.reentry_frame]]  # (Hf, Wf, C)
            dense_sim = torch.matmul(reentry_fmap, support_desc)  # (Hf, Wf)
            flat_idx = int(torch.argmax(dense_sim).item())
            hf, wf = dense_sim.shape
            pred_y = flat_idx // wf
            pred_x = flat_idx % wf

            gt_feat_y = int(np.clip(round(float(gt_yx_norm[0]) * (hf - 1)), 0, hf - 1))
            gt_feat_x = int(np.clip(round(float(gt_yx_norm[1]) * (wf - 1)), 0, wf - 1))
            gt_dense_rank1 = bool(pred_y == gt_feat_y and pred_x == gt_feat_x)

            bg_desc = gather_normalized_descriptors(
                fmaps_bthwc=fmaps,
                frame_idx=frame_remap[e.reentry_frame],
                points_yx_norm=torch.from_numpy(bg).to(device),
            )

            gt_sim = float(torch.dot(support_desc, gt_desc).item())
            bg_sims = torch.matmul(bg_desc, support_desc.unsqueeze(-1)).squeeze(-1).detach().cpu().numpy()
            bg_mean = float(np.mean(bg_sims))
            bg_max = float(np.max(bg_sims))
            rank1 = bool(gt_sim > bg_max)
            margin = gt_sim - bg_max

            by_sequence.setdefault(seq_name, []).append(margin)
            per_event.append(
                {
                    **asdict(e),
                    "occ_bucket": occ_bucket(e.occ_length),
                    "support_frames": support_frames,
                    "gt_similarity": gt_sim,
                    "bg_similarity_mean": bg_mean,
                    "bg_similarity_max": bg_max,
                    "gt_dense_rank1": gt_dense_rank1,
                    "gt_dense_rank1_pred_feat_yx": [int(pred_y), int(pred_x)],
                    "gt_dense_rank1_gt_feat_yx": [int(gt_feat_y), int(gt_feat_x)],
                    "gt_minus_bg_max": margin,
                    "gt_beats_bg_max": rank1,
                }
            )

    seq_margin = {k: float(np.mean(v)) for k, v in by_sequence.items()}
    summary = summarize_events(per_event)
    summary["per_sequence_margin_mean"] = seq_margin

    by_occ_bucket = {}
    for bucket_name in ("20-99", "100-199", "200-499", "500+"):
        bucket_events = [x for x in per_event if x["occ_bucket"] == bucket_name]
        by_occ_bucket[bucket_name] = summarize_events(bucket_events)

    pass_flags = [
        summary["pass_gt_beats_bg_max_frac"],
        summary["pass_gt_minus_bg_max_mean_vs_std"],
        summary["pass_rank1_frac"],
    ]
    if all(pass_flags):
        decision = "pass"
    elif not any(pass_flags):
        decision = "fail"
    else:
        decision = "warning"
    summary["stage0_decision"] = decision

    out = {
        "config": vars(args),
        "dataset_stats": dataset_stats,
        "summary": summary,
        "thresholds": {
            "gt_beats_bg_max_frac_min": 0.65,
            "gt_minus_bg_max_mean_min_ratio_to_margin_std": 0.5,
            "rank1_frac_min": 0.30,
            "decision_rule": {
                "pass": "all three thresholds pass",
                "warning": "some but not all thresholds pass",
                "fail": "none of the thresholds pass",
            },
        },
        "by_occ_bucket": by_occ_bucket,
        "per_event": per_event,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
