#!/usr/bin/env python3
"""Build training data for ReEntry-VisCalibrator.

The dataset is built from existing unified caches.  Each sample is a candidate
re-entry window found by prediction-only base/override visibility rules.  The
label asks a metric-aware question:

    Is it safe to mark this frame visible while keeping the base coordinates?

So labels require GT visible AND base coordinate proximity.  GT is used only for
training/evaluation labels, never for inference triggers.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.reentry_viscalibrator_features import (  # noqa: E402
    FEATURE_NAMES,
    CandidateConfig,
    LabelConfig,
    check_alignment,
    iter_candidate_examples,
)

DEFAULT_BASE = "outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_offline_rgb_stacking_10video.pt"
DEFAULT_OVERRIDE = "outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_online_rgb_stacking_10video.pt"
DEFAULT_OUT = "outputs/paper_discovery_2026-06-27/reentry_viscalibrator/datasets/rgb_dev10_w16p2_candidates.npz"


def load_cache(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build ReEntry-VisCalibrator candidate-window dataset.")
    ap.add_argument("--base-cache", action="append", default=None, help="Base/offline cache. Can be repeated.")
    ap.add_argument("--override-cache", action="append", default=None, help="Override/online cache. Can be repeated.")
    ap.add_argument("--setting", action="append", default=None, help="Setting name for each cache pair. Can be repeated.")
    ap.add_argument("--out-npz", default=DEFAULT_OUT)
    ap.add_argument("--out-manifest", default="")
    ap.add_argument("--context-before", type=int, default=16)
    ap.add_argument("--context-after", type=int, default=16)
    ap.add_argument("--candidate-pre", type=int, default=1)
    ap.add_argument("--candidate-post", type=int, default=16)
    ap.add_argument("--trigger-k", type=int, default=1)
    ap.add_argument("--trigger-persist", type=int, default=2)
    ap.add_argument("--val-fraction", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--max-examples", type=int, default=0, help="Optional cap for smoke builds.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base_paths = args.base_cache or [DEFAULT_BASE]
    override_paths = args.override_cache or [DEFAULT_OVERRIDE]
    settings = args.setting or ["rgb_dev10_natural"]
    if not (len(base_paths) == len(override_paths) == len(settings)):
        raise ValueError("--base-cache, --override-cache, and --setting must have the same count")

    random.seed(args.seed)
    np.random.seed(args.seed)

    candidate_cfg = CandidateConfig(
        context_before=args.context_before,
        context_after=args.context_after,
        candidate_pre=args.candidate_pre,
        candidate_post=args.candidate_post,
        trigger_k=args.trigger_k,
        trigger_persist=args.trigger_persist,
    )
    label_cfg = LabelConfig()

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    yh: List[np.ndarray] = []
    valid_masks: List[np.ndarray] = []
    loss_masks: List[np.ndarray] = []
    setting_ids: List[int] = []
    record_indices: List[int] = []
    query_indices: List[int] = []
    trigger_ts: List[int] = []
    query_ts: List[int] = []
    video_ids: List[str] = []

    per_setting = []
    stopped = False
    for sid, (setting, base_path, override_path) in enumerate(zip(settings, base_paths, override_paths)):
        print({"loading": setting, "base": base_path, "override": override_path}, flush=True)
        base = load_cache(base_path)
        override = load_cache(override_path)
        check_alignment(base, override, setting)
        before = len(xs)
        for ri, (br, orr) in enumerate(zip(base["records"], override["records"])):
            for ex in iter_candidate_examples(br, orr, candidate_cfg, label_cfg):
                xs.append(ex["x"])
                ys.append(ex["y_soft"])
                yh.append(ex["y_hard"])
                valid_masks.append(ex["valid_mask"])
                loss_masks.append(ex["loss_mask"])
                setting_ids.append(sid)
                record_indices.append(ri)
                query_indices.append(int(ex["query_idx"]))
                trigger_ts.append(int(ex["trigger_t"]))
                query_ts.append(int(ex["query_t"]))
                video_ids.append(str(ex["video_id"]))
                if args.max_examples and len(xs) >= int(args.max_examples):
                    stopped = True
                    break
            if stopped:
                break
        per_setting.append({"setting": setting, "examples": len(xs) - before, "base_cache": base_path, "override_cache": override_path})
        if stopped:
            break

    if not xs:
        raise RuntimeError("No candidate examples generated")

    x = np.stack(xs).astype(np.float32)
    y_soft = np.stack(ys).astype(np.float32)
    y_hard = np.stack(yh).astype(np.float32)
    valid_mask = np.stack(valid_masks).astype(bool)
    loss_mask = np.stack(loss_masks).astype(bool)
    setting_ids_a = np.asarray(setting_ids, dtype=np.int32)
    record_indices_a = np.asarray(record_indices, dtype=np.int32)
    query_indices_a = np.asarray(query_indices, dtype=np.int32)
    trigger_ts_a = np.asarray(trigger_ts, dtype=np.int32)
    query_ts_a = np.asarray(query_ts, dtype=np.int32)
    video_ids_a = np.asarray(video_ids, dtype=object)

    unique_videos = sorted(set(video_ids))
    rng = random.Random(args.seed)
    shuffled = unique_videos[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * float(args.val_fraction)))) if len(shuffled) > 1 else 0
    val_videos = set(shuffled[:n_val])
    split = np.asarray([1 if v in val_videos else 0 for v in video_ids], dtype=np.int32)  # 0=train, 1=val
    if np.all(split == 1) and len(split) > 0:
        split[0] = 0
    if np.all(split == 0) and len(unique_videos) > 1:
        split[np.where(video_ids_a == shuffled[0])[0]] = 1

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        x=x,
        y_soft=y_soft,
        y_hard=y_hard,
        valid_mask=valid_mask,
        loss_mask=loss_mask,
        split=split,
        setting_ids=setting_ids_a,
        record_indices=record_indices_a,
        query_indices=query_indices_a,
        trigger_ts=trigger_ts_a,
        query_ts=query_ts_a,
        video_ids=video_ids_a,
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
        setting_names=np.asarray(settings, dtype=object),
    )

    train_mask = split == 0
    val_mask = split == 1
    def masked_rate(arr: np.ndarray, sample_mask: np.ndarray) -> float | None:
        frame_mask = loss_mask & sample_mask[:, None]
        if not np.any(frame_mask):
            return None
        return round(float(arr[frame_mask].mean()), 6)

    manifest = {
        "out_npz": str(out_npz),
        "n_examples": int(x.shape[0]),
        "sequence_len": int(x.shape[1]),
        "feature_dim": int(x.shape[2]),
        "feature_names": FEATURE_NAMES,
        "candidate_config": candidate_cfg.__dict__,
        "label_config": {
            "coord_thresholds": list(label_cfg.coord_thresholds),
            "coord_values": list(label_cfg.coord_values),
            "label_mean_all_loss_frames": masked_rate(y_soft, np.ones(x.shape[0], dtype=bool)),
            "hard_positive_rate_all_loss_frames": masked_rate(y_hard, np.ones(x.shape[0], dtype=bool)),
        },
        "split": {
            "train_examples": int(train_mask.sum()),
            "val_examples": int(val_mask.sum()),
            "train_videos": sorted(set(video_ids_a[train_mask].tolist())) if train_mask.any() else [],
            "val_videos": sorted(set(video_ids_a[val_mask].tolist())) if val_mask.any() else [],
            "val_fraction_requested": float(args.val_fraction),
            "seed": int(args.seed),
        },
        "per_setting": per_setting,
        "notes": [
            "Labels are safe-visible labels: GT visible plus base coordinate proximity.",
            "Split is video-level, not query-level, to reduce leakage.",
            "GT is used only for labels/evaluation, not inference triggers.",
        ],
    }
    out_manifest = Path(args.out_manifest) if args.out_manifest else out_npz.with_suffix(".manifest.json")
    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "out_npz": str(out_npz),
        "out_manifest": str(out_manifest),
        "n_examples": manifest["n_examples"],
        "sequence_len": manifest["sequence_len"],
        "feature_dim": manifest["feature_dim"],
        "train_examples": manifest["split"]["train_examples"],
        "val_examples": manifest["split"]["val_examples"],
        "hard_positive_rate": manifest["label_config"]["hard_positive_rate_all_loss_frames"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
