#!/usr/bin/env python3
"""
Preprocess Kubric TFDS (MOVi-E) into sharded pickle files.

Why this script:
- Avoid expensive on-the-fly flow->track generation during training.
- Keep memory usage bounded by writing fixed-size shards.
- Produce a JSON manifest consumable by TAPVidKubricShardedIterableDataset.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import tensorflow as tf  # noqa: F401
    import tensorflow_datasets as tfds
except Exception as exc:
    raise RuntimeError(
        "This script requires tensorflow and tensorflow-datasets. "
        "Install them in your training environment first."
    ) from exc

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Ensure project root is importable when launched as `python scripts/...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the same logic as runtime TFDS dataset to keep labels consistent.
from datasets.tapvid_kubric import (  # noqa: E402
    _build_tfds_read_config,
    _canonical_tfds_split,
    _decode_kubric_flow,
    _generate_sparse_tracks_from_kubric,
    _load_tfds_dataset,
)


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess Kubric TFDS to sharded pickle")
    p.add_argument(
        "--tfds-root",
        type=str,
        required=True,
        help="Root containing Kubric TFDS files (e.g., /gemini/code/datasets/tapvid_kubric)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for shards and manifest",
    )
    p.add_argument("--split", type=str, default="train", choices=["train", "validation", "test", "val"])
    p.add_argument("--tfds-name", type=str, default="movi_e/256x256")
    p.add_argument("--num-points", type=int, default=256)
    p.add_argument("--num-frames", type=int, default=24)
    p.add_argument("--resolution", type=int, nargs=2, default=[256, 256], metavar=("H", "W"))
    p.add_argument("--shard-size", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--sampling-strategy", type=str, default="uniform")
    p.add_argument("--hard-fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle-files", action="store_true", help="Enable TFDS file-level shuffle when reading")
    p.add_argument(
        "--read-buffer-size",
        type=int,
        default=1 << 20,
        help="TFDS read buffer size in bytes (default: 1MiB)",
    )
    p.add_argument(
        "--low-memory",
        action="store_true",
        help="Use TFDS low-memory read config (recommended for cgroup-limited envs)",
    )
    return p.parse_args()


def resize_video(video: np.ndarray, resolution: Tuple[int, int]) -> np.ndarray:
    """Resize (T,H,W,3) uint8 video to target resolution with bilinear interpolation."""
    h_new, w_new = int(resolution[0]), int(resolution[1])
    if video.shape[1] == h_new and video.shape[2] == w_new:
        return video
    # Use TensorFlow resize to avoid extra dependencies.
    video_f = tf.convert_to_tensor(video, dtype=tf.float32)
    resized = tf.image.resize(video_f, size=[h_new, w_new], method="bilinear", antialias=True)
    resized = tf.clip_by_value(resized, 0.0, 255.0)
    return tf.cast(tf.round(resized), tf.uint8).numpy()


def convert_sample(
    sample: Dict,
    *,
    num_points: int,
    num_frames: int,
    resolution: Optional[Tuple[int, int]],
    sampling_strategy: str,
    hard_fraction: float,
    rng: np.random.RandomState,
) -> Dict:
    video = np.asarray(sample["video"])  # (T,H,W,3)
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"Invalid TFDS video shape: {video.shape}")

    metadata = sample.get("metadata", None)
    ff_range = None
    bf_range = None
    if isinstance(metadata, dict):
        ff_range = metadata.get("forward_flow_range", None)
        bf_range = metadata.get("backward_flow_range", None)

    forward_flow = _decode_kubric_flow(sample.get("forward_flow", None), ff_range)
    backward_flow = _decode_kubric_flow(sample.get("backward_flow", None), bf_range)
    segmentations = sample.get("segmentations", None)

    # Optional temporal windowing (MOVi-E is usually already 24).
    t_full = int(video.shape[0])
    if num_frames is not None and int(num_frames) > 0 and t_full > int(num_frames):
        start = int(rng.randint(0, t_full - int(num_frames) + 1))
        end = start + int(num_frames)
        video = video[start:end]
        forward_flow = forward_flow[start:end]
        backward_flow = backward_flow[start:end]
        if segmentations is not None:
            segmentations = np.asarray(segmentations)[start:end]

    query_points, target_points, occluded = _generate_sparse_tracks_from_kubric(
        forward_flow=forward_flow,
        backward_flow=backward_flow,
        segmentations=segmentations,
        num_points=int(num_points),
        rng=rng,
        sampling_strategy=str(sampling_strategy),
        hard_fraction=float(hard_fraction),
    )

    if resolution is not None:
        video = resize_video(video, resolution)

    return {
        "video": np.asarray(video, dtype=np.uint8),
        "query_points": np.asarray(query_points, dtype=np.float32),
        "target_points": np.asarray(target_points, dtype=np.float32),
        "occluded": np.asarray(occluded, dtype=bool),
    }


def main():
    args = parse_args()

    tfds_root = Path(args.tfds_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = _canonical_tfds_split(str(args.split))
    read_config = _build_tfds_read_config(
        low_memory=bool(args.low_memory),
        deterministic=True,
        shuffle_seed=None,
        override_buffer_size=int(args.read_buffer_size) if args.read_buffer_size else None,
    )
    ds, source = _load_tfds_dataset(
        tfds_name=str(args.tfds_name),
        split=split,
        root=tfds_root,
        shuffle_files=bool(args.shuffle_files),
        read_config=read_config,
    )
    np_ds = tfds.as_numpy(ds)

    shard_size = max(1, int(args.shard_size))
    rng_seed = int(args.seed)
    max_samples = int(args.max_samples) if args.max_samples is not None else None
    resolution = tuple(args.resolution) if args.resolution is not None else None

    shard_items = []
    shard_meta = []
    processed = 0
    shard_idx = 0

    def flush_shard():
        nonlocal shard_items, shard_idx
        if not shard_items:
            return
        shard_name = f"{split}_{shard_idx:05d}.pkl"
        shard_path = out_dir / shard_name
        with open(shard_path, "wb") as f:
            pickle.dump(shard_items, f, protocol=4)
        shard_meta.append({"path": shard_name, "num_samples": len(shard_items)})
        shard_items = []
        shard_idx += 1

    iterator = np_ds
    if tqdm is not None:
        iterator = tqdm(iterator, total=max_samples, desc=f"preprocess:{split}")

    for i, sample in enumerate(iterator):
        if max_samples is not None and processed >= max_samples:
            break
        rng = np.random.RandomState((rng_seed + i * 1013) % (2**32))
        converted = convert_sample(
            sample,
            num_points=int(args.num_points),
            num_frames=int(args.num_frames),
            resolution=resolution,
            sampling_strategy=str(args.sampling_strategy),
            hard_fraction=float(args.hard_fraction),
            rng=rng,
        )
        converted["video_name"] = f"kubric_{split}_{i:07d}"
        shard_items.append(converted)
        processed += 1

        if len(shard_items) >= shard_size:
            flush_shard()

    flush_shard()

    manifest = {
        "version": 1,
        "dataset": "tapvid_kubric",
        "source": f"tfds:{source}",
        "split": split,
        "tfds_name": str(args.tfds_name),
        "num_samples": int(processed),
        "num_shards": int(len(shard_meta)),
        "num_points": int(args.num_points),
        "num_frames": int(args.num_frames),
        "resolution": list(resolution) if resolution is not None else None,
        "sampling_strategy": str(args.sampling_strategy),
        "hard_fraction": float(args.hard_fraction),
        "seed": int(args.seed),
        "shards": shard_meta,
    }
    manifest_path = out_dir / f"{split}.index.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)

    print(f"[done] split={split} samples={processed} shards={len(shard_meta)}")
    print(f"[done] manifest={manifest_path}")


if __name__ == "__main__":
    sys.exit(main())
