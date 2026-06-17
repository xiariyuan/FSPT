#!/usr/bin/env python3
"""
Parallel wrapper for build_prt_candidate_dataset.py.
Builds per-sequence caches in parallel, then merges them.

Usage:
  python3 scripts/build_prt_parallel.py \
    --data-root /gemini/code/FSPT/datasets/pointodyssey \
    --splits val \
    --max-sequences 15 \
    --per-seq-samples 300 \
    --topk 5 \
    --output-dir outputs/prt_candidate_val_15seq_300each
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def discover_val_sequences(data_root: str, split: str = "val") -> list:
    split_dir = Path(data_root) / split
    return sorted([d for d in split_dir.iterdir() if d.is_dir() and (d / "anno.npz").exists()])


def build_single_sequence(args_tuple):
    """Build cache for a single sequence in a subprocess."""
    seq_path, output_dir, per_seq_samples, topk, num_support, patch_size, \
        query_crop_size, search_crop_size, depth_noise_sigma, min_occ_length, \
        min_camera_motion, weights, seed = args_tuple

    seq_output = Path(output_dir) / seq_path.name
    seq_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "build_prt_candidate_dataset.py"),
        "--data-root", str(seq_path.parent.parent),
        "--splits", seq_path.parent.name,
        "--max-sequences-per-split", "0",  # we pass specific seq via hack
        "--max-samples", str(per_seq_samples),
        "--topk", str(topk),
        "--num-support", str(num_support),
        "--patch-size", str(patch_size),
        "--query-crop-size", str(query_crop_size),
        "--search-crop-size", str(search_crop_size),
        "--depth-noise-sigma", str(depth_noise_sigma),
        "--min-occ-length", str(min_occ_length),
        "--min-camera-motion", str(min_camera_motion),
        "--weights", str(weights),
        "--seed", str(seed),
        "--output-dir", str(seq_output),
    ]

    # We need to limit to just this one sequence.
    # The simplest approach: create a temp symlink directory with only this seq.
    # Actually, let's use a different approach - modify max_sequences to 1 and
    # use a temp split dir with just this sequence.

    # Use a temp dir with just this sequence symlinked
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_split = Path(tmpdir) / seq_path.parent.name
        tmp_split.mkdir()
        (tmp_split / seq_path.name).symlink_to(seq_path)

        cmd_hack = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "build_prt_candidate_dataset.py"),
            "--data-root", tmpdir,
            "--splits", seq_path.parent.name,
            "--max-sequences-per-split", "1",
            "--max-samples", str(per_seq_samples),
            "--topk", str(topk),
            "--num-support", str(num_support),
            "--patch-size", str(patch_size),
            "--query-crop-size", str(query_crop_size),
            "--search-crop-size", str(search_crop_size),
            "--depth-noise-sigma", str(depth_noise_sigma),
            "--min-occ-length", str(min_occ_length),
            "--min-camera-motion", str(min_camera_motion),
            "--weights", str(weights),
            "--seed", str(seed),
            "--output-dir", str(seq_output),
        ]

        result = subprocess.run(cmd_hack, capture_output=True, text=True)

    cache_path = seq_output / "dataset_cache.npz"
    if cache_path.exists():
        c = np.load(cache_path, allow_pickle=False)
        return seq_path.name, int(c["query_patch"].shape[0]), None
    else:
        return seq_path.name, 0, result.stderr[-500:] if result.stderr else "unknown error"


def merge_caches(seq_dirs: list, output_path: Path):
    """Merge per-sequence caches into one."""
    all_data = {}
    first = True
    total_n = 0

    for seq_dir in seq_dirs:
        cache_path = seq_dir / "dataset_cache.npz"
        if not cache_path.exists():
            print(f"  Skipping {seq_dir.name}: no cache")
            continue
        c = np.load(cache_path, allow_pickle=False)
        n = c["query_patch"].shape[0]
        if n == 0:
            continue

        if first:
            for k in c.files:
                all_data[k] = [c[k]]
            first = False
        else:
            for k in c.files:
                all_data[k].append(c[k])
        total_n += n
        print(f"  {seq_dir.name}: {n} samples")

    if total_n == 0:
        print("ERROR: No samples collected!")
        return

    merged = {}
    for k in all_data:
        merged[k] = np.concatenate(all_data[k], axis=0)

    np.savez(output_path, **merged)
    print(f"\nMerged: {total_n} total samples -> {output_path}")

    # Also merge meta files
    meta_lines = []
    sample_id = 0
    for seq_dir in seq_dirs:
        meta_path = seq_dir / "samples_meta.jsonl"
        if meta_path.exists():
            for line in meta_path.read_text().strip().split("\n"):
                if line:
                    d = json.loads(line)
                    d["sample_id"] = sample_id
                    meta_lines.append(json.dumps(d))
                    sample_id += 1

    if meta_lines:
        (output_path.parent / "samples_meta.jsonl").write_text("\n".join(meta_lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Parallel PRT cache builder")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--splits", type=str, default="val")
    parser.add_argument("--max-sequences", type=int, default=15)
    parser.add_argument("--per-seq-samples", type=int, default=300)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--num-support", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--query-crop-size", type=int, default=112)
    parser.add_argument("--search-crop-size", type=int, default=224)
    parser.add_argument("--depth-noise-sigma", type=float, default=0.10)
    parser.add_argument("--min-occ-length", type=int, default=20)
    parser.add_argument("--min-camera-motion", type=float, default=0.30)
    parser.add_argument("--weights", type=str,
                        default="/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    parser.add_argument("--seed", type=int, default=789)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover sequences
    all_seqs = []
    for split in args.splits.split(","):
        all_seqs.extend(discover_val_sequences(args.data_root, split.strip()))

    if args.max_sequences > 0:
        all_seqs = all_seqs[:args.max_sequences]

    print(f"Building cache for {len(all_seqs)} sequences, {args.per_seq_samples} samples each")
    print(f"Total target: {len(all_seqs) * args.per_seq_samples} samples")
    print(f"Workers: {args.workers}")
    print()

    # Build per-sequence caches in parallel
    task_args = [
        (seq_path, str(output_dir), args.per_seq_samples, args.topk,
         args.num_support, args.patch_size, args.query_crop_size,
         args.search_crop_size, args.depth_noise_sigma, args.min_occ_length,
         args.min_camera_motion, args.weights, args.seed + i)
        for i, seq_path in enumerate(all_seqs)
    ]

    results = []
    import time
    start_time = time.time()
    n_total = len(all_seqs)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(build_single_sequence, ta): ta[0].name for ta in task_args}
        for future in as_completed(futures):
            seq_name = futures[future]
            elapsed = time.time() - start_time
            done_count = len(results) + 1
            try:
                name, count, error = future.result()
                status = f"{count} samples" if error is None else f"FAILED: {error}"
                print(f"[{done_count}/{n_total}] {name}: {status}  ({elapsed/60:.0f}min elapsed)", flush=True)
                results.append((name, count, error))
            except Exception as e:
                print(f"[{done_count}/{n_total}] {seq_name}: EXCEPTION: {e}  ({elapsed/60:.0f}min elapsed)", flush=True)
                results.append((seq_name, 0, str(e)))

    # Merge
    print("\nMerging caches...")
    seq_dirs = [output_dir / seq_path.name for seq_path in all_seqs]
    merge_caches(seq_dirs, output_dir / "dataset_cache.npz")

    # Stats
    cache_path = output_dir / "dataset_cache.npz"
    if cache_path.exists():
        c = np.load(cache_path, allow_pickle=False)
        stats = {
            "n_sequences": len(all_seqs),
            "per_seq_samples": args.per_seq_samples,
            "total_samples": int(c["query_patch"].shape[0]),
            "topk": int(c["cand_patches"].shape[1]),
            "sequences": {r[0]: r[1] for r in results},
        }
        (output_dir / "build_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        print(f"\nFinal: {c['query_patch'].shape[0]} samples from {len(all_seqs)} sequences")
        print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
