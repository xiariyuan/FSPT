#!/usr/bin/env python3
"""
Serial PRT cache builder with optional stratified sampling.

Builds one sequence at a time and writes one per-sequence cache directory.
When enabled, stratified sampling operates inside build_prt_candidate_dataset.py
using (reentry_type, occ_length_bucket) caps. After all sequences complete,
this script merges per-sequence caches into one root-level dataset_cache.npz
and samples_meta.jsonl for downstream training/evaluation.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def discover_val_sequences(data_root: str, split: str = "val") -> list:
    split_dir = Path(data_root) / split
    return sorted([d for d in split_dir.iterdir() if d.is_dir() and (d / "anno.npz").exists()])


def build_single_sequence(seq_path, output_dir, per_seq_samples, topk, num_support,
                          patch_size, query_crop_size, search_crop_size,
                          depth_noise_sigma, min_occ_length, min_camera_motion,
                          weights, seed, stratified_sampling,
                          occ_short_max, occ_medium_max, occ_long_max,
                          bucket_cap_inframe_short, bucket_cap_inframe_medium,
                          bucket_cap_inframe_long, bucket_cap_inframe_very_long,
                          bucket_cap_offscreen, bucket_cap_mixed_short,
                          bucket_cap_mixed_medium, bucket_cap_mixed_long,
                          bucket_cap_mixed_very_long):
    """Build cache for a single sequence. Returns (seq_name, n_samples, elapsed_sec)."""
    import subprocess as sp
    t0 = time.time()
    seq_output = Path(output_dir) / seq_path.name
    seq_output.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_split = Path(tmpdir) / seq_path.parent.name
            tmp_split.mkdir()
            (tmp_split / seq_path.name).symlink_to(seq_path)

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "build_prt_candidate_dataset.py"),
                "--data-root", str(tmpdir),
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
            if stratified_sampling:
                cmd.extend([
                    "--stratified-sampling",
                    "--occ-short-max", str(occ_short_max),
                    "--occ-medium-max", str(occ_medium_max),
                    "--occ-long-max", str(occ_long_max),
                    "--bucket-cap-inframe-short", str(bucket_cap_inframe_short),
                    "--bucket-cap-inframe-medium", str(bucket_cap_inframe_medium),
                    "--bucket-cap-inframe-long", str(bucket_cap_inframe_long),
                    "--bucket-cap-inframe-very-long", str(bucket_cap_inframe_very_long),
                    "--bucket-cap-offscreen", str(bucket_cap_offscreen),
                    "--bucket-cap-mixed-short", str(bucket_cap_mixed_short),
                    "--bucket-cap-mixed-medium", str(bucket_cap_mixed_medium),
                    "--bucket-cap-mixed-long", str(bucket_cap_mixed_long),
                    "--bucket-cap-mixed-very-long", str(bucket_cap_mixed_very_long),
                ])

            result = sp.run(cmd, timeout=None)
            if result.returncode != 0:
                print(f"  WARNING: {seq_path.name} exited with code {result.returncode}")

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  EXCEPTION building {seq_path.name} after {elapsed/60:.0f}min: {e}")
        return seq_path.name, 0, elapsed

    elapsed = time.time() - t0
    cache_path = seq_output / "dataset_cache.npz"
    if cache_path.exists():
        try:
            c = np.load(cache_path, allow_pickle=False)
            n = int(c["query_patch"].shape[0])
        except Exception:
            n = "err"
        print(f"  [{elapsed/60:.0f}min] {seq_path.name}: {n} samples")
        return seq_path.name, n, elapsed
    else:
        print(f"  [{elapsed/60:.0f}min] {seq_path.name}: no cache produced!")
        return seq_path.name, 0, elapsed


def merge_caches(seq_dirs: list[Path], output_dir: Path) -> int:
    all_data = {}
    first = True
    total_n = 0

    for seq_dir in seq_dirs:
        cache_path = seq_dir / "dataset_cache.npz"
        if not cache_path.exists():
            print(f"  Skipping {seq_dir.name}: no cache")
            continue
        c = np.load(cache_path, allow_pickle=False)
        n = int(c["query_patch"].shape[0])
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
        print(f"  Merge {seq_dir.name}: {n} samples")

    if total_n == 0:
        print("  No per-sequence caches to merge")
        return 0

    merged = {k: np.concatenate(v, axis=0) for k, v in all_data.items()}
    np.savez_compressed(output_dir / "dataset_cache.npz", **merged)

    meta_lines = []
    sample_id = 0
    for seq_dir in seq_dirs:
        meta_path = seq_dir / "samples_meta.jsonl"
        if not meta_path.exists():
            continue
        for line in meta_path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            d["sample_id"] = sample_id
            meta_lines.append(json.dumps(d))
            sample_id += 1
    if meta_lines:
        (output_dir / "samples_meta.jsonl").write_text("\n".join(meta_lines) + "\n")

    return total_n


def main():
    parser = argparse.ArgumentParser(description="Serial PRT cache builder (monitorable)")
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
    parser.add_argument("--stratified-sampling", action="store_true", default=True)
    parser.add_argument("--no-stratified-sampling", dest="stratified_sampling", action="store_false")
    parser.add_argument("--occ-short-max", type=int, default=100)
    parser.add_argument("--occ-medium-max", type=int, default=200)
    parser.add_argument("--occ-long-max", type=int, default=500)
    parser.add_argument("--bucket-cap-inframe-short", type=int, default=100)
    parser.add_argument("--bucket-cap-inframe-medium", type=int, default=150)
    parser.add_argument("--bucket-cap-inframe-long", type=int, default=250)
    parser.add_argument("--bucket-cap-inframe-very-long", type=int, default=50)
    parser.add_argument("--bucket-cap-offscreen", type=int, default=100)
    parser.add_argument("--bucket-cap-mixed-short", type=int, default=50)
    parser.add_argument("--bucket-cap-mixed-medium", type=int, default=75)
    parser.add_argument("--bucket-cap-mixed-long", type=int, default=100)
    parser.add_argument("--bucket-cap-mixed-very-long", type=int, default=25)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_seqs = []
    for split in args.splits.split(","):
        all_seqs.extend(discover_val_sequences(args.data_root, split.strip()))
    if args.max_sequences > 0:
        all_seqs = all_seqs[:args.max_sequences]

    print(f"Serial builder: {len(all_seqs)} sequences, {args.per_seq_samples} samples each")
    print(f"Target: {len(all_seqs) * args.per_seq_samples} total samples")
    print(f"Session PID: {os.getpid()}")
    print()

    results = []
    total_start = time.time()

    for seq_idx, seq_path in enumerate(all_seqs):
        seed = args.seed + seq_idx
        # Skip if already completed
        cache_path = output_dir / seq_path.name / "dataset_cache.npz"
        if cache_path.exists():
            try:
                c = np.load(cache_path, allow_pickle=False)
                n = int(c["query_patch"].shape[0])
            except Exception:
                n = "?"
            print(f"[{seq_idx+1}/{len(all_seqs)}] {seq_path.name}: already done ({n} samples), skipping")
            results.append((seq_path.name, n, 0))
            continue

        print(f"[{seq_idx+1}/{len(all_seqs)}] Building {seq_path.name} (seed={seed}) ...")
        sys.stdout.flush()

        name, count, elapsed = build_single_sequence(
            seq_path, str(output_dir),
            args.per_seq_samples, args.topk, args.num_support,
            args.patch_size, args.query_crop_size, args.search_crop_size,
            args.depth_noise_sigma, args.min_occ_length, args.min_camera_motion,
            args.weights, seed, args.stratified_sampling,
            args.occ_short_max, args.occ_medium_max, args.occ_long_max,
            args.bucket_cap_inframe_short, args.bucket_cap_inframe_medium,
            args.bucket_cap_inframe_long, args.bucket_cap_inframe_very_long,
            args.bucket_cap_offscreen, args.bucket_cap_mixed_short,
            args.bucket_cap_mixed_medium, args.bucket_cap_mixed_long,
            args.bucket_cap_mixed_very_long,
        )
        results.append((name, count, elapsed))

        done = seq_idx + 1
        total_elapsed = time.time() - total_start
        avg = total_elapsed / done
        remaining = avg * (len(all_seqs) - done)
        print(f"  --- Overall: {done}/{len(all_seqs)} done, "
              f"{total_elapsed/60:.0f}min elapsed, "
              f"~{remaining/60:.0f}min remaining ---")
        sys.stdout.flush()

    print("\nMerging per-sequence caches...")
    seq_dirs = [output_dir / seq_path.name for seq_path in all_seqs]
    merged_total = merge_caches(seq_dirs, output_dir)

    # Save summary
    stats = {
        "n_sequences": len(all_seqs),
        "per_seq_samples": args.per_seq_samples,
        "sampling_mode": "stratified" if args.stratified_sampling else "random",
        "total_samples": sum(r[1] for r in results if isinstance(r[1], int)),
        "merged_total_samples": int(merged_total),
        "sequences": {r[0]: {"samples": r[1], "minutes": round(r[2]/60, 1)} for r in results},
    }
    (output_dir / "build_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    total_samples = sum(r[1] for r in results if isinstance(r[1], int))
    print(f"\nDone! {total_samples} samples from {len(results)} sequences in {time.time()-total_start:.0f}s")
    print(f"Saved build_stats.json -> {output_dir}")


if __name__ == "__main__":
    main()
