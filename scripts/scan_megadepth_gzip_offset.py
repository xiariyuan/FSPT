#!/usr/bin/env python3
"""Scan MegaDepth_v1.tar.gz and report the first gzip/zlib failure offset.

This is a read-only diagnostic helper. It streams the archive with zlib's
gzip decoder, so a corruption anywhere in the compressed stream will stop the
scan at the first failing input chunk.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zlib


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{value} B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="/gemini/code/FSPT/datasets/megadepth/MegaDepth_v1.tar.gz",
        help="Path to MegaDepth_v1.tar.gz",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256 * 1024 * 1024,
        help="Compressed input chunk size to read each step (default: 256MiB)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=4 * 1024 * 1024 * 1024,
        help="Print a progress line every N bytes processed (default: 4GiB)",
    )
    parser.add_argument(
        "--limit-bytes",
        type=int,
        default=0,
        help="Stop after processing at most N compressed input bytes (0 = full file)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.path
    chunk_size = args.chunk_size
    progress_every = args.progress_every
    limit_bytes = args.limit_bytes

    file_size = os.path.getsize(path)
    print(f"[scan] file={path}")
    print(f"[scan] size={file_size} ({human_bytes(file_size)})")
    print(f"[scan] chunk_size={chunk_size} ({human_bytes(chunk_size)})")
    if limit_bytes > 0:
        print(f"[scan] limit_bytes={limit_bytes} ({human_bytes(limit_bytes)})")

    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    read_off = 0
    next_progress = progress_every
    start = time.time()

    with open(path, "rb") as f:
        while True:
            if limit_bytes > 0 and read_off >= limit_bytes:
                print(f"[scan] reached prefix limit at {read_off} bytes without zlib error")
                return 0

            chunk = f.read(chunk_size)
            if not chunk:
                break

            if limit_bytes > 0:
                remaining = limit_bytes - read_off
                if remaining <= 0:
                    print(f"[scan] reached prefix limit at {read_off} bytes without zlib error")
                    return 0
                if len(chunk) > remaining:
                    chunk = chunk[:remaining]
            try:
                decoder.decompress(chunk)
            except zlib.error as exc:
                print(
                    f"[scan] FAIL at compressed input offset ~{read_off}..{read_off + len(chunk) - 1}"
                )
                print(f"[scan] zlib_error={exc}")
                return 2

            read_off += len(chunk)
            if read_off >= next_progress:
                elapsed = max(time.time() - start, 1e-6)
                rate = read_off / elapsed
                print(
                    f"[scan] progress {read_off}/{file_size} "
                    f"({read_off / file_size * 100:.2f}%), {rate / 1024 / 1024:.2f} MiB/s"
                )
                next_progress += progress_every

        try:
            decoder.flush()
        except zlib.error as exc:
            print(f"[scan] FAIL during flush at end of input offset {read_off}")
            print(f"[scan] zlib_error={exc}")
            return 2

    if not decoder.eof:
        if limit_bytes > 0:
            print(f"[scan] reached prefix limit at {read_off} bytes without zlib error")
            return 0
        print("[scan] truncated stream: gzip EOF not reached")
        return 2

    print("[scan] OK: gzip stream decompresses cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
