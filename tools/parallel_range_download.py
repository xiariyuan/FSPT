#!/usr/bin/env python3
from __future__ import annotations
import argparse, concurrent.futures, os, time, urllib.request
from pathlib import Path

URL_DEFAULT = 'https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip'
EXPECTED_DEFAULT = 1668710491

def fetch_range(url: str, start: int, end: int, out: Path, timeout: int = 90) -> tuple[bool, str, int]:
    tmp = out.with_suffix(out.suffix + '.tmp')
    if out.exists() and out.stat().st_size == end - start + 1:
        return True, out.name, out.stat().st_size
    try:
        req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        need = end - start + 1
        if len(data) != need:
            return False, f'{out.name}: got {len(data)} need {need}', len(data)
        tmp.write_bytes(data)
        tmp.replace(out)
        return True, out.name, len(data)
    except Exception as e:
        return False, f'{out.name}: {type(e).__name__}: {e}', 0

def assemble(parts_dir: Path, dest: Path, total: int, chunk: int) -> bool:
    n = (total + chunk - 1) // chunk
    for i in range(n):
        start = i * chunk
        end = min(total - 1, start + chunk - 1)
        p = parts_dir / f'part_{i:05d}.bin'
        if not p.exists() or p.stat().st_size != end - start + 1:
            return False
    tmp = dest.with_suffix(dest.suffix + '.assembling')
    with tmp.open('wb') as f:
        for i in range(n):
            f.write((parts_dir / f'part_{i:05d}.bin').read_bytes())
    if tmp.stat().st_size != total:
        return False
    tmp.replace(dest)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default=URL_DEFAULT)
    ap.add_argument('--dest', default='downloads/tapvid_davis.zip')
    ap.add_argument('--total', type=int, default=EXPECTED_DEFAULT)
    ap.add_argument('--chunk-mb', type=int, default=4)
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--seconds', type=int, default=240)
    ap.add_argument('--range-timeout', type=int, default=45)
    args = ap.parse_args()
    dest = Path(args.dest)
    parts_dir = dest.parent / (dest.name + '.parts')
    parts_dir.mkdir(parents=True, exist_ok=True)
    chunk = args.chunk_mb * 1024 * 1024
    n = (args.total + chunk - 1) // chunk
    # Seed from existing partial contiguous file, if any.
    if dest.exists() and dest.stat().st_size > 0 and dest.stat().st_size < args.total:
        with dest.open('rb') as f:
            idx = 0
            while True:
                start = idx * chunk
                if start >= dest.stat().st_size:
                    break
                end = min(start + chunk, dest.stat().st_size)
                data = f.read(end - start)
                if len(data) != end - start:
                    break
                if len(data) == chunk or end == args.total:
                    p = parts_dir / f'part_{idx:05d}.bin'
                    if not p.exists():
                        p.write_bytes(data)
                idx += 1
    deadline = time.time() + args.seconds
    def complete(i: int) -> bool:
        start = i * chunk
        end = min(args.total - 1, start + chunk - 1)
        p = parts_dir / f'part_{i:05d}.bin'
        return p.exists() and p.stat().st_size == end - start + 1
    print({'total_parts': n, 'complete_parts': sum(complete(i) for i in range(n)), 'chunk_mb': args.chunk_mb, 'workers': args.workers}, flush=True)
    while time.time() < deadline:
        missing = [i for i in range(n) if not complete(i)]
        if not missing:
            break
        batch = missing[:args.workers * 3]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = []
            for i in batch:
                start = i * chunk
                end = min(args.total - 1, start + chunk - 1)
                out = parts_dir / f'part_{i:05d}.bin'
                futs.append(ex.submit(fetch_range, args.url, start, end, out, args.range_timeout))
            ok = fail = got = 0
            for fut in concurrent.futures.as_completed(futs):
                good, msg, size = fut.result()
                got += size
                if good:
                    ok += 1
                else:
                    fail += 1
                    print('FAIL', msg, flush=True)
        done = sum(complete(i) for i in range(n))
        bytes_done = 0
        for i in range(n):
            if complete(i):
                bytes_done += (min(args.total - 1, i * chunk + chunk - 1) - i * chunk + 1)
        print({'ok_batch': ok, 'fail_batch': fail, 'complete_parts': done, 'pct': round(bytes_done / args.total * 100, 2), 'mb_done': round(bytes_done / 1024 / 1024, 1), 'seconds_left': round(deadline - time.time(), 1)}, flush=True)
        if ok == 0 and fail > 0:
            break
    if sum(complete(i) for i in range(n)) == n:
        print('assembling', flush=True)
        print({'assembled': assemble(parts_dir, dest, args.total, chunk), 'dest': str(dest), 'size': dest.stat().st_size if dest.exists() else 0}, flush=True)
    else:
        done = sum(complete(i) for i in range(n))
        print({'status': 'partial', 'complete_parts': done, 'total_parts': n}, flush=True)

if __name__ == '__main__':
    main()
