#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


def npy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def video_index(video_id: str) -> int | None:
    m = re.search(r'rgb_stacking_(\d{6})', str(video_id))
    if not m:
        return None
    return int(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-cache', required=True)
    ap.add_argument('--output-cache', required=True)
    ap.add_argument('--output-report', required=True)
    ap.add_argument('--start-index', type=int, required=True)
    ap.add_argument('--end-index', type=int, required=True, help='inclusive')
    args = ap.parse_args()

    src = torch.load(args.input_cache, map_location='cpu', weights_only=False)
    records = []
    skipped = []
    kept_video_ids = []
    n_queries = 0
    for r in src.get('records', []):
        vid = str(r.get('video_id', r.get('source_video_id', '')))
        idx = video_index(vid)
        if idx is not None and args.start_index <= idx <= args.end_index:
            records.append(r)
            kept_video_ids.append(vid)
            q = r.get('query_points')
            if q is not None:
                n_queries += int(npy(q).shape[0])
        else:
            skipped.append(vid)
    out = dict(src)
    out['records'] = records
    out['filtered_range'] = {'start_index': args.start_index, 'end_index': args.end_index}
    out['source_cache'] = str(args.input_cache)
    Path(args.output_cache).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output_cache)
    report = {
        'input_cache': str(args.input_cache),
        'output_cache': str(args.output_cache),
        'start_index': args.start_index,
        'end_index': args.end_index,
        'n_records': len(records),
        'n_queries': int(n_queries),
        'video_ids': kept_video_ids,
        'skipped_count': len(skipped),
    }
    Path(args.output_report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
