#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb-pkl', default='/gemini/code/datasets/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl')
    ap.add_argument('--out-dir', default='outputs/paper_discovery_2026-06-27/b2wa_video_cache/rgb_dev10')
    ap.add_argument('--start-index', type=int, default=0)
    ap.add_argument('--num-videos', type=int, default=10)
    ap.add_argument('--compressed', action='store_true', help='Use savez_compressed; slower but smaller.')
    args = ap.parse_args()

    pkl = Path(args.rgb_pkl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'loading {pkl} ({pkl.stat().st_size/1024/1024:.1f} MB) ...', flush=True)
    with pkl.open('rb') as f:
        data = pickle.load(f)
    print(f'loaded entries={len(data)}', flush=True)
    start = max(0, int(args.start_index))
    end = min(len(data), start + int(args.num_videos))
    manifest = {
        'source_pkl': str(pkl),
        'start_index': start,
        'end_index_exclusive': end,
        'num_videos': end - start,
        'compressed': bool(args.compressed),
        'videos': [],
    }
    for i in range(start, end):
        entry = data[i]
        video = np.asarray(entry['video'])
        points = np.asarray(entry['points'], dtype=np.float32)
        occ = np.asarray(entry['occluded'], dtype=bool)
        vid = f'rgb_stacking_{i:06d}'
        out = out_dir / f'{vid}.npz'
        payload = {
            'video': video,
            'points': points,
            'occluded': occ,
            'sequence_index': np.asarray(i, dtype=np.int32),
        }
        if args.compressed:
            np.savez_compressed(out, **payload)
        else:
            np.savez(out, **payload)
        st = out.stat().st_size
        manifest['videos'].append({
            'video_id': vid,
            'path': str(out),
            'shape': list(video.shape),
            'dtype': str(video.dtype),
            'size_mb': round(st/1024/1024, 3),
        })
        print(f'wrote {vid} shape={video.shape} size={st/1024/1024:.1f} MB', flush=True)
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f'wrote manifest {out_dir / "manifest.json"}', flush=True)


if __name__ == '__main__':
    main()
