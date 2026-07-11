#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/gemini/code/FSPT_v9a52_clean')
SOURCE_NPZ = ROOT / 'outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_history_preserving_beam_rows.npz'
OUT = ROOT / 'docs/v9a52_independent_state_event_manifest_2026-07-11.json'
EXPECTED_SOURCE_SHA256 = '05b450fb2031ce1d762ed7436bcef34a47dfaf37c10d1c33bd8fd74c1d0381a0'
EVENTS_PER_CLIP = 8
MAX_EVENT_FRAME = 87
HASH_PREFIX = 'v9a52-independent-state-event-v1'
HORIZONS = (1, 4, 8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def event_digest(clip_id: str, query_idx: int, frame_tau: int) -> str:
    payload = f'{HASH_PREFIX}:{clip_id}:{query_idx}:{frame_tau}'.encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    source_hash = sha256(SOURCE_NPZ)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f'V9-A5.1c NPZ hash mismatch: {source_hash}')

    data = np.load(SOURCE_NPZ, allow_pickle=True)
    allowed = {'clip_id', 'query_idx', 'frame_tau', 'risk'}
    loaded = {name: data[name] for name in allowed}
    clip_id = loaded['clip_id'].astype(str)
    query_idx = loaded['query_idx'].astype(int)
    frame_tau = loaded['frame_tau'].astype(int)
    risk = loaded['risk'].astype(bool)

    row_keys = list(zip(clip_id.tolist(), frame_tau.tolist(), query_idx.tolist()))
    if len(row_keys) != 27_648 or len(set(row_keys)) != 27_648:
        raise RuntimeError('unexpected or duplicate V9-A5.1c row keys')

    clips = sorted(set(clip_id.tolist()))
    selected = []
    per_clip = {}
    for clip in clips:
        eligible = []
        for query in range(32):
            rows = np.flatnonzero(
                (clip_id == clip)
                & (query_idx == query)
                & risk
                & (frame_tau > 0)
                & (frame_tau <= MAX_EVENT_FRAME)
            )
            if len(rows) == 0:
                continue
            first_row = int(rows[np.argmin(frame_tau[rows])])
            event_frame = int(frame_tau[first_row])
            digest = event_digest(clip, query, event_frame)
            eligible.append(
                {
                    'clip_id': clip,
                    'query_idx': int(query),
                    'frame_tau': event_frame,
                    'source_row_index': first_row,
                    'selection_sha256': digest,
                }
            )
        eligible.sort(
            key=lambda row: (
                row['selection_sha256'],
                row['frame_tau'],
                row['query_idx'],
            )
        )
        if len(eligible) < EVENTS_PER_CLIP:
            raise RuntimeError(
                f'{clip} has only {len(eligible)} eligible first-risk events'
            )
        chosen = eligible[:EVENTS_PER_CLIP]
        selected.extend(chosen)
        per_clip[clip] = {
            'eligible_first_risk_events': int(len(eligible)),
            'selected_events': int(len(chosen)),
            'eligible_frame_min': int(min(row['frame_tau'] for row in eligible)),
            'eligible_frame_median': float(
                np.median([row['frame_tau'] for row in eligible])
            ),
            'eligible_frame_max': int(max(row['frame_tau'] for row in eligible)),
            'selected_frame_min': int(min(row['frame_tau'] for row in chosen)),
            'selected_frame_median': float(
                np.median([row['frame_tau'] for row in chosen])
            ),
            'selected_frame_max': int(max(row['frame_tau'] for row in chosen)),
        }

    event_keys = [
        (row['clip_id'], row['query_idx'], row['frame_tau']) for row in selected
    ]
    if len(selected) != len(clips) * EVENTS_PER_CLIP:
        raise RuntimeError('selected event count mismatch')
    if len(set(event_keys)) != len(event_keys):
        raise RuntimeError('duplicate selected event keys')

    manifest = {
        'created_at': '2026-07-11',
        'experiment': 'V9-A5.2 independent model-state branching feasibility',
        'selection_protocol': {
            'source_fields_read': sorted(allowed),
            'gt_fields_read_for_event_selection': False,
            'risk_rule': 'committed V9-A5.1c native risk flag',
            'event_definition': (
                'first native-risk frame per query with 1 <= frame_tau <= 87'
            ),
            'selection_method': (
                'lowest stable SHA256 event keys within each clip'
            ),
            'hash_prefix': HASH_PREFIX,
            'events_per_clip': EVENTS_PER_CLIP,
            'horizons': list(HORIZONS),
            'maximum_event_frame': MAX_EVENT_FRAME,
        },
        'source': {
            'path': str(SOURCE_NPZ),
            'size_bytes': int(SOURCE_NPZ.stat().st_size),
            'sha256': source_hash,
            'rows': int(len(row_keys)),
            'unique_row_keys': int(len(set(row_keys))),
        },
        'clips': clips,
        'per_clip': per_clip,
        'selected_events': selected,
        'totals': {
            'clips': int(len(clips)),
            'events': int(len(selected)),
            'event_horizon_rows_before_visibility_filter': int(
                len(selected) * len(HORIZONS)
            ),
            'minimum_eligible_events_per_clip': int(
                min(row['eligible_first_risk_events'] for row in per_clip.values())
            ),
        },
    }
    OUT.write_text(json.dumps(manifest, indent=2) + '\n')
    print(
        json.dumps(
            {
                'ok': True,
                'out': str(OUT),
                'events': len(selected),
                'events_per_clip': EVENTS_PER_CLIP,
                'minimum_eligible_events_per_clip': manifest['totals'][
                    'minimum_eligible_events_per_clip'
                ],
                'source_sha256': source_hash,
                'manifest_sha256': sha256(OUT),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
