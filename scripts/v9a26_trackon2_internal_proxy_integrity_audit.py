#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/paper_discovery_2026-07-05'
V9A2 = BASE / 'v9a2_anchor_uncertainty_reacquisition'
OUTDIR = BASE / 'v9a26_trackon2_internal_proxy'
JOINT = V9A2 / 'v9a2_joint_w8_common_plus_w16_extension_v3.npz'
FULL = OUTDIR / 'v9a26_trackon2_internal_proxy_features.npz'
FULL_REPORT = FULL.with_suffix('.report.json')
SMOKE_OLD = OUTDIR / 'v9a26_internal_proxy_smoke1video.npz'
SMOKE_NEW = OUTDIR / 'v9a26_internal_proxy_smoke1video_v2.npz'
SMOKE_NEW_REPORT = SMOKE_NEW.with_suffix('.report.json')
OUT_JSON = OUTDIR / 'v9a26_trackon2_internal_proxy_integrity_audit.json'
OUT_DOC = ROOT / 'docs/v9a26_trackon2_internal_proxy_integrity_audit_result_2026-07-10.md'


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_summary(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        'mean': float(np.mean(x)),
        'median': float(np.median(x)),
        'p95': float(np.quantile(x, 0.95)),
        'max': float(np.max(x)),
        'std': float(np.std(x)),
    }


def parity_max(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    return {
        key: max((float(row[key]) for row in rows), default=None)
        for key in ['p_max_abs', 'v_max_abs', 'q_max_abs']
    }


def main() -> None:
    joint = np.load(JOINT, allow_pickle=True)
    full = np.load(FULL, allow_pickle=True)
    smoke_old = np.load(SMOKE_OLD, allow_pickle=True)
    smoke_new = np.load(SMOKE_NEW, allow_pickle=True)
    full_report = json.loads(FULL_REPORT.read_text())
    smoke_new_report = json.loads(SMOKE_NEW_REPORT.read_text())

    expected_indices = np.where(joint['is_w16_extension'].astype(bool))[0]
    checks = {
        'full_shape': tuple(full['X_internal'].shape) == (557, 64),
        'full_finite': bool(np.all(np.isfinite(full['X_internal']))),
        'full_indices_match_joint': bool(np.array_equal(full['joint_indices'], expected_indices)),
        'full_row_keys_match_joint': bool(
            np.array_equal(full['row_key_json'], joint['row_key_json'][expected_indices])
        ),
        'unique_feature_names': len(set(full['feature_names'].tolist())) == 64,
        'old_smoke_matches_full_prefix': bool(
            np.array_equal(smoke_old['row_key_json'], full['row_key_json'][: len(smoke_old['joint_indices'])])
            and np.array_equal(smoke_old['feature_names'], full['feature_names'])
            and np.array_equal(smoke_old['X_internal'], full['X_internal'][: len(smoke_old['joint_indices'])])
        ),
        'enhanced_smoke_matches_old_smoke': bool(
            np.array_equal(smoke_new['row_key_json'], smoke_old['row_key_json'])
            and np.array_equal(smoke_new['feature_names'], smoke_old['feature_names'])
            and np.array_equal(smoke_new['X_internal'], smoke_old['X_internal'])
        ),
        'enhanced_smoke_matches_full_prefix': bool(
            np.array_equal(smoke_new['X_internal'], full['X_internal'][: len(smoke_new['joint_indices'])])
        ),
    }

    full_parity = parity_max(full_report['official_track_frame_parity'])
    enhanced_parity = parity_max(smoke_new_report['official_track_frame_parity'])
    checks['full_first_active_frame_parity'] = bool(
        len(full_report['official_track_frame_parity']) == 20
        and all(value == 0.0 for value in full_parity.values())
    )
    checks['enhanced_long_sequence_parity'] = bool(
        len(smoke_new_report['official_track_frame_parity']) == 3
        and len({row['frame'] for row in smoke_new_report['official_track_frame_parity']}) == 3
        and all(value == 0.0 for value in enhanced_parity.values())
    )

    if not all(checks.values()):
        raise RuntimeError({'failed_checks': [name for name, passed in checks.items() if not passed]})

    names = [str(x) for x in full['feature_names'].tolist()]
    old_dist = full['X_internal'][:, names.index('proxy_old_candidate_dist_model_px')]
    native_dist = full['X_internal'][:, names.index('proxy_native_dist_model_px')]
    report = {
        'date': '2026-07-10',
        'script': 'scripts/v9a26_trackon2_internal_proxy_integrity_audit.py',
        'provenance_boundary': (
            'TrackOn2 256-space M24 support-grid20 rerun is an internal-state proxy. '
            'It is not claimed to be the exact latent state that generated the historical first-input cache.'
        ),
        'artifacts': {
            'joint': {'path': str(JOINT), 'sha256': sha256_file(JOINT)},
            'full_features': {'path': str(FULL), 'sha256': sha256_file(FULL)},
            'full_report': {'path': str(FULL_REPORT), 'sha256': sha256_file(FULL_REPORT)},
            'old_smoke': {'path': str(SMOKE_OLD), 'sha256': sha256_file(SMOKE_OLD)},
            'enhanced_smoke': {'path': str(SMOKE_NEW), 'sha256': sha256_file(SMOKE_NEW)},
            'enhanced_smoke_report': {
                'path': str(SMOKE_NEW_REPORT),
                'sha256': sha256_file(SMOKE_NEW_REPORT),
            },
        },
        'checks': checks,
        'full_matrix': {
            'shape': list(full['X_internal'].shape),
            'finite_rate': float(np.mean(np.isfinite(full['X_internal']))),
            'videos': len(full_report['videos']),
            'seconds_original_full_build': float(full_report['seconds']),
        },
        'official_forward_parity': {
            'full_first_active_frame_checks': len(full_report['official_track_frame_parity']),
            'full_max_abs': full_parity,
            'enhanced_target_frames': [
                {
                    'video_id': row['video_id'],
                    'frame': int(row['frame']),
                    'active_queries': int(row['active_queries']),
                }
                for row in smoke_new_report['official_track_frame_parity']
            ],
            'enhanced_max_abs': enhanced_parity,
        },
        'trackon_config': smoke_new_report['trackon_config'],
        'hashes_from_enhanced_builder': smoke_new_report['integrity'],
        'proxy_alignment_model_pixels': {
            'proxy_to_old_candidate': array_summary(old_dist),
            'proxy_to_native': array_summary(native_dist),
        },
        'decision': {
            'status': 'PASS_PROXY_FEATURE_LINEAGE_AND_FORWARD_PARITY',
            'interpretation': (
                'The patched builder preserves the original 64-dimensional features exactly on the smoke prefix, '
                'the full matrix is strictly aligned to all 557 extension rows, and the copied diagnostic forward '
                'matches official TrackOn2 position/visibility/query tensors exactly at both first-active and '
                'long-sequence target frames. The historical-latent provenance limitation remains explicit.'
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    lines = [
        '# V9-A2.6 TrackOn2 Internal Proxy Integrity Audit',
        '',
        'Date: 2026-07-10',
        '',
        '## Provenance boundary',
        '',
        report['provenance_boundary'],
        '',
        '## Integrity checks',
        '',
        '| Check | Result |',
        '|---|---|',
    ]
    for name, passed in checks.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |")
    lines += [
        '',
        '## Official-forward parity',
        '',
        f"- Full build: {len(full_report['official_track_frame_parity'])} videos, first-active-frame max errors {full_parity}.",
        f"- Enhanced smoke target frames: {report['official_forward_parity']['enhanced_target_frames']}.",
        f"- Enhanced smoke max errors: {enhanced_parity}.",
        '',
        '## Proxy alignment',
        '',
        f"- Proxy-to-old-candidate internal-model distance: {report['proxy_alignment_model_pixels']['proxy_to_old_candidate']}.",
        f"- Proxy-to-native internal-model distance: {report['proxy_alignment_model_pixels']['proxy_to_native']}.",
        '',
        '## Decision',
        '',
        '**PASS_PROXY_FEATURE_LINEAGE_AND_FORWARD_PARITY**',
        '',
        report['decision']['interpretation'],
    ]
    OUT_DOC.write_text('\n'.join(lines) + '\n')

    print(json.dumps({
        'ok': True,
        'json': str(OUT_JSON),
        'doc': str(OUT_DOC),
        'checks': checks,
        'full_parity_max': full_parity,
        'enhanced_parity_max': enhanced_parity,
        'proxy_old_candidate_p95_model_px': report['proxy_alignment_model_pixels']['proxy_to_old_candidate']['p95'],
        'decision': report['decision']['status'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
