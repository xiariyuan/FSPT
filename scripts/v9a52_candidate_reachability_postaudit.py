#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path('/gemini/code/FSPT_v9a52_clean')
FORMAL_SCRIPT = ROOT / 'scripts/v9a52_independent_state_branching_audit.py'
FORMAL_JSON = ROOT / 'outputs/paper_discovery_2026-07-05/v9a52_independent_state/v9a52_independent_state_branching.json'
FORMAL_NPZ = ROOT / 'outputs/paper_discovery_2026-07-05/v9a52_independent_state/v9a52_independent_state_branching_rows.npz'
OUT_JSON = ROOT / 'outputs/paper_discovery_2026-07-05/v9a52_independent_state/v9a52_candidate_reachability_postaudit.json'
OUT_DOC = ROOT / 'docs/v9a52_candidate_reachability_postaudit_2026-07-11.md'

EXPECTED_HASHES = {
    FORMAL_SCRIPT: 'd400e8a8b0e713df00620fa113bda1e1bddcf2612821968dd3682811913c7104',
    FORMAL_JSON: '929e87feb67795448cfece53d2d6d3c0eb0cc7174d30f211790e4c2eeac12d74',
    FORMAL_NPZ: 'fd84754a5c5bb8192c9ce8251de2f1bd51b567d6ead6147b6d81849d453a19b5',
}

SEQUENCES = ('ani', 'animal3', 'r4_new_f')
HORIZONS = (1, 4, 8)
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 20260717
TOL = 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def summary(candidate: np.ndarray, baseline: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.float64)[mask]
    baseline = np.asarray(baseline, dtype=np.float64)[mask]
    difference = candidate - baseline
    if len(candidate) == 0:
        return {
            'n': 0,
            'candidate_mean': None,
            'baseline_mean': None,
            'mean_difference': None,
            'median_difference': None,
            'better': 0,
            'worse': 0,
            'equal': 0,
            'candidate_safe4': 0,
            'baseline_safe4': 0,
            'candidate_safe8': 0,
            'baseline_safe8': 0,
            'candidate_safe16': 0,
            'baseline_safe16': 0,
        }
    return {
        'n': int(len(candidate)),
        'candidate_mean': float(np.mean(candidate)),
        'baseline_mean': float(np.mean(baseline)),
        'mean_difference': float(np.mean(difference)),
        'median_difference': float(np.median(difference)),
        'better': int(np.sum(difference < -TOL)),
        'worse': int(np.sum(difference > TOL)),
        'equal': int(np.sum(np.abs(difference) <= TOL)),
        'candidate_safe4': int(np.sum(candidate <= 4.0)),
        'baseline_safe4': int(np.sum(baseline <= 4.0)),
        'candidate_safe8': int(np.sum(candidate <= 8.0)),
        'baseline_safe8': int(np.sum(baseline <= 8.0)),
        'candidate_safe16': int(np.sum(candidate <= 16.0)),
        'baseline_safe16': int(np.sum(baseline <= 16.0)),
    }


def clip_bootstrap(
    clip_id: np.ndarray,
    mask: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, Any]:
    clips = sorted(set(clip_id[mask].tolist()))
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(
        baseline, dtype=np.float64
    )
    clip_values = np.asarray(
        [np.mean(difference[mask & (clip_id == clip)]) for clip in clips],
        dtype=np.float64,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    chunk = 1000
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        count = min(chunk, BOOTSTRAP_RESAMPLES - start)
        indices = rng.integers(0, len(clips), size=(count, len(clips)))
        bootstrap[start : start + count] = np.mean(clip_values[indices], axis=1)
    return {
        'n_clips': int(len(clips)),
        'bootstrap_resamples': BOOTSTRAP_RESAMPLES,
        'bootstrap_seed': BOOTSTRAP_SEED,
        'per_clip_difference': {
            clip: float(value) for clip, value in zip(clips, clip_values)
        },
        'clip_better': int(np.sum(clip_values < -TOL)),
        'clip_worse': int(np.sum(clip_values > TOL)),
        'clip_equal': int(np.sum(np.abs(clip_values) <= TOL)),
        'mean_difference_95_ci': [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
        ],
        'probability_mean_lt_zero': float(np.mean(bootstrap < 0.0)),
    }


def comparison_bundle(
    candidate: np.ndarray,
    baseline: np.ndarray,
    visible: np.ndarray,
    horizon: np.ndarray,
    sequence: np.ndarray,
    clip_id: np.ndarray,
) -> dict[str, Any]:
    return {
        'pooled_visible': summary(candidate, baseline, visible),
        'horizon8_visible': summary(
            candidate, baseline, visible & (horizon == 8)
        ),
        'per_horizon_visible': {
            str(value): summary(
                candidate,
                baseline,
                visible & (horizon == value),
            )
            for value in HORIZONS
        },
        'per_sequence_pooled_visible': {
            name: summary(
                candidate,
                baseline,
                visible & (sequence == name),
            )
            for name in SEQUENCES
        },
        'pooled_clip_bootstrap': clip_bootstrap(
            clip_id, visible, candidate, baseline
        ),
        'horizon8_clip_bootstrap': clip_bootstrap(
            clip_id,
            visible & (horizon == 8),
            candidate,
            baseline,
        ),
    }


def write_doc(result: dict[str, Any]) -> None:
    lines = [
        '# V9-A5.2 Candidate-Reachability Post-Audit',
        '',
        'Date: 2026-07-11',
        '',
        'This is a non-gating post-formal diagnostic. It does not alter the predeclared V9-A5.2 decision.',
        '',
        '## Branch B K64 versus shared-state K64',
        '',
        '```json',
        json.dumps(result['comparisons']['branch_b_vs_shared_k64'], indent=2),
        '```',
        '',
        '## Union K64 versus shared-state K64',
        '',
        '```json',
        json.dumps(result['comparisons']['union_vs_shared_k64'], indent=2),
        '```',
        '',
        '## Novelty counts',
        '',
        '```json',
        json.dumps(result['novelty_counts'], indent=2),
        '```',
        '',
        '## Interpretation',
        '',
        result['decision'],
        '',
        '## Integrity',
        '',
        '```json',
        json.dumps(result['integrity'], indent=2),
        '```',
    ]
    OUT_DOC.write_text('\n'.join(lines) + '\n')


def main() -> None:
    checked = {}
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f'hash mismatch: {path} -> {actual}')
        checked[str(path)] = actual

    formal = json.loads(FORMAL_JSON.read_text())
    data = np.load(FORMAL_NPZ, allow_pickle=True)
    required = {
        'clip_id',
        'sequence',
        'horizon',
        'gt_visible',
        'shared_refined_oracle_k16',
        'shared_refined_oracle_k64',
        'branch_b_refined_oracle_k16',
        'branch_b_refined_oracle_k64',
        'union_refined_oracle_k16',
        'union_refined_oracle_k64',
        'branch_b_candidate_beats_shared_k16_by1',
        'branch_b_candidate_beats_shared_k64_by1',
        'branch_b_candidate_le4_shared_k16_gt4',
        'branch_b_candidate_le4_shared_k64_gt4',
    }
    if not required.issubset(set(data.files)):
        raise RuntimeError(f'missing NPZ arrays: {sorted(required - set(data.files))}')
    if len(data['clip_id']) != 216:
        raise RuntimeError('unexpected formal row count')

    clip_id = data['clip_id'].astype(str)
    sequence = data['sequence'].astype(str)
    horizon = data['horizon'].astype(int)
    visible = data['gt_visible'].astype(bool)
    shared16 = data['shared_refined_oracle_k16'].astype(np.float64)
    shared64 = data['shared_refined_oracle_k64'].astype(np.float64)
    branch16 = data['branch_b_refined_oracle_k16'].astype(np.float64)
    branch64 = data['branch_b_refined_oracle_k64'].astype(np.float64)
    union16 = data['union_refined_oracle_k16'].astype(np.float64)
    union64 = data['union_refined_oracle_k64'].astype(np.float64)

    if np.max(np.abs(union16 - np.minimum(shared16, branch16))) > TOL:
        raise RuntimeError('K16 union formula mismatch')
    if np.max(np.abs(union64 - np.minimum(shared64, branch64))) > TOL:
        raise RuntimeError('K64 union formula mismatch')

    comparisons = {
        'branch_b_vs_shared_k16': comparison_bundle(
            branch16, shared16, visible, horizon, sequence, clip_id
        ),
        'branch_b_vs_shared_k64': comparison_bundle(
            branch64, shared64, visible, horizon, sequence, clip_id
        ),
        'union_vs_shared_k16': comparison_bundle(
            union16, shared16, visible, horizon, sequence, clip_id
        ),
        'union_vs_shared_k64': comparison_bundle(
            union64, shared64, visible, horizon, sequence, clip_id
        ),
    }

    novelty_counts = {
        'visible_rows': int(np.sum(visible)),
        'branch_b_beats_shared_k16_by_gt1': int(
            np.sum(data['branch_b_candidate_beats_shared_k16_by1'] & visible)
        ),
        'branch_b_beats_shared_k64_by_gt1': int(
            np.sum(data['branch_b_candidate_beats_shared_k64_by1'] & visible)
        ),
        'branch_b_le4_shared_k16_gt4': int(
            np.sum(data['branch_b_candidate_le4_shared_k16_gt4'] & visible)
        ),
        'branch_b_le4_shared_k64_gt4': int(
            np.sum(data['branch_b_candidate_le4_shared_k64_gt4'] & visible)
        ),
        'branch_b_beats_shared_k64_by_gt1_per_sequence': {
            name: int(
                np.sum(
                    data['branch_b_candidate_beats_shared_k64_by1']
                    & visible
                    & (sequence == name)
                )
            )
            for name in SEQUENCES
        },
        'branch_b_beats_shared_k64_by_gt1_per_horizon': {
            str(value): int(
                np.sum(
                    data['branch_b_candidate_beats_shared_k64_by1']
                    & visible
                    & (horizon == value)
                )
            )
            for value in HORIZONS
        },
        'branch_b_le4_shared_k64_gt4_per_sequence': {
            name: int(
                np.sum(
                    data['branch_b_candidate_le4_shared_k64_gt4']
                    & visible
                    & (sequence == name)
                )
            )
            for name in SEQUENCES
        },
    }

    branch64_row = comparisons['branch_b_vs_shared_k64']['pooled_visible']
    branch64_ci = comparisons['branch_b_vs_shared_k64'][
        'pooled_clip_bootstrap'
    ]['mean_difference_95_ci']
    union64_row = comparisons['union_vs_shared_k64']['pooled_visible']
    union64_ci = comparisons['union_vs_shared_k64'][
        'pooled_clip_bootstrap'
    ]['mean_difference_95_ci']
    if (
        branch64_row['mean_difference'] < 0.0
        and branch64_ci[1] < 0.0
        and all(
            comparisons['branch_b_vs_shared_k64'][
                'per_sequence_pooled_visible'
            ][name]['mean_difference'] < 0.0
            for name in SEQUENCES
        )
    ):
        interpretation = (
            'CANDIDATE_POOL_INCREMENTAL_SIGNAL: Branch B K64 is robustly better '
            'than shared-state K64. This remains non-gating and cannot reopen the '
            'failed V9-A5.2 route without a new preregistration.'
        )
    else:
        interpretation = (
            'CANDIDATE_POOL_INCREMENTAL_FAIL: independent state changes the future '
            'candidate pool, but Branch B K64 does not robustly beat the shared-state '
            'K64 pool across sequences/clips. The union oracle gain is sparse and '
            'structural; it does not rescue the failed capacity-2 final-output gate. '
            'Keep INDEPENDENT_STATE_FAIL and close the tested temporal multi-state route.'
        )

    result = {
        'date': '2026-07-11',
        'type': 'non-gating post-formal candidate reachability audit',
        'protocol': {
            'formal_rows': 216,
            'visible_rows': int(np.sum(visible)),
            'horizons': list(HORIZONS),
            'bootstrap_unit': 'clip_id',
            'bootstrap_resamples': BOOTSTRAP_RESAMPLES,
            'bootstrap_seed': BOOTSTRAP_SEED,
            'training': False,
            'davis_read': False,
            'changes_predeclared_gate': False,
        },
        'formal_decision': formal['gates']['decision'],
        'comparisons': comparisons,
        'novelty_counts': novelty_counts,
        'decision': interpretation,
        'integrity': {
            'checked_hashes': checked,
            'formal_rows': int(len(data['clip_id'])),
            'visible_rows': int(np.sum(visible)),
            'union_k16_formula_exact': True,
            'union_k64_formula_exact': True,
            'formal_gate_still_failed': formal['gates']['pass_all'] is False,
            'script_sha256': sha256(Path(__file__).resolve()),
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + '\n')
    write_doc(result)
    print(
        json.dumps(
            {
                'ok': True,
                'json': str(OUT_JSON),
                'doc': str(OUT_DOC),
                'branch_b_vs_shared_k64': {
                    'mean_difference': branch64_row['mean_difference'],
                    'ci': branch64_ci,
                    'better_worse_equal': [
                        branch64_row['better'],
                        branch64_row['worse'],
                        branch64_row['equal'],
                    ],
                },
                'union_vs_shared_k64': {
                    'mean_difference': union64_row['mean_difference'],
                    'ci': union64_ci,
                    'better_worse_equal': [
                        union64_row['better'],
                        union64_row['worse'],
                        union64_row['equal'],
                    ],
                },
                'novelty_counts': novelty_counts,
                'decision': interpretation,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
