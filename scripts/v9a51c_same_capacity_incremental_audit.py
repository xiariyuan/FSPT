#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path('/gemini/code/FSPT_v9a51c_clean')
V9B_NPZ = ROOT / 'outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_candidate_conditioned_refinement_rows.npz'
V9C_SCRIPT = ROOT / 'scripts/v9a51c_history_preserving_beam_audit.py'
V9C_JSON = ROOT / 'outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_history_preserving_beam.json'
V9C_NPZ = ROOT / 'outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_history_preserving_beam_rows.npz'
OUT_JSON = ROOT / 'outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_same_capacity_incremental_audit.json'
OUT_DOC = ROOT / 'docs/v9a51c_same_capacity_incremental_review_2026-07-10.md'

EXPECTED_HASHES = {
    str(V9B_NPZ): 'eb9736cbe7fc1af08e5bfe23248b333e18ae01c826fd26fc65aa40945746d266',
    str(V9C_SCRIPT): 'ee96041ce451c0639c94f7558771e6958907dc5030ea1b429228ec97ce3f4da1',
    str(V9C_JSON): 'ae6bd3eabd36aa8d64014298f48bcc24a31b0611854c8827455a96aa69a6d00b',
    str(V9C_NPZ): '05b450fb2031ce1d762ed7436bcef34a47dfaf37c10d1c33bd8fd74c1d0381a0',
}

SEQUENCES = ('ani', 'animal3', 'r4_new_f')
PRIMARY = 'native_dynamic_B4'
TOL = 1e-6
BOOTSTRAP_SEED = 20260715
BOOTSTRAP_RESAMPLES = 100_000

POLICIES = {
    'native_dynamic_B1': {'budget_on_risk': 64, 'width': 1},
    'native_dynamic_B4': {'budget_on_risk': 64, 'width': 4},
    'native_dynamic_B8': {'budget_on_risk': 64, 'width': 8},
    'fixed16_B4': {'budget_on_risk': 16, 'width': 4},
    'fixed64_B4': {'budget_on_risk': 64, 'width': 4},
}


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
    return {
        'n': int(len(candidate)),
        'candidate_mean': float(np.mean(candidate)),
        'baseline_mean': float(np.mean(baseline)),
        'mean_difference': float(np.mean(difference)),
        'median_difference': float(np.median(difference)),
        'candidate_safe16': int(np.sum(candidate <= 16.0)),
        'baseline_safe16': int(np.sum(baseline <= 16.0)),
        'better': int(np.sum(difference < -TOL)),
        'worse': int(np.sum(difference > TOL)),
        'equal': int(np.sum(np.abs(difference) <= TOL)),
    }


def clip_bootstrap(
    clip_id: np.ndarray,
    mask: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, Any]:
    clips = sorted(set(clip_id.tolist()))
    clip_difference = np.asarray(
        [
            np.mean((candidate - baseline)[mask & (clip_id == clip)])
            for clip in clips
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    chunk = 1000
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        count = min(chunk, BOOTSTRAP_RESAMPLES - start)
        indices = rng.integers(0, len(clips), size=(count, len(clips)))
        values[start:start + count] = np.mean(clip_difference[indices], axis=1)
    return {
        'per_clip_difference': {
            clip: float(value) for clip, value in zip(clips, clip_difference)
        },
        'mean_difference_95_ci': [
            float(value) for value in np.quantile(values, [0.025, 0.975])
        ],
        'probability_mean_lt_zero': float(np.mean(values < 0.0)),
        'all_clip_nonpositive': bool(np.all(clip_difference <= TOL)),
    }


def frame_local_readouts(
    official: np.ndarray,
    risk: np.ndarray,
    frame_tau: np.ndarray,
    scores: np.ndarray,
    refined_error: np.ndarray,
    budget: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    top1 = np.asarray(official, dtype=np.float64).copy()
    oracle = np.asarray(official, dtype=np.float64).copy()
    active = np.flatnonzero(risk & (frame_tau > 0))
    for row in active:
        order = np.argsort(-scores[row, :budget], kind='mergesort')
        top1[row] = float(refined_error[row, order[0]])
        oracle[row] = min(
            float(official[row]),
            float(np.min(refined_error[row, order[:width]])),
        )
    return top1, oracle


def build_gate(
    candidate: np.ndarray,
    baseline: np.ndarray,
    visible: np.ndarray,
    sequence: np.ndarray,
    reentry_first: np.ndarray,
    reentry_early8: np.ndarray,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    per_sequence = {
        name: summary(candidate, baseline, visible & (sequence == name))
        for name in SEQUENCES
    }
    global_row = summary(candidate, baseline, visible)
    first_row = summary(candidate, baseline, visible & reentry_first)
    early8_row = summary(candidate, baseline, visible & reentry_early8)
    checks = {
        'sequence_mean_improved': {
            name: row['candidate_mean'] < row['baseline_mean']
            for name, row in per_sequence.items()
        },
        'sequence_safe16_not_decreased': {
            name: row['candidate_safe16'] >= row['baseline_safe16']
            for name, row in per_sequence.items()
        },
        'global_better_gt_worse': global_row['better'] > global_row['worse'],
        'clip_bootstrap_ci_upper_lt_zero': bootstrap['mean_difference_95_ci'][1] < 0.0,
        'first_reentry_mean_improved': first_row['candidate_mean'] < first_row['baseline_mean'],
        'early8_mean_improved': early8_row['candidate_mean'] < early8_row['baseline_mean'],
        'all_clip_mean_differences_nonpositive': bootstrap['all_clip_nonpositive'],
    }
    checks['pass_all'] = bool(
        all(checks['sequence_mean_improved'].values())
        and all(checks['sequence_safe16_not_decreased'].values())
        and checks['global_better_gt_worse']
        and checks['clip_bootstrap_ci_upper_lt_zero']
        and checks['first_reentry_mean_improved']
        and checks['early8_mean_improved']
        and checks['all_clip_mean_differences_nonpositive']
    )
    checks['summaries'] = {
        'all_visible': global_row,
        'reentry_first': first_row,
        'reentry_early8': early8_row,
        'per_sequence': per_sequence,
    }
    return checks


def write_doc(result: dict[str, Any]) -> None:
    primary = result['policies'][PRIMARY]
    lines = [
        '# V9-A5.1c Same-Capacity Incremental Temporal-Value Review',
        '',
        'Date: 2026-07-10',
        '',
        'This is a post-formal conservative comparator review. It does not alter the original predeclared V9-A5.1c gates.',
        '',
        '## Primary B4 fair comparison',
        '',
        '```json',
        json.dumps(primary, indent=2),
        '```',
        '',
        '## All policy comparisons',
        '',
        '```json',
        json.dumps(result['policy_overview'], indent=2),
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
    checked_hashes = {}
    for path_text, expected in EXPECTED_HASHES.items():
        path = Path(path_text)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f'hash mismatch: {path}')
        checked_hashes[path_text] = actual

    v9c_report = json.loads(V9C_JSON.read_text())
    v9b = np.load(V9B_NPZ, allow_pickle=True)
    v9c = np.load(V9C_NPZ, allow_pickle=True)

    keys_b = list(
        zip(
            v9b['clip_id'].astype(str).tolist(),
            v9b['frame_tau'].astype(int).tolist(),
            v9b['query_idx'].astype(int).tolist(),
        )
    )
    keys_c = list(
        zip(
            v9c['clip_id'].astype(str).tolist(),
            v9c['frame_tau'].astype(int).tolist(),
            v9c['query_idx'].astype(int).tolist(),
        )
    )
    if keys_b != keys_c or len(set(keys_c)) != 27_648:
        raise RuntimeError('row-key parity failed')

    frame_tau = v9c['frame_tau'].astype(int)
    visible = v9c['gt_visible'].astype(bool) & (frame_tau > 0)
    risk = v9c['risk'].astype(bool)
    sequence = v9c['sequence'].astype(str)
    clip_id = v9c['clip_id'].astype(str)
    official = v9c['official_final'].astype(np.float64)
    score = v9b['candidate_score'].astype(np.float64)
    refined_error = v9b['refined_candidate_error'].astype(np.float64)

    policy_results = {}
    overview = {}
    for policy, config in POLICIES.items():
        local_top1, local_oracle = frame_local_readouts(
            official,
            risk,
            frame_tau,
            score,
            refined_error,
            budget=int(config['budget_on_risk']),
            width=int(config['width']),
        )
        temporal_top1 = v9c[f'beam_top1_{policy}'].astype(np.float64)
        temporal_oracle = v9c[f'beam_oracle_{policy}'].astype(np.float64)

        top_bootstrap = clip_bootstrap(
            clip_id, visible, temporal_top1, local_top1
        )
        oracle_bootstrap = clip_bootstrap(
            clip_id, visible, temporal_oracle, local_oracle
        )
        top_gate = build_gate(
            temporal_top1,
            local_top1,
            visible,
            sequence,
            v9c['reentry_first'].astype(bool),
            v9c['reentry_early8'].astype(bool),
            top_bootstrap,
        )
        oracle_gate = build_gate(
            temporal_oracle,
            local_oracle,
            visible,
            sequence,
            v9c['reentry_first'].astype(bool),
            v9c['reentry_early8'].astype(bool),
            oracle_bootstrap,
        )
        policy_results[policy] = {
            'configuration': config,
            'temporal_top1_vs_frame_local_score_top1': {
                'gate': top_gate,
                'bootstrap': top_bootstrap,
            },
            'temporal_oracle_vs_frame_local_same_capacity_oracle': {
                'gate': oracle_gate,
                'bootstrap': oracle_bootstrap,
            },
        }
        overview[policy] = {
            'top1_mean_difference': top_gate['summaries']['all_visible']['mean_difference'],
            'top1_ci': top_bootstrap['mean_difference_95_ci'],
            'top1_incremental_pass': top_gate['pass_all'],
            'oracle_mean_difference': oracle_gate['summaries']['all_visible']['mean_difference'],
            'oracle_ci': oracle_bootstrap['mean_difference_95_ci'],
            'oracle_incremental_pass': oracle_gate['pass_all'],
        }

    primary = policy_results[PRIMARY]
    primary_top_pass = primary[
        'temporal_top1_vs_frame_local_score_top1'
    ]['gate']['pass_all']
    primary_oracle_pass = primary[
        'temporal_oracle_vs_frame_local_same_capacity_oracle'
    ]['gate']['pass_all']
    decision = (
        'HISTORY_INCREMENTAL_VALUE_PASS: the primary temporal beam adds robust '
        'same-capacity selection/reachability beyond the current-frame baseline.'
        if primary_top_pass or primary_oracle_pass
        else 'HISTORY_INCREMENTAL_VALUE_FAIL: the primary history beam is useful '
        'relative to official final, but it is weaker than the same-frame, same-capacity '
        'score-top4 candidate oracle. The original V9-A5.1c system-level reachability '
        'pass is valid, but it is not evidence that temporal history adds incremental '
        'candidate reachability. Do not proceed directly to a learned temporal beam '
        'readout; first redesign pruning/state so it beats the frame-local capacity-matched baseline.'
    )

    result = {
        'date': '2026-07-10',
        'type': 'post-formal conservative same-capacity comparator review',
        'protocol': {
            'frame0_excluded': True,
            'visible_rows': int(np.sum(visible)),
            'risk_flags_replayed_from_v9a51c': True,
            'same_candidate_budget_and_width': True,
            'bootstrap_unit': 'clip_id',
            'bootstrap_resamples': BOOTSTRAP_RESAMPLES,
            'bootstrap_seed': BOOTSTRAP_SEED,
            'training': False,
            'davis_read': False,
        },
        'original_v9a51c_decision': v9c_report['gates']['decision'],
        'policies': policy_results,
        'policy_overview': overview,
        'primary_policy': PRIMARY,
        'primary_incremental_top1_pass': bool(primary_top_pass),
        'primary_incremental_oracle_pass': bool(primary_oracle_pass),
        'decision': decision,
        'integrity': {
            'checked_hashes': checked_hashes,
            'row_keys_exact': True,
            'unique_rows': int(len(set(keys_c))),
            'v9a51c_original_replay_zero': all(
                value == 0 or value == 0.0
                for value in v9c_report['integrity']['v9a51b_replay'].values()
            ),
            'v9a51c_official_parity_zero': max(
                v9c_report['integrity']['official_p_max_abs'],
                v9c_report['integrity']['official_v_max_abs'],
                v9c_report['integrity']['official_q_max_abs'],
            ) == 0.0,
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
                'decision': decision,
                'primary': overview[PRIMARY],
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
