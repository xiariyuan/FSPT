#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path('/gemini/code/FSPT_v9a60_clean')
EXPORT_SCRIPT = ROOT / 'scripts/v9a60_export_fused_top129.py'
EXPORT_JSON = ROOT / 'outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_fused_top129_export.json'
EXPORT_NPZ = ROOT / 'outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_fused_top129_export.npz'
DESIGN = ROOT / 'docs/v9a60_bounded_correlation_rank_shift_design_2026-07-11.md'
OUT_JSON = ROOT / 'outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_bounded_rank_shift_audit.json'
OUT_NPZ = ROOT / 'outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_bounded_rank_shift_audit_rows.npz'
OUT_DOC = ROOT / 'docs/v9a60_bounded_rank_shift_audit_result_2026-07-11.md'

EXPECTED_HEAD = '7a8dc06638630ec83c22f9ea76e04870b28b545b'
EXPECTED_BRANCH = 'v9a60-correlation-rank-shift-20260711'
EXPECTED_HASHES = {
    EXPORT_SCRIPT: '684e7a6243d7fccddd23bef79355356dd78ef2b40d2257d1483f790ac680ed2c',
    EXPORT_JSON: '72fb654381eef236c73b54d8d52d5bfbe4a7b3e4cfaf7ebcf850f1cdf8472cbb',
    EXPORT_NPZ: '0849bb9ab700e8f5224bf751f897aea7a38e0efd6363d6d83297e04ecbfac3ac',
    DESIGN: '98893e60568e9788a72af505fd7048bb3cc26f4c2eea0c0923c684369ae82869',
}

SEQUENCES = ('ani', 'animal3', 'r4_new_f')
G_REF = 0.003143310546875
SPAN_MULTIPLIERS = np.asarray([0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float64)
SPAN_BUDGETS = G_REF * SPAN_MULTIPLIERS
PRIMARY_BUDGET_INDEX = 2
TARGET_START = 16
TARGET_STOP = 64
RADIUS_PX = 4.0
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 20260718
TOL = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_environment() -> dict[str, Any]:
    checked = {}
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f'hash mismatch: {path} -> {actual}')
        checked[str(path)] = actual
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    branch = subprocess.check_output(['git', '-C', str(ROOT), 'branch', '--show-current'], text=True).strip()
    tracked = subprocess.check_output(
        ['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no'],
        text=True,
    ).strip()
    if head != EXPECTED_HEAD:
        raise RuntimeError(f'HEAD mismatch: {head}')
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f'branch mismatch: {branch}')
    if tracked:
        raise RuntimeError(f'tracked worktree dirty: {tracked}')
    return {
        'head': head,
        'branch': branch,
        'tracked_status': tracked,
        'checked_hashes': checked,
        'script': {
            'path': str(Path(__file__).resolve()),
            'sha256': sha256(Path(__file__).resolve()),
        },
    }


def distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        'n': int(len(values)),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'p10': float(np.quantile(values, 0.10)),
        'median': float(np.median(values)),
        'p75': float(np.quantile(values, 0.75)),
        'p90': float(np.quantile(values, 0.90)),
        'p95': float(np.quantile(values, 0.95)),
        'p99': float(np.quantile(values, 0.99)),
        'max': float(np.max(values)),
    }


def conversion_metrics(
    converted: np.ndarray,
    sequence: np.ndarray,
    clip_id: np.ndarray,
    native_risk: np.ndarray,
    reentry_first: np.ndarray,
    reentry_early8: np.ndarray,
    target_rank: np.ndarray,
    total_rows: int,
    total_hard_rows: int,
) -> dict[str, Any]:
    converted = np.asarray(converted, dtype=bool)
    n = len(converted)
    count = int(np.sum(converted))
    result = {
        'opportunity_rows': int(n),
        'promoted_rows': count,
        'opportunity_conversion': float(count / n),
        'implied_global_recall4_gain': float(count / total_rows),
        'implied_gt_hard_recall4_gain': float(count / total_hard_rows),
        'per_sequence': {},
        'per_clip': {},
    }
    for name in SEQUENCES:
        mask = sequence == name
        count_rows = int(np.sum(mask))
        result['per_sequence'][name] = {
            'opportunities': count_rows,
            'promoted': int(np.sum(converted & mask)),
            'conversion': (
                float(np.mean(converted[mask])) if count_rows > 0 else None
            ),
        }
    for clip in sorted(set(clip_id.tolist())):
        mask = clip_id == clip
        result['per_clip'][clip] = {
            'opportunities': int(np.sum(mask)),
            'promoted': int(np.sum(converted & mask)),
            'conversion': float(np.mean(converted[mask])),
        }
    risk_count = int(np.sum(converted & native_risk))
    risk_opp = int(np.sum(native_risk))
    result['native_risk'] = {
        'opportunities': risk_opp,
        'promoted': risk_count,
        'conversion_within_risk_opportunities': float(risk_count / risk_opp),
        'implied_global_recall4_gain': float(risk_count / total_rows),
        'opportunity_coverage_ceiling': float(risk_opp / n),
    }
    result['reentry'] = {
        'first': {
            'opportunities': int(np.sum(reentry_first)),
            'promoted': int(np.sum(converted & reentry_first)),
            'conversion': (
                float(np.mean(converted[reentry_first]))
                if np.any(reentry_first)
                else None
            ),
        },
        'early8': {
            'opportunities': int(np.sum(reentry_early8)),
            'promoted': int(np.sum(converted & reentry_early8)),
            'conversion': (
                float(np.mean(converted[reentry_early8]))
                if np.any(reentry_early8)
                else None
            ),
        },
    }
    rank_bins = ((17, 24), (25, 32), (33, 48), (49, 64))
    result['target_rank_bins'] = {}
    for lower, upper in rank_bins:
        mask = (target_rank >= lower) & (target_rank <= upper)
        result['target_rank_bins'][f'{lower}-{upper}'] = {
            'opportunities': int(np.sum(mask)),
            'promoted': int(np.sum(converted & mask)),
            'conversion': float(np.mean(converted[mask])) if np.any(mask) else None,
        }
    return result


def clip_block_bootstrap(
    converted: np.ndarray,
    clip_id: np.ndarray,
    all_clip_id: np.ndarray,
    all_is_hard: np.ndarray,
) -> dict[str, Any]:
    clips = sorted(set(all_clip_id.tolist()))
    opp_count = np.asarray([np.sum(clip_id == clip) for clip in clips], dtype=np.int64)
    converted_count = np.asarray(
        [np.sum(converted & (clip_id == clip)) for clip in clips], dtype=np.int64
    )
    total_count = np.asarray([np.sum(all_clip_id == clip) for clip in clips], dtype=np.int64)
    hard_count = np.asarray(
        [np.sum((all_clip_id == clip) & all_is_hard) for clip in clips], dtype=np.int64
    )
    if np.any(opp_count <= 0) or np.any(total_count <= 0) or np.any(hard_count <= 0):
        raise RuntimeError('invalid clip denominator for bootstrap')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    conversion = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    global_gain = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    hard_gain = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    chunk = 1000
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        count = min(chunk, BOOTSTRAP_RESAMPLES - start)
        indices = rng.integers(0, len(clips), size=(count, len(clips)))
        selected_promoted = np.sum(converted_count[indices], axis=1)
        conversion[start : start + count] = selected_promoted / np.sum(
            opp_count[indices], axis=1
        )
        global_gain[start : start + count] = selected_promoted / np.sum(
            total_count[indices], axis=1
        )
        hard_gain[start : start + count] = selected_promoted / np.sum(
            hard_count[indices], axis=1
        )
    leave_one_out = {}
    for index, clip in enumerate(clips):
        keep = np.arange(len(clips)) != index
        leave_one_out[clip] = float(
            np.sum(converted_count[keep]) / np.sum(opp_count[keep])
        )
    return {
        'clips': clips,
        'per_clip_opportunities': {
            clip: int(value) for clip, value in zip(clips, opp_count)
        },
        'per_clip_promoted': {
            clip: int(value) for clip, value in zip(clips, converted_count)
        },
        'per_clip_conversion': {
            clip: float(promoted / opportunity)
            for clip, promoted, opportunity in zip(
                clips, converted_count, opp_count
            )
        },
        'conversion_95_ci': [
            float(value) for value in np.quantile(conversion, [0.025, 0.975])
        ],
        'global_gain_95_ci': [
            float(value) for value in np.quantile(global_gain, [0.025, 0.975])
        ],
        'hard_gain_95_ci': [
            float(value) for value in np.quantile(hard_gain, [0.025, 0.975])
        ],
        'probability_conversion_ge_0_50': float(np.mean(conversion >= 0.50)),
        'leave_one_clip_out_conversion': leave_one_out,
        'bootstrap_resamples': BOOTSTRAP_RESAMPLES,
        'bootstrap_seed': BOOTSTRAP_SEED,
    }


def component_diagnostics(
    delta: np.ndarray,
    weights: np.ndarray,
    sequence: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    weighted = delta * weights[None, :]
    result = {'all': {}, 'per_sequence': {}}
    for index, name in enumerate(names):
        result['all'][name] = {
            'delta_distribution': distribution(delta[:, index]),
            'positive_fraction': float(np.mean(delta[:, index] > 0)),
            'weighted_delta_distribution': distribution(weighted[:, index]),
        }
    for seq_name in SEQUENCES:
        mask = sequence == seq_name
        result['per_sequence'][seq_name] = {}
        for index, name in enumerate(names):
            if not np.any(mask):
                result['per_sequence'][seq_name][name] = {
                    'n': 0,
                    'median_delta': None,
                    'positive_fraction': None,
                    'median_weighted_delta': None,
                }
            else:
                result['per_sequence'][seq_name][name] = {
                    'n': int(np.sum(mask)),
                    'median_delta': float(np.median(delta[mask, index])),
                    'positive_fraction': float(np.mean(delta[mask, index] > 0)),
                    'median_weighted_delta': float(
                        np.median(weighted[mask, index])
                    ),
                }
    return result


def write_doc(result: dict[str, Any], path: Path) -> None:
    primary = result['budgets'][str(float(SPAN_MULTIPLIERS[PRIMARY_BUDGET_INDEX]))]
    lines = [
        '# V9-A6.0 Bounded Correlation Rank-Shift Audit Result',
        '',
        'Date: 2026-07-11',
        '',
        'This is a no-training GT-directed score-magnitude necessary-condition audit. It is not an adapter evaluation.',
        '',
        '## Primary budget',
        '',
        '```json',
        json.dumps(primary, indent=2),
        '```',
        '',
        '## Magnitude distributions',
        '',
        '```json',
        json.dumps(result['magnitude'], indent=2),
        '```',
        '',
        '## Gates',
        '',
        '```json',
        json.dumps(result['gates'], indent=2),
        '```',
        '',
        '## Decision',
        '',
        result['gates']['decision'],
        '',
        '## Integrity',
        '',
        '```json',
        json.dumps(result['integrity'], indent=2),
        '```',
    ]
    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-opportunities', type=int, default=0)
    parser.add_argument('--out-json', type=Path, default=OUT_JSON)
    parser.add_argument('--out-npz', type=Path, default=OUT_NPZ)
    parser.add_argument('--out-doc', type=Path, default=OUT_DOC)
    args = parser.parse_args()

    environment = verify_environment()
    export_report = json.loads(EXPORT_JSON.read_text())
    data = np.load(EXPORT_NPZ, allow_pickle=True)
    if export_report['decision'].split(':', 1)[0] != 'EXPORT_PASS':
        raise RuntimeError('formal export did not pass')
    if len(data['source_index']) != 4878:
        raise RuntimeError('unexpected export row count')

    all_clip = data['clip_id'].astype(str)
    all_sequence = data['sequence'].astype(str)
    all_hard = data['is_hard'].astype(bool)
    opportunity = (~data['recall4_k16'].astype(bool)) & data['recall4_k64'].astype(bool)
    opportunity_indices = np.flatnonzero(opportunity)
    if len(opportunity_indices) != 760:
        raise RuntimeError(f'opportunity count mismatch: {len(opportunity_indices)}')
    if args.max_opportunities > 0:
        selected = opportunity_indices[: args.max_opportunities]
    else:
        selected = opportunity_indices
    formal_run = args.max_opportunities == 0

    top_scores = data['top_scores'][selected].astype(np.float64)
    top_errors = data['top_errors_px'][selected].astype(np.float64)
    top_components = data['top_components'][selected].astype(np.float64)
    fused_std = data['fused_std'][selected].astype(np.float64)
    sequence = data['sequence'][selected].astype(str)
    clip_id = data['clip_id'][selected].astype(str)
    native_risk = data['native_risk'][selected].astype(bool)
    reentry_first = data['reentry_first'][selected].astype(bool)
    reentry_early8 = data['reentry_early8'][selected].astype(bool)
    weights = data['component_weights'].astype(np.float64)
    component_names = data['component_names'].tolist()

    good_window = top_errors[:, TARGET_START:TARGET_STOP] <= RADIUS_PX
    if not np.all(np.any(good_window, axis=1)):
        raise RuntimeError('opportunity without <=4px candidate in ranks 17-64')
    first_good_offset = np.argmax(good_window, axis=1)
    target_position = TARGET_START + first_good_offset
    row = np.arange(len(selected))
    target_rank = target_position + 1
    target_score = top_scores[row, target_position]
    target_error = top_errors[row, target_position]
    boundary_score = top_scores[:, 15]
    boundary_error = top_errors[:, 15]
    span = boundary_score - target_score
    positive_epsilon = np.nextafter(boundary_score, np.inf) - target_score
    signed_epsilon = span / 2.0
    span_over_std = span / fused_std
    positive_over_std = positive_epsilon / fused_std
    signed_over_std = signed_epsilon / fused_std
    span_over_gref = span / G_REF
    target_component = top_components[row, target_position]
    boundary_component = top_components[:, 15]
    component_delta = target_component - boundary_component
    weighted_component_delta = component_delta * weights[None, :]

    if np.any(target_rank < 17) or np.any(target_rank > 64):
        raise RuntimeError('target rank outside 17-64')
    if np.any(target_error > RADIUS_PX):
        raise RuntimeError('target error exceeds 4px')
    if np.any(boundary_error <= RADIUS_PX):
        raise RuntimeError('K16-miss row has <=4px rank16 boundary')
    if np.any(span < -TOL):
        raise RuntimeError('negative rank-shift span')
    for index, position in enumerate(target_position):
        if np.any(top_errors[index, TARGET_START:position] <= RADIUS_PX):
            raise RuntimeError('target is not the first <=4px rank17-64 candidate')
    if np.max(np.abs(np.sum(weighted_component_delta, axis=1) + span)) > 1e-5:
        raise RuntimeError('component delta does not reconstruct score span')
    if np.max(np.abs(2.0 * signed_epsilon - span)) > TOL:
        raise RuntimeError('signed epsilon formula mismatch')
    if np.any(positive_epsilon < span):
        raise RuntimeError('positive epsilon is smaller than score span')
    numeric = [
        target_score,
        target_error,
        boundary_score,
        boundary_error,
        span,
        positive_epsilon,
        signed_epsilon,
        span_over_std,
        positive_over_std,
        signed_over_std,
        span_over_gref,
        component_delta,
        weighted_component_delta,
    ]
    if not all(np.all(np.isfinite(value)) for value in numeric):
        raise RuntimeError('non-finite rank-shift output')

    conversion = span[:, None] <= SPAN_BUDGETS[None, :]
    positive_conversion = positive_epsilon[:, None] <= SPAN_BUDGETS[None, :]
    budgets = {}
    for index, multiplier in enumerate(SPAN_MULTIPLIERS.tolist()):
        metrics = conversion_metrics(
            conversion[:, index],
            sequence,
            clip_id,
            native_risk,
            reentry_first,
            reentry_early8,
            target_rank,
            total_rows=4878,
            total_hard_rows=2419,
        )
        metrics['span_multiplier'] = float(multiplier)
        metrics['span_budget'] = float(SPAN_BUDGETS[index])
        metrics['positive_strict_promoted_rows'] = int(
            np.sum(positive_conversion[:, index])
        )
        metrics['signed_pairwise_per_cell_epsilon'] = float(
            SPAN_BUDGETS[index] / 2.0
        )
        budgets[str(float(multiplier))] = metrics

    primary_converted = conversion[:, PRIMARY_BUDGET_INDEX]
    bootstrap = (
        clip_block_bootstrap(
            primary_converted,
            clip_id,
            all_clip,
            all_hard,
        )
        if formal_run
        else {
            'smoke_only': True,
            'bootstrap_disabled': True,
            'reason': 'subset does not contain all preregistered clips/sequences',
        }
    )
    magnitude = {
        'span': distribution(span),
        'positive_epsilon': distribution(positive_epsilon),
        'signed_epsilon': distribution(signed_epsilon),
        'span_over_fused_std': distribution(span_over_std),
        'positive_epsilon_over_fused_std': distribution(positive_over_std),
        'signed_epsilon_over_fused_std': distribution(signed_over_std),
        'span_over_g_ref': distribution(span_over_gref),
        'per_sequence': {
            name: (
                {
                    'span': distribution(span[sequence == name]),
                    'span_over_fused_std': distribution(
                        span_over_std[sequence == name]
                    ),
                    'target_rank': distribution(target_rank[sequence == name]),
                }
                if np.any(sequence == name)
                else {'n': 0}
            )
            for name in SEQUENCES
        },
        'target_rank': distribution(target_rank),
        'target_error': distribution(target_error),
        'rank16_error': distribution(boundary_error),
    }
    components = component_diagnostics(
        component_delta,
        weights,
        sequence,
        component_names,
    )

    primary = budgets[str(float(SPAN_MULTIPLIERS[PRIMARY_BUDGET_INDEX]))]

    if formal_run:
        clips_ge_030 = sum(
            row_value['conversion'] >= 0.30
            for row_value in primary['per_clip'].values()
        )
        sequence_conversion_gate = {
            name: primary['per_sequence'][name]['conversion'] >= 0.40
            for name in SEQUENCES
        }
        sequence_magnitude_gate = {
            name: magnitude['per_sequence'][name]['span_over_fused_std']['median']
            <= 0.02
            for name in SEQUENCES
        }
        gate_checks = {
            'global_conversion_ge_0_50': primary['opportunity_conversion'] >= 0.50,
            'sequence_conversion_ge_0_40': sequence_conversion_gate,
            'clips_ge_0_30_at_least_7': clips_ge_030 >= 7,
            'bootstrap_conversion_ci_lower_gt_0_35': bootstrap[
                'conversion_95_ci'
            ][0]
            > 0.35,
            'implied_global_gain_ge_0_075': primary[
                'implied_global_recall4_gain'
            ]
            >= 0.075,
            'implied_hard_gain_ge_0_12': primary[
                'implied_gt_hard_recall4_gain'
            ]
            >= 0.12,
            'sequence_median_span_over_std_le_0_02': sequence_magnitude_gate,
            'global_p90_span_over_std_le_0_10': magnitude[
                'span_over_fused_std'
            ]['p90']
            <= 0.10,
        }
        pass_all = bool(
            gate_checks['global_conversion_ge_0_50']
            and all(sequence_conversion_gate.values())
            and gate_checks['clips_ge_0_30_at_least_7']
            and gate_checks['bootstrap_conversion_ci_lower_gt_0_35']
            and gate_checks['implied_global_gain_ge_0_075']
            and gate_checks['implied_hard_gain_ge_0_12']
            and all(sequence_magnitude_gate.values())
            and gate_checks['global_p90_span_over_std_le_0_10']
        )
        decision = (
            'RANK_SHIFT_MAGNITUDE_PASS: a small, preregistered target-directed score span promotes a meaningful and sequence-consistent fraction of K16-to-K64 opportunity rows. This authorizes only a new sequence-heldout V9-A6.1 bounded residual adapter preregistration; it does not establish adapter safety or final tracking gain.'
            if pass_all
            else 'RANK_SHIFT_MAGNITUDE_FAIL: even the optimistic target-directed oracle does not satisfy the preregistered small-span, sequence/clip-consistency and implied-recall gates. Close the bounded correlation residual route; do not train or read DAVIS.'
        )
        gates = {
            'formal_run': True,
            'checks': gate_checks,
            'clips_conversion_ge_0_30': int(clips_ge_030),
            'pass_all': pass_all,
            'decision': decision,
        }
    else:
        gates = {
            'formal_run': False,
            'pass_all': False,
            'decision': 'RANK_SHIFT_SMOKE_ONLY: target extraction and score-span formulas are validated on a subset. Formal gates are disabled.',
        }

    arrays = {
        'source_index': data['source_index'][selected].astype(np.int64),
        'clip_id': clip_id.astype(object),
        'sequence': sequence.astype(object),
        'frame_tau': data['frame_tau'][selected].astype(np.int32),
        'query_idx': data['query_idx'][selected].astype(np.int32),
        'native_risk': native_risk,
        'is_hard': data['is_hard'][selected].astype(bool),
        'reentry_first': reentry_first,
        'reentry_early8': reentry_early8,
        'target_position': target_position.astype(np.int16),
        'target_rank': target_rank.astype(np.int16),
        'target_score': target_score.astype(np.float64),
        'target_error_px': target_error.astype(np.float64),
        'rank16_score': boundary_score.astype(np.float64),
        'rank16_error_px': boundary_error.astype(np.float64),
        'span': span.astype(np.float64),
        'positive_epsilon': positive_epsilon.astype(np.float64),
        'signed_epsilon': signed_epsilon.astype(np.float64),
        'fused_std': fused_std.astype(np.float64),
        'span_over_fused_std': span_over_std.astype(np.float64),
        'positive_epsilon_over_fused_std': positive_over_std.astype(np.float64),
        'signed_epsilon_over_fused_std': signed_over_std.astype(np.float64),
        'span_over_g_ref': span_over_gref.astype(np.float64),
        'component_delta': component_delta.astype(np.float64),
        'weighted_component_delta': weighted_component_delta.astype(np.float64),
        'component_names': np.asarray(component_names, dtype=object),
        'component_weights': weights.astype(np.float64),
        'span_multipliers': SPAN_MULTIPLIERS,
        'span_budgets': SPAN_BUDGETS,
        'converted_by_budget': conversion,
        'positive_strict_converted_by_budget': positive_conversion,
    }

    result = {
        'date': '2026-07-11',
        'protocol': {
            'formal_run': formal_run,
            'source_rows': 4878,
            'opportunity_rows_available': 760,
            'opportunity_rows_evaluated': int(len(selected)),
            'target_rank_range': [17, 64],
            'target_radius_px': RADIUS_PX,
            'g_ref': G_REF,
            'span_multipliers': SPAN_MULTIPLIERS.tolist(),
            'span_budgets': SPAN_BUDGETS.tolist(),
            'primary_multiplier': float(
                SPAN_MULTIPLIERS[PRIMARY_BUDGET_INDEX]
            ),
            'primary_span_budget': float(
                SPAN_BUDGETS[PRIMARY_BUDGET_INDEX]
            ),
            'training': False,
            'davis_read': False,
            'trackon_execution': False,
        },
        'integrity': {
            'environment': environment,
            'export_decision': export_report['decision'],
            'target_exists_every_row': True,
            'target_rank_in_17_64': True,
            'target_error_le4': True,
            'rank16_error_gt4': True,
            'target_is_first_good_candidate': True,
            'span_nonnegative': True,
            'component_score_reconstruction_max_abs': float(
                np.max(np.abs(np.sum(weighted_component_delta, axis=1) + span))
            ),
            'signed_formula_max_abs': float(
                np.max(np.abs(2.0 * signed_epsilon - span))
            ),
            'positive_epsilon_ge_span': True,
            'all_outputs_finite': True,
        },
        'magnitude': magnitude,
        'budgets': budgets,
        'primary_bootstrap': bootstrap,
        'component_diagnostics_non_gating': components,
        'gates': gates,
        'provenance': {
            'branch': environment['branch'],
            'head': environment['head'],
            'script_sha256': sha256(Path(__file__).resolve()),
            'export_script_sha256': sha256(EXPORT_SCRIPT),
            'export_json_sha256': sha256(EXPORT_JSON),
            'export_npz_sha256': sha256(EXPORT_NPZ),
            'design_sha256': sha256(DESIGN),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_doc.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(args.out_npz, **arrays)
    write_doc(result, args.out_doc)
    print(
        json.dumps(
            {
                'ok': True,
                'json': str(args.out_json),
                'npz': str(args.out_npz),
                'doc': str(args.out_doc),
                'formal_run': formal_run,
                'opportunities': int(len(selected)),
                'primary': primary,
                'primary_bootstrap': bootstrap,
                'magnitude': {
                    'span': magnitude['span'],
                    'span_over_fused_std': magnitude['span_over_fused_std'],
                    'target_rank': magnitude['target_rank'],
                },
                'gates': gates,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
