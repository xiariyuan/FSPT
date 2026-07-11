#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path('/gemini/code/FSPT_v9a60_clean')
SOURCE_POOL_SCRIPT = Path('/gemini/code/FSPT/scripts/v9a38_build_pointodyssey_hypothesis_pool.py')
EXPORT_JSON = ROOT / 'outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_fused_top129_export.json'
EXPORT_NPZ = ROOT / 'outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_fused_top129_export.npz'
AUDIT_JSON = ROOT / 'outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_bounded_rank_shift_audit.json'
AUDIT_NPZ = ROOT / 'outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_bounded_rank_shift_audit_rows.npz'
OUT_JSON = ROOT / 'outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_visibility_stratified_postaudit.json'
OUT_DOC = ROOT / 'docs/v9a60_visibility_stratified_postaudit_2026-07-11.md'

EXPECTED_HEAD = 'a1c697de250e1d15f4653cdb77aa85e7e7ce8899'
EXPECTED_BRANCH = 'v9a60-correlation-rank-shift-20260711'
EXPECTED_HASHES = {
    SOURCE_POOL_SCRIPT: 'ff8aa0639872c2cf39fa43420982a95910bb155e8cad6275d59fd346c3724c30',
    EXPORT_JSON: '72fb654381eef236c73b54d8d52d5bfbe4a7b3e4cfaf7ebcf850f1cdf8472cbb',
    EXPORT_NPZ: '0849bb9ab700e8f5224bf751f897aea7a38e0efd6363d6d83297e04ecbfac3ac',
    AUDIT_JSON: 'a541d4475ca647a798827ac05a5befff588d33ceccfefb1229b85dff3a4f9370',
    AUDIT_NPZ: '24c3b49d85635aa6879ef084630422ccf63f145951eeae808d72a24d58da30a6',
}

SEQUENCES = ('ani', 'animal3', 'r4_new_f')
PRIMARY_INDEX = 2
WIDE_INDEX = 4


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


def conversion_summary(converted: np.ndarray, sequence: np.ndarray, clip_id: np.ndarray) -> dict[str, Any]:
    converted = np.asarray(converted, dtype=bool)
    result = {
        'n': int(len(converted)),
        'promoted': int(np.sum(converted)),
        'conversion': float(np.mean(converted)),
        'per_sequence': {},
        'per_clip': {},
    }
    for name in SEQUENCES:
        mask = sequence == name
        result['per_sequence'][name] = {
            'n': int(np.sum(mask)),
            'promoted': int(np.sum(converted & mask)),
            'conversion': float(np.mean(converted[mask])),
        }
    for clip in sorted(set(clip_id.tolist())):
        mask = clip_id == clip
        result['per_clip'][clip] = {
            'n': int(np.sum(mask)),
            'promoted': int(np.sum(converted & mask)),
            'conversion': float(np.mean(converted[mask])),
        }
    return result


def distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        'n': int(len(values)),
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'p90': float(np.quantile(values, 0.90)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
    }


def write_doc(result: dict[str, Any]) -> None:
    lines = [
        '# V9-A6.0 Visibility-Stratified Post-Audit',
        '',
        'Date: 2026-07-11',
        '',
        'This is a non-gating saved-artifact review. It does not alter the V9-A6.0 protocol, budgets, gates, or route decision.',
        '',
        '## Sampling boundary',
        '',
        '```json',
        json.dumps(result['sampling_boundary'], indent=2),
        '```',
        '',
        '## Opportunity visibility',
        '',
        '```json',
        json.dumps(result['opportunity_visibility'], indent=2),
        '```',
        '',
        '## Visible-only formal metrics',
        '',
        '```json',
        json.dumps(result['visible_only_metrics'], indent=2),
        '```',
        '',
        '## Decision',
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
    environment = verify_environment()
    export_report = json.loads(EXPORT_JSON.read_text())
    audit_report = json.loads(AUDIT_JSON.read_text())
    export = np.load(EXPORT_NPZ, allow_pickle=True)
    audit = np.load(AUDIT_NPZ, allow_pickle=True)

    if export_report['decision'].split(':', 1)[0] != 'EXPORT_PASS':
        raise RuntimeError('top129 export did not pass')
    if audit_report['gates']['decision'].split(':', 1)[0] != 'RANK_SHIFT_MAGNITUDE_FAIL':
        raise RuntimeError('unexpected formal rank-shift decision')
    if len(export['source_index']) != 4878 or len(audit['source_index']) != 760:
        raise RuntimeError('unexpected row count')

    source_index = audit['source_index'].astype(np.int64)
    if len(np.unique(source_index)) != 760:
        raise RuntimeError('duplicate opportunity source index')
    if not np.array_equal(export['source_index'][source_index], source_index):
        raise RuntimeError('opportunity/export source-index mismatch')

    all_visible = export['gt_visible'].astype(bool)
    opportunity_visible = all_visible[source_index]
    sequence = audit['sequence'].astype(str)
    clip_id = audit['clip_id'].astype(str)
    if not np.array_equal(sequence, export['sequence'][source_index].astype(str)):
        raise RuntimeError('sequence mismatch')
    if not np.array_equal(clip_id, export['clip_id'][source_index].astype(str)):
        raise RuntimeError('clip mismatch')

    primary_converted = audit['converted_by_budget'][:, PRIMARY_INDEX].astype(bool)
    wide_converted = audit['converted_by_budget'][:, WIDE_INDEX].astype(bool)
    if not np.all(all_visible):
        raise RuntimeError('canonical V9-A3.8/V9-A5C.0 pool unexpectedly contains invisible rows')
    if not np.all(opportunity_visible):
        raise RuntimeError('rank-shift opportunity set unexpectedly contains invisible rows')

    sampling_boundary = {
        'canonical_rows': 4878,
        'canonical_visible_rows': int(np.sum(all_visible)),
        'canonical_invisible_rows': int(np.sum(~all_visible)),
        'pool_builder_rule': (
            'V9-A3.8 skips a row before append when visible[t,track_id] is false '
            'or the trajectory coordinate is non-finite.'
        ),
        'pool_builder_source': str(SOURCE_POOL_SCRIPT),
        'pool_builder_sha256': sha256(SOURCE_POOL_SCRIPT),
        'implication': (
            'V9-A5C.0 and V9-A6.0 are visible-frame candidate audits by construction. '
            'They do not measure invisible-frame candidate recall or rank-shift magnitude.'
        ),
    }
    opportunity_visibility = {
        'opportunity_rows': 760,
        'visible_rows': int(np.sum(opportunity_visible)),
        'invisible_rows': int(np.sum(~opportunity_visible)),
        'per_sequence': {
            name: {
                'visible': int(np.sum(opportunity_visible & (sequence == name))),
                'invisible': int(np.sum((~opportunity_visible) & (sequence == name))),
            }
            for name in SEQUENCES
        },
        'per_clip': {
            clip: {
                'visible': int(np.sum(opportunity_visible & (clip_id == clip))),
                'invisible': int(np.sum((~opportunity_visible) & (clip_id == clip))),
            }
            for clip in sorted(set(clip_id.tolist()))
        },
        'visibility_stratification_available': False,
        'reason': 'The canonical pool and every rank-shift opportunity row are GT-visible.',
    }
    visible_only_metrics = {
        'primary_2x_g_ref': conversion_summary(primary_converted, sequence, clip_id),
        'wide_8x_g_ref': conversion_summary(wide_converted, sequence, clip_id),
        'span': distribution(audit['span']),
        'span_over_fused_std': distribution(audit['span_over_fused_std']),
        'target_rank': distribution(audit['target_rank']),
        'native_risk': {
            'opportunities': int(np.sum(audit['native_risk'])),
            'primary_promoted': int(np.sum(primary_converted & audit['native_risk'])),
            'wide_promoted': int(np.sum(wide_converted & audit['native_risk'])),
        },
        'reentry': {
            'first_opportunities': int(np.sum(audit['reentry_first'])),
            'first_primary_promoted': int(np.sum(primary_converted & audit['reentry_first'])),
            'early8_opportunities': int(np.sum(audit['reentry_early8'])),
            'early8_primary_promoted': int(np.sum(primary_converted & audit['reentry_early8'])),
        },
        'matches_formal_primary_count': int(np.sum(primary_converted)) == 67,
        'matches_formal_wide_count': int(np.sum(wide_converted)) == 264,
    }

    decision = (
        'VISIBILITY_STRATIFICATION_NOT_APPLICABLE: all 4,878 canonical candidate-audit '
        'rows and all 760 rank-shift opportunity rows are GT-visible by construction. '
        'The V9-A6.0 negative result already applies to the complete evaluated opportunity '
        'set and is not driven by invisible/occluded rows. Preserve the formal gate and '
        'route closure. Add an explicit visible-only sampling caveat to the paper; do not '
        'claim invisible-frame correlation recall coverage.'
    )
    result = {
        'date': '2026-07-11',
        'type': 'non-gating visibility stratification post-audit',
        'protocol': {
            'trackon_execution': False,
            'training': False,
            'davis_read': False,
            'changes_formal_gate': False,
            'adds_score_budget': False,
        },
        'sampling_boundary': sampling_boundary,
        'opportunity_visibility': opportunity_visibility,
        'visible_only_metrics': visible_only_metrics,
        'formal_decision': audit_report['gates']['decision'],
        'decision': decision,
        'integrity': {
            'environment': environment,
            'source_indices_exact': True,
            'sequence_keys_exact': True,
            'clip_keys_exact': True,
            'all_canonical_rows_visible': True,
            'all_opportunity_rows_visible': True,
            'formal_primary_count_reproduced': visible_only_metrics['matches_formal_primary_count'],
            'formal_wide_count_reproduced': visible_only_metrics['matches_formal_wide_count'],
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
                'canonical_visible': int(np.sum(all_visible)),
                'canonical_invisible': int(np.sum(~all_visible)),
                'opportunity_visible': int(np.sum(opportunity_visible)),
                'opportunity_invisible': int(np.sum(~opportunity_visible)),
                'decision': decision,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
