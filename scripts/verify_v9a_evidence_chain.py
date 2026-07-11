#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INDEX_JSON = ROOT / 'outputs/paper_discovery_2026-07-05/v9a_evidence/v9a_evidence_index.json'
OUT_JSON = ROOT / 'outputs/paper_discovery_2026-07-05/v9a_evidence/v9a_evidence_verification.json'
OUT_DOC = ROOT / 'docs/V9A_EVIDENCE_VERIFICATION_2026-07-11.md'
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


def git(*args: str) -> str:
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def assert_close(name: str, actual: float, expected: float, tolerance: float = TOL) -> None:
    if abs(float(actual) - float(expected)) > tolerance:
        raise RuntimeError(f'{name}: {actual} != {expected} within {tolerance}')


def route_map(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row['id']: row for row in index['routes']}


def verify_index_and_artifacts(index: dict[str, Any]) -> dict[str, Any]:
    if index['schema_version'] != 'v9a-evidence-index-v1':
        raise RuntimeError(f"unexpected schema: {index['schema_version']}")
    current_head = git('rev-parse', 'HEAD')
    generated_head = index['generated_from']['head']
    subprocess.run(
        ['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', generated_head, current_head],
        check=True,
    )
    checked_artifacts = []
    for route in index['routes']:
        subprocess.run(
            ['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', route['commit_full'], current_head],
            check=True,
        )
        for item in route['artifacts']:
            path = ROOT / item['path']
            if not path.exists():
                raise RuntimeError(f"missing indexed artifact: {item['path']}")
            actual_hash = sha256(path)
            actual_size = path.stat().st_size
            if actual_hash != item['sha256'] or actual_size != item['size_bytes']:
                raise RuntimeError(
                    f"artifact mismatch: {item['path']} -> {actual_size}/{actual_hash}"
                )
            checked_artifacts.append(item['path'])
    return {
        'current_head': current_head,
        'generated_head': generated_head,
        'routes': len(index['routes']),
        'artifacts_checked': len(checked_artifacts),
        'unique_artifacts_checked': len(set(checked_artifacts)),
    }


def verify_v9a45(route: dict[str, Any]) -> dict[str, Any]:
    result = json.loads((ROOT / route['result_json']).read_text())
    teacher = result['validation']['teacher']
    student = result['validation']['student']
    comparison = result['validation']['comparison']
    assert_close(
        'V9-A4.5 student-teacher mean difference',
        student['mean_error'] - teacher['mean_error'],
        comparison['mean_difference_student_minus_teacher'],
        1e-12,
    )
    if result['validation']['gate']['pass_all']:
        raise RuntimeError('V9-A4.5 unexpectedly passes')
    if not result['decision'].startswith('Integrity passes, but heldout synthetic ranking collapses'):
        raise RuntimeError('V9-A4.5 decision text mismatch')
    return {
        'teacher_mean': teacher['mean_error'],
        'student_mean': student['mean_error'],
        'difference': comparison['mean_difference_student_minus_teacher'],
        'gate_pass': False,
    }


def verify_v9a50(route: dict[str, Any]) -> dict[str, Any]:
    result = json.loads((ROOT / route['result_json']).read_text())
    if result['gates']['passing_sampled_causal']:
        raise RuntimeError('V9-A5.0 sampled-causal policy unexpectedly passes')
    if len(result['gates']['passing_oracle_state']) < 1:
        raise RuntimeError('V9-A5.0 oracle-state headroom missing')
    sampled = result['gates']['best_sampled_causal_by_global_mean']
    oracle = result['gates']['best_oracle_state_by_global_mean']
    if result['metrics'][oracle]['all']['mean_difference_vs_teacher'] >= 0:
        raise RuntimeError('V9-A5.0 oracle state does not improve teacher')
    if not result['gates']['decision'].startswith('ORACLE_STATE_HEADROOM_ONLY'):
        raise RuntimeError('V9-A5.0 decision mismatch')
    return {
        'best_sampled': sampled,
        'sampled_difference': result['metrics'][sampled]['all']['mean_difference_vs_teacher'],
        'best_oracle': oracle,
        'oracle_difference': result['metrics'][oracle]['all']['mean_difference_vs_teacher'],
        'sampled_pass_count': 0,
        'oracle_pass_count': len(result['gates']['passing_oracle_state']),
    }


def verify_v9a5c0(route: dict[str, Any]) -> dict[str, Any]:
    result = json.loads((ROOT / route['result_json']).read_text())
    npz_path = next(ROOT / item['path'] for item in route['artifacts'] if item['role'] == 'result_npz')
    data = np.load(npz_path, allow_pickle=True)
    map_i = data['map_names'].tolist().index('fused')
    k16_i = data['k_values'].tolist().index(16)
    k64_i = data['k_values'].tolist().index(64)
    r4_i = data['radii'].tolist().index(4.0)
    sequence = data['sequence'].astype(str)
    recall16 = data['recall'][:, map_i, k16_i, r4_i].astype(bool)
    recall64 = data['recall'][:, map_i, k64_i, r4_i].astype(bool)
    for name in ('ani', 'animal3', 'r4_new_f'):
        mask = sequence == name
        expected = result['gate']['per_sequence'][name]
        assert_close(f'{name} K16 recall', np.mean(recall16[mask]), expected['fused_top16_recall4'], 1e-12)
        assert_close(f'{name} K64 recall', np.mean(recall64[mask]), expected['fused_top64_recall4'], 1e-12)
        assert_close(
            f'{name} headroom',
            np.mean(recall64[mask]) - np.mean(recall16[mask]),
            expected['fused_headroom'],
            1e-12,
        )
        if not expected['pass_fused_headroom_ge_0.02']:
            raise RuntimeError(f'V9-A5C.0 sequence gate fails: {name}')
    if not result['gate']['pass_all_sequences']:
        raise RuntimeError('V9-A5C.0 global gate unexpectedly fails')
    return {
        'rows': len(sequence),
        'global_k16_recall4': float(np.mean(recall16)),
        'global_k64_recall4': float(np.mean(recall64)),
        'opportunity_rows': int(np.sum((~recall16) & recall64)),
        'gate_pass': True,
    }


def verify_v9a51a(route: dict[str, Any]) -> dict[str, Any]:
    result = json.loads((ROOT / route['result_json']).read_text())
    npz_path = next(ROOT / item['path'] for item in route['artifacts'] if item['role'] == 'result_npz')
    data = np.load(npz_path, allow_pickle=True)
    rows = np.arange(len(data['candidate_error']))
    visible = data['visible'].astype(bool)
    candidate_error = data['candidate_error'].astype(np.float64)
    teacher = candidate_error[rows, data['selection_teacher_top1'].astype(int)]
    beam_top1 = candidate_error[rows, data['selection_beam4_motion_latent_top1'].astype(int)]
    teacher_oracle = candidate_error[rows, data['selection_teacher_top4_oracle'].astype(int)]
    beam_oracle = candidate_error[rows, data['selection_beam4_motion_latent_oracle'].astype(int)]
    assert_close('V9-A5.1a teacher mean', np.mean(teacher[visible]), result['metrics']['teacher_top1']['all_visible']['mean_error'], 1e-6)
    assert_close('V9-A5.1a beam top1 mean', np.mean(beam_top1[visible]), result['metrics']['beam4_motion_latent_top1']['all_visible']['mean_error'], 1e-6)
    assert_close('V9-A5.1a teacher oracle mean', np.mean(teacher_oracle[visible]), result['metrics']['teacher_top4_oracle']['all_visible']['mean_error'], 1e-6)
    assert_close('V9-A5.1a beam oracle mean', np.mean(beam_oracle[visible]), result['metrics']['beam4_motion_latent_oracle']['all_visible']['mean_error'], 1e-6)
    if result['gates']['deterministic_top1']['pass_all'] or result['gates']['beam_reachability']['pass_all']:
        raise RuntimeError('V9-A5.1a unexpectedly passes')
    return {
        'rows': len(rows),
        'visible_rows': int(np.sum(visible)),
        'teacher_top1_mean': float(np.mean(teacher[visible])),
        'beam_top1_mean': float(np.mean(beam_top1[visible])),
        'teacher_top4_oracle_mean': float(np.mean(teacher_oracle[visible])),
        'beam_top4_oracle_mean': float(np.mean(beam_oracle[visible])),
    }


def verify_v9a51b(route: dict[str, Any]) -> dict[str, Any]:
    result = json.loads((ROOT / route['result_json']).read_text())
    npz_path = next(ROOT / item['path'] for item in route['artifacts'] if item['role'] == 'result_npz')
    data = np.load(npz_path, allow_pickle=True)
    active = data['risk'].astype(bool) & (data['frame_tau'].astype(int) > 0)
    official = data['official_final'].astype(np.float64)
    refined64 = data['refined_oracle_K64'].astype(np.float64)
    expected = np.where(active, np.minimum(official, refined64), official)
    if np.max(np.abs(expected - data['hybrid_refined_oracle'])) > TOL:
        raise RuntimeError('V9-A5.1b hybrid oracle formula mismatch')
    visible = data['gt_visible'].astype(bool) & (data['frame_tau'].astype(int) > 0)
    assert_close('V9-A5.1b official mean', np.mean(official[visible]), result['summaries']['official_final']['all_visible']['mean_error'], 1e-9)
    assert_close('V9-A5.1b oracle mean', np.mean(expected[visible]), result['summaries']['hybrid_refined_oracle']['all_visible']['mean_error'], 1e-9)
    if not result['gates']['pass_all']:
        raise RuntimeError('V9-A5.1b oracle gate unexpectedly fails')
    return {
        'rows': len(official),
        'visible_rows': int(np.sum(visible)),
        'official_mean': float(np.mean(official[visible])),
        'hybrid_oracle_mean': float(np.mean(expected[visible])),
        'gate_pass': True,
    }


def verify_v9a51c(route: dict[str, Any]) -> dict[str, Any]:
    result = json.loads((ROOT / route['result_json']).read_text())
    fair = json.loads((ROOT / route['extra_json']).read_text())
    npz_path = next(ROOT / item['path'] for item in route['artifacts'] if item['role'] == 'result_npz')
    data = np.load(npz_path, allow_pickle=True)
    active = data['risk'].astype(bool) & (data['frame_tau'].astype(int) > 0)
    official = data['official_final'].astype(np.float64)
    refined = data['beam_refined_error_native_dynamic_B4'].astype(np.float64)
    top1 = np.where(active, refined[:, 0], official)
    oracle = np.where(active, np.minimum(official, np.min(refined, axis=1)), official)
    if np.max(np.abs(top1 - data['beam_top1_native_dynamic_B4'])) > TOL:
        raise RuntimeError('V9-A5.1c top1 formula mismatch')
    if np.max(np.abs(oracle - data['beam_oracle_native_dynamic_B4'])) > TOL:
        raise RuntimeError('V9-A5.1c oracle formula mismatch')
    if result['gates']['deterministic_top1']['pass_all']:
        raise RuntimeError('V9-A5.1c deterministic gate unexpectedly passes')
    if not result['gates']['beam_reachability']['pass_all']:
        raise RuntimeError('V9-A5.1c raw oracle gate unexpectedly fails')
    if fair['primary_incremental_top1_pass'] or fair['primary_incremental_oracle_pass']:
        raise RuntimeError('V9-A5.1c fair incremental gate unexpectedly passes')
    if fair['policy_overview']['native_dynamic_B4']['oracle_mean_difference'] <= 0:
        raise RuntimeError('V9-A5.1c fair oracle is not worse than frame-local control')
    return {
        'raw_oracle_pass': True,
        'deterministic_pass': False,
        'fair_top1_difference': fair['policy_overview']['native_dynamic_B4']['top1_mean_difference'],
        'fair_oracle_difference': fair['policy_overview']['native_dynamic_B4']['oracle_mean_difference'],
        'final_incremental_pass': False,
    }


def verify_v9a52(route: dict[str, Any]) -> dict[str, Any]:
    result = json.loads((ROOT / route['result_json']).read_text())
    npz_path = next(ROOT / item['path'] for item in route['artifacts'] if item['role'] == 'result_npz')
    data = np.load(npz_path, allow_pickle=True)
    shared = np.minimum(data['official_error'], data['shared_score_top1_error'])
    independent = np.minimum(data['branch_a_error'], data['branch_b_error'])
    if np.max(np.abs(shared - data['shared_b2_oracle'])) > TOL:
        raise RuntimeError('V9-A5.2 shared B2 formula mismatch')
    if np.max(np.abs(independent - data['independent_b2_oracle'])) > TOL:
        raise RuntimeError('V9-A5.2 independent B2 formula mismatch')
    mask = data['gt_visible'].astype(bool) & (data['horizon'].astype(int) == 8)
    assert_close(
        'V9-A5.2 horizon8 difference',
        np.mean(independent[mask] - shared[mask]),
        result['summaries']['horizon8_all_visible']['mean_difference'],
        1e-9,
    )
    if result['gates']['pass_all']:
        raise RuntimeError('V9-A5.2 unexpectedly passes')
    return {
        'rows': len(shared),
        'horizon8_visible_rows': int(np.sum(mask)),
        'horizon8_difference': float(np.mean(independent[mask] - shared[mask])),
        'q_new_divergence_fraction': float(np.mean(data['target_q_new_l2'][data['gt_visible']] > 1e-4)),
        'useful_novel_rows': int(np.sum(data['useful_novel'] & data['gt_visible'])),
        'gate_pass': False,
    }


def verify_v9a60(route: dict[str, Any]) -> dict[str, Any]:
    result = json.loads((ROOT / route['result_json']).read_text())
    boundary = json.loads((ROOT / route['extra_json']).read_text())
    npz_path = next(ROOT / item['path'] for item in route['artifacts'] if item['role'] == 'result_npz')
    data = np.load(npz_path, allow_pickle=True)
    if len(data['source_index']) != 760 or len(np.unique(data['source_index'])) != 760:
        raise RuntimeError('V9-A6.0 opportunity row mismatch')
    if np.max(np.abs(data['span'] - (data['rank16_score'] - data['target_score']))) > 1e-12:
        raise RuntimeError('V9-A6.0 span formula mismatch')
    if np.max(np.abs(2 * data['signed_epsilon'] - data['span'])) > 1e-12:
        raise RuntimeError('V9-A6.0 signed epsilon formula mismatch')
    if np.max(np.abs(np.sum(data['weighted_component_delta'], axis=1) + data['span'])) > 1e-5:
        raise RuntimeError('V9-A6.0 component decomposition mismatch')
    expected_conversion = data['span'][:, None] <= data['span_budgets'][None, :]
    if not np.array_equal(expected_conversion, data['converted_by_budget']):
        raise RuntimeError('V9-A6.0 budget conversion mismatch')
    primary_count = int(np.sum(expected_conversion[:, 2]))
    wide_count = int(np.sum(expected_conversion[:, 4]))
    if primary_count != result['budgets']['2.0']['promoted_rows'] or wide_count != result['budgets']['8.0']['promoted_rows']:
        raise RuntimeError('V9-A6.0 promoted counts mismatch')
    if result['gates']['pass_all']:
        raise RuntimeError('V9-A6.0 unexpectedly passes')
    if boundary['sampling_boundary']['canonical_invisible_rows'] != 0 or boundary['opportunity_visibility']['invisible_rows'] != 0:
        raise RuntimeError('V9-A6.0 visible-only boundary mismatch')
    return {
        'opportunities': len(data['source_index']),
        'primary_promoted': primary_count,
        'primary_conversion': float(primary_count / len(data['source_index'])),
        'wide_promoted': wide_count,
        'wide_conversion': float(wide_count / len(data['source_index'])),
        'all_opportunities_visible': True,
        'gate_pass': False,
    }


def verify_routes(index: dict[str, Any]) -> dict[str, Any]:
    routes = route_map(index)
    checks = {
        'V9-A4.5': verify_v9a45(routes['V9-A4.5']),
        'V9-A5.0': verify_v9a50(routes['V9-A5.0']),
        'V9-A5C.0': verify_v9a5c0(routes['V9-A5C.0']),
        'V9-A5.1a': verify_v9a51a(routes['V9-A5.1a']),
        'V9-A5.1b': verify_v9a51b(routes['V9-A5.1b']),
        'V9-A5.1c': verify_v9a51c(routes['V9-A5.1c']),
        'V9-A5.2': verify_v9a52(routes['V9-A5.2']),
        'V9-A6.0': verify_v9a60(routes['V9-A6.0']),
    }
    return checks


def write_doc(result: dict[str, Any]) -> None:
    lines = [
        '# V9-A Evidence Chain Verification',
        '',
        'Date: 2026-07-11',
        '',
        f"Overall status: **{'PASS' if result['pass_all'] else 'FAIL'}**",
        '',
        '## Index and artifact checks',
        '',
        '```json',
        json.dumps(result['index_and_artifacts'], indent=2),
        '```',
        '',
        '## Independent route checks',
        '',
        '```json',
        json.dumps(result['route_checks'], indent=2),
        '```',
        '',
        '## Boundary',
        '',
        'This verification checks committed artifacts and independently recomputes key saved-array formulas. It does not rerun TrackOn2 GPU inference or reproduce external datasets/checkpoints.',
    ]
    OUT_DOC.write_text('\n'.join(lines) + '\n')


def main() -> None:
    if not INDEX_JSON.exists():
        raise RuntimeError(
            f'missing evidence index: {INDEX_JSON}. Run scripts/build_v9a_evidence_index.py first.'
        )
    index = json.loads(INDEX_JSON.read_text())
    index_checks = verify_index_and_artifacts(index)
    route_checks = verify_routes(index)
    result = {
        'date': '2026-07-11',
        'type': 'CPU-side committed-artifact verification',
        'pass_all': True,
        'index_sha256': sha256(INDEX_JSON),
        'index_and_artifacts': index_checks,
        'route_checks': route_checks,
        'limitations': {
            'trackon_gpu_rerun': False,
            'external_assets_rehashed': False,
            'checks_commit_ancestry': True,
            'checks_artifact_sha256': True,
            'checks_key_npz_formulas': True,
        },
        'provenance': {
            'head': git('rev-parse', 'HEAD'),
            'branch': git('branch', '--show-current'),
            'script_sha256': sha256(Path(__file__).resolve()),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2) + '\n')
    write_doc(result)
    print(
        json.dumps(
            {
                'ok': True,
                'pass_all': True,
                'routes_verified': len(route_checks),
                'artifacts_checked': index_checks['artifacts_checked'],
                'json': str(OUT_JSON),
                'doc': str(OUT_DOC),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
