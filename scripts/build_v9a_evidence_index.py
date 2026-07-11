#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / 'outputs/paper_discovery_2026-07-05/v9a_evidence'
OUT_JSON = OUTDIR / 'v9a_evidence_index.json'
OUT_INDEX_DOC = ROOT / 'docs/V9A_EVIDENCE_INDEX_2026-07-11.md'
OUT_MATRIX_DOC = ROOT / 'docs/V9A_ROUTE_CLOSURE_MATRIX_2026-07-11.md'
OUT_REPRO_DOC = ROOT / 'docs/REPRODUCE_V9A_EVIDENCE.md'

SCHEMA_VERSION = 'v9a-evidence-index-v1'


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


def load_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text())


def artifact(path_text: str, role: str) -> dict[str, Any]:
    path = ROOT / path_text
    if not path.exists():
        raise RuntimeError(f'missing evidence artifact: {path_text}')
    return {
        'path': path_text,
        'role': role,
        'size_bytes': int(path.stat().st_size),
        'sha256': sha256(path),
    }


def route_specs() -> list[dict[str, Any]]:
    return [
        {
            'id': 'V9-A4.5',
            'title': 'Conservative residual reranking',
            'commit': '8fe78db',
            'question': 'Can a bounded residual reranker improve the frozen candidate ordering without damaging safe rows?',
            'dataset': 'PointOdyssey synthetic holdout; ani held out after sequence split',
            'fair_baseline': 'Frozen teacher C1 ranking on the same rows/candidates',
            'primary_gate': 'Mean improvement, better>worse, safe16 non-decrease, retention, and oracle-regret improvement',
            'raw_result': 'Heldout student ranking is worse than the frozen teacher.',
            'final_interpretation': 'Fixed-topK residual reranking adaptation is closed.',
            'status': 'closed',
            'result_json': 'outputs/paper_discovery_2026-07-05/v9a45_conservative_residual/v9a45_smoke_holdout_ani_e1_seed20260710.json',
            'artifacts': [
                ('scripts/v9a45_conservative_residual_end_to_end.py', 'execution_script'),
                ('outputs/paper_discovery_2026-07-05/v9a45_conservative_residual/v9a45_smoke_holdout_ani_e1_seed20260710.json', 'result_json'),
                ('docs/v9a45_conservative_residual_end_to_end_design_2026-07-10.md', 'design'),
                ('docs/v9a45_route_closure_and_v9a5_next_step_2026-07-10.md', 'review'),
            ],
        },
        {
            'id': 'V9-A5.0',
            'title': 'Sampled-causal temporal-state feasibility',
            'commit': '7c24367',
            'question': 'Does causal prior state contain useful temporal information, and is self-state error propagation the blocker?',
            'dataset': 'PointOdyssey canonical 4,878-row visible candidate pool',
            'fair_baseline': 'Frozen teacher/current-frame selection',
            'primary_gate': 'Sampled self-state must improve robustly; oracle/past-GT states are diagnostic headroom only',
            'raw_result': 'Sampled self-state fails; causal past-oracle/past-GT state has large sequence-consistent headroom.',
            'final_interpretation': 'Temporal information exists, but single-state error propagation blocks deployable use.',
            'status': 'diagnostic_headroom_only',
            'result_json': 'outputs/paper_discovery_2026-07-05/v9a50_temporal_feasibility/v9a50_temporal_feasibility.json',
            'artifacts': [
                ('scripts/v9a50_temporal_multihypothesis_feasibility.py', 'execution_script'),
                ('outputs/paper_discovery_2026-07-05/v9a50_temporal_feasibility/v9a50_temporal_feasibility.json', 'result_json'),
                ('outputs/paper_discovery_2026-07-05/v9a50_temporal_feasibility/v9a50_temporal_feasibility_selections.npz', 'result_npz'),
                ('docs/v9a50_temporal_multihypothesis_feasibility_design_2026-07-10.md', 'design'),
                ('docs/v9a50_temporal_multihypothesis_feasibility_result_2026-07-10.md', 'result_doc'),
            ],
        },
        {
            'id': 'V9-A5C.0',
            'title': 'Fused correlation candidate-recall oracle',
            'commit': '448c095',
            'question': 'Is useful <=4px candidate recall materially higher at fused K64 than at fused K16?',
            'dataset': 'PointOdyssey canonical 4,878-row visible candidate pool',
            'fair_baseline': 'Fused C1 top16 on the same correlation map',
            'primary_gate': 'Every sequence must have fused K64-K16 recall@4 headroom >=0.02',
            'raw_result': 'All three sequences pass; hard-row headroom is large.',
            'final_interpretation': 'Synthetic upstream candidate availability exists, but does not establish deployable selection.',
            'status': 'oracle_headroom_pass',
            'result_json': 'outputs/paper_discovery_2026-07-05/v9a5c0_candidate_recall/v9a5c0_pointodyssey_candidate_recall.json',
            'artifacts': [
                ('scripts/v9a5c0_candidate_recall_correlation_oracle.py', 'execution_script'),
                ('outputs/paper_discovery_2026-07-05/v9a5c0_candidate_recall/v9a5c0_pointodyssey_candidate_recall.json', 'result_json'),
                ('outputs/paper_discovery_2026-07-05/v9a5c0_candidate_recall/v9a5c0_pointodyssey_candidate_recall.npz', 'result_npz'),
                ('docs/v9a5c0_candidate_recall_correlation_oracle_design_2026-07-10.md', 'design'),
                ('docs/v9a5c0_pointodyssey_candidate_recall_result_2026-07-10.md', 'result_doc'),
            ],
        },
        {
            'id': 'V9-A5.1a',
            'title': 'Fixed-K16 shared-state beam baseline',
            'commit': 'd8e5ca9',
            'question': 'Can fixed-K16 temporal path scoring preserve or improve current-frame candidate reachability?',
            'dataset': '9 PointOdyssey clips; 27,648 query-frame rows',
            'fair_baseline': 'Current-frame teacher top1/topB at equal capacity',
            'primary_gate': 'Deterministic top1 and same-capacity beam oracle must improve sequence/clip/reentry metrics',
            'raw_result': 'Both deterministic and same-capacity reachability fail.',
            'final_interpretation': 'The exact fixed-K16/raw-C1/shared-state beam configuration is closed.',
            'status': 'closed',
            'result_json': 'outputs/paper_discovery_2026-07-05/v9a51a_fixed_k16_shared_state_beam/v9a51a_fixed_k16_shared_state_beam.json',
            'artifacts': [
                ('scripts/v9a51a_fixed_k16_shared_state_beam.py', 'execution_script'),
                ('outputs/paper_discovery_2026-07-05/v9a51a_fixed_k16_shared_state_beam/v9a51a_fixed_k16_shared_state_beam.json', 'result_json'),
                ('outputs/paper_discovery_2026-07-05/v9a51a_fixed_k16_shared_state_beam/v9a51a_fixed_k16_shared_state_beam.npz', 'result_npz'),
                ('docs/v9a51a_fixed_k16_baseline_freeze_2026-07-10.md', 'review'),
            ],
        },
        {
            'id': 'V9-A5.1b',
            'title': 'Candidate-conditioned downstream refinement',
            'commit': '29512f0',
            'question': 'Can a frozen fused C1 candidate drive a useful singleton-conditioned C2/head refined coordinate?',
            'dataset': '9 PointOdyssey clips; 27,648 query-frame rows',
            'fair_baseline': 'Official final TrackOn2 output with risk-gated fallback',
            'primary_gate': 'Risk-gated refined candidate oracle must improve all sequences, clip CI, and reentry summaries',
            'raw_result': 'GT-only refined candidate oracle passes strongly; frozen score-top1 remains unreliable.',
            'final_interpretation': 'Candidate refinement has oracle headroom only; it authorizes a temporal candidate-set audit, not deployment.',
            'status': 'oracle_headroom_pass',
            'result_json': 'outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_candidate_conditioned_refinement.json',
            'artifacts': [
                ('scripts/v9a51b_candidate_conditioned_refinement_audit.py', 'execution_script'),
                ('outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_candidate_conditioned_refinement.json', 'result_json'),
                ('outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_candidate_conditioned_refinement_rows.npz', 'result_npz'),
                ('docs/v9a51b_candidate_conditioned_refinement_review_2026-07-10.md', 'review'),
            ],
        },
        {
            'id': 'V9-A5.1c',
            'title': 'History-preserving shared-state beam',
            'commit': '53f475a',
            'question': 'Does finite-window history-preserving beam pruning add value beyond a same-frame same-capacity candidate set?',
            'dataset': '9 PointOdyssey clips; 27,648 query-frame rows',
            'fair_baseline': 'Same-frame frozen-score topB candidate oracle at matched B/K',
            'primary_gate': 'Deterministic top1 and capacity-matched temporal oracle must improve across sequence/clip/reentry',
            'raw_result': 'System-level min(official,beam) oracle passes versus official; deterministic top1 fails.',
            'final_interpretation': 'Capacity-matched review reverses the temporal claim: primary B4 is worse than frame-local top4.',
            'status': 'closed_after_fair_control',
            'result_json': 'outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_history_preserving_beam.json',
            'extra_json': 'outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_same_capacity_incremental_audit.json',
            'artifacts': [
                ('scripts/v9a51c_history_preserving_beam_audit.py', 'execution_script'),
                ('scripts/v9a51c_same_capacity_incremental_audit.py', 'fair_control_script'),
                ('outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_history_preserving_beam.json', 'result_json'),
                ('outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_history_preserving_beam_rows.npz', 'result_npz'),
                ('outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_same_capacity_incremental_audit.json', 'fair_control_json'),
                ('docs/v9a51c_comprehensive_review_and_v9a52_next_step_2026-07-10.md', 'review'),
            ],
        },
        {
            'id': 'V9-A5.2',
            'title': 'Independent full-model-state branching',
            'commit': 'bb87918',
            'question': 'Does a complete independent 432-query TrackOn2 state produce useful future outputs beyond a shared-state capacity-2 control?',
            'dataset': '72 frozen first-risk events across 9 PointOdyssey clips; horizons 1/4/8',
            'fair_baseline': 'Shared-state capacity-2 oracle: official final plus current-state score-top1 refined output',
            'primary_gate': 'Horizon-8 sequence/clip/CI/safe16 plus pooled, early8, and mechanistic novelty gates',
            'raw_result': 'State and C1 sets diverge, but horizon-8, pooled, early8, and useful-novelty gates fail.',
            'final_interpretation': 'The tested independent score-top1 singleton state branch is closed.',
            'status': 'closed',
            'result_json': 'outputs/paper_discovery_2026-07-05/v9a52_independent_state/v9a52_independent_state_branching.json',
            'extra_json': 'outputs/paper_discovery_2026-07-05/v9a52_independent_state/v9a52_candidate_reachability_postaudit.json',
            'artifacts': [
                ('scripts/v9a52_independent_state_branching_audit.py', 'execution_script'),
                ('scripts/v9a52_candidate_reachability_postaudit.py', 'postaudit_script'),
                ('outputs/paper_discovery_2026-07-05/v9a52_independent_state/v9a52_independent_state_branching.json', 'result_json'),
                ('outputs/paper_discovery_2026-07-05/v9a52_independent_state/v9a52_independent_state_branching_rows.npz', 'result_npz'),
                ('outputs/paper_discovery_2026-07-05/v9a52_independent_state/v9a52_candidate_reachability_postaudit.json', 'postaudit_json'),
                ('docs/v9a52_comprehensive_review_and_route_closure_2026-07-11.md', 'review'),
            ],
        },
        {
            'id': 'V9-A6.0',
            'title': 'Bounded fused-correlation rank shift',
            'commit': 'a1c697d',
            'question': 'Are K16-miss/K64-hit candidates close enough to the K16 score boundary for a small identity-preserving residual?',
            'dataset': '760 visible K16-miss/K64-hit opportunities from the 4,878-row canonical pool',
            'fair_baseline': 'Original fused top16 boundary; GT-directed target boost is an optimistic magnitude lower bound',
            'primary_gate': 'At span<=2*g_ref: >=50% global conversion, >=40% each sequence, clip/CI/gain and normalized-magnitude gates',
            'raw_result': 'Only 67/760 opportunities convert at 2*g_ref; every gate fails.',
            'final_interpretation': 'Small bounded pre-topK correlation residual promotion is closed; the evaluated pool is visible-only.',
            'status': 'closed',
            'result_json': 'outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_bounded_rank_shift_audit.json',
            'extra_json': 'outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_visibility_stratified_postaudit.json',
            'artifacts': [
                ('scripts/v9a60_export_fused_top129.py', 'export_script'),
                ('scripts/v9a60_bounded_rank_shift_audit.py', 'execution_script'),
                ('scripts/v9a60_visibility_stratified_postaudit.py', 'boundary_postaudit_script'),
                ('outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_fused_top129_export.json', 'export_json'),
                ('outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_fused_top129_export.npz', 'export_npz'),
                ('outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_bounded_rank_shift_audit.json', 'result_json'),
                ('outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_bounded_rank_shift_audit_rows.npz', 'result_npz'),
                ('outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_visibility_stratified_postaudit.json', 'boundary_postaudit_json'),
                ('docs/v9a60_comprehensive_review_and_algorithmic_route_closure_2026-07-11.md', 'review'),
            ],
        },
    ]


def extract_metrics(route: dict[str, Any]) -> dict[str, Any]:
    route_id = route['id']
    result = load_json(route['result_json'])
    if route_id == 'V9-A4.5':
        teacher = result['validation']['teacher']
        student = result['validation']['student']
        comparison = result['validation']['comparison']
        return {
            'teacher_mean_error': teacher['mean_error'],
            'student_mean_error': student['mean_error'],
            'student_minus_teacher': comparison['mean_difference_student_minus_teacher'],
            'better_worse_equal': [comparison['better'], comparison['worse'], comparison['equal']],
            'teacher_safe16': teacher['safe16'],
            'student_safe16': student['safe16'],
            'gate_pass': result['validation']['gate']['pass_all'],
            'decision': result['decision'],
        }
    if route_id == 'V9-A5.0':
        sampled_name = result['gates']['best_sampled_causal_by_global_mean']
        oracle_name = result['gates']['best_oracle_state_by_global_mean']
        return {
            'best_sampled_policy': sampled_name,
            'sampled_mean_error': result['metrics'][sampled_name]['all']['mean_error'],
            'sampled_difference_vs_teacher': result['metrics'][sampled_name]['all']['mean_difference_vs_teacher'],
            'best_oracle_policy': oracle_name,
            'oracle_mean_error': result['metrics'][oracle_name]['all']['mean_error'],
            'oracle_difference_vs_teacher': result['metrics'][oracle_name]['all']['mean_difference_vs_teacher'],
            'sampled_gate_pass': result['gates']['passing_sampled_causal'],
            'oracle_gate_pass': result['gates']['passing_oracle_state'],
            'decision': result['gates']['decision'],
        }
    if route_id == 'V9-A5C.0':
        return {
            'per_sequence': result['gate']['per_sequence'],
            'gate_pass': result['gate']['pass_all_sequences'],
            'hard_fused_k16_recall4': result['summaries']['hard']['maps']['fused']['budgets'][3]['recall']['4'],
            'hard_fused_k64_recall4': result['summaries']['hard']['maps']['fused']['budgets'][5]['recall']['4'],
            'decision': result['gate']['decision'],
        }
    if route_id == 'V9-A5.1a':
        return {
            'teacher_top1_mean': result['metrics']['teacher_top1']['all_visible']['mean_error'],
            'beam4_top1_mean': result['metrics']['beam4_motion_latent_top1']['all_visible']['mean_error'],
            'teacher_top4_oracle_mean': result['metrics']['teacher_top4_oracle']['all_visible']['mean_error'],
            'beam4_oracle_mean': result['metrics']['beam4_motion_latent_oracle']['all_visible']['mean_error'],
            'deterministic_pass': result['gates']['deterministic_top1']['pass_all'],
            'reachability_pass': result['gates']['beam_reachability']['pass_all'],
            'decision': result['gates']['decision'],
        }
    if route_id == 'V9-A5.1b':
        official = result['summaries']['official_final']['all_visible']
        oracle = result['summaries']['hybrid_refined_oracle']['all_visible']
        return {
            'official_mean_error': official['mean_error'],
            'hybrid_oracle_mean_error': oracle['mean_error'],
            'oracle_difference_vs_official': oracle['mean_difference_vs_official'],
            'better_worse_equal': [oracle['better'], oracle['worse'], oracle['equal']],
            'gate_pass': result['gates']['pass_all'],
            'decision': result['gates']['decision'],
        }
    if route_id == 'V9-A5.1c':
        fair = load_json(route['extra_json'])
        official = result['summaries']['official_final']['all_visible']
        top1 = result['summaries']['beam_top1_native_dynamic_B4']['all_visible']
        oracle = result['summaries']['beam_oracle_native_dynamic_B4']['all_visible']
        primary_fair = fair['policy_overview']['native_dynamic_B4']
        return {
            'official_mean_error': official['mean_error'],
            'temporal_top1_mean_error': top1['mean_error'],
            'system_oracle_mean_error': oracle['mean_error'],
            'raw_deterministic_pass': result['gates']['deterministic_top1']['pass_all'],
            'raw_oracle_pass': result['gates']['beam_reachability']['pass_all'],
            'fair_top1_difference': primary_fair['top1_mean_difference'],
            'fair_top1_ci': primary_fair['top1_ci'],
            'fair_oracle_difference': primary_fair['oracle_mean_difference'],
            'fair_oracle_ci': primary_fair['oracle_ci'],
            'final_incremental_pass': fair['primary_incremental_top1_pass'] or fair['primary_incremental_oracle_pass'],
            'decision': fair['decision'],
        }
    if route_id == 'V9-A5.2':
        h8 = result['summaries']['horizon8_all_visible']
        pooled = result['summaries']['pooled_all_visible']
        return {
            'horizon8_independent_mean': h8['candidate_mean'],
            'horizon8_shared_mean': h8['baseline_mean'],
            'horizon8_difference': h8['mean_difference'],
            'horizon8_better_worse_equal': [h8['better'], h8['worse'], h8['equal']],
            'pooled_difference': pooled['mean_difference'],
            'q_new_divergence_fraction': result['mechanistic']['target_q_new_l2_gt_1e4_fraction_visible'],
            'top16_changed_fraction': result['mechanistic']['top16_set_changed_fraction_visible'],
            'useful_novel_rows': result['mechanistic']['useful_novel_rows_visible'],
            'gate_pass': result['gates']['pass_all'],
            'decision': result['gates']['decision'],
        }
    if route_id == 'V9-A6.0':
        visibility = load_json(route['extra_json'])
        primary = result['budgets']['2.0']
        wide = result['budgets']['8.0']
        return {
            'opportunities': primary['opportunity_rows'],
            'primary_promoted': primary['promoted_rows'],
            'primary_conversion': primary['opportunity_conversion'],
            'primary_global_gain': primary['implied_global_recall4_gain'],
            'primary_hard_gain': primary['implied_gt_hard_recall4_gain'],
            'primary_conversion_ci': result['primary_bootstrap']['conversion_95_ci'],
            'wide_conversion': wide['opportunity_conversion'],
            'span_over_std_median': result['magnitude']['span_over_fused_std']['median'],
            'span_over_std_p90': result['magnitude']['span_over_fused_std']['p90'],
            'canonical_visible_rows': visibility['sampling_boundary']['canonical_visible_rows'],
            'canonical_invisible_rows': visibility['sampling_boundary']['canonical_invisible_rows'],
            'gate_pass': result['gates']['pass_all'],
            'decision': result['gates']['decision'],
            'boundary_decision': visibility['decision'],
        }
    raise RuntimeError(f'unsupported route: {route_id}')


def build_index() -> dict[str, Any]:
    head = git('rev-parse', 'HEAD')
    branch = git('branch', '--show-current')
    tracked = git('status', '--porcelain', '--untracked-files=no')
    if tracked:
        raise RuntimeError(f'tracked worktree dirty: {tracked}')
    routes = []
    for spec in route_specs():
        subprocess.run(
            ['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', spec['commit'], head],
            check=True,
        )
        row = {key: value for key, value in spec.items() if key != 'artifacts'}
        row['commit_full'] = git('rev-parse', spec['commit'])
        row['metrics'] = extract_metrics(spec)
        row['artifacts'] = [artifact(path, role) for path, role in spec['artifacts']]
        routes.append(row)
    return {
        'schema_version': SCHEMA_VERSION,
        'created_at': '2026-07-11',
        'generated_from': {
            'head': head,
            'branch': branch,
            'tracked_status': tracked,
            'builder_script': {
                'path': str(Path(__file__).resolve().relative_to(ROOT)),
                'sha256': sha256(Path(__file__).resolve()),
            },
        },
        'project_decision': {
            'route_a': 'diagnostic paper active/stable',
            'route_b': 'current SOTA-compatible rescue hypothesis tree exhausted',
            'route_c': 'old student/pseudo-label/DINO-local routes closed',
            'next_work': 'evidence consolidation, reproducibility, manuscript tables/figures, discussion and limitations',
        },
        'routes': routes,
    }


def write_index_doc(index: dict[str, Any]) -> None:
    lines = [
        '# V9-A Evidence Index',
        '',
        'Date: 2026-07-11',
        '',
        f"Generated from `{index['generated_from']['branch']}` at `{index['generated_from']['head']}`.",
        '',
        'This index separates the original preregistered result from the final fair interpretation. Artifact hashes are machine-readable in the accompanying JSON.',
        '',
    ]
    for route in index['routes']:
        metrics = json.dumps(route['metrics'], indent=2)
        lines.extend(
            [
                f"## {route['id']} — {route['title']}",
                '',
                f"- Commit: `{route['commit_full']}`",
                f"- Status: `{route['status']}`",
                f"- Question: {route['question']}",
                f"- Data: {route['dataset']}",
                f"- Fair baseline: {route['fair_baseline']}",
                f"- Primary gate: {route['primary_gate']}",
                f"- Original result: {route['raw_result']}",
                f"- Final interpretation: {route['final_interpretation']}",
                '',
                '```json',
                metrics,
                '```',
                '',
                'Artifacts:',
                '',
            ]
        )
        for item in route['artifacts']:
            lines.append(
                f"- `{item['path']}` — {item['role']} — `{item['sha256']}`"
            )
        lines.append('')
    OUT_INDEX_DOC.write_text('\n'.join(lines) + '\n')


def write_matrix_doc(index: dict[str, Any]) -> None:
    lines = [
        '# V9-A Route Closure Matrix',
        '',
        'Date: 2026-07-11',
        '',
        '| Route | Primary question | Data | Fair baseline | Gate/result | Final decision |',
        '|---|---|---|---|---|---|',
    ]
    for route in index['routes']:
        gate_result = route['raw_result'].replace('|', '/')
        decision = route['final_interpretation'].replace('|', '/')
        lines.append(
            f"| {route['id']} | {route['question']} | {route['dataset']} | "
            f"{route['fair_baseline']} | {gate_result} | **{decision}** |"
        )
    lines.extend(
        [
            '',
            '## Project conclusion',
            '',
            'Substantial candidate-level oracle headroom exists beyond fused top16, but the tested fixed-pool reranking, shared-state temporal, independent-state temporal, and small bounded pre-topK score-shift routes do not convert it into sequence-consistent deployable gains under the committed synthetic gates.',
            '',
            'This matrix is a project evidence closure, not a universal impossibility theorem for all tracking architectures.',
        ]
    )
    OUT_MATRIX_DOC.write_text('\n'.join(lines) + '\n')


def write_reproduce_doc(index: dict[str, Any]) -> None:
    lines = [
        '# Reproduce and Verify the V9-A Evidence Chain',
        '',
        '## Artifact verification from the final checkout',
        '',
        '```bash',
        'python scripts/verify_v9a_evidence_chain.py',
        '```',
        '',
        'This CPU-side verification checks commit ancestry, SHA256 hashes, JSON parsing, NPZ shapes, key formulas, gate decisions, and evidence-index consistency. It does not rerun TrackOn2 GPU inference.',
        '',
        '## Full GPU reproduction versus artifact verification',
        '',
        'The committed evidence can be verified from the final checkout. Full GPU recomputation additionally requires the immutable TrackOn2 checkpoint, DINOv3 weights, PointOdyssey annotations/frames, the recorded historical execution state, and the original absolute-path layout or an environment wrapper that maps equivalent roots.',
        '',
        'Several formal scripts were executed from a historical HEAD before the execution script/result commit was created. The result JSON stores the execution HEAD and script SHA256. Therefore exact GPU reproduction should reconstruct the recorded historical code tree and inject the hash-matched execution script, rather than assuming the final checkout can rerun every script unchanged.',
        '',
        '## Frozen versus generated artifacts',
        '',
        '- Frozen scientific artifacts: route result JSON/NPZ, execution scripts, design/review documents.',
        '- Generated evidence artifacts: evidence index, route matrix, verification report.',
        '- Do not modify frozen scripts to replace absolute paths; use an external wrapper or matching mount layout.',
        '',
        '## Paper-use boundary',
        '',
        '- V9-A5C.0/V9-A6.0 are visible-frame candidate audits by construction.',
        '- GT-only oracle/union results must be labeled as upper bounds.',
        '- V9-A5.1c final interpretation must use the same-capacity comparator, not the raw min(official,beam) oracle pass.',
        '- No official leaderboard or universal-improvement claim is authorized.',
        '',
        f"Evidence index generated from `{index['generated_from']['head']}`.",
    ]
    OUT_REPRO_DOC.write_text('\n'.join(lines) + '\n')


def main() -> None:
    index = build_index()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(index, indent=2) + '\n')
    write_index_doc(index)
    write_matrix_doc(index)
    write_reproduce_doc(index)
    print(
        json.dumps(
            {
                'ok': True,
                'routes': len(index['routes']),
                'json': str(OUT_JSON),
                'index_doc': str(OUT_INDEX_DOC),
                'matrix_doc': str(OUT_MATRIX_DOC),
                'reproduce_doc': str(OUT_REPRO_DOC),
                'generated_from': index['generated_from'],
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
