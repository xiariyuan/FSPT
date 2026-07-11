#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

WORKTREE_ROOT = Path('/gemini/code/FSPT_v9a52_clean')
SOURCE_ROOT = Path('/gemini/code/FSPT')
HELPER_SCRIPT = WORKTREE_ROOT / 'scripts/v9a52_independent_state_branching_smoke.py'
INPUT_MANIFEST = WORKTREE_ROOT / 'docs/v9a52_input_manifest_2026-07-11.json'
EVENT_MANIFEST = WORKTREE_ROOT / 'docs/v9a52_independent_state_event_manifest_2026-07-11.json'
V9B_NPZ = WORKTREE_ROOT / 'outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_candidate_conditioned_refinement_rows.npz'
V9C_NPZ = WORKTREE_ROOT / 'outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_history_preserving_beam_rows.npz'
POD_ROOT = SOURCE_ROOT / 'datasets/pointodyssey/train'
CHECKPOINT = SOURCE_ROOT / 'baselines/track_on/checkpoints_trackon2_dinov3.pt'
CONFIG = WORKTREE_ROOT / 'baselines/track_on/config/test.yaml'
OUTDIR = WORKTREE_ROOT / 'outputs/paper_discovery_2026-07-05/v9a52_independent_state'
DEFAULT_JSON = OUTDIR / 'v9a52_independent_state_branching.json'
DEFAULT_NPZ = OUTDIR / 'v9a52_independent_state_branching_rows.npz'
DEFAULT_DOC = WORKTREE_ROOT / 'docs/v9a52_independent_state_branching_result_2026-07-11.md'

EXPECTED_HEAD = 'ee17e03176c9e0b256cf555feab82b0f9be1f041'
EXPECTED_BRANCH = 'v9a52-independent-state-branching-20260711'
EXPECTED_INPUT_MANIFEST_SHA256 = '61e6cfe418032b7d3466458c1aa7c1e81b59f8bb0c9ab44e5ff286a4ca0086be'
EXPECTED_HELPER_SHA256 = '1e0a0324bf0170bf7f9db1d91de51075bcd690410f4014b534c834628da3ec37'
EXPECTED_EVENT_MANIFEST_SHA256 = 'e5f5fa34c840e5159726ada1c7752e8bfa273b48ad6cd215e7519ed71721fb4b'

SEQUENCES = ('ani', 'animal3', 'r4_new_f')
CLIP_STARTS = (0, 256, 512)
LENGTH = 96
NQ = 32
SUPPORT_GRID_SIZE = 20
EXPECTED_N = 432
EXPECTED_M = 24
EXPECTED_D = 256
HORIZONS = (1, 4, 8)
K64 = 64
TOL = 1e-6
CANDIDATE_TOL = 1e-4
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 20260716


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


if sha256(HELPER_SCRIPT) != EXPECTED_HELPER_SHA256:
    raise RuntimeError('frozen smoke helper hash mismatch before import')
_SPEC = importlib.util.spec_from_file_location('v9a52_smoke_frozen', HELPER_SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError('unable to import frozen V9-A5.2 smoke helper')
H = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = H
_SPEC.loader.exec_module(H)


@dataclass
class ActiveEvent:
    event_id: int
    clip_id: str
    sequence: str
    clip_start: int
    query_idx: int
    event_frame: int
    selection_sha256: str
    branch_a: Any
    branch_b: Any


def verify_inputs(verify_all_frames: bool, selected_clips: list[str]) -> dict[str, Any]:
    if sha256(INPUT_MANIFEST) != EXPECTED_INPUT_MANIFEST_SHA256:
        raise RuntimeError('formal input manifest hash mismatch')
    if sha256(EVENT_MANIFEST) != EXPECTED_EVENT_MANIFEST_SHA256:
        raise RuntimeError('event manifest hash mismatch')
    manifest = json.loads(INPUT_MANIFEST.read_text())
    checked = []
    for row in manifest['files']:
        path = Path(row['path'])
        if not path.exists():
            raise RuntimeError(f'missing input: {path}')
        actual = sha256(path)
        if path.stat().st_size != int(row['size_bytes']) or actual != row['sha256']:
            raise RuntimeError(f'input hash/size mismatch: {path}')
        checked.append(
            {
                'path': str(path),
                'size_bytes': int(path.stat().st_size),
                'sha256': actual,
            }
        )

    head = subprocess.check_output(
        ['git', '-C', str(WORKTREE_ROOT), 'rev-parse', 'HEAD'],
        text=True,
    ).strip()
    branch = subprocess.check_output(
        ['git', '-C', str(WORKTREE_ROOT), 'branch', '--show-current'],
        text=True,
    ).strip()
    tracked = subprocess.check_output(
        [
            'git',
            '-C',
            str(WORKTREE_ROOT),
            'status',
            '--porcelain',
            '--untracked-files=no',
        ],
        text=True,
    ).strip()
    if head != EXPECTED_HEAD or head != manifest['clean_head']:
        raise RuntimeError(f'HEAD mismatch: {head}')
    if branch != EXPECTED_BRANCH or branch != manifest['clean_branch']:
        raise RuntimeError(f'branch mismatch: {branch}')
    if tracked:
        raise RuntimeError(f'tracked worktree is dirty: {tracked}')

    frame_meta = manifest['frame_manifest']
    frame_manifest = json.loads(Path(frame_meta['path']).read_text())
    root = Path(frame_manifest['root'])
    selected_set = set(selected_clips)
    rows = (
        frame_manifest['files']
        if verify_all_frames
        else [
            row
            for row in frame_manifest['files']
            if ':'.join(
                [
                    row['relative_path'].split('/')[0],
                    str(
                        (
                            int(Path(row['relative_path']).stem.split('_')[-1])
                            // 256
                        )
                        * 256
                    ),
                ]
            )
            in selected_set
        ]
    )
    if not rows:
        raise RuntimeError('no frame rows selected for verification')
    verified_frames = []
    aggregate = hashlib.sha256()
    total = 0
    for row in rows:
        relative = str(row['relative_path'])
        path = root / relative
        actual = sha256(path)
        size = path.stat().st_size
        if size != int(row['size_bytes']) or actual != row['sha256']:
            raise RuntimeError(f'frame hash/size mismatch: {path}')
        total += size
        verified_frames.append(relative)
        if verify_all_frames:
            aggregate.update(relative.encode())
            aggregate.update(b'\0')
            aggregate.update(str(size).encode())
            aggregate.update(b'\0')
            aggregate.update(actual.encode())
            aggregate.update(b'\n')
    if verify_all_frames:
        if len(rows) != int(frame_meta['file_count']):
            raise RuntimeError('full frame count mismatch')
        if total != int(frame_meta['total_bytes']):
            raise RuntimeError('full frame byte count mismatch')
        if aggregate.hexdigest() != frame_meta['aggregate_sha256']:
            raise RuntimeError('full frame aggregate mismatch')

    imported = {}
    for name in (
        'model.trackon_predictor',
        'model.trackon',
        'model.reranking',
        'model.prediction_head',
        'model.modules',
        'utils.coord_utils',
        'utils.train_utils',
    ):
        module = sys.modules.get(name)
        value = '' if module is None else str(Path(module.__file__).resolve())
        imported[name] = value
        if not value.startswith(str(H.TRACKON_ROOT.resolve()) + '/'):
            raise RuntimeError(f'non-clean TrackOn import: {name} -> {value}')

    branch_functions = (
        create_active_event,
        forward_active_event,
    )
    forbidden_hits = {}
    for function in branch_functions:
        source = inspect.getsource(function).lower()
        forbidden_hits[function.__name__] = [
            token
            for token in ('gt_', 'ground_truth', 'error_px', 'oracle_index')
            if token in source
        ]
    if any(forbidden_hits.values()):
        raise RuntimeError(f'GT/error token in formal branch logic: {forbidden_hits}')

    return {
        'pass': True,
        'head': head,
        'branch': branch,
        'tracked_status': tracked,
        'checked_inputs': checked,
        'verified_frame_count': int(len(rows)),
        'verified_frame_total_bytes': int(total),
        'full_frame_hash': bool(verify_all_frames),
        'imported_trackon_code_paths': imported,
        'formal_branch_forbidden_hits': forbidden_hits,
        'helper_script': {
            'path': str(HELPER_SCRIPT),
            'sha256': sha256(HELPER_SCRIPT),
        },
        'script': {
            'path': str(Path(__file__).resolve()),
            'sha256': sha256(Path(__file__).resolve()),
        },
    }


@torch.no_grad()
def create_active_event(
    model,
    event: dict[str, Any],
    official_state: Any,
    official_q_new: torch.Tensor,
    candidate_bundle: Any,
) -> tuple[ActiveEvent, dict[str, Any]]:
    target = int(event['query_idx'])
    bundle = candidate_bundle
    score_index = int(torch.argmax(bundle.score[0]).item())
    alternative_q2 = bundle.q2[0, score_index]
    branch_a, branch_b = H.split_event_states(
        model,
        official_state,
        official_q_new,
        target,
        alternative_q2,
    )
    split = {
        'score_index': score_index,
        'grid_index': int(bundle.correlation_indices[0, score_index].detach().cpu()),
        'score': float(bundle.score[0, score_index].detach().cpu()),
        'alternative_q2_vs_official_q_new': H.vector_divergence(
            alternative_q2,
            official_q_new[target],
        ),
        'storage_disjoint_within_event': H.storage_is_disjoint(
            official_state,
            branch_a,
            branch_b,
        ),
    }
    a_before = float(branch_a.point_memory[target, -1, 0].detach().cpu())
    b_before = float(branch_b.point_memory[target, -1, 0].detach().cpu())
    branch_b.point_memory[target, -1, 0] += 1.0
    a_after = float(branch_a.point_memory[target, -1, 0].detach().cpu())
    branch_b.point_memory[target, -1, 0] = b_before
    b_after = float(branch_b.point_memory[target, -1, 0].detach().cpu())
    split['mutation_isolation'] = bool(a_before == a_after and b_before == b_after)
    active = ActiveEvent(
        event_id=int(event['event_id']),
        clip_id=str(event['clip_id']),
        sequence=str(event['sequence']),
        clip_start=int(event['clip_start']),
        query_idx=target,
        event_frame=int(event['frame_tau']),
        selection_sha256=str(event['selection_sha256']),
        branch_a=branch_a,
        branch_b=branch_b,
    )
    return active, split


@torch.no_grad()
def forward_active_event(
    model,
    active: ActiveEvent,
    frame_features: tuple[torch.Tensor, ...],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    diag_a = H.diagnostic_track_frame(
        model,
        active.branch_a,
        frame_features,
    )
    diag_b = H.diagnostic_track_frame(
        model,
        active.branch_b,
        frame_features,
    )
    return diag_a, diag_b


def summary(
    candidate: np.ndarray,
    baseline: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    candidate = np.asarray(candidate, dtype=np.float64)[mask]
    baseline = np.asarray(baseline, dtype=np.float64)[mask]
    if len(candidate) == 0:
        return {
            'n': 0,
            'candidate_mean': None,
            'baseline_mean': None,
            'mean_difference': None,
            'median_difference': None,
            'candidate_safe4': 0,
            'baseline_safe4': 0,
            'candidate_safe8': 0,
            'baseline_safe8': 0,
            'candidate_safe16': 0,
            'baseline_safe16': 0,
            'better': 0,
            'worse': 0,
            'equal': 0,
        }
    difference = candidate - baseline
    return {
        'n': int(len(candidate)),
        'candidate_mean': float(np.mean(candidate)),
        'baseline_mean': float(np.mean(baseline)),
        'mean_difference': float(np.mean(difference)),
        'median_difference': float(np.median(difference)),
        'candidate_safe4': int(np.sum(candidate <= 4.0)),
        'baseline_safe4': int(np.sum(baseline <= 4.0)),
        'candidate_safe8': int(np.sum(candidate <= 8.0)),
        'baseline_safe8': int(np.sum(baseline <= 8.0)),
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
    clips = sorted(set(clip_id[mask].tolist()))
    if not clips:
        return {
            'n_clips': 0,
            'per_clip_difference': {},
            'mean_difference_95_ci': [None, None],
            'probability_mean_lt_zero': None,
            'clip_better': 0,
            'clip_worse': 0,
            'clip_equal': 0,
            'leave_one_clip_out_mean_difference': {},
        }
    difference = candidate - baseline
    values = np.asarray(
        [np.mean(difference[mask & (clip_id == clip)]) for clip in clips],
        dtype=np.float64,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    chunk = 1000
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        count = min(chunk, BOOTSTRAP_RESAMPLES - start)
        indices = rng.integers(0, len(clips), size=(count, len(clips)))
        bootstrap[start : start + count] = np.mean(values[indices], axis=1)
    leave_one_out = {}
    if len(clips) > 1:
        for index, clip in enumerate(clips):
            leave_one_out[clip] = float(np.mean(np.delete(values, index)))
    return {
        'n_clips': int(len(clips)),
        'bootstrap_resamples': BOOTSTRAP_RESAMPLES,
        'bootstrap_seed': BOOTSTRAP_SEED,
        'per_clip_difference': {
            clip: float(value) for clip, value in zip(clips, values)
        },
        'mean_difference_95_ci': [
            float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
        ],
        'probability_mean_lt_zero': float(np.mean(bootstrap < 0.0)),
        'clip_better': int(np.sum(values < -TOL)),
        'clip_worse': int(np.sum(values > TOL)),
        'clip_equal': int(np.sum(np.abs(values) <= TOL)),
        'leave_one_clip_out_mean_difference': leave_one_out,
    }


def write_doc(result: dict[str, Any], path: Path) -> None:
    primary = result['summaries']['horizon8_all_visible']
    pooled = result['summaries']['pooled_all_visible']
    lines = [
        '# V9-A5.2 Independent Model-State Branching Result',
        '',
        'Date: 2026-07-11',
        '',
        'No training and no DAVIS data were used.',
        '',
        '## Primary horizon-8 comparison',
        '',
        '```json',
        json.dumps(primary, indent=2),
        '```',
        '',
        '## Pooled horizons comparison',
        '',
        '```json',
        json.dumps(pooled, indent=2),
        '```',
        '',
        '## Mechanistic diagnostics',
        '',
        '```json',
        json.dumps(result['mechanistic'], indent=2),
        '```',
        '',
        '## Candidate-reachability supplement (non-gating)',
        '',
        '```json',
        json.dumps(result['candidate_reachability_supplement'], indent=2),
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


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-clips', type=int, default=0)
    parser.add_argument('--max-events-per-clip', type=int, default=0)
    parser.add_argument('--skip-full-frame-hash', action='store_true')
    parser.add_argument('--out-json', type=Path, default=DEFAULT_JSON)
    parser.add_argument('--out-npz', type=Path, default=DEFAULT_NPZ)
    parser.add_argument('--out-doc', type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()

    start_all = time.perf_counter()
    event_manifest = json.loads(EVENT_MANIFEST.read_text())
    selected_events = []
    for event_id, row in enumerate(event_manifest['selected_events']):
        sequence, start_text = str(row['clip_id']).split(':')
        selected_events.append(
            {
                **row,
                'event_id': int(event_id),
                'sequence': sequence,
                'clip_start': int(start_text),
            }
        )
    all_clips = [f'{sequence}:{start}' for sequence in SEQUENCES for start in CLIP_STARTS]
    clips = all_clips[: args.max_clips] if args.max_clips > 0 else all_clips
    selected_events = [row for row in selected_events if row['clip_id'] in clips]
    if args.max_events_per_clip > 0:
        limited = []
        for clip in clips:
            rows = [row for row in selected_events if row['clip_id'] == clip]
            limited.extend(rows[: args.max_events_per_clip])
        selected_events = limited
    if not selected_events:
        raise RuntimeError('no selected events')
    expected_events_per_clip = defaultdict(int)
    for row in selected_events:
        expected_events_per_clip[row['clip_id']] += 1

    input_audit = verify_inputs(
        verify_all_frames=not args.skip_full_frame_hash,
        selected_clips=clips,
    )
    v9b = np.load(V9B_NPZ, allow_pickle=True)
    v9c = np.load(V9C_NPZ, allow_pickle=True)
    v9b_key = {
        (str(clip), int(frame), int(query)): index
        for index, (clip, frame, query) in enumerate(
            zip(v9b['clip_id'], v9b['frame_tau'], v9b['query_idx'])
        )
    }
    v9c_key = {
        (str(clip), int(frame), int(query)): index
        for index, (clip, frame, query) in enumerate(
            zip(v9c['clip_id'], v9c['frame_tau'], v9c['query_idx'])
        )
    }
    if len(v9b_key) != 27_648 or set(v9b_key) != set(v9c_key):
        raise RuntimeError('V9-A5.1b/c row-key mismatch')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = H.load_args_from_yaml(str(CONFIG))
    config.grad_checkpoint = False
    config.M_i = config.M
    predictor = H.Predictor(
        config,
        checkpoint_path=str(CHECKPOINT),
        support_grid_size=SUPPORT_GRID_SIZE,
    ).to(device).eval()
    model = predictor.model
    if predictor.model.M != EXPECTED_M or predictor.model.D != EXPECTED_D:
        raise RuntimeError(f'unexpected model dimensions: M={model.M}, D={model.D}')
    if str(model.memory_update_policy) != 'unconditional':
        raise RuntimeError(f'unexpected memory policy: {model.memory_update_policy}')

    scalar: dict[str, list[Any]] = defaultdict(list)
    split_rows = []
    runtime = []
    official_track_parity = {'p': 0.0, 'v_logit': 0.0, 'q_new': 0.0}
    branch_a_output_parity = {
        key: 0.0
        for key in ('p', 'v_logit', 'u_logit', 'q_new', 'q_pre', 'c1', 'c2')
    }
    branch_a_state_parity = {
        'q_init': 0.0,
        'point_memory': 0.0,
        'temporal_mask_mismatch': 0,
    }
    online_risk_mismatch = 0
    gt_visibility_mismatch = 0
    candidate_replay = {
        'rows': 0,
        'score_max_abs': 0.0,
        'raw_error_max_abs': 0.0,
        'refined_error_max_abs': 0.0,
        'official_error_max_abs': 0.0,
        'shared_score_top1_error_max_abs': 0.0,
        'risk_mismatch': 0,
    }
    storage_disjoint_failures = 0
    mutation_isolation_failures = 0
    cross_event_storage_alias_violations = 0
    event_key_mismatch = 0
    first_risk_mismatch = 0
    active_peak = 0

    for clip_id in clips:
        sequence, start_text = clip_id.split(':')
        clip_start = int(start_text)
        clip_events = [row for row in selected_events if row['clip_id'] == clip_id]
        events_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in clip_events:
            events_by_frame[int(row['frame_tau'])].append(row)
        sequence_dir = POD_ROOT / sequence
        annotation = np.load(
            sequence_dir / 'anno.npz',
            allow_pickle=True,
            mmap_mode='r',
        )
        trajectories = np.asarray(
            annotation['trajs_2d'][clip_start : clip_start + LENGTH],
            dtype=np.float32,
        )
        raw_visibility = np.asarray(
            annotation['visibs'][clip_start : clip_start + LENGTH],
            dtype=bool,
        )
        valid = (
            np.asarray(
                annotation['valids'][clip_start : clip_start + LENGTH],
                dtype=bool,
            )
            if 'valids' in annotation.files
            else np.ones_like(raw_visibility, dtype=bool)
        )
        finite = np.isfinite(trajectories).all(axis=2)
        visible_all = raw_visibility & valid & finite
        eligible = np.where(
            visible_all[0] & (np.sum(visible_all, axis=0) >= 24)
        )[0]
        chosen = np.sort(
            np.random.default_rng(H.stable_seed(sequence, clip_start)).choice(
                eligible,
                NQ,
                replace=False,
            )
        )
        gt_xy = trajectories[:, chosen]
        visible = visible_all[:, chosen]
        frame_paths = [
            sequence_dir / 'rgbs' / f'rgb_{clip_start + frame:05d}.jpg'
            for frame in range(LENGTH)
        ]
        width, height = Image.open(frame_paths[0]).size
        query_xy = np.empty((NQ, 2), dtype=np.float32)
        query_xy[:, 0] = gt_xy[0, :, 0] * H.UNIFIED_DENOM / float(width - 1)
        query_xy[:, 1] = gt_xy[0, :, 1] * H.UNIFIED_DENOM / float(height - 1)
        queries = torch.from_numpy(query_xy).float().to(device)
        support = H.get_points_on_a_grid(
            SUPPORT_GRID_SIZE,
            (256, 256),
            device,
        ).squeeze(0)
        combined = torch.cat([queries, support], dim=0)

        predictor.reset()
        predictor.initial_capacity = len(combined)
        first_image = H.load_frame(frame_paths[0], device)
        first_features = model.extract_frame_features(first_image)
        predictor.init_queries((first_features[-1], device), combined, 256, 256)
        if predictor.N != EXPECTED_N:
            raise RuntimeError(f'unexpected active query count: {predictor.N}')
        official_state = H.TrackerState(
            q_init=predictor.q_init[: predictor.N].clone(),
            point_memory=predictor.point_memory[: predictor.N].clone(),
            temporal_mask=predictor.temporal_mask[: predictor.N].clone(),
        )
        active: dict[int, ActiveEvent] = {}
        clip_start_time = time.perf_counter()

        for frame in range(LENGTH):
            if frame == 0:
                frame_features = first_features
            else:
                image = H.load_frame(frame_paths[frame], device)
                frame_features = model.extract_frame_features(image)
            official_diag = H.diagnostic_track_frame(
                model,
                official_state,
                frame_features,
            )
            if frame in {0, LENGTH // 2, LENGTH - 1} or frame in events_by_frame:
                p_ref, v_ref, q_ref = model.track_frame(
                    official_state.q_init,
                    official_state.temporal_mask,
                    official_state.point_memory,
                    frame_features,
                    256,
                    256,
                )
                official_track_parity['p'] = max(
                    official_track_parity['p'],
                    H.max_abs(official_diag['p'], p_ref),
                )
                official_track_parity['v_logit'] = max(
                    official_track_parity['v_logit'],
                    H.max_abs(official_diag['v_logit'], v_ref),
                )
                official_track_parity['q_new'] = max(
                    official_track_parity['q_new'],
                    H.max_abs(official_diag['q_new'], q_ref),
                )
            official_next = H.update_state(
                model,
                official_state,
                official_diag['q_new'],
            )

            online_risk = (
                torch.sigmoid(official_diag['v_logit'][:NQ]) < float(model.delta_v)
            ) | (
                torch.sigmoid(official_diag['u_logit'][:NQ]) >= 0.5
            )
            frozen_risk = np.asarray(
                [
                    bool(v9c['risk'][v9c_key[(clip_id, frame, query)]])
                    for query in range(NQ)
                ],
                dtype=bool,
            )
            online_risk_mismatch += int(
                np.sum(online_risk.detach().cpu().numpy() != frozen_risk)
            )
            gt_visibility_mismatch += int(
                np.sum(
                    visible[frame]
                    != np.asarray(
                        [
                            bool(
                                v9c['gt_visible'][
                                    v9c_key[(clip_id, frame, query)]
                                ]
                            )
                            for query in range(NQ)
                        ],
                        dtype=bool,
                    )
                )
            )

            candidate_cache: dict[int, Any] = {}

            def official_bundle(target: int) -> Any:
                if target not in candidate_cache:
                    candidate_cache[target] = H.build_candidate_bundle(
                        model,
                        official_diag['q_pre'][target : target + 1],
                        frame_features,
                        official_diag['c1'][target : target + 1],
                        k=K64,
                    )
                return candidate_cache[target]

            next_active: dict[int, ActiveEvent] = {}
            for event_id, event_state in active.items():
                diag_a, diag_b = forward_active_event(
                    model,
                    event_state,
                    frame_features,
                )
                for key, value in H.compare_diag(diag_a, official_diag).items():
                    branch_a_output_parity[key] = max(
                        branch_a_output_parity[key],
                        value,
                    )
                branch_a_state_parity['q_init'] = max(
                    branch_a_state_parity['q_init'],
                    H.max_abs(event_state.branch_a.q_init, official_state.q_init),
                )
                branch_a_state_parity['point_memory'] = max(
                    branch_a_state_parity['point_memory'],
                    H.max_abs(
                        event_state.branch_a.point_memory,
                        official_state.point_memory,
                    ),
                )
                branch_a_state_parity['temporal_mask_mismatch'] += int(
                    torch.sum(
                        event_state.branch_a.temporal_mask
                        != official_state.temporal_mask
                    ).detach().cpu()
                )

                horizon = frame - event_state.event_frame
                if horizon in HORIZONS:
                    target = event_state.query_idx
                    shared_bundle = official_bundle(target)
                    branch_b_bundle = H.build_candidate_bundle(
                        model,
                        diag_b['q_pre'][target : target + 1],
                        frame_features,
                        diag_b['c1'][target : target + 1],
                        k=K64,
                    )
                    gt_yx = np.asarray(
                        [
                            gt_xy[frame, target, 1] / float(height - 1),
                            gt_xy[frame, target, 0] / float(width - 1),
                        ],
                        dtype=np.float32,
                    )
                    official_yx = H.output_xy_to_yx_norm(
                        official_diag['p'][target]
                    )
                    branch_a_yx = H.output_xy_to_yx_norm(diag_a['p'][target])
                    branch_b_yx = H.output_xy_to_yx_norm(diag_b['p'][target])
                    shared_refined = H.model_xy_to_yx_norm(
                        shared_bundle.refined_model_xy[0],
                        model,
                    )
                    shared_score_index = int(
                        torch.argmax(shared_bundle.score[0]).item()
                    )
                    shared_score_yx = shared_refined[shared_score_index]
                    official_error = float(
                        np.linalg.norm(
                            (official_yx - gt_yx) * H.UNIFIED_DENOM
                        )
                    )
                    shared_score_error = float(
                        np.linalg.norm(
                            (shared_score_yx - gt_yx) * H.UNIFIED_DENOM
                        )
                    )
                    branch_a_error = float(
                        np.linalg.norm(
                            (branch_a_yx - gt_yx) * H.UNIFIED_DENOM
                        )
                    )
                    branch_b_error = float(
                        np.linalg.norm(
                            (branch_b_yx - gt_yx) * H.UNIFIED_DENOM
                        )
                    )
                    shared_oracle = min(official_error, shared_score_error)
                    independent_oracle = min(branch_a_error, branch_b_error)

                    frozen_row = v9b_key[(clip_id, frame, target)]
                    raw_positions = H.model_xy_to_yx_norm(
                        shared_bundle.positions_model_xy[0],
                        model,
                    )
                    refined_positions = H.model_xy_to_yx_norm(
                        shared_bundle.refined_model_xy[0],
                        model,
                    )
                    raw_error = np.linalg.norm(
                        (raw_positions - gt_yx[None, :]) * H.UNIFIED_DENOM,
                        axis=1,
                    ).astype(np.float32)
                    refined_error = np.linalg.norm(
                        (refined_positions - gt_yx[None, :])
                        * H.UNIFIED_DENOM,
                        axis=1,
                    ).astype(np.float32)
                    score_numpy = (
                        shared_bundle.score[0].detach().cpu().float().numpy()
                    )
                    candidate_replay['rows'] += 1
                    candidate_replay['score_max_abs'] = max(
                        candidate_replay['score_max_abs'],
                        float(
                            np.max(
                                np.abs(
                                    score_numpy
                                    - v9b['candidate_score'][frozen_row]
                                )
                            )
                        ),
                    )
                    candidate_replay['raw_error_max_abs'] = max(
                        candidate_replay['raw_error_max_abs'],
                        float(
                            np.max(
                                np.abs(
                                    raw_error
                                    - v9b['raw_candidate_error'][frozen_row]
                                )
                            )
                        ),
                    )
                    candidate_replay['refined_error_max_abs'] = max(
                        candidate_replay['refined_error_max_abs'],
                        float(
                            np.max(
                                np.abs(
                                    refined_error
                                    - v9b['refined_candidate_error'][frozen_row]
                                )
                            )
                        ),
                    )
                    candidate_replay['official_error_max_abs'] = max(
                        candidate_replay['official_error_max_abs'],
                        abs(
                            official_error
                            - float(v9b['official_final'][frozen_row])
                        ),
                    )
                    candidate_replay['shared_score_top1_error_max_abs'] = max(
                        candidate_replay['shared_score_top1_error_max_abs'],
                        abs(
                            shared_score_error
                            - float(
                                v9b['refined_score_top1_K64'][frozen_row]
                            )
                        ),
                    )
                    candidate_replay['risk_mismatch'] += int(
                        bool(online_risk[target])
                        != bool(v9b['risk'][frozen_row])
                    )

                    top16_a = torch.topk(
                        diag_a['c1'][target],
                        k=16,
                        largest=True,
                        sorted=True,
                    ).indices
                    top16_b = torch.topk(
                        diag_b['c1'][target],
                        k=16,
                        largest=True,
                        sorted=True,
                    ).indices
                    top64_a = torch.topk(
                        diag_a['c1'][target],
                        k=64,
                        largest=True,
                        sorted=True,
                    ).indices
                    top64_b = torch.topk(
                        diag_b['c1'][target],
                        k=64,
                        largest=True,
                        sorted=True,
                    ).indices
                    set16 = H.candidate_set_metrics(model, top16_a, top16_b)
                    set64 = H.candidate_set_metrics(model, top64_a, top64_b)
                    branch_b_refined = H.model_xy_to_yx_norm(
                        branch_b_bundle.refined_model_xy[0],
                        model,
                    )
                    branch_b_refined_error = np.linalg.norm(
                        (branch_b_refined - gt_yx[None, :])
                        * H.UNIFIED_DENOM,
                        axis=1,
                    )
                    shared_refined_oracle_k16 = float(np.min(refined_error[:16]))
                    shared_refined_oracle_k64 = float(np.min(refined_error))
                    branch_b_refined_oracle_k16 = float(
                        np.min(branch_b_refined_error[:16])
                    )
                    branch_b_refined_oracle_k64 = float(
                        np.min(branch_b_refined_error)
                    )
                    union_refined_oracle_k16 = min(
                        shared_refined_oracle_k16,
                        branch_b_refined_oracle_k16,
                    )
                    union_refined_oracle_k64 = min(
                        shared_refined_oracle_k64,
                        branch_b_refined_oracle_k64,
                    )
                    horizon_key = v9c_key[(clip_id, frame, target)]
                    scalar['event_id'].append(event_id)
                    scalar['clip_id'].append(clip_id)
                    scalar['sequence'].append(sequence)
                    scalar['query_idx'].append(target)
                    scalar['event_frame'].append(event_state.event_frame)
                    scalar['horizon'].append(horizon)
                    scalar['frame_tau'].append(frame)
                    scalar['gt_visible'].append(bool(visible[frame, target]))
                    scalar['reentry_first'].append(
                        bool(v9c['reentry_first'][horizon_key])
                    )
                    scalar['reentry_early8'].append(
                        bool(v9c['reentry_early8'][horizon_key])
                    )
                    scalar['official_error'].append(official_error)
                    scalar['shared_score_top1_error'].append(shared_score_error)
                    scalar['shared_b2_oracle'].append(shared_oracle)
                    scalar['branch_a_error'].append(branch_a_error)
                    scalar['branch_b_error'].append(branch_b_error)
                    scalar['independent_b2_oracle'].append(independent_oracle)
                    scalar['shared_refined_oracle_k16'].append(
                        shared_refined_oracle_k16
                    )
                    scalar['shared_refined_oracle_k64'].append(
                        shared_refined_oracle_k64
                    )
                    scalar['branch_b_refined_oracle_k16'].append(
                        branch_b_refined_oracle_k16
                    )
                    scalar['branch_b_refined_oracle_k64'].append(
                        branch_b_refined_oracle_k64
                    )
                    scalar['union_refined_oracle_k16'].append(
                        union_refined_oracle_k16
                    )
                    scalar['union_refined_oracle_k64'].append(
                        union_refined_oracle_k64
                    )
                    scalar['target_q_new_l2'].append(
                        H.tensor_l2(
                            diag_a['q_new'][target],
                            diag_b['q_new'][target],
                        )
                    )
                    scalar['target_q_new_max_abs'].append(
                        H.max_abs(
                            diag_a['q_new'][target],
                            diag_b['q_new'][target],
                        )
                    )
                    scalar['target_c1_l2'].append(
                        H.tensor_l2(
                            diag_a['c1'][target],
                            diag_b['c1'][target],
                        )
                    )
                    scalar['target_c1_max_abs'].append(
                        H.max_abs(
                            diag_a['c1'][target],
                            diag_b['c1'][target],
                        )
                    )
                    scalar['top16_exact_equal'].append(
                        bool(set16['exact_set_equal'])
                    )
                    scalar['top16_overlap'].append(int(set16['overlap']))
                    scalar['top16_jaccard'].append(float(set16['jaccard']))
                    scalar['top16_hausdorff_px'].append(
                        float(set16['hausdorff_px'])
                    )
                    scalar['top64_exact_equal'].append(
                        bool(set64['exact_set_equal'])
                    )
                    scalar['top64_overlap'].append(int(set64['overlap']))
                    scalar['top64_jaccard'].append(float(set64['jaccard']))
                    scalar['top64_hausdorff_px'].append(
                        float(set64['hausdorff_px'])
                    )
                    scalar['final_coordinate_divergence_px'].append(
                        float(
                            np.linalg.norm(
                                (branch_a_yx - branch_b_yx)
                                * H.UNIFIED_DENOM
                            )
                        )
                    )
                    scalar['useful_novel'].append(
                        bool(branch_b_error < shared_oracle - 1.0)
                    )
                    scalar['branch_b_final_le4_while_shared_gt4'].append(
                        bool(branch_b_error <= 4.0 and shared_oracle > 4.0)
                    )
                    scalar['branch_b_candidate_novel_k16'].append(
                        bool(
                            np.min(branch_b_refined_error[:16]) <= 4.0
                            and shared_oracle > 4.0
                        )
                    )
                    scalar['branch_b_candidate_novel_k64'].append(
                        bool(
                            branch_b_refined_oracle_k64 <= 4.0
                            and shared_oracle > 4.0
                        )
                    )
                    scalar['branch_b_candidate_beats_shared_k16_by1'].append(
                        bool(
                            branch_b_refined_oracle_k16
                            < shared_refined_oracle_k16 - 1.0
                        )
                    )
                    scalar['branch_b_candidate_beats_shared_k64_by1'].append(
                        bool(
                            branch_b_refined_oracle_k64
                            < shared_refined_oracle_k64 - 1.0
                        )
                    )
                    scalar['branch_b_candidate_le4_shared_k16_gt4'].append(
                        bool(
                            branch_b_refined_oracle_k16 <= 4.0
                            and shared_refined_oracle_k16 > 4.0
                        )
                    )
                    scalar['branch_b_candidate_le4_shared_k64_gt4'].append(
                        bool(
                            branch_b_refined_oracle_k64 <= 4.0
                            and shared_refined_oracle_k64 > 4.0
                        )
                    )

                branch_a_next = H.update_state(
                    model,
                    event_state.branch_a,
                    diag_a['q_new'],
                )
                branch_b_next = H.update_state(
                    model,
                    event_state.branch_b,
                    diag_b['q_new'],
                )
                branch_a_state_parity['point_memory'] = max(
                    branch_a_state_parity['point_memory'],
                    H.max_abs(
                        branch_a_next.point_memory,
                        official_next.point_memory,
                    ),
                )
                branch_a_state_parity['temporal_mask_mismatch'] += int(
                    torch.sum(
                        branch_a_next.temporal_mask
                        != official_next.temporal_mask
                    ).detach().cpu()
                )
                if horizon < max(HORIZONS):
                    event_state.branch_a = branch_a_next
                    event_state.branch_b = branch_b_next
                    next_active[event_id] = event_state

            for event in events_by_frame.get(frame, []):
                target = int(event['query_idx'])
                v9c_row = v9c_key[(clip_id, frame, target)]
                frozen_row = v9b_key[(clip_id, frame, target)]
                if int(event['source_row_index']) != v9c_row:
                    event_key_mismatch += 1
                if not bool(online_risk[target]):
                    event_key_mismatch += 1
                if any(
                    bool(v9c['risk'][v9c_key[(clip_id, prior, target)]])
                    for prior in range(1, frame)
                ):
                    first_risk_mismatch += 1
                bundle = official_bundle(target)
                split_gt_yx = np.asarray(
                    [
                        gt_xy[frame, target, 1] / float(height - 1),
                        gt_xy[frame, target, 0] / float(width - 1),
                    ],
                    dtype=np.float32,
                )
                split_raw_positions = H.model_xy_to_yx_norm(
                    bundle.positions_model_xy[0],
                    model,
                )
                split_refined_positions = H.model_xy_to_yx_norm(
                    bundle.refined_model_xy[0],
                    model,
                )
                split_raw_error = np.linalg.norm(
                    (split_raw_positions - split_gt_yx[None, :])
                    * H.UNIFIED_DENOM,
                    axis=1,
                ).astype(np.float32)
                split_refined_error = np.linalg.norm(
                    (split_refined_positions - split_gt_yx[None, :])
                    * H.UNIFIED_DENOM,
                    axis=1,
                ).astype(np.float32)
                split_official_yx = H.output_xy_to_yx_norm(
                    official_diag['p'][target]
                )
                split_official_error = float(
                    np.linalg.norm(
                        (split_official_yx - split_gt_yx) * H.UNIFIED_DENOM
                    )
                )
                split_score_numpy = (
                    bundle.score[0].detach().cpu().float().numpy()
                )
                split_score_index = int(np.argmax(split_score_numpy))
                split_score_top1_error = float(
                    split_refined_error[split_score_index]
                )
                candidate_replay['rows'] += 1
                candidate_replay['score_max_abs'] = max(
                    candidate_replay['score_max_abs'],
                    float(
                        np.max(
                            np.abs(
                                split_score_numpy
                                - v9b['candidate_score'][frozen_row]
                            )
                        )
                    ),
                )
                candidate_replay['raw_error_max_abs'] = max(
                    candidate_replay['raw_error_max_abs'],
                    float(
                        np.max(
                            np.abs(
                                split_raw_error
                                - v9b['raw_candidate_error'][frozen_row]
                            )
                        )
                    ),
                )
                candidate_replay['refined_error_max_abs'] = max(
                    candidate_replay['refined_error_max_abs'],
                    float(
                        np.max(
                            np.abs(
                                split_refined_error
                                - v9b['refined_candidate_error'][frozen_row]
                            )
                        )
                    ),
                )
                candidate_replay['official_error_max_abs'] = max(
                    candidate_replay['official_error_max_abs'],
                    abs(
                        split_official_error
                        - float(v9b['official_final'][frozen_row])
                    ),
                )
                candidate_replay['shared_score_top1_error_max_abs'] = max(
                    candidate_replay['shared_score_top1_error_max_abs'],
                    abs(
                        split_score_top1_error
                        - float(v9b['refined_score_top1_K64'][frozen_row])
                    ),
                )
                candidate_replay['risk_mismatch'] += int(
                    bool(online_risk[target])
                    != bool(v9b['risk'][frozen_row])
                )
                active_event, split = create_active_event(
                    model,
                    event,
                    official_state,
                    official_diag['q_new'],
                    bundle,
                )
                if not split['storage_disjoint_within_event']:
                    storage_disjoint_failures += 1
                if not split['mutation_isolation']:
                    mutation_isolation_failures += 1
                all_states = [official_next]
                for existing in next_active.values():
                    all_states.extend([existing.branch_a, existing.branch_b])
                all_states.extend([active_event.branch_a, active_event.branch_b])
                for field in ('q_init', 'point_memory', 'temporal_mask'):
                    pointers = [H.state_storage(state)[field] for state in all_states]
                    if len(set(pointers)) != len(pointers):
                        cross_event_storage_alias_violations += 1
                split_rows.append(
                    {
                        'event_id': int(event['event_id']),
                        'clip_id': clip_id,
                        'query_idx': target,
                        'event_frame': frame,
                        **split,
                    }
                )
                branch_a_state_parity['q_init'] = max(
                    branch_a_state_parity['q_init'],
                    H.max_abs(active_event.branch_a.q_init, official_next.q_init),
                )
                branch_a_state_parity['point_memory'] = max(
                    branch_a_state_parity['point_memory'],
                    H.max_abs(
                        active_event.branch_a.point_memory,
                        official_next.point_memory,
                    ),
                )
                branch_a_state_parity['temporal_mask_mismatch'] += int(
                    torch.sum(
                        active_event.branch_a.temporal_mask
                        != official_next.temporal_mask
                    ).detach().cpu()
                )
                next_active[int(event['event_id'])] = active_event

            official_state = official_next
            active = next_active
            active_peak = max(active_peak, len(active))

        if active:
            raise RuntimeError(f'active events remain after clip: {list(active)}')
        runtime.append(
            {
                'clip_id': clip_id,
                'events': int(len(clip_events)),
                'seconds': float(time.perf_counter() - clip_start_time),
            }
        )
        print(json.dumps(runtime[-1]), flush=True)

    arrays: dict[str, np.ndarray] = {
        'event_id': np.asarray(scalar['event_id'], dtype=np.int16),
        'clip_id': np.asarray(scalar['clip_id'], dtype=object),
        'sequence': np.asarray(scalar['sequence'], dtype=object),
        'query_idx': np.asarray(scalar['query_idx'], dtype=np.int16),
        'event_frame': np.asarray(scalar['event_frame'], dtype=np.int16),
        'horizon': np.asarray(scalar['horizon'], dtype=np.int16),
        'frame_tau': np.asarray(scalar['frame_tau'], dtype=np.int16),
        'gt_visible': np.asarray(scalar['gt_visible'], dtype=bool),
        'reentry_first': np.asarray(scalar['reentry_first'], dtype=bool),
        'reentry_early8': np.asarray(scalar['reentry_early8'], dtype=bool),
        'official_error': np.asarray(scalar['official_error'], dtype=np.float32),
        'shared_score_top1_error': np.asarray(
            scalar['shared_score_top1_error'], dtype=np.float32
        ),
        'shared_b2_oracle': np.asarray(
            scalar['shared_b2_oracle'], dtype=np.float32
        ),
        'branch_a_error': np.asarray(scalar['branch_a_error'], dtype=np.float32),
        'branch_b_error': np.asarray(scalar['branch_b_error'], dtype=np.float32),
        'independent_b2_oracle': np.asarray(
            scalar['independent_b2_oracle'], dtype=np.float32
        ),
        'shared_refined_oracle_k16': np.asarray(
            scalar['shared_refined_oracle_k16'], dtype=np.float32
        ),
        'shared_refined_oracle_k64': np.asarray(
            scalar['shared_refined_oracle_k64'], dtype=np.float32
        ),
        'branch_b_refined_oracle_k16': np.asarray(
            scalar['branch_b_refined_oracle_k16'], dtype=np.float32
        ),
        'branch_b_refined_oracle_k64': np.asarray(
            scalar['branch_b_refined_oracle_k64'], dtype=np.float32
        ),
        'union_refined_oracle_k16': np.asarray(
            scalar['union_refined_oracle_k16'], dtype=np.float32
        ),
        'union_refined_oracle_k64': np.asarray(
            scalar['union_refined_oracle_k64'], dtype=np.float32
        ),
        'target_q_new_l2': np.asarray(scalar['target_q_new_l2'], dtype=np.float32),
        'target_q_new_max_abs': np.asarray(
            scalar['target_q_new_max_abs'], dtype=np.float32
        ),
        'target_c1_l2': np.asarray(scalar['target_c1_l2'], dtype=np.float32),
        'target_c1_max_abs': np.asarray(
            scalar['target_c1_max_abs'], dtype=np.float32
        ),
        'top16_exact_equal': np.asarray(
            scalar['top16_exact_equal'], dtype=bool
        ),
        'top16_overlap': np.asarray(scalar['top16_overlap'], dtype=np.int16),
        'top16_jaccard': np.asarray(scalar['top16_jaccard'], dtype=np.float32),
        'top16_hausdorff_px': np.asarray(
            scalar['top16_hausdorff_px'], dtype=np.float32
        ),
        'top64_exact_equal': np.asarray(
            scalar['top64_exact_equal'], dtype=bool
        ),
        'top64_overlap': np.asarray(scalar['top64_overlap'], dtype=np.int16),
        'top64_jaccard': np.asarray(scalar['top64_jaccard'], dtype=np.float32),
        'top64_hausdorff_px': np.asarray(
            scalar['top64_hausdorff_px'], dtype=np.float32
        ),
        'final_coordinate_divergence_px': np.asarray(
            scalar['final_coordinate_divergence_px'], dtype=np.float32
        ),
        'useful_novel': np.asarray(scalar['useful_novel'], dtype=bool),
        'branch_b_final_le4_while_shared_gt4': np.asarray(
            scalar['branch_b_final_le4_while_shared_gt4'], dtype=bool
        ),
        'branch_b_candidate_novel_k16': np.asarray(
            scalar['branch_b_candidate_novel_k16'], dtype=bool
        ),
        'branch_b_candidate_novel_k64': np.asarray(
            scalar['branch_b_candidate_novel_k64'], dtype=bool
        ),
        'branch_b_candidate_beats_shared_k16_by1': np.asarray(
            scalar['branch_b_candidate_beats_shared_k16_by1'], dtype=bool
        ),
        'branch_b_candidate_beats_shared_k64_by1': np.asarray(
            scalar['branch_b_candidate_beats_shared_k64_by1'], dtype=bool
        ),
        'branch_b_candidate_le4_shared_k16_gt4': np.asarray(
            scalar['branch_b_candidate_le4_shared_k16_gt4'], dtype=bool
        ),
        'branch_b_candidate_le4_shared_k64_gt4': np.asarray(
            scalar['branch_b_candidate_le4_shared_k64_gt4'], dtype=bool
        ),
    }
    expected_rows = len(selected_events) * len(HORIZONS)
    if len(arrays['event_id']) != expected_rows:
        raise RuntimeError(
            f'event-horizon row mismatch: {len(arrays["event_id"])} != {expected_rows}'
        )
    row_keys = list(
        zip(
            arrays['event_id'].astype(int).tolist(),
            arrays['horizon'].astype(int).tolist(),
        )
    )
    if len(set(row_keys)) != expected_rows:
        raise RuntimeError('duplicate event-horizon rows')
    for key, value in arrays.items():
        if value.dtype in (object, bool):
            continue
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f'non-finite array: {key}')

    visible = arrays['gt_visible']
    h8 = visible & (arrays['horizon'] == 8)
    pooled = visible
    early8 = visible & arrays['reentry_early8']
    first_reentry = visible & arrays['reentry_first']
    candidate = arrays['independent_b2_oracle'].astype(np.float64)
    baseline = arrays['shared_b2_oracle'].astype(np.float64)

    summaries = {
        'horizon8_all_visible': summary(candidate, baseline, h8),
        'pooled_all_visible': summary(candidate, baseline, pooled),
        'pooled_early8': summary(candidate, baseline, early8),
        'pooled_first_reentry_descriptive': summary(
            candidate, baseline, first_reentry
        ),
        'per_horizon': {
            str(horizon): summary(
                candidate,
                baseline,
                visible & (arrays['horizon'] == horizon),
            )
            for horizon in HORIZONS
        },
        'horizon8_per_sequence': {
            sequence: summary(
                candidate,
                baseline,
                h8 & (arrays['sequence'] == sequence),
            )
            for sequence in SEQUENCES
        },
        'pooled_per_sequence': {
            sequence: summary(
                candidate,
                baseline,
                pooled & (arrays['sequence'] == sequence),
            )
            for sequence in SEQUENCES
        },
    }
    bootstrap = {
        'horizon8': clip_bootstrap(
            arrays['clip_id'], h8, candidate, baseline
        ),
        'pooled': clip_bootstrap(
            arrays['clip_id'], pooled, candidate, baseline
        ),
        'early8': clip_bootstrap(
            arrays['clip_id'], early8, candidate, baseline
        ),
    }

    visible_count = int(np.sum(visible))
    q_divergent = arrays['target_q_new_l2'] > 1e-4
    c1_changed = ~arrays['top16_exact_equal']
    useful_novel = arrays['useful_novel'] & visible
    useful_per_sequence = {
        sequence: int(
            np.sum(useful_novel & (arrays['sequence'] == sequence))
        )
        for sequence in SEQUENCES
    }
    shared_candidate_k16 = arrays['shared_refined_oracle_k16'].astype(np.float64)
    shared_candidate_k64 = arrays['shared_refined_oracle_k64'].astype(np.float64)
    branch_candidate_k16 = arrays['branch_b_refined_oracle_k16'].astype(np.float64)
    branch_candidate_k64 = arrays['branch_b_refined_oracle_k64'].astype(np.float64)
    union_candidate_k16 = arrays['union_refined_oracle_k16'].astype(np.float64)
    union_candidate_k64 = arrays['union_refined_oracle_k64'].astype(np.float64)
    candidate_reachability_supplement = {
        'non_gating': True,
        'interpretation': (
            'Post-formal conservative diagnostic. It compares the complete '
            'Branch B refined candidate pool against the complete shared-state '
            'candidate pool at matched K. It cannot alter the predeclared gates.'
        ),
        'pooled_visible': {
            'branch_b_vs_shared_k16': summary(
                branch_candidate_k16, shared_candidate_k16, visible
            ),
            'branch_b_vs_shared_k64': summary(
                branch_candidate_k64, shared_candidate_k64, visible
            ),
            'union_vs_shared_k16': summary(
                union_candidate_k16, shared_candidate_k16, visible
            ),
            'union_vs_shared_k64': summary(
                union_candidate_k64, shared_candidate_k64, visible
            ),
        },
        'horizon8_visible': {
            'branch_b_vs_shared_k16': summary(
                branch_candidate_k16, shared_candidate_k16, h8
            ),
            'branch_b_vs_shared_k64': summary(
                branch_candidate_k64, shared_candidate_k64, h8
            ),
            'union_vs_shared_k16': summary(
                union_candidate_k16, shared_candidate_k16, h8
            ),
            'union_vs_shared_k64': summary(
                union_candidate_k64, shared_candidate_k64, h8
            ),
        },
        'per_horizon_visible': {
            str(horizon): {
                'branch_b_vs_shared_k16': summary(
                    branch_candidate_k16,
                    shared_candidate_k16,
                    visible & (arrays['horizon'] == horizon),
                ),
                'branch_b_vs_shared_k64': summary(
                    branch_candidate_k64,
                    shared_candidate_k64,
                    visible & (arrays['horizon'] == horizon),
                ),
                'union_vs_shared_k16': summary(
                    union_candidate_k16,
                    shared_candidate_k16,
                    visible & (arrays['horizon'] == horizon),
                ),
                'union_vs_shared_k64': summary(
                    union_candidate_k64,
                    shared_candidate_k64,
                    visible & (arrays['horizon'] == horizon),
                ),
            }
            for horizon in HORIZONS
        },
        'per_sequence_pooled_visible': {
            sequence: {
                'branch_b_vs_shared_k16': summary(
                    branch_candidate_k16,
                    shared_candidate_k16,
                    visible & (arrays['sequence'] == sequence),
                ),
                'branch_b_vs_shared_k64': summary(
                    branch_candidate_k64,
                    shared_candidate_k64,
                    visible & (arrays['sequence'] == sequence),
                ),
                'union_vs_shared_k16': summary(
                    union_candidate_k16,
                    shared_candidate_k16,
                    visible & (arrays['sequence'] == sequence),
                ),
                'union_vs_shared_k64': summary(
                    union_candidate_k64,
                    shared_candidate_k64,
                    visible & (arrays['sequence'] == sequence),
                ),
            }
            for sequence in SEQUENCES
        },
        'novelty_counts_visible': {
            'branch_b_beats_shared_k16_by_gt1': int(
                np.sum(
                    arrays['branch_b_candidate_beats_shared_k16_by1'] & visible
                )
            ),
            'branch_b_beats_shared_k64_by_gt1': int(
                np.sum(
                    arrays['branch_b_candidate_beats_shared_k64_by1'] & visible
                )
            ),
            'branch_b_le4_shared_k16_gt4': int(
                np.sum(
                    arrays['branch_b_candidate_le4_shared_k16_gt4'] & visible
                )
            ),
            'branch_b_le4_shared_k64_gt4': int(
                np.sum(
                    arrays['branch_b_candidate_le4_shared_k64_gt4'] & visible
                )
            ),
            'branch_b_beats_shared_k64_by_gt1_per_sequence': {
                sequence: int(
                    np.sum(
                        arrays['branch_b_candidate_beats_shared_k64_by1']
                        & visible
                        & (arrays['sequence'] == sequence)
                    )
                )
                for sequence in SEQUENCES
            },
            'branch_b_le4_shared_k64_gt4_per_sequence': {
                sequence: int(
                    np.sum(
                        arrays['branch_b_candidate_le4_shared_k64_gt4']
                        & visible
                        & (arrays['sequence'] == sequence)
                    )
                )
                for sequence in SEQUENCES
            },
        },
    }

    mechanistic = {
        'visible_event_horizon_rows': visible_count,
        'target_q_new_l2_gt_1e4_fraction_visible': (
            float(np.mean(q_divergent[visible])) if visible_count else None
        ),
        'top16_set_changed_fraction_visible': (
            float(np.mean(c1_changed[visible])) if visible_count else None
        ),
        'target_q_new_l2': {
            'mean_visible': float(np.mean(arrays['target_q_new_l2'][visible]))
            if visible_count
            else None,
            'median_visible': float(
                np.median(arrays['target_q_new_l2'][visible])
            )
            if visible_count
            else None,
            'max_visible': float(np.max(arrays['target_q_new_l2'][visible]))
            if visible_count
            else None,
        },
        'top16_overlap_mean_visible': float(
            np.mean(arrays['top16_overlap'][visible])
        )
        if visible_count
        else None,
        'top64_overlap_mean_visible': float(
            np.mean(arrays['top64_overlap'][visible])
        )
        if visible_count
        else None,
        'final_coordinate_divergence_mean_visible': float(
            np.mean(arrays['final_coordinate_divergence_px'][visible])
        )
        if visible_count
        else None,
        'useful_novel_rows_visible': int(np.sum(useful_novel)),
        'useful_novel_per_sequence': useful_per_sequence,
        'branch_b_final_le4_while_shared_gt4': int(
            np.sum(
                arrays['branch_b_final_le4_while_shared_gt4'] & visible
            )
        ),
        'branch_b_candidate_novel_k16': int(
            np.sum(arrays['branch_b_candidate_novel_k16'] & visible)
        ),
        'branch_b_candidate_novel_k64': int(
            np.sum(arrays['branch_b_candidate_novel_k64'] & visible)
        ),
    }

    formal_run = (
        args.max_clips == 0
        and args.max_events_per_clip == 0
        and len(clips) == 9
        and len(selected_events) == 72
    )
    if not formal_run:
        gates = {
            'preflight_only': True,
            'pass_all': False,
            'decision': (
                'PREFLIGHT_ONLY: event scheduling and integrity are evaluated; '
                'the frozen 72-event scientific gates are disabled.'
            ),
        }
    else:
        h8_sequence_mean = {
            sequence: bool(
                summaries['horizon8_per_sequence'][sequence]['candidate_mean']
                < summaries['horizon8_per_sequence'][sequence]['baseline_mean']
            )
            for sequence in SEQUENCES
        }
        h8_sequence_safe16 = {
            sequence: bool(
                summaries['horizon8_per_sequence'][sequence][
                    'candidate_safe16'
                ]
                >= summaries['horizon8_per_sequence'][sequence][
                    'baseline_safe16'
                ]
            )
            for sequence in SEQUENCES
        }
        primary_gate = {
            'sequence_mean_improved': h8_sequence_mean,
            'sequence_safe16_not_decreased': h8_sequence_safe16,
            'global_better_gt_worse': bool(
                summaries['horizon8_all_visible']['better']
                > summaries['horizon8_all_visible']['worse']
            ),
            'clip_bootstrap_ci_upper_lt_zero': bool(
                bootstrap['horizon8']['mean_difference_95_ci'][1] < 0.0
            ),
            'clip_better_gt_worse': bool(
                bootstrap['horizon8']['clip_better']
                > bootstrap['horizon8']['clip_worse']
            ),
        }
        primary_gate['pass_all'] = bool(
            all(h8_sequence_mean.values())
            and all(h8_sequence_safe16.values())
            and primary_gate['global_better_gt_worse']
            and primary_gate['clip_bootstrap_ci_upper_lt_zero']
            and primary_gate['clip_better_gt_worse']
        )
        pooled_gate = {
            'mean_difference_lt_zero': bool(
                summaries['pooled_all_visible']['mean_difference'] < 0.0
            ),
            'better_gt_worse': bool(
                summaries['pooled_all_visible']['better']
                > summaries['pooled_all_visible']['worse']
            ),
            'clip_bootstrap_ci_upper_lt_zero': bool(
                bootstrap['pooled']['mean_difference_95_ci'][1] < 0.0
            ),
        }
        pooled_gate['pass_all'] = bool(all(pooled_gate.values()))
        early8_gate = {
            'n_at_least_20': summaries['pooled_early8']['n'] >= 20,
            'mean_not_increased': bool(
                summaries['pooled_early8']['candidate_mean']
                <= summaries['pooled_early8']['baseline_mean']
            )
            if summaries['pooled_early8']['n'] > 0
            else False,
        }
        early8_gate['pass_all'] = bool(
            early8_gate['n_at_least_20']
            and early8_gate['mean_not_increased']
        )
        nondegenerate_gate = {
            'q_new_divergent_fraction_ge_0_10': bool(
                mechanistic['target_q_new_l2_gt_1e4_fraction_visible']
                >= 0.10
            ),
            'top16_changed_fraction_ge_0_10': bool(
                mechanistic['top16_set_changed_fraction_visible'] >= 0.10
            ),
            'useful_novel_each_sequence': {
                sequence: useful_per_sequence[sequence] >= 1
                for sequence in SEQUENCES
            },
            'useful_novel_global_ge_10': bool(
                mechanistic['useful_novel_rows_visible'] >= 10
            ),
        }
        nondegenerate_gate['pass_all'] = bool(
            nondegenerate_gate['q_new_divergent_fraction_ge_0_10']
            and nondegenerate_gate['top16_changed_fraction_ge_0_10']
            and all(
                nondegenerate_gate['useful_novel_each_sequence'].values()
            )
            and nondegenerate_gate['useful_novel_global_ge_10']
        )
        pass_all = bool(
            primary_gate['pass_all']
            and pooled_gate['pass_all']
            and early8_gate['pass_all']
            and nondegenerate_gate['pass_all']
        )
        decision = (
            'INDEPENDENT_STATE_PASS: the independent B2 state branch beats the '
            'shared-state capacity-2 control and shows useful non-degenerate future '
            'state/candidate novelty. A later full-stream independent-state B2 beam '
            'audit is authorized; do not train or read DAVIS.'
            if pass_all
            else 'INDEPENDENT_STATE_FAIL: the frozen independent-state branch does '
            'not satisfy the capacity-matched statistical and mechanistic gates. '
            'Close the temporal multi-state route; do not train or read DAVIS.'
        )
        gates = {
            'primary_horizon8': primary_gate,
            'pooled_horizons': pooled_gate,
            'early8_secondary': early8_gate,
            'mechanistic_nondegeneracy': nondegenerate_gate,
            'pass_all': pass_all,
            'decision': decision,
        }

    integrity = {
        'input_audit': input_audit,
        'formal_run': formal_run,
        'selected_clips': clips,
        'selected_events': int(len(selected_events)),
        'expected_events_per_clip': dict(expected_events_per_clip),
        'event_horizon_rows': int(len(arrays['event_id'])),
        'expected_event_horizon_rows': int(expected_rows),
        'unique_event_horizon_keys': int(len(set(row_keys))),
        'split_rows': int(len(split_rows)),
        'active_event_peak': int(active_peak),
        'official_track_parity': official_track_parity,
        'branch_a_output_parity': branch_a_output_parity,
        'branch_a_state_parity': branch_a_state_parity,
        'online_risk_mismatch': int(online_risk_mismatch),
        'gt_visibility_mismatch': int(gt_visibility_mismatch),
        'candidate_replay': candidate_replay,
        'expected_candidate_replay_rows': int(
            len(selected_events) * (1 + len(HORIZONS))
        ),
        'storage_disjoint_failures': int(storage_disjoint_failures),
        'mutation_isolation_failures': int(mutation_isolation_failures),
        'cross_event_storage_alias_violations': int(
            cross_event_storage_alias_violations
        ),
        'event_key_mismatch': int(event_key_mismatch),
        'first_risk_mismatch': int(first_risk_mismatch),
        'all_outputs_finite': True,
    }
    if max(official_track_parity.values()) > TOL:
        raise RuntimeError(f'official track parity failed: {official_track_parity}')
    if max(branch_a_output_parity.values()) > TOL:
        raise RuntimeError(f'Branch A output parity failed: {branch_a_output_parity}')
    if (
        branch_a_state_parity['q_init'] > TOL
        or branch_a_state_parity['point_memory'] > TOL
        or branch_a_state_parity['temporal_mask_mismatch'] != 0
    ):
        raise RuntimeError(f'Branch A state parity failed: {branch_a_state_parity}')
    if online_risk_mismatch or gt_visibility_mismatch:
        raise RuntimeError('online V9-A5.1c replay mismatch')
    expected_candidate_replay_rows = len(selected_events) * (1 + len(HORIZONS))
    if candidate_replay['rows'] != expected_candidate_replay_rows:
        raise RuntimeError(
            f'candidate replay row mismatch: {candidate_replay["rows"]} '
            f'!= {expected_candidate_replay_rows}'
        )
    if candidate_replay['score_max_abs'] > 6e-5:
        raise RuntimeError(f'candidate score replay failed: {candidate_replay}')
    if max(
        candidate_replay['raw_error_max_abs'],
        candidate_replay['refined_error_max_abs'],
        candidate_replay['shared_score_top1_error_max_abs'],
    ) > CANDIDATE_TOL:
        raise RuntimeError(f'candidate error replay failed: {candidate_replay}')
    if candidate_replay['official_error_max_abs'] > TOL:
        raise RuntimeError(f'official error replay failed: {candidate_replay}')
    if candidate_replay['risk_mismatch']:
        raise RuntimeError(f'candidate risk replay failed: {candidate_replay}')
    if any(
        [
            storage_disjoint_failures,
            mutation_isolation_failures,
            cross_event_storage_alias_violations,
            event_key_mismatch,
            first_risk_mismatch,
        ]
    ):
        raise RuntimeError('event/state integrity counter is nonzero')

    result = {
        'date': '2026-07-11',
        'script': 'scripts/v9a52_independent_state_branching_audit.py',
        'protocol': {
            'dataset': 'PointOdyssey frozen event audit',
            'clips': clips,
            'events': int(len(selected_events)),
            'horizons': list(HORIZONS),
            'primary_horizon': 8,
            'active_queries': EXPECTED_N,
            'memory_size': EXPECTED_M,
            'feature_dim': EXPECTED_D,
            'shared_control': (
                'min(official final, official-state K64 score-top1 singleton refined)'
            ),
            'independent_output': 'min(Branch A final, Branch B final)',
            'bootstrap_unit': 'clip_id',
            'bootstrap_resamples': BOOTSTRAP_RESAMPLES,
            'bootstrap_seed': BOOTSTRAP_SEED,
            'training': False,
            'davis_read': False,
            'threshold_tuning': False,
            'seconds': float(time.perf_counter() - start_all),
        },
        'integrity': integrity,
        'split_diagnostics': split_rows,
        'per_clip_runtime': runtime,
        'summaries': summaries,
        'bootstrap': bootstrap,
        'mechanistic': mechanistic,
        'candidate_reachability_supplement': candidate_reachability_supplement,
        'gates': gates,
        'provenance': {
            'branch': input_audit['branch'],
            'head': input_audit['head'],
            'worktree_root': str(WORKTREE_ROOT),
            'source_asset_root': str(SOURCE_ROOT),
            'script_sha256': sha256(Path(__file__).resolve()),
            'helper_sha256': sha256(HELPER_SCRIPT),
            'input_manifest_sha256': sha256(INPUT_MANIFEST),
            'event_manifest_sha256': sha256(EVENT_MANIFEST),
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
                'events': len(selected_events),
                'rows': len(arrays['event_id']),
                'visible_rows': int(np.sum(visible)),
                'decision': gates['decision'],
                'gate_pass': bool(gates.get('pass_all', False)),
                'integrity': {
                    key: value
                    for key, value in integrity.items()
                    if key not in {'input_audit'}
                },
                'primary': summaries['horizon8_all_visible'],
                'mechanistic': mechanistic,
                'candidate_reachability_supplement': {
                    'pooled_branch_b_vs_shared_k64': (
                        candidate_reachability_supplement['pooled_visible'][
                            'branch_b_vs_shared_k64'
                        ]
                    ),
                    'pooled_union_vs_shared_k64': (
                        candidate_reachability_supplement['pooled_visible'][
                            'union_vs_shared_k64'
                        ]
                    ),
                    'novelty_counts_visible': (
                        candidate_reachability_supplement[
                            'novelty_counts_visible'
                        ]
                    ),
                },
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
