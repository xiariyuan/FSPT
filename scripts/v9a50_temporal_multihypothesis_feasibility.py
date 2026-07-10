#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

WORKTREE_ROOT = Path('/gemini/code/FSPT_v9a45_clean')
SOURCE_ROOT = Path('/gemini/code/FSPT')
SOURCE_BASE = SOURCE_ROOT / 'outputs/paper_discovery_2026-07-05'
POOL_PATH = SOURCE_BASE / 'v9a38_pointodyssey_pool/v9a38_pointodyssey_hypothesis_pool.npz'
PREFUSION_PATH = SOURCE_BASE / 'v9a43_prefusion_latents/v9a43_pointodyssey_c1_prefusion_latents.npz'
POD_ROOT = SOURCE_ROOT / 'datasets/pointodyssey/train'
MANIFEST_PATH = WORKTREE_ROOT / 'docs/v9a50_input_manifest_2026-07-10.json'
CHECKPOINT_PATH = SOURCE_ROOT / 'baselines/track_on/checkpoints_trackon2_dinov3.pt'
OUTDIR = WORKTREE_ROOT / 'outputs/paper_discovery_2026-07-05/v9a50_temporal_feasibility'
OUT_JSON = OUTDIR / 'v9a50_temporal_feasibility.json'
OUT_NPZ = OUTDIR / 'v9a50_temporal_feasibility_selections.npz'
OUT_DOC = WORKTREE_ROOT / 'docs/v9a50_temporal_multihypothesis_feasibility_result_2026-07-10.md'

SEQUENCES = ('ani', 'animal3', 'r4_new_f')
NQ = 32
LENGTH = 96
MAX_TEMPORAL_GAP = 8
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 20260710
TOL = 1e-6

SAMPLED_CAUSAL_POLICIES = (
    'causal_teacher_motion',
    'causal_teacher_latent',
    'causal_teacher_motion_latent',
)
ORACLE_STATE_POLICIES = (
    'past_oracle_motion',
    'past_oracle_latent',
    'past_oracle_motion_latent',
    'past_gt_motion',
    'past_gt_motion_plus_teacher',
)
ALL_POLICIES = (
    'teacher',
    *SAMPLED_CAUSAL_POLICIES,
    *ORACLE_STATE_POLICIES,
    'frame_oracle',
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_seed(sequence: str, start: int) -> int:
    digest = hashlib.sha256(f'{sequence}:{start}'.encode()).digest()
    return 20260710 + int.from_bytes(digest[:4], 'little')


def verify_inputs() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest_by_path = {str(Path(row['path']).resolve()): row for row in manifest['files']}
    required = [
        POOL_PATH,
        PREFUSION_PATH,
        CHECKPOINT_PATH,
        POD_ROOT / 'ani/anno.npz',
        POD_ROOT / 'animal3/anno.npz',
        POD_ROOT / 'r4_new_f/anno.npz',
        *[Path(value) for value in manifest['dimension_frames']],
    ]
    checked = []
    for path in required:
        resolved = str(path.resolve())
        if resolved not in manifest_by_path:
            raise RuntimeError(f'required V9-A5.0 input absent from manifest: {resolved}')
        expected = manifest_by_path[resolved]
        if not path.exists():
            raise RuntimeError(f'manifest input missing: {path}')
        size = path.stat().st_size
        digest = sha256(path)
        if size != int(expected['size_bytes']) or digest != expected['sha256']:
            raise RuntimeError(f'input hash/size mismatch: {path}')
        checked.append({'path': resolved, 'size_bytes': int(size), 'sha256': digest})

    head = subprocess.check_output(
        ['git', '-C', str(WORKTREE_ROOT), 'rev-parse', 'HEAD'], text=True
    ).strip()
    branch = subprocess.check_output(
        ['git', '-C', str(WORKTREE_ROOT), 'branch', '--show-current'], text=True
    ).strip()
    tracked_status = subprocess.check_output(
        ['git', '-C', str(WORKTREE_ROOT), 'status', '--porcelain', '--untracked-files=no'], text=True
    ).strip()
    if head != manifest['clean_head']:
        raise RuntimeError(f'HEAD mismatch: {head} != {manifest["clean_head"]}')
    if branch != manifest['clean_branch']:
        raise RuntimeError(f'branch mismatch: {branch} != {manifest["clean_branch"]}')
    if tracked_status:
        raise RuntimeError(f'tracked worktree is not clean: {tracked_status}')

    dimension_frames = []
    for value in manifest['dimension_frames']:
        path = Path(value)
        width, height = Image.open(path).size
        sequence = path.parts[-3]
        frame_id = int(path.stem.split('_')[-1])
        dimension_frames.append(
            {
                'sequence': sequence,
                'frame_id': frame_id,
                'path': str(path),
                'size_bytes': int(path.stat().st_size),
                'sha256': sha256(path),
                'width': int(width),
                'height': int(height),
            }
        )
    return {
        'pass': True,
        'head': head,
        'branch': branch,
        'tracked_status': tracked_status,
        'checked_inputs': checked,
        'dimension_frames': dimension_frames,
        'script': {'path': str(Path(__file__).resolve()), 'sha256': sha256(Path(__file__).resolve())},
    }

def build_pointodyssey_gt(pool: Any) -> tuple[np.ndarray, dict[str, Any]]:
    gt = np.zeros((len(pool['clip_id']), 2), dtype=np.float32)
    clip_stats = []
    for clip_id in sorted(set(pool['clip_id'].tolist())):
        sequence, start_s = str(clip_id).split(':')
        start = int(start_s)
        sequence_dir = POD_ROOT / sequence
        annotation = np.load(sequence_dir / 'anno.npz', allow_pickle=True, mmap_mode='r')
        trajectories = np.asarray(annotation['trajs_2d'][start:start + LENGTH], dtype=np.float32)
        visibility = np.asarray(annotation['visibs'][start:start + LENGTH], dtype=bool)
        valid = (
            np.asarray(annotation['valids'][start:start + LENGTH], dtype=bool)
            if 'valids' in annotation.files
            else np.ones_like(visibility, dtype=bool)
        )
        eligible = np.where(
            visibility[0]
            & valid[0]
            & (np.sum(visibility & valid, axis=0) >= 24)
        )[0]
        rng = np.random.default_rng(stable_seed(sequence, start))
        chosen = np.sort(rng.choice(eligible, NQ, replace=False))
        width, height = Image.open(sequence_dir / 'rgbs' / f'rgb_{start:05d}.jpg').size
        rows = np.where(pool['clip_id'] == clip_id)[0]
        for row in rows:
            tau = int(pool['frame_tau'][row])
            query_idx = int(pool['query_idx'][row])
            xy = trajectories[tau, chosen[query_idx]]
            gt[row] = [xy[1] / float(height - 1), xy[0] / float(width - 1)]
        clip_stats.append(
            {
                'clip_id': str(clip_id),
                'rows': int(len(rows)),
                'eligible_tracks': int(len(eligible)),
                'selected_tracks': int(len(chosen)),
                'width': int(width),
                'height': int(height),
            }
        )
    reproduced = np.linalg.norm(
        (pool['candidates'].astype(np.float32) - gt[:, None, :]) * 255.0,
        axis=-1,
    )
    difference = np.abs(reproduced - pool['error_px'].astype(np.float32))
    max_abs = float(np.max(difference))
    if max_abs > 1e-4:
        raise RuntimeError(f'GT reconstruction error reproduction failed: {max_abs}')
    return gt, {
        'max_abs_error_reproduction': max_abs,
        'mean_abs_error_reproduction': float(np.mean(difference)),
        'clip_stats': clip_stats,
    }


def audit_candidate_latent_alignment(pool: Any, prefusion: Any) -> dict[str, Any]:
    for key in ['clip_id', 'sequence', 'frame_tau', 'query_idx']:
        if not np.array_equal(pool[key], prefusion[key]):
            raise RuntimeError(f'pool/prefusion alignment mismatch: {key}')
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    weight = checkpoint['reranking_head.score_layer.weight'].detach().cpu().numpy().astype(np.float32)
    bias = checkpoint['reranking_head.score_layer.bias'].detach().cpu().numpy().astype(np.float32)
    postfusion = prefusion['c1_topk_latent'].astype(np.float32)
    reconstructed = np.einsum('nkd,od->nko', postfusion, weight) + bias.reshape(1, 1, -1)
    reconstructed = reconstructed[..., 0]
    feature_names = [str(value) for value in pool['local_feature_names'].tolist()]
    score_index = feature_names.index('exact_rerank_s')
    reference = pool['Xlocal'][:, 2:18, score_index].astype(np.float32)
    difference = np.abs(reconstructed - reference)
    reconstructed_top1 = np.argmax(reconstructed, axis=1)
    reference_top1 = np.argmax(reference, axis=1)
    top1_match_rate = float(np.mean(reconstructed_top1 == reference_top1))
    max_abs = float(np.max(difference))
    if top1_match_rate != 1.0:
        raise RuntimeError(f'candidate/latent top1 alignment failed: {top1_match_rate}')
    if max_abs > 0.005:
        raise RuntimeError(f'candidate/latent score reconstruction failed: {max_abs}')
    prefusion_latent = prefusion['c1_prefusion_latent'].astype(np.float32)
    prefusion_norm = np.linalg.norm(prefusion_latent, axis=2)
    return {
        'rows': int(len(reference)),
        'candidate_count': int(reference.shape[1]),
        'postfusion_score_max_abs': max_abs,
        'postfusion_score_mean_abs': float(np.mean(difference)),
        'postfusion_score_p99_abs': float(np.quantile(difference, 0.99)),
        'top1_match_rate': top1_match_rate,
        'top1_mismatch_rows': int(np.sum(reconstructed_top1 != reference_top1)),
        'prefusion_latent_norm_min': float(np.min(prefusion_norm)),
        'prefusion_latent_norm_p01': float(np.quantile(prefusion_norm, 0.01)),
        'prefusion_latent_norm_mean': float(np.mean(prefusion_norm)),
        'finite': bool(
            np.all(np.isfinite(reconstructed))
            and np.all(np.isfinite(prefusion_latent))
        ),
    }


def ordinal_rank(values: np.ndarray, descending: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    key = -values if descending else values
    order = np.argsort(key, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values), dtype=np.int64)
    return ranks


def choose_borda(teacher_rank: np.ndarray, *extra_ranks: np.ndarray) -> int:
    total = teacher_rank.astype(np.int64).copy()
    for rank in extra_ranks:
        total += np.asarray(rank, dtype=np.int64)
    index = np.arange(len(total), dtype=np.int64)
    order = np.lexsort((index, teacher_rank, total))
    return int(order[0])


def predict_motion(history: list[dict[str, Any]], current_t: int) -> np.ndarray:
    last = history[-1]
    if len(history) < 2:
        return np.asarray(last['coord'], dtype=np.float32)
    previous = history[-2]
    dt = max(1, int(last['t']) - int(previous['t']))
    velocity = (np.asarray(last['coord']) - np.asarray(previous['coord'])) / float(dt)
    horizon = int(current_t) - int(last['t'])
    return (np.asarray(last['coord']) + velocity * float(horizon)).astype(np.float32)


def motion_rank(candidates: np.ndarray, history: list[dict[str, Any]], current_t: int) -> np.ndarray:
    prediction = predict_motion(history, current_t)
    distance = np.linalg.norm((candidates - prediction[None, :]) * 255.0, axis=1)
    return ordinal_rank(distance, descending=False)


def latent_rank(current_latent: np.ndarray, previous_latent: np.ndarray) -> np.ndarray:
    similarity = current_latent @ np.asarray(previous_latent, dtype=np.float32)
    return ordinal_rank(similarity, descending=True)


def append_state(
    history: list[dict[str, Any]],
    t: int,
    index: int,
    candidates: np.ndarray,
    latents: np.ndarray,
) -> None:
    history.append(
        {
            't': int(t),
            'index': int(index),
            'coord': np.asarray(candidates[index], dtype=np.float32),
            'latent': np.asarray(latents[index], dtype=np.float32),
        }
    )
    if len(history) > 2:
        del history[:-2]


def append_gt_state(history: list[dict[str, Any]], t: int, coord: np.ndarray) -> None:
    history.append({'t': int(t), 'coord': np.asarray(coord, dtype=np.float32)})
    if len(history) > 2:
        del history[:-2]


def simulate_policies(
    pool: Any,
    candidates: np.ndarray,
    errors: np.ndarray,
    teacher_scores: np.ndarray,
    latents: np.ndarray,
    gt: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n_rows, n_candidates = errors.shape
    if n_candidates != 16:
        raise RuntimeError(f'expected 16 C1 candidates, got {n_candidates}')
    teacher_index = np.argmax(teacher_scores, axis=1).astype(np.int64)
    oracle_index = np.argmin(errors, axis=1).astype(np.int64)
    selected = {name: np.full(n_rows, -1, dtype=np.int64) for name in ALL_POLICIES}
    gap = np.full(n_rows, -1, dtype=np.int32)
    temporal_eligible = np.zeros(n_rows, dtype=bool)
    first_row = np.zeros(n_rows, dtype=bool)

    tracks: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row, (clip_id, query_idx) in enumerate(zip(pool['clip_id'], pool['query_idx'])):
        tracks[(str(clip_id), int(query_idx))].append(int(row))

    for key, rows in tracks.items():
        rows = sorted(rows, key=lambda row: int(pool['frame_tau'][row]))
        times = [int(pool['frame_tau'][row]) for row in rows]
        if len(times) != len(set(times)):
            raise RuntimeError(f'duplicate frame_tau within track {key}')
        self_histories = {name: [] for name in SAMPLED_CAUSAL_POLICIES}
        oracle_history: list[dict[str, Any]] = []
        gt_history: list[dict[str, Any]] = []
        previous_t = None
        for position, row in enumerate(rows):
            t = int(pool['frame_tau'][row])
            c = candidates[row]
            e = errors[row]
            score = teacher_scores[row]
            latent = latents[row]
            trank = ordinal_rank(score, descending=True)
            teacher = int(teacher_index[row])
            oracle = int(oracle_index[row])
            selected['teacher'][row] = teacher
            selected['frame_oracle'][row] = oracle

            if previous_t is None:
                first_row[row] = True
                current_gap = -1
            else:
                current_gap = t - previous_t
                if current_gap <= 0:
                    raise RuntimeError(f'non-increasing time in track {key}')
                gap[row] = current_gap
                temporal_eligible[row] = current_gap <= MAX_TEMPORAL_GAP
                if current_gap > MAX_TEMPORAL_GAP:
                    # A long observation gap starts a new causal segment. Do not let
                    # a stale second-last state leak into the next velocity estimate.
                    for history in self_histories.values():
                        history.clear()
                    oracle_history.clear()
                    gt_history.clear()

            for policy in SAMPLED_CAUSAL_POLICIES:
                history = self_histories[policy]
                if not history or not temporal_eligible[row]:
                    index = teacher
                else:
                    mrank = motion_rank(c, history, t)
                    lrank = latent_rank(latent, history[-1]['latent'])
                    if policy == 'causal_teacher_motion':
                        index = choose_borda(trank, mrank)
                    elif policy == 'causal_teacher_latent':
                        index = choose_borda(trank, lrank)
                    elif policy == 'causal_teacher_motion_latent':
                        index = choose_borda(trank, mrank, lrank)
                    else:
                        raise RuntimeError(policy)
                selected[policy][row] = index
                append_state(history, t, index, c, latent)

            if oracle_history and temporal_eligible[row]:
                oracle_mrank = motion_rank(c, oracle_history, t)
                oracle_lrank = latent_rank(latent, oracle_history[-1]['latent'])
                selected['past_oracle_motion'][row] = choose_borda(trank, oracle_mrank)
                selected['past_oracle_latent'][row] = choose_borda(trank, oracle_lrank)
                selected['past_oracle_motion_latent'][row] = choose_borda(
                    trank, oracle_mrank, oracle_lrank
                )
            else:
                selected['past_oracle_motion'][row] = teacher
                selected['past_oracle_latent'][row] = teacher
                selected['past_oracle_motion_latent'][row] = teacher

            if gt_history and temporal_eligible[row]:
                prediction = predict_motion(gt_history, t)
                distance = np.linalg.norm((c - prediction[None, :]) * 255.0, axis=1)
                grank = ordinal_rank(distance, descending=False)
                selected['past_gt_motion'][row] = int(np.argmin(distance))
                selected['past_gt_motion_plus_teacher'][row] = choose_borda(trank, grank)
            else:
                selected['past_gt_motion'][row] = teacher
                selected['past_gt_motion_plus_teacher'][row] = teacher

            append_state(oracle_history, t, oracle, c, latent)
            append_gt_state(gt_history, t, gt[row])
            previous_t = t

    for name, values in selected.items():
        if np.any(values < 0) or np.any(values >= n_candidates):
            raise RuntimeError(f'invalid selections for {name}')
    audit = {
        'tracks': int(len(tracks)),
        'first_rows': int(np.sum(first_row)),
        'temporal_eligible_rows': int(np.sum(temporal_eligible)),
        'gap_gt8_rows': int(np.sum(gap > MAX_TEMPORAL_GAP)),
        'gap_histogram': {
            str(value): int(count)
            for value, count in zip(*np.unique(gap[gap > 0], return_counts=True))
        },
    }
    return selected, {
        'teacher_index': teacher_index,
        'oracle_index': oracle_index,
        'gap': gap,
        'temporal_eligible': temporal_eligible,
        'first_row': first_row,
        'track_rows': {f'{key[0]}::{key[1]}': rows for key, rows in tracks.items()},
        'audit': audit,
    }


def summarize(
    chosen_error: np.ndarray,
    teacher_error: np.ndarray,
    oracle_error: np.ndarray,
    chosen_index: np.ndarray,
    oracle_index: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    chosen = chosen_error[mask]
    teacher = teacher_error[mask]
    oracle = oracle_error[mask]
    index = chosen_index[mask]
    oindex = oracle_index[mask]
    difference = chosen - teacher
    better = int(np.sum(difference < -TOL))
    worse = int(np.sum(difference > TOL))
    equal = int(len(difference) - better - worse)
    return {
        'n': int(len(chosen)),
        'mean_error': float(np.mean(chosen)),
        'median_error': float(np.median(chosen)),
        'safe4': int(np.sum(chosen <= 4.0)),
        'safe8': int(np.sum(chosen <= 8.0)),
        'safe16': int(np.sum(chosen <= 16.0)),
        'oracle_regret': float(np.mean(chosen - oracle)),
        'exact_best_rate': float(np.mean(index == oindex)),
        'within1px_best_rate': float(np.mean(chosen <= oracle + 1.0)),
        'mean_difference_vs_teacher': float(np.mean(difference)),
        'median_difference_vs_teacher': float(np.median(difference)),
        'better': better,
        'worse': worse,
        'equal': equal,
    }


def build_metrics(
    pool: Any,
    errors: np.ndarray,
    selections: dict[str, np.ndarray],
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    row = np.arange(len(errors))
    teacher_index = state['teacher_index']
    oracle_index = state['oracle_index']
    teacher_error = errors[row, teacher_index]
    oracle_error = errors[row, oracle_index]
    sequence = np.asarray(pool['sequence']).astype(str)
    hard = np.asarray(pool['is_hard']).astype(bool)
    gap = state['gap']
    masks = {
        'all': np.ones(len(errors), dtype=bool),
        'hard': hard,
        'easy': ~hard,
        'first': state['first_row'],
        'temporal_eligible': state['temporal_eligible'],
        'gap_1': gap == 1,
        'gap_2_4': (gap >= 2) & (gap <= 4),
        'gap_5_8': (gap >= 5) & (gap <= 8),
        'gap_gt8': gap > 8,
    }
    for seq in SEQUENCES:
        masks[f'sequence_{seq}'] = sequence == seq

    metrics = {}
    chosen_errors = {}
    for policy, index in selections.items():
        chosen = errors[row, index]
        chosen_errors[policy] = chosen
        metrics[policy] = {
            name: summarize(chosen, teacher_error, oracle_error, index, oracle_index, mask)
            for name, mask in masks.items()
        }
    return metrics, chosen_errors


def cluster_bootstrap(
    pool: Any,
    chosen_errors: dict[str, np.ndarray],
    teacher_error: np.ndarray,
    cluster_unit: str,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    cluster_rows: dict[Any, list[int]] = defaultdict(list)
    if cluster_unit == 'track':
        for row, (clip_id, query_idx) in enumerate(zip(pool['clip_id'], pool['query_idx'])):
            cluster_rows[(str(clip_id), int(query_idx))].append(int(row))
        seed = BOOTSTRAP_SEED
    elif cluster_unit == 'clip':
        for row, clip_id in enumerate(pool['clip_id']):
            cluster_rows[str(clip_id)].append(int(row))
        seed = BOOTSTRAP_SEED + 1
    else:
        raise ValueError(f'unknown cluster unit: {cluster_unit}')
    cluster_keys = sorted(cluster_rows, key=str)
    policies = [name for name in ALL_POLICIES if name != 'teacher']
    matrix = np.zeros((len(policies), len(cluster_keys)), dtype=np.float64)
    per_cluster: dict[str, dict[str, float]] = {name: {} for name in policies}
    for pi, policy in enumerate(policies):
        difference = chosen_errors[policy] - teacher_error
        for ci, key in enumerate(cluster_keys):
            value = float(np.mean(difference[cluster_rows[key]]))
            matrix[pi, ci] = value
            if isinstance(key, tuple):
                key_text = f'{key[0]}::{key[1]}'
            else:
                key_text = str(key)
            per_cluster[policy][key_text] = value

    rng = np.random.default_rng(seed)
    bootstrap = np.empty((len(policies), BOOTSTRAP_RESAMPLES), dtype=np.float64)
    chunk = 1000
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        count = min(chunk, BOOTSTRAP_RESAMPLES - start)
        indices = rng.integers(0, len(cluster_keys), size=(count, len(cluster_keys)))
        sampled = matrix[:, indices]
        bootstrap[:, start:start + count] = np.mean(sampled, axis=2)
    report = {}
    for pi, policy in enumerate(policies):
        values = matrix[pi]
        boot = bootstrap[pi]
        report[policy] = {
            'cluster_unit': cluster_unit,
            'n_clusters': int(len(cluster_keys)),
            'cluster_mean_difference': float(np.mean(values)),
            'cluster_median_difference': float(np.median(values)),
            'cluster_better': int(np.sum(values < -TOL)),
            'cluster_worse': int(np.sum(values > TOL)),
            'cluster_equal': int(np.sum(np.abs(values) <= TOL)),
            'bootstrap_resamples': int(BOOTSTRAP_RESAMPLES),
            'bootstrap_seed': int(seed),
            'mean_difference_95_ci': [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            'probability_mean_lt_zero': float(np.mean(boot < 0.0)),
        }
    return report, per_cluster

def evaluate_gates(metrics: dict[str, Any], clip_bootstrap: dict[str, Any]) -> dict[str, Any]:
    teacher = metrics['teacher']
    sampled_causal = {}
    for policy in SAMPLED_CAUSAL_POLICIES:
        sequence_mean_improved = {
            seq: bool(
                metrics[policy][f'sequence_{seq}']['mean_error']
                < teacher[f'sequence_{seq}']['mean_error']
            )
            for seq in SEQUENCES
        }
        sequence_safe16 = {
            seq: bool(
                metrics[policy][f'sequence_{seq}']['safe16']
                >= teacher[f'sequence_{seq}']['safe16']
            )
            for seq in SEQUENCES
        }
        global_better = metrics[policy]['all']['better'] > metrics[policy]['all']['worse']
        ci_upper_negative = clip_bootstrap[policy]['mean_difference_95_ci'][1] < 0.0
        gate = {
            'sequence_mean_improved': sequence_mean_improved,
            'sequence_safe16_not_decreased': sequence_safe16,
            'global_better_gt_worse': bool(global_better),
            'clip_bootstrap_ci_upper_lt_zero': bool(ci_upper_negative),
        }
        gate['pass_all'] = bool(
            all(sequence_mean_improved.values())
            and all(sequence_safe16.values())
            and global_better
            and ci_upper_negative
        )
        sampled_causal[policy] = gate

    oracle_state = {}
    for policy in ORACLE_STATE_POLICIES:
        improvements = {
            seq: float(
                teacher[f'sequence_{seq}']['mean_error']
                - metrics[policy][f'sequence_{seq}']['mean_error']
            )
            for seq in SEQUENCES
        }
        global_better = metrics[policy]['all']['better'] > metrics[policy]['all']['worse']
        gate = {
            'sequence_mean_improvement_px': improvements,
            'each_sequence_improves_at_least_0.25px': bool(
                all(value >= 0.25 for value in improvements.values())
            ),
            'global_better_gt_worse': bool(global_better),
        }
        gate['pass_headroom'] = bool(
            gate['each_sequence_improves_at_least_0.25px'] and global_better
        )
        oracle_state[policy] = gate

    passing_sampled = [name for name, gate in sampled_causal.items() if gate['pass_all']]
    passing_oracle = [name for name, gate in oracle_state.items() if gate['pass_headroom']]
    best_sampled = min(
        SAMPLED_CAUSAL_POLICIES,
        key=lambda name: metrics[name]['all']['mean_error'],
    )
    best_oracle = min(
        ORACLE_STATE_POLICIES,
        key=lambda name: metrics[name]['all']['mean_error'],
    )
    if passing_sampled:
        decision = (
            'SAMPLED_CAUSAL_TEMPORAL_PASS: at least one deterministic self-state policy passes '
            'all three sequence gates and the conservative clip-block bootstrap gate on the '
            'balanced sampled pool. Because intermediate online frames are absent, the next step '
            'is a full-stream deterministic causal audit, not a learned belief head and not DAVIS.'
        )
    elif passing_oracle:
        decision = (
            'ORACLE_STATE_HEADROOM_ONLY: sampled self-state policies fail, but causal past-oracle '
            'or past-GT state improves every sequence by at least 0.25 px. Temporal information '
            'has headroom while state-error propagation is the blocker. Next build an explicit '
            'multi-hypothesis/beam full-stream audit; do not train another single-state reranker.'
        )
    else:
        decision = (
            'TEMPORAL_FEASIBILITY_FAIL: neither sampled-causal self-state nor causal past-oracle/'
            'past-GT temporal policies show consistent headroom. Stop temporal reranking and move '
            'to candidate-generation/correlation supervision.'
        )
    return {
        'sampled_causal': sampled_causal,
        'oracle_state': oracle_state,
        'passing_sampled_causal': passing_sampled,
        'passing_oracle_state': passing_oracle,
        'best_sampled_causal_by_global_mean': best_sampled,
        'best_oracle_state_by_global_mean': best_oracle,
        'decision': decision,
    }

def fmt(value: float, digits: int = 4, signed: bool = False) -> str:
    return f'{value:+.{digits}f}' if signed else f'{value:.{digits}f}'


def write_doc(result: dict[str, Any]) -> None:
    metrics = result['metrics']
    track_bootstrap = result['track_bootstrap']
    clip_bootstrap = result['clip_bootstrap']
    lines = [
        '# V9-A5.0 Sampled-Causal Temporal State Feasibility Audit Result',
        '',
        'Date: 2026-07-10',
        '',
        'No training and no DAVIS data were used. State updates occur only on the balanced sampled PointOdyssey pool rows, not on every online frame. Temporal evidence is enabled only for observation gaps <= 8.',
        '',
        '## Global results',
        '',
        '| Policy | Mean | Median | Safe4/8/16 | Oracle regret | Exact best | Better/Worse/Equal | Mean Δ vs teacher | Track 95% CI | Clip 95% CI |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|---|',
    ]
    for policy in ALL_POLICIES:
        row = metrics[policy]['all']
        if policy == 'teacher':
            track_interval = [0.0, 0.0]
            clip_interval = [0.0, 0.0]
        else:
            track_interval = track_bootstrap[policy]['mean_difference_95_ci']
            clip_interval = clip_bootstrap[policy]['mean_difference_95_ci']
        lines.append(
            f"| {policy} | {row['mean_error']:.4f} | {row['median_error']:.4f} | "
            f"{row['safe4']}/{row['safe8']}/{row['safe16']} | {row['oracle_regret']:.4f} | "
            f"{row['exact_best_rate']:.4f} | {row['better']}/{row['worse']}/{row['equal']} | "
            f"{row['mean_difference_vs_teacher']:+.4f} | "
            f"[{track_interval[0]:+.4f},{track_interval[1]:+.4f}] | "
            f"[{clip_interval[0]:+.4f},{clip_interval[1]:+.4f}] |"
        )

    lines += [
        '',
        '## Per-sequence mean error',
        '',
        '| Policy | ani | animal3 | r4_new_f |',
        '|---|---:|---:|---:|',
    ]
    for policy in ALL_POLICIES:
        lines.append(
            f"| {policy} | "
            f"{metrics[policy]['sequence_ani']['mean_error']:.4f} | "
            f"{metrics[policy]['sequence_animal3']['mean_error']:.4f} | "
            f"{metrics[policy]['sequence_r4_new_f']['mean_error']:.4f} |"
        )

    lines += [
        '',
        '## Temporal-eligible subset',
        '',
        '| Policy | N | Mean | Better/Worse/Equal | Mean Δ |',
        '|---|---:|---:|---:|---:|',
    ]
    for policy in ALL_POLICIES:
        row = metrics[policy]['temporal_eligible']
        lines.append(
            f"| {policy} | {row['n']} | {row['mean_error']:.4f} | "
            f"{row['better']}/{row['worse']}/{row['equal']} | {row['mean_difference_vs_teacher']:+.4f} |"
        )

    lines += [
        '',
        '## Gates',
        '',
        '### Sampled-causal self-state policies',
        '',
        '```json',
        json.dumps(result['gates']['sampled_causal'], indent=2),
        '```',
        '',
        '### Causal past-oracle / past-GT state headroom',
        '',
        '```json',
        json.dumps(result['gates']['oracle_state'], indent=2),
        '```',
        '',
        '## Decision',
        '',
        result['gates']['decision'],
        '',
        '## Integrity and audit counts',
        '',
        '```json',
        json.dumps(
            {
                'input_audit': result['input_audit'],
                'candidate_latent_alignment': result['candidate_latent_alignment'],
                'gt_audit': result['gt_audit'],
                'simulation_audit': result['simulation_audit'],
            },
            indent=2,
        ),
        '```',
    ]
    OUT_DOC.write_text('\n'.join(lines) + '\n')

def main() -> None:
    input_audit = verify_inputs()
    pool = np.load(POOL_PATH, allow_pickle=True)
    prefusion = np.load(PREFUSION_PATH, allow_pickle=True)
    if pool['Xlocal'].shape[:2] != (len(pool['clip_id']), 34):
        raise RuntimeError('unexpected Xlocal shape')
    if prefusion['c1_prefusion_latent'].shape[:2] != (len(pool['clip_id']), 16):
        raise RuntimeError('unexpected C1 prefusion latent shape')

    candidate_latent_alignment = audit_candidate_latent_alignment(pool, prefusion)
    feature_names = [str(value) for value in pool['local_feature_names'].tolist()]
    score_index = feature_names.index('exact_rerank_s')
    candidates = pool['candidates'][:, 2:18].astype(np.float32)
    errors = pool['error_px'][:, 2:18].astype(np.float32)
    teacher_scores = pool['Xlocal'][:, 2:18, score_index].astype(np.float32)
    raw_latents = prefusion['c1_prefusion_latent'].astype(np.float32)
    latent_norm = np.linalg.norm(raw_latents, axis=2, keepdims=True)
    if float(np.min(latent_norm)) <= 1e-8:
        raise RuntimeError('near-zero C1 prefusion latent norm')
    latents = raw_latents / latent_norm
    if not all(
        np.all(np.isfinite(value))
        for value in [candidates, errors, teacher_scores, latents]
    ):
        raise RuntimeError('non-finite audit input')

    gt, gt_audit = build_pointodyssey_gt(pool)
    selections, state = simulate_policies(
        pool, candidates, errors, teacher_scores, latents, gt
    )
    metrics, chosen_errors = build_metrics(pool, errors, selections, state)
    row = np.arange(len(errors))
    teacher_error = errors[row, state['teacher_index']]
    track_bootstrap, per_track = cluster_bootstrap(
        pool, chosen_errors, teacher_error, cluster_unit='track'
    )
    clip_bootstrap, per_clip = cluster_bootstrap(
        pool, chosen_errors, teacher_error, cluster_unit='clip'
    )
    gates = evaluate_gates(metrics, clip_bootstrap)

    result = {
        'script': 'scripts/v9a50_temporal_multihypothesis_feasibility.py',
        'date': '2026-07-10',
        'provenance': {
            'script_sha256': sha256(Path(__file__).resolve()),
            'branch': input_audit['branch'],
            'head': input_audit['head'],
            'worktree_root': str(WORKTREE_ROOT),
            'source_asset_root': str(SOURCE_ROOT),
        },
        'protocol': {
            'dataset': 'PointOdyssey canonical balanced sampled pool',
            'rows': int(len(errors)),
            'tracks': int(state['audit']['tracks']),
            'clips': int(len(set(pool['clip_id'].tolist()))),
            'sequences': list(SEQUENCES),
            'max_temporal_gap': int(MAX_TEMPORAL_GAP),
            'state_observation_stream': (
                'sampled pool observations only; intermediate online frames are absent'
            ),
            'training': False,
            'davis_read': False,
            'threshold_tuning': False,
            'rank_fusion': 'unweighted ordinal Borda with teacher-rank tie-break',
            'bootstrap_units': ['clip_id/query_idx track', 'clip_id block'],
            'bootstrap_resamples': int(BOOTSTRAP_RESAMPLES),
            'track_bootstrap_seed': int(BOOTSTRAP_SEED),
            'clip_bootstrap_seed': int(BOOTSTRAP_SEED + 1),
        },
        'input_audit': input_audit,
        'candidate_latent_alignment': candidate_latent_alignment,
        'gt_audit': gt_audit,
        'simulation_audit': state['audit'],
        'metrics': metrics,
        'track_bootstrap': track_bootstrap,
        'clip_bootstrap': clip_bootstrap,
        'per_track_mean_difference': per_track,
        'per_clip_mean_difference': per_clip,
        'gates': gates,
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    np.savez_compressed(
        OUT_NPZ,
        row_key=np.asarray(
            [
                json.dumps([str(c), int(q), int(t)])
                for c, q, t in zip(
                    pool['clip_id'], pool['query_idx'], pool['frame_tau']
                )
            ],
            dtype=object,
        ),
        **{f'selection_{name}': values for name, values in selections.items()},
        gap=state['gap'],
        temporal_eligible=state['temporal_eligible'].astype(np.int8),
    )
    write_doc(result)
    print(
        json.dumps(
            {
                'ok': True,
                'json': str(OUT_JSON),
                'npz': str(OUT_NPZ),
                'doc': str(OUT_DOC),
                'decision': gates['decision'],
                'passing_sampled_causal': gates['passing_sampled_causal'],
                'passing_oracle_state': gates['passing_oracle_state'],
                'best_sampled_causal': gates['best_sampled_causal_by_global_mean'],
                'best_oracle_state': gates['best_oracle_state_by_global_mean'],
                'candidate_latent_top1_match_rate': (
                    candidate_latent_alignment['top1_match_rate']
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == '__main__':
    main()
