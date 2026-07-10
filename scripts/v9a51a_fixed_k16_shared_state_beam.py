#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

WORKTREE_ROOT = Path('/gemini/code/FSPT_v9a51_clean')
SOURCE_ROOT = Path('/gemini/code/FSPT')
CLEAN_TRACKON_ROOT = WORKTREE_ROOT / 'baselines/track_on'
SOURCE_BASE = SOURCE_ROOT / 'outputs/paper_discovery_2026-07-05'
POOL_PATH = SOURCE_BASE / 'v9a38_pointodyssey_pool/v9a38_pointodyssey_hypothesis_pool.npz'
PREFUSION_PATH = SOURCE_BASE / 'v9a43_prefusion_latents/v9a43_pointodyssey_c1_prefusion_latents.npz'
CHECKPOINT = SOURCE_ROOT / 'baselines/track_on/checkpoints_trackon2_dinov3.pt'
CONFIG = CLEAN_TRACKON_ROOT / 'config/test.yaml'
POD_ROOT = SOURCE_ROOT / 'datasets/pointodyssey/train'
MANIFEST = WORKTREE_ROOT / 'docs/v9a51_input_manifest_2026-07-10.json'
OUTDIR = WORKTREE_ROOT / 'outputs/paper_discovery_2026-07-05/v9a51_fullstream_beam'
OUT_JSON = OUTDIR / 'v9a51_fullstream_beam_reachability.json'
OUT_NPZ = OUTDIR / 'v9a51_fullstream_beam_reachability.npz'
OUT_DOC = WORKTREE_ROOT / 'docs/v9a51_fullstream_beam_reachability_result_2026-07-10.md'
EXPECTED_HEAD = '7c24367cf8aeb1467946d2ced3b294299032538e'
EXPECTED_BRANCH = 'v9a51-fullstream-beam-20260710'

SEQUENCES = ('ani', 'animal3', 'r4_new_f')
CLIP_STARTS = (0, 256, 512)
NQ = 32
LENGTH = 96
K = 16
TOL = 1e-6
RADII = (1.0, 2.0, 4.0, 8.0)
BOOTSTRAP_RESAMPLES = 100_000
TRACK_BOOTSTRAP_SEED = 20260710
CLIP_BOOTSTRAP_SEED = 20260711

BEAM_CONFIGS = {
    'beam1_motion_latent': {'width': 1, 'motion': True, 'latent': True},
    'beam4_motion': {'width': 4, 'motion': True, 'latent': False},
    'beam4_latent': {'width': 4, 'motion': False, 'latent': True},
    'beam4_motion_latent': {'width': 4, 'motion': True, 'latent': True},
    'beam8_motion_latent': {'width': 8, 'motion': True, 'latent': True},
}
PRIMARY_BEAM = 'beam4_motion_latent'

if str(CLEAN_TRACKON_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_TRACKON_ROOT))
os.environ['DINOV3_LOCAL_DIR'] = str(
    SOURCE_ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m'
)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from model.trackon_predictor import Predictor  # noqa: E402
from utils.coord_utils import get_points_on_a_grid, indices_to_coords  # noqa: E402
from utils.train_utils import load_args_from_yaml  # noqa: E402


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


def verify_inputs(verify_frames: bool) -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text())
    by_path = {str(Path(row['path']).resolve()): row for row in manifest['files']}
    required = [
        POOL_PATH,
        PREFUSION_PATH,
        WORKTREE_ROOT / 'outputs/paper_discovery_2026-07-05/v9a50_temporal_feasibility/v9a50_temporal_feasibility.json',
        CHECKPOINT,
        SOURCE_ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m/model.safetensors',
        SOURCE_ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m/config.json',
        CONFIG,
        POD_ROOT / 'ani/anno.npz',
        POD_ROOT / 'animal3/anno.npz',
        POD_ROOT / 'r4_new_f/anno.npz',
        Path(manifest['frame_manifest']['path']),
    ]
    checked = []
    for path in required:
        resolved = str(path.resolve())
        if resolved not in by_path:
            raise RuntimeError(f'required input absent from manifest: {resolved}')
        expected = by_path[resolved]
        if not path.exists():
            raise RuntimeError(f'missing input: {path}')
        size = path.stat().st_size
        digest = sha256(path)
        if size != int(expected['size_bytes']) or digest != expected['sha256']:
            raise RuntimeError(f'input hash/size mismatch: {path}')
        checked.append({'path': resolved, 'size_bytes': int(size), 'sha256': digest})

    frame_report = None
    if verify_frames:
        meta = manifest['frame_manifest']
        frame_path = Path(meta['path'])
        frame_manifest = json.loads(frame_path.read_text())
        root = Path(frame_manifest['root'])
        aggregate = hashlib.sha256()
        total = 0
        for row in frame_manifest['files']:
            relative = str(row['relative_path'])
            path = root / relative
            size = path.stat().st_size
            digest = sha256(path)
            if size != int(row['size_bytes']) or digest != row['sha256']:
                raise RuntimeError(f'frame hash/size mismatch: {path}')
            total += size
            aggregate.update(relative.encode())
            aggregate.update(b'\0')
            aggregate.update(str(size).encode())
            aggregate.update(b'\0')
            aggregate.update(digest.encode())
            aggregate.update(b'\n')
        aggregate_digest = aggregate.hexdigest()
        if len(frame_manifest['files']) != int(meta['file_count']):
            raise RuntimeError('frame count mismatch')
        if total != int(meta['total_bytes']):
            raise RuntimeError('frame byte-count mismatch')
        if aggregate_digest != meta['aggregate_sha256']:
            raise RuntimeError('frame aggregate mismatch')
        frame_report = {
            'path': str(frame_path),
            'file_count': int(len(frame_manifest['files'])),
            'total_bytes': int(total),
            'aggregate_sha256': aggregate_digest,
        }

    head = subprocess.check_output(
        ['git', '-C', str(WORKTREE_ROOT), 'rev-parse', 'HEAD'], text=True
    ).strip()
    branch = subprocess.check_output(
        ['git', '-C', str(WORKTREE_ROOT), 'branch', '--show-current'], text=True
    ).strip()
    tracked = subprocess.check_output(
        ['git', '-C', str(WORKTREE_ROOT), 'status', '--porcelain', '--untracked-files=no'], text=True
    ).strip()
    if head != EXPECTED_HEAD:
        raise RuntimeError(f'HEAD mismatch: {head} != {EXPECTED_HEAD}')
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f'branch mismatch: {branch} != {EXPECTED_BRANCH}')
    if tracked:
        raise RuntimeError(f'tracked worktree is dirty: {tracked}')

    imported = {}
    for name in ['model.trackon_predictor', 'model.trackon', 'model.reranking', 'utils.coord_utils']:
        module = sys.modules.get(name)
        value = '' if module is None else str(Path(module.__file__).resolve())
        imported[name] = value
        if not value.startswith(str(CLEAN_TRACKON_ROOT.resolve()) + '/'):
            raise RuntimeError(f'non-clean TrackOn import: {name} -> {value}')
    return {
        'pass': True,
        'head': head,
        'branch': branch,
        'tracked_status': tracked,
        'checked_inputs': checked,
        'frame_manifest': frame_report,
        'imported_trackon_code_paths': imported,
        'script': {'path': str(Path(__file__).resolve()), 'sha256': sha256(Path(__file__).resolve())},
    }


def load_frame(path: Path, device: torch.device) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().unsqueeze(0)
    tensor = F.interpolate(tensor, size=(256, 256), mode='bilinear', align_corners=False)
    return tensor.to(device)


@torch.no_grad()
def track_frame_diag(model, q_init, temporal_mask, point_memory, frame_features, h_in: int, w_in: int):
    n, d = q_init.shape
    memory_size = point_memory.shape[1]
    f4, f8, f16, f32, fused = frame_features
    q_t = q_init.unsqueeze(0).clone()
    memory = point_memory.clone()
    mask = torch.zeros(n, memory_size + 1, device=q_init.device, dtype=torch.bool)
    mask[:, :-1] = temporal_mask.clone()
    qkv = torch.zeros(n, memory_size + 1, d, device=q_init.device, dtype=memory.dtype)
    qkv[:, :-1] = memory
    for layer in range(model.decoder_layer_num):
        q_t = model.feature_attention[layer](q_t, fused, fused)
        q_t = model.query_attention[layer](q_t, q_t, q_t)
        qkv[:, -1] = q_t.view(n, d)
        qkv = model.memory_attention[layer](
            qkv + model.t_embedding,
            qkv + model.t_embedding,
            qkv,
            mask,
        )
        q_t = qkv[:, -1].unsqueeze(0)
        qkv[:, :-1] = qkv[:, :-1].clone()
    q_pre = model.projection1(q_t)
    c1 = model.multiscale_correlation(q_pre, f4, f8, f16, f32)
    q_rerank, p_topk, u_topk, s_topk = model.reranking_head(q_pre, f4, f8, f16, f32, c1)
    q_new = model.projection2(q_rerank)
    c2 = model.multiscale_correlation(q_new, f4, f8, f16, f32)
    p_patch = indices_to_coords(
        torch.argmax(c2, dim=-1).unsqueeze(1), model.input_size, model.stride
    ).squeeze(1)
    offsets, v_logit, u_logit = model.prediction_head(q_new, f4, f8, f16, f32, p_patch)
    p_model = p_patch[0] + offsets[-1]
    p = p_model.clone()
    p[..., 0] = (p[..., 0] / model.W) * w_in
    p[..., 1] = (p[..., 1] / model.H) * h_in
    return p, v_logit, q_new.squeeze(0), {
        'c1': c1.squeeze(0),
        'c2': c2.squeeze(0),
        'p_topk': p_topk.squeeze(0),
        'u_topk': u_topk.squeeze(0),
        's_topk': s_topk.squeeze(0),
        'u_logit': u_logit,
        'offsets': offsets,
        'p_model': p_model,
        'q_pre': q_pre.squeeze(0),
    }


def model_xy_to_yx_norm(xy: torch.Tensor, model) -> np.ndarray:
    value = xy.detach().cpu().float().numpy().copy()
    value[..., 0] *= 256.0 / float(model.W)
    value[..., 1] *= 256.0 / float(model.H)
    return (value[..., [1, 0]] / 255.0).astype(np.float32)


def ordinal_rank(values: np.ndarray, descending: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-values if descending else values, kind='mergesort')
    rank = np.empty(len(values), dtype=np.int64)
    rank[order] = np.arange(len(values), dtype=np.int64)
    return rank


@dataclass
class BeamPath:
    cost: int
    current_index: int
    history: tuple[tuple[int, np.ndarray, np.ndarray], ...]


def predicted_coord(history: tuple[tuple[int, np.ndarray, np.ndarray], ...], current_t: int) -> np.ndarray:
    last_t, last_coord, _ = history[-1]
    if len(history) < 2:
        return last_coord
    previous_t, previous_coord, _ = history[-2]
    dt = max(1, int(last_t) - int(previous_t))
    velocity = (last_coord - previous_coord) / float(dt)
    return last_coord + velocity * float(current_t - int(last_t))


def initialize_beam(
    coords: np.ndarray,
    latents: np.ndarray,
    teacher_rank: np.ndarray,
    current_t: int,
    width: int,
) -> list[BeamPath]:
    order = np.lexsort((np.arange(K), teacher_rank))
    beam = []
    for index in order[:width]:
        state = (int(current_t), coords[index].copy(), latents[index].copy())
        beam.append(BeamPath(cost=int(teacher_rank[index]), current_index=int(index), history=(state,)))
    return beam


def update_beam(
    previous: list[BeamPath],
    coords: np.ndarray,
    latents: np.ndarray,
    teacher_rank: np.ndarray,
    current_t: int,
    width: int,
    use_motion: bool,
    use_latent: bool,
) -> list[BeamPath]:
    expansions: list[tuple[int, int, int, int, BeamPath]] = []
    for parent_slot, parent in enumerate(previous):
        step = teacher_rank.astype(np.int64).copy()
        if use_motion:
            prediction = predicted_coord(parent.history, current_t)
            distance = np.linalg.norm((coords - prediction[None, :]) * 255.0, axis=1)
            step += ordinal_rank(distance, descending=False)
        if use_latent:
            similarity = latents @ parent.history[-1][2]
            step += ordinal_rank(similarity, descending=True)
        for candidate_index in range(K):
            state = (int(current_t), coords[candidate_index].copy(), latents[candidate_index].copy())
            history = (parent.history + (state,))[-2:]
            path = BeamPath(
                cost=int(parent.cost + int(step[candidate_index])),
                current_index=int(candidate_index),
                history=history,
            )
            expansions.append(
                (
                    path.cost,
                    int(teacher_rank[candidate_index]),
                    int(candidate_index),
                    int(parent_slot),
                    path,
                )
            )
    expansions.sort(key=lambda row: row[:4])
    selected: list[BeamPath] = []
    seen = set()
    for _, _, candidate_index, _, path in expansions:
        if candidate_index in seen:
            continue
        seen.add(candidate_index)
        selected.append(path)
        if len(selected) == width:
            break
    if len(selected) != width:
        raise RuntimeError(f'beam underflow: {len(selected)} != {width}')
    return selected


def build_visibility_subsets(visible: np.ndarray) -> dict[str, np.ndarray]:
    frames, queries = visible.shape
    reentry_first = np.zeros_like(visible, dtype=bool)
    reentry_early4 = np.zeros_like(visible, dtype=bool)
    reentry_early8 = np.zeros_like(visible, dtype=bool)
    reentry_after_occ4 = np.zeros_like(visible, dtype=bool)
    for query in range(queries):
        occlusion_run = 0
        post_reentry_visible_count = 0
        for frame in range(frames):
            if visible[frame, query]:
                is_reentry = frame > 0 and not visible[frame - 1, query]
                if is_reentry:
                    reentry_first[frame, query] = True
                    if occlusion_run >= 4:
                        reentry_after_occ4[frame, query] = True
                    post_reentry_visible_count = 1
                elif post_reentry_visible_count > 0:
                    post_reentry_visible_count += 1
                if 1 <= post_reentry_visible_count <= 4:
                    reentry_early4[frame, query] = True
                if 1 <= post_reentry_visible_count <= 8:
                    reentry_early8[frame, query] = True
                occlusion_run = 0
            else:
                occlusion_run += 1
                post_reentry_visible_count = 0
    return {
        'reentry_first': reentry_first,
        'reentry_early4': reentry_early4,
        'reentry_early8': reentry_early8,
        'reentry_after_occ4': reentry_after_occ4,
    }


def summarize_errors(
    errors: np.ndarray,
    frame_oracle: np.ndarray,
    selected_index: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    values = errors[mask]
    oracle = frame_oracle[mask]
    indices = selected_index[mask]
    oracle_index = np.argmin(errors_matrix_global[mask], axis=1)
    return {
        'n': int(np.sum(mask)),
        'mean_error': float(np.mean(values)),
        'median_error': float(np.median(values)),
        'safe4': int(np.sum(values <= 4.0)),
        'safe8': int(np.sum(values <= 8.0)),
        'safe16': int(np.sum(values <= 16.0)),
        'frame_oracle_regret': float(np.mean(values - oracle)),
        'exact_best_rate': float(np.mean(indices == oracle_index)),
        'within1px_best_rate': float(np.mean(values <= oracle + 1.0)),
    }


# Bound at runtime after replay; kept global only to avoid repeatedly slicing a 3D array.
errors_matrix_global: np.ndarray


def compare_errors(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    difference = a[mask] - b[mask]
    return {
        'n': int(np.sum(mask)),
        'mean_difference': float(np.mean(difference)),
        'median_difference': float(np.median(difference)),
        'better': int(np.sum(difference < -TOL)),
        'worse': int(np.sum(difference > TOL)),
        'equal': int(np.sum(np.abs(difference) <= TOL)),
    }


def cluster_bootstrap(
    difference: np.ndarray,
    mask: np.ndarray,
    clip_id: np.ndarray,
    query_idx: np.ndarray,
    unit: str,
    seed: int,
) -> dict[str, Any]:
    clusters: dict[Any, list[int]] = defaultdict(list)
    rows = np.where(mask)[0]
    for row in rows.tolist():
        key = (str(clip_id[row]), int(query_idx[row])) if unit == 'track' else str(clip_id[row])
        clusters[key].append(row)
    keys = sorted(clusters, key=str)
    values = np.asarray([float(np.mean(difference[clusters[key]])) for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    chunk = 1000
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        count = min(chunk, BOOTSTRAP_RESAMPLES - start)
        indices = rng.integers(0, len(keys), size=(count, len(keys)))
        bootstrap[start:start + count] = np.mean(values[indices], axis=1)
    return {
        'unit': unit,
        'n_clusters': int(len(keys)),
        'cluster_mean_difference': float(np.mean(values)),
        'cluster_median_difference': float(np.median(values)),
        'cluster_better': int(np.sum(values < -TOL)),
        'cluster_worse': int(np.sum(values > TOL)),
        'cluster_equal': int(np.sum(np.abs(values) <= TOL)),
        'resamples': int(BOOTSTRAP_RESAMPLES),
        'seed': int(seed),
        'mean_difference_95_ci': [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
        'probability_mean_lt_zero': float(np.mean(bootstrap < 0.0)),
    }


def beam_reachability(errors: np.ndarray, beam_indices: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    rows = np.where(mask)[0]
    selected = beam_indices[rows]
    valid = selected >= 0
    gathered = np.full(selected.shape, np.inf, dtype=np.float32)
    for column in range(selected.shape[1]):
        column_valid = valid[:, column]
        gathered[column_valid, column] = errors[rows[column_valid], selected[column_valid, column]]
    minimum = np.min(gathered, axis=1)
    unique_size = np.asarray([len(set(row[row >= 0].tolist())) for row in selected], dtype=np.float32)
    return {
        'n': int(len(rows)),
        'mean_min_error': float(np.mean(minimum)),
        'median_min_error': float(np.median(minimum)),
        'unique_beam_size_mean': float(np.mean(unique_size)),
        'unique_beam_size_min': int(np.min(unique_size)),
        'recall': {str(int(radius)): float(np.mean(minimum <= radius)) for radius in RADII},
    }


def write_doc(result: dict[str, Any]) -> None:
    metrics = result['metrics']
    pairs = result['pairwise']
    lines = [
        '# V9-A5.1 Full-Stream Deterministic Beam Reachability Audit Result',
        '',
        'Date: 2026-07-10',
        '',
        f"Rows={result['protocol']['rows']}; GT-visible rows={result['protocol']['visible_rows']}; reentry-early8 rows={result['protocol']['reentry_early8_rows']}.",
        '',
        '## All-visible top1/readout metrics',
        '',
        '| Policy | Mean | Median | Safe4/8/16 | Oracle regret | Exact best | Within1px |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for policy in result['policy_order']:
        row = metrics[policy]['all_visible']
        lines.append(
            f"| {policy} | {row['mean_error']:.4f} | {row['median_error']:.4f} | "
            f"{row['safe4']}/{row['safe8']}/{row['safe16']} | {row['frame_oracle_regret']:.4f} | "
            f"{row['exact_best_rate']:.4f} | {row['within1px_best_rate']:.4f} |"
        )
    lines += [
        '',
        '## Primary pairwise results',
        '',
        '| Comparison | Mean Δ | Better/Worse/Equal | Track CI | Clip CI |',
        '|---|---:|---:|---|---|',
    ]
    for name in ['primary_top1_vs_teacher', 'primary_oracle_vs_teacher_top4']:
        row = pairs[name]['all_visible']
        track = pairs[name]['track_bootstrap']
        clip = pairs[name]['clip_bootstrap']
        lines.append(
            f"| {name} | {row['mean_difference']:+.4f} | "
            f"{row['better']}/{row['worse']}/{row['equal']} | "
            f"[{track['mean_difference_95_ci'][0]:+.4f},{track['mean_difference_95_ci'][1]:+.4f}] | "
            f"[{clip['mean_difference_95_ci'][0]:+.4f},{clip['mean_difference_95_ci'][1]:+.4f}] |"
        )
    lines += [
        '',
        '## Per-sequence primary means',
        '',
        '| Sequence | Teacher | Beam4 top1 | Teacher top4 oracle | Beam4 oracle |',
        '|---|---:|---:|---:|---:|',
    ]
    for sequence in SEQUENCES:
        subset = f'sequence_{sequence}'
        if subset not in metrics['teacher_top1']:
            continue
        lines.append(
            f"| {sequence} | {metrics['teacher_top1'][subset]['mean_error']:.4f} | "
            f"{metrics['beam4_motion_latent_top1'][subset]['mean_error']:.4f} | "
            f"{metrics['teacher_top4_oracle'][subset]['mean_error']:.4f} | "
            f"{metrics['beam4_motion_latent_oracle'][subset]['mean_error']:.4f} |"
        )
    lines += [
        '',
        '## Primary beam reachability',
        '',
        '```json',
        json.dumps(result['beam_reachability'][PRIMARY_BEAM], indent=2),
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
    OUT_DOC.write_text('\n'.join(lines) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-clips', type=int, default=0)
    parser.add_argument('--skip-full-frame-hash', action='store_true')
    args = parser.parse_args()

    start_all = time.perf_counter()
    input_audit = verify_inputs(verify_frames=not args.skip_full_frame_hash)
    if any(name in inspect.signature(update_beam).parameters for name in ['gt', 'error', 'label']):
        raise RuntimeError('beam update function accepts forbidden GT/error/label input')

    pool = np.load(POOL_PATH, allow_pickle=True)
    pool_names = [str(value) for value in pool['local_feature_names'].tolist()]
    pool_score_index = pool_names.index('exact_rerank_s')
    pool_keys = {
        (str(pool['clip_id'][i]), int(pool['frame_tau'][i]), int(pool['query_idx'][i])): i
        for i in range(len(pool['clip_id']))
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_args_from_yaml(str(CONFIG))
    config.grad_checkpoint = False
    config.M_i = config.M
    predictor = Predictor(config, checkpoint_path=str(CHECKPOINT), support_grid_size=20).to(device).eval()
    model = predictor.model

    clips = [(sequence, start) for sequence in SEQUENCES for start in CLIP_STARTS]
    if args.max_clips > 0:
        clips = clips[:args.max_clips]

    row_clip = []
    row_sequence = []
    row_frame = []
    row_query = []
    row_visible = []
    row_reentry_first = []
    row_reentry_early4 = []
    row_reentry_early8 = []
    row_reentry_after_occ4 = []
    candidate_errors = []
    teacher_scores_all = []
    selections: dict[str, list[int]] = defaultdict(list)
    beam_sets: dict[str, list[np.ndarray]] = defaultdict(list)

    official_p = []
    official_v = []
    official_q = []
    sampled_coord_ordered = []
    sampled_coord_set = []
    sampled_score_diff = []
    sampled_top1_match = []
    beam_unique_violations = 0
    processed_rows = 0

    for sequence, start in clips:
        clip_id = f'{sequence}:{start}'
        sequence_dir = POD_ROOT / sequence
        annotation = np.load(sequence_dir / 'anno.npz', allow_pickle=True, mmap_mode='r')
        trajectories = np.asarray(annotation['trajs_2d'][start:start + LENGTH], dtype=np.float32)
        raw_visibility = np.asarray(annotation['visibs'][start:start + LENGTH], dtype=bool)
        valid = (
            np.asarray(annotation['valids'][start:start + LENGTH], dtype=bool)
            if 'valids' in annotation.files
            else np.ones_like(raw_visibility, dtype=bool)
        )
        finite = np.isfinite(trajectories).all(axis=2)
        eligible = np.where(
            raw_visibility[0]
            & valid[0]
            & finite[0]
            & (np.sum(raw_visibility & valid & finite, axis=0) >= 24)
        )[0]
        chosen = np.sort(
            np.random.default_rng(stable_seed(sequence, start)).choice(eligible, NQ, replace=False)
        )
        gt_xy = trajectories[:, chosen]
        visible = raw_visibility[:, chosen] & valid[:, chosen] & finite[:, chosen]
        subset_flags = build_visibility_subsets(visible)
        frame_paths = [sequence_dir / 'rgbs' / f'rgb_{start + frame:05d}.jpg' for frame in range(LENGTH)]
        width, height = Image.open(frame_paths[0]).size
        query_xy = np.empty((NQ, 2), dtype=np.float32)
        query_xy[:, 0] = gt_xy[0, :, 0] * 255.0 / float(width - 1)
        query_xy[:, 1] = gt_xy[0, :, 1] * 255.0 / float(height - 1)
        queries = torch.from_numpy(query_xy).float().to(device)
        support = get_points_on_a_grid(20, (256, 256), device).squeeze(0)
        combined = torch.cat([queries, support], dim=0)
        predictor.reset()
        predictor.initial_capacity = len(combined)
        beam_state: dict[str, list[list[BeamPath] | None]] = {
            name: [None] * NQ for name in BEAM_CONFIGS
        }
        clip_start = time.perf_counter()

        for frame in range(LENGTH):
            image = load_frame(frame_paths[frame], device)
            f4, f8, f16, f32, fused = model.extract_frame_features(image)
            if frame == 0:
                predictor.init_queries((fused, device), combined, 256, 256)
            active_q = predictor.q_init[:predictor.N]
            active_mask = predictor.temporal_mask[:predictor.N]
            active_memory = predictor.point_memory[:predictor.N]
            capture: dict[str, torch.Tensor] = {}

            def prehook(_module, inputs):
                capture['prefusion'] = inputs[0][:, :NQ].detach().clone()
                return None

            hook = model.reranking_head.fusion_layer.register_forward_pre_hook(prehook)
            p, v_logit, q_new, diag = track_frame_diag(
                model,
                active_q,
                active_mask,
                active_memory,
                (f4, f8, f16, f32, fused),
                256,
                256,
            )
            hook.remove()
            if 'prefusion' not in capture:
                raise RuntimeError('failed to capture pre-fusion candidate latents')
            prefusion = capture['prefusion'].squeeze(0).float().cpu().numpy()
            prefusion /= np.maximum(np.linalg.norm(prefusion, axis=2, keepdims=True), 1e-8)
            coords_all = model_xy_to_yx_norm(diag['p_topk'][:NQ], model)
            scores_all = diag['s_topk'][:NQ].detach().cpu().float().numpy()

            if frame == 0:
                p_ref, v_ref, q_ref = model.track_frame(
                    active_q,
                    active_mask,
                    active_memory,
                    (f4, f8, f16, f32, fused),
                    256,
                    256,
                )
                official_p.append(float(torch.max(torch.abs(p - p_ref))))
                official_v.append(float(torch.max(torch.abs(v_logit - v_ref))))
                official_q.append(float(torch.max(torch.abs(q_new - q_ref))))

            for query in range(NQ):
                coords = coords_all[query]
                scores = scores_all[query]
                latents = prefusion[query]
                gt = np.asarray(
                    [gt_xy[frame, query, 1] / float(height - 1), gt_xy[frame, query, 0] / float(width - 1)],
                    dtype=np.float32,
                )
                errors = np.linalg.norm((coords - gt[None, :]) * 255.0, axis=1).astype(np.float32)
                teacher_rank = ordinal_rank(scores, descending=True)
                teacher_order = np.lexsort((np.arange(K), teacher_rank))
                teacher_index = int(teacher_order[0])
                frame_oracle_index = int(np.argmin(errors))

                row_clip.append(clip_id)
                row_sequence.append(sequence)
                row_frame.append(frame)
                row_query.append(query)
                row_visible.append(bool(visible[frame, query]))
                row_reentry_first.append(bool(subset_flags['reentry_first'][frame, query]))
                row_reentry_early4.append(bool(subset_flags['reentry_early4'][frame, query]))
                row_reentry_early8.append(bool(subset_flags['reentry_early8'][frame, query]))
                row_reentry_after_occ4.append(bool(subset_flags['reentry_after_occ4'][frame, query]))
                candidate_errors.append(errors)
                teacher_scores_all.append(scores)
                selections['teacher_top1'].append(teacher_index)
                selections['frame_top16_oracle'].append(frame_oracle_index)
                selections['teacher_top4_oracle'].append(int(teacher_order[:4][np.argmin(errors[teacher_order[:4]])]))
                selections['teacher_top8_oracle'].append(int(teacher_order[:8][np.argmin(errors[teacher_order[:8]])]))

                for beam_name, beam_config in BEAM_CONFIGS.items():
                    previous = beam_state[beam_name][query]
                    if previous is None:
                        beam = initialize_beam(
                            coords,
                            latents,
                            teacher_rank,
                            frame,
                            int(beam_config['width']),
                        )
                    else:
                        beam = update_beam(
                            previous,
                            coords,
                            latents,
                            teacher_rank,
                            frame,
                            int(beam_config['width']),
                            bool(beam_config['motion']),
                            bool(beam_config['latent']),
                        )
                    beam_state[beam_name][query] = beam
                    indices = np.asarray([path.current_index for path in beam], dtype=np.int16)
                    if len(set(indices.tolist())) != len(indices):
                        beam_unique_violations += 1
                    beam_sets[beam_name].append(indices)
                    selections[f'{beam_name}_top1'].append(int(indices[0]))
                    selections[f'{beam_name}_oracle'].append(int(indices[np.argmin(errors[indices])]))

                sampled_key = (clip_id, frame, query)
                if sampled_key in pool_keys:
                    pool_index = pool_keys[sampled_key]
                    pool_coords = pool['candidates'][pool_index, 2:18].astype(np.float32)
                    sampled_coord_ordered.append(float(np.max(np.abs(coords - pool_coords))))
                    distance = np.linalg.norm(coords[:, None, :] - pool_coords[None, :, :], axis=-1)
                    sampled_coord_set.append(
                        float(max(np.min(distance, axis=1).max(), np.min(distance, axis=0).max()))
                    )
                    pool_scores = pool['Xlocal'][pool_index, 2:18, pool_score_index].astype(np.float32)
                    sampled_score_diff.append(float(np.max(np.abs(scores - pool_scores))))
                    sampled_top1_match.append(int(np.argmax(scores) == np.argmax(pool_scores)))
                processed_rows += 1

            write_mask = torch.ones(predictor.N, dtype=torch.bool, device=device)
            updated_memory, updated_mask = model._update_point_memory(
                active_memory,
                active_mask,
                q_new,
                write_mask,
            )
            predictor.point_memory[:predictor.N] = updated_memory
            predictor.temporal_mask[:predictor.N] = updated_mask
            predictor.t += 1

        print(
            json.dumps(
                {
                    'clip': clip_id,
                    'rows': LENGTH * NQ,
                    'visible_rows': int(np.sum(visible)),
                    'reentry_first': int(np.sum(subset_flags['reentry_first'])),
                    'seconds': float(time.perf_counter() - clip_start),
                }
            ),
            flush=True,
        )

    global errors_matrix_global
    errors_matrix_global = np.stack(candidate_errors).astype(np.float32)
    teacher_scores_array = np.stack(teacher_scores_all).astype(np.float32)
    clip_array = np.asarray(row_clip, dtype=object)
    sequence_array = np.asarray(row_sequence, dtype=object)
    frame_array = np.asarray(row_frame, dtype=np.int16)
    query_array = np.asarray(row_query, dtype=np.int16)
    visible_array = np.asarray(row_visible, dtype=bool)
    flags = {
        'reentry_first': np.asarray(row_reentry_first, dtype=bool),
        'reentry_early4': np.asarray(row_reentry_early4, dtype=bool),
        'reentry_early8': np.asarray(row_reentry_early8, dtype=bool),
        'reentry_after_occ4': np.asarray(row_reentry_after_occ4, dtype=bool),
    }
    selection_arrays = {name: np.asarray(value, dtype=np.int16) for name, value in selections.items()}
    beam_arrays = {}
    for name, values in beam_sets.items():
        width = int(BEAM_CONFIGS[name]['width'])
        beam_arrays[name] = np.stack(values).reshape(processed_rows, width).astype(np.int16)

    expected_rows = len(clips) * LENGTH * NQ
    if processed_rows != expected_rows:
        raise RuntimeError(f'row count mismatch: {processed_rows} != {expected_rows}')
    if beam_unique_violations != 0:
        raise RuntimeError(f'beam uniqueness violations: {beam_unique_violations}')
    if not all(
        np.all(np.isfinite(value))
        for value in [errors_matrix_global, teacher_scores_array]
    ):
        raise RuntimeError('non-finite replay output')

    row = np.arange(processed_rows)
    chosen_errors = {
        name: errors_matrix_global[row, index]
        for name, index in selection_arrays.items()
    }
    frame_oracle_error = chosen_errors['frame_top16_oracle']
    teacher_error = chosen_errors['teacher_top1']
    teacher_hard = teacher_error > 4.0

    masks: dict[str, np.ndarray] = {
        'all_visible': visible_array,
        'teacher_hard': visible_array & teacher_hard,
        'teacher_easy': visible_array & ~teacher_hard,
        'reentry_first': visible_array & flags['reentry_first'],
        'reentry_early4': visible_array & flags['reentry_early4'],
        'reentry_early8': visible_array & flags['reentry_early8'],
        'reentry_after_occ4': visible_array & flags['reentry_after_occ4'],
    }
    for sequence in SEQUENCES:
        masks[f'sequence_{sequence}'] = visible_array & (sequence_array == sequence)

    policy_order = [
        'teacher_top1',
        'beam1_motion_latent_top1',
        'beam4_motion_top1',
        'beam4_latent_top1',
        'beam4_motion_latent_top1',
        'beam8_motion_latent_top1',
        'teacher_top4_oracle',
        'teacher_top8_oracle',
        'beam4_motion_oracle',
        'beam4_latent_oracle',
        'beam4_motion_latent_oracle',
        'beam8_motion_latent_oracle',
        'frame_top16_oracle',
    ]
    metrics = {
        policy: {
            subset: summarize_errors(
                chosen_errors[policy],
                frame_oracle_error,
                selection_arrays[policy],
                mask,
            )
            for subset, mask in masks.items()
            if np.any(mask)
        }
        for policy in policy_order
    }

    primary_top1 = 'beam4_motion_latent_top1'
    primary_oracle = 'beam4_motion_latent_oracle'
    pair_specs = {
        'primary_top1_vs_teacher': (primary_top1, 'teacher_top1'),
        'primary_oracle_vs_teacher_top4': (primary_oracle, 'teacher_top4_oracle'),
    }
    pairwise = {}
    for name, (a_name, b_name) in pair_specs.items():
        difference = chosen_errors[a_name] - chosen_errors[b_name]
        pairwise[name] = {
            subset: compare_errors(chosen_errors[a_name], chosen_errors[b_name], mask)
            for subset, mask in masks.items()
            if np.any(mask)
        }
        pairwise[name]['track_bootstrap'] = cluster_bootstrap(
            difference,
            masks['all_visible'],
            clip_array,
            query_array,
            'track',
            TRACK_BOOTSTRAP_SEED,
        )
        pairwise[name]['clip_bootstrap'] = cluster_bootstrap(
            difference,
            masks['all_visible'],
            clip_array,
            query_array,
            'clip',
            CLIP_BOOTSTRAP_SEED,
        )

    reachability = {
        name: {
            subset: beam_reachability(errors_matrix_global, values, mask)
            for subset, mask in masks.items()
            if np.any(mask)
        }
        for name, values in beam_arrays.items()
    }

    if args.max_clips > 0:
        deterministic_gate = {'smoke_only': True, 'pass_all': False}
        reachability_gate = {'smoke_only': True, 'pass_all': False}
    else:
        primary_sequence_top1 = {}
        primary_sequence_oracle = {}
        for sequence in SEQUENCES:
            subset = f'sequence_{sequence}'
            primary_sequence_top1[sequence] = {
                'mean_improved': bool(
                    metrics[primary_top1][subset]['mean_error']
                    < metrics['teacher_top1'][subset]['mean_error']
                ),
                'safe16_not_decreased': bool(
                    metrics[primary_top1][subset]['safe16']
                    >= metrics['teacher_top1'][subset]['safe16']
                ),
            }
            primary_sequence_oracle[sequence] = {
                'mean_difference_vs_teacher_top4': float(
                    metrics[primary_oracle][subset]['mean_error']
                    - metrics['teacher_top4_oracle'][subset]['mean_error']
                ),
                'not_worse': bool(
                    metrics[primary_oracle][subset]['mean_error']
                    <= metrics['teacher_top4_oracle'][subset]['mean_error'] + TOL
                ),
                'safe16_not_decreased': bool(
                    metrics[primary_oracle][subset]['safe16']
                    >= metrics['teacher_top4_oracle'][subset]['safe16']
                ),
            }

        top1_pair = pairwise['primary_top1_vs_teacher']['all_visible']
        top1_reentry = pairwise['primary_top1_vs_teacher']['reentry_early8']
        top1_clip_ci = pairwise['primary_top1_vs_teacher']['clip_bootstrap']['mean_difference_95_ci']
        deterministic_gate = {
            'per_sequence': primary_sequence_top1,
            'global_better_gt_worse': bool(top1_pair['better'] > top1_pair['worse']),
            'clip_ci_upper_lt_zero': bool(top1_clip_ci[1] < 0.0),
            'reentry_early8_mean_improved': bool(top1_reentry['mean_difference'] < 0.0),
            'reentry_early8_safe16_not_decreased': bool(
                metrics[primary_top1]['reentry_early8']['safe16']
                >= metrics['teacher_top1']['reentry_early8']['safe16']
            ),
        }
        deterministic_gate['pass_all'] = bool(
            all(row['mean_improved'] and row['safe16_not_decreased'] for row in primary_sequence_top1.values())
            and deterministic_gate['global_better_gt_worse']
            and deterministic_gate['clip_ci_upper_lt_zero']
            and deterministic_gate['reentry_early8_mean_improved']
            and deterministic_gate['reentry_early8_safe16_not_decreased']
        )

        oracle_pair = pairwise['primary_oracle_vs_teacher_top4']['all_visible']
        oracle_reentry = pairwise['primary_oracle_vs_teacher_top4']['reentry_early8']
        oracle_clip_ci = pairwise['primary_oracle_vs_teacher_top4']['clip_bootstrap']['mean_difference_95_ci']
        reachability_gate = {
            'global_mean_improvement_ge_0.10px': bool(oracle_pair['mean_difference'] <= -0.10),
            'per_sequence': primary_sequence_oracle,
            'global_better_gt_worse': bool(oracle_pair['better'] > oracle_pair['worse']),
            'clip_ci_upper_lt_zero': bool(oracle_clip_ci[1] < 0.0),
            'reentry_early8_mean_improvement_ge_0.25px': bool(oracle_reentry['mean_difference'] <= -0.25),
            'reentry_early8_safe16_not_decreased': bool(
                metrics[primary_oracle]['reentry_early8']['safe16']
                >= metrics['teacher_top4_oracle']['reentry_early8']['safe16']
            ),
        }
        reachability_gate['pass_all'] = bool(
            reachability_gate['global_mean_improvement_ge_0.10px']
            and all(row['not_worse'] and row['safe16_not_decreased'] for row in primary_sequence_oracle.values())
            and reachability_gate['global_better_gt_worse']
            and reachability_gate['clip_ci_upper_lt_zero']
            and reachability_gate['reentry_early8_mean_improvement_ge_0.25px']
            and reachability_gate['reentry_early8_safe16_not_decreased']
        )

    if args.max_clips > 0:
        decision = 'SMOKE_ONLY: full nine-clip gates were not evaluated.'
    elif deterministic_gate['pass_all']:
        decision = (
            'DETERMINISTIC_BEAM_PASS: beam4 motion+latent top1 passes all full-stream synthetic gates. '
            'Freeze the deterministic policy and proceed to robustness packaging before any DAVIS diagnostic.'
        )
    elif reachability_gate['pass_all']:
        decision = (
            'BEAM_REACHABILITY_ONLY: deterministic top1 fails, but the retained beam adds statistically '
            'stable reachability beyond teacher top4, including re-entry. Proceed to sequence-heldout '
            'learned beam scoring/readout; do not train another single-state reranker.'
        )
    else:
        decision = (
            'FULLSTREAM_BEAM_FAIL: neither deterministic top1 nor GT-only beam reachability passes. '
            'Stop the temporal multi-hypothesis branch and return to candidate-recall/correlation-map analysis.'
        )

    integrity = {
        'input_audit': input_audit,
        'processed_rows': int(processed_rows),
        'expected_rows': int(expected_rows),
        'official_p_max_abs': float(np.max(official_p)),
        'official_v_max_abs': float(np.max(official_v)),
        'official_q_max_abs': float(np.max(official_q)),
        'sampled_rows_checked': int(len(sampled_top1_match)),
        'sampled_candidate_ordered_max_abs': float(np.max(sampled_coord_ordered)),
        'sampled_candidate_set_hausdorff_max': float(np.max(sampled_coord_set)),
        'sampled_teacher_score_max_abs': float(np.max(sampled_score_diff)),
        'sampled_teacher_top1_match_rate': float(np.mean(sampled_top1_match)),
        'beam_unique_violations': int(beam_unique_violations),
        'beam_update_signature': list(inspect.signature(update_beam).parameters),
        'beam_update_uses_gt_or_error': False,
        'all_finite': True,
    }
    if max(integrity['official_p_max_abs'], integrity['official_v_max_abs'], integrity['official_q_max_abs']) > 1e-6:
        raise RuntimeError(f'official forward parity failed: {integrity}')
    if integrity['sampled_candidate_set_hausdorff_max'] > 1e-6:
        raise RuntimeError(f'sampled candidate parity failed: {integrity}')
    if integrity['sampled_teacher_score_max_abs'] > 0.005:
        raise RuntimeError(f'sampled teacher-score parity failed: {integrity}')
    if integrity['sampled_teacher_top1_match_rate'] != 1.0:
        raise RuntimeError(f'sampled teacher top1 parity failed: {integrity}')

    result = {
        'script': 'scripts/v9a51_fullstream_beam_reachability.py',
        'date': '2026-07-10',
        'protocol': {
            'dataset': 'PointOdyssey full stream',
            'clips': [f'{sequence}:{start}' for sequence, start in clips],
            'rows': int(processed_rows),
            'visible_rows': int(np.sum(visible_array)),
            'reentry_first_rows': int(np.sum(masks['reentry_first'])),
            'reentry_early4_rows': int(np.sum(masks['reentry_early4'])),
            'reentry_early8_rows': int(np.sum(masks['reentry_early8'])),
            'reentry_after_occ4_rows': int(np.sum(masks['reentry_after_occ4'])),
            'beam_configs': BEAM_CONFIGS,
            'primary_beam': PRIMARY_BEAM,
            'training': False,
            'davis_read': False,
            'threshold_tuning': False,
            'seconds': float(time.perf_counter() - start_all),
        },
        'policy_order': policy_order,
        'metrics': metrics,
        'pairwise': pairwise,
        'beam_reachability': reachability,
        'gates': {
            'deterministic_top1': deterministic_gate,
            'beam_reachability': reachability_gate,
            'decision': decision,
        },
        'integrity': integrity,
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        clip_id=clip_array,
        sequence=sequence_array,
        frame_tau=frame_array,
        query_idx=query_array,
        visible=visible_array.astype(np.int8),
        reentry_first=flags['reentry_first'].astype(np.int8),
        reentry_early4=flags['reentry_early4'].astype(np.int8),
        reentry_early8=flags['reentry_early8'].astype(np.int8),
        reentry_after_occ4=flags['reentry_after_occ4'].astype(np.int8),
        candidate_error=errors_matrix_global,
        teacher_score=teacher_scores_array,
        **{f'selection_{name}': value for name, value in selection_arrays.items()},
        **{f'beam_{name}': value for name, value in beam_arrays.items()},
    )
    saved = np.load(OUT_NPZ, allow_pickle=True)
    if not np.array_equal(saved['selection_teacher_top1'], selection_arrays['teacher_top1']):
        raise RuntimeError('saved NPZ teacher selection mismatch')
    saved_primary_error = saved['candidate_error'][row, saved['selection_beam4_motion_latent_top1']]
    saved_mean = float(np.mean(saved_primary_error[saved['visible'].astype(bool)]))
    expected_mean = metrics['beam4_motion_latent_top1']['all_visible']['mean_error']
    if abs(saved_mean - expected_mean) > 1e-6:
        raise RuntimeError(f'saved NPZ metric reproduction failed: {saved_mean} != {expected_mean}')
    result['integrity']['saved_npz_primary_mean_max_abs'] = float(abs(saved_mean - expected_mean))

    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    write_doc(result)
    print(
        json.dumps(
            {
                'ok': True,
                'json': str(OUT_JSON),
                'npz': str(OUT_NPZ),
                'doc': str(OUT_DOC),
                'decision': decision,
                'deterministic_gate': deterministic_gate,
                'reachability_gate': reachability_gate,
                'primary_top1_mean': metrics['beam4_motion_latent_top1']['all_visible']['mean_error'],
                'teacher_mean': metrics['teacher_top1']['all_visible']['mean_error'],
                'primary_oracle_mean': metrics['beam4_motion_latent_oracle']['all_visible']['mean_error'],
                'teacher_top4_oracle_mean': metrics['teacher_top4_oracle']['all_visible']['mean_error'],
            },
            ensure_ascii=False,
        )
    )


if __name__ == '__main__':
    main()
