#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

WORKTREE_ROOT = Path('/gemini/code/FSPT_v9a45_clean')
SOURCE_ROOT = Path('/gemini/code/FSPT')
CLEAN_TRACKON_ROOT = WORKTREE_ROOT / 'baselines/track_on'
SOURCE_BASE = SOURCE_ROOT / 'outputs/paper_discovery_2026-07-05'
POOL_PATH = SOURCE_BASE / 'v9a38_pointodyssey_pool/v9a38_pointodyssey_hypothesis_pool.npz'
POD_ROOT = SOURCE_ROOT / 'datasets/pointodyssey/train'
CHECKPOINT = SOURCE_ROOT / 'baselines/track_on/checkpoints_trackon2_dinov3.pt'
CONFIG = CLEAN_TRACKON_ROOT / 'config/test.yaml'
ASSET_MANIFEST = WORKTREE_ROOT / 'docs/v9a5c0_input_manifest_2026-07-10.json'
OUTDIR = WORKTREE_ROOT / 'outputs/paper_discovery_2026-07-05/v9a5c0_candidate_recall'
DEFAULT_JSON = OUTDIR / 'v9a5c0_pointodyssey_candidate_recall.json'
DEFAULT_NPZ = OUTDIR / 'v9a5c0_pointodyssey_candidate_recall.npz'
DEFAULT_DOC = WORKTREE_ROOT / 'docs/v9a5c0_pointodyssey_candidate_recall_result_2026-07-10.md'
EXPECTED_HEAD = '7c24367cf8aeb1467946d2ced3b294299032538e'
EXPECTED_BRANCH = 'v9a45-conservative-residual-20260710'

K_VALUES = np.asarray([1, 4, 8, 16, 32, 64], dtype=np.int32)
RADII = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
MAP_NAMES = ('fused', 'c4', 'c8', 'c16', 'c32', 'union_raw_equal')
SCORE_MAP_NAMES = ('fused', 'c4', 'c8', 'c16', 'c32')
STRIDES = {'fused': 4, 'c4': 4, 'c8': 8, 'c16': 16, 'c32': 32}
SEQUENCES = ('ani', 'animal3', 'r4_new_f')
NQ = 32
LENGTH = 96
UNIFIED_SIZE = 256.0
UNIFIED_DENOM = 255.0
HEADROOM_THRESHOLD = 0.02

if str(CLEAN_TRACKON_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEAN_TRACKON_ROOT))
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.append(str(WORKTREE_ROOT))
os.environ['DINOV3_LOCAL_DIR'] = str(
    SOURCE_ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m'
)

from model.trackon_predictor import Predictor  # noqa: E402
from utils.coord_utils import get_points_on_a_grid, indices_to_coords  # noqa: E402
from utils.train_utils import load_args_from_yaml  # noqa: E402
from scripts.v9a26_build_trackon2_internal_proxy_features import track_frame_diag  # noqa: E402

# v9a26 resolves its ROOT from the clean worktree and therefore overwrites
# DINOV3_LOCAL_DIR with a non-existent clean-worktree asset path. Restore the
# hash-verified immutable source asset after all imports and force offline mode.
os.environ['DINOV3_LOCAL_DIR'] = str(
    SOURCE_ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m'
)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


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


def verify_inputs(verify_all_frames: bool) -> dict[str, Any]:
    manifest = json.loads(ASSET_MANIFEST.read_text())
    manifest_by_path = {str(Path(row['path']).resolve()): row for row in manifest['files']}
    required = [
        POOL_PATH,
        CHECKPOINT,
        SOURCE_ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m/model.safetensors',
        SOURCE_ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m/config.json',
        POD_ROOT / 'ani/anno.npz',
        POD_ROOT / 'animal3/anno.npz',
        POD_ROOT / 'r4_new_f/anno.npz',
        CONFIG,
        Path(manifest['frame_manifest']['path']),
    ]
    checked = []
    for path in required:
        resolved = str(path.resolve())
        if resolved not in manifest_by_path:
            raise RuntimeError(f'required input missing from asset manifest: {resolved}')
        expected = manifest_by_path[resolved]
        size = path.stat().st_size
        digest = sha256(path)
        if size != int(expected['size_bytes']) or digest != expected['sha256']:
            raise RuntimeError(f'asset hash/size mismatch: {path}')
        checked.append({'path': resolved, 'size_bytes': int(size), 'sha256': digest})

    frame_report = None
    if verify_all_frames:
        frame_meta = manifest['frame_manifest']
        frame_manifest_path = Path(frame_meta['path'])
        frame_manifest = json.loads(frame_manifest_path.read_text())
        root = Path(frame_manifest['root'])
        aggregate = hashlib.sha256()
        total_bytes = 0
        for row in frame_manifest['files']:
            relative = str(row['relative_path'])
            path = root / relative
            if not path.exists():
                raise RuntimeError(f'frame missing: {path}')
            size = path.stat().st_size
            digest = sha256(path)
            if size != int(row['size_bytes']) or digest != row['sha256']:
                raise RuntimeError(f'frame hash/size mismatch: {path}')
            total_bytes += size
            aggregate.update(relative.encode())
            aggregate.update(b'\0')
            aggregate.update(str(size).encode())
            aggregate.update(b'\0')
            aggregate.update(digest.encode())
            aggregate.update(b'\n')
        aggregate_digest = aggregate.hexdigest()
        if len(frame_manifest['files']) != int(frame_meta['file_count']):
            raise RuntimeError('frame manifest file-count mismatch')
        if total_bytes != int(frame_meta['total_bytes']):
            raise RuntimeError('frame manifest total-byte mismatch')
        if aggregate_digest != frame_meta['aggregate_sha256']:
            raise RuntimeError('frame manifest aggregate mismatch')
        frame_report = {
            'path': str(frame_manifest_path),
            'file_count': int(len(frame_manifest['files'])),
            'total_bytes': int(total_bytes),
            'aggregate_sha256': aggregate_digest,
        }

    head = subprocess.check_output(
        ['git', '-C', str(WORKTREE_ROOT), 'rev-parse', 'HEAD'], text=True
    ).strip()
    branch = subprocess.check_output(
        ['git', '-C', str(WORKTREE_ROOT), 'branch', '--show-current'], text=True
    ).strip()
    tracked_status = subprocess.check_output(
        ['git', '-C', str(WORKTREE_ROOT), 'status', '--porcelain', '--untracked-files=no'],
        text=True,
    ).strip()
    if head != EXPECTED_HEAD:
        raise RuntimeError(f'HEAD mismatch: {head} != {EXPECTED_HEAD}')
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f'branch mismatch: {branch} != {EXPECTED_BRANCH}')
    if tracked_status:
        raise RuntimeError(f'tracked worktree is not clean: {tracked_status}')

    imported_paths = {}
    for name in [
        'model.trackon_predictor',
        'model.trackon',
        'model.reranking',
        'utils.coord_utils',
        'utils.train_utils',
    ]:
        module = sys.modules.get(name)
        path = '' if module is None else str(Path(module.__file__).resolve())
        imported_paths[name] = path
        if not path.startswith(str(CLEAN_TRACKON_ROOT.resolve()) + '/'):
            raise RuntimeError(f'TrackOn code not imported from clean worktree: {name} -> {path}')

    return {
        'pass': True,
        'head': head,
        'branch': branch,
        'tracked_status': tracked_status,
        'checked_inputs': checked,
        'frame_manifest': frame_report,
        'imported_trackon_code_paths': imported_paths,
        'script': {'path': str(Path(__file__).resolve()), 'sha256': sha256(Path(__file__).resolve())},
    }


def load_frame(path: Path, device: torch.device) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().unsqueeze(0)
    tensor = F.interpolate(tensor, size=(256, 256), mode='bilinear', align_corners=False)
    return tensor.to(device)


def model_xy_to_unified_yx_norm(xy: torch.Tensor, model) -> np.ndarray:
    value = xy.detach().cpu().float().numpy().copy()
    value[..., 0] *= UNIFIED_SIZE / float(model.W)
    value[..., 1] *= UNIFIED_SIZE / float(model.H)
    return (value[..., [1, 0]] / UNIFIED_DENOM).astype(np.float32)


def grid_yx_norm(model, stride: int, device: torch.device) -> np.ndarray:
    height = model.H // stride
    width = model.W // stride
    indices = torch.arange(height * width, device=device, dtype=torch.long).view(1, 1, -1)
    xy = indices_to_coords(indices, model.input_size, stride)[0, 0]
    return model_xy_to_unified_yx_norm(xy, model)


def correlation_components(model, q_pre, f4, f8, f16, f32) -> dict[str, torch.Tensor]:
    n_queries = q_pre.shape[1]
    q_normalized = F.normalize(q_pre, p=2, dim=-1)
    flat = {
        'c4': torch.einsum('bnd,bpd->bnp', q_normalized, F.normalize(f4, p=2, dim=-1)),
        'c8': torch.einsum('bnd,bpd->bnp', q_normalized, F.normalize(f8, p=2, dim=-1)),
        'c16': torch.einsum('bnd,bpd->bnp', q_normalized, F.normalize(f16, p=2, dim=-1)),
        'c32': torch.einsum('bnd,bpd->bnp', q_normalized, F.normalize(f32, p=2, dim=-1)),
    }
    native = {
        'c4': flat['c4'].view(n_queries, model.Hf, model.Wf),
        'c8': flat['c8'].view(n_queries, model.Hf // 2, model.Wf // 2),
        'c16': flat['c16'].view(n_queries, model.Hf // 4, model.Wf // 4),
        'c32': flat['c32'].view(n_queries, model.Hf // 8, model.Wf // 8),
    }
    upsampled = [
        native['c4'].unsqueeze(1),
        F.interpolate(native['c8'].unsqueeze(1), size=(model.Hf, model.Wf), mode='bilinear', align_corners=False),
        F.interpolate(native['c16'].unsqueeze(1), size=(model.Hf, model.Wf), mode='bilinear', align_corners=False),
        F.interpolate(native['c32'].unsqueeze(1), size=(model.Hf, model.Wf), mode='bilinear', align_corners=False),
    ]
    stacked = torch.cat(upsampled, dim=1)
    fused = model.ms_corr_proj(stacked).view(1, n_queries, model.P)
    return {
        'fused': fused.squeeze(0),
        'c4': flat['c4'].squeeze(0),
        'c8': flat['c8'].squeeze(0),
        'c16': flat['c16'].squeeze(0),
        'c32': flat['c32'].squeeze(0),
    }


def top_order(scores: np.ndarray, maximum: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    count = min(int(maximum), len(scores))
    if count == len(scores):
        return np.argsort(-scores, kind='mergesort')
    partition = np.argpartition(-scores, count - 1)[:count]
    return partition[np.argsort(-scores[partition], kind='mergesort')]


def unique_coords(coords: np.ndarray) -> np.ndarray:
    if len(coords) == 0:
        return coords
    rounded = np.round(np.asarray(coords, dtype=np.float32), decimals=8)
    _, index = np.unique(rounded, axis=0, return_index=True)
    return np.asarray(coords)[np.sort(index)]


def evaluate_score_map(
    scores: np.ndarray,
    coords: np.ndarray,
    gt_yx: np.ndarray,
    order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float, float, float, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float32)
    order = np.asarray(order, dtype=np.int64)
    maximum = min(int(K_VALUES[-1]), len(scores))
    if order.shape != (maximum,):
        raise RuntimeError(f'unexpected top-order shape: {order.shape} != {(maximum,)}')
    if len(np.unique(order)) != len(order) or np.any(order < 0) or np.any(order >= len(scores)):
        raise RuntimeError('invalid top-order indices')
    top_coords = coords[order]
    top_errors = np.linalg.norm((top_coords - gt_yx[None, :]) * UNIFIED_DENOM, axis=1)
    min_errors = np.empty(len(K_VALUES), dtype=np.float32)
    candidate_counts = np.empty(len(K_VALUES), dtype=np.int32)
    for ki, k in enumerate(K_VALUES.tolist()):
        count = min(int(k), len(top_errors))
        min_errors[ki] = float(np.min(top_errors[:count]))
        candidate_counts[ki] = count

    all_distance = np.linalg.norm((coords - gt_yx[None, :]) * UNIFIED_DENOM, axis=1)
    nearest_index = int(np.argmin(all_distance))
    nearest_distance = float(all_distance[nearest_index])
    nearest_score = float(scores[nearest_index])
    top1_score = float(np.max(scores))
    nearest_rank = int(1 + np.sum(scores > nearest_score))
    margin = nearest_score - top1_score
    return min_errors, candidate_counts, nearest_rank, nearest_distance, margin, top1_score, order


def summarize_subset(
    min_error: np.ndarray,
    candidate_count: np.ndarray,
    nearest_rank: np.ndarray,
    nearest_distance: np.ndarray,
    nearest_margin: np.ndarray,
    boundary_gap: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    result: dict[str, Any] = {'n': int(np.sum(mask)), 'maps': {}}
    for mi, map_name in enumerate(MAP_NAMES):
        rows = []
        for ki, k in enumerate(K_VALUES.tolist()):
            values = min_error[mask, mi, ki]
            row = {
                'k': int(k),
                'candidate_count_mean': float(np.mean(candidate_count[mask, mi, ki])),
                'min_error_mean': float(np.mean(values)),
                'min_error_median': float(np.median(values)),
                'recall': {
                    str(int(radius)): float(np.mean(values <= radius))
                    for radius in RADII.tolist()
                },
            }
            rows.append(row)
        entry: dict[str, Any] = {'budgets': rows}
        if map_name in SCORE_MAP_NAMES:
            score_index = SCORE_MAP_NAMES.index(map_name)
            ranks = nearest_rank[mask, score_index]
            distances = nearest_distance[mask, score_index]
            margins = nearest_margin[mask, score_index]
            entry['nearest_grid'] = {
                'rank_mean': float(np.mean(ranks)),
                'rank_median': float(np.median(ranks)),
                'rank_p90': float(np.quantile(ranks, 0.90)),
                'rank_le16': float(np.mean(ranks <= 16)),
                'rank_le32': float(np.mean(ranks <= 32)),
                'rank_le64': float(np.mean(ranks <= 64)),
                'distance_mean': float(np.mean(distances)),
                'distance_median': float(np.median(distances)),
                'margin_mean': float(np.mean(margins)),
                'margin_median': float(np.median(margins)),
            }
            entry['topk_boundary'] = {
                str(int(k)): {
                    'gap_mean': float(np.mean(boundary_gap[mask, score_index, ki])),
                    'gap_median': float(np.median(boundary_gap[mask, score_index, ki])),
                    'exact_tie_rate_le_1e-8': float(
                        np.mean(boundary_gap[mask, score_index, ki] <= 1e-8)
                    ),
                    'near_tie_rate_le_1e-6': float(
                        np.mean(boundary_gap[mask, score_index, ki] <= 1e-6)
                    ),
                }
                for ki, k in enumerate(K_VALUES.tolist())
            }
        result['maps'][map_name] = entry
    return result


def budget_row(summary: dict[str, Any], map_name: str, k: int) -> dict[str, Any]:
    rows = summary['maps'][map_name]['budgets']
    return next(row for row in rows if int(row['k']) == int(k))


def compute_gate(summaries: dict[str, Any]) -> dict[str, Any]:
    per_sequence = {}
    for sequence in SEQUENCES:
        summary = summaries[f'sequence_{sequence}']
        current = budget_row(summary, 'fused', 16)['recall']['4']
        expanded_fused = budget_row(summary, 'fused', 64)['recall']['4']
        expanded_union_raw = budget_row(summary, 'union_raw_equal', 64)['recall']['4']
        headroom = expanded_fused - current
        per_sequence[sequence] = {
            'fused_top16_recall4': float(current),
            'fused_top64_recall4': float(expanded_fused),
            'union_raw_equal_top64_recall4_diagnostic': float(expanded_union_raw),
            'fused_headroom': float(headroom),
            'pass_fused_headroom_ge_0.02': bool(headroom >= HEADROOM_THRESHOLD),
        }
    passed = bool(all(row['pass_fused_headroom_ge_0.02'] for row in per_sequence.values()))
    if passed:
        decision = (
            'POINTODYSSEY_CANDIDATE_RECALL_HEADROOM_PASS: every sequence has at least '
            'two percentage points of fused-map recall@4px between top16 and top64. Freeze '
            'this as synthetic upstream headroom. Do not automatically read DAVIS or train '
            'an adapter; combine this result with the V9-A5.0 temporal-state audit to choose '
            'the next full-stream synthetic experiment.'
        )
    else:
        decision = (
            'POINTODYSSEY_CANDIDATE_RECALL_HEADROOM_FAIL: at least one sequence lacks '
            'the predeclared top16-to-candidate-pool recall@4px headroom. Stop the current '
            'correlation-adapter branch; the remaining gap is not sequence-consistent candidate recall.'
        )
    return {
        'threshold': float(HEADROOM_THRESHOLD),
        'per_sequence': per_sequence,
        'pass_all_sequences': passed,
        'decision': decision,
    }


def write_doc(result: dict[str, Any], path: Path) -> None:
    summaries = result['summaries']
    gate = result['gate']
    lines = [
        '# V9-A5C.0 PointOdyssey Candidate-Recall / Correlation-Map Oracle Audit',
        '',
        'Date: 2026-07-10',
        '',
        f"Rows audited: {result['protocol']['rows']}; clips: {result['protocol']['clips']}; device: {result['protocol']['device']}.",
        '',
        '## Global recall@4px',
        '',
        '| Map | K1 | K4 | K8 | K16 | K32 | K64 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    global_summary = summaries['all']
    for map_name in MAP_NAMES:
        values = [budget_row(global_summary, map_name, int(k))['recall']['4'] for k in K_VALUES]
        lines.append('| ' + map_name + ' | ' + ' | '.join(f'{value:.4f}' for value in values) + ' |')

    if 'per_sequence' in gate:
        lines += [
            '',
            '## Per-sequence primary headroom',
            '',
            '| Sequence | Fused K16 R@4 | Fused K64 R@4 | Raw-union K64 R@4 (diag.) | Fused headroom | Pass |',
            '|---|---:|---:|---:|---:|---|',
        ]
        for sequence in SEQUENCES:
            row = gate['per_sequence'][sequence]
            lines.append(
                f"| {sequence} | {row['fused_top16_recall4']:.4f} | "
                f"{row['fused_top64_recall4']:.4f} | "
                f"{row['union_raw_equal_top64_recall4_diagnostic']:.4f} | "
                f"{row['fused_headroom']:+.4f} | "
                f"{row['pass_fused_headroom_ge_0.02']} |"
            )
    else:
        lines += ['', '## Per-sequence primary headroom', '', 'Smoke run only; the three-sequence gate was not evaluated.']

    lines += [
        '',
        '## Nearest-GT grid rank',
        '',
        '| Map | Median rank | P90 rank | <=16 | <=32 | <=64 | Nearest distance median | Margin median |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for map_name in SCORE_MAP_NAMES:
        row = global_summary['maps'][map_name]['nearest_grid']
        lines.append(
            f"| {map_name} | {row['rank_median']:.1f} | {row['rank_p90']:.1f} | "
            f"{row['rank_le16']:.4f} | {row['rank_le32']:.4f} | {row['rank_le64']:.4f} | "
            f"{row['distance_median']:.3f} | {row['margin_median']:+.4f} |"
        )

    lines += [
        '',
        '## Integrity',
        '',
        '```json',
        json.dumps(result['integrity'], indent=2),
        '```',
        '',
        '## Decision',
        '',
        gate['decision'],
    ]
    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-clips', type=int, default=0)
    parser.add_argument('--skip-full-frame-hash', action='store_true')
    parser.add_argument('--out-json', type=Path, default=DEFAULT_JSON)
    parser.add_argument('--out-npz', type=Path, default=DEFAULT_NPZ)
    parser.add_argument('--out-doc', type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()

    start_all = time.perf_counter()
    input_audit = verify_inputs(verify_all_frames=not args.skip_full_frame_hash)
    pool = np.load(POOL_PATH, allow_pickle=True)
    row_keys = [
        (str(pool['clip_id'][i]), int(pool['frame_tau'][i]), int(pool['query_idx'][i]))
        for i in range(len(pool['clip_id']))
    ]
    if len(set(row_keys)) != len(row_keys):
        raise RuntimeError('duplicate PointOdyssey pool row keys')
    key_to_index = {key: i for i, key in enumerate(row_keys)}
    clips = sorted(set(key[0] for key in row_keys))
    if args.max_clips > 0:
        clips = clips[: args.max_clips]
    selected_indices = np.asarray(
        [i for i, key in enumerate(row_keys) if key[0] in set(clips)], dtype=np.int64
    )
    local_index = {int(global_i): local_i for local_i, global_i in enumerate(selected_indices.tolist())}
    n_rows = len(selected_indices)
    if n_rows == 0:
        raise RuntimeError('no rows selected')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_args_from_yaml(str(CONFIG))
    config.grad_checkpoint = False
    config.M_i = config.M
    predictor = Predictor(
        config,
        checkpoint_path=str(CHECKPOINT),
        support_grid_size=20,
    ).to(device).eval()
    model = predictor.model
    if str(model.memory_update_policy) != 'unconditional':
        raise RuntimeError(
            f'unexpected TrackOn2 memory policy: {model.memory_update_policy}'
        )

    grid_coords = {
        name: grid_yx_norm(model, stride, device)
        for name, stride in STRIDES.items()
    }

    min_error = np.full((n_rows, len(MAP_NAMES), len(K_VALUES)), np.nan, dtype=np.float32)
    candidate_count = np.zeros((n_rows, len(MAP_NAMES), len(K_VALUES)), dtype=np.int32)
    nearest_rank = np.zeros((n_rows, len(SCORE_MAP_NAMES)), dtype=np.int32)
    nearest_distance = np.full((n_rows, len(SCORE_MAP_NAMES)), np.nan, dtype=np.float32)
    nearest_margin = np.full((n_rows, len(SCORE_MAP_NAMES)), np.nan, dtype=np.float32)
    top1_score = np.full((n_rows, len(SCORE_MAP_NAMES)), np.nan, dtype=np.float32)
    boundary_gap = np.full(
        (n_rows, len(SCORE_MAP_NAMES), len(K_VALUES)), np.nan, dtype=np.float32
    )
    gt_yx = np.full((n_rows, 2), np.nan, dtype=np.float32)
    parity_fused = []
    parity_p = []
    parity_v = []
    parity_q = []
    pool_ordered_diff = []
    pool_set_diff = []
    audit_fused_top16_ordered_diff = []
    audit_fused_top16_set_diff = []
    processed = np.zeros(n_rows, dtype=bool)

    for clip in clips:
        sequence, start_s = clip.split(':')
        window_start = int(start_s)
        sequence_path = POD_ROOT / sequence
        annotation = np.load(sequence_path / 'anno.npz', allow_pickle=True, mmap_mode='r')
        trajectories = np.asarray(
            annotation['trajs_2d'][window_start:window_start + LENGTH], dtype=np.float32
        )
        visibility = np.asarray(
            annotation['visibs'][window_start:window_start + LENGTH], dtype=bool
        )
        valid = (
            np.asarray(annotation['valids'][window_start:window_start + LENGTH], dtype=bool)
            if 'valids' in annotation.files
            else np.ones_like(visibility, dtype=bool)
        )
        eligible = np.where(
            visibility[0]
            & valid[0]
            & (np.sum(visibility & valid, axis=0) >= 24)
        )[0]
        rng = np.random.default_rng(stable_seed(sequence, window_start))
        chosen = np.sort(rng.choice(eligible, NQ, replace=False))
        selected_gt = trajectories[:, chosen]
        frame_paths = [
            sequence_path / 'rgbs' / f'rgb_{window_start + t:05d}.jpg'
            for t in range(LENGTH)
        ]
        width, height = Image.open(frame_paths[0]).size
        query_xy = np.empty((NQ, 2), dtype=np.float32)
        query_xy[:, 0] = selected_gt[0, :, 0] * UNIFIED_DENOM / float(width - 1)
        query_xy[:, 1] = selected_gt[0, :, 1] * UNIFIED_DENOM / float(height - 1)
        queries = torch.from_numpy(query_xy).float().to(device)
        support = get_points_on_a_grid(20, (256, 256), device).squeeze(0)
        combined = torch.cat([queries, support], dim=0)
        predictor.reset()
        predictor.initial_capacity = len(combined)

        clip_targets = {
            (frame_t, query_i): local_index[key_to_index[(clip, frame_t, query_i)]]
            for clip_id, frame_t, query_i in row_keys
            if clip_id == clip and key_to_index[(clip_id, frame_t, query_i)] in local_index
        }
        max_t = max(frame_t for frame_t, _ in clip_targets)
        clip_start = time.perf_counter()
        for t in range(max_t + 1):
            frame = load_frame(frame_paths[t], device)
            f4, f8, f16, f32, fused_features = model.extract_frame_features(frame)
            if t == 0:
                predictor.init_queries((fused_features, device), combined, 256, 256)
            active_q = predictor.q_init[: predictor.N]
            active_mask = predictor.temporal_mask[: predictor.N]
            active_memory = predictor.point_memory[: predictor.N]
            frame_features = (f4, f8, f16, f32, fused_features)
            p, v_logit, q_new, diag = track_frame_diag(
                model,
                active_q,
                active_mask,
                active_memory,
                frame_features,
                256,
                256,
            )
            components = correlation_components(
                model,
                diag['q_pre'].unsqueeze(0),
                f4,
                f8,
                f16,
                f32,
            )
            parity_fused.append(
                float(torch.max(torch.abs(components['fused'] - diag['c1'])))
            )
            if not parity_p:
                p_ref, v_ref, q_ref = model.track_frame(
                    active_q,
                    active_mask,
                    active_memory,
                    frame_features,
                    256,
                    256,
                )
                parity_p.append(float(torch.max(torch.abs(p - p_ref))))
                parity_v.append(float(torch.max(torch.abs(v_logit - v_ref))))
                parity_q.append(float(torch.max(torch.abs(q_new - q_ref))))

            for query_i in range(NQ):
                key = (t, query_i)
                if key not in clip_targets:
                    continue
                out_i = clip_targets[key]
                global_i = int(selected_indices[out_i])
                gt_xy = selected_gt[t, query_i]
                gt_norm = np.asarray(
                    [gt_xy[1] / float(height - 1), gt_xy[0] / float(width - 1)],
                    dtype=np.float32,
                )
                gt_yx[out_i] = gt_norm

                orders: dict[str, np.ndarray] = {}
                for score_map_i, map_name in enumerate(SCORE_MAP_NAMES):
                    score_tensor = components[map_name][query_i].float()
                    maximum = min(int(K_VALUES[-1]), int(score_tensor.numel()))
                    audit_count = min(maximum + 1, int(score_tensor.numel()))
                    topk_result = torch.topk(
                        score_tensor,
                        k=audit_count,
                        dim=0,
                        largest=True,
                        sorted=True,
                    )
                    order = topk_result.indices[:maximum].detach().cpu().numpy()
                    ordered_values = topk_result.values.detach().cpu().float().numpy()
                    for ki, k in enumerate(K_VALUES.tolist()):
                        if int(k) >= len(ordered_values):
                            raise RuntimeError(
                                f'cannot audit K boundary for {map_name}: K={k}, values={len(ordered_values)}'
                            )
                        boundary_gap[out_i, score_map_i, ki] = float(
                            ordered_values[int(k) - 1] - ordered_values[int(k)]
                        )
                    scores = score_tensor.detach().cpu().numpy()
                    values = grid_coords[map_name]
                    (
                        map_min_error,
                        map_candidate_count,
                        map_nearest_rank,
                        map_nearest_distance,
                        map_margin,
                        map_top1_score,
                        order,
                    ) = evaluate_score_map(scores, values, gt_norm, order)
                    map_i = MAP_NAMES.index(map_name)
                    min_error[out_i, map_i] = map_min_error
                    candidate_count[out_i, map_i] = map_candidate_count
                    nearest_rank[out_i, score_map_i] = map_nearest_rank
                    nearest_distance[out_i, score_map_i] = map_nearest_distance
                    nearest_margin[out_i, score_map_i] = map_margin
                    top1_score[out_i, score_map_i] = map_top1_score
                    orders[map_name] = order

                union_map_i = MAP_NAMES.index('union_raw_equal')
                for ki, k in enumerate(K_VALUES.tolist()):
                    per_scale = int(math.ceil(float(k) / 4.0))
                    coords = np.concatenate(
                        [
                            grid_coords[name][orders[name][: min(per_scale, len(orders[name]))]]
                            for name in ('c4', 'c8', 'c16', 'c32')
                        ],
                        axis=0,
                    )
                    coords = unique_coords(coords)
                    errors = np.linalg.norm((coords - gt_norm[None, :]) * UNIFIED_DENOM, axis=1)
                    min_error[out_i, union_map_i, ki] = float(np.min(errors))
                    candidate_count[out_i, union_map_i, ki] = int(len(coords))

                c1_coords = model_xy_to_unified_yx_norm(diag['p_topk'][query_i], model)
                audit_fused_coords = grid_coords['fused'][orders['fused'][:16]]
                audit_fused_top16_ordered_diff.append(
                    float(np.max(np.abs(audit_fused_coords - c1_coords)))
                )
                audit_distance = np.linalg.norm(
                    audit_fused_coords[:, None, :] - c1_coords[None, :, :], axis=-1
                )
                audit_fused_top16_set_diff.append(
                    float(max(
                        np.min(audit_distance, axis=1).max(),
                        np.min(audit_distance, axis=0).max(),
                    ))
                )
                pool_coords = pool['candidates'][global_i, 2:18].astype(np.float32)
                pool_ordered_diff.append(float(np.max(np.abs(c1_coords - pool_coords))))
                distance = np.linalg.norm(
                    c1_coords[:, None, :] - pool_coords[None, :, :], axis=-1
                )
                pool_set_diff.append(
                    float(max(np.min(distance, axis=1).max(), np.min(distance, axis=0).max()))
                )
                processed[out_i] = True

            write_mask = torch.ones(predictor.N, dtype=torch.bool, device=device)
            updated_memory, updated_mask = model._update_point_memory(
                active_memory,
                active_mask,
                q_new,
                write_mask,
            )
            predictor.point_memory[: predictor.N] = updated_memory
            predictor.temporal_mask[: predictor.N] = updated_mask
            predictor.t += 1

        print(
            json.dumps(
                {
                    'clip': clip,
                    'targets': int(len(clip_targets)),
                    'max_t': int(max_t),
                    'seconds': float(time.perf_counter() - clip_start),
                }
            ),
            flush=True,
        )

    if not np.all(processed):
        missing = np.where(~processed)[0].tolist()[:20]
        raise RuntimeError(f'missing audited rows: {missing}')
    if not all(
        np.all(np.isfinite(value))
        for value in [min_error, nearest_distance, nearest_margin, top1_score, boundary_gap, gt_yx]
    ):
        raise RuntimeError('non-finite audit output')

    fused_map_i = MAP_NAMES.index('fused')
    k16_i = int(np.where(K_VALUES == 16)[0][0])
    frozen_c1_oracle = np.min(pool['error_px'][selected_indices, 2:18].astype(np.float32), axis=1)
    fused_k16_pool_error_max_abs = float(
        np.max(np.abs(min_error[:, fused_map_i, k16_i] - frozen_c1_oracle))
    )
    min_error_monotonic = bool(np.all(np.diff(min_error, axis=2) <= 1e-5))
    candidate_count_monotonic = bool(np.all(np.diff(candidate_count, axis=2) >= 0))
    ms_corr_weight = model.ms_corr_proj.weight.detach().cpu().float().numpy().reshape(-1)
    ms_corr_bias = model.ms_corr_proj.bias.detach().cpu().float().numpy().reshape(-1)
    integrity = {
        'input_audit': input_audit,
        'rows_processed': int(np.sum(processed)),
        'fused_recompute_max_abs': float(np.max(parity_fused)),
        'official_p_max_abs': float(np.max(parity_p)),
        'official_v_max_abs': float(np.max(parity_v)),
        'official_q_max_abs': float(np.max(parity_q)),
        'pool_c1_ordered_max_abs': float(np.max(pool_ordered_diff)),
        'pool_c1_set_hausdorff_max': float(np.max(pool_set_diff)),
        'audit_fused_top16_official_ordered_max_abs': float(
            np.max(audit_fused_top16_ordered_diff)
        ),
        'audit_fused_top16_official_set_hausdorff_max': float(
            np.max(audit_fused_top16_set_diff)
        ),
        'fused_k16_pool_oracle_error_max_abs': fused_k16_pool_error_max_abs,
        'min_error_monotonic_in_k': min_error_monotonic,
        'candidate_count_monotonic_in_k': candidate_count_monotonic,
        'ms_corr_proj_weight_order_c4_c8_c16_c32': ms_corr_weight.tolist(),
        'ms_corr_proj_bias': ms_corr_bias.tolist(),
    }
    if integrity['fused_recompute_max_abs'] > 1e-6:
        raise RuntimeError(f'fused correlation parity failed: {integrity}')
    if max(
        integrity['official_p_max_abs'],
        integrity['official_v_max_abs'],
        integrity['official_q_max_abs'],
    ) > 1e-6:
        raise RuntimeError(f'official forward parity failed: {integrity}')
    if integrity['pool_c1_set_hausdorff_max'] > 1e-6:
        raise RuntimeError(f'canonical pool C1 set parity failed: {integrity}')
    if integrity['audit_fused_top16_official_set_hausdorff_max'] > 1e-6:
        raise RuntimeError(f'audit fused top16/official top16 parity failed: {integrity}')
    if integrity['fused_k16_pool_oracle_error_max_abs'] > 1e-4:
        raise RuntimeError(f'fused K16/pool oracle-error parity failed: {integrity}')
    if not integrity['min_error_monotonic_in_k']:
        raise RuntimeError('minimum candidate error is not monotonic in K')
    if not integrity['candidate_count_monotonic_in_k']:
        raise RuntimeError('candidate count is not monotonic in K')

    sequence = pool['sequence'][selected_indices].astype(str)
    hard = pool['is_hard'][selected_indices].astype(bool)
    masks = {
        'all': np.ones(n_rows, dtype=bool),
        'hard': hard,
        'easy': ~hard,
    }
    for name in SEQUENCES:
        masks[f'sequence_{name}'] = sequence == name
    summaries = {
        name: summarize_subset(
            min_error,
            candidate_count,
            nearest_rank,
            nearest_distance,
            nearest_margin,
            boundary_gap,
            mask,
        )
        for name, mask in masks.items()
        if np.any(mask)
    }
    gate = compute_gate(summaries) if args.max_clips == 0 else {
        'smoke_only': True,
        'decision': 'SMOKE_ONLY: full three-sequence headroom gate not evaluated.',
    }

    result = {
        'script': 'scripts/v9a5c0_candidate_recall_correlation_oracle.py',
        'date': '2026-07-10',
        'protocol': {
            'dataset': 'PointOdyssey canonical balanced pool',
            'rows': int(n_rows),
            'clips': int(len(clips)),
            'clip_ids': clips,
            'device': str(device),
            'k_values': K_VALUES.tolist(),
            'radii_px': RADII.tolist(),
            'map_names': list(MAP_NAMES),
            'union_policy': (
                'diagnostic raw-cosine top ceil(K/4) from each native scale; '
                'exact-coordinate deduplication; not used by the primary gate'
            ),
            'native_scale_semantics': (
                'raw normalized cosine maps; ms_corr_proj sign/weight is not applied'
            ),
            'training': False,
            'davis_read': False,
            'threshold_tuning': False,
            'seconds': float(time.perf_counter() - start_all),
        },
        'integrity': integrity,
        'summaries': summaries,
        'gate': gate,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_doc.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    np.savez_compressed(
        args.out_npz,
        selected_indices=selected_indices,
        clip_id=pool['clip_id'][selected_indices],
        sequence=sequence,
        frame_tau=pool['frame_tau'][selected_indices],
        query_idx=pool['query_idx'][selected_indices],
        is_hard=hard,
        gt_yx=gt_yx,
        map_names=np.asarray(MAP_NAMES, dtype=object),
        score_map_names=np.asarray(SCORE_MAP_NAMES, dtype=object),
        k_values=K_VALUES,
        radii=RADII,
        min_error=min_error,
        candidate_count=candidate_count,
        recall=(min_error[..., None] <= RADII[None, None, None, :]).astype(np.int8),
        nearest_rank=nearest_rank,
        nearest_distance=nearest_distance,
        nearest_margin=nearest_margin,
        top1_score=top1_score,
        boundary_gap=boundary_gap,
    )
    write_doc(result, args.out_doc)
    print(
        json.dumps(
            {
                'ok': True,
                'json': str(args.out_json),
                'npz': str(args.out_npz),
                'doc': str(args.out_doc),
                'rows': int(n_rows),
                'integrity': {
                    key: value
                    for key, value in integrity.items()
                    if key != 'input_audit'
                },
                'gate': gate,
            },
            ensure_ascii=False,
        )
    )


if __name__ == '__main__':
    main()
