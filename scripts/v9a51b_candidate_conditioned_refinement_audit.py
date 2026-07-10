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
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

WORKTREE_ROOT = Path('/gemini/code/FSPT_v9a51b_clean')
SOURCE_ROOT = Path('/gemini/code/FSPT')
TRACKON_ROOT = WORKTREE_ROOT / 'baselines/track_on'
POD_ROOT = SOURCE_ROOT / 'datasets/pointodyssey/train'
CHECKPOINT = SOURCE_ROOT / 'baselines/track_on/checkpoints_trackon2_dinov3.pt'
CONFIG = TRACKON_ROOT / 'config/test.yaml'
MANIFEST = WORKTREE_ROOT / 'docs/v9a51b_input_manifest_2026-07-10.json'
OUTDIR = WORKTREE_ROOT / 'outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement'
DEFAULT_JSON = OUTDIR / 'v9a51b_candidate_conditioned_refinement.json'
DEFAULT_NPZ = OUTDIR / 'v9a51b_candidate_conditioned_refinement_rows.npz'
DEFAULT_DOC = WORKTREE_ROOT / 'docs/v9a51b_candidate_conditioned_refinement_result_2026-07-10.md'

EXPECTED_HEAD = 'd8e5ca9fb190ce57a3a35725603447a89a177ce3'
EXPECTED_BRANCH = 'v9a51b-candidate-conditioned-refinement-20260710'
SEQUENCES = ('ani', 'animal3', 'r4_new_f')
CLIP_STARTS = (0, 256, 512)
LENGTH = 96
NQ = 32
K16 = 16
K64 = 64
UNIFIED_DENOM = 255.0
TOL = 1e-6
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 20260713

if str(TRACKON_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKON_ROOT))
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
    checked = []
    for row in manifest['files']:
        path = Path(row['path'])
        if not path.exists():
            raise RuntimeError(f'missing manifest input: {path}')
        size = path.stat().st_size
        digest = sha256(path)
        if size != int(row['size_bytes']) or digest != row['sha256']:
            raise RuntimeError(f'input hash/size mismatch: {path}')
        checked.append({'path': str(path), 'size_bytes': int(size), 'sha256': digest})

    frame_report = None
    meta = manifest['frame_manifest']
    frame_manifest_path = Path(meta['path'])
    if str(frame_manifest_path.resolve()) not in by_path:
        raise RuntimeError('frame manifest is absent from input manifest')
    if verify_frames:
        frame_manifest = json.loads(frame_manifest_path.read_text())
        root = Path(frame_manifest['root'])
        aggregate = hashlib.sha256()
        total = 0
        for row in frame_manifest['files']:
            relative = str(row['relative_path'])
            path = root / relative
            if not path.exists():
                raise RuntimeError(f'missing frame: {path}')
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
            raise RuntimeError('frame byte count mismatch')
        if aggregate_digest != meta['aggregate_sha256']:
            raise RuntimeError('frame aggregate mismatch')
        frame_report = {
            'path': str(frame_manifest_path),
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
        ['git', '-C', str(WORKTREE_ROOT), 'status', '--porcelain', '--untracked-files=no'],
        text=True,
    ).strip()
    if head != EXPECTED_HEAD or head != manifest['clean_head']:
        raise RuntimeError(f'HEAD mismatch: {head}')
    if branch != EXPECTED_BRANCH or branch != manifest['clean_branch']:
        raise RuntimeError(f'branch mismatch: {branch}')
    if tracked:
        raise RuntimeError(f'tracked worktree is dirty: {tracked}')

    imported = {}
    for name in [
        'model.trackon_predictor',
        'model.trackon',
        'model.reranking',
        'model.prediction_head',
        'utils.coord_utils',
        'utils.train_utils',
    ]:
        module = sys.modules.get(name)
        value = '' if module is None else str(Path(module.__file__).resolve())
        imported[name] = value
        if not value.startswith(str(TRACKON_ROOT.resolve()) + '/'):
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
def track_frame_diag(
    model,
    q_init: torch.Tensor,
    temporal_mask: torch.Tensor,
    point_memory: torch.Tensor,
    frame_features: tuple[torch.Tensor, ...],
    h_in: int,
    w_in: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
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
    q_rerank, p_topk, u_topk, s_topk = model.reranking_head(
        q_pre, f4, f8, f16, f32, c1
    )
    q_new = model.projection2(q_rerank)
    c2 = model.multiscale_correlation(q_new, f4, f8, f16, f32)
    p_patch = indices_to_coords(
        torch.argmax(c2, dim=-1).unsqueeze(1), model.input_size, model.stride
    ).squeeze(1)
    offsets, v_logit, u_logit = model.prediction_head(
        q_new, f4, f8, f16, f32, p_patch
    )
    p_model = p_patch[0] + offsets[-1]
    p = p_model.clone()
    p[..., 0] = (p[..., 0] / model.W) * w_in
    p[..., 1] = (p[..., 1] / model.H) * h_in
    return p, v_logit, q_new.squeeze(0), {
        'q_pre': q_pre.squeeze(0),
        'c1': c1.squeeze(0),
        'p_topk': p_topk.squeeze(0),
        'u_topk': u_topk.squeeze(0),
        's_topk': s_topk.squeeze(0),
        'u_logit': u_logit,
    }


def model_xy_to_yx_norm(xy: torch.Tensor | np.ndarray, model) -> np.ndarray:
    if isinstance(xy, torch.Tensor):
        value = xy.detach().cpu().float().numpy().copy()
    else:
        value = np.asarray(xy, dtype=np.float32).copy()
    value[..., 0] *= 256.0 / float(model.W)
    value[..., 1] *= 256.0 / float(model.H)
    return (value[..., [1, 0]] / UNIFIED_DENOM).astype(np.float32)


def output_xy_to_yx_norm(xy: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(xy, torch.Tensor):
        value = xy.detach().cpu().float().numpy()
    else:
        value = np.asarray(xy, dtype=np.float32)
    return (value[..., [1, 0]] / UNIFIED_DENOM).astype(np.float32)


@torch.no_grad()
def extended_candidates(
    model,
    q_pre: torch.Tensor,
    frame_features: tuple[torch.Tensor, ...],
    c1: torch.Tensor,
    k: int = K64,
) -> dict[str, torch.Tensor]:
    f4, f8, f16, f32, _ = frame_features
    head = model.reranking_head
    q = q_pre.unsqueeze(0)
    n_queries = q.shape[1]
    d_model = int(head.D)
    top = torch.topk(c1.unsqueeze(0), k=k, dim=-1, largest=True, sorted=True)
    positions = indices_to_coords(top.indices, head.size, head.stride)
    positions_norm = positions / torch.tensor(
        [head.size[1], head.size[0]], device=q.device, dtype=positions.dtype
    )
    positions_norm = torch.clamp(positions_norm, 0, 1)
    positions_norm = positions_norm.view(1, n_queries * k, 1, 2)
    positions_norm = positions_norm.expand(-1, -1, head.num_level, -1)
    feature_scales = torch.cat([f4, f8, f16, f32], dim=1)
    decoded = q.unsqueeze(2).expand(-1, -1, k, -1).reshape(
        1, n_queries * k, d_model
    )
    for layer in head.local_decoder:
        decoded = layer(
            q=decoded,
            k=feature_scales,
            v=feature_scales,
            reference_points=positions_norm,
            spatial_shapes=head.spatial_shapes,
            start_levels=head.start_levels,
        )
    decoded = decoded.view(1, n_queries, k, d_model)
    pre_fusion = torch.cat(
        [decoded, q.unsqueeze(2).expand(-1, -1, k, -1)], dim=-1
    )
    post_fusion = head.fusion_layer(pre_fusion)
    score = head.score_layer(post_fusion).squeeze(-1).squeeze(0)
    certainty = head.certainty_layer(post_fusion).squeeze(-1).squeeze(0)
    return {
        'positions_model_xy': positions.squeeze(0),
        'correlation_values': top.values.squeeze(0),
        'correlation_indices': top.indices.squeeze(0),
        'post_fusion': post_fusion.squeeze(0),
        'score': score,
        'certainty': certainty,
    }


@torch.no_grad()
def candidate_conditioned_refinement(
    model,
    q_pre: torch.Tensor,
    post_fusion: torch.Tensor,
    frame_features: tuple[torch.Tensor, ...],
) -> dict[str, torch.Tensor]:
    f4, f8, f16, f32, _ = frame_features
    n_queries, k, d_model = post_fusion.shape
    q_base = q_pre[:, None, :].expand(n_queries, k, d_model).reshape(
        n_queries * k, 1, d_model
    )
    candidate = post_fusion.reshape(n_queries * k, 1, d_model)
    q_conditioned = model.reranking_head.fusion(
        q_base, candidate, candidate
    )
    q_conditioned = model.reranking_head.final_projection_layer(
        torch.cat([q_conditioned, q_base], dim=-1)
    )
    q2 = model.projection2(q_conditioned.squeeze(1).unsqueeze(0))
    c2 = model.multiscale_correlation(q2, f4, f8, f16, f32)
    c2_index = torch.argmax(c2, dim=-1)
    p_patch = indices_to_coords(
        c2_index.unsqueeze(1), model.input_size, model.stride
    ).squeeze(1)
    offsets, v_logit, u_logit = model.prediction_head(
        q2, f4, f8, f16, f32, p_patch
    )
    refined = (p_patch[0] + offsets[-1]).view(n_queries, k, 2)
    return {
        'refined_model_xy': refined,
        'c2_index': c2_index.view(n_queries, k),
        'v_logit': v_logit.view(n_queries, k),
        'u_logit': u_logit.view(n_queries, k),
    }


def build_visibility_subsets(visible: np.ndarray) -> dict[str, np.ndarray]:
    frames, queries = visible.shape
    reentry_first = np.zeros_like(visible, dtype=bool)
    reentry_early4 = np.zeros_like(visible, dtype=bool)
    reentry_early8 = np.zeros_like(visible, dtype=bool)
    occ_length = np.zeros_like(visible, dtype=np.int16)
    for query in range(queries):
        occlusion_run = 0
        post_reentry_visible_count = 0
        last_occ_length = 0
        for frame in range(frames):
            if bool(visible[frame, query]):
                is_reentry = frame > 0 and not bool(visible[frame - 1, query])
                if is_reentry:
                    reentry_first[frame, query] = True
                    last_occ_length = occlusion_run
                    occ_length[frame, query] = int(occlusion_run)
                    post_reentry_visible_count = 1
                elif post_reentry_visible_count > 0:
                    post_reentry_visible_count += 1
                    occ_length[frame, query] = int(last_occ_length)
                if 1 <= post_reentry_visible_count <= 4:
                    reentry_early4[frame, query] = True
                if 1 <= post_reentry_visible_count <= 8:
                    reentry_early8[frame, query] = True
                occlusion_run = 0
            else:
                occlusion_run += 1
                post_reentry_visible_count = 0
                last_occ_length = 0
    return {
        'reentry_first': reentry_first,
        'reentry_early4': reentry_early4,
        'reentry_early8': reentry_early8,
        'occ_length': occ_length,
    }


def spatial_cluster_count(coords_yx: np.ndarray, radius_px: float = 4.0) -> int:
    centers: list[np.ndarray] = []
    for coord in np.asarray(coords_yx, dtype=np.float32):
        if not centers:
            centers.append(coord)
            continue
        distance = np.linalg.norm(
            (np.asarray(centers) - coord[None, :]) * UNIFIED_DENOM,
            axis=1,
        )
        if float(np.min(distance)) > radius_px:
            centers.append(coord)
    return int(len(centers))


def metric_summary(
    error: np.ndarray,
    official_error: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    chosen = np.asarray(error, dtype=np.float64)[mask]
    official = np.asarray(official_error, dtype=np.float64)[mask]
    if len(chosen) == 0:
        return {
            'n': 0,
            'mean_error': None,
            'median_error': None,
            'safe4': 0,
            'safe8': 0,
            'safe16': 0,
            'mean_difference_vs_official': None,
            'median_difference_vs_official': None,
            'better': 0,
            'worse': 0,
            'equal': 0,
        }
    difference = chosen - official
    better = int(np.sum(difference < -TOL))
    worse = int(np.sum(difference > TOL))
    return {
        'n': int(len(chosen)),
        'mean_error': float(np.mean(chosen)),
        'median_error': float(np.median(chosen)),
        'safe4': int(np.sum(chosen <= 4.0)),
        'safe8': int(np.sum(chosen <= 8.0)),
        'safe16': int(np.sum(chosen <= 16.0)),
        'mean_difference_vs_official': float(np.mean(difference)),
        'median_difference_vs_official': float(np.median(difference)),
        'better': better,
        'worse': worse,
        'equal': int(len(chosen) - better - worse),
    }


def clip_bootstrap(
    clip_id: np.ndarray,
    visible: np.ndarray,
    official_error: np.ndarray,
    readouts: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    clips = sorted(set(clip_id.tolist()))
    names = list(readouts)
    matrix = np.zeros((len(names), len(clips)), dtype=np.float64)
    per_clip: dict[str, dict[str, float]] = {name: {} for name in names}
    for name_i, name in enumerate(names):
        difference = np.asarray(readouts[name], dtype=np.float64) - official_error
        for clip_i, clip in enumerate(clips):
            mask = visible & (clip_id == clip)
            value = float(np.mean(difference[mask]))
            matrix[name_i, clip_i] = value
            per_clip[name][clip] = value
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty((len(names), BOOTSTRAP_RESAMPLES), dtype=np.float64)
    chunk = 1000
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        count = min(chunk, BOOTSTRAP_RESAMPLES - start)
        indices = rng.integers(0, len(clips), size=(count, len(clips)))
        bootstrap[:, start:start + count] = np.mean(matrix[:, indices], axis=2)
    report = {}
    for name_i, name in enumerate(names):
        values = matrix[name_i]
        boot = bootstrap[name_i]
        report[name] = {
            'n_clips': int(len(clips)),
            'clip_mean_difference': float(np.mean(values)),
            'clip_median_difference': float(np.median(values)),
            'clip_better': int(np.sum(values < -TOL)),
            'clip_worse': int(np.sum(values > TOL)),
            'clip_equal': int(np.sum(np.abs(values) <= TOL)),
            'bootstrap_resamples': int(BOOTSTRAP_RESAMPLES),
            'bootstrap_seed': int(BOOTSTRAP_SEED),
            'mean_difference_95_ci': [
                float(value) for value in np.quantile(boot, [0.025, 0.975])
            ],
            'probability_mean_lt_zero': float(np.mean(boot < 0.0)),
        }
    return report, per_clip


def write_doc(result: dict[str, Any], path: Path) -> None:
    summaries = result['summaries']
    bootstrap = result['clip_bootstrap']
    display = [
        'official_final',
        'raw_score_top1_K16',
        'raw_score_top1_K64',
        'raw_oracle_K16',
        'raw_oracle_K64',
        'refined_score_top1_K16',
        'refined_score_top1_K64',
        'refined_oracle_K16',
        'refined_oracle_K64',
        'hybrid_refined_score_top1',
        'hybrid_refined_oracle',
    ]
    lines = [
        '# V9-A5.1b Candidate-Conditioned Refinement Audit Result',
        '',
        'Date: 2026-07-10',
        '',
        'No training and no DAVIS data were used. Official TrackOn2 memory remained on the original q_new path.',
        '',
        '## Global visible-frame results',
        '',
        '| Readout | Mean | Median | Safe4/8/16 | Better/Worse/Equal | Mean Δ | Clip 95% CI |',
        '|---|---:|---:|---:|---:|---:|---|',
    ]
    for name in display:
        row = summaries[name]['all_visible']
        interval = [0.0, 0.0] if name == 'official_final' else bootstrap[name]['mean_difference_95_ci']
        lines.append(
            f"| {name} | {row['mean_error']:.4f} | {row['median_error']:.4f} | "
            f"{row['safe4']}/{row['safe8']}/{row['safe16']} | "
            f"{row['better']}/{row['worse']}/{row['equal']} | "
            f"{row['mean_difference_vs_official']:+.4f} | "
            f"[{interval[0]:+.4f},{interval[1]:+.4f}] |"
        )

    lines += [
        '',
        '## Primary hybrid refined oracle',
        '',
        '| Subset | Official mean | Hybrid oracle mean | N |',
        '|---|---:|---:|---:|',
    ]
    for subset in [
        'all_visible',
        'risk_visible',
        'nonrisk_visible',
        'hard_official',
        'opportunity',
        'reentry_first',
        'reentry_early4',
        'reentry_early8',
    ]:
        official = summaries['official_final'][subset]
        hybrid = summaries['hybrid_refined_oracle'][subset]
        official_mean = float('nan') if official['mean_error'] is None else official['mean_error']
        hybrid_mean = float('nan') if hybrid['mean_error'] is None else hybrid['mean_error']
        lines.append(
            f'| {subset} | {official_mean:.4f} | {hybrid_mean:.4f} | {official["n"]} |'
        )

    lines += [
        '',
        '## Risk activation',
        '',
        '```json',
        json.dumps(result['risk_audit'], indent=2),
        '```',
        '',
        '## Refinement diagnostics',
        '',
        '```json',
        json.dumps(result['refinement_diagnostics'], indent=2),
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
    parser.add_argument('--max-clips', type=int, default=0)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--skip-full-frame-hash', action='store_true')
    parser.add_argument('--out-json', type=Path, default=DEFAULT_JSON)
    parser.add_argument('--out-npz', type=Path, default=DEFAULT_NPZ)
    parser.add_argument('--out-doc', type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()

    if args.max_frames < 0 or args.max_frames > LENGTH:
        raise SystemExit(f'--max-frames must be in [0,{LENGTH}]')
    frames_to_run = LENGTH if args.max_frames == 0 else args.max_frames
    if frames_to_run < 2:
        raise SystemExit('at least two frames are required')

    start_all = time.perf_counter()
    input_audit = verify_inputs(verify_frames=not args.skip_full_frame_hash)
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
    if int(model.K) != K16:
        raise RuntimeError(f'unexpected official K: {model.K}')
    if str(model.memory_update_policy) != 'unconditional':
        raise RuntimeError(f'unexpected memory policy: {model.memory_update_policy}')
    if abs(float(model.delta_v) - 0.8) > 1e-12:
        raise RuntimeError(f'unexpected delta_v: {model.delta_v}')

    clips = [f'{sequence}:{start}' for sequence in SEQUENCES for start in CLIP_STARTS]
    if args.max_clips > 0:
        clips = clips[: args.max_clips]

    scalar: dict[str, list[Any]] = defaultdict(list)
    raw_error_rows: list[np.ndarray] = []
    refined_error_rows: list[np.ndarray] = []
    score_rows: list[np.ndarray] = []
    displacement_rows: list[np.ndarray] = []
    refined_c2_index_rows: list[np.ndarray] = []
    refined_out_of_range_rows: list[np.ndarray] = []
    cluster16_rows: list[int] = []
    cluster64_rows: list[int] = []

    parity: dict[str, list[float]] = {
        'official_p_max_abs': [],
        'official_v_max_abs': [],
        'official_q_max_abs': [],
        'extended_first16_ordered_position_max_abs': [],
        'extended_first16_set_hausdorff_max': [],
        'extended_first16_matched_score_max_abs': [],
        'extended_first16_matched_certainty_max_abs': [],
    }
    runtime = []

    for clip_id in clips:
        sequence, start_s = clip_id.split(':')
        start = int(start_s)
        sequence_dir = POD_ROOT / sequence
        annotation = np.load(sequence_dir / 'anno.npz', allow_pickle=True, mmap_mode='r')
        trajectories = np.asarray(
            annotation['trajs_2d'][start : start + LENGTH], dtype=np.float32
        )
        raw_visibility = np.asarray(
            annotation['visibs'][start : start + LENGTH], dtype=bool
        )
        valid = (
            np.asarray(annotation['valids'][start : start + LENGTH], dtype=bool)
            if 'valids' in annotation.files
            else np.ones_like(raw_visibility, dtype=bool)
        )
        finite = np.isfinite(trajectories).all(axis=2)
        visible_all = raw_visibility & valid & finite
        eligible = np.where(
            visible_all[0] & (np.sum(visible_all, axis=0) >= 24)
        )[0]
        chosen = np.sort(
            np.random.default_rng(stable_seed(sequence, start)).choice(
                eligible, NQ, replace=False
            )
        )
        gt_xy = trajectories[:, chosen]
        visible = visible_all[:, chosen]
        subset_flags = build_visibility_subsets(visible)
        frame_paths = [
            sequence_dir / 'rgbs' / f'rgb_{start + frame:05d}.jpg'
            for frame in range(frames_to_run)
        ]
        width, height = Image.open(frame_paths[0]).size
        query_xy = np.empty((NQ, 2), dtype=np.float32)
        query_xy[:, 0] = gt_xy[0, :, 0] * UNIFIED_DENOM / float(width - 1)
        query_xy[:, 1] = gt_xy[0, :, 1] * UNIFIED_DENOM / float(height - 1)
        queries = torch.from_numpy(query_xy).float().to(device)
        support = get_points_on_a_grid(20, (256, 256), device).squeeze(0)
        combined = torch.cat([queries, support], dim=0)

        predictor.reset()
        predictor.initial_capacity = len(combined)
        clip_start = time.perf_counter()

        for frame in range(frames_to_run):
            image = load_frame(frame_paths[frame], device)
            frame_features = model.extract_frame_features(image)
            if frame == 0:
                predictor.init_queries((frame_features[-1], device), combined, 256, 256)
            active_q = predictor.q_init[: predictor.N]
            active_mask = predictor.temporal_mask[: predictor.N]
            active_memory = predictor.point_memory[: predictor.N]
            p, v_logit, q_new, diag = track_frame_diag(
                model,
                active_q,
                active_mask,
                active_memory,
                frame_features,
                256,
                256,
            )
            extended = extended_candidates(
                model,
                diag['q_pre'][:NQ],
                frame_features,
                diag['c1'][:NQ],
                k=K64,
            )
            refined = candidate_conditioned_refinement(
                model,
                diag['q_pre'][:NQ],
                extended['post_fusion'],
                frame_features,
            )

            official_positions = model_xy_to_yx_norm(diag['p_topk'][:NQ], model)
            extended_positions = model_xy_to_yx_norm(
                extended['positions_model_xy'], model
            )
            extended_top16 = extended_positions[:, :K16]
            official_scores = diag['s_topk'][:NQ].detach().cpu().float().numpy()
            official_certainty = diag['u_topk'][:NQ].detach().cpu().float().numpy()
            extended_scores = extended['score'].detach().cpu().float().numpy()
            extended_certainty = extended['certainty'].detach().cpu().float().numpy()
            official_to_extended = np.full((NQ, K16), -1, dtype=np.int32)
            ordered_difference = float(
                np.max(np.abs(extended_top16 - official_positions))
            )
            set_hausdorff = 0.0
            matched_score_difference = 0.0
            matched_certainty_difference = 0.0
            for query in range(NQ):
                distance = np.linalg.norm(
                    extended_top16[query, :, None, :]
                    - official_positions[query, None, :, :],
                    axis=-1,
                )
                ext_for_official = np.argmin(distance, axis=0)
                official_for_ext = np.argmin(distance, axis=1)
                if (
                    len(np.unique(ext_for_official)) != K16
                    or len(np.unique(official_for_ext)) != K16
                ):
                    raise RuntimeError('non-bijective K64/K16 coordinate matching')
                official_to_extended[query] = ext_for_official.astype(np.int32)
                set_hausdorff = max(
                    set_hausdorff,
                    float(
                        max(
                            np.min(distance, axis=0).max(),
                            np.min(distance, axis=1).max(),
                        )
                    ),
                )
                matched_score_difference = max(
                    matched_score_difference,
                    float(
                        np.max(
                            np.abs(
                                extended_scores[query, ext_for_official]
                                - official_scores[query]
                            )
                        )
                    ),
                )
                matched_certainty_difference = max(
                    matched_certainty_difference,
                    float(
                        np.max(
                            np.abs(
                                extended_certainty[query, ext_for_official]
                                - official_certainty[query]
                            )
                        )
                    ),
                )
            parity['extended_first16_ordered_position_max_abs'].append(
                ordered_difference
            )
            parity['extended_first16_set_hausdorff_max'].append(set_hausdorff)
            parity['extended_first16_matched_score_max_abs'].append(
                matched_score_difference
            )
            parity['extended_first16_matched_certainty_max_abs'].append(
                matched_certainty_difference
            )

            parity_frames = {0, max(0, frames_to_run // 2), frames_to_run - 1}
            if frame in parity_frames:
                p_ref, v_ref, q_ref = model.track_frame(
                    active_q,
                    active_mask,
                    active_memory,
                    frame_features,
                    256,
                    256,
                )
                parity['official_p_max_abs'].append(
                    float(torch.max(torch.abs(p - p_ref)))
                )
                parity['official_v_max_abs'].append(
                    float(torch.max(torch.abs(v_logit - v_ref)))
                )
                parity['official_q_max_abs'].append(
                    float(torch.max(torch.abs(q_new - q_ref)))
                )

            refined_positions = model_xy_to_yx_norm(
                refined['refined_model_xy'], model
            )
            official_yx = output_xy_to_yx_norm(p[:NQ])
            v_conf = torch.sigmoid(v_logit[:NQ]).detach().cpu().numpy()
            u_conf = torch.sigmoid(diag['u_logit'][:NQ]).detach().cpu().numpy()
            risk_visibility = v_conf < float(model.delta_v)
            risk_uncertainty = u_conf >= 0.5
            risk = risk_visibility | risk_uncertainty
            refined_c2_index = refined['c2_index'].detach().cpu().numpy().astype(np.int32)

            raw_model_xy = extended['positions_model_xy'].detach().cpu().float().numpy()
            refined_model_xy = refined['refined_model_xy'].detach().cpu().float().numpy()
            raw_scaled = raw_model_xy.copy()
            refined_scaled = refined_model_xy.copy()
            raw_scaled[..., 0] *= 256.0 / float(model.W)
            raw_scaled[..., 1] *= 256.0 / float(model.H)
            refined_scaled[..., 0] *= 256.0 / float(model.W)
            refined_scaled[..., 1] *= 256.0 / float(model.H)
            displacement = np.linalg.norm(refined_scaled - raw_scaled, axis=-1).astype(
                np.float32
            )
            refined_out_of_range = np.any(
                (refined_positions < 0.0) | (refined_positions > 1.0), axis=-1
            )

            for query in range(NQ):
                gt_yx = np.asarray(
                    [
                        gt_xy[frame, query, 1] / float(height - 1),
                        gt_xy[frame, query, 0] / float(width - 1),
                    ],
                    dtype=np.float32,
                )
                raw_error = np.linalg.norm(
                    (extended_positions[query] - gt_yx[None, :]) * UNIFIED_DENOM,
                    axis=1,
                ).astype(np.float32)
                refined_error = np.linalg.norm(
                    (refined_positions[query] - gt_yx[None, :]) * UNIFIED_DENOM,
                    axis=1,
                ).astype(np.float32)
                official_error = float(
                    np.linalg.norm((official_yx[query] - gt_yx) * UNIFIED_DENOM)
                )

                official_top1 = int(np.argmax(official_scores[query]))
                score_top1_k16 = int(
                    official_to_extended[query, official_top1]
                )
                score_top1_k64 = int(np.argmax(extended_scores[query]))
                dynamic_k = K64 if bool(risk[query]) else K16
                raw_oracle_k16 = float(np.min(raw_error[:K16]))
                raw_oracle_k64 = float(np.min(raw_error))
                refined_oracle_k16 = float(np.min(refined_error[:K16]))
                refined_oracle_k64 = float(np.min(refined_error))
                raw_oracle_dynamic = (
                    raw_oracle_k64 if dynamic_k == K64 else raw_oracle_k16
                )
                refined_oracle_dynamic = (
                    refined_oracle_k64
                    if dynamic_k == K64
                    else refined_oracle_k16
                )
                hybrid_score = (
                    float(refined_error[score_top1_k64])
                    if bool(risk[query])
                    else official_error
                )
                hybrid_oracle = (
                    min(official_error, refined_oracle_k64)
                    if bool(risk[query])
                    else official_error
                )
                opportunity = bool(
                    bool(visible[frame, query])
                    and official_error > 4.0
                    and refined_oracle_k64 < official_error - 1.0
                )

                scalar['clip_id'].append(clip_id)
                scalar['sequence'].append(sequence)
                scalar['frame_tau'].append(int(frame))
                scalar['query_idx'].append(int(query))
                scalar['gt_visible'].append(bool(visible[frame, query]))
                scalar['risk'].append(bool(risk[query]))
                scalar['risk_visibility'].append(bool(risk_visibility[query]))
                scalar['risk_uncertainty'].append(bool(risk_uncertainty[query]))
                scalar['v_conf'].append(float(v_conf[query]))
                scalar['u_conf'].append(float(u_conf[query]))
                scalar['reentry_first'].append(
                    bool(subset_flags['reentry_first'][frame, query])
                )
                scalar['reentry_early4'].append(
                    bool(subset_flags['reentry_early4'][frame, query])
                )
                scalar['reentry_early8'].append(
                    bool(subset_flags['reentry_early8'][frame, query])
                )
                scalar['occ_length'].append(
                    int(subset_flags['occ_length'][frame, query])
                )
                scalar['hard_official'].append(
                    bool(visible[frame, query] and official_error > 4.0)
                )
                scalar['opportunity'].append(opportunity)
                scalar['official_final'].append(official_error)
                scalar['raw_score_top1_K16'].append(
                    float(raw_error[score_top1_k16])
                )
                scalar['raw_score_top1_K64'].append(
                    float(raw_error[score_top1_k64])
                )
                scalar['raw_oracle_K16'].append(raw_oracle_k16)
                scalar['raw_oracle_K64'].append(raw_oracle_k64)
                scalar['raw_oracle_dynamicK'].append(raw_oracle_dynamic)
                scalar['refined_score_top1_K16'].append(
                    float(refined_error[score_top1_k16])
                )
                scalar['refined_score_top1_K64'].append(
                    float(refined_error[score_top1_k64])
                )
                scalar['refined_oracle_K16'].append(refined_oracle_k16)
                scalar['refined_oracle_K64'].append(refined_oracle_k64)
                scalar['refined_oracle_dynamicK'].append(refined_oracle_dynamic)
                scalar['hybrid_refined_score_top1'].append(hybrid_score)
                scalar['hybrid_refined_oracle'].append(hybrid_oracle)
                scalar['refined_unique_c2_K16'].append(
                    int(len(np.unique(refined_c2_index[query, :K16])))
                )
                scalar['refined_unique_c2_K64'].append(
                    int(len(np.unique(refined_c2_index[query])))
                )

                raw_error_rows.append(raw_error)
                refined_error_rows.append(refined_error)
                score_rows.append(extended_scores[query].astype(np.float32))
                displacement_rows.append(displacement[query])
                refined_c2_index_rows.append(refined_c2_index[query])
                refined_out_of_range_rows.append(refined_out_of_range[query])
                cluster16_rows.append(
                    spatial_cluster_count(refined_positions[query, :K16], 4.0)
                )
                cluster64_rows.append(
                    spatial_cluster_count(refined_positions[query], 4.0)
                )

            write_mask = torch.ones(
                predictor.N, dtype=torch.bool, device=device
            )
            updated_memory, updated_mask = model._update_point_memory(
                active_memory,
                active_mask,
                q_new,
                write_mask,
            )
            predictor.point_memory[: predictor.N] = updated_memory
            predictor.temporal_mask[: predictor.N] = updated_mask
            predictor.t += 1

        runtime.append(
            {
                'clip_id': clip_id,
                'frames': int(frames_to_run),
                'queries': int(NQ),
                'seconds': float(time.perf_counter() - clip_start),
            }
        )
        print(json.dumps(runtime[-1]), flush=True)

    arrays: dict[str, np.ndarray] = {
        'clip_id': np.asarray(scalar['clip_id'], dtype=object),
        'sequence': np.asarray(scalar['sequence'], dtype=object),
        'frame_tau': np.asarray(scalar['frame_tau'], dtype=np.int16),
        'query_idx': np.asarray(scalar['query_idx'], dtype=np.int16),
        'gt_visible': np.asarray(scalar['gt_visible'], dtype=bool),
        'risk': np.asarray(scalar['risk'], dtype=bool),
        'risk_visibility': np.asarray(scalar['risk_visibility'], dtype=bool),
        'risk_uncertainty': np.asarray(scalar['risk_uncertainty'], dtype=bool),
        'v_conf': np.asarray(scalar['v_conf'], dtype=np.float32),
        'u_conf': np.asarray(scalar['u_conf'], dtype=np.float32),
        'reentry_first': np.asarray(scalar['reentry_first'], dtype=bool),
        'reentry_early4': np.asarray(scalar['reentry_early4'], dtype=bool),
        'reentry_early8': np.asarray(scalar['reentry_early8'], dtype=bool),
        'occ_length': np.asarray(scalar['occ_length'], dtype=np.int16),
        'hard_official': np.asarray(scalar['hard_official'], dtype=bool),
        'opportunity': np.asarray(scalar['opportunity'], dtype=bool),
        'official_final': np.asarray(scalar['official_final'], dtype=np.float32),
        'raw_score_top1_K16': np.asarray(scalar['raw_score_top1_K16'], dtype=np.float32),
        'raw_score_top1_K64': np.asarray(scalar['raw_score_top1_K64'], dtype=np.float32),
        'raw_oracle_K16': np.asarray(scalar['raw_oracle_K16'], dtype=np.float32),
        'raw_oracle_K64': np.asarray(scalar['raw_oracle_K64'], dtype=np.float32),
        'raw_oracle_dynamicK': np.asarray(scalar['raw_oracle_dynamicK'], dtype=np.float32),
        'refined_score_top1_K16': np.asarray(scalar['refined_score_top1_K16'], dtype=np.float32),
        'refined_score_top1_K64': np.asarray(scalar['refined_score_top1_K64'], dtype=np.float32),
        'refined_oracle_K16': np.asarray(scalar['refined_oracle_K16'], dtype=np.float32),
        'refined_oracle_K64': np.asarray(scalar['refined_oracle_K64'], dtype=np.float32),
        'refined_oracle_dynamicK': np.asarray(scalar['refined_oracle_dynamicK'], dtype=np.float32),
        'hybrid_refined_score_top1': np.asarray(
            scalar['hybrid_refined_score_top1'], dtype=np.float32
        ),
        'hybrid_refined_oracle': np.asarray(
            scalar['hybrid_refined_oracle'], dtype=np.float32
        ),
        'refined_unique_c2_K16': np.asarray(
            scalar['refined_unique_c2_K16'], dtype=np.int16
        ),
        'refined_unique_c2_K64': np.asarray(
            scalar['refined_unique_c2_K64'], dtype=np.int16
        ),
        'refined_cluster4_K16': np.asarray(cluster16_rows, dtype=np.int16),
        'refined_cluster4_K64': np.asarray(cluster64_rows, dtype=np.int16),
        'raw_candidate_error': np.stack(raw_error_rows).astype(np.float32),
        'refined_candidate_error': np.stack(refined_error_rows).astype(np.float32),
        'candidate_score': np.stack(score_rows).astype(np.float32),
        'raw_refined_displacement_px': np.stack(displacement_rows).astype(np.float32),
        'refined_c2_index': np.stack(refined_c2_index_rows).astype(np.int32),
        'refined_out_of_range': np.stack(refined_out_of_range_rows).astype(bool),
    }
    expected_rows = len(clips) * frames_to_run * NQ
    if len(arrays['clip_id']) != expected_rows:
        raise RuntimeError(
            f'row count mismatch: {len(arrays["clip_id"])} != {expected_rows}'
        )

    for key, value in arrays.items():
        if value.dtype == object or value.dtype == bool:
            continue
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f'non-finite output: {key}')

    frame_positive = arrays['frame_tau'] > 0
    visible = arrays['gt_visible'] & frame_positive
    risk_visible = visible & arrays['risk']
    nonrisk_visible = visible & ~arrays['risk']
    hard_visible = visible & arrays['hard_official']
    opportunity = visible & arrays['opportunity']
    reentry_first = visible & arrays['reentry_first']
    reentry_early4 = visible & arrays['reentry_early4']
    reentry_early8 = visible & arrays['reentry_early8']

    masks: dict[str, np.ndarray] = {
        'all_visible': visible,
        'risk_visible': risk_visible,
        'nonrisk_visible': nonrisk_visible,
        'hard_official': hard_visible,
        'opportunity': opportunity,
        'reentry_first': reentry_first,
        'reentry_early4': reentry_early4,
        'reentry_early8': reentry_early8,
        'reentry_occ1': reentry_first & (arrays['occ_length'] == 1),
        'reentry_occ2_4': reentry_first
        & (arrays['occ_length'] >= 2)
        & (arrays['occ_length'] <= 4),
        'reentry_occ5_8': reentry_first
        & (arrays['occ_length'] >= 5)
        & (arrays['occ_length'] <= 8),
        'reentry_occ_gt8': reentry_first & (arrays['occ_length'] > 8),
    }
    for sequence in SEQUENCES:
        masks[f'sequence_{sequence}'] = visible & (
            arrays['sequence'] == sequence
        )
        masks[f'reentry_sequence_{sequence}'] = reentry_first & (
            arrays['sequence'] == sequence
        )
    for clip_id in clips:
        masks[f'clip_{clip_id}'] = visible & (arrays['clip_id'] == clip_id)

    readout_names = [
        'official_final',
        'raw_score_top1_K16',
        'raw_score_top1_K64',
        'raw_oracle_K16',
        'raw_oracle_K64',
        'raw_oracle_dynamicK',
        'refined_score_top1_K16',
        'refined_score_top1_K64',
        'refined_oracle_K16',
        'refined_oracle_K64',
        'refined_oracle_dynamicK',
        'hybrid_refined_score_top1',
        'hybrid_refined_oracle',
    ]
    official_error = arrays['official_final'].astype(np.float64)
    summaries = {
        name: {
            mask_name: metric_summary(arrays[name], official_error, mask)
            for mask_name, mask in masks.items()
        }
        for name in readout_names
    }
    bootstrap, per_clip_difference = clip_bootstrap(
        arrays['clip_id'],
        visible,
        official_error,
        {
            name: arrays[name]
            for name in readout_names
            if name != 'official_final'
        },
    )
    bootstrap['official_final'] = {
        'n_clips': int(len(clips)),
        'clip_mean_difference': 0.0,
        'clip_median_difference': 0.0,
        'clip_better': 0,
        'clip_worse': 0,
        'clip_equal': int(len(clips)),
        'bootstrap_resamples': int(BOOTSTRAP_RESAMPLES),
        'bootstrap_seed': int(BOOTSTRAP_SEED),
        'mean_difference_95_ci': [0.0, 0.0],
        'probability_mean_lt_zero': 0.0,
    }

    risk_audit = {
        'all_rows': float(np.mean(arrays['risk'])),
        'visibility_trigger_all_rows': float(
            np.mean(arrays['risk_visibility'])
        ),
        'uncertainty_trigger_all_rows': float(
            np.mean(arrays['risk_uncertainty'])
        ),
        'trigger_overlap_all_rows': float(
            np.mean(arrays['risk_visibility'] & arrays['risk_uncertainty'])
        ),
        'visible_rows': (
            float(np.mean(arrays['risk'][visible])) if np.any(visible) else None
        ),
        'invisible_rows': (
            float(
                np.mean(
                    arrays['risk'][frame_positive & ~arrays['gt_visible']]
                )
            )
            if np.any(frame_positive & ~arrays['gt_visible'])
            else None
        ),
        'reentry_first_rows': (
            float(np.mean(arrays['risk'][reentry_first]))
            if np.any(reentry_first)
            else None
        ),
        'opportunity_rows': (
            float(np.mean(arrays['risk'][opportunity]))
            if np.any(opportunity)
            else None
        ),
        'conceptual_mean_candidate_count': float(
            16.0 + 48.0 * np.mean(arrays['risk'][frame_positive])
        ),
        'actual_audit_candidate_count': int(K64),
        'delta_v': float(model.delta_v),
        'uncertainty_threshold': 0.5,
        'per_sequence': {
            sequence: (
                {
                    'activation_rate': float(np.mean(arrays['risk'][sequence_mask])),
                    'conceptual_mean_candidate_count': float(
                        16.0 + 48.0 * np.mean(arrays['risk'][sequence_mask])
                    ),
                }
                if np.any(sequence_mask)
                else {
                    'activation_rate': None,
                    'conceptual_mean_candidate_count': None,
                }
            )
            for sequence in SEQUENCES
            for sequence_mask in [
                frame_positive & (arrays['sequence'] == sequence)
            ]
        },
    }

    candidate_mask = np.repeat(visible[:, None], K64, axis=1)
    displacement_visible = arrays['raw_refined_displacement_px'][candidate_mask]
    refinement_diagnostics = {
        'raw_to_refined_displacement_px': {
            'mean': float(np.mean(displacement_visible)),
            'median': float(np.median(displacement_visible)),
            'p95': float(np.quantile(displacement_visible, 0.95)),
            'max': float(np.max(displacement_visible)),
        },
        'raw_K64_oracle_mean_visible': summaries['raw_oracle_K64']['all_visible'][
            'mean_error'
        ],
        'refined_K64_oracle_mean_visible': summaries[
            'refined_oracle_K64'
        ]['all_visible']['mean_error'],
        'refined_score_oracle_gap_K64_mean_visible': float(
            np.mean(
                arrays['refined_score_top1_K64'][visible]
                - arrays['refined_oracle_K64'][visible]
            )
        ),
        'refined_unique_c2_K16_mean_visible': float(
            np.mean(arrays['refined_unique_c2_K16'][visible])
        ),
        'refined_unique_c2_K64_mean_visible': float(
            np.mean(arrays['refined_unique_c2_K64'][visible])
        ),
        'refined_cluster4_K16_mean_visible': float(
            np.mean(arrays['refined_cluster4_K16'][visible])
        ),
        'refined_cluster4_K64_mean_visible': float(
            np.mean(arrays['refined_cluster4_K64'][visible])
        ),
        'refined_candidate_out_of_range_rate_visible': float(
            np.mean(arrays['refined_out_of_range'][visible])
        ),
        'per_sequence': {
            sequence: {
                'raw_K64_oracle_mean': summaries['raw_oracle_K64'][
                    f'sequence_{sequence}'
                ]['mean_error'],
                'refined_K64_oracle_mean': summaries['refined_oracle_K64'][
                    f'sequence_{sequence}'
                ]['mean_error'],
            }
            for sequence in SEQUENCES
        },
    }

    primary = 'hybrid_refined_oracle'
    if args.max_clips > 0 or frames_to_run < LENGTH:
        gates = {
            'smoke_only': True,
            'pass_all': False,
            'decision': 'SMOKE_ONLY: full three-sequence refinement gate not evaluated.',
        }
    else:
        sequence_mean = {
            sequence: bool(
                summaries[primary][f'sequence_{sequence}']['mean_error']
                < summaries['official_final'][f'sequence_{sequence}']['mean_error']
            )
            for sequence in SEQUENCES
        }
        sequence_safe16 = {
            sequence: bool(
                summaries[primary][f'sequence_{sequence}']['safe16']
                >= summaries['official_final'][f'sequence_{sequence}']['safe16']
            )
            for sequence in SEQUENCES
        }
        reentry_nonincrease = {}
        for sequence in SEQUENCES:
            official_row = summaries['official_final'][
                f'reentry_sequence_{sequence}'
            ]
            hybrid_row = summaries[primary][f'reentry_sequence_{sequence}']
            reentry_nonincrease[sequence] = (
                None
                if official_row['n'] < 10
                else bool(hybrid_row['mean_error'] <= official_row['mean_error'])
            )
        gate = {
            'sequence_mean_improved': sequence_mean,
            'sequence_safe16_not_decreased': sequence_safe16,
            'global_better_gt_worse': bool(
                summaries[primary]['all_visible']['better']
                > summaries[primary]['all_visible']['worse']
            ),
            'clip_bootstrap_ci_upper_lt_zero': bool(
                bootstrap[primary]['mean_difference_95_ci'][1] < 0.0
            ),
            'first_reentry_mean_improved': bool(
                summaries[primary]['reentry_first']['n'] > 0
                and summaries[primary]['reentry_first']['mean_error']
                < summaries['official_final']['reentry_first']['mean_error']
            ),
            'early8_mean_improved': bool(
                summaries[primary]['reentry_early8']['n'] > 0
                and summaries[primary]['reentry_early8']['mean_error']
                < summaries['official_final']['reentry_early8']['mean_error']
            ),
            'reentry_sequence_nonincrease_when_n_ge10': reentry_nonincrease,
        }
        gate['pass_all'] = bool(
            all(sequence_mean.values())
            and all(sequence_safe16.values())
            and gate['global_better_gt_worse']
            and gate['clip_bootstrap_ci_upper_lt_zero']
            and gate['first_reentry_mean_improved']
            and gate['early8_mean_improved']
            and all(value is None or value for value in reentry_nonincrease.values())
        )
        gate['decision'] = (
            'CANDIDATE_REFINEMENT_PASS: risk-gated candidate-conditioned refinement '
            'has sequence-consistent system-level oracle headroom. Proceed to '
            'V9-A5.1c history-preserving beam; do not read DAVIS or train yet.'
            if gate['pass_all']
            else 'CANDIDATE_REFINEMENT_FAIL: the frozen candidate-conditioned '
            'proposal/refinement path does not satisfy the full synthetic gate. '
            'Do not rebuild the beam.'
        )
        gates = gate

    integrity = {
        'input_audit': input_audit,
        'processed_rows': int(len(arrays['clip_id'])),
        'expected_rows': int(expected_rows),
        'clips': int(len(clips)),
        'frames_per_clip': int(frames_to_run),
        'official_p_max_abs': float(max(parity['official_p_max_abs'])),
        'official_v_max_abs': float(max(parity['official_v_max_abs'])),
        'official_q_max_abs': float(max(parity['official_q_max_abs'])),
        'extended_first16_ordered_position_max_abs': float(
            max(parity['extended_first16_ordered_position_max_abs'])
        ),
        'extended_first16_set_hausdorff_max': float(
            max(parity['extended_first16_set_hausdorff_max'])
        ),
        'extended_first16_matched_score_max_abs': float(
            max(parity['extended_first16_matched_score_max_abs'])
        ),
        'extended_first16_matched_certainty_max_abs': float(
            max(parity['extended_first16_matched_certainty_max_abs'])
        ),
        'candidate_refinement_signature': list(
            inspect.signature(candidate_conditioned_refinement).parameters
        ),
        'candidate_refinement_uses_gt_or_error': any(
            token in inspect.getsource(candidate_conditioned_refinement).lower()
            for token in ['gt_', 'ground_truth', 'error_px']
        ),
        'all_outputs_finite': True,
    }
    if max(
        integrity['official_p_max_abs'],
        integrity['official_v_max_abs'],
        integrity['official_q_max_abs'],
    ) > 1e-6:
        raise RuntimeError(f'official forward parity failed: {integrity}')
    integrity['extended_parity_tolerance'] = 6e-5
    if max(
        integrity['extended_first16_set_hausdorff_max'],
        integrity['extended_first16_matched_score_max_abs'],
        integrity['extended_first16_matched_certainty_max_abs'],
    ) > integrity['extended_parity_tolerance']:
        raise RuntimeError(f'extended K64/K16 parity failed: {integrity}')
    if integrity['candidate_refinement_uses_gt_or_error']:
        raise RuntimeError('candidate refinement source unexpectedly references GT/error')

    result = {
        'script': 'scripts/v9a51b_candidate_conditioned_refinement_audit.py',
        'date': '2026-07-10',
        'provenance': {
            'script_sha256': sha256(Path(__file__).resolve()),
            'branch': input_audit['branch'],
            'head': input_audit['head'],
            'worktree_root': str(WORKTREE_ROOT),
            'source_asset_root': str(SOURCE_ROOT),
        },
        'protocol': {
            'dataset': 'PointOdyssey full-stream deterministic clips',
            'sequences': list(SEQUENCES),
            'clip_ids': clips,
            'clip_length': int(frames_to_run),
            'queries_per_clip': int(NQ),
            'frame0_excluded_from_metrics': True,
            'candidate_budgets': [K16, K64],
            'risk_rule': 'sigmoid(v_logit) < 0.8 OR sigmoid(u_logit) >= 0.5',
            'primary_readout': primary,
            'training': False,
            'davis_read': False,
            'threshold_tuning': False,
            'bootstrap_unit': 'clip_id',
            'bootstrap_resamples': int(BOOTSTRAP_RESAMPLES),
            'bootstrap_seed': int(BOOTSTRAP_SEED),
            'seconds': float(time.perf_counter() - start_all),
        },
        'integrity': integrity,
        'per_clip_runtime': runtime,
        'risk_audit': risk_audit,
        'refinement_diagnostics': refinement_diagnostics,
        'summaries': summaries,
        'clip_bootstrap': bootstrap,
        'per_clip_mean_difference': per_clip_difference,
        'gates': gates,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_doc.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    np.savez_compressed(args.out_npz, **arrays)
    write_doc(result, args.out_doc)
    print(
        json.dumps(
            {
                'ok': True,
                'json': str(args.out_json),
                'npz': str(args.out_npz),
                'doc': str(args.out_doc),
                'rows': int(len(arrays['clip_id'])),
                'decision': gates['decision'],
                'gate_pass': bool(gates.get('pass_all', False)),
                'official_mean_visible': summaries['official_final']['all_visible'][
                    'mean_error'
                ],
                'hybrid_oracle_mean_visible': summaries[primary]['all_visible'][
                    'mean_error'
                ],
                'risk_rate': risk_audit['all_rows'],
                'integrity': {
                    key: value
                    for key, value in integrity.items()
                    if key != 'input_audit'
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == '__main__':
    main()
