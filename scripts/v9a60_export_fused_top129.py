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

WORKTREE_ROOT = Path('/gemini/code/FSPT_v9a60_clean')
SOURCE_ROOT = Path('/gemini/code/FSPT')
TRACKON_ROOT = WORKTREE_ROOT / 'baselines/track_on'
SOURCE_BASE = SOURCE_ROOT / 'outputs/paper_discovery_2026-07-05'
POOL_PATH = SOURCE_BASE / 'v9a38_pointodyssey_pool/v9a38_pointodyssey_hypothesis_pool.npz'
V9C_JSON = WORKTREE_ROOT / 'outputs/paper_discovery_2026-07-05/v9a5c0_candidate_recall/v9a5c0_pointodyssey_candidate_recall.json'
V9C_NPZ = WORKTREE_ROOT / 'outputs/paper_discovery_2026-07-05/v9a5c0_candidate_recall/v9a5c0_pointodyssey_candidate_recall.npz'
V9B_NPZ = WORKTREE_ROOT / 'outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_candidate_conditioned_refinement_rows.npz'
POD_ROOT = SOURCE_ROOT / 'datasets/pointodyssey/train'
CHECKPOINT = SOURCE_ROOT / 'baselines/track_on/checkpoints_trackon2_dinov3.pt'
CONFIG = TRACKON_ROOT / 'config/test.yaml'
INPUT_MANIFEST = WORKTREE_ROOT / 'docs/v9a60_input_manifest_2026-07-11.json'
OUTDIR = WORKTREE_ROOT / 'outputs/paper_discovery_2026-07-05/v9a60_rank_shift'
DEFAULT_JSON = OUTDIR / 'v9a60_fused_top129_export.json'
DEFAULT_NPZ = OUTDIR / 'v9a60_fused_top129_export.npz'
DEFAULT_DOC = WORKTREE_ROOT / 'docs/v9a60_fused_top129_export_result_2026-07-11.md'

EXPECTED_HEAD = 'db4f79d8c128a208dcfe636de330fb9309888fba'
EXPECTED_PARENT_HEAD = 'bb879183ffff03129cb652824c286f56f9201804'
EXPECTED_BRANCH = 'v9a60-correlation-rank-shift-20260711'
EXPECTED_MANIFEST_SHA256 = '953f6f63738a7a90e7db0c31525ea55255f70ace759aa2c310384a77c3ddccb3'

SEQUENCES = ('ani', 'animal3', 'r4_new_f')
NQ = 32
LENGTH = 96
TOP_EXPORT = 129
UNIFIED_SIZE = 256.0
UNIFIED_DENOM = 255.0
COMPONENT_NAMES = ('c4', 'c8', 'c16', 'c32')
K_VALUES = (16, 64)
RADIUS_PX = 4.0
PARITY_TOL = 1e-6
ERROR_TOL = 1e-4

if str(TRACKON_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKON_ROOT))
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.append(str(WORKTREE_ROOT))
os.environ['DINOV3_LOCAL_DIR'] = str(
    SOURCE_ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m'
)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from model.trackon_predictor import Predictor  # noqa: E402
from utils.coord_utils import get_points_on_a_grid, indices_to_coords  # noqa: E402
from utils.train_utils import load_args_from_yaml  # noqa: E402
from scripts.v9a26_build_trackon2_internal_proxy_features import track_frame_diag  # noqa: E402

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


def fused_grid_coords(model, device: torch.device) -> np.ndarray:
    indices = torch.arange(model.P, device=device, dtype=torch.long).view(1, 1, -1)
    xy = indices_to_coords(indices, model.input_size, model.stride)[0, 0]
    return model_xy_to_unified_yx_norm(xy, model)


@torch.no_grad()
def correlation_components_upsampled(
    model,
    q_pre: torch.Tensor,
    f4: torch.Tensor,
    f8: torch.Tensor,
    f16: torch.Tensor,
    f32: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    upsampled = torch.cat(
        [
            native['c4'].unsqueeze(1),
            F.interpolate(native['c8'].unsqueeze(1), size=(model.Hf, model.Wf), mode='bilinear', align_corners=False),
            F.interpolate(native['c16'].unsqueeze(1), size=(model.Hf, model.Wf), mode='bilinear', align_corners=False),
            F.interpolate(native['c32'].unsqueeze(1), size=(model.Hf, model.Wf), mode='bilinear', align_corners=False),
        ],
        dim=1,
    )
    fused = model.ms_corr_proj(upsampled).view(n_queries, model.P)
    return fused, upsampled.view(n_queries, len(COMPONENT_NAMES), model.P)


def verify_inputs(verify_frames: bool) -> dict[str, Any]:
    if sha256(INPUT_MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError('input manifest hash mismatch')
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
        ['git', '-C', str(WORKTREE_ROOT), 'rev-parse', 'HEAD'], text=True
    ).strip()
    parent = subprocess.check_output(
        ['git', '-C', str(WORKTREE_ROOT), 'rev-parse', 'HEAD^'], text=True
    ).strip()
    branch = subprocess.check_output(
        ['git', '-C', str(WORKTREE_ROOT), 'branch', '--show-current'], text=True
    ).strip()
    tracked = subprocess.check_output(
        ['git', '-C', str(WORKTREE_ROOT), 'status', '--porcelain', '--untracked-files=no'],
        text=True,
    ).strip()
    if head != EXPECTED_HEAD:
        raise RuntimeError(f'HEAD mismatch: {head}')
    if parent != EXPECTED_PARENT_HEAD or manifest['clean_head'] != EXPECTED_PARENT_HEAD:
        raise RuntimeError(f'parent/base mismatch: {parent} / {manifest["clean_head"]}')
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f'branch mismatch: {branch}')
    if tracked:
        raise RuntimeError(f'tracked worktree dirty: {tracked}')

    frame_report = None
    if verify_frames:
        meta = manifest['frame_manifest']
        frame_manifest = json.loads(Path(meta['path']).read_text())
        root = Path(frame_manifest['root'])
        aggregate = hashlib.sha256()
        total = 0
        for row in frame_manifest['files']:
            relative = str(row['relative_path'])
            path = root / relative
            actual = sha256(path)
            size = path.stat().st_size
            if size != int(row['size_bytes']) or actual != row['sha256']:
                raise RuntimeError(f'frame hash/size mismatch: {path}')
            total += size
            aggregate.update(relative.encode())
            aggregate.update(b'\0')
            aggregate.update(str(size).encode())
            aggregate.update(b'\0')
            aggregate.update(actual.encode())
            aggregate.update(b'\n')
        if len(frame_manifest['files']) != int(meta['file_count']):
            raise RuntimeError('frame count mismatch')
        if total != int(meta['total_bytes']):
            raise RuntimeError('frame bytes mismatch')
        if aggregate.hexdigest() != meta['aggregate_sha256']:
            raise RuntimeError('frame aggregate mismatch')
        frame_report = {
            'path': str(meta['path']),
            'file_count': int(len(frame_manifest['files'])),
            'total_bytes': int(total),
            'aggregate_sha256': aggregate.hexdigest(),
        }

    imported = {}
    for name in (
        'model.trackon_predictor',
        'model.trackon',
        'model.reranking',
        'utils.coord_utils',
        'utils.train_utils',
    ):
        module = sys.modules.get(name)
        value = '' if module is None else str(Path(module.__file__).resolve())
        imported[name] = value
        if not value.startswith(str(TRACKON_ROOT.resolve()) + '/'):
            raise RuntimeError(f'non-clean TrackOn import: {name} -> {value}')

    return {
        'pass': True,
        'head': head,
        'parent_head': parent,
        'branch': branch,
        'tracked_status': tracked,
        'checked_inputs': checked,
        'frame_manifest': frame_report,
        'imported_trackon_code_paths': imported,
        'script': {
            'path': str(Path(__file__).resolve()),
            'sha256': sha256(Path(__file__).resolve()),
        },
    }


def write_doc(result: dict[str, Any], path: Path) -> None:
    lines = [
        '# V9-A6.0 Fused Top129 Export Result',
        '',
        'Date: 2026-07-11',
        '',
        'This is a read-only export. It does not train or evaluate a correlation adapter.',
        '',
        '## Protocol',
        '',
        '```json',
        json.dumps(result['protocol'], indent=2),
        '```',
        '',
        '## Integrity',
        '',
        '```json',
        json.dumps(result['integrity'], indent=2),
        '```',
        '',
        '## Opportunity preview',
        '',
        '```json',
        json.dumps(result['opportunity_preview'], indent=2),
        '```',
        '',
        '## Decision',
        '',
        result['decision'],
    ]
    path.write_text('\n'.join(lines) + '\n')


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-clips', type=int, default=0)
    parser.add_argument('--max-rows-per-clip', type=int, default=0)
    parser.add_argument('--skip-full-frame-hash', action='store_true')
    parser.add_argument('--out-json', type=Path, default=DEFAULT_JSON)
    parser.add_argument('--out-npz', type=Path, default=DEFAULT_NPZ)
    parser.add_argument('--out-doc', type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()

    start_all = time.perf_counter()
    formal_run = args.max_clips == 0 and args.max_rows_per_clip == 0
    input_audit = verify_inputs(verify_frames=formal_run and not args.skip_full_frame_hash)

    pool = np.load(POOL_PATH, allow_pickle=True)
    v9c = np.load(V9C_NPZ, allow_pickle=True)
    v9c_report = json.loads(V9C_JSON.read_text())
    v9b = np.load(V9B_NPZ, allow_pickle=True)
    selected_indices_all = v9c['selected_indices'].astype(np.int64)
    if len(selected_indices_all) != 4878:
        raise RuntimeError(f'unexpected V9-A5C row count: {len(selected_indices_all)}')
    if not np.array_equal(selected_indices_all, np.arange(len(selected_indices_all))):
        raise RuntimeError('V9-A5C selected_indices are not the full ordered pool')
    row_keys_all = list(
        zip(
            v9c['clip_id'].astype(str).tolist(),
            v9c['frame_tau'].astype(int).tolist(),
            v9c['query_idx'].astype(int).tolist(),
        )
    )
    pool_keys = list(
        zip(
            pool['clip_id'].astype(str).tolist(),
            pool['frame_tau'].astype(int).tolist(),
            pool['query_idx'].astype(int).tolist(),
        )
    )
    if row_keys_all != pool_keys or len(set(row_keys_all)) != 4878:
        raise RuntimeError('V9-A5C/pool row-key mismatch')
    v9b_key = {
        (str(clip), int(frame), int(query)): index
        for index, (clip, frame, query) in enumerate(
            zip(v9b['clip_id'], v9b['frame_tau'], v9b['query_idx'])
        )
    }
    if not set(row_keys_all).issubset(set(v9b_key)):
        raise RuntimeError('V9-A5C rows missing from V9-A5.1b fullstream rows')

    clips_all = sorted(set(v9c['clip_id'].astype(str).tolist()))
    clips = clips_all[: args.max_clips] if args.max_clips > 0 else clips_all
    selected_local = []
    for clip in clips:
        rows = np.flatnonzero(v9c['clip_id'].astype(str) == clip)
        if args.max_rows_per_clip > 0:
            rows = rows[: args.max_rows_per_clip]
        selected_local.extend(rows.tolist())
    selected_local = np.asarray(selected_local, dtype=np.int64)
    if len(selected_local) == 0:
        raise RuntimeError('no rows selected')
    n_rows = len(selected_local)
    local_index = {int(global_row): local for local, global_row in enumerate(selected_local.tolist())}

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
    if model.M != 24 or model.D != 256 or model.K != 16:
        raise RuntimeError(f'unexpected model dimensions: M={model.M}, D={model.D}, K={model.K}')
    if str(model.memory_update_policy) != 'unconditional':
        raise RuntimeError(f'unexpected memory policy: {model.memory_update_policy}')

    grid_coords = fused_grid_coords(model, device)
    feature_names = pool['internal_feature_names'].tolist()
    visibility_conf_i = feature_names.index('visibility_conf')
    uncertainty_i = feature_names.index('uncertainty_sigmoid')

    clip_id_out = np.asarray(v9c['clip_id'][selected_local], dtype=object)
    sequence_out = np.asarray(v9c['sequence'][selected_local], dtype=object)
    frame_tau_out = v9c['frame_tau'][selected_local].astype(np.int32)
    query_idx_out = v9c['query_idx'][selected_local].astype(np.int32)
    source_index_out = selected_indices_all[selected_local].astype(np.int64)
    gt_yx_out = np.full((n_rows, 2), np.nan, dtype=np.float32)
    gt_visible_out = np.zeros(n_rows, dtype=bool)
    is_hard_out = pool['is_hard'][source_index_out].astype(bool)
    old_error_out = pool['old_error_px'][source_index_out].astype(np.float32)
    x_internal = pool['X_internal'][source_index_out]
    visibility_conf_out = x_internal[:, visibility_conf_i].astype(np.float32)
    uncertainty_out = x_internal[:, uncertainty_i].astype(np.float32)
    native_risk_out = (visibility_conf_out < 0.8) | (uncertainty_out >= 0.5)
    reentry_first_out = np.asarray(
        [v9b['reentry_first'][v9b_key[key]] for key in [row_keys_all[i] for i in selected_local]],
        dtype=bool,
    )
    reentry_early8_out = np.asarray(
        [v9b['reentry_early8'][v9b_key[key]] for key in [row_keys_all[i] for i in selected_local]],
        dtype=bool,
    )

    top_indices = np.full((n_rows, TOP_EXPORT), -1, dtype=np.int32)
    top_scores = np.full((n_rows, TOP_EXPORT), np.nan, dtype=np.float32)
    top_errors = np.full((n_rows, TOP_EXPORT), np.nan, dtype=np.float32)
    top_coords = np.full((n_rows, TOP_EXPORT, 2), np.nan, dtype=np.float32)
    top_components = np.full(
        (n_rows, TOP_EXPORT, len(COMPONENT_NAMES)), np.nan, dtype=np.float32
    )
    fused_mean = np.full(n_rows, np.nan, dtype=np.float32)
    fused_std = np.full(n_rows, np.nan, dtype=np.float32)
    fused_min = np.full(n_rows, np.nan, dtype=np.float32)
    fused_max = np.full(n_rows, np.nan, dtype=np.float32)
    min_error_k16 = np.full(n_rows, np.nan, dtype=np.float32)
    min_error_k64 = np.full(n_rows, np.nan, dtype=np.float32)
    recall4_k16 = np.zeros(n_rows, dtype=bool)
    recall4_k64 = np.zeros(n_rows, dtype=bool)
    nearest_rank = np.zeros(n_rows, dtype=np.int32)
    nearest_distance = np.full(n_rows, np.nan, dtype=np.float32)
    nearest_margin = np.full(n_rows, np.nan, dtype=np.float32)
    boundary_gap_k16 = np.full(n_rows, np.nan, dtype=np.float32)
    processed = np.zeros(n_rows, dtype=bool)

    component_weights = (
        model.ms_corr_proj.weight.detach().cpu().float().numpy().reshape(-1).astype(np.float32)
    )
    component_bias = (
        model.ms_corr_proj.bias.detach().cpu().float().numpy().reshape(-1).astype(np.float32)
    )
    if component_weights.shape != (4,) or component_bias.shape != (1,):
        raise RuntimeError(
            f'unexpected fusion parameters: {component_weights.shape}, {component_bias.shape}'
        )

    parity = {
        'fused_max_abs': 0.0,
        'top129_recompose_max_abs': 0.0,
        'p_max_abs': 0.0,
        'v_max_abs': 0.0,
        'q_max_abs': 0.0,
        'k16_error_max_abs': 0.0,
        'k64_error_max_abs': 0.0,
        'nearest_rank_mismatch': 0,
        'nearest_distance_max_abs': 0.0,
        'nearest_margin_max_abs': 0.0,
        'boundary_gap_k16_max_abs': 0.0,
        'official_top16_set_hausdorff_max': 0.0,
        'pool_top16_ordered_max_abs': 0.0,
        'pool_top16_set_hausdorff_max': 0.0,
    }
    clip_runtime = []

    fused_map_i = v9c['map_names'].tolist().index('fused')
    fused_score_i = v9c['score_map_names'].tolist().index('fused')
    k16_i = v9c['k_values'].tolist().index(16)
    k64_i = v9c['k_values'].tolist().index(64)
    r4_i = v9c['radii'].tolist().index(4.0)

    for clip in clips:
        sequence, start_text = clip.split(':')
        window_start = int(start_text)
        sequence_path = POD_ROOT / sequence
        annotation = np.load(sequence_path / 'anno.npz', allow_pickle=True, mmap_mode='r')
        trajectories = np.asarray(
            annotation['trajs_2d'][window_start : window_start + LENGTH], dtype=np.float32
        )
        raw_visibility = np.asarray(
            annotation['visibs'][window_start : window_start + LENGTH], dtype=bool
        )
        valid = (
            np.asarray(
                annotation['valids'][window_start : window_start + LENGTH], dtype=bool
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
            np.random.default_rng(stable_seed(sequence, window_start)).choice(
                eligible, NQ, replace=False
            )
        )
        selected_gt = trajectories[:, chosen]
        selected_visible = visible_all[:, chosen]
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
            (int(v9c['frame_tau'][global_row]), int(v9c['query_idx'][global_row])): local_index[int(global_row)]
            for global_row in selected_local
            if str(v9c['clip_id'][global_row]) == clip
        }
        max_t = max(frame for frame, _ in clip_targets)
        clip_start_time = time.perf_counter()
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
            fused, components = correlation_components_upsampled(
                model,
                diag['q_pre'].unsqueeze(0),
                f4,
                f8,
                f16,
                f32,
            )
            parity['fused_max_abs'] = max(
                parity['fused_max_abs'],
                float(torch.max(torch.abs(fused - diag['c1'])).detach().cpu()),
            )
            p_ref, v_ref, q_ref = model.track_frame(
                active_q,
                active_mask,
                active_memory,
                frame_features,
                256,
                256,
            )
            parity['p_max_abs'] = max(parity['p_max_abs'], float(torch.max(torch.abs(p - p_ref)).detach().cpu()))
            parity['v_max_abs'] = max(parity['v_max_abs'], float(torch.max(torch.abs(v_logit - v_ref)).detach().cpu()))
            parity['q_max_abs'] = max(parity['q_max_abs'], float(torch.max(torch.abs(q_new - q_ref)).detach().cpu()))

            for query_i in range(NQ):
                key = (t, query_i)
                if key not in clip_targets:
                    continue
                out_i = clip_targets[key]
                global_row = int(selected_local[out_i])
                gt_xy = selected_gt[t, query_i]
                gt_norm = np.asarray(
                    [gt_xy[1] / float(height - 1), gt_xy[0] / float(width - 1)],
                    dtype=np.float32,
                )
                gt_yx_out[out_i] = gt_norm
                gt_visible_out[out_i] = bool(selected_visible[t, query_i])

                score_tensor = fused[query_i].float()
                top = torch.topk(
                    score_tensor,
                    k=TOP_EXPORT,
                    dim=0,
                    largest=True,
                    sorted=True,
                )
                indices_np = top.indices.detach().cpu().numpy().astype(np.int32)
                scores_np = top.values.detach().cpu().float().numpy().astype(np.float32)
                coords_np = grid_coords[indices_np]
                errors_np = np.linalg.norm(
                    (coords_np - gt_norm[None, :]) * UNIFIED_DENOM,
                    axis=1,
                ).astype(np.float32)
                comp_np = (
                    components[query_i, :, top.indices]
                    .permute(1, 0)
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                    .astype(np.float32)
                )
                top_indices[out_i] = indices_np
                top_scores[out_i] = scores_np
                top_coords[out_i] = coords_np
                top_errors[out_i] = errors_np
                top_components[out_i] = comp_np
                recomposed = comp_np @ component_weights + float(component_bias[0])
                parity['top129_recompose_max_abs'] = max(
                    parity['top129_recompose_max_abs'],
                    float(np.max(np.abs(recomposed - scores_np))),
                )
                all_scores = score_tensor.detach().cpu().float().numpy().astype(np.float32)
                fused_mean[out_i] = float(np.mean(all_scores))
                fused_std[out_i] = float(np.std(all_scores))
                fused_min[out_i] = float(np.min(all_scores))
                fused_max[out_i] = float(np.max(all_scores))
                min_error_k16[out_i] = float(np.min(errors_np[:16]))
                min_error_k64[out_i] = float(np.min(errors_np[:64]))
                recall4_k16[out_i] = bool(min_error_k16[out_i] <= RADIUS_PX)
                recall4_k64[out_i] = bool(min_error_k64[out_i] <= RADIUS_PX)
                boundary_gap_k16[out_i] = float(scores_np[15] - scores_np[16])

                all_error = np.linalg.norm(
                    (grid_coords - gt_norm[None, :]) * UNIFIED_DENOM,
                    axis=1,
                )
                nearest_index = int(np.argmin(all_error))
                nearest_distance[out_i] = float(all_error[nearest_index])
                nearest_score = float(all_scores[nearest_index])
                nearest_rank[out_i] = int(1 + np.sum(all_scores > nearest_score))
                nearest_margin[out_i] = nearest_score - float(np.max(all_scores))

                parity['k16_error_max_abs'] = max(
                    parity['k16_error_max_abs'],
                    abs(
                        float(min_error_k16[out_i])
                        - float(v9c['min_error'][global_row, fused_map_i, k16_i])
                    ),
                )
                parity['k64_error_max_abs'] = max(
                    parity['k64_error_max_abs'],
                    abs(
                        float(min_error_k64[out_i])
                        - float(v9c['min_error'][global_row, fused_map_i, k64_i])
                    ),
                )
                parity['nearest_rank_mismatch'] += int(
                    int(nearest_rank[out_i])
                    != int(v9c['nearest_rank'][global_row, fused_score_i])
                )
                parity['nearest_distance_max_abs'] = max(
                    parity['nearest_distance_max_abs'],
                    abs(
                        float(nearest_distance[out_i])
                        - float(v9c['nearest_distance'][global_row, fused_score_i])
                    ),
                )
                parity['nearest_margin_max_abs'] = max(
                    parity['nearest_margin_max_abs'],
                    abs(
                        float(nearest_margin[out_i])
                        - float(v9c['nearest_margin'][global_row, fused_score_i])
                    ),
                )
                parity['boundary_gap_k16_max_abs'] = max(
                    parity['boundary_gap_k16_max_abs'],
                    abs(
                        float(boundary_gap_k16[out_i])
                        - float(v9c['boundary_gap'][global_row, fused_score_i, k16_i])
                    ),
                )
                c1_coords = model_xy_to_unified_yx_norm(
                    diag['p_topk'][query_i], model
                )
                distance = np.linalg.norm(
                    top_coords[out_i, :16, None, :] - c1_coords[None, :, :],
                    axis=-1,
                )
                parity['official_top16_set_hausdorff_max'] = max(
                    parity['official_top16_set_hausdorff_max'],
                    float(
                        max(
                            np.min(distance, axis=1).max(),
                            np.min(distance, axis=0).max(),
                        )
                    ),
                )
                pool_coords = pool['candidates'][source_index_out[out_i], 2:18].astype(np.float32)
                parity['pool_top16_ordered_max_abs'] = max(
                    parity['pool_top16_ordered_max_abs'],
                    float(np.max(np.abs(top_coords[out_i, :16] - pool_coords))),
                )
                pool_distance = np.linalg.norm(
                    top_coords[out_i, :16, None, :] - pool_coords[None, :, :],
                    axis=-1,
                )
                parity['pool_top16_set_hausdorff_max'] = max(
                    parity['pool_top16_set_hausdorff_max'],
                    float(
                        max(
                            np.min(pool_distance, axis=1).max(),
                            np.min(pool_distance, axis=0).max(),
                        )
                    ),
                )
                if recall4_k16[out_i] != bool(v9c['recall'][global_row, fused_map_i, k16_i, r4_i]):
                    raise RuntimeError(f'K16 recall mismatch at row {global_row}')
                if recall4_k64[out_i] != bool(v9c['recall'][global_row, fused_map_i, k64_i, r4_i]):
                    raise RuntimeError(f'K64 recall mismatch at row {global_row}')
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

        clip_runtime.append(
            {
                'clip_id': clip,
                'rows': int(len(clip_targets)),
                'max_t': int(max_t),
                'seconds': float(time.perf_counter() - clip_start_time),
            }
        )
        print(json.dumps(clip_runtime[-1]), flush=True)

    if not np.all(processed):
        raise RuntimeError(f'missing rows: {np.flatnonzero(~processed)[:20].tolist()}')
    numeric_arrays = [
        gt_yx_out,
        old_error_out,
        visibility_conf_out,
        uncertainty_out,
        top_scores,
        top_errors,
        top_coords,
        top_components,
        fused_mean,
        fused_std,
        fused_min,
        fused_max,
        min_error_k16,
        min_error_k64,
        nearest_distance,
        nearest_margin,
        boundary_gap_k16,
    ]
    if not all(np.all(np.isfinite(value)) for value in numeric_arrays):
        raise RuntimeError('non-finite export output')
    if np.any(top_indices < 0) or np.any(top_indices >= model.P):
        raise RuntimeError('invalid top129 indices')
    if any(len(np.unique(row)) != TOP_EXPORT for row in top_indices):
        raise RuntimeError('duplicate top129 index')
    if np.any(np.diff(top_scores, axis=1) > 1e-7):
        raise RuntimeError('top129 scores not non-increasing')
    if np.any(fused_std <= 0):
        raise RuntimeError('non-positive fused std')

    if max(
        parity['fused_max_abs'],
        parity['top129_recompose_max_abs'],
        parity['p_max_abs'],
        parity['v_max_abs'],
        parity['q_max_abs'],
        parity['boundary_gap_k16_max_abs'],
        parity['official_top16_set_hausdorff_max'],
        parity['pool_top16_set_hausdorff_max'],
    ) > PARITY_TOL:
        raise RuntimeError(f'parity gate failed: {parity}')
    if max(
        parity['k16_error_max_abs'],
        parity['k64_error_max_abs'],
        parity['nearest_distance_max_abs'],
        parity['nearest_margin_max_abs'],
    ) > ERROR_TOL:
        raise RuntimeError(f'error parity gate failed: {parity}')
    if parity['nearest_rank_mismatch'] != 0:
        raise RuntimeError(f'nearest-rank mismatch: {parity}')

    opportunity = (~recall4_k16) & recall4_k64
    expected_opportunity = int(np.sum(opportunity))
    if formal_run and expected_opportunity != 760:
        raise RuntimeError(f'opportunity count mismatch: {expected_opportunity}')
    opportunity_preview = {
        'rows': int(n_rows),
        'opportunity_rows': expected_opportunity,
        'opportunity_per_sequence': {
            sequence: int(np.sum(opportunity & (sequence_out.astype(str) == sequence)))
            for sequence in SEQUENCES
        },
        'native_risk_rows': int(np.sum(native_risk_out)),
        'native_risk_opportunity_rows': int(np.sum(native_risk_out & opportunity)),
        'gt_hard_rows': int(np.sum(is_hard_out)),
        'opportunity_gt_hard_rows': int(np.sum(is_hard_out & opportunity)),
    }

    arrays = {
        'source_index': source_index_out,
        'clip_id': clip_id_out,
        'sequence': sequence_out,
        'frame_tau': frame_tau_out,
        'query_idx': query_idx_out,
        'gt_yx': gt_yx_out,
        'gt_visible': gt_visible_out,
        'old_error_px': old_error_out,
        'is_hard': is_hard_out,
        'visibility_conf': visibility_conf_out,
        'uncertainty_sigmoid': uncertainty_out,
        'native_risk': native_risk_out,
        'reentry_first': reentry_first_out,
        'reentry_early8': reentry_early8_out,
        'top_indices': top_indices,
        'top_scores': top_scores,
        'top_errors_px': top_errors,
        'top_coords_yx': top_coords,
        'top_components': top_components,
        'component_names': np.asarray(COMPONENT_NAMES, dtype=object),
        'component_weights': component_weights,
        'component_bias': component_bias,
        'fused_mean': fused_mean,
        'fused_std': fused_std,
        'fused_min': fused_min,
        'fused_max': fused_max,
        'min_error_k16': min_error_k16,
        'min_error_k64': min_error_k64,
        'recall4_k16': recall4_k16,
        'recall4_k64': recall4_k64,
        'nearest_rank': nearest_rank,
        'nearest_distance': nearest_distance,
        'nearest_margin': nearest_margin,
        'boundary_gap_k16': boundary_gap_k16,
    }

    decision = (
        'EXPORT_PASS: the formal 4,878-row top129 export reproduces V9-A5C.0 and may be frozen for the no-training rank-shift audit.'
        if formal_run
        else 'EXPORT_SMOKE_PASS: subset export parity passes. This is not the formal 4,878-row export.'
    )
    result = {
        'date': '2026-07-11',
        'protocol': {
            'formal_run': formal_run,
            'rows': int(n_rows),
            'clips': clips,
            'top_export': TOP_EXPORT,
            'component_names': list(COMPONENT_NAMES),
            'training': False,
            'davis_read': False,
            'seconds': float(time.perf_counter() - start_all),
        },
        'integrity': {
            'input_audit': input_audit,
            'processed_rows': int(np.sum(processed)),
            'expected_rows': int(n_rows),
            'row_keys_unique': int(len(set((str(c), int(t), int(q)) for c, t, q in zip(clip_id_out, frame_tau_out, query_idx_out)))),
            'top129_unique_every_row': True,
            'top129_nonincreasing_every_row': True,
            'all_outputs_finite': True,
            'parity': parity,
            'pool_top16_ordered_difference_is_diagnostic_only': True,
            'pool_top16_set_is_hard_gate': True,
            'v9a5c_decision': v9c_report['gate']['decision'],
        },
        'opportunity_preview': opportunity_preview,
        'per_clip_runtime': clip_runtime,
        'decision': decision,
        'provenance': {
            'branch': input_audit['branch'],
            'head': input_audit['head'],
            'parent_head': input_audit['parent_head'],
            'script_sha256': sha256(Path(__file__).resolve()),
            'input_manifest_sha256': sha256(INPUT_MANIFEST),
            'v9a5c_json_sha256': sha256(V9C_JSON),
            'v9a5c_npz_sha256': sha256(V9C_NPZ),
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
                'rows': n_rows,
                'decision': decision,
                'parity': parity,
                'opportunity_preview': opportunity_preview,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
