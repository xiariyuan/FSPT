#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

WORKTREE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path('/gemini/code/FSPT')
TRACKON_CODE_ROOT = WORKTREE_ROOT / 'baselines/track_on'
TRACKON_ASSET_ROOT = SOURCE_ROOT / 'baselines/track_on'
if str(TRACKON_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKON_CODE_ROOT))
os.environ['DINOV3_LOCAL_DIR'] = str(
    SOURCE_ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m'
)

from model.trackon_predictor import Predictor  # noqa: E402
from utils.coord_utils import indices_to_coords  # noqa: E402
from utils.train_utils import load_args_from_yaml  # noqa: E402

SOURCE_BASE = SOURCE_ROOT / 'outputs/paper_discovery_2026-07-05'
POOL_PATH = SOURCE_BASE / 'v9a38_pointodyssey_pool/v9a38_pointodyssey_hypothesis_pool.npz'
PREFUSION_PATH = SOURCE_BASE / 'v9a43_prefusion_latents/v9a43_pointodyssey_c1_prefusion_latents.npz'
POD_ROOT = SOURCE_ROOT / 'datasets/pointodyssey/train'
CHECKPOINT = TRACKON_ASSET_ROOT / 'checkpoints_trackon2_dinov3.pt'
CONFIG = TRACKON_CODE_ROOT / 'config/test.yaml'
MANIFEST = WORKTREE_ROOT / 'docs/v9a45_input_manifest_2026-07-10.json'
OUTDIR = WORKTREE_ROOT / 'outputs/paper_discovery_2026-07-05/v9a45_conservative_residual'

NQ = 32
LENGTH = 96
SEQUENCES = ('ani', 'animal3', 'r4_new_f')
INITIAL_ALPHA = 0.05
MAXIMUM_ALPHA = 0.25
TIE_MARGIN_PX = 0.5
IMPROVEMENT_TRIGGER_PX = 1.0
SCORE_MARGIN = 0.5


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def stable_seed(sequence: str, start: int) -> int:
    digest = hashlib.sha256(f'{sequence}:{start}'.encode()).digest()
    return 20260710 + int.from_bytes(digest[:4], 'little')


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    checked = []
    for row in manifest['files']:
        file_path = Path(row['path'])
        if not file_path.exists():
            raise RuntimeError(f'manifest input missing: {file_path}')
        size = file_path.stat().st_size
        if size != int(row['size_bytes']):
            raise RuntimeError(
                f'manifest size mismatch: {file_path}: {size} != {row["size_bytes"]}'
            )
        digest = file_sha256(file_path)
        if digest != row['sha256']:
            raise RuntimeError(f'manifest hash mismatch: {file_path}')
        checked.append({'path': str(file_path), 'size_bytes': size, 'sha256': digest})

    frame_meta = manifest.get('frame_manifest')
    frame_audit = None
    if frame_meta and frame_meta.get('verify_each_file'):
        frame_manifest_path = Path(frame_meta['path'])
        frame_manifest = json.loads(frame_manifest_path.read_text())
        frame_root = Path(frame_manifest['root'])
        aggregate = hashlib.sha256()
        total_bytes = 0
        for row in frame_manifest['files']:
            relative_path = str(row['relative_path'])
            frame_path = frame_root / relative_path
            if not frame_path.exists():
                raise RuntimeError(f'frame input missing: {frame_path}')
            size = frame_path.stat().st_size
            if size != int(row['size_bytes']):
                raise RuntimeError(f'frame size mismatch: {frame_path}')
            digest = file_sha256(frame_path)
            if digest != row['sha256']:
                raise RuntimeError(f'frame hash mismatch: {frame_path}')
            total_bytes += size
            aggregate.update(relative_path.encode())
            aggregate.update(b'\0')
            aggregate.update(str(size).encode())
            aggregate.update(b'\0')
            aggregate.update(digest.encode())
            aggregate.update(b'\n')
        aggregate_digest = aggregate.hexdigest()
        if len(frame_manifest['files']) != int(frame_meta['file_count']):
            raise RuntimeError('frame-manifest file-count mismatch')
        if total_bytes != int(frame_meta['total_bytes']):
            raise RuntimeError('frame-manifest byte-count mismatch')
        if aggregate_digest != frame_meta['aggregate_sha256']:
            raise RuntimeError('frame-manifest aggregate hash mismatch')
        frame_audit = {
            'path': str(frame_manifest_path),
            'file_count': int(len(frame_manifest['files'])),
            'total_bytes': int(total_bytes),
            'aggregate_sha256': aggregate_digest,
        }

    clean_head = subprocess.check_output(
        ['git', '-C', str(WORKTREE_ROOT), 'rev-parse', 'HEAD'], text=True
    ).strip()
    clean_branch = subprocess.check_output(
        ['git', '-C', str(WORKTREE_ROOT), 'branch', '--show-current'], text=True
    ).strip()
    if clean_head != manifest['source_head']:
        raise RuntimeError(f'worktree HEAD mismatch: {clean_head} != {manifest["source_head"]}')
    if clean_branch != manifest['clean_branch']:
        raise RuntimeError(
            f'worktree branch mismatch: {clean_branch} != {manifest["clean_branch"]}'
        )
    tracked_status = subprocess.check_output(
        ['git', '-C', str(WORKTREE_ROOT), 'status', '--porcelain', '--untracked-files=no'],
        text=True,
    ).strip()
    if tracked_status:
        raise RuntimeError(f'worktree has tracked modifications: {tracked_status}')
    return {
        'pass': True,
        'clean_head': clean_head,
        'clean_branch': clean_branch,
        'tracked_status': tracked_status,
        'checked_files': checked,
        'frame_manifest': frame_audit,
    }


def load_frame(clip_id: str, tau: int) -> torch.Tensor:
    sequence, start_s = clip_id.split(':')
    frame_id = int(start_s) + int(tau)
    path = POD_ROOT / sequence / 'rgbs' / f'rgb_{frame_id:05d}.jpg'
    array = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().unsqueeze(0)
    return F.interpolate(tensor, size=(256, 256), mode='bilinear', align_corners=False)


def build_pointodyssey_gt(pool: Any) -> tuple[np.ndarray, float]:
    gt = np.zeros((len(pool['clip_id']), 2), dtype=np.float32)
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
    reproduction = np.linalg.norm(
        (pool['candidates'].astype(np.float32) - gt[:, None, :]) * 255.0,
        axis=-1,
    )
    max_abs = float(np.max(np.abs(reproduction - pool['error_px'].astype(np.float32))))
    if max_abs > 1e-4:
        raise RuntimeError(f'PointOdyssey GT reconstruction failed: max_abs={max_abs}')
    return gt, max_abs


def candidate_yx_from_model(position_xy: torch.Tensor) -> np.ndarray:
    output = position_xy.detach().cpu().float().numpy().copy()
    output[..., 0] *= 256.0 / 512.0
    output[..., 1] *= 256.0 / 384.0
    return (output[..., [1, 0]] / 255.0).astype(np.float32)


def candidate_set_hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    distance = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(max(distance.min(axis=1).max(), distance.min(axis=0).max()))


def row_errors(candidate_yx: np.ndarray, gt_yx: np.ndarray) -> np.ndarray:
    return np.linalg.norm((candidate_yx - gt_yx[:, None, :]) * 255.0, axis=-1).astype(np.float32)


def partial_rerank_hidden(
    head: nn.Module,
    q_t: torch.Tensor,
    f4: torch.Tensor,
    f8: torch.Tensor,
    f16: torch.Tensor,
    f32: torch.Tensor,
    correlation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_queries = q_t.shape[1]
    k = int(head.K)
    d_model = int(head.D)
    top_k_indices = torch.topk(correlation, k, dim=-1)[1]
    positions = indices_to_coords(top_k_indices, head.size, head.stride)
    positions_norm = positions / torch.tensor(
        [head.size[1], head.size[0]], device=q_t.device, dtype=positions.dtype
    )
    positions_norm = torch.clamp(positions_norm, 0, 1)
    positions_norm = positions_norm.view(1, n_queries * k, 1, 2)
    positions_norm = positions_norm.expand(-1, -1, head.num_level, -1)
    feature_scales = torch.cat([f4, f8, f16, f32], dim=1)
    q_top_k = q_t.unsqueeze(2).expand(-1, -1, k, -1).reshape(1, n_queries * k, d_model)
    for layer in head.local_decoder:
        q_top_k = layer(
            q=q_top_k,
            k=feature_scales,
            v=feature_scales,
            reference_points=positions_norm,
            spatial_shapes=head.spatial_shapes,
            start_levels=head.start_levels,
        )
    q_top_k = q_top_k.view(1, n_queries, k, d_model)
    q_top_k = torch.cat(
        [q_top_k, q_t.unsqueeze(2).expand(-1, -1, k, -1)], dim=-1
    )
    hidden = head.fusion_layer(q_top_k)
    return positions, hidden


class ConservativeResidualAdapter(nn.Module):
    def __init__(self, teacher_head: nn.Module) -> None:
        super().__init__()
        self.student_head = copy.deepcopy(teacher_head)
        for parameter in self.student_head.parameters():
            parameter.requires_grad = False
        for module in [self.student_head.local_decoder, self.student_head.fusion_layer]:
            for parameter in module.parameters():
                parameter.requires_grad = True
        self.delta_head = nn.Linear(int(teacher_head.D), 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        alpha_fraction = INITIAL_ALPHA / MAXIMUM_ALPHA
        alpha_logit = math.log(alpha_fraction / (1.0 - alpha_fraction))
        self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32))

    def alpha(self) -> torch.Tensor:
        return MAXIMUM_ALPHA * torch.sigmoid(self.alpha_logit)

    def forward(
        self,
        teacher_head: nn.Module,
        q_pre: torch.Tensor,
        frame_features: tuple[torch.Tensor, ...],
        correlation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        f4, f8, f16, f32, _ = frame_features
        with torch.no_grad():
            teacher_positions, teacher_hidden = partial_rerank_hidden(
                teacher_head, q_pre, f4, f8, f16, f32, correlation
            )
            teacher_score = teacher_head.score_layer(teacher_hidden).squeeze(-1).squeeze(0)
        student_positions, student_hidden = partial_rerank_hidden(
            self.student_head, q_pre, f4, f8, f16, f32, correlation
        )
        delta_score = self.delta_head(student_hidden).squeeze(-1).squeeze(0)
        score_residual = self.alpha() * delta_score
        student_score = teacher_score + score_residual
        return {
            'teacher_positions': teacher_positions,
            'student_positions': student_positions,
            'teacher_score': teacher_score,
            'student_score': student_score,
            'delta_score': delta_score,
            'score_residual': score_residual,
        }


def group_rows(pool: Any, indices: np.ndarray) -> dict[tuple[str, int], list[int]]:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in indices.tolist():
        groups[(str(pool['clip_id'][row]), int(pool['frame_tau'][row]))].append(int(row))
    return dict(groups)


def prepare_batch(
    base_model: nn.Module,
    pool: Any,
    prefusion: Any,
    gt: np.ndarray,
    rows: list[int],
    clip_id: str,
    tau: int,
    device: torch.device,
) -> dict[str, Any]:
    frame = load_frame(clip_id, tau).to(device)
    with torch.no_grad():
        frame_features = base_model.extract_frame_features(frame)
        q_pre = torch.from_numpy(prefusion['q_pre_latent'][rows].astype(np.float32)).to(device).unsqueeze(0)
        f4, f8, f16, f32, _ = frame_features
        correlation = base_model.multiscale_correlation(q_pre, f4, f8, f16, f32)
    return {
        'frame_features': frame_features,
        'q_pre': q_pre,
        'correlation': correlation,
        'gt': gt[rows],
        'rows': rows,
        'historical_c1': pool['candidates'][rows, 2:18].astype(np.float32),
    }


def compute_loss(
    student_score: torch.Tensor,
    teacher_score: torch.Tensor,
    error_px: torch.Tensor,
    score_residual: torch.Tensor,
    adapter: ConservativeResidualAdapter,
    anchor_state: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    oracle_error = torch.min(error_px, dim=1).values
    tie_mask = error_px <= (oracle_error[:, None] + TIE_MARGIN_PX)
    tie_target = tie_mask.float() / torch.clamp(tie_mask.sum(dim=1, keepdim=True).float(), min=1.0)
    log_student = torch.log_softmax(student_score, dim=1)
    loss_rank = -(tie_target * log_student).sum(dim=1).mean()

    teacher_probability = torch.softmax(teacher_score.detach(), dim=1)
    loss_kl = F.kl_div(log_student, teacher_probability, reduction='batchmean')

    teacher_index = torch.argmax(teacher_score.detach(), dim=1)
    teacher_error = error_px.gather(1, teacher_index[:, None]).squeeze(1)
    teacher_good = teacher_error <= (oracle_error + 1.0)
    if bool(torch.any(teacher_good)):
        loss_retention = F.cross_entropy(student_score[teacher_good], teacher_index[teacher_good])
    else:
        loss_retention = student_score.sum() * 0.0

    improvement = teacher_error > (oracle_error + IMPROVEMENT_TRIGGER_PX)
    if bool(torch.any(improvement)):
        masked_best = student_score.masked_fill(~tie_mask, -torch.inf)
        best_logit = torch.max(masked_best, dim=1).values
        teacher_logit = student_score.gather(1, teacher_index[:, None]).squeeze(1)
        loss_improvement = F.relu(
            SCORE_MARGIN - (best_logit[improvement] - teacher_logit[improvement])
        ).mean()
    else:
        loss_improvement = student_score.sum() * 0.0

    loss_delta = torch.mean(score_residual ** 2)
    loss_anchor = torch.zeros((), device=student_score.device)
    for name, parameter in adapter.named_parameters():
        if name in anchor_state:
            loss_anchor = loss_anchor + torch.mean((parameter - anchor_state[name]) ** 2)

    total = (
        loss_rank
        + 2.0 * loss_kl
        + 1.0 * loss_retention
        + 0.5 * loss_improvement
        + 0.05 * loss_delta
        + 1e-4 * loss_anchor
    )
    return total, {
        'total': float(total.detach()),
        'rank': float(loss_rank.detach()),
        'kl': float(loss_kl.detach()),
        'retention': float(loss_retention.detach()),
        'improvement': float(loss_improvement.detach()),
        'delta': float(loss_delta.detach()),
        'anchor': float(loss_anchor.detach()),
        'teacher_good_rate': float(torch.mean(teacher_good.float()).detach()),
        'improvement_rate': float(torch.mean(improvement.float()).detach()),
    }


def make_optimizer(adapter: ConservativeResidualAdapter) -> torch.optim.Optimizer:
    local_parameters = [p for p in adapter.student_head.local_decoder.parameters() if p.requires_grad]
    fusion_parameters = [p for p in adapter.student_head.fusion_layer.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        [
            {'params': local_parameters, 'lr': 1e-6},
            {'params': fusion_parameters, 'lr': 5e-6},
            {'params': list(adapter.delta_head.parameters()), 'lr': 5e-5},
            {'params': [adapter.alpha_logit], 'lr': 1e-3},
        ],
        weight_decay=1e-4,
    )


def gradient_groups(adapter: ConservativeResidualAdapter) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
    for name, parameter in adapter.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if name.startswith('student_head.local_decoder.'):
            group = 'local_decoder'
        elif name.startswith('student_head.fusion_layer.'):
            group = 'fusion_layer'
        elif name.startswith('delta_head.'):
            group = 'delta_head'
        elif name == 'alpha_logit':
            group = 'alpha_logit'
        else:
            group = 'other'
        grouped[group].append(parameter.grad.detach().float().reshape(-1))
    result = {}
    for group in ['local_decoder', 'fusion_layer', 'delta_head', 'alpha_logit']:
        values = grouped.get(group, [])
        if values:
            vector = torch.cat(values)
            result[group] = {
                'norm': float(torch.linalg.vector_norm(vector)),
                'nonzero': int(torch.count_nonzero(vector).item()),
                'elements': int(vector.numel()),
            }
        else:
            result[group] = {'norm': 0.0, 'nonzero': 0, 'elements': 0}
    return result


def initial_integrity_check(
    teacher_head: nn.Module,
    adapter: ConservativeResidualAdapter,
    batch: dict[str, Any],
) -> dict[str, Any]:
    teacher_head.eval()
    adapter.eval()
    f4, f8, f16, f32, _ = batch['frame_features']
    with torch.no_grad():
        _, full_positions, _, full_score = teacher_head(
            batch['q_pre'], f4, f8, f16, f32, batch['correlation']
        )
        output = adapter(
            teacher_head, batch['q_pre'], batch['frame_features'], batch['correlation']
        )
    forward_position_diff = float(torch.max(torch.abs(full_positions - output['teacher_positions'])))
    forward_score_diff = float(torch.max(torch.abs(full_score.squeeze(0) - output['teacher_score'])))
    initial_score_diff = float(torch.max(torch.abs(output['teacher_score'] - output['student_score'])))
    candidate_diff = float(torch.max(torch.abs(output['teacher_positions'] - output['student_positions'])))
    if forward_position_diff > 1e-7 or forward_score_diff > 1e-6:
        raise RuntimeError(
            f'partial rerank mismatch: position={forward_position_diff}, score={forward_score_diff}'
        )
    if initial_score_diff > 1e-7:
        raise RuntimeError(f'initial score mismatch: {initial_score_diff}')
    if candidate_diff > 1e-8:
        raise RuntimeError(f'candidate mismatch: {candidate_diff}')
    positions_yx = candidate_yx_from_model(output['teacher_positions'].squeeze(0))
    historical_ordered = float(np.max(np.abs(positions_yx - batch['historical_c1'])))
    historical_set = max(
        candidate_set_hausdorff(a, b)
        for a, b in zip(positions_yx, batch['historical_c1'])
    )
    return {
        'partial_vs_original_position_max_abs': forward_position_diff,
        'partial_vs_original_score_max_abs': forward_score_diff,
        'initial_student_teacher_score_max_abs': initial_score_diff,
        'teacher_student_candidate_max_abs': candidate_diff,
        'historical_ordered_candidate_max_abs': historical_ordered,
        'historical_candidate_set_hausdorff_max': float(historical_set),
        'initial_alpha': float(adapter.alpha().detach()),
    }


def two_step_gradient_integrity(
    teacher_head: nn.Module,
    adapter: ConservativeResidualAdapter,
    batch: dict[str, Any],
    anchor_state: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    probe = copy.deepcopy(adapter).to(device).eval()
    probe_anchor = {
        name: parameter.detach().clone()
        for name, parameter in probe.named_parameters()
        if parameter.requires_grad
    }
    optimizer = make_optimizer(probe)
    steps = []
    for step in range(2):
        output = probe(
            teacher_head, batch['q_pre'], batch['frame_features'], batch['correlation']
        )
        positions_yx = candidate_yx_from_model(output['teacher_positions'].squeeze(0))
        error = torch.from_numpy(row_errors(positions_yx, batch['gt'])).to(device)
        loss, components = compute_loss(
            output['student_score'], output['teacher_score'], error,
            output['score_residual'], probe, probe_anchor
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        groups = gradient_groups(probe)
        all_grads = [
            p.grad.detach().float().reshape(-1)
            for p in probe.parameters()
            if p.requires_grad and p.grad is not None
        ]
        if not all_grads:
            raise RuntimeError(f'no gradients at integrity step {step}')
        total_norm = float(torch.linalg.vector_norm(torch.cat(all_grads)))
        if not np.isfinite(total_norm) or total_norm <= 0:
            raise RuntimeError(f'invalid gradient norm at step {step}: {total_norm}')
        steps.append({
            'step': step,
            'loss': components,
            'gradient_norm': total_norm,
            'gradient_by_group': groups,
            'alpha_before_step': float(probe.alpha().detach()),
        })
        torch.nn.utils.clip_grad_norm_([p for p in probe.parameters() if p.requires_grad], 1.0)
        optimizer.step()
    if steps[0]['gradient_by_group']['delta_head']['norm'] <= 0:
        raise RuntimeError('delta head has zero first-step gradient')
    if steps[1]['gradient_by_group']['local_decoder']['norm'] <= 0:
        raise RuntimeError('local decoder has zero second-step gradient')
    if steps[1]['gradient_by_group']['fusion_layer']['norm'] <= 0:
        raise RuntimeError('fusion layer has zero second-step gradient')
    return {
        'steps': steps,
        'expected_first_step_zero_upstream': True,
        'second_step_upstream_gradient_pass': True,
        'probe_final_alpha': float(probe.alpha().detach()),
    }


def train_one_epoch(
    base_model: nn.Module,
    teacher_head: nn.Module,
    adapter: ConservativeResidualAdapter,
    optimizer: torch.optim.Optimizer,
    pool: Any,
    prefusion: Any,
    gt: np.ndarray,
    groups: dict[tuple[str, int], list[int]],
    anchor_state: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    base_model.eval()
    teacher_head.eval()
    adapter.eval()
    order = sorted(groups.keys())
    random.Random(seed).shuffle(order)
    sums: dict[str, float] = defaultdict(float)
    row_count = 0
    grad_norms = []
    ordered_diffs = []
    set_diffs = []
    group_grad_sums: dict[str, float] = defaultdict(float)
    group_grad_nonzero_batches: dict[str, int] = defaultdict(int)
    for clip_id, tau in order:
        rows = groups[(clip_id, tau)]
        batch = prepare_batch(base_model, pool, prefusion, gt, rows, clip_id, tau, device)
        output = adapter(teacher_head, batch['q_pre'], batch['frame_features'], batch['correlation'])
        candidate_diff = float(torch.max(torch.abs(
            output['teacher_positions'] - output['student_positions']
        )))
        if candidate_diff > 1e-8:
            raise RuntimeError(f'candidate mismatch at {clip_id}:{tau}: {candidate_diff}')
        positions = candidate_yx_from_model(output['student_positions'].squeeze(0))
        error = torch.from_numpy(row_errors(positions, batch['gt'])).to(device)
        loss, components = compute_loss(
            output['student_score'], output['teacher_score'], error,
            output['score_residual'], adapter, anchor_state
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f'non-finite loss at {clip_id}:{tau}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        group_grads = gradient_groups(adapter)
        for group, stats in group_grads.items():
            group_grad_sums[group] += float(stats['norm'])
            group_grad_nonzero_batches[group] += int(float(stats['norm']) > 0.0)
        trainable = [p for p in adapter.parameters() if p.requires_grad]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        if not np.isfinite(grad_norm):
            raise RuntimeError(f'non-finite gradient norm at {clip_id}:{tau}')
        optimizer.step()
        n_rows = len(rows)
        for key, value in components.items():
            sums[key] += float(value) * n_rows
        row_count += n_rows
        grad_norms.append(grad_norm)
        ordered_diffs.extend(np.max(np.abs(positions - batch['historical_c1']), axis=(1, 2)).tolist())
        set_diffs.extend(
            candidate_set_hausdorff(a, b)
            for a, b in zip(positions, batch['historical_c1'])
        )
    return {
        'rows': int(row_count),
        'batches': int(len(order)),
        'mean_losses': {key: float(value / max(1, row_count)) for key, value in sums.items()},
        'grad_norm_mean': float(np.mean(grad_norms)),
        'grad_norm_p95': float(np.quantile(grad_norms, 0.95)),
        'gradient_by_group': {
            group: {
                'mean_norm_over_batches': float(group_grad_sums[group] / max(1, len(order))),
                'nonzero_batches': int(group_grad_nonzero_batches[group]),
                'total_batches': int(len(order)),
            }
            for group in ['local_decoder', 'fusion_layer', 'delta_head', 'alpha_logit']
        },
        'alpha': float(adapter.alpha().detach()),
        'historical_ordered_candidate_max_abs': float(np.max(ordered_diffs)),
        'historical_candidate_set_hausdorff_max': float(np.max(set_diffs)),
    }


def metric_summary(error: np.ndarray, oracle: np.ndarray, selected_index: np.ndarray, oracle_index: np.ndarray) -> dict[str, Any]:
    return {
        'n_rows': int(len(error)),
        'mean_error': float(np.mean(error)),
        'median_error': float(np.median(error)),
        'safe4': int(np.sum(error <= 4.0)),
        'safe8': int(np.sum(error <= 8.0)),
        'safe16': int(np.sum(error <= 16.0)),
        'oracle_regret': float(np.mean(error - oracle)),
        'exact_best_rate': float(np.mean(selected_index == oracle_index)),
        'within1px_best_rate': float(np.mean(error <= oracle + 1.0)),
    }


def comparison_summary(
    teacher: np.ndarray,
    student: np.ndarray,
    oracle: np.ndarray,
    teacher_index: np.ndarray,
    student_index: np.ndarray,
) -> dict[str, Any]:
    tol = 1e-6
    better = int(np.sum(student < teacher - tol))
    worse = int(np.sum(student > teacher + tol))
    equal = int(len(student) - better - worse)
    teacher_good = teacher <= oracle + 1.0
    retention = float(np.mean(student[teacher_good] <= oracle[teacher_good] + 1.0)) if np.any(teacher_good) else None
    opportunity = teacher > oracle + 1.0
    opportunity_success = (
        float(np.mean(student[opportunity] < teacher[opportunity] - tol))
        if np.any(opportunity) else None
    )
    return {
        'mean_difference_student_minus_teacher': float(np.mean(student - teacher)),
        'median_difference_student_minus_teacher': float(np.median(student - teacher)),
        'better': better,
        'worse': worse,
        'equal': equal,
        'teacher_student_top1_match_rate': float(np.mean(student_index == teacher_index)),
        'teacher_good_rows': int(np.sum(teacher_good)),
        'teacher_good_retention': retention,
        'improvement_opportunity_rows': int(np.sum(opportunity)),
        'improvement_opportunity_success': opportunity_success,
    }


@torch.no_grad()
def evaluate_sequence(
    base_model: nn.Module,
    teacher_head: nn.Module,
    adapter: ConservativeResidualAdapter,
    pool: Any,
    prefusion: Any,
    gt: np.ndarray,
    groups: dict[tuple[str, int], list[int]],
    device: torch.device,
) -> dict[str, Any]:
    base_model.eval()
    teacher_head.eval()
    adapter.eval()
    records = []
    historical_ordered = []
    historical_set = []
    score_changes = []
    for clip_id, tau in sorted(groups.keys()):
        rows = groups[(clip_id, tau)]
        batch = prepare_batch(base_model, pool, prefusion, gt, rows, clip_id, tau, device)
        output = adapter(teacher_head, batch['q_pre'], batch['frame_features'], batch['correlation'])
        candidate_diff = float(torch.max(torch.abs(
            output['teacher_positions'] - output['student_positions']
        )))
        if candidate_diff > 1e-8:
            raise RuntimeError(f'eval candidate mismatch at {clip_id}:{tau}: {candidate_diff}')
        positions = candidate_yx_from_model(output['teacher_positions'].squeeze(0))
        error = row_errors(positions, batch['gt'])
        teacher_index = torch.argmax(output['teacher_score'], dim=1).cpu().numpy()
        student_index = torch.argmax(output['student_score'], dim=1).cpu().numpy()
        oracle_index = np.argmin(error, axis=1)
        local = np.arange(len(rows))
        for j, row_id in enumerate(rows):
            records.append({
                'row_id': int(row_id),
                'clip_id': str(clip_id),
                'sequence': str(pool['sequence'][row_id]),
                'is_hard': bool(pool['is_hard'][row_id]),
                'teacher_error': float(error[local[j], teacher_index[j]]),
                'student_error': float(error[local[j], student_index[j]]),
                'oracle_error': float(error[local[j], oracle_index[j]]),
                'teacher_index': int(teacher_index[j]),
                'student_index': int(student_index[j]),
                'oracle_index': int(oracle_index[j]),
            })
        score_changes.extend(
            torch.max(torch.abs(output['student_score'] - output['teacher_score']), dim=1)
            .values.cpu().tolist()
        )
        historical_ordered.extend(np.max(np.abs(positions - batch['historical_c1']), axis=(1, 2)).tolist())
        historical_set.extend(
            candidate_set_hausdorff(a, b)
            for a, b in zip(positions, batch['historical_c1'])
        )

    teacher = np.asarray([r['teacher_error'] for r in records], dtype=np.float64)
    student = np.asarray([r['student_error'] for r in records], dtype=np.float64)
    oracle = np.asarray([r['oracle_error'] for r in records], dtype=np.float64)
    teacher_index = np.asarray([r['teacher_index'] for r in records], dtype=np.int64)
    student_index = np.asarray([r['student_index'] for r in records], dtype=np.int64)
    oracle_index = np.asarray([r['oracle_index'] for r in records], dtype=np.int64)

    teacher_summary = metric_summary(teacher, oracle, teacher_index, oracle_index)
    student_summary = metric_summary(student, oracle, student_index, oracle_index)
    comparison = comparison_summary(teacher, student, oracle, teacher_index, student_index)
    gate = {
        'mean_error_improved': bool(student_summary['mean_error'] < teacher_summary['mean_error']),
        'better_gt_worse': bool(comparison['better'] > comparison['worse']),
        'safe16_not_decreased': bool(student_summary['safe16'] >= teacher_summary['safe16']),
        'teacher_good_retention_ge_095': bool(
            comparison['teacher_good_retention'] is not None
            and comparison['teacher_good_retention'] >= 0.95
        ),
        'oracle_regret_improved': bool(
            student_summary['oracle_regret'] < teacher_summary['oracle_regret']
        ),
    }
    gate['pass_all'] = bool(all(gate.values()))

    subgroup = {}
    masks = {
        'hard': np.asarray([r['is_hard'] for r in records], dtype=bool),
        'easy': ~np.asarray([r['is_hard'] for r in records], dtype=bool),
    }
    for name, mask in masks.items():
        subgroup[name] = {
            'teacher': metric_summary(
                teacher[mask], oracle[mask], teacher_index[mask], oracle_index[mask]
            ),
            'student': metric_summary(
                student[mask], oracle[mask], student_index[mask], oracle_index[mask]
            ),
            'comparison': comparison_summary(
                teacher[mask], student[mask], oracle[mask],
                teacher_index[mask], student_index[mask]
            ),
        }

    per_clip = {}
    for clip_id in sorted(set(r['clip_id'] for r in records)):
        mask = np.asarray([r['clip_id'] == clip_id for r in records], dtype=bool)
        per_clip[clip_id] = {
            'teacher': metric_summary(
                teacher[mask], oracle[mask], teacher_index[mask], oracle_index[mask]
            ),
            'student': metric_summary(
                student[mask], oracle[mask], student_index[mask], oracle_index[mask]
            ),
            'comparison': comparison_summary(
                teacher[mask], student[mask], oracle[mask],
                teacher_index[mask], student_index[mask]
            ),
        }

    return {
        'n_rows': int(len(records)),
        'teacher': teacher_summary,
        'student': student_summary,
        'comparison': comparison,
        'subgroups': subgroup,
        'per_clip': per_clip,
        'candidate_audit': {
            'historical_ordered_candidate_max_abs': float(np.max(historical_ordered)),
            'historical_candidate_set_hausdorff_max': float(np.max(historical_set)),
            'score_change_max_abs_mean': float(np.mean(score_changes)),
            'score_change_max_abs_p95': float(np.quantile(score_changes, 0.95)),
        },
        'gate': gate,
    }


def clone_states(module: nn.Module, trainable: bool) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
        if bool(parameter.requires_grad) == trainable
    }


def parameter_audit(
    base_model: nn.Module,
    teacher_head: nn.Module,
    adapter: ConservativeResidualAdapter,
    base_probe_before: torch.Tensor,
    base_probe: torch.Tensor,
    trainable_before: dict[str, torch.Tensor],
    frozen_student_before: dict[str, torch.Tensor],
    teacher_before: dict[str, torch.Tensor],
) -> dict[str, Any]:
    grouped: dict[str, dict[str, float | int]] = {}
    max_trainable = 0.0
    for name, parameter in adapter.named_parameters():
        if name not in trainable_before:
            continue
        delta = parameter.detach().float() - trainable_before[name].float()
        max_abs = float(torch.max(torch.abs(delta)))
        l2 = float(torch.linalg.vector_norm(delta.reshape(-1)))
        max_trainable = max(max_trainable, max_abs)
        if name.startswith('student_head.local_decoder.'):
            group = 'local_decoder'
        elif name.startswith('student_head.fusion_layer.'):
            group = 'fusion_layer'
        elif name.startswith('delta_head.'):
            group = 'delta_head'
        elif name == 'alpha_logit':
            group = 'alpha_logit'
        else:
            group = 'other'
        row = grouped.setdefault(group, {'max_abs': 0.0, 'l2_squared': 0.0, 'parameters': 0})
        row['max_abs'] = max(float(row['max_abs']), max_abs)
        row['l2_squared'] = float(row['l2_squared']) + l2 * l2
        row['parameters'] = int(row['parameters']) + int(parameter.numel())
    grouped_out = {
        name: {
            'max_abs': float(row['max_abs']),
            'l2': float(math.sqrt(float(row['l2_squared']))),
            'parameters': int(row['parameters']),
        }
        for name, row in grouped.items()
    }

    adapter_params = dict(adapter.named_parameters())
    frozen_student_max = max(
        float(torch.max(torch.abs(adapter_params[name].detach() - before)))
        for name, before in frozen_student_before.items()
    ) if frozen_student_before else 0.0
    teacher_params = dict(teacher_head.named_parameters())
    teacher_max = max(
        float(torch.max(torch.abs(teacher_params[name].detach() - before)))
        for name, before in teacher_before.items()
    ) if teacher_before else 0.0
    frozen_grad_tensors = sum(
        1
        for parameter in adapter.parameters()
        if not parameter.requires_grad
        and parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
    )
    return {
        'max_trainable_parameter_change': max_trainable,
        'trainable_change_by_group': grouped_out,
        'frozen_base_probe_change': float(torch.max(torch.abs(base_probe.detach() - base_probe_before))),
        'frozen_student_max_parameter_change': frozen_student_max,
        'teacher_max_parameter_change': teacher_max,
        'frozen_student_grad_tensors': int(frozen_grad_tensors),
        'trainable_parameter_count': int(sum(p.numel() for p in adapter.parameters() if p.requires_grad)),
        'total_base_parameter_count': int(sum(p.numel() for p in base_model.parameters())),
        'final_alpha': float(adapter.alpha().detach()),
    }


def write_result_doc(result: dict[str, Any], path: Path) -> None:
    validation = result['validation']
    teacher = validation['teacher']
    student = validation['student']
    comparison = validation['comparison']
    gate = validation['gate']
    lines = [
        '# V9-A4.5 Conservative Residual End-to-End Smoke Result',
        '',
        f"Holdout sequence: `{result['protocol']['holdout_sequence']}`; train rows={result['protocol']['train_rows']}; validation rows={result['protocol']['validation_rows']}; epochs={result['protocol']['epochs']}.",
        '',
        '## Integrity',
        '',
        '```json',
        json.dumps(result['integrity'], indent=2),
        '```',
        '',
        '## Training',
        '',
        '| Epoch | Total | Rank | KL | Retention | Improve | Alpha | Grad p95 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in result['training_history']:
        loss = row['mean_losses']
        lines.append(
            f"| {row['epoch']} | {loss['total']:.4f} | {loss['rank']:.4f} | {loss['kl']:.4f} | "
            f"{loss['retention']:.4f} | {loss['improvement']:.4f} | {row['alpha']:.6f} | {row['grad_norm_p95']:.4f} |"
        )
    lines += [
        '',
        '## Heldout synthetic sequence',
        '',
        '| Model | Mean | Median | Safe4/8/16 | Oracle regret | Exact best | Within1px |',
        '|---|---:|---:|---:|---:|---:|---:|',
        f"| teacher | {teacher['mean_error']:.4f} | {teacher['median_error']:.4f} | {teacher['safe4']}/{teacher['safe8']}/{teacher['safe16']} | {teacher['oracle_regret']:.4f} | {teacher['exact_best_rate']:.4f} | {teacher['within1px_best_rate']:.4f} |",
        f"| student | {student['mean_error']:.4f} | {student['median_error']:.4f} | {student['safe4']}/{student['safe8']}/{student['safe16']} | {student['oracle_regret']:.4f} | {student['exact_best_rate']:.4f} | {student['within1px_best_rate']:.4f} |",
        '',
        f"Better/Worse/Equal: {comparison['better']}/{comparison['worse']}/{comparison['equal']}",
        '',
        f"Teacher-good retention: {comparison['teacher_good_retention']:.4f}",
        '',
        f"Improvement-opportunity success: {comparison['improvement_opportunity_success']:.4f}",
        '',
        '## Hard/easy diagnostics',
        '',
        '| Subset | Teacher mean | Student mean | Better/Worse/Equal | Teacher-good retention |',
        '|---|---:|---:|---|---:|',
    ]
    for name, row in validation['subgroups'].items():
        lines.append(
            f"| {name} | {row['teacher']['mean_error']:.4f} | {row['student']['mean_error']:.4f} | "
            f"{row['comparison']['better']}/{row['comparison']['worse']}/{row['comparison']['equal']} | "
            f"{row['comparison']['teacher_good_retention']:.4f} |"
        )
    lines += [
        '',
        '## Synthetic gate',
        '',
        '```json',
        json.dumps(gate, indent=2),
        '```',
        '',
        '## Decision',
        '',
        result['decision'],
    ]
    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--holdout', choices=SEQUENCES, default='ani')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20260710)
    parser.add_argument('--integrity-only', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1:
        raise SystemExit('--epochs must be positive')

    start_all = time.perf_counter()
    manifest_audit = verify_manifest(MANIFEST)
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pool = np.load(POOL_PATH, allow_pickle=True)
    prefusion = np.load(PREFUSION_PATH, allow_pickle=True)
    for key in ['clip_id', 'sequence', 'frame_tau', 'query_idx']:
        if not np.array_equal(pool[key], prefusion[key]):
            raise RuntimeError(f'pool/prefusion row alignment mismatch: {key}')
    gt, gt_reproduction = build_pointodyssey_gt(pool)
    sequence = np.asarray(pool['sequence']).astype(str)
    train_indices = np.where(sequence != args.holdout)[0]
    validation_indices = np.where(sequence == args.holdout)[0]
    train_sequences = sorted(set(sequence[train_indices].tolist()))
    validation_sequences = sorted(set(sequence[validation_indices].tolist()))
    overlap = sorted(set(train_sequences) & set(validation_sequences))
    if overlap:
        raise RuntimeError(f'train/validation leakage: {overlap}')
    train_groups = group_rows(pool, train_indices)
    validation_groups = group_rows(pool, validation_indices)

    config = load_args_from_yaml(str(CONFIG))
    config.grad_checkpoint = False
    config.M_i = config.M
    predictor = Predictor(
        config, checkpoint_path=str(CHECKPOINT), support_grid_size=20
    ).to(device).eval()
    base_model = predictor.model
    for parameter in base_model.parameters():
        parameter.requires_grad = False
    teacher_head = copy.deepcopy(base_model.reranking_head).to(device).eval()
    for parameter in teacher_head.parameters():
        parameter.requires_grad = False
    adapter = ConservativeResidualAdapter(teacher_head).to(device).eval()
    optimizer = make_optimizer(adapter)

    trainable_before = clone_states(adapter, trainable=True)
    frozen_student_before = clone_states(adapter, trainable=False)
    teacher_before = {
        name: parameter.detach().clone()
        for name, parameter in teacher_head.named_parameters()
    }
    base_probe_name, base_probe = next(
        (name, parameter)
        for name, parameter in base_model.named_parameters()
        if parameter.numel() > 1000
    )
    base_probe_before = base_probe.detach().clone()

    first_key = sorted(train_groups.keys())[0]
    first_batch = prepare_batch(
        base_model, pool, prefusion, gt, train_groups[first_key],
        first_key[0], first_key[1], device
    )
    integrity = initial_integrity_check(teacher_head, adapter, first_batch)
    integrity.update({
        'manifest': manifest_audit,
        'gt_error_reproduction_max_abs': gt_reproduction,
        'pool_prefusion_alignment_pass': True,
        'train_sequences': train_sequences,
        'validation_sequences': validation_sequences,
        'train_validation_sequence_overlap': overlap,
        'device': str(device),
        'trackon_code_root': str(TRACKON_CODE_ROOT),
        'checkpoint_path': str(CHECKPOINT),
        'base_probe_name': base_probe_name,
        'two_step_gradient': two_step_gradient_integrity(
            teacher_head, adapter, first_batch, trainable_before, device
        ),
    })

    if args.integrity_only:
        print(json.dumps({'ok': True, 'integrity': integrity}, ensure_ascii=False, indent=2))
        return

    history = []
    for epoch in range(args.epochs):
        epoch_result = train_one_epoch(
            base_model, teacher_head, adapter, optimizer, pool, prefusion, gt,
            train_groups, trainable_before, device, args.seed + epoch
        )
        epoch_result['epoch'] = int(epoch)
        history.append(epoch_result)
        print(json.dumps({
            'epoch': epoch,
            'mean_total_loss': epoch_result['mean_losses']['total'],
            'alpha': epoch_result['alpha'],
            'rows': epoch_result['rows'],
        }), flush=True)

    validation = evaluate_sequence(
        base_model, teacher_head, adapter, pool, prefusion, gt,
        validation_groups, device
    )
    parameters = parameter_audit(
        base_model, teacher_head, adapter,
        base_probe_before, base_probe,
        trainable_before, frozen_student_before, teacher_before
    )
    if parameters['frozen_base_probe_change'] != 0.0:
        raise RuntimeError('frozen base parameter changed')
    if parameters['frozen_student_max_parameter_change'] != 0.0:
        raise RuntimeError('frozen student parameter changed')
    if parameters['teacher_max_parameter_change'] != 0.0:
        raise RuntimeError('teacher parameter changed')
    if parameters['frozen_student_grad_tensors'] != 0:
        raise RuntimeError('frozen student parameter received gradient')
    if parameters['max_trainable_parameter_change'] <= 0.0:
        raise RuntimeError('trainable parameters did not move')

    collapse = (
        validation['student']['mean_error'] > validation['teacher']['mean_error'] + 0.25
        and validation['comparison']['worse'] > validation['comparison']['better']
    )
    if validation['gate']['pass_all']:
        decision = (
            'Synthetic heldout gate passes on the smoke fold. Proceed to the remaining two '
            'predeclared sequence-heldout folds before any DAVIS evaluation.'
        )
    elif collapse:
        decision = (
            'Integrity passes, but heldout synthetic ranking collapses materially. Stop before '
            'DAVIS and do not scale this objective.'
        )
    else:
        decision = (
            'Integrity passes but the one-epoch smoke does not satisfy every synthetic gate. '
            'Record the miss and inspect whether it is close or directionally harmful before continuing.'
        )

    result = {
        'script': 'scripts/v9a45_conservative_residual_end_to_end.py',
        'provenance': {
            'script_sha256': file_sha256(Path(__file__)),
            'worktree_root': str(WORKTREE_ROOT),
            'trackon_code_root': str(TRACKON_CODE_ROOT),
            'source_asset_root': str(SOURCE_ROOT),
            'branch': manifest_audit['clean_branch'],
            'head': manifest_audit['clean_head'],
        },
        'protocol': {
            'holdout_sequence': args.holdout,
            'train_sequences': train_sequences,
            'validation_sequences': validation_sequences,
            'train_rows': int(len(train_indices)),
            'validation_rows': int(len(validation_indices)),
            'train_unique_frames': int(len(train_groups)),
            'validation_unique_frames': int(len(validation_groups)),
            'epochs': int(args.epochs),
            'seed': int(args.seed),
            'davis_labels_used': False,
            'davis_evaluation_run': False,
            'model_selection_on_davis': False,
            'trajectory_threshold_sweep': False,
            'initial_alpha': INITIAL_ALPHA,
            'maximum_alpha': MAXIMUM_ALPHA,
            'dropout_disabled_during_adaptation': True,
        },
        'integrity': integrity,
        'training_history': history,
        'parameter_audit': parameters,
        'validation': validation,
        'decision': decision,
        'seconds': float(time.perf_counter() - start_all),
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    stem = f'v9a45_smoke_holdout_{args.holdout}_e{args.epochs}_seed{args.seed}'
    out_json = OUTDIR / f'{stem}.json'
    out_doc = WORKTREE_ROOT / 'docs' / f'{stem}_result_2026-07-10.md'
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    write_result_doc(result, out_doc)
    print(json.dumps({
        'ok': True,
        'json': str(out_json),
        'doc': str(out_doc),
        'gate': validation['gate'],
        'teacher_mean_error': validation['teacher']['mean_error'],
        'student_mean_error': validation['student']['mean_error'],
        'better_worse_equal': [
            validation['comparison']['better'],
            validation['comparison']['worse'],
            validation['comparison']['equal'],
        ],
        'teacher_good_retention': validation['comparison']['teacher_good_retention'],
        'decision': decision,
        'seconds': result['seconds'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
