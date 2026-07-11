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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

WORKTREE_ROOT = Path('/gemini/code/FSPT_v9a52_clean')
SOURCE_ROOT = Path('/gemini/code/FSPT')
TRACKON_ROOT = WORKTREE_ROOT / 'baselines/track_on'
POD_ROOT = SOURCE_ROOT / 'datasets/pointodyssey/train'
CHECKPOINT = SOURCE_ROOT / 'baselines/track_on/checkpoints_trackon2_dinov3.pt'
DINOV3_WEIGHTS = SOURCE_ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m/model.safetensors'
DINOV3_CONFIG = SOURCE_ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m/config.json'
CONFIG = TRACKON_ROOT / 'config/test.yaml'
EVENT_MANIFEST = WORKTREE_ROOT / 'docs/v9a52_independent_state_event_manifest_2026-07-11.json'
FRAME_MANIFEST = WORKTREE_ROOT / 'docs/v9a51c_pointodyssey_frame_manifest_2026-07-10.json'
V9B_SCRIPT = WORKTREE_ROOT / 'scripts/v9a51b_candidate_conditioned_refinement_audit.py'
V9B_NPZ = WORKTREE_ROOT / 'outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_candidate_conditioned_refinement_rows.npz'
V9C_NPZ = WORKTREE_ROOT / 'outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_history_preserving_beam_rows.npz'
OUTDIR = WORKTREE_ROOT / 'outputs/paper_discovery_2026-07-05/v9a52_independent_state'
DEFAULT_JSON = OUTDIR / 'v9a52_independent_state_smoke.json'
DEFAULT_NPZ = OUTDIR / 'v9a52_independent_state_smoke_rows.npz'
DEFAULT_DOC = WORKTREE_ROOT / 'docs/v9a52_independent_state_smoke_result_2026-07-11.md'

EXPECTED_HEAD = '74882d3c8a7ff2539fe7319482c7f236e8ff2a0c'
EXPECTED_BRANCH = 'v9a52-independent-state-branching-20260711'
EXPECTED_HASHES = {
    CHECKPOINT: '0e319c279cbdf51a5fc761b47dc1969520e8cfccfb57dc5a019a8c56e1039cd4',
    DINOV3_WEIGHTS: '208146e499dace99e4c9376ddb8a26f77d64c31c46c4dc4b86ff8bc63b0235e2',
    DINOV3_CONFIG: '6f4ac67fea1761fe684d2a7db3139bab2d0dfdf94c05063d5992717c4c1da0ac',
    CONFIG: '34c08c71be511b2cda6aaafb6e2e3af7e21c6fd5197de98a54fa4b9ac7a78c8e',
    POD_ROOT / 'ani/anno.npz': 'cc20ff10b815d607a09a48895a8d76e9cb8169c108ddb9fdacb2955dc26c2368',
    EVENT_MANIFEST: 'e5f5fa34c840e5159726ada1c7752e8bfa273b48ad6cd215e7519ed71721fb4b',
    FRAME_MANIFEST: 'b1a92a0c5a627fb9a554cd45201dc172bd2b7edaaf13ce78a40eecaddf7effba',
    V9B_SCRIPT: '1a0e4a75db2f48e75fae1d3cab1b11708f77ef1ea96964a26beba545d6e129f5',
    V9B_NPZ: 'eb9736cbe7fc1af08e5bfe23248b333e18ae01c826fd26fc65aa40945746d266',
    V9C_NPZ: '05b450fb2031ce1d762ed7436bcef34a47dfaf37c10d1c33bd8fd74c1d0381a0',
}

NQ = 32
SUPPORT_GRID_SIZE = 20
EXPECTED_ACTIVE_QUERIES = NQ + SUPPORT_GRID_SIZE * SUPPORT_GRID_SIZE
EXPECTED_MEMORY = 24
D_MODEL = 256
K64 = 64
UNIFIED_DENOM = 255.0
HORIZONS = (1, 4, 8)
PARITY_TOL = 1e-6
CANDIDATE_TOL = 1e-4

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


@dataclass
class TrackerState:
    q_init: torch.Tensor
    point_memory: torch.Tensor
    temporal_mask: torch.Tensor


@dataclass
class CandidateBundle:
    positions_model_xy: torch.Tensor
    correlation_indices: torch.Tensor
    post_fusion: torch.Tensor
    score: torch.Tensor
    certainty: torch.Tensor
    q2: torch.Tensor
    refined_model_xy: torch.Tensor
    c2_index: torch.Tensor
    v_logit: torch.Tensor
    u_logit: torch.Tensor


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


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a - b)).detach().cpu())


def tensor_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm((a - b).float()).detach().cpu())


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    left = a.reshape(-1).float()
    right = b.reshape(-1).float()
    value = F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0), dim=1)
    return float((1.0 - value[0]).detach().cpu())


def clone_state(state: TrackerState) -> TrackerState:
    return TrackerState(
        q_init=state.q_init.clone(),
        point_memory=state.point_memory.clone(),
        temporal_mask=state.temporal_mask.clone(),
    )


def state_storage(state: TrackerState) -> dict[str, int]:
    return {
        'q_init': int(state.q_init.untyped_storage().data_ptr()),
        'point_memory': int(state.point_memory.untyped_storage().data_ptr()),
        'temporal_mask': int(state.temporal_mask.untyped_storage().data_ptr()),
    }


def storage_is_disjoint(*states: TrackerState) -> bool:
    for field in ('q_init', 'point_memory', 'temporal_mask'):
        pointers = [state_storage(state)[field] for state in states]
        if len(set(pointers)) != len(pointers):
            return False
    return True


def load_frame(path: Path, device: torch.device) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(256, 256),
        mode='bilinear',
        align_corners=False,
    )
    return tensor.to(device)


def model_xy_to_yx_norm(
    xy: torch.Tensor | np.ndarray,
    model,
) -> np.ndarray:
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
def diagnostic_track_frame(
    model,
    state: TrackerState,
    frame_features: tuple[torch.Tensor, ...],
    h_in: int = 256,
    w_in: int = 256,
) -> dict[str, torch.Tensor]:
    q_init = state.q_init
    temporal_mask = state.temporal_mask
    point_memory = state.point_memory
    n, d_model = q_init.shape
    memory_size = point_memory.shape[1]
    f4, f8, f16, f32, fused = frame_features

    q_t = q_init.unsqueeze(0).clone()
    memory = point_memory.clone()
    mask = torch.zeros(
        n,
        memory_size + 1,
        device=q_init.device,
        dtype=torch.bool,
    )
    mask[:, :-1] = temporal_mask.clone()
    qkv = torch.zeros(
        n,
        memory_size + 1,
        d_model,
        device=q_init.device,
        dtype=memory.dtype,
    )
    qkv[:, :-1] = memory
    for layer in range(model.decoder_layer_num):
        q_t = model.feature_attention[layer](q_t, fused, fused)
        q_t = model.query_attention[layer](q_t, q_t, q_t)
        qkv[:, -1] = q_t.view(n, d_model)
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
    q_rerank, p_topk, certainty_topk, score_topk = model.reranking_head(
        q_pre,
        f4,
        f8,
        f16,
        f32,
        c1,
    )
    q_new = model.projection2(q_rerank)
    c2 = model.multiscale_correlation(q_new, f4, f8, f16, f32)
    p_patch = indices_to_coords(
        torch.argmax(c2, dim=-1).unsqueeze(1),
        model.input_size,
        model.stride,
    ).squeeze(1)
    offsets, v_logit, u_logit = model.prediction_head(
        q_new,
        f4,
        f8,
        f16,
        f32,
        p_patch,
    )
    p_model = p_patch[0] + offsets[-1]
    p_output = p_model.clone()
    p_output[..., 0] = (p_output[..., 0] / model.W) * w_in
    p_output[..., 1] = (p_output[..., 1] / model.H) * h_in
    return {
        'p': p_output,
        'p_model': p_model,
        'p_patch': p_patch.squeeze(0),
        'v_logit': v_logit,
        'u_logit': u_logit,
        'q_new': q_new.squeeze(0),
        'q_pre': q_pre.squeeze(0),
        'c1': c1.squeeze(0),
        'c2': c2.squeeze(0),
        'official_topk_positions': p_topk.squeeze(0),
        'official_topk_certainty': certainty_topk.squeeze(0),
        'official_topk_score': score_topk.squeeze(0),
    }


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
    top = torch.topk(
        c1.unsqueeze(0),
        k=k,
        dim=-1,
        largest=True,
        sorted=True,
    )
    positions = indices_to_coords(top.indices, head.size, head.stride)
    positions_norm = positions / torch.tensor(
        [head.size[1], head.size[0]],
        device=q.device,
        dtype=positions.dtype,
    )
    positions_norm = torch.clamp(positions_norm, 0, 1)
    positions_norm = positions_norm.view(1, n_queries * k, 1, 2)
    positions_norm = positions_norm.expand(-1, -1, head.num_level, -1)
    feature_scales = torch.cat([f4, f8, f16, f32], dim=1)
    decoded = q.unsqueeze(2).expand(-1, -1, k, -1).reshape(
        1,
        n_queries * k,
        d_model,
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
        [decoded, q.unsqueeze(2).expand(-1, -1, k, -1)],
        dim=-1,
    )
    post_fusion = head.fusion_layer(pre_fusion)
    score = head.score_layer(post_fusion).squeeze(-1).squeeze(0)
    certainty = head.certainty_layer(post_fusion).squeeze(-1).squeeze(0)
    return {
        'positions_model_xy': positions.squeeze(0),
        'correlation_indices': top.indices.squeeze(0),
        'post_fusion': post_fusion.squeeze(0),
        'score': score,
        'certainty': certainty,
    }


@torch.no_grad()
def candidate_conditioned_states(
    model,
    q_pre: torch.Tensor,
    post_fusion: torch.Tensor,
    frame_features: tuple[torch.Tensor, ...],
) -> CandidateBundle:
    f4, f8, f16, f32, _ = frame_features
    n_queries, k, d_model = post_fusion.shape
    q_base = q_pre[:, None, :].expand(n_queries, k, d_model).reshape(
        n_queries * k,
        1,
        d_model,
    )
    candidate = post_fusion.reshape(n_queries * k, 1, d_model)
    q_conditioned = model.reranking_head.fusion(
        q_base,
        candidate,
        candidate,
    )
    q_conditioned = model.reranking_head.final_projection_layer(
        torch.cat([q_conditioned, q_base], dim=-1)
    )
    q2_flat = model.projection2(q_conditioned.squeeze(1).unsqueeze(0))
    c2 = model.multiscale_correlation(q2_flat, f4, f8, f16, f32)
    c2_index = torch.argmax(c2, dim=-1)
    p_patch = indices_to_coords(
        c2_index.unsqueeze(1),
        model.input_size,
        model.stride,
    ).squeeze(1)
    offsets, v_logit, u_logit = model.prediction_head(
        q2_flat,
        f4,
        f8,
        f16,
        f32,
        p_patch,
    )
    refined = (p_patch[0] + offsets[-1]).view(n_queries, k, 2)
    q2 = q2_flat.squeeze(0).view(n_queries, k, d_model)
    return CandidateBundle(
        positions_model_xy=torch.empty(0, device=q_pre.device),
        correlation_indices=torch.empty(0, device=q_pre.device, dtype=torch.long),
        post_fusion=post_fusion,
        score=torch.empty(0, device=q_pre.device),
        certainty=torch.empty(0, device=q_pre.device),
        q2=q2,
        refined_model_xy=refined,
        c2_index=c2_index.view(n_queries, k),
        v_logit=v_logit.view(n_queries, k),
        u_logit=u_logit.view(n_queries, k),
    )


@torch.no_grad()
def build_candidate_bundle(
    model,
    q_pre: torch.Tensor,
    frame_features: tuple[torch.Tensor, ...],
    c1: torch.Tensor,
    k: int = K64,
) -> CandidateBundle:
    extended = extended_candidates(model, q_pre, frame_features, c1, k=k)
    conditioned = candidate_conditioned_states(
        model,
        q_pre,
        extended['post_fusion'],
        frame_features,
    )
    return CandidateBundle(
        positions_model_xy=extended['positions_model_xy'],
        correlation_indices=extended['correlation_indices'],
        post_fusion=extended['post_fusion'],
        score=extended['score'],
        certainty=extended['certainty'],
        q2=conditioned.q2,
        refined_model_xy=conditioned.refined_model_xy,
        c2_index=conditioned.c2_index,
        v_logit=conditioned.v_logit,
        u_logit=conditioned.u_logit,
    )


@torch.no_grad()
def update_state(
    model,
    state: TrackerState,
    q_features: torch.Tensor,
) -> TrackerState:
    write_mask = torch.ones(
        q_features.shape[0],
        dtype=torch.bool,
        device=q_features.device,
    )
    memory, mask = model._update_point_memory(
        state.point_memory,
        state.temporal_mask,
        q_features,
        write_mask,
    )
    return TrackerState(
        q_init=state.q_init,
        point_memory=memory,
        temporal_mask=mask,
    )


@torch.no_grad()
def split_event_states(
    model,
    pre_update_state: TrackerState,
    official_q_new: torch.Tensor,
    target_query: int,
    alternative_q2: torch.Tensor,
) -> tuple[TrackerState, TrackerState]:
    state_a = clone_state(pre_update_state)
    state_b = clone_state(pre_update_state)
    q_a = official_q_new.clone()
    q_b = official_q_new.clone()
    q_b[target_query] = alternative_q2
    state_a = update_state(model, state_a, q_a)
    state_b = update_state(model, state_b, q_b)
    return state_a, state_b


def compare_diag(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> dict[str, float]:
    keys = ('p', 'v_logit', 'u_logit', 'q_new', 'q_pre', 'c1', 'c2')
    return {key: max_abs(left[key], right[key]) for key in keys}


def candidate_set_metrics(
    model,
    indices_a: torch.Tensor,
    indices_b: torch.Tensor,
) -> dict[str, Any]:
    a = indices_a.detach().cpu().numpy().astype(np.int64)
    b = indices_b.detach().cpu().numpy().astype(np.int64)
    set_a = set(a.tolist())
    set_b = set(b.tolist())
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    coords_a = model_xy_to_yx_norm(
        indices_to_coords(
            indices_a.view(1, 1, -1),
            model.input_size,
            model.stride,
        ).view(-1, 2),
        model,
    )
    coords_b = model_xy_to_yx_norm(
        indices_to_coords(
            indices_b.view(1, 1, -1),
            model.input_size,
            model.stride,
        ).view(-1, 2),
        model,
    )
    distance = np.linalg.norm(
        (coords_a[:, None, :] - coords_b[None, :, :]) * UNIFIED_DENOM,
        axis=-1,
    )
    hausdorff = float(
        max(
            np.min(distance, axis=1).max(),
            np.min(distance, axis=0).max(),
        )
    )
    return {
        'k': int(len(a)),
        'exact_set_equal': bool(set_a == set_b),
        'overlap': int(intersection),
        'jaccard': float(intersection / union),
        'hausdorff_px': hausdorff,
    }


def vector_divergence(
    a: torch.Tensor,
    b: torch.Tensor,
) -> dict[str, float]:
    return {
        'max_abs': max_abs(a, b),
        'l2': tensor_l2(a, b),
        'cosine_distance': cosine_distance(a, b),
    }


def row_l2_summary(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    value = torch.linalg.vector_norm((a - b).float(), dim=-1)
    return {
        'mean': float(torch.mean(value).detach().cpu()),
        'max': float(torch.max(value).detach().cpu()),
    }


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def write_doc(result: dict[str, Any], path: Path) -> None:
    lines = [
        '# V9-A5.2 Independent Model-State Branching Smoke Result',
        '',
        'Date: 2026-07-11',
        '',
        'This is an integrity smoke only. It is not a scientific result and does not evaluate the 72-event gate.',
        '',
        '## Event',
        '',
        '```json',
        json.dumps(result['event'], indent=2),
        '```',
        '',
        '## Integrity gates',
        '',
        '```json',
        json.dumps(result['integrity']['gates'], indent=2),
        '```',
        '',
        '## Event split',
        '',
        '```json',
        json.dumps(result['split'], indent=2),
        '```',
        '',
        '## Horizons',
        '',
        '```json',
        json.dumps(result['horizons'], indent=2),
        '```',
        '',
        '## Decision',
        '',
        result['decision'],
        '',
        '## Provenance',
        '',
        '```json',
        json.dumps(result['provenance'], indent=2),
        '```',
    ]
    path.write_text('\n'.join(lines) + '\n')


def verify_environment(event: dict[str, Any]) -> dict[str, Any]:
    checked = []
    for path, expected in EXPECTED_HASHES.items():
        if not path.exists():
            raise RuntimeError(f'missing input: {path}')
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f'hash mismatch: {path} -> {actual}')
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
    if head != EXPECTED_HEAD:
        raise RuntimeError(f'HEAD mismatch: {head}')
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f'branch mismatch: {branch}')
    if tracked:
        raise RuntimeError(f'tracked worktree is dirty: {tracked}')

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
        if not value.startswith(str(TRACKON_ROOT.resolve()) + '/'):
            raise RuntimeError(f'non-clean TrackOn import: {name} -> {value}')

    frame_manifest = json.loads(FRAME_MANIFEST.read_text())
    by_relative = {
        row['relative_path']: row for row in frame_manifest['files']
    }
    last_frame = int(event['frame_tau']) + max(HORIZONS)
    frame_checks = []
    for frame in range(last_frame + 1):
        relative = f'ani/rgbs/rgb_{frame:05d}.jpg'
        row = by_relative.get(relative)
        if row is None:
            raise RuntimeError(f'frame absent from manifest: {relative}')
        path = Path(frame_manifest['root']) / relative
        actual = sha256(path)
        if path.stat().st_size != int(row['size_bytes']) or actual != row['sha256']:
            raise RuntimeError(f'frame hash mismatch: {path}')
        frame_checks.append(
            {
                'relative_path': relative,
                'size_bytes': int(row['size_bytes']),
                'sha256': actual,
            }
        )

    branch_functions = (
        diagnostic_track_frame,
        extended_candidates,
        candidate_conditioned_states,
        build_candidate_bundle,
        update_state,
        split_event_states,
    )
    forbidden_hits = {}
    for function in branch_functions:
        source = inspect.getsource(function).lower()
        hits = [
            token
            for token in ('gt_', 'ground_truth', 'error_px', 'oracle_index')
            if token in source
        ]
        forbidden_hits[function.__name__] = hits
    if any(forbidden_hits.values()):
        raise RuntimeError(f'GT/error token in branch generation: {forbidden_hits}')

    return {
        'head': head,
        'branch': branch,
        'tracked_status': tracked,
        'checked_inputs': checked,
        'checked_smoke_frames': frame_checks,
        'imported_trackon_code_paths': imported,
        'branch_function_forbidden_hits': forbidden_hits,
        'script': {
            'path': str(Path(__file__).resolve()),
            'sha256': sha256(Path(__file__).resolve()),
        },
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-json', type=Path, default=DEFAULT_JSON)
    parser.add_argument('--out-npz', type=Path, default=DEFAULT_NPZ)
    parser.add_argument('--out-doc', type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()

    start_time = time.perf_counter()
    manifest = json.loads(EVENT_MANIFEST.read_text())
    ani_events = [
        row for row in manifest['selected_events'] if row['clip_id'] == 'ani:0'
    ]
    if not ani_events:
        raise RuntimeError('no frozen ani:0 smoke event')
    event = sorted(
        ani_events,
        key=lambda row: (
            int(row['frame_tau']),
            int(row['query_idx']),
            row['selection_sha256'],
        ),
    )[0]
    if (
        event['clip_id'] != 'ani:0'
        or int(event['query_idx']) != 20
        or int(event['frame_tau']) != 2
    ):
        raise RuntimeError(f'unexpected frozen smoke event: {event}')

    environment = verify_environment(event)
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
    config = load_args_from_yaml(str(CONFIG))
    config.grad_checkpoint = False
    config.M_i = config.M
    predictor = Predictor(
        config,
        checkpoint_path=str(CHECKPOINT),
        support_grid_size=SUPPORT_GRID_SIZE,
    ).to(device).eval()
    model = predictor.model
    if int(model.K) != 16:
        raise RuntimeError(f'unexpected K: {model.K}')
    if int(model.M) != EXPECTED_MEMORY:
        raise RuntimeError(f'unexpected M: {model.M}')
    if int(model.D) != D_MODEL:
        raise RuntimeError(f'unexpected D: {model.D}')
    if str(model.memory_update_policy) != 'unconditional':
        raise RuntimeError(f'unexpected memory policy: {model.memory_update_policy}')

    annotation = np.load(
        POD_ROOT / 'ani/anno.npz',
        allow_pickle=True,
        mmap_mode='r',
    )
    trajectories = np.asarray(annotation['trajs_2d'][:96], dtype=np.float32)
    raw_visibility = np.asarray(annotation['visibs'][:96], dtype=bool)
    valid = (
        np.asarray(annotation['valids'][:96], dtype=bool)
        if 'valids' in annotation.files
        else np.ones_like(raw_visibility, dtype=bool)
    )
    finite = np.isfinite(trajectories).all(axis=2)
    visible_all = raw_visibility & valid & finite
    eligible = np.where(
        visible_all[0] & (np.sum(visible_all, axis=0) >= 24)
    )[0]
    chosen = np.sort(
        np.random.default_rng(stable_seed('ani', 0)).choice(
            eligible,
            NQ,
            replace=False,
        )
    )
    gt_xy = trajectories[:, chosen]
    visible = visible_all[:, chosen]
    frame_paths = [
        POD_ROOT / 'ani/rgbs' / f'rgb_{frame:05d}.jpg'
        for frame in range(int(event['frame_tau']) + max(HORIZONS) + 1)
    ]
    width, height = Image.open(frame_paths[0]).size
    query_xy = np.empty((NQ, 2), dtype=np.float32)
    query_xy[:, 0] = gt_xy[0, :, 0] * UNIFIED_DENOM / float(width - 1)
    query_xy[:, 1] = gt_xy[0, :, 1] * UNIFIED_DENOM / float(height - 1)
    queries = torch.from_numpy(query_xy).float().to(device)
    support = get_points_on_a_grid(
        SUPPORT_GRID_SIZE,
        (256, 256),
        device,
    ).squeeze(0)
    combined = torch.cat([queries, support], dim=0)

    predictor.reset()
    predictor.initial_capacity = len(combined)
    first_image = load_frame(frame_paths[0], device)
    first_features = model.extract_frame_features(first_image)
    predictor.init_queries((first_features[-1], device), combined, 256, 256)
    if predictor.N != EXPECTED_ACTIVE_QUERIES:
        raise RuntimeError(f'unexpected active queries: {predictor.N}')
    official_state = TrackerState(
        q_init=predictor.q_init[: predictor.N].clone(),
        point_memory=predictor.point_memory[: predictor.N].clone(),
        temporal_mask=predictor.temporal_mask[: predictor.N].clone(),
    )

    target = int(event['query_idx'])
    event_frame = int(event['frame_tau'])
    branch_a: TrackerState | None = None
    branch_b: TrackerState | None = None
    split_report: dict[str, Any] = {}
    horizon_report: dict[str, Any] = {}
    online_risk_mismatch = 0
    official_track_parity = {
        key: 0.0
        for key in ('p', 'v_logit', 'q_new')
    }
    branch_a_parity = {
        key: 0.0
        for key in ('p', 'v_logit', 'u_logit', 'q_new', 'q_pre', 'c1', 'c2')
    }
    branch_a_state_parity = {
        'q_init': 0.0,
        'point_memory': 0.0,
        'temporal_mask_mismatch': 0,
    }
    order_parity = {
        'branch_a': {
            key: 0.0
            for key in ('p', 'v_logit', 'u_logit', 'q_new', 'q_pre', 'c1', 'c2')
        },
        'branch_b': {
            key: 0.0
            for key in ('p', 'v_logit', 'u_logit', 'q_new', 'q_pre', 'c1', 'c2')
        },
    }
    candidate_replay = {
        'score_max_abs': 0.0,
        'raw_error_max_abs': 0.0,
        'refined_error_max_abs': 0.0,
        'official_error_max_abs': 0.0,
        'risk_mismatch': 0,
        'shared_score_top1_error_max_abs': 0.0,
    }
    saved_arrays: dict[str, list[Any]] = {
        'horizon': [],
        'official_error': [],
        'shared_score_top1_error': [],
        'shared_b2_oracle': [],
        'branch_a_error': [],
        'branch_b_error': [],
        'independent_b2_oracle': [],
        'target_q_new_l2': [],
        'target_c1_l2': [],
        'top16_overlap': [],
        'top64_overlap': [],
    }

    for frame in range(event_frame + max(HORIZONS) + 1):
        if frame == 0:
            frame_features = first_features
        else:
            image = load_frame(frame_paths[frame], device)
            frame_features = model.extract_frame_features(image)

        official_diag = diagnostic_track_frame(
            model,
            official_state,
            frame_features,
        )
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
            max_abs(official_diag['p'], p_ref),
        )
        official_track_parity['v_logit'] = max(
            official_track_parity['v_logit'],
            max_abs(official_diag['v_logit'], v_ref),
        )
        official_track_parity['q_new'] = max(
            official_track_parity['q_new'],
            max_abs(official_diag['q_new'], q_ref),
        )

        online_risk = (
            torch.sigmoid(official_diag['v_logit'][:NQ]) < float(model.delta_v)
        ) | (
            torch.sigmoid(official_diag['u_logit'][:NQ]) >= 0.5
        )
        frozen_risk = np.asarray(
            [
                bool(v9c['risk'][v9c_key[('ani:0', frame, query)]])
                for query in range(NQ)
            ],
            dtype=bool,
        )
        online_risk_mismatch += int(
            np.sum(online_risk.detach().cpu().numpy() != frozen_risk)
        )

        target_bundle: CandidateBundle | None = None
        if frame == event_frame or frame - event_frame in HORIZONS:
            target_bundle = build_candidate_bundle(
                model,
                official_diag['q_pre'][target : target + 1],
                frame_features,
                official_diag['c1'][target : target + 1],
                k=K64,
            )
            frozen_row = v9b_key[('ani:0', frame, target)]
            gt_yx = np.asarray(
                [
                    gt_xy[frame, target, 1] / float(height - 1),
                    gt_xy[frame, target, 0] / float(width - 1),
                ],
                dtype=np.float32,
            )
            raw_positions = model_xy_to_yx_norm(
                target_bundle.positions_model_xy[0],
                model,
            )
            refined_positions = model_xy_to_yx_norm(
                target_bundle.refined_model_xy[0],
                model,
            )
            raw_error = np.linalg.norm(
                (raw_positions - gt_yx[None, :]) * UNIFIED_DENOM,
                axis=1,
            ).astype(np.float32)
            refined_error = np.linalg.norm(
                (refined_positions - gt_yx[None, :]) * UNIFIED_DENOM,
                axis=1,
            ).astype(np.float32)
            official_yx = output_xy_to_yx_norm(official_diag['p'][target])
            official_error = float(
                np.linalg.norm((official_yx - gt_yx) * UNIFIED_DENOM)
            )
            score_numpy = target_bundle.score[0].detach().cpu().float().numpy()
            score_index = int(np.argmax(score_numpy))
            shared_score_top1_error = float(refined_error[score_index])
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
                abs(official_error - float(v9b['official_final'][frozen_row])),
            )
            candidate_replay['risk_mismatch'] += int(
                bool(online_risk[target])
                != bool(v9b['risk'][frozen_row])
            )
            candidate_replay['shared_score_top1_error_max_abs'] = max(
                candidate_replay['shared_score_top1_error_max_abs'],
                abs(
                    shared_score_top1_error
                    - float(v9b['refined_score_top1_K64'][frozen_row])
                ),
            )

        if frame == event_frame:
            if not bool(online_risk[target]):
                raise RuntimeError('frozen smoke event is not online-risk')
            if any(
                bool(v9c['risk'][v9c_key[('ani:0', prior, target)]])
                for prior in range(1, event_frame)
            ):
                raise RuntimeError('smoke event is not the first risk frame')
            if target_bundle is None:
                raise RuntimeError('missing event candidate bundle')
            score_index = int(torch.argmax(target_bundle.score[0]).item())
            alternative_q2 = target_bundle.q2[0, score_index]
            branch_a, branch_b = split_event_states(
                model,
                official_state,
                official_diag['q_new'],
                target,
                alternative_q2,
            )
            official_after = update_state(
                model,
                official_state,
                official_diag['q_new'],
            )
            storage = {
                'official': state_storage(official_after),
                'branch_a': state_storage(branch_a),
                'branch_b': state_storage(branch_b),
            }
            disjoint = storage_is_disjoint(official_after, branch_a, branch_b)
            a_before = float(branch_a.point_memory[target, -1, 0].detach().cpu())
            b_before = float(branch_b.point_memory[target, -1, 0].detach().cpu())
            branch_b.point_memory[target, -1, 0] += 1.0
            a_after_mutation = float(
                branch_a.point_memory[target, -1, 0].detach().cpu()
            )
            branch_b.point_memory[target, -1, 0] = b_before
            b_restored = float(
                branch_b.point_memory[target, -1, 0].detach().cpu()
            )
            split_report = {
                'candidate_score_index': score_index,
                'candidate_grid_index': int(
                    target_bundle.correlation_indices[0, score_index].detach().cpu()
                ),
                'candidate_score': float(
                    target_bundle.score[0, score_index].detach().cpu()
                ),
                'alternative_q2_vs_official_q_new': vector_divergence(
                    alternative_q2,
                    official_diag['q_new'][target],
                ),
                'storage_pointers': storage,
                'storage_disjoint': disjoint,
                'mutation_isolation': {
                    'branch_a_before': a_before,
                    'branch_a_after_branch_b_mutation': a_after_mutation,
                    'branch_a_unchanged': a_before == a_after_mutation,
                    'branch_b_before': b_before,
                    'branch_b_restored': b_restored,
                    'branch_b_restore_exact': b_before == b_restored,
                },
                'post_write_target_memory_divergence': vector_divergence(
                    branch_a.point_memory[target],
                    branch_b.point_memory[target],
                ),
                'post_write_full_memory_max_abs': max_abs(
                    branch_a.point_memory,
                    branch_b.point_memory,
                ),
                'branch_a_official_post_write': {
                    'q_init_max_abs': max_abs(
                        branch_a.q_init,
                        official_after.q_init,
                    ),
                    'memory_max_abs': max_abs(
                        branch_a.point_memory,
                        official_after.point_memory,
                    ),
                    'mask_mismatch': int(
                        torch.sum(
                            branch_a.temporal_mask
                            != official_after.temporal_mask
                        ).detach().cpu()
                    ),
                },
            }
            official_state = official_after
            continue

        if frame < event_frame:
            official_state = update_state(
                model,
                official_state,
                official_diag['q_new'],
            )
            continue

        if branch_a is None or branch_b is None:
            raise RuntimeError('branches missing after split')

        branch_a_pre = clone_state(branch_a)
        branch_b_pre = clone_state(branch_b)
        diag_a_first = diagnostic_track_frame(
            model,
            branch_a_pre,
            frame_features,
        )
        diag_b_second = diagnostic_track_frame(
            model,
            branch_b_pre,
            frame_features,
        )
        diag_b_first = diagnostic_track_frame(
            model,
            branch_b_pre,
            frame_features,
        )
        diag_a_second = diagnostic_track_frame(
            model,
            branch_a_pre,
            frame_features,
        )
        for key, value in compare_diag(diag_a_first, diag_a_second).items():
            order_parity['branch_a'][key] = max(
                order_parity['branch_a'][key],
                value,
            )
        for key, value in compare_diag(diag_b_first, diag_b_second).items():
            order_parity['branch_b'][key] = max(
                order_parity['branch_b'][key],
                value,
            )
        for key, value in compare_diag(diag_a_first, official_diag).items():
            branch_a_parity[key] = max(branch_a_parity[key], value)
        branch_a_state_parity['q_init'] = max(
            branch_a_state_parity['q_init'],
            max_abs(branch_a.q_init, official_state.q_init),
        )
        branch_a_state_parity['point_memory'] = max(
            branch_a_state_parity['point_memory'],
            max_abs(branch_a.point_memory, official_state.point_memory),
        )
        branch_a_state_parity['temporal_mask_mismatch'] += int(
            torch.sum(
                branch_a.temporal_mask != official_state.temporal_mask
            ).detach().cpu()
        )

        horizon = frame - event_frame
        if horizon in HORIZONS:
            if target_bundle is None:
                raise RuntimeError('missing shared-state horizon candidate bundle')
            bundle_b = build_candidate_bundle(
                model,
                diag_b_first['q_pre'][target : target + 1],
                frame_features,
                diag_b_first['c1'][target : target + 1],
                k=K64,
            )
            gt_yx = np.asarray(
                [
                    gt_xy[frame, target, 1] / float(height - 1),
                    gt_xy[frame, target, 0] / float(width - 1),
                ],
                dtype=np.float32,
            )
            official_yx = output_xy_to_yx_norm(official_diag['p'][target])
            branch_a_yx = output_xy_to_yx_norm(diag_a_first['p'][target])
            branch_b_yx = output_xy_to_yx_norm(diag_b_first['p'][target])
            shared_refined = model_xy_to_yx_norm(
                target_bundle.refined_model_xy[0],
                model,
            )
            shared_score_index = int(
                torch.argmax(target_bundle.score[0]).item()
            )
            shared_score_yx = shared_refined[shared_score_index]
            official_error = float(
                np.linalg.norm((official_yx - gt_yx) * UNIFIED_DENOM)
            )
            shared_score_error = float(
                np.linalg.norm((shared_score_yx - gt_yx) * UNIFIED_DENOM)
            )
            branch_a_error = float(
                np.linalg.norm((branch_a_yx - gt_yx) * UNIFIED_DENOM)
            )
            branch_b_error = float(
                np.linalg.norm((branch_b_yx - gt_yx) * UNIFIED_DENOM)
            )
            shared_oracle = min(official_error, shared_score_error)
            independent_oracle = min(branch_a_error, branch_b_error)

            top16_a = torch.topk(
                diag_a_first['c1'][target],
                k=16,
                largest=True,
                sorted=True,
            ).indices
            top16_b = torch.topk(
                diag_b_first['c1'][target],
                k=16,
                largest=True,
                sorted=True,
            ).indices
            top64_a = torch.topk(
                diag_a_first['c1'][target],
                k=64,
                largest=True,
                sorted=True,
            ).indices
            top64_b = torch.topk(
                diag_b_first['c1'][target],
                k=64,
                largest=True,
                sorted=True,
            ).indices
            set16 = candidate_set_metrics(model, top16_a, top16_b)
            set64 = candidate_set_metrics(model, top64_a, top64_b)

            branch_b_refined = model_xy_to_yx_norm(
                bundle_b.refined_model_xy[0],
                model,
            )
            branch_b_refined_error = np.linalg.norm(
                (branch_b_refined - gt_yx[None, :]) * UNIFIED_DENOM,
                axis=1,
            )
            useful_novel = branch_b_error < shared_oracle - 1.0
            candidate_novel_k16 = bool(
                np.min(branch_b_refined_error[:16]) <= 4.0
                and shared_oracle > 4.0
            )
            candidate_novel_k64 = bool(
                np.min(branch_b_refined_error) <= 4.0
                and shared_oracle > 4.0
            )

            full_q = row_l2_summary(
                diag_a_first['q_new'],
                diag_b_first['q_new'],
            )
            non_target_eval = torch.ones(
                NQ,
                dtype=torch.bool,
                device=device,
            )
            non_target_eval[target] = False
            non_target_q = row_l2_summary(
                diag_a_first['q_new'][:NQ][non_target_eval],
                diag_b_first['q_new'][:NQ][non_target_eval],
            )
            support_q = row_l2_summary(
                diag_a_first['q_new'][NQ:],
                diag_b_first['q_new'][NQ:],
            )
            horizon_report[str(horizon)] = {
                'frame_tau': frame,
                'gt_visible_valid': bool(visible[frame, target]),
                'errors_px': {
                    'official': official_error,
                    'shared_score_top1': shared_score_error,
                    'shared_b2_oracle': shared_oracle,
                    'branch_a': branch_a_error,
                    'branch_b': branch_b_error,
                    'independent_b2_oracle': independent_oracle,
                    'independent_minus_shared': independent_oracle - shared_oracle,
                },
                'useful_novel': bool(useful_novel),
                'branch_b_final_le4_while_shared_gt4': bool(
                    branch_b_error <= 4.0 and shared_oracle > 4.0
                ),
                'branch_b_candidate_novel_k16': candidate_novel_k16,
                'branch_b_candidate_novel_k64': candidate_novel_k64,
                'target_state_divergence': {
                    'q_new': vector_divergence(
                        diag_a_first['q_new'][target],
                        diag_b_first['q_new'][target],
                    ),
                    'q_pre': vector_divergence(
                        diag_a_first['q_pre'][target],
                        diag_b_first['q_pre'][target],
                    ),
                    'c1': vector_divergence(
                        diag_a_first['c1'][target],
                        diag_b_first['c1'][target],
                    ),
                    'c2': vector_divergence(
                        diag_a_first['c2'][target],
                        diag_b_first['c2'][target],
                    ),
                    'final_coordinate_l2_px': float(
                        np.linalg.norm(
                            (branch_a_yx - branch_b_yx) * UNIFIED_DENOM
                        )
                    ),
                    'v_logit_abs': float(
                        torch.abs(
                            diag_a_first['v_logit'][target]
                            - diag_b_first['v_logit'][target]
                        ).detach().cpu()
                    ),
                    'u_logit_abs': float(
                        torch.abs(
                            diag_a_first['u_logit'][target]
                            - diag_b_first['u_logit'][target]
                        ).detach().cpu()
                    ),
                },
                'full_q_new_row_l2': full_q,
                'non_target_evaluated_q_new_row_l2': non_target_q,
                'support_q_new_row_l2': support_q,
                'pre_update_memory': {
                    'target': vector_divergence(
                        branch_a.point_memory[target],
                        branch_b.point_memory[target],
                    ),
                    'full_max_abs': max_abs(
                        branch_a.point_memory,
                        branch_b.point_memory,
                    ),
                },
                'candidate_sets': {
                    'top16': set16,
                    'top64': set64,
                },
                'branch_b_refined_oracle_px': {
                    'k16': float(np.min(branch_b_refined_error[:16])),
                    'k64': float(np.min(branch_b_refined_error)),
                },
            }
            saved_arrays['horizon'].append(horizon)
            saved_arrays['official_error'].append(official_error)
            saved_arrays['shared_score_top1_error'].append(shared_score_error)
            saved_arrays['shared_b2_oracle'].append(shared_oracle)
            saved_arrays['branch_a_error'].append(branch_a_error)
            saved_arrays['branch_b_error'].append(branch_b_error)
            saved_arrays['independent_b2_oracle'].append(independent_oracle)
            saved_arrays['target_q_new_l2'].append(
                horizon_report[str(horizon)]['target_state_divergence']['q_new']['l2']
            )
            saved_arrays['target_c1_l2'].append(
                horizon_report[str(horizon)]['target_state_divergence']['c1']['l2']
            )
            saved_arrays['top16_overlap'].append(set16['overlap'])
            saved_arrays['top64_overlap'].append(set64['overlap'])

        official_next = update_state(
            model,
            official_state,
            official_diag['q_new'],
        )
        branch_a_next = update_state(
            model,
            branch_a,
            diag_a_first['q_new'],
        )
        branch_b_next = update_state(
            model,
            branch_b,
            diag_b_first['q_new'],
        )
        branch_a_state_parity['point_memory'] = max(
            branch_a_state_parity['point_memory'],
            max_abs(branch_a_next.point_memory, official_next.point_memory),
        )
        branch_a_state_parity['temporal_mask_mismatch'] += int(
            torch.sum(
                branch_a_next.temporal_mask
                != official_next.temporal_mask
            ).detach().cpu()
        )
        official_state = official_next
        branch_a = branch_a_next
        branch_b = branch_b_next

    if set(horizon_report) != {'1', '4', '8'}:
        raise RuntimeError(f'missing smoke horizons: {horizon_report.keys()}')

    parity_max = max(
        *official_track_parity.values(),
        *branch_a_parity.values(),
        branch_a_state_parity['q_init'],
        branch_a_state_parity['point_memory'],
    )
    order_max = max(
        *order_parity['branch_a'].values(),
        *order_parity['branch_b'].values(),
    )
    candidate_max = max(
        candidate_replay['score_max_abs'],
        candidate_replay['raw_error_max_abs'],
        candidate_replay['refined_error_max_abs'],
        candidate_replay['official_error_max_abs'],
        candidate_replay['shared_score_top1_error_max_abs'],
    )
    all_horizon_formulas = all(
        abs(
            row['errors_px']['shared_b2_oracle']
            - min(
                row['errors_px']['official'],
                row['errors_px']['shared_score_top1'],
            )
        ) <= 1e-9
        and abs(
            row['errors_px']['independent_b2_oracle']
            - min(
                row['errors_px']['branch_a'],
                row['errors_px']['branch_b'],
            )
        ) <= 1e-9
        for row in horizon_report.values()
    )
    gates = {
        'official_diagnostic_parity_le_1e6': max(
            official_track_parity.values()
        ) <= PARITY_TOL,
        'branch_a_full_output_parity_le_1e6': max(
            branch_a_parity.values()
        ) <= PARITY_TOL,
        'branch_a_state_parity': (
            branch_a_state_parity['q_init'] <= PARITY_TOL
            and branch_a_state_parity['point_memory'] <= PARITY_TOL
            and branch_a_state_parity['temporal_mask_mismatch'] == 0
        ),
        'storage_disjoint': bool(split_report['storage_disjoint']),
        'mutation_isolation': bool(
            split_report['mutation_isolation']['branch_a_unchanged']
            and split_report['mutation_isolation']['branch_b_restore_exact']
        ),
        'call_order_invariance_le_1e6': order_max <= PARITY_TOL,
        'online_risk_exact': online_risk_mismatch == 0,
        'event_candidate_replay': (
            candidate_replay['score_max_abs'] <= 6e-5
            and candidate_replay['raw_error_max_abs'] <= CANDIDATE_TOL
            and candidate_replay['refined_error_max_abs'] <= CANDIDATE_TOL
            and candidate_replay['official_error_max_abs'] <= PARITY_TOL
            and candidate_replay['shared_score_top1_error_max_abs']
            <= CANDIDATE_TOL
            and candidate_replay['risk_mismatch'] == 0
        ),
        'shared_and_independent_b2_formulas_exact': all_horizon_formulas,
        'all_outputs_finite': finite_tree(
            {
                'split': split_report,
                'horizons': horizon_report,
                'official_track_parity': official_track_parity,
                'branch_a_parity': branch_a_parity,
                'order_parity': order_parity,
                'candidate_replay': candidate_replay,
            }
        ),
        'branch_generation_has_no_gt_error_tokens': not any(
            environment['branch_function_forbidden_hits'].values()
        ),
        'expected_state_dimensions': (
            predictor.N == EXPECTED_ACTIVE_QUERIES
            and model.M == EXPECTED_MEMORY
            and model.D == D_MODEL
        ),
    }
    pass_all = bool(all(gates.values()))
    decision = (
        'SMOKE_PASS: full 432-query independent-state splitting, official Branch A '
        'parity, storage isolation, call-order invariance, candidate replay and B2 '
        'readout formulas all pass. The frozen 72-event audit may be implemented; '
        'this smoke is not a scientific result.'
        if pass_all
        else 'SMOKE_FAIL: at least one independent-state integrity gate failed. '
        'Do not implement or run the 72-event audit until corrected.'
    )

    result = {
        'date': '2026-07-11',
        'type': 'V9-A5.2 one-event integrity smoke',
        'event': {
            **event,
            'smoke_selection': (
                'earliest frozen ani:0 event by frame_tau, query_idx, selection hash'
            ),
            'horizons': list(HORIZONS),
            'active_queries': int(predictor.N),
            'evaluated_queries': NQ,
            'support_queries': SUPPORT_GRID_SIZE * SUPPORT_GRID_SIZE,
            'memory_size': int(model.M),
            'feature_dim': int(model.D),
        },
        'split': split_report,
        'horizons': horizon_report,
        'integrity': {
            'gates': gates,
            'pass_all': pass_all,
            'official_track_parity': official_track_parity,
            'branch_a_output_parity': branch_a_parity,
            'branch_a_state_parity': branch_a_state_parity,
            'call_order_parity': order_parity,
            'online_risk_mismatch': int(online_risk_mismatch),
            'candidate_replay': candidate_replay,
            'parity_max_abs': float(parity_max),
            'call_order_max_abs': float(order_max),
            'candidate_replay_max_abs': float(candidate_max),
        },
        'decision': decision,
        'provenance': {
            'branch': environment['branch'],
            'head': environment['head'],
            'worktree_root': str(WORKTREE_ROOT),
            'source_asset_root': str(SOURCE_ROOT),
            'device': str(device),
            'seconds': float(time.perf_counter() - start_time),
            'environment': environment,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_doc.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(
        args.out_npz,
        **{
            key: np.asarray(value)
            for key, value in saved_arrays.items()
        },
    )
    write_doc(result, args.out_doc)
    print(
        json.dumps(
            {
                'ok': True,
                'json': str(args.out_json),
                'npz': str(args.out_npz),
                'doc': str(args.out_doc),
                'decision': decision,
                'gate_pass': pass_all,
                'event': result['event'],
                'parity_max_abs': parity_max,
                'call_order_max_abs': order_max,
                'candidate_replay_max_abs': candidate_max,
                'horizon_summary': {
                    horizon: {
                        'independent_minus_shared': row['errors_px'][
                            'independent_minus_shared'
                        ],
                        'target_q_new_l2': row['target_state_divergence'][
                            'q_new'
                        ]['l2'],
                        'top16_overlap': row['candidate_sets']['top16'][
                            'overlap'
                        ],
                    }
                    for horizon, row in horizon_report.items()
                },
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
