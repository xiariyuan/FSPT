#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
from PIL import Image

WORKTREE_ROOT = Path('/gemini/code/FSPT_v9a51c_clean')
SOURCE_ROOT = Path('/gemini/code/FSPT')
TRACKON_ROOT = WORKTREE_ROOT / 'baselines/track_on'
POD_ROOT = SOURCE_ROOT / 'datasets/pointodyssey/train'
CHECKPOINT = SOURCE_ROOT / 'baselines/track_on/checkpoints_trackon2_dinov3.pt'
CONFIG = TRACKON_ROOT / 'config/test.yaml'
MANIFEST = WORKTREE_ROOT / 'docs/v9a51c_input_manifest_2026-07-10.json'
V9A51B_SCRIPT = WORKTREE_ROOT / 'scripts/v9a51b_candidate_conditioned_refinement_audit.py'
V9A51B_JSON = WORKTREE_ROOT / 'outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_candidate_conditioned_refinement.json'
V9A51B_NPZ = WORKTREE_ROOT / 'outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_candidate_conditioned_refinement_rows.npz'
OUTDIR = WORKTREE_ROOT / 'outputs/paper_discovery_2026-07-05/v9a51c_history_beam'
DEFAULT_JSON = OUTDIR / 'v9a51c_history_preserving_beam.json'
DEFAULT_NPZ = OUTDIR / 'v9a51c_history_preserving_beam_rows.npz'
DEFAULT_DOC = WORKTREE_ROOT / 'docs/v9a51c_history_preserving_beam_result_2026-07-10.md'
EXPECTED_HEAD = '29512f0d95d8b4069b73b9130e60b75d1a892b98'
EXPECTED_BRANCH = 'v9a51c-history-preserving-beam-20260710'

SEQUENCES = ('ani', 'animal3', 'r4_new_f')
CLIP_STARTS = (0, 256, 512)
LENGTH = 96
NQ = 32
K16 = 16
K64 = 64
UNIFIED_DENOM = 255.0
COST_WINDOW = 8
TOL = 1e-6
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 20260714
PRIMARY_POLICY = 'native_dynamic_B4'
POLICIES = {
    'native_dynamic_B1': {'mode': 'dynamic', 'width': 1},
    'native_dynamic_B4': {'mode': 'dynamic', 'width': 4},
    'native_dynamic_B8': {'mode': 'dynamic', 'width': 8},
    'fixed16_B4': {'mode': 'fixed16', 'width': 4},
    'fixed64_B4': {'mode': 'fixed64', 'width': 4},
}

if str(TRACKON_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKON_ROOT))
os.environ['DINOV3_LOCAL_DIR'] = str(
    SOURCE_ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m'
)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from model.trackon_predictor import Predictor  # noqa: E402
from utils.coord_utils import get_points_on_a_grid  # noqa: E402
from utils.train_utils import load_args_from_yaml  # noqa: E402

# Reuse the exact committed V9-A5.1b candidate/refinement functions. TrackOn
# modules have already been imported from the current clean worktree, so the
# frozen module resolves those existing modules rather than importing another
# worktree's TrackOn implementation.
_SPEC = importlib.util.spec_from_file_location('v9a51b_frozen', V9A51B_SCRIPT)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError('unable to load frozen V9-A5.1b script')
V9B = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(V9B)
# Restore current TrackOn path precedence after the frozen module import.
sys.path = [entry for entry in sys.path if entry != '/gemini/code/FSPT_v9a51b_clean/baselines/track_on']
if str(TRACKON_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKON_ROOT))
os.environ['DINOV3_LOCAL_DIR'] = str(
    SOURCE_ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m'
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


def verify_inputs(verify_frames: bool) -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text())
    checked = []
    for row in manifest['files']:
        path = Path(row['path'])
        if not path.exists():
            raise RuntimeError(f'missing input: {path}')
        size = path.stat().st_size
        digest = sha256(path)
        if size != int(row['size_bytes']) or digest != row['sha256']:
            raise RuntimeError(f'input hash/size mismatch: {path}')
        checked.append({'path': str(path), 'size_bytes': int(size), 'sha256': digest})

    frame_report = None
    meta = manifest['frame_manifest']
    if verify_frames:
        frame_manifest = json.loads(Path(meta['path']).read_text())
        root = Path(frame_manifest['root'])
        aggregate = hashlib.sha256()
        total = 0
        for row in frame_manifest['files']:
            relative = str(row['relative_path'])
            path = root / relative
            size = path.stat().st_size
            digest = sha256(path)
            if size != int(row['size_bytes']) or digest != row['sha256']:
                raise RuntimeError(f'frame mismatch: {path}')
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
            raise RuntimeError('frame bytes mismatch')
        if aggregate_digest != meta['aggregate_sha256']:
            raise RuntimeError('frame aggregate mismatch')
        frame_report = {
            'path': str(meta['path']),
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

    replay = manifest['v9a51b_replay']
    if sha256(V9A51B_SCRIPT) != replay['script_sha256']:
        raise RuntimeError('V9-A5.1b script hash mismatch')
    if sha256(V9A51B_JSON) != replay['json_sha256']:
        raise RuntimeError('V9-A5.1b JSON hash mismatch')
    if sha256(V9A51B_NPZ) != replay['npz_sha256']:
        raise RuntimeError('V9-A5.1b NPZ hash mismatch')

    return {
        'pass': True,
        'head': head,
        'branch': branch,
        'tracked_status': tracked,
        'checked_inputs': checked,
        'frame_manifest': frame_report,
        'imported_trackon_code_paths': imported,
        'v9a51b_replay': replay,
        'script': {'path': str(Path(__file__).resolve()), 'sha256': sha256(Path(__file__).resolve())},
    }


def ordinal_rank(values: np.ndarray, descending: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-values if descending else values, kind='mergesort')
    rank = np.empty(len(values), dtype=np.int64)
    rank[order] = np.arange(len(values), dtype=np.int64)
    return rank


@dataclass
class BeamState:
    raw_coord: np.ndarray
    previous_raw_coord: np.ndarray | None
    refined_coord: np.ndarray
    descriptor: np.ndarray
    grid_history: tuple[int, ...]
    cost_history: tuple[float, ...]
    window_cost: float
    step_cost: float
    emission_rank: int


def initialize_state(
    query_coord: np.ndarray,
    raw_coords: np.ndarray,
    descriptors: np.ndarray,
    grid_indices: np.ndarray,
) -> BeamState:
    distance = np.linalg.norm(
        (raw_coords - query_coord[None, :]) * UNIFIED_DENOM, axis=1
    )
    index = int(np.argmin(distance))
    return BeamState(
        raw_coord=np.asarray(query_coord, dtype=np.float32).copy(),
        previous_raw_coord=None,
        refined_coord=np.asarray(query_coord, dtype=np.float32).copy(),
        descriptor=np.asarray(descriptors[index], dtype=np.float32).copy(),
        grid_history=(int(grid_indices[index]),),
        cost_history=(),
        window_cost=0.0,
        step_cost=0.0,
        emission_rank=0,
    )


def update_beam(
    parents: list[BeamState],
    raw_coords: np.ndarray,
    refined_coords: np.ndarray,
    descriptors: np.ndarray,
    scores: np.ndarray,
    grid_indices: np.ndarray,
    candidate_budget: int,
    beam_width: int,
) -> list[BeamState]:
    raw = np.asarray(raw_coords[:candidate_budget], dtype=np.float32)
    refined = np.asarray(refined_coords[:candidate_budget], dtype=np.float32)
    desc = np.asarray(descriptors[:candidate_budget], dtype=np.float32)
    score = np.asarray(scores[:candidate_budget], dtype=np.float64)
    grid = np.asarray(grid_indices[:candidate_budget], dtype=np.int64)
    denominator = float(max(1, candidate_budget - 1))
    emission_rank = ordinal_rank(score, descending=True)
    emission = emission_rank.astype(np.float64) / denominator
    transitions: list[tuple[tuple[float, float, int, int, int], BeamState]] = []
    for parent_slot, parent in enumerate(parents):
        prediction = (
            parent.raw_coord
            if parent.previous_raw_coord is None
            else parent.raw_coord + (parent.raw_coord - parent.previous_raw_coord)
        )
        motion_distance = np.linalg.norm(
            (raw - prediction[None, :]) * UNIFIED_DENOM, axis=1
        )
        motion = ordinal_rank(motion_distance, descending=False).astype(np.float64) / denominator
        similarity = desc @ parent.descriptor
        identity = ordinal_rank(similarity, descending=True).astype(np.float64) / denominator
        step = emission + motion + identity
        for candidate_index in range(candidate_budget):
            step_cost = float(step[candidate_index])
            cost_history = (parent.cost_history + (step_cost,))[-COST_WINDOW:]
            grid_history = (
                parent.grid_history + (int(grid[candidate_index]),)
            )[-3:]
            state = BeamState(
                raw_coord=raw[candidate_index].copy(),
                previous_raw_coord=parent.raw_coord.copy(),
                refined_coord=refined[candidate_index].copy(),
                descriptor=desc[candidate_index].copy(),
                grid_history=grid_history,
                cost_history=cost_history,
                window_cost=float(sum(cost_history)),
                step_cost=step_cost,
                emission_rank=int(emission_rank[candidate_index]),
            )
            key = (
                state.window_cost,
                state.step_cost,
                state.emission_rank,
                int(grid[candidate_index]),
                int(parent_slot),
            )
            transitions.append((key, state))

    best_by_signature: dict[tuple[int, ...], tuple[tuple[float, float, int, int, int], BeamState]] = {}
    for key, state in transitions:
        previous = best_by_signature.get(state.grid_history)
        if previous is None or key < previous[0]:
            best_by_signature[state.grid_history] = (key, state)
    ordered = sorted(best_by_signature.values(), key=lambda item: item[0])
    result = [state for _, state in ordered[:beam_width]]
    if len(result) != beam_width:
        raise RuntimeError(f'beam underflow: {len(result)} != {beam_width}')
    return result


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
        'v9a51b_hybrid_score_top1',
        'v9a51b_full_candidate_oracle',
        f'beam_top1_{PRIMARY_POLICY}',
        f'beam_oracle_{PRIMARY_POLICY}',
        f'beam_raw_min_{PRIMARY_POLICY}',
        'beam_top1_native_dynamic_B1',
        'beam_oracle_native_dynamic_B1',
        'beam_top1_native_dynamic_B8',
        'beam_oracle_native_dynamic_B8',
        'beam_top1_fixed16_B4',
        'beam_oracle_fixed16_B4',
        'beam_top1_fixed64_B4',
        'beam_oracle_fixed64_B4',
    ]
    lines = [
        '# V9-A5.1c History-Preserving Risk-Gated Beam Result',
        '',
        'Date: 2026-07-10',
        '',
        'No training and no DAVIS data were used. Official TrackOn2 query/memory state remained unchanged.',
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
        '## Primary beam diagnostics',
        '',
        '```json',
        json.dumps(result['beam_diagnostics'][PRIMARY_POLICY], indent=2),
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
        '## Replay and integrity',
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
    frames_to_run = LENGTH if args.max_frames == 0 else args.max_frames
    if frames_to_run < 2 or frames_to_run > LENGTH:
        raise SystemExit(f'--max-frames must resolve to [2,{LENGTH}]')
    if frames_to_run < LENGTH and args.max_clips != 1:
        raise SystemExit(
            'partial-frame replay is valid only with --max-clips 1 because the '
            'frozen V9-A5.1b NPZ is stored in full-clip row order'
        )

    start_all = time.perf_counter()
    input_audit = verify_inputs(verify_frames=not args.skip_full_frame_hash)
    frozen = np.load(V9A51B_NPZ, allow_pickle=True)
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

    clips = [f'{sequence}:{start}' for sequence in SEQUENCES for start in CLIP_STARTS]
    if args.max_clips > 0:
        clips = clips[: args.max_clips]

    scalar: dict[str, list[Any]] = defaultdict(list)
    beam_refined_rows: dict[str, list[np.ndarray]] = {name: [] for name in POLICIES}
    beam_raw_rows: dict[str, list[np.ndarray]] = {name: [] for name in POLICIES}
    beam_cost_rows: dict[str, list[np.ndarray]] = {name: [] for name in POLICIES}
    beam_current_grid_rows: dict[str, list[np.ndarray]] = {name: [] for name in POLICIES}
    beam_signature_rows: dict[str, list[np.ndarray]] = {name: [] for name in POLICIES}
    beam_raw_cluster_rows: dict[str, list[int]] = {name: [] for name in POLICIES}
    beam_refined_cluster_rows: dict[str, list[int]] = {name: [] for name in POLICIES}
    runtime = []
    replay = {
        'row_key_mismatch': 0,
        'official_max_abs': 0.0,
        'risk_mismatch': 0,
        'score_max_abs': 0.0,
        'raw_error_max_abs': 0.0,
        'refined_error_max_abs': 0.0,
        'hybrid_oracle_max_abs': 0.0,
    }
    official_parity = {'p': [], 'v': [], 'q': []}
    row_cursor = 0

    for clip_id in clips:
        sequence, start_s = clip_id.split(':')
        start = int(start_s)
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
        visible_all = raw_visibility & valid & finite
        eligible = np.where(visible_all[0] & (np.sum(visible_all, axis=0) >= 24))[0]
        chosen = np.sort(
            np.random.default_rng(V9B.stable_seed(sequence, start)).choice(
                eligible, NQ, replace=False
            )
        )
        gt_xy = trajectories[:, chosen]
        visible = visible_all[:, chosen]
        subset_flags = V9B.build_visibility_subsets(visible)
        frame_paths = [
            sequence_dir / 'rgbs' / f'rgb_{start + frame:05d}.jpg'
            for frame in range(frames_to_run)
        ]
        width, height = Image.open(frame_paths[0]).size
        query_xy = np.empty((NQ, 2), dtype=np.float32)
        query_xy[:, 0] = gt_xy[0, :, 0] * UNIFIED_DENOM / float(width - 1)
        query_xy[:, 1] = gt_xy[0, :, 1] * UNIFIED_DENOM / float(height - 1)
        query_yx = np.stack(
            [
                gt_xy[0, :, 1] / float(height - 1),
                gt_xy[0, :, 0] / float(width - 1),
            ],
            axis=1,
        ).astype(np.float32)
        queries = torch.from_numpy(query_xy).float().to(device)
        support = get_points_on_a_grid(20, (256, 256), device).squeeze(0)
        combined = torch.cat([queries, support], dim=0)
        predictor.reset()
        predictor.initial_capacity = len(combined)
        beams: dict[str, list[list[BeamState]]] = {
            name: [[] for _ in range(NQ)] for name in POLICIES
        }
        clip_start = time.perf_counter()

        for frame in range(frames_to_run):
            image = V9B.load_frame(frame_paths[frame], device)
            frame_features = model.extract_frame_features(image)
            if frame == 0:
                predictor.init_queries((frame_features[-1], device), combined, 256, 256)
            active_q = predictor.q_init[: predictor.N]
            active_mask = predictor.temporal_mask[: predictor.N]
            active_memory = predictor.point_memory[: predictor.N]
            p, v_logit, q_new, diag = V9B.track_frame_diag(
                model,
                active_q,
                active_mask,
                active_memory,
                frame_features,
                256,
                256,
            )
            extended = V9B.extended_candidates(
                model,
                diag['q_pre'][:NQ],
                frame_features,
                diag['c1'][:NQ],
                K64,
            )
            refined = V9B.candidate_conditioned_refinement(
                model,
                diag['q_pre'][:NQ],
                extended['post_fusion'],
                frame_features,
            )
            raw_positions = V9B.model_xy_to_yx_norm(
                extended['positions_model_xy'], model
            )
            refined_positions = V9B.model_xy_to_yx_norm(
                refined['refined_model_xy'], model
            )
            scores = extended['score'].detach().cpu().float().numpy()
            descriptors = extended['post_fusion'].detach().cpu().float().numpy()
            descriptors /= np.maximum(
                np.linalg.norm(descriptors, axis=2, keepdims=True), 1e-8
            )
            grid_indices = extended['correlation_indices'].detach().cpu().numpy().astype(np.int32)
            official_yx = V9B.output_xy_to_yx_norm(p[:NQ])
            v_conf = torch.sigmoid(v_logit[:NQ]).detach().cpu().numpy()
            u_conf = torch.sigmoid(diag['u_logit'][:NQ]).detach().cpu().numpy()
            risk = (v_conf < float(model.delta_v)) | (u_conf >= 0.5)

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
                official_parity['p'].append(float(torch.max(torch.abs(p - p_ref))))
                official_parity['v'].append(float(torch.max(torch.abs(v_logit - v_ref))))
                official_parity['q'].append(float(torch.max(torch.abs(q_new - q_ref))))

            for query in range(NQ):
                frozen_row = row_cursor
                if (
                    str(frozen['clip_id'][frozen_row]) != clip_id
                    or str(frozen['sequence'][frozen_row]) != sequence
                    or int(frozen['frame_tau'][frozen_row]) != frame
                    or int(frozen['query_idx'][frozen_row]) != query
                ):
                    replay['row_key_mismatch'] += 1
                gt_yx = np.asarray(
                    [
                        gt_xy[frame, query, 1] / float(height - 1),
                        gt_xy[frame, query, 0] / float(width - 1),
                    ],
                    dtype=np.float32,
                )
                raw_error = np.linalg.norm(
                    (raw_positions[query] - gt_yx[None, :]) * UNIFIED_DENOM,
                    axis=1,
                ).astype(np.float32)
                refined_error = np.linalg.norm(
                    (refined_positions[query] - gt_yx[None, :]) * UNIFIED_DENOM,
                    axis=1,
                ).astype(np.float32)
                official_error = float(
                    np.linalg.norm((official_yx[query] - gt_yx) * UNIFIED_DENOM)
                )
                full_candidate_oracle = (
                    min(official_error, float(np.min(refined_error)))
                    if bool(risk[query])
                    else official_error
                )
                score_index = int(np.argmax(scores[query]))
                frozen_score_hybrid = (
                    float(refined_error[score_index])
                    if bool(risk[query])
                    else official_error
                )

                replay['official_max_abs'] = max(
                    replay['official_max_abs'],
                    abs(official_error - float(frozen['official_final'][frozen_row])),
                )
                replay['risk_mismatch'] += int(
                    bool(risk[query]) != bool(frozen['risk'][frozen_row])
                )
                replay['score_max_abs'] = max(
                    replay['score_max_abs'],
                    float(
                        np.max(
                            np.abs(
                                scores[query]
                                - frozen['candidate_score'][frozen_row]
                            )
                        )
                    ),
                )
                replay['raw_error_max_abs'] = max(
                    replay['raw_error_max_abs'],
                    float(
                        np.max(
                            np.abs(
                                raw_error
                                - frozen['raw_candidate_error'][frozen_row]
                            )
                        )
                    ),
                )
                replay['refined_error_max_abs'] = max(
                    replay['refined_error_max_abs'],
                    float(
                        np.max(
                            np.abs(
                                refined_error
                                - frozen['refined_candidate_error'][frozen_row]
                            )
                        )
                    ),
                )
                replay['hybrid_oracle_max_abs'] = max(
                    replay['hybrid_oracle_max_abs'],
                    abs(
                        full_candidate_oracle
                        - float(frozen['hybrid_refined_oracle'][frozen_row])
                    ),
                )

                scalar['clip_id'].append(clip_id)
                scalar['sequence'].append(sequence)
                scalar['frame_tau'].append(int(frame))
                scalar['query_idx'].append(int(query))
                scalar['gt_visible'].append(bool(visible[frame, query]))
                scalar['risk'].append(bool(risk[query]))
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
                scalar['opportunity'].append(
                    bool(frozen['opportunity'][frozen_row])
                )
                scalar['official_final'].append(official_error)
                scalar['v9a51b_hybrid_score_top1'].append(frozen_score_hybrid)
                scalar['v9a51b_full_candidate_oracle'].append(
                    full_candidate_oracle
                )

                if frame == 0:
                    for policy in POLICIES:
                        beams[policy][query] = [
                            initialize_state(
                                query_yx[query],
                                raw_positions[query],
                                descriptors[query],
                                grid_indices[query],
                            )
                        ]
                else:
                    for policy, config_row in POLICIES.items():
                        mode = str(config_row['mode'])
                        width_value = int(config_row['width'])
                        if mode == 'fixed16':
                            budget = K16
                        elif mode == 'fixed64':
                            budget = K64
                        elif mode == 'dynamic':
                            budget = K64 if bool(risk[query]) else K16
                        else:
                            raise RuntimeError(mode)
                        beams[policy][query] = update_beam(
                            beams[policy][query],
                            raw_positions[query],
                            refined_positions[query],
                            descriptors[query],
                            scores[query],
                            grid_indices[query],
                            budget,
                            width_value,
                        )

                for policy, config_row in POLICIES.items():
                    width_value = int(config_row['width'])
                    states = beams[policy][query]
                    if frame == 0:
                        raw_state_error = np.zeros(width_value, dtype=np.float32)
                        refined_state_error = np.zeros(width_value, dtype=np.float32)
                        cost = np.zeros(width_value, dtype=np.float32)
                        current_grid = np.full(
                            width_value,
                            int(states[0].grid_history[-1]),
                            dtype=np.int32,
                        )
                        signature = np.full(
                            (width_value, 3), -1, dtype=np.int32
                        )
                        signature[:, -1] = int(states[0].grid_history[-1])
                    else:
                        raw_state_error = np.asarray(
                            [
                                np.linalg.norm(
                                    (state.raw_coord - gt_yx) * UNIFIED_DENOM
                                )
                                for state in states
                            ],
                            dtype=np.float32,
                        )
                        refined_state_error = np.asarray(
                            [
                                np.linalg.norm(
                                    (state.refined_coord - gt_yx)
                                    * UNIFIED_DENOM
                                )
                                for state in states
                            ],
                            dtype=np.float32,
                        )
                        cost = np.asarray(
                            [state.window_cost for state in states],
                            dtype=np.float32,
                        )
                        current_grid = np.asarray(
                            [state.grid_history[-1] for state in states],
                            dtype=np.int32,
                        )
                        signature = np.full(
                            (width_value, 3), -1, dtype=np.int32
                        )
                        for state_i, state in enumerate(states):
                            signature[state_i, -len(state.grid_history):] = np.asarray(
                                state.grid_history, dtype=np.int32
                            )

                    beam_raw_rows[policy].append(raw_state_error)
                    beam_refined_rows[policy].append(refined_state_error)
                    beam_cost_rows[policy].append(cost)
                    beam_current_grid_rows[policy].append(current_grid)
                    beam_signature_rows[policy].append(signature)
                    beam_raw_cluster_rows[policy].append(
                        V9B.spatial_cluster_count(
                            np.stack([state.raw_coord for state in states]), 4.0
                        )
                    )
                    beam_refined_cluster_rows[policy].append(
                        V9B.spatial_cluster_count(
                            np.stack([state.refined_coord for state in states]), 4.0
                        )
                    )
                    system_top1 = (
                        float(refined_state_error[0])
                        if bool(risk[query]) and frame > 0
                        else official_error
                    )
                    system_oracle = (
                        min(official_error, float(np.min(refined_state_error)))
                        if bool(risk[query]) and frame > 0
                        else official_error
                    )
                    scalar[f'beam_top1_{policy}'].append(system_top1)
                    scalar[f'beam_oracle_{policy}'].append(system_oracle)
                    scalar[f'beam_raw_min_{policy}'].append(
                        float(np.min(raw_state_error))
                    )
                row_cursor += 1

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

    expected_rows = len(clips) * frames_to_run * NQ
    arrays: dict[str, np.ndarray] = {
        'clip_id': np.asarray(scalar['clip_id'], dtype=object),
        'sequence': np.asarray(scalar['sequence'], dtype=object),
        'frame_tau': np.asarray(scalar['frame_tau'], dtype=np.int16),
        'query_idx': np.asarray(scalar['query_idx'], dtype=np.int16),
        'gt_visible': np.asarray(scalar['gt_visible'], dtype=bool),
        'risk': np.asarray(scalar['risk'], dtype=bool),
        'reentry_first': np.asarray(scalar['reentry_first'], dtype=bool),
        'reentry_early4': np.asarray(scalar['reentry_early4'], dtype=bool),
        'reentry_early8': np.asarray(scalar['reentry_early8'], dtype=bool),
        'occ_length': np.asarray(scalar['occ_length'], dtype=np.int16),
        'hard_official': np.asarray(scalar['hard_official'], dtype=bool),
        'opportunity': np.asarray(scalar['opportunity'], dtype=bool),
        'official_final': np.asarray(scalar['official_final'], dtype=np.float32),
        'v9a51b_hybrid_score_top1': np.asarray(
            scalar['v9a51b_hybrid_score_top1'], dtype=np.float32
        ),
        'v9a51b_full_candidate_oracle': np.asarray(
            scalar['v9a51b_full_candidate_oracle'], dtype=np.float32
        ),
    }
    for policy, config_row in POLICIES.items():
        width_value = int(config_row['width'])
        arrays[f'beam_top1_{policy}'] = np.asarray(
            scalar[f'beam_top1_{policy}'], dtype=np.float32
        )
        arrays[f'beam_oracle_{policy}'] = np.asarray(
            scalar[f'beam_oracle_{policy}'], dtype=np.float32
        )
        arrays[f'beam_raw_min_{policy}'] = np.asarray(
            scalar[f'beam_raw_min_{policy}'], dtype=np.float32
        )
        arrays[f'beam_refined_error_{policy}'] = np.stack(
            beam_refined_rows[policy]
        ).reshape(expected_rows, width_value)
        arrays[f'beam_raw_error_{policy}'] = np.stack(
            beam_raw_rows[policy]
        ).reshape(expected_rows, width_value)
        arrays[f'beam_cost_{policy}'] = np.stack(
            beam_cost_rows[policy]
        ).reshape(expected_rows, width_value)
        arrays[f'beam_current_grid_{policy}'] = np.stack(
            beam_current_grid_rows[policy]
        ).reshape(expected_rows, width_value)
        arrays[f'beam_signature_{policy}'] = np.stack(
            beam_signature_rows[policy]
        ).reshape(expected_rows, width_value, 3)
        arrays[f'beam_raw_cluster4_{policy}'] = np.asarray(
            beam_raw_cluster_rows[policy], dtype=np.int16
        )
        arrays[f'beam_refined_cluster4_{policy}'] = np.asarray(
            beam_refined_cluster_rows[policy], dtype=np.int16
        )

    if len(arrays['clip_id']) != expected_rows or row_cursor != expected_rows:
        raise RuntimeError('row count mismatch')
    for key, value in arrays.items():
        if value.dtype in (object, bool):
            continue
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f'non-finite output: {key}')

    frame_positive = arrays['frame_tau'] > 0
    visible = arrays['gt_visible'] & frame_positive
    risk_visible = visible & arrays['risk']
    masks: dict[str, np.ndarray] = {
        'all_visible': visible,
        'risk_visible': risk_visible,
        'nonrisk_visible': visible & ~arrays['risk'],
        'hard_official': visible & arrays['hard_official'],
        'opportunity': visible & arrays['opportunity'],
        'reentry_first': visible & arrays['reentry_first'],
        'reentry_early4': visible & arrays['reentry_early4'],
        'reentry_early8': visible & arrays['reentry_early8'],
        'reentry_occ1': visible & arrays['reentry_first'] & (arrays['occ_length'] == 1),
        'reentry_occ2_4': visible & arrays['reentry_first'] & (arrays['occ_length'] >= 2) & (arrays['occ_length'] <= 4),
        'reentry_occ5_8': visible & arrays['reentry_first'] & (arrays['occ_length'] >= 5) & (arrays['occ_length'] <= 8),
        'reentry_occ_gt8': visible & arrays['reentry_first'] & (arrays['occ_length'] > 8),
    }
    for sequence in SEQUENCES:
        masks[f'sequence_{sequence}'] = visible & (arrays['sequence'] == sequence)
        masks[f'reentry_sequence_{sequence}'] = masks['reentry_first'] & (
            arrays['sequence'] == sequence
        )
    for clip_id in clips:
        masks[f'clip_{clip_id}'] = visible & (arrays['clip_id'] == clip_id)

    readout_names = [
        'official_final',
        'v9a51b_hybrid_score_top1',
        'v9a51b_full_candidate_oracle',
    ]
    for policy in POLICIES:
        readout_names.extend(
            [
                f'beam_top1_{policy}',
                f'beam_oracle_{policy}',
                f'beam_raw_min_{policy}',
            ]
        )
    official_error = arrays['official_final'].astype(np.float64)
    summaries = {
        name: {
            mask_name: V9B.metric_summary(arrays[name], official_error, mask)
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

    beam_diagnostics = {}
    full_oracle = arrays['v9a51b_full_candidate_oracle']
    full_gain = float(np.sum((official_error - full_oracle)[visible]))
    for policy, config_row in POLICIES.items():
        width_value = int(config_row['width'])
        refined_matrix = arrays[f'beam_refined_error_{policy}']
        raw_matrix = arrays[f'beam_raw_error_{policy}']
        current_grid = arrays[f'beam_current_grid_{policy}']
        signature = arrays[f'beam_signature_{policy}']
        system_oracle = arrays[f'beam_oracle_{policy}']
        beam_gain = float(np.sum((official_error - system_oracle)[visible]))
        beam_diagnostics[policy] = {
            'width': width_value,
            'raw_survival_risk_visible': {
                str(radius): float(
                    np.mean(np.any(raw_matrix[risk_visible] <= float(radius), axis=1))
                )
                if np.any(risk_visible)
                else None
                for radius in [1, 2, 4, 8]
            },
            'refined_survival_risk_visible': {
                str(radius): float(
                    np.mean(
                        np.any(
                            refined_matrix[risk_visible] <= float(radius), axis=1
                        )
                    )
                )
                if np.any(risk_visible)
                else None
                for radius in [1, 2, 4, 8]
            },
            'mean_unique_current_grid_risk_visible': float(
                np.mean(
                    [
                        len(np.unique(row))
                        for row in current_grid[risk_visible]
                    ]
                )
            )
            if np.any(risk_visible)
            else None,
            'mean_unique_signature_risk_visible': float(
                np.mean(
                    [
                        len({tuple(value) for value in row})
                        for row in signature[risk_visible]
                    ]
                )
            )
            if np.any(risk_visible)
            else None,
            'raw_cluster4_mean_risk_visible': float(
                np.mean(arrays[f'beam_raw_cluster4_{policy}'][risk_visible])
            )
            if np.any(risk_visible)
            else None,
            'refined_cluster4_mean_risk_visible': float(
                np.mean(
                    arrays[f'beam_refined_cluster4_{policy}'][risk_visible]
                )
            )
            if np.any(risk_visible)
            else None,
            'top1_oracle_gap_mean_risk_visible': float(
                np.mean(
                    arrays[f'beam_top1_{policy}'][risk_visible]
                    - arrays[f'beam_oracle_{policy}'][risk_visible]
                )
            )
            if np.any(risk_visible)
            else None,
            'raw_state_min_mean_visible': float(
                np.mean(arrays[f'beam_raw_min_{policy}'][visible])
            ),
            'raw_state_min_mean_risk_visible': (
                float(np.mean(arrays[f'beam_raw_min_{policy}'][risk_visible]))
                if np.any(risk_visible)
                else None
            ),
            'beam_oracle_gap_to_full_candidate_mean_visible': float(
                np.mean(system_oracle[visible] - full_oracle[visible])
            ),
            'full_candidate_headroom_retained_fraction': (
                beam_gain / full_gain if full_gain > 0 else None
            ),
        }

    primary_top1 = f'beam_top1_{PRIMARY_POLICY}'
    primary_oracle = f'beam_oracle_{PRIMARY_POLICY}'
    if args.max_clips > 0 or frames_to_run < LENGTH:
        gates = {
            'smoke_only': True,
            'pass_all': False,
            'decision': 'SMOKE_ONLY: full three-sequence beam gates not evaluated.',
        }
    else:
        def build_gate(name: str, require_all_clips_nonpositive: bool) -> dict[str, Any]:
            sequence_mean = {
                sequence: bool(
                    summaries[name][f'sequence_{sequence}']['mean_error']
                    < summaries['official_final'][f'sequence_{sequence}']['mean_error']
                )
                for sequence in SEQUENCES
            }
            sequence_safe16 = {
                sequence: bool(
                    summaries[name][f'sequence_{sequence}']['safe16']
                    >= summaries['official_final'][f'sequence_{sequence}']['safe16']
                )
                for sequence in SEQUENCES
            }
            reentry_nonincrease = {}
            for sequence in SEQUENCES:
                official_row = summaries['official_final'][
                    f'reentry_sequence_{sequence}'
                ]
                beam_row = summaries[name][f'reentry_sequence_{sequence}']
                reentry_nonincrease[sequence] = (
                    None
                    if official_row['n'] < 10
                    else bool(beam_row['mean_error'] <= official_row['mean_error'])
                )
            clip_nonpositive = all(
                value <= TOL
                for value in per_clip_difference[name].values()
            )
            gate = {
                'sequence_mean_improved': sequence_mean,
                'sequence_safe16_not_decreased': sequence_safe16,
                'global_better_gt_worse': bool(
                    summaries[name]['all_visible']['better']
                    > summaries[name]['all_visible']['worse']
                ),
                'clip_bootstrap_ci_upper_lt_zero': bool(
                    bootstrap[name]['mean_difference_95_ci'][1] < 0.0
                ),
                'first_reentry_mean_improved': bool(
                    summaries[name]['reentry_first']['n'] > 0
                    and summaries[name]['reentry_first']['mean_error']
                    < summaries['official_final']['reentry_first']['mean_error']
                ),
                'early8_mean_improved': bool(
                    summaries[name]['reentry_early8']['n'] > 0
                    and summaries[name]['reentry_early8']['mean_error']
                    < summaries['official_final']['reentry_early8']['mean_error']
                ),
                'reentry_sequence_nonincrease_when_n_ge10': reentry_nonincrease,
                'all_clip_mean_differences_nonpositive': clip_nonpositive,
            }
            gate['pass_all'] = bool(
                all(sequence_mean.values())
                and all(sequence_safe16.values())
                and gate['global_better_gt_worse']
                and gate['clip_bootstrap_ci_upper_lt_zero']
                and gate['first_reentry_mean_improved']
                and gate['early8_mean_improved']
                and all(value is None or value for value in reentry_nonincrease.values())
                and (clip_nonpositive or not require_all_clips_nonpositive)
            )
            return gate

        deterministic_gate = build_gate(primary_top1, False)
        reachability_gate = build_gate(primary_oracle, True)
        if deterministic_gate['pass_all']:
            decision = (
                'HISTORY_BEAM_TOP1_PASS: frozen history-preserving dynamic B4 '
                'improves official final on all synthetic gates. Do not read DAVIS yet.'
            )
        elif reachability_gate['pass_all']:
            decision = (
                'HISTORY_BEAM_REACHABILITY_ONLY: deterministic path scoring fails, '
                'but the surviving beam preserves sequence-consistent refined alternatives. '
                'A later sequence-heldout learned beam readout is justified; do not read DAVIS.'
            )
        else:
            decision = (
                'HISTORY_BEAM_PRUNING_FAIL: the beam does not preserve the committed '
                'V9-A5.1b headroom. Redesign state/pruning before any training.'
            )
        gates = {
            'primary_policy': PRIMARY_POLICY,
            'deterministic_top1': deterministic_gate,
            'beam_reachability': reachability_gate,
            'pass_all': bool(
                deterministic_gate['pass_all'] or reachability_gate['pass_all']
            ),
            'decision': decision,
        }

    signature_unique_violations = {}
    beam_width_shape_pass = {}
    for policy, config_row in POLICIES.items():
        width_value = int(config_row['width'])
        signature_matrix = arrays[f'beam_signature_{policy}']
        beam_width_shape_pass[policy] = bool(
            signature_matrix.shape == (expected_rows, width_value, 3)
            and arrays[f'beam_refined_error_{policy}'].shape
            == (expected_rows, width_value)
            and arrays[f'beam_raw_error_{policy}'].shape
            == (expected_rows, width_value)
        )
        violations = 0
        for row in signature_matrix[frame_positive]:
            if len({tuple(value.tolist()) for value in row}) != width_value:
                violations += 1
        signature_unique_violations[policy] = int(violations)

    integrity = {
        'input_audit': input_audit,
        'processed_rows': int(len(arrays['clip_id'])),
        'expected_rows': int(expected_rows),
        'clips': int(len(clips)),
        'frames_per_clip': int(frames_to_run),
        'official_p_max_abs': float(max(official_parity['p'])),
        'official_v_max_abs': float(max(official_parity['v'])),
        'official_q_max_abs': float(max(official_parity['q'])),
        'v9a51b_replay': replay,
        'beam_width_shape_pass': beam_width_shape_pass,
        'signature_unique_violations': signature_unique_violations,
        'beam_update_signature': list(inspect.signature(update_beam).parameters),
        'beam_update_uses_gt_or_error': any(
            token in inspect.getsource(update_beam).lower()
            for token in ['gt_', 'ground_truth', 'error_px']
        ),
        'all_outputs_finite': True,
    }
    if max(
        integrity['official_p_max_abs'],
        integrity['official_v_max_abs'],
        integrity['official_q_max_abs'],
    ) > 1e-6:
        raise RuntimeError(f'official parity failed: {integrity}')
    if replay['row_key_mismatch'] != 0 or replay['risk_mismatch'] != 0:
        raise RuntimeError(f'V9-A5.1b row/risk replay failed: {replay}')
    if replay['official_max_abs'] > 1e-6:
        raise RuntimeError(f'official replay failed: {replay}')
    if replay['score_max_abs'] > 6e-5:
        raise RuntimeError(f'score replay failed: {replay}')
    if max(
        replay['raw_error_max_abs'],
        replay['refined_error_max_abs'],
    ) > 1e-4:
        raise RuntimeError(f'candidate error replay failed: {replay}')
    if replay['hybrid_oracle_max_abs'] > 1e-6:
        raise RuntimeError(f'hybrid oracle replay failed: {replay}')
    if integrity['beam_update_uses_gt_or_error']:
        raise RuntimeError('beam update source references GT/error')
    if not all(beam_width_shape_pass.values()):
        raise RuntimeError(f'beam width/shape audit failed: {beam_width_shape_pass}')
    if any(value != 0 for value in signature_unique_violations.values()):
        raise RuntimeError(
            f'beam signature uniqueness failed: {signature_unique_violations}'
        )

    result = {
        'script': 'scripts/v9a51c_history_preserving_beam_audit.py',
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
            'cost_window': int(COST_WINDOW),
            'policies': POLICIES,
            'primary_policy': PRIMARY_POLICY,
            'risk_rule': 'sigmoid(v_logit) < 0.8 OR sigmoid(u_logit) >= 0.5',
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
        'summaries': summaries,
        'beam_diagnostics': beam_diagnostics,
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
                'official_mean': summaries['official_final']['all_visible']['mean_error'],
                'primary_top1_mean': summaries[f'beam_top1_{PRIMARY_POLICY}']['all_visible']['mean_error'],
                'primary_oracle_mean': summaries[f'beam_oracle_{PRIMARY_POLICY}']['all_visible']['mean_error'],
                'replay': replay,
            },
            ensure_ascii=False,
        )
    )


if __name__ == '__main__':
    main()
