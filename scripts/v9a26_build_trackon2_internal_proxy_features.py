#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TRACKON = ROOT / 'baselines/track_on'
if str(TRACKON) not in sys.path:
    sys.path.insert(0, str(TRACKON))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
DINO_LOCAL = ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m'
os.environ['DINOV3_LOCAL_DIR'] = str(DINO_LOCAL)
BASE = ROOT / 'outputs/paper_discovery_2026-07-05'
OUTDIR = BASE / 'v9a26_trackon2_internal_proxy'
JOINT = BASE / 'v9a2_anchor_uncertainty_reacquisition/v9a2_joint_w8_common_plus_w16_extension_v3.npz'
OLD_CACHE = ROOT / 'outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt'
DAVIS = ROOT / 'datasets/tapvid_davis/tapvid_davis.pkl'
CKPT = TRACKON / 'checkpoints_trackon2_dinov3.pt'
CONFIG = TRACKON / 'config/test.yaml'

from model.trackon_predictor import Predictor
from utils.coord_utils import get_points_on_a_grid, indices_to_coords
from utils.train_utils import load_args_from_yaml


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def correlation_stats(c: torch.Tensor, indices: dict[str, int]) -> tuple[list[float], list[str]]:
    # c: (P,)
    cf = c.float()
    top = torch.topk(cf, k=2).values
    prob = torch.softmax(cf, dim=0)
    entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum() / math.log(max(2, cf.numel()))
    vals = [float(cf.max()), float(cf.mean()), float(cf.std(unbiased=False)), float(top[0] - top[1]), float(entropy)]
    names = ['max', 'mean', 'std', 'top12_margin', 'softmax_entropy_norm']
    for name, idx in indices.items():
        score = float(cf[int(idx)])
        rank_frac = float(torch.mean((cf <= cf[int(idx)]).float()))
        vals.extend([score, rank_frac, float(cf.max() - cf[int(idx)])])
        names.extend([f'{name}_score', f'{name}_percentile', f'{name}_gap_to_max'])
    return vals, names


def norm_yx_to_patch_index(yx: np.ndarray, hf: int, wf: int) -> int:
    y = float(np.clip(yx[0], 0.0, 1.0)); x = float(np.clip(yx[1], 0.0, 1.0))
    row = int(np.clip(np.rint(y * hf - 0.5), 0, hf - 1))
    col = int(np.clip(np.rint(x * wf - 0.5), 0, wf - 1))
    return row * wf + col


def topk_summary(x: torch.Tensor, prefix: str) -> tuple[list[float], list[str]]:
    xf = x.float().reshape(-1)
    sorted_vals = torch.sort(xf, descending=True).values
    vals = [float(xf.max()), float(xf.mean()), float(xf.std(unbiased=False)), float(sorted_vals[0] - sorted_vals[1]), float(xf.min())]
    names = [f'{prefix}_max', f'{prefix}_mean', f'{prefix}_std', f'{prefix}_top12_margin', f'{prefix}_min']
    return vals, names


def memory_summary(q_init: torch.Tensor, q_new: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor) -> tuple[list[float], list[str]]:
    valid = ~mask.bool()
    vals = []
    names = []
    q0 = F.normalize(q_init.float(), dim=0)
    q1 = F.normalize(q_new.float(), dim=0)
    vals.extend([float(F.cosine_similarity(q0, q1, dim=0)), float(torch.norm(q_init.float() - q_new.float())), float(torch.norm(q_new.float()))])
    names.extend(['qinit_qnew_cos', 'qinit_qnew_l2', 'qnew_norm'])
    n_valid = int(valid.sum())
    vals.append(float(n_valid)); names.append('memory_valid_count')
    if n_valid > 0:
        mem = F.normalize(memory[valid].float(), dim=1)
        s0 = mem @ q0; s1 = mem @ q1
        for prefix, s in [('qinit_mem', s0), ('qnew_mem', s1)]:
            vals.extend([float(s.max()), float(s.mean()), float(s.std(unbiased=False)), float(s.min()), float(s[-1])])
            names.extend([f'{prefix}_cos_max', f'{prefix}_cos_mean', f'{prefix}_cos_std', f'{prefix}_cos_min', f'{prefix}_last_cos'])
    else:
        vals.extend([0.0] * 10)
        names.extend([f'{prefix}_{stat}' for prefix in ['qinit_mem','qnew_mem'] for stat in ['cos_max','cos_mean','cos_std','cos_min','last_cos']])
    return vals, names


@torch.no_grad()
def track_frame_diag(model, q_init, temporal_mask, point_memory, frame_features, h_in: int, w_in: int):
    n, d = q_init.shape
    m = point_memory.shape[1]
    f4, f8, f16, f32, f_fused = frame_features
    q_t = q_init.unsqueeze(0).clone()
    memory = point_memory.clone()
    mask = torch.zeros(n, m + 1, device=q_init.device, dtype=torch.bool)
    mask[:, :-1] = temporal_mask.clone(); mask[:, -1] = False
    qkv = torch.zeros(n, m + 1, d, device=q_init.device, dtype=memory.dtype)
    qkv[:, :-1] = memory
    for i in range(model.decoder_layer_num):
        q_t = model.feature_attention[i](q_t, f_fused, f_fused)
        q_t = model.query_attention[i](q_t, q_t, q_t)
        qkv[:, -1] = q_t.view(n, d)
        qkv = model.memory_attention[i](qkv + model.t_embedding, qkv + model.t_embedding, qkv, mask)
        q_t = qkv[:, -1].unsqueeze(0)
        qkv[:, :-1] = qkv[:, :-1].clone()
    q_pre = model.projection1(q_t)
    c1 = model.multiscale_correlation(q_pre, f4, f8, f16, f32)
    q_rerank, p_topk, u_topk, s_topk = model.reranking_head(q_pre, f4, f8, f16, f32, c1)
    q_new = model.projection2(q_rerank)
    c2 = model.multiscale_correlation(q_new, f4, f8, f16, f32)
    p_patch = indices_to_coords(torch.argmax(c2, dim=-1).unsqueeze(1), model.input_size, model.stride).squeeze(1)
    offsets, v_logit, u_logit = model.prediction_head(q_new, f4, f8, f16, f32, p_patch)
    p_model = p_patch[0] + offsets[-1]
    p = p_model.clone()
    p[..., 0] = (p[..., 0] / model.W) * w_in
    p[..., 1] = (p[..., 1] / model.H) * h_in
    return p, v_logit, q_new.squeeze(0), {
        'c1': c1.squeeze(0), 'c2': c2.squeeze(0), 'p_topk': p_topk.squeeze(0),
        'u_topk': u_topk.squeeze(0), 's_topk': s_topk.squeeze(0), 'u_logit': u_logit,
        'offsets': offsets, 'p_model': p_model, 'q_pre': q_pre.squeeze(0),
    }


def build_row_feature(
    diag: dict,
    pos: int,
    q_init: torch.Tensor,
    q_new: torch.Tensor,
    memory: torch.Tensor,
    mask: torch.Tensor,
    old_cand_yx: np.ndarray,
    native_yx: np.ndarray,
    old_vis: bool,
    hf: int,
    wf: int,
    model_h: int,
    model_w: int,
) -> tuple[np.ndarray, list[str]]:
    cand_idx = norm_yx_to_patch_index(old_cand_yx, hf, wf)
    native_idx = norm_yx_to_patch_index(native_yx, hf, wf)
    indices = {'old_candidate': cand_idx, 'native': native_idx}
    vals = []
    names = []
    for prefix, corr in [('c1', diag['c1'][pos]), ('c2', diag['c2'][pos])]:
        v, n = correlation_stats(corr, indices)
        vals.extend(v); names.extend([f'{prefix}_{x}' for x in n])
    for prefix, tensor in [('rerank_u', diag['u_topk'][pos]), ('rerank_s', diag['s_topk'][pos])]:
        v, n = topk_summary(tensor, prefix)
        vals.extend(v); names.extend(n)
    ptop = diag['p_topk'][pos].float()  # K,2 model x,y
    old_xy_model = torch.tensor(
        [float(old_cand_yx[1]) * float(model_w), float(old_cand_yx[0]) * float(model_h)],
        device=ptop.device,
    )
    native_xy_model = torch.tensor(
        [float(native_yx[1]) * float(model_w), float(native_yx[0]) * float(model_h)],
        device=ptop.device,
    )
    for prefix, target in [('old_candidate', old_xy_model), ('native', native_xy_model)]:
        dist = torch.norm(ptop - target[None], dim=1)
        vals.extend([float(dist.min()), float(dist.mean()), float((dist <= 16.0).sum()), float((dist <= 32.0).sum())])
        names.extend([f'topk_{prefix}_dist_min', f'topk_{prefix}_dist_mean', f'topk_{prefix}_within16', f'topk_{prefix}_within32'])
    vlogit = float(diag['v_logit'][pos]) if 'v_logit' in diag else float('nan')
    ulogit = float(diag['u_logit'][pos])
    vals.extend([vlogit, float(torch.sigmoid(torch.tensor(vlogit))), ulogit, float(torch.sigmoid(torch.tensor(ulogit))), float(old_vis)])
    names.extend(['visibility_logit', 'visibility_conf', 'uncertainty_logit', 'uncertainty_sigmoid', 'old_candidate_visible'])
    offs = diag['offsets'][:, pos].float()
    vals.extend([float(torch.norm(x)) for x in offs])
    names.extend([f'offset_layer{i}_norm' for i in range(offs.shape[0])])
    proxy_xy = diag['p_model'][pos].float()
    vals.extend([float(torch.norm(proxy_xy - old_xy_model)), float(torch.norm(proxy_xy - native_xy_model))])
    names.extend(['proxy_old_candidate_dist_model_px', 'proxy_native_dist_model_px'])
    mv, mn = memory_summary(q_init, q_new, memory, mask)
    vals.extend(mv); names.extend(mn)
    return np.asarray(vals, dtype=np.float32), names


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--joint', type=Path, default=JOINT)
    ap.add_argument('--out', type=Path, default=OUTDIR / 'v9a26_trackon2_internal_proxy_features.npz')
    ap.add_argument('--max-videos', type=int, default=0)
    ap.add_argument('--max-rows', type=int, default=0)
    ap.add_argument('--validate-official', action='store_true')
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    z = np.load(args.joint, allow_pickle=True)
    ext_idx_all = np.where(z['is_w16_extension'].astype(bool))[0]
    if args.max_rows > 0:
        ext_idx_all = ext_idx_all[:args.max_rows]
    metas_all = [json.loads(str(z['meta_json'][i])) for i in ext_idx_all]
    videos_order = []
    for m in metas_all:
        if m['video_id'] not in videos_order:
            videos_order.append(m['video_id'])
    if args.max_videos > 0:
        allowed = set(videos_order[:args.max_videos])
        keep = [i for i, m in enumerate(metas_all) if m['video_id'] in allowed]
        ext_indices = ext_idx_all[np.asarray(keep, dtype=np.int64)]
        metas = [metas_all[i] for i in keep]
    else:
        ext_indices = ext_idx_all
        metas = metas_all

    old = torch.load(OLD_CACHE, map_location='cpu', weights_only=False)
    old_by = {str(r['video_id']): r for r in old['records']}
    with DAVIS.open('rb') as f:
        davis = pickle.load(f)
    rows_by_video: dict[str, list[tuple[int, dict]]] = {}
    for local_i, m in enumerate(metas):
        rows_by_video.setdefault(str(m['video_id']), []).append((local_i, m))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_args = load_args_from_yaml(str(CONFIG))
    model_args.grad_checkpoint = False
    model_args.M_i = model_args.M
    predictor = Predictor(model_args, checkpoint_path=str(CKPT), support_grid_size=20).to(device).eval()
    memory_policy = str(predictor.model.memory_update_policy)
    if memory_policy not in {'unconditional', 'visibility_selective'}:
        raise RuntimeError(f'unsupported TrackOn2 memory policy: {memory_policy}')

    X_rows: list[np.ndarray | None] = [None] * len(metas)
    feature_names = None
    parity = []
    start_all = time.perf_counter()
    for vi, (vid, targets) in enumerate(rows_by_video.items()):
        rec = old_by[vid]
        video_np = np.asarray(davis[vid]['video'], dtype=np.float32)
        vt = torch.from_numpy(video_np).permute(0, 3, 1, 2).float()
        vt = F.interpolate(vt, size=(256, 256), mode='bilinear', align_corners=False).to(device)
        q = npy(rec['query_points'], np.float32)
        n_orig = q.shape[0]
        q_model = np.zeros_like(q, dtype=np.float32)
        q_model[:, 0] = q[:, 0]; q_model[:, 1] = q[:, 2] * 255.0; q_model[:, 2] = q[:, 1] * 255.0
        queries = torch.from_numpy(q_model).float().to(device)
        extra = get_points_on_a_grid(20, (256, 256), device).squeeze(0)
        extra_queries = torch.cat([torch.zeros(extra.shape[0], 1, device=device), extra], dim=1)
        combined = torch.cat([queries, extra_queries], dim=0)
        query_times = combined[:, 0].long(); query_coords = combined[:, 1:]
        predictor.reset(); predictor.initial_capacity = combined.shape[0]
        tracking_to_original: list[int] = []
        max_tau = max(int(m['frame_tau']) for _, m in targets)
        target_at: dict[int, list[tuple[int, dict]]] = {}
        for local_i, m in targets:
            target_at.setdefault(int(m['frame_tau']), []).append((local_i, m))
        target_frames = sorted(target_at)
        parity_frames: set[int] = set()
        if args.validate_official and target_frames:
            parity_frames = {
                target_frames[0],
                target_frames[len(target_frames) // 2],
                target_frames[-1],
            }
        video_start = time.perf_counter()
        for t in range(max_tau + 1):
            frame = vt[t].unsqueeze(0)
            new_mask = query_times == t
            new_queries = None
            if new_mask.any():
                new_queries = query_coords[new_mask]
                tracking_to_original.extend(new_mask.nonzero(as_tuple=True)[0].tolist())
            f4, f8, f16, f32, fused = predictor.model.extract_frame_features(frame)
            if new_queries is not None and new_queries.shape[0] > 0:
                predictor.init_queries((fused, device), new_queries, 256, 256)
            if predictor.N == 0:
                continue
            active_q = predictor.q_init[:predictor.N]
            active_mask = predictor.temporal_mask[:predictor.N]
            active_memory = predictor.point_memory[:predictor.N]
            frame_features = (f4, f8, f16, f32, fused)
            p, vlogit, qnew, diag = track_frame_diag(predictor.model, active_q, active_mask, active_memory, frame_features, 256, 256)
            diag['v_logit'] = vlogit
            if args.validate_official and t in parity_frames:
                pref, vref, qref = predictor.model.track_frame(
                    active_q, active_mask, active_memory, frame_features, 256, 256
                )
                parity.append({
                    'video_id': vid,
                    'frame': int(t),
                    'active_queries': int(predictor.N),
                    'p_max_abs': float(torch.max(torch.abs(p - pref))),
                    'v_max_abs': float(torch.max(torch.abs(vlogit - vref))),
                    'q_max_abs': float(torch.max(torch.abs(qnew - qref))),
                })
            if t in target_at:
                reverse = {orig: pos for pos, orig in enumerate(tracking_to_original)}
                for local_i, m in target_at[t]:
                    qi = int(m['query_idx'])
                    if qi not in reverse:
                        raise RuntimeError(f'query {qi} inactive at {vid}:{t}')
                    pos = reverse[qi]
                    if qi < 0 or qi >= n_orig:
                        raise RuntimeError(f'query index out of range: {vid} qi={qi} n={n_orig}')
                    expected_query_t = int(m['query_t'])
                    actual_query_t = int(round(float(q[qi, 0])))
                    if expected_query_t != actual_query_t:
                        raise RuntimeError(
                            f'query-time mismatch: {vid} qi={qi} meta={expected_query_t} cache={actual_query_t}'
                        )
                    old_cand = npy(rec['pred_tracks'], np.float32)[qi, t]
                    native_yx = np.asarray(m['native_yx'], dtype=np.float32)
                    old_vis = bool(npy(rec['pred_visibility'], bool)[qi, t])
                    feat, names = build_row_feature(
                        diag,
                        pos,
                        active_q[pos],
                        qnew[pos],
                        active_memory[pos],
                        active_mask[pos],
                        old_cand,
                        native_yx,
                        old_vis,
                        predictor.model.Hf,
                        predictor.model.Wf,
                        predictor.model.H,
                        predictor.model.W,
                    )
                    X_rows[local_i] = feat
                    if feature_names is None:
                        feature_names = names
                    elif feature_names != names:
                        raise RuntimeError('feature names mismatch')
            v_t = vlogit.sigmoid() >= predictor.delta_v
            new_count = 0 if new_queries is None else int(new_queries.shape[0])
            new_query_mask = torch.zeros(predictor.N, dtype=torch.bool, device=device)
            if new_count > 0:
                new_query_mask[predictor.N - new_count:predictor.N] = True
            if memory_policy == 'visibility_selective':
                write_mask = v_t | new_query_mask
            else:
                write_mask = torch.ones(predictor.N, dtype=torch.bool, device=device)
            updated_memory, updated_mask = predictor.model._update_point_memory(
                active_memory, active_mask, qnew, write_mask
            )
            predictor.point_memory[:predictor.N] = updated_memory
            predictor.temporal_mask[:predictor.N] = updated_mask
            predictor.t += 1
        print(json.dumps({'video': vid, 'targets': len(targets), 'max_tau': max_tau, 'seconds': time.perf_counter() - video_start}), flush=True)

    if any(x is None for x in X_rows):
        missing = [i for i, x in enumerate(X_rows) if x is None]
        raise RuntimeError(f'missing features: {missing[:20]}')
    X = np.stack([x for x in X_rows if x is not None], axis=0).astype(np.float32)
    if not np.all(np.isfinite(X)):
        raise RuntimeError('non-finite internal proxy feature')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X_internal=X,
        feature_names=np.asarray(feature_names or [], dtype=object),
        joint_indices=np.asarray(ext_indices, dtype=np.int64),
        row_key_json=z['row_key_json'][ext_indices], event_key_json=z['event_key_json'][ext_indices], meta_json=z['meta_json'][ext_indices],
        y_candidate_good=z['y_candidate_good'][ext_indices], y_candidate_bad=z['y_candidate_bad'][ext_indices], y_false_visible=z['y_false_visible'][ext_indices], y_candidate_worse_px=z['y_candidate_worse_px'][ext_indices],
    )
    parity_max = {
        key: (max((row[key] for row in parity), default=None))
        for key in ['p_max_abs', 'v_max_abs', 'q_max_abs']
    }
    parity_pass = bool(
        parity
        and all(value is not None and value <= 1e-6 for value in parity_max.values())
    ) if args.validate_official else None
    proxy_idx = (feature_names or []).index('proxy_old_candidate_dist_model_px')
    proxy_dist = X[:, proxy_idx].astype(np.float64)
    report = {
        'out': str(args.out),
        'joint': str(args.joint),
        'old_cache': str(OLD_CACHE),
        'n_rows': int(X.shape[0]),
        'feature_dim': int(X.shape[1]),
        'finite_rate': float(np.mean(np.isfinite(X))),
        'videos': list(rows_by_video),
        'device': str(device),
        'seconds': float(time.perf_counter() - start_all),
        'provenance': 'TrackOn2 256-space M24 support-grid20 internal-state proxy; not exact old-cache latent state',
        'trackon_config': {
            'input_size': [int(x) for x in predictor.model.input_size],
            'M_training': int(model_args.M),
            'M_inference': int(model_args.M_i),
            'support_grid_size': int(predictor.support_grid_size),
            'delta_v': float(predictor.delta_v),
            'memory_update_policy': memory_policy,
        },
        'integrity': {
            'checkpoint_sha256': sha256_file(CKPT),
            'config_sha256': sha256_file(CONFIG),
            'old_cache_sha256': sha256_file(OLD_CACHE),
            'row_key_alignment_preserved': True,
            'query_time_alignment_pass': True,
        },
        'official_track_frame_parity': parity,
        'official_track_frame_parity_max': parity_max,
        'official_track_frame_parity_pass': parity_pass,
        'proxy_vs_old_candidate_model_px': {
            'mean': float(np.mean(proxy_dist)),
            'median': float(np.median(proxy_dist)),
            'p95': float(np.quantile(proxy_dist, 0.95)),
            'max': float(np.max(proxy_dist)),
        },
        'feature_names': feature_names or [],
    }
    report_path = args.out.with_suffix('.report.json')
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    if args.validate_official and not parity_pass:
        raise RuntimeError(f'official TrackOn2 parity failed: {parity_max}')
    print(json.dumps({
        'ok': True,
        'out': str(args.out),
        'report': str(report_path),
        'n_rows': report['n_rows'],
        'feature_dim': report['feature_dim'],
        'finite_rate': report['finite_rate'],
        'seconds': report['seconds'],
        'parity_pass': parity_pass,
        'parity_max': parity_max,
        'proxy_old_candidate_p95_model_px': report['proxy_vs_old_candidate_model_px']['p95'],
    }))


if __name__ == '__main__':
    main()
