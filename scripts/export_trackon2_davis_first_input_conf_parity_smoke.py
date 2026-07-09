#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/gemini/code/FSPT')
TRACKON_DIR = ROOT / 'baselines/track_on'
CKPT = TRACKON_DIR / 'checkpoints_trackon2_dinov3.pt'
CONFIG = TRACKON_DIR / 'config/test.yaml'
DINOV3_LOCAL_DIR = ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m'
OLD_CACHE = ROOT / 'outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt'
DAVIS_PKL = ROOT / 'datasets/tapvid_davis/tapvid_davis.pkl'
OUT_DIR = ROOT / 'outputs/paper_discovery_2026-07-05/trackon2_true_base_conf'
OUT_DIR.mkdir(parents=True, exist_ok=True)

if DINOV3_LOCAL_DIR.is_dir():
    os.environ['DINOV3_LOCAL_DIR'] = str(DINOV3_LOCAL_DIR)

# TrackOn2 imports use local model/utils package names.
os.chdir(str(TRACKON_DIR))
sys.path.insert(0, str(TRACKON_DIR))
sys.path.append(str(ROOT))

from model.trackon_predictor import Predictor  # noqa: E402
from utils.coord_utils import get_points_on_a_grid  # noqa: E402
from utils.train_utils import load_args_from_yaml  # noqa: E402


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


@torch.no_grad()
def forward_frame_with_logit(predictor: Predictor, frame: torch.Tensor, new_queries: torch.Tensor | None = None):
    """Mirror Predictor.forward_frame, but preserve raw visibility logits."""
    device = frame.device
    _, _, H, W = frame.shape
    f4_t, f8_t, f16_t, f32_t, f_fused_t = predictor.model.extract_frame_features(frame)
    if new_queries is not None and new_queries.shape[0] > 0:
        predictor.init_queries((f_fused_t, device), new_queries, H, W)
    if predictor.q_init is None or predictor.N == 0:
        return (
            torch.empty(0, 2, device=device),
            torch.empty(0, dtype=torch.bool, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )

    frame_features = (f4_t, f8_t, f16_t, f32_t, f_fused_t)
    p, v_logit, q_new = predictor.model.track_frame(
        predictor.q_init[:predictor.N],
        predictor.temporal_mask[:predictor.N],
        predictor.point_memory[:predictor.N],
        frame_features,
        H,
        W,
    )
    v_t = (v_logit.sigmoid() >= predictor.delta_v)

    if new_queries is not None and new_queries.shape[0] > 0:
        new_count = int(new_queries.shape[0])
        new_query_mask = torch.zeros(predictor.N, dtype=torch.bool, device=device)
        new_query_mask[predictor.N - new_count:predictor.N] = True
    else:
        new_query_mask = torch.zeros(predictor.N, dtype=torch.bool, device=device)

    if predictor.model.memory_update_policy == 'visibility_selective':
        write_mask = v_t | new_query_mask
    elif predictor.model.memory_update_policy == 'unconditional':
        write_mask = torch.ones(predictor.N, dtype=torch.bool, device=device)
    else:
        raise ValueError(f'Unknown memory_update_policy: {predictor.model.memory_update_policy}')

    updated_memory, updated_mask = predictor.model._update_point_memory(
        predictor.point_memory[:predictor.N],
        predictor.temporal_mask[:predictor.N],
        q_new,
        write_mask,
    )
    predictor.point_memory[:predictor.N] = updated_memory
    predictor.temporal_mask[:predictor.N] = updated_mask
    predictor.t += 1
    return p, v_t, v_logit.float()


@torch.no_grad()
def forward_with_logits(predictor: Predictor, video: torch.Tensor, queries: torch.Tensor):
    """Mirror Predictor.forward with support grid and logit preservation.

    video: (1,T,3,H,W), H=W=256 for parity with old input256 bridge.
    queries: (1,N,3), [t,x,y] in input256 pixel space.
    """
    _, T, _, H, W = video.shape
    device = video.device
    queries = queries.squeeze(0)
    N_orig = queries.shape[0]
    query_times = queries[:, 0].long()
    query_coords = queries[:, 1:]

    if predictor.support_grid_size > 0:
        extra = get_points_on_a_grid(predictor.support_grid_size, (H, W), device).squeeze(0)
        extra_queries = torch.cat([torch.zeros(extra.shape[0], 1, device=device), extra], dim=1)
        queries = torch.cat([queries, extra_queries], dim=0)
        query_times = queries[:, 0].long()
        query_coords = queries[:, 1:]

    N_total = queries.shape[0]
    predictor.reset()
    predictor.initial_capacity = N_total

    pred_trajectory = torch.zeros(1, T, N_orig, 2, device='cpu', dtype=torch.float32)
    pred_visibility = torch.zeros(1, T, N_orig, device='cpu', dtype=torch.bool)
    pred_vis_logit = torch.zeros(1, T, N_orig, device='cpu', dtype=torch.float32)
    tracking_to_original: list[int] = []

    for t in range(T):
        frame = video[0, t].unsqueeze(0)
        new_queries_mask = query_times == t
        new_queries_this_frame = None
        if new_queries_mask.any():
            new_queries_this_frame = query_coords[new_queries_mask]
            new_indices = new_queries_mask.nonzero(as_tuple=True)[0]
            tracking_to_original.extend(new_indices.tolist())
        if predictor.N > 0 or new_queries_this_frame is not None:
            p_t, v_t, vlogit_t = forward_frame_with_logit(predictor, frame, new_queries_this_frame)
            if predictor.N > 0:
                tracking_to_original_tensor = torch.tensor(tracking_to_original, dtype=torch.long, device='cpu')
                orig_mask = tracking_to_original_tensor < N_orig
                if orig_mask.any():
                    orig_indices = tracking_to_original_tensor[orig_mask]
                    pred_trajectory[0, t, orig_indices] = p_t[orig_mask].detach().cpu()
                    pred_visibility[0, t, orig_indices] = v_t[orig_mask].detach().cpu()
                    pred_vis_logit[0, t, orig_indices] = vlogit_t[orig_mask].detach().cpu()
    return pred_trajectory, pred_visibility, pred_vis_logit, torch.sigmoid(pred_vis_logit)


def load_resized_video_256(video_id: str, device: torch.device) -> torch.Tensor:
    with open(DAVIS_PKL, 'rb') as f:
        data = pickle.load(f)
    if video_id not in data:
        raise KeyError(f'video_id {video_id} not found in {DAVIS_PKL}')
    video_np = data[video_id]['video'].astype(np.float32)  # T,H,W,3 [0,255]
    vt = torch.from_numpy(video_np).permute(0, 3, 1, 2).float()
    vt = F.interpolate(vt, size=(256, 256), mode='bilinear', align_corners=False)
    return vt.unsqueeze(0).to(device)  # 1,T,3,256,256


def build_parity_record(old_record: dict, predictor: Predictor, device: torch.device) -> dict:
    video_id = str(old_record['video_id'])
    video = load_resized_video_256(video_id, device=device)
    old_q = npy(old_record['query_points'], np.float32)

    # Old bridge query_points are [t,y,x] normalized input256, where norm denominator is 255.
    q_model = np.zeros_like(old_q, dtype=np.float32)
    q_model[:, 0] = old_q[:, 0]
    q_model[:, 1] = old_q[:, 2] * 255.0  # x px
    q_model[:, 2] = old_q[:, 1] * 255.0  # y px
    q_model_t = torch.from_numpy(q_model).float().to(device).unsqueeze(0)

    t0 = time.time()
    pred_xy, pred_vis, pred_logit, pred_conf = forward_with_logits(predictor, video, q_model_t)
    infer_sec = time.time() - t0

    pred_xy_tn = pred_xy[0].numpy()  # T,N,2 xy in input256 pixel space
    pred_vis_tn = pred_vis[0].numpy()
    pred_logit_tn = pred_logit[0].numpy()
    pred_conf_tn = pred_conf[0].numpy()

    pred_yx_norm = pred_xy_tn.transpose(1, 0, 2)[..., [1, 0]].astype(np.float32) / 255.0
    pred_vis_nt = pred_vis_tn.transpose(1, 0).astype(bool)
    pred_logit_nt = pred_logit_tn.transpose(1, 0).astype(np.float32)
    pred_conf_nt = pred_conf_tn.transpose(1, 0).astype(np.float32)

    nr = dict(old_record)
    # Preserve schema/query/GT from old cache exactly.
    nr['query_points'] = npy(old_record['query_points'], np.float32).copy()
    nr['gt_tracks'] = npy(old_record['gt_tracks'], np.float32).copy()
    nr['gt_visibility'] = npy(old_record['gt_visibility'], bool).copy()
    nr['original_size'] = npy(old_record['original_size'], np.int32).copy()
    nr['model_input_size'] = np.asarray([256, 256], dtype=np.int32)
    # Replace only predictions and add confidence/logit.
    nr['pred_tracks'] = pred_yx_norm
    nr['pred_visibility'] = pred_vis_nt
    nr['pred_vis_logit'] = pred_logit_nt
    nr['pred_vis_conf'] = pred_conf_nt
    nr['adapter_version'] = 'trackon2_forward_online_conf_parity_smoke_v1'
    nr['raw_coordinate_note'] = 'Parity export: old first/input bridge query/GT/schema preserved; video resized to input256; TrackOn2 xy output converted to yx/255.'
    nr['infer_sec'] = round(float(infer_sec), 3)
    return nr


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print({'device': str(device), 'dinov3_local_dir': os.environ.get('DINOV3_LOCAL_DIR', '')}, flush=True)
    model_args = load_args_from_yaml(str(CONFIG))
    model_args.grad_checkpoint = False
    predictor = Predictor(model_args, checkpoint_path=str(CKPT), support_grid_size=20).to(device).eval()
    print('predictor loaded', flush=True)

    old = torch.load(OLD_CACHE, map_location='cpu', weights_only=False)
    old_record = old['records'][0]
    new_record = build_parity_record(old_record, predictor, device)

    out_cache = OUT_DIR / 'trackon2_davis_first_input_conf_parity_smoke_1video.pt'
    out_report = OUT_DIR / 'trackon2_davis_first_input_conf_parity_smoke_1video_report.json'
    payload = dict(old)
    payload['schema_version'] = 'trackon2_forward_online_conf_parity_smoke_v1'
    payload['model_name'] = 'trackon2_dinov3_forward_online_conf_parity_smoke'
    payload['records'] = [new_record]
    payload['checkpoint_path'] = str(CKPT)
    payload['config_path'] = str(CONFIG)
    payload['dinov3_local_dir'] = os.environ.get('DINOV3_LOCAL_DIR', '')
    torch.save(payload, out_cache)

    # Strict schema preservation checks.
    schema_checks = {}
    for key in ['query_points', 'gt_tracks', 'gt_visibility', 'original_size']:
        a = npy(old_record[key])
        b = npy(new_record[key])
        schema_checks[key] = {
            'same_shape': bool(a.shape == b.shape),
            'max_abs_diff': float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.shape == b.shape and a.dtype != bool else None,
            'diff_rate': float(np.mean(a != b)) if a.shape == b.shape and (a.dtype == bool or b.dtype == bool) else None,
        }
    q = new_record['query_points']
    pred = new_record['pred_tracks']
    H, W = 256, 256
    anchor_err = []
    for i in range(q.shape[0]):
        t = int(round(float(q[i, 0])))
        anchor_err.append(float(np.linalg.norm((pred[i, t] - q[i, 1:]) * np.array([H - 1, W - 1], dtype=np.float32))))
    pv = new_record['pred_visibility']
    conf = new_record['pred_vis_conf']
    logit = new_record['pred_vis_logit']
    report = {
        'out_cache': str(out_cache),
        'old_cache': str(OLD_CACHE),
        'video_id': str(new_record['video_id']),
        'queries': int(q.shape[0]),
        'frames': int(new_record['frame_count']),
        'infer_sec': new_record['infer_sec'],
        'schema_checks': schema_checks,
        'pred_vis_rate': float(pv.mean()),
        'conf_min': float(conf.min()),
        'conf_max': float(conf.max()),
        'conf_mean': float(conf.mean()),
        'logit_min': float(logit.min()),
        'logit_max': float(logit.max()),
        'vis_equals_conf_ge_delta_v': bool(np.all(pv == (conf >= predictor.delta_v))),
        'delta_v': float(predictor.delta_v),
        'anchor_err_px_mean_input256': float(np.mean(anchor_err)),
        'anchor_err_px_max_input256': float(np.max(anchor_err)),
    }
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
