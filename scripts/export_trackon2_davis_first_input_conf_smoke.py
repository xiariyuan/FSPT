#!/usr/bin/env python3
from __future__ import annotations
import json, os, pickle, sys, time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path('/gemini/code/FSPT')
TRACKON_DIR = ROOT / 'baselines/track_on'
CKPT = TRACKON_DIR / 'checkpoints_trackon2_dinov3.pt'
CONFIG = TRACKON_DIR / 'config/test.yaml'
DINOV3_LOCAL_DIR = ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m'
if DINOV3_LOCAL_DIR.is_dir():
    os.environ['DINOV3_LOCAL_DIR'] = str(DINOV3_LOCAL_DIR)
DAVIS_PKL = ROOT / 'datasets/tapvid_davis/tapvid_davis.pkl'
OUT_DIR = ROOT / 'outputs/paper_discovery_2026-07-05/trackon2_true_base_conf'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# TrackOn2 imports use local model/utils package names.
os.chdir(str(TRACKON_DIR))
sys.path.insert(0, str(TRACKON_DIR))
sys.path.append(str(ROOT))

from model.trackon_predictor import Predictor  # noqa: E402
from utils.train_utils import load_args_from_yaml  # noqa: E402
from utils.coord_utils import get_points_on_a_grid  # noqa: E402


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


@torch.no_grad()
def forward_frame_with_logit(predictor: Predictor, frame: torch.Tensor, new_queries: torch.Tensor | None = None):
    """Mirror Predictor.forward_frame, but also return raw visibility logits."""
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
    """Mirror Predictor.forward, preserving support-grid behavior, plus logits/confidence.

    video: (1,T,3,H,W) [0,255]
    queries: (1,N,3) [t,x,y] pixel
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
        new_queries_mask = (query_times == t)
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
    pred_vis_conf = torch.sigmoid(pred_vis_logit)
    return pred_trajectory, pred_visibility, pred_vis_logit, pred_vis_conf


def build_one_record(video_name: str, entry: dict, predictor: Predictor, vi: int, device: torch.device) -> dict:
    video_np = entry['video'].astype(np.float32)
    T, H, W = video_np.shape[:3]
    video = torch.from_numpy(video_np).permute(0, 3, 1, 2).float().unsqueeze(0).to(device)

    points_np = entry['points']
    occluded_np = entry['occluded']
    # TAPVid points are normalized [0,1] xy. Existing first_input bridge used PIX=255 for eval space;
    # TrackOn2 input needs original pixels, so use W-1/H-1 for model queries.
    valid = np.sum(~occluded_np, axis=1) > 0
    pts_valid = points_np[valid]
    occ_valid = occluded_np[valid]
    query_model = []
    query_cache = []
    for i in range(pts_valid.shape[0]):
        first_vis = int(np.where(occ_valid[i] == 0)[0][0])
        x_norm, y_norm = float(pts_valid[i, first_vis, 0]), float(pts_valid[i, first_vis, 1])
        query_model.append([first_vis, x_norm * max(W - 1, 1), y_norm * max(H - 1, 1)])  # [t,x,y] px
        query_cache.append([first_vis, y_norm, x_norm])  # [t,y,x] norm
    query_model_t = torch.tensor(query_model, dtype=torch.float32, device=device).unsqueeze(0)

    t0 = time.time()
    pred_xy, pred_vis, pred_logit, pred_conf = forward_with_logits(predictor, video, query_model_t)
    infer_sec = time.time() - t0

    pred_xy_tn = pred_xy[0].numpy()  # T,N,2 xy pixel
    pred_vis_tn = pred_vis[0].numpy()
    pred_logit_tn = pred_logit[0].numpy()
    pred_conf_tn = pred_conf[0].numpy()

    pred_yx_norm = pred_xy_tn.transpose(1, 0, 2)[..., [1, 0]].astype(np.float32)
    pred_yx_norm[..., 0] /= max(H - 1, 1)
    pred_yx_norm[..., 1] /= max(W - 1, 1)
    pred_vis_nt = pred_vis_tn.transpose(1, 0).astype(bool)
    pred_logit_nt = pred_logit_tn.transpose(1, 0).astype(np.float32)
    pred_conf_nt = pred_conf_tn.transpose(1, 0).astype(np.float32)

    gt_yx_norm = pts_valid[..., ::-1].astype(np.float32)
    q_cache = np.asarray(query_cache, dtype=np.float32)

    return {
        'video_id': video_name,
        'sequence_index': vi,
        'frame_count': int(T),
        'query_points': q_cache,
        'pred_tracks': pred_yx_norm,
        'pred_visibility': pred_vis_nt,
        'pred_vis_logit': pred_logit_nt,
        'pred_vis_conf': pred_conf_nt,
        'gt_tracks': gt_yx_norm,
        'gt_visibility': (~occ_valid).astype(bool),
        'original_size': np.array([H, W], dtype=np.int32),
        'model_input_size': np.array([384, 512], dtype=np.int32),
        'adapter_version': 'trackon2_forward_online_conf_smoke_v1',
        'raw_coordinate_note': 'TrackOn2 Predictor.forward behavior replicated with support grid; output xy pixels converted to normalized yx by original_size [H-1,W-1]. Queries are TAPVid first visible [t,y,x] normalized.',
        'infer_sec': round(float(infer_sec), 3),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device, flush=True)
    model_args = load_args_from_yaml(str(CONFIG))
    model_args.grad_checkpoint = False
    # Match existing successful TrackOn2 exports unless overridden elsewhere.
    predictor = Predictor(model_args, checkpoint_path=str(CKPT), support_grid_size=20).to(device).eval()
    print('predictor loaded', flush=True)
    with open(DAVIS_PKL, 'rb') as f:
        data = pickle.load(f)
    video_names = sorted(data.keys())
    records = []
    # Smoke: first video only.
    rec = build_one_record(video_names[0], data[video_names[0]], predictor, 0, device)
    records.append(rec)
    out_cache = OUT_DIR / 'trackon2_davis_first_input_conf_smoke_1video.pt'
    out_report = OUT_DIR / 'trackon2_davis_first_input_conf_smoke_1video_report.json'
    payload = {
        'schema_version': 'trackon2_forward_online_conf_smoke_v1',
        'model_name': 'trackon2_dinov3_forward_online_conf',
        'dataset_name': 'tapvid_davis',
        'protocol': 'first+input',
        'checkpoint_path': str(CKPT),
        'config_path': str(CONFIG),
        'dinov3_local_dir': os.environ.get('DINOV3_LOCAL_DIR', ''),
        'records': records,
    }
    torch.save(payload, out_cache)
    r = records[0]
    q = r['query_points']; pred = r['pred_tracks']; pv = r['pred_visibility']; conf = r['pred_vis_conf']; logit = r['pred_vis_logit']
    H, W = r['original_size']
    anchor_err = []
    for i in range(q.shape[0]):
        t = int(q[i,0])
        err = np.linalg.norm((pred[i,t] - q[i,1:]) * np.array([H-1, W-1], dtype=np.float32))
        anchor_err.append(float(err))
    report = {
        'out_cache': str(out_cache),
        'video_id': r['video_id'],
        'queries': int(q.shape[0]),
        'frames': int(r['frame_count']),
        'infer_sec': r['infer_sec'],
        'pred_vis_rate': float(pv.mean()),
        'conf_min': float(conf.min()),
        'conf_max': float(conf.max()),
        'conf_mean': float(conf.mean()),
        'logit_min': float(logit.min()),
        'logit_max': float(logit.max()),
        'vis_equals_conf_ge_delta_v': bool(np.all(pv == (conf >= predictor.delta_v))),
        'delta_v': float(predictor.delta_v),
        'anchor_err_px_mean': float(np.mean(anchor_err)),
        'anchor_err_px_max': float(np.max(anchor_err)),
    }
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
