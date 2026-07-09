#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path('/gemini/code/FSPT')
COTRACKER_ROOT = ROOT / 'baselines/cotracker'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(COTRACKER_ROOT))

from cotracker.predictor import CoTrackerOnlinePredictor
from cotracker.datasets.tap_vid_datasets import TapVidDataset

DEFAULT_BASE_CACHE = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_true_streaming_v3_decoupled_subset_eval/full30_overlap_soft_tau055_w2_j4_min1_confirm4_p081/cotracker3_true_streaming_native_subset.pt'
DEFAULT_DAVIS = ROOT / 'datasets/tapvid_davis/tapvid_davis.pkl'
DEFAULT_CKPT = ROOT / 'baselines/cotracker/checkpoints/scaled_online.pth'
DEFAULT_OUT_CACHE = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_smoke.pt'
DEFAULT_OUT_REPORT = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a4_raw_visconf/cotracker3_v7a4_raw_visconf_smoke_report.json'


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def load_video_for_record(record: dict, ds: TapVidDataset, device: str):
    vid = str(record['video_id'])
    idx = ds.video_names.index(vid)
    sample = ds[idx]
    rgbs = sample.video.unsqueeze(0).to(device).float()
    q_norm = npy(record['query_points'], np.float32)
    q = torch.zeros((1, q_norm.shape[0], 3), device=device, dtype=torch.float32)
    q[0, :, 0] = torch.from_numpy(q_norm[:, 0]).to(device)
    q[0, :, 1] = torch.from_numpy(q_norm[:, 2] * 255.0).to(device)  # x
    q[0, :, 2] = torch.from_numpy(q_norm[:, 1] * 255.0).to(device)  # y
    return rgbs, q


def run_true_streaming(model: CoTrackerOnlinePredictor, video: torch.Tensor, queries_xy: torch.Tensor, device: str):
    B, T, C, H, W = video.shape
    model(video_chunk=video, is_first_step=True, queries=queries_xy, add_support_grid=False, grid_size=0)
    autocast_enabled = device.startswith('cuda')
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=autocast_enabled):
        for ind in range(0, T - model.step, model.step):
            chunk = video[:, ind: ind + model.step * 2]
            model(video_chunk=chunk, is_first_step=False, add_support_grid=False, grid_size=0)
    return model


def make_record_from_state(base_record: dict, model: CoTrackerOnlinePredictor, name: str):
    T = int(npy(base_record['gt_tracks']).shape[1])
    N = int(npy(base_record['gt_tracks']).shape[0])
    coords_xy = model.model.online_coords_predicted[0, :T, :N].detach().float().cpu().numpy()
    raw_v_tn = model.model.online_vis_predicted[0, :T, :N].detach().float().cpu()
    raw_c_tn = model.model.online_conf_predicted[0, :T, :N].detach().float().cpu()
    vis_prob_tn = torch.sigmoid(raw_v_tn)
    conf_prob_tn = torch.sigmoid(raw_c_tn)
    score_tn = (vis_prob_tn * conf_prob_tn).numpy()
    pred_vis_tn = score_tn > 0.6
    interp_h, interp_w = model.interp_shape
    coords_xy[..., 0] *= 255.0 / max(float(interp_w - 1), 1.0)
    coords_xy[..., 1] *= 255.0 / max(float(interp_h - 1), 1.0)
    pred_yx_nt = coords_xy.transpose(1, 0, 2)[..., [1, 0]].astype(np.float32) / 255.0
    rec = dict(base_record)
    rec['pred_tracks'] = pred_yx_nt.astype(np.float32)
    rec['pred_visibility'] = pred_vis_tn.transpose(1, 0).astype(bool)
    rec['pred_vis_score'] = score_tn.transpose(1, 0).astype(np.float32)
    rec['pred_raw_vis_logit'] = raw_v_tn.numpy().transpose(1, 0).astype(np.float32)
    rec['pred_raw_conf_logit'] = raw_c_tn.numpy().transpose(1, 0).astype(np.float32)
    rec['pred_vis_prob_component'] = vis_prob_tn.numpy().transpose(1, 0).astype(np.float32)
    rec['pred_conf_prob_component'] = conf_prob_tn.numpy().transpose(1, 0).astype(np.float32)
    rec['model_name'] = name
    rec['adapter_version'] = 'cotracker3_true_streaming_raw_visconf_v1'
    rec['raw_coordinate_note'] = 'True CoTrackerOnlinePredictor streaming loop with raw visibility/confidence components exported; normalized yx output.'
    return rec


def parity_info(new_rec: dict, base_rec: dict) -> dict:
    new_score = npy(new_rec['pred_vis_score'], np.float32)
    base_score = npy(base_rec.get('pred_vis_score', np.zeros_like(new_score)), np.float32)
    new_vis = npy(new_rec['pred_visibility'], bool)
    base_vis = npy(base_rec['pred_visibility'], bool)
    new_tracks = npy(new_rec['pred_tracks'], np.float32)
    base_tracks = npy(base_rec['pred_tracks'], np.float32)
    vis_component = npy(new_rec['pred_vis_prob_component'], np.float32)
    conf_component = npy(new_rec['pred_conf_prob_component'], np.float32)
    product = vis_component * conf_component
    return {
        'score_max_abs_diff_vs_base': float(np.max(np.abs(new_score - base_score))) if base_score.shape == new_score.shape else None,
        'score_mean_abs_diff_vs_base': float(np.mean(np.abs(new_score - base_score))) if base_score.shape == new_score.shape else None,
        'score_product_internal_max_abs_diff': float(np.max(np.abs(product - new_score))),
        'visibility_diff_count_vs_base': int(np.sum(new_vis != base_vis)) if base_vis.shape == new_vis.shape else None,
        'visibility_diff_rate_vs_base': float(np.mean(new_vis != base_vis)) if base_vis.shape == new_vis.shape else None,
        'track_max_abs_diff_vs_base': float(np.max(np.abs(new_tracks - base_tracks))) if base_tracks.shape == new_tracks.shape else None,
        'track_mean_abs_diff_vs_base': float(np.mean(np.abs(new_tracks - base_tracks))) if base_tracks.shape == new_tracks.shape else None,
        'vis_prob_min': float(np.min(vis_component)),
        'vis_prob_max': float(np.max(vis_component)),
        'conf_prob_min': float(np.min(conf_component)),
        'conf_prob_max': float(np.max(conf_component)),
        'score_min': float(np.min(new_score)),
        'score_max': float(np.max(new_score)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-cache', default=str(DEFAULT_BASE_CACHE))
    ap.add_argument('--davis-pkl', default=str(DEFAULT_DAVIS))
    ap.add_argument('--checkpoint', default=str(DEFAULT_CKPT))
    ap.add_argument('--out-cache', default=str(DEFAULT_OUT_CACHE))
    ap.add_argument('--out-report', default=str(DEFAULT_OUT_REPORT))
    ap.add_argument('--start-index', type=int, default=0)
    ap.add_argument('--max-videos', type=int, default=1)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    base_cache = Path(args.base_cache)
    out_cache = Path(args.out_cache); out_report = Path(args.out_report)
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    base_payload = torch.load(base_cache, map_location='cpu', weights_only=False)
    base_records = base_payload['records']
    end = len(base_records) if args.max_videos <= 0 else min(len(base_records), args.start_index + args.max_videos)
    selected = list(range(args.start_index, end))
    print({'selected_indices': selected, 'device': device}, flush=True)
    ds = TapVidDataset(str(args.davis_pkl), dataset_type='davis', resize_to=[256, 256], queried_first=True)
    model = CoTrackerOnlinePredictor(checkpoint=str(args.checkpoint)).to(device).eval()
    records = []
    per_video = []
    t_all = time.time()
    for idx in selected:
        base_rec = base_records[idx]
        vid = str(base_rec['video_id'])
        t0 = time.time()
        video, queries_xy = load_video_for_record(base_rec, ds, device)
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        model = run_true_streaming(model, video, queries_xy, device)
        rec = make_record_from_state(base_rec, model, 'cotracker3_true_streaming_v7a4_raw_visconf')
        pinfo = parity_info(rec, base_rec)
        info = {
            'index': int(idx),
            'video_id': vid,
            'frames': int(video.shape[1]),
            'queries': int(queries_xy.shape[1]),
            'sec': round(float(time.time() - t0), 3),
            'visible_rate': float(np.mean(rec['pred_visibility'])),
            'parity': pinfo,
            'peak_mem_mb': round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1) if device.startswith('cuda') else None,
        }
        records.append(rec)
        per_video.append(info)
        print(json.dumps(info, ensure_ascii=False), flush=True)
        del video, queries_xy
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    payload = dict(base_payload)
    payload['model_name'] = 'cotracker3_true_streaming_v7a4_raw_visconf'
    payload['schema_version'] = 'cotracker3_true_streaming_raw_visconf_v1'
    payload['records'] = records
    payload['source_base_cache'] = str(base_cache)
    payload['checkpoint_path'] = str(args.checkpoint)
    payload['protocol'] = 'true_streaming_first_input_raw_visconf'
    torch.save(payload, out_cache)
    report = {
        'script': 'scripts/export_cotracker3_online_v7a4_raw_visconf_components.py',
        'base_cache': str(base_cache),
        'out_cache': str(out_cache),
        'checkpoint': str(args.checkpoint),
        'davis_pkl': str(args.davis_pkl),
        'start_index': int(args.start_index),
        'num_videos': len(records),
        'per_video': per_video,
        'total_sec': round(float(time.time() - t_all), 2),
    }
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
