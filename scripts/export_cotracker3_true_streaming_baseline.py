#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
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

DEFAULT_REF_CACHE = ROOT / 'outputs/attempt0_2026-06-15_recovery/prediction_caches/cotracker3_baseline_davis_first_input_bridge.pt'
DEFAULT_DAVIS = ROOT / 'datasets/tapvid_davis/tapvid_davis.pkl'
DEFAULT_CKPT = ROOT / 'baselines/cotracker/checkpoints/scaled_online.pth'
DEFAULT_OUT_CACHE = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_true_streaming/cotracker3_true_streaming_davis_first_input.pt'
DEFAULT_OUT_REPORT = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_true_streaming/cotracker3_true_streaming_davis_first_input_report.json'
DEFAULT_OUT_METRICS = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_true_streaming/cotracker3_true_streaming_davis_first_input_metric_compare.json'


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def eval_metric_compare(ref_cache: Path, new_cache: Path, out_json: Path) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / 'scripts/eval_external_baseline_smoke_cache.py'),
        '--items',
        f'cotracker3_existing_online_arch={ref_cache}',
        f'cotracker3_true_streaming={new_cache}',
        '--max-records',
        '0',
        '--out-json',
        str(out_json),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return json.loads(out_json.read_text())


def load_video_for_record(record: dict, ds: TapVidDataset):
    vid = str(record['video_id'])
    idx = ds.video_names.index(vid)
    sample = ds[idx]
    rgbs = sample.video.unsqueeze(0).cuda().float()
    q_norm = npy(record['query_points'], np.float32)
    q = torch.zeros((1, q_norm.shape[0], 3), device='cuda', dtype=torch.float32)
    q[0, :, 0] = torch.from_numpy(q_norm[:, 0]).cuda()
    q[0, :, 1] = torch.from_numpy(q_norm[:, 2] * 255.0).cuda()  # x
    q[0, :, 2] = torch.from_numpy(q_norm[:, 1] * 255.0).cuda()  # y
    return rgbs, q


def run_true_streaming(model: CoTrackerOnlinePredictor, video: torch.Tensor, queries_xy: torch.Tensor):
    B, T, C, H, W = video.shape
    model(video_chunk=video, is_first_step=True, queries=queries_xy, add_support_grid=False, grid_size=0)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        for ind in range(0, T - model.step, model.step):
            chunk = video[:, ind: ind + model.step * 2]
            model(video_chunk=chunk, is_first_step=False, add_support_grid=False, grid_size=0)
    return model


def make_record_from_state(base_record: dict, model: CoTrackerOnlinePredictor, name: str):
    T = int(npy(base_record['gt_tracks']).shape[1])
    N = int(npy(base_record['gt_tracks']).shape[0])
    coords_xy = model.model.online_coords_predicted[0, :T, :N].detach().float().cpu().numpy()
    raw_v = model.model.online_vis_predicted[0, :T, :N].detach().float().cpu()
    raw_c = model.model.online_conf_predicted[0, :T, :N].detach().float().cpu()
    score_tn = (torch.sigmoid(raw_v) * torch.sigmoid(raw_c)).numpy()
    pred_vis_tn = score_tn > 0.6
    interp_h, interp_w = model.interp_shape
    coords_xy[..., 0] *= 255.0 / max(float(interp_w - 1), 1.0)
    coords_xy[..., 1] *= 255.0 / max(float(interp_h - 1), 1.0)
    pred_yx_nt = coords_xy.transpose(1, 0, 2)[..., [1, 0]].astype(np.float32) / 255.0
    pred_vis_nt = pred_vis_tn.transpose(1, 0).astype(bool)
    score_nt = score_tn.transpose(1, 0).astype(np.float32)
    rec = dict(base_record)
    rec['pred_tracks'] = pred_yx_nt.astype(np.float32)
    rec['pred_visibility'] = pred_vis_nt.astype(bool)
    rec['pred_vis_score'] = score_nt
    rec['model_name'] = name
    rec['adapter_version'] = 'cotracker3_true_streaming_export_v1'
    rec['raw_coordinate_note'] = 'True CoTrackerOnlinePredictor streaming loop; coords read from online state, scaled to 256 input, converted to normalized yx.'
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref-cache', default=str(DEFAULT_REF_CACHE))
    ap.add_argument('--davis-pkl', default=str(DEFAULT_DAVIS))
    ap.add_argument('--checkpoint', default=str(DEFAULT_CKPT))
    ap.add_argument('--out-cache', default=str(DEFAULT_OUT_CACHE))
    ap.add_argument('--out-report', default=str(DEFAULT_OUT_REPORT))
    ap.add_argument('--out-metrics', default=str(DEFAULT_OUT_METRICS))
    ap.add_argument('--max-videos', type=int, default=0)
    ap.add_argument('--start-index', type=int, default=0)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    ref_cache = Path(args.ref_cache)
    out_cache = Path(args.out_cache); out_report = Path(args.out_report); out_metrics = Path(args.out_metrics)
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    ref_payload = torch.load(ref_cache, map_location='cpu', weights_only=False)
    ref_records = ref_payload['records']
    end = len(ref_records) if args.max_videos <= 0 else min(len(ref_records), args.start_index + args.max_videos)
    selected = list(range(args.start_index, end))
    records = []
    per_video = []
    t_all = time.time()
    print('loading DAVIS dataset once', flush=True)
    ds = TapVidDataset(str(args.davis_pkl), dataset_type='davis', resize_to=[256, 256], queried_first=True)
    print('loading CoTrackerOnlinePredictor once', flush=True)
    model = CoTrackerOnlinePredictor(checkpoint=str(args.checkpoint)).cuda().eval()
    for idx in selected:
        base_record = ref_records[idx]
        vid = str(base_record['video_id'])
        t0 = time.time()
        video, queries_xy = load_video_for_record(base_record, ds)
        torch.cuda.reset_peak_memory_stats()
        model = run_true_streaming(model, video, queries_xy)
        rec = make_record_from_state(base_record, model, 'cotracker3_true_streaming_davis_first_input')
        tr_diff = np.abs(npy(rec['pred_tracks'], np.float32) - npy(base_record['pred_tracks'], np.float32))
        vis_diff = npy(rec['pred_visibility'], bool) != npy(base_record['pred_visibility'], bool)
        info = {
            'index': int(idx),
            'video_id': vid,
            'frames': int(video.shape[1]),
            'queries': int(queries_xy.shape[1]),
            'sec': round(float(time.time() - t0), 3),
            'visible_rate': float(rec['pred_visibility'].mean()),
            'diff_vs_existing_online_arch': {
                'track_max_abs_norm': float(tr_diff.max()),
                'track_mean_abs_norm': float(tr_diff.mean()),
                'vis_diff_count': int(vis_diff.sum()),
                'vis_diff_rate': float(vis_diff.mean()),
            },
            'peak_mem_mb': round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
        }
        records.append(rec)
        per_video.append(info)
        print(json.dumps(info, ensure_ascii=False), flush=True)
        del video, queries_xy
        torch.cuda.empty_cache()
    payload = dict(ref_payload)
    payload['model_name'] = 'cotracker3_true_streaming_davis_first_input'
    payload['schema_version'] = 'cotracker3_true_streaming_export_v1'
    payload['records'] = records
    payload['source_ref_cache'] = str(ref_cache)
    payload['checkpoint_path'] = str(args.checkpoint)
    payload['protocol'] = 'true_streaming_first_input'
    torch.save(payload, out_cache)
    metrics = None
    if args.start_index == 0 and len(records) == len(ref_records):
        metrics = eval_metric_compare(ref_cache, out_cache, out_metrics)
    report = {
        'script': 'scripts/export_cotracker3_true_streaming_baseline.py',
        'ref_cache': str(ref_cache),
        'out_cache': str(out_cache),
        'out_metrics': str(out_metrics) if metrics is not None else None,
        'checkpoint': str(args.checkpoint),
        'davis_pkl': str(args.davis_pkl),
        'num_videos': len(records),
        'start_index': int(args.start_index),
        'total_sec': round(float(time.time() - t_all), 2),
        'per_video': per_video,
        'metrics': metrics,
        'note': 'This is true streaming CoTrackerOnlinePredictor export. Compare state-writeback experiments against this, not against the old full-sequence online-architecture cache.'
    }
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({'out_cache': str(out_cache), 'out_report': str(out_report), 'out_metrics': str(out_metrics) if metrics else None, 'num_videos': len(records), 'metrics': metrics}, indent=2, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
