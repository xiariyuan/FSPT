#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, subprocess, sys, tempfile
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
from datasets.metrics import compute_tapvid_metrics

DEFAULT_BASE_CACHE = ROOT / 'outputs/attempt0_2026-06-15_recovery/prediction_caches/cotracker3_baseline_davis_first_input_bridge.pt'
DEFAULT_DAVIS = ROOT / 'datasets/tapvid_davis/tapvid_davis.pkl'
DEFAULT_CKPT = ROOT / 'baselines/cotracker/checkpoints/scaled_online.pth'
DEFAULT_OUTDIR = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_state_reentry_smoke'


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def find_visible_segments(vis_1d: np.ndarray):
    T = len(vis_1d); segs=[]; t=0
    while t < T:
        if not vis_1d[t]:
            t += 1; continue
        s=t
        while t < T and vis_1d[t]:
            t += 1
        segs.append((s,t))
    return segs


def delayed_recovery_mask(score_nt: np.ndarray, coords_yx_nt: np.ndarray, base_vis_nt: np.ndarray, *, tau_low: float, pre_window: int, max_jump_px: float, min_len: int):
    """Recover invisible suffix immediately before native visible re-entry segments.

    Inputs are N,T arrays. coords_yx_nt are normalized yx in input 256 space.
    """
    N,T = base_vis_nt.shape
    open_mask = np.zeros_like(base_vis_nt, dtype=bool)
    stats = {'candidate_frames':0,'opened_frames':0,'rejected_jump':0,'rejected_short':0}
    for q in range(N):
        for s,e in find_visible_segments(base_vis_nt[q]):
            if s <= 0 or base_vis_nt[q, s-1]:
                continue
            lo = max(0, s - int(pre_window))
            cand=[]
            for t in range(lo, s):
                if base_vis_nt[q,t]:
                    continue
                stats['candidate_frames'] += 1
                if score_nt[q,t] < tau_low:
                    continue
                if math.isfinite(max_jump_px):
                    jump = float(np.linalg.norm((coords_yx_nt[q,t] - coords_yx_nt[q,s]) * 255.0))
                    if jump > max_jump_px:
                        stats['rejected_jump'] += 1
                        continue
                cand.append(t)
            if not cand:
                continue
            candset=set(cand); suffix=[]; t=s-1
            while t in candset:
                suffix.append(t); t -= 1
            suffix=list(reversed(suffix))
            if len(suffix) < int(min_len):
                stats['rejected_short'] += len(cand)
                continue
            open_mask[q, suffix] = True
            stats['opened_frames'] += len(suffix)
    return open_mask, stats


def make_record_from_state(base_record: dict, model: CoTrackerOnlinePredictor, *, pred_vis_nt: np.ndarray | None = None, score_nt: np.ndarray | None = None, name: str):
    # model.model stores coords in model-resolution xy pixels, B,T,N,2.
    coords_xy = model.model.online_coords_predicted[0].detach().float().cpu().numpy()
    T = int(base_record['gt_tracks'].shape[1])
    N = int(base_record['gt_tracks'].shape[0])
    coords_xy = coords_xy[:T, :N]
    # Scale from model interpolation resolution to 256 input resolution.
    interp_h, interp_w = model.interp_shape
    coords_xy[..., 0] *= 255.0 / max(float(interp_w - 1), 1.0)
    coords_xy[..., 1] *= 255.0 / max(float(interp_h - 1), 1.0)
    pred_yx_nt = coords_xy.transpose(1,0,2)[..., [1,0]].astype(np.float32) / 255.0
    if pred_vis_nt is None:
        raw_v = model.model.online_vis_predicted[0, :T, :N].detach().float().cpu()
        raw_c = model.model.online_conf_predicted[0, :T, :N].detach().float().cpu()
        sc = (torch.sigmoid(raw_v) * torch.sigmoid(raw_c)).numpy().T
        pred_vis_nt = sc > 0.6
        score_nt = sc
    rec = dict(base_record)
    rec['pred_tracks'] = pred_yx_nt.astype(np.float32)
    rec['pred_visibility'] = pred_vis_nt.astype(bool)
    if score_nt is not None:
        rec['pred_vis_score'] = score_nt.astype(np.float32)
    rec['model_name'] = name
    rec['adapter_version'] = name
    rec['raw_coordinate_note'] = 'CoTracker3 online state smoke; coords read from online state, scaled to 256, converted to normalized yx.'
    return rec


def eval_one_record(rec: dict, cache_name: str, outdir: Path):
    pred = torch.from_numpy(npy(rec['pred_tracks'], np.float32))
    gt = torch.from_numpy(npy(rec['gt_tracks'], np.float32))
    pv = torch.from_numpy(npy(rec['pred_visibility'], bool))
    gv = torch.from_numpy(npy(rec['gt_visibility'], bool))
    q = torch.from_numpy(npy(rec['query_points'], np.float32))
    m = compute_tapvid_metrics(pred, gt, pv, gv, q, resolution=256, query_mode='first')
    std = {k: float(m.get(k, 0.0)) for k in ['AJ','OA','average_pts_within_thresh','pts_within_4','average_jaccard','occlusion_accuracy']}
    payload = {'schema_version':1,'model_name':cache_name,'dataset_name':'tapvid_davis','split':'test','protocol':'first_input_1video_smoke','records':[rec]}
    tmp = outdir / f'{cache_name}.pt'
    torch.save(payload, tmp)
    ajrd_json = outdir / f'{cache_name}_ajrd.json'
    subprocess.run([sys.executable, str(ROOT/'scripts/eval_aj_rd_from_cache.py'), '--cache-path', str(tmp), '--output-json', str(ajrd_json)], cwd=str(ROOT), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ajrd = json.loads(ajrd_json.read_text())
    return {
        'cache': str(tmp),
        'AJ': std['AJ'] * 100.0,
        'OA': std['OA'] * 100.0,
        'delta_avg': std['average_pts_within_thresh'] * 100.0,
        'delta_4px': std['pts_within_4'] * 100.0,
        'AJ_RD': ajrd.get('true_AJ_RD'),
        'AJ_RD_256': ajrd.get('true_AJ_RD_256'),
        'first_reentry_frame_proxy': ajrd.get('first_reentry_frame_proxy'),
    }


def load_video_for_record(record: dict, davis_pkl: Path):
    ds = TapVidDataset(str(davis_pkl), dataset_type='davis', resize_to=[256,256], queried_first=True)
    vid = str(record['video_id'])
    idx = ds.video_names.index(vid)
    sample = ds[idx]
    rgbs = sample.video.unsqueeze(0).cuda().float()  # B,T,C,H,W, 0..255
    q_norm = npy(record['query_points'], np.float32)
    q = torch.zeros((1, q_norm.shape[0], 3), device='cuda', dtype=torch.float32)
    q[0,:,0] = torch.from_numpy(q_norm[:,0]).cuda()
    q[0,:,1] = torch.from_numpy(q_norm[:,2] * 255.0).cuda()  # x
    q[0,:,2] = torch.from_numpy(q_norm[:,1] * 255.0).cuda()  # y
    return rgbs, q


def run_variant(base_record: dict, video: torch.Tensor, queries_xy: torch.Tensor, *, checkpoint: Path, variant: str, params: dict):
    model = CoTrackerOnlinePredictor(checkpoint=str(checkpoint)).cuda().eval()
    model(video_chunk=video, is_first_step=True, queries=queries_xy, add_support_grid=False, grid_size=0)
    T = int(video.shape[1]); N = int(queries_xy.shape[1])
    state_stats_total = {'candidate_frames':0,'opened_frames':0,'rejected_jump':0,'rejected_short':0}
    high_logit = float(logit(params.get('state_prob', 0.97)))
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        for ind in range(0, T - model.step, model.step):
            chunk = video[:, ind: ind + model.step * 2]
            model(video_chunk=chunk, is_first_step=False, add_support_grid=False, grid_size=0)
            # State-level writeback after each window/chunk.
            if variant == 'state_writeback':
                raw_v = model.model.online_vis_predicted[0].detach().float()
                raw_c = model.model.online_conf_predicted[0].detach().float()
                cur_T = min(int(raw_v.shape[0]), T)
                cur_N = min(int(raw_v.shape[1]), N)
                score_tn = (torch.sigmoid(raw_v[:cur_T,:cur_N]) * torch.sigmoid(raw_c[:cur_T,:cur_N])).cpu().numpy()
                # coords state in model resolution -> normalized yx input
                coords_xy = model.model.online_coords_predicted[0, :cur_T, :cur_N].detach().float().cpu().numpy()
                interp_h, interp_w = model.interp_shape
                coords_xy[...,0] *= 255.0 / max(float(interp_w - 1), 1.0)
                coords_xy[...,1] *= 255.0 / max(float(interp_h - 1), 1.0)
                coords_yx_nt = coords_xy.transpose(1,0,2)[..., [1,0]].astype(np.float32) / 255.0
                score_nt = score_tn.T
                base_vis_nt = score_nt > 0.6
                open_mask_nt, st = delayed_recovery_mask(score_nt, coords_yx_nt, base_vis_nt, tau_low=params['tau_low'], pre_window=params['pre_window'], max_jump_px=params['max_jump_px'], min_len=params['min_len'])
                # Only write newly opened frames into raw state.
                q_idx, t_idx = np.where(open_mask_nt)
                if len(q_idx) > 0:
                    model.model.online_vis_predicted[0, torch.from_numpy(t_idx).cuda(), torch.from_numpy(q_idx).cuda()] = high_logit
                    model.model.online_conf_predicted[0, torch.from_numpy(t_idx).cuda(), torch.from_numpy(q_idx).cuda()] = high_logit
                for k in state_stats_total:
                    state_stats_total[k] += int(st.get(k,0))
    # Final score and output-level optional correction.
    raw_v = model.model.online_vis_predicted[0, :T, :N].detach().float().cpu()
    raw_c = model.model.online_conf_predicted[0, :T, :N].detach().float().cpu()
    score_nt = (torch.sigmoid(raw_v) * torch.sigmoid(raw_c)).numpy().T
    pred_vis_nt = score_nt > 0.6
    output_stats = None
    if variant == 'output_only':
        rec_tmp = make_record_from_state(base_record, model, name='tmp')
        open_mask_nt, output_stats = delayed_recovery_mask(score_nt, rec_tmp['pred_tracks'], pred_vis_nt, tau_low=params['tau_low'], pre_window=params['pre_window'], max_jump_px=params['max_jump_px'], min_len=params['min_len'])
        pred_vis_nt = pred_vis_nt.copy()
        pred_vis_nt[open_mask_nt] = True
    rec = make_record_from_state(base_record, model, pred_vis_nt=pred_vis_nt, score_nt=score_nt, name=f'cotracker3_online_{variant}')
    stats = {'state_stats': state_stats_total, 'output_stats': output_stats or {}, 'visible_rate': float(pred_vis_nt.mean())}
    return rec, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-cache', default=str(DEFAULT_BASE_CACHE))
    ap.add_argument('--davis-pkl', default=str(DEFAULT_DAVIS))
    ap.add_argument('--checkpoint', default=str(DEFAULT_CKPT))
    ap.add_argument('--outdir', default=str(DEFAULT_OUTDIR))
    ap.add_argument('--video-index', type=int, default=0)
    ap.add_argument('--tau-low', type=float, default=0.40)
    ap.add_argument('--pre-window', type=int, default=4)
    ap.add_argument('--max-jump-px', type=float, default=24.0)
    ap.add_argument('--min-len', type=int, default=1)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for CoTracker3 online smoke')
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    base_payload = torch.load(args.base_cache, map_location='cpu', weights_only=False)
    base_record = dict(base_payload['records'][args.video_index])
    video, queries_xy = load_video_for_record(base_record, Path(args.davis_pkl))
    params = {'tau_low': args.tau_low, 'pre_window': args.pre_window, 'max_jump_px': args.max_jump_px, 'min_len': args.min_len, 'state_prob': 0.97}
    # Existing cache record metric for exact historical baseline.
    existing_metric = eval_one_record(base_record, 'existing_cotracker3_online_cache_1video', outdir)
    variants = {}
    for variant in ['native_rerun','output_only','state_writeback']:
        vname = 'native' if variant == 'native_rerun' else variant
        rec, stats = run_variant(base_record, video, queries_xy, checkpoint=Path(args.checkpoint), variant=('native' if variant=='native_rerun' else variant), params=params)
        metric = eval_one_record(rec, f'cotracker3_online_{vname}_1video', outdir)
        # parity/diff to existing cache record
        tr_diff = np.abs(npy(rec['pred_tracks'], np.float32) - npy(base_record['pred_tracks'], np.float32))
        vis_diff = npy(rec['pred_visibility'], bool) != npy(base_record['pred_visibility'], bool)
        variants[vname] = {'metric': metric, 'stats': stats, 'diff_vs_existing_cache': {'track_max_abs_norm': float(tr_diff.max()), 'track_mean_abs_norm': float(tr_diff.mean()), 'vis_diff_count': int(vis_diff.sum()), 'vis_diff_rate': float(vis_diff.mean())}}
        print(json.dumps({'variant': vname, **variants[vname]}, indent=2, ensure_ascii=False), flush=True)
    native = variants['native']['metric']
    for k in ['output_only','state_writeback']:
        m = variants[k]['metric']
        variants[k]['delta_vs_native_rerun'] = {kk: (m[kk] - native[kk] if isinstance(m.get(kk), (int,float)) and isinstance(native.get(kk), (int,float)) else None) for kk in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256','first_reentry_frame_proxy']}
    report = {'video_index': args.video_index, 'video_id': str(base_record['video_id']), 'params': params, 'existing_metric': existing_metric, 'variants': variants, 'note': '1-video smoke only. Native rerun should be close to existing cache before interpreting output/state variants.'}
    out_report = outdir / 'cotracker3_online_state_reentry_smoke_report.json'
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({'out_report': str(out_report), 'video_id': report['video_id'], 'summary': report}, indent=2, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
