#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TRACKON_ROOT = PROJECT_ROOT / 'baselines' / 'track_on'
if str(TRACKON_ROOT) not in sys.path:
    sys.path.insert(0, str(TRACKON_ROOT))

from model.trackon_predictor import Predictor as TrackOnPredictor
from utils.train_utils import load_args_from_yaml


def _git_commit() -> str:
    try:
        return subprocess.check_output(['git','rev-parse','HEAD'], cwd=str(PROJECT_ROOT), text=True).strip()
    except Exception:
        return ''


def npy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def make_record(src: Dict[str, Any], pred_tracks_xy_tn: torch.Tensor, pred_vis_tn: torch.Tensor, idx: int) -> Dict[str, Any]:
    tracks_xy_tn = pred_tracks_xy_tn[0].detach().cpu().float().numpy()  # T,N,2 xy pixel
    vis_tn = pred_vis_tn[0].detach().cpu().bool().numpy()  # T,N
    tracks_yx_nt = tracks_xy_tn.transpose(1, 0, 2)[..., [1, 0]].astype(np.float32)
    pred_vis_nt = vis_tn.transpose(1, 0).astype(np.bool_)
    q = npy(src['query_points']).astype(np.float32)
    gt = npy(src['gt_tracks']).astype(np.float32)
    gv = npy(src['gt_visibility']).astype(np.bool_)
    osz = npy(src['original_size']).astype(np.int32).reshape(-1)
    h, w = int(osz[0]), int(osz[1])
    pred_norm = np.empty_like(tracks_yx_nt, dtype=np.float32)
    pred_norm[..., 0] = tracks_yx_nt[..., 0] / max(h - 1, 1)
    pred_norm[..., 1] = tracks_yx_nt[..., 1] / max(w - 1, 1)
    rec = {
        'video_id': str(src.get('video_id', f'stress_{idx:06d}')),
        'source_video_id': str(src.get('source_video_id', '')),
        'sequence_index': int(src.get('sequence_index', idx)),
        'frame_count': int(gt.shape[1]),
        'query_points': q.astype(np.float32),
        'pred_tracks': pred_norm.astype(np.float32),
        'pred_visibility': pred_vis_nt,
        'gt_tracks': gt.astype(np.float32),
        'target_points': gt.astype(np.float32),
        'gt_visibility': gv,
        'occluded': (~gv).astype(np.bool_),
        'original_size': np.asarray([h, w], dtype=np.int32),
        'model_input_size': np.asarray(src.get('model_input_size', [h, w]), dtype=np.int32),
        'stress_type': src.get('stress_type', ''),
        'stress_params': src.get('stress_params', {}),
        'adapter_version': 'trackon2_reentry_stress_v1',
        'raw_coordinate_note': 'TrackOn2 output xy pixel; converted to unified normalized yx by original_size [H-1,W-1]. Queries are [t,y,x] normalized.',
        'model_name': 'trackon2_dinov3_reentry_stress',
    }
    for k in ['source_query_indices','stress_shift_xy_px']:
        if k in src:
            rec[k] = src[k]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stress-dataset', required=True)
    ap.add_argument('--checkpoint', default='baselines/track_on/checkpoints_trackon2_dinov3.pt')
    ap.add_argument('--config', default='baselines/track_on/config/test.yaml')
    ap.add_argument('--support-grid-size', type=int, default=20)
    ap.add_argument('--out-cache', required=True)
    ap.add_argument('--out-report', required=True)
    ap.add_argument('--max-videos', type=int, default=0)
    ap.add_argument('--start-record', type=int, default=0)
    ap.add_argument('--max-queries', type=int, default=0)
    ap.add_argument('--memory-policy', default='unconditional', choices=['unconditional','visibility_selective'])
    ap.add_argument('--delta-v', type=float, default=None, help='Override TrackOn2 visibility threshold; default uses config value.')
    ap.add_argument('--input-scale', default='uint8', choices=['uint8','unit'], help='uint8 passes 0..255 frames; unit passes 0..1 frames.')
    args = ap.parse_args()
    payload = torch.load(args.stress_dataset, map_location='cpu', weights_only=False)
    records_in = payload['records']
    lo = int(args.start_record)
    hi = len(records_in) if args.max_videos <= 0 else min(len(records_in), lo + int(args.max_videos))
    records_in = records_in[lo:hi]
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    print('loading TrackOn2...', flush=True)
    t0=time.time()
    model_args = load_args_from_yaml(args.config)
    model_args.grad_checkpoint = False
    # Optional memory policy; unconditional is official-style, visibility_selective is local fixed code path.
    model_args.memory_update_policy = args.memory_policy
    if args.delta_v is not None:
        model_args.delta_v = float(args.delta_v)
    model = TrackOnPredictor(model_args, checkpoint_path=args.checkpoint, support_grid_size=int(args.support_grid_size)).cuda().eval()
    print({'model_load_sec': round(time.time()-t0,2), 'input_size': list(model.model.input_size), 'memory_policy': args.memory_policy, 'delta_v': float(model_args.delta_v), 'input_scale': args.input_scale}, flush=True)
    records: List[Dict[str, Any]]=[]; per_video=[]
    for idx, src0 in enumerate(records_in):
        src=dict(src0)
        if args.max_queries and args.max_queries>0:
            mq=int(args.max_queries)
            for k in ['query_points','gt_tracks','target_points','gt_visibility','occluded','source_query_indices']:
                if k in src:
                    src[k]=npy(src[k])[:mq]
        vid=str(src.get('video_id', f'stress_{idx:06d}'))
        video_np=npy(src['video']) # T,H,W,3 uint8
        h,w=int(src['original_size'][0]), int(src['original_size'][1])
        video=torch.from_numpy(video_np).permute(0,3,1,2).float().unsqueeze(0).cuda(non_blocking=True)
        if args.input_scale == 'unit':
            video = video / 255.0
        q=torch.from_numpy(npy(src['query_points']).astype(np.float32)).unsqueeze(0).cuda(non_blocking=True)
        queries=torch.zeros_like(q)
        queries[:,:,0]=q[:,:,0]
        queries[:,:,1]=q[:,:,2]*max(w-1,1)
        queries[:,:,2]=q[:,:,1]*max(h-1,1)
        torch.cuda.reset_peak_memory_stats()
        t1=time.time()
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            tracks, vis = model(video, queries)
        torch.cuda.synchronize()
        sec=time.time()-t1
        rec=make_record(src, tracks, vis, idx+lo)
        records.append(rec)
        info={'idx': idx+lo, 'video_id': vid, 'frames': int(video_np.shape[0]), 'queries': int(npy(src['query_points']).shape[0]), 'size_hw':[h,w], 'sec': round(sec,3), 'vis_rate': round(float(rec['pred_visibility'].mean()),4), 'peak_mem_mb': round(torch.cuda.max_memory_allocated()/1024/1024,1)}
        per_video.append(info)
        print(info, flush=True)
        del video,q,queries,tracks,vis
        torch.cuda.empty_cache()
    out={'schema_version':1,'model_name':'trackon2_dinov3_reentry_stress','repo_commit':_git_commit(),'checkpoint_path':args.checkpoint,'config_path':args.config,'source_stress_dataset':args.stress_dataset,'stress_name':payload.get('stress_name'),'stress_type':payload.get('stress_type'),'support_grid_size':int(args.support_grid_size),'memory_policy':args.memory_policy,'delta_v':float(model_args.delta_v),'input_scale':args.input_scale,'records':records}
    out_cache=Path(args.out_cache); out_report=Path(args.out_report)
    out_cache.parent.mkdir(parents=True,exist_ok=True)
    torch.save(out,out_cache)
    report={'out_cache':str(out_cache),'source_stress_dataset':args.stress_dataset,'n_records':len(records),'n_queries':int(sum(r['query_points'].shape[0] for r in records)),'total_sec':round(sum(v['sec'] for v in per_video),3),'mean_vis_rate':round(float(np.mean([v['vis_rate'] for v in per_video])),4) if per_video else None,'per_video':per_video}
    out_report.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print('CACHE_OK', json.dumps(report, ensure_ascii=False), flush=True)
if __name__=='__main__': main()
