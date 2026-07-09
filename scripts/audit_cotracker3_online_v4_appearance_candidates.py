#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

ROOT = Path('/gemini/code/FSPT')
COTRACKER_ROOT = ROOT / 'baselines/cotracker'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(COTRACKER_ROOT))

from cotracker.datasets.tap_vid_datasets import TapVidDataset

DEFAULT_NATIVE = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_true_streaming_v3_decoupled_subset_eval/full30_overlap_soft_tau055_w2_j4_min1_confirm4_p081/cotracker3_true_streaming_native_subset.pt'
DEFAULT_DAVIS = ROOT / 'datasets/tapvid_davis/tapvid_davis.pkl'
DEFAULT_MODEL = ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m'
DEFAULT_OUT = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_v4_appearance_audit/davis_true_streaming_v4_appearance_audit.json'
DEFAULT_NPZ = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_v4_appearance_audit/davis_true_streaming_v4_appearance_candidates.npz'


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def find_visible_segments(vis: np.ndarray) -> list[tuple[int,int]]:
    out=[]; t=0; T=len(vis)
    while t<T:
        if not vis[t]:
            t+=1; continue
        s=t
        while t<T and vis[t]:
            t+=1
        out.append((s,t))
    return out


def norm_yx_to_pixel(yx: np.ndarray, h: int, w: int) -> tuple[float,float]:
    return float(yx[0])*max(h-1,1), float(yx[1])*max(w-1,1)


def crop_square(img: np.ndarray, y: float, x: float, size: int) -> Image.Image:
    half=int(size)//2; cy=int(round(float(y))); cx=int(round(float(x)))
    h,w=img.shape[:2]
    y0,y1=cy-half, cy+half+1; x0,x1=cx-half, cx+half+1
    patch=np.zeros((size,size,3),dtype=np.uint8)
    sy0,sy1=max(0,y0), min(h,y1); sx0,sx1=max(0,x0), min(w,x1)
    if sy1>sy0 and sx1>sx0:
        py0,px0=sy0-y0, sx0-x0
        patch[py0:py0+(sy1-sy0), px0:px0+(sx1-sx0)] = img[sy0:sy1, sx0:sx1]
    return Image.fromarray(patch, mode='RGB')


def manual_preprocess(images: list[Image.Image], image_size: int) -> torch.Tensor:
    mean=np.asarray([0.485,0.456,0.406],np.float32)
    std=np.asarray([0.229,0.224,0.225],np.float32)
    arrs=[]
    for img in images:
        img=img.convert('RGB').resize((int(image_size),int(image_size)), Image.BILINEAR)
        arr=np.asarray(img).astype(np.float32)/255.0
        arr=(arr-mean[None,None,:])/std[None,None,:]
        arrs.append(np.transpose(arr,(2,0,1)))
    return torch.from_numpy(np.stack(arrs,axis=0)).float()


def embed_images(images: list[Image.Image], model: torch.nn.Module, device: str, image_size: int, batch_size: int) -> np.ndarray:
    outs=[]
    for i in range(0,len(images),batch_size):
        batch=manual_preprocess(images[i:i+batch_size], image_size).to(device)
        with torch.no_grad():
            out=model(pixel_values=batch)
        emb=out.last_hidden_state[:,0].float()
        emb=emb/torch.clamp(torch.linalg.vector_norm(emb, dim=1, keepdim=True), min=1e-8)
        outs.append(emb.detach().cpu().numpy().astype(np.float32))
        del batch, out, emb
    return np.concatenate(outs,axis=0) if outs else np.zeros((0,384),np.float32)


def load_video(ds: TapVidDataset, video_id: str) -> np.ndarray:
    idx=ds.video_names.index(str(video_id))
    sample=ds[idx]
    v=sample.video.detach().cpu().numpy()  # T,C,H,W float 0..255
    if v.shape[1] in (1,3):
        v=np.transpose(v,(0,2,3,1))
    return np.clip(v,0,255).astype(np.uint8)


def choose_refs(record: dict, qi: int, t: int, min_score: float=0.80) -> dict:
    pred_vis=npy(record['pred_visibility'], bool)[qi]
    score=npy(record.get('pred_vis_score', pred_vis.astype(np.float32)), np.float32)[qi]
    qpts=npy(record['query_points'], np.float32)
    query_t=int(round(float(qpts[qi,0])))
    # strict reliable visible frames before candidate.
    reliable=np.where(pred_vis[:max(0,t)] & (score[:max(0,t)] >= float(min_score)))[0]
    any_visible=np.where(pred_vis[:max(0,t)])[0]
    last_strict=int(reliable[-1]) if reliable.size else None
    last_any=int(any_visible[-1]) if any_visible.size else None
    if last_strict is None:
        last_strict=last_any if last_any is not None else query_t
    if last_any is None:
        last_any=query_t
    return {'query_t':query_t, 'last_strict_t':int(last_strict), 'last_any_t':int(last_any), 'has_strict_ref': bool(reliable.size), 'has_any_ref': bool(any_visible.size)}


def collect_candidates(payload: dict, *, pre_window:int, confirm_visible_len:int, tau_low:float, trend_min:float, max_jump_px:float, use_overlap:bool=True) -> list[dict]:
    rows=[]
    for ri,r in enumerate(payload['records']):
        vid=str(r['video_id'])
        base_vis=npy(r['pred_visibility'], bool)
        score=npy(r['pred_vis_score'], np.float32)
        coords=npy(r['pred_tracks'], np.float32)
        gt=npy(r['gt_visibility'], bool)
        N,T=base_vis.shape
        for q in range(N):
            for s,e in find_visible_segments(base_vis[q]):
                if s<=0 or base_vis[q,s-1]:
                    continue
                if s+confirm_visible_len > T or not bool(base_vis[q,s:s+confirm_visible_len].all()):
                    continue
                lo=max(0,s-pre_window)
                for t in range(lo,s):
                    if base_vis[q,t]:
                        continue
                    if use_overlap:
                        # True streaming CoTracker3 uses step=8. Candidate must be in a window region
                        # that could be copied into the next call: [ind+8, ind+16). This is equivalent
                        # to t % 8 in [0..7] for some next window, but we approximate using the same
                        # allowed-frame construction from V3 by checking the previous chunk start.
                        # For each t, at least one ind such that t in [ind+8, ind+16). With ind multiple of 8.
                        # This is true for most t>=8; keep t>=8 and t<T.
                        if t < 8:
                            continue
                    jump=float(np.linalg.norm((coords[q,t]-coords[q,s])*255.0))
                    trend=float(score[q,t]-score[q,t-1]) if t>0 else 0.0
                    numeric_pass=bool(score[q,t]>=tau_low and trend>=trend_min and jump<=max_jump_px)
                    refs=choose_refs(r,q,t)
                    rows.append({
                        'record_index':ri,'video_id':vid,'query_idx':int(q),'frame_t':int(t),'reentry_s':int(s),'reentry_e':int(e),
                        'query_t':refs['query_t'],'last_strict_t':refs['last_strict_t'],'last_any_t':refs['last_any_t'],
                        'has_strict_ref':refs['has_strict_ref'],'has_any_ref':refs['has_any_ref'],
                        'score_t':float(score[q,t]),'score_prev':float(score[q,t-1]) if t>0 else None,'score_s':float(score[q,s]),
                        'trend':trend,'jump_to_s':jump,'d_to_s':int(s-t),'native_vis_seg_len':int(e-s),
                        'numeric_pass':numeric_pass,'gt_visible':bool(gt[q,t]),
                    })
    return rows


def make_patches(row: dict, record: dict, video: np.ndarray, crop_size:int) -> list[Image.Image]:
    q=int(row['query_idx']); t=int(row['frame_t'])
    h,w=video.shape[1], video.shape[2]
    coords=npy(record['pred_tracks'], np.float32)
    qpts=npy(record['query_points'], np.float32)
    q_yx=np.asarray([qpts[q,1], qpts[q,2]], np.float32)
    last_any_yx=coords[q,int(row['last_any_t'])]
    last_strict_yx=coords[q,int(row['last_strict_t'])]
    cand_yx=coords[q,t]
    patches=[]
    for frame,yx in [(int(row['query_t']),q_yx),(int(row['last_any_t']),last_any_yx),(int(row['last_strict_t']),last_strict_yx),(t,cand_yx)]:
        frame=max(0,min(video.shape[0]-1,frame))
        y,x=norm_yx_to_pixel(yx,h,w)
        patches.append(crop_square(video[frame],y,x,crop_size))
    return patches


def summarize_feature(values: np.ndarray, labels: np.ndarray) -> dict:
    out={}
    for name,mask in [('pos', labels.astype(bool)), ('neg', ~labels.astype(bool))]:
        v=values[mask]
        if v.size:
            out[name]={'n':int(v.size),'mean':float(v.mean()),'median':float(np.median(v)),'min':float(v.min()),'max':float(v.max()),'p25':float(np.percentile(v,25)),'p75':float(np.percentile(v,75))}
        else:
            out[name]={'n':0}
    return out


def threshold_sweep(scores: np.ndarray, labels: np.ndarray, *, direction:str) -> list[dict]:
    rows=[]
    if scores.size==0:
        return rows
    if direction=='high_is_good':
        qs=np.unique(np.quantile(scores, [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))
        for thr in qs:
            keep=scores>=thr
            if keep.sum()==0: continue
            rows.append({'direction':direction,'threshold':float(thr),'keep':int(keep.sum()),'gt_visible':int((keep&labels).sum()),'gt_occluded':int((keep&~labels).sum()),'precision':float((keep&labels).sum()/max(keep.sum(),1)),'recall_visible':float((keep&labels).sum()/max(labels.sum(),1))})
    else:
        qs=np.unique(np.quantile(scores, [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]))
        for thr in qs:
            keep=scores<=thr
            if keep.sum()==0: continue
            rows.append({'direction':direction,'threshold':float(thr),'keep':int(keep.sum()),'gt_visible':int((keep&labels).sum()),'gt_occluded':int((keep&~labels).sum()),'precision':float((keep&labels).sum()/max(keep.sum(),1)),'recall_visible':float((keep&labels).sum()/max(labels.sum(),1))})
    rows.sort(key=lambda r:(r['precision'], r['gt_visible'], -r['keep']), reverse=True)
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--native-cache', default=str(DEFAULT_NATIVE))
    ap.add_argument('--davis-pkl', default=str(DEFAULT_DAVIS))
    ap.add_argument('--model-dir', default=str(DEFAULT_MODEL))
    ap.add_argument('--out-json', default=str(DEFAULT_OUT))
    ap.add_argument('--out-npz', default=str(DEFAULT_NPZ))
    ap.add_argument('--pre-window', type=int, default=2)
    ap.add_argument('--confirm-visible-len', type=int, default=4)
    ap.add_argument('--tau-low', type=float, default=0.55)
    ap.add_argument('--trend-min', type=float, default=-0.02)
    ap.add_argument('--max-jump-px', type=float, default=4.0)
    ap.add_argument('--crop-size', type=int, default=33)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()

    t0=time.time()
    payload=torch.load(args.native_cache, map_location='cpu', weights_only=False)
    candidates=collect_candidates(payload, pre_window=args.pre_window, confirm_visible_len=args.confirm_visible_len, tau_low=args.tau_low, trend_min=args.trend_min, max_jump_px=args.max_jump_px)
    print({'candidates_total':len(candidates),'numeric_pass':sum(c['numeric_pass'] for c in candidates)}, flush=True)
    ds=TapVidDataset(str(args.davis_pkl), dataset_type='davis', resize_to=[256,256], queried_first=True)
    video_cache={}
    images=[]
    for i,c in enumerate(candidates):
        vid=c['video_id']
        if vid not in video_cache:
            video_cache[vid]=load_video(ds,vid)
        r=payload['records'][c['record_index']]
        images.extend(make_patches(c,r,video_cache[vid],args.crop_size))
        if (i+1)%200==0:
            print({'prepared':i+1,'videos_loaded':len(video_cache)}, flush=True)
    print({'loading_model':args.model_dir,'images':len(images)}, flush=True)
    model=AutoModel.from_pretrained(str(args.model_dir), local_files_only=True).to(args.device).eval()
    emb=embed_images(images, model, args.device, args.image_size, args.batch_size)
    if emb.shape[0] != len(candidates)*4:
        raise RuntimeError(f'emb count mismatch {emb.shape[0]} vs {len(candidates)*4}')
    feats=[]
    for i,c in enumerate(candidates):
        q,last_any,last_strict,cand = emb[4*i], emb[4*i+1], emb[4*i+2], emb[4*i+3]
        q_c=float(np.dot(q,cand)); any_c=float(np.dot(last_any,cand)); strict_c=float(np.dot(last_strict,cand))
        q_any=float(np.dot(q,last_any)); q_strict=float(np.dot(q,last_strict))
        feats.append([q_c, any_c, strict_c, max(q_c,any_c,strict_c), min(q_c,any_c,strict_c), q_any, q_strict, float(np.linalg.norm(last_strict-cand)), float(np.linalg.norm(last_any-cand))])
    X=np.asarray(feats,np.float32)
    labels=np.asarray([c['gt_visible'] for c in candidates], bool)
    numeric=np.asarray([c['numeric_pass'] for c in candidates], bool)
    names=['query_candidate_cosine','last_any_candidate_cosine','last_strict_candidate_cosine','best_ref_candidate_cosine','worst_ref_candidate_cosine','query_last_any_cosine','query_last_strict_cosine','last_strict_candidate_l2','last_any_candidate_l2']

    summaries={}
    for subset_name,mask in [('all', np.ones(len(candidates), bool)), ('numeric_pass', numeric), ('numeric_fail', ~numeric)]:
        summaries[subset_name]={'n':int(mask.sum()),'gt_visible':int((mask&labels).sum()),'gt_occluded':int((mask&~labels).sum()),'precision':float((mask&labels).sum()/max(mask.sum(),1)), 'features':{}, 'sweeps':{}}
        for j,n in enumerate(names):
            vals=X[mask,j]
            labs=labels[mask]
            summaries[subset_name]['features'][n]=summarize_feature(vals,labs)
            direction='low_is_good' if n.endswith('_l2') else 'high_is_good'
            summaries[subset_name]['sweeps'][n]=threshold_sweep(vals,labs,direction=direction)[:10]

    meta_json=np.asarray([json.dumps(c,ensure_ascii=False) for c in candidates], dtype=object)
    out_npz=Path(args.out_npz); out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, X=X, feature_names=np.asarray(names,dtype=object), y_gt_visible=labels.astype(np.float32), numeric_pass=numeric.astype(np.float32), meta_json=meta_json, native_cache=str(args.native_cache))
    report={'script':'scripts/audit_cotracker3_online_v4_appearance_candidates.py','native_cache':str(args.native_cache),'out_npz':str(out_npz),'params':vars(args),'n_candidates':len(candidates),'numeric_pass':int(numeric.sum()),'numeric_pass_precision':float((numeric&labels).sum()/max(numeric.sum(),1)),'summaries':summaries,'sec':round(time.time()-t0,2)}
    out=Path(args.out_json); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'out_json':str(out),'out_npz':str(out_npz),'n_candidates':len(candidates),'numeric_pass':int(numeric.sum()),'numeric_pass_precision':report['numeric_pass_precision'],'sec':report['sec']}, indent=2, ensure_ascii=False), flush=True)

if __name__=='__main__':
    main()
