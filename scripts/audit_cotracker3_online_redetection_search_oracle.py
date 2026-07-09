#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

ROOT = Path('/gemini/code/FSPT')
COTRACKER_ROOT = ROOT / 'baselines/cotracker'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(COTRACKER_ROOT))
from cotracker.datasets.tap_vid_datasets import TapVidDataset

DEFAULT_CACHE = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_true_streaming_v3_decoupled_subset_eval/full30_overlap_soft_tau055_w2_j4_min1_confirm4_p081/cotracker3_true_streaming_native_subset.pt'
DEFAULT_DAVIS = ROOT / 'datasets/tapvid_davis/tapvid_davis.pkl'
DEFAULT_MODEL = ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m'
DEFAULT_OUT = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_redetection_search_oracle/smoke3_stride8.json'


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a=np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def norm_yx_to_pixel(yx, h, w):
    return float(yx[0])*max(h-1,1), float(yx[1])*max(w-1,1)


def crop_square(img: np.ndarray, y: float, x: float, size: int) -> Image.Image:
    half=size//2; cy=int(round(y)); cx=int(round(x)); h,w=img.shape[:2]
    y0,y1=cy-half,cy+half+1; x0,x1=cx-half,cx+half+1
    patch=np.zeros((size,size,3),np.uint8)
    sy0,sy1=max(0,y0),min(h,y1); sx0,sx1=max(0,x0),min(w,x1)
    if sy1>sy0 and sx1>sx0:
        py0,px0=sy0-y0,sx0-x0
        patch[py0:py0+(sy1-sy0), px0:px0+(sx1-sx0)] = img[sy0:sy1, sx0:sx1]
    return Image.fromarray(patch, mode='RGB')


def manual_preprocess(images, image_size):
    mean=np.asarray([0.485,0.456,0.406],np.float32); std=np.asarray([0.229,0.224,0.225],np.float32)
    arrs=[]
    for img in images:
        img=img.convert('RGB').resize((image_size,image_size), Image.BILINEAR)
        arr=np.asarray(img).astype(np.float32)/255.0
        arr=(arr-mean[None,None,:])/std[None,None,:]
        arrs.append(np.transpose(arr,(2,0,1)))
    return torch.from_numpy(np.stack(arrs,axis=0)).float()


def embed_images(images, model, device, image_size, batch_size):
    outs=[]
    for i in range(0,len(images),batch_size):
        batch=manual_preprocess(images[i:i+batch_size], image_size).to(device)
        with torch.no_grad():
            out=model(pixel_values=batch)
        emb=out.last_hidden_state[:,0].float()
        emb=emb/torch.clamp(torch.linalg.vector_norm(emb,dim=1,keepdim=True),min=1e-8)
        outs.append(emb.detach().cpu().numpy().astype(np.float32))
        del batch, out, emb
    return np.concatenate(outs,axis=0) if outs else np.zeros((0,384),np.float32)


def to_uint8_video(sample_video):
    v=sample_video.detach().cpu().numpy() if isinstance(sample_video,torch.Tensor) else np.asarray(sample_video)
    if v.ndim!=4: raise ValueError(v.shape)
    if v.shape[1] in (1,3): v=np.transpose(v,(0,2,3,1))
    return np.clip(v,0,255).astype(np.uint8)


def find_gt_reentry_events(gt_vis: np.ndarray):
    events=[]; N,T=gt_vis.shape
    for q in range(N):
        for t in range(1,T):
            if bool(gt_vis[q,t]) and not bool(gt_vis[q,t-1]):
                events.append((q,t))
    return events


def choose_support(record, q, t, min_score=0.80):
    pred_vis=npy(record['pred_visibility'], bool)[q]
    score=npy(record.get('pred_vis_score', pred_vis.astype(np.float32)), np.float32)[q]
    qpts=npy(record['query_points'], np.float32)
    query_t=int(round(float(qpts[q,0])))
    strict=np.where(pred_vis[:t] & (score[:t]>=min_score))[0]
    anyv=np.where(pred_vis[:t])[0]
    last_strict=int(strict[-1]) if strict.size else None
    last_any=int(anyv[-1]) if anyv.size else None
    if last_strict is None: last_strict = last_any if last_any is not None else query_t
    if last_any is None: last_any=query_t
    return query_t, int(last_any), int(last_strict), bool(strict.size)


def make_grid(h,w,stride):
    ys=list(range(stride//2, h, stride)); xs=list(range(stride//2, w, stride))
    pts=np.asarray([(y,x) for y in ys for x in xs], np.float32)
    return pts


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--native-cache', default=str(DEFAULT_CACHE))
    ap.add_argument('--davis-pkl', default=str(DEFAULT_DAVIS))
    ap.add_argument('--model-dir', default=str(DEFAULT_MODEL))
    ap.add_argument('--out-json', default=str(DEFAULT_OUT))
    ap.add_argument('--max-records', type=int, default=3)
    ap.add_argument('--max-events', type=int, default=80)
    ap.add_argument('--grid-stride', type=int, default=8)
    ap.add_argument('--crop-size', type=int, default=33)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()
    t0=time.time()
    payload=torch.load(args.native_cache,map_location='cpu',weights_only=False)
    ds=TapVidDataset(str(args.davis_pkl), dataset_type='davis', resize_to=[256,256], queried_first=True)
    model=AutoModel.from_pretrained(str(args.model_dir), local_files_only=True).to(args.device).eval()
    grid=make_grid(256,256,args.grid_stride)
    rows=[]
    records=payload['records'][:args.max_records if args.max_records>0 else len(payload['records'])]
    total_events=0
    for ri,r in enumerate(records):
        vid=str(r['video_id'])
        sample=ds[ds.video_names.index(vid)]
        video=to_uint8_video(sample.video)
        gt_vis=npy(r['gt_visibility'], bool); gt_tracks=npy(r['gt_tracks'], np.float32)
        pred_tracks=npy(r['pred_tracks'], np.float32); pred_vis=npy(r['pred_visibility'], bool)
        qpts=npy(r['query_points'], np.float32)
        events=find_gt_reentry_events(gt_vis)
        for q,t in events:
            if total_events>=args.max_events: break
            h,w=video.shape[1],video.shape[2]
            gt_y,gt_x=norm_yx_to_pixel(gt_tracks[q,t],h,w)
            native_y,native_x=norm_yx_to_pixel(pred_tracks[q,t],h,w)
            native_err=float(np.linalg.norm(np.asarray([native_y,native_x])-np.asarray([gt_y,gt_x])))
            query_t,last_any,last_strict,has_strict=choose_support(r,q,t)
            q_yx=np.asarray([qpts[q,1], qpts[q,2]], np.float32)
            refs=[]
            for frame,yx in [(query_t,q_yx),(last_any,pred_tracks[q,last_any]),(last_strict,pred_tracks[q,last_strict])]:
                yy,xx=norm_yx_to_pixel(yx,h,w)
                refs.append(crop_square(video[max(0,min(video.shape[0]-1,frame))],yy,xx,args.crop_size))
            cand_imgs=[crop_square(video[t], float(y), float(x), args.crop_size) for y,x in grid]
            emb=embed_images(refs+cand_imgs, model, args.device, args.image_size, args.batch_size)
            ref_emb=emb[:3]; cand_emb=emb[3:]
            sims=cand_emb @ ref_emb.T
            best_sim=sims.max(axis=1)
            order=np.argsort(-best_sim)
            dists=np.linalg.norm(grid - np.asarray([gt_y,gt_x],np.float32)[None,:],axis=1)
            row={
                'record_index':ri,'video_id':vid,'query_idx':int(q),'frame_t':int(t),
                'native_visible':bool(pred_vis[q,t]),'native_err_px':native_err,
                'has_strict_ref':has_strict,'query_t':int(query_t),'last_any_t':int(last_any),'last_strict_t':int(last_strict),
                'nearest_grid_dist_px':float(dists.min()),
                'best_grid_possible_at4':bool(dists.min()<=4),'best_grid_possible_at8':bool(dists.min()<=8),'best_grid_possible_at16':bool(dists.min()<=16),
            }
            for K in [1,5,10,20,50]:
                top=order[:K]
                row[f'top{K}_min_dist_px']=float(dists[top].min())
                row[f'top{K}_recall4']=bool(dists[top].min()<=4)
                row[f'top{K}_recall8']=bool(dists[top].min()<=8)
                row[f'top{K}_recall16']=bool(dists[top].min()<=16)
            # record top1 coord/sim for debugging
            top1=order[0]
            row['top1_yx_px']=[float(grid[top1,0]),float(grid[top1,1])]
            row['top1_sim']=float(best_sim[top1])
            row['gt_yx_px']=[float(gt_y),float(gt_x)]
            rows.append(row); total_events+=1
        if total_events>=args.max_events: break
        print({'processed_record':ri,'video_id':vid,'events_total':total_events}, flush=True)
    # summaries
    def rate(key, subset):
        if not subset: return None
        return float(np.mean([bool(r[key]) for r in subset]))
    subsets={
        'all':rows,
        'native_invisible':[r for r in rows if not r['native_visible']],
        'native_err_gt8':[r for r in rows if r['native_err_px']>8],
        'native_invis_or_err_gt8':[r for r in rows if (not r['native_visible']) or r['native_err_px']>8],
    }
    summary={}
    for name,sub in subsets.items():
        summary[name]={'n':len(sub),'native_visible_rate':float(np.mean([r['native_visible'] for r in sub])) if sub else None,'native_err_mean':float(np.mean([r['native_err_px'] for r in sub])) if sub else None,'nearest_grid_recall8':rate('best_grid_possible_at8',sub)}
        for K in [1,5,10,20,50]:
            summary[name][f'top{K}_recall4']=rate(f'top{K}_recall4',sub)
            summary[name][f'top{K}_recall8']=rate(f'top{K}_recall8',sub)
            summary[name][f'top{K}_recall16']=rate(f'top{K}_recall16',sub)
            summary[name][f'top{K}_min_dist_mean']=float(np.mean([r[f'top{K}_min_dist_px'] for r in sub])) if sub else None
    report={'script':'scripts/audit_cotracker3_online_redetection_search_oracle.py','params':vars(args),'n_events':len(rows),'summary':summary,'rows':rows,'sec':round(time.time()-t0,2)}
    out=Path(args.out_json); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'out_json':str(out),'n_events':len(rows),'summary':summary,'sec':report['sec']},indent=2,ensure_ascii=False), flush=True)

if __name__=='__main__':
    main()
