#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

ROOT=Path('/gemini/code/FSPT')
COTRACKER_ROOT=ROOT/'baselines/cotracker'
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(COTRACKER_ROOT))
from cotracker.datasets.tap_vid_datasets import TapVidDataset

DEFAULT_NATIVE=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_true_streaming_v3_decoupled_subset_eval/full30_overlap_soft_tau055_w2_j4_min1_confirm4_p081/cotracker3_true_streaming_native_subset.pt'
DEFAULT_REPORT=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v5b_hard_candidate_pool_oracle/v5b_hard_candidate_pool_oracle_report.json'
DEFAULT_DAVIS=ROOT/'datasets/tapvid_davis/tapvid_davis.pkl'
DEFAULT_MODEL=ROOT/'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m'
DEFAULT_OUT=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v5c_verifier/cotracker3_v5c_local96_top20_candidate_dataset.npz'

def npy(x:Any,dtype=None)->np.ndarray:
    if isinstance(x,torch.Tensor): x=x.detach().cpu().numpy()
    a=np.asarray(x)
    return a.astype(dtype) if dtype is not None else a

def to_uint8_video(sample_video:torch.Tensor)->np.ndarray:
    v=sample_video.detach().cpu().numpy()
    if v.shape[1] in (1,3): v=np.transpose(v,(0,2,3,1))
    return np.clip(v,0,255).astype(np.uint8)

def crop_square(img:np.ndarray,y:float,x:float,size:int)->Image.Image:
    half=size//2; cy=int(round(float(y))); cx=int(round(float(x))); h,w=img.shape[:2]
    y0,y1=cy-half,cy+half+1; x0,x1=cx-half,cx+half+1
    patch=np.zeros((size,size,3),np.uint8)
    sy0,sy1=max(0,y0),min(h,y1); sx0,sx1=max(0,x0),min(w,x1)
    if sy1>sy0 and sx1>sx0:
        py0,px0=sy0-y0,sx0-x0
        patch[py0:py0+(sy1-sy0),px0:px0+(sx1-sx0)]=img[sy0:sy1,sx0:sx1]
    return Image.fromarray(patch,mode='RGB')

def preprocess(images:list[Image.Image],image_size:int)->torch.Tensor:
    mean=np.asarray([0.485,0.456,0.406],np.float32); std=np.asarray([0.229,0.224,0.225],np.float32)
    arrs=[]
    for im in images:
        im=im.convert('RGB').resize((image_size,image_size),Image.BILINEAR)
        arr=np.asarray(im).astype(np.float32)/255.0
        arr=(arr-mean[None,None,:])/std[None,None,:]
        arrs.append(np.transpose(arr,(2,0,1)))
    return torch.from_numpy(np.stack(arrs,0)).float()

def embed(images:list[Image.Image],model,device:str,image_size:int,batch_size:int)->np.ndarray:
    outs=[]
    for i in range(0,len(images),batch_size):
        batch=preprocess(images[i:i+batch_size],image_size).to(device)
        with torch.no_grad(): out=model(pixel_values=batch)
        emb=out.last_hidden_state[:,0].float()
        emb=emb/torch.clamp(torch.linalg.vector_norm(emb,dim=1,keepdim=True),min=1e-8)
        outs.append(emb.detach().cpu().numpy().astype(np.float32))
        del batch,out,emb
    return np.concatenate(outs,0) if outs else np.zeros((0,384),np.float32)

def norm_yx_to_px(yx,H,W): return float(yx[0])*max(H-1,1), float(yx[1])*max(W-1,1)

def make_local_grid(native_y,native_x,H,W,radius,stride):
    pts=[]
    for dy in range(-radius,radius+1,stride):
        for dx in range(-radius,radius+1,stride):
            y=float(min(max(native_y+dy,0),H-1)); x=float(min(max(native_x+dx,0),W-1))
            pts.append((y,x))
    return list(dict.fromkeys(pts))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--native-cache',default=str(DEFAULT_NATIVE))
    ap.add_argument('--v5b-report',default=str(DEFAULT_REPORT))
    ap.add_argument('--davis-pkl',default=str(DEFAULT_DAVIS))
    ap.add_argument('--model-dir',default=str(DEFAULT_MODEL))
    ap.add_argument('--out-npz',default=str(DEFAULT_OUT))
    ap.add_argument('--mode',default='local96_s8')
    ap.add_argument('--topk',type=int,default=20)
    ap.add_argument('--crop-size',type=int,default=33)
    ap.add_argument('--image-size',type=int,default=224)
    ap.add_argument('--batch-size',type=int,default=96)
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()
    if args.mode!='local96_s8':
        raise ValueError('This dataset builder currently fixes mode=local96_s8 for minimal V5-C audit')
    t0=time.time()
    native=torch.load(args.native_cache,map_location='cpu',weights_only=False)
    records={str(r['video_id']):r for r in native['records']}
    report=json.loads(Path(args.v5b_report).read_text())
    events=[e for e in report['events'] if f'{args.mode}_top{args.topk}_oracle_err_px' in e]
    print({'events':len(events),'mode':args.mode,'topk':args.topk},flush=True)
    ds=TapVidDataset(str(args.davis_pkl),dataset_type='davis',resize_to=[256,256],queried_first=True)
    print({'loading_model':args.model_dir},flush=True)
    model=AutoModel.from_pretrained(str(args.model_dir),local_files_only=True).to(args.device).eval()
    video_cache={}
    rows=[]; X=[]; y_good=[]; y_safe8=[]; y_better2=[]; meta=[]
    feature_names=[
        'dino_score',
        'dino_rank_norm',
        'dino_score_gap_to_top1',
        'dino_score_gap_to_next',
        'candidate_dist_to_native_px',
        'candidate_dist_to_native_norm',
        'candidate_y_norm',
        'candidate_x_norm',
        'native_visible',
        'native_score',
        'support_age_norm',
        'frame_t_norm',
        'reentry_offset_norm',
        'ref_t_norm',
        'query_idx_norm',
    ]
    for ei,e in enumerate(events):
        vid=str(e['video_id']); r=records[vid]
        if vid not in video_cache:
            sample=ds[ds.video_names.index(vid)]
            video_cache[vid]=to_uint8_video(sample.video)
        video=video_cache[vid]; H,W=video.shape[1],video.shape[2]
        pred=npy(r['pred_tracks'],np.float32); gt=npy(r['gt_tracks'],np.float32); score=npy(r.get('pred_vis_score',np.zeros(pred.shape[:2],np.float32)),np.float32); pvis=npy(r['pred_visibility'],bool)
        q=int(e['query_idx']); t=int(e['frame_t']); ref_t=int(e['ref_t']); s=int(e['reentry_s'])
        ref_y,ref_x=norm_yx_to_px(pred[q,ref_t],H,W); native_y,native_x=norm_yx_to_px(pred[q,t],H,W); gt_y,gt_x=norm_yx_to_px(gt[q,t],H,W)
        pts=make_local_grid(native_y,native_x,H,W,96,8)
        imgs=[crop_square(video[ref_t],ref_y,ref_x,args.crop_size)] + [crop_square(video[t],y,x,args.crop_size) for y,x in pts]
        em=embed(imgs,model,args.device,args.image_size,args.batch_size)
        ref=em[0]; cand=em[1:]
        sims=cand@ref
        order=np.argsort(-sims)
        sorted_sims=sims[order]
        # Keep top-k DINO-ranked candidates as verifier candidates.
        for rank_pos,idx in enumerate(order[:args.topk]):
            y,x=pts[int(idx)]
            cand_err=float(math.hypot(y-gt_y,x-gt_x))
            native_err=float(e['native_err_px'])
            cand_dist_native=float(math.hypot(y-native_y,x-native_x))
            next_score=float(sorted_sims[rank_pos+1]) if rank_pos+1<len(sorted_sims) else float(sorted_sims[rank_pos])
            feat=[
                float(sims[idx]),
                float(rank_pos/max(args.topk-1,1)),
                float(sorted_sims[0]-sims[idx]),
                float(sims[idx]-next_score),
                cand_dist_native,
                cand_dist_native/255.0,
                float(y/max(H-1,1)),
                float(x/max(W-1,1)),
                1.0 if bool(pvis[q,t]) else 0.0,
                float(score[q,t]),
                float(t-ref_t)/255.0,
                float(t)/255.0,
                float(t-s)/max(int(report['params'].get('event_horizon',8)),1),
                float(ref_t)/255.0,
                float(q)/max(pred.shape[0]-1,1),
            ]
            good=bool((cand_err+2.0<native_err) and cand_err<=8.0)
            X.append(feat); y_good.append(1.0 if good else 0.0); y_safe8.append(1.0 if cand_err<=8.0 else 0.0); y_better2.append(1.0 if cand_err+2.0<native_err else 0.0)
            meta.append(json.dumps({
                'event_index':ei,'video_id':vid,'query_idx':q,'frame_t':t,'ref_t':ref_t,'reentry_s':s,'rank':rank_pos+1,
                'candidate_yx_px':[y,x],'gt_yx_px':[gt_y,gt_x],'native_yx_px':[native_y,native_x],
                'candidate_err_px':cand_err,'native_err_px':native_err,'candidate_dist_to_native_px':cand_dist_native,
                'native_visible':bool(pvis[q,t]),'native_score':float(score[q,t]),'dino_score':float(sims[idx]),
            },ensure_ascii=False))
        if (ei+1)%10==0:
            print({'processed_events':ei+1,'rows':len(X),'videos_loaded':len(video_cache)},flush=True)
    X=np.asarray(X,np.float32); y_good=np.asarray(y_good,np.float32); y_safe8=np.asarray(y_safe8,np.float32); y_better2=np.asarray(y_better2,np.float32); meta=np.asarray(meta,dtype=object)
    out=Path(args.out_npz); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,X=X,feature_names=np.asarray(feature_names,dtype=object),y_good=y_good,y_safe8=y_safe8,y_better2=y_better2,meta_json=meta,native_cache=str(args.native_cache),v5b_report=str(args.v5b_report),mode=args.mode,topk=args.topk)
    summary={'out_npz':str(out),'n_samples':int(X.shape[0]),'n_events':len(events),'positive_good':int(y_good.sum()),'positive_good_rate':float(y_good.mean()) if X.shape[0] else 0.0,'safe8_rate':float(y_safe8.mean()) if X.shape[0] else 0.0,'better2_rate':float(y_better2.mean()) if X.shape[0] else 0.0,'videos':sorted({json.loads(m)['video_id'] for m in meta.tolist()}),'sec':round(time.time()-t0,2)}
    out.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
