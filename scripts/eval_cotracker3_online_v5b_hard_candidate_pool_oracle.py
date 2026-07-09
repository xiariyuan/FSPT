#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, subprocess, sys, time
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
from datasets.metrics import compute_tapvid_metrics

DEFAULT_CACHE=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_true_streaming_v3_decoupled_subset_eval/full30_overlap_soft_tau055_w2_j4_min1_confirm4_p081/cotracker3_true_streaming_native_subset.pt'
DEFAULT_DAVIS=ROOT/'datasets/tapvid_davis/tapvid_davis.pkl'
DEFAULT_MODEL=ROOT/'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m'
DEFAULT_OUTDIR=ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v5b_hard_candidate_pool_oracle'

def npy(x:Any,dtype=None)->np.ndarray:
    if isinstance(x,torch.Tensor): x=x.detach().cpu().numpy()
    a=np.asarray(x)
    return a.astype(dtype) if dtype is not None else a

def to_uint8_video(sample_video:torch.Tensor)->np.ndarray:
    v=sample_video.detach().cpu().numpy()
    if v.shape[1] in (1,3): v=np.transpose(v,(0,2,3,1))
    return np.clip(v,0,255).astype(np.uint8)

def crop_square(img:np.ndarray,y:float,x:float,size:int)->Image.Image:
    half=size//2; cy=int(round(y)); cx=int(round(x)); h,w=img.shape[:2]
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

def norm_yx_to_px(yx,h,w): return float(yx[0])*max(h-1,1), float(yx[1])*max(w-1,1)
def px_to_norm_yx(y,x,h,w): return np.asarray([float(y)/max(h-1,1),float(x)/max(w-1,1)],np.float32)

def gt_reentry_segments(vis:np.ndarray):
    out=[]; t=0; T=len(vis)
    while t<T:
        if not vis[t]: t+=1; continue
        s=t
        while t<T and vis[t]: t+=1
        if s>0 and not vis[s-1]: out.append((s,t))
    return out

def collect_hard_events(record:dict, *, max_events:int, support_score:float, event_horizon:int, low_score:float, hard_err_px:float):
    pred_vis=npy(record['pred_visibility'],bool); gt_vis=npy(record['gt_visibility'],bool)
    pred=npy(record['pred_tracks'],np.float32); gt=npy(record['gt_tracks'],np.float32)
    score=npy(record.get('pred_vis_score',pred_vis.astype(np.float32)),np.float32)
    N,T=pred_vis.shape; events=[]
    for q in range(N):
        for s,e in gt_reentry_segments(gt_vis[q]):
            support_idx=np.where(pred_vis[q,:s] & (score[q,:s]>=support_score))[0]
            if support_idx.size==0:
                support_idx=np.where(pred_vis[q,:s])[0]
            if support_idx.size==0: continue
            ref_t=int(support_idx[-1])
            for t in range(s,min(e,s+event_horizon)):
                if not gt_vis[q,t]: continue
                err=float(np.linalg.norm((pred[q,t]-gt[q,t])*255.0))
                is_hard=(not bool(pred_vis[q,t])) or float(score[q,t])<low_score or err>=hard_err_px
                if not is_hard: continue
                events.append({'video_id':str(record['video_id']),'query_idx':int(q),'frame_t':int(t),'reentry_s':int(s),'ref_t':ref_t,'native_visible':bool(pred_vis[q,t]),'native_score':float(score[q,t]),'native_err_px':err})
                break
        if len(events)>=max_events: break
    return events[:max_events]

def make_grid(mode:str,native_y:float,native_x:float,H:int,W:int):
    if mode.startswith('local'):
        # local64_s8 / local96_s8
        rest=mode[len('local'):]
        rad=int(rest.split('_s')[0]); stride=int(rest.split('_s')[1])
        pts=[]
        for dy in range(-rad,rad+1,stride):
            for dx in range(-rad,rad+1,stride):
                y=min(max(native_y+dy,0),H-1); x=min(max(native_x+dx,0),W-1)
                pts.append((float(y),float(x)))
        # unique after clipping
        return list(dict.fromkeys(pts))
    if mode.startswith('global_s'):
        stride=int(mode.split('global_s')[1])
        pts=[]
        for y in range(0,H,stride):
            for x in range(0,W,stride):
                pts.append((float(y),float(x)))
        return pts
    raise ValueError(mode)

def eval_records(records):
    vals=[]; n=0
    for r in records:
        pred=torch.from_numpy(npy(r['pred_tracks'],np.float32)); gt=torch.from_numpy(npy(r['gt_tracks'],np.float32))
        pv=torch.from_numpy(npy(r['pred_visibility'],bool)); gv=torch.from_numpy(npy(r['gt_visibility'],bool)); q=torch.from_numpy(npy(r['query_points'],np.float32)); n+=int(q.shape[0])
        m=compute_tapvid_metrics(pred,gt,pv,gv,q,resolution=256,query_mode='first')
        vals.append({'AJ':float(m['AJ'])*100,'OA':float(m['OA'])*100,'delta_avg':float(m['average_pts_within_thresh'])*100,'delta_4px':float(m['pts_within_4'])*100})
    return {k:float(np.mean([v[k] for v in vals])) for k in ['AJ','OA','delta_avg','delta_4px']} | {'n_records':len(vals),'n_queries':n}

def run_ajrd(cache,out_json):
    subprocess.run([sys.executable,str(ROOT/'scripts/eval_aj_rd_from_cache.py'),'--cache-path',str(cache),'--output-json',str(out_json)],cwd=str(ROOT),check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    return json.loads(Path(out_json).read_text())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--native-cache',default=str(DEFAULT_CACHE)); ap.add_argument('--davis-pkl',default=str(DEFAULT_DAVIS)); ap.add_argument('--model-dir',default=str(DEFAULT_MODEL)); ap.add_argument('--outdir',default=str(DEFAULT_OUTDIR))
    ap.add_argument('--max-videos',type=int,default=10); ap.add_argument('--max-events-per-video',type=int,default=12); ap.add_argument('--modes',default='local64_s8,local96_s8,global_s16'); ap.add_argument('--topk-list',default='1,5,10,20')
    ap.add_argument('--crop-size',type=int,default=33); ap.add_argument('--image-size',type=int,default=224); ap.add_argument('--batch-size',type=int,default=96); ap.add_argument('--support-score',type=float,default=0.80); ap.add_argument('--event-horizon',type=int,default=8); ap.add_argument('--low-score',type=float,default=0.60); ap.add_argument('--hard-err-px',type=float,default=8.0); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args(); outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    modes=[m.strip() for m in args.modes.split(',') if m.strip()]
    topks=[int(x) for x in args.topk_list.split(',') if x.strip()]
    t0=time.time(); payload=torch.load(args.native_cache,map_location='cpu',weights_only=False); records=payload['records'][:args.max_videos]
    ds=TapVidDataset(str(args.davis_pkl),dataset_type='davis',resize_to=[256,256],queried_first=True)
    print({'loading_model':args.model_dir,'modes':modes,'topks':topks},flush=True)
    model=AutoModel.from_pretrained(str(args.model_dir),local_files_only=True).to(args.device).eval()
    # output record copies for each mode/topk
    record_sets={}
    for mode in modes:
        for k in topks:
            key=f'{mode}_top{k}_oracle'
            record_sets[key]=[dict(r) for r in records]
            for rr in record_sets[key]:
                rr['pred_tracks']=npy(rr['pred_tracks'],np.float32).copy(); rr['pred_visibility']=npy(rr['pred_visibility'],bool).copy()
    all_events=[]
    for rec_i,record in enumerate(records):
        vid=str(record['video_id']); sample=ds[ds.video_names.index(vid)]; video=to_uint8_video(sample.video); H,W=video.shape[1],video.shape[2]
        events=collect_hard_events(record,max_events=args.max_events_per_video,support_score=args.support_score,event_horizon=args.event_horizon,low_score=args.low_score,hard_err_px=args.hard_err_px)
        print({'video':vid,'hard_events':len(events)},flush=True)
        pred=npy(record['pred_tracks'],np.float32); gt=npy(record['gt_tracks'],np.float32)
        for ev in events:
            q=ev['query_idx']; t=ev['frame_t']; ref_t=ev['ref_t']
            ref_y,ref_x=norm_yx_to_px(pred[q,ref_t],H,W); native_y,native_x=norm_yx_to_px(pred[q,t],H,W); gt_y,gt_x=norm_yx_to_px(gt[q,t],H,W)
            ev_out=dict(ev); ev_out['gt_yx_px']=[gt_y,gt_x]; ev_out['native_yx_px']=[native_y,native_x]
            # support embedding once
            ref_emb=embed([crop_square(video[ref_t],ref_y,ref_x,args.crop_size)],model,args.device,args.image_size,args.batch_size)[0]
            for mode in modes:
                pts=make_grid(mode,native_y,native_x,H,W)
                imgs=[crop_square(video[t],y,x,args.crop_size) for y,x in pts]
                em=embed(imgs,model,args.device,args.image_size,args.batch_size)
                scores=em@ref_emb
                order=np.argsort(-scores)
                dists=np.asarray([math.hypot(pts[i][0]-gt_y,pts[i][1]-gt_x) for i in order],np.float32)
                ev_out[f'{mode}_n_candidates']=len(pts)
                for k in topks:
                    kk=min(k,len(order)); best_local=order[:kk][int(np.argmin(dists[:kk]))]
                    y,x=pts[best_local]
                    err=float(math.hypot(y-gt_y,x-gt_x))
                    ev_out[f'{mode}_top{k}_oracle_err_px']=err
                    ev_out[f'{mode}_top{k}_dino_top_err_px']=float(dists[0])
                    rs=record_sets[f'{mode}_top{k}_oracle']
                    rs[rec_i]['pred_tracks'][q,t]=px_to_norm_yx(y,x,H,W); rs[rec_i]['pred_visibility'][q,t]=True
            all_events.append(ev_out)
    def save_eval(name,records_eval):
        out_payload=dict(payload); out_payload['records']=records_eval; out_payload['model_name']=name
        cache=outdir/f'{name}.pt'; torch.save(out_payload,cache)
        std=eval_records(records_eval); aj=run_ajrd(cache,outdir/f'{name}_ajrd.json')
        return {'variant':name,'cache':str(cache),**std,'AJ_RD':aj['true_AJ_RD'],'AJ_RD_256':aj['true_AJ_RD_256']}
    native_row=save_eval('native_ref',records)
    rows=[native_row]
    for name,rs in record_sets.items(): rows.append(save_eval(name,rs))
    for r in rows:
        r['delta_vs_native']={k:(r[k]-native_row[k] if k in r and k in native_row else None) for k in ['AJ','OA','delta_avg','delta_4px','AJ_RD','AJ_RD_256']}
    recall={}
    if all_events:
        keys=['native_err_px']+[f'{m}_top{k}_oracle_err_px' for m in modes for k in topks]+[f'{m}_top{k}_dino_top_err_px' for m in modes for k in topks]
        for key in keys:
            if key not in all_events[0]: continue
            arr=np.asarray([e[key] for e in all_events],np.float32)
            recall[key]={'n':int(arr.size),'median':float(np.median(arr)),'mean':float(arr.mean()),'r4':float((arr<=4).mean()),'r8':float((arr<=8).mean()),'r16':float((arr<=16).mean())}
    report={'script':'scripts/eval_cotracker3_online_v5b_hard_candidate_pool_oracle.py','params':vars(args),'n_events':len(all_events),'recall':recall,'rows':rows,'events':all_events,'sec':round(time.time()-t0,2)}
    (outdir/'v5b_hard_candidate_pool_oracle_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'n_events':len(all_events),'recall':recall,'rows':rows,'sec':report['sec']},indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
