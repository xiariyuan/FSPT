#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from PIL import Image

ROWS = Path('outputs/paper_discovery_2026-06-27/b2wa_appearance_pilot/rgb_dev10_patch_rows.jsonl')
VIDEO_CACHE = Path('outputs/paper_discovery_2026-06-27/b2wa_video_cache/rgb_dev10')
BASE_CACHE = Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_offline_rgb_stacking_10video.pt')
OVER_CACHE = Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_online_rgb_stacking_10video.pt')
OUT = Path('outputs/paper_discovery_2026-06-27/b2wa_clip_pilot')


def load_rows(path: Path, max_rows: int) -> List[Dict[str, Any]]:
    rows=[]
    with path.open() as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows[:max_rows] if max_rows > 0 else rows

def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor): x=x.detach().cpu().numpy()
    a=np.asarray(x); return a.astype(dtype) if dtype is not None else a

def yx_to_px(yx, h, w):
    y=int(round(float(yx[0])*(h-1))); x=int(round(float(yx[1])*(w-1)))
    return max(0,min(h-1,y)), max(0,min(w-1,x))

def crop(frame: np.ndarray, yx, size: int) -> Image.Image:
    h,w=frame.shape[:2]
    y,x=yx_to_px(yx,h,w)
    r=size//2
    pad=r+2
    fr=np.pad(frame, ((pad,pad),(pad,pad),(0,0)), mode='edge')
    yp, xp = y+pad, x+pad
    c=fr[yp-r:yp+r, xp-r:xp+r]
    if c.shape[0] != size or c.shape[1] != size:
        c=np.asarray(Image.fromarray(c.astype('uint8')).resize((size,size)))
    return Image.fromarray(c.astype('uint8'))

def last_visible_before(v, t):
    for j in range(int(t)-1,-1,-1):
        if bool(v[j]): return int(j)
    return None

def encode_batches(model, preprocess, images: List[Image.Image], device: str, batch_size: int) -> np.ndarray:
    feats=[]
    with torch.no_grad():
        for i in range(0,len(images),batch_size):
            x=torch.stack([preprocess(im) for im in images[i:i+batch_size]]).to(device)
            f=model.encode_image(x)
            f=f/torch.clamp(f.norm(dim=-1, keepdim=True), min=1e-6)
            feats.append(f.detach().cpu().float().numpy())
            if (i//batch_size+1)%10==0:
                print(f'encoded {min(i+batch_size,len(images))}/{len(images)}', flush=True)
    return np.concatenate(feats, axis=0)

def cos(a,b): return float(np.dot(a,b)/(max(np.linalg.norm(a)*np.linalg.norm(b),1e-9)))
def l2(a,b): return float(np.linalg.norm(a-b))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rows', default=str(ROWS))
    ap.add_argument('--video-cache', default=str(VIDEO_CACHE))
    ap.add_argument('--base-cache', default=str(BASE_CACHE))
    ap.add_argument('--override-cache', default=str(OVER_CACHE))
    ap.add_argument('--out-dir', default=str(OUT))
    ap.add_argument('--max-rows', type=int, default=2000)
    ap.add_argument('--crop-size', type=int, default=64)
    ap.add_argument('--batch-size', type=int, default=128)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows=load_rows(Path(args.rows), args.max_rows)
    base=torch.load(args.base_cache,map_location='cpu',weights_only=False)
    over=torch.load(args.override_cache,map_location='cpu',weights_only=False)
    base_by_vid={str(r['video_id']):r for r in base['records']}
    over_by_vid={str(r['video_id']):r for r in over['records']}
    import open_clip
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,_,preprocess=open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai', device=device)
    model.eval()
    videos={}
    images=[]; meta=[]
    for idx,row in enumerate(rows):
        vid=str(row['video_id'])
        if vid not in videos:
            videos[vid]=np.load(Path(args.video_cache)/f'{vid}.npz')['video']
        video=videos[vid]
        b=base_by_vid[vid]; o=over_by_vid[vid]
        qi=int(row['query_idx']); t=int(row['trigger_t']); q=int(row['query_t'])
        bt=npy(b['pred_tracks'],np.float32)[qi]; bv=npy(b['pred_visibility'],bool)[qi]
        ot=npy(o['pred_tracks'],np.float32)[qi]
        lv=last_visible_before(bv,t)
        images.append(crop(video[q], bt[q], args.crop_size)); meta.append((idx,'query'))
        images.append(crop(video[lv], bt[lv], args.crop_size) if lv is not None else crop(video[q], bt[q], args.crop_size)); meta.append((idx,'last'))
        images.append(crop(video[t], ot[t], args.crop_size)); meta.append((idx,'cand'))
        # 4 local negatives around candidate in trigger frame
        h,w=video.shape[1:3]; y,x=yx_to_px(ot[t],h,w); off=max(12,args.crop_size//2)
        for name,dy,dx in [('neg_u',-off,0),('neg_d',off,0),('neg_l',0,-off),('neg_r',0,off)]:
            yy=max(0,min(h-1,y+dy))/(h-1); xx=max(0,min(w-1,x+dx))/(w-1)
            images.append(crop(video[t], [yy,xx], args.crop_size)); meta.append((idx,name))
    print('encoding images',len(images),'device',device, flush=True)
    feats=encode_batches(model,preprocess,images,device,args.batch_size)
    by=[{} for _ in rows]
    for (idx,name),f in zip(meta,feats): by[idx][name]=f
    out_rows=[]
    for i,row in enumerate(rows):
        d=by[i]
        q,l,c=d['query'],d['last'],d['cand']
        negs=[d[k] for k in ['neg_u','neg_d','neg_l','neg_r']]
        qneg=max(cos(q,n) for n in negs); lneg=max(cos(l,n) for n in negs)
        clip={
            'clip_query_cand_cos': cos(q,c),
            'clip_last_cand_cos': cos(l,c),
            'clip_query_last_cos': cos(q,l),
            'clip_query_cand_l2': l2(q,c),
            'clip_last_cand_l2': l2(l,c),
            'clip_query_margin': cos(q,c)-qneg,
            'clip_last_margin': cos(l,c)-lneg,
            'clip_neg_max_query_cos': qneg,
            'clip_neg_max_last_cos': lneg,
        }
        r2=dict(row); r2['clip_features']=clip; out_rows.append(r2)
    out_jsonl=out/'rgb_dev10_clip_patch_rows.jsonl'
    with out_jsonl.open('w') as f:
        for r in out_rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    summary={'n_rows':len(out_rows),'n_images_encoded':len(images),'model':'open_clip ViT-B-16/openai','crop_size':args.crop_size,'out_jsonl':str(out_jsonl)}
    (out/'clip_feature_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2), flush=True)
if __name__=='__main__': main()
