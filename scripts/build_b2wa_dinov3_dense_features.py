#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ROWS=Path('outputs/paper_discovery_2026-06-27/b2wa_appearance_pilot/rgb_dev10_patch_rows.jsonl')
VIDEO_CACHE=Path('outputs/paper_discovery_2026-06-27/b2wa_video_cache/rgb_dev10')
BASE_CACHE=Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_offline_rgb_stacking_10video.pt')
OVER_CACHE=Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_online_rgb_stacking_10video.pt')
DINO_DIR=Path('/gemini/code/FSPT/third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m')
OUT=Path('outputs/paper_discovery_2026-06-27/b2wa_dinov3_dense_pilot')


def load_rows(p,max_rows):
    rows=[]
    for line in Path(p).open():
        if line.strip(): rows.append(json.loads(line))
    return rows[:max_rows] if max_rows>0 else rows


def npy(x:Any,dtype=None):
    if isinstance(x,torch.Tensor): x=x.detach().cpu().numpy()
    a=np.asarray(x); return a.astype(dtype) if dtype is not None else a


def last_visible_before(v,t):
    for j in range(int(t)-1,-1,-1):
        if bool(v[j]): return int(j)
    return None


def prep(frames:np.ndarray, device):
    x=torch.from_numpy(frames).permute(0,3,1,2).float()
    if float(x.max())>1.5: x=x/255.0
    x=F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False)
    mean=torch.tensor([0.485,0.456,0.406],device=x.device).view(1,3,1,1)
    std=torch.tensor([0.229,0.224,0.225],device=x.device).view(1,3,1,1)
    x=(x-mean)/std
    return x.to(device)


def sample_feat(feat:torch.Tensor, yx_norm) -> np.ndarray:
    # feat C,H,W, yx_norm normalized y,x
    y=float(yx_norm[0]); x=float(yx_norm[1])
    grid=torch.tensor([[[[x*2-1,y*2-1]]]],device=feat.device,dtype=feat.dtype)
    v=F.grid_sample(feat.unsqueeze(0),grid,mode='bilinear',padding_mode='border',align_corners=False)
    v=v[0,:,0,0]
    v=F.normalize(v,dim=0)
    return v.detach().cpu().float().numpy()


def cos(a,b): return float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-9))
def l2(a,b): return float(np.linalg.norm(a-b))

def offset_yx(yx,dy,dx):
    return [min(1,max(0,float(yx[0])+dy)), min(1,max(0,float(yx[1])+dx))]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rows',default=str(ROWS)); ap.add_argument('--max-rows',type=int,default=2000)
    ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--out-dir',default=str(OUT))
    ap.add_argument('--dino-dir',default=str(DINO_DIR))
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=load_rows(args.rows,args.max_rows)
    base=torch.load(BASE_CACHE,map_location='cpu',weights_only=False); over=torch.load(OVER_CACHE,map_location='cpu',weights_only=False)
    base_by={str(r['video_id']):r for r in base['records']}; over_by={str(r['video_id']):r for r in over['records']}
    need:Dict[Tuple[str,int],List[Tuple[int,str,List[float]]]]={}
    row_points=[]
    for i,r in enumerate(rows):
        vid=str(r['video_id']); qi=int(r['query_idx']); t=int(r['trigger_t']); q=int(r['query_t'])
        b=base_by[vid]; o=over_by[vid]
        bt=npy(b['pred_tracks'],np.float32)[qi]; bv=npy(b['pred_visibility'],bool)[qi]; ot=npy(o['pred_tracks'],np.float32)[qi]
        lv=last_visible_before(bv,t)
        pts={'query':(q,bt[q].tolist()), 'last':(lv if lv is not None else q, (bt[lv] if lv is not None else bt[q]).tolist()), 'cand':(t,ot[t].tolist())}
        off=16/255.0
        for name,(dy,dx) in {'neg_u':(-off,0),'neg_d':(off,0),'neg_l':(0,-off),'neg_r':(0,off),'neg_uu':(-2*off,0),'neg_dd':(2*off,0),'neg_ll':(0,-2*off),'neg_rr':(0,2*off)}.items():
            pts[name]=(t,offset_yx(ot[t],dy,dx))
        row_points.append((vid,pts))
        for name,(ft,yx) in pts.items(): need.setdefault((vid,int(ft)),[]).append((i,name,yx))
    from transformers import AutoModel
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=AutoModel.from_pretrained(args.dino_dir,local_files_only=True).eval().to(device)
    videos={}; feat_store=[{} for _ in rows]
    keys=sorted(need.keys())
    print('unique frames',len(keys),'rows',len(rows),'device',device,flush=True)
    num_reg=int(getattr(model.config,'num_register_tokens',4)); patch_size=int(getattr(model.config,'patch_size',16)); image_size=int(getattr(model.config,'image_size',224))
    grid_hw=image_size//patch_size
    for start in range(0,len(keys),args.batch_size):
        batch_keys=keys[start:start+args.batch_size]
        frames=[]
        for vid,ft in batch_keys:
            if vid not in videos: videos[vid]=np.load(VIDEO_CACHE/f'{vid}.npz')['video']
            frames.append(videos[vid][ft])
        x=prep(np.stack(frames),device)
        with torch.no_grad():
            outm=model(pixel_values=x)
            tokens=outm.last_hidden_state[:,1+num_reg:,:]
            feat=tokens.reshape(tokens.shape[0],grid_hw,grid_hw,tokens.shape[-1]).permute(0,3,1,2).contiguous()
        for bi,key in enumerate(batch_keys):
            f=feat[bi]
            for ri,name,yx in need[key]:
                feat_store[ri][name]=sample_feat(f,yx)
        print('encoded frames',min(start+args.batch_size,len(keys)),'/',len(keys),flush=True)
    out_rows=[]
    for i,r in enumerate(rows):
        d=feat_store[i]; q,l,c=d['query'],d['last'],d['cand']; negs=[d[k] for k in ['neg_u','neg_d','neg_l','neg_r','neg_uu','neg_dd','neg_ll','neg_rr']]
        qneg=max(cos(q,n) for n in negs); lneg=max(cos(l,n) for n in negs)
        qavg=(q+l)/2; qavg=qavg/max(np.linalg.norm(qavg),1e-9); aneg=max(cos(qavg,n) for n in negs)
        dense={
            'dinov3_query_cand_cos':cos(q,c),'dinov3_last_cand_cos':cos(l,c),'dinov3_query_last_cos':cos(q,l),
            'dinov3_memavg_cand_cos':cos(qavg,c),
            'dinov3_query_cand_l2':l2(q,c),'dinov3_last_cand_l2':l2(l,c),'dinov3_memavg_cand_l2':l2(qavg,c),
            'dinov3_query_margin':cos(q,c)-qneg,'dinov3_last_margin':cos(l,c)-lneg,'dinov3_memavg_margin':cos(qavg,c)-aneg,
            'dinov3_neg_max_query_cos':qneg,'dinov3_neg_max_last_cos':lneg,'dinov3_neg_max_memavg_cos':aneg,
        }
        r2=dict(r); r2['dinov3_dense_features']=dense; out_rows.append(r2)
    out_json=out/'rgb_dev10_dinov3_dense_rows.jsonl'
    with out_json.open('w') as f:
        for r in out_rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    summ={'n_rows':len(out_rows),'unique_frames':len(keys),'model':str(args.dino_dir),'out_jsonl':str(out_json)}
    (out/'dinov3_dense_feature_summary.json').write_text(json.dumps(summ,indent=2)); print(json.dumps(summ,indent=2),flush=True)
if __name__=='__main__': main()
