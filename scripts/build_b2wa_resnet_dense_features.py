#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F

ROWS=Path('outputs/paper_discovery_2026-06-27/b2wa_appearance_pilot/rgb_dev10_patch_rows.jsonl')
VIDEO_CACHE=Path('outputs/paper_discovery_2026-06-27/b2wa_video_cache/rgb_dev10')
BASE_CACHE=Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_offline_rgb_stacking_10video.pt')
OVER_CACHE=Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_online_rgb_stacking_10video.pt')
OUT=Path('outputs/paper_discovery_2026-06-27/b2wa_resnet_dense_pilot')

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
    mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    x=(x-mean)/std
    return x.to(device)

def sample_feat(feat:torch.Tensor, yx_norm) -> np.ndarray:
    # feat: C,H,W. yx_norm in [0,1]. grid_sample expects x,y in [-1,1]
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
    ap.add_argument('--batch-size',type=int,default=64); ap.add_argument('--out-dir',default=str(OUT))
    args=ap.parse_args(); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=load_rows(args.rows,args.max_rows)
    base=torch.load(BASE_CACHE,map_location='cpu',weights_only=False); over=torch.load(OVER_CACHE,map_location='cpu',weights_only=False)
    base_by={str(r['video_id']):r for r in base['records']}; over_by={str(r['video_id']):r for r in over['records']}
    # collect events and per-frame points
    need:Dict[Tuple[str,int],List[Tuple[int,str,List[float]]]]={}
    row_points=[]
    for i,r in enumerate(rows):
        vid=str(r['video_id']); qi=int(r['query_idx']); t=int(r['trigger_t']); q=int(r['query_t'])
        b=base_by[vid]; o=over_by[vid]
        bt=npy(b['pred_tracks'],np.float32)[qi]; bv=npy(b['pred_visibility'],bool)[qi]; ot=npy(o['pred_tracks'],np.float32)[qi]
        lv=last_visible_before(bv,t)
        pts={'query':(q,bt[q].tolist()), 'last':(lv if lv is not None else q, (bt[lv] if lv is not None else bt[q]).tolist()), 'cand':(t,ot[t].tolist())}
        # local negatives around candidate, offsets in normalized coords ~ 16 px / 256
        off=16/255.0
        for name,(dy,dx) in {'neg_u':(-off,0),'neg_d':(off,0),'neg_l':(0,-off),'neg_r':(0,off)}.items():
            pts[name]=(t,offset_yx(ot[t],dy,dx))
        row_points.append((vid,pts))
        for name,(ft,yx) in pts.items(): need.setdefault((vid,int(ft)),[]).append((i,name,yx))
    import timm
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=timm.create_model('resnet50.a1_in1k',pretrained=True,features_only=True,out_indices=(2,3)).eval().to(device)
    videos={}; feat_store=[{} for _ in rows]
    keys=sorted(need.keys())
    print('unique frames',len(keys),'rows',len(rows),'device',device,flush=True)
    for start in range(0,len(keys),args.batch_size):
        batch_keys=keys[start:start+args.batch_size]
        frames=[]
        for vid,ft in batch_keys:
            if vid not in videos: videos[vid]=np.load(VIDEO_CACHE/f'{vid}.npz')['video']
            frames.append(videos[vid][ft])
        x=prep(np.stack(frames),device)
        with torch.no_grad(): feats=model(x)
        # use stride8 and stride16 feature maps, sample and concatenate
        f2=feats[0]; f3=feats[1]
        for bi,key in enumerate(batch_keys):
            for ri,name,yx in need[key]:
                v2=sample_feat(f2[bi],yx); v3=sample_feat(f3[bi],yx)
                feat_store[ri][name]=np.concatenate([v2,v3]).astype(np.float32)
        print('encoded frames',min(start+args.batch_size,len(keys)),'/',len(keys),flush=True)
    out_rows=[]
    for i,r in enumerate(rows):
        d=feat_store[i]; q,l,c=d['query'],d['last'],d['cand']; negs=[d[k] for k in ['neg_u','neg_d','neg_l','neg_r']]
        qneg=max(cos(q,n) for n in negs); lneg=max(cos(l,n) for n in negs)
        dense={'resnet_query_cand_cos':cos(q,c),'resnet_last_cand_cos':cos(l,c),'resnet_query_last_cos':cos(q,l),'resnet_query_cand_l2':l2(q,c),'resnet_last_cand_l2':l2(l,c),'resnet_query_margin':cos(q,c)-qneg,'resnet_last_margin':cos(l,c)-lneg,'resnet_neg_max_query_cos':qneg,'resnet_neg_max_last_cos':lneg}
        r2=dict(r); r2['resnet_dense_features']=dense; out_rows.append(r2)
    out_json=out/'rgb_dev10_resnet_dense_rows.jsonl'
    with out_json.open('w') as f:
        for r in out_rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    summ={'n_rows':len(out_rows),'unique_frames':len(keys),'model':'timm resnet50.a1_in1k features_only out_indices=(2,3)','out_jsonl':str(out_json)}
    (out/'resnet_dense_feature_summary.json').write_text(json.dumps(summ,indent=2)); print(json.dumps(summ,indent=2),flush=True)
if __name__=='__main__': main()
