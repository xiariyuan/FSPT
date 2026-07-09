#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))

ROWS=Path('outputs/paper_discovery_2026-06-27/b2wv_counterfactual_dev/rgb_dev10_window_rows.jsonl')
VIDEO_CACHE=Path('outputs/paper_discovery_2026-06-27/b2wa_video_cache/rgb_dev10')
BASE_CACHE=Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_offline_rgb_stacking_10video.pt')
OVER_CACHE=Path('outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_online_rgb_stacking_10video.pt')
DINO_DIR=Path('/gemini/code/FSPT/third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m')
OUT=Path('outputs/paper_discovery_2026-06-27/reentry_id_lite_audit')


def npy(x:Any,dtype=None):
    if isinstance(x,torch.Tensor): x=x.detach().cpu().numpy()
    a=np.asarray(x); return a.astype(dtype) if dtype is not None else a

def read_rows(max_rows:int, seed:int=20260702):
    pos=[]
    with ROWS.open() as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            if r.get('eligible_window_has_reentry') and r.get('first_eligible_reentry_t') is not None:
                pos.append(r)
    rng=np.random.default_rng(seed)
    if max_rows>0 and len(pos)>max_rows:
        idx=rng.permutation(len(pos))[:max_rows]
        pos=[pos[i] for i in idx]
    return pos

def last_visible_before(v,t):
    for j in range(int(t)-1,-1,-1):
        if bool(v[j]): return int(j)
    return None

def prep(frames:np.ndarray,device):
    x=torch.from_numpy(frames).permute(0,3,1,2).float()
    if float(x.max())>1.5: x=x/255.0
    x=F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False)
    mean=torch.tensor([0.485,0.456,0.406],device=x.device).view(1,3,1,1)
    std=torch.tensor([0.229,0.224,0.225],device=x.device).view(1,3,1,1)
    return ((x-mean)/std).to(device)

def sample_feat(feat,yx):
    y=float(yx[0]); x=float(yx[1])
    grid=torch.tensor([[[[x*2-1,y*2-1]]]],device=feat.device,dtype=feat.dtype)
    v=F.grid_sample(feat.unsqueeze(0),grid,mode='bilinear',padding_mode='border',align_corners=False)[0,:,0,0]
    v=F.normalize(v,dim=0)
    return v.detach().cpu().float().numpy()

def cos(a,b): return float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-9))
def offset_yx(yx,dy,dx): return [min(1,max(0,float(yx[0])+dy)), min(1,max(0,float(yx[1])+dx))]

def auc_from_scores(pos_scores, neg_scores):
    y=np.asarray([1]*len(pos_scores)+[0]*len(neg_scores))
    s=np.asarray(pos_scores+neg_scores,dtype=np.float32)
    if len(np.unique(y))<2: return None
    return float(roc_auc_score(y,s))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--max-rows',type=int,default=1000)
    ap.add_argument('--batch-size',type=int,default=48)
    args=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    rows=read_rows(args.max_rows)
    base=torch.load(BASE_CACHE,map_location='cpu',weights_only=False); over=torch.load(OVER_CACHE,map_location='cpu',weights_only=False)
    base_by={str(r['video_id']):r for r in base['records']}; over_by={str(r['video_id']):r for r in over['records']}
    # points: query, last, gt_reentry, online_at_reentry, online_trigger, local/global negatives at reentry
    need:Dict[Tuple[str,int],List[Tuple[int,str,List[float]]]]={}
    rng=np.random.default_rng(20260702)
    for i,r in enumerate(rows):
        vid=str(r['video_id']); qi=int(r['query_idx']); q=int(r['query_t']); tr=int(r['trigger_t']); rt=int(r['first_eligible_reentry_t'])
        b=base_by[vid]; o=over_by[vid]
        bt=npy(b['pred_tracks'],np.float32)[qi]; bv=npy(b['pred_visibility'],bool)[qi]; ot=npy(o['pred_tracks'],np.float32)[qi]; gt=npy(b['gt_tracks'],np.float32)[qi]
        lv=last_visible_before(bv,tr)
        pts={'query':(q,bt[q].tolist()), 'last':(lv if lv is not None else q,(bt[lv] if lv is not None else bt[q]).tolist()), 'gt_reentry':(rt,gt[rt].tolist()), 'online_reentry':(rt,ot[rt].tolist()), 'online_trigger':(tr,ot[tr].tolist()), 'base_reentry':(rt,bt[rt].tolist())}
        off=16/255.0
        for name,(dy,dx) in {'neg_u':(-off,0),'neg_d':(off,0),'neg_l':(0,-off),'neg_r':(0,off),'neg_uu':(-2*off,0),'neg_dd':(2*off,0),'neg_ll':(0,-2*off),'neg_rr':(0,2*off)}.items():
            pts[name]=(rt,offset_yx(gt[rt],dy,dx))
        for k in range(8):
            pts[f'rand{k}']=(rt,[float(rng.uniform(0,1)),float(rng.uniform(0,1))])
        for name,(ft,yx) in pts.items(): need.setdefault((vid,int(ft)),[]).append((i,name,yx))
    from transformers import AutoModel
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=AutoModel.from_pretrained(str(DINO_DIR),local_files_only=True).eval().to(device)
    num_reg=int(getattr(model.config,'num_register_tokens',4)); grid_hw=int(getattr(model.config,'image_size',224))//int(getattr(model.config,'patch_size',16))
    feat_store=[{} for _ in rows]; videos={}; keys=sorted(need.keys())
    print('rows',len(rows),'unique_frames',len(keys),'device',device,flush=True)
    for start in range(0,len(keys),args.batch_size):
        bkeys=keys[start:start+args.batch_size]; frames=[]
        for vid,ft in bkeys:
            if vid not in videos: videos[vid]=np.load(VIDEO_CACHE/f'{vid}.npz')['video']
            frames.append(videos[vid][ft])
        x=prep(np.stack(frames),device)
        with torch.no_grad():
            out=model(pixel_values=x)
            tokens=out.last_hidden_state[:,1+num_reg:,:]
            fmap=tokens.reshape(tokens.shape[0],grid_hw,grid_hw,tokens.shape[-1]).permute(0,3,1,2).contiguous()
        for bi,key in enumerate(bkeys):
            f=fmap[bi]
            for ri,name,yx in need[key]: feat_store[ri][name]=sample_feat(f,yx)
        print('encoded',min(start+args.batch_size,len(keys)),'/',len(keys),flush=True)
    out_rows=[]; pos_last=[]; neg_last=[]; pos_query=[]; neg_query=[]; pos_mem=[]; neg_mem=[]
    online_scores=[]; base_scores=[]; gt_scores=[]
    for i,r in enumerate(rows):
        d=feat_store[i]; q=d['query']; l=d['last']; gt=d['gt_reentry']; mem=(q+l)/2; mem=mem/max(np.linalg.norm(mem),1e-9)
        negs=[d[k] for k in ['neg_u','neg_d','neg_l','neg_r','neg_uu','neg_dd','neg_ll','neg_rr']]+[d[f'rand{k}'] for k in range(8)]
        qgt=cos(q,gt); lgt=cos(l,gt); mgt=cos(mem,gt)
        qnegs=[cos(q,n) for n in negs]; lnegs=[cos(l,n) for n in negs]; mnegs=[cos(mem,n) for n in negs]
        pos_query.append(qgt); neg_query.extend(qnegs); pos_last.append(lgt); neg_last.extend(lnegs); pos_mem.append(mgt); neg_mem.extend(mnegs)
        online=cos(mem,d['online_reentry']); base=cos(mem,d['base_reentry'])
        online_scores.append(online); base_scores.append(base); gt_scores.append(mgt)
        rank=1+sum(1 for s in mnegs if s>=mgt)
        out_rows.append({'video_id':r['video_id'],'query_idx':int(r['query_idx']),'query_t':int(r['query_t']),'trigger_t':int(r['trigger_t']),'reentry_t':int(r['first_eligible_reentry_t']),'mem_gt_cos':mgt,'query_gt_cos':qgt,'last_gt_cos':lgt,'mem_online_reentry_cos':online,'mem_base_reentry_cos':base,'mem_gt_rank_among_17':int(rank),'mem_gt_beats_local_negs':bool(mgt>max(mnegs[:8])),'mem_gt_beats_all_negs':bool(mgt>max(mnegs)),'mem_gt_margin_local':float(mgt-max(mnegs[:8])),'mem_gt_margin_all':float(mgt-max(mnegs))})
    summary={
        'n_rows':len(rows),'unique_frames':len(keys),
        'auc_query_gt_vs_neg':auc_from_scores(pos_query,neg_query),
        'auc_last_gt_vs_neg':auc_from_scores(pos_last,neg_last),
        'auc_mem_gt_vs_neg':auc_from_scores(pos_mem,neg_mem),
        'mean_mem_gt_cos':float(np.mean(pos_mem)),'mean_mem_neg_cos':float(np.mean(neg_mem)),
        'mean_mem_online_reentry_cos':float(np.mean(online_scores)),'mean_mem_base_reentry_cos':float(np.mean(base_scores)),
        'frac_gt_beats_local_negs':float(np.mean([r['mem_gt_beats_local_negs'] for r in out_rows])),
        'frac_gt_beats_all_negs':float(np.mean([r['mem_gt_beats_all_negs'] for r in out_rows])),
        'median_gt_rank_among_17':float(np.median([r['mem_gt_rank_among_17'] for r in out_rows])),
    }
    (OUT/'dinov3_point_identity_retrieval_rows.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in out_rows)+'\n')
    (OUT/'dinov3_point_identity_retrieval_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
