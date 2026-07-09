#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path('/gemini/code/FSPT')
COTRACKER_ROOT = ROOT / 'baselines/cotracker'
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(COTRACKER_ROOT))
from cotracker.predictor import CoTrackerOnlinePredictor
from cotracker.datasets.tap_vid_datasets import TapVidDataset

DEFAULT_NATIVE = ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_true_streaming_v3_decoupled_subset_eval/full30_overlap_soft_tau055_w2_j4_min1_confirm4_p081/cotracker3_true_streaming_native_subset.pt'
DEFAULT_LABELS = ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v6a3_reentry_utility/cotracker3_v6a3_reentry_utility_labels.npz'
DEFAULT_DAVIS = ROOT/'datasets/tapvid_davis/tapvid_davis.pkl'
DEFAULT_CKPT = ROOT/'baselines/cotracker/checkpoints/scaled_online.pth'
DEFAULT_OUT = ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a_internal_corr/v7a1_internal_corr_first10.npz'

def npy(x:Any,dtype=None)->np.ndarray:
    if isinstance(x, torch.Tensor): x=x.detach().cpu().numpy()
    a=np.asarray(x); return a.astype(dtype) if dtype is not None else a

def load_video_tensor(ds, video_id, device, interp_shape):
    sample=ds[ds.video_names.index(video_id)]
    v=sample.video.detach().float()[None].to(device)
    B,T,C,H,W=v.shape
    v2=F.interpolate(v.reshape(B*T,C,H,W), tuple(interp_shape), mode='bilinear', align_corners=True)
    return v2.reshape(B,T,C,interp_shape[0],interp_shape[1])

def compute_fmaps(model, video, chunk=32):
    B,T,C,H,W=video.shape; outs=[]
    with torch.no_grad():
        for s in range(0,T,chunk):
            vc=video[:,s:s+chunk]
            f=model.fnet(vc.reshape(-1,C,H,W))
            outs.append(f.reshape(B,vc.shape[1],f.shape[1],f.shape[2],f.shape[3]))
        fmaps=torch.cat(outs,dim=1)
        fmaps=fmaps.permute(0,1,3,4,2)
        fmaps=fmaps/torch.sqrt(torch.clamp(torch.sum(fmaps.square(),dim=-1,keepdim=True),min=1e-12))
        fmaps=fmaps.permute(0,1,4,2,3).contiguous()
        pyr=[fmaps]; cur=fmaps
        for _ in range(model.corr_levels-1):
            B,T,D,Hf,Wf=cur.shape
            pooled=F.avg_pool2d(cur.reshape(B*T,D,Hf,Wf),2,stride=2)
            cur=pooled.reshape(B,T,D,pooled.shape[-2],pooled.shape[-1])
            pyr.append(cur)
    return pyr

def yx_to_xy_feat(yx_norm, interp_hw, level=0, stride=8):
    y=float(yx_norm[0])*(interp_hw[0]-1)/stride/(2**level)
    x=float(yx_norm[1])*(interp_hw[1]-1)/stride/(2**level)
    return x,y

def sample_center(fmap, frame_idx, yx_norm, interp_hw, level=0):
    B,T,D,Hf,Wf=fmap.shape
    x,y=yx_to_xy_feat(yx_norm, interp_hw, level=level)
    yy=int(round(max(0,min(Hf-1,y)))); xx=int(round(max(0,min(Wf-1,x))))
    return fmap[0,frame_idx,:,yy,xx]

def soft_entropy(vals, temp=10.0):
    a=np.asarray(vals,np.float32)
    e=np.exp((a-a.max())*temp); p=e/max(float(e.sum()),1e-12)
    return float(-(p*np.log(p+1e-12)).sum()/math.log(len(p)))

def exact_corr_volume_stats(model, fmap, pred, q, support_t, t, interp_hw, level=0):
    # Faithful level-0 corr_volume summary following forward_window shape logic.
    device=fmap.device; r=model.corr_radius; rr=2*r+1
    sx,sy=yx_to_xy_feat(pred[q,support_t], interp_hw, level=level)
    tx,ty=yx_to_xy_feat(pred[q,t], interp_hw, level=level)
    queried_frames=torch.tensor([[support_t]],device=device,dtype=torch.long)
    queried_coords=torch.tensor([[[sx,sy]]],device=device,dtype=fmap.dtype)
    with torch.no_grad():
        _, support = model.get_track_feat(fmap, queried_frames, queried_coords, support_radius=r)
        # support: B, rr*rr, N, D -> rr,rr,D
        supp = support[0,:,0,:].reshape(rr,rr,-1).float()
        # Use only the requested frame. get_correlation_feat flattens B*T internally,
        # so fmap_one has B*T=1 and curr_coords [1,1,2] is correctly aligned.
        curr_coords=torch.tensor([[[tx,ty]]],device=device,dtype=fmap.dtype)
        fmap_one=fmap[:, t:t+1].contiguous()
        corr_feat=model.get_correlation_feat(fmap_one, curr_coords)[0,0,0].float()  # rr,rr,D
        vol=torch.einsum('hwc,ijc->hwij', corr_feat, supp).reshape(-1)
        arr=vol.detach().cpu().numpy().astype(np.float32)
    arrs=np.sort(arr)[::-1]
    peak=float(arrs[0]); top2=float(arrs[1]) if len(arrs)>1 else peak
    center_index=(rr//2)*rr*rr*rr + (rr//2)*rr*rr + (rr//2)*rr + (rr//2)
    center=float(arr[center_index]) if center_index<len(arr) else float(arr.mean())
    return {
        'exact_l0_peak':peak,
        'exact_l0_margin':peak-top2,
        'exact_l0_mean':float(arr.mean()),
        'exact_l0_std':float(arr.std()),
        'exact_l0_entropy':soft_entropy(arr),
        'exact_l0_center':center,
        'exact_l0_center_minus_peak':center-peak,
    }

def approx_local_stats(fmap, pred, q, support_t, last_t, t, interp_hw, level=0, radius=3):
    B,T,D,Hf,Wf=fmap.shape
    supp=F.normalize(sample_center(fmap,support_t,pred[q,support_t],interp_hw,level=level).float(),dim=0)
    last=F.normalize(sample_center(fmap,last_t,pred[q,last_t],interp_hw,level=level).float(),dim=0)
    cur=F.normalize(sample_center(fmap,t,pred[q,t],interp_hw,level=level).float(),dim=0)
    cos_support=float(torch.dot(cur,supp).detach().cpu()); cos_last=float(torch.dot(cur,last).detach().cpu())
    x0,y0=yx_to_xy_feat(pred[q,t],interp_hw,level=level); vals=[]; fmap_frame=fmap[0,t].float()
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            yy=int(round(max(0,min(Hf-1,y0+dy)))); xx=int(round(max(0,min(Wf-1,x0+dx))))
            f=F.normalize(fmap_frame[:,yy,xx],dim=0)
            vals.append(float(torch.dot(f,supp).detach().cpu()))
    arr=np.asarray(vals,np.float32); s=np.sort(arr)[::-1]
    return {'approx_cos_support':cos_support,'approx_cos_last':cos_last,'approx_peak':float(s[0]),'approx_margin':float(s[0]-s[1]),'approx_entropy':soft_entropy(arr),'approx_center_minus_peak':float(cos_support-s[0])}

def extract_features(model,fmaps,rec,event,interp_hw,post=4,exact=True):
    pred=npy(rec['pred_tracks'],np.float32); vis=npy(rec['pred_visibility'],bool); score=npy(rec.get('pred_vis_score',vis.astype(np.float32)),np.float32)
    q=int(event['query_idx']); t0=int(event['frame_t']); support_t=int(event.get('support_t',max(0,t0-1))); last_t=int(event.get('last_visible_t',support_t)); T=pred.shape[1]
    names=[]; feats=[]
    seqs={}
    for off in range(0,post+1):
        tt=min(T-1,t0+off)
        d={}
        d.update(approx_local_stats(fmaps[0],pred,q,support_t,last_t,tt,interp_hw,level=0))
        if exact:
            d.update(exact_corr_volume_stats(model,fmaps[0],pred,q,support_t,tt,interp_hw,level=0))
        for k,v in d.items():
            names.append(f'{k}_dt{off}'); feats.append(float(v)); seqs.setdefault(k,[]).append(float(v))
        names.append(f'native_score_dt{off}'); feats.append(float(score[q,tt]))
        names.append(f'native_vis_dt{off}'); feats.append(1.0 if bool(vis[q,tt]) else 0.0)
    for k,seq in seqs.items():
        a=np.asarray(seq,np.float32)
        for stat,val in [('mean',a.mean()),('max',a.max()),('min',a.min()),('std',a.std()),('slope',a[-1]-a[0]),('recovery',a.max()-a[0])]:
            names.append(f'{k}_{stat}'); feats.append(float(val))
    return feats,names

def feature_auc(X,y,names,idxs=None):
    if idxs is None: idxs=range(X.shape[1])
    rows=[]
    if y.sum()==0 or y.sum()==len(y): return rows
    for j in idxs:
        vals=X[:,j]
        if len(np.unique(vals))<2: continue
        ap=average_precision_score(y,vals); auc=roc_auc_score(y,vals); apn=average_precision_score(y,-vals); aucn=roc_auc_score(y,-vals)
        if apn>ap: rows.append({'feature':names[j],'direction':'-','ap':float(apn),'auc':float(max(auc,1-auc))})
        else: rows.append({'feature':names[j],'direction':'+','ap':float(ap),'auc':float(max(auc,1-auc))})
    return sorted(rows,key=lambda r:(r['ap'],r['auc']),reverse=True)

def loov_logreg(X,y,groups,idxs):
    if y.sum()==0 or y.sum()==len(y): return None
    oof=np.zeros(len(y),np.float32); logo=LeaveOneGroupOut()
    for tr,te in logo.split(X,y,groups):
        if len(np.unique(y[tr]))<2: return None
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced',solver='liblinear'))
        clf.fit(X[tr][:,idxs],y[tr]); oof[te]=clf.predict_proba(X[te][:,idxs])[:,1]
    return {'ap':float(average_precision_score(y,oof)),'auc':float(roc_auc_score(y,oof))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--native-cache',default=str(DEFAULT_NATIVE)); ap.add_argument('--labels',default=str(DEFAULT_LABELS)); ap.add_argument('--davis-pkl',default=str(DEFAULT_DAVIS)); ap.add_argument('--checkpoint',default=str(DEFAULT_CKPT)); ap.add_argument('--out-npz',default=str(DEFAULT_OUT))
    ap.add_argument('--max-videos',type=int,default=10); ap.add_argument('--max-events-per-video',type=int,default=9999); ap.add_argument('--no-exact',action='store_true'); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--sample-label', choices=['none','y_reentry_early4','y_reentry_early8','y_useful_open_t'], default='none')
    ap.add_argument('--neg-ratio', type=int, default=2)
    ap.add_argument('--sample-seed', type=int, default=0)
    args=ap.parse_args(); t0=time.time(); out=Path(args.out_npz); out.parent.mkdir(parents=True,exist_ok=True)
    labels=np.load(args.labels,allow_pickle=True); all_meta=[json.loads(str(m)) for m in labels['meta_json'].tolist()]
    y4_all=np.asarray(labels['y_reentry_early4'],np.float32).astype(bool); y8_all=np.asarray(labels['y_reentry_early8'],np.float32).astype(bool); yu_all=np.asarray(labels['y_useful_open_t'],np.float32).astype(bool)
    label_map={'y_reentry_early4': y4_all, 'y_reentry_early8': y8_all, 'y_useful_open_t': yu_all}
    native=torch.load(args.native_cache,map_location='cpu',weights_only=False); rec_by_vid={str(r['video_id']):r for r in native['records']}; videos=[str(r['video_id']) for r in native['records'][:args.max_videos]]
    allowed_video_set=set(videos)
    sampled_indices_set=None
    if args.sample_label != 'none':
        rng=np.random.default_rng(args.sample_seed)
        y_sample=label_map[args.sample_label]
        pos=np.asarray([i for i,m in enumerate(all_meta) if m['video_id'] in allowed_video_set and y_sample[i]], dtype=np.int64)
        neg=np.asarray([i for i,m in enumerate(all_meta) if m['video_id'] in allowed_video_set and (not y_sample[i])], dtype=np.int64)
        n_neg=min(len(neg), max(1,args.neg_ratio)*max(1,len(pos)))
        neg_sel=rng.choice(neg, size=n_neg, replace=False) if n_neg>0 else np.asarray([], dtype=np.int64)
        sampled_indices_set=set(np.concatenate([pos,neg_sel]).tolist())
        print({'sample_label':args.sample_label,'pos':int(len(pos)),'neg_selected':int(len(neg_sel)),'total_sampled':int(len(sampled_indices_set))}, flush=True)
    ds=TapVidDataset(str(args.davis_pkl),dataset_type='davis',resize_to=[256,256],queried_first=True)
    predictor=CoTrackerOnlinePredictor(checkpoint=str(args.checkpoint)).to(args.device).eval(); model=predictor.model; interp_hw=tuple(predictor.interp_shape)
    X=[]; names=None; used_meta=[]; used_idx=[]
    for vid in videos:
        ev=[i for i,m in enumerate(all_meta) if m['video_id']==vid]
        if sampled_indices_set is not None:
            ev=[i for i in ev if i in sampled_indices_set]
        if args.max_events_per_video < len(ev): ev=ev[:args.max_events_per_video]
        if not ev: continue
        print({'video':vid,'events':len(ev)},flush=True)
        video=load_video_tensor(ds,vid,args.device,interp_hw)
        with torch.no_grad(): fmaps=compute_fmaps(model,video)
        rec=rec_by_vid[vid]
        for si in ev:
            feats,fn=extract_features(model,fmaps,rec,all_meta[si],interp_hw,post=4,exact=(not args.no_exact))
            if names is None: names=fn
            elif names!=fn: raise RuntimeError('feature names mismatch')
            X.append(feats); used_meta.append(json.dumps({'source_index':si,**all_meta[si]},ensure_ascii=False)); used_idx.append(si)
        del video,fmaps
        if args.device.startswith('cuda'): torch.cuda.empty_cache()
    X=np.asarray(X,np.float32); used_idx=np.asarray(used_idx,np.int64); groups=np.asarray([json.loads(m)['video_id'] for m in used_meta],object)
    y4=y4_all[used_idx]; y8=y8_all[used_idx]; yu=yu_all[used_idx]
    np.savez_compressed(out,X=X,feature_names=np.asarray(names,dtype=object),meta_json=np.asarray(used_meta,dtype=object),source_indices=used_idx,y_reentry_early4=y4.astype(np.float32),y_reentry_early8=y8.astype(np.float32),y_useful_open_t=yu.astype(np.float32),videos=np.asarray(videos,dtype=object),labels_source=str(args.labels),exact_enabled=not args.no_exact)
    fams={'native':[i for i,n in enumerate(names) if n.startswith('native_')], 'approx_internal':[i for i,n in enumerate(names) if n.startswith('approx_')], 'exact_corr':[i for i,n in enumerate(names) if n.startswith('exact_')], 'all_internal':[i for i,n in enumerate(names) if not n.startswith('native_')], 'all':list(range(len(names)))}
    audits={}
    for lname,y in [('early4',y4),('early8',y8),('useful_open_t',yu)]:
        audits[lname]={'n_pos':int(y.sum()),'base_rate':float(y.mean()) if len(y) else 0.0,'families':{}}
        for fam,idxs in fams.items():
            top=feature_auc(X,y,names,idxs)[:15]
            loov=loov_logreg(X,y,groups,idxs) if (len(set(groups))>=2 and len(idxs)>0) else None
            audits[lname]['families'][fam]={'n_features':len(idxs),'top_features':top,'loov_logreg':loov}
    report={'script':'scripts/export_cotracker3_online_v7a1_internal_corr_first10.py','out_npz':str(out),'videos':videos,'effective_videos':sorted(set(groups.tolist())),'n_events':int(X.shape[0]),'feature_dim':int(X.shape[1]) if X.ndim==2 else 0,'exact_enabled':not args.no_exact,'audits':audits,'sec':round(time.time()-t0,2)}
    out.with_suffix('.report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
