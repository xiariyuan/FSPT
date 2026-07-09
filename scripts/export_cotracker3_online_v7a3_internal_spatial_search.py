#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path('/gemini/code/FSPT')
COTRACKER_ROOT = ROOT / 'baselines/cotracker'
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(COTRACKER_ROOT))
from cotracker.predictor import CoTrackerOnlinePredictor
from cotracker.datasets.tap_vid_datasets import TapVidDataset

DEFAULT_NATIVE = ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_true_streaming_v3_decoupled_subset_eval/full30_overlap_soft_tau055_w2_j4_min1_confirm4_p081/cotracker3_true_streaming_native_subset.pt'
DEFAULT_LABELS = ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v6a3_reentry_utility/cotracker3_v6a3_reentry_utility_labels.npz'
DEFAULT_DAVIS = ROOT/'datasets/tapvid_davis/tapvid_davis.pkl'
DEFAULT_CKPT = ROOT/'baselines/cotracker/checkpoints/scaled_online.pth'
DEFAULT_OUT = ROOT/'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a3_spatial_search/v7a3_spatial_search_smoke.npz'


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def load_video_tensor(ds, video_id, device, interp_shape):
    sample = ds[ds.video_names.index(video_id)]
    v = sample.video.detach().float()[None].to(device)
    B,T,C,H,W = v.shape
    v2 = F.interpolate(v.reshape(B*T,C,H,W), tuple(interp_shape), mode='bilinear', align_corners=True)
    return v2.reshape(B,T,C,interp_shape[0],interp_shape[1])


def compute_fmaps(model, video, chunk=32):
    # Match CoTracker3 forward(): video -> 2*(video/255)-1 before fnet.
    video = 2.0 * (video / 255.0) - 1.0
    B,T,C,H,W = video.shape
    outs=[]
    with torch.no_grad():
        for s in range(0,T,chunk):
            vc=video[:,s:s+chunk]
            f=model.fnet(vc.reshape(-1,C,H,W))
            outs.append(f.reshape(B,vc.shape[1],f.shape[1],f.shape[2],f.shape[3]))
        fmaps=torch.cat(outs,dim=1)
        # Match CoTracker normalization.
        fmaps=fmaps.permute(0,1,3,4,2)
        fmaps=fmaps/torch.sqrt(torch.clamp(torch.sum(fmaps.square(),dim=-1,keepdim=True), min=1e-12))
        fmaps=fmaps.permute(0,1,4,2,3).contiguous()
    return fmaps


def yx_to_grid(yx_norm, interp_hw, stride=8):
    y = float(yx_norm[0]) * (interp_hw[0]-1) / stride
    x = float(yx_norm[1]) * (interp_hw[1]-1) / stride
    return x, y


def sample_feat(fmap, frame_idx, yx_norm, interp_hw):
    B,T,D,Hf,Wf=fmap.shape
    x,y=yx_to_grid(yx_norm, interp_hw)
    yy=int(round(max(0,min(Hf-1,y)))); xx=int(round(max(0,min(Wf-1,x))))
    return fmap[0,frame_idx,:,yy,xx]


def entropy(vals, temp=10.0):
    a=np.asarray(vals,np.float32)
    if len(a)==0: return 0.0
    e=np.exp((a-a.max())*temp); p=e/max(float(e.sum()),1e-12)
    return float(-(p*np.log(p+1e-12)).sum()/math.log(max(len(p),2)))


def topk_search_frame(fmap, support_feat, native_yx, interp_hw, frame_idx, radius_px=64, topk=5):
    B,T,D,Hf,Wf=fmap.shape
    x0,y0=yx_to_grid(native_yx, interp_hw)
    r_feat = max(1, int(round(radius_px / 8.0)))
    xmin=max(0,int(math.floor(x0-r_feat))); xmax=min(Wf-1,int(math.ceil(x0+r_feat)))
    ymin=max(0,int(math.floor(y0-r_feat))); ymax=min(Hf-1,int(math.ceil(y0+r_feat)))
    patch=fmap[0,frame_idx,:,ymin:ymax+1,xmin:xmax+1].float() # D,h,w
    sf=F.normalize(support_feat.float(), dim=0)
    corr=torch.einsum('dhw,d->hw', patch, sf).detach().cpu().numpy().astype(np.float32)
    flat=corr.reshape(-1)
    order=np.argsort(flat)[::-1]
    k=min(topk,len(order))
    peaks=[]
    for rank in range(k):
        idx=int(order[rank]); yy=idx//corr.shape[1]; xx=idx%corr.shape[1]
        feat_x=xmin+xx; feat_y=ymin+yy
        y_norm=(feat_y*8.0)/(interp_hw[0]-1); x_norm=(feat_x*8.0)/(interp_hw[1]-1)
        peaks.append({'rank':rank+1,'score':float(flat[idx]),'yx':(float(y_norm),float(x_norm)),'feat_xy':(int(feat_x),int(feat_y))})
    sorted_vals=flat[order]
    peak=float(sorted_vals[0]) if len(sorted_vals) else 0.0
    margin=float(sorted_vals[0]-sorted_vals[1]) if len(sorted_vals)>1 else 0.0
    return peaks, {'peak':peak,'margin':margin,'entropy':entropy(flat),'mean':float(flat.mean()),'std':float(flat.std()),'search_cells':int(len(flat))}


def dist_px(a_yx, b_yx):
    return float(np.linalg.norm((np.asarray(a_yx,np.float32)-np.asarray(b_yx,np.float32))*255.0))


def event_features_and_recall(fmap, rec, event, interp_hw, radius_px=64, topk=5, post=4):
    pred=npy(rec['pred_tracks'],np.float32); gt=npy(rec['gt_tracks'],np.float32); gt_vis=npy(rec['gt_visibility'],bool); pred_vis=npy(rec['pred_visibility'],bool); score=npy(rec.get('pred_vis_score',pred_vis.astype(np.float32)),np.float32)
    q=int(event['query_idx']); t0=int(event['frame_t']); T=pred.shape[1]
    support_t=int(event.get('support_t',max(0,t0-1)))
    support_t=max(0,min(T-1,support_t))
    support_feat=sample_feat(fmap,support_t,pred[q,support_t],interp_hw)
    names=[]; feats=[]
    top1_errs=[]; top5_errs=[]; native_errs=[]; peak_scores=[]; margins=[]; entropies=[]; disp_from_native=[]
    best_top1=1e9; best_top5=1e9; best_native=1e9
    for off in range(post+1):
        tt=min(T-1,t0+off)
        peaks,stats=topk_search_frame(fmap,support_feat,pred[q,tt],interp_hw,tt,radius_px=radius_px,topk=topk)
        gt_yx=gt[q,tt]
        nat_err=dist_px(pred[q,tt], gt_yx) if bool(gt_vis[q,tt]) else 1e9
        t1_err=dist_px(peaks[0]['yx'], gt_yx) if peaks and bool(gt_vis[q,tt]) else 1e9
        t5_err=min([dist_px(p['yx'], gt_yx) for p in peaks], default=1e9) if bool(gt_vis[q,tt]) else 1e9
        top1_errs.append(t1_err); top5_errs.append(t5_err); native_errs.append(nat_err)
        best_top1=min(best_top1,t1_err); best_top5=min(best_top5,t5_err); best_native=min(best_native,nat_err)
        peak_scores.append(stats['peak']); margins.append(stats['margin']); entropies.append(stats['entropy'])
        disp=dist_px(peaks[0]['yx'], pred[q,tt]) if peaks else 1e9
        disp_from_native.append(disp)
        for k,v in [('peak',stats['peak']),('margin',stats['margin']),('entropy',stats['entropy']),('mean',stats['mean']),('std',stats['std']),('top1_err_gt_px',t1_err),('top5_err_gt_px',t5_err),('native_err_gt_px',nat_err),('top1_disp_native_px',disp),('native_score',float(score[q,tt])),('native_visible',1.0 if bool(pred_vis[q,tt]) else 0.0),('gt_visible',1.0 if bool(gt_vis[q,tt]) else 0.0)]:
            names.append(f'{k}_dt{off}'); feats.append(float(v))
    def add_stats(prefix, arr):
        a=np.asarray(arr,np.float32)
        for k,v in [('mean',a.mean()),('max',a.max()),('min',a.min()),('std',a.std()),('slope',a[-1]-a[0]),('recovery',a.max()-a[0])]:
            names.append(f'{prefix}_{k}'); feats.append(float(v))
    add_stats('peak',peak_scores); add_stats('margin',margins); add_stats('entropy',entropies); add_stats('top1_disp_native_px',disp_from_native)
    # Recall labels / oracle stats for this event-window.
    recall = {
        'best_native_err':float(best_native), 'best_top1_err':float(best_top1), 'best_top5_err':float(best_top5),
        'top1_r8':bool(best_top1<=8), 'top1_r16':bool(best_top1<=16), 'top5_r8':bool(best_top5<=8), 'top5_r16':bool(best_top5<=16),
        'native_r8':bool(best_native<=8), 'native_r16':bool(best_native<=16),
    }
    return feats,names,recall


def summarize(rows, labels):
    out={}
    for label_name,y in labels.items():
        y=np.asarray(y,bool)
        idx=np.ones(len(y),bool)
        out[label_name]={'n':int(len(y)),'pos':int(y.sum()),'base_rate':float(y.mean()) if len(y) else 0.0}
        for subset_name,mask in [('all',idx),('positive',y),('negative',~y)]:
            if mask.sum()==0: continue
            sub=[rows[i] for i in np.where(mask)[0]]
            out[label_name][subset_name]={}
            for key in ['native_r16','top1_r16','top5_r16','native_r8','top1_r8','top5_r8']:
                out[label_name][subset_name][key]=float(np.mean([r[key] for r in sub]))
            for key in ['best_native_err','best_top1_err','best_top5_err']:
                vals=[r[key] for r in sub if r[key]<1e8]
                out[label_name][subset_name][key+'_median']=float(np.median(vals)) if vals else None
    return out


def loov_ap_auc(X,y,groups):
    if y.sum()==0 or y.sum()==len(y) or len(set(groups))<2: return None
    oof=np.zeros(len(y),np.float32)
    for tr,te in LeaveOneGroupOut().split(X,y,groups):
        if len(np.unique(y[tr]))<2: return None
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced',solver='liblinear'))
        clf.fit(X[tr],y[tr]); oof[te]=clf.predict_proba(X[te])[:,1]
    return {'ap':float(average_precision_score(y,oof)),'auc':float(roc_auc_score(y,oof))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--native-cache',default=str(DEFAULT_NATIVE)); ap.add_argument('--labels',default=str(DEFAULT_LABELS)); ap.add_argument('--davis-pkl',default=str(DEFAULT_DAVIS)); ap.add_argument('--checkpoint',default=str(DEFAULT_CKPT)); ap.add_argument('--out-npz',default=str(DEFAULT_OUT))
    ap.add_argument('--max-videos',type=int,default=10); ap.add_argument('--sample-label',choices=['none','y_reentry_early4','y_reentry_early8','y_useful_open_t'],default='y_reentry_early8'); ap.add_argument('--neg-ratio',type=int,default=2); ap.add_argument('--sample-seed',type=int,default=0); ap.add_argument('--radius-px',type=int,default=64); ap.add_argument('--topk',type=int,default=5); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args(); t0=time.time(); out=Path(args.out_npz); out.parent.mkdir(parents=True,exist_ok=True)
    labels=np.load(args.labels,allow_pickle=True); all_meta=[json.loads(str(m)) for m in labels['meta_json'].tolist()]
    y4_all=np.asarray(labels['y_reentry_early4'],np.float32).astype(bool); y8_all=np.asarray(labels['y_reentry_early8'],np.float32).astype(bool); yu_all=np.asarray(labels['y_useful_open_t'],np.float32).astype(bool)
    label_map={'y_reentry_early4':y4_all,'y_reentry_early8':y8_all,'y_useful_open_t':yu_all}
    native=torch.load(args.native_cache,map_location='cpu',weights_only=False); rec_by_vid={str(r['video_id']):r for r in native['records']}; videos=[str(r['video_id']) for r in native['records'][:args.max_videos]]; allowed=set(videos)
    sampled=None
    if args.sample_label!='none':
        rng=np.random.default_rng(args.sample_seed); y=label_map[args.sample_label]
        pos=np.asarray([i for i,m in enumerate(all_meta) if m['video_id'] in allowed and y[i]],np.int64)
        neg=np.asarray([i for i,m in enumerate(all_meta) if m['video_id'] in allowed and not y[i]],np.int64)
        n_neg=min(len(neg),args.neg_ratio*max(1,len(pos)))
        neg_sel=rng.choice(neg,size=n_neg,replace=False) if n_neg>0 else np.asarray([],np.int64)
        sampled=set(np.concatenate([pos,neg_sel]).tolist())
        print({'sample_label':args.sample_label,'pos':len(pos),'neg':len(neg_sel),'total':len(sampled)},flush=True)
    ds=TapVidDataset(str(args.davis_pkl),dataset_type='davis',resize_to=[256,256],queried_first=True)
    predictor=CoTrackerOnlinePredictor(checkpoint=str(args.checkpoint)).to(args.device).eval(); model=predictor.model; interp_hw=tuple(predictor.interp_shape)
    X=[]; names=None; used_idx=[]; used_meta=[]; recall_rows=[]
    for vid in videos:
        ev=[i for i,m in enumerate(all_meta) if m['video_id']==vid and (sampled is None or i in sampled)]
        if not ev: continue
        print({'video':vid,'events':len(ev)},flush=True)
        video=load_video_tensor(ds,vid,args.device,interp_hw)
        with torch.no_grad(): fmap=compute_fmaps(model,video)
        rec=rec_by_vid[vid]
        for si in ev:
            feats,fn,recall=event_features_and_recall(fmap,rec,all_meta[si],interp_hw,radius_px=args.radius_px,topk=args.topk,post=4)
            if names is None: names=fn
            elif names!=fn: raise RuntimeError('feature name mismatch')
            X.append(feats); used_idx.append(si); used_meta.append(json.dumps({'source_index':si,**all_meta[si]},ensure_ascii=False)); recall_rows.append(recall)
        del video,fmap
        if args.device.startswith('cuda'): torch.cuda.empty_cache()
    X=np.asarray(X,np.float32); used_idx=np.asarray(used_idx,np.int64); groups=np.asarray([json.loads(m)['video_id'] for m in used_meta],object)
    y4=y4_all[used_idx]; y8=y8_all[used_idx]; yu=yu_all[used_idx]
    np.savez_compressed(out,X=X,feature_names=np.asarray(names,dtype=object),meta_json=np.asarray(used_meta,dtype=object),source_indices=used_idx,y_reentry_early4=y4.astype(np.float32),y_reentry_early8=y8.astype(np.float32),y_useful_open_t=yu.astype(np.float32),recall_json=np.asarray([json.dumps(r) for r in recall_rows],dtype=object),videos=np.asarray(videos,dtype=object),radius_px=args.radius_px,topk=args.topk)
    labels_used={'early4':y4,'early8':y8,'useful_open_t':yu}
    recall_summary=summarize(recall_rows,labels_used)
    # Remove GT err feature columns for classifier audit; use search stats only, not oracle errors/GT visibility.
    audit_idx=[i for i,n in enumerate(names) if ('err_gt' not in n and not n.startswith('gt_'))]
    audit={}
    for lname,y in labels_used.items():
        audit[lname]={'n_pos':int(y.sum()),'base_rate':float(y.mean()) if len(y) else 0.0,'loov_search_stats':loov_ap_auc(X[:,audit_idx],y,groups)}
    report={'script':'scripts/export_cotracker3_online_v7a3_internal_spatial_search.py','out_npz':str(out),'videos':videos,'effective_videos':sorted(set(groups.tolist())),'n_events':int(X.shape[0]),'feature_dim':int(X.shape[1]),'radius_px':args.radius_px,'topk':args.topk,'sample_label':args.sample_label,'recall_summary':recall_summary,'audit':audit,'sec':round(time.time()-t0,2)}
    out.with_suffix('.report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
