#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch


def first_events(gv,qt):
    T=len(gv); t=max(0,qt+1); out=[]
    while t<T:
        if gv[t]: t+=1; continue
        s=t
        while t<T and not gv[t]: t+=1
        if t<T and gv[t]:
            rt=t
            e=rt
            while e<T and gv[e]: e+=1
            out.append((s,rt,e,t-s))
        t+=1
    return out

def audit_setting(name, paths, out_path):
    caches={k:torch.load(v,map_location='cpu',weights_only=False) for k,v in paths.items()}
    sums={k:{'events':0,'frames':0,'coord8':0,'coord16':0,'predvis':0,'joint8':0,'joint16':0,'first_coord8':0,'first_predvis':0,'first_joint8':0} for k in ['offline','online','b2']}
    per_video=[]
    for ri in range(len(caches['offline']['records'])):
        recs={k:caches[k]['records'][ri] for k in sums}
        gt=np.asarray(recs['offline']['gt_tracks'],np.float32); gv=np.asarray(recs['offline']['gt_visibility'],bool); q=np.asarray(recs['offline']['query_points'],np.float32)
        pv={k:np.asarray(recs[k]['pred_visibility'],bool) for k in sums}
        pp={k:np.asarray(recs[k]['pred_tracks'],np.float32) for k in sums}
        vstat={k:{'events':0,'frames':0,'coord8':0,'coord16':0,'predvis':0,'joint8':0,'joint16':0,'first_coord8':0,'first_predvis':0,'first_joint8':0} for k in sums}
        for qi in range(q.shape[0]):
            qt=int(round(float(q[qi,0])))
            for s,rt,e,occ in first_events(gv[qi],qt):
                seg=slice(rt,e); L=e-rt
                if L<=0: continue
                for k in sums:
                    dist=np.linalg.norm(pp[k][qi,seg]-gt[qi,seg],axis=-1)
                    c8=dist<8; c16=dist<16; vis=pv[k][qi,seg]
                    vals={
                        'events':1,'frames':L,
                        'coord8':int(np.sum(c8)),'coord16':int(np.sum(c16)),
                        'predvis':int(np.sum(vis)),
                        'joint8':int(np.sum(vis & c8)),'joint16':int(np.sum(vis & c16)),
                        'first_coord8':int(np.linalg.norm(pp[k][qi,rt]-gt[qi,rt])<8),
                        'first_predvis':int(pv[k][qi,rt]),
                        'first_joint8':int(pv[k][qi,rt] and np.linalg.norm(pp[k][qi,rt]-gt[qi,rt])<8),
                    }
                    for kk,v in vals.items():
                        sums[k][kk]+=v; vstat[k][kk]+=v
        def norm(d):
            return {
                'events':d['events'],'frames':d['frames'],
                'coord8_rate':round(d['coord8']/max(d['frames'],1),6),
                'coord16_rate':round(d['coord16']/max(d['frames'],1),6),
                'predvis_recall':round(d['predvis']/max(d['frames'],1),6),
                'joint8_rate':round(d['joint8']/max(d['frames'],1),6),
                'joint16_rate':round(d['joint16']/max(d['frames'],1),6),
                'first_coord8_rate':round(d['first_coord8']/max(d['events'],1),6),
                'first_predvis_rate':round(d['first_predvis']/max(d['events'],1),6),
                'first_joint8_rate':round(d['first_joint8']/max(d['events'],1),6),
            }
        per_video.append({'video_id':str(recs['offline']['video_id']),'methods':{k:norm(vstat[k]) for k in sums}})
    def norm(d):
        return {
            'events':d['events'],'frames':d['frames'],
            'coord8_rate':round(d['coord8']/max(d['frames'],1),6),
            'coord16_rate':round(d['coord16']/max(d['frames'],1),6),
            'predvis_recall':round(d['predvis']/max(d['frames'],1),6),
            'joint8_rate':round(d['joint8']/max(d['frames'],1),6),
            'joint16_rate':round(d['joint16']/max(d['frames'],1),6),
            'first_coord8_rate':round(d['first_coord8']/max(d['events'],1),6),
            'first_predvis_rate':round(d['first_predvis']/max(d['events'],1),6),
            'first_joint8_rate':round(d['first_joint8']/max(d['events'],1),6),
        }
    out={'setting':name,'methods':{k:norm(v) for k,v in sums.items()},'per_video':per_video}
    Path(out_path).parent.mkdir(parents=True,exist_ok=True); Path(out_path).write_text(json.dumps(out,indent=2,ensure_ascii=False))
    return out

base='outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10'
settings={
 'dev_translate_L16': {
   'offline':f'{base}/translate_L16/predictions/cotracker3_offline_translate_L16.pt',
   'online':f'{base}/translate_L16/predictions/cotracker3_online_translate_L16.pt',
   'b2':f'{base}/translate_L16/predictions/b2_w16_p2_translate_L16.pt'},
 'dev_occluder_L16': {
   'offline':f'{base}/occluder_L16/predictions/cotracker3_offline_occluder_L16.pt',
   'online':f'{base}/occluder_L16/predictions/cotracker3_online_occluder_L16.pt',
   'b2':f'{base}/occluder_L16/predictions/b2_w16_p2_occluder_L16.pt'},
}
res=[]
for name,paths in settings.items():
    res.append(audit_setting(name,paths,f'outputs/paper_discovery_2026-06-27/reentry_identity_audit/{name}_segment_coord_visibility.json'))
print(json.dumps([{ 'setting':r['setting'], 'methods':r['methods']} for r in res], indent=2, ensure_ascii=False))
