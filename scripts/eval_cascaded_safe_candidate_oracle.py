#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_candidate_pool_oracle import npy, check_alignment, per_query_ajrd_map, eval_one

OUTDIR = Path('outputs/paper_discovery_2026-06-27/cascaded_safe_candidate_oracle')
OUTDIR.mkdir(parents=True, exist_ok=True)

SAFE_CANDIDATES = ['offline','b2_fullpost_p1','b2_w8_p2','b2_w16_p1','b2_w16_p2','b2_w32_p2']

PATHS = {
    'offline': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/offline_rgb_stacking_fresh20_49.pt'),
    'b2_fullpost_p1': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_natural_ablation/b2_fullpost_p1_rgb_fresh20_49.pt'),
    'b2_w8_p2': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_natural_ablation/b2_w8_p2_rgb_fresh20_49.pt'),
    'b2_w16_p1': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_natural_ablation/b2_w16_p1_rgb_fresh20_49.pt'),
    'b2_w16_p2': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_natural_ablation/b2_w16_p2_rgb_fresh20_49.pt'),
    'b2_w32_p2': Path('outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_natural_ablation/b2_w32_p2_rgb_fresh20_49.pt'),
    'guard_rf_thr0.40': Path('outputs/paper_discovery_2026-06-27/reentry_guard_v2_sklearn/rgb_fresh20_49_natural/random_forest/random_forest_thr0.40.pt'),
}


def query_changed(a: Dict[str,Any], b: Dict[str,Any], qi: int, tol: float = 1e-4) -> bool:
    ap=npy(a['pred_tracks'],np.float32)[qi]
    bp=npy(b['pred_tracks'],np.float32)[qi]
    av=npy(a['pred_visibility'],bool)[qi]
    bv=npy(b['pred_visibility'],bool)[qi]
    return bool(np.max(np.abs(ap-bp)) > tol or np.any(av != bv))


def build_safe_oracle(candidate_paths: Dict[str,Path], out_path: Path, only_guard_selected: bool = False) -> Dict[str,Any]:
    caches={k:torch.load(v,map_location='cpu',weights_only=False) for k,v in candidate_paths.items()}
    base=caches['offline']
    for k,c in caches.items():
        if k!='offline': check_alignment(base,c,k)
    records=[]; counts=Counter(); total_re=0; gate_selected=0; eligible_gate_re=0
    per_video=[]
    for rec_i, br in enumerate(base['records']):
        maps={cname:per_query_ajrd_map(caches[cname]['records'][rec_i]) for cname in SAFE_CANDIDATES}
        n=int(npy(br['query_points']).shape[0])
        chosen=np.array(['offline']*n,dtype=object)
        for qi,bval in maps['offline'].items():
            total_re += 1
            if only_guard_selected:
                if not query_changed(br, caches['guard_rf_thr0.40']['records'][rec_i], qi):
                    counts['offline'] += 1
                    continue
                gate_selected += 1
                eligible_gate_re += 1
            best='offline'; bestv=float(bval)
            for cname in SAFE_CANDIDATES:
                val=maps[cname].get(qi)
                if val is not None and float(val)>bestv+1e-9:
                    best=cname; bestv=float(val)
            chosen[qi]=best; counts[best]+=1
        pred_p=npy(br['pred_tracks'],np.float32).copy(); pred_v=npy(br['pred_visibility'],bool).copy()
        for cname in SAFE_CANDIDATES:
            if cname=='offline': continue
            use=chosen==cname
            if np.any(use):
                cr=caches[cname]['records'][rec_i]
                pred_p[use]=npy(cr['pred_tracks'],np.float32)[use]
                pred_v[use]=npy(cr['pred_visibility'],bool)[use]
        nr=dict(br); nr['pred_tracks']=pred_p.astype(np.float32); nr['pred_visibility']=pred_v.astype(bool)
        nr['cascaded_safe_oracle']={'only_guard_selected':only_guard_selected,'candidate_counts':dict(Counter(chosen.tolist()))}
        records.append(nr)
        per_video.append({'video_id':str(br['video_id']),'candidate_counts':dict(Counter(chosen.tolist()))})
    payload=dict(base); payload['records']=records; payload['model_name']='cascaded_safe_oracle_guard_selected' if only_guard_selected else 'safe_candidate_pool_oracle'
    payload['cascaded_safe_oracle']={
        'only_guard_selected':only_guard_selected,
        'safe_candidates':SAFE_CANDIDATES,
        'eligible_reentry_queries':int(total_re),
        'guard_selected_reentry_queries':int(gate_selected),
        'candidate_counts':dict(counts),
        'candidate_rates':{k:round(v/max(total_re,1),6) for k,v in counts.items()},
        'per_video':per_video,
    }
    out_path.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,out_path)
    return payload['cascaded_safe_oracle']


def main():
    missing=[str(p) for p in PATHS.values() if not p.exists()]
    if missing: raise FileNotFoundError('\n'.join(missing))
    safe_path=OUTDIR/'rgb_fresh20_49_natural_safe_candidate_pool_oracle.pt'
    gated_path=OUTDIR/'rgb_fresh20_49_natural_guard_gated_safe_candidate_oracle.pt'
    safe_sel=build_safe_oracle(PATHS,safe_path,only_guard_selected=False)
    gated_sel=build_safe_oracle(PATHS,gated_path,only_guard_selected=True)
    method_paths={k:v for k,v in PATHS.items()}
    method_paths['safe_candidate_pool_oracle']=safe_path
    method_paths['guard_gated_safe_oracle']=gated_path
    method_paths['candidate_pool_oracle']=Path('outputs/paper_discovery_2026-06-27/candidate_pool_oracle/rgb_fresh20_49_natural_candidate_pool_oracle.pt')
    method_paths['expanded_candidate_pool_oracle']=Path('outputs/paper_discovery_2026-06-27/candidate_pool_oracle_expanded/rgb_fresh20_49_natural_expanded_candidate_pool_oracle.pt')
    rows=[eval_one(name,path) for name,path in method_paths.items() if path.exists()]
    by={r['name']:r for r in rows}
    gains={}
    for name,r in by.items():
        if name=='offline': continue
        gains[f'{name}_vs_offline']={'AJ_RD_256':round(float(r['AJ_RD_256'])-float(by['offline']['AJ_RD_256']),6),'AJ_256':round(float(r['AJ_256'])-float(by['offline']['AJ_256']),6),'OA_256':round(float(r['OA_256'])-float(by['offline']['OA_256']),6)}
        if 'guard_rf_thr0.40' in by:
            gains[f'{name}_vs_guard']={'AJ_RD_256':round(float(r['AJ_RD_256'])-float(by['guard_rf_thr0.40']['AJ_RD_256']),6),'AJ_256':round(float(r['AJ_256'])-float(by['guard_rf_thr0.40']['AJ_256']),6),'OA_256':round(float(r['OA_256'])-float(by['guard_rf_thr0.40']['OA_256']),6)}
    summary={'setting':'rgb_fresh20_49_natural','safe_candidates':SAFE_CANDIDATES,'methods':rows,'gains':gains,'selection':{'safe_candidate_pool_oracle':safe_sel,'guard_gated_safe_oracle':gated_sel}}
    out=OUTDIR/'rgb_fresh20_49_natural_cascaded_safe_oracle_summary.json'
    out.write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    compact={
        'out_json':str(out),
        'methods':[{k:r[k] for k in ['name','AJ_RD_256','AJ_256','OA_256']} for r in rows],
        'key_gains':{k:v for k,v in gains.items() if 'safe' in k or 'guard_gated' in k or k.startswith('guard_rf')},
        'selection':summary['selection'],
    }
    print(json.dumps(compact,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
