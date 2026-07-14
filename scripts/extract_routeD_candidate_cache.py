#!/usr/bin/env python3
"""Extract Route-D candidate caches from a frozen MMP checkpoint."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from projects.mmp_tracker.mmp_tracker import MMPTracker
from projects.mmp_tracker.train_mmp import config_from_dict, resolve_dataset
from projects.mmp_tracker.mmp_tracker.routeD_candidate_cache import extract_routeD_candidate_cache_batch, merge_routeD_candidate_cache_batches

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--split',default='train')
    p.add_argument('--dataset-root',default=None)
    p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,default=None)
    p.add_argument('--device',default='cuda')
    args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    cfg.setdefault('model',{}).setdefault('tracking',{})['enable_multi_hypothesis_diagnostics']=True
    model=MMPTracker(config_from_dict(cfg))
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model'],strict=True)
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    
    if args.dataset_root:
        cfg.setdefault('data',{}).setdefault('train',{})['root']=args.dataset_root
    dataset,_=resolve_dataset(cfg,args.split,False)
    loader=DataLoader(dataset,batch_size=1,shuffle=False)
    caches=[]
    with torch.no_grad():
        for i,batch in enumerate(loader):
            if args.limit is not None and i>=args.limit: break
            video=batch['video'].to(device)
            q=batch['query_points'].to(device)
            _,_,info=model(video,q,return_info=True)
            cache=extract_routeD_candidate_cache_batch(
                info,batch['target_points'].to(device),batch['occluded'].to(device),q,
                video_height=video.shape[-2],video_width=video.shape[-1],sample_ids=torch.tensor([i],device=device)
            )
            caches.append(cache)
            print('processed',i,'rows',int(cache['oracle_index'].shape[0]))
    merged=merge_routeD_candidate_cache_batches(caches)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'cache':merged,'metadata':{'split':args.split,'samples':len(caches)}},out)
    summary={k:(list(v.shape) if hasattr(v,'shape') else str(v)) for k,v in merged.items()}
    out.with_suffix('.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
