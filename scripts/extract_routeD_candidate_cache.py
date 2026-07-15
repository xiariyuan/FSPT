#!/usr/bin/env python3
"""Extract Route-D candidate caches from a frozen MMP checkpoint."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
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
    p.add_argument('--annotation-file',default=None)
    p.add_argument('--subset',type=int,default=None)
    p.add_argument('--dataset-split',default=None)
    p.add_argument('--dataset',default=None)
    p.add_argument('--cache-role',choices=['train','validation','diagnostic'],default='diagnostic')
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
    
    split_cfg = cfg.setdefault('data',{}).setdefault(args.split,{})
    if args.dataset_root:
        split_cfg['root']=args.dataset_root
    if args.annotation_file:
        split_cfg['annotation_file']=args.annotation_file
    if args.subset is not None:
        split_cfg['subset']=int(args.subset)
    if args.dataset_split:
        split_cfg['split']=args.dataset_split
    if args.dataset:
        split_cfg['dataset']=args.dataset
    dataset,resolved_limit=resolve_dataset(cfg,args.split,False)
    effective_limit = args.limit if args.limit is not None else resolved_limit
    loader=DataLoader(dataset,batch_size=1,shuffle=False)
    caches=[]
    with torch.no_grad():
        for i,batch in enumerate(loader):
            if effective_limit is not None and i>=effective_limit: break
            video=batch['video'].to(device)
            q=batch['query_points'].to(device)
            _,_,info=model(video,q,return_info=True)
            cache=extract_routeD_candidate_cache_batch(
                info,batch['target_points'].to(device),batch['occluded'].to(device),q,
                video_height=video.shape[-2],video_width=video.shape[-1],sample_ids=torch.tensor([i],device=device)
            )
            caches.append(cache)
            print('processed',i,'rows',int(cache['oracle_index'].shape[0]))
    if not caches:
        raise RuntimeError('No candidate-cache rows were extracted')
    merged=merge_routeD_candidate_cache_batches(caches)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    def sha256(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
        return h.hexdigest()
    try:
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except Exception:
        git_head=None
    metadata={
        'format_version':2,
        'kind':'routeD_frozen_candidate_cache',
        'evidence_tier':'development_diagnostic_only',
        'paper_claim_eligible':False,
        'split_key':args.split,
        'cache_role':args.cache_role,
        'dataset_split':split_cfg.get('split'),
        'dataset_root':str(Path(split_cfg.get('root','')).resolve()),
        'annotation_file':str(split_cfg.get('annotation_file','')),
        'configured_subset':split_cfg.get('subset'),
        'effective_limit':effective_limit,
        'samples_processed':len(caches),
        'rows':int(merged['features'].shape[0]),
        'candidate_count':int(merged['features'].shape[1]),
        'feature_dim':int(merged['features'].shape[2]),
        'candidate_representation':'local_plus_coarse_global_topk',
        'feature_contract':'causal_pre_frame_state_v2',
        'previous_confidence_timing':'pre_frame_state',
        'coordinate_order':'yx',
        'coordinate_normalization':'pixel_center_divide_by_size_minus_one',
        'config_path':str(Path(args.config).resolve()),
        'config_sha256':sha256(args.config),
        'checkpoint_path':str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256':sha256(args.checkpoint),
        'git_head':git_head,
        'device':str(device),
    }
    torch.save({'cache':merged,'metadata':metadata},out)
    summary={'metadata':metadata,'tensors':{k:(list(v.shape) if hasattr(v,'shape') else str(v)) for k,v in merged.items()}}
    out.with_suffix('.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
