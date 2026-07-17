#!/usr/bin/env python3
"""Build resumable frozen local-token caches for P0h comparator training."""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import yaml

REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_feature_cache import load_complete_feature_index
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_cache import (
    CMCP_PAIRWISE_TOKEN_CACHE_SCHEMA_VERSION, CMCP_PAIRWISE_TOKEN_INDEX_SCHEMA_VERSION,
    verify_pairwise_token_artifact,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_safety import (
    CMCPLocalPairwiseSafetyComparator, CMCPLocalSafetyConfig,
    build_cmcp_local_candidate_tokens,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_training import load_cmcp_video
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import canonical_json_sha256, file_sha256
from projects.mmp_tracker.mmp_tracker.routeD_multi_memory_proposal import (
    CMCPConfig, CausalMultiMemoryProposalGenerator,
    build_causal_multi_memory_correlations, extract_proposal_candidates,
)
from projects.mmp_tracker.mmp_tracker.routeD_musr_training import state_dict_sha256

DEFAULT_CONFIG=REPO_ROOT/'configs/routeD_cmcp_local_pairwise_safety_v0.yaml'
DEFAULT_FEATURE_ROOT=REPO_ROOT/'outputs/routeD_cmcp_feature_cache_20260717'
DEFAULT_OUTPUT_ROOT=REPO_ROOT/'outputs/routeD_cmcp_pairwise_token_cache_20260717'
PARTITIONS=('fit','model_validation')


def _deterministic(seed:int)->None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=False)


def _load_generator(config:dict[str,Any],device:str):
    path=(REPO_ROOT/config['frozen_generator']['checkpoint']).resolve()
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    model=CausalMultiMemoryProposalGenerator(CMCPConfig(**checkpoint['model_config']))
    model.load_state_dict(checkpoint['model_state'],strict=True); model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    if state_dict_sha256(model.state_dict())!=checkpoint['model_state_sha256']: raise RuntimeError('generator state hash mismatch')
    if checkpoint['model_state_sha256']!=config['frozen_generator']['model_state_sha256']: raise RuntimeError('configured state hash mismatch')
    return model,checkpoint,path


def _build(generator,bundle,*,point_chunk:int,device:str):
    tensors=bundle.tensors; fmaps=bundle.feature_maps.float().to(device)
    native_all=tensors['native_coords_xy_px']; query_all=tensors['query_points_tyx']
    collected={key:[] for key in ('candidate_tokens','candidate_coords_xy_px','candidate_scores','candidate_valid_mask','frame_valid')}
    for start in range(0,native_all.shape[0],point_chunk):
        end=min(native_all.shape[0],start+point_chunk)
        native=native_all[start:end].to(device); query=query_all[start:end].to(device)
        corr,motion,valid=build_causal_multi_memory_correlations(
            fmaps,native,query,input_height=generator.config.input_height,input_width=generator.config.input_width,
            ema_alpha=generator.config.ema_alpha,motion_sigma_cells=generator.config.motion_sigma_cells,
        )
        state=None; chunk={key:[] for key in collected}
        zero_summary=torch.zeros(end-start,4,device=device)
        with torch.no_grad():
            for frame in range(native.shape[1]):
                dense,state=generator.step(corr[:,frame],motion[:,frame],state,frame_valid=valid[:,frame])
                proposal=extract_proposal_candidates(dense['proposal_score'],native[:,frame],dense['native_logit'],generator.config)
                candidate_valid=proposal['candidate_valid_mask'].clone(); candidate_valid[:,1:] &= valid[:,frame,None]
                token=build_cmcp_local_candidate_tokens(
                    hidden_map=dense['hidden_map'],recurrent_input=dense['recurrent_input'],
                    utility_logit=dense['utility_logit'],risk_logit=dense['risk_logit'],proposal_score=dense['proposal_score'],
                    candidate_coords_xy_px=proposal['candidate_coords_xy_px'],candidate_scores=proposal['candidate_scores'],
                    candidate_valid_mask=candidate_valid,
                    native_visibility_probability=tensors['native_visibility_probability'][start:end,frame].to(device),
                    native_confidence_probability=tensors['native_confidence_probability'][start:end,frame].to(device),
                    native_joint_probability=tensors['native_joint_probability'][start:end,frame].to(device),
                    previous_decision_summary=zero_summary,
                    input_height=generator.config.input_height,input_width=generator.config.input_width,
                )
                chunk['candidate_tokens'].append(token.cpu())
                chunk['candidate_coords_xy_px'].append(proposal['candidate_coords_xy_px'].cpu())
                chunk['candidate_scores'].append(proposal['candidate_scores'].cpu())
                chunk['candidate_valid_mask'].append(candidate_valid.cpu())
                chunk['frame_valid'].append(valid[:,frame].cpu())
        for key in collected: collected[key].append(torch.stack(chunk[key],dim=1))
    result={key:torch.cat(value,dim=0) for key,value in collected.items()}
    result['native_coords_xy_px']=tensors['native_coords_xy_px'].clone()
    if not torch.equal(result['candidate_coords_xy_px'][...,0,:],result['native_coords_xy_px']): raise RuntimeError('native parity drift')
    if torch.count_nonzero(result['candidate_tokens'][...,-4:]).item()!=0: raise RuntimeError('dynamic placeholder is nonzero')
    return result


def _paths(out:Path,index:int):
    stem=f'video_{index:05d}'; return out/f'{stem}.pt',out/f'{stem}.json'


def _valid(sidecar:Path,report:Path,*,generator_hash:str,base_hash:str)->bool:
    if not sidecar.exists() or not report.exists(): return False
    try:
        r=json.loads(report.read_text())
        if r.get('sidecar_sha256')!=file_sha256(sidecar): return False
        verify_pairwise_token_artifact(sidecar,expected_generator_state_sha256=generator_hash,expected_base_sidecar_sha256=base_hash)
        return True
    except Exception: return False


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default=str(DEFAULT_CONFIG)); ap.add_argument('--feature-root',default=str(DEFAULT_FEATURE_ROOT))
    ap.add_argument('--output-root',default=str(DEFAULT_OUTPUT_ROOT)); ap.add_argument('--partition',choices=PARTITIONS,required=True)
    ap.add_argument('--start-offset',type=int,default=0); ap.add_argument('--max-videos',type=int,default=0)
    ap.add_argument('--point-chunk',type=int,default=4); ap.add_argument('--resume',action='store_true')
    ap.add_argument('--replay-indices',default='0,47,48,63'); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args(); _deterministic(17000)
    config_path=Path(args.config).resolve(); config=yaml.safe_load(config_path.read_text())
    feature_index_path=Path(args.feature_root).resolve()/args.partition/'cache_index.json'
    feature_index=load_complete_feature_index(feature_index_path,expected_partition=args.partition)
    expected=[int(v) for v in feature_index['expected_source_indices']]
    selected=expected[args.start_offset:];
    if args.max_videos>0: selected=selected[:args.max_videos]
    rows={int(row['source_index']):row for row in feature_index['videos']}
    generator,checkpoint,checkpoint_path=_load_generator(config,args.device)
    out=Path(args.output_root).resolve()/args.partition; out.mkdir(parents=True,exist_ok=True)
    replay_indices={int(v) for v in args.replay_indices.split(',') if v.strip()}
    reports={}
    if args.resume:
        for source_index in expected:
            sidecar,report=_paths(out,source_index); row=rows[source_index]
            if _valid(sidecar,report,generator_hash=checkpoint['model_state_sha256'],base_hash=row['base_sidecar_sha256']): reports[source_index]=json.loads(report.read_text())
    print(json.dumps({'stage':'pairwise_token_partition_start','partition':args.partition,'selected_indices':selected,'generator_state_sha256':checkpoint['model_state_sha256']}),flush=True)
    for ordinal,source_index in enumerate(selected,1):
        if source_index in reports:
            print(json.dumps({'stage':'resume_skip','source_index':source_index}),flush=True); continue
        started=time.time(); row=rows[source_index]; bundle=load_cmcp_video(row)
        print(json.dumps({'stage':'pairwise_token_video_start','ordinal':ordinal,'selected_count':len(selected),'source_index':source_index,'video_name':bundle.video_name}),flush=True)
        tensors=_build(generator,bundle,point_chunk=args.point_chunk,device=args.device)
        replay=None
        if source_index in replay_indices:
            second=_build(generator,bundle,point_chunk=args.point_chunk,device=args.device)
            replay={key:torch.equal(tensors[key],second[key]) for key in tensors}
            replay['exact']=all(replay.values())
            if not replay['exact']: raise RuntimeError(f'replay mismatch {source_index}: {replay}')
        hashes={key:tensor_sha256(value) for key,value in tensors.items()}
        sidecar,report_path=_paths(out,source_index)
        artifact={
            'schema_version':CMCP_PAIRWISE_TOKEN_CACHE_SCHEMA_VERSION,
            'provenance':{
                'partition':args.partition,'source_index':source_index,'sample_identity':row['sample_identity'],
                'config_sha256':file_sha256(config_path),'feature_index_sha256':feature_index['_index_sha256'],
                'feature_sidecar_sha256':row['sidecar_sha256'],'base_sidecar':row['base_sidecar'],
                'base_sidecar_sha256':row['base_sidecar_sha256'],'generator_checkpoint_sha256':file_sha256(checkpoint_path),
                'generator_model_state_sha256':checkpoint['model_state_sha256'],
                'ground_truth_used_for_tokens':False,'dynamic_summary_placeholder_zero':True,
            },
            'tensors':tensors,'tensor_hashes':hashes,
        }
        torch.save(artifact,sidecar)
        report={
            'schema_version':'routeD_cmcp_pairwise_token_video_report_v0','partition':args.partition,
            'source_index':source_index,'sample_identity':row['sample_identity'],'sidecar':str(sidecar),
            'sidecar_sha256':file_sha256(sidecar),'base_sidecar_sha256':row['base_sidecar_sha256'],
            'feature_sidecar_sha256':row['sidecar_sha256'],'generator_model_state_sha256':checkpoint['model_state_sha256'],
            'tensor_hashes':hashes,'deterministic_replay':replay,'seconds':round(time.time()-started,3),
            'integrity':{'ground_truth_used_for_tokens':False,'candidate_coordinates_modified':False,'dynamic_summary_placeholder_zero':True,
                         'calibration_read':False,'final_holdout_read':False,'tapvid_davis_read':False,'tapvid_kinetics_read':False},
        }
        report_path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n'); reports[source_index]=report
        print(json.dumps({'stage':'pairwise_token_video_complete','source_index':source_index,'candidate_coordinate_sha256':hashes['candidate_coords_xy_px'],'replay_exact':None if replay is None else replay['exact'],'seconds':report['seconds']}),flush=True)
    completed=sorted(reports)
    ordered=[reports[i] for i in completed]
    payload={
        'schema_version':CMCP_PAIRWISE_TOKEN_INDEX_SCHEMA_VERSION,'partition':args.partition,
        'expected_source_indices':expected,'completed_source_indices':completed,'expected_count':len(expected),'completed_count':len(completed),
        'complete':completed==expected,'config_sha256':file_sha256(config_path),'feature_index_sha256':feature_index['_index_sha256'],
        'generator_checkpoint_sha256':file_sha256(checkpoint_path),'generator_model_state_sha256':checkpoint['model_state_sha256'],
        'candidate_coordinate_combined_sha256':canonical_json_sha256([r['tensor_hashes']['candidate_coords_xy_px'] for r in ordered]),
        'videos':ordered,
    }
    payload['cache_index_payload_sha256']=canonical_json_sha256(payload)
    index_path=out/'cache_index.json'; index_path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'stage':'pairwise_token_partition_complete','cache_index':str(index_path),'completed_count':len(completed),'expected_count':len(expected),'complete':payload['complete'],'candidate_coordinate_combined_sha256':payload['candidate_coordinate_combined_sha256']},indent=2),flush=True)

if __name__=='__main__': main()
