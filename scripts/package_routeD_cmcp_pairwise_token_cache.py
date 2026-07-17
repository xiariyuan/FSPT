#!/usr/bin/env python3
"""Package and verify complete P0h frozen local-token caches."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Any

REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_cache import (
    CMCP_PAIRWISE_TOKEN_INDEX_SCHEMA_VERSION,verify_pairwise_token_artifact,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256


def _read(path:Path)->dict[str,Any]: return json.loads(path.read_text())


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--fit-index',default=str(REPO_ROOT/'outputs/routeD_cmcp_pairwise_token_cache_20260717/fit/cache_index.json'))
    ap.add_argument('--validation-index',default=str(REPO_ROOT/'outputs/routeD_cmcp_pairwise_token_cache_20260717/model_validation/cache_index.json'))
    ap.add_argument('--config',default=str(REPO_ROOT/'configs/routeD_cmcp_local_pairwise_safety_v0.yaml'))
    ap.add_argument('--interface-summary',default=str(REPO_ROOT/'docs/generated/ROUTED_CMCP_PAIRWISE_SAFETY_INTERFACE_SUMMARY_2026-07-17.json'))
    ap.add_argument('--output',default=str(REPO_ROOT/'docs/generated/ROUTED_CMCP_PAIRWISE_TOKEN_CACHE_SUMMARY_2026-07-17.json'))
    args=ap.parse_args()
    fit_path=Path(args.fit_index).resolve(); val_path=Path(args.validation_index).resolve(); config_path=Path(args.config).resolve()
    fit=_read(fit_path); val=_read(val_path); interface=_read(Path(args.interface_summary).resolve())
    for name,index,expected in [('fit',fit,48),('model_validation',val,16)]:
        if index.get('schema_version')!=CMCP_PAIRWISE_TOKEN_INDEX_SCHEMA_VERSION: raise RuntimeError(f'schema mismatch {name}')
        if not index.get('complete') or index.get('completed_count')!=expected: raise RuntimeError(f'incomplete {name}')
        if index.get('expected_source_indices')!=index.get('completed_source_indices'): raise RuntimeError(f'membership mismatch {name}')
        for row in index['videos']:
            verify_pairwise_token_artifact(row['sidecar'],expected_generator_state_sha256=index['generator_model_state_sha256'],expected_base_sidecar_sha256=row['base_sidecar_sha256'])
            if row['integrity']['ground_truth_used_for_tokens'] or row['integrity']['candidate_coordinates_modified']:
                raise RuntimeError(f'integrity failure {name}/{row["source_index"]}')
    anchors=[]
    for name,index,ids in [('fit',fit,{0,47}),('model_validation',val,{48,63})]:
        for row in index['videos']:
            if row['source_index'] in ids:
                anchors.append({'partition':name,'source_index':row['source_index'],'exact':row['deterministic_replay']['exact'],'candidate_coordinate_sha256':row['tensor_hashes']['candidate_coords_xy_px']})
    if len(anchors)!=4 or not all(a['exact'] for a in anchors): raise RuntimeError('replay anchors incomplete')
    if anchors[0]['candidate_coordinate_sha256']!=interface['frozen_generator']['candidate_coordinate_sha256']:
        raise RuntimeError('fit0 candidate hash differs from interface')
    summary={
        'schema_version':'routeD_cmcp_pairwise_token_cache_summary_v0','date':'2026-07-17','status':'completed_pass',
        'decision':'ALLOW_PAIRWISE_COMPARATOR_ONLY_TRAINING',
        'claim_boundary':'Frozen token-cache integrity result; not learned comparator performance.',
        'contract':{
            'local_token_dim':88,'static_token_dim':84,'dynamic_causal_summary_dim':4,
            'dynamic_summary_placeholder_zero':True,'generator_model_state_sha256':fit['generator_model_state_sha256'],
            'config_path':str(config_path),'config_sha256':file_sha256(config_path),
        },
        'partitions':{
            'fit':{'videos':fit['completed_count'],'index_sha256':file_sha256(fit_path),'payload_sha256':fit['cache_index_payload_sha256'],'candidate_coordinate_combined_sha256':fit['candidate_coordinate_combined_sha256']},
            'model_validation':{'videos':val['completed_count'],'index_sha256':file_sha256(val_path),'payload_sha256':val['cache_index_payload_sha256'],'candidate_coordinate_combined_sha256':val['candidate_coordinate_combined_sha256']},
        },
        'replay_anchors':anchors,
        'integrity':{
            'all_generator_states_frozen':True,'all_candidate_coordinates_unmodified':True,'all_static_tokens_ground_truth_free':True,
            'calibration_read':False,'final_holdout_read':False,'tapvid_davis_read':False,'tapvid_kinetics_read':False,
        },
        'next_gate':{
            'training':'comparator-only on fit','checkpoint_selection':'complete model_validation only',
            'candidate_coordinate_hashes_must_remain_exact':True,'generator_updates_authorized':False,
        },
    }
    output=Path(args.output).resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'output':str(output),'sha256':file_sha256(output),'fit_candidate_hash':fit['candidate_coordinate_combined_sha256'],'validation_candidate_hash':val['candidate_coordinate_combined_sha256'],'decision':summary['decision']},indent=2))

if __name__=='__main__': main()
