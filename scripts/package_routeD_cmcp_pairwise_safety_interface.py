#!/usr/bin/env python3
"""Package and verify the P0h local pairwise safety interface audit."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any

REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_safety import (
    CMCP_PAIRWISE_LOCAL_TOKEN_DIM, CMCP_PAIRWISE_SAFETY_SCHEMA_VERSION,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256


def _read(path:Path)->dict[str,Any]: return json.loads(path.read_text())


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--primary',default=str(REPO_ROOT/'outputs/routeD_cmcp_pairwise_safety_20260717/interface_smoke.json'))
    ap.add_argument('--replay',default=str(REPO_ROOT/'outputs/routeD_cmcp_pairwise_safety_20260717/interface_smoke_replay.json'))
    ap.add_argument('--config',default=str(REPO_ROOT/'configs/routeD_cmcp_local_pairwise_safety_v0.yaml'))
    ap.add_argument('--output',default=str(REPO_ROOT/'docs/generated/ROUTED_CMCP_PAIRWISE_SAFETY_INTERFACE_SUMMARY_2026-07-17.json'))
    args=ap.parse_args()
    primary_path=Path(args.primary).resolve(); replay_path=Path(args.replay).resolve(); config_path=Path(args.config).resolve()
    a=_read(primary_path); b=_read(replay_path)
    exact_fields=(
        'comparator_schema_version','source_partition','source_index','video_name','points','frames',
        'candidate_count','local_token_dim','comparator_trainable_parameters','config_sha256',
        'frozen_generator_checkpoint_sha256','frozen_generator_model_state_sha256','feature_index_sha256',
        'base_sidecar_sha256','candidate_zero_native_parity','zero_step_selected_native_all_active',
        'generator_parameters_frozen','in_process_replay_checks','tensor_hashes','integrity',
    )
    checks={key:a[key]==b[key] for key in exact_fields}
    if not all(checks.values()): raise RuntimeError(f'P0h interface replay mismatch: {checks}')
    if a['comparator_schema_version']!=CMCP_PAIRWISE_SAFETY_SCHEMA_VERSION: raise RuntimeError('schema mismatch')
    if a['local_token_dim']!=CMCP_PAIRWISE_LOCAL_TOKEN_DIM: raise RuntimeError('token dimension mismatch')
    if a['config_sha256']!=file_sha256(config_path): raise RuntimeError('config hash mismatch')
    if not a['candidate_zero_native_parity'] or not a['zero_step_selected_native_all_active']:
        raise RuntimeError('native-safe interface failed')
    if not a['generator_parameters_frozen']: raise RuntimeError('generator is not frozen')
    locked=('calibration_read','final_holdout_read','tapvid_davis_read','tapvid_kinetics_read')
    if any(a['integrity'][key] for key in locked): raise RuntimeError('locked data were read')
    if a['integrity']['ground_truth_used_for_tokens_or_selection']: raise RuntimeError('GT contamination')
    if a['integrity']['candidate_coordinates_modified']: raise RuntimeError('candidate coordinates modified')
    summary={
        'schema_version':'routeD_cmcp_pairwise_safety_interface_summary_v0',
        'date':'2026-07-17',
        'status':'completed_pass',
        'decision':'ALLOW_FROZEN_CMCP_TOKEN_CACHE_AND_COMPARATOR_ONLY_TRAINING',
        'claim_boundary':'Zero-step interface result on one authorized fit video; not learned comparator performance.',
        'frozen_generator':{
            'checkpoint_sha256':a['frozen_generator_checkpoint_sha256'],
            'model_state_sha256':a['frozen_generator_model_state_sha256'],
            'parameters_frozen':True,
            'candidate_coordinate_sha256':a['tensor_hashes']['candidate_coords_xy_px'],
        },
        'comparator':{
            'schema_version':a['comparator_schema_version'],
            'local_token_dim':a['local_token_dim'],
            'trainable_parameters':a['comparator_trainable_parameters'],
            'config_path':str(config_path),
            'config_sha256':a['config_sha256'],
            'zero_step_native_safe':True,
        },
        'sample':{
            'partition':a['source_partition'],'source_index':a['source_index'],'video_name':a['video_name'],
            'points':a['points'],'frames':a['frames'],'candidate_count':a['candidate_count'],
        },
        'audit':{
            'candidate_zero_native_parity':True,
            'zero_step_selected_native_all_active':True,
            'in_process_replay_exact':True,
            'independent_full_video_tensor_replay_exact':True,
            'tensor_hashes':a['tensor_hashes'],
            'replay_checks':checks,
        },
        'artifacts':{
            'primary_report_sha256':file_sha256(primary_path),
            'primary_sidecar_sha256':a['sidecar_sha256'],
            'replay_report_sha256':file_sha256(replay_path),
            'replay_sidecar_sha256':b['sidecar_sha256'],
            'sidecar_hash_expected_to_differ_by_embedded_path':True,
        },
        'next_gate':{
            'step':'export frozen local candidate tokens for fit and model_validation',
            'training':'comparator-only on fit',
            'model_validation':'checkpoint selection and formal gates only',
            'candidate_generator_updates_authorized':False,
        },
        'integrity':a['integrity'],
    }
    output=Path(args.output).resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'output':str(output),'sha256':file_sha256(output),'candidate_coordinate_sha256':a['tensor_hashes']['candidate_coords_xy_px'],'trainable_parameters':a['comparator_trainable_parameters'],'decision':summary['decision']},indent=2))

if __name__=='__main__': main()
