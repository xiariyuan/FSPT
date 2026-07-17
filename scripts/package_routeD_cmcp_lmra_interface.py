#!/usr/bin/env python3
"""Package and verify the P0i LMRA zero-step interface audit."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Any

REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_late_metric_adapter import (
    LMRA_SCHEMA_VERSION, LMRA_TRAINABLE_PARAMETERS,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256


def _read(path:Path)->dict[str,Any]: return json.loads(path.read_text())


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--primary',default=str(REPO_ROOT/'outputs/routeD_cmcp_lmra_20260717/interface_smoke.json'))
    ap.add_argument('--replay',default=str(REPO_ROOT/'outputs/routeD_cmcp_lmra_20260717/interface_smoke_replay.json'))
    ap.add_argument('--config',default=str(REPO_ROOT/'configs/routeD_cmcp_lmra_v0.yaml'))
    ap.add_argument('--output',default=str(REPO_ROOT/'docs/generated/ROUTED_CMCP_LMRA_INTERFACE_SUMMARY_2026-07-17.json'))
    args=ap.parse_args(); primary_path=Path(args.primary).resolve(); replay_path=Path(args.replay).resolve(); config_path=Path(args.config).resolve()
    a=_read(primary_path); b=_read(replay_path)
    fields=('adapter_schema_version','source_partition','source_index','video_name','points','frames','candidate_count','config_sha256','feature_index_sha256','token_index_sha256','base_sidecar_sha256','cmcp_checkpoint_sha256','cmcp_model_state_sha256','comparator_checkpoint_sha256','comparator_model_state_sha256','adapter_state_sha256','trainable_parameters','total_trainable_parameters','zero_step_formal_p0h_checks','in_process_replay_checks','tensor_hashes','integrity')
    checks={key:a[key]==b[key] for key in fields}
    if not all(checks.values()): raise RuntimeError(f'LMRA interface replay mismatch: {checks}')
    if a['adapter_schema_version']!=LMRA_SCHEMA_VERSION: raise RuntimeError('LMRA schema mismatch')
    if a['config_sha256']!=file_sha256(config_path): raise RuntimeError('LMRA config hash mismatch')
    if a['trainable_parameters']['lmra']!=LMRA_TRAINABLE_PARAMETERS: raise RuntimeError('LMRA parameter count mismatch')
    if not a['zero_step_formal_p0h_exact'] or not a['in_process_replay_exact']: raise RuntimeError('LMRA zero-step/replay failed')
    if a['tensor_hashes']['frozen_feature_maps']!=a['tensor_hashes']['adapted_feature_maps']: raise RuntimeError('feature-map byte identity failed')
    locked=('validation_read','calibration_read','final_holdout_read','tapvid_davis_read','tapvid_kinetics_read')
    if any(a['integrity'][key] for key in locked): raise RuntimeError('locked data read')
    if a['integrity']['native_trajectory_modified'] or a['integrity']['candidate_zero_modified'] or a['integrity']['ground_truth_used_for_adapter_or_selection']:
        raise RuntimeError('LMRA integrity failure')
    summary={
      'schema_version':'routeD_cmcp_lmra_interface_summary_v0','date':'2026-07-17','status':'completed_pass',
      'decision':'ALLOW_LMRA_FIT_ONLY_JOINT_TRAINING',
      'claim_boundary':'Zero-step interface result on one authorized fit video; not learned LMRA performance.',
      'adapter':{
        'schema_version':a['adapter_schema_version'],'rank':32,'feature_dim':128,'trainable_parameters':a['trainable_parameters']['lmra'],
        'state_sha256':a['adapter_state_sha256'],'config_path':str(config_path),'config_sha256':a['config_sha256'],
        'frozen_feature_sha256':a['tensor_hashes']['frozen_feature_maps'],'adapted_feature_sha256':a['tensor_hashes']['adapted_feature_maps'],
        'byte_exact_identity':True,
      },
      'initialized_components':{
        'cmcp_model_state_sha256':a['cmcp_model_state_sha256'],'cmcp_trainable_parameters':a['trainable_parameters']['cmcp'],
        'comparator_model_state_sha256':a['comparator_model_state_sha256'],'comparator_trainable_parameters':a['trainable_parameters']['comparator'],
        'total_joint_trainable_parameters':a['total_trainable_parameters'],
      },
      'sample':{'partition':a['source_partition'],'source_index':a['source_index'],'video_name':a['video_name'],'points':a['points'],'frames':a['frames'],'candidate_count':a['candidate_count']},
      'audit':{
        'zero_step_formal_p0h_exact':True,'independent_full_video_replay_exact':True,'replay_checks':checks,
        'candidate_coordinate_sha256':a['tensor_hashes']['candidate_coords_xy_px'],'candidate_token_sha256':a['tensor_hashes']['candidate_tokens'],
        'selected_coordinate_sha256':a['tensor_hashes']['selected_coord_xy_px'],'tensor_hashes':a['tensor_hashes'],
      },
      'artifacts':{'primary_report_sha256':file_sha256(primary_path),'primary_sidecar_sha256':a['sidecar_sha256'],'replay_report_sha256':file_sha256(replay_path),'replay_sidecar_sha256':b['sidecar_sha256'],'sidecar_hash_expected_to_differ_by_embedded_path':True},
      'next_gate':{'training':'LMRA + initialized CMCP + initialized comparator on fit only','checkpoint_selection':'complete model_validation only','native_tracker_gradients_authorized':False,'rank_or_layer_sweep_authorized':False},
      'integrity':a['integrity'],
    }
    output=Path(args.output).resolve(); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'output':str(output),'sha256':file_sha256(output),'feature_identity_sha256':a['tensor_hashes']['adapted_feature_maps'],'candidate_coordinate_sha256':a['tensor_hashes']['candidate_coords_xy_px'],'total_trainable_parameters':a['total_trainable_parameters'],'decision':summary['decision']},indent=2))

if __name__=='__main__': main()
