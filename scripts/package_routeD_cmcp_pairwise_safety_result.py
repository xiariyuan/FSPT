#!/usr/bin/env python3
"""Package and verify the formal P0h comparator-only result."""
from __future__ import annotations
import argparse,json,statistics,sys
from pathlib import Path
from typing import Any

REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_cache import CMCP_PAIRWISE_TOKEN_INDEX_SCHEMA_VERSION


def _read(path:Path)->dict[str,Any]: return json.loads(path.read_text())

def _threshold_gains(final:dict[str,Any],key:str)->dict[str,float]:
    return {t:100*(float(final[key][t])-float(final['native_metrics'][t])) for t in ('<1px','<2px','<4px','<8px','<16px')}


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--primary',default=str(REPO_ROOT/'outputs/routeD_cmcp_pairwise_safety_training_20260717/seed17/metrics.json'))
    ap.add_argument('--replay',default=str(REPO_ROOT/'outputs/routeD_cmcp_pairwise_safety_training_20260717/seed17_replay/metrics.json'))
    ap.add_argument('--fit-index',default=str(REPO_ROOT/'outputs/routeD_cmcp_pairwise_token_cache_20260717/fit/cache_index.json'))
    ap.add_argument('--validation-index',default=str(REPO_ROOT/'outputs/routeD_cmcp_pairwise_token_cache_20260717/model_validation/cache_index.json'))
    ap.add_argument('--config',default=str(REPO_ROOT/'configs/routeD_cmcp_local_pairwise_safety_v0.yaml'))
    ap.add_argument('--p0g-summary',default=str(REPO_ROOT/'docs/generated/ROUTED_CMCP_PROPOSAL_TRAINING_SUMMARY_2026-07-17.json'))
    ap.add_argument('--output',default=str(REPO_ROOT/'docs/generated/ROUTED_CMCP_PAIRWISE_SAFETY_TRAINING_SUMMARY_2026-07-17.json'))
    args=ap.parse_args(); paths={k:Path(v).resolve() for k,v in vars(args).items() if k!='output'}
    a=_read(paths['primary']); b=_read(paths['replay']); fit=_read(paths['fit_index']); val=_read(paths['validation_index']); p0g=_read(paths['p0g_summary'])
    fields=('model_state_sha256','best_epoch','config_sha256','fit_cache_index_sha256','validation_cache_index_sha256','fit_candidate_coordinate_combined_sha256','validation_candidate_coordinate_combined_sha256')
    checks={k:a[k]==b[k] for k in fields}
    checks.update({'model_config':a['model_config']==b['model_config'],'loss_config':a['loss_config']==b['loss_config'],'normalization':a['normalization']==b['normalization'],'optimizer':a['optimizer']==b['optimizer'],'initialization_validation':a['initialization_validation']==b['initialization_validation'],'history':a['history']==b['history'],'final_validation':a['final_validation']==b['final_validation'],'gate':a['gate']==b['gate'],'external_data_read':a['external_data_read']==b['external_data_read']})
    if not all(checks.values()): raise RuntimeError(f'P0h replay mismatch: {checks}')
    for name,index,expected in [('fit',fit,48),('model_validation',val,16)]:
        if index.get('schema_version')!=CMCP_PAIRWISE_TOKEN_INDEX_SCHEMA_VERSION or not index.get('complete') or index.get('completed_count')!=expected:
            raise RuntimeError(f'incomplete token cache {name}')
    if a['smoke_limits']!={'max_train_videos':0,'max_validation_videos':0} or b['smoke_limits']!={'max_train_videos':0,'max_validation_videos':0}: raise RuntimeError('not full scale')
    if any(a['external_data_read'].values()): raise RuntimeError('locked data read')
    final=a['final_validation']
    if final['candidate_coordinate_combined_sha256']!=val['candidate_coordinate_combined_sha256']: raise RuntimeError('candidate coordinate drift')
    p0g_final=p0g['best_model_validation']
    if final['oracle_gain_points']!=p0g_final['oracle_gain_points']: raise RuntimeError('P0h oracle differs from frozen P0g pool')
    selected=[float(r['selected_AJ_gain_points']) for r in final['per_video']]
    summary={
      'schema_version':'routeD_cmcp_pairwise_safety_training_summary_v0','date':'2026-07-17','status':'completed_positive_but_below_magnitude_gate',
      'formal_decision':a['gate']['decision'],
      'research_interpretation':'The local comparator is safe and statistically positive, but beneficial recall and AJ magnitude remain below the preregistered gate.',
      'claim_boundary':'Kubric fit/model-validation comparator-only result; not calibration, final holdout, DAVIS, or Kinetics evidence.',
      'frozen_contract':{
        'fit_candidate_coordinate_combined_sha256':fit['candidate_coordinate_combined_sha256'],
        'validation_candidate_coordinate_combined_sha256':val['candidate_coordinate_combined_sha256'],
        'candidate_oracle_unchanged_from_p0g':True,
        'generator_model_state_sha256':fit['generator_model_state_sha256'],
        'config_path':str(paths['config']),'config_sha256':file_sha256(paths['config']),
      },
      'training':{
        'seed':a['seed'],'best_epoch':a['best_epoch'],'epochs_executed':len(a['history']),'model_state_sha256':a['model_state_sha256'],
        'checkpoint_sha256':a['checkpoint_sha256'],'metrics_sha256':file_sha256(paths['primary']),'exact_seed_replay':all(checks.values()),
        'replay_model_state_sha256':b['model_state_sha256'],'replay_checkpoint_sha256':b['checkpoint_sha256'],'replay_metrics_sha256':file_sha256(paths['replay']),
        'checkpoint_rule':a['optimizer']['checkpoint_rule'],'normalization':a['normalization'],'optimizer':a['optimizer'],
      },
      'best_model_validation':{
        'native_metrics':final['native_metrics'],'selected_metrics':final['selected_metrics'],'oracle_metrics':final['oracle_metrics'],
        'selected_gain_points':final['selected_gain_points'],'oracle_gain_points':final['oracle_gain_points'],
        'selected_threshold_gain_points':_threshold_gains(final,'selected_metrics'),'oracle_threshold_gain_points':_threshold_gains(final,'oracle_metrics'),
        'paired_video_selected_AJ_gain_CI':final['paired_video_selected_AJ_gain_CI'],'paired_video_oracle_AJ_gain_CI':final['paired_video_oracle_AJ_gain_CI'],
        'paired_video_selected_delta_gain_CI':final['paired_video_selected_delta_gain_CI'],'severe_16px_rate':final['severe_16px_rate'],'behavior':final['behavior'],
        'per_video_summary':{'min':min(selected),'median':statistics.median(selected),'mean':statistics.mean(selected),'max':max(selected),'positive_videos':sum(v>0 for v in selected),'negative_videos':sum(v<0 for v in selected),'videos':len(selected)},
        'per_video':final['per_video'],
      },
      'gate':a['gate'],
      'diagnosis':{
        'safety_gate_passes':a['gate']['harmful_non_native_rate_at_most_0_01'],'paired_consistency_gate_passes':a['gate']['paired_top1_AJ_CI_lower_positive'],
        'magnitude_gate_passes':a['gate']['direct_top1_AJ_gain_at_least_0_5'],'candidate_quality_is_bottleneck':False,
        'safe_beneficial_recall_is_bottleneck':True,'next_route':'zero-initialized rank-32 late feature metric residual adapter with frozen native trajectory',
      },
      'external_data_read':a['external_data_read'],
    }
    output=Path(args.output).resolve(); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'output':str(output),'sha256':file_sha256(output),'model_state_sha256':a['model_state_sha256'],'AJ_gain_points':final['selected_gain_points']['AJ'],'AJ_CI':final['paired_video_selected_AJ_gain_CI'],'harmful_non_native_rate':final['behavior']['harmful_non_native_rate'],'formal_decision':a['gate']['decision']},indent=2))

if __name__=='__main__': main()
