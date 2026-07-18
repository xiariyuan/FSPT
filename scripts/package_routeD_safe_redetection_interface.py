#!/usr/bin/env python3
"""Package and verify the Route-D safe-redetection interface milestone."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from projects.mmp_tracker.mmp_tracker.routeD_safe_redetection import SafeRedetectionConfig,SafeRedetectionModel
from projects.mmp_tracker.mmp_tracker.routeD_safe_redetection_cache import file_sha256


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default=str(REPO_ROOT/'configs/routeD_safe_redetection_v0.yaml'))
    ap.add_argument('--protocol',default=str(REPO_ROOT/'configs/routeD_safe_redetection_pointodyssey_protocol_v0.json'))
    ap.add_argument('--baseline',default=str(REPO_ROOT/'docs/generated/OFFICIAL_COTRACKER3_DAVIS_FIRST_ALIGNMENT_2026-07-18.json'))
    ap.add_argument('--ajrd-parity',default=str(REPO_ROOT/'docs/generated/OFFICIAL_TAPNEXTPP_AJRD_PARITY_2026-07-18.json'))
    ap.add_argument('--runtime-alignment',default=str(REPO_ROOT/'docs/generated/ROUTED_SAFE_REDETECTION_RUNTIME_ALIGNMENT_2026-07-18.json'))
    ap.add_argument('--output',default=str(REPO_ROOT/'docs/generated/ROUTED_SAFE_REDETECTION_INTERFACE_SUMMARY_2026-07-18.json'))
    args=ap.parse_args()
    paths={key:Path(value).resolve() for key,value in vars(args).items() if key!='output'}
    config=yaml.safe_load(paths['config'].read_text())
    protocol=json.loads(paths['protocol'].read_text())
    baseline=json.loads(paths['baseline'].read_text())
    parity=json.loads(paths['ajrd_parity'].read_text())
    runtime=json.loads(paths['runtime_alignment'].read_text())
    prerequisites=protocol['official_prerequisites']
    prerequisite_paths={
        key:(REPO_ROOT/Path(prerequisites[key])).resolve()
        for key in ('cotracker3_alignment_summary','official_ajrd_parity')
    }
    for key,path in prerequisite_paths.items():
        if Path(prerequisites[key]).is_absolute():
            raise RuntimeError(f'protocol prerequisite must be repository-relative: {key}')
        try: path.relative_to(REPO_ROOT)
        except ValueError as exc: raise RuntimeError(f'protocol prerequisite escapes repository: {key}') from exc
    checks={
        'official_baseline_pass':baseline['gate']['pass'],
        'official_AJ_RD_parity_pass':parity['pass'],
        'official_AJ_RD_zero_difference':parity['maximum_absolute_scalar_difference']==0.0,
        'portable_official_evidence_paths':all(not Path(prerequisites[key]).is_absolute() for key in prerequisite_paths),
        'protocol_baseline_hash_exact':file_sha256(prerequisite_paths['cotracker3_alignment_summary'])==prerequisites['cotracker3_alignment_summary_sha256']==file_sha256(paths['baseline']),
        'protocol_AJ_RD_hash_exact':file_sha256(prerequisite_paths['official_ajrd_parity'])==prerequisites['official_ajrd_parity_sha256']==file_sha256(paths['ajrd_parity']),
        'runtime_alignment_pass':runtime['pass'],
        'runtime_visibility_exact':runtime['visibility_exact'],
        'runtime_zero_intervention':all(row['interventions']==0 and row['writes']==0 for row in runtime['runtime_diagnostics']),
        'fit_events_240':protocol['partitions']['fit']['counts']['total']==240,
        'validation_events_160':protocol['partitions']['model_validation']['counts']['total']==160,
        'internal_holdout_unread':not protocol['partitions']['locked_internal_holdout']['events_parsed'],
        'pointodyssey_test_unread':not protocol['partitions']['locked_pointodyssey_test']['events_parsed'],
        'kinetics_locked':'frozen' in protocol['data_exposure']['tapvid_kinetics_official_1144'].lower() and 'not rerun' in protocol['data_exposure']['tapvid_kinetics_official_1144'].lower(),
    }
    if not all(checks.values()): raise RuntimeError(checks)
    model_cfg=dict(config['model']); model_cfg.update(input_height=256,input_width=256)
    model=SafeRedetectionModel(SafeRedetectionConfig(**model_cfg))
    proposal_parameters=sum(p.numel() for p in model.proposal.parameters())
    comparator_parameters=sum(p.numel() for p in model.comparator.parameters())
    total_parameters=sum(p.numel() for p in model.parameters())
    summary={
        'schema_version':'routeD_safe_redetection_interface_summary_v0',
        'date':'2026-07-18',
        'status':'completed_authorize_committed_event_cache_export',
        'formal_decision':'ALLOW_COMMITTED_POINTODYSSEY_EVENT_CACHE_EXPORT',
        'research_direction':'risk-controlled long-occlusion re-detection for a frozen point tracker',
        'claim_boundary':'CoTracker3 instantiation; a second distinct backbone is required for a general plug-in claim.',
        'checks':checks,
        'official_alignment':{
            'source_commit':baseline['official_source']['commit'],
            'checkpoint_sha256':baseline['checkpoint']['sha256'],
            'DAVIS_first_AJ':baseline['metrics']['average_jaccard'],
            'DAVIS_first_OA':baseline['metrics']['occlusion_accuracy'],
            'DAVIS_first_delta_average':baseline['metrics']['average_pts_within_thresh'],
            'single_point':baseline['frozen_config']['single_point'],
            'summary_sha256':file_sha256(paths['baseline']),
        },
        'official_AJ_RD':{
            'source_commit':parity['official_commit'],
            'source_sha256':parity['official_source_sha256'],
            'randomized_cases':parity['cases'],
            'maximum_absolute_difference':parity['maximum_absolute_scalar_difference'],
            'summary_sha256':file_sha256(paths['ajrd_parity']),
        },
        'runtime_alignment':{
            'real_queries':runtime['query_count'],
            'track_max_abs_px':runtime['track_max_abs_px'],
            'track_mean_abs_px':runtime['track_mean_abs_px'],
            'visibility_exact':runtime['visibility_exact'],
            'interventions':sum(row['interventions'] for row in runtime['runtime_diagnostics']),
            'writes':sum(row['writes'] for row in runtime['runtime_diagnostics']),
            'summary_sha256':file_sha256(paths['runtime_alignment']),
        },
        'model':{
            'proposal_parameters':proposal_parameters,
            'comparator_parameters':comparator_parameters,
            'total_parameters':total_parameters,
            'config':model.config.__dict__,
            'native_tracker_trainable':False,
            'candidate_zero_native_exact':True,
            'coordinate_visibility_joint_recovery':True,
            'temporal_confirmation_frames':model.config.confirmation_frames,
            'future_only_bounded_writeback':True,
        },
        'pointodyssey_protocol':{
            'fit_scenes':len(protocol['partitions']['fit']['scenes']),
            'fit_events':protocol['partitions']['fit']['counts']['total'],
            'model_validation_scenes':len(protocol['partitions']['model_validation']['scenes']),
            'model_validation_events':protocol['partitions']['model_validation']['counts']['total'],
            'locked_internal_holdout_scenes':len(protocol['partitions']['locked_internal_holdout']['scenes']),
            'locked_test_scenes':len(protocol['partitions']['locked_pointodyssey_test']['scenes']),
            'protocol_sha256':file_sha256(paths['protocol']),
            'payload_sha256':protocol['payload_sha256'],
        },
        'config_sha256':file_sha256(paths['config']),
        'external_data_read':{
            'pointodyssey_internal_holdout':False,
            'pointodyssey_test':False,
            'tapvid_kinetics_official_1144':False,
        },
    }
    output=Path(args.output).resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(summary,indent=2,allow_nan=True)+'\n')
    print(json.dumps({'output':str(output),'sha256':file_sha256(output),'parameters':summary['model'],'decision':summary['formal_decision']},indent=2))

if __name__=='__main__': main()
