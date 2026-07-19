#!/usr/bin/env python3
"""Fit-0 exact compatibility audit for the sealed P0l cache format."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_bounded_writeback import (
    normalize_adapted_feature_maps,
)
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_final_holdout import load_final_holdout_config
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_feature_cache import load_complete_feature_index
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_training import load_cmcp_video
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256
from scripts.audit_routeD_cmcp_bounded_writeback_interface import _predict_from_fixed_trajectory
from scripts.eval_routeD_cmcp_final_holdout import _load_model

DEFAULT_CONFIG=REPO_ROOT/'configs/routeD_cmcp_final_holdout_v0.yaml'
DEFAULT_SMOKE=REPO_ROOT/'outputs/routeD_cmcp_final_holdout_20260719/implementation_smoke/cache_index.json'
DEFAULT_REFERENCE=REPO_ROOT/'outputs/routeD_cmcp_feature_cache_20260717/fit/cache_index.json'
DEFAULT_OUTPUT=REPO_ROOT/'docs/generated/ROUTED_STRONG_BACKBONE_FINAL_HOLDOUT_SMOKE_SUMMARY_2026-07-19.json'


def _compare(left:torch.Tensor,right:torch.Tensor):
    exact=torch.equal(left,right)
    max_abs=float((left.float()-right.float()).abs().max().item()) if left.dtype.is_floating_point and left.shape==right.shape else None
    return {'exact':exact,'max_abs':max_abs,'left_sha256':tensor_sha256(left),'right_sha256':tensor_sha256(right)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(DEFAULT_CONFIG)); ap.add_argument('--smoke-index',default=str(DEFAULT_SMOKE)); ap.add_argument('--reference-index',default=str(DEFAULT_REFERENCE)); ap.add_argument('--output',default=str(DEFAULT_OUTPUT)); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); args=ap.parse_args()
    config=load_final_holdout_config(args.config)
    smoke_index=load_complete_feature_index(args.smoke_index,expected_partition='implementation_smoke')
    ref_index=load_complete_feature_index(args.reference_index,expected_partition='fit')
    smoke_row=smoke_index['videos'][0]; ref_row=next(row for row in ref_index['videos'] if int(row['source_index'])==0)
    smoke=load_cmcp_video(smoke_row); reference=load_cmcp_video(ref_row)
    adapter,cmcp,comparator,normalization,_,combined_sha=_load_model(config,args.device)
    tensor_keys=('native_coords_xy_px','native_visibility_probability','native_confidence_probability','native_joint_probability','native_visibility','query_points_tyx','gt_tracks_yx','gt_occluded')
    base_checks={key:_compare(smoke.tensors[key],reference.tensors[key]) for key in tensor_keys}
    feature_check=_compare(smoke.feature_maps,reference.feature_maps)
    with torch.no_grad():
        smoke_maps=normalize_adapted_feature_maps(adapter,smoke.feature_maps.to(args.device,dtype=torch.float32))
        ref_maps=normalize_adapted_feature_maps(adapter,reference.feature_maps.to(args.device,dtype=torch.float32))
    def predict(bundle,maps):
        return _predict_from_fixed_trajectory(
            adapted_maps=maps,
            native=bundle.tensors['native_coords_xy_px'].float(),
            vis=bundle.tensors['native_visibility_probability'].float(),
            conf=bundle.tensors['native_confidence_probability'].float(),
            queries=bundle.tensors['query_points_tyx'].float(),
            cmcp=cmcp,comparator=comparator,normalization=normalization,point_batch_size=4,
        )
    left=predict(smoke,smoke_maps); right=predict(reference,ref_maps)
    output_checks={key:_compare(left[key],right[key]) for key in ('candidate_coords_xy_px','candidate_valid_mask','selected_candidate_index','selected_coords_xy_px','decision_summary')}
    checks={'base':base_checks,'feature_maps_f16':feature_check,'variant_C_outputs':output_checks}
    exact=all(row['exact'] for row in base_checks.values()) and feature_check['exact'] and all(row['exact'] for row in output_checks.values())
    if not exact: raise RuntimeError(f'P0l smoke mismatch: {checks}')
    summary={
        'schema_version':'routeD_cmcp_final_holdout_smoke_summary_v0','date':'2026-07-19','status':'completed_pass','decision':'ALLOW_P0L_FINAL_HOLDOUT_CACHE_BUILD',
        'source_index':0,'video_name':smoke.video_name,'config_sha256':config['_config_sha256'],'combined_model_state_sha256':combined_sha,
        'smoke_index_sha256':file_sha256(Path(args.smoke_index)),'reference_index_sha256':file_sha256(Path(args.reference_index)),
        'checks':checks,'performance_metrics_computed':False,'calibration_read':False,'final_holdout_read':False,'external_data_read':{'tapvid_davis':False,'tapvid_kinetics':False},
    }
    output=Path(args.output).resolve(); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'output':str(output),'sha256':file_sha256(output),'decision':summary['decision'],'exact':exact,'performance_metrics_computed':False},indent=2))

if __name__=='__main__': main()
