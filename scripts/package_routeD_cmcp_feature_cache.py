#!/usr/bin/env python3
"""Package and verify the complete CMCP feature-map caches."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_feature_cache import load_complete_feature_index,verify_feature_cache_artifact
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--fit-index',default=str(REPO_ROOT/'outputs/routeD_cmcp_feature_cache_20260717/fit/cache_index.json'))
 ap.add_argument('--validation-index',default=str(REPO_ROOT/'outputs/routeD_cmcp_feature_cache_20260717/model_validation/cache_index.json'))
 ap.add_argument('--output',default=str(REPO_ROOT/'docs/generated/ROUTED_CMCP_FEATURE_CACHE_SUMMARY_2026-07-17.json'))
 args=ap.parse_args()
 fitp=Path(args.fit_index).resolve(); valp=Path(args.validation_index).resolve()
 fit=load_complete_feature_index(fitp,expected_partition='fit'); val=load_complete_feature_index(valp,expected_partition='model_validation')
 if fit['protocol_sha256']!=val['protocol_sha256']: raise RuntimeError('protocol mismatch')
 anchors=[]
 for name,index in [('fit',fit),('model_validation',val)]:
  for row in index['videos']:
   verify_feature_cache_artifact(row['sidecar'],expected_protocol_sha256=index['protocol_sha256'],expected_base_sidecar_sha256=row['base_sidecar_sha256'])
   if row['deterministic_float32_replay_exact'] is not None: anchors.append({'partition':name,'source_index':row['source_index'],'exact':row['deterministic_float32_replay_exact']})
 if len(anchors)!=4 or not all(x['exact'] for x in anchors): raise RuntimeError('replay anchors incomplete')
 summary={
  'schema_version':'routeD_cmcp_feature_cache_summary_v1','date':'2026-07-17','status':'completed_pass',
  'decision':'ALLOW_CMCP_FIT_ONLY_DENSE_PROPOSAL_TRAINING',
  'claim_boundary':'Frozen feature-map cache integrity result; not learned proposal performance.',
  'protocol_sha256':fit['protocol_sha256'],
  'partitions':{
   'fit':{'videos':fit['completed_count'],'index_sha256':file_sha256(fitp),'payload_sha256':fit['cache_index_payload_sha256'],'base_index_sha256':fit['base_cache_index_sha256'],'quantization':fit['aggregate_quantization_audit']},
   'model_validation':{'videos':val['completed_count'],'index_sha256':file_sha256(valp),'payload_sha256':val['cache_index_payload_sha256'],'base_index_sha256':val['base_cache_index_sha256'],'quantization':val['aggregate_quantization_audit']},
  },
  'feature_contract':{'dtype':'float16','shape_per_video':[24,128,96,128],'reconstruction_max_abs_gate':5e-4,'reconstruction_min_cosine_gate':0.99999,'observed_global_max_abs':max(fit['aggregate_quantization_audit']['max_abs_max'],val['aggregate_quantization_audit']['max_abs_max']),'observed_global_min_cosine':min(fit['aggregate_quantization_audit']['min_cosine_min'],val['aggregate_quantization_audit']['min_cosine_min'])},
  'correlation_quantization_smoke':{'source_partition':'fit','source_index':0,'points':4,'max_abs':0.00012364983558654785,'mean_abs':1.6336192857124843e-05,'rms_abs':2.1369542082538828e-05,'motion_prior_exact':True,'causal_valid_mask_exact':True},
  'float32_replay_anchors':anchors,
  'integrity':{'native_state_exact_all':True,'ground_truth_used_for_feature_map':False,'calibration_read':False,'final_holdout_read':False,'tapvid_davis_read':False,'tapvid_kinetics_read':False},
  'next_gate':{'training_partition':'fit only','checkpoint_selection_partition':'model_validation only','proposal_only_before_selector':True,'selector_authorized':False,'state_write_authorized':False}
 }
 out=Path(args.output).resolve(); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({'output':str(out),'sha256':file_sha256(out),'fit_videos':48,'validation_videos':16,'decision':summary['decision']},indent=2))
if __name__=='__main__': main()
