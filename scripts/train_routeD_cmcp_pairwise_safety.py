#!/usr/bin/env python3
"""Formal P0h comparator-only training on frozen CMCP local tokens."""
from __future__ import annotations
import argparse,json,os,random,sys
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
import yaml

REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0,str(REPO_ROOT))
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_cache import load_complete_pairwise_token_index
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_safety import CMCPLocalPairwiseSafetyComparator,CMCPLocalSafetyConfig
from projects.mmp_tracker.mmp_tracker.routeD_cmcp_pairwise_training import (
 PairwiseSafetyLossConfig,StaticTokenNormalization,compute_static_token_normalization,evaluate_pairwise_index,
 safety_feasible,train_pairwise_epoch,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256
from projects.mmp_tracker.mmp_tracker.routeD_musr_training import state_dict_sha256

DEFAULT_CONFIG=REPO_ROOT/'configs/routeD_cmcp_local_pairwise_safety_v0.yaml'
DEFAULT_FIT=REPO_ROOT/'outputs/routeD_cmcp_pairwise_token_cache_20260717/fit/cache_index.json'
DEFAULT_VAL=REPO_ROOT/'outputs/routeD_cmcp_pairwise_token_cache_20260717/model_validation/cache_index.json'
DEFAULT_OUTPUT=REPO_ROOT/'outputs/routeD_cmcp_pairwise_safety_training_20260717/seed17'


def deterministic(seed:int):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
 torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
 torch.use_deterministic_algorithms(True,warn_only=False)


def clone_state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(DEFAULT_CONFIG)); ap.add_argument('--fit-index',default=str(DEFAULT_FIT)); ap.add_argument('--validation-index',default=str(DEFAULT_VAL)); ap.add_argument('--output',default=str(DEFAULT_OUTPUT)); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
 ap.add_argument('--epochs',type=int,default=0); ap.add_argument('--patience',type=int,default=0); ap.add_argument('--batch-size',type=int,default=0); ap.add_argument('--eval-batch-size',type=int,default=0); ap.add_argument('--bootstrap-samples',type=int,default=0); ap.add_argument('--seed',type=int,default=17); ap.add_argument('--max-train-videos',type=int,default=0); ap.add_argument('--max-validation-videos',type=int,default=0)
 args=ap.parse_args(); deterministic(args.seed)
 config_path=Path(args.config).resolve(); cfg=yaml.safe_load(config_path.read_text()); train_cfg=cfg['training']; loss_cfg=cfg['loss_weights']
 model_kwargs=dict(cfg['model']);
 for key in ('class','native_safe_initialization'): model_kwargs.pop(key,None)
 model_config=CMCPLocalSafetyConfig(**model_kwargs)
 loss_config=PairwiseSafetyLossConfig(utility_bce_weight=loss_cfg['utility_bce'],risk_bce_weight=loss_cfg['catastrophic_risk_bce'],pairwise_preference_weight=loss_cfg['pairwise_preference'],abstention_weight=loss_cfg['abstention_to_native'],no_harm_weight=loss_cfg['no_harm_margin'],harmful_multiplier=loss_cfg['harmful_non_native_multiplier'],pairwise_margin=loss_cfg['pairwise_margin'],grad_clip_norm=train_cfg['grad_clip_norm'])
 model=CMCPLocalPairwiseSafetyComparator(model_config).to(args.device)
 normalization=compute_static_token_normalization(args.fit_index)
 epochs=args.epochs or train_cfg['epochs']; patience=args.patience or train_cfg['patience']; batch=args.batch_size or train_cfg['point_sequence_batch_size']; eval_batch=args.eval_batch_size or train_cfg['eval_point_sequence_batch_size']; bootstrap=args.bootstrap_samples or train_cfg['bootstrap_samples']
 initial=evaluate_pairwise_index(model,args.validation_index,normalization,expected_partition='model_validation',device=args.device,point_batch_size=eval_batch,bootstrap_samples=bootstrap,bootstrap_seed=17000,max_videos=args.max_validation_videos)
 if initial['selected_gain_points']['AJ']!=0.0 or initial['behavior']['selected_non_native_rate']!=0.0: raise RuntimeError('zero-step native parity failed')
 best_state=clone_state(model); best_validation=initial; best_epoch=-1; best_safe_aj=float(initial['selected_gain_points']['AJ']); stale=0; history=[]
 optimizer=torch.optim.AdamW(model.parameters(),lr=train_cfg['learning_rate'],weight_decay=train_cfg['weight_decay']); generator=torch.Generator().manual_seed(args.seed)
 for epoch in range(epochs):
  train_metrics=train_pairwise_epoch(model,args.fit_index,normalization,optimizer,loss_config,device=args.device,point_batch_size=batch,generator=generator,max_videos=args.max_train_videos)
  validation=evaluate_pairwise_index(model,args.validation_index,normalization,expected_partition='model_validation',device=args.device,point_batch_size=eval_batch,bootstrap_samples=bootstrap,bootstrap_seed=17000+epoch,max_videos=args.max_validation_videos)
  safe=safety_feasible(validation); aj=float(validation['selected_gain_points']['AJ']); improved=safe and aj>best_safe_aj+1e-12
  if improved: best_state=clone_state(model); best_validation=validation; best_epoch=epoch; best_safe_aj=aj; stale=0
  else: stale+=1
  row={'epoch':epoch,'train':train_metrics,'validation':validation,'safety_feasible':safe,'checkpoint_improved':improved}; history.append(row)
  print(json.dumps({'epoch':epoch,'train_loss':train_metrics['loss'],'AJ_gain_points':aj,'AJ_CI':validation['paired_video_selected_AJ_gain_CI'],'harmful_non_native_rate':validation['behavior']['harmful_non_native_rate'],'selected_non_native_rate':validation['behavior']['selected_non_native_rate'],'severe_16px_delta':validation['severe_16px_rate']['selected_delta'],'safety_feasible':safe}),flush=True)
  if stale>=patience: break
 model.load_state_dict(best_state,strict=True)
 final=evaluate_pairwise_index(model,args.validation_index,normalization,expected_partition='model_validation',device=args.device,point_batch_size=eval_batch,bootstrap_samples=bootstrap,bootstrap_seed=17999,max_videos=args.max_validation_videos)
 fit_index=load_complete_pairwise_token_index(args.fit_index,expected_partition='fit'); val_index=load_complete_pairwise_token_index(args.validation_index,expected_partition='model_validation')
 full_scale=args.max_train_videos==0 and args.max_validation_videos==0
 gate={
  'candidate_coordinate_hash_exact': final['candidate_coordinate_combined_sha256']==val_index['candidate_coordinate_combined_sha256'],
  'candidate_oracle_AJ_gain_at_least_3':final['oracle_gain_points']['AJ']>=3.0,
  'direct_top1_AJ_gain_at_least_0_5':final['selected_gain_points']['AJ']>=0.5,
  'paired_top1_AJ_CI_lower_positive':final['paired_video_selected_AJ_gain_CI']['lower']>0,
  'delta_gain_positive':final['selected_gain_points']['delta_average']>0,
  'severe_16px_not_worse':final['severe_16px_rate']['selected_delta']<=0,
  'harmful_non_native_rate_at_most_0_01':final['behavior']['harmful_non_native_rate']<=0.01,
 }
 gate['pass']=all(gate.values()) and full_scale; gate['decision']='ALLOW_LOCAL_SAFETY_COMPARATOR_AND_REOPEN_MUSR_ABLATION' if gate['pass'] else 'STOP_PAIRWISE_COMPARATOR_AND_CONSIDER_LATE_BACKBONE_FINETUNING'
 output=Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=True); checkpoint_path=output/'best.pt'; metrics_path=output/'metrics.json'
 bundle={'schema_version':'routeD_cmcp_pairwise_safety_training_bundle_v0','seed':args.seed,'best_epoch':best_epoch,'model_config':asdict(model_config),'loss_config':asdict(loss_config),'model_state':best_state,'model_state_sha256':state_dict_sha256(best_state),'normalization':normalization.to_json(),'optimizer':{'type':'AdamW','learning_rate':train_cfg['learning_rate'],'weight_decay':train_cfg['weight_decay'],'epochs_requested':epochs,'patience':patience,'point_sequence_batch_size':batch,'eval_point_sequence_batch_size':eval_batch,'checkpoint_rule':'safety_feasible_then_max_direct_AJ'},'config_path':str(config_path),'config_sha256':file_sha256(config_path),'fit_cache_index_sha256':fit_index['_index_sha256'],'validation_cache_index_sha256':val_index['_index_sha256'],'fit_candidate_coordinate_combined_sha256':fit_index['candidate_coordinate_combined_sha256'],'validation_candidate_coordinate_combined_sha256':val_index['candidate_coordinate_combined_sha256'],'initialization_validation':initial,'final_validation':final,'gate':gate,'history':history,'external_data_read':{'calibration':False,'final_holdout':False,'tapvid_davis':False,'tapvid_kinetics':False},'smoke_limits':{'max_train_videos':args.max_train_videos,'max_validation_videos':args.max_validation_videos}}
 torch.save(bundle,checkpoint_path); bundle['checkpoint_path']=str(checkpoint_path); bundle['checkpoint_sha256']=file_sha256(checkpoint_path); serial={k:v for k,v in bundle.items() if k not in ('model_state',)}; metrics_path.write_text(json.dumps(serial,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps({'checkpoint':str(checkpoint_path),'checkpoint_sha256':bundle['checkpoint_sha256'],'model_state_sha256':bundle['model_state_sha256'],'best_epoch':best_epoch,'selected_gain_points':final['selected_gain_points'],'oracle_gain_points':final['oracle_gain_points'],'selected_AJ_CI':final['paired_video_selected_AJ_gain_CI'],'behavior':final['behavior'],'severe_16px_rate':final['severe_16px_rate'],'gate':gate},indent=2))

if __name__=='__main__': main()
