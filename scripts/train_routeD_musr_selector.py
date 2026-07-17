#!/usr/bin/env python3
"""Stage-A utility-aligned candidate selector pretraining for MUSR."""
from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_routeD_musr import build_configs
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256
from projects.mmp_tracker.mmp_tracker.routeD_musr_training import (
    SelectorLossConfig,
    clone_state_dict_cpu,
    compute_feature_normalization,
    evaluate_raw_selector_on_cache_index,
    load_training_rows,
    state_dict_sha256,
    train_selector_one_epoch,
)
from projects.mmp_tracker.mmp_tracker.routeD_recovery_network import MultiHypothesisStateRecoveryNetwork

CACHE_ROOT = REPO_ROOT / "outputs/routeD_musr_kubric_cache_20260717"


def set_deterministic(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default=str(REPO_ROOT/'configs/routeD_musr_cotracker3_stage0.yaml'))
    ap.add_argument('--fit-index',default=str(CACHE_ROOT/'fit/cache_index.json'))
    ap.add_argument('--validation-index',default=str(CACHE_ROOT/'model_validation/cache_index.json'))
    ap.add_argument('--qualification-summary',default=str(REPO_ROOT/'docs/generated/ROUTED_MUSR_KUBRIC_QUALIFICATION_SUMMARY_2026-07-17.json'))
    ap.add_argument('--output',default=str(REPO_ROOT/'outputs/routeD_musr_selector_20260717/seed17'))
    ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--patience',type=int,default=4)
    ap.add_argument('--batch-size',type=int,default=512); ap.add_argument('--eval-batch-size',type=int,default=2048)
    ap.add_argument('--learning-rate',type=float,default=3e-4); ap.add_argument('--weight-decay',type=float,default=1e-4)
    ap.add_argument('--bootstrap-samples',type=int,default=5000); ap.add_argument('--seed',type=int,default=17)
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args()
    qualification_path=Path(args.qualification_summary).resolve(); qualification=json.loads(qualification_path.read_text())
    if not qualification['qualification']['training_authorized'] or qualification['integrity']['final_holdout_read']:
        raise RuntimeError('qualification does not authorize clean selector training')
    set_deterministic(args.seed)
    network_config,_,raw_config=build_configs(Path(args.config).resolve())
    rows,fit_index=load_training_rows(args.fit_index,expected_partition='fit')
    _,val_index=load_training_rows(args.validation_index,expected_partition='model_validation')
    if fit_index['protocol_sha256'] != val_index['protocol_sha256'] or fit_index['protocol_sha256'] != qualification['protocol']['sha256']:
        raise RuntimeError('protocol mismatch')
    for name, index in (('fit', fit_index), ('model_validation', val_index)):
        if int(index.get('candidate_feature_dim', network_config.candidate_feature_dim)) != network_config.candidate_feature_dim:
            raise RuntimeError(f'{name} candidate feature dimension mismatch')
        if int(index.get('state_feature_dim', network_config.state_feature_dim)) != network_config.state_feature_dim:
            raise RuntimeError(f'{name} state feature dimension mismatch')
        integrity=index.get('integrity',{})
        if any(bool(integrity.get(key,False)) for key in ('calibration_read','final_holdout_read','tapvid_davis_read','tapvid_kinetics_read')):
            raise RuntimeError(f'{name} cache reports locked-data contamination')
    normalization=compute_feature_normalization(rows)
    model=MultiHypothesisStateRecoveryNetwork(network_config).to(args.device)
    # State-write heads are excluded from Stage A by construction.
    for module in (model.abstention_head, model.state_write_head):
        for parameter in module.parameters(): parameter.requires_grad=False
    selector_config=SelectorLossConfig()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate,weight_decay=args.weight_decay)
    generator=torch.Generator(device='cpu').manual_seed(args.seed)
    init_eval=evaluate_raw_selector_on_cache_index(model,args.validation_index,normalization,expected_partition='model_validation',device=args.device,batch_size=args.eval_batch_size,bootstrap_samples=min(args.bootstrap_samples,1000),bootstrap_seed=args.seed*1000-1)
    if abs(init_eval['gain_points']['AJ'])>1e-10 or abs(init_eval['gain_points']['delta_average'])>1e-10:
        raise RuntimeError('selector native-safe initialization failed')
    history=[]; best=None; best_epoch=-1; best_AJ=float('-inf'); stale=0
    for epoch in range(args.epochs):
        train=train_selector_one_epoch(model,rows,normalization,optimizer,network_config,selector_config,device=args.device,batch_size=args.batch_size,generator=generator)
        val=evaluate_raw_selector_on_cache_index(model,args.validation_index,normalization,expected_partition='model_validation',device=args.device,batch_size=args.eval_batch_size,bootstrap_samples=args.bootstrap_samples,bootstrap_seed=args.seed*1000+epoch)
        history.append({'epoch':epoch,'train':train,'validation':val})
        print(json.dumps({'epoch':epoch,'train_loss':train['loss'],'selector_accuracy':train['selector_accuracy'],'train_regret':train['mean_utility_regret'],'validation_AJ_gain_points':val['gain_points']['AJ'],'validation_AJ_CI':val['paired_video_AJ_gain_CI'],'behavior':val['behavior'],'severe_16px_delta':val['severe_16px_rate']['delta']},ensure_ascii=False),flush=True)
        score=float(val['selected_metrics']['AJ'])
        if score>best_AJ+1e-8: best_AJ=score; best_epoch=epoch; best=clone_state_dict_cpu(model); stale=0
        else: stale+=1
        if stale>=args.patience: break
    if best is None: raise RuntimeError('no selector checkpoint')
    model.load_state_dict(best,strict=True)
    final=evaluate_raw_selector_on_cache_index(model,args.validation_index,normalization,expected_partition='model_validation',device=args.device,batch_size=args.eval_batch_size,bootstrap_samples=args.bootstrap_samples,bootstrap_seed=args.seed*1000+999)
    gate={
      'raw_selector_AJ_gain_at_least_0_5': final['gain_points']['AJ']>=0.5,
      'paired_AJ_CI_lower_positive': final['paired_video_AJ_gain_CI']['lower']>0,
      'delta_gain_positive': final['gain_points']['delta_average']>0,
      'severe_16px_not_worse': final['severe_16px_rate']['delta']<=0,
      'oracle_utility_match_rate_at_least_0_25': final['behavior']['oracle_utility_match_rate']>=0.25,
      'harmful_global_selection_rate_at_most_0_01': final['behavior']['harmful_global_selection_rate']<=0.01,
    }; gate['pass']=all(gate.values()); gate['decision']='ALLOW_STAGE_B_STATE_WRITE_TRAINING' if gate['pass'] else 'STOP_STAGE_B_AND_REDESIGN_CANDIDATE_REPRESENTATION'
    out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    config_path=Path(args.config).resolve()
    bundle={'schema_version':'routeD_musr_selector_bundle_v1','seed':args.seed,'best_epoch':best_epoch,'network_config':asdict(network_config),'selector_loss_config':asdict(selector_config),'model_state':best,'model_state_sha256':state_dict_sha256(best),'normalization':normalization.to_serializable(),'optimizer':{'type':'AdamW','learning_rate':args.learning_rate,'weight_decay':args.weight_decay,'batch_size':args.batch_size,'epochs_requested':args.epochs,'patience':args.patience},'config_path':str(config_path),'config_sha256':file_sha256(config_path),'representation_schema_version':fit_index.get('representation_schema_version'),'protocol_sha256':fit_index['protocol_sha256'],'fit_cache_index_sha256':fit_index['_index_sha256'],'validation_cache_index_sha256':val_index['_index_sha256'],'qualification_summary_sha256':file_sha256(qualification_path),'initialization_validation':init_eval,'final_validation':final,'gate':gate,'history':history,'external_data_read':{'calibration':False,'final_holdout':False,'tapvid_davis':False,'tapvid_kinetics':False}}
    ckpt=out/'best.pt'; torch.save(bundle,ckpt)
    metrics={k:v for k,v in bundle.items() if k!='model_state'}; metrics['checkpoint_path']=str(ckpt); metrics['checkpoint_sha256']=file_sha256(ckpt)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'checkpoint':str(ckpt),'checkpoint_sha256':metrics['checkpoint_sha256'],'model_state_sha256':bundle['model_state_sha256'],'best_epoch':best_epoch,'validation_gain_points':final['gain_points'],'validation_AJ_CI':final['paired_video_AJ_gain_CI'],'behavior':final['behavior'],'severe_16px_rate':final['severe_16px_rate'],'gate':gate},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
