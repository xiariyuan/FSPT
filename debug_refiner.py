#!/usr/bin/env python3
"""诊断 refiner 为什么越训越差"""
import torch
import sys
sys.path.insert(0, '/gemini/code/FSPT')

from omegaconf import OmegaConf
from pathlib import Path

# 加载最新checkpoint
ckpt_dir = Path('/gemini/code/FSPT/checkpoints/fspt_routeA_r50_corr7cache_20260212_101650')
latest = ckpt_dir / 'latest.pth'
if not latest.exists():
    print(f'No checkpoint found at {latest}')
    sys.exit(1)

ckpt = torch.load(latest, map_location='cpu')
state_dict = ckpt.get('model_state_dict', ckpt)

print('=== Residual Decoder Weights ===')
for k, v in state_dict.items():
    if 'residual_decoder' in k:
        print(f'{k}: shape={tuple(v.shape)}, mean={v.mean().item():.6f}, std={v.std().item():.6f}, abs_max={v.abs().max().item():.6f}')

print()
print('=== Position Head Final Layer (should be near zero for identity init) ===')
pos_weight = state_dict.get('residual_decoder.position_head.4.weight')
pos_bias = state_dict.get('residual_decoder.position_head.4.bias')
if pos_weight is not None:
    print(f'position_head final weight: mean={pos_weight.mean().item():.6f}, std={pos_weight.std().item():.6f}')
if pos_bias is not None:
    print(f'position_head final bias: {pos_bias.tolist()}')

print()
print('=== Check delta_scale ===')
# 从config检查
config_path = '/gemini/code/FSPT/configs/fspt_cotracker_refine_posbalanced_r50pretrained_corr7.yaml'
cfg = OmegaConf.load(config_path)
delta_scale = cfg.model.refiner.get('delta_scale', 0.2)
print(f'delta_scale from config: {delta_scale}')

print()
print('=== Temporal Transformer Weights ===')
for k, v in state_dict.items():
    if 'temporal_transformer' in k and 'weight' in k:
        print(f'{k}: mean={v.mean().item():.6f}, std={v.std().item():.6f}')
        break

print()
print('=== Local Correlation Weights ===')
for k, v in state_dict.items():
    if 'local_correlation' in k:
        print(f'{k}: shape={tuple(v.shape)}, mean={v.mean().item():.6f}, std={v.std().item():.6f}')
