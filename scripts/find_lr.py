#!/usr/bin/env python3
"""
学习率查找脚本

自动寻找最佳学习率范围

使用方法:
    python scripts/find_lr.py --config configs/fspt_base.yaml
"""

import os
import sys
import argparse
import logging
import math
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='FSPT Learning Rate Finder')
    parser.add_argument('--config', type=str, default='configs/fspt_base.yaml',
                        help='Path to config file')
    parser.add_argument('--start-lr', type=float, default=1e-7,
                        help='Starting learning rate')
    parser.add_argument('--end-lr', type=float, default=1.0,
                        help='Ending learning rate')
    parser.add_argument('--num-iter', type=int, default=100,
                        help='Number of iterations')
    parser.add_argument('--output', type=str, default='outputs/lr_finder.png',
                        help='Output plot path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.config and not Path(args.config).is_absolute():
        args.config = str(project_root / args.config)
    if args.output and not Path(args.output).is_absolute():
        args.output = str(project_root / args.output)
    if args.config and not Path(args.config).exists():
        logger.error(f"Config not found: {args.config}")
        return
    
    # 设备
    device_str = str(args.device)
    if device_str.startswith('cuda'):
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            device = torch.device('cpu')
        else:
            index = 0
            if ':' in device_str:
                try:
                    index = int(device_str.split(':', 1)[1])
                except ValueError:
                    index = 0
            if index >= torch.cuda.device_count():
                logger.warning(f"CUDA device index {index} out of range; using cuda:0")
                index = 0
            device = torch.device(f'cuda:{index}')
    else:
        device = torch.device(device_str)
    
    logger.info(f"Using device: {device}")
    
    # 加载配置（支持defaults继承）
    from train import load_config, _get_amp_dtype
    config = load_config(args.config)

    def _resolve_path(path_str: str) -> str:
        if not path_str:
            return path_str
        path = Path(path_str)
        return str(path if path.is_absolute() else project_root / path)

    if hasattr(config, 'data'):
        if hasattr(config.data, 'train') and hasattr(config.data.train, 'root'):
            config.data.train.root = _resolve_path(config.data.train.root)
        if hasattr(config.data, 'val') and hasattr(config.data.val, 'root'):
            config.data.val.root = _resolve_path(config.data.val.root)
        if hasattr(config.data, 'pseudo') and hasattr(config.data.pseudo, 'root'):
            config.data.pseudo.root = _resolve_path(config.data.pseudo.root)
    
    # 禁用语义编码器（加快速度）
    if hasattr(config, 'model') and hasattr(config.model, 'semantic'):
        config.model.semantic.enabled = False
    
    # 混合精度配置
    amp_cfg = getattr(getattr(config, 'training', None), 'amp', None)
    amp_enabled = bool(amp_cfg is not None and getattr(amp_cfg, 'enabled', False))
    amp_dtype = _get_amp_dtype(config) if amp_enabled else None
    grad_cfg = getattr(getattr(config, 'training', None), 'gradient', None)
    accum_steps_for_lr = int(getattr(grad_cfg, 'accumulation_steps', 1) or 1)
    opt_cfg = getattr(getattr(config, 'training', None), 'optimizer', None)
    opt_type = str(getattr(opt_cfg, 'type', 'AdamW')).lower() if opt_cfg is not None else 'adamw'
    if opt_type == 'adamw':
        optimizer_cls = torch.optim.AdamW
    elif opt_type == 'adam':
        optimizer_cls = torch.optim.Adam
    else:
        raise ValueError(f"Unsupported optimizer for LR finder: {opt_type}")
    optimizer_kwargs = {}
    betas = getattr(opt_cfg, 'betas', None) if opt_cfg is not None else None
    if betas is not None:
        optimizer_kwargs['betas'] = tuple(betas)

    # 创建模型
    from train import create_model, create_dataloaders, PointTrackingLoss
    
    model = create_model(config).to(device)
    logger.info("Model created")
    
    # 创建数据加载器
    train_loader, _ = create_dataloaders(config, debug=True)
    
    if train_loader is None:
        logger.error("Failed to create data loader")
        return
    try:
        if len(train_loader) == 0:
            logger.error("Data loader is empty")
            return
    except TypeError:
        pass
    
    try:
        num_batches = len(train_loader)
    except TypeError:
        num_batches = None
    if num_batches is not None:
        logger.info(f"Data loader created with {num_batches} batches")
    else:
        logger.info("Data loader created (num batches unknown)")
    
    # 创建损失函数
    criterion = PointTrackingLoss(config)
    
    # 运行LR finder
    from utils.lr_finder import find_optimal_lr
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    weight_decay = 0.0
    if opt_cfg is not None:
        weight_decay = float(getattr(opt_cfg, 'weight_decay', 0.0) or 0.0)

    suggested_lr = find_optimal_lr(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        device=device,
        optimizer_cls=optimizer_cls,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        num_iter=args.num_iter,
        weight_decay=weight_decay,
        plot_path=output_path,
        accumulation_steps=accum_steps_for_lr,
        use_amp=amp_enabled,
        amp_dtype=amp_dtype,
        optimizer_kwargs=optimizer_kwargs,
    )
    
    logger.info(f"\n{'='*50}")
    if suggested_lr is None or not math.isfinite(float(suggested_lr)):
        logger.warning("Suggested learning rate is not available.")
        logger.info(f"LR range plot saved to: {output_path}")
        logger.info(f"{'='*50}")
        return
    logger.info(f"Suggested learning rate: {suggested_lr:.2e}")
    logger.info(f"LR range plot saved to: {output_path}")
    logger.info(f"{'='*50}")
    
    # 更新配置建议
    logger.info("\nTo use this learning rate, update your config:")
    logger.info(f"  training.optimizer.lr: {suggested_lr:.2e}")


if __name__ == '__main__':
    main()
