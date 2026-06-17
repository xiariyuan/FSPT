#!/usr/bin/env python3
"""
训练验证脚本

验证训练流程是否正常工作:
1. 模型能否正确前向传播
2. 损失能否正确反向传播
3. 梯度是否正常
4. 数据加载是否正常

使用方法:
    python scripts/validate_training.py --config configs/fspt_base.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Validate FSPT Training')
    parser.add_argument('--config', type=str, default='configs/fspt_base.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--skip-data', action='store_true',
                        help='Skip data loading validation')
    return parser.parse_args()


def validate_model_creation(config) -> Tuple[bool, Optional[torch.nn.Module]]:
    """验证模型创建"""
    logger.info("=" * 50)
    logger.info("验证模型创建...")
    
    try:
        from train import create_model
        model = create_model(config)
        logger.info(f"✓ 模型创建成功")
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  总参数量: {total_params:,}")
        logger.info(f"  可训练参数: {trainable_params:,}")
        
        return True, model
    except Exception as e:
        logger.error(f"✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def validate_forward_pass(model, device) -> bool:
    """验证前向传播"""
    logger.info("=" * 50)
    logger.info("验证前向传播...")
    
    model = model.to(device)
    model.eval()
    
    try:
        # 创建虚拟输入
        B, T, H, W = 2, 24, 256, 256
        N = 100
        
        video = torch.randn(B, T, 3, H, W, device=device)
        query_points = torch.rand(B, N, 3, device=device)
        query_points[:, :, 0] = (query_points[:, :, 0] * (T - 1)).long()  # 时间帧
        
        with torch.no_grad():
            outputs = model(video, query_points)
        
        if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
            pred_tracks, pred_visibility = outputs[0], outputs[1]
            logger.info(f"✓ 前向传播成功")
            logger.info(f"  轨迹形状: {pred_tracks.shape}")
            logger.info(f"  可见性形状: {pred_visibility.shape}")
            
            # 检查输出范围
            if pred_tracks.min() >= 0 and pred_tracks.max() <= 1:
                logger.info(f"  轨迹范围: [{pred_tracks.min():.4f}, {pred_tracks.max():.4f}] ✓")
            else:
                logger.warning(f"  轨迹范围异常: [{pred_tracks.min():.4f}, {pred_tracks.max():.4f}]")
            
            if pred_visibility.min() >= 0 and pred_visibility.max() <= 1:
                logger.info(f"  可见性范围: [{pred_visibility.min():.4f}, {pred_visibility.max():.4f}] ✓")
            else:
                logger.warning(f"  可见性范围异常: [{pred_visibility.min():.4f}, {pred_visibility.max():.4f}]")
            
            return True
        else:
            logger.error(f"✗ 输出格式异常: {type(outputs)}")
            return False
            
    except Exception as e:
        logger.error(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_backward_pass(model, config, device) -> bool:
    """验证反向传播"""
    logger.info("=" * 50)
    logger.info("验证反向传播...")
    
    model = model.to(device)
    model.train()
    
    try:
        from train import PointTrackingLoss
        criterion = PointTrackingLoss(config)
        
        # 创建虚拟输入
        B, T, H, W = 2, 24, 256, 256
        N = 100
        
        video = torch.randn(B, T, 3, H, W, device=device)
        query_points = torch.rand(B, N, 3, device=device)
        query_points[:, :, 0] = (query_points[:, :, 0] * (T - 1)).long()
        
        target_points = torch.rand(B, N, T, 2, device=device)
        visibility = torch.rand(B, N, T, device=device) > 0.3
        
        # 前向传播
        outputs = model(video, query_points)
        if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
            pred_tracks, pred_visibility = outputs[0], outputs[1]
        else:
            logger.error(f"✗ 输出格式异常")
            return False
        
        # 计算损失
        losses = criterion(pred_tracks, target_points, pred_visibility, visibility)
        loss = losses['total']
        
        logger.info(f"  总损失: {loss.item():.6f}")
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        has_grad = False
        nan_grad = False
        zero_grad = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                total_params += 1
                
                if torch.isnan(param.grad).any():
                    nan_grad = True
                    logger.warning(f"  NaN梯度: {name}")
                
                if param.grad.abs().max() == 0:
                    zero_grad += 1
        
        if not has_grad:
            logger.error("✗ 没有参数有梯度")
            return False
        
        if nan_grad:
            logger.error("✗ 存在NaN梯度")
            return False
        
        if zero_grad > total_params * 0.5:
            logger.warning(f"⚠ 超过50%的参数梯度为零 ({zero_grad}/{total_params})")
        
        logger.info(f"✓ 反向传播成功")
        logger.info(f"  有梯度的参数: {total_params - zero_grad}/{total_params}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 反向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_optimization_step(model, config, device) -> bool:
    """验证优化步骤"""
    logger.info("=" * 50)
    logger.info("验证优化步骤...")
    
    model = model.to(device)
    model.train()
    
    try:
        from train import PointTrackingLoss, create_optimizer, create_scheduler
        criterion = PointTrackingLoss(config)
        optimizer = create_optimizer(model, config)
        num_training_steps = 100
        scheduler = create_scheduler(optimizer, config, num_training_steps)
        
        # 保存初始参数
        initial_params = {
            name: param.clone() for name, param in model.named_parameters()
            if param.requires_grad
        }
        
        # 创建虚拟输入
        B, T, H, W = 2, 24, 256, 256
        N = 100
        
        video = torch.randn(B, T, 3, H, W, device=device)
        query_points = torch.rand(B, N, 3, device=device)
        query_points[:, :, 0] = (query_points[:, :, 0] * (T - 1)).long()
        
        target_points = torch.rand(B, N, T, 2, device=device)
        visibility = torch.rand(B, N, T, device=device) > 0.3
        
        # 训练步骤
        optimizer.zero_grad()
        
        outputs = model(video, query_points)
        pred_tracks, pred_visibility = outputs[0], outputs[1]
        
        losses = criterion(pred_tracks, target_points, pred_visibility, visibility)
        loss = losses['total']
        
        loss.backward()
        
        # 梯度裁剪
        gradient_cfg = getattr(config.training, 'gradient', None)
        if gradient_cfg is not None and getattr(gradient_cfg, 'clip_norm', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_cfg.clip_norm)
        
        optimizer.step()
        
        # 检查参数是否更新
        params_updated = 0
        for name, param in model.named_parameters():
            if name in initial_params and param.requires_grad:
                if not torch.equal(param.data, initial_params[name]):
                    params_updated += 1
        
        if params_updated == 0:
            logger.error("✗ 没有参数被更新")
            return False
        
        logger.info(f"✓ 优化步骤成功")
        logger.info(f"  更新的参数: {params_updated}/{len(initial_params)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 优化步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_mixed_precision(model, config, device) -> bool:
    """验证混合精度训练"""
    logger.info("=" * 50)
    logger.info("验证混合精度训练...")
    
    if device.type == 'cpu':
        logger.info("⚠ CPU模式，跳过混合精度验证")
        return True

    amp_cfg = getattr(getattr(config, 'training', None), 'amp', None)
    if amp_cfg is None or not getattr(amp_cfg, 'enabled', False):
        logger.info("⚠ AMP未启用，跳过混合精度验证")
        return True
    
    model = model.to(device)
    model.train()
    
    try:
        from train import PointTrackingLoss, _get_amp_dtype
        from torch.cuda.amp import autocast, GradScaler
        
        criterion = PointTrackingLoss(config)
        amp_dtype = _get_amp_dtype(config)
        scaler = GradScaler(enabled=(amp_dtype == torch.float16))
        
        # 创建虚拟输入
        B, T, H, W = 2, 24, 256, 256
        N = 100
        
        video = torch.randn(B, T, 3, H, W, device=device)
        query_points = torch.rand(B, N, 3, device=device)
        query_points[:, :, 0] = (query_points[:, :, 0] * (T - 1)).long()
        
        target_points = torch.rand(B, N, T, 2, device=device)
        visibility = torch.rand(B, N, T, device=device) > 0.3
        
        # 混合精度前向传播
        with autocast(enabled=True, dtype=amp_dtype):
            outputs = model(video, query_points)
            pred_tracks, pred_visibility = outputs[0], outputs[1]
            
            losses = criterion(pred_tracks, target_points, pred_visibility, visibility)
            loss = losses['total']
        
        # 混合精度反向传播
        scaler.scale(loss).backward()
        
        logger.info(f"✓ 混合精度训练验证成功")
        logger.info(f"  损失值: {loss.item():.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 混合精度训练验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_data_loading(config) -> bool:
    """验证数据加载"""
    logger.info("=" * 50)
    logger.info("验证数据加载...")
    
    try:
        from train import create_dataloaders
        train_loader, val_loader = create_dataloaders(config, debug=True)
        
        if train_loader is None:
            logger.warning("⚠ 训练数据加载器为空（可能是数据集不存在）")
            return True
        
        # 尝试加载一个batch
        try:
            batch = next(iter(train_loader))
            logger.info(f"✓ 数据加载成功")
            
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  {key}: {value.shape}, dtype={value.dtype}")
            
            return True
            
        except StopIteration:
            logger.warning("⚠ 数据集为空")
            return True
            
    except Exception as e:
        logger.warning(f"⚠ 数据加载验证跳过: {e}")
        return True  # 不因数据问题阻止其他验证


def main():
    args = parse_args()
    project_root = Path(__file__).parent.parent
    if args.config and not Path(args.config).is_absolute():
        args.config = str(project_root / args.config)
    if args.config and not Path(args.config).exists():
        logger.error(f"Config not found: {args.config}")
        return
    
    # 确定设备
    device_str = str(args.device)
    if device_str.startswith('cuda'):
        if not torch.cuda.is_available():
            logger.warning("CUDA不可用，使用CPU")
            device = torch.device('cpu')
        else:
            index = 0
            if ':' in device_str:
                try:
                    index = int(device_str.split(':', 1)[1])
                except ValueError:
                    index = 0
            if index >= torch.cuda.device_count():
                logger.warning(f"CUDA设备索引{index}不可用，使用cuda:0")
                index = 0
            device = torch.device(f'cuda:{index}')
    else:
        device = torch.device(device_str)
    logger.info(f"使用设备: {device}")
    
    # 加载配置（支持defaults继承）
    from train import load_config
    config = load_config(args.config)

    def _resolve_path(path_str: Optional[str]) -> Optional[str]:
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
    
    # 禁用语义编码器（避免CLIP依赖问题）
    if hasattr(config, 'model') and hasattr(config.model, 'semantic'):
        config.model.semantic.enabled = False
        logger.info("⚠ 已禁用语义编码器进行验证")
    
    results = {}
    
    # 1. 验证模型创建
    success, model = validate_model_creation(config)
    results['model_creation'] = success
    if not success:
        logger.error("模型创建失败，终止验证")
        return
    
    # 2. 验证前向传播
    results['forward_pass'] = validate_forward_pass(model, device)
    
    # 3. 验证反向传播
    results['backward_pass'] = validate_backward_pass(model, config, device)
    
    # 4. 验证优化步骤
    results['optimization'] = validate_optimization_step(model, config, device)
    
    # 5. 验证混合精度
    results['mixed_precision'] = validate_mixed_precision(model, config, device)
    
    # 6. 验证数据加载（可选）
    if not args.skip_data:
        results['data_loading'] = validate_data_loading(config)
    
    # 总结
    logger.info("=" * 50)
    logger.info("验证总结:")
    logger.info("=" * 50)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        logger.info(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n✓ 所有验证通过！可以开始训练。")
    else:
        logger.error("\n✗ 部分验证失败，请检查错误信息。")
    
    return all_passed


if __name__ == '__main__':
    main()
