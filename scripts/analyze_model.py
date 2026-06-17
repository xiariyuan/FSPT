#!/usr/bin/env python3
"""
模型分析工具

分析FSPT模型的:
1. 参数量统计（按模块）
2. FLOPs计算
3. 内存占用估算
4. 推理速度测试

使用方法:
    python scripts/analyze_model.py --config configs/fspt_base.yaml
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple
import time

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
    parser = argparse.ArgumentParser(description='Analyze FSPT Model')
    parser.add_argument('--config', type=str, default='configs/fspt_base.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for analysis')
    parser.add_argument('--num-frames', type=int, default=24,
                        help='Number of frames')
    parser.add_argument('--num-points', type=int, default=256,
                        help='Number of query points')
    parser.add_argument('--resolution', type=int, nargs=2, default=[256, 256],
                        help='Input resolution (H W)')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Warmup iterations for speed test')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Iterations for speed test')
    return parser.parse_args()


def count_parameters(model: nn.Module) -> Dict[str, Tuple[int, int]]:
    """
    统计模型参数量
    
    Returns:
        dict: {module_name: (total_params, trainable_params)}
    """
    stats = {}
    
    # 统计顶层模块
    for name, module in model.named_children():
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        stats[name] = (total, trainable)
    
    # 总计
    total_all = sum(p.numel() for p in model.parameters())
    trainable_all = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stats['_total_'] = (total_all, trainable_all)
    
    return stats


def format_params(num: int) -> str:
    """格式化参数数量"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)


def estimate_flops(model: nn.Module, input_shape: Tuple[int, ...], 
                   query_shape: Tuple[int, ...], device: torch.device) -> int:
    """
    估算模型FLOPs
    
    使用简化的计算方法，基于线性层和卷积层
    """
    from collections import defaultdict
    
    flops = defaultdict(int)
    
    # Hook函数
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(module, nn.Linear):
                # FLOPs = 2 * in_features * out_features * batch_elements
                batch_elements = input[0].numel() // input[0].shape[-1]
                flops[name] += 2 * module.in_features * module.out_features * batch_elements
                
            elif isinstance(module, nn.Conv2d):
                # FLOPs = 2 * kernel_size^2 * in_channels * out_channels * output_elements
                output_elements = output.numel() // output.shape[0] // output.shape[1]
                flops[name] += 2 * module.kernel_size[0] * module.kernel_size[1] * \
                               module.in_channels * module.out_channels * output_elements * output.shape[0]
                
            elif isinstance(module, nn.MultiheadAttention):
                # Attention FLOPs ≈ 4 * seq_len * dim^2 + 2 * seq_len^2 * dim
                if len(input) > 0 and input[0] is not None:
                    if input[0].dim() != 3:
                        return
                    dim = input[0].shape[-1]
                    if getattr(module, "batch_first", False):
                        batch = input[0].shape[0]
                        seq_len = input[0].shape[1]
                    else:
                        seq_len = input[0].shape[0]
                        batch = input[0].shape[1]
                    flops[name] += (4 * seq_len * dim * dim + 2 * seq_len * seq_len * dim) * batch
                    
        return hook
    
    # 注册hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.MultiheadAttention)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # 前向传播
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        video = torch.randn(*input_shape, device=device)
        query_points = torch.rand(*query_shape, device=device)
        query_points[:, :, 0] = (query_points[:, :, 0] * (input_shape[1] - 1)).long()
        
        try:
            model(video, query_points)
        except Exception as e:
            logger.warning(f"FLOPs计算时前向传播失败: {e}")
    
    # 移除hooks
    for h in hooks:
        h.remove()
    
    total_flops = sum(flops.values())
    return total_flops


def estimate_memory(model: nn.Module, input_shape: Tuple[int, ...],
                    query_shape: Tuple[int, ...], device: torch.device) -> Dict[str, float]:
    """
    估算模型内存占用
    
    Returns:
        dict: 各项内存占用（MB）
    """
    if device.type != 'cuda':
        return {"note": "Memory estimation requires CUDA"}
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # 模型参数内存
    model = model.to(device)
    params_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # 前向传播内存
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        video = torch.randn(*input_shape, device=device)
        query_points = torch.rand(*query_shape, device=device)
        query_points[:, :, 0] = (query_points[:, :, 0] * (input_shape[1] - 1)).long()
        
        try:
            model(video, query_points)
        except Exception as e:
            logger.warning(f"内存估算时前向传播失败: {e}")
    
    peak_memory = torch.cuda.max_memory_allocated()
    
    # 训练内存估算（约3倍推理内存）
    train_memory_estimate = peak_memory * 3
    
    return {
        "parameters_mb": params_memory / 1024 / 1024,
        "inference_peak_mb": peak_memory / 1024 / 1024,
        "training_estimate_mb": train_memory_estimate / 1024 / 1024,
    }


def benchmark_speed(model: nn.Module, input_shape: Tuple[int, ...],
                    query_shape: Tuple[int, ...], device: torch.device,
                    warmup: int = 10, iterations: int = 100) -> Dict[str, float]:
    """
    测试模型推理速度
    
    Returns:
        dict: 速度统计
    """
    model = model.to(device)
    model.eval()
    
    video = torch.randn(*input_shape, device=device)
    query_points = torch.rand(*query_shape, device=device)
    query_points[:, :, 0] = (query_points[:, :, 0] * (input_shape[1] - 1)).long()
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            try:
                model(video, query_points)
            except Exception as e:
                logger.warning(f"速度测试时前向传播失败: {e}")
                return {"error": str(e)}
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 计时
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            model(video, query_points)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            times.append(time.perf_counter() - start)
    
    times = sorted(times)
    
    return {
        "mean_ms": sum(times) / len(times) * 1000,
        "median_ms": times[len(times) // 2] * 1000,
        "min_ms": times[0] * 1000,
        "max_ms": times[-1] * 1000,
        "fps": 1.0 / (sum(times) / len(times)),
    }


def main():
    args = parse_args()
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
    
    # 如果CLIP不可用则禁用语义编码器
    try:
        from models.semantic_encoder import CLIP_AVAILABLE
    except Exception:
        CLIP_AVAILABLE = False
    if (
        not CLIP_AVAILABLE
        and hasattr(config, 'model')
        and hasattr(config.model, 'semantic')
        and getattr(config.model.semantic, 'enabled', False)
    ):
        logger.warning("CLIP not available; disabling semantic encoder for analysis.")
        config.model.semantic.enabled = False
    
    # 创建模型
    from models.point_tracker import FSPTTracker
    model = FSPTTracker(config.model)
    
    # 输入形状
    B = args.batch_size
    T = args.num_frames
    H, W = args.resolution
    N = args.num_points
    
    input_shape = (B, T, 3, H, W)
    query_shape = (B, N, 3)
    
    logger.info(f"输入形状: video={input_shape}, query={query_shape}")
    
    # 1. 参数统计
    logger.info("\n" + "=" * 60)
    logger.info("参数量统计")
    logger.info("=" * 60)
    
    param_stats = count_parameters(model)
    
    print(f"\n{'模块':<30} {'总参数':<15} {'可训练':<15}")
    print("-" * 60)
    
    for name, (total, trainable) in param_stats.items():
        if name != '_total_':
            print(f"{name:<30} {format_params(total):<15} {format_params(trainable):<15}")
    
    print("-" * 60)
    total, trainable = param_stats['_total_']
    print(f"{'总计':<30} {format_params(total):<15} {format_params(trainable):<15}")
    
    # 2. FLOPs估算
    logger.info("\n" + "=" * 60)
    logger.info("FLOPs估算")
    logger.info("=" * 60)
    
    try:
        flops = estimate_flops(model, input_shape, query_shape, device)
        
        print(f"\n估算FLOPs: {format_params(flops)}")
        print(f"GFLOPs: {flops / 1e9:.2f}")
    except Exception as e:
        logger.warning(f"FLOPs估算失败: {e}")
    
    # 3. 内存估算
    if device.type == 'cuda':
        logger.info("\n" + "=" * 60)
        logger.info("内存估算")
        logger.info("=" * 60)
        
        try:
            memory_stats = estimate_memory(model, input_shape, query_shape, device)
            
            print(f"\n参数内存: {memory_stats.get('parameters_mb', 0):.2f} MB")
            print(f"推理峰值: {memory_stats.get('inference_peak_mb', 0):.2f} MB")
            print(f"训练估算: {memory_stats.get('training_estimate_mb', 0):.2f} MB")
        except Exception as e:
            logger.warning(f"内存估算失败: {e}")
    
    # 4. 速度测试
    logger.info("\n" + "=" * 60)
    logger.info("速度测试")
    logger.info("=" * 60)
    
    try:
        speed_stats = benchmark_speed(
            model, input_shape, query_shape, device,
            warmup=args.warmup, iterations=args.iterations
        )
        
        if 'error' not in speed_stats:
            print(f"\n平均延迟: {speed_stats['mean_ms']:.2f} ms")
            print(f"中位延迟: {speed_stats['median_ms']:.2f} ms")
            print(f"最小延迟: {speed_stats['min_ms']:.2f} ms")
            print(f"最大延迟: {speed_stats['max_ms']:.2f} ms")
            print(f"FPS: {speed_stats['fps']:.2f}")
        else:
            print(f"速度测试失败: {speed_stats['error']}")
    except Exception as e:
        logger.warning(f"速度测试失败: {e}")
    
    # 5. 模型结构摘要
    logger.info("\n" + "=" * 60)
    logger.info("模型结构摘要")
    logger.info("=" * 60)
    
    print(f"\n骨干网络: {config.model.backbone.type}")
    print(f"Transformer层数: {config.model.temporal.num_layers}")
    print(f"注意力头数: {config.model.temporal.num_heads}")
    print(f"特征维度: {config.model.temporal.dim}")
    print(f"频率分解: {'启用' if config.model.frequency.enabled else '禁用'}")
    print(f"频带数量: {config.model.frequency.num_bands}")
    print(f"语义编码: {'启用' if config.model.semantic.enabled else '禁用'}")
    print(f"遮挡预测: {'启用' if config.model.occlusion.enabled else '禁用'}")
    print(f"迭代精化次数: {getattr(config.model.temporal, 'num_refinement_iters', 1)}")
    print(f"多尺度融合: {'启用' if getattr(config.model.backbone, 'use_multiscale', True) else '禁用'}")
    
    logger.info("\n分析完成！")


if __name__ == '__main__':
    main()
