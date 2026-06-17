#!/usr/bin/env python3
"""
FSPT ONNX 导出脚本

将训练好的模型导出为ONNX格式，用于部署和推理优化

使用方法:
    python scripts/export_onnx.py --checkpoint checkpoints/fspt_base/best.pth --output fspt.onnx
    
    # 使用TensorRT优化
    python scripts/export_onnx.py --checkpoint checkpoints/fspt_base/best.pth --output fspt.onnx --optimize
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FSPTONNXWrapper(nn.Module):
    """
    ONNX导出包装器
    
    简化模型接口以便于ONNX导出
    """
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        # 禁用一些ONNX不支持的特性
        if hasattr(self.model, 'use_semantic'):
            self.model.use_semantic = False
        if hasattr(self.model, 'semantic_encoder'):
            self.model.semantic_encoder = None
        if hasattr(self.model, 'use_occlusion'):
            self.model.use_occlusion = False
        if hasattr(self.model, 'occlusion_predictor'):
            self.model.occlusion_predictor = None
    
    def forward(self, video: torch.Tensor, query_points: torch.Tensor):
        """
        简化的前向传播
        
        Args:
            video: (B, T, 3, H, W) 视频帧
            query_points: (B, N, 3) 查询点 [t, y, x]
            
        Returns:
            tracks: (B, N, T, 2)
            visibility: (B, N, T)
        """
        tracks, visibility = self.model(video, query_points)
        return tracks, visibility


def _align_state_dict_keys(state_dict, model_state):
    if not state_dict or not model_state:
        return state_dict
    sd_keys = list(state_dict.keys())
    ms_keys = list(model_state.keys())
    if not sd_keys or not ms_keys:
        return state_dict
    has_module = sd_keys[0].startswith('module.')
    model_has_module = ms_keys[0].startswith('module.')
    if has_module and not model_has_module:
        return {k[7:]: v for k, v in state_dict.items()}
    if not has_module and model_has_module:
        return {f'module.{k}': v for k, v in state_dict.items()}
    return state_dict


def load_model(checkpoint_path: str, config_path: str = None, use_ema: Optional[bool] = None):
    """加载模型"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if config_path:
        from train import load_config
        config = load_config(config_path)
    else:
        from omegaconf import OmegaConf
        if 'config' not in checkpoint:
            raise ValueError("Checkpoint missing config; please provide --config.")
        config = OmegaConf.create(checkpoint['config'])
    
    # 禁用CLIP以简化ONNX导出
    if hasattr(config, 'model') and hasattr(config.model, 'semantic'):
        config.model.semantic.enabled = False
    # 禁用频率正交损失，避免导出中引入随机操作
    if hasattr(config, 'model') and hasattr(config.model, 'frequency'):
        if hasattr(config.model.frequency, 'ortho_weight'):
            config.model.frequency.ortho_weight = 0.0
        if hasattr(config.model.frequency, 'feature_ortho_weight'):
            config.model.frequency.feature_ortho_weight = 0.0
    
    from train import create_model
    model = create_model(config)
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if not isinstance(state_dict, dict) or not all(isinstance(v, torch.Tensor) for v in state_dict.values()):
        raise ValueError("Checkpoint missing model_state_dict and is not a raw state dict.")
    if use_ema is None:
        ema_cfg = getattr(getattr(config, 'training', None), 'ema', None)
        use_ema = bool(ema_cfg is not None and getattr(ema_cfg, 'use_for_eval', False))
        if use_ema:
            logger.info("Using EMA weights by default (config.training.ema.use_for_eval=true).")
    if use_ema:
        ema_state = checkpoint.get('ema_state_dict', None)
        if ema_state is not None and 'ema_model' in ema_state:
            state_dict = ema_state['ema_model']
        else:
            logger.warning("EMA weights not found; falling back to model_state_dict")
    state_dict = _align_state_dict_keys(state_dict, model.state_dict())
    
    # 只加载匹配的权重
    model_state = model.state_dict()
    filtered_state = {k: v for k, v in state_dict.items() 
                      if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered_state, strict=False)
    
    model.eval()
    logger.info(f"Loaded model from {checkpoint_path}")
    
    return model, config


def export_onnx(
    model: nn.Module,
    output_path: str,
    batch_size: int = 1,
    num_frames: int = 24,
    num_points: int = 256,
    height: int = 256,
    width: int = 256,
    opset_version: int = 14,
    dynamic_axes: bool = True,
):
    """
    导出ONNX模型
    
    Args:
        model: PyTorch模型
        output_path: 输出路径
        batch_size: 批大小
        num_frames: 帧数
        num_points: 点数
        height: 图像高度
        width: 图像宽度
        opset_version: ONNX opset版本
        dynamic_axes: 是否使用动态轴
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = str(output_path)

    device = torch.device('cpu')  # ONNX导出使用CPU
    model = model.to(device)
    model.eval()
    
    # 包装模型
    wrapper = FSPTONNXWrapper(model)
    wrapper.eval()
    
    # 创建示例输入
    video = torch.randn(batch_size, num_frames, 3, height, width, device=device)
    query_points = torch.rand(batch_size, num_points, 3, device=device)
    query_points[:, :, 0] = 0  # 查询帧
    
    # 测试前向传播
    logger.info("Testing forward pass...")
    with torch.no_grad():
        tracks, visibility = wrapper(video, query_points)
    logger.info(f"Tracks shape: {tracks.shape}, Visibility shape: {visibility.shape}")
    
    # 配置动态轴
    if dynamic_axes:
        dynamic_axes_config = {
            'video': {0: 'batch', 1: 'frames'},
            'query_points': {0: 'batch', 1: 'points'},
            'tracks': {0: 'batch', 1: 'points', 2: 'frames'},
            'visibility': {0: 'batch', 1: 'points', 2: 'frames'},
        }
    else:
        dynamic_axes_config = None
    
    # 导出ONNX
    logger.info(f"Exporting ONNX to {output_path}...")
    
    torch.onnx.export(
        wrapper,
        (video, query_points),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['video', 'query_points'],
        output_names=['tracks', 'visibility'],
        dynamic_axes=dynamic_axes_config,
    )
    
    logger.info(f"ONNX model exported to {output_path}")
    
    # 验证ONNX模型
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model validation passed!")
    except ImportError:
        logger.warning("onnx package not installed, skipping validation")
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")
    
    return output_path


def optimize_onnx(input_path: str, output_path: str = None):
    """
    优化ONNX模型
    
    使用onnxruntime进行优化
    """
    try:
        import onnxruntime as ort
        from onnxruntime.transformers import optimizer
    except ImportError:
        logger.error("onnxruntime not installed. Run: pip install onnxruntime")
        return None
    
    if output_path is None:
        output_path = input_path.replace('.onnx', '_optimized.onnx')
    
    logger.info(f"Optimizing ONNX model: {input_path} -> {output_path}")
    
    # 使用onnxruntime优化
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = output_path
    
    # 创建会话触发优化
    _ = ort.InferenceSession(input_path, sess_options)
    
    logger.info(f"Optimized ONNX model saved to {output_path}")
    
    return output_path


def benchmark_onnx(onnx_path: str, num_runs: int = 100):
    """
    ONNX模型性能测试
    """
    try:
        import onnxruntime as ort
        import time
        import numpy as np
    except ImportError:
        logger.error("onnxruntime or numpy not installed")
        return
    
    logger.info(f"Benchmarking ONNX model: {onnx_path}")
    
    # 创建会话
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
    except Exception:
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    logger.info(f"Using providers: {session.get_providers()}")
    
    # 获取输入信息
    input_info = session.get_inputs()
    for inp in input_info:
        logger.info(f"Input: {inp.name}, shape: {inp.shape}, dtype: {inp.type}")
    
    # 创建测试输入
    video = np.random.randn(1, 24, 3, 256, 256).astype(np.float32)
    query_points = np.random.rand(1, 100, 3).astype(np.float32)
    query_points[:, :, 0] = 0
    
    # 预热
    for _ in range(10):
        _ = session.run(None, {'video': video, 'query_points': query_points})
    
    # 计时
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, {'video': video, 'query_points': query_points})
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    
    logger.info(f"Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
    logger.info(f"Throughput: {1000 / avg_time:.2f} FPS")
    
    return {'avg_ms': avg_time, 'std_ms': std_time, 'fps': 1000 / avg_time}


def main():
    parser = argparse.ArgumentParser(description='FSPT ONNX Export')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--use-ema', action='store_true', default=None,
                        help='Use EMA weights if available in checkpoint (default: follow config.training.ema.use_for_eval)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--output', type=str, default='fspt.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-frames', type=int, default=24)
    parser.add_argument('--num-points', type=int, default=256)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--opset', type=int, default=14)
    parser.add_argument('--optimize', action='store_true',
                        help='Optimize ONNX model with onnxruntime')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark ONNX model')
    parser.add_argument('--no-dynamic', action='store_true',
                        help='Disable dynamic axes')
    
    args = parser.parse_args()
    project_root = Path(__file__).parent.parent
    if args.checkpoint and not Path(args.checkpoint).is_absolute():
        args.checkpoint = str(project_root / args.checkpoint)
    if args.config and not Path(args.config).is_absolute():
        args.config = str(project_root / args.config)
    if args.output and not Path(args.output).is_absolute():
        args.output = str(project_root / args.output)
    if args.checkpoint and not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.config and not Path(args.config).exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    
    # 加载模型
    model, config = load_model(args.checkpoint, args.config, use_ema=args.use_ema)
    
    # 导出ONNX
    onnx_path = export_onnx(
        model,
        args.output,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        num_points=args.num_points,
        height=args.height,
        width=args.width,
        opset_version=args.opset,
        dynamic_axes=not args.no_dynamic,
    )
    
    # 优化
    if args.optimize:
        onnx_path = optimize_onnx(onnx_path)
    
    # 性能测试
    if args.benchmark and onnx_path:
        benchmark_onnx(onnx_path)


if __name__ == '__main__':
    main()
