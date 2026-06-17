#!/usr/bin/env python3
"""
FSPT 推理脚本

快速推理和可视化

使用方法:
    # 从视频文件推理
    python scripts/inference.py --checkpoint checkpoints/fspt_base/best.pth --video input.mp4 --output output/
    
    # 从图像序列推理
    python scripts/inference.py --checkpoint checkpoints/fspt_base/best.pth --images frames/*.png --output output/
    
    # 使用ONNX模型
    python scripts/inference.py --onnx fspt.onnx --video input.mp4 --output output/
"""

import argparse
import logging
import sys
import time
import re
from pathlib import Path
from typing import List, Tuple, Optional
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_video(video_path: str, max_frames: int = None) -> Tuple[np.ndarray, float]:
    """
    加载视频文件
    
    Returns:
        frames: (T, H, W, 3) uint8
        fps: 帧率
    """
    try:
        import cv2
    except Exception as exc:
        raise ImportError("OpenCV is required to load videos. Please install opencv-python.") from exc

    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        logger.warning(f"Invalid FPS ({fps}) for {video_path}, using 30.0")
        fps = 30.0
    
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None or frame.size == 0:
            logger.warning(f"Skipping empty frame at index {frame_idx} in {video_path}")
            frame_idx += 1
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_idx += 1
        
        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()
    if not frames:
        raise ValueError(f"No valid frames read from video: {video_path}")
    return np.stack(frames), float(fps)


def load_images(image_paths: List[str]) -> np.ndarray:
    """加载图像序列"""
    from PIL import Image
    
    frames = []
    if not image_paths:
        raise ValueError("No image paths provided.")
    def _natural_key(value: str):
        parts = re.split(r'(\d+)', str(value))
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    for path in sorted(image_paths, key=_natural_key):
        try:
            img = Image.open(path).convert('RGB')
            if img.size[0] == 0 or img.size[1] == 0:
                logger.warning(f"Skipping empty image: {path}")
                continue
            frames.append(np.array(img))
        except Exception as exc:
            logger.warning(f"Failed to load image {path}: {exc}")
            continue
    if not frames:
        raise ValueError("No valid images loaded.")
    return np.stack(frames)


def preprocess_video(
    video: np.ndarray,
    resolution: Tuple[int, int] = (256, 256),
) -> torch.Tensor:
    """
    预处理视频
    
    Args:
        video: (T, H, W, 3) uint8
        resolution: 目标分辨率
        
    Returns:
        video_tensor: (1, T, 3, H, W) float32
    """
    T, H, W, C = video.shape
    
    # 转换为tensor（支持已归一化输入）
    video_tensor = torch.from_numpy(video).float()
    if video_tensor.max() > 1.5:
        video_tensor = video_tensor / 255.0
    video_tensor = video_tensor.permute(0, 3, 1, 2)  # (T, 3, H, W)
    
    # 调整分辨率
    if resolution and (H, W) != resolution:
        video_tensor = F.interpolate(
            video_tensor, size=resolution, mode='bilinear', align_corners=False
        )
    
    # 添加batch维度
    video_tensor = video_tensor.unsqueeze(0)  # (1, T, 3, H, W)
    
    return video_tensor


def generate_grid_points(
    height: int,
    width: int,
    grid_size: int = 20,
    query_frame: int = 0,
) -> torch.Tensor:
    """
    生成网格查询点
    
    Returns:
        query_points: (1, N, 3) [t, y, x]
    """
    grid_size = int(grid_size)
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    ys = np.linspace(0.1, 0.9, grid_size)
    xs = np.linspace(0.1, 0.9, grid_size)
    
    points = []
    for y in ys:
        for x in xs:
            points.append([query_frame, y, x])
    
    query_points = torch.tensor(points, dtype=torch.float32)
    query_points = query_points.unsqueeze(0)  # (1, N, 3)
    
    return query_points


def run_pytorch_inference(
    model,
    video: torch.Tensor,
    query_points: torch.Tensor,
    device: str = 'cuda',
) -> Tuple[np.ndarray, np.ndarray]:
    """PyTorch模型推理"""
    device = torch.device(device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            index = device.index if device.index is not None else 0
            if index >= torch.cuda.device_count():
                logger.warning(f"CUDA device index {index} out of range; using cuda:0")
                index = 0
            device = torch.device(f'cuda:{index}')
    model = model.to(device)
    model.eval()
    
    video = video.to(device)
    query_points = query_points.to(device)
    
    with torch.no_grad():
        start = time.perf_counter()
        tracks, visibility = model(video, query_points)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    
    logger.info(f"Inference time: {elapsed * 1000:.2f} ms")
    
    return tracks[0].cpu().numpy(), (visibility[0] > 0.5).cpu().numpy()


def run_onnx_inference(
    session,
    video: np.ndarray,
    query_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """ONNX模型推理"""
    start = time.perf_counter()
    tracks, visibility = session.run(
        None, {'video': video, 'query_points': query_points}
    )
    elapsed = time.perf_counter() - start
    
    logger.info(f"Inference time: {elapsed * 1000:.2f} ms")
    
    return tracks[0], visibility[0] > 0.5


def visualize_tracks(
    video: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    output_dir: str,
    fps: float = 30.0,
):
    """
    可视化追踪结果
    
    Args:
        video: (T, H, W, 3) uint8
        tracks: (N, T, 2) 归一化坐标
        visibility: (N, T) bool
        output_dir: 输出目录
    """
    try:
        import cv2
    except Exception as exc:
        logger.warning(f"Visualization skipped (cv2 unavailable): {exc}")
        return
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        colors = list(mcolors.TABLEAU_COLORS.values())
    except Exception as exc:
        logger.warning(f"matplotlib unavailable, using fallback colors: {exc}")
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    T, H, W, _ = video.shape
    N = tracks.shape[0]
    
    # 生成颜色
    color_map = []
    for i in range(N):
        color = colors[i % len(colors)]
        if isinstance(color, tuple):
            color_map.append(tuple(int(c * 255) for c in color))
        else:
            # hex string
            color_map.append(tuple(int(color[j:j+2], 16) for j in (1, 3, 5)))
    
    # 绘制每帧
    frames_with_tracks = []
    
    for t in range(T):
        frame = video[t].copy()
        
        for i in range(N):
            if visibility[i, t]:
                y, x = tracks[i, t]
                px, py = int(x * W), int(y * H)
                
                # 绘制点
                cv2.circle(frame, (px, py), 4, color_map[i], -1)
                cv2.circle(frame, (px, py), 5, (255, 255, 255), 1)
                
                # 绘制轨迹
                for t2 in range(max(0, t - 10), t):
                    if visibility[i, t2] and visibility[i, t2 + 1]:
                        y1, x1 = tracks[i, t2]
                        y2, x2 = tracks[i, t2 + 1]
                        px1, py1 = int(x1 * W), int(y1 * H)
                        px2, py2 = int(x2 * W), int(y2 * H)
                        cv2.line(frame, (px1, py1), (px2, py2), color_map[i], 2)
        
        frames_with_tracks.append(frame)
        
        # 保存帧
        cv2.imwrite(str(output_dir / f'frame_{t:04d}.png'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # 保存视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = str(output_dir / 'tracking_result.mp4')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
    
    for frame in frames_with_tracks:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    writer.release()
    
    logger.info(f"Saved visualization to {output_dir}")
    logger.info(f"Saved video to {video_path}")
    
    # 保存GIF
    try:
        import imageio
        gif_path = str(output_dir / 'tracking_result.gif')
        imageio.mimsave(gif_path, frames_with_tracks[::2], fps=fps / 2)
        logger.info(f"Saved GIF to {gif_path}")
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description='FSPT Inference')
    
    # 模型选项
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--onnx', type=str, default=None,
                        help='Path to ONNX model')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--use-ema', action='store_true', default=None,
                        help='Use EMA weights if available in checkpoint (default: follow config.training.ema.use_for_eval)')
    
    # 输入选项
    parser.add_argument('--video', type=str, default=None,
                        help='Path to input video')
    parser.add_argument('--images', type=str, nargs='+', default=None,
                        help='Paths to input images')
    
    # 输出选项
    parser.add_argument('--output', type=str, default='outputs/inference',
                        help='Output directory')
    
    # 推理选项
    parser.add_argument('--resolution', type=int, nargs=2, default=[256, 256],
                        help='Input resolution (H, W)')
    parser.add_argument('--grid-size', type=int, default=15,
                        help='Grid size for query points')
    parser.add_argument('--query-frame', type=int, default=0,
                        help='Query frame index')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    project_root = Path(__file__).parent.parent
    if args.checkpoint and not Path(args.checkpoint).is_absolute():
        args.checkpoint = str(project_root / args.checkpoint)
    if args.onnx and not Path(args.onnx).is_absolute():
        args.onnx = str(project_root / args.onnx)
    if args.config and not Path(args.config).is_absolute():
        args.config = str(project_root / args.config)
    if args.output and not Path(args.output).is_absolute():
        args.output = str(project_root / args.output)
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if args.onnx:
        onnx_path = Path(args.onnx)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
    if args.video:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = project_root / video_path
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        args.video = str(video_path)
    if args.images:
        expanded = []
        for img_path in args.images:
            has_glob = any(ch in img_path for ch in ["*", "?", "["])
            candidates = []
            if has_glob:
                candidates = glob(img_path)
                if not candidates and not Path(img_path).is_absolute():
                    candidates = glob(str(project_root / img_path))
            else:
                candidates = [img_path]
            for cand in candidates:
                cand_path = Path(cand)
                if not cand_path.is_absolute():
                    cand_path = project_root / cand_path
                if cand_path.exists():
                    expanded.append(str(cand_path))
        if not expanded:
            raise FileNotFoundError("No input images matched or exist.")
        args.images = expanded
    
    # 检查输入
    if args.video is None and args.images is None:
        parser.error('Either --video or --images must be specified')
    
    if args.checkpoint is None and args.onnx is None:
        parser.error('Either --checkpoint or --onnx must be specified')
    
    # 加载视频
    if args.video:
        logger.info(f"Loading video from {args.video}")
        video, fps = load_video(args.video, args.max_frames)
    else:
        logger.info(f"Loading images from {args.images}")
        video = load_images(args.images)
        fps = 30.0
        if args.max_frames and video.shape[0] > args.max_frames:
            video = video[:args.max_frames]
    
    logger.info(f"Video shape: {video.shape}")
    original_size = video.shape[1:3]
    num_frames = int(video.shape[0])
    query_frame = int(args.query_frame)
    if num_frames > 0:
        query_frame = max(0, min(query_frame, num_frames - 1))
    if query_frame != args.query_frame:
        logger.warning(f"Query frame clamped from {args.query_frame} to {query_frame}")
    
    # 预处理
    video_tensor = preprocess_video(video, tuple(args.resolution))
    # 使用与模型一致的分辨率进行可视化，避免长宽比变化导致偏移
    video_vis = video_tensor.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
    if video_vis.max() <= 1.5:
        video_vis = (video_vis * 255.0).clip(0, 255).astype(np.uint8)
    else:
        video_vis = video_vis.clip(0, 255).astype(np.uint8)
    
    # 生成查询点
    query_points = generate_grid_points(
        args.resolution[0], args.resolution[1],
        grid_size=args.grid_size,
        query_frame=query_frame,
    )
    
    logger.info(f"Query points: {query_points.shape[1]} points")
    
    # 推理
    if args.checkpoint:
        logger.info(f"Loading PyTorch model from {args.checkpoint}")
        from omegaconf import OmegaConf
        
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if args.config is None:
            if 'config' not in checkpoint:
                raise ValueError("Checkpoint missing config; please provide --config.")
            config = OmegaConf.create(checkpoint['config'])
        else:
            from train import load_config
            config = load_config(args.config)
        # 推理阶段禁用频率正交损失，避免额外开销
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
        use_ema = args.use_ema
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
        sd_keys = list(state_dict.keys())
        if sd_keys and sd_keys[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        
        tracks, visibility = run_pytorch_inference(
            model, video_tensor, query_points, args.device
        )
    else:
        logger.info(f"Loading ONNX model from {args.onnx}")
        import onnxruntime as ort
        
        available = ort.get_available_providers()
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in available:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(args.onnx, providers=providers)
        
        tracks, visibility = run_onnx_inference(
            session,
            video_tensor.numpy(),
            query_points.numpy(),
        )
    
    logger.info(f"Tracks shape: {tracks.shape}")
    
    # 可视化
    visualize_tracks(video_vis, tracks, visibility, args.output, fps)


if __name__ == '__main__':
    main()
