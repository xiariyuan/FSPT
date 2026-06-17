#!/usr/bin/env python3
"""
FSPT 可视化工具

可视化点追踪结果和频率分解
"""

import os
import re
import sys
import argparse
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def visualize_tracking_results(
    video: np.ndarray,
    pred_tracks: np.ndarray,
    gt_tracks: np.ndarray,
    pred_visibility: np.ndarray,
    gt_visibility: np.ndarray,
    output_dir: str,
    video_name: str = 'video',
    max_points: int = 20,
    fps: int = 10,
):
    """
    可视化追踪结果
    
    Args:
        video: (T, H, W, 3) 视频帧
        pred_tracks: (N, T, 2) 预测轨迹
        gt_tracks: (N, T, 2) 真实轨迹
        pred_visibility: (N, T) 预测可见性
        gt_visibility: (N, T) 真实可见性
        output_dir: 输出目录
        video_name: 视频名称
        max_points: 最大显示点数
        fps: 输出GIF帧率
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", str(video_name))
    
    if video is None or video.size == 0:
        raise ValueError("video is empty")
    if pred_tracks is None or pred_tracks.size == 0:
        raise ValueError("pred_tracks is empty")
    if pred_visibility is None or pred_visibility.size == 0:
        raise ValueError("pred_visibility is empty")
    if gt_tracks is None or gt_tracks.size == 0:
        raise ValueError("gt_tracks is empty")
    if gt_visibility is None or gt_visibility.size == 0:
        raise ValueError("gt_visibility is empty")
    if video.ndim != 4 or pred_tracks.ndim != 3:
        raise ValueError("Invalid input dimensions for visualization")

    # 统一可见性类型
    if pred_visibility.dtype != np.bool_:
        pred_visibility = pred_visibility > 0.5
    if gt_visibility.dtype != np.bool_:
        gt_visibility = gt_visibility > 0.5

    T, H, W, _ = video.shape
    N = pred_tracks.shape[0]

    # Auto-detect coordinate format: if max > 1.5, assume pixel coords
    if isinstance(pred_tracks, torch.Tensor):
        _pred_max = float(pred_tracks.max().item()) if pred_tracks.numel() > 0 else 0.0
    else:
        _pred_max = float(np.max(pred_tracks)) if np.size(pred_tracks) > 0 else 0.0
    if isinstance(gt_tracks, torch.Tensor):
        _gt_max = float(gt_tracks.max().item()) if gt_tracks.numel() > 0 else 0.0
    else:
        _gt_max = float(np.max(gt_tracks)) if np.size(gt_tracks) > 0 else 0.0
    _is_pixel_coords = _pred_max > 1.5 or _gt_max > 1.5
    if gt_tracks.shape[0] != N:
        raise ValueError("pred_tracks and gt_tracks point counts mismatch")
    if pred_visibility.shape[0] != N or gt_visibility.shape[0] != N:
        raise ValueError("Visibility point counts mismatch")
    if pred_tracks.shape[1] != T or gt_tracks.shape[1] != T:
        raise ValueError("Track length does not match video frames")
    if pred_visibility.shape[1] != T or gt_visibility.shape[1] != T:
        raise ValueError("Visibility length does not match video frames")
    
    # 选择要显示的点
    if N > max_points:
        indices = np.linspace(0, N-1, max_points, dtype=int)
    else:
        indices = np.arange(N)
    
    # 颜色
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    # 生成帧
    frames = []
    
    for t in tqdm(range(T), desc=f'Visualizing {video_name}'):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 左图：预测
        axes[0].imshow(video[t])
        axes[0].set_title(f'Prediction (Frame {t})')
        
        for j, idx in enumerate(indices):
            color = colors[j % len(colors)]
            
            if pred_visibility[idx, t]:
                y, x = pred_tracks[idx, t]
                if not _is_pixel_coords:
                    y, x = y * H, x * W
                circle = Circle((x, y), 4, color=color, fill=True, alpha=0.8)
                axes[0].add_patch(circle)
                
                # 画轨迹
                for t2 in range(max(0, t-5), t):
                    if pred_visibility[idx, t2]:
                        y1, x1 = pred_tracks[idx, t2]
                        if not _is_pixel_coords:
                            y1, x1 = y1 * H, x1 * W
                        y2, x2 = pred_tracks[idx, min(t2+1, t)]
                        if not _is_pixel_coords:
                            y2, x2 = y2 * H, x2 * W
                        axes[0].plot([x1, x2], [y1, y2], color=color, alpha=0.5, linewidth=2)
        
        # 中图：Ground Truth
        axes[1].imshow(video[t])
        axes[1].set_title(f'Ground Truth (Frame {t})')
        
        for j, idx in enumerate(indices):
            color = colors[j % len(colors)]
            
            if gt_visibility[idx, t]:
                y, x = gt_tracks[idx, t]
                if not _is_pixel_coords:
                    y, x = y * H, x * W
                circle = Circle((x, y), 4, color=color, fill=True, alpha=0.8)
                axes[1].add_patch(circle)
                
                # 画轨迹
                for t2 in range(max(0, t-5), t):
                    if gt_visibility[idx, t2]:
                        y1, x1 = gt_tracks[idx, t2]
                        if not _is_pixel_coords:
                            y1, x1 = y1 * H, x1 * W
                        y2, x2 = gt_tracks[idx, min(t2+1, t)]
                        if not _is_pixel_coords:
                            y2, x2 = y2 * H, x2 * W
                        axes[1].plot([x1, x2], [y1, y2], color=color, alpha=0.5, linewidth=2)
        
        # 右图：误差可视化
        axes[2].imshow(video[t])
        axes[2].set_title(f'Error (Frame {t})')
        
        for j, idx in enumerate(indices):
            if pred_visibility[idx, t] and gt_visibility[idx, t]:
                py, px = pred_tracks[idx, t]
                gy, gx = gt_tracks[idx, t]
                if not _is_pixel_coords:
                    py, px = py * H, px * W
                    gy, gx = gy * H, gx * W
                
                # 误差线
                error = np.sqrt((py-gy)**2 + (px-gx)**2)
                color = 'green' if error < 4 else 'orange' if error < 8 else 'red'
                
                axes[2].plot([px, gx], [py, gy], color=color, linewidth=2)
                axes[2].scatter([px], [py], c='blue', s=30, marker='o')
                axes[2].scatter([gx], [gy], c='red', s=30, marker='x')
        
        for ax in axes:
            ax.axis('off')
        
        plt.tight_layout()
        
        # 保存帧
        frame_path = output_dir / f'{safe_name}_frame_{t:04d}.png'
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        
        # 读取为numpy数组用于GIF
        fig.canvas.draw()
        frame_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame_data = frame_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame_data)
        
        plt.close()
    
    # 创建GIF
    try:
        import imageio
        gif_path = output_dir / f'{safe_name}.gif'
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"Saved GIF to {gif_path}")
    except ImportError:
        print("imageio not available, skipping GIF creation")
    except Exception as exc:
        print(f"Failed to create GIF: {exc}")


def visualize_frequency_decomposition(
    band_features: Dict[str, np.ndarray],
    output_path: str,
    point_idx: int = 0,
):
    """
    可视化频率分解结果
    
    Args:
        band_features: 各频带特征
        output_path: 输出路径
        point_idx: 要可视化的点索引
    """
    band_keys = sorted([
        k for k in band_features.keys()
        if k.startswith('band_') and band_features.get(k) is not None
    ])
    if not band_keys:
        print("No band_* features found, skipping frequency visualization")
        return
    
    num_bands = len(band_keys)
    fig, axes = plt.subplots(2, num_bands, figsize=(4 * num_bands, 8), squeeze=False)
    
    for i, band_key in enumerate(band_keys):
        
        band = band_features[band_key]
        
        # 如果是5D，取第一个batch和group
        while band.ndim > 3:
            band = band[0]
        
        # (T, N, C) -> 取特定点
        if band.ndim == 3:
            point_band = band[:, point_idx, :]  # (T, C)
        else:
            point_band = band
        
        T, C = point_band.shape
        
        # 时域可视化
        axes[0, i].imshow(point_band.T, aspect='auto', cmap='coolwarm')
        axes[0, i].set_title(f'Band {i} (Time Domain)')
        axes[0, i].set_xlabel('Time')
        axes[0, i].set_ylabel('Channel')
        
        # 频域可视化
        fft = np.abs(np.fft.fft(point_band, axis=0))[:T//2]
        axes[1, i].imshow(fft.T, aspect='auto', cmap='viridis')
        axes[1, i].set_title(f'Band {i} (Frequency Domain)')
        axes[1, i].set_xlabel('Frequency')
        axes[1, i].set_ylabel('Channel')
    
    plt.tight_layout()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved frequency decomposition to {output_path}")


def visualize_occlusion_prediction(
    video: np.ndarray,
    pred_visibility: np.ndarray,
    gt_visibility: np.ndarray,
    output_dir: str,
    video_name: str = 'video',
):
    """
    可视化遮挡预测结果
    
    Args:
        video: (T, H, W, 3)
        pred_visibility: (N, T)
        gt_visibility: (N, T)
        output_dir: 输出目录
        video_name: 视频名称
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[\\/:*?"<>|]', "_", str(video_name))
    if pred_visibility is None or pred_visibility.size == 0:
        raise ValueError("pred_visibility is empty")
    if gt_visibility is None or gt_visibility.size == 0:
        raise ValueError("gt_visibility is empty")
    if pred_visibility.shape != gt_visibility.shape:
        raise ValueError("pred_visibility and gt_visibility shapes mismatch")
    if pred_visibility.dtype != np.bool_:
        pred_visibility = pred_visibility > 0.5
    if gt_visibility.dtype != np.bool_:
        gt_visibility = gt_visibility > 0.5
    
    N, T = pred_visibility.shape
    
    # 计算每帧的遮挡准确率
    frame_accuracy = []
    for t in range(T):
        pred = pred_visibility[:, t] > 0.5
        gt = gt_visibility[:, t]
        acc = (pred == gt).mean()
        frame_accuracy.append(acc)
    
    # 绘制遮挡统计
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 每帧遮挡准确率
    axes[0, 0].plot(frame_accuracy, 'b-', linewidth=2)
    axes[0, 0].axhline(y=np.mean(frame_accuracy), color='r', linestyle='--', label=f'Mean: {np.mean(frame_accuracy):.3f}')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Occlusion Accuracy')
    axes[0, 0].set_title('Per-Frame Occlusion Accuracy')
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1.05])
    
    # 2. 预测vs真实遮挡比例
    pred_occ_ratio = 1 - pred_visibility.mean(axis=0)
    gt_occ_ratio = 1 - gt_visibility.mean(axis=0)
    axes[0, 1].plot(pred_occ_ratio, 'b-', label='Predicted', linewidth=2)
    axes[0, 1].plot(gt_occ_ratio, 'r--', label='Ground Truth', linewidth=2)
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Occlusion Ratio')
    axes[0, 1].set_title('Occlusion Ratio over Time')
    axes[0, 1].legend()
    
    # 3. 混淆矩阵
    pred_binary = (pred_visibility > 0.5).flatten()
    gt_binary = gt_visibility.flatten()
    
    tp = ((pred_binary == 1) & (gt_binary == 1)).sum()
    tn = ((pred_binary == 0) & (gt_binary == 0)).sum()
    fp = ((pred_binary == 1) & (gt_binary == 0)).sum()
    fn = ((pred_binary == 0) & (gt_binary == 1)).sum()
    
    confusion = np.array([[tn, fp], [fn, tp]])
    im = axes[1, 0].imshow(confusion, cmap='Blues')
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_xticklabels(['Occluded', 'Visible'])
    axes[1, 0].set_yticklabels(['Occluded', 'Visible'])
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Ground Truth')
    axes[1, 0].set_title('Confusion Matrix')
    
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, f'{confusion[i, j]}', ha='center', va='center', fontsize=14)
    
    # 4. 每个点的遮挡持续时间
    gt_occ_duration = (1 - gt_visibility).sum(axis=1)
    pred_occ_duration = (pred_visibility < 0.5).sum(axis=1)
    
    axes[1, 1].scatter(gt_occ_duration, pred_occ_duration, alpha=0.5)
    axes[1, 1].plot([0, T], [0, T], 'r--', label='Perfect')
    axes[1, 1].set_xlabel('GT Occlusion Duration')
    axes[1, 1].set_ylabel('Predicted Occlusion Duration')
    axes[1, 1].set_title('Occlusion Duration per Point')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{safe_name}_occlusion_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved occlusion analysis to {output_dir / f'{safe_name}_occlusion_analysis.png'}")


def create_comparison_video(
    video: np.ndarray,
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    gt_tracks: np.ndarray,
    gt_visibility: np.ndarray,
    output_path: str,
    max_points: int = 10,
    fps: int = 10,
):
    """
    创建多方法对比视频
    
    Args:
        video: (T, H, W, 3)
        results_dict: {method_name: (tracks, visibility)}
        gt_tracks: (N, T, 2)
        gt_visibility: (N, T)
        output_path: 输出路径
        max_points: 最大显示点数
        fps: 帧率
    """
    if video is None or video.size == 0:
        raise ValueError("video is empty")
    if gt_tracks is None or gt_tracks.size == 0:
        raise ValueError("gt_tracks is empty")
    if gt_visibility is None or gt_visibility.size == 0:
        raise ValueError("gt_visibility is empty")
    if video.ndim != 4 or gt_tracks.ndim != 3:
        raise ValueError("Invalid input dimensions for comparison video")

    if gt_visibility.dtype != np.bool_:
        gt_visibility = gt_visibility > 0.5
    T, H, W, _ = video.shape
    N = gt_tracks.shape[0]

    # Auto-detect coordinate format
    _all_tracks = [gt_tracks] + [t for t, _ in results_dict.values()]
    _max_coord = max(float(t.max()) if t.size else 0.0 for t in _all_tracks)
    _is_pixel_coords = _max_coord > 1.5
    if gt_tracks.shape[1] != T or gt_visibility.shape[1] != T:
        raise ValueError("Track length does not match video frames")
    if results_dict is None or len(results_dict) == 0:
        raise ValueError("results_dict is empty")
    num_methods = len(results_dict) + 1  # +1 for GT
    
    # 选择点
    if N > max_points:
        indices = np.linspace(0, N-1, max_points, dtype=int)
    else:
        indices = np.arange(N)
    
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    frames = []
    
    for t in tqdm(range(T), desc='Creating comparison'):
        fig, axes = plt.subplots(1, num_methods, figsize=(5 * num_methods, 5), squeeze=False)
        axes = axes[0]
        
        # Ground Truth
        axes[0].imshow(video[t])
        axes[0].set_title('Ground Truth')
        
        for j, idx in enumerate(indices):
            if gt_visibility[idx, t]:
                y, x = gt_tracks[idx, t]
                if not _is_pixel_coords:
                    y, x = y * H, x * W
                circle = Circle((x, y), 4, color=colors[j % len(colors)], fill=True, alpha=0.8)
                axes[0].add_patch(circle)
        axes[0].axis('off')
        
        # 各方法结果
        for i, (method_name, (tracks, visibility)) in enumerate(results_dict.items()):
            axes[i+1].imshow(video[t])
            axes[i+1].set_title(method_name)
            
            for j, idx in enumerate(indices):
                if visibility[idx, t] > 0.5:
                    y, x = tracks[idx, t]
                    if not _is_pixel_coords:
                        y, x = y * H, x * W
                    circle = Circle((x, y), 4, color=colors[j % len(colors)], fill=True, alpha=0.8)
                    axes[i+1].add_patch(circle)
            axes[i+1].axis('off')
        
        plt.tight_layout()
        
        fig.canvas.draw()
        frame_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame_data = frame_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame_data)
        
        plt.close()
    
    # 保存GIF
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio
        imageio.mimsave(str(output_path), frames, fps=fps)
        print(f"Saved comparison video to {output_path}")
    except ImportError:
        print("imageio not available, skipping video creation")
    except Exception as exc:
        print(f"Failed to create comparison video: {exc}")


if __name__ == '__main__':
    # 测试可视化函数
    print("Testing visualization functions...")
    
    # 创建模拟数据
    T, H, W = 24, 256, 256
    N = 50
    
    video = np.random.rand(T, H, W, 3)
    pred_tracks = np.random.rand(N, T, 2)
    gt_tracks = pred_tracks + np.random.randn(N, T, 2) * 0.02
    pred_visibility = np.random.rand(N, T) > 0.2
    gt_visibility = pred_visibility.copy()
    
    # 测试追踪可视化
    output_dir = Path(tempfile.mkdtemp(prefix='fspt_vis_test_'))
    visualize_tracking_results(
        video, pred_tracks, gt_tracks,
        pred_visibility, gt_visibility,
        str(output_dir), 'test_video',
        max_points=10, fps=5,
    )
    
    # 测试遮挡分析
    visualize_occlusion_prediction(
        video, pred_visibility, gt_visibility,
        str(output_dir), 'test_video',
    )
    
    print("Visualization test complete!")
