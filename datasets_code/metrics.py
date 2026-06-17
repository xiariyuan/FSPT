"""
TAP-Vid 评估指标

实现官方的评估指标:
- AJ (Average Jaccard): 综合位置和可见性的Jaccard指数
- < δ^x_avg: 各阈值下的位置精度
- OA (Occlusion Accuracy): 遮挡预测准确率

参考: https://github.com/deepmind/tapnet
"""

import logging
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def _extract_predictions(outputs):
    if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
        return outputs[0], outputs[1]
    raise ValueError(f"Unexpected model output type: {type(outputs)}")


def _empty_metrics(thresholds: List[int]) -> Dict[str, float]:
    metrics = {f'<{t}px': 0.0 for t in thresholds}
    metrics['<avg'] = 0.0
    metrics['OA'] = 0.0
    metrics['AJ'] = 0.0
    metrics['avg_error_px'] = 0.0
    metrics['median_error_px'] = 0.0
    return metrics


def _resolve_resolution_from_batch(
    batch: Dict[str, Any],
    index: int,
    fallback: Tuple[int, int],
) -> Tuple[int, int]:
    """从batch中解析原始分辨率（若存在），否则使用fallback。"""
    if not isinstance(batch, dict):
        return fallback
    original_size = batch.get('original_size', None)
    if original_size is None:
        return fallback
    try:
        size = None
        if isinstance(original_size, torch.Tensor):
            if original_size.ndim == 2 and original_size.shape[0] > index:
                size = original_size[index].tolist()
            else:
                size = original_size.tolist()
        elif isinstance(original_size, (list, tuple)):
            if len(original_size) > 0 and isinstance(original_size[0], (list, tuple, torch.Tensor)):
                size = original_size[index] if index < len(original_size) else original_size[0]
                if hasattr(size, 'tolist'):
                    size = size.tolist()
            else:
                size = original_size
        else:
            size = original_size
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            return (int(size[0]), int(size[1]))
    except Exception:
        return fallback
    return fallback


def compute_tapvid_metrics(
    pred_tracks: torch.Tensor,
    gt_tracks: torch.Tensor,
    pred_visibility: torch.Tensor,
    gt_visibility: torch.Tensor,
    query_points: torch.Tensor,
    thresholds: List[int] = [1, 2, 4, 8, 16],
    resolution: Optional[Union[int, Tuple[int, int]]] = 256,
    exclude_query_frame: bool = True,
    query_mode: Optional[str] = None,
) -> Dict[str, float]:
    """
    计算TAP-Vid评估指标
    
    Args:
        pred_tracks: (N, T, 2) 预测轨迹，归一化坐标[0,1]，格式为[y, x]
        gt_tracks: (N, T, 2) 真实轨迹，归一化坐标[0,1]，格式为[y, x]
        pred_visibility: (N, T) 预测可见性 (True=可见)
        gt_visibility: (N, T) 真实可见性 (True=可见)
        query_points: (N, 3) 查询点 [t, y, x]
        thresholds: 位置阈值列表（像素）
        resolution: 用于将归一化坐标转换为像素的分辨率 (H, W) 或单一整数
        exclude_query_frame: 是否排除查询帧参与指标计算
        query_mode: TAP-Vid评估模式，可选 'first' 或 'strided'
            - 'first': 仅评估查询帧之后的帧（官方 first-query 协议）
            - 'strided': 评估除查询帧外的所有帧（官方 strided 协议）
            - None: 兼容旧逻辑，使用 exclude_query_frame 决定掩码
        
    Returns:
        metrics: 包含各指标的字典
    """
    if thresholds is None or len(thresholds) == 0:
        thresholds = [1, 2, 4, 8, 16]
    device = pred_tracks.device
    if gt_tracks.device != device:
        gt_tracks = gt_tracks.to(device)
    if pred_visibility.device != device:
        pred_visibility = pred_visibility.to(device)
    if gt_visibility.device != device:
        gt_visibility = gt_visibility.to(device)
    if query_points.device != device:
        query_points = query_points.to(device)
    N, T = pred_tracks.shape[:2]
    min_n = min(
        pred_tracks.shape[0],
        gt_tracks.shape[0],
        pred_visibility.shape[0],
        gt_visibility.shape[0],
        query_points.shape[0],
    )
    min_t = min(
        pred_tracks.shape[1],
        gt_tracks.shape[1],
        pred_visibility.shape[1],
        gt_visibility.shape[1],
    )
    if min_n == 0 or min_t == 0:
        logger.warning("Empty sequence for metric computation, returning zeros.")
        return _empty_metrics(thresholds)
    if min_n != N or min_t != T:
        pred_tracks = pred_tracks[:min_n, :min_t]
        gt_tracks = gt_tracks[:min_n, :min_t]
        pred_visibility = pred_visibility[:min_n, :min_t]
        gt_visibility = gt_visibility[:min_n, :min_t]
        query_points = query_points[:min_n]
        N, T = pred_tracks.shape[:2]
    if query_points is None or query_points.numel() == 0:
        logger.warning("Empty query_points for metric computation, returning zeros.")
        return _empty_metrics(thresholds)

    if torch.isnan(pred_tracks).any() or torch.isinf(pred_tracks).any():
        logger.warning("NaN/Inf in pred_tracks, returning zeros.")
        return _empty_metrics(thresholds)
    if torch.isnan(gt_tracks).any() or torch.isinf(gt_tracks).any():
        logger.warning("NaN/Inf in gt_tracks, returning zeros.")
        return _empty_metrics(thresholds)
    
    # 确保是布尔类型
    if pred_visibility.dtype != torch.bool:
        pred_visibility = pred_visibility > 0.5
    if gt_visibility.dtype != torch.bool:
        gt_visibility = gt_visibility > 0.5
    
    # 将归一化坐标转换为像素坐标
    if resolution is None:
        resolution = 256
    if isinstance(resolution, (tuple, list)):
        scale = torch.tensor([resolution[0], resolution[1]], device=device, dtype=pred_tracks.dtype)
    else:
        scale = torch.tensor([resolution, resolution], device=device, dtype=pred_tracks.dtype)
    pred_min = float(pred_tracks.min())
    pred_max = float(pred_tracks.max())
    if pred_min < 0.0 or pred_max > 1.0:
        logger.warning("Pred tracks out of [0,1], clamping.")
        pred_tracks = pred_tracks.clamp(0.0, 1.0)
    gt_min = float(gt_tracks.min())
    gt_max = float(gt_tracks.max())
    if gt_min < 0.0 or gt_max > 1.0:
        logger.warning("GT tracks out of [0,1], clamping.")
        gt_tracks = gt_tracks.clamp(0.0, 1.0)

    pred_tracks_px = pred_tracks * scale
    gt_tracks_px = gt_tracks * scale
    
    # 计算位置误差 (欧氏距离)
    position_error = torch.norm(pred_tracks_px - gt_tracks_px, dim=-1)  # (N, T)
    
    metrics = {}
    
    # 创建评估掩码
    query_frame_mask = torch.zeros(N, T, dtype=torch.bool, device=device)
    query_t = query_points[:, 0].round().long().clamp(0, T - 1)
    query_frame_mask[torch.arange(N, device=device), query_t] = True
    if query_mode is not None:
        mode = str(query_mode).lower().strip()
        if mode == 'strided':
            # 官方 strided：除查询帧外全部评估
            valid_mask = ~query_frame_mask
        elif mode == 'first':
            # 官方 first：仅评估查询帧之后的帧
            frame_idx = torch.arange(T, device=device).view(1, T)
            valid_mask = frame_idx > query_t.view(N, 1)
        else:
            raise ValueError(f"Unknown query_mode: {query_mode}")
    else:
        valid_mask = ~query_frame_mask if exclude_query_frame else torch.ones_like(
            query_frame_mask, dtype=torch.bool
        )

    # =========================================================================
    # 1. 各阈值下的精度 (< δ^x_avg)
    # 只在GT可见的帧上计算（可选排除查询帧）
    # =========================================================================
    for thresh in thresholds:
        within_thresh = position_error < thresh  # (N, T)
        
        # 只在GT可见时计算
        visible_mask = gt_visibility & valid_mask
        if visible_mask.sum() > 0:
            correct_visible = (within_thresh & visible_mask).sum().float()
            total_visible = visible_mask.sum().float()
            accuracy = correct_visible / total_visible
        else:
            accuracy = torch.tensor(0.0, device=device)
        
        metrics[f'<{thresh}px'] = accuracy.item()
    
    # 平均位置精度
    metrics['<avg'] = np.mean([metrics[f'<{t}px'] for t in thresholds])
    
    # =========================================================================
    # 2. 遮挡准确率 (OA)
    # 可选不在查询帧计算
    # =========================================================================
    non_query_mask = valid_mask
    pred_occ = ~pred_visibility
    gt_occ = ~gt_visibility
    
    if non_query_mask.sum() > 0:
        occ_correct = ((pred_occ == gt_occ) & non_query_mask).sum().float()
        occ_total = non_query_mask.sum().float()
        metrics['OA'] = (occ_correct / occ_total).item()
    else:
        metrics['OA'] = 0.0
    
    # =========================================================================
    # 3. Average Jaccard (AJ)
    # AJ = 平均(所有阈值下的Jaccard)
    # Jaccard考虑位置准确性和可见性预测
    # =========================================================================
    jaccard_values = []
    
    for thresh in thresholds:
        within_thresh = position_error < thresh
        
        # True Positive: 预测可见且位置正确，真实也可见
        true_positive = within_thresh & pred_visibility & gt_visibility
        
        # False Positive: 
        # 1. 预测可见但真实不可见
        # 2. 预测可见且真实可见，但位置错误
        fp_wrong_vis = pred_visibility & ~gt_visibility
        fp_wrong_pos = pred_visibility & gt_visibility & ~within_thresh
        false_positive = fp_wrong_vis | fp_wrong_pos
        
        # 排除查询帧（可选）
        true_positive = true_positive & valid_mask
        false_positive = false_positive & valid_mask
        
        # TAP-Vid official aggregation:
        # Jaccard = TP / (GT positives + FP), where
        # GT positives = number of visible GT points at evaluation frames.
        # This is equivalent to TP / (TP + FN + FP), but computed with global
        # counts over the whole video for numerical parity with official code.
        true_positive_count = true_positive.sum().float()
        gt_positive_count = (gt_visibility & valid_mask).sum().float()
        false_positive_count = false_positive.sum().float()
        denom = gt_positive_count + false_positive_count
        if denom > 0:
            jaccard = true_positive_count / denom
        else:
            jaccard = torch.tensor(0.0, device=device)

        jaccard_values.append(jaccard.item())
    
    metrics['AJ'] = np.mean(jaccard_values)
    
    # =========================================================================
    # 4. 额外指标
    # =========================================================================
    # 平均位置误差（仅可见点，可选排除查询帧）
    valid_visible = gt_visibility & valid_mask
    if valid_visible.sum() > 0:
        avg_error = position_error[valid_visible].mean()
        metrics['avg_error_px'] = avg_error.item()
    else:
        metrics['avg_error_px'] = 0.0
    
    # 中位数位置误差
    if valid_visible.sum() > 0:
        median_error = position_error[valid_visible].median()
        metrics['median_error_px'] = median_error.item()
    else:
        metrics['median_error_px'] = 0.0
    
    return metrics


def compute_dataset_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    verbose: bool = True,
    exclude_query_frame: bool = True,
    query_mode: Optional[str] = None,
) -> Dict[str, float]:
    """
    在整个数据集上计算评估指标
    
    Args:
        model: 点追踪模型，接收 (video, query_points) 返回 (tracks, visibility)
        dataloader: 数据加载器
        device: 设备
        verbose: 是否打印进度
        exclude_query_frame: 是否排除查询帧
        query_mode: TAP-Vid评估模式，可选 'first' 或 'strided'
        
    Returns:
        dataset_metrics: 数据集级别的平均指标
    """
    from tqdm import tqdm
    
    if dataloader is None:
        logger.warning("No dataloader provided; skipping dataset metrics.")
        return {}
    try:
        if len(dataloader) == 0:
            logger.warning("Empty dataloader; skipping dataset metrics.")
            return {}
    except TypeError:
        pass

    device = torch.device(device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            index = device.index if device.index is not None else 0
            if index >= torch.cuda.device_count():
                logger.warning(f"CUDA device index {index} out of range; using cuda:0")
                device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()
    
    all_metrics = []
    
    iterator = tqdm(dataloader, desc='Evaluating') if verbose else dataloader
    
    with torch.no_grad():
        for batch in iterator:
            try:
                if batch is None or not isinstance(batch, dict):
                    logger.warning("Empty or invalid batch in dataset metrics, skipping.")
                    continue
                required_keys = ('video', 'query_points', 'target_points', 'occluded')
                missing_keys = [k for k in required_keys if k not in batch]
                if missing_keys:
                    logger.warning(f"Batch missing keys {missing_keys} in dataset metrics, skipping.")
                    continue
                video = batch['video'].to(device)
                query_points = batch['query_points'].to(device)
                target_points = batch['target_points'].to(device)
                occluded = batch['occluded'].to(device)
                if occluded.dtype != torch.bool:
                    occluded = occluded > 0.5
                if video.ndim == 4:
                    video = video.unsqueeze(0)
                    if query_points.ndim == 2:
                        query_points = query_points.unsqueeze(0)
                    if target_points.ndim == 3:
                        target_points = target_points.unsqueeze(0)
                    if occluded.ndim == 2:
                        occluded = occluded.unsqueeze(0)

                # 模型预测
                outputs = model(video, query_points)
                pred_tracks, pred_visibility = _extract_predictions(outputs)

                # 计算每个样本的指标
                batch_size = video.shape[0]
                for i in range(batch_size):
                    resolution = _resolve_resolution_from_batch(
                        batch, i, tuple(video.shape[-2:])
                    )
                    metrics = compute_tapvid_metrics(
                        pred_tracks[i],
                        target_points[i],
                        pred_visibility[i],
                        ~occluded[i],  # 转换为可见性
                        query_points[i],
                        resolution=resolution,
                        exclude_query_frame=exclude_query_frame,
                        query_mode=query_mode,
                    )
                    all_metrics.append(metrics)
            except Exception as exc:
                logger.warning(f"Skipping invalid batch during metric computation: {exc}")
                continue
    
    # 平均所有样本的指标
    if len(all_metrics) == 0:
        logger.warning("No valid metrics computed for dataset.")
        return {}
    
    def _safe_float(value):
        try:
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    return None
                value = value.detach().cpu().item()
            return float(value)
        except Exception:
            return None

    dataset_metrics = {}
    for key in all_metrics[0].keys():
        values = []
        for m in all_metrics:
            v = _safe_float(m.get(key, None))
            if v is None or not np.isfinite(v):
                continue
            values.append(v)
        if values:
            dataset_metrics[key] = float(np.mean(values))
            dataset_metrics[f'{key}_std'] = float(np.std(values))
        else:
            dataset_metrics[key] = 0.0
            dataset_metrics[f'{key}_std'] = 0.0
    
    return dataset_metrics


def compute_per_sequence_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    exclude_query_frame: bool = True,
    query_mode: Optional[str] = None,
) -> List[Dict]:
    """
    计算每个序列的指标
    
    Returns:
        per_seq_metrics: 每个序列的指标列表
    """
    from tqdm import tqdm
    
    if dataloader is None:
        logger.warning("No dataloader provided; skipping per-sequence metrics.")
        return []
    try:
        if len(dataloader) == 0:
            logger.warning("Empty dataloader; skipping per-sequence metrics.")
            return []
    except TypeError:
        pass

    device = torch.device(device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            device = torch.device('cpu')
        else:
            index = device.index if device.index is not None else 0
            if index >= torch.cuda.device_count():
                logger.warning(f"CUDA device index {index} out of range; using cuda:0")
                device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()
    
    per_seq_metrics = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            try:
                if batch is None or not isinstance(batch, dict):
                    logger.warning("Empty or invalid batch in per-sequence metrics, skipping.")
                    continue
                required_keys = ('video', 'query_points', 'target_points', 'occluded')
                missing_keys = [k for k in required_keys if k not in batch]
                if missing_keys:
                    logger.warning(f"Batch missing keys {missing_keys} in per-sequence metrics, skipping.")
                    continue
                video = batch['video'].to(device)
                query_points = batch['query_points'].to(device)
                target_points = batch['target_points'].to(device)
                occluded = batch['occluded'].to(device)
                if occluded.dtype != torch.bool:
                    occluded = occluded > 0.5
                if video.ndim == 4:
                    video = video.unsqueeze(0)
                    if query_points.ndim == 2:
                        query_points = query_points.unsqueeze(0)
                    if target_points.ndim == 3:
                        target_points = target_points.unsqueeze(0)
                    if occluded.ndim == 2:
                        occluded = occluded.unsqueeze(0)

                # 模型预测
                outputs = model(video, query_points)
                pred_tracks, pred_visibility = _extract_predictions(outputs)

                # 计算指标
                batch_size = int(video.shape[0])
                video_names = batch.get('video_name', None)
                for i in range(batch_size):
                    resolution = _resolve_resolution_from_batch(
                        batch, i, tuple(video.shape[-2:])
                    )
                    if isinstance(video_names, (list, tuple)):
                        video_name = str(video_names[i]) if i < len(video_names) else f"unknown_{i}"
                    elif isinstance(video_names, torch.Tensor):
                        if video_names.ndim == 0:
                            video_name = str(video_names.item())
                        elif i < len(video_names):
                            video_name = str(video_names[i])
                        else:
                            video_name = f"unknown_{i}"
                    elif video_names is not None:
                        video_name = str(video_names)
                    else:
                        video_name = f"unknown_{i}"

                    metrics = compute_tapvid_metrics(
                        pred_tracks[i],
                        target_points[i],
                        pred_visibility[i],
                        ~occluded[i],
                        query_points[i],
                        resolution=resolution,
                        exclude_query_frame=exclude_query_frame,
                        query_mode=query_mode,
                    )

                    per_seq_metrics.append({
                        'video_name': video_name,
                        **metrics,
                    })
            except Exception as exc:
                logger.warning(f"Skipping invalid sample in per-sequence metrics: {exc}")
                continue
    
    return per_seq_metrics


def summarize_metrics(per_seq_metrics: List[Dict]) -> Dict[str, Dict]:
    """
    汇总每个序列的指标
    
    Returns:
        summary: 包含mean, std, min, max的汇总
    """
    if len(per_seq_metrics) == 0:
        return {}
    
    # 获取所有指标名
    metric_names = [k for k in per_seq_metrics[0].keys() if k != 'video_name']
    
    summary = {}
    for name in metric_names:
        values = [m[name] for m in per_seq_metrics]
        summary[name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
        }
    
    return summary


if __name__ == '__main__':
    # 测试指标计算
    print("Testing TAP-Vid metrics...")
    
    # 创建模拟数据
    N, T = 10, 50
    
    # 完美预测
    gt_tracks = torch.rand(N, T, 2)
    pred_tracks = gt_tracks.clone()
    gt_visibility = torch.ones(N, T, dtype=torch.bool)
    pred_visibility = gt_visibility.clone()
    query_points = torch.zeros(N, 3)
    query_points[:, 0] = 0  # 查询帧
    query_points[:, 1:3] = gt_tracks[:, 0]  # 查询位置
    
    metrics = compute_tapvid_metrics(
        pred_tracks, gt_tracks, pred_visibility, gt_visibility, query_points
    )
    
    print("\nPerfect prediction:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    assert metrics['AJ'] > 0.99, "Perfect prediction should have AJ close to 1"
    
    # 加入噪声
    pred_tracks_noisy = gt_tracks + torch.randn_like(gt_tracks) * 0.02
    
    metrics_noisy = compute_tapvid_metrics(
        pred_tracks_noisy, gt_tracks, pred_visibility, gt_visibility, query_points
    )
    
    print("\nNoisy prediction:")
    for k, v in metrics_noisy.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nTest passed!")

