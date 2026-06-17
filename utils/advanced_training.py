"""
先进训练策略

基于近几年顶会论文（CoTracker3, TAPIR, BootsTAP, TAPNext等）的训练技巧

包含:
1. 渐进式训练（分辨率、视频长度、点数量）
2. 伪标签自训练框架
3. 置信度加权损失
4. 合成到真实域适应
5. 时序一致性正则化
"""

import math
import logging
from typing import Optional, Dict, Tuple, List, Any, Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# =============================================================================
# 1. 渐进式训练调度器 (Progressive Training Scheduler)
# =============================================================================

@dataclass
class ProgressiveStage:
    """渐进式训练阶段配置"""
    epochs: int  # 该阶段的epoch数
    resolution: Tuple[int, int]  # 图像分辨率
    num_frames: int  # 视频帧数
    num_points: int  # 查询点数量
    batch_size: int  # 批次大小
    learning_rate: float  # 学习率


class ProgressiveTrainingScheduler:
    """
    渐进式训练调度器
    
    基于CoTracker3和TAPIR的训练策略：
    - 从低分辨率/短视频开始
    - 逐步增加难度
    - 稳定训练过程
    
    用法:
        scheduler = ProgressiveTrainingScheduler([
            ProgressiveStage(epochs=20, resolution=(128, 128), num_frames=8, ...),
            ProgressiveStage(epochs=40, resolution=(256, 256), num_frames=16, ...),
            ProgressiveStage(epochs=40, resolution=(256, 256), num_frames=24, ...),
        ])
        
        for epoch in range(total_epochs):
            stage = scheduler.get_current_stage(epoch)
            # 使用stage.resolution, stage.num_frames等配置数据加载
    """
    
    def __init__(
        self,
        stages: List[ProgressiveStage],
        smooth_transition: bool = True,
        transition_epochs: int = 2,
    ):
        """
        Args:
            stages: 训练阶段列表
            smooth_transition: 是否平滑过渡
            transition_epochs: 过渡epoch数
        """
        self.stages = stages
        self.smooth_transition = smooth_transition
        self.transition_epochs = transition_epochs
        
        # 计算累积epoch
        self.stage_boundaries = []
        cumsum = 0
        for stage in stages:
            cumsum += stage.epochs
            self.stage_boundaries.append(cumsum)
        
        self.total_epochs = cumsum
        logger.info(f"ProgressiveTraining: {len(stages)} stages, {self.total_epochs} total epochs")
    
    def get_current_stage(self, epoch: int) -> ProgressiveStage:
        """获取当前阶段配置"""
        for i, boundary in enumerate(self.stage_boundaries):
            if epoch < boundary:
                return self.stages[i]
        return self.stages[-1]
    
    def get_stage_index(self, epoch: int) -> int:
        """获取当前阶段索引"""
        for i, boundary in enumerate(self.stage_boundaries):
            if epoch < boundary:
                return i
        return len(self.stages) - 1
    
    def get_interpolated_config(self, epoch: int) -> Dict[str, Any]:
        """
        获取插值配置（平滑过渡）
        
        在阶段切换时平滑过渡分辨率和帧数
        """
        stage_idx = self.get_stage_index(epoch)
        current_stage = self.stages[stage_idx]
        
        if not self.smooth_transition or stage_idx == 0:
            return {
                'resolution': current_stage.resolution,
                'num_frames': current_stage.num_frames,
                'num_points': current_stage.num_points,
                'batch_size': current_stage.batch_size,
                'learning_rate': current_stage.learning_rate,
            }
        
        # 计算阶段内的位置
        stage_start = 0 if stage_idx == 0 else self.stage_boundaries[stage_idx - 1]
        epoch_in_stage = epoch - stage_start
        
        # 过渡期间插值
        if self.transition_epochs <= 0:
            return {
                'resolution': current_stage.resolution,
                'num_frames': current_stage.num_frames,
                'num_points': current_stage.num_points,
                'batch_size': current_stage.batch_size,
                'learning_rate': current_stage.learning_rate,
            }
        if epoch_in_stage < self.transition_epochs:
            prev_stage = self.stages[stage_idx - 1]
            alpha = epoch_in_stage / self.transition_epochs
            
            # 分辨率插值（取整）
            h = int(prev_stage.resolution[0] * (1 - alpha) + current_stage.resolution[0] * alpha)
            w = int(prev_stage.resolution[1] * (1 - alpha) + current_stage.resolution[1] * alpha)
            # 确保是32的倍数，且不为0
            h = max(32, (h // 32) * 32)
            w = max(32, (w // 32) * 32)
            
            return {
                'resolution': (h, w),
                'num_frames': current_stage.num_frames,  # 帧数不插值
                'num_points': current_stage.num_points,
                'batch_size': current_stage.batch_size,
                'learning_rate': prev_stage.learning_rate * (1 - alpha) + current_stage.learning_rate * alpha,
            }
        
        return {
            'resolution': current_stage.resolution,
            'num_frames': current_stage.num_frames,
            'num_points': current_stage.num_points,
            'batch_size': current_stage.batch_size,
            'learning_rate': current_stage.learning_rate,
        }


def create_progressive_stages(config) -> List[ProgressiveStage]:
    """
    根据配置创建渐进式训练阶段
    
    默认策略（基于CoTracker3）：
    Stage 1: 低分辨率，短视频，热身
    Stage 2: 中等设置
    Stage 3: 完整设置
    """
    base_lr = float(config.training.optimizer.lr)
    base_epochs = int(config.training.epochs)

    resolution_cfg = getattr(config.data.train, 'resolution', None)
    if resolution_cfg is None:
        target_resolution = (256, 256)
    else:
        target_resolution = tuple(resolution_cfg)
    target_frames = int(getattr(config.data.train, 'num_frames', 24) or 24)
    if target_frames <= 0:
        target_frames = 24
    target_points = int(getattr(config.data.train, 'num_points', 256) or 256)
    if target_points <= 0:
        target_points = 256
    batch_size = int(getattr(config.training, 'batch_size', 1) or 1)

    if base_epochs <= 0:
        raise ValueError("training.epochs must be positive for progressive training")

    if base_epochs < 3:
        return [
            ProgressiveStage(
                epochs=base_epochs,
                resolution=target_resolution,
                num_frames=target_frames,
                num_points=target_points,
                batch_size=batch_size,
                learning_rate=base_lr,
            )
        ]
    
    stage1_epochs = max(1, int(round(base_epochs * 0.2)))
    stage2_epochs = max(1, int(round(base_epochs * 0.3)))
    stage3_epochs = base_epochs - stage1_epochs - stage2_epochs
    if stage3_epochs < 1:
        stage3_epochs = 1
        stage2_epochs = max(1, base_epochs - stage1_epochs - stage3_epochs)
        if stage1_epochs + stage2_epochs + stage3_epochs > base_epochs:
            stage1_epochs = max(1, base_epochs - stage2_epochs - stage3_epochs)

    stage1_resolution = (
        max(1, target_resolution[0] // 2),
        max(1, target_resolution[1] // 2),
    )
    stage1_frames = max(1, min(target_frames, max(8, target_frames // 3)))
    stage2_frames = max(1, min(target_frames, max(12, target_frames // 2)))
    stage1_points = max(1, min(target_points, max(64, target_points // 4)))
    stage2_points = max(1, min(target_points, max(128, target_points // 2)))

    stages = [
        # Stage 1: 热身（20%）
        ProgressiveStage(
            epochs=stage1_epochs,
            resolution=stage1_resolution,
            num_frames=stage1_frames,
            num_points=stage1_points,
            batch_size=batch_size * 2,  # 低分辨率可用更大batch
            learning_rate=base_lr * 0.5,  # 热身学习率
        ),
        # Stage 2: 中等（30%）
        ProgressiveStage(
            epochs=stage2_epochs,
            resolution=target_resolution,
            num_frames=stage2_frames,
            num_points=stage2_points,
            batch_size=batch_size,
            learning_rate=base_lr,
        ),
        # Stage 3: 完整（50%）
        ProgressiveStage(
            epochs=stage3_epochs,
            resolution=target_resolution,
            num_frames=target_frames,
            num_points=target_points,
            batch_size=batch_size,
            learning_rate=base_lr * 0.5,  # 后期降低学习率
        ),
    ]
    
    return stages


# =============================================================================
# 2. 置信度加权损失 (Confidence-Weighted Loss)
# =============================================================================

class ConfidenceWeightedLoss(nn.Module):
    """
    置信度加权损失
    
    基于CoTracker3的设计：
    - 模型预测轨迹置信度
    - 使用置信度加权损失
    - 高置信度样本权重更大
    
    损失 = mean(confidence * position_loss + λ * confidence_regularization)
    """
    
    def __init__(
        self,
        position_loss_type: str = 'smooth_l1',
        confidence_weight: float = 0.1,
        min_confidence: float = 0.1,
    ):
        """
        Args:
            position_loss_type: 位置损失类型 ('l1', 'l2', 'smooth_l1', 'huber')
            confidence_weight: 置信度正则化权重
            min_confidence: 最小置信度（防止忽略所有样本）
        """
        super().__init__()
        
        self.confidence_weight = confidence_weight
        self.min_confidence = min_confidence
        
        if position_loss_type == 'l1':
            self.position_loss_fn = nn.L1Loss(reduction='none')
        elif position_loss_type == 'l2':
            self.position_loss_fn = nn.MSELoss(reduction='none')
        elif position_loss_type == 'smooth_l1':
            self.position_loss_fn = nn.SmoothL1Loss(reduction='none', beta=0.05)
        elif position_loss_type == 'huber':
            self.position_loss_fn = nn.HuberLoss(reduction='none', delta=0.5)
        else:
            raise ValueError(f"Unknown position loss type: {position_loss_type}")
    
    def forward(
        self,
        pred_tracks: torch.Tensor,
        gt_tracks: torch.Tensor,
        confidence: torch.Tensor,
        visibility: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_tracks: (B, N, T, 2) 预测轨迹
            gt_tracks: (B, N, T, 2) 真实轨迹
            confidence: (B, N, T) 预测置信度
            visibility: (B, N, T) 可见性掩码
            
        Returns:
            损失字典
        """
        # 位置损失
        position_loss = self.position_loss_fn(pred_tracks, gt_tracks)  # (B, N, T, 2)
        position_loss = position_loss.sum(dim=-1)  # (B, N, T)
        
        # 置信度加权
        # 高置信度 -> 更信任预测 -> 如果错了惩罚更大
        confidence_clamped = torch.clamp(confidence, min=self.min_confidence)
        weighted_loss = confidence_clamped * position_loss
        
        # 只在可见帧计算
        valid_mask = visibility.float()
        
        if valid_mask.sum() > 0:
            weighted_position_loss = (weighted_loss * valid_mask).sum() / valid_mask.sum()
        else:
            weighted_position_loss = torch.tensor(0.0, device=pred_tracks.device)
        
        # 置信度正则化：鼓励高置信度
        # 但不能太高，否则模型会过于自信
        confidence_reg = -torch.log(confidence_clamped + 1e-8) * valid_mask
        if valid_mask.sum() > 0:
            confidence_reg = confidence_reg.sum() / valid_mask.sum()
        else:
            confidence_reg = torch.tensor(0.0, device=pred_tracks.device)
        
        total_loss = weighted_position_loss + self.confidence_weight * confidence_reg
        
        return {
            'total': total_loss,
            'weighted_position': weighted_position_loss,
            'confidence_reg': confidence_reg,
            'mean_confidence': confidence.mean(),
        }


# =============================================================================
# 3. 伪标签自训练 (Pseudo-Label Self-Training)
# =============================================================================

class PseudoLabelTrainer:
    """
    伪标签自训练框架
    
    基于CoTracker3和BootsTAP的设计：
    - 使用教师模型在无标签真实视频上生成伪标签
    - 学生模型在伪标签上训练
    - 逐步更新教师模型
    
    用法:
        teacher = create_model()
        student = create_model()
        trainer = PseudoLabelTrainer(teacher, student)
        
        for batch in unlabeled_dataloader:
            loss = trainer.train_step(batch, optimizer)
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        confidence_threshold: float = 0.5,
        teacher_momentum: float = 0.999,
        use_soft_labels: bool = True,
        confidence_source: str = "auto",
    ):
        """
        Args:
            teacher_model: 教师模型（生成伪标签）
            student_model: 学生模型（被训练）
            confidence_threshold: 伪标签置信度阈值
            teacher_momentum: 教师模型EMA动量
            use_soft_labels: 是否使用软标签
        """
        self.teacher = teacher_model
        self.student = student_model
        self.confidence_threshold = confidence_threshold
        self.teacher_momentum = teacher_momentum
        self.use_soft_labels = use_soft_labels
        self.confidence_source = self._resolve_confidence_source(confidence_source)
        
        # 教师模型冻结
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        logger.info(
            "PseudoLabelTrainer initialized: "
            f"threshold={confidence_threshold}, confidence_source={self.confidence_source}"
        )

    @staticmethod
    def _resolve_confidence_source(source: Optional[str]) -> str:
        src = str(source or "auto").strip().lower()
        if src in ("", "none", "null"):
            return "auto"
        return src

    @staticmethod
    def _confidence_keys(source: str) -> Tuple[str, ...]:
        aliases = {
            "auto": (
                "verifier_scores",
                "relocal_acceptor",
                "policy_gate",
                "confidence",
                "pred_confidence",
                "confidence_map",
                "visibility",
                "pred_visibility",
            ),
            "verifier_scores": (
                "verifier_scores",
                "relocal_acceptor",
                "verifier",
                "confidence",
                "pred_confidence",
                "confidence_map",
                "visibility",
                "pred_visibility",
            ),
            "relocal_acceptor": (
                "relocal_acceptor",
                "verifier_scores",
                "verifier",
                "confidence",
                "pred_confidence",
                "confidence_map",
                "visibility",
                "pred_visibility",
            ),
            "policy_gate": (
                "policy_gate",
                "confidence",
                "pred_confidence",
                "confidence_map",
                "visibility",
                "pred_visibility",
            ),
            "confidence": (
                "confidence",
                "pred_confidence",
                "confidence_map",
                "visibility",
                "pred_visibility",
            ),
            "pred_visibility": ("pred_visibility", "visibility"),
            "neighbor_deviation": (
                "neighbor_deviation_confidence",
                "neighbor_deviation_reliability",
                "neighbor_deviation_raw_risk",
                "neighbor_deviation_risk",
            ),
        }
        return aliases.get(source, aliases["auto"])

    def _extract_teacher_confidence(
        self,
        outputs: Tuple[object, ...],
        fallback_confidence: torch.Tensor,
    ) -> torch.Tensor:
        confidence = fallback_confidence
        if len(outputs) < 3 or not isinstance(outputs[2], dict):
            return confidence

        extra = outputs[2]
        for key in self._confidence_keys(self.confidence_source):
            value = extra.get(key, None)
            if isinstance(value, torch.Tensor):
                confidence = value
                break
        return confidence
    
    @torch.no_grad()
    def generate_pseudo_labels(
        self,
        video: torch.Tensor,
        query_points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使用教师模型生成伪标签
        
        Returns:
            pseudo_tracks: 伪标签轨迹
            pseudo_visibility: 伪标签可见性
            confidence_mask: 高置信度掩码
        """
        self.teacher.eval()

        with torch.no_grad():
            try:
                outputs = self.teacher(video, query_points, return_info=True)
            except TypeError:
                outputs = self.teacher(video, query_points)
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
            raise ValueError("Unexpected teacher output format")

        pred_tracks, pred_visibility = outputs[0], outputs[1]

        # Compute neighbor_deviation risk from predicted tracks if requested
        if self.confidence_source in ("neighbor_deviation",):
            from utils.neighbor_deviation import compute_neighbor_deviation_risk, normalize_risk_to_01

            nd_risk = compute_neighbor_deviation_risk(pred_tracks, pred_visibility, K=16)
            nd_risk_norm = normalize_risk_to_01(nd_risk, pred_visibility)
            # Pseudo-label filtering expects confidence semantics: higher = safer.
            # Neighbor deviation is a risk score: higher = worse, so invert it here.
            vis_mask = pred_visibility > 0.5
            nd_conf = torch.where(vis_mask, 1.0 - nd_risk_norm, torch.zeros_like(nd_risk_norm))

            extra = {}
            if len(outputs) >= 3 and isinstance(outputs[2], dict):
                extra = dict(outputs[2])
            extra["neighbor_deviation_raw_risk"] = nd_risk_norm
            extra["neighbor_deviation_risk"] = nd_risk_norm
            extra["neighbor_deviation_confidence"] = nd_conf
            extra["neighbor_deviation_reliability"] = nd_conf
            outputs = (outputs[0], outputs[1], extra)

        confidence = self._extract_teacher_confidence(outputs, pred_visibility)

        if confidence.dim() == 4 and confidence.shape[-1] == 1:
            confidence = confidence.squeeze(-1)
        # 对齐置信度形状为 (B, N, T)
        if confidence.dim() == 3 and pred_visibility.dim() == 3:
            if confidence.shape != pred_visibility.shape:
                if confidence.shape[1] == pred_visibility.shape[2] and confidence.shape[2] == pred_visibility.shape[1]:
                    confidence = confidence.permute(0, 2, 1)
        if confidence.dim() == 2 and pred_visibility.dim() == 3:
            B, N, T = pred_visibility.shape
            if confidence.shape[0] == B * N and confidence.shape[1] == T:
                confidence = confidence.view(B, N, T)
            elif confidence.shape[0] == B and confidence.shape[1] == N:
                confidence = confidence.unsqueeze(-1).expand_as(pred_visibility)
        if confidence.shape != pred_visibility.shape:
            logger.warning(
                f"Confidence shape mismatch: {confidence.shape} vs {pred_visibility.shape}, "
                "falling back to pred_visibility."
            )
            confidence = pred_visibility
        if confidence.device != pred_visibility.device:
            confidence = confidence.to(pred_visibility.device)
        
        # 生成高置信度掩码
        confidence_mask = confidence > self.confidence_threshold

        pseudo_visibility = pred_visibility
        if not self.use_soft_labels:
            pseudo_visibility = pred_visibility > 0.5
        
        return pred_tracks, pseudo_visibility, confidence_mask
    
    def train_step(
        self,
        video: torch.Tensor,
        query_points: torch.Tensor,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        amp_dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, float]:
        """
        单步伪标签训练
        """
        # 生成伪标签
        pseudo_tracks, pseudo_visibility, confidence_mask = self.generate_pseudo_labels(
            video, query_points
        )
        has_valid = confidence_mask.any()
        if dist.is_available() and dist.is_initialized():
            flag = has_valid.float()
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            global_has_valid = bool(flag.item() > 0)
        else:
            global_has_valid = bool(has_valid.item())
        if not global_has_valid:
            return {'total': 0.0, 'skipped': True, 'did_step': False}
        skip_local = not bool(has_valid.item())
        
        # 学生模型前向传播
        self.student.train()
        
        use_amp = (amp_dtype is not None) or (scaler is not None)
        amp_enabled = use_amp and video.device.type == 'cuda' and torch.cuda.is_available()
        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype if amp_enabled else None):
            outputs = self.student(video, query_points)
            
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
                student_tracks, student_visibility = outputs[0], outputs[1]
            else:
                raise ValueError("Unexpected student output format")
            
            # 只在高置信度样本上计算损失
            loss_dict = criterion(
                student_tracks,
                pseudo_tracks,
                student_visibility,
                pseudo_visibility,
                visibility_mask=confidence_mask,
            )
            loss = loss_dict['total']
            if skip_local:
                loss = loss * 0.0
        
        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        use_scaler = scaler is not None and amp_enabled and scaler.is_enabled()
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # 更新教师模型（EMA）
        self._update_teacher()

        if skip_local:
            for key, value in list(loss_dict.items()):
                if isinstance(value, torch.Tensor):
                    loss_dict[key] = value.detach() * 0.0
                else:
                    loss_dict[key] = 0.0
            loss_dict['skipped'] = True
        loss_dict['did_step'] = True

        return {k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in loss_dict.items()}
    
    @torch.no_grad()
    def _update_teacher(self):
        """使用EMA更新教师模型"""
        student_params = dict(self.student.named_parameters())
        teacher_params = dict(self.teacher.named_parameters())
        for name, teacher_param in teacher_params.items():
            student_param = student_params.get(name, None)
            if student_param is None:
                student_param = student_params.get(f"module.{name}", None)
            if student_param is None:
                continue
            teacher_param.data.mul_(self.teacher_momentum)
            teacher_param.data.add_(student_param.data, alpha=1 - self.teacher_momentum)

        # 同步buffers（如BatchNorm的running_mean/var）
        student_buffers = dict(self.student.named_buffers())
        for name, teacher_buffer in self.teacher.named_buffers():
            student_buffer = student_buffers.get(name, None)
            if student_buffer is None:
                student_buffer = student_buffers.get(f"module.{name}", None)
            if student_buffer is not None:
                teacher_buffer.data.copy_(student_buffer.data)


# =============================================================================
# 4. 时序一致性正则化 (Temporal Consistency Regularization)
# =============================================================================

class TemporalConsistencyLoss(nn.Module):
    """
    时序一致性正则化
    
    基于TAPIR和LocoTrack的设计：
    - 鼓励轨迹在时间上平滑
    - 惩罚突变（除非被遮挡）
    - 双向一致性
    
    L_temporal = ||x_t - 2*x_{t-1} + x_{t-2}||  (二阶平滑)
    """
    
    def __init__(
        self,
        order: int = 2,
        weight: float = 0.01,
        use_visibility_mask: bool = True,
    ):
        """
        Args:
            order: 平滑阶数 (1=一阶差分, 2=二阶差分)
            weight: 损失权重
            use_visibility_mask: 是否使用可见性掩码
        """
        super().__init__()
        self.order = order
        self.weight = weight
        self.use_visibility_mask = use_visibility_mask
    
    def forward(
        self,
        tracks: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tracks: (B, N, T, 2) 轨迹
            visibility: (B, N, T) 可见性
            
        Returns:
            时序一致性损失
        """
        B, N, T, D = tracks.shape
        
        if self.order == 1:
            # 一阶差分：||x_t - x_{t-1}||
            diff = tracks[:, :, 1:] - tracks[:, :, :-1]  # (B, N, T-1, 2)
            smoothness = torch.norm(diff, dim=-1)  # (B, N, T-1)
            
        elif self.order == 2:
            # 二阶差分：||x_t - 2*x_{t-1} + x_{t-2}||（加速度）
            if T < 3:
                return torch.tensor(0.0, device=tracks.device)
            
            diff = tracks[:, :, 2:] - 2 * tracks[:, :, 1:-1] + tracks[:, :, :-2]
            smoothness = torch.norm(diff, dim=-1)  # (B, N, T-2)
        else:
            raise ValueError(f"Unsupported order: {self.order}")
        
        # 可见性掩码
        if self.use_visibility_mask and visibility is not None:
            if self.order == 1:
                # 两帧都可见才计算
                valid = visibility[:, :, 1:] * visibility[:, :, :-1]
            else:
                # 三帧都可见才计算
                valid = visibility[:, :, 2:] * visibility[:, :, 1:-1] * visibility[:, :, :-2]
            
            if valid.sum() > 0:
                loss = (smoothness * valid).sum() / valid.sum()
            else:
                # 无可见帧时不施加平滑约束
                loss = torch.tensor(0.0, device=tracks.device)
        else:
            loss = smoothness.mean()
        
        return self.weight * loss


# =============================================================================
# 5. 合成到真实域适应 (Synthetic-to-Real Domain Adaptation)
# =============================================================================

class DomainAdaptationLoss(nn.Module):
    """
    合成到真实域适应损失
    
    基于BootsTAP的设计：
    - 特征对齐损失
    - 对抗性域判别
    - 一致性正则化
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        domain_weight: float = 0.1,
        use_gradient_reversal: bool = True,
    ):
        """
        Args:
            feature_dim: 特征维度
            domain_weight: 域适应损失权重
            use_gradient_reversal: 是否使用梯度反转
        """
        super().__init__()
        
        self.domain_weight = domain_weight
        self.use_gradient_reversal = use_gradient_reversal
        
        # 域判别器
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            source_features: 源域（合成）特征 (B, *, D)
            target_features: 目标域（真实）特征 (B, *, D)
            
        Returns:
            损失字典
        """
        # 展平特征
        source_flat = source_features.reshape(-1, source_features.shape[-1])
        target_flat = target_features.reshape(-1, target_features.shape[-1])
        
        # 域判别
        if self.use_gradient_reversal:
            source_flat = GradientReversalFunction.apply(source_flat)
            target_flat = GradientReversalFunction.apply(target_flat)
        
        source_pred = self.domain_discriminator(source_flat)
        target_pred = self.domain_discriminator(target_flat)
        
        # 域判别损失
        source_labels = torch.zeros_like(source_pred)  # 合成 = 0
        target_labels = torch.ones_like(target_pred)   # 真实 = 1
        
        domain_loss = F.binary_cross_entropy(source_pred, source_labels) + \
                      F.binary_cross_entropy(target_pred, target_labels)
        
        # 特征对齐损失（MMD）
        mmd_loss = self._compute_mmd(source_flat, target_flat)
        
        total_loss = self.domain_weight * (domain_loss + mmd_loss)
        
        return {
            'domain_total': total_loss,
            'domain_disc': domain_loss,
            'mmd': mmd_loss,
        }
    
    def _compute_mmd(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        kernel: str = 'rbf',
    ) -> torch.Tensor:
        """计算Maximum Mean Discrepancy"""
        n_s, n_t = source.shape[0], target.shape[0]
        
        if kernel == 'rbf':
            # RBF核
            gamma = 1.0 / source.shape[-1]
            
            ss = torch.exp(-gamma * torch.cdist(source, source) ** 2)
            tt = torch.exp(-gamma * torch.cdist(target, target) ** 2)
            st = torch.exp(-gamma * torch.cdist(source, target) ** 2)
            
            mmd = ss.mean() + tt.mean() - 2 * st.mean()
        else:
            # 线性核
            mean_s = source.mean(dim=0)
            mean_t = target.mean(dim=0)
            mmd = torch.norm(mean_s - mean_t)
        
        return mmd


class GradientReversalFunction(torch.autograd.Function):
    """梯度反转层"""
    
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


# =============================================================================
# 6. 多迭代监督 (Multi-Iteration Supervision)
# =============================================================================

class MultiIterationLoss(nn.Module):
    """
    多迭代监督损失
    
    基于CoTracker的设计：
    - 对每次迭代更新都计算损失
    - 后期迭代权重更大
    - 鼓励渐进式改进
    """
    
    def __init__(
        self,
        num_iters: int = 4,
        gamma: float = 0.8,
        loss_fn: Optional[nn.Module] = None,
    ):
        """
        Args:
            num_iters: 迭代次数
            gamma: 权重衰减因子（越小早期权重越低）
            loss_fn: 基础损失函数
        """
        super().__init__()
        
        self.num_iters = num_iters
        self.gamma = gamma
        self.loss_fn = loss_fn or nn.SmoothL1Loss(reduction='none')
        
        # 预计算权重
        weights = [gamma ** (num_iters - 1 - i) for i in range(num_iters)]
        weight_sum = sum(weights)
        self.weights = [w / weight_sum for w in weights]
        
        logger.info(f"MultiIterationLoss: weights={[f'{w:.3f}' for w in self.weights]}")
    
    def forward(
        self,
        all_predictions: List[torch.Tensor],
        targets: torch.Tensor,
        visibility: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            all_predictions: 每次迭代的预测列表
            targets: 目标轨迹
            visibility: 可见性掩码
            
        Returns:
            损失字典
        """
        total_loss = 0.0
        iter_losses = []
        
        for i, pred in enumerate(all_predictions):
            loss = self.loss_fn(pred, targets)
            
            if visibility is not None:
                mask = visibility.unsqueeze(-1).expand_as(loss).float()
                if mask.sum() > 0:
                    loss = (loss * mask).sum() / mask.sum()
                else:
                    loss = torch.tensor(0.0, device=targets.device)
            else:
                loss = loss.mean()
            
            weighted_loss = self.weights[i] * loss
            total_loss = total_loss + weighted_loss
            iter_losses.append(loss.item())
        
        return {
            'total': total_loss,
            'iter_losses': iter_losses,
            'final_iter_loss': iter_losses[-1] if iter_losses else 0.0,
        }


# =============================================================================
# 工厂函数
# =============================================================================

def create_advanced_training_components(config) -> Dict[str, Any]:
    """
    根据配置创建高级训练组件
    
    Returns:
        包含各种训练组件的字典
    """
    components = {}
    
    # 渐进式训练
    progressive_config = getattr(config.training, 'progressive', None)
    if progressive_config is not None and getattr(progressive_config, 'enabled', False):
        stages = create_progressive_stages(config)
        components['progressive_scheduler'] = ProgressiveTrainingScheduler(stages)
    
    # 时序一致性
    temporal_config = getattr(config.loss, 'temporal_consistency', None)
    if temporal_config is not None and getattr(temporal_config, 'enabled', True):
        components['temporal_consistency_loss'] = TemporalConsistencyLoss(
            order=getattr(temporal_config, 'order', 2),
            weight=getattr(temporal_config, 'weight', 0.01),
        )
    
    # 多迭代监督
    multi_iter_cfg = getattr(config.loss, 'multi_iteration', None)
    num_iters = getattr(config.model.temporal, 'num_refinement_iters', 1)
    if num_iters > 1 and multi_iter_cfg is not None and getattr(multi_iter_cfg, 'enabled', False):
        components['multi_iter_loss'] = MultiIterationLoss(
            num_iters=num_iters,
            gamma=getattr(multi_iter_cfg, 'gamma', 0.8),
        )
    
    return components
