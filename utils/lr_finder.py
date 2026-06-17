"""
学习率查找器 (Learning Rate Finder)

自动寻找最佳学习率范围，基于Leslie Smith的论文
"""

import math
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Union
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LRFinder:
    """
    学习率查找器
    
    通过指数增长学习率并记录损失，找到最佳学习率范围
    
    用法:
        lr_finder = LRFinder(model, optimizer, criterion, device)
        lr_finder.range_test(train_loader, start_lr=1e-7, end_lr=10, num_iter=100)
        lr_finder.plot()  # 绘制学习率-损失曲线
        suggested_lr = lr_finder.suggest_lr()
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ):
        """
        Args:
            model: 模型
            optimizer: 优化器
            criterion: 损失函数
            device: 设备
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
        # 保存初始状态
        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())
        
        # 记录
        self.history = {
            'lr': [],
            'loss': [],
        }
        self.best_loss = float('inf')
        self._accum_step = 0
    
    def range_test(
        self,
        train_loader: DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 10,
        num_iter: int = 100,
        smooth_f: float = 0.05,
        diverge_th: float = 5.0,
        accumulation_steps: int = 1,
        use_amp: bool = False,
        amp_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[List[float], List[float]]:
        """
        执行学习率范围测试
        
        Args:
            train_loader: 训练数据加载器
            start_lr: 起始学习率
            end_lr: 结束学习率
            num_iter: 迭代次数
            smooth_f: 损失平滑因子
            diverge_th: 发散阈值（相对于最佳损失的倍数）
            accumulation_steps: 梯度累积步数
            use_amp: 是否使用混合精度
            
        Returns:
            (学习率列表, 损失列表)
        """
        if train_loader is None:
            raise ValueError("train_loader is None")
        try:
            if len(train_loader) == 0:
                raise ValueError("train_loader is empty")
        except TypeError:
            pass
        if num_iter <= 0:
            raise ValueError("num_iter must be positive")
        if start_lr <= 0 or end_lr <= 0:
            raise ValueError("start_lr and end_lr must be positive")
        if end_lr <= start_lr:
            logger.warning("end_lr <= start_lr; learning rate will not increase")
        # 重置
        self.history = {'lr': [], 'loss': []}
        self.best_loss = float('inf')
        
        # 恢复初始状态
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        
        # 设置初始学习率
        self._set_lr(start_lr)
        
        # 计算学习率增长因子（按优化器step数）
        accumulation_steps = max(1, int(accumulation_steps))
        lr_updates = max(1, math.ceil(num_iter / accumulation_steps))
        lr_mult = (end_lr / start_lr) ** (1 / lr_updates)
        
        # 混合精度（仅CUDA）
        if use_amp and self.device.type != 'cuda':
            logger.warning("AMP requested on non-CUDA device; disabling AMP.")
        use_amp = bool(use_amp and self.device.type == 'cuda' and torch.cuda.is_available())
        if use_amp and amp_dtype is None:
            amp_dtype = torch.float16
        scaler_enabled = use_amp and (amp_dtype is None or amp_dtype == torch.float16)
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        
        self.model.train()
        smoothed_loss = 0.0
        self._accum_step = 0
        
        data_iter = iter(train_loader)
        pbar = tqdm(range(num_iter), desc="LR Range Test")
        
        for iteration in pbar:
            # 获取数据
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            # 计算损失
            loss, did_step = self._train_batch(
                batch, scaler, accumulation_steps, use_amp, amp_dtype
            )
            
            if loss is None:
                logger.warning("Skipping invalid batch during LR range test.")
                continue
            if not math.isfinite(loss):
                logger.warning(f"Stopping: loss became {loss}")
                break
            
            # 平滑损失
            if iteration == 0:
                smoothed_loss = loss
            else:
                smoothed_loss = smooth_f * loss + (1 - smooth_f) * smoothed_loss
            
            # 记录
            current_lr = self._get_lr()
            self.history['lr'].append(current_lr)
            self.history['loss'].append(smoothed_loss)
            
            # 更新最佳损失
            if smoothed_loss < self.best_loss:
                self.best_loss = smoothed_loss
            
            # 检查发散
            if smoothed_loss > diverge_th * self.best_loss:
                logger.info(f"Stopping: loss diverged (threshold: {diverge_th}x best)")
                break
            
            # 更新学习率
            if did_step:
                self._set_lr(current_lr * lr_mult)
            pbar.set_postfix({'lr': f'{current_lr:.2e}', 'loss': f'{smoothed_loss:.4f}'})
        
        # 恢复初始状态
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        
        logger.info(f"LR Range Test completed: {len(self.history['lr'])} iterations")
        
        return self.history['lr'], self.history['loss']
    
    def _train_batch(
        self,
        batch: dict,
        scaler: torch.cuda.amp.GradScaler,
        accumulation_steps: int,
        use_amp: bool,
        amp_dtype: Optional[torch.dtype],
    ) -> Tuple[Optional[float], bool]:
        """训练单个batch"""
        accumulation_steps = max(1, int(accumulation_steps))
        if self._accum_step == 0:
            self.optimizer.zero_grad(set_to_none=True)
        
        if not isinstance(batch, dict):
            logger.warning("Batch must be a dict with required keys; skipping.")
            return None, False
        required_keys = ('video', 'query_points', 'target_points', 'occluded')
        missing_keys = [k for k in required_keys if k not in batch]
        if missing_keys:
            logger.warning(f"Missing keys in batch: {missing_keys}; skipping.")
            return None, False

        # 移动数据到设备
        video = batch['video'].to(self.device)
        query_points = batch['query_points'].to(self.device)
        target_points = batch['target_points'].to(self.device)
        occluded = batch['occluded'].to(self.device)
        if occluded.dtype != torch.bool:
            occluded = occluded > 0.5
        
        try:
            with torch.cuda.amp.autocast(
                enabled=use_amp and self.device.type == 'cuda',
                dtype=amp_dtype,
            ):
                outputs = self.model(video, query_points)
                
                if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
                    pred_tracks, pred_visibility = outputs[0], outputs[1]
                else:
                    raise ValueError("Unexpected model output format")
                
                losses = self.criterion(
                    pred_tracks, target_points,
                    pred_visibility, ~occluded,
                )
                if isinstance(losses, dict):
                    loss = losses.get('total', None)
                    if loss is None:
                        loss = sum(
                            v for v in losses.values() if isinstance(v, torch.Tensor)
                        )
                elif isinstance(losses, torch.Tensor):
                    loss = losses
                else:
                    raise ValueError("Unexpected loss output format")
            
            loss_value = float(loss.detach().item())
            scaled_loss = loss / accumulation_steps

            # 反向传播
            use_scaler = use_amp and scaler.is_enabled()
            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            self._accum_step += 1
            did_step = False

            if self._accum_step >= accumulation_steps:
                if use_scaler:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()
                self._accum_step = 0
                did_step = True
            
            return loss_value, did_step
            
        except Exception as e:
            logger.warning(f"Error in batch: {e}")
            return None, False
    
    def _get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def _set_lr(self, lr: float):
        """设置学习率"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def suggest_lr(
        self,
        skip_start: int = 10,
        skip_end: int = 5,
        steepest: bool = True,
    ) -> float:
        """
        建议最佳学习率
        
        Args:
            skip_start: 跳过开始的几个点
            skip_end: 跳过结束的几个点
            steepest: 是否选择梯度最陡处（否则选择最小损失前的点）
            
        Returns:
            建议的学习率
        """
        if len(self.history['lr']) < skip_start + skip_end + 1:
            logger.warning("Not enough data points to suggest LR")
            return 1e-4
        
        lrs = self.history['lr'][skip_start:-skip_end]
        losses = self.history['loss'][skip_start:-skip_end]
        
        if steepest:
            # 找到梯度最陡的点
            gradients = []
            for i in range(1, len(losses)):
                lr_diff = math.log10(lrs[i]) - math.log10(lrs[i-1])
                if abs(lr_diff) < 1e-12:
                    continue
                grad = (losses[i] - losses[i-1]) / lr_diff
                gradients.append((i, grad))
            if not gradients:
                logger.warning("Unable to compute LR gradients; falling back to min-loss heuristic.")
                min_loss_idx = losses.index(min(losses))
                suggested = lrs[max(0, min_loss_idx - 1)]
            else:
                min_grad_idx = min(gradients, key=lambda x: x[1])[0]
                suggested = lrs[min_grad_idx]
        else:
            # 找到最小损失前的点
            min_loss_idx = losses.index(min(losses))
            suggested = lrs[max(0, min_loss_idx - 1)]
        
        # 建议使用建议值的1/10作为实际学习率
        suggested /= 10
        
        logger.info(f"Suggested learning rate: {suggested:.2e}")
        return suggested
    
    def plot(
        self,
        save_path: Optional[Union[str, Path]] = None,
        skip_start: int = 10,
        skip_end: int = 5,
        log_scale: bool = True,
        show: bool = True,
    ):
        """
        绘制学习率-损失曲线
        
        Args:
            save_path: 保存路径
            skip_start: 跳过开始的几个点
            skip_end: 跳过结束的几个点
            log_scale: 是否使用对数坐标
            show: 是否显示图像
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return
        
        if len(self.history['lr']) < skip_start + skip_end + 1:
            logger.warning("Not enough data points to plot")
            return
        
        lrs = self.history['lr'][skip_start:-skip_end] if skip_end > 0 else self.history['lr'][skip_start:]
        losses = self.history['loss'][skip_start:-skip_end] if skip_end > 0 else self.history['loss'][skip_start:]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(lrs, losses, 'b-', linewidth=2)
        
        if log_scale:
            ax.set_xscale('log')
        
        ax.set_xlabel('Learning Rate', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Learning Rate Range Test', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 标记建议的学习率
        suggested = self.suggest_lr(skip_start, skip_end)
        ax.axvline(x=suggested, color='r', linestyle='--', label=f'Suggested: {suggested:.2e}')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            logger.info(f"LR finder plot saved to {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def reset(self):
        """重置到初始状态"""
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.history = {'lr': [], 'loss': []}
        self.best_loss = float('inf')


def find_optimal_lr(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer_cls: type = torch.optim.AdamW,
    start_lr: float = 1e-7,
    end_lr: float = 1.0,
    num_iter: int = 100,
    weight_decay: float = 1e-4,
    plot_path: Optional[Union[str, Path]] = None,
    accumulation_steps: int = 1,
    use_amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    optimizer_kwargs: Optional[dict] = None,
) -> float:
    """
    一键寻找最佳学习率
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        device: 设备
        optimizer_cls: 优化器类
        start_lr: 起始学习率
        end_lr: 结束学习率
        num_iter: 迭代次数
        weight_decay: 权重衰减
        plot_path: 图像保存路径
        accumulation_steps: 梯度累积步数
        optimizer_kwargs: 额外优化器参数（如 betas）
        
    Returns:
        建议的学习率
    """
    # 创建临时优化器
    optimizer_args = dict(optimizer_kwargs or {})
    optimizer_args.setdefault('weight_decay', weight_decay)
    optimizer = optimizer_cls(
        model.parameters(),
        lr=start_lr,
        **optimizer_args,
    )
    
    # 创建LR finder
    lr_finder = LRFinder(model, optimizer, criterion, device)
    
    # 执行范围测试
    lr_finder.range_test(
        train_loader,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter,
        accumulation_steps=accumulation_steps,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    
    # 绘图
    if plot_path:
        lr_finder.plot(save_path=plot_path, show=False)
    
    # 获取建议值
    suggested_lr = lr_finder.suggest_lr()
    
    # 重置模型
    lr_finder.reset()
    
    return suggested_lr
