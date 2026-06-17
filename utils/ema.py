"""
模型指数移动平均 (Exponential Moving Average)

EMA可以平滑训练过程中的权重更新，提高模型稳定性和泛化能力
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Optional, Dict, Any
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class ModelEMA:
    """
    模型指数移动平均
    
    维护模型参数的滑动平均，用于评估和推理
    
    用法:
        model = MyModel()
        ema = ModelEMA(model, decay=0.999)
        
        for batch in dataloader:
            # 正常训练
            loss = model(batch)
            loss.backward()
            optimizer.step()
            
            # 更新EMA
            ema.update(model)
        
        # 使用EMA模型评估
        ema_model = ema.ema_model
        ema_model.eval()
        output = ema_model(test_input)
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 2000,
        update_after_step: int = 100,
        use_warmup: bool = True,
    ):
        """
        Args:
            model: 原始模型
            decay: EMA衰减率（0.9999表示强平滑）
            warmup_steps: 预热步数（逐渐增加decay）
            update_after_step: 开始更新的步数
            use_warmup: 是否使用预热
        """
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.update_after_step = update_after_step
        self.use_warmup = use_warmup
        self.step = 0
        
        # 创建EMA模型（深拷贝）
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)
        
        # 记录参数数量
        num_params = sum(p.numel() for p in self.ema_model.parameters())
        logger.info(f"ModelEMA initialized with decay={decay}, params={num_params:,}")
    
    def _get_decay(self) -> float:
        """获取当前衰减率（带预热）"""
        if not self.use_warmup or self.step >= self.warmup_steps:
            return self.decay
        
        # 线性预热
        return min(self.decay, (1 + self.step) / (10 + self.step))
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        更新EMA权重
        
        Args:
            model: 当前训练模型
        """
        self.step += 1
        
        # 跳过早期步骤
        if self.step < self.update_after_step:
            return
        
        decay = self._get_decay()
        
        # 更新EMA参数
        model_params = dict(model.named_parameters())
        ema_params = dict(self.ema_model.named_parameters())
        
        for name, ema_param in ema_params.items():
            model_param = model_params.get(name)
            if model_param is None:
                if name.startswith('module.'):
                    model_param = model_params.get(name[7:])
                else:
                    model_param = model_params.get(f'module.{name}')
            if model_param is not None:
                model_data = model_param.data
                if model_data.device != ema_param.device or model_data.dtype != ema_param.dtype:
                    model_data = model_data.to(device=ema_param.device, dtype=ema_param.dtype)
                ema_param.data.mul_(decay).add_(model_data, alpha=1 - decay)
            else:
                logger.debug(f"EMA param missing in model: {name}")
        
        # 同步buffers（如BatchNorm的running_mean/var）
        model_buffers = dict(model.named_buffers())
        ema_buffers = dict(self.ema_model.named_buffers())
        
        for name, ema_buffer in ema_buffers.items():
            model_buffer = model_buffers.get(name)
            if model_buffer is None:
                if name.startswith('module.'):
                    model_buffer = model_buffers.get(name[7:])
                else:
                    model_buffer = model_buffers.get(f'module.{name}')
            if model_buffer is not None:
                buffer_data = model_buffer.data
                if buffer_data.device != ema_buffer.device or buffer_data.dtype != ema_buffer.dtype:
                    buffer_data = buffer_data.to(device=ema_buffer.device, dtype=ema_buffer.dtype)
                ema_buffer.data.copy_(buffer_data)
            else:
                logger.debug(f"EMA buffer missing in model: {name}")
    
    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        """
        将EMA权重复制到模型
        
        Args:
            model: 目标模型
        """
        ema_params = dict(self.ema_model.named_parameters())
        model_params = dict(model.named_parameters())
        
        for name, model_param in model_params.items():
            ema_param = ema_params.get(name)
            if ema_param is None:
                if name.startswith('module.'):
                    ema_param = ema_params.get(name[7:])
                else:
                    ema_param = ema_params.get(f'module.{name}')
            if ema_param is not None:
                ema_data = ema_param.data
                if ema_data.device != model_param.device or ema_data.dtype != model_param.dtype:
                    ema_data = ema_data.to(device=model_param.device, dtype=model_param.dtype)
                model_param.data.copy_(ema_data)
            else:
                logger.debug(f"EMA param missing for copy: {name}")
        
        # 同步buffers
        ema_buffers = dict(self.ema_model.named_buffers())
        model_buffers = dict(model.named_buffers())
        for name, model_buffer in model_buffers.items():
            ema_buffer = ema_buffers.get(name)
            if ema_buffer is None:
                if name.startswith('module.'):
                    ema_buffer = ema_buffers.get(name[7:])
                else:
                    ema_buffer = ema_buffers.get(f'module.{name}')
            if ema_buffer is not None:
                buffer_data = ema_buffer.data
                if buffer_data.device != model_buffer.device or buffer_data.dtype != model_buffer.dtype:
                    buffer_data = buffer_data.to(device=model_buffer.device, dtype=model_buffer.dtype)
                model_buffer.data.copy_(buffer_data)
            else:
                logger.debug(f"EMA buffer missing for copy: {name}")
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典（用于保存）"""
        return {
            'ema_model': self.ema_model.state_dict(),
            'decay': self.decay,
            'step': self.step,
            'warmup_steps': self.warmup_steps,
            'update_after_step': self.update_after_step,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态字典"""
        ema_model_state = state_dict.get('ema_model')
        if ema_model_state is not None:
            self.ema_model.load_state_dict(ema_model_state)
        self.decay = state_dict.get('decay', self.decay)
        self.step = state_dict.get('step', 0)
        self.warmup_steps = state_dict.get('warmup_steps', self.warmup_steps)
        self.update_after_step = state_dict.get('update_after_step', self.update_after_step)
        should_log = True
        if dist.is_available() and dist.is_initialized():
            should_log = dist.get_rank() == 0
        if should_log:
            if ema_model_state is None:
                logger.warning("EMA state missing 'ema_model', skipping weight load.")
            logger.info(f"Loaded EMA state at step {self.step}")
    
    def to(self, device: torch.device):
        """移动EMA模型到指定设备"""
        self.ema_model = self.ema_model.to(device)
        return self


class AveragedModel(nn.Module):
    """
    PyTorch风格的平均模型包装器
    
    可以使用不同的平均策略
    """
    
    def __init__(
        self,
        model: nn.Module,
        avg_fn: Optional[callable] = None,
        use_buffers: bool = True,
    ):
        """
        Args:
            model: 原始模型
            avg_fn: 自定义平均函数 fn(avg_param, model_param, num_averaged)
            use_buffers: 是否平均buffers
        """
        super().__init__()
        
        self.module = deepcopy(model)
        self.module.requires_grad_(False)
        
        if avg_fn is None:
            # 默认使用简单平均
            def avg_fn(avg, model, num):
                return avg + (model - avg) / (num + 1)
        
        self.avg_fn = avg_fn
        self.use_buffers = use_buffers
        self.n_averaged = 0
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    @torch.no_grad()
    def update_parameters(self, model: nn.Module):
        """更新平均参数"""
        self_params = dict(self.module.named_parameters())
        model_params = dict(model.named_parameters())
        
        for name, self_param in self_params.items():
            model_param = model_params.get(name)
            if model_param is None:
                if name.startswith('module.'):
                    model_param = model_params.get(name[7:])
                else:
                    model_param = model_params.get(f'module.{name}')
            if model_param is not None:
                self_param.data = self.avg_fn(
                    self_param.data, model_param.data, self.n_averaged
                )
        
        if self.use_buffers:
            self_buffers = dict(self.module.named_buffers())
            model_buffers = dict(model.named_buffers())
            
            for name, self_buffer in self_buffers.items():
                model_buffer = model_buffers.get(name)
                if model_buffer is None:
                    if name.startswith('module.'):
                        model_buffer = model_buffers.get(name[7:])
                    else:
                        model_buffer = model_buffers.get(f'module.{name}')
                if model_buffer is not None:
                    self_buffer.data.copy_(model_buffer.data)
        
        self.n_averaged += 1


def create_ema(
    model: nn.Module,
    config,
    device: torch.device,
) -> Optional[ModelEMA]:
    """
    根据配置创建EMA
    
    Args:
        model: 原始模型
        config: 配置
        device: 设备
        
    Returns:
        ModelEMA实例或None
    """
    # 检查配置中是否启用EMA
    training_config = getattr(config, 'training', None)
    if training_config is None:
        return None
    
    ema_config = getattr(training_config, 'ema', None)
    if ema_config is None or not getattr(ema_config, 'enabled', False):
        return None
    
    decay = getattr(ema_config, 'decay', 0.9999)
    warmup_steps = getattr(ema_config, 'warmup_steps', 2000)
    update_after_step = getattr(ema_config, 'update_after_step', 100)
    
    ema = ModelEMA(
        model=model,
        decay=decay,
        warmup_steps=warmup_steps,
        update_after_step=update_after_step,
    )
    ema.to(device)
    
    return ema
