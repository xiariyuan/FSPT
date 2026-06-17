"""
早停机制 (Early Stopping)

当验证指标不再改善时提前停止训练，防止过拟合
"""

import logging
from pathlib import Path
from typing import Optional, Union, Callable
import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    早停机制
    
    监控验证指标，当指标在指定轮数内不再改善时停止训练
    
    用法:
        early_stopping = EarlyStopping(patience=10, mode='max')
        
        for epoch in range(num_epochs):
            train(...)
            val_metric = validate(...)
            
            if early_stopping(val_metric, model, epoch):
                print("Early stopping triggered!")
                break
        
        # 加载最佳模型
        early_stopping.load_best_model(model)
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max',
        save_path: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ):
        """
        Args:
            patience: 等待改善的epoch数
            min_delta: 最小改善阈值
            mode: 'max'表示指标越大越好，'min'表示越小越好
            save_path: 最佳模型保存路径
            verbose: 是否输出日志
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = Path(save_path) if save_path else None
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.best_epoch = -1
        self.early_stop = False
        self.best_model_state = None
        
        # 根据模式设置比较函数
        if mode == 'max':
            self.is_better = lambda current, best: current > best + min_delta
            self.best_score = float('-inf')
        elif mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
            self.best_score = float('inf')
        else:
            raise ValueError(f"mode must be 'max' or 'min', got {mode}")
        
        logger.info(f"EarlyStopping initialized: patience={patience}, mode={mode}")
    
    def __call__(
        self,
        score: float,
        model: Optional[torch.nn.Module] = None,
        epoch: int = -1,
    ) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前验证指标
            model: 模型（用于保存最佳状态）
            epoch: 当前epoch
            
        Returns:
            是否应该停止训练
        """
        if self.is_better(score, self.best_score):
            # 指标改善
            if self.verbose:
                logger.info(
                    f"EarlyStopping: score improved "
                    f"{self.best_score:.6f} -> {score:.6f}"
                )
            
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            
            # 保存最佳模型
            if model is not None:
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                if self.save_path is not None:
                    try:
                        self.save_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save({
                            'model_state_dict': self.best_model_state,
                            'score': score,
                            'epoch': epoch,
                        }, self.save_path)
                        if self.verbose:
                            logger.info(f"Best model saved to {self.save_path}")
                    except Exception as exc:
                        logger.warning(f"Failed to save best model to {self.save_path}: {exc}")
        else:
            # 指标未改善
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"EarlyStopping: no improvement for {self.counter}/{self.patience} epochs "
                    f"(best: {self.best_score:.6f} at epoch {self.best_epoch})"
                )
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(
                        f"EarlyStopping triggered! "
                        f"Best score: {self.best_score:.6f} at epoch {self.best_epoch}"
                    )
        
        return self.early_stop
    
    def load_best_model(self, model: torch.nn.Module):
        """
        加载最佳模型权重
        
        Args:
            model: 目标模型
        """
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model from epoch {self.best_epoch}")
        elif self.save_path is not None and self.save_path.exists():
            checkpoint = torch.load(self.save_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {self.save_path}")
        else:
            logger.warning("No best model found to load")
    
    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.early_stop = False
        if self.mode == 'max':
            self.best_score = float('-inf')
        else:
            self.best_score = float('inf')
        self.best_epoch = -1
        self.best_model_state = None
    
    def state_dict(self) -> dict:
        """获取状态字典"""
        state = {
            'counter': self.counter,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'early_stop': self.early_stop,
            'patience': self.patience,
            'mode': self.mode,
        }
        if self.best_model_state is not None:
            state['best_model_state'] = self.best_model_state
        return state
    
    def load_state_dict(self, state_dict: dict):
        """加载状态字典"""
        required_keys = ['counter', 'best_score', 'best_epoch', 'early_stop']
        missing_keys = [k for k in required_keys if k not in state_dict]
        if missing_keys:
            logger.warning(f"EarlyStopping state missing keys: {missing_keys}")
        self.counter = state_dict.get('counter', 0)
        self.best_score = state_dict.get('best_score', self.best_score)
        self.best_epoch = state_dict.get('best_epoch', -1)
        self.early_stop = state_dict.get('early_stop', False)
        if 'best_model_state' in state_dict:
            self.best_model_state = state_dict.get('best_model_state', self.best_model_state)


class ReduceLROnPlateau:
    """
    基于验证指标的学习率调整
    
    当验证指标不再改善时降低学习率
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = 'max',
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-7,
        verbose: bool = True,
    ):
        """
        Args:
            optimizer: 优化器
            mode: 'max'或'min'
            factor: 学习率缩放因子
            patience: 等待改善的epoch数
            min_lr: 最小学习率
            verbose: 是否输出日志
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.counter = 0
        if mode == 'max':
            self.best_score = float('-inf')
            self.is_better = lambda current, best: current > best
        else:
            self.best_score = float('inf')
            self.is_better = lambda current, best: current < best
    
    def step(self, score: float):
        """
        根据指标调整学习率
        
        Args:
            score: 验证指标
        """
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                self._reduce_lr()
                self.counter = 0
    
    def _reduce_lr(self):
        """降低学习率"""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            
            if self.verbose and new_lr < old_lr:
                logger.info(f"ReduceLROnPlateau: lr {old_lr:.2e} -> {new_lr:.2e}")


def create_early_stopping(config) -> Optional[EarlyStopping]:
    """
    根据配置创建早停
    
    Args:
        config: 配置
        
    Returns:
        EarlyStopping实例或None
    """
    training_config = getattr(config, 'training', None)
    if training_config is None:
        return None
    
    early_stop_config = getattr(training_config, 'early_stopping', None)
    if early_stop_config is None or not getattr(early_stop_config, 'enabled', False):
        return None
    
    patience = getattr(early_stop_config, 'patience', 10)
    min_delta = getattr(early_stop_config, 'min_delta', 0.0)
    mode = getattr(early_stop_config, 'mode', 'max')
    
    # 保存路径
    paths_config = getattr(config, 'paths', None)
    save_path = None
    if paths_config is not None:
        checkpoint_dir = getattr(paths_config, 'checkpoint_dir', 'outputs/checkpoints')
        experiment_name = getattr(config.experiment, 'name', 'fspt')
        save_path = Path(checkpoint_dir) / experiment_name / 'early_stop_best.pth'
    
    return EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        mode=mode,
        save_path=save_path,
    )
