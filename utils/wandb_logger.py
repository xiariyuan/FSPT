"""
WandB 实验追踪集成

提供统一的实验日志记录接口，支持WandB和TensorBoard回退
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from omegaconf import OmegaConf, DictConfig
import torch

logger = logging.getLogger(__name__)

# 尝试导入wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.info("WandB not available. Install with: pip install wandb")


class ExperimentLogger:
    """
    统一的实验日志记录器
    
    支持WandB和TensorBoard，自动回退
    
    用法:
        logger = ExperimentLogger(config, output_dir)
        logger.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100)
        logger.log_image("predictions", image_tensor)
        logger.finish()
    """
    
    def __init__(
        self,
        config: DictConfig,
        output_dir: Union[str, Path],
        project_name: str = "FSPT",
        run_name: Optional[str] = None,
        use_wandb: bool = True,
        use_tensorboard: bool = True,
        entity: Optional[str] = None,
        watch_log_freq: int = 100,
        resume: Optional[str] = None,
    ):
        """
        Args:
            config: 实验配置
            output_dir: 输出目录
            project_name: WandB项目名
            run_name: 运行名称（默认使用config.experiment.name）
            use_wandb: 是否使用WandB
            use_tensorboard: 是否使用TensorBoard
            resume: WandB run ID用于恢复
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.wandb_run = None
        self.tb_writer = None
        self._watched = False
        self._watch_log_freq = watch_log_freq
        self._entity = entity
        
        # 运行名称
        if run_name is None:
            run_name = getattr(config.experiment, 'name', 'fspt_run')
        self.run_name = run_name
        
        # 初始化WandB
        if use_wandb and WANDB_AVAILABLE:
            self._init_wandb(config, project_name, run_name, resume, entity)
        
        # 初始化TensorBoard
        if use_tensorboard:
            self._init_tensorboard()
    
    def _init_wandb(
        self,
        config: DictConfig,
        project_name: str,
        run_name: str,
        resume: Optional[str] = None,
        entity: Optional[str] = None,
    ):
        """初始化WandB"""
        try:
            # 转换config为字典
            config_dict = OmegaConf.to_container(config, resolve=True)
            
            # 初始化wandb
            resume_mode = "allow" if resume else None
            resume_id = resume if isinstance(resume, str) and resume else None
            self.wandb_run = wandb.init(
                project=project_name,
                name=run_name,
                entity=entity,
                config=config_dict,
                dir=str(self.output_dir),
                resume=resume_mode,
                id=resume_id,
            )
            
            logger.info(f"WandB initialized: {self.wandb_run.url}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            self.wandb_run = None
    
    def _init_tensorboard(self):
        """初始化TensorBoard"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self.output_dir / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(str(tb_dir))
            logger.info(f"TensorBoard initialized: {tb_dir}")
        except ImportError:
            logger.warning("TensorBoard not available")
            self.tb_writer = None
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: str = "",
    ):
        """
        记录指标
        
        Args:
            metrics: 指标字典
            step: 当前步数
            prefix: 指标前缀（如 "train/", "val/"）
        """
        import torch

        # 添加前缀
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        # 规范化标量类型
        try:
            import numpy as np
        except Exception:
            np = None
        import math
        normalized = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    value = value.detach().cpu().item()
                else:
                    continue
            if np is not None:
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        value = float(value)
                    else:
                        continue
                elif isinstance(value, np.generic):
                    value = float(value)
            if isinstance(value, (int, float, bool)):
                if isinstance(value, bool) or math.isfinite(float(value)):
                    normalized[key] = value
        metrics = normalized
        if not metrics:
            return

        # WandB
        if self.wandb_run is not None and metrics:
            wandb.log(metrics, step=step)
        
        # TensorBoard
        if self.tb_writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)
    
    def log_image(
        self,
        tag: str,
        image,
        step: Optional[int] = None,
        caption: Optional[str] = None,
    ):
        """
        记录图像
        
        Args:
            tag: 图像标签
            image: 图像（numpy array或torch tensor）
            step: 当前步数
            caption: 图像说明
        """
        import numpy as np
        import torch
        
        # 转换为numpy
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        
        # WandB
        if self.wandb_run is not None:
            wandb_image = wandb.Image(image, caption=caption)
            wandb.log({tag: wandb_image}, step=step)
        
        # TensorBoard
        if self.tb_writer is not None:
            if image.ndim == 3:
                # (H, W, C) -> (C, H, W)
                if image.shape[-1] in [1, 3, 4]:
                    image = np.transpose(image, (2, 0, 1))
            self.tb_writer.add_image(tag, image, step)
    
    def log_video(
        self,
        tag: str,
        video,
        step: Optional[int] = None,
        fps: int = 10,
    ):
        """
        记录视频
        
        Args:
            tag: 视频标签
            video: 视频 (T, H, W, C) 或 (T, C, H, W)
            step: 当前步数
            fps: 帧率
        """
        import numpy as np
        import torch
        
        # 转换为numpy
        if isinstance(video, torch.Tensor):
            video = video.detach().cpu().numpy()
        
        # WandB
        if self.wandb_run is not None:
            # WandB期望 (T, C, H, W)
            if video.shape[-1] in [1, 3, 4]:
                video = np.transpose(video, (0, 3, 1, 2))
            wandb_video = wandb.Video(video, fps=fps)
            wandb.log({tag: wandb_video}, step=step)
        
        # TensorBoard
        if self.tb_writer is not None:
            # TensorBoard期望 (N, T, C, H, W)
            if video.ndim == 4:
                video = video[np.newaxis, ...]
            if video.shape[-1] in [1, 3, 4]:
                video = np.transpose(video, (0, 1, 4, 2, 3))
            self.tb_writer.add_video(tag, video, step, fps=fps)
    
    def log_histogram(
        self,
        tag: str,
        values,
        step: Optional[int] = None,
    ):
        """记录直方图"""
        import torch
        
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        if self.wandb_run is not None:
            wandb.log({tag: wandb.Histogram(values)}, step=step)
        
        if self.tb_writer is not None:
            self.tb_writer.add_histogram(tag, values, step)
    
    def log_model_gradients(self, model, log_freq: Optional[int] = None):
        """记录模型梯度分布"""
        if self.wandb_run is None or self._watched:
            return
        if log_freq is not None and int(log_freq) <= 0:
            return
        wandb.watch(model, log="gradients", log_freq=log_freq or self._watch_log_freq)
        self._watched = True
    
    def log_artifact(
        self,
        name: str,
        artifact_type: str,
        path: Union[str, Path],
        metadata: Optional[Dict] = None,
    ):
        """
        记录工件（模型、数据集等）
        
        Args:
            name: 工件名称
            artifact_type: 工件类型 ("model", "dataset", etc.)
            path: 工件路径
            metadata: 元数据
        """
        if self.wandb_run is not None:
            artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata)
            
            path = Path(path)
            if path.is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))
            
            self.wandb_run.log_artifact(artifact)
            logger.info(f"Logged artifact: {name}")
    
    def log_table(
        self,
        tag: str,
        columns: list,
        data: list,
        step: Optional[int] = None,
    ):
        """记录表格数据"""
        if self.wandb_run is not None:
            table = wandb.Table(columns=columns, data=data)
            wandb.log({tag: table}, step=step)
    
    def save_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        upload_to_wandb: bool = True,
    ):
        """
        保存并可选上传检查点
        
        Args:
            checkpoint_path: 检查点路径
            upload_to_wandb: 是否上传到WandB
        """
        if upload_to_wandb and self.wandb_run is not None:
            self.log_artifact(
                name=f"{self.run_name}-checkpoint",
                artifact_type="model",
                path=checkpoint_path,
                metadata={"run_id": self.wandb_run.id},
            )
    
    def finish(self):
        """结束日志记录"""
        if self.wandb_run is not None:
            self.wandb_run.finish()
            logger.info("WandB run finished")
        
        if self.tb_writer is not None:
            self.tb_writer.close()
            logger.info("TensorBoard writer closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False


def create_experiment_logger(
    config: DictConfig,
    output_dir: Union[str, Path],
    is_main_process: bool = True,
) -> ExperimentLogger:
    """
    创建实验日志记录器的工厂函数
    
    Args:
        config: 实验配置
        output_dir: 输出目录
        is_main_process: 是否为主进程（分布式训练时只有主进程记录）
        
    Returns:
        ExperimentLogger实例
    """
    # 从配置读取日志设置
    logging_config = getattr(config, 'logging', None)
    
    use_wandb = True
    use_tensorboard = True
    project_name = "FSPT"
    entity = None
    watch_log_freq = 100
    
    if logging_config is not None:
        wandb_cfg = getattr(logging_config, 'wandb', None)
        if wandb_cfg is not None:
            use_wandb = bool(getattr(wandb_cfg, 'enabled', True))
            project_name = getattr(wandb_cfg, 'project', project_name)
            entity = getattr(wandb_cfg, 'entity', entity)
            log_freq = getattr(wandb_cfg, 'log_freq', None)
            if log_freq is not None:
                watch_log_freq = int(log_freq)
        else:
            use_wandb = bool(getattr(logging_config, 'use_wandb', True))
            project_name = getattr(logging_config, 'wandb_project', project_name)

        if hasattr(logging_config, 'tensorboard'):
            use_tensorboard = bool(getattr(logging_config.tensorboard, 'enabled', True))
        else:
            use_tensorboard = bool(getattr(logging_config, 'use_tensorboard', True))
    
    # 非主进程不记录
    if not is_main_process:
        use_wandb = False
        use_tensorboard = False
    
    return ExperimentLogger(
        config=config,
        output_dir=output_dir,
        project_name=project_name,
        use_wandb=use_wandb,
        use_tensorboard=use_tensorboard,
        entity=entity,
        watch_log_freq=watch_log_freq,
    )
