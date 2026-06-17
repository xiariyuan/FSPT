"""
分布式训练工具

支持多GPU数据并行训练
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def setup_distributed(
    backend: str = 'nccl',
    init_method: str = 'env://',
) -> Tuple[int, int, bool]:
    """
    初始化分布式训练环境
    
    Args:
        backend: 分布式后端 ('nccl' 推荐用于GPU)
        init_method: 初始化方法
        
    Returns:
        rank: 当前进程rank
        world_size: 总进程数
        is_main_process: 是否为主进程
    """
    # 检查是否在分布式环境中
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        logger.info("Not in distributed environment, running in single GPU mode")
        return 0, 1, True
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    if 'LOCAL_RANK' not in os.environ:
        logger.warning("LOCAL_RANK not set; defaulting to 0.")
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # 设置当前设备
    if backend == 'nccl' and dist.is_available() and not dist.is_nccl_available():
        logger.warning("NCCL backend not available; falling back to gloo.")
        backend = 'gloo'
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    else:
        if backend == 'nccl':
            logger.warning("CUDA not available; falling back to gloo backend.")
            backend = 'gloo'
    
    # 初始化进程组
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
    
    is_main_process = rank == 0
    
    if is_main_process:
        logger.info(f"Distributed training initialized: rank={rank}, world_size={world_size}")
    
    return rank, world_size, is_main_process


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device(local_rank: Optional[int] = None) -> torch.device:
    """
    获取当前设备
    
    Args:
        local_rank: 本地rank (用于多GPU)
        
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        if local_rank is not None:
            return torch.device(f'cuda:{local_rank}')
        return torch.device('cuda')
    return torch.device('cpu')


def wrap_model_ddp(
    model: torch.nn.Module,
    device: torch.device,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True,
) -> torch.nn.Module:
    """
    包装模型为DistributedDataParallel
    
    Args:
        model: 原始模型
        device: 设备
        find_unused_parameters: 是否查找未使用参数
        gradient_as_bucket_view: 梯度作为bucket视图（节省内存）
        
    Returns:
        DDP包装的模型
    """
    if not dist.is_initialized():
        return model
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    model = model.to(device)
    if device.type == 'cuda':
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )
    else:
        model = DDP(
            model,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )
    
    return model


def create_distributed_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """
    创建分布式DataLoader
    
    Args:
        dataset: 数据集
        batch_size: 批次大小（每个GPU）
        num_workers: 数据加载线程数
        shuffle: 是否打乱
        pin_memory: 是否使用pinned memory
        drop_last: 是否丢弃最后不完整批次
        seed: 随机种子
        
    Returns:
        dataloader: DataLoader
        sampler: DistributedSampler (用于设置epoch)
    """
    if dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            seed=seed,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        return dataloader, sampler
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        return dataloader, None


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    跨进程reduce tensor
    
    Args:
        tensor: 输入tensor
        average: 是否取平均
        
    Returns:
        reduced tensor
    """
    if not dist.is_initialized():
        return tensor
    
    world_size = dist.get_world_size()
    
    if world_size == 1:
        return tensor

    if tensor.numel() == 0:
        return tensor
    
    with torch.no_grad():
        dist.all_reduce(tensor)
        if average:
            tensor /= world_size
    
    return tensor


def all_gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    收集所有进程的tensor
    
    Args:
        tensor: 输入tensor
        
    Returns:
        gathered tensor (concatenated)
    """
    if not dist.is_initialized():
        return tensor
    
    world_size = dist.get_world_size()
    
    if world_size == 1:
        return tensor

    if tensor.numel() == 0:
        return tensor
    
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    
    return torch.cat(gathered, dim=0)


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """
    广播Python对象
    
    Args:
        obj: Python对象
        src: 源进程rank
        
    Returns:
        广播后的对象
    """
    if not dist.is_initialized():
        return obj
    
    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def barrier():
    """进程同步屏障"""
    if dist.is_initialized():
        dist.barrier()


def is_main_process() -> bool:
    """检查是否为主进程"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """获取总进程数"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """获取当前进程rank"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


class DistributedMetricLogger:
    """
    分布式指标记录器
    
    自动同步跨进程的指标
    """
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, key: str, value: float, count: int = 1):
        """更新指标"""
        if key not in self.metrics:
            self.metrics[key] = 0.0
            self.counts[key] = 0
        
        self.metrics[key] += value * count
        self.counts[key] += count
    
    def sync_and_get(self, key: str, device: Optional[torch.device] = None) -> float:
        """同步并获取指标"""
        if key not in self.metrics:
            return 0.0
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        value_tensor = torch.tensor([self.metrics[key]], device=device)
        count_tensor = torch.tensor([self.counts[key]], device=device)
        
        value_tensor = reduce_tensor(value_tensor, average=False)
        count_tensor = reduce_tensor(count_tensor, average=False)
        
        if count_tensor.item() > 0:
            return value_tensor.item() / count_tensor.item()
        return 0.0
    
    def sync_all(self) -> dict:
        """同步所有指标"""
        result = {}
        for key in self.metrics:
            result[key] = self.sync_and_get(key)
        return result
    
    def reset(self):
        """重置指标"""
        self.metrics = {}
        self.counts = {}
