"""
数据预取器

异步将数据加载到GPU，减少数据传输延迟
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional, Iterator, Dict, Any


class DataPrefetcher:
    """
    数据预取器
    
    在GPU训练的同时异步加载下一批数据到GPU
    可显著减少数据传输造成的训练延迟
    
    用法:
        prefetcher = DataPrefetcher(dataloader, device)
        for batch in prefetcher:
            # 训练代码
            pass
    """
    
    def __init__(
        self,
        loader: DataLoader,
        device: torch.device,
        non_blocking: bool = True,
    ):
        """
        Args:
            loader: PyTorch DataLoader
            device: 目标设备
            non_blocking: 是否使用非阻塞传输
        """
        self.loader = loader
        self.device = device
        self.non_blocking = non_blocking
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self.loader_iter = iter(self.loader)
        self._preload()
        return self
    
    def __next__(self) -> Dict[str, Any]:
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        
        # 确保数据完全传输
        if self.stream is not None:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    value.record_stream(torch.cuda.current_stream())
        
        self._preload()
        return batch
    
    def _preload(self):
        """预加载下一批数据"""
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.next_batch = None
            return
        
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_batch = self._to_device(batch)
        else:
            self.next_batch = self._to_device(batch)
    
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """将batch移动到目标设备"""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device, non_blocking=self.non_blocking)
            elif isinstance(value, dict):
                result[key] = self._to_device(value)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                result[key] = [v.to(self.device, non_blocking=self.non_blocking) for v in value]
            else:
                result[key] = value
        return result
    
    def __len__(self) -> int:
        return len(self.loader)


class CUDAGraphWrapper:
    """
    CUDA Graph包装器
    
    将重复的训练步骤捕获为CUDA Graph以减少CPU开销
    注意：仅适用于固定形状的输入
    
    用法:
        graph_wrapper = CUDAGraphWrapper(model, sample_batch, device)
        for batch in dataloader:
            loss = graph_wrapper.run(batch)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        sample_batch: Dict[str, torch.Tensor],
        device: torch.device,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        use_amp: bool = True,
        amp_dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            model: 训练模型
            loss_fn: 损失函数
            optimizer: 优化器
            sample_batch: 用于捕获graph的样例batch
            device: 训练设备
            scaler: 混合精度scaler
            use_amp: 是否使用混合精度
            amp_dtype: autocast dtype（如 torch.float16 / torch.bfloat16）
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scaler = scaler
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        
        # 检查CUDA Graph可用性
        if device.type != 'cuda' or not torch.cuda.is_available():
            self.graph = None
            self.static_input = None
            return
        
        # 创建静态输入缓冲区
        self.static_input = {}
        for key, value in sample_batch.items():
            if isinstance(value, torch.Tensor):
                self.static_input[key] = value.clone().to(device)
        
        # 预热
        self.model.train()
        for _ in range(3):
            self._forward_backward(self.static_input)
        
        # 捕获CUDA Graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_loss = self._forward_backward(self.static_input)
    
    def _forward_backward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """执行前向和反向传播"""
        self.optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(
            enabled=self.use_amp and self.device.type == 'cuda',
            dtype=self.amp_dtype,
        ):
            video = batch['video']
            query_points = batch['query_points']
            target_points = batch['target_points']
            occluded = batch['occluded']
            if isinstance(occluded, torch.Tensor) and occluded.dtype != torch.bool:
                occluded = occluded > 0.5
            
            outputs = self.model(video, query_points)
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
                pred_tracks, pred_visibility = outputs[0], outputs[1]
            else:
                raise ValueError("Model output format not supported")
            
            losses = self.loss_fn(
                pred_tracks,
                target_points,
                pred_visibility,
                ~occluded,
            )
            loss = losses['total']
        
        use_scaler = self.scaler is not None and (self.amp_dtype is None or self.amp_dtype == torch.float16)
        if use_scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        return loss
    
    def run(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """执行训练步骤"""
        if self.graph is None:
            # 回退到普通训练
            return self._forward_backward(batch)
        
        # 复制数据到静态缓冲区
        for key, value in batch.items():
            if key in self.static_input and isinstance(value, torch.Tensor):
                value = value.to(self.device)
                self.static_input[key].copy_(value)
        
        # 重放CUDA Graph
        self.graph.replay()
        
        return self.static_loss


def create_prefetched_dataloader(
    loader: DataLoader,
    device: torch.device,
) -> DataPrefetcher:
    """
    创建预取数据加载器
    
    Args:
        loader: 原始DataLoader
        device: 目标设备
        
    Returns:
        DataPrefetcher实例
    """
    return DataPrefetcher(loader, device)
