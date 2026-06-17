"""
频率感知遮挡预测器 (Frequency-Aware Occlusion Predictor)

核心思想：
1. 高频异常检测遮挡：高频信号突变表示遮挡
2. 低频轨迹外推：用低频轨迹预测遮挡期间的位置
3. 语义一致性传播：同一物体的点应有相似遮挡状态
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def _first_valid_band(band_features: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    for value in band_features.values():
        if value is not None:
            return value
    return None


class OcclusionDetector(nn.Module):
    """
    遮挡检测器
    
    基于频率特征检测遮挡
    高频信号突变往往表示遮挡发生
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_bands: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        num_bands = int(num_bands)
        if num_bands < 1:
            raise ValueError("num_bands must be >= 1")

        self.dim = dim
        self.num_bands = num_bands
        
        # 融合所有频带特征（考虑高频带权重更高）
        self.band_fusion = nn.Sequential(
            nn.Linear(dim * num_bands, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # 高频特征额外处理（高频突变是遮挡的重要信号）
        self.high_freq_detector = nn.Sequential(
            nn.Linear(dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )
        
        # 时序上下文
        self.temporal_context = nn.GRU(
            input_size=hidden_dim + hidden_dim // 4,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        
        # 遮挡分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        band_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            band_features: 各频带特征
                每个tensor形状为 (B, T, N, C)
                
        Returns:
            occlusion_prob: 遮挡概率 (B, T, N, 1)
        """
        # 获取形状（跳过None）
        first_band = _first_valid_band(band_features)
        if first_band is None:
            raise ValueError("No valid band features for occlusion detection")
        B, T, N, C = first_band.shape
        
        # 拼接所有频带（处理band数量不足的情况）
        available_bands = [band_features[f'band_{i}'] for i in range(self.num_bands) 
                          if f'band_{i}' in band_features and band_features[f'band_{i}'] is not None]
        
        # 如果band数量不足，复制最后一个band来填充
        while len(available_bands) < self.num_bands:
            if len(available_bands) > 0:
                available_bands.append(available_bands[-1])
            else:
                # 极端情况：没有有效band，使用零张量
                available_bands.append(torch.zeros(B, T, N, C, device=first_band.device, dtype=first_band.dtype))
        
        all_bands = torch.cat(available_bands, dim=-1)  # (B, T, N, C * num_bands)
        
        # 融合频带特征
        fused = self.band_fusion(all_bands)  # (B, T, N, hidden_dim)
        
        # 提取高频特征（最后一个频带通常是最高频）
        high_freq_key = f'band_{len(available_bands) - 1}'
        high_freq = band_features.get(high_freq_key, None)
        if high_freq is None:
            high_freq = available_bands[-1]
        high_freq_feat = self.high_freq_detector(high_freq)  # (B, T, N, hidden_dim//4)
        
        # 合并特征
        combined = torch.cat([fused, high_freq_feat], dim=-1)  # (B, T, N, hidden_dim + hidden_dim//4)
        
        # 时序上下文 (每个点独立处理)
        combined_flat = combined.permute(0, 2, 1, 3).reshape(B * N, T, -1)  # (B*N, T, ...)
        temporal, _ = self.temporal_context(combined_flat)  # (B*N, T, hidden_dim*2)
        temporal = temporal.reshape(B, N, T, -1).permute(0, 2, 1, 3)  # (B, T, N, hidden_dim*2)
        
        # 分类
        occlusion_prob = self.classifier(temporal)  # (B, T, N, 1)
        
        return occlusion_prob


class TrajectoryPredictor(nn.Module):
    """
    轨迹预测器
    
    基于低频分量预测遮挡期间的位置
    """
    
    def __init__(
        self,
        dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.dim = dim
        
        # 特征编码
        self.feature_encoder = nn.Linear(dim, hidden_dim)
        
        # 双向GRU用于轨迹建模
        self.trajectory_rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        
        # 位移预测头
        self.motion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),  # 预测 (dy, dx)
        )
        
        # 置信度预测
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        low_freq_feat: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            low_freq_feat: 低频特征 (B, T, N, C)
            positions: 当前位置 (B, T, N, 2)
            
        Returns:
            predicted_positions: 预测位置 (B, T, N, 2)
            confidence: 预测置信度 (B, T, N, 1)
        """
        B, T, N, C = low_freq_feat.shape
        
        # 编码特征
        feat = self.feature_encoder(low_freq_feat)  # (B, T, N, hidden_dim)
        
        # 每个点独立处理
        feat_flat = feat.permute(0, 2, 1, 3).reshape(B * N, T, -1)  # (B*N, T, hidden_dim)
        
        # 轨迹建模
        trajectory, _ = self.trajectory_rnn(feat_flat)  # (B*N, T, hidden_dim*2)
        trajectory = trajectory.reshape(B, N, T, -1).permute(0, 2, 1, 3)  # (B, T, N, hidden_dim*2)
        
        # 预测运动
        motion = self.motion_head(trajectory)  # (B, T, N, 2)
        
        # 预测位置 = 当前位置 + 预测运动
        predicted_positions = positions + motion
        predicted_positions = torch.clamp(predicted_positions, 0, 1)
        
        # 预测置信度
        confidence = self.confidence_head(trajectory)  # (B, T, N, 1)
        
        return predicted_positions, confidence


class SemanticConsistencyModule(nn.Module):
    """
    语义一致性模块
    
    传播遮挡信息到语义相似的点
    """
    
    def __init__(
        self,
        semantic_dim: int = 512,
        temperature: float = 0.1,
        max_chunk_size: Optional[int] = 256,
    ):
        super().__init__()
        
        self.temperature = temperature
        if max_chunk_size is not None:
            try:
                max_chunk_size = int(max_chunk_size)
            except (TypeError, ValueError):
                max_chunk_size = None
        if max_chunk_size is not None and max_chunk_size <= 0:
            max_chunk_size = None
        self.max_chunk_size = max_chunk_size
        
        # 语义相似度投影
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim // 2),
            nn.GELU(),
            nn.Linear(semantic_dim // 2, semantic_dim // 4),
        )
        
    def forward(
        self,
        occlusion_prob: torch.Tensor,
        semantic_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            occlusion_prob: 遮挡概率 (B, T, N, 1)
            semantic_feat: 语义特征 (B, T, N, semantic_dim)
            
        Returns:
            refined_prob: 精化后的遮挡概率 (B, T, N, 1)
        """
        B, T, N, _ = occlusion_prob.shape
        
        # 投影语义特征
        semantic_proj = self.semantic_proj(semantic_feat)  # (B, T, N, dim)
        
        # 计算语义相似度矩阵
        semantic_flat = semantic_proj.reshape(B * T, N, -1)  # (B*T, N, dim)
        
        # 归一化
        semantic_norm = F.normalize(semantic_flat, dim=-1)
        
        # 相似度矩阵
        occ_flat = occlusion_prob.reshape(B * T, N, 1)  # (B*T, N, 1)
        semantic_norm_t = semantic_norm.transpose(1, 2)  # (B*T, C, N)
        chunk_size = self.max_chunk_size
        if chunk_size is None or chunk_size >= N:
            similarity = torch.bmm(semantic_norm, semantic_norm_t)  # (B*T, N, N)
            # 温度缩放的softmax
            attention = F.softmax(similarity / self.temperature, dim=-1)
            # 传播遮挡信息
            refined = torch.bmm(attention, occ_flat)  # (B*T, N, 1)
        else:
            refined_chunks = []
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                q = semantic_norm[:, start:end, :]  # (B*T, chunk, C)
                similarity = torch.bmm(q, semantic_norm_t)  # (B*T, chunk, N)
                attention = F.softmax(similarity / self.temperature, dim=-1)
                refined_chunk = torch.bmm(attention, occ_flat)  # (B*T, chunk, 1)
                refined_chunks.append(refined_chunk)
            refined = torch.cat(refined_chunks, dim=1)
        
        refined_prob = refined.reshape(B, T, N, 1)
        
        return refined_prob


class FrequencyAwareOcclusionPredictor(nn.Module):
    """
    频率感知遮挡预测器
    
    完整模块，结合:
    1. 遮挡检测
    2. 轨迹预测
    3. 语义一致性
    """
    
    def __init__(
        self,
        dim: int = 256,
        semantic_dim: int = 512,
        num_bands: int = 4,
        use_semantic_consistency: bool = True,
        use_low_freq_prediction: bool = True,
    ):
        super().__init__()

        num_bands = int(num_bands)
        if num_bands < 1:
            raise ValueError("num_bands must be >= 1")

        self.dim = dim
        self.num_bands = num_bands
        self.use_semantic_consistency = use_semantic_consistency
        self.use_low_freq_prediction = use_low_freq_prediction
        
        # 遮挡检测器
        self.occlusion_detector = OcclusionDetector(
            dim=dim,
            num_bands=num_bands,
        )
        
        # 轨迹预测器（可选）
        if self.use_low_freq_prediction:
            self.trajectory_predictor = TrajectoryPredictor(
                dim=dim,
            )
        else:
            self.trajectory_predictor = None
        
        # 语义一致性模块
        if use_semantic_consistency:
            self.semantic_consistency = SemanticConsistencyModule(
                semantic_dim=semantic_dim,
            )
        
    def forward(
        self,
        band_features: Dict[str, torch.Tensor],
        semantic_feat: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            band_features: 各频带特征
            semantic_feat: 语义特征 (B, T, N, semantic_dim)
            positions: 当前位置 (B, T, N, 2)
            
        Returns:
            occlusion_prob: 遮挡概率 (B, T, N)
            predicted_positions: 预测位置 (B, T, N, 2)
            confidence: 预测置信度 (B, T, N)
        """
        # 1. 遮挡检测
        occlusion_prob = self.occlusion_detector(band_features)  # (B, T, N, 1)
        
        # 2. 语义一致性精化
        if self.use_semantic_consistency:
            occlusion_prob = self.semantic_consistency(occlusion_prob, semantic_feat)
        
        # 3. 低频轨迹预测
        if self.use_low_freq_prediction:
            low_freq_feat = band_features.get('band_0', None)
            if low_freq_feat is None:
                low_freq_feat = _first_valid_band(band_features)
            if low_freq_feat is None:
                raise ValueError("No valid band features for trajectory prediction")
            if self.trajectory_predictor is None:
                raise RuntimeError("Trajectory predictor is disabled but low-freq prediction is enabled.")
            predicted_positions, confidence = self.trajectory_predictor(low_freq_feat, positions)
        else:
            predicted_positions = positions
            confidence = torch.ones_like(occlusion_prob)
        
        return (
            occlusion_prob.squeeze(-1),  # (B, T, N)
            predicted_positions,  # (B, T, N, 2)
            confidence.squeeze(-1),  # (B, T, N)
        )


if __name__ == '__main__':
    # 测试
    print("Testing FrequencyAwareOcclusionPredictor...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = FrequencyAwareOcclusionPredictor(
        dim=256,
        semantic_dim=512,
        num_bands=4,
        use_semantic_consistency=True,
    )
    model = model.to(device)
    
    # 测试输入
    B, T, N = 2, 24, 100
    band_features = {
        f'band_{i}': torch.randn(B, T, N, 256, device=device)
        for i in range(4)
    }
    semantic_feat = torch.randn(B, T, N, 512, device=device)
    positions = torch.rand(B, T, N, 2, device=device)
    
    # 前向传播
    occlusion_prob, predicted_positions, confidence = model(
        band_features, semantic_feat, positions
    )
    
    print(f"Occlusion prob shape: {occlusion_prob.shape}")
    print(f"Predicted positions shape: {predicted_positions.shape}")
    print(f"Confidence shape: {confidence.shape}")
    print(f"Occlusion prob range: [{occlusion_prob.min():.3f}, {occlusion_prob.max():.3f}]")
    
    print("Test passed!")
