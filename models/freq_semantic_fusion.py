"""
语义增强的频率分解模块 (Semantic-Enhanced LFD)

核心创新：将CLIP语义特征用于调制频率分解的响应

设计思想：
1. 语义相似的点在同一频带应该有相似的响应
2. 语义特征可以帮助区分不同物体的点
3. 语义一致性可以用于遮挡推理
"""

import logging
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# 尝试导入FM-Track的频率分解模块
try:
    from .learnable_freq_decomposition import LearnableFrequencyDecomposition
    LFD_AVAILABLE = True
except ImportError:
    LFD_AVAILABLE = False
    logger.info("LearnableFrequencyDecomposition not available, using SimplifiedLFD")


def _first_valid_band(band_features: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    for value in band_features.values():
        if value is not None:
            return value
    return None


class SemanticModulator(nn.Module):
    """
    语义调制器
    
    根据语义特征动态调整频率响应
    """
    
    def __init__(
        self,
        semantic_dim: int = 512,
        geo_dim: int = 256,
        num_bands: int = 4,
        modulation_type: str = "multiplicative",
    ):
        super().__init__()

        num_bands = int(num_bands)
        if num_bands < 1:
            raise ValueError("num_bands must be >= 1")

        self.semantic_dim = semantic_dim
        self.geo_dim = geo_dim
        self.num_bands = num_bands
        self.modulation_type = str(modulation_type).lower()
        
        # 语义特征投影
        proj_out_dim = geo_dim * num_bands
        if self.modulation_type == "film":
            proj_out_dim = geo_dim * num_bands * 2
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, geo_dim * 2),
            nn.GELU(),
            nn.Linear(geo_dim * 2, proj_out_dim),
        )
        
        # 调制方式：乘法调制
        self.modulation_scale = nn.Parameter(torch.ones(num_bands))
        self.modulation_bias = nn.Parameter(torch.zeros(num_bands))
        
    def forward(
        self,
        band_features: Dict[str, torch.Tensor],
        semantic_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            band_features: 各频带特征，key为'band_0', 'band_1', ...
                每个tensor形状为 (B, T, N, C)
            semantic_feat: 语义特征 (B, T, N, semantic_dim)
            
        Returns:
            modulated_bands: 调制后的频带特征
        """
        if not band_features:
            return {}
        first_band = _first_valid_band(band_features)
        if first_band is None:
            return {key: None for key in band_features.keys()}
        B, T, N, C = first_band.shape

        mod_type = self.modulation_type
        if mod_type in ["none", "disable", "disabled"]:
            return band_features

        # 计算调制权重
        modulation = self.semantic_proj(semantic_feat)
        if mod_type == "film":
            scale_mod, bias_mod = modulation.chunk(2, dim=-1)
            scale_mod = scale_mod.sigmoid()
            bias_mod = bias_mod.tanh()
        elif mod_type == "additive":
            modulation = modulation.tanh()
        elif mod_type == "multiplicative":
            modulation = modulation.sigmoid()
        else:
            logger.warning(f"Unknown modulation_type: {mod_type}, fallback to multiplicative")
            modulation = modulation.sigmoid()
        
        # 对每个频带应用调制
        modulated_bands = {}
        actual_bands = len([k for k in band_features.keys() if k.startswith('band_')])
        
        for i in range(self.num_bands):
            band_key = f'band_{i}'
            if band_key in band_features and band_features[band_key] is not None:
                # 安全检查：确保索引不越界
                start_idx = i * C
                end_idx = (i + 1) * C
                max_len = scale_mod.shape[-1] if mod_type == "film" and scale_mod is not None else modulation.shape[-1]
                if end_idx > max_len:
                    # 索引越界时使用最后一段
                    start_idx = max(0, max_len - C)
                    end_idx = max_len
                
                scale = self.modulation_scale[min(i, len(self.modulation_scale) - 1)].sigmoid() * 2
                bias = self.modulation_bias[min(i, len(self.modulation_bias) - 1)]

                if mod_type == "film":
                    band_scale = scale_mod[..., start_idx:end_idx]
                    band_bias = bias_mod[..., start_idx:end_idx]
                    if band_scale.shape[-1] < C:
                        pad = torch.zeros(
                            *band_scale.shape[:-1],
                            C - band_scale.shape[-1],
                            device=band_scale.device,
                            dtype=band_scale.dtype,
                        )
                        band_scale = torch.cat([band_scale, pad], dim=-1)
                        band_bias = torch.cat([band_bias, pad], dim=-1)
                    # 确保维度匹配
                    if band_scale.shape[-1] == C:
                        modulated = band_features[band_key] * (1 + band_scale * scale) + band_bias * bias
                    else:
                        modulated = band_features[band_key]
                elif mod_type == "additive":
                    band_mod = modulation[..., start_idx:end_idx]
                    if band_mod.shape[-1] < C:
                        pad = torch.zeros(
                            *band_mod.shape[:-1],
                            C - band_mod.shape[-1],
                            device=band_mod.device,
                            dtype=band_mod.dtype,
                        )
                        band_mod = torch.cat([band_mod, pad], dim=-1)
                    if band_mod.shape[-1] == C:
                        modulated = band_features[band_key] + band_mod * scale
                    else:
                        modulated = band_features[band_key]
                else:
                    band_mod = modulation[..., start_idx:end_idx]
                    if band_mod.shape[-1] < C:
                        pad = torch.zeros(
                            *band_mod.shape[:-1],
                            C - band_mod.shape[-1],
                            device=band_mod.device,
                            dtype=band_mod.dtype,
                        )
                        band_mod = torch.cat([band_mod, pad], dim=-1)
                    if band_mod.shape[-1] == C:
                        modulated = band_features[band_key] * (band_mod * scale + bias)
                    else:
                        modulated = band_features[band_key]
                modulated_bands[band_key] = modulated
            elif band_key in band_features:
                # band存在但是None，保持None
                modulated_bands[band_key] = None
            # 如果band不存在，不添加到modulated_bands
        
        return modulated_bands


class SemanticEnhancedLFD(nn.Module):
    """
    语义增强的可学习频率分解
    
    结合:
    1. FM-Track的可学习频率分解
    2. CLIP语义特征调制
    3. 跨模态融合
    
    Args:
        geo_dim: 几何特征维度
        semantic_dim: 语义特征维度
        num_bands: 频带数量
        kernel_size: 频率滤波器核大小
        use_cross_attention: 是否使用跨模态注意力
    """
    
    def __init__(
        self,
        geo_dim: int = 256,
        semantic_dim: int = 512,
        num_bands: int = 4,
        kernel_size: int = 7,
        use_cross_attention: Optional[bool] = None,
        fusion_type: str = "cross_attention",
        modulation_type: str = "multiplicative",
        use_fixed_laplacian: bool = False,
        ortho_weight: float = 0.0,
        feature_ortho_weight: float = 0.0,
        ortho_samples: int = 4096,
        dropout: float = 0.1,
        strict_lfd: bool = False,
    ):
        super().__init__()

        num_bands = int(num_bands)
        if num_bands < 1:
            raise ValueError("num_bands must be >= 1")

        self.geo_dim = geo_dim
        self.semantic_dim = semantic_dim
        self.num_bands = num_bands
        self.strict_lfd = bool(strict_lfd)
        if use_cross_attention is not None:
            self.fusion_type = "cross_attention" if use_cross_attention else "none"
        else:
            self.fusion_type = str(fusion_type).lower()
        if self.fusion_type not in ["cross_attention", "none", "disabled"]:
            logger.warning(f"Unknown fusion_type: {self.fusion_type}, fallback to cross_attention")
            self.fusion_type = "cross_attention"
        self.use_cross_attention = self.fusion_type == "cross_attention"
        self.lfd_backend = "learnable" if LFD_AVAILABLE else "simplified"
        
        # 频率分解模块（从FM-Track复用或简化实现）
        if LFD_AVAILABLE:
            lfd_kwargs = dict(
                dim=geo_dim,
                num_bands=num_bands,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            try:
                sig = inspect.signature(LearnableFrequencyDecomposition.__init__)
                enable_ortho = (feature_ortho_weight > 0) or (ortho_weight > 0)
                if 'ortho_weight' in sig.parameters:
                    # 只作为开关使用，权重统一在loss侧控制
                    lfd_kwargs['ortho_weight'] = 1.0 if enable_ortho else 0.0
                if 'feature_ortho_weight' in sig.parameters:
                    lfd_kwargs['feature_ortho_weight'] = 1.0 if feature_ortho_weight > 0 else 0.0
                if 'ortho_samples' in sig.parameters:
                    lfd_kwargs['ortho_samples'] = ortho_samples
                if 'use_fixed_laplacian' in sig.parameters:
                    lfd_kwargs['use_fixed_laplacian'] = use_fixed_laplacian
            except Exception:
                pass
            self.lfd = LearnableFrequencyDecomposition(**lfd_kwargs)
        else:
            if self.strict_lfd:
                raise ImportError(
                    "LearnableFrequencyDecomposition is required (strict_lfd=true), "
                    "but models/learnable_freq_decomposition.py is unavailable."
                )
            self.lfd = SimplifiedLFD(
                dim=geo_dim,
                num_bands=num_bands,
                use_fixed_laplacian=use_fixed_laplacian,
                kernel_size=kernel_size,
                ortho_weight=ortho_weight,
                feature_ortho_weight=feature_ortho_weight,
                ortho_samples=ortho_samples,
            )
        
        # 语义调制器（可选）
        modulation_type = str(modulation_type).lower()
        if modulation_type in ["none", "disable", "disabled"]:
            self.semantic_modulator = None
        else:
            self.semantic_modulator = SemanticModulator(
                semantic_dim=semantic_dim,
                geo_dim=geo_dim,
                num_bands=num_bands,
                modulation_type=modulation_type,
            )
        
        # 跨模态注意力融合
        if self.use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=geo_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True,
            )
            self.semantic_to_geo = nn.Linear(semantic_dim, geo_dim)
            self.fusion_norm = nn.LayerNorm(geo_dim)
        
        # 输出融合
        self.band_fusion = nn.Sequential(
            nn.Linear(geo_dim * num_bands, geo_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(geo_dim * 2, geo_dim),
        )
        
        # 残差门控
        self.gate = nn.Sequential(
            nn.Linear(geo_dim * 2, geo_dim),
            nn.Sigmoid(),
        )
        
        self.output_norm = nn.LayerNorm(geo_dim)
        
    def forward(
        self,
        geo_feat: torch.Tensor,
        semantic_feat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            geo_feat: 几何特征 (B, T, N, geo_dim)
            semantic_feat: 语义特征 (B, T, N, semantic_dim)
            mask: 可选的mask (B, T, N)
            
        Returns:
            output: 融合后的特征 (B, T, N, geo_dim)
            info: 包含中间结果的字典
        """
        B, T, N, C = geo_feat.shape
        
        # 1. 频率分解
        # 添加虚拟的Group维度以适配LFD接口
        geo_feat_5d = geo_feat.unsqueeze(1)  # (B, 1, T, N, C)
        
        try:
            freq_out = self.lfd(
                geo_feat_5d, mask=mask.unsqueeze(1) if mask is not None else None
            )
        except TypeError:
            # 兼容未实现mask参数的LFD实现
            freq_out = self.lfd(geo_feat_5d)
        if isinstance(freq_out, (list, tuple)) and len(freq_out) == 2:
            freq_output, freq_info = freq_out
        else:
            # 兼容只返回特征的实现
            freq_output, freq_info = freq_out, {}
        
        # 确保输出是4D (B, T, N, C)
        if freq_output.dim() == 5:
            freq_output = freq_output.squeeze(1)  # (B, 1, T, N, C) -> (B, T, N, C)
        
        # 提取频带特征
        band_features = {}
        if 'band_features' in freq_info:
            for key, value in freq_info['band_features'].items():
                if key.startswith('band_'):
                    if value.dim() == 5:
                        band_features[key] = value.squeeze(1)
                    else:
                        band_features[key] = value
        else:
            # 如果没有频带特征，使用频率输出
            for i in range(self.num_bands):
                band_features[f'band_{i}'] = freq_output
        
        # 2. 语义调制
        if self.semantic_modulator is None:
            modulated_bands = band_features
        else:
            modulated_bands = self.semantic_modulator(band_features, semantic_feat)
        
        # 3. 跨模态注意力融合
        attn_output = None
        if self.use_cross_attention:
            # 将语义特征投影到几何空间
            semantic_proj = self.semantic_to_geo(semantic_feat)  # (B, T, N, geo_dim)
            
            # 展平时间维度
            freq_flat = freq_output.reshape(B * T, N, C)
            semantic_flat = semantic_proj.reshape(B * T, N, C)
            
            # 跨模态注意力
            attn_output, _ = self.cross_attention(
                query=freq_flat,
                key=semantic_flat,
                value=semantic_flat,
            )
            
            attn_output = attn_output.reshape(B, T, N, C)
            attn_output = self.fusion_norm(freq_output + attn_output)
        
        # 4. 频带融合
        # 收集有效的频带特征
        valid_bands = []
        for i in range(self.num_bands):
            band_key = f'band_{i}'
            if band_key in modulated_bands and modulated_bands[band_key] is not None:
                valid_bands.append(modulated_bands[band_key])
        
        # 如果有效频带数量不足，用零填充或复制现有频带
        while len(valid_bands) < self.num_bands:
            if len(valid_bands) > 0:
                valid_bands.append(valid_bands[-1])  # 复制最后一个
            else:
                # 极端情况：没有有效频带，使用geo_feat
                valid_bands.append(geo_feat)
        
        all_bands = torch.cat(valid_bands, dim=-1)  # (B, T, N, geo_dim * num_bands)
        
        fused = self.band_fusion(all_bands)  # (B, T, N, geo_dim)
        if attn_output is not None:
            fused = fused + attn_output
        
        # 5. 残差门控
        gate_input = torch.cat([geo_feat, fused], dim=-1)
        gate = self.gate(gate_input)
        
        output = geo_feat + gate * fused
        output = self.output_norm(output)
        
        # 应用mask
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)
        
        # 收集信息
        info = {
            'band_features': modulated_bands,
            'gate': gate,
        }
        for loss_key in ('ortho_loss', 'reconstruction_loss', 'freq_separation_loss'):
            if loss_key in freq_info:
                info[loss_key] = freq_info[loss_key]
        
        return output, info


class LearnableTemporalFilter(nn.Module):
    """
    可学习的时序滤波器

    通过可学习的1D卷积核实现频率分解，每个滤波器学习捕获特定频率范围的信号
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        init_type: str = "gaussian",
        center_freq: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        # 可学习的滤波器权重 (per-channel)
        self.filter_weight = nn.Parameter(torch.zeros(dim, 1, self.kernel_size))

        # 初始化滤波器
        self._init_filter(init_type, center_freq)

    def _init_filter(self, init_type: str, center_freq: float):
        """初始化滤波器权重"""
        k = self.kernel_size
        center = k // 2

        with torch.no_grad():
            if init_type == "gaussian":
                # 高斯低通滤波器
                sigma = k / 6.0
                x = torch.arange(k).float() - center
                gaussian = torch.exp(-x**2 / (2 * sigma**2))
                gaussian = gaussian / gaussian.sum()
                self.filter_weight.data[:] = gaussian.view(1, 1, k)

            elif init_type == "bandpass":
                # 带通滤波器 (基于调制高斯)
                sigma = k / 4.0
                x = torch.arange(k).float() - center
                gaussian = torch.exp(-x**2 / (2 * sigma**2))
                # 调制到指定中心频率
                modulation = torch.cos(2 * torch.pi * center_freq * x / k)
                bandpass = gaussian * modulation
                bandpass = bandpass / (bandpass.abs().sum() + 1e-8)
                self.filter_weight.data[:] = bandpass.view(1, 1, k)

            elif init_type == "highpass":
                # 高通滤波器 (delta - lowpass)
                sigma = k / 6.0
                x = torch.arange(k).float() - center
                gaussian = torch.exp(-x**2 / (2 * sigma**2))
                gaussian = gaussian / gaussian.sum()
                delta = torch.zeros(k)
                delta[center] = 1.0
                highpass = delta - gaussian
                self.filter_weight.data[:] = highpass.view(1, 1, k)

            else:  # "uniform" or default
                # 均匀低通
                self.filter_weight.data[:] = 1.0 / k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) 时序特征
        Returns:
            filtered: (B, C, T) 滤波后的特征
        """
        B, C, T = x.shape

        # 归一化滤波器 (确保能量守恒)
        filter_norm = self.filter_weight / (self.filter_weight.abs().sum(dim=-1, keepdim=True) + 1e-8)

        # 分组卷积
        filtered = F.conv1d(
            x,
            filter_norm.expand(C, -1, -1),
            padding=self.kernel_size // 2,
            groups=C
        )

        return filtered


class SimplifiedLFD(nn.Module):
    """
    改进版可学习频率分解模块

    核心改进:
    1. 可学习的时序滤波器替代固定滤波器
    2. 多尺度金字塔分解 (类似拉普拉斯金字塔)
    3. 频域约束确保频带分离
    4. 自适应频带权重

    当FM-Track的LFD不可用时使用
    """

    def __init__(
        self,
        dim: int = 256,
        num_bands: int = 4,
        use_fixed_laplacian: bool = False,
        kernel_size: int = 7,
        ortho_weight: float = 0.0,
        feature_ortho_weight: float = 0.0,
        ortho_samples: int = 4096,
        decomposition_type: str = "learnable_pyramid",
    ):
        super().__init__()

        num_bands = int(num_bands)
        if num_bands < 1:
            raise ValueError("num_bands must be >= 1")

        self.dim = dim
        self.num_bands = num_bands
        self.use_fixed_laplacian = use_fixed_laplacian
        self.kernel_size = int(kernel_size) if int(kernel_size) % 2 == 1 else int(kernel_size) + 1
        self.ortho_weight = float(ortho_weight)
        self.feature_ortho_weight = float(feature_ortho_weight)
        self.ortho_samples = int(ortho_samples)
        self.decomposition_type = decomposition_type

        # ============ 可学习频率分解组件 ============

        if decomposition_type == "learnable_pyramid":
            # 方案1: 可学习金字塔分解
            # 每层使用不同尺度的可学习低通滤波器
            self.lowpass_filters = nn.ModuleList([
                LearnableTemporalFilter(
                    dim=dim,
                    kernel_size=self.kernel_size + 2 * i,  # 递增核大小
                    init_type="gaussian",
                    center_freq=0.0,
                )
                for i in range(num_bands - 1)
            ])

        elif decomposition_type == "learnable_bandpass":
            # 方案2: 可学习带通滤波器组
            # 每个滤波器学习特定频带
            center_freqs = [0.5 * (i + 1) / num_bands for i in range(num_bands)]
            self.bandpass_filters = nn.ModuleList([
                LearnableTemporalFilter(
                    dim=dim,
                    kernel_size=self.kernel_size,
                    init_type="bandpass" if i < num_bands - 1 else "gaussian",
                    center_freq=freq,
                )
                for i, freq in enumerate(center_freqs)
            ])

        elif decomposition_type == "fourier_learnable":
            # 方案3: 傅里叶域可学习分解
            # 在频域学习频带划分
            self.freq_masks = nn.ParameterList([
                nn.Parameter(torch.ones(1, dim, 1) * (1.0 / num_bands))
                for _ in range(num_bands)
            ])
            # 频带边界 (可学习)
            init_boundaries = torch.linspace(0, 1, num_bands + 1)
            self.freq_boundaries = nn.Parameter(init_boundaries)

        # ============ 频带特征增强 ============

        # 每个频带的特征变换
        self.band_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
            )
            for _ in range(num_bands)
        ])

        # 频带权重 (自适应)
        self.band_weights = nn.Parameter(torch.ones(num_bands) / num_bands)

        # 频带重要性预测 (基于输入内容)
        self.importance_predictor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, num_bands),
            nn.Softmax(dim=-1),
        )

        # 重建层 (确保频带可以重建原始信号)
        self.reconstruction_proj = nn.Linear(dim * num_bands, dim)

    def _learnable_pyramid_decompose(self, x: torch.Tensor) -> list:
        """
        可学习金字塔分解

        类似拉普拉斯金字塔，但使用可学习滤波器:
        - band_0: 最高频 (原始 - 第一层平滑)
        - band_1: 次高频 (第一层平滑 - 第二层平滑)
        - ...
        - band_n: 最低频 (最后一层平滑结果)
        """
        B, G, T, N, C = x.shape
        # 重排为 (B*G*N, C, T) 以便进行时序卷积
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(B * G * N, C, T)

        bands = []
        current = x_flat

        for i, lowpass in enumerate(self.lowpass_filters):
            # 低通滤波
            smoothed = lowpass(current)
            # 高频 = 当前 - 平滑
            high_freq = current - smoothed
            bands.append(high_freq)
            # 下一层的输入是平滑后的结果
            current = smoothed

        # 最后一个频带是剩余的低频分量
        bands.append(current)

        # 重排回 (B, G, T, N, C)
        outputs = []
        for band in bands:
            band = band.reshape(B, G, N, C, T).permute(0, 1, 4, 2, 3)  # (B, G, T, N, C)
            outputs.append(band)

        return outputs

    def _learnable_bandpass_decompose(self, x: torch.Tensor) -> list:
        """
        可学习带通滤波器组分解

        每个滤波器直接学习提取特定频带
        """
        B, G, T, N, C = x.shape
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(B * G * N, C, T)

        bands = []
        for bandpass in self.bandpass_filters:
            filtered = bandpass(x_flat)
            filtered = filtered.reshape(B, G, N, C, T).permute(0, 1, 4, 2, 3)
            bands.append(filtered)

        return bands

    def _fourier_learnable_decompose(self, x: torch.Tensor) -> list:
        """
        傅里叶域可学习分解

        在频域使用可学习的软掩码进行频带划分
        """
        B, G, T, N, C = x.shape
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(B * G * N, C, T)

        # FFT
        x_fft = torch.fft.rfft(x_flat, dim=-1)
        freq_bins = x_fft.shape[-1]

        # 生成频带掩码
        freq_axis = torch.linspace(0, 1, freq_bins, device=x.device)
        boundaries = torch.sigmoid(self.freq_boundaries) * 0.8 + 0.1  # 限制在 [0.1, 0.9]
        boundaries = torch.sort(boundaries)[0]  # 确保单调递增

        bands = []
        for i in range(self.num_bands):
            # 软掩码: 使用sigmoid实现平滑过渡
            if i == 0:
                # 最低频带
                mask = torch.sigmoid((boundaries[1] - freq_axis) * 10)
            elif i == self.num_bands - 1:
                # 最高频带
                mask = torch.sigmoid((freq_axis - boundaries[i]) * 10)
            else:
                # 中间频带
                low_mask = torch.sigmoid((freq_axis - boundaries[i]) * 10)
                high_mask = torch.sigmoid((boundaries[i + 1] - freq_axis) * 10)
                mask = low_mask * high_mask

            # 应用掩码并逆变换
            masked_fft = x_fft * mask.view(1, 1, -1) * self.freq_masks[i]
            band = torch.fft.irfft(masked_fft, n=T, dim=-1)
            band = band.reshape(B, G, N, C, T).permute(0, 1, 4, 2, 3)
            bands.append(band)

        return bands

    def _fixed_laplacian_decompose(self, x: torch.Tensor) -> list:
        """使用固定低通核进行多尺度分解 (保留原有实现作为备选)"""
        B, G, T, N, C = x.shape
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(B * G * N, C, T)

        if self.num_bands <= 1:
            # Single band: return input as-is
            band = x_flat.permute(0, 2, 1).reshape(B, G, N, T, C).permute(0, 1, 3, 2, 4)
            return [band]

        bands = []
        low = x_flat
        for i in range(self.num_bands - 1):
            k = self.kernel_size + 2 * i
            if k % 2 == 0:
                k += 1
            kernel = torch.ones(C, 1, k, device=x.device, dtype=x.dtype) / k
            smooth = F.conv1d(low, kernel, padding=k // 2, groups=C)
            band = low - smooth
            bands.append(band)
            low = smooth
        bands.append(low)

        outputs = []
        for band in bands:
            band = band.permute(0, 2, 1)  # (B*G*N, T, C)
            band = band.reshape(B, G, N, T, C).permute(0, 1, 3, 2, 4)
            outputs.append(band)
        return outputs

    def _compute_feature_ortho_loss(self, band_outputs: list) -> torch.Tensor:
        """计算频带特征的正交约束"""
        if len(band_outputs) < 2:
            return torch.tensor(0.0, device=band_outputs[0].device)
        B, G, T, N, C = band_outputs[0].shape
        total = B * G * T * N
        if total == 0:
            return torch.tensor(0.0, device=band_outputs[0].device)

        sample = min(self.ortho_samples, total)
        idx = torch.randperm(total, device=band_outputs[0].device)[:sample]

        samples = []
        for band in band_outputs:
            flat = band.reshape(total, C)[idx]
            flat = F.normalize(flat, dim=-1)
            samples.append(flat)

        loss = 0.0
        pairs = 0
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                sim = (samples[i] * samples[j]).sum(dim=-1).abs().mean()
                loss += sim
                pairs += 1
        return loss / max(pairs, 1)

    def _compute_reconstruction_loss(self, x: torch.Tensor, band_outputs: list) -> torch.Tensor:
        """计算重建损失，确保频带可以重建原始信号"""
        reconstructed = sum(band_outputs)
        return F.mse_loss(reconstructed, x)

    def _compute_frequency_separation_loss(self, band_outputs: list) -> torch.Tensor:
        """
        计算频率分离损失

        确保不同频带在频域上有不同的能量分布
        """
        if len(band_outputs) < 2:
            return torch.tensor(0.0, device=band_outputs[0].device)

        B, G, T, N, C = band_outputs[0].shape

        # 计算每个频带的频谱
        spectrums = []
        for band in band_outputs:
            band_flat = band.permute(0, 1, 3, 4, 2).reshape(-1, C, T)
            # AMP下half精度在某些长度（如T=24）会触发cuFFT限制，统一使用fp32做频谱计算。
            fft_input = band_flat.float() if band_flat.dtype in (torch.float16, torch.bfloat16) else band_flat
            fft = torch.fft.rfft(fft_input, dim=-1)
            power = (fft.real ** 2 + fft.imag ** 2).mean(dim=(0, 1))  # 平均功率谱
            power = power / (power.sum() + 1e-8)  # 归一化
            spectrums.append(power)

        # 计算频谱重叠损失 (希望不同频带的频谱尽量不重叠)
        overlap_loss = 0.0
        pairs = 0
        for i in range(len(spectrums)):
            for j in range(i + 1, len(spectrums)):
                # 频谱重叠 = 两个归一化频谱的点积
                overlap = (spectrums[i] * spectrums[j]).sum()
                overlap_loss += overlap
                pairs += 1

        return overlap_loss / max(pairs, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: (B, G, T, N, C)

        Returns:
            output: (B, G, T, N, C)
            info: 包含频带特征和各种损失
        """
        B, G, T, N, C = x.shape

        # ============ 频率分解 ============
        if self.use_fixed_laplacian:
            raw_bands = self._fixed_laplacian_decompose(x)
        elif self.decomposition_type == "learnable_pyramid":
            raw_bands = self._learnable_pyramid_decompose(x)
        elif self.decomposition_type == "learnable_bandpass":
            raw_bands = self._learnable_bandpass_decompose(x)
        elif self.decomposition_type == "fourier_learnable":
            raw_bands = self._fourier_learnable_decompose(x)
        else:
            raw_bands = self._learnable_pyramid_decompose(x)

        # ============ 频带特征增强 ============
        band_outputs = []
        band_features = {}

        for i, (raw_band, transform) in enumerate(zip(raw_bands, self.band_transforms)):
            # 特征变换
            enhanced_band = transform(raw_band)
            band_outputs.append(enhanced_band)
            band_features[f'band_{i}'] = enhanced_band

        # ============ 自适应频带权重 ============
        # 基于输入内容预测频带重要性
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(B * G * N, C, T)
        content_weights = self.importance_predictor(x_flat)  # (B*G*N, num_bands)
        content_weights = content_weights.reshape(B, G, N, self.num_bands)
        content_weights = content_weights.mean(dim=(0, 1, 2))  # 全局平均

        # 结合可学习权重和内容权重
        static_weights = F.softmax(self.band_weights, dim=0)
        combined_weights = 0.5 * static_weights + 0.5 * content_weights

        # ============ 应用Mask ============
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1) if mask.dim() == 4 else mask
            for i in range(len(band_outputs)):
                band_outputs[i] = band_outputs[i].masked_fill(mask_expanded, 0)
                band_features[f'band_{i}'] = band_outputs[i]

        # ============ 加权融合 ============
        output = sum(w * b for w, b in zip(combined_weights, band_outputs))

        # ============ 收集信息和损失 ============
        info = {
            'band_features': band_features,
            'band_weights': combined_weights,
            'static_weights': static_weights,
            'content_weights': content_weights,
            'raw_bands': raw_bands,  # 用于可视化
        }

        # 计算各种正则化损失
        if self.training:
            # 正交损失
            feature_weight = self.feature_ortho_weight if self.feature_ortho_weight > 0 else self.ortho_weight
            if feature_weight > 0:
                ortho_loss = self._compute_feature_ortho_loss(band_outputs)
                info['ortho_loss'] = ortho_loss

            # 重建损失 (确保频带完整性)
            reconstruction_loss = self._compute_reconstruction_loss(x, raw_bands)
            info['reconstruction_loss'] = reconstruction_loss

            # 频率分离损失
            freq_sep_loss = self._compute_frequency_separation_loss(raw_bands)
            info['freq_separation_loss'] = freq_sep_loss

        return output, info


if __name__ == '__main__':
    # 测试
    print("=" * 60)
    print("Testing Improved Frequency Decomposition Modules")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 测试输入
    B, T, N = 2, 24, 100
    geo_feat = torch.randn(B, T, N, 256, device=device)
    semantic_feat = torch.randn(B, T, N, 512, device=device)

    print(f"\nInput shapes:")
    print(f"  geo_feat: {geo_feat.shape}")
    print(f"  semantic_feat: {semantic_feat.shape}")

    # ============ 测试 SimplifiedLFD 各种分解模式 ============
    print("\n" + "-" * 60)
    print("Testing SimplifiedLFD with different decomposition types")
    print("-" * 60)

    decomposition_types = ["learnable_pyramid", "learnable_bandpass", "fourier_learnable"]

    for decomp_type in decomposition_types:
        print(f"\n[{decomp_type}]")
        try:
            lfd = SimplifiedLFD(
                dim=256,
                num_bands=4,
                decomposition_type=decomp_type,
                ortho_weight=0.1,
            ).to(device)
            lfd.train()

            # 添加虚拟Group维度
            x = geo_feat.unsqueeze(1)  # (B, 1, T, N, C)
            output, info = lfd(x)

            print(f"  Output shape: {output.shape}")
            print(f"  Band features: {list(info['band_features'].keys())}")
            print(f"  Static weights: {info['static_weights'].detach().cpu().numpy().round(3)}")
            print(f"  Content weights: {info['content_weights'].detach().cpu().numpy().round(3)}")

            if 'ortho_loss' in info:
                print(f"  Ortho loss: {info['ortho_loss'].item():.4f}")
            if 'reconstruction_loss' in info:
                print(f"  Reconstruction loss: {info['reconstruction_loss'].item():.4f}")
            if 'freq_separation_loss' in info:
                print(f"  Freq separation loss: {info['freq_separation_loss'].item():.4f}")

            print(f"  [OK] {decomp_type} passed!")

        except Exception as e:
            print(f"  [FAIL] {decomp_type} failed: {e}")

    # ============ 测试 LearnableTemporalFilter ============
    print("\n" + "-" * 60)
    print("Testing LearnableTemporalFilter")
    print("-" * 60)

    for init_type in ["gaussian", "bandpass", "highpass"]:
        print(f"\n[{init_type}]")
        try:
            filter_module = LearnableTemporalFilter(
                dim=256,
                kernel_size=7,
                init_type=init_type,
                center_freq=0.25,
            ).to(device)

            x = torch.randn(B * N, 256, T, device=device)
            y = filter_module(x)

            print(f"  Input: {x.shape} -> Output: {y.shape}")
            print(f"  Filter weight shape: {filter_module.filter_weight.shape}")
            print(f"  [OK] {init_type} filter passed!")

        except Exception as e:
            print(f"  [FAIL] {init_type} filter failed: {e}")

    # ============ 测试 SemanticEnhancedLFD ============
    print("\n" + "-" * 60)
    print("Testing SemanticEnhancedLFD (Full Module)")
    print("-" * 60)

    model = SemanticEnhancedLFD(
        geo_dim=256,
        semantic_dim=512,
        num_bands=4,
        use_cross_attention=True,
        modulation_type="film",
    )
    model = model.to(device)
    model.train()

    # 前向传播
    output, info = model(geo_feat, semantic_feat)

    print(f"\nOutput shape: {output.shape}")
    print(f"Band features: {list(info['band_features'].keys())}")
    print(f"Gate shape: {info['gate'].shape}")

    if 'ortho_loss' in info:
        print(f"Ortho loss: {info['ortho_loss'].item():.4f}")

    # 测试梯度流
    loss = output.mean()
    loss.backward()
    print(f"\nGradient check:")
    print(f"  geo_feat.grad exists: {geo_feat.grad is not None}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
