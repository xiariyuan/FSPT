"""
FSPT: Frequency-Semantic Point Tracker

完整的点追踪模型，结合:
1. 骨干网络特征提取
2. CLIP语义编码
3. 语义增强频率分解
4. 频率感知遮挡预测
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, Optional, Tuple, Union, List
import logging

from .semantic_encoder import SemanticEncoder, CLIP_AVAILABLE
from .freq_semantic_fusion import SemanticEnhancedLFD
from .occlusion_predictor import FrequencyAwareOcclusionPredictor

logger = logging.getLogger(__name__)


def _checkpoint_wrapper(func, use_checkpoint: bool, *args, **kwargs):
    """梯度检查点包装器"""
    if use_checkpoint and torch.is_grad_enabled():
        # checkpoint不支持kwargs，需要转换
        return checkpoint(func, *args, use_reentrant=False)
    return func(*args, **kwargs)


def _has_valid_band_features(band_features: Optional[Dict[str, torch.Tensor]]) -> bool:
    if not band_features:
        return False
    for value in band_features.values():
        if value is not None:
            return True
    return False


def _normalized_yx_to_grid_xy(points_yx: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Convert normalized [y, x] coordinates to grid_sample [x, y] coordinates in [-1, 1].

    Dataset points are normalized with divisors (H, W). We recover pixel coordinates
    first and then map to an align_corners=True grid to avoid sampling bias.
    """
    h = max(int(height), 1)
    w = max(int(width), 1)

    y_px = (points_yx[..., 0] * float(h)).clamp(0.0, float(h - 1))
    x_px = (points_yx[..., 1] * float(w)).clamp(0.0, float(w - 1))

    if h > 1:
        y_grid = (y_px / float(h - 1)) * 2.0 - 1.0
    else:
        y_grid = torch.zeros_like(y_px)
    if w > 1:
        x_grid = (x_px / float(w - 1)) * 2.0 - 1.0
    else:
        x_grid = torch.zeros_like(x_px)

    return torch.stack([x_grid, y_grid], dim=-1)


class GeometricBackbone(nn.Module):
    """
    几何特征提取骨干网络
    
    支持多尺度特征融合
    """
    
    def __init__(
        self,
        backbone_type: str = 'resnet50',
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        freeze_stages: Optional[List[str]] = None,
        freeze_bn: bool = False,
        output_dim: int = 256,
        use_multiscale: bool = True,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.freeze_bn = freeze_bn
        self.use_multiscale = use_multiscale
        self.pretrained = bool(pretrained)
        self.pretrained_path = str(pretrained_path) if pretrained_path else None
        self.freeze_stages = [str(s).lower().strip() for s in (freeze_stages or []) if str(s).strip()]

        # ImageNet normalization for timm pretrained backbones.
        # Many timm weights assume inputs are in [0,1] and then normalized by mean/std.
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        
        # 使用timm加载预训练模型（支持本地权重路径）
        # Detect DINOv2-style ViT backbones (all feature levels share the same
        # channel count, so FPN multi-scale fusion is not meaningful).
        self._is_dinov2 = "dinov2" in backbone_type.lower()
        try:
            import timm
            use_hub_pretrained = self.pretrained and not self.pretrained_path
            if self._is_dinov2:
                # DINOv2 ViT outputs uniform-dim features at all levels.
                # Use a single output level and project to output_dim.
                self.backbone = timm.create_model(
                    backbone_type,
                    pretrained=use_hub_pretrained,
                    features_only=True,
                    out_indices=[3],  # last block only
                    img_size=256,
                )
            else:
                self.backbone = timm.create_model(
                    backbone_type,
                    pretrained=use_hub_pretrained,
                    features_only=True,
                    out_indices=[1, 2, 3, 4],  # 多尺度特征
                )

            if self.pretrained and self.pretrained_path:
                if not os.path.isfile(self.pretrained_path):
                    raise FileNotFoundError(
                        f"Backbone pretrained_path not found: {self.pretrained_path}"
                    )

                checkpoint = torch.load(
                    self.pretrained_path, map_location='cpu', weights_only=False
                )
                if isinstance(checkpoint, dict):
                    if isinstance(checkpoint.get('state_dict', None), dict):
                        checkpoint = checkpoint['state_dict']
                    elif isinstance(checkpoint.get('model', None), dict):
                        checkpoint = checkpoint['model']

                if not isinstance(checkpoint, dict):
                    raise TypeError(
                        f"Unsupported checkpoint format at {self.pretrained_path}: {type(checkpoint)}"
                    )

                cleaned = {}
                for key, value in checkpoint.items():
                    k = str(key)
                    for prefix in ("module.", "model.", "backbone."):
                        if k.startswith(prefix):
                            k = k[len(prefix):]
                    cleaned[k] = value

                # DINOv2 Meta checkpoints have no "model." prefix but timm wraps
                # them under model.*  Detect and apply the prefix if needed.
                if self._is_dinov2:
                    model_keys = set(self.backbone.state_dict().keys())
                    if model_keys and not any(k.startswith("model.") for k in cleaned):
                        cleaned = {"model." + k: v for k, v in cleaned.items()}
                    # Interpolate positional embedding if input size differs.
                    pos_key = "model.pos_embed"
                    if pos_key in cleaned and pos_key in self.backbone.state_dict():
                        target_shape = self.backbone.state_dict()[pos_key].shape
                        if cleaned[pos_key].shape != target_shape:
                            old = cleaned[pos_key]
                            cls_tok = old[:, :1]
                            spatial = old[:, 1:]
                            old_grid = int(spatial.shape[1] ** 0.5)
                            new_grid = int(target_shape[1] - 1) ** 0.5
                            new_grid = int(new_grid)
                            dim = spatial.shape[-1]
                            spatial = spatial.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
                            spatial = F.interpolate(spatial, size=(new_grid, new_grid), mode="bilinear", align_corners=False)
                            spatial = spatial.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, dim)
                            cleaned[pos_key] = torch.cat([cls_tok, spatial], dim=1)

                missing, unexpected = self.backbone.load_state_dict(cleaned, strict=False)
                logger.info(
                    f"Loaded local backbone weights from {self.pretrained_path} "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})"
                )

            self.backbone_channels = self.backbone.feature_info.channels()
            logger.info(f"Loaded {backbone_type} backbone with channels: {self.backbone_channels}")
        except ImportError:
            logger.warning("timm not available, using simplified backbone")
            self.backbone = None
            self.backbone_channels = [64, 128, 256, 512]

        if self.backbone is not None and self.freeze_stages:
            self._freeze_backbone_stages(self.backbone, self.freeze_stages)
        
        # 特征投影
        if self.backbone is not None:
            if self._is_dinov2:
                # DINOv2 outputs uniform-dim features; use single projection.
                self.feature_proj = nn.Conv2d(
                    self.backbone_channels[-1],
                    output_dim,
                    kernel_size=1,
                )
            elif use_multiscale and len(self.backbone_channels) >= 3:
                # 多尺度特征融合：FPN风格
                self.lateral_convs = nn.ModuleList([
                    nn.Conv2d(ch, output_dim, kernel_size=1)
                    for ch in self.backbone_channels[-3:]  # 使用最后3个尺度
                ])
                self.fpn_convs = nn.ModuleList([
                    nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
                    for _ in range(3)
                ])
                self.fusion_conv = nn.Conv2d(output_dim * 3, output_dim, kernel_size=1)
                logger.info("Multi-scale feature fusion enabled (FPN-style)")
            else:
                self.feature_proj = nn.Conv2d(
                    self.backbone_channels[-1],
                    output_dim,
                    kernel_size=1,
                )
        else:
            # 简化骨干
            self.simple_backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, output_dim, 3, padding=1),
            )

        if self.freeze_bn:
            backbone = self.backbone if self.backbone is not None else self.simple_backbone
            self._freeze_batch_norm(backbone)

    @staticmethod
    def _freeze_batch_norm(module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    @staticmethod
    def _param_in_stage(param_name: str, stage: str) -> bool:
        if stage == "all":
            return True
        if stage == "stem":
            return param_name.startswith("conv1.") or param_name.startswith("bn1.")
        if stage in {"layer1", "layer2", "layer3", "layer4"}:
            return param_name.startswith(f"{stage}.")
        return False

    def _freeze_backbone_stages(self, module: nn.Module, stages: List[str]) -> None:
        stage_set = {s.lower().strip() for s in stages if s}
        if not stage_set:
            return
        frozen_params = 0
        for name, param in module.named_parameters():
            if any(self._param_in_stage(name, stage) for stage in stage_set):
                param.requires_grad = False
                frozen_params += param.numel()
        logger.info(
            f"Frozen backbone stages: {sorted(stage_set)} (params={frozen_params})"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_bn:
            backbone = self.backbone if self.backbone is not None else self.simple_backbone
            self._freeze_batch_norm(backbone)
        return self
        
    def forward(self, images: torch.Tensor, return_pyramid: bool = False):
        """
        Args:
            images: (B, C, H, W) 或 (B, T, C, H, W)
            
        Returns:
            features: (B, H', W', output_dim) 或 (B, T, H', W', output_dim)
        """
        has_temporal = images.dim() == 5
        
        if has_temporal:
            B, T, C, H, W = images.shape
            images = images.reshape(B * T, C, H, W)

        # Normalize inputs for pretrained timm backbones.
        if self.backbone is not None and self.pretrained:
            images = images.float()
            if images.numel() > 0 and float(images.max()) > 1.5:
                # Likely [0, 255] range; normalize to [0, 1]
                images = images / 255.0
            elif images.numel() > 0 and float(images.min()) < 0.0:
                # Already normalized with mean/std, skip range normalization
                pass
            mean = self.imagenet_mean.to(device=images.device, dtype=images.dtype)
            std = self.imagenet_std.to(device=images.device, dtype=images.dtype)
            images = (images - mean) / std
        
        pyramid = None
        if self.backbone is not None:
            multi_scale_features = self.backbone(images)  # list of features

            if return_pyramid:
                if self._is_dinov2:
                    # DINOv2: single level, map to "c2" for downstream compatibility
                    feat = multi_scale_features[-1].permute(0, 2, 3, 1).contiguous()
                    if has_temporal:
                        h_out, w_out, c_out = feat.shape[1:]
                        feat = feat.reshape(B, T, h_out, w_out, c_out)
                    pyramid = {"c2": feat}
                else:
                    # timm `features_only=True` with `out_indices=[1,2,3,4]` yields:
                    #   c2=layer1, c3=layer2, c4=layer3, c5=layer4
                    names = ("c2", "c3", "c4", "c5")
                    pyramid = {}
                    for name, feat in zip(names, multi_scale_features):
                        feat = feat.permute(0, 2, 3, 1).contiguous()  # (B*T, H, W, C)
                        if has_temporal:
                            h_out, w_out, c_out = feat.shape[1:]
                            feat = feat.reshape(B, T, h_out, w_out, c_out)
                        pyramid[name] = feat

            if self._is_dinov2:
                # DINOv2: single-level projection (no FPN needed)
                feat = multi_scale_features[-1]  # (B*T, C, H, W)
                features = self.feature_proj(feat)
            elif self.use_multiscale and hasattr(self, 'lateral_convs'):
                # FPN风格多尺度融合
                # 取最后3个尺度
                feats = multi_scale_features[-3:]
                
                # 横向连接
                laterals = [conv(f) for conv, f in zip(self.lateral_convs, feats)]
                
                # 自顶向下融合
                for i in range(len(laterals) - 1, 0, -1):
                    h, w = laterals[i - 1].shape[-2:]
                    laterals[i - 1] = laterals[i - 1] + F.interpolate(
                        laterals[i], size=(h, w), mode='bilinear', align_corners=False
                    )
                
                # FPN卷积
                fpn_feats = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
                
                # 上采样到相同尺度并拼接
                target_size = fpn_feats[0].shape[-2:]
                aligned_feats = [
                    F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
                    if f.shape[-2:] != target_size else f
                    for f in fpn_feats
                ]
                
                # 融合
                features = self.fusion_conv(torch.cat(aligned_feats, dim=1))
            else:
                features = multi_scale_features[-1]  # 最后一个尺度
                features = self.feature_proj(features)
        else:
            features = self.simple_backbone(images)
        
        # 转换为 (B, H', W', C) 格式
        features = features.permute(0, 2, 3, 1)
        
        if has_temporal:
            _, H_out, W_out, C = features.shape
            features = features.reshape(B, T, H_out, W_out, C)
        
        if not return_pyramid:
            return features
        return features, (pyramid or {})


class TemporalTransformer(nn.Module):
    """
    时序Transformer
    
    对点特征进行时序建模
    支持FlashAttention加速（PyTorch 2.0+）
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_window_size: Optional[int] = None,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        
        self.dim = dim
        self.max_window_size = int(max_window_size) if max_window_size else None
        
        # 检测FlashAttention可用性
        self.use_flash_attention = use_flash_attention and hasattr(F, 'scaled_dot_product_attention')
        if self.use_flash_attention:
            logger.info("FlashAttention enabled for TemporalTransformer")
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.zeros(1, 1000, dim))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        
        # Transformer编码器
        # PyTorch 2.0+ 会自动使用FlashAttention（如果启用了SDPA）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 启用PyTorch 2.0的编译优化（如果可用）
        if self.use_flash_attention and torch.cuda.is_available():
            try:
                # 设置默认使用FlashAttention
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            except AttributeError:
                # PyTorch版本不支持这些设置
                pass
        
        self.norm = nn.LayerNorm(dim)

    def _get_pos_encoding(self, length: int) -> torch.Tensor:
        """获取适配长度的相对位置编码"""
        max_len = self.pos_encoding.shape[1]
        if length <= max_len:
            return self.pos_encoding[:, :length, :]
        # 线性插值扩展到更长序列
        pos = self.pos_encoding.permute(0, 2, 1)  # (1, C, L)
        pos = F.interpolate(pos, size=length, mode='linear', align_corners=False)
        return pos.permute(0, 2, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, C)
            
        Returns:
            output: (B, T, N, C)
        """
        B, T, N, C = x.shape
        
        # 每个点独立处理时序
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, C)  # (B*N, T, C)
        
        # 添加位置编码
        if self.max_window_size is None or T <= self.max_window_size:
            pos = self._get_pos_encoding(T)
            x_flat = x_flat + pos
            output = self.transformer(x_flat)  # (B*N, T, C)
        else:
            pos_full = self._get_pos_encoding(T)
            output = x_flat.new_zeros(x_flat.shape)
            counts = x_flat.new_zeros(B * N, T, 1)
            stride = max(1, self.max_window_size // 2)
            start = 0
            while start < T:
                end = min(start + self.max_window_size, T)
                chunk = x_flat[:, start:end, :] + pos_full[:, start:end, :]
                chunk_out = self.transformer(chunk)
                output[:, start:end, :] += chunk_out
                counts[:, start:end, :] += 1
                if end == T:
                    break
                start += stride
            output = output / counts
        
        output = self.norm(output)
        
        # 恢复形状
        output = output.reshape(B, N, T, C).permute(0, 2, 1, 3)  # (B, T, N, C)
        
        return output


class PositionDecoder(nn.Module):
    """
    位置解码器
    
    预测点的位置更新
    """
    
    def __init__(
        self,
        dim: int = 256,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.position_head = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),  # (dy, dx)
        )
        
        self.visibility_head = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        features: torch.Tensor,
        init_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, T, N, C)
            init_positions: (B, N, 2) 初始位置
            
        Returns:
            positions: (B, N, T, 2) 预测轨迹
            visibility: (B, N, T) 可见性
        """
        B, T, N, C = features.shape
        
        # 预测位置增量（直接回归每帧相对于初始位置的偏移）
        position_delta = self.position_head(features)  # (B, T, N, 2)

        # 初始位置 + 每帧偏移
        positions = init_positions.unsqueeze(2) + position_delta.permute(0, 2, 1, 3)  # (B, N, T, 2)
        positions = torch.clamp(positions, 0, 1)
        
        # 预测可见性
        visibility = self.visibility_head(features).squeeze(-1)  # (B, T, N)
        visibility = visibility.permute(0, 2, 1)  # (B, N, T)
        
        return positions, visibility


class FSPTTracker(nn.Module):
    """
    FSPT: Frequency-Semantic Point Tracker
    
    完整的点追踪模型
    
    Args:
        config: 模型配置
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # 几何骨干
        self.geo_backbone = GeometricBackbone(
            backbone_type=config.backbone.type,
            pretrained=config.backbone.pretrained,
            pretrained_path=getattr(config.backbone, 'pretrained_path', None),
            freeze_stages=getattr(config.backbone, 'freeze_stages', None),
            freeze_bn=getattr(config.backbone, 'freeze_bn', False),
            output_dim=config.temporal.dim,
            use_multiscale=getattr(config.backbone, 'use_multiscale', True),
        )
        
        # 语义编码器
        if CLIP_AVAILABLE and config.semantic.enabled:
            self.semantic_encoder = SemanticEncoder(
                clip_model=config.clip.model,
                output_dim=config.temporal.dim,
                freeze=config.clip.freeze,
                cache_dir=getattr(config.clip, 'cache_dir', None),
            )
            self.use_semantic = True
        else:
            self.semantic_encoder = None
            self.use_semantic = False
            logger.warning("Semantic encoder disabled (CLIP not available or disabled in config)")
        self._semantic_spatial_available = True
        self._warned_semantic_fallback = False

        # Optional CLIP feature cache (recommended for evaluation-only on server)
        self.semantic_cache = None
        self.semantic_cache_only_eval = True
        self.semantic_cache_mode = "spatial"
        semantic_cfg = getattr(config, "semantic", None)
        cache_cfg = getattr(semantic_cfg, "cache", None) if semantic_cfg is not None else None
        if self.use_semantic and cache_cfg is not None and bool(getattr(cache_cfg, "enabled", False)):
            try:
                from utils.tensor_cache import DiskTensorCache

                root_dir = getattr(cache_cfg, "root_dir", "outputs/semantic_cache")
                write = bool(getattr(cache_cfg, "write", True))
                self.semantic_cache_only_eval = bool(getattr(cache_cfg, "only_eval", True))
                self.semantic_cache_mode = str(getattr(cache_cfg, "mode", "spatial")).lower().strip()
                keep_in_memory_cfg = getattr(cache_cfg, "keep_in_memory", None)
                if keep_in_memory_cfg is None:
                    keep_in_memory = self.semantic_cache_mode != "spatial"
                else:
                    keep_in_memory = bool(keep_in_memory_cfg)
                self.semantic_cache = DiskTensorCache(
                    root_dir=str(root_dir),
                    enabled=True,
                    write=write,
                    keep_in_memory=keep_in_memory,
                )
            except Exception as exc:
                logger.warning(f"Semantic cache disabled (init failed): {exc}")
        
        # 语义增强频率分解
        if config.frequency.enabled:
            fusion_type = getattr(config.semantic, 'fusion_type', 'cross_attention')
            if not self.use_semantic:
                fusion_type = 'none'
            modulation_type = getattr(config.semantic, 'modulation_type', 'multiplicative')
            if not self.use_semantic:
                modulation_type = 'none'
            # 注意：semantic_dim应该是SemanticEncoder的输出维度(temporal.dim)，而不是CLIP原始维度(clip.dim)
            # 因为SemanticEncoder内部会将CLIP特征投影到output_dim=temporal.dim
            self.freq_semantic = SemanticEnhancedLFD(
                geo_dim=config.temporal.dim,
                semantic_dim=config.temporal.dim,  # 使用投影后的维度
                num_bands=config.frequency.num_bands,
                kernel_size=getattr(config.frequency, 'kernel_size', 7),
                fusion_type=fusion_type,
                modulation_type=modulation_type,
                use_fixed_laplacian=getattr(config.frequency, 'use_fixed_laplacian', False),
                ortho_weight=getattr(config.frequency, 'ortho_weight', 0.0),
                feature_ortho_weight=getattr(config.frequency, 'feature_ortho_weight', 0.0),
                strict_lfd=bool(getattr(config.frequency, 'strict_lfd', False)),
            )
            self.use_frequency = True
        else:
            self.freq_semantic = None
            self.use_frequency = False
        
        # 时序Transformer
        self.temporal_transformer = TemporalTransformer(
            dim=config.temporal.dim,
            num_layers=config.temporal.num_layers,
            num_heads=config.temporal.num_heads,
            dropout=config.temporal.dropout,
            max_window_size=getattr(config.temporal, 'max_window_size', None),
            use_flash_attention=getattr(config.temporal, 'use_flash_attention', True),
        )
        
        # 遮挡预测器
        occlusion_enabled = bool(config.occlusion.enabled)
        if occlusion_enabled:
            # 同样，semantic_dim使用投影后的维度(temporal.dim)
            self.occlusion_predictor = FrequencyAwareOcclusionPredictor(
                dim=config.temporal.dim,
                semantic_dim=config.temporal.dim,  # 使用投影后的维度
                num_bands=config.frequency.num_bands,
                use_semantic_consistency=bool(config.occlusion.use_semantic_propagation and self.use_semantic),
                use_low_freq_prediction=getattr(config.occlusion, 'use_low_freq_prediction', True),
            )
            self.use_occlusion = True
        else:
            self.occlusion_predictor = None
            self.use_occlusion = False
        # Track fusion policy for occlusion branch:
        # - train: default off to avoid trajectory predictor interfering with position supervision
        # - eval: default on to keep occlusion-aware refinement behavior
        self.occlusion_track_fusion_train = bool(
            getattr(config.occlusion, 'track_fusion_train', False)
        )
        self.occlusion_track_fusion_eval = bool(
            getattr(config.occlusion, 'track_fusion_eval', True)
        )
        self._warned_missing_band_features = False
        
        # 位置解码器
        self.position_decoder = PositionDecoder(
            dim=config.temporal.dim,
        )
        
        # 迭代精化配置
        self.num_refinement_iters = getattr(config, 'num_refinement_iters', 
                                            getattr(config.temporal, 'num_refinement_iters', 1))
        if self.num_refinement_iters > 1:
            # 精化模块：基于当前轨迹位置的特征更新位置
            self.refinement_encoder = nn.Sequential(
                nn.Linear(config.temporal.dim + 2, config.temporal.dim),
                nn.GELU(),
                nn.Linear(config.temporal.dim, config.temporal.dim),
            )
            self.refinement_decoder = nn.Sequential(
                nn.Linear(config.temporal.dim, config.temporal.dim // 2),
                nn.GELU(),
                nn.Linear(config.temporal.dim // 2, 2),  # 位置增量
            )
            logger.info(f"Iterative refinement enabled with {self.num_refinement_iters} iterations")
        
        # 梯度检查点配置（节省显存，适用于长视频）
        self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing',
                                                   getattr(config.temporal, 'use_gradient_checkpointing', False))
        if self.use_gradient_checkpointing:
            logger.info("Gradient checkpointing enabled for memory optimization")
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for name, m in self.named_modules():
            if not name:
                continue
            # Do not destroy pretrained weights.
            if name.startswith("geo_backbone.backbone."):
                continue
            if name.startswith("semantic_encoder.clip_model."):
                continue

            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _sample_features(
        self,
        features: torch.Tensor,
        points: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        在指定点位置采样特征
        
        Args:
            features: (B, T, H, W, C) 空间特征
            points: (B, N, 2) 或 (B, T, N, 2) 点坐标 [y, x]，归一化到[0, 1]
            
        Returns:
            point_features: (B, T, N, C)
        """
        B, T, H, W, C = features.shape
        
        # 重排特征为 (B*T, C, H, W)
        features_flat = features.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
        
        # 处理静态或动态点坐标
        if points.dim() == 3:
            # (B, N, 2) -> 扩展到所有帧
            N = points.shape[1]
            points_bt = points.unsqueeze(1).expand(B, T, N, 2)  # (B, T, N, 2)
        else:
            # (B, T, N, 2) -> 每帧使用对应位置
            N = points.shape[2]
            points_bt = points
        
        grid = _normalized_yx_to_grid_xy(points_bt, H, W)
        grid = grid.reshape(B * T, N, 1, 2)  # (B*T, N, 1, 2)
        
        # 采样
        sampled = F.grid_sample(features_flat, grid, mode='bilinear', align_corners=True)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (B*T, N, C)
        
        point_features = sampled.reshape(B, T, N, C)
        
        return point_features

    def _align_tracks_to_query(
        self,
        tracks: torch.Tensor,
        init_positions: torch.Tensor,
        query_t: torch.Tensor,
    ) -> torch.Tensor:
        """对齐查询帧，确保轨迹经过给定查询点。"""
        B, N, T, _ = tracks.shape
        delta = tracks - init_positions.unsqueeze(2)  # (B, N, T, 2)
        batch_idx = torch.arange(B, device=tracks.device).view(B, 1).expand(B, N)
        point_idx = torch.arange(N, device=tracks.device).view(1, N).expand(B, N)
        delta_at_query = delta[batch_idx, point_idx, query_t]  # (B, N, 2)
        delta = delta - delta_at_query.unsqueeze(2)
        aligned = init_positions.unsqueeze(2) + delta
        return torch.clamp(aligned, 0, 1)
    
    def forward(
        self,
        video: torch.Tensor,
        query_points: torch.Tensor,
        meta: Optional[Dict] = None,
        return_info: bool = False,
        return_iter_tracks: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """
        前向传播
        
        Args:
            video: (B, T, 3, H, W) 视频帧
            query_points: (B, N, 3) 查询点 [t, y, x]，t为帧索引[0, T-1]，y/x归一化到[0, 1]
            return_info: 是否返回额外信息（频率/语义/置信度等）
            return_iter_tracks: 是否返回迭代精化过程中的中间轨迹
            
        Returns:
            tracks: (B, N, T, 2) 预测轨迹 [y, x]
            visibility: (B, N, T) 可见性概率
            info (可选): 额外信息字典，可能包含
                - freq_info: 频率分解信息
                - semantic_features: 语义特征（采样在点上）
                - iter_tracks: 迭代精化的中间轨迹列表
                - confidence: 置信度 (B, N, T)
        """
        B, T, C, H, W = video.shape
        N = query_points.shape[1]

        video_names = None
        if isinstance(meta, dict):
            raw_name = meta.get("video_name", None)
            if raw_name is not None:
                if isinstance(raw_name, (list, tuple)) and len(raw_name) == B:
                    video_names = [str(v) for v in raw_name]
                elif isinstance(raw_name, torch.Tensor):
                    try:
                        if raw_name.ndim == 0 and B == 1:
                            video_names = [str(raw_name.item())]
                        elif raw_name.ndim == 1 and int(raw_name.shape[0]) == B:
                            video_names = [str(v.item()) for v in raw_name]
                    except Exception:
                        video_names = None
                elif B == 1:
                    video_names = [str(raw_name)]

        use_semantic_cache = (
            self.semantic_cache is not None
            and video_names is not None
            and (not self.training or not self.semantic_cache_only_eval)
        )
        semantic_cache = self.semantic_cache if use_semantic_cache else None
        semantic_cache_key = video_names if use_semantic_cache else None
        semantic_cache_mode = self.semantic_cache_mode if use_semantic_cache else None
        
        # 1. 几何特征提取
        geo_features = self.geo_backbone(video)  # (B, T, H', W', dim)
        
        # 2. 语义特征提取
        if self.use_semantic:
            semantic_features = None
            if semantic_cache_mode != "global" and self._semantic_spatial_available:
                try:
                    semantic_features = self.semantic_encoder(
                        video,
                        mode='spatial',
                        cache=semantic_cache,
                        cache_key=semantic_cache_key,
                    )  # (B, T, H', W', dim)
                except Exception as e:
                    self._semantic_spatial_available = False
                    if not self._warned_semantic_fallback:
                        logger.warning(f"Spatial semantic extraction failed, fallback to global: {e}")
                        self._warned_semantic_fallback = True
            if semantic_features is None:
                global_features = self.semantic_encoder(
                    video,
                    mode='global',
                    cache=semantic_cache,
                    cache_key=semantic_cache_key,
                )  # (B, T, dim)
                Hs, Ws = geo_features.shape[2], geo_features.shape[3]
                semantic_features = global_features[:, :, None, None, :].expand(-1, -1, Hs, Ws, -1)
        else:
            semantic_features = geo_features  # 使用几何特征作为替代
        
        # 3. 在查询点位置采样特征
        init_positions = query_points[:, :, 1:3]  # (B, N, 2) [y, x]
        
        geo_point_features = self._sample_features(geo_features, init_positions)  # (B, T, N, dim)
        
        if self.use_semantic:
            semantic_point_features = self._sample_features(semantic_features, init_positions)  # (B, T, N, dim)
        else:
            semantic_point_features = geo_point_features
        
        # 4. 语义增强频率分解
        if self.use_frequency:
            # 注意：freq_semantic返回dict，checkpoint不支持非Tensor输出
            freq_output, freq_info = self.freq_semantic(
                geo_point_features,
                semantic_point_features,
            )
        else:
            freq_output = geo_point_features
            freq_info = {}
        
        # 5. 时序建模
        if self.use_gradient_checkpointing and self.training:
            # 使用梯度检查点包装时序Transformer
            try:
                temporal_features = checkpoint(
                    self.temporal_transformer, freq_output,
                    use_reentrant=False
                )
            except TypeError:
                # 兼容旧版PyTorch不支持use_reentrant参数
                temporal_features = checkpoint(self.temporal_transformer, freq_output)
        else:
            temporal_features = self.temporal_transformer(freq_output)  # (B, T, N, dim)
        
        # 6. 位置解码
        tracks, visibility = self.position_decoder(temporal_features, init_positions)

        # 6.1 对齐查询帧位置，确保轨迹在查询帧通过给定点
        query_t = query_points[:, :, 0].round().long().clamp(0, T - 1)  # (B, N)
        tracks = self._align_tracks_to_query(tracks, init_positions, query_t)
        
        confidence = None
        band_features = freq_info.get('band_features') if isinstance(freq_info, dict) else None
        if self.use_occlusion and not _has_valid_band_features(band_features):
            # 频率分支关闭或band信息缺失时，回退到时序特征，保证遮挡分支仍可训练/评估。
            band_features = {'band_0': freq_output}
            if isinstance(freq_info, dict):
                freq_info['band_features'] = band_features
            if not self._warned_missing_band_features:
                if self.use_frequency:
                    logger.warning(
                        "Occlusion predictor fallback: missing/empty band_features, using temporal features as band_0."
                    )
                else:
                    logger.info(
                        "Occlusion predictor fallback: frequency disabled, using temporal features as band_0."
                    )
                self._warned_missing_band_features = True

        # 7. 遮挡感知位置修正
        if self.use_occlusion and _has_valid_band_features(band_features):
            # 在当前预测轨迹位置采样语义特征（更准确）
            if self.use_semantic:
                tracks_for_sample = tracks.permute(0, 2, 1, 3)  # (B, T, N, 2)
                semantic_at_tracks = self._sample_features(semantic_features, tracks_for_sample)
            else:
                semantic_at_tracks = semantic_point_features
            
            occ_prob, pred_positions, confidence_pred = self.occlusion_predictor(
                band_features,
                semantic_at_tracks,
                tracks.permute(0, 2, 1, 3),  # (B, T, N, 2)
            )
            confidence = confidence_pred.permute(0, 2, 1)  # (B, N, T)

            # Optional occlusion-guided track fusion.
            # Default policy: disable in training, enable in evaluation.
            apply_track_fusion = (
                self.occlusion_track_fusion_train if self.training else self.occlusion_track_fusion_eval
            )
            if apply_track_fusion:
                occ_mask = occ_prob.unsqueeze(-1)  # (B, T, N, 1)
                pred_positions_transposed = pred_positions.permute(0, 2, 1, 3)  # (B, N, T, 2)

                # 软融合：根据遮挡概率加权
                tracks = (1 - occ_mask.permute(0, 2, 1, 3)) * tracks + \
                         occ_mask.permute(0, 2, 1, 3) * pred_positions_transposed
                tracks = torch.clamp(tracks, 0, 1)

            # 更新可见性
            visibility = 1 - occ_prob.permute(0, 2, 1)  # (B, N, T)
        
        iter_tracks = None
        # 8. 迭代精化
        if self.num_refinement_iters > 1:
            if return_iter_tracks:
                tracks, iter_tracks = self._iterative_refinement(
                    tracks, geo_features, semantic_features if self.use_semantic else geo_features,
                    init_positions, query_t, return_all=True
                )
            else:
                tracks = self._iterative_refinement(
                    tracks, geo_features, semantic_features if self.use_semantic else geo_features,
                    init_positions, query_t
                )
        elif return_iter_tracks:
            iter_tracks = [tracks]

        # 8.1 迭代/遮挡更新后再次对齐查询帧，保证硬约束
        tracks = self._align_tracks_to_query(tracks, init_positions, query_t)
        if return_iter_tracks and iter_tracks is not None:
            iter_tracks = [
                self._align_tracks_to_query(t, init_positions, query_t) for t in iter_tracks
            ]
        
        if return_info:
            info = {'freq_info': freq_info}
            if self.use_semantic:
                info['semantic_features'] = semantic_point_features
            if return_iter_tracks and iter_tracks is not None:
                info['iter_tracks'] = iter_tracks
            if confidence is not None:
                info['confidence'] = confidence
            return tracks, visibility, info
        return tracks, visibility
    
    def _iterative_refinement(
        self,
        tracks: torch.Tensor,
        geo_features: torch.Tensor,
        semantic_features: torch.Tensor,
        init_positions: torch.Tensor,
        query_t: torch.Tensor,
        return_all: bool = False,
    ) -> torch.Tensor:
        """
        迭代精化轨迹预测
        
        Args:
            tracks: (B, N, T, 2) 初始轨迹预测
            geo_features: (B, T, H, W, C) 几何特征
            semantic_features: (B, T, H, W, C) 语义特征
            init_positions: (B, N, 2) 查询点位置
            query_t: (B, N) 查询帧索引
            return_all: 是否返回每次迭代的轨迹列表
            
        Returns:
            refined_tracks: (B, N, T, 2) 精化后的轨迹
            (可选) all_tracks: List[(B, N, T, 2)]，每次迭代的轨迹
        """
        B, N, T, _ = tracks.shape
        
        tracks_all = [tracks] if return_all else None

        for iter_idx in range(1, self.num_refinement_iters):
            # 在当前轨迹位置采样特征
            tracks_for_sample = tracks.permute(0, 2, 1, 3)  # (B, T, N, 2)
            
            # 融合几何和语义特征
            geo_at_tracks = self._sample_features(geo_features, tracks_for_sample)  # (B, T, N, C)
            sem_at_tracks = self._sample_features(semantic_features, tracks_for_sample)  # (B, T, N, C)
            
            # 平均融合
            fused_features = (geo_at_tracks + sem_at_tracks) / 2  # (B, T, N, C)
            
            # 拼接当前位置信息
            tracks_normalized = tracks.permute(0, 2, 1, 3)  # (B, T, N, 2)
            refinement_input = torch.cat([fused_features, tracks_normalized], dim=-1)  # (B, T, N, C+2)
            
            # 编码和解码得到位置增量
            encoded = self.refinement_encoder(refinement_input)  # (B, T, N, C)
            delta = self.refinement_decoder(encoded)  # (B, T, N, 2)
            
            # 更新轨迹
            tracks = tracks + delta.permute(0, 2, 1, 3) * 0.1  # 使用较小的步长
            
            # 确保查询帧位置不变
            batch_idx = torch.arange(B, device=tracks.device).view(B, 1).expand(B, N)
            point_idx = torch.arange(N, device=tracks.device).view(1, N).expand(B, N)
            current_delta = tracks - init_positions.unsqueeze(2)
            delta_at_query = current_delta[batch_idx, point_idx, query_t]
            tracks = init_positions.unsqueeze(2) + (current_delta - delta_at_query.unsqueeze(2))
            tracks = torch.clamp(tracks, 0, 1)
            if return_all:
                tracks_all.append(tracks)
        
        if return_all:
            return tracks, tracks_all
        return tracks


def create_fspt_tracker(config) -> FSPTTracker:
    """
    创建FSPT追踪器
    
    Args:
        config: 配置对象
        
    Returns:
        model: FSPTTracker实例
    """
    return FSPTTracker(config)


if __name__ == '__main__':
    # 测试
    from omegaconf import OmegaConf
    
    print("Testing FSPTTracker...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试配置
    config = OmegaConf.create({
        'backbone': {
            'type': 'resnet18',  # 使用小模型测试
            'pretrained': False,
        },
        'clip': {
            'model': 'ViT-B/16',
            'dim': 512,
            'freeze': True,
        },
        'frequency': {
            'enabled': True,
            'num_bands': 4,
        },
        'semantic': {
            'enabled': False,  # 测试时禁用CLIP
        },
        'temporal': {
            'dim': 256,
            'num_layers': 2,
            'num_heads': 8,
            'dropout': 0.1,
        },
        'occlusion': {
            'enabled': True,
            'use_semantic_propagation': True,
        },
    })
    
    # 创建模型
    model = FSPTTracker(config)
    model = model.to(device)
    model.eval()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试输入
    B, T, H, W = 2, 24, 256, 256
    N = 100
    
    video = torch.randn(B, T, 3, H, W, device=device)
    query_points = torch.rand(B, N, 3, device=device)
    query_points[:, :, 0] = 0  # 查询帧为0
    
    # 前向传播
    with torch.no_grad():
        tracks, visibility = model(video, query_points)
    
    print(f"Input video shape: {video.shape}")
    print(f"Input query_points shape: {query_points.shape}")
    print(f"Output tracks shape: {tracks.shape}")
    print(f"Output visibility shape: {visibility.shape}")
    print(f"Tracks range: [{tracks.min():.3f}, {tracks.max():.3f}]")
    print(f"Visibility range: [{visibility.min():.3f}, {visibility.max():.3f}]")
    
    print("Test passed!")
