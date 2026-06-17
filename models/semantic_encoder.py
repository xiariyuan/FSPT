"""
Semantic Encoder for FSPT

利用CLIP提取语义特征，用于引导点追踪
"""

import os
import importlib.util
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Sequence, List
import logging

logger = logging.getLogger(__name__)

try:
    from utils.tensor_cache import DiskTensorCache
except Exception:
    DiskTensorCache = None  # type: ignore[assignment]


def _detect_clip_backend() -> Optional[str]:
    """
    Detect an available CLIP backend without importing the heavy package at module load.

    Importing `clip` in this environment can trigger protocol initialization issues
    during package import, so we only probe for module availability here and delay
    the actual import until SemanticEncoder construction.
    """
    if importlib.util.find_spec("open_clip") is not None:
        return "open_clip"
    if importlib.util.find_spec("clip") is not None:
        return "openai"
    return None


CLIP_BACKEND = _detect_clip_backend()
CLIP_AVAILABLE = CLIP_BACKEND is not None
if not CLIP_AVAILABLE:
    logger.warning("CLIP backend not available. Install openai/CLIP or open-clip-torch.")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_clip_cache_dir(cache_dir: Optional[str]) -> str:
    """
    Resolve the CLIP cache/download directory.

    Preference order:
    1. Explicit config path
    2. Repo-local weights/clip directory

    Relative paths are resolved against the project root so configs remain
    portable when the working directory changes.
    """
    if cache_dir:
        path = Path(str(cache_dir)).expanduser()
        if not path.is_absolute():
            path = _project_root() / path
        return str(path)
    return str(_project_root() / "weights" / "clip")


def _find_local_clip_checkpoint(clip_model: str, cache_dir: Optional[str]) -> Optional[str]:
    """
    Look for a repo-local CLIP checkpoint before triggering a network download.
    """
    model_name = str(clip_model).replace("/", "-") + ".pt"
    candidates = []

    if cache_dir:
        cache_path = Path(cache_dir).expanduser()
        if not cache_path.is_absolute():
            cache_path = _project_root() / cache_path
        if cache_path.is_file():
            candidates.append(cache_path)
        else:
            candidates.append(cache_path / model_name)

    candidates.append(_project_root() / "weights" / "clip" / model_name)

    direct_path = Path(str(clip_model)).expanduser()
    if direct_path.is_file():
        candidates.insert(0, direct_path)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _import_clip_backend(backend: str):
    if backend == "openai":
        import clip as openai_clip

        return openai_clip
    if backend == "open_clip":
        import open_clip

        return open_clip
    raise ValueError(f"Unsupported CLIP backend: {backend}")


def _normalized_yx_to_grid_xy(points_yx: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert normalized [y, x] coordinates to grid_sample [x, y] in [-1, 1]."""
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


class SemanticEncoder(nn.Module):
    """
    语义编码器
    
    使用冻结的CLIP模型提取语义特征
    """
    
    def __init__(
        self,
        clip_model: str = "ViT-B/16",
        output_dim: int = 256,
        freeze: bool = True,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.freeze = freeze
        self.clip_model_name = str(clip_model)
        self.cache_dir = _resolve_clip_cache_dir(cache_dir)
        self._clip_fp32_train_logged = False
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        if not CLIP_AVAILABLE:
            raise RuntimeError("CLIP is required for SemanticEncoder")

        # 选择设备：由上层训练入口显式决定，避免在构造期额外探测 CUDA 状态。
        resolved_device = torch.device(device)

        # 加载CLIP模型
        # 优先使用仓库内的 JIT archive，但只把它当作权重来源。
        # 最终模型仍然使用 open_clip 的 eager 实现，以避免 TorchScript
        # forward / device handling 的不稳定性。
        clip_backend = "open_clip"
        local_clip_path = _find_local_clip_checkpoint(clip_model, self.cache_dir)
        if local_clip_path is not None:
            try:
                open_clip = _import_clip_backend("open_clip")
                model_name = clip_model.replace("/", "-")
                logger.info(
                    f"Loading CLIP from local TorchScript checkpoint into eager open_clip model: {local_clip_path}"
                )
                self.clip_model = open_clip.create_model(
                    model_name,
                    pretrained=None,
                    device="cpu",
                    cache_dir=self.cache_dir,
                )
                state_dict = torch.jit.load(local_clip_path, map_location="cpu").state_dict()
                missing, unexpected = self.clip_model.load_state_dict(state_dict, strict=False)
                logger.info(
                    f"Loaded local CLIP weights into eager open_clip model "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})"
                )
            except Exception as exc:
                logger.warning(
                    f"Local TorchScript CLIP checkpoint failed to load into eager model ({local_clip_path}): {exc}. "
                    "Falling back to backend loader."
                )
                local_clip_path = None

        if local_clip_path is None:
            clip_backend = CLIP_BACKEND or _detect_clip_backend()
            if clip_backend is None:
                raise RuntimeError("CLIP is required for SemanticEncoder")
            if clip_backend == "open_clip":
                open_clip = _import_clip_backend("open_clip")
                model_name = clip_model.replace("/", "-")
                self.clip_model = open_clip.create_model(
                    model_name,
                    pretrained="openai",
                    device=resolved_device,
                    cache_dir=self.cache_dir,
                )
            else:
                openai_clip = _import_clip_backend("openai")
                load_kwargs = {"device": resolved_device}
                load_kwargs["download_root"] = self.cache_dir
                self.clip_model, _ = openai_clip.load(clip_model, **load_kwargs)

        self.clip_backend = clip_backend

        self.clip_dim = getattr(self.clip_model.visual, 'output_dim', None) \
            or getattr(self.clip_model.visual, 'embed_dim', None) \
            or getattr(self.clip_model.visual, 'num_features', 512)

        # 输入分辨率
        self.input_resolution = None
        visual = self.clip_model.visual
        if hasattr(visual, "input_resolution"):
            self.input_resolution = (visual.input_resolution, visual.input_resolution)
        elif hasattr(visual, "image_size"):
            size = visual.image_size
            self.input_resolution = (size, size) if isinstance(size, int) else tuple(size)

        # CLIP标准归一化参数
        self.register_buffer('pixel_mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]))
        self.register_buffer('pixel_std', torch.tensor([0.26862954, 0.26130258, 0.27577711]))
        
        # 冻结CLIP
        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self.clip_model.eval()
        
        # 投影层：将CLIP特征投影到追踪特征空间
        self.projection = nn.Sequential(
            nn.Linear(self.clip_dim, self.clip_dim),
            nn.GELU(),
            nn.Linear(self.clip_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        input_res = self.input_resolution if self.input_resolution is not None else ("?", "?")
        self._cache_prefix = (
            # v2: cache RAW CLIP features (pre-projection) so caches remain valid even if
            # semantic_encoder.projection is trained/changed across checkpoints.
            f"semantic:v2raw:{self.clip_backend}:{self.clip_model_name}:clip{self.clip_dim}:"
            f"out{output_dim}:in{input_res[0]}x{input_res[1]}"
        )

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """对输入图像进行CLIP标准预处理"""
        images = images.float()
        if images.max() > 1.0:
            images = images / 255.0

        if self.input_resolution is not None:
            target_h, target_w = self.input_resolution
            if images.shape[-2:] != (target_h, target_w):
                images = F.interpolate(
                    images, size=(target_h, target_w), mode="bilinear", align_corners=False
                )

        mean = self.pixel_mean.to(images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = self.pixel_std.to(images.device, dtype=images.dtype).view(1, 3, 1, 1)
        images = (images - mean) / std
        return images

    def train(self, mode: bool = True):
        super().train(mode)
        # 训练模式应由参数实际可训练状态决定；
        # 若训练阶段动态解冻了部分CLIP参数，需允许其进入train模式。
        if mode and self._clip_has_trainable_params():
            self._ensure_clip_trainable_fp32()
            self.clip_model.train()
        else:
            self.clip_model.eval()
        return self

    def _clip_has_trainable_params(self) -> bool:
        for param in self.clip_model.parameters():
            if bool(param.requires_grad):
                return True
        return False

    def _ensure_clip_trainable_fp32(self) -> None:
        if not self._clip_has_trainable_params():
            return
        needs_cast = False
        for param in self.clip_model.parameters():
            if param.dtype in (torch.float16, torch.bfloat16):
                needs_cast = True
                break
        if not needs_cast:
            return
        self.clip_model.float()
        if not self._clip_fp32_train_logged:
            logger.info("Trainable CLIP params detected in low precision; casting CLIP to fp32 for AMP compatibility.")
            self._clip_fp32_train_logged = True

    def _clip_grad_context(self):
        # 动态调度场景下，freeze 可能仍为 True，但部分参数已被外部 phase schedule 解冻。
        # 梯度开关必须以 requires_grad 的真实状态为准。
        if self._clip_has_trainable_params():
            self._ensure_clip_trainable_fp32()
            return torch.enable_grad()
        return torch.no_grad()
        
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        提取图像级CLIP特征
        
        Args:
            images: (B, C, H, W) 或 (B, T, C, H, W)
            
        Returns:
            features: (B, clip_dim) 或 (B, T, clip_dim)
        """
        has_temporal = images.dim() == 5
        original_dtype = images.dtype
        
        if has_temporal:
            B, T, C, H, W = images.shape
            images = images.reshape(B * T, C, H, W)
        
        images = self._preprocess_images(images)
        
        # CLIP始终在fp32下运行以保持精度（禁用autocast）
        with self._clip_grad_context():
            with torch.cuda.amp.autocast(enabled=False):
                # 确保输入是fp32
                images_fp32 = images.float()
                # openai CLIP在CUDA上常见权重为fp16；输入需与权重dtype/device一致
                # 否则会触发 "Input type ... and weight type ... should be the same"。
                try:
                    first_param = next(self.clip_model.parameters())
                    images_fp32 = images_fp32.to(device=first_param.device)
                    clip_dtype = first_param.dtype
                except StopIteration:
                    clip_dtype = images_fp32.dtype
                images_clip = images_fp32.to(dtype=clip_dtype)

                features = self.clip_model.encode_image(images_clip)
                features = features.float()  # 确保输出是fp32
        
        if has_temporal:
            features = features.reshape(B, T, -1)
            
        return features
    
    def extract_spatial_features(
        self, 
        images: torch.Tensor,
        return_cls: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        提取空间CLIP特征 (patch-level)
        
        Args:
            images: (B, C, H, W) 或 (B, T, C, H, W)
            return_cls: 是否返回CLS token
            
        Returns:
            spatial_features: (B, H', W', clip_dim) 或 (B, T, H', W', clip_dim)
            cls_token: (B, clip_dim) 或 (B, T, clip_dim) if return_cls
        """
        has_temporal = images.dim() == 5
        
        if has_temporal:
            B, T, C, H, W = images.shape
            images = images.reshape(B * T, C, H, W)
        else:
            B = images.shape[0]
            T = 1
        
        images = self._preprocess_images(images)
        
        # CLIP始终在fp32下运行以保持精度（禁用autocast）
        with self._clip_grad_context():
            with torch.cuda.amp.autocast(enabled=False):
                # 确保输入是fp32
                images_fp32 = images.float()
                
                # 获取ViT的patch embeddings
                visual = self.clip_model.visual
                required = ["conv1", "class_embedding", "positional_embedding", "ln_pre", "transformer", "ln_post"]
                if not all(hasattr(visual, attr) for attr in required):
                    raise RuntimeError("CLIP visual backbone does not support spatial features extraction")

                # openai CLIP 在 CUDA 上常见权重为 fp16；手工走 visual 路径时需对齐输入dtype
                # 否则会出现 "Input type ... and weight type ... should be the same"。
                conv1_weight = visual.conv1.weight
                images_clip = images_fp32.to(dtype=conv1_weight.dtype)

                # 预处理
                x = visual.conv1(images_clip)  # (B*T, C, H/patch, W/patch)
                x = x.reshape(x.shape[0], x.shape[1], -1)  # (B*T, C, HW)
                x = x.permute(0, 2, 1)  # (B*T, HW, C)
                
                # 添加位置编码
                x = torch.cat([
                    visual.class_embedding.to(x.dtype) + 
                    torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x
                ], dim=1)
                x = x + visual.positional_embedding.to(x.dtype)
                x = visual.ln_pre(x)
                
                # Transformer
                x = x.permute(1, 0, 2)  # (L, B*T, C)
                x = visual.transformer(x)
                x = x.permute(1, 0, 2)  # (B*T, L, C)
                
                # 分离CLS和patch tokens
                cls_token = visual.ln_post(x[:, 0, :])
                if hasattr(visual, 'proj') and visual.proj is not None:
                    cls_token = cls_token @ visual.proj
                
                patch_tokens = x[:, 1:, :]
                # 与CLS一致，patch token也需经过visual.proj映射到clip_dim。
                if hasattr(visual, 'proj') and visual.proj is not None:
                    patch_tokens = patch_tokens @ visual.proj
                patch_tokens = patch_tokens.float()  # (B*T, HW, clip_dim) 确保fp32
        
        # 重塑为空间维度
        num_patches = patch_tokens.shape[1]
        # 尝试推断空间尺寸（假设正方形，否则使用原始图像比例）
        H_out = W_out = int(num_patches ** 0.5)
        if H_out * W_out != num_patches:
            # 非正方形patch grid，根据输入图像比例推断
            if self.input_resolution is not None:
                h_in, w_in = self.input_resolution
                patch_size = getattr(self.clip_model.visual, 'patch_size', 16)
                if isinstance(patch_size, tuple):
                    patch_size = patch_size[0]
                H_out = h_in // patch_size
                W_out = w_in // patch_size
            else:
                # 回退到最接近的整数分解
                import math
                H_out = int(math.sqrt(num_patches))
                W_out = num_patches // H_out
        spatial_features = patch_tokens.reshape(B * T, H_out, W_out, -1)
        
        if has_temporal:
            spatial_features = spatial_features.reshape(B, T, H_out, W_out, -1)
            cls_token = cls_token.reshape(B, T, -1)
        
        if return_cls:
            return spatial_features, cls_token
        return spatial_features, None
    
    def forward(
        self,
        images: torch.Tensor,
        mode: str = "global",
        cache: Optional["DiskTensorCache"] = None,
        cache_key: Optional[Union[str, Sequence[str], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            images: (B, T, C, H, W) 视频帧
            mode: "global" 或 "spatial"
            
        Returns:
            features: 语义特征
        """
        cache_obj = cache if DiskTensorCache is not None else None
        keys: Optional[List[str]] = None
        if cache_obj is not None and cache_key is not None:
            batch_size = int(images.shape[0]) if images.dim() >= 4 else 1
            if isinstance(cache_key, (list, tuple)):
                candidate = [str(k) for k in cache_key]
                if len(candidate) == batch_size:
                    keys = candidate
            elif isinstance(cache_key, torch.Tensor):
                try:
                    if cache_key.ndim == 0 and batch_size == 1:
                        keys = [str(cache_key.item())]
                    elif cache_key.ndim == 1 and int(cache_key.shape[0]) == batch_size:
                        keys = [str(v.item()) for v in cache_key]
                except Exception:
                    keys = None
            elif batch_size == 1:
                keys = [str(cache_key)]

        def _full_key(video_name: str) -> str:
            T = int(images.shape[1]) if images.dim() == 5 else 1
            # Include input resolution to avoid accidentally reusing cached features across
            # different dataset/model resize settings (e.g., 256 vs 480).
            if images.dim() >= 4:
                H = int(images.shape[-2])
                W = int(images.shape[-1])
            else:
                H, W = -1, -1
            return f"{self._cache_prefix}|{mode}|{video_name}|T{T}|HW{H}x{W}"

        if cache_obj is not None and keys is not None:
            cached = [cache_obj.get(_full_key(k)) for k in keys]
            if all(isinstance(v, torch.Tensor) for v in cached):
                raw = torch.stack([v for v in cached], dim=0).to(device=images.device)
                if mode == "global":
                    return self.projection(raw)
                if mode == "spatial":
                    if raw.dim() != 5:
                        # Expected (B,T,H,W,C). Fallback: ignore cache for unexpected shapes.
                        raw = None
                    else:
                        B, T, H, W, C = raw.shape
                        raw_flat = raw.reshape(B * T * H * W, C)
                        proj = self.projection(raw_flat)
                        return proj.reshape(B, T, H, W, -1)
                if raw is None:
                    # fallthrough to compute
                    pass

        if mode == "global":
            raw_features = self.extract_features(images)  # (B,T,clip_dim)
            features = self.projection(raw_features)
        elif mode == "spatial":
            raw_features, _ = self.extract_spatial_features(images)  # (B,T,H,W,clip_dim)
            B, T, H, W, C = raw_features.shape
            raw_flat = raw_features.reshape(B * T * H * W, C)
            proj = self.projection(raw_flat)
            features = proj.reshape(B, T, H, W, -1)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if cache_obj is not None and keys is not None:
            for i, k in enumerate(keys):
                try:
                    # Store RAW CLIP features (pre-projection).
                    cache_obj.set(_full_key(k), raw_features[i])
                except Exception:
                    continue
        
        return features
    
    def sample_point_features(
        self,
        spatial_features: torch.Tensor,
        points: torch.Tensor,
        mode: str = "bilinear"
    ) -> torch.Tensor:
        """
        在指定点位置采样语义特征
        
        Args:
            spatial_features: (B, T, H, W, C)
            points: (B, T, N, 2) 点坐标 (归一化到[0,1])
            mode: 插值模式
            
        Returns:
            point_features: (B, T, N, C)
        """
        B, T, H, W, C = spatial_features.shape
        N = points.shape[2]
        
        # 将特征重排为 (B*T, C, H, W)
        features = spatial_features.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
        
        # 将点坐标转换为grid_sample格式 [-1, 1]
        # 输入点默认是 [y, x]，grid_sample需要 [x, y]
        grid = _normalized_yx_to_grid_xy(points, H, W).reshape(B * T, N, 1, 2)
        
        # 采样
        sampled = F.grid_sample(features, grid, mode=mode, align_corners=True)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (B*T, N, C)
        
        point_features = sampled.reshape(B, T, N, C)
        
        return point_features


if __name__ == "__main__":
    # 测试代码
    print("Testing SemanticEncoder...")
    
    if not CLIP_AVAILABLE:
        print("CLIP not available, skipping test")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = SemanticEncoder(clip_model="ViT-B/16", output_dim=256)
        encoder = encoder.to(device)
        
        # 测试输入
        video = torch.randn(2, 8, 3, 224, 224).to(device)
        
        # 测试全局特征
        global_feat = encoder(video, mode="global")
        print(f"Global features shape: {global_feat.shape}")  # (2, 8, 256)
        
        # 测试空间特征
        spatial_feat = encoder(video, mode="spatial")
        print(f"Spatial features shape: {spatial_feat.shape}")  # (2, 8, 14, 14, 256)
        
        # 测试点采样
        points = torch.rand(2, 8, 100, 2).to(device)  # 100个点
        point_feat = encoder.sample_point_features(spatial_feat, points)
        print(f"Point features shape: {point_feat.shape}")  # (2, 8, 100, 256)
        
        print("SemanticEncoder test passed!")
