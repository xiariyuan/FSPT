from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


_MODEL_BUILDERS = {
    "dinov2_vits14": "dinov2_vits14",
    "dinov2_vitb14": "dinov2_vitb14",
    "dinov2_vitl14": "dinov2_vitl14",
    "dinov2_vitg14": "dinov2_vitg14",
    "dinov2_vits14_reg": "dinov2_vits14_reg",
    "dinov2_vitb14_reg": "dinov2_vitb14_reg",
    "dinov2_vitl14_reg": "dinov2_vitl14_reg",
    "dinov2_vitg14_reg": "dinov2_vitg14_reg",
}

_TIMM_MODEL_NAMES = {
    "dinov2_vits14": "vit_small_patch14_dinov2",
    "dinov2_vitb14": "vit_base_patch14_dinov2",
    "dinov2_vitl14": "vit_large_patch14_dinov2",
    "dinov2_vitg14": "vit_giant_patch14_dinov2",
    "dinov2_vits14_reg": "vit_small_patch14_reg4_dinov2",
    "dinov2_vitb14_reg": "vit_base_patch14_reg4_dinov2",
    "dinov2_vitl14_reg": "vit_large_patch14_reg4_dinov2",
    "dinov2_vitg14_reg": "vit_giant_patch14_reg4_dinov2",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_external_dinov2_on_path() -> None:
    external_dir = _repo_root() / "external" / "dinov2"
    if external_dir.is_dir():
        external_path = str(external_dir)
        if external_path not in sys.path:
            sys.path.insert(0, external_path)


def _load_dinov2_builder(model_name: str):
    _ensure_external_dinov2_on_path()
    try:
        from dinov2.hub import backbones as dinov2_backbones
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import local DINOv2 code. Ensure `external/dinov2` exists and its dependencies are installed."
        ) from exc
    try:
        return getattr(dinov2_backbones, _MODEL_BUILDERS[model_name])
    except KeyError as exc:
        raise ValueError("Unsupported DINOv2 model: {0}".format(model_name)) from exc


def _build_timm_model(model_name: str, pretrained: bool, weights: Optional[str]):
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("Fallback DINOv2 path requires `timm` to be installed.") from exc
    try:
        timm_name = _TIMM_MODEL_NAMES[model_name]
    except KeyError as exc:
        raise ValueError("Unsupported DINOv2 model for timm fallback: {0}".format(model_name)) from exc
    model = timm.create_model(timm_name, pretrained=bool(pretrained) and not weights, dynamic_img_size=True)
    if weights:
        state_dict = torch.load(weights, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    return model


def _as_pair(value) -> Tuple[int, int]:
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


@dataclass
class DinoV2Output:
    feature_maps: Tuple[torch.Tensor, ...]
    cls_tokens: Tuple[torch.Tensor, ...]
    patch_grid: Tuple[int, int]
    patch_size: Tuple[int, int]
    input_size: Tuple[int, int]
    resized_input_size: Tuple[int, int]
    embedding_dim: int


class DinoV2Backbone(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        pretrained: bool = True,
        weights: Optional[str] = None,
        out_indices: Sequence[int] = (8, 10, 11),
        train_backbone: bool = False,
        apply_layer_norm: bool = True,
        return_cls_tokens: bool = True,
        resize_to_patch_multiple: bool = True,
        pixel_mean: Sequence[float] = (0.485, 0.456, 0.406),
        pixel_std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        if not out_indices:
            raise ValueError("out_indices must not be empty.")
        self.backend = "external"
        builder_kwargs = {"pretrained": bool(pretrained)}
        if weights:
            builder_kwargs["weights"] = weights
        try:
            builder = _load_dinov2_builder(model_name)
            self.model = builder(**builder_kwargs)
        except Exception as exc:
            warnings.warn(
                "Falling back to timm DINOv2 backbone because external/dinov2 could not be imported: {0}".format(exc)
            )
            self.backend = "timm"
            self.model = _build_timm_model(model_name, pretrained=bool(pretrained), weights=weights)
        self.out_indices = tuple(int(index) for index in out_indices)
        self.apply_layer_norm = bool(apply_layer_norm)
        self.return_cls_tokens = bool(return_cls_tokens)
        self.resize_to_patch_multiple = bool(resize_to_patch_multiple)
        self.embedding_dim = int(getattr(self.model, "embed_dim"))
        self.patch_size = _as_pair(getattr(getattr(self.model, "patch_embed", None), "patch_size", 14))
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        if not bool(train_backbone):
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float()
        if torch.isfinite(images).all() and float(images.detach().max()) > 1.5:
            images = images / 255.0
        return (images - self.pixel_mean) / self.pixel_std

    def _resize_inputs(self, images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        height, width = int(images.shape[-2]), int(images.shape[-1])
        if not self.resize_to_patch_multiple:
            return images, (height, width)
        patch_h, patch_w = self.patch_size
        target_h = max(patch_h, int((height + patch_h - 1) // patch_h) * patch_h)
        target_w = max(patch_w, int((width + patch_w - 1) // patch_w) * patch_w)
        if target_h == height and target_w == width:
            return images, (height, width)
        resized = F.interpolate(images, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return resized, (target_h, target_w)

    @staticmethod
    def _reshape_leading_dims(tensor: torch.Tensor, batch: int, time: Optional[int]) -> torch.Tensor:
        if time is None:
            return tensor
        return tensor.reshape(batch, time, tensor.shape[1], tensor.shape[2], tensor.shape[3])

    @staticmethod
    def _reshape_cls_tokens(tensor: torch.Tensor, batch: int, time: Optional[int]) -> torch.Tensor:
        if time is None:
            return tensor
        return tensor.reshape(batch, time, tensor.shape[-1])

    def forward(self, images: torch.Tensor) -> DinoV2Output:
        if images.ndim == 4:
            batch = int(images.shape[0])
            time = None
            flat_images = images
        elif images.ndim == 5:
            batch, time, channels, height, width = images.shape
            flat_images = images.reshape(batch * time, channels, height, width)
        else:
            raise ValueError("Expected image tensor with shape (B, C, H, W) or (B, T, C, H, W).")

        input_size = (int(flat_images.shape[-2]), int(flat_images.shape[-1]))
        normalized = self._normalize(flat_images)
        normalized, resized_input_size = self._resize_inputs(normalized)
        if self.backend == "external":
            outputs = self.model.get_intermediate_layers(
                normalized,
                n=self.out_indices,
                reshape=True,
                return_class_token=self.return_cls_tokens,
                norm=self.apply_layer_norm,
            )
            if self.return_cls_tokens:
                feature_maps = tuple(item[0] for item in outputs)
                cls_tokens = tuple(item[1] for item in outputs)
            else:
                feature_maps = tuple(outputs)
                cls_tokens = tuple()
        else:
            timm_outputs = self.model.forward_intermediates(
                normalized,
                indices=self.out_indices,
                return_prefix_tokens=self.return_cls_tokens,
                norm=self.apply_layer_norm,
                output_fmt="NCHW",
            )
            if self.return_cls_tokens:
                _, features_with_prefix = timm_outputs
                feature_maps = tuple(item[0] for item in features_with_prefix)
                cls_tokens = tuple(item[1].squeeze(1) for item in features_with_prefix)
            else:
                if isinstance(timm_outputs, tuple):
                    _, feature_maps = timm_outputs
                else:
                    feature_maps = timm_outputs
                feature_maps = tuple(feature_maps)
                cls_tokens = tuple()
        feature_maps = tuple(self._reshape_leading_dims(feature_map, batch, time) for feature_map in feature_maps)
        cls_tokens = tuple(self._reshape_cls_tokens(token, batch, time) for token in cls_tokens)
        patch_grid = (int(feature_maps[0].shape[-2]), int(feature_maps[0].shape[-1]))
        return DinoV2Output(
            feature_maps=feature_maps,
            cls_tokens=cls_tokens,
            patch_grid=patch_grid,
            patch_size=self.patch_size,
            input_size=input_size,
            resized_input_size=resized_input_size,
            embedding_dim=self.embedding_dim,
        )


def build_dinov2_backbone(**kwargs) -> DinoV2Backbone:
    return DinoV2Backbone(**kwargs)
