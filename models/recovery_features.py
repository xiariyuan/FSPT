"""
Shared recovery feature extractors for the online recovery pipeline.

Provides feature sources that can be used by the relocalization / recovery
branch in CoTrackerFSPTRefiner. Currently supports:
  - cotracker: uses the base tracker's fnet features (default)
  - dino: uses DINOv2 ViT-S/14 for stronger appearance matching

Usage in cotracker_refiner.py:
    from models.recovery_features import get_recovery_feature_extractor
    extractor = get_recovery_feature_extractor("dino", device=device)
    feat_map = extractor.feature_map(image_rgb_uint8)  # (D, H, W)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import cv2 as _cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DINO_WEIGHTS = str(_PROJECT_ROOT / "weights" / "dinov2" / "dinov2_vits14_pretrain.pth")


class DINORecoveryExtractor(nn.Module):
    """DINOv2 ViT-S/14 feature extractor for recovery-only use.

    This is a torch Module so it can be registered as a submodule of the refiner.
    It is always eval-mode and frozen (no gradient).
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        img_size: int = 518,
        feat_dim: int = 384,
    ):
        super().__init__()
        import timm

        self.img_size = img_size
        self.feat_dim = feat_dim

        wp = weights_path or _DEFAULT_DINO_WEIGHTS
        self.weights_path = str(wp)

        self.model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=False,
            features_only=True,
            out_indices=[3],
            img_size=img_size,
        )

        self._loaded = False

    def _ensure_loaded(self, device: torch.device):
        if self._loaded:
            return
        sd = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        model_keys = set(self.model.state_dict().keys())
        if model_keys and not any(k.startswith("model.") for k in sd):
            sd = {"model." + k: v for k, v in sd.items()}
        pos_key = "model.pos_embed"
        if pos_key in sd and pos_key in self.model.state_dict():
            target_shape = self.model.state_dict()[pos_key].shape
            if sd[pos_key].shape != target_shape:
                old = sd[pos_key]
                cls_tok = old[:, :1]
                spatial = old[:, 1:]
                old_grid = int(spatial.shape[1] ** 0.5)
                new_grid = int((target_shape[1] - 1) ** 0.5)
                dim = spatial.shape[-1]
                spatial = spatial.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
                spatial = F.interpolate(spatial, size=(new_grid, new_grid), mode="bilinear", align_corners=False)
                spatial = spatial.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, dim)
                sd[pos_key] = torch.cat([cls_tok, spatial], dim=1)
        self.model.load_state_dict(sd, strict=False)
        self.model = self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._loaded = True
        logger.info(f"DINORecoveryExtractor loaded from {self.weights_path}")

    @torch.no_grad()
    def feature_map(self, image_rgb_uint8: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Extract DINOv2 feature map from a single RGB image.

        Args:
            image_rgb_uint8: (H, W, 3) uint8 or float [0, 255] tensor on any device
            device: target device for inference

        Returns:
            (D, H_feat, W_feat) float tensor on CPU (normalized)
        """
        self._ensure_loaded(device)
        x = image_rgb_uint8.float()
        if x.dim() == 3 and x.shape[-1] == 3:
            x = x.permute(2, 0, 1)  # (3, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)  # (1, 3, H, W)
        x = x / 255.0
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        x = x.to(device)
        feat = self.model(x)[-1]  # (1, D, h, w)
        feat = F.normalize(feat.float(), dim=1)
        return feat[0].cpu()  # (D, h, w)

    @torch.no_grad()
    def encode_patch(self, patch_rgb_uint8: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Encode a single patch into a DINOv2 feature vector.

        Args:
            patch_rgb_uint8: (H, W, 3) or (3, H, W) uint8/float tensor
            device: target device

        Returns:
            (D,) normalized float tensor on CPU
        """
        fmap = self.feature_map(patch_rgb_uint8, device)  # (D, h, w)
        return fmap.mean(dim=[-2, -1])  # (D,)

    @torch.no_grad()
    def encode_patches_batch(
        self,
        patches_rgb_uint8: torch.Tensor,
        device: torch.device,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Encode a batch of patches.

        Args:
            patches_rgb_uint8: (N, H, W, 3) or (N, 3, H, W) tensor
            device: target device
            batch_size: inference batch size

        Returns:
            (N, D) normalized float tensor on CPU
        """
        self._ensure_loaded(device)
        patches = patches_rgb_uint8.float()
        if patches.dim() == 4 and patches.shape[-1] == 3:
            patches = patches.permute(0, 3, 1, 2)  # (N, 3, H, W)
        patches = patches / 255.0

        out = []
        for start in range(0, len(patches), batch_size):
            batch = patches[start:start + batch_size]
            batch = F.interpolate(batch, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
            batch = batch.to(device)
            feat = self.model(batch)[-1].mean(dim=[-2, -1]).float()  # (bs, D)
            feat = F.normalize(feat, dim=-1)
            out.append(feat.cpu())
        return torch.cat(out, dim=0)


# --- Factory ---

_extractor_cache = {}


def get_recovery_feature_extractor(
    source: str,
    device: Optional[torch.device] = None,
    dino_weights: Optional[str] = None,
) -> Optional[nn.Module]:
    """Get or create a recovery feature extractor.

    Args:
        source: "cotracker" | "dino" | "geo"
        device: torch device (only needed for dino)
        dino_weights: path to DINOv2 weights (only for dino)

    Returns:
        nn.Module for dino, None for cotracker/geo (those use the base tracker's features)
    """
    source = str(source).lower().strip()

    if source in ("cotracker", "geo", ""):
        return None  # Uses base tracker features, no extra extractor needed

    if source == "dino":
        cache_key = ("dino", dino_weights or _DEFAULT_DINO_WEIGHTS)
        if cache_key not in _extractor_cache:
            _extractor_cache[cache_key] = DINORecoveryExtractor(weights_path=dino_weights)
        return _extractor_cache[cache_key]

    logger.warning(f"Unknown recovery feature source: {source!r}")
    return None


# --- M1 Gate Feature Builder ---

GATE_FEATURE_NAMES = [
    "delta_x",           # anchor_x - base_x
    "delta_y",           # anchor_y - base_y
    "delta_norm",        # ||delta_xy||
    "relocal_conf",      # relocalization confidence
    "base_vis",          # base tracker visibility at t0
    "occ_len_norm",      # occlusion length / 300.0
]


def build_gate_features_scalar(
    base_xy: np.ndarray,      # (2,) normalized [x, y]
    anchor_xy: np.ndarray,    # (2,) normalized [x, y]
    relocal_conf: float,
    base_vis: float,
    occ_len: int,
) -> np.ndarray:
    """Build scalar gate features from online-compatible signals.

    All inputs are available in real-time from the tracker pipeline.
    No GT leakage.

    Returns:
        (6,) float32 feature vector
    """
    delta = anchor_xy - base_xy
    return np.array([
        float(delta[0]),
        float(delta[1]),
        float(np.linalg.norm(delta)),
        float(relocal_conf),
        float(base_vis),
        float(occ_len) / 300.0,
    ], dtype=np.float32)


def build_gate_labels(
    base_err_px: float,
    anchor_err_px: float,
    margin_px: float = 2.0,
) -> Dict:
    """Build gate labels from error values.

    Args:
        base_err_px: base tracker pixel error at t0
        anchor_err_px: anchor pixel error at t0
        margin_px: margin for ambiguous zone

    Returns:
        dict with accept_label, improve_px, base_good, base_bad, base_very_bad
    """
    improve = base_err_px - anchor_err_px
    if improve >= margin_px:
        accept_label = 1
    elif improve <= -margin_px:
        accept_label = 0
    else:
        accept_label = -1  # ambiguous

    return {
        "improve_px": round(float(improve), 2),
        "accept_label": int(accept_label),
        "base_good": bool(base_err_px <= 4.0),
        "base_bad": bool(base_err_px > 16.0),
        "base_very_bad": bool(base_err_px > 32.0),
    }
