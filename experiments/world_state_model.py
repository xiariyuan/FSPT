from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScratchPatchEncoder(nn.Module):
    """Small CNN patch encoder used by the original appearance-v3 runs."""

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(out_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        feat = self.conv(patch)
        feat = feat.mean(dim=[-2, -1])
        return self.proj(feat)


class FrozenDINOv2PatchEncoder(nn.Module):
    """Frozen DINOv2 ViT-S/14 to encode 64x64 appearance patches."""

    def __init__(self, out_dim: int = 128, weights_path: str = ""):
        super().__init__()
        import sys

        sys.path.insert(0, "/gemini/code/Depth-Anything-V2")
        import timm

        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=False,
            features_only=True,
            out_indices=[3],
            img_size=518,
        )

        if weights_path:
            sd = torch.load(weights_path, map_location="cpu", weights_only=True)
            model_keys = set(self.backbone.state_dict().keys())
            if model_keys and not any(k.startswith("model.") for k in sd):
                sd = {"model." + k: v for k, v in sd.items()}
            pos_key = "model.pos_embed"
            if pos_key in sd and pos_key in self.backbone.state_dict():
                target_shape = self.backbone.state_dict()[pos_key].shape
                if sd[pos_key].shape != target_shape:
                    old = sd[pos_key]
                    cls_tok = old[:, :1]
                    spatial = old[:, 1:]
                    old_grid = int(spatial.shape[1] ** 0.5)
                    new_grid = int((target_shape[1] - 1) ** 0.5)
                    dim = spatial.shape[-1]
                    spatial = spatial.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
                    spatial = F.interpolate(
                        spatial,
                        size=(new_grid, new_grid),
                        mode="bilinear",
                        align_corners=False,
                    )
                    spatial = spatial.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, dim)
                    sd[pos_key] = torch.cat([cls_tok, spatial], dim=1)
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            print(f"DINOv2 loaded: missing={len(missing)}, unexpected={len(unexpected)}")

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.proj = nn.Linear(384, out_dim)
        self.out_dim = out_dim

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(patch, size=(518, 518), mode="bilinear", align_corners=False)
        with torch.no_grad():
            feats = self.backbone(x)
            feat = feats[-1].mean(dim=[-2, -1])
        return self.proj(feat.float())


class TemporalWorldStateRefiner(nn.Module):
    """
    Stage 2 world-state refiner with switchable appearance encoder.
    """

    def __init__(
        self,
        feature_dim: int = 13,
        history_dim: int = 12,
        hidden_dim: int = 256,
        history_len: int = 8,
        patch_feat_dim: int = 128,
        appearance_mode: str = "scratch",
        dinov2_weights: str = "",
        use_query_prev_patch: bool = False,
        gate_bias_init: float = -1.0,
        delta_scale: float = 3.0,
    ) -> None:
        super().__init__()
        self.history_len = int(history_len)
        self.appearance_mode = str(appearance_mode)
        self.use_query_prev_patch = bool(use_query_prev_patch)
        self.delta_scale = float(delta_scale)

        if self.appearance_mode == "scratch":
            self.query_patch_enc = ScratchPatchEncoder(out_dim=patch_feat_dim)
            self.reentry_patch_enc = ScratchPatchEncoder(out_dim=patch_feat_dim)
            self.query_prev_patch_enc = (
                ScratchPatchEncoder(out_dim=patch_feat_dim) if self.use_query_prev_patch else None
            )
        elif self.appearance_mode == "dino":
            shared_enc = FrozenDINOv2PatchEncoder(out_dim=patch_feat_dim, weights_path=dinov2_weights)
            self.query_patch_enc = shared_enc
            self.reentry_patch_enc = shared_enc
            self.query_prev_patch_enc = shared_enc if self.use_query_prev_patch else None
        else:
            raise ValueError(f"Unknown appearance_mode: {self.appearance_mode}")

        self.history_proj = nn.Sequential(
            nn.Linear(history_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.history_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        num_patch_streams = 2 + int(self.use_query_prev_patch)
        context_in_dim = feature_dim + patch_feat_dim * num_patch_streams
        self.context_proj = nn.Sequential(
            nn.Linear(context_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.delta_head = nn.Linear(hidden_dim, 3)
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.visibility_head = nn.Linear(hidden_dim, 1)

        nn.init.constant_(self.gate_head.bias, float(gate_bias_init))

    def forward(
        self,
        features: torch.Tensor,
        input_world: torch.Tensor,
        history_xy: torch.Tensor,
        history_xy_delta: torch.Tensor,
        history_world: torch.Tensor,
        history_world_delta: torch.Tensor,
        history_vis: torch.Tensor,
        history_camrot: torch.Tensor,
        query_patch: Optional[torch.Tensor] = None,
        reentry_patch: Optional[torch.Tensor] = None,
        query_prev_patch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        history = torch.cat(
            [
                history_xy,
                history_xy_delta,
                history_world,
                history_world_delta,
                history_vis,
                history_camrot,
            ],
            dim=-1,
        )
        bsz, steps, dim = history.shape
        history = history.view(bsz * steps, dim)
        history = self.history_proj(history).view(bsz, steps, -1)
        _, h = self.history_gru(history)
        history_feat = h[-1]

        patch_feats = []
        if query_patch is not None:
            patch_feats.append(self.query_patch_enc(query_patch))
        if reentry_patch is not None:
            patch_feats.append(self.reentry_patch_enc(reentry_patch))
        if self.use_query_prev_patch:
            if query_prev_patch is not None:
                patch_feats.append(self.query_prev_patch_enc(query_prev_patch))
            else:
                patch_feats.append(torch.zeros(bsz, self.query_patch_enc.out_dim, device=features.device))

        context_input = torch.cat([features] + patch_feats, dim=-1)
        context_feat = self.context_proj(context_input)

        fused = self.fusion(torch.cat([history_feat, context_feat], dim=-1))
        raw_delta = self.delta_head(fused)
        gate = torch.sigmoid(self.gate_head(fused))
        delta = torch.tanh(raw_delta) * gate * self.delta_scale
        refined_world = input_world + delta
        visibility_logit = self.visibility_head(fused).squeeze(-1)

        return {
            "delta_world": delta,
            "refined_world": refined_world,
            "gate": gate.squeeze(-1),
            "visibility_logit": visibility_logit,
        }
