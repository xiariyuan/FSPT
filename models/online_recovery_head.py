"""
Recovery heads for the online recovery pipeline.

OnlineRecoveryHead: heatmap-based geometric recovery head (M0 offline)
OnlineRecoveryGateHead: scalar feature gate for accept/reject decision (M1 online)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineRecoveryGateHead(nn.Module):
    """Lightweight gate: accept or reject a relocal anchor.

    Input: scalar feature vector [delta_x, delta_y, delta_norm, relocal_conf, base_vis, occ_len_norm]
    Output: override_logit (sigmoid -> P(override))

    Design: very small MLP, bias initialized toward conservative (reject).
    """

    def __init__(self, input_dim: int = 6, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize last layer bias negative -> conservative default (reject)
        nn.init.constant_(self.net[-1].bias, -1.0)
        nn.init.xavier_uniform_(self.net[-1].weight)

    def forward(self, features: torch.Tensor) -> dict:
        """
        Args:
            features: (B, input_dim) scalar gate features

        Returns:
            dict with:
                logit: (B,) raw logit
                prob: (B,) sigmoid probability
        """
        logit = self.net(features).squeeze(-1)
        prob = torch.sigmoid(logit)
        return {"logit": logit, "prob": prob}

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class OnlineRecoveryHead(nn.Module):
    """Minimal recovery head: conv tower over fused search+support features.

    Input:
        search_feature_map: (B, D, H, W) DINO features of search crop
        support_descriptor: (B, D) pooled support memory descriptor
        tracker_vis: (B, 1) optional visibility signal

    Output:
        heatmap: (B, 1, H, W) logits (before sigmoid)
        pred_xy: (B, 2) normalized coordinates [0, 1] via soft-argmax
    """

    def __init__(
        self,
        feat_dim: int = 384,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.feat_dim = feat_dim

        # Project search features to lower dim
        self.search_proj = nn.Conv2d(feat_dim, hidden_dim, kernel_size=1)

        # Project support descriptor to same dim
        self.support_proj = nn.Linear(feat_dim, hidden_dim)

        # +2 for relative (y, x) coordinate channels
        # +1 for tracker_visibility broadcast
        # search_proj: hidden, supp_map: hidden, yy: 1, xx: 1, vis: 1
        in_channels = hidden_dim * 2 + 3

        # Conv tower
        layers = []
        ch = in_channels
        for i in range(num_layers):
            out_ch = hidden_dim // (2 ** i) if i < num_layers - 1 else 32
            out_ch = max(out_ch, 16)
            layers.extend([
                nn.Conv2d(ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            ch = out_ch
        # Final 1x1 conv -> 1 channel heatmap
        layers.append(nn.Conv2d(ch, 1, kernel_size=1))
        self.conv_tower = nn.Sequential(*layers)

    def forward(
        self,
        search_feature_map: torch.Tensor,
        support_descriptor: torch.Tensor,
        tracker_vis: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            search_feature_map: (B, D, H, W)
            support_descriptor: (B, D)
            tracker_vis: (B, 1) optional

        Returns:
            dict with 'heatmap' (B, 1, H, W) and 'pred_xy' (B, 2)
        """
        B, D, H, W = search_feature_map.shape
        device = search_feature_map.device

        # Project search features
        s_proj = self.search_proj(search_feature_map)  # (B, hidden, H, W)

        # Project support descriptor and broadcast
        supp_proj = self.support_proj(support_descriptor)  # (B, hidden)
        supp_map = supp_proj.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # (B, hidden, H, W)

        # Relative coordinate channels
        yy = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
        xx = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)

        # Tracker visibility broadcast
        if tracker_vis is not None:
            vis_map = tracker_vis.unsqueeze(-1).unsqueeze(-1).expand(B, 1, H, W)
        else:
            vis_map = torch.zeros(B, 1, H, W, device=device)

        # Concatenate all features
        x = torch.cat([s_proj, supp_map, yy, xx, vis_map], dim=1)  # (B, hidden+3, H, W)

        # Conv tower -> heatmap logits
        heatmap = self.conv_tower(x)  # (B, 1, H, W)

        # Soft-argmax readout
        prob = torch.softmax(heatmap.reshape(B, -1), dim=-1).reshape(B, H, W)
        yy_idx = torch.linspace(0, 1, H, device=device).view(1, H, 1)
        xx_idx = torch.linspace(0, 1, W, device=device).view(1, 1, W)
        pred_y = (prob * yy_idx).sum(dim=[1, 2])  # (B,)
        pred_x = (prob * xx_idx).sum(dim=[1, 2])  # (B,)
        pred_xy = torch.stack([pred_x, pred_y], dim=-1)  # (B, 2)

        return {
            "heatmap": heatmap,
            "pred_xy": pred_xy,
        }

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
