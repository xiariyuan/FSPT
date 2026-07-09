"""TAPNext++ Predictor for FSPT evaluation pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path("/gemini/code/FSPT")
REPO = ROOT / "external/tapnextpp/repo"
CKPT = ROOT / "checkpoints/tapnextpp/tapnextpp_ckpt.pt"

import sys
sys.path.insert(0, str(REPO))

from tapnet.tapnext.tapnext_torch import TAPNext


class TAPNextPlusPlusPredictor(torch.nn.Module):
    """TAPNext++ Predictor.

    Interface compatible with the FSPT evaluation pipeline:
        model(video, query_points) -> (tracks, visibility)

    where:
        video: (B, T, C, H, W) float32 in [0, 255], B=1
        query_points: (B, N, 3) float32 [t, y, x] in pixel coords at ORIGINAL resolution
        tracks: (B, T, N, 2) float32 [x, y] in pixel coords at ORIGINAL resolution
        visibility: (B, T, N) bool, True = visible
    """

    def __init__(self, checkpoint_path: str | None = None):
        super().__init__()
        ckpt = checkpoint_path or str(CKPT)

        self.interp_shape = (256, 256)
        self.model = TAPNext(image_size=self.interp_shape)
        state = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict({
            k.replace("tapnext.", ""): v
            for k, v in state["state_dict"].items()
        })
        self.model.eval()

    def forward(
        self, rgbs: torch.Tensor, queries: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C, H, W = rgbs.shape
        device = rgbs.device

        # ---- 1. Resize video to 256x256 ----
        rgbs_flat = rgbs.reshape(B * T, C, H, W)
        rgbs_resized = F.interpolate(
            rgbs_flat, self.interp_shape, mode="bilinear", align_corners=False
        )  # (B*T, C, 256, 256)
        rgbs_resized = rgbs_resized.reshape(B, T, C, *self.interp_shape)

        # TAPNext expects (B, T, H, W, C) in [-1, 1]
        video = (
            rgbs_resized[0].permute(0, 2, 3, 1).unsqueeze(0).float()  # (1, T, 256, 256, 3)
        )
        video = video / 127.5 - 1.0  # [0, 255] → [-1, 1]

        # ---- 2. Scale query points from original pixel to model pixel ----
        # queries: (B, N, 3) = [t, y, x] in ORIGINAL pixel coords
        queries = queries.clone().float()
        queries[:, :, 1] *= self.interp_shape[0] / float(H)  # y: [0,H] → [0,256)
        queries[:, :, 2] *= self.interp_shape[1] / float(W)  # x: [0,W] → [0,256)

        if torch.isnan(queries).any():
            raise ValueError("Queries contain NaN values")

        # ---- 3. Online frame-by-frame inference ----
        pred_tracks_list = []
        pred_visible_list = []

        with torch.no_grad():
            tr, _, vl, state = self.model(
                video=video[:, :1].to(device),
                query_points=queries.to(device),
            )
            # tr: (1, 1, N, 2) in [y_pred, x_pred] pixel at 256
            pred_tracks_list.append(tr.cpu())
            pred_visible_list.append((vl > 0).cpu())

            for f in range(1, T):
                tr, _, vl, state = self.model(
                    video=video[:, f:f+1].to(device),
                    state=state,
                )
                pred_tracks_list.append(tr.cpu())
                pred_visible_list.append((vl > 0).cpu())

        # ---- 4. Assemble outputs ----
        # cat along time: (1, T, N, 2)
        tracks_model = torch.cat(pred_tracks_list, dim=1)   # (1, T, N, 2) [y, x]
        visible_model = torch.cat(pred_visible_list, dim=1)  # (1, T, N, 1)

        # Flip from [y, x] to [x, y] for TAP-Vid convention
        tracks_model = tracks_model.flip(-1)  # (1, T, N, 2) → [x, y]

        # Squeeze visibility last dim: (1, T, N)
        visibility = visible_model.squeeze(-1).bool()

        # ---- 5. Rescale tracks back to original resolution ----
        tracks_model[:, :, :, 0] *= W / float(self.interp_shape[1])  # x
        tracks_model[:, :, :, 1] *= H / float(self.interp_shape[0])  # y

        tracks = tracks_model.to(device)
        visibility = visibility.to(device)

        return tracks, visibility
