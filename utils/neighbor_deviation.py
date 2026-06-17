"""
Neighbor deviation computation from predicted tracks.

Computes per-track reliability scores using K-nearest-neighbor motion consistency.
All inputs are predicted (no GT leakage). High deviation = high risk.

Usage:
  from utils.neighbor_deviation import compute_neighbor_deviation_risk

  risk = compute_neighbor_deviation_risk(pred_tracks, pred_visibility, K=16)
  # risk: (B, N, T) float tensor, higher = less reliable
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_neighbor_deviation_risk(
    pred_tracks: torch.Tensor,   # (B, N, T, 2)
    pred_visibility: torch.Tensor,  # (B, N, T)
    K: int = 16,
) -> torch.Tensor:
    """Compute neighbor-deviation risk score from predicted tracks.

    For each visible frame, measures how far a track deviates from its
    K-nearest neighbors' centroid. Higher = more deviation = higher risk.

    Args:
        pred_tracks: (B, N, T, 2) predicted track positions (normalized)
        pred_visibility: (B, N, T) visibility mask (1=visible, 0=occluded)
        K: number of neighbors

    Returns:
        risk: (B, N, T) risk score. Higher = less reliable. Occluded frames = 0.
    """
    B, N, T, _ = pred_tracks.shape
    device = pred_tracks.device

    risk = torch.zeros(B, N, T, device=device, dtype=torch.float32)

    for b in range(B):
        tracks_b = pred_tracks[b]   # (N, T, 2)
        vis_b = pred_visibility[b]  # (N, T)

        for t in range(T):
            vis_t = vis_b[:, t]  # (N,)
            visible_count = vis_t.sum().item()
            if visible_count < K + 1:
                continue  # Not enough visible neighbors

            # Positions of visible tracks at this frame
            vis_idx = vis_t.nonzero(as_tuple=True)[0]  # (V,)
            vis_pos = tracks_b[vis_idx, t]  # (V, 2)

            # For each visible track, compute distance to K-nearest neighbors
            for j, idx in enumerate(vis_idx):
                q_pos = vis_pos[j:j+1]  # (1, 2)

                # Distance to all other visible tracks
                dists = torch.norm(vis_pos - q_pos, dim=-1)  # (V,)
                # Exclude self (distance=0)
                dists[j] = float('inf')

                # Find K nearest (excluding self)
                k_actual = min(K, vis_t.sum().item() - 1)
                _, nn_idx = dists.topk(k_actual, largest=False)  # (k_actual,)

                # Compute centroid of neighbors
                nn_pos = vis_pos[nn_idx]  # (k_actual, 2)
                centroid = nn_pos.mean(dim=0)  # (2,)

                # Deviation = distance to centroid
                deviation = torch.norm(q_pos - centroid, dim=-1).item()

                # Normalize to make it comparable (using track-level normalization later)
                risk[b, idx.item(), t] = deviation

    return risk


@torch.no_grad()
def compute_neighbor_deviation_risk_fast(
    pred_tracks: torch.Tensor,   # (B, N, T, 2)
    pred_visibility: torch.Tensor,  # (B, N, T)
    K: int = 16,
) -> torch.Tensor:
    """Vectorized version: faster but approximate (uses all-N computation).

    For each frame, computes pairwise distance matrix, masks out invisible,
    and finds K-nearest neighbors for all tracks at once.

    Returns risk: (B, N, T)
    """
    B, N, T, _ = pred_tracks.shape
    device = pred_tracks.device
    risk = torch.zeros(B, N, T, device=device, dtype=torch.float32)

    for b in range(B):
        tracks_b = pred_tracks[b]   # (N, T, 2)
        vis_b = pred_visibility[b]  # (N, T)

        for t in range(T):
            vis_t = vis_b[:, t]  # (N,)
            V = int(vis_t.sum().item())
            if V < K + 1:
                continue

            vis_idx = vis_t.nonzero(as_tuple=True)[0]  # (V,)
            vis_pos = tracks_b[vis_idx, t]  # (V, 2)

            # Pairwise distances (V, V)
            diff = vis_pos.unsqueeze(1) - vis_pos.unsqueeze(0)  # (V, V, 2)
            dists = torch.norm(diff, dim=-1)  # (V, V)

            # Mask self (diagonal = inf)
            dists.fill_diagonal_(float('inf'))

            # K-nearest centroid deviation for each
            k = min(K, V - 1)
            nn_dists, _ = dists.topk(k, largest=False)  # (V, k)
            # Mean distance to K-nearest neighbors
            dev_per_track = nn_dists.mean(dim=-1)  # (V,)

            # Write back to risk tensor
            for j, idx in enumerate(vis_idx):
                risk[b, idx.item(), t] = dev_per_track[j].item()

    return risk


def normalize_risk_to_01(risk: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    """Min-max normalize risk per-batch over visible tracks.

    risk: (B, N, T), higher = worse
    vis: (B, N, T)
    Returns: (B, N, T) risk in [0, 1], occluded = 0.
    """
    B, N, T = risk.shape
    out = torch.zeros_like(risk)
    for b in range(B):
        r = risk[b]
        m = vis[b].bool()
        r_vis = r[m]
        if r_vis.numel() == 0:
            continue
        rmin, rmax = r_vis.min(), r_vis.max()
        if rmax > rmin:
            out[b] = torch.where(m, (risk[b] - rmin) / (rmax - rmin), torch.zeros_like(risk[b]))
        else:
            out[b] = torch.zeros_like(risk[b])
    return out
