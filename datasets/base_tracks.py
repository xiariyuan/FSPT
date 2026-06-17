from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseTracksCacheWrapper(Dataset):
    """
    Dataset wrapper that injects precomputed base tracker outputs.

    Expected cache file format (.pt):
      {
        "video_name": str,
        "query_points": Tensor (N,3) optional,
        "base_tracks": Tensor (N,T,2) normalized [y,x],
        "base_visibility": Tensor (N,T) float/bool,
      }
    """

    def __init__(
        self,
        dataset: Dataset,
        base_tracks_dir: str,
        dataset_name: Optional[str] = None,
        strict: bool = False,
        query_tol: float = 1e-4,
    ) -> None:
        self.dataset = dataset
        self.dataset_name = str(dataset_name).lower().strip() if dataset_name else None
        self.strict = bool(strict)
        self.query_tol = float(query_tol)

        root = Path(str(base_tracks_dir))
        self.base_tracks_dir = root

        self._warned_missing = False
        self._warned_mismatch = False

    def __len__(self) -> int:
        return len(self.dataset)

    def _resolve_cache_path(self, video_name: str) -> Optional[Path]:
        # Prefer per-dataset subdir if present.
        candidates = []
        if self.dataset_name:
            candidates.append(self.base_tracks_dir / self.dataset_name / f"{video_name}.pt")
        candidates.append(self.base_tracks_dir / f"{video_name}.pt")
        for path in candidates:
            if path.exists() and path.stat().st_size > 0:
                return path
        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]
        if not isinstance(sample, dict):
            return sample
        video_name = sample.get("video_name", None)
        if video_name is None:
            return sample
        video_name = str(video_name)

        path = self._resolve_cache_path(video_name)
        if path is None:
            if self.strict:
                raise FileNotFoundError(f"Base tracks cache not found for {video_name}")
            if not self._warned_missing:
                logger.warning(
                    "Base tracks cache enabled but missing files (showing once). "
                    f"Example: {video_name}"
                )
                self._warned_missing = True
            return sample

        try:
            payload = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception as exc:
            if self.strict:
                raise
            if not self._warned_missing:
                logger.warning(f"Failed to load base tracks cache {path}: {exc}")
                self._warned_missing = True
            return sample

        if not isinstance(payload, dict):
            return sample

        base_tracks = payload.get("base_tracks", None)
        base_visibility = payload.get("base_visibility", None)
        if not isinstance(base_tracks, torch.Tensor) or not isinstance(base_visibility, torch.Tensor):
            return sample

        # Optional safety check: query_points should match (evaluation splits are deterministic).
        try:
            cached_q = payload.get("query_points", None)
            cur_q = sample.get("query_points", None)
            if isinstance(cached_q, torch.Tensor) and isinstance(cur_q, torch.Tensor):
                if cached_q.shape == cur_q.shape:
                    diff = (cached_q.float() - cur_q.detach().cpu().float()).abs().max().item()
                    if diff > self.query_tol:
                        if self.strict:
                            raise ValueError(
                                f"query_points mismatch for {video_name}: max_abs_diff={diff}"
                            )
                        if not self._warned_mismatch:
                            logger.warning(
                                "Base tracks cache query_points mismatch (showing once). "
                                f"Example: {video_name}, max_abs_diff={diff:.3e}"
                            )
                            self._warned_mismatch = True
                        return sample
        except Exception:
            if self.strict:
                raise

        # Attach to sample (DataLoader will collate to (B,N,T,2)/(B,N,T)).
        sample["base_tracks"] = base_tracks
        sample["base_visibility"] = base_visibility
        return sample

