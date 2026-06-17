import math
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.eval_world_state_stage0 import discover_sequences, find_reentry_queries, load_sequence, project_3d_to_2d


def _load_rgb_frame(seq_path: Path, frame_idx: int, seq_name: str = "") -> Optional[np.ndarray]:
    """Load a single RGB frame from PointOdyssey rgbs/ directory."""
    rgb_dir = seq_path / "rgbs"
    if not rgb_dir.is_dir():
        return None
    fname = rgb_dir / f"rgb_{frame_idx:05d}.jpg"
    if not fname.exists():
        return None
    img = cv2.imread(str(fname))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _extract_patch(
    image: np.ndarray,
    center_yx: np.ndarray,
    patch_size: int = 64,
) -> np.ndarray:
    """Extract a square patch around a 2D point. Returns (C, patch_size, patch_size)."""
    H, W = image.shape[:2]
    cy = int(np.clip(center_yx[0], 0, H - 1))
    cx = int(np.clip(center_yx[1], 0, W - 1))
    half = patch_size // 2
    y0 = max(0, cy - half)
    y1 = min(H, cy + half)
    x0 = max(0, cx - half)
    x1 = min(W, cx + half)
    patch = image[y0:y1, x0:x1]
    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    patch = patch.astype(np.float32) / 255.0
    return patch.transpose(2, 0, 1)  # (C, H, W)


def _extract_patch_from_xy(
    image: np.ndarray,
    center_xy: np.ndarray,
    patch_size: int = 64,
) -> np.ndarray:
    """PointOdyssey trajs_2d are pixel-space (x, y); convert to patch-space (y, x)."""
    if not np.all(np.isfinite(center_xy)):
        return np.zeros((3, patch_size, patch_size), dtype=np.float32)
    center_yx = np.array([center_xy[1], center_xy[0]], dtype=np.float32)
    return _extract_patch(image, center_yx, patch_size=patch_size)


def _lift_points_to_world(points_xy: np.ndarray, depths: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    """Lift 2D points with depth to 3D world coordinates."""
    k_inv = np.linalg.inv(intrinsics)
    pixels_h = np.concatenate([points_xy, np.ones((points_xy.shape[0], 1), dtype=np.float32)], axis=1)
    pts_cam = (k_inv @ pixels_h.T).T * depths[:, None]
    e_inv = np.linalg.inv(extrinsics)
    return (e_inv[:3, :3] @ pts_cam.T).T + e_inv[:3, 3]


class PointOdysseyWorldStateDataset(Dataset):
    """
    Stage 2 dataset with hard-subset awareness and temporal history.
    """

    def __init__(
        self,
        data_root: str,
        splits: str = "train",
        min_occ_length: int = 10,
        max_sequences: int = 0,
        max_samples: int = 0,
        depth_noise_sigma: float = 0.10,
        clip_log_abs: float = 0.35,
        history_len: int = 8,
        min_camera_motion: float = 0.0,
        hard_only: bool = False,
        preload_patches: bool = False,
        causal_mode: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.splits = [s.strip() for s in splits.split(",") if s.strip()]
        self.min_occ_length = int(min_occ_length)
        self.depth_noise_sigma = float(depth_noise_sigma)
        self.clip_log_abs = float(clip_log_abs)
        self.history_len = int(history_len)
        self.min_camera_motion = float(min_camera_motion)
        self.hard_only = bool(hard_only)
        self.preload_patches = bool(preload_patches)
        self.causal_mode = bool(causal_mode)
        self.rng = np.random.default_rng(seed)
        self._frame_cache: Dict[str, np.ndarray] = {}
        self._frame_cache_max = 256

        self.samples: List[Dict] = []
        sequence_paths = discover_sequences(self.data_root, self.splits)
        if max_sequences > 0:
            sequence_paths = sequence_paths[:max_sequences]

        for seq_path in sequence_paths:
            seq = load_sequence(seq_path)
            queries = find_reentry_queries(seq, min_occ_length=self.min_occ_length)
            if not queries:
                continue

            trajs_2d = seq["trajs_2d"]
            trajs_3d = seq["trajs_3d"]
            visibs = seq["visibs"]
            intrinsics = seq["intrinsics"]
            extrinsics = seq["extrinsics"]

            for q in queries:
                t_q = q.query_frame
                t_re = q.reentry_frame
                i = q.point_idx

                hold_3d = trajs_3d[t_q, i].astype(np.float32)
                gt_re_3d = trajs_3d[t_re, i].astype(np.float32)
                pt_cam = extrinsics[t_q][:3, :3] @ hold_3d + extrinsics[t_q][:3, 3]
                z_depth = float(pt_cam[2])
                if not np.isfinite(z_depth) or z_depth <= 1e-6:
                    continue

                e_rel = extrinsics[t_re] @ np.linalg.inv(extrinsics[t_q])
                camera_motion = float(np.linalg.norm(e_rel[:3, :3] - np.eye(3)))
                hard_mask = bool(q.occ_length >= 20 and camera_motion >= 0.30)
                if camera_motion < self.min_camera_motion:
                    continue
                if self.hard_only and not hard_mask:
                    continue

                self.samples.append(
                    {
                        "seq_path": str(seq_path),
                        "point_idx": int(i),
                        "query_frame": int(t_q),
                        "reentry_frame": int(t_re),
                        "occ_length": int(q.occ_length),
                        "camera_motion_rotation": camera_motion,
                        "query_xy": trajs_2d[t_q, i].astype(np.float32),
                        "reentry_xy": trajs_2d[t_re, i].astype(np.float32),
                        "hold_3d": hold_3d,
                        "gt_re_3d": gt_re_3d,
                        "query_intrinsics": intrinsics[t_q].astype(np.float32),
                        "query_extrinsics": extrinsics[t_q].astype(np.float32),
                        "reentry_intrinsics": intrinsics[t_re].astype(np.float32),
                        "reentry_extrinsics": extrinsics[t_re].astype(np.float32),
                        "query_depth": z_depth,
                        "reentry_visible": float(visibs[t_re, i]),
                        "hard_mask": float(hard_mask),
                        "trajs_2d": trajs_2d[:, i].astype(np.float32),
                        "trajs_3d": trajs_3d[:, i].astype(np.float32),
                        "visibs": visibs[:, i].astype(np.float32),
                        "extrinsics_all": extrinsics.astype(np.float32),
                        "depth_noise_eps": 0.0,
                    }
                )

                if max_samples > 0 and len(self.samples) >= max_samples:
                    break

            if max_samples > 0 and len(self.samples) >= max_samples:
                break

        if self.preload_patches and self.samples:
            self._preload_all_patches()

        if self.samples and self.depth_noise_sigma > 0:
            eps_all = self.rng.normal(0.0, self.depth_noise_sigma, size=len(self.samples)).astype(np.float32)
            eps_all = np.clip(eps_all, -self.clip_log_abs, self.clip_log_abs)
            for item, eps in zip(self.samples, eps_all):
                item["depth_noise_eps"] = float(eps)

    def __len__(self) -> int:
        return len(self.samples)

    def _get_cached_rgb_frame(self, seq_path: Path, frame_idx: int) -> Optional[np.ndarray]:
        key = f"{seq_path}:{frame_idx}"
        img = self._frame_cache.get(key)
        if img is not None:
            return img
        img = _load_rgb_frame(seq_path, frame_idx)
        if img is None:
            return None
        if len(self._frame_cache) >= self._frame_cache_max:
            self._frame_cache.pop(next(iter(self._frame_cache)))
        self._frame_cache[key] = img
        return img

    def _build_patch_triplet(
        self,
        item: Dict,
        reentry_center_xy: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        zero_patch = np.zeros((3, 64, 64), dtype=np.float32)
        query_patch = zero_patch.copy()
        reentry_patch = zero_patch.copy()
        query_prev_patch = zero_patch.copy()

        seq_path = Path(item["seq_path"])
        t_q = int(item["query_frame"])
        img_q = self._get_cached_rgb_frame(seq_path, t_q)
        if img_q is not None:
            query_patch = _extract_patch_from_xy(img_q, item["query_xy"])

        img_re = self._get_cached_rgb_frame(seq_path, int(item["reentry_frame"]))
        if img_re is not None:
            center_xy = item["reentry_xy"] if reentry_center_xy is None else reentry_center_xy
            reentry_patch = _extract_patch_from_xy(img_re, center_xy)

        t_prev = max(0, t_q - 1)
        while t_prev > 0 and item["visibs"][t_prev] < 0.5:
            t_prev -= 1
        if t_prev != t_q:
            img_prev = self._get_cached_rgb_frame(seq_path, t_prev)
            if img_prev is not None and np.all(np.isfinite(item["trajs_2d"][t_prev])):
                query_prev_patch = _extract_patch_from_xy(img_prev, item["trajs_2d"][t_prev])

        return query_patch, reentry_patch, query_prev_patch

    def _preload_all_patches(self) -> None:
        for item in self.samples:
            query_patch, reentry_patch, query_prev_patch = self._build_patch_triplet(item)
            item["query_patch_cached"] = query_patch.astype(np.float16, copy=False)
            item["reentry_patch_cached"] = reentry_patch.astype(np.float16, copy=False)
            item["query_prev_patch_cached"] = query_prev_patch.astype(np.float16, copy=False)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.samples[index]
        eps = float(item.get("depth_noise_eps", 0.0))

        noisy_depth = float(item["query_depth"] * math.exp(eps))
        lifted_world = _lift_points_to_world(
            item["query_xy"][None, :],
            np.array([noisy_depth], dtype=np.float32),
            item["query_intrinsics"],
            item["query_extrinsics"],
        )[0].astype(np.float32)

        baseline_reproj = project_3d_to_2d(
            lifted_world,
            item["reentry_intrinsics"],
            item["reentry_extrinsics"],
        ).astype(np.float32)
        baseline_reproj_error_px = float(np.linalg.norm(baseline_reproj - item["reentry_xy"]))

        if self.causal_mode:
            feature = np.concatenate(
                [
                    item["query_xy"] / 1024.0,
                    baseline_reproj / 1024.0,
                    np.array(
                        [
                            noisy_depth / 10.0,
                            item["occ_length"] / 100.0,
                            item["camera_motion_rotation"],
                            baseline_reproj_error_px / 256.0,
                        ],
                        dtype=np.float32,
                    ),
                    lifted_world / 10.0,
                ],
                axis=0,
            ).astype(np.float32)
        else:
            feature = np.concatenate(
                [
                    item["query_xy"] / 1024.0,
                    item["reentry_xy"] / 1024.0,
                    np.array(
                        [
                            noisy_depth / 10.0,
                            item["occ_length"] / 100.0,
                            item["camera_motion_rotation"],
                            item["reentry_visible"],
                            item["hard_mask"],
                            baseline_reproj_error_px / 256.0,
                        ],
                        dtype=np.float32,
                    ),
                    lifted_world / 10.0,
                ],
                axis=0,
            ).astype(np.float32)

        t_q = int(item["query_frame"])
        start = max(0, t_q - self.history_len + 1)
        history_xy = []
        history_xy_delta = []
        history_world = []
        history_world_delta = []
        history_vis = []
        history_camrot = []
        # Find first finite values for initialization
        first_xy = item["trajs_2d"][start].copy()
        first_world = item["trajs_3d"][start].copy()
        for t_init in range(start, t_q + 1):
            if np.all(np.isfinite(item["trajs_2d"][t_init])):
                first_xy = item["trajs_2d"][t_init].copy()
                first_world = item["trajs_3d"][t_init].copy()
                break
        prev_xy = first_xy.copy()
        prev_world = first_world.copy()
        prev_extr = item["extrinsics_all"][t_q]
        for t in range(start, t_q + 1):
            xy = item["trajs_2d"][t].copy()
            world = item["trajs_3d"][t].copy()
            # Replace inf/nan with last valid value
            if not np.all(np.isfinite(xy)):
                xy = prev_xy.copy()
            if not np.all(np.isfinite(world)):
                world = prev_world.copy()
            history_xy.append(xy / 1024.0)
            delta_xy = xy - prev_xy
            delta_xy = np.where(np.isfinite(delta_xy), delta_xy, 0.0)
            history_xy_delta.append(delta_xy / 128.0)
            if self.causal_mode:
                history_world.append(np.zeros(3, dtype=np.float32))
                history_world_delta.append(np.zeros(3, dtype=np.float32))
            else:
                history_world.append(world / 10.0)
                delta_w = world - prev_world
                delta_w = np.where(np.isfinite(delta_w), delta_w, 0.0)
                history_world_delta.append(delta_w / 2.0)
            history_vis.append([item["visibs"][t]])
            rel = item["extrinsics_all"][t] @ np.linalg.inv(prev_extr)
            history_camrot.append([float(np.linalg.norm(rel[:3, :3] - np.eye(3)))])
            prev_xy = xy
            prev_world = world
        while len(history_xy) < self.history_len:
            history_xy.insert(0, history_xy[0].copy())
            history_xy_delta.insert(0, np.zeros_like(history_xy_delta[0]))
            history_world.insert(0, history_world[0].copy())
            history_world_delta.insert(0, np.zeros_like(history_world_delta[0]))
            history_vis.insert(0, history_vis[0].copy())
            history_camrot.insert(0, history_camrot[0].copy())

        history_xy = np.asarray(history_xy, dtype=np.float32)
        history_xy_delta = np.asarray(history_xy_delta, dtype=np.float32)
        history_world = np.asarray(history_world, dtype=np.float32)
        history_world_delta = np.asarray(history_world_delta, dtype=np.float32)
        history_vis = np.asarray(history_vis, dtype=np.float32)
        history_camrot = np.asarray(history_camrot, dtype=np.float32)

        target_world = item["gt_re_3d"].astype(np.float32)
        target_delta = (target_world - lifted_world).astype(np.float32)

        if "query_patch_cached" in item:
            query_patch = item["query_patch_cached"].astype(np.float32, copy=False)
            reentry_patch = item["reentry_patch_cached"].astype(np.float32, copy=False)
            query_prev_patch = item["query_prev_patch_cached"].astype(np.float32, copy=False)
        else:
            if self.causal_mode:
                query_patch, reentry_patch, query_prev_patch = self._build_patch_triplet(item, reentry_center_xy=baseline_reproj)
            else:
                query_patch, reentry_patch, query_prev_patch = self._build_patch_triplet(item)

        return {
            "features": torch.from_numpy(feature),
            "history_xy": torch.from_numpy(history_xy),
            "history_xy_delta": torch.from_numpy(history_xy_delta),
            "history_world": torch.from_numpy(history_world),
            "history_world_delta": torch.from_numpy(history_world_delta),
            "history_vis": torch.from_numpy(history_vis),
            "history_camrot": torch.from_numpy(history_camrot),
            "input_world": torch.from_numpy(lifted_world),
            "target_world": torch.from_numpy(target_world),
            "target_delta": torch.from_numpy(target_delta),
            "query_xy": torch.from_numpy(item["query_xy"]),
            "reentry_xy": torch.from_numpy(item["reentry_xy"]),
            "reentry_visible": torch.tensor(item["reentry_visible"], dtype=torch.float32),
            "hard_mask": torch.tensor(item["hard_mask"], dtype=torch.float32),
            "baseline_reproj_error_px": torch.tensor(baseline_reproj_error_px, dtype=torch.float32),
            "reentry_intrinsics": torch.from_numpy(item["reentry_intrinsics"]),
            "reentry_extrinsics": torch.from_numpy(item["reentry_extrinsics"]),
            "query_patch": torch.from_numpy(query_patch),
            "reentry_patch": torch.from_numpy(reentry_patch),
            "query_prev_patch": torch.from_numpy(query_prev_patch),
        }
