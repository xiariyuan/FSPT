"""
TAP-Vid Kubric Dataset

Kubric是一个合成数据生成框架，提供完美的点追踪标注。
TAP-Vid使用Kubric生成的MOVi-E数据集进行训练。
"""

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Optional, Tuple, Dict, List
from pathlib import Path
from .augmentation import PointTrackingAugmentation, TemporalAugmentation
from .coord_utils import normalize_points_yx, normalize_query_points_tyx
try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    OmegaConf = None

# 尝试导入tensorflow-datasets
try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: tensorflow-datasets not available. TAPVidKubricDataset will have limited functionality.")


def _canonical_tfds_split(split: str) -> str:
    split = str(split).lower().strip()
    if split in ("val", "valid"):
        return "validation"
    if split in ("train", "training"):
        return "train"
    if split in ("test", "testing"):
        return "test"
    return split


def _parse_version_key(version_name: str) -> Tuple[int, int, int, str]:
    """Parse semantic-ish version folder names like '1.0.0' for sorting."""
    nums: List[int] = []
    for token in str(version_name).replace("-", ".").split("."):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            nums.append(int(token))
            continue
        digits = "".join(ch for ch in token if ch.isdigit())
        nums.append(int(digits) if digits else -1)
    while len(nums) < 3:
        nums.append(-1)
    return nums[0], nums[1], nums[2], str(version_name)


def _pick_builder_dir(base_dir: Path) -> Optional[Path]:
    """
    Resolve a TFDS builder directory.

    A valid builder dir is either:
    - a directory containing dataset_info.json directly, or
    - a parent directory containing version subdirs, each with dataset_info.json.
    """
    info_file = base_dir / "dataset_info.json"
    if info_file.exists():
        return base_dir
    if not base_dir.exists() or not base_dir.is_dir():
        return None

    version_dirs = []
    try:
        for child in base_dir.iterdir():
            if child.is_dir() and (child / "dataset_info.json").exists():
                version_dirs.append(child)
    except OSError:
        return None

    if not version_dirs:
        return None
    version_dirs.sort(key=lambda p: _parse_version_key(p.name))
    return version_dirs[-1]


def _find_tfds_builder_dir(root: Path, tfds_name: str) -> Optional[Path]:
    """
    Find a usable TFDS builder directory from common layouts.

    Supports both canonical layout:
      <root>/movi_e/256x256/1.0.0/...
    and server layout seen in this project:
      <root>/tapvid_kubric/256x256/1.0.0/...
      <root>/256x256/1.0.0/...
    """
    parts = [p for p in str(tfds_name or "").strip().split("/") if p]
    dataset_name = parts[0] if parts else ""
    config_name = parts[1] if len(parts) > 1 else ""

    candidates: List[Path] = []
    if dataset_name and config_name:
        candidates.append(root / dataset_name / config_name)
    if config_name:
        candidates.extend(
            [
                root / config_name,
                root / "movi_e" / config_name,
                root / "tapvid_kubric" / config_name,
            ]
        )
    if dataset_name:
        candidates.append(root / dataset_name)
    candidates.extend([root / "movi_e", root / "tapvid_kubric", root])

    seen = set()
    for base in candidates:
        key = str(base)
        if key in seen:
            continue
        seen.add(key)
        builder_dir = _pick_builder_dir(base)
        if builder_dir is not None:
            return builder_dir
    return None


def _resolve_tfds_builder(root: Path, tfds_name: str):
    """Resolve TFDS builder via standard name, then fallback directory mode."""
    named_error: Optional[Exception] = None
    try:
        builder = tfds.builder(tfds_name, data_dir=str(root))
        return builder, "name", None
    except Exception as exc:
        named_error = exc

    builder_dir = _find_tfds_builder_dir(root, tfds_name)
    if builder_dir is not None and hasattr(tfds, "builder_from_directory"):
        builder = tfds.builder_from_directory(str(builder_dir))
        return builder, "directory", builder_dir

    raise RuntimeError(
        f"Failed to resolve TFDS builder for tfds_name={tfds_name!r}, root={root}. "
        "Tried tfds.builder(...) and common fallback directories "
        "(movi_e / tapvid_kubric / config-only layouts). "
        f"Original error: {named_error}"
    ) from named_error


def _load_tfds_dataset(
    *,
    tfds_name: str,
    split: str,
    root: Path,
    shuffle_files: bool,
    read_config=None,
):
    """Load TFDS split with fallback to builder_from_directory when layout is non-canonical."""
    builder, mode, builder_dir = _resolve_tfds_builder(root, tfds_name)
    if mode == "name":
        load_kwargs = dict(
            split=split,
            data_dir=str(root),
            shuffle_files=shuffle_files,
        )
        if read_config is not None:
            load_kwargs["read_config"] = read_config
        dataset = tfds.load(tfds_name, **load_kwargs)
        source = f"name:{tfds_name}"
        return dataset, source

    # directory fallback
    try:
        as_dataset_kwargs = dict(split=split, shuffle_files=shuffle_files)
        if read_config is not None:
            as_dataset_kwargs["read_config"] = read_config
        dataset = builder.as_dataset(**as_dataset_kwargs)
    except TypeError:
        try:
            dataset = builder.as_dataset(split=split, shuffle_files=shuffle_files)
        except TypeError:
            dataset = builder.as_dataset(split=split)
    source = f"directory:{builder_dir}"
    return dataset, source


def _build_tfds_read_config(
    *,
    low_memory: bool = False,
    deterministic: bool = False,
    shuffle_seed: Optional[int] = None,
    override_buffer_size: Optional[int] = None,
):
    """
    Build TFDS ReadConfig with best-effort compatibility across TFDS versions.

    In low-memory mode we reduce tf.data parallelism/prefetch/autocache to avoid
    memory spikes under strict cgroup limits.
    """
    if not TF_AVAILABLE:
        return None

    read_kwargs = {}
    options = None

    try:
        options = tf.data.Options()
        if deterministic:
            try:
                options.experimental_deterministic = True
            except Exception:
                pass
        if low_memory:
            try:
                options.autotune.enabled = False
            except Exception:
                pass
            try:
                options.threading.private_threadpool_size = 1
                options.threading.max_intra_op_parallelism = 1
            except Exception:
                try:
                    options.experimental_threading.private_threadpool_size = 1
                    options.experimental_threading.max_intra_op_parallelism = 1
                except Exception:
                    pass
    except Exception:
        options = None

    if options is not None:
        read_kwargs["options"] = options

    if shuffle_seed is not None:
        read_kwargs["shuffle_seed"] = int(shuffle_seed)
        read_kwargs["shuffle_reshuffle_each_iteration"] = True

    if low_memory:
        read_kwargs["try_autocache"] = False
        read_kwargs["skip_prefetch"] = True
        read_kwargs["interleave_cycle_length"] = 1
        read_kwargs["interleave_block_length"] = 1
        read_kwargs["num_parallel_calls_for_decode"] = 1
        read_kwargs["num_parallel_calls_for_interleave_files"] = 1
        if override_buffer_size is not None:
            try:
                buf = int(override_buffer_size)
                if buf > 0:
                    read_kwargs["override_buffer_size"] = buf
            except (TypeError, ValueError):
                pass
        elif TF_AVAILABLE:
            # Default 1MiB TFRecord read buffer in low-memory mode.
            read_kwargs["override_buffer_size"] = 1 << 20

    try:
        return tfds.ReadConfig(**read_kwargs)
    except TypeError:
        # Older TFDS may not support some kwargs. Retry with supported keys only.
        try:
            import inspect

            sig = inspect.signature(tfds.ReadConfig)
            filtered = {k: v for k, v in read_kwargs.items() if k in sig.parameters}
            return tfds.ReadConfig(**filtered)
        except Exception:
            return None


def _bilinear_sample_flow(flow: np.ndarray, y: float, x: float) -> np.ndarray:
    """Bilinear sample a single optical-flow vector at float (y, x). Returns (dx, dy)."""
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"Expected flow with shape (H, W, 2), got {flow.shape}.")
    H, W, _ = flow.shape
    # Clamp to valid range for safe indexing.
    y = float(np.clip(y, 0.0, max(0.0, H - 1.0)))
    x = float(np.clip(x, 0.0, max(0.0, W - 1.0)))
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    y1 = min(y0 + 1, H - 1)
    x1 = min(x0 + 1, W - 1)
    wy = y - y0
    wx = x - x0
    f00 = flow[y0, x0]
    f01 = flow[y0, x1]
    f10 = flow[y1, x0]
    f11 = flow[y1, x1]
    f0 = (1.0 - wx) * f00 + wx * f01
    f1 = (1.0 - wx) * f10 + wx * f11
    return (1.0 - wy) * f0 + wy * f1


def _bilinear_sample_flow_batch(flow: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """
    Vectorized bilinear sampling for multiple points on one flow frame.

    Args:
        flow: (H, W, 2), values as (dx, dy)
        ys: (N,) float y coordinates
        xs: (N,) float x coordinates

    Returns:
        sampled flow: (N, 2)
    """
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"Expected flow with shape (H, W, 2), got {flow.shape}.")

    ys = np.asarray(ys, dtype=np.float32)
    xs = np.asarray(xs, dtype=np.float32)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    H, W, _ = flow.shape
    ys = np.clip(ys, 0.0, max(0.0, H - 1.0))
    xs = np.clip(xs, 0.0, max(0.0, W - 1.0))

    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)

    wy = (ys - y0.astype(np.float32)).reshape(-1, 1)
    wx = (xs - x0.astype(np.float32)).reshape(-1, 1)

    f00 = flow[y0, x0].astype(np.float32, copy=False)
    f01 = flow[y0, x1].astype(np.float32, copy=False)
    f10 = flow[y1, x0].astype(np.float32, copy=False)
    f11 = flow[y1, x1].astype(np.float32, copy=False)

    f0 = (1.0 - wx) * f00 + wx * f01
    f1 = (1.0 - wx) * f10 + wx * f11
    return (1.0 - wy) * f0 + wy * f1


def _decode_kubric_flow(flow: Optional[np.ndarray], flow_range: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Decode Kubric uint16 flow to pixel displacement using metadata ranges.

    Kubric TFDS stores forward/backward flow as uint16 and provides min/max in:
      sample["metadata"]["forward_flow_range"] / ["backward_flow_range"].
    If range metadata is missing, we conservatively cast to float32.
    """
    if flow is None:
        return None
    arr = np.asarray(flow)
    if arr.size == 0:
        return arr.astype(np.float32, copy=False)

    if arr.dtype == np.uint16 and flow_range is not None:
        try:
            fr = np.asarray(flow_range, dtype=np.float32).reshape(-1)
            if fr.size >= 2:
                minv = float(fr[0])
                maxv = float(fr[1])
                if np.isfinite(minv) and np.isfinite(maxv) and maxv > minv:
                    return (arr.astype(np.float32) / 65535.0) * (maxv - minv) + minv
        except Exception:
            pass

    return arr.astype(np.float32, copy=False)


def _generate_sparse_tracks_from_kubric(
    *,
    forward_flow: Optional[np.ndarray],
    backward_flow: Optional[np.ndarray],
    segmentations: Optional[np.ndarray],
    num_points: int,
    rng: np.random.RandomState,
    sampling_strategy: str = "uniform",
    hard_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate sparse point tracks from Kubric TFDS fields.

    Uses ground-truth optical flow (forward/backward) to propagate point coordinates.
    Visibility is approximated using instance segmentations (if available): a point is
    considered visible at frame t if it stays in-bounds and remains on the same instance id.

    Returns:
        query_points: (N, 3) float32 with (t, y_norm, x_norm)
        target_points: (N, T, 2) float32 with (y_norm, x_norm)
        occluded: (N, T) bool
    """
    if forward_flow is None or backward_flow is None:
        raise ValueError("forward_flow/backward_flow are required to generate tracks from TFDS.")

    forward_flow = np.asarray(forward_flow)
    backward_flow = np.asarray(backward_flow)
    if forward_flow.ndim != 4 or forward_flow.shape[-1] != 2:
        raise ValueError(f"Invalid forward_flow shape: {forward_flow.shape}")
    if backward_flow.shape != forward_flow.shape:
        raise ValueError(f"backward_flow shape {backward_flow.shape} != forward_flow shape {forward_flow.shape}")

    T, H, W, _ = forward_flow.shape
    if T <= 0 or H <= 0 or W <= 0:
        raise ValueError(f"Invalid TFDS flow shape: {forward_flow.shape}")

    seg = None
    if segmentations is not None:
        seg = np.asarray(segmentations)
        if seg.ndim == 4 and seg.shape[-1] == 1:
            seg = seg[..., 0]
        if seg.shape[:3] != (T, H, W):
            seg = None

    sampling_strategy = str(sampling_strategy or "uniform").lower().strip()
    hard_fraction = float(hard_fraction) if hard_fraction is not None else 0.5
    hard_fraction = max(0.0, min(1.0, hard_fraction))

    valid_coords_by_t: Optional[List[Optional[np.ndarray]]] = None
    if seg is not None:
        valid_coords_by_t = []
        for t in range(T):
            coords = np.argwhere(seg[t] != 0)
            valid_coords_by_t.append(coords if coords.size > 0 else None)

    def _sample_points(k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Sample (t0, y0, x0) in pixel coords; prefer non-background pixels when segmentations available.
        t0 = rng.randint(0, T, size=(k,))
        ys = np.empty((k,), dtype=np.float32)
        xs = np.empty((k,), dtype=np.float32)
        for i in range(k):
            t = int(t0[i])
            if valid_coords_by_t is not None:
                coords = valid_coords_by_t[t]
                if coords is not None:
                    yy, xx = coords[int(rng.randint(0, coords.shape[0]))]
                    # Keep sub-pixel jitter inside the valid image domain.  The
                    # previous yy+U[0,1) rule could sample beyond H-1/W-1 on
                    # the final row or column, making a query incorrectly
                    # occluded at its own query frame.
                    ys[i] = min(float(yy) + float(rng.rand()), float(H - 1))
                    xs[i] = min(float(xx) + float(rng.rand()), float(W - 1))
                    continue
            ys[i] = float(rng.rand() * (H - 1))
            xs[i] = float(rng.rand() * (W - 1))
        return t0.astype(np.int64), ys, xs

    def _track_batch(t0_arr: np.ndarray, y0_arr: np.ndarray, x0_arr: np.ndarray):
        """
        Vectorized propagation for a batch of query points.
        """
        t0_arr = np.asarray(t0_arr, dtype=np.int64)
        y0_arr = np.asarray(y0_arr, dtype=np.float32)
        x0_arr = np.asarray(x0_arr, dtype=np.float32)
        N = int(t0_arr.shape[0])

        coords = np.full((N, T, 2), np.nan, dtype=np.float32)
        idx = np.arange(N, dtype=np.int64)
        coords[idx, t0_arr, 0] = y0_arr
        coords[idx, t0_arr, 1] = x0_arr

        # Forward propagation: t -> t+1.
        y_f = y0_arr.copy()
        x_f = x0_arr.copy()
        for t in range(T - 1):
            active = (t0_arr <= t)
            if not np.any(active):
                continue
            delta = _bilinear_sample_flow_batch(forward_flow[t], y_f[active], x_f[active])  # (dx, dy)
            x_f[active] += delta[:, 0]
            y_f[active] += delta[:, 1]
            coords[active, t + 1, 0] = y_f[active]
            coords[active, t + 1, 1] = x_f[active]

        # Backward propagation: t -> t-1.
        y_b = y0_arr.copy()
        x_b = x0_arr.copy()
        for t in range(T - 1, 0, -1):
            active = (t0_arr >= t)
            if not np.any(active):
                continue
            delta = _bilinear_sample_flow_batch(backward_flow[t], y_b[active], x_b[active])  # (dx, dy)
            x_b[active] += delta[:, 0]
            y_b[active] += delta[:, 1]
            coords[active, t - 1, 0] = y_b[active]
            coords[active, t - 1, 1] = x_b[active]

        in_bounds = (
            (coords[:, :, 0] >= 0.0) & (coords[:, :, 0] <= (H - 1.0)) &
            (coords[:, :, 1] >= 0.0) & (coords[:, :, 1] <= (W - 1.0))
        )
        occ = ~in_bounds

        # Optional segmentation-consistency occlusion.
        if seg is not None:
            y_q = np.clip(np.rint(y0_arr).astype(np.int64), 0, H - 1)
            x_q = np.clip(np.rint(x0_arr).astype(np.int64), 0, W - 1)
            obj_ids = seg[t0_arr, y_q, x_q].astype(np.int32)
            trackable = obj_ids != 0
            if np.any(trackable):
                for t in range(T):
                    valid = in_bounds[:, t] & trackable
                    if not np.any(valid):
                        continue
                    yy = np.clip(np.rint(coords[valid, t, 0]).astype(np.int64), 0, H - 1)
                    xx = np.clip(np.rint(coords[valid, t, 1]).astype(np.int64), 0, W - 1)
                    occ[valid, t] |= (seg[t, yy, xx].astype(np.int32) != obj_ids[valid])

        # Normalize coordinates to [0, 1].
        coords[:, :, 0] = np.clip(coords[:, :, 0] / float(H), 0.0, 1.0)
        coords[:, :, 1] = np.clip(coords[:, :, 1] / float(W), 0.0, 1.0)
        # Fill rare NaN/Inf (shouldn't happen in normal cases) to keep tensors finite.
        coords = np.nan_to_num(coords, nan=0.0, posinf=1.0, neginf=0.0)
        return coords.astype(np.float32, copy=False), occ.astype(bool, copy=False)

    if num_points <= 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, T, 2), dtype=np.float32),
            np.zeros((0, T), dtype=bool),
        )

    if sampling_strategy not in ("uniform", "random", "occlusion_balanced", "occ_balanced", "balanced_occlusion"):
        sampling_strategy = "uniform"

    # Candidate pool for optional occlusion-balanced sampling.
    pool = int(num_points) if sampling_strategy == "uniform" else int(min(max(num_points * 4, num_points + 64), 1024))
    t0s, ys, xs = _sample_points(pool)
    tracks, occs = _track_batch(t0s, ys, xs)

    # Build query_points from the (t0, coords[t0]).
    query_points = np.zeros((pool, 3), dtype=np.float32)
    query_points[:, 0] = t0s.astype(np.float32)
    # coords at query time are already normalized.
    query_points[:, 1] = tracks[np.arange(pool), t0s, 0]
    query_points[:, 2] = tracks[np.arange(pool), t0s, 1]

    if sampling_strategy == "uniform":
        chosen = np.arange(pool, dtype=np.int64)
        if pool > num_points:
            chosen = rng.choice(pool, int(num_points), replace=False)
    else:
        q_t = np.clip(np.round(query_points[:, 0]).astype(np.int64), 0, T - 1)
        visible_at_query = ~occs[np.arange(pool), q_t]
        occluded_any = occs.any(axis=1)
        hard_mask = visible_at_query & occluded_any
        easy_mask = visible_at_query & (~occluded_any)
        hard_idx = np.nonzero(hard_mask)[0]
        easy_idx = np.nonzero(easy_mask)[0]
        vis_idx = np.nonzero(visible_at_query)[0]

        k = int(num_points)
        k_hard = int(round(k * hard_fraction))
        k_easy = int(k - k_hard)
        chosen_list: List[int] = []
        if hard_idx.size > 0 and k_hard > 0:
            take = min(int(hard_idx.size), int(k_hard))
            chosen_list.extend(rng.choice(hard_idx, take, replace=False).tolist())
        if easy_idx.size > 0 and k_easy > 0:
            take = min(int(easy_idx.size), int(k_easy))
            chosen_list.extend(rng.choice(easy_idx, take, replace=False).tolist())
        remaining = k - len(chosen_list)
        if remaining > 0:
            if vis_idx.size > 0:
                chosen_list.extend(rng.choice(vis_idx, remaining, replace=vis_idx.size < remaining).tolist())
            else:
                chosen_list.extend(rng.choice(pool, remaining, replace=pool < remaining).tolist())
        chosen = np.array(chosen_list[:k], dtype=np.int64)

    return (
        query_points[chosen].astype(np.float32),
        tracks[chosen].astype(np.float32),
        occs[chosen].astype(bool),
    )


class TAPVidKubricDataset(Dataset):
    """
    TAP-Vid Kubric训练数据集
    
    可以从以下来源加载:
    1. TAP-Vid官方Kubric标注pickle文件（推荐）
    2. TensorFlow Datasets (需要tensorflow-datasets，仅用于调试)
    3. 预处理的pickle文件或逐视频标注文件
    
    Args:
        root: 数据集根目录
        split: 数据集划分 ('train', 'validation')
        num_frames: 采样帧数
        num_points: 每个视频采样的点数
        resolution: 目标分辨率 (H, W)
        augmentation: 是否进行数据增强
        annotation_file: 指定标注文件名（如 tapvid_kubric_train.pkl）
        use_tfds: 是否使用TensorFlow Datasets加载（调试用）
        allow_synthetic_tracks: 是否允许使用合成轨迹（默认False）
        max_retries: 单个样本失败时的最大重试次数
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        num_frames: int = 24,
        num_points: int = 256,
        resolution: Tuple[int, int] = (256, 256),
        augmentation: bool = True,
        annotation_file: Optional[str] = None,
        use_tfds: bool = False,
        allow_synthetic_tracks: bool = False,
        max_retries: int = 5,
        deterministic_sampling: bool = False,
        deterministic_seed: Optional[int] = None,
        sampling: Optional[object] = None,
        tfds_name: str = "movi_e/256x256",
    ):
        self.root = Path(root)
        self.split = split
        self.num_frames = num_frames
        self.num_points = num_points
        self.resolution = resolution
        self.deterministic_sampling = bool(deterministic_sampling)
        self.deterministic_seed = int(deterministic_seed) if deterministic_seed is not None else None
        self.aug_cfg = self._normalize_aug_config(augmentation)
        self.augmentation = self.aug_cfg.get('enabled', False)
        self.sampling_cfg = self._normalize_sampling_config(sampling)
        self.sampling_strategy = str(self.sampling_cfg.get('strategy', 'uniform')).lower().strip()
        self.sampling_hard_fraction = float(self.sampling_cfg.get('hard_fraction', 0.5) or 0.5)
        self.sampling_hard_fraction = max(0.0, min(1.0, self.sampling_hard_fraction))
        self.tfds_name = str(tfds_name)
        self.annotation_file = annotation_file
        self.allow_synthetic_tracks = allow_synthetic_tracks
        self.max_retries = max(0, int(max_retries))
        self.spatial_aug = None
        self.temporal_aug = None
        if self.augmentation:
            cfg = self.aug_cfg
            random_scale = cfg.get('random_scale', None)
            if isinstance(random_scale, list):
                random_scale = tuple(random_scale)
            crop_size = cfg.get('crop_size', self.resolution or (256, 256))
            random_rotation = float(cfg.get('random_rotation', 0.0) or 0.0)
            self.spatial_aug = PointTrackingAugmentation(
                random_crop=cfg.get('random_crop', True),
                random_flip=cfg.get('random_flip', True),
                color_jitter=cfg.get('color_jitter', 0.4),
                random_scale=random_scale,
                random_rotation=random_rotation,
                crop_size=crop_size,
            )

            temporal_cfg = cfg.get('temporal', {})
            if not isinstance(temporal_cfg, dict):
                temporal_cfg = {}
            temporal_enabled = bool(temporal_cfg.get('enabled', False))
            if any(k in cfg for k in ['random_reverse', 'random_speed', 'random_start']):
                temporal_enabled = True
            if temporal_enabled:
                random_speed = temporal_cfg.get('random_speed', cfg.get('random_speed', (0.5, 2.0)))
                if isinstance(random_speed, list):
                    random_speed = tuple(random_speed)
                self.temporal_aug = TemporalAugmentation(
                    random_reverse=temporal_cfg.get('random_reverse', cfg.get('random_reverse', True)),
                    random_speed=random_speed,
                    random_start=temporal_cfg.get('random_start', cfg.get('random_start', True)),
                )
        
        # 尝试加载数据
        self.samples = []
        self.data = None
        self.video_names = []
        self.data_format = None
        
        # 方式1: TAP-Vid官方或预处理的pickle文件
        anno_path = self._find_annotation_file()
        if anno_path is not None:
            print(f"Loading from pickle: {anno_path}")
            with open(anno_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                self.data = data
                self.video_names = sorted(list(self.data.keys()))
                self.data_format = 'dict'
            elif isinstance(data, list):
                self.samples = data
                self.data_format = 'list'
            else:
                raise ValueError(
                    f"Unsupported kubric pickle format: {type(data)}. "
                    "Expected dict or list."
                )
        
        # 方式2: 使用TFDS（调试用）
        elif use_tfds and TF_AVAILABLE:
            print(f"Loading from TensorFlow Datasets...")
            self._load_from_tfds()
            self.data_format = 'list'
        elif use_tfds and not TF_AVAILABLE:
            raise RuntimeError(
                "use_tfds=True but tensorflow-datasets is not available. "
                "Install tensorflow-datasets or provide official Kubric pickle files."
            )
        
        # 方式3: 从单独的视频文件加载
        else:
            print(f"Scanning for video files...")
            self._scan_video_files()
            self.data_format = 'list'

        if self.data_format == 'dict':
            num_samples = len(self.video_names)
        else:
            num_samples = len(self.samples)
        print(f"Loaded {num_samples} samples from TAP-Vid Kubric ({split})")
        if num_samples == 0:
            raise FileNotFoundError(
                "No Kubric samples found. "
                "Please ensure the TAP-Vid Kubric pickle is present (e.g., "
                "tapvid_kubric_train.pkl) or set use_tfds=true for debugging."
            )

    def _rng_for_sample(self, idx: int, retry: int = 0):
        """
        Deterministic RNG helper.

        This is useful for:
        - Reproducible debugging
        - Precomputing base tracks when query_points sampling must match.
        """
        if not self.deterministic_sampling:
            return np.random
        base = int(self.deterministic_seed) if self.deterministic_seed is not None else 0
        # Mix (seed, idx, retry) into a uint32-ish range.
        mixed = (base * 1000003 + int(idx) * 9176 + int(retry) * 1013) % (2**32)
        return np.random.RandomState(int(mixed))

    def _normalize_aug_config(self, augmentation) -> Dict:
        """统一增强配置"""
        if augmentation is None:
            return {'enabled': False}
        if isinstance(augmentation, bool):
            return {'enabled': augmentation}
        if OMEGACONF_AVAILABLE and OmegaConf.is_config(augmentation):
            cfg = OmegaConf.to_container(augmentation, resolve=True)
        elif isinstance(augmentation, dict):
            cfg = dict(augmentation)
        else:
            try:
                cfg = dict(augmentation)
            except Exception:
                cfg = {'enabled': True}
        cfg.setdefault('enabled', True)
        return cfg

    def _normalize_sampling_config(self, sampling) -> Dict:
        """Normalize point sampling config (training-only)."""
        if sampling is None:
            return {'strategy': 'uniform'}
        if isinstance(sampling, str):
            return {'strategy': sampling}
        if OMEGACONF_AVAILABLE and OmegaConf is not None:
            try:
                if OmegaConf.is_config(sampling):
                    cfg = OmegaConf.to_container(sampling, resolve=True)
                else:
                    cfg = sampling
            except Exception:
                cfg = sampling
        else:
            cfg = sampling
        if isinstance(cfg, dict):
            cfg = dict(cfg)
        else:
            try:
                cfg = dict(cfg)
            except Exception:
                cfg = {'strategy': 'uniform'}
        cfg.setdefault('strategy', 'uniform')
        return cfg

    def _ensure_query_visible(
        self,
        query_points: torch.Tensor,
        target_points: torch.Tensor,
        occluded: torch.Tensor,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """确保查询帧点可见，避免增强后出现不可见查询点。"""
        if query_points.numel() == 0 or target_points.numel() == 0 or occluded.numel() == 0:
            return query_points, target_points, occluded
        if occluded.dim() != 2:
            return query_points, target_points, occluded
        T = occluded.shape[1]
        query_t = query_points[:, 0].round().long().clamp(0, T - 1)
        point_idx = torch.arange(occluded.shape[0], device=occluded.device)
        visible_mask = ~occluded[point_idx, query_t]
        if visible_mask.all():
            return query_points, target_points, occluded
        valid_idx = torch.nonzero(visible_mask, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            return query_points, target_points, occluded
        if self.num_points is not None and self.num_points > 0:
            if valid_idx.numel() >= self.num_points:
                if rng is not None:
                    local = rng.choice(int(valid_idx.numel()), int(self.num_points), replace=False)
                    perm = torch.as_tensor(local, device=valid_idx.device, dtype=torch.long)
                else:
                    perm = torch.randperm(valid_idx.numel(), device=valid_idx.device)[:self.num_points]
                chosen = valid_idx[perm]
            else:
                if rng is not None:
                    local = rng.randint(0, int(valid_idx.numel()), size=(int(self.num_points),))
                    rand = torch.as_tensor(local, device=valid_idx.device, dtype=torch.long)
                else:
                    rand = torch.randint(0, valid_idx.numel(), (self.num_points,), device=valid_idx.device)
                chosen = valid_idx[rand]
            return query_points[chosen], target_points[chosen], occluded[chosen]
        return query_points[valid_idx], target_points[valid_idx], occluded[valid_idx]

    def _find_annotation_file(self) -> Optional[Path]:
        """查找Kubric标注文件"""
        if self.annotation_file:
            anno_path = self.root / self.annotation_file
            if anno_path.exists():
                return anno_path
            raise FileNotFoundError(
                f"Annotation file not found at {anno_path}. "
                "Please check data.train.annotation_file."
            )

        split = str(self.split).lower()
        if split in ['val', 'valid']:
            split = 'validation'

        candidates = [
            f'tapvid_kubric_{split}.pkl',
            'tapvid_kubric.pkl',
            f'kubric_{split}.pkl',
            'kubric.pkl',
        ]
        for name in candidates:
            path = self.root / name
            if path.exists():
                return path
        return None

    @staticmethod
    def _get_field(data: Dict, keys: List[str]):
        for key in keys:
            if key in data:
                return data[key]
        return None
        
    def _load_from_tfds(self):
        """从TensorFlow Datasets加载"""
        if not self.allow_synthetic_tracks:
            raise RuntimeError(
                "TFDS does not provide point tracks. "
                "Please preprocess Kubric to pickle or set allow_synthetic_tracks=true for debugging."
            )
        try:
            split = _canonical_tfds_split(self.split)
            ds, source = _load_tfds_dataset(
                tfds_name=self.tfds_name,
                split=split,
                root=self.root,
                shuffle_files=True,
            )
            print(f"Loading TFDS split={split} via {source}")

            # 转换为列表（较慢但便于索引）
            for sample in ds:
                self.samples.append(self._convert_tfds_sample(sample))
        except Exception as e:
            print(f"Failed to load from TFDS: {e}")
    
    def _convert_tfds_sample(self, sample: Dict) -> Dict:
        """转换TFDS样本格式"""
        video = sample['video'].numpy()  # (T, H, W, 3)
        if not self.allow_synthetic_tracks:
            raise RuntimeError("Synthetic tracks are disabled. Cannot convert TFDS sample.")

        metadata = sample.get("metadata", None)
        ff_range = None
        bf_range = None
        if metadata is not None:
            try:
                ff_range = metadata.get("forward_flow_range", None)
                bf_range = metadata.get("backward_flow_range", None)
                if ff_range is not None and hasattr(ff_range, "numpy"):
                    ff_range = ff_range.numpy()
                if bf_range is not None and hasattr(bf_range, "numpy"):
                    bf_range = bf_range.numpy()
            except Exception:
                ff_range = None
                bf_range = None

        fwd = sample.get("forward_flow", None).numpy() if "forward_flow" in sample else None
        bwd = sample.get("backward_flow", None).numpy() if "backward_flow" in sample else None
        fwd = _decode_kubric_flow(fwd, ff_range)
        bwd = _decode_kubric_flow(bwd, bf_range)

        rng = np.random.RandomState()
        query_points, target_points, occluded = _generate_sparse_tracks_from_kubric(
            forward_flow=fwd,
            backward_flow=bwd,
            segmentations=sample.get("segmentations", None).numpy() if "segmentations" in sample else None,
            num_points=int(min(self.num_points or 0, 256) if (self.num_points or 0) > 0 else 256),
            rng=rng,
            sampling_strategy=getattr(self, "sampling_strategy", "uniform"),
            hard_fraction=getattr(self, "sampling_hard_fraction", 0.5),
        )
        return {
            'video': video,
            'query_points': np.array(query_points, dtype=np.float32),
            'target_points': np.array(target_points, dtype=np.float32),
            'occluded': np.array(occluded, dtype=bool),
        }
    
    def _scan_video_files(self):
        """扫描视频文件"""
        video_dir = self.root / 'videos'
        if not video_dir.exists():
            print(f"No video directory found at {video_dir}")
            return
        
        # 查找所有视频文件
        video_files = list(video_dir.glob('*.mp4')) + list(video_dir.glob('*.avi'))
        
        for video_path in video_files:
            # 查找对应的标注文件
            anno_path = video_path.with_suffix('.pkl')
            if anno_path.exists():
                with open(anno_path, 'rb') as f:
                    anno = pickle.load(f)
                self.samples.append({
                    'video_path': str(video_path),
                    **anno,
                })
    
    def __len__(self):
        if self.data_format == 'dict':
            return len(self.video_names)
        return len(self.samples)
    
    def __getitem__(self, idx, retry: int = 0):
        rng = self._rng_for_sample(int(idx), int(retry))
        video_name = None
        video_path = None
        original_size = None
        query_generated = False
        if self.data_format == 'dict':
            video_name = self.video_names[idx]
            video_data = self.data[video_name]

            video = video_data.get('video')
            if video is None:
                video_path = video_data.get('video_path')
                if video_path is None:
                    raise KeyError("Missing 'video' or 'video_path' in kubric sample.")
                if not os.path.isabs(video_path):
                    video_path = str(self.root / video_path)
                try:
                    video = self._load_video(video_path)
                except Exception as e:
                    if retry < self.max_retries:
                        new_idx = int(rng.randint(0, len(self)))
                        return self.__getitem__(new_idx, retry=retry + 1)
                    raise e
            points = self._get_field(video_data, ['points', 'target_points', 'tracks'])
            occluded = self._get_field(video_data, ['occluded', 'occlusions', 'occ'])
            if occluded is None:
                visibility = self._get_field(video_data, ['visibility', 'vis'])
                if visibility is not None:
                    occluded = ~np.array(visibility, dtype=bool)
            query_points = self._get_field(video_data, ['query_points', 'queries'])

            if points is None:
                raise KeyError("Missing required field 'points' in kubric sample.")

            if isinstance(points, torch.Tensor):
                points = points.detach().cpu().numpy()
            if isinstance(occluded, torch.Tensor):
                occluded = occluded.detach().cpu().numpy()
            if isinstance(query_points, torch.Tensor):
                query_points = query_points.detach().cpu().numpy()
            points = np.array(points, dtype=np.float32)
            occluded = np.array(occluded, dtype=bool) if occluded is not None else None
            query_points = np.array(query_points, dtype=np.float32) if query_points is not None else None
        else:
            sample = self.samples[idx]
            
            # 加载视频
            if 'video_path' in sample:
                video_path = sample['video_path']
                if not os.path.isabs(video_path):
                    video_path = str(self.root / video_path)
                try:
                    video = self._load_video(video_path)
                except Exception as e:
                    if retry < self.max_retries:
                        new_idx = int(rng.randint(0, len(self)))
                        return self.__getitem__(new_idx, retry=retry + 1)
                    raise e
            else:
                video = sample['video']
            query_points = self._get_field(sample, ['query_points', 'queries'])
            target_points = self._get_field(sample, ['target_points', 'points', 'tracks'])
            occluded = self._get_field(sample, ['occluded', 'occlusions', 'occ'])
            if target_points is None:
                raise KeyError("Missing target_points/points/tracks in kubric sample.")
            if isinstance(query_points, torch.Tensor):
                query_points = query_points.detach().cpu().numpy()
            if isinstance(target_points, torch.Tensor):
                target_points = target_points.detach().cpu().numpy()
            if isinstance(occluded, torch.Tensor):
                occluded = occluded.detach().cpu().numpy()
            query_points = np.array(query_points, dtype=np.float32) if query_points is not None else None
            target_points = np.array(target_points, dtype=np.float32)
            occluded = np.array(occluded, dtype=bool) if occluded is not None else None

        if video is None:
            if retry < self.max_retries:
                new_idx = int(rng.randint(0, len(self)))
                return self.__getitem__(new_idx, retry=retry + 1)
            raise ValueError("Empty video sample encountered.")
        if isinstance(video, torch.Tensor):
            video = video.detach().cpu().numpy()
        video = np.asarray(video)
        if video.ndim == 1 and video.size > 0 and isinstance(video[0], np.ndarray):
            video = np.stack(video)
        if video.size == 0 or video.shape[0] == 0:
            if retry < self.max_retries:
                new_idx = int(rng.randint(0, len(self)))
                return self.__getitem__(new_idx, retry=retry + 1)
            raise ValueError("Empty video sample encountered.")
        
        T, H, W, C = video.shape
        # Keep original resolution as a tensor so DataLoader collate produces (B,2),
        # avoiding ambiguous list-of-tensors shapes like (heights, widths).
        original_size = torch.tensor([int(H), int(W)], dtype=torch.int32)

        if self.data_format == 'dict':
            if points.ndim == 3 and points.shape[0] == T and points.shape[1] != T:
                points = np.transpose(points, (1, 0, 2))
            if occluded is not None and occluded.ndim == 2 and occluded.shape[0] == T and occluded.shape[1] != T:
                occluded = np.transpose(occluded, (1, 0))
            # 归一化点坐标到[0, 1]（如果是像素坐标）
            points = normalize_points_yx(points, H, W)
            if query_points is None or query_points.shape[0] == 0:
                query_points = np.stack(
                    [
                        np.zeros(points.shape[0], dtype=np.float32),
                        points[:, 0, 0],
                        points[:, 0, 1],
                    ],
                    axis=1,
                )
                query_generated = True
            query_points = normalize_query_points_tyx(query_points, H, W)
            if occluded is None:
                occluded = np.zeros(points.shape[:2], dtype=bool)
            target_points = points
        else:
            if target_points.ndim == 3 and target_points.shape[0] == T and target_points.shape[1] != T:
                target_points = np.transpose(target_points, (1, 0, 2))
            if occluded is not None and occluded.ndim == 2 and occluded.shape[0] == T and occluded.shape[1] != T:
                occluded = np.transpose(occluded, (1, 0))
            target_points = normalize_points_yx(target_points, H, W)
            if query_points is None or query_points.shape[0] == 0:
                query_points = np.stack(
                    [
                        np.zeros(target_points.shape[0], dtype=np.float32),
                        target_points[:, 0, 0],
                        target_points[:, 0, 1],
                    ],
                    axis=1,
                )
                query_generated = True
            query_points = normalize_query_points_tyx(query_points, H, W)
            if occluded is None:
                occluded = np.zeros(target_points.shape[:2], dtype=bool)

        if target_points.shape[1] != T:
            min_t = min(target_points.shape[1], T)
            if min_t <= 0:
                if retry < self.max_retries:
                    new_idx = int(rng.randint(0, len(self)))
                    return self.__getitem__(new_idx, retry=retry + 1)
                raise ValueError("Invalid temporal alignment.")
            video = video[:min_t]
            target_points = target_points[:, :min_t]
            occluded = occluded[:, :min_t]
            T = min_t

        if target_points.shape[0] == 0:
            if retry < self.max_retries:
                new_idx = int(rng.randint(0, len(self)))
                return self.__getitem__(new_idx, retry=retry + 1)
            raise ValueError("No points available in kubric sample.")

        if (
            target_points.shape[0] != query_points.shape[0]
            or occluded.shape[0] != target_points.shape[0]
        ):
            min_len = min(target_points.shape[0], query_points.shape[0], occluded.shape[0])
            if min_len == 0:
                if retry < self.max_retries:
                    new_idx = int(rng.randint(0, len(self)))
                    return self.__getitem__(new_idx, retry=retry + 1)
                raise ValueError("No valid query points remain after alignment.")
            target_points = target_points[:min_len]
            occluded = occluded[:min_len]
            query_points = query_points[:min_len]
        
        # 采样帧（确保至少包含一个查询帧）
        if self.num_frames > 0 and T > self.num_frames:
            if query_points.shape[0] > 0:
                anchor_idx = int(rng.randint(0, query_points.shape[0]))
                anchor_t = int(np.round(query_points[anchor_idx, 0]))
                anchor_t = int(np.clip(anchor_t, 0, T - 1))
                min_start = max(0, anchor_t - self.num_frames + 1)
                max_start = min(anchor_t, T - self.num_frames)
                if max_start < min_start:
                    min_start = max_start = max(0, min(anchor_t, T - self.num_frames))
                start_idx = int(rng.randint(min_start, max_start + 1))
            else:
                start_idx = int(rng.randint(0, T - self.num_frames + 1))
            frame_indices = np.arange(start_idx, start_idx + self.num_frames)
        else:
            start_idx = 0
            frame_indices = np.arange(T)
        
        video = video[frame_indices]
        target_points = target_points[:, frame_indices]
        occluded = occluded[:, frame_indices]
        
        # 如果查询点为空，使用窗口起始帧位置构造查询点
        if query_points.shape[0] == 0 and target_points.shape[0] > 0:
            anchor_t = float(start_idx)
            query_points = np.stack(
                [
                    np.full(target_points.shape[0], anchor_t, dtype=np.float32),
                    target_points[:, 0, 0],
                    target_points[:, 0, 1],
                ],
                axis=1,
            )

        # 更新查询帧索引，并过滤不在窗口内的点
        new_query_t = query_points[:, 0] - start_idx
        valid_mask = (new_query_t >= 0) & (new_query_t < len(frame_indices))
        if valid_mask.any():
            query_points = query_points[valid_mask]
            target_points = target_points[valid_mask]
            occluded = occluded[valid_mask]
            new_query_t = new_query_t[valid_mask]
            query_points[:, 0] = new_query_t
        else:
            # 兜底：如果没有点落入窗口，保持原始点并裁剪时间索引
            if target_points.shape[0] == 0:
                if retry < self.max_retries:
                    new_idx = int(rng.randint(0, len(self)))
                    return self.__getitem__(new_idx, retry=retry + 1)
                raise ValueError("No valid query points remain after temporal cropping.")
            new_query_t = np.clip(new_query_t, 0, len(frame_indices) - 1)
            query_points[:, 0] = new_query_t
            t_idx = new_query_t.astype(np.int64)
            query_points[:, 1] = target_points[np.arange(target_points.shape[0]), t_idx, 0]
            query_points[:, 2] = target_points[np.arange(target_points.shape[0]), t_idx, 1]

        if (
            target_points.shape[0] != query_points.shape[0]
            or occluded.shape[0] != target_points.shape[0]
        ):
            min_len = min(target_points.shape[0], query_points.shape[0], occluded.shape[0])
            if min_len == 0:
                if retry < self.max_retries:
                    new_idx = int(rng.randint(0, len(self)))
                    return self.__getitem__(new_idx, retry=retry + 1)
                raise ValueError("No valid query points remain after alignment.")
            target_points = target_points[:min_len]
            occluded = occluded[:min_len]
            query_points = query_points[:min_len]
        
        # 采样点（保证训练时N固定）
        N = target_points.shape[0]
        if self.num_points is not None and self.num_points > 0:
            if N == 0:
                if retry < self.max_retries:
                    new_idx = int(rng.randint(0, len(self)))
                    return self.__getitem__(new_idx, retry=retry + 1)
                raise ValueError("No valid query points remain after temporal cropping.")

            k = int(self.num_points)
            strategy = getattr(self, "sampling_strategy", "uniform")
            if strategy in ("uniform", "random", None, ""):
                if N >= k:
                    indices = rng.choice(N, k, replace=False)
                else:
                    indices = rng.choice(N, k, replace=True) if N > 0 else np.array([], dtype=int)
            elif strategy in ("occlusion_balanced", "occ_balanced", "balanced_occlusion"):
                try:
                    q_t = np.round(query_points[:, 0]).astype(np.int64)
                    q_t = np.clip(q_t, 0, occluded.shape[1] - 1)
                    point_idx = np.arange(N)
                    visible_at_query = ~occluded[point_idx, q_t]
                    occluded_any = occluded.any(axis=1)

                    # Hard points: visible at query but occluded at some other time.
                    hard_mask = visible_at_query & occluded_any
                    easy_mask = visible_at_query & (~occluded_any)
                    hard_idx = np.nonzero(hard_mask)[0]
                    easy_idx = np.nonzero(easy_mask)[0]
                    vis_idx = np.nonzero(visible_at_query)[0]

                    hard_fraction = float(getattr(self, "sampling_hard_fraction", 0.5) or 0.5)
                    hard_fraction = max(0.0, min(1.0, hard_fraction))
                    k_hard = int(round(k * hard_fraction))
                    k_easy = int(k - k_hard)

                    chosen = []
                    if hard_idx.size > 0 and k_hard > 0:
                        chosen.append(rng.choice(hard_idx, k_hard, replace=(hard_idx.size < k_hard)))
                    else:
                        k_easy = k

                    if k_easy > 0:
                        pool = easy_idx if easy_idx.size > 0 else (vis_idx if vis_idx.size > 0 else np.arange(N))
                        chosen.append(rng.choice(pool, k_easy, replace=(pool.size < k_easy)))

                    if len(chosen) > 0:
                        indices = np.concatenate(chosen, axis=0)
                    else:
                        indices = rng.choice(N, k, replace=(N < k)) if N > 0 else np.array([], dtype=int)
                    if indices.size > 1:
                        rng.shuffle(indices)
                except Exception:
                    indices = rng.choice(N, k, replace=(N < k)) if N > 0 else np.array([], dtype=int)
            else:
                # Unknown strategy -> uniform.
                if N >= k:
                    indices = rng.choice(N, k, replace=False)
                else:
                    indices = rng.choice(N, k, replace=True) if N > 0 else np.array([], dtype=int)

            if indices.size > 0:
                query_points = query_points[indices]
                target_points = target_points[indices]
                occluded = occluded[indices]
        
        # 转换为PyTorch tensor（用于统一增强与后处理）
        video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
        if video.max() > 1.5:
            video = video / 255.0
        query_points = torch.from_numpy(query_points).float()
        query_points[:, 0] = torch.clamp(query_points[:, 0], 0, video.shape[0] - 1)
        query_points[:, 1:3] = torch.clamp(query_points[:, 1:3], 0.0, 1.0)
        target_points = torch.from_numpy(target_points).float()
        occluded = torch.from_numpy(occluded.astype(bool))

        # 数据增强
        if self.augmentation and self.spatial_aug is not None:
            outputs = self.spatial_aug(video, target_points, occluded, query_points)
            if len(outputs) == 4:
                video, target_points, occluded, query_points = outputs
            else:
                video, target_points, occluded = outputs
        if self.augmentation and self.temporal_aug is not None:
            video, target_points, occluded, query_points = self.temporal_aug(
                video, target_points, occluded, query_points
            )
        query_points[:, 0] = torch.clamp(query_points[:, 0], 0, video.shape[0] - 1)
        if self.augmentation or query_generated:
            query_points, target_points, occluded = self._ensure_query_visible(
                query_points,
                target_points,
                occluded,
                rng=rng if self.deterministic_sampling else None,
            )

        # 调整分辨率
        if self.resolution is not None:
            new_H, new_W = self.resolution
            if video.shape[-2:] != (new_H, new_W):
                video = F.interpolate(video, size=(new_H, new_W), mode='bilinear', align_corners=False)
        
        if video_name is None:
            if video_path:
                video_name = Path(video_path).stem
            else:
                video_name = f"sample_{idx}"
        
        output = {
            'video': video,  # (T, 3, H, W)
            'query_points': query_points,  # (N, 3)
            'target_points': target_points,  # (N, T, 2)
            'occluded': occluded,  # (N, T)
        }
        if video_name is not None:
            output['video_name'] = video_name
            output['original_size'] = original_size
        return output
    
    def _load_video(self, path: str) -> np.ndarray:
        """加载视频文件"""
        try:
            import cv2
        except ImportError as e:
            raise ImportError("cv2 is required to load video files.") from e
        
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if not frames:
            raise FileNotFoundError(f"Failed to read video: {path}")
        return np.stack(frames)
    
    def _resize_video(self, video: np.ndarray, resolution: Tuple[int, int]) -> np.ndarray:
        """调整视频分辨率"""
        from PIL import Image
        
        T, H, W, C = video.shape
        new_H, new_W = resolution
        
        resized = np.zeros((T, new_H, new_W, C), dtype=video.dtype)
        for t in range(T):
            img = Image.fromarray(video[t])
            img = img.resize((new_W, new_H), Image.BILINEAR)
            resized[t] = np.array(img)
        
        return resized
    
    def _augment(
        self,
        video: np.ndarray,
        target_points: np.ndarray,
        occluded: np.ndarray,
        query_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """数据增强"""
        cfg = self.aug_cfg

        random_flip = cfg.get('random_flip', True)
        random_crop = cfg.get('random_crop', False)
        random_scale = cfg.get('random_scale', None)
        color_jitter = cfg.get('color_jitter', 0.4)
        crop_size = cfg.get('crop_size', self.resolution or (256, 256))

        # 随机水平翻转
        if random_flip and np.random.rand() > 0.5:
            video = video[:, :, ::-1, :].copy()
            target_points[:, :, 1] = 1 - target_points[:, :, 1]
            query_points[:, 2] = 1 - query_points[:, 2]

        # 随机缩放
        if random_scale is not None:
            if isinstance(random_scale, list):
                random_scale = tuple(random_scale)
            video, target_points, occluded, query_points = self._random_scale(
                video, target_points, occluded, query_points, random_scale
            )

        # 随机裁剪
        if random_crop and crop_size is not None:
            video, target_points, occluded, query_points = self._random_crop(
                video, target_points, occluded, query_points, crop_size
            )

        # 颜色抖动
        if color_jitter and np.random.rand() > 0.5:
            video = self._color_jitter(video, brightness=color_jitter, contrast=color_jitter, saturation=color_jitter)

        return video, target_points, occluded, query_points

    def _random_scale(
        self,
        video: np.ndarray,
        target_points: np.ndarray,
        occluded: np.ndarray,
        query_points: np.ndarray,
        scale_range: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """随机缩放并保持输出大小不变"""
        T, H, W, C = video.shape
        scale = np.random.uniform(*scale_range)
        new_H = max(1, int(H * scale))
        new_W = max(1, int(W * scale))

        scaled = self._resize_video(video, (new_H, new_W))

        if scale > 1:
            start_y = np.random.randint(0, new_H - H + 1)
            start_x = np.random.randint(0, new_W - W + 1)
            video = scaled[:, start_y:start_y + H, start_x:start_x + W]

            target_points[:, :, 0] = target_points[:, :, 0] * scale - start_y / H
            target_points[:, :, 1] = target_points[:, :, 1] * scale - start_x / W
            query_points[:, 1] = query_points[:, 1] * scale - start_y / H
            query_points[:, 2] = query_points[:, 2] * scale - start_x / W
        else:
            pad_y = (H - new_H) // 2
            pad_x = (W - new_W) // 2
            video = np.zeros((T, H, W, C), dtype=video.dtype)
            video[:, pad_y:pad_y + new_H, pad_x:pad_x + new_W] = scaled

            target_points[:, :, 0] = target_points[:, :, 0] * scale + pad_y / H
            target_points[:, :, 1] = target_points[:, :, 1] * scale + pad_x / W
            query_points[:, 1] = query_points[:, 1] * scale + pad_y / H
            query_points[:, 2] = query_points[:, 2] * scale + pad_x / W

        out_of_bounds = (
            (target_points[:, :, 0] < 0) | (target_points[:, :, 0] > 1) |
            (target_points[:, :, 1] < 0) | (target_points[:, :, 1] > 1)
        )
        occluded = occluded | out_of_bounds
        target_points = np.clip(target_points, 0, 1)
        query_points[:, 1:3] = np.clip(query_points[:, 1:3], 0, 1)

        return video, target_points, occluded, query_points

    def _random_crop(
        self,
        video: np.ndarray,
        target_points: np.ndarray,
        occluded: np.ndarray,
        query_points: np.ndarray,
        crop_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """随机裁剪"""
        T, H, W, C = video.shape
        crop_H, crop_W = crop_size

        if H <= crop_H and W <= crop_W:
            return video, target_points, occluded, query_points

        start_y = np.random.randint(0, max(1, H - crop_H + 1))
        start_x = np.random.randint(0, max(1, W - crop_W + 1))

        video = video[:, start_y:start_y + crop_H, start_x:start_x + crop_W]

        target_points[:, :, 0] = (target_points[:, :, 0] * H - start_y) / crop_H
        target_points[:, :, 1] = (target_points[:, :, 1] * W - start_x) / crop_W
        query_points[:, 1] = (query_points[:, 1] * H - start_y) / crop_H
        query_points[:, 2] = (query_points[:, 2] * W - start_x) / crop_W

        out_of_bounds = (
            (target_points[:, :, 0] < 0) | (target_points[:, :, 0] > 1) |
            (target_points[:, :, 1] < 0) | (target_points[:, :, 1] > 1)
        )
        occluded = occluded | out_of_bounds
        target_points = np.clip(target_points, 0, 1)
        query_points[:, 1:3] = np.clip(query_points[:, 1:3], 0, 1)

        return video, target_points, occluded, query_points
    
    def _color_jitter(
        self,
        video: np.ndarray,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
    ) -> np.ndarray:
        """颜色抖动"""
        video = video.astype(np.float32)
        
        # 亮度
        video += np.random.uniform(-brightness, brightness) * 255
        
        # 对比度
        mean = video.mean()
        video = (video - mean) * np.random.uniform(1-contrast, 1+contrast) + mean
        
        return np.clip(video, 0, 255).astype(np.uint8)


class TAPVidKubricTFDSIterableDataset(IterableDataset):
    """
    Streaming TFDS-based Kubric dataset.

    Motivation:
    - The legacy `TAPVidKubricDataset(use_tfds=True)` path materializes the entire TFDS split into memory,
      which is only suitable for debugging.
    - This iterable variant streams samples from TFDS and generates sparse point tracks on-the-fly from
      ground-truth optical flow + segmentations.

    Notes:
    - Requires `tensorflow` and `tensorflow-datasets`.
    - For paper-quality training you should still prefer official TAP-Vid Kubric annotations when available.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_frames: int = 24,
        num_points: int = 256,
        resolution: Tuple[int, int] = (256, 256),
        augmentation: bool = True,
        annotation_file: Optional[str] = None,
        use_tfds: bool = True,
        allow_synthetic_tracks: bool = False,
        max_retries: int = 5,
        deterministic_sampling: bool = False,
        deterministic_seed: Optional[int] = None,
        sampling: Optional[object] = None,
        tfds_name: str = "movi_e/256x256",
        shuffle_buffer: int = 1024,
        shuffle_files: Optional[bool] = None,
        tfds_low_memory: bool = False,
        tfds_read_buffer_size: Optional[int] = None,
    ):
        if not TF_AVAILABLE:
            raise RuntimeError(
                "TAPVidKubricTFDSIterableDataset requires tensorflow-datasets. "
                "Install tensorflow and tensorflow-datasets, or use official Kubric pickle annotations."
            )
        if not use_tfds:
            raise ValueError("TAPVidKubricTFDSIterableDataset must be constructed with use_tfds=True.")
        if not allow_synthetic_tracks:
            raise RuntimeError(
                "TFDS does not provide TAP-Vid point tracks directly. "
                "Set allow_synthetic_tracks=true to generate sparse tracks from ground-truth flow."
            )

        self.root = Path(root)
        self.split = str(split)
        self.tfds_split = _canonical_tfds_split(self.split)
        self.tfds_name = str(tfds_name)
        self.num_frames = int(num_frames) if num_frames is not None else -1
        self.num_points = int(num_points) if num_points is not None else 0
        self.resolution = resolution
        self.allow_synthetic_tracks = True
        self.max_retries = max(0, int(max_retries))
        self.deterministic_sampling = bool(deterministic_sampling)
        self.deterministic_seed = int(deterministic_seed) if deterministic_seed is not None else None
        self.annotation_file = annotation_file  # kept for signature compatibility; unused for TFDS

        self.aug_cfg = self._normalize_aug_config(augmentation)
        self.augmentation = self.aug_cfg.get("enabled", False)
        self.sampling_cfg = self._normalize_sampling_config(sampling)
        self.sampling_strategy = str(self.sampling_cfg.get("strategy", "uniform")).lower().strip()
        self.sampling_hard_fraction = float(self.sampling_cfg.get("hard_fraction", 0.5) or 0.5)
        self.sampling_hard_fraction = max(0.0, min(1.0, self.sampling_hard_fraction))
        self.shuffle_buffer = max(0, int(shuffle_buffer))
        self.shuffle_files = None if shuffle_files is None else bool(shuffle_files)
        self.tfds_low_memory = bool(tfds_low_memory)
        if tfds_read_buffer_size is None:
            self.tfds_read_buffer_size = None
        else:
            try:
                parsed_size = int(tfds_read_buffer_size)
                self.tfds_read_buffer_size = parsed_size if parsed_size > 0 else None
            except (TypeError, ValueError):
                self.tfds_read_buffer_size = None
        self._tfds_source_mode = "name"
        self._tfds_builder_dir: Optional[str] = None

        self.spatial_aug = None
        self.temporal_aug = None
        if self.augmentation:
            cfg = self.aug_cfg
            random_scale = cfg.get("random_scale", None)
            if isinstance(random_scale, list):
                random_scale = tuple(random_scale)
            crop_size = cfg.get("crop_size", self.resolution or (256, 256))
            random_rotation = float(cfg.get("random_rotation", 0.0) or 0.0)
            self.spatial_aug = PointTrackingAugmentation(
                random_crop=cfg.get("random_crop", True),
                random_flip=cfg.get("random_flip", True),
                color_jitter=cfg.get("color_jitter", 0.4),
                random_scale=random_scale,
                random_rotation=random_rotation,
                crop_size=crop_size,
            )

            temporal_cfg = cfg.get("temporal", {})
            if not isinstance(temporal_cfg, dict):
                temporal_cfg = {}
            temporal_enabled = bool(temporal_cfg.get("enabled", False))
            if any(k in cfg for k in ["random_reverse", "random_speed", "random_start"]):
                temporal_enabled = True
            if temporal_enabled:
                random_speed = temporal_cfg.get("random_speed", cfg.get("random_speed", (0.5, 2.0)))
                if isinstance(random_speed, list):
                    random_speed = tuple(random_speed)
                self.temporal_aug = TemporalAugmentation(
                    random_reverse=temporal_cfg.get("random_reverse", cfg.get("random_reverse", True)),
                    random_speed=random_speed,
                    random_start=temporal_cfg.get("random_start", cfg.get("random_start", True)),
                )

        # Resolve split size once for __len__ / progress bars.
        try:
            builder, source_mode, builder_dir = _resolve_tfds_builder(self.root, self.tfds_name)
            self._tfds_source_mode = str(source_mode)
            self._tfds_builder_dir = str(builder_dir) if builder_dir is not None else None
            info = builder.info
            if self.tfds_split not in info.splits:
                raise KeyError(self.tfds_split)
            self._num_examples = int(info.splits[self.tfds_split].num_examples)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build TFDS dataset '{self.tfds_name}' at root={self.root} "
                f"(split={self.tfds_split}). Make sure TFDS data exists under root. "
                f"Tried standard path and fallback layouts (movi_e / tapvid_kubric / config-only); "
                f"original error: {exc}"
            ) from exc

    def __len__(self):
        # For DDP we shard inside TFDS as well; report per-rank length to keep progress bars reasonable.
        world = 1
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                world = int(torch.distributed.get_world_size())
        except Exception:
            world = 1
        world = max(1, world)
        return int((self._num_examples + world - 1) // world)

    def _normalize_aug_config(self, augmentation) -> Dict:
        if augmentation is None:
            return {"enabled": False}
        if isinstance(augmentation, bool):
            return {"enabled": augmentation}
        if OMEGACONF_AVAILABLE and OmegaConf.is_config(augmentation):
            cfg = OmegaConf.to_container(augmentation, resolve=True)
        elif isinstance(augmentation, dict):
            cfg = dict(augmentation)
        else:
            try:
                cfg = dict(augmentation)
            except Exception:
                cfg = {"enabled": True}
        cfg.setdefault("enabled", True)
        return cfg

    def _normalize_sampling_config(self, sampling) -> Dict:
        if sampling is None:
            return {"strategy": "uniform"}
        if isinstance(sampling, str):
            return {"strategy": sampling}
        if OMEGACONF_AVAILABLE and OmegaConf is not None:
            try:
                if OmegaConf.is_config(sampling):
                    cfg = OmegaConf.to_container(sampling, resolve=True)
                else:
                    cfg = sampling
            except Exception:
                cfg = sampling
        else:
            cfg = sampling
        if isinstance(cfg, dict):
            cfg = dict(cfg)
        else:
            try:
                cfg = dict(cfg)
            except Exception:
                cfg = {"strategy": "uniform"}
        cfg.setdefault("strategy", "uniform")
        return cfg

    def _ensure_query_visible(
        self,
        query_points: torch.Tensor,
        target_points: torch.Tensor,
        occluded: torch.Tensor,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if query_points.numel() == 0 or target_points.numel() == 0 or occluded.numel() == 0:
            return query_points, target_points, occluded
        if occluded.dim() != 2:
            return query_points, target_points, occluded
        T = occluded.shape[1]
        query_t = query_points[:, 0].round().long().clamp(0, T - 1)
        point_idx = torch.arange(occluded.shape[0], device=occluded.device)
        visible_mask = ~occluded[point_idx, query_t]
        if visible_mask.all():
            return query_points, target_points, occluded
        valid_idx = torch.nonzero(visible_mask, as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            return query_points, target_points, occluded
        if self.num_points is not None and self.num_points > 0:
            if valid_idx.numel() >= self.num_points:
                if rng is not None:
                    local = rng.choice(int(valid_idx.numel()), int(self.num_points), replace=False)
                    perm = torch.as_tensor(local, device=valid_idx.device, dtype=torch.long)
                else:
                    perm = torch.randperm(valid_idx.numel(), device=valid_idx.device)[: self.num_points]
                chosen = valid_idx[perm]
            else:
                if rng is not None:
                    local = rng.randint(0, int(valid_idx.numel()), size=(int(self.num_points),))
                    rand = torch.as_tensor(local, device=valid_idx.device, dtype=torch.long)
                else:
                    rand = torch.randint(0, valid_idx.numel(), (self.num_points,), device=valid_idx.device)
                chosen = valid_idx[rand]
            return query_points[chosen], target_points[chosen], occluded[chosen]
        return query_points[valid_idx], target_points[valid_idx], occluded[valid_idx]

    def _iter_tfds(self):
        # Determine global rank for optional DDP sharding.
        rank = 0
        world = 1
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = int(torch.distributed.get_rank())
                world = int(torch.distributed.get_world_size())
        except Exception:
            rank, world = 0, 1
        rank = max(0, rank)
        world = max(1, world)

        worker = torch.utils.data.get_worker_info()
        worker_id = 0
        num_workers = 1
        if worker is not None:
            worker_id = int(worker.id)
            num_workers = int(worker.num_workers)
        num_workers = max(1, num_workers)

        shard_count = world * num_workers
        shard_index = rank * num_workers + worker_id

        default_shuffle_files = (self.tfds_split == "train") and (not self.deterministic_sampling)
        shuffle_files = default_shuffle_files if self.shuffle_files is None else bool(self.shuffle_files)
        seed = int(torch.initial_seed() % (2**31 - 1))
        read_config = _build_tfds_read_config(
            low_memory=self.tfds_low_memory,
            deterministic=bool(self.deterministic_sampling),
            shuffle_seed=seed if shuffle_files else None,
            override_buffer_size=self.tfds_read_buffer_size,
        )

        if self._tfds_source_mode == "directory":
            if self._tfds_builder_dir is None:
                raise RuntimeError("TFDS source mode is directory but builder directory is not set.")
            builder = tfds.builder_from_directory(self._tfds_builder_dir)
            try:
                as_dataset_kwargs = dict(split=self.tfds_split, shuffle_files=shuffle_files)
                if read_config is not None:
                    as_dataset_kwargs["read_config"] = read_config
                ds = builder.as_dataset(**as_dataset_kwargs)
            except TypeError:
                try:
                    ds = builder.as_dataset(split=self.tfds_split, shuffle_files=shuffle_files)
                except TypeError:
                    ds = builder.as_dataset(split=self.tfds_split)
        else:
            load_kwargs = dict(
                split=self.tfds_split,
                data_dir=str(self.root),
                shuffle_files=shuffle_files,
            )
            if read_config is not None:
                load_kwargs["read_config"] = read_config
            ds = tfds.load(self.tfds_name, **load_kwargs)
        if shard_count > 1:
            ds = ds.shard(num_shards=shard_count, index=shard_index)

        if self.tfds_split == "train" and not self.deterministic_sampling and self.shuffle_buffer > 0:
            buf = int(min(self.shuffle_buffer, max(1, self._num_examples // shard_count)))
            ds = ds.shuffle(buffer_size=buf, seed=seed, reshuffle_each_iteration=True)

        return tfds.as_numpy(ds)

    def __iter__(self):
        base_seed = int(torch.initial_seed() % (2**32))
        failures = 0
        for sample_idx, sample in enumerate(self._iter_tfds()):
            rng_seed = (base_seed + int(sample_idx) * 1013) % (2**32)
            rng = np.random.RandomState(int(rng_seed))

            try:
                video = np.asarray(sample["video"])  # (T, H, W, 3) uint8
                metadata = sample.get("metadata", None)
                ff_range = None
                bf_range = None
                if isinstance(metadata, dict):
                    ff_range = metadata.get("forward_flow_range", None)
                    bf_range = metadata.get("backward_flow_range", None)

                forward_flow = _decode_kubric_flow(sample.get("forward_flow", None), ff_range)
                backward_flow = _decode_kubric_flow(sample.get("backward_flow", None), bf_range)
                segmentations = sample.get("segmentations", None)

                # Optional temporal windowing (usually MOVi-E has T=24 already).
                T_full = int(video.shape[0])
                start = 0
                if self.num_frames is not None and int(self.num_frames) > 0 and T_full > int(self.num_frames):
                    start = int(rng.randint(0, T_full - int(self.num_frames) + 1))
                    end = start + int(self.num_frames)
                    video = video[start:end]
                    forward_flow = forward_flow[start:end]
                    backward_flow = backward_flow[start:end]
                    if segmentations is not None:
                        segmentations = np.asarray(segmentations)[start:end]

                query_points, target_points, occluded = _generate_sparse_tracks_from_kubric(
                    forward_flow=forward_flow,
                    backward_flow=backward_flow,
                    segmentations=segmentations,
                    num_points=int(self.num_points),
                    rng=rng,
                    sampling_strategy=self.sampling_strategy,
                    hard_fraction=self.sampling_hard_fraction,
                )

                # Convert to torch (same format as TAPVidKubricDataset).
                T, H, W, C = video.shape
                original_size = torch.tensor([int(H), int(W)], dtype=torch.int32)
                video_t = torch.from_numpy(video).permute(0, 3, 1, 2).float()  # (T, 3, H, W)
                if video_t.numel() > 0 and float(video_t.max()) > 1.5:
                    video_t = video_t / 255.0

                query_t = torch.from_numpy(query_points).float()
                query_t[:, 0] = torch.clamp(query_t[:, 0], 0, T - 1)
                query_t[:, 1:3] = torch.clamp(query_t[:, 1:3], 0.0, 1.0)
                target_t = torch.from_numpy(target_points).float()
                occluded_t = torch.from_numpy(occluded.astype(bool))

                if self.augmentation and self.spatial_aug is not None:
                    outputs = self.spatial_aug(video_t, target_t, occluded_t, query_t)
                    if len(outputs) == 4:
                        video_t, target_t, occluded_t, query_t = outputs
                    else:
                        video_t, target_t, occluded_t = outputs
                if self.augmentation and self.temporal_aug is not None:
                    video_t, target_t, occluded_t, query_t = self.temporal_aug(
                        video_t, target_t, occluded_t, query_t
                    )
                query_t[:, 0] = torch.clamp(query_t[:, 0], 0, video_t.shape[0] - 1)
                if self.augmentation:
                    query_t, target_t, occluded_t = self._ensure_query_visible(query_t, target_t, occluded_t, rng=rng)

                if self.resolution is not None:
                    new_H, new_W = self.resolution
                    if video_t.shape[-2:] != (new_H, new_W):
                        video_t = F.interpolate(video_t, size=(new_H, new_W), mode="bilinear", align_corners=False)

                yield {
                    "video": video_t,
                    "query_points": query_t,
                    "target_points": target_t,
                    "occluded": occluded_t,
                    "video_name": f"tfds_{self.tfds_split}_{sample_idx:06d}",
                    "original_size": original_size,
                }
                failures = 0
            except Exception as exc:
                failures += 1
                if failures <= self.max_retries:
                    continue
                raise


def create_kubric_dataloader(
    root: str,
    batch_size: int = 8,
    num_workers: int = 8,
    **kwargs,
) -> DataLoader:
    """
    创建Kubric数据加载器
    
    Args:
        root: 数据集根目录
        batch_size: 批大小
        num_workers: 数据加载线程数
        **kwargs: 传递给TAPVidKubricDataset的参数
        
    Returns:
        DataLoader实例
    """
    dataset = TAPVidKubricDataset(root, **kwargs)
    
    drop_last = len(dataset) >= batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    
    return dataloader


def preprocess_kubric_to_pickle(
    tfds_root: str,
    output_path: str,
    split: str = 'train',
    max_samples: int = None,
):
    """
    预处理Kubric数据集并保存为pickle格式
    
    这可以加速后续加载
    
    Args:
        tfds_root: TensorFlow Datasets目录
        output_path: 输出pickle文件路径
        split: 数据集划分
        max_samples: 最大样本数（用于调试）
    """
    if not TF_AVAILABLE:
        print("TensorFlow not available, cannot preprocess")
        return
    
    print(f"Loading {split} split from {tfds_root}...")
    ds = tfds.load(
        'movi_e/256x256',
        split=split,
        data_dir=tfds_root,
    )
    
    samples = []
    for i, sample in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        
        video = sample['video'].numpy()
        
        # 生成点追踪标注（简化版本）
        T, H, W, _ = video.shape
        num_points = 100
        
        query_points = []
        target_points = []
        occluded = []
        
        for _ in range(num_points):
            t = np.random.randint(0, T)
            y, x = np.random.rand(), np.random.rand()
            
            query_points.append([t, y, x])
            
            # 简化轨迹
            track = [[np.clip(y + 0.01 * (t2-t), 0, 1),
                      np.clip(x + 0.01 * (t2-t), 0, 1)]
                     for t2 in range(T)]
            occ = [False] * T
            
            target_points.append(track)
            occluded.append(occ)
        
        samples.append({
            'video': video,
            'query_points': np.array(query_points, dtype=np.float32),
            'target_points': np.array(target_points, dtype=np.float32),
            'occluded': np.array(occluded, dtype=bool),
        })
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} samples...")
    
    print(f"Saving {len(samples)} samples to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(samples, f)
    
    print("Done!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='datasets/tapvid_kubric')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess TFDS to pickle')
    args = parser.parse_args()
    
    if args.preprocess:
        preprocess_kubric_to_pickle(
            args.root,
            f'{args.root}/tapvid_kubric_train.pkl',
            split='train',
            max_samples=1000,
        )
    else:
        # 测试数据集
        dataset = TAPVidKubricDataset(
            args.root,
            split='train',
            num_frames=24,
            num_points=256,
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample 0:")
            print(f"  Video shape: {sample['video'].shape}")
            print(f"  Query points shape: {sample['query_points'].shape}")
            print(f"  Target points shape: {sample['target_points'].shape}")
            print(f"  Occluded shape: {sample['occluded'].shape}")
