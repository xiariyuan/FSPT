"""
TAP-Vid Kinetics Dataset

Kinetics是一个大规模真实视频数据集，TAP-Vid在其上标注了点追踪数据。
"""

import os
import csv
import pickle
import glob
import io
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict, List, Any
try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    OmegaConf = None
from .augmentation import PointTrackingAugmentation, TemporalAugmentation
from .coord_utils import normalize_points_yx, normalize_query_points_tyx
from .tapvid_official_eval import sample_queries_first, sample_queries_strided
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class TAPVidKineticsDataset(Dataset):
    """
    TAP-Vid Kinetics测试数据集

    数据格式:
    - video: (T, H, W, 3) uint8
    - points: (N, T, 2) float32, 像素坐标
    - occluded: (N, T) bool
    - query_points: (N, 3) float32, [t, y, x] 像素坐标

    Args:
        root: 数据集根目录，应包含 tapvid_kinetics.pkl / tapvid_kinetics.csv
              或 *_of_*.pkl 分片
        resolution: 目标分辨率 (H, W)，None表示使用原始分辨率
        augmentation: 数据增强配置或开关（默认关闭）
        max_frames: 最大帧数（用于长视频截断）
        num_points: 可选点数上限（默认使用全部点）
        max_retries: 单个样本失败时的最大重试次数
    """

    def __init__(
        self,
        root: str,
        resolution: Optional[Tuple[int, int]] = None,
        augmentation: Optional[object] = None,
        max_frames: Optional[int] = 250,
        num_points: Optional[int] = None,
        max_retries: int = 5,
        split: Optional[str] = None,
        annotation_file: Optional[str] = None,
        query_mode: str = "strided",
        query_stride: int = 5,
        points_order: str = "xy",
    ):
        self.root = root
        self.resolution = resolution
        self.max_frames = max_frames
        self.num_points = num_points
        self.max_retries = max(0, int(max_retries))
        self.split = self._normalize_split(split)
        self.annotation_file = annotation_file
        self.query_mode = self._normalize_query_mode(query_mode)
        self.query_stride = max(1, int(query_stride))
        self.points_order = self._normalize_points_order(points_order)
        self.aug_cfg = self._normalize_aug_config(augmentation)
        self.augmentation = self.aug_cfg.get('enabled', False)
        self.spatial_aug = None
        self.temporal_aug = None
        if self.augmentation:
            cfg = self.aug_cfg
            random_scale = cfg.get('random_scale', None)
            if isinstance(random_scale, list):
                random_scale = tuple(random_scale)
            crop_size = cfg.get('crop_size', self.resolution)
            random_rotation = float(cfg.get('random_rotation', 0.0) or 0.0)
            self.spatial_aug = PointTrackingAugmentation(
                random_crop=cfg.get('random_crop', True),
                random_flip=cfg.get('random_flip', True),
                color_jitter=cfg.get('color_jitter', 0.4),
                random_scale=random_scale,
                random_rotation=random_rotation,
                crop_size=crop_size or (256, 256),
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

        # 加载标注
        self.data = self._load_annotations(root, annotation_file=annotation_file)

        self.video_names = sorted(list(self.data.keys()))
        self.use_video_files = False
        if self.video_names:
            sample = self.data[self.video_names[0]]
            self.use_video_files = 'video' not in sample
        if self.use_video_files:
            if not CV2_AVAILABLE:
                raise ImportError("cv2 is required to load Kinetics videos from files.")
            filtered = []
            for name in self.video_names:
                sample = self.data.get(name, {})
                video_path = sample.get('video_path')
                if video_path is not None:
                    if not os.path.isabs(video_path):
                        video_path = os.path.join(self.root, video_path)
                    if os.path.exists(video_path):
                        filtered.append(name)
                        continue
                video_path = os.path.join(self.root, 'videos', f'{name}.mp4')
                if os.path.exists(video_path):
                    filtered.append(name)
            self.video_names = filtered
            if not self.video_names:
                raise FileNotFoundError(
                    f"No videos found under {os.path.join(self.root, 'videos')}. "
                    "Please download Kinetics videos or provide embedded video data."
                )

        print(f"Loaded TAP-Vid Kinetics dataset with {len(self.video_names)} videos")

    def _coerce_annotation_mapping(self, payload: Any, source_hint: str) -> Dict[str, Dict[str, Any]]:
        """
        Normalize TAP-Vid annotations into {video_name: sample_dict}.

        Supported payload formats:
        1) dict[str, dict] (legacy project format)
        2) list[dict] (official shard dumps commonly seen in mirrors)
        """
        if isinstance(payload, dict):
            normalized: Dict[str, Dict[str, Any]] = {}
            for key, value in payload.items():
                if not isinstance(value, dict):
                    raise TypeError(
                        f"Expected dict sample under key={key!r} in {source_hint}, got {type(value)}"
                    )
                name = str(key)
                normalized[name] = value
            return normalized

        if isinstance(payload, list):
            normalized: Dict[str, Dict[str, Any]] = {}
            for idx, sample in enumerate(payload):
                if not isinstance(sample, dict):
                    raise TypeError(
                        f"Expected dict sample at index={idx} in {source_hint}, got {type(sample)}"
                    )

                name = sample.get('video_name') or sample.get('name') or sample.get('id')
                if name is None or str(name).strip() == "":
                    # Official shards usually do not carry video names; synthesize stable IDs.
                    name = f"{source_hint}_{idx:06d}"
                else:
                    name = str(name)

                # Avoid accidental key overwrite on duplicated names.
                if name in normalized:
                    base = name
                    suffix = 1
                    while f"{base}_{suffix:03d}" in normalized:
                        suffix += 1
                    name = f"{base}_{suffix:03d}"

                normalized[name] = sample
            return normalized

        raise TypeError(
            f"Unsupported annotation payload type in {source_hint}: {type(payload)}; "
            "expected dict or list."
        )

    def _normalize_split(self, split: Optional[str]) -> Optional[str]:
        if split is None:
            return None
        normalized = str(split).strip().lower()
        aliases = {
            "": None,
            "all": None,
            "full": None,
            "none": None,
            "train": "train",
            "val": "val",
            "valid": "val",
            "validation": "val",
            "test": "test",
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported kinetics split={split}. Expected train|val|test|all.")
        return aliases[normalized]

    def _resolve_annotation_path(self, root: str, annotation_file: Optional[str]) -> Optional[str]:
        if annotation_file:
            candidate = annotation_file
            if not os.path.isabs(candidate):
                candidate = os.path.join(root, candidate)
            return candidate

        default_pkl = os.path.join(root, 'tapvid_kinetics.pkl')
        if os.path.exists(default_pkl):
            return default_pkl

        default_csv = os.path.join(root, 'tapvid_kinetics.csv')
        if os.path.exists(default_csv):
            return default_csv

        return default_pkl

    def _load_split_video_ids(self, root: str) -> Optional[set]:
        if self.split is None:
            return None

        split_file = os.path.join(root, f'{self.split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Requested Kinetics split={self.split}, but split file not found: {split_file}"
            )

        with open(split_file, 'r', encoding='utf-8') as f:
            video_ids = {line.strip() for line in f if line.strip()}
        return video_ids

    def _discover_kinetics_segment_index(self, root: str) -> Dict[str, str]:
        """
        Build a basename -> absolute-path index for unpacked Kinetics clips.

        Official tar archives extract into directories such as:
          videos_val_unpacked/<label>/<youtubeid>_<start>_<end>.mp4
          videos_test_unpacked/Kinetics700-2020-test/<youtubeid>_<start>_<end>.mp4
        """
        index: Dict[str, str] = {}
        source_dirs = sorted(glob.glob(os.path.join(root, "videos_*_unpacked")))
        for source_dir in source_dirs:
            if not os.path.isdir(source_dir):
                continue
            for mp4_path in Path(source_dir).rglob("*.mp4"):
                key = mp4_path.name
                if key not in index:
                    index[key] = str(mp4_path.resolve())
        return index

    def _load_official_csv_annotations(self, root: str, csv_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load the official TAP-Vid Kinetics CSV package.

        Format per row:
          video_id, meta_1, meta_2, x_0, y_0, vis_0, x_1, y_1, vis_1, ...

        The official package stores one point track per row. Coordinates are
        normalized to [0, 1], and visibility is {0,1}.
        """
        split_video_ids = self._load_split_video_ids(root)
        segment_index = self._discover_kinetics_segment_index(root)
        grouped: Dict[str, Dict[str, Any]] = {}
        num_rows = 0

        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader):
                if not row:
                    continue
                video_name = row[0].strip()
                if not video_name:
                    continue
                if split_video_ids is not None and video_name not in split_video_ids:
                    continue
                if len(row) < 6 or (len(row) - 3) % 3 != 0:
                    raise ValueError(
                        f"Malformed TAP-Vid Kinetics CSV row #{row_idx}: expected "
                        f"'video_id,meta_1,meta_2,(x,y,vis)*', got {len(row)} columns."
                    )

                triples = np.asarray(row[3:], dtype=np.float32).reshape(-1, 3)
                points_xy = triples[:, :2]
                occluded = triples[:, 2] <= 0.5
                segment_name = f"{video_name}_{int(float(row[1])):06d}_{int(float(row[2])):06d}.mp4"
                segment_path = segment_index.get(segment_name)

                sample = grouped.setdefault(
                    video_name,
                    {
                        'points': [],
                        'occluded': [],
                    },
                )
                if segment_path is not None and 'video_path' not in sample:
                    sample['video_path'] = segment_path
                sample['points'].append(points_xy)
                sample['occluded'].append(occluded)
                num_rows += 1

        if not grouped:
            split_hint = f" for split={self.split}" if self.split is not None else ""
            raise FileNotFoundError(
                f"No TAP-Vid Kinetics tracks were loaded from {csv_path}{split_hint}."
            )

        normalized: Dict[str, Dict[str, Any]] = {}
        for video_name, sample in grouped.items():
            points = np.stack(sample['points'], axis=0).astype(np.float32)   # (N,T,2) [x,y]
            occluded = np.stack(sample['occluded'], axis=0).astype(bool)     # (N,T)
            normalized[video_name] = {
                'points': points,
                'occluded': occluded,
            }

        split_hint = f" split={self.split}" if self.split is not None else ""
        print(
            f"Loaded official TAP-Vid Kinetics CSV from {csv_path} "
            f"with {len(normalized)} videos / {num_rows} tracks{split_hint}"
        )
        return normalized

    def _load_annotations(self, root: str, annotation_file: Optional[str] = None) -> Dict:
        """Load annotations from pkl, official csv, or sharded pkl files."""
        anno_path = self._resolve_annotation_path(root, annotation_file)
        if os.path.exists(anno_path):
            if anno_path.endswith('.csv'):
                return self._load_official_csv_annotations(root, anno_path)
            with open(anno_path, 'rb') as f:
                data = pickle.load(f)
            source_hint = os.path.basename(anno_path).replace('.pkl', '')
            return self._coerce_annotation_mapping(data, source_hint)

        shard_paths = sorted(glob.glob(os.path.join(root, '*_of_*.pkl')))
        if not shard_paths:
            raise FileNotFoundError(
                f"Annotation file not found at {anno_path}, and no sharded files like "
                f"{os.path.join(root, '*_of_*.pkl')} were found."
            )

        merged: Dict = {}
        for shard_idx, shard_path in enumerate(shard_paths):
            with open(shard_path, 'rb') as f:
                shard_data = pickle.load(f)
            shard_name = os.path.basename(shard_path).replace('.pkl', '')
            normalized = self._coerce_annotation_mapping(
                shard_data, f"{shard_name}_s{shard_idx:03d}"
            )

            overlap = set(merged.keys()) & set(normalized.keys())
            if overlap:
                # Keep deterministic behavior and avoid silent overwrite.
                normalized = {
                    (f"{k}__dup{shard_idx:03d}" if k in overlap else k): v
                    for k, v in normalized.items()
                }
            merged.update(normalized)

        print(f"Loaded {len(shard_paths)} Kinetics annotation shards from {root}")
        return merged

    def _decode_video_if_needed(self, video: np.ndarray) -> np.ndarray:
        """
        Decode TAP-Vid encoded-frame format when needed.

        Some official Kinetics dumps store `video` as shape (T,) where each item is
        JPEG bytes (np.bytes_ / bytes). This helper converts it to (T,H,W,3) uint8.
        """
        if not isinstance(video, np.ndarray):
            video = np.asarray(video)
        if video.ndim != 1 or video.size == 0:
            return video

        first = video[0]
        is_encoded_bytes = isinstance(first, (bytes, np.bytes_))
        is_encoded_uint8 = isinstance(first, np.ndarray) and first.dtype == np.uint8 and first.ndim == 1
        if not (is_encoded_bytes or is_encoded_uint8):
            return video

        decoded_frames: List[np.ndarray] = []
        for frame_idx, item in enumerate(video):
            if isinstance(item, np.ndarray):
                frame_bytes = item.tobytes()
            else:
                frame_bytes = bytes(item)

            if CV2_AVAILABLE:
                frame_buf = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame_bgr = cv2.imdecode(frame_buf, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    raise ValueError(f"Failed to decode encoded frame #{frame_idx} with OpenCV.")
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                decoded_frames.append(frame_rgb)
            else:
                # Pillow fallback when OpenCV is unavailable.
                from PIL import Image  # type: ignore
                with Image.open(io.BytesIO(frame_bytes)) as img:
                    decoded_frames.append(np.asarray(img.convert('RGB')))

        return np.stack(decoded_frames, axis=0)
        
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

    def _normalize_query_mode(self, query_mode: Optional[str]) -> str:
        mode = str(query_mode).strip().lower() if query_mode is not None else "strided"
        if mode in ("", "none", "null"):
            mode = "strided"
        if mode not in ("strided", "first"):
            raise ValueError(f"Unsupported query_mode={query_mode}. Expected 'strided' or 'first'.")
        return mode

    def _normalize_points_order(self, points_order: Optional[str]) -> str:
        order = str(points_order).strip().lower() if points_order is not None else "xy"
        aliases = {
            "xy": "xy",
            "yx": "yx",
            "auto": "auto",
            "x_y": "xy",
            "y_x": "yx",
        }
        if order not in aliases:
            raise ValueError(f"Unsupported points_order={points_order}. Expected one of: xy|yx|auto.")
        return aliases[order]

    def _infer_points_order(
        self,
        points: np.ndarray,
        query_points: Optional[np.ndarray],
        height: int,
        width: int,
    ) -> str:
        if (
            query_points is not None
            and query_points.ndim == 2
            and query_points.shape[1] >= 3
            and points.ndim == 3
            and points.shape[0] == query_points.shape[0]
            and points.shape[1] > 0
            and query_points.shape[0] > 0
        ):
            query_t = np.clip(np.round(query_points[:, 0]).astype(np.int64), 0, points.shape[1] - 1)
            p_at_q = points[np.arange(points.shape[0]), query_t]
            q_yx = query_points[:, 1:3]
            err_as_yx = np.linalg.norm(p_at_q - q_yx, axis=1).mean()
            err_as_xy = np.linalg.norm(p_at_q[:, [1, 0]] - q_yx, axis=1).mean()
            if np.isfinite(err_as_xy) and np.isfinite(err_as_yx) and abs(err_as_xy - err_as_yx) > 1e-6:
                return "xy" if err_as_xy < err_as_yx else "yx"

        if points.size > 0 and float(np.nanmax(points)) > 1.5:
            invalid_if_xy = (
                (points[..., 0] < 0)
                | (points[..., 0] > width)
                | (points[..., 1] < 0)
                | (points[..., 1] > height)
            ).mean()
            invalid_if_yx = (
                (points[..., 0] < 0)
                | (points[..., 0] > height)
                | (points[..., 1] < 0)
                | (points[..., 1] > width)
            ).mean()
            if abs(float(invalid_if_xy) - float(invalid_if_yx)) > 1e-6:
                return "xy" if invalid_if_xy < invalid_if_yx else "yx"

        return "xy"

    def _convert_points_to_internal_yx(
        self,
        points: np.ndarray,
        query_points: Optional[np.ndarray],
        height: int,
        width: int,
    ) -> np.ndarray:
        order = self.points_order
        if order == "auto":
            order = self._infer_points_order(points, query_points, height, width)
        if order == "xy":
            return points[..., [1, 0]]
        return points

    def _generate_queries_from_tracks(
        self,
        points_yx: np.ndarray,
        occluded: np.ndarray,
        mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mode = self._normalize_query_mode(mode or self.query_mode)
        num_tracks, num_frames = points_yx.shape[:2]
        if num_tracks == 0 or num_frames == 0:
            return points_yx, occluded, np.zeros((0, 3), dtype=np.float32)

        points_yx = np.asarray(points_yx, dtype=np.float32)
        occluded = np.asarray(occluded, dtype=bool)
        visible = ~occluded

        # Convert to official sampler input: target_points in [x, y] normalized.
        points_xy = points_yx[..., [1, 0]].astype(np.float32)
        use_pixel_scale = float(np.nanmax(np.abs(points_xy))) > 1.5 if points_xy.size > 0 else False
        if use_pixel_scale:
            h_est = max(1.0, float(np.nanmax(points_yx[..., 0])) if points_yx.size > 0 else 1.0)
            w_est = max(1.0, float(np.nanmax(points_yx[..., 1])) if points_yx.size > 0 else 1.0)
            points_xy_norm = points_xy / np.array([max(w_est - 1.0, 1.0), max(h_est - 1.0, 1.0)], dtype=np.float32)
        else:
            h_est, w_est = 1.0, 1.0
            points_xy_norm = points_xy

        dummy_frames = np.zeros((num_frames, 1, 1, 3), dtype=np.float32)

        try:
            if mode == "first":
                if not np.any(visible.any(axis=1)):
                    return (
                        np.zeros((0, num_frames, 2), dtype=np.float32),
                        np.zeros((0, num_frames), dtype=bool),
                        np.zeros((0, 3), dtype=np.float32),
                    )
                converted = sample_queries_first(occluded, points_xy_norm, dummy_frames)
            else:
                has_strided = any(np.any(visible[:, t]) for t in range(0, num_frames, self.query_stride))
                if not has_strided:
                    return self._generate_queries_from_tracks(points_yx, occluded, mode="first")
                converted = sample_queries_strided(
                    occluded, points_xy_norm, dummy_frames, query_stride=self.query_stride
                )
        except Exception:
            if mode != "first":
                return self._generate_queries_from_tracks(points_yx, occluded, mode="first")
            return (
                np.zeros((0, num_frames, 2), dtype=np.float32),
                np.zeros((0, num_frames), dtype=bool),
                np.zeros((0, 3), dtype=np.float32),
            )

        target_xy_norm = converted["target_points"][0].astype(np.float32)  # (M,T,2) [x,y]
        target_occ = converted["occluded"][0].astype(bool)
        query_tyx = converted["query_points"][0].astype(np.float32)  # (M,3) [t,y,x] normalized

        target_yx = target_xy_norm[..., [1, 0]]
        query_yx = query_tyx[:, 1:3]

        if use_pixel_scale:
            target_yx = target_yx * np.array([max(h_est - 1.0, 1.0), max(w_est - 1.0, 1.0)], dtype=np.float32)
            query_yx = query_yx * np.array([max(h_est - 1.0, 1.0), max(w_est - 1.0, 1.0)], dtype=np.float32)

        queries = np.concatenate([query_tyx[:, :1], query_yx], axis=1).astype(np.float32)
        return target_yx.astype(np.float32), target_occ, queries

    def _ensure_query_visible(
        self,
        query_points: torch.Tensor,
        target_points: torch.Tensor,
        occluded: torch.Tensor,
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
                perm = torch.randperm(valid_idx.numel(), device=valid_idx.device)[:self.num_points]
                chosen = valid_idx[perm]
            else:
                rand = torch.randint(0, valid_idx.numel(), (self.num_points,), device=valid_idx.device)
                chosen = valid_idx[rand]
            return query_points[chosen], target_points[chosen], occluded[chosen]
        return query_points[valid_idx], target_points[valid_idx], occluded[valid_idx]

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx, retry: int = 0):
        video_name = self.video_names[idx]
        video_data = self.data[video_name]
        query_generated = False

        # 获取数据
        if 'video' in video_data:
            video = video_data['video']  # (T, H, W, 3)
        else:
            video_path = video_data.get('video_path')
            if video_path is None:
                video_path = os.path.join(self.root, 'videos', f'{video_name}.mp4')
            elif not os.path.isabs(video_path):
                video_path = os.path.join(self.root, video_path)
            try:
                video = self._load_video(video_path)
            except Exception as e:
                if retry < self.max_retries:
                    new_idx = np.random.randint(0, len(self))
                    return self.__getitem__(new_idx, retry=retry + 1)
                raise e

        if video is None:
            if retry < self.max_retries:
                new_idx = np.random.randint(0, len(self))
                return self.__getitem__(new_idx, retry=retry + 1)
            raise ValueError("Empty video sample encountered.")
        if isinstance(video, torch.Tensor):
            video = video.detach().cpu().numpy()
        video = np.asarray(video)
        video = self._decode_video_if_needed(video)
        if video.ndim == 1 and video.size > 0 and isinstance(video[0], np.ndarray):
            video = np.stack(video)
        if video.size == 0 or video.shape[0] == 0:
            if retry < self.max_retries:
                new_idx = np.random.randint(0, len(self))
                return self.__getitem__(new_idx, retry=retry + 1)
            raise ValueError("Empty video sample encountered.")
        points = video_data.get('points')  # (N, T, 2) 像素坐标
        if points is None:
            raise KeyError("Missing 'points' in Kinetics sample.")
        occluded = video_data.get('occluded')  # (N, T)
        if occluded is None:
            visibility = video_data.get('visibility') or video_data.get('vis')
            if visibility is not None:
                occluded = ~np.array(visibility, dtype=bool)
        if occluded is None:
            raise KeyError("Missing 'occluded' or 'visibility' in Kinetics sample.")
        query_points = video_data.get('query_points')  # (N, 3) [t, y, x]
        if query_points is None:
            query_points = video_data.get('queries')
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        if isinstance(occluded, torch.Tensor):
            occluded = occluded.detach().cpu().numpy()
        if isinstance(query_points, torch.Tensor):
            query_points = query_points.detach().cpu().numpy()
        points = np.array(points, dtype=np.float32)
        occluded = np.array(occluded, dtype=bool)
        query_points = np.array(query_points, dtype=np.float32) if query_points is not None else None

        T, H, W, _ = video.shape
        if points.ndim == 3 and points.shape[0] == T and points.shape[1] != T:
            points = np.transpose(points, (1, 0, 2))
        if occluded.ndim == 2 and occluded.shape[0] == T and occluded.shape[1] != T:
            occluded = np.transpose(occluded, (1, 0))
        if points.shape[1] != T:
            min_t = min(points.shape[1], T)
            if min_t <= 0:
                if retry < self.max_retries:
                    new_idx = np.random.randint(0, len(self))
                    return self.__getitem__(new_idx, retry=retry + 1)
                raise ValueError("Invalid temporal alignment.")
            video = video[:min_t]
            points = points[:, :min_t]
            occluded = occluded[:, :min_t]
            T = min_t
        points = self._convert_points_to_internal_yx(points, query_points, H, W)
        # Keep original resolution as a tensor so DataLoader collate produces (B,2),
        # avoiding ambiguous list-of-tensors shapes like (heights, widths).
        original_size = torch.tensor([int(H), int(W)], dtype=torch.int32)
        has_query_points = bool(query_points is not None and query_points.shape[0] > 0)
        if has_query_points and (
            points.shape[0] != query_points.shape[0] or occluded.shape[0] != points.shape[0]
        ):
            min_len = min(points.shape[0], query_points.shape[0], occluded.shape[0])
            if min_len == 0:
                if retry < self.max_retries:
                    new_idx = np.random.randint(0, len(self))
                    return self.__getitem__(new_idx, retry=retry + 1)
                raise ValueError("No valid points remain after alignment.")
            points = points[:min_len]
            occluded = occluded[:min_len]
            query_points = query_points[:min_len]
            has_query_points = query_points.shape[0] > 0
        if points.shape[0] == 0:
            if retry < self.max_retries:
                new_idx = np.random.randint(0, len(self))
                return self.__getitem__(new_idx, retry=retry + 1)
            raise ValueError("No points available in kinetics sample.")
        
        # 截断到最大帧数
        if self.max_frames is not None and T > self.max_frames:
            video = video[:self.max_frames]
            points = points[:, :self.max_frames]
            occluded = occluded[:, :self.max_frames]
            T = self.max_frames
            if has_query_points:
                # 过滤超出截断范围的查询点
                valid_mask = query_points[:, 0] < T
                if valid_mask.any():
                    points = points[valid_mask]
                    occluded = occluded[valid_mask]
                    query_points = query_points[valid_mask]
                else:
                    # 兜底：夹紧并用当前帧位置更新查询点坐标
                    query_points = query_points.copy()
                    if points.shape[0] != query_points.shape[0]:
                        min_len = min(points.shape[0], query_points.shape[0])
                        if min_len == 0:
                            if retry < self.max_retries:
                                new_idx = np.random.randint(0, len(self))
                                return self.__getitem__(new_idx, retry=retry + 1)
                            raise ValueError("No valid points remain after truncation.")
                        points = points[:min_len]
                        occluded = occluded[:min_len]
                        query_points = query_points[:min_len]
                    query_points[:, 0] = np.clip(query_points[:, 0], 0, T - 1)
                    t_idx = query_points[:, 0].astype(np.int64)
                    query_points[:, 1] = points[np.arange(points.shape[0]), t_idx, 0]
                    query_points[:, 2] = points[np.arange(points.shape[0]), t_idx, 1]
                has_query_points = query_points.shape[0] > 0

        if (not has_query_points) and points.shape[0] > 0:
            points, occluded, query_points = self._generate_queries_from_tracks(points, occluded)
            query_generated = True

        # 归一化点坐标到[0, 1]（若已归一化则保持）
        points_normalized = normalize_points_yx(points, H, W)

        query_points_normalized = normalize_query_points_tyx(query_points, H, W)

        # 可选采样点数
        if self.num_points is not None and self.num_points > 0 and points_normalized.shape[0] > self.num_points:
            indices = np.random.choice(points_normalized.shape[0], self.num_points, replace=False)
            points_normalized = points_normalized[indices]
            occluded = occluded[indices]
            query_points_normalized = query_points_normalized[indices]

        # 转换为PyTorch tensor
        video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
        if video.max() > 1.5:
            video = video / 255.0
        target_points = torch.from_numpy(points_normalized).float()
        occluded = torch.from_numpy(occluded.astype(bool))
        query_points = torch.from_numpy(query_points_normalized).float()
        query_points[:, 1:3] = torch.clamp(query_points[:, 1:3], 0.0, 1.0)
        query_points[:, 0] = torch.clamp(query_points[:, 0], 0, T - 1)

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
                query_points, target_points, occluded
            )

        # 调整分辨率
        if self.resolution is not None:
            new_H, new_W = self.resolution
            if video.shape[-2:] != (new_H, new_W):
                video = F.interpolate(video, size=(new_H, new_W), mode='bilinear', align_corners=False)

        return {
            'video': video,  # (T, 3, H, W)
            'query_points': query_points,  # (N, 3) [t, y, x] 归一化
            'target_points': target_points,  # (N, T, 2) [y, x] 归一化
            'occluded': occluded,  # (N, T)
            'video_name': video_name,
            'original_size': original_size,
        }

    def _load_video(self, path: str) -> np.ndarray:
        """读取视频文件"""
        if not CV2_AVAILABLE:
            raise ImportError("cv2 is required to load Kinetics videos from files.")
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
        """调整视频分辨率（保留兼容接口）"""
        T, H, W, C = video.shape
        new_H, new_W = resolution
        video_t = torch.from_numpy(video).permute(0, 3, 1, 2).float()
        video_t = F.interpolate(video_t, size=(new_H, new_W), mode='bilinear', align_corners=False)
        video_t = video_t.permute(0, 2, 3, 1).clamp(0, 255)
        if np.issubdtype(video.dtype, np.integer):
            return video_t.round().to(torch.uint8).cpu().numpy()
        return video_t.cpu().numpy().astype(video.dtype)

    def get_video_info(self, idx):
        """获取视频信息"""
        video_name = self.video_names[idx]
        video_data = self.data[video_name]

        if 'video' in video_data:
            video = video_data['video']
            T, H, W, _ = video.shape
        else:
            if not CV2_AVAILABLE:
                raise ImportError("cv2 is required to load Kinetics videos from files.")
            video_path = video_data.get('video_path')
            if video_path is None:
                video_path = os.path.join(self.root, 'videos', f'{video_name}.mp4')
            cap = cv2.VideoCapture(video_path)
            T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        N = video_data['points'].shape[0]

        return {
            'video_name': video_name,
            'num_frames': T,
            'resolution': (H, W),
            'num_points': N,
        }


def create_kinetics_dataloader(
    root: str,
    batch_size: int = 1,
    num_workers: int = 4,
    **kwargs
):
    """创建Kinetics数据加载器"""
    from torch.utils.data import DataLoader

    dataset = TAPVidKineticsDataset(root, **kwargs)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


if __name__ == '__main__':
    # 测试数据集
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='datasets/tapvid_kinetics')
    args = parser.parse_args()

    dataset = TAPVidKineticsDataset(args.root)

    print(f"\nDataset size: {len(dataset)}")

    # 测试第一个样本
    sample = dataset[0]

    print(f"\nSample 0:")
    print(f"  Video name: {sample['video_name']}")
    print(f"  Video shape: {sample['video'].shape}")
    print(f"  Query points shape: {sample['query_points'].shape}")
    print(f"  Target points shape: {sample['target_points'].shape}")
    print(f"  Occluded shape: {sample['occluded'].shape}")
    print(f"  Original size: {sample['original_size']}")
