from __future__ import annotations

import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import map_coordinates
from torch.utils.data import Dataset

from .augmentation import PointTrackingAugmentation

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    h5py = None

logger = logging.getLogger(__name__)


def _configure_opencv_runtime() -> None:
    """
    Keep OpenCV from spawning its own thread pool inside each DataLoader worker.

    This is usually a net win when PyTorch already parallelizes data loading with
    multiple workers, because otherwise each worker can oversubscribe the CPU.
    """
    try:
        cv2.setNumThreads(max(0, _env_int("FSPT_CV2_NUM_THREADS", 0)))
    except Exception:
        pass
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, None)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; falling back to %d", name, raw, default)
        return int(default)


# Keep dataset caches conservative by default to avoid per-worker RSS blowups.
_RGB_CACHE_SIZE = max(0, _env_int("FSPT_MEGADEPTH_RGB_CACHE_SIZE", 64))
_DEPTH_CACHE_SIZE = max(0, _env_int("FSPT_MEGADEPTH_DEPTH_CACHE_SIZE", 16))
_configure_opencv_runtime()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _resolve_image_from_depth(root: Path, image_value: Any, depth_value: Any) -> Optional[Path]:
    """
    Resolve a MegaDepth image path.

    The prepared pair JSONL stores image paths in the original MegaDepth layout
    (e.g. Undistorted_SfM/.../images/...), but the local tree available in this
    repo keeps the actual RGB frames beside the depth maps as:

      phoenix/S6/.../MegaDepth_v1/<scene>/denseX/imgs/<frame>.jpg

    This helper first tries the raw annotation path and then derives the local
    sibling `imgs/` path from the paired depth file, which is the reliable local
    representation we have in this workspace.
    """
    candidates: List[Path] = []

    if image_value is not None:
        raw_image = _resolve_path(root, image_value)
        candidates.append(raw_image)

    depth_path: Optional[Path] = None
    if depth_value is not None:
        depth_path = _resolve_path(root, depth_value)
        if depth_path.exists():
            parent = depth_path.parent.parent
            stem = depth_path.stem
            candidates.append(parent / "imgs" / f"{stem}.jpg")
            candidates.append(parent / "imgs" / f"{stem}.png")

    if image_value is not None and depth_path is not None:
        image_name = Path(str(image_value)).name
        candidates.append(depth_path.parent.parent / "imgs" / image_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _normalize_resolution(resolution: Optional[Any]) -> Tuple[int, int]:
    if resolution is None:
        return (256, 256)
    if isinstance(resolution, (int, np.integer)):
        size = int(resolution)
        return (size, size)
    if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
        return (int(resolution[0]), int(resolution[1]))
    raise ValueError(f"Invalid resolution: {resolution!r}")


def _normalize_aug_config(augmentation: Optional[object]) -> Dict[str, Any]:
    if augmentation is None:
        return {"enabled": False}
    if isinstance(augmentation, bool):
        return {"enabled": bool(augmentation)}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(augmentation):
            augmentation = OmegaConf.to_container(augmentation, resolve=True)
    except Exception:
        pass
    if isinstance(augmentation, dict):
        cfg = dict(augmentation)
    else:
        try:
            cfg = dict(augmentation)  # type: ignore[arg-type]
        except Exception:
            cfg = {"enabled": True}
    cfg.setdefault("enabled", True)
    return cfg


def _aug_enabled(augmentation: Optional[object]) -> bool:
    if augmentation is None:
        return False
    if isinstance(augmentation, bool):
        return bool(augmentation)
    try:
        enabled = getattr(augmentation, "enabled", None)
        if enabled is not None:
            return bool(enabled)
    except Exception:
        pass
    if isinstance(augmentation, dict):
        return bool(augmentation.get("enabled", True))
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(augmentation):
            return bool(getattr(augmentation, "enabled", True))
    except Exception:
        pass
    return True


def _scene_bucket(scene_id: str, modulo: int = 1000) -> int:
    digest = hashlib.sha1(str(scene_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % int(max(modulo, 1))


def _parse_matrix(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :] = array
        return out
    if array.shape == (4, 4):
        return array.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported matrix shape: {array.shape}")


def _parse_intrinsics(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (3, 3):
        raise ValueError(f"Unsupported intrinsics shape: {array.shape}")
    return array


def _load_jsonl_records(annotation_file: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with annotation_file.open("r", encoding="utf-8") as f:
        for line_idx, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Failed to parse {annotation_file} line {line_idx}: {exc}") from exc
    return records


@lru_cache(maxsize=_RGB_CACHE_SIZE)
def _load_rgb_image(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


@lru_cache(maxsize=_DEPTH_CACHE_SIZE)
def _load_depth_map(path: str) -> np.ndarray:
    if h5py is None:  # pragma: no cover - exercised only on the real dataset
        raise ImportError(
            "h5py is required to load MegaDepth depth maps. Install it with `pip install h5py`."
        )
    with h5py.File(path, "r") as f:  # type: ignore[operator]
        if "depth" not in f:
            keys = ", ".join(sorted(f.keys()))
            raise KeyError(f"Depth key 'depth' not found in {path}. Available keys: {keys}")
        depth = np.asarray(f["depth"], dtype=np.float32)
    return depth


def _resize_image(image: np.ndarray, resolution: Tuple[int, int]) -> np.ndarray:
    out_h, out_w = int(resolution[0]), int(resolution[1])
    if image.shape[0] == out_h and image.shape[1] == out_w:
        return image
    return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def _resize_depth(depth: np.ndarray, resolution: Tuple[int, int]) -> np.ndarray:
    out_h, out_w = int(resolution[0]), int(resolution[1])
    if depth.shape[0] == out_h and depth.shape[1] == out_w:
        return depth
    return cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_NEAREST)


def _sample_depth(depth: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    coords = np.vstack([ys, xs])
    sampled = map_coordinates(depth, coords, order=1, mode="constant", cval=np.nan)
    return sampled.astype(np.float32, copy=False)


def _normalize_xy(xs: np.ndarray, ys: np.ndarray, width: int, height: int) -> np.ndarray:
    denom_x = float(max(int(width) - 1, 1))
    denom_y = float(max(int(height) - 1, 1))
    return np.stack([ys / denom_y, xs / denom_x], axis=1).astype(np.float32, copy=False)


def _world_to_camera(world_from_camera: np.ndarray) -> np.ndarray:
    return np.linalg.inv(world_from_camera).astype(np.float32, copy=False)


class MegaDepthPairDataset(Dataset):
    """
    Supervised two-view correspondence dataset built from the prepared MegaDepth
    JSONL pair lists.

    Each sample is converted to a TAP-Vid-style 2-frame clip:
    - video: (2, 3, H, W)
    - query_points: (N, 3) with query frame index t in {0, 1}
    - target_points: (N, 2, 2) normalized [y, x] coordinates
    - occluded: (N, 2) bool visibility mask

    The dataset supports a deterministic scene-level split so that a single JSONL
    can be reused for both train and validation without leaking scenes.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_frames: Optional[int] = 2,
        num_points: Optional[int] = 256,
        resolution: Optional[Tuple[int, int]] = (256, 256),
        augmentation: Optional[object] = None,
        annotation_file: Optional[str] = None,
        split_strategy: str = "scene_hash",
        val_fraction: float = 0.1,
        bidirectional: bool = True,
        depth_tolerance: float = 0.05,
        seed: int = 42,
        deterministic_sampling: bool = True,
        deterministic_seed: Optional[int] = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"MegaDepth root not found: {self.root}")

        if num_frames is not None and int(num_frames) not in (2,):
            raise ValueError(
                f"MegaDepthPairDataset only supports 2 frames per sample, got num_frames={num_frames}"
            )
        self.num_frames = 2
        self.num_points = int(num_points) if num_points is not None else 256
        if self.num_points <= 0:
            raise ValueError(f"num_points must be positive, got {self.num_points}")

        self.resolution = _normalize_resolution(resolution)
        self.split = str(split or "train").strip().lower()
        self.annotation_file = annotation_file
        self.split_strategy = str(split_strategy or "scene_hash").strip().lower()
        self.val_fraction = float(val_fraction)
        self.bidirectional = bool(bidirectional)
        self.depth_tolerance = float(depth_tolerance)
        self.seed = int(deterministic_seed if deterministic_seed is not None else seed)
        self.deterministic_sampling = bool(deterministic_sampling)

        self.augmentation_cfg = _normalize_aug_config(augmentation)
        self.enabled_augmentation = _aug_enabled(augmentation)
        self.spatial_aug: Optional[PointTrackingAugmentation] = None
        self.temporal_aug = None
        if self.enabled_augmentation:
            crop_size = self.resolution
            random_scale = self.augmentation_cfg.get("random_scale", None)
            if isinstance(random_scale, list):
                random_scale = tuple(random_scale)
            if isinstance(crop_size, list):
                crop_size = tuple(crop_size)
            self.spatial_aug = PointTrackingAugmentation(
                random_crop=bool(self.augmentation_cfg.get("random_crop", True)),
                random_flip=bool(self.augmentation_cfg.get("random_flip", True)),
                color_jitter=float(self.augmentation_cfg.get("color_jitter", 0.0) or 0.0),
                random_scale=random_scale if random_scale is not None else (1.0, 1.0),
                random_rotation=float(self.augmentation_cfg.get("random_rotation", 0.0) or 0.0),
                crop_size=crop_size,
            )
            temporal_cfg = self.augmentation_cfg.get("temporal", {})
            if not isinstance(temporal_cfg, dict):
                temporal_cfg = {}
            if bool(temporal_cfg.get("enabled", False)):
                from .augmentation import TemporalAugmentation

                random_speed = temporal_cfg.get("random_speed", (1.0, 1.0))
                if isinstance(random_speed, list):
                    random_speed = tuple(random_speed)
                self.temporal_aug = TemporalAugmentation(
                    random_reverse=bool(temporal_cfg.get("random_reverse", False)),
                    random_speed=random_speed,
                    random_start=bool(temporal_cfg.get("random_start", False)),
                )

        if annotation_file is None:
            annotation_file = "data/two_view/megadepth_scene_info_pairs_sampled.jsonl"
        annotation_path = Path(str(annotation_file))
        if not annotation_path.is_absolute():
            annotation_path = (_project_root() / annotation_path).resolve()
        if not annotation_path.exists():
            raise FileNotFoundError(f"MegaDepth annotation file not found: {annotation_path}")

        raw_records = _load_jsonl_records(annotation_path)
        self.records = self._prepare_records(raw_records)
        if not self.records:
            raise RuntimeError(
                f"No MegaDepth pair records left after applying split={self.split!r} "
                f"and split_strategy={self.split_strategy!r}."
            )

    def _prepare_records(self, raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if self.split_strategy not in ("scene_hash", "scene", "scene_split", "hash", "all", "none", ""):
            logger.warning(
                "Unknown split_strategy=%r; falling back to no split.", self.split_strategy
            )
        scene_split = self.split_strategy in ("scene_hash", "scene", "scene_split", "hash")
        use_split = scene_split and self.split in ("train", "training", "val", "valid", "validation")
        cutoff = int(round(max(min(self.val_fraction, 1.0), 0.0) * 1000.0))
        cutoff = max(0, min(1000, cutoff))
        keep_val = self.split in ("val", "valid", "validation")
        keep_train = self.split in ("train", "training")

        for idx, record in enumerate(raw_records):
            try:
                scene_id = str(record.get("scene_id", ""))
                image0_raw = str(record.get("image0", ""))
                image1_raw = str(record.get("image1", ""))
                if use_split and cutoff > 0:
                    bucket = _scene_bucket(scene_id, modulo=1000)
                    is_val = bucket < cutoff
                else:
                    is_val = False
                if keep_val and not is_val:
                    continue
                if keep_train and is_val:
                    continue
                intrinsics0 = _parse_intrinsics(record["intrinsics0"])
                intrinsics1 = _parse_intrinsics(record["intrinsics1"])
                if "pose0" not in record or "pose1" not in record:
                    raise KeyError("MegaDepth record must contain absolute pose0 and pose1 matrices")
                pose0 = _parse_matrix(record["pose0"])
                pose1 = _parse_matrix(record["pose1"])

                pair_name = (
                    f"{scene_id}_{Path(image0_raw).stem}__{Path(image1_raw).stem}"
                    if scene_id
                    else f"pair_{idx:06d}"
                )
                out.append(
                    {
                        "dataset": str(record.get("dataset", "megadepth_pairs")),
                        "scene_id": scene_id,
                        "pair_name": pair_name,
                        "image0": str(record.get("image0", "")),
                        "image1": str(record.get("image1", "")),
                        "depth0": str(record.get("depth0", "")),
                        "depth1": str(record.get("depth1", "")),
                        "intrinsics0": intrinsics0,
                        "intrinsics1": intrinsics1,
                        "pose0": pose0,
                        "pose1": pose1,
                    }
                )
            except Exception as exc:
                logger.warning(
                    "Skipping invalid MegaDepth record at index %d: %s", idx, exc
                )
        return out

    def __len__(self) -> int:
        return len(self.records)

    def _make_transform(self, pose_src: np.ndarray, pose_tgt: np.ndarray) -> np.ndarray:
        # The prepared MegaDepth pairs store camera-to-world extrinsics.
        w2c_src = _world_to_camera(pose_src)
        c2w_tgt = pose_tgt
        return (w2c_src @ c2w_tgt).astype(np.float32, copy=False)

    def _sample_direction(
        self,
        *,
        src_idx: int,
        record: Dict[str, Any],
        image_src: np.ndarray,
        image_tgt: np.ndarray,
        depth_src: np.ndarray,
        depth_tgt: np.ndarray,
        intr_src: np.ndarray,
        intr_tgt: np.ndarray,
        pose_src: np.ndarray,
        pose_tgt: np.ndarray,
        num_points: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid_mask = np.isfinite(depth_src) & (depth_src > 0)
        ys, xs = np.nonzero(valid_mask)
        if ys.size == 0:
            raise RuntimeError(f"No valid depth pixels in source view for {record['pair_name']}")

        replace = ys.size < num_points
        choice = rng.choice(ys.size, size=num_points, replace=replace)
        ys_sel = ys[choice].astype(np.float32, copy=False)
        xs_sel = xs[choice].astype(np.float32, copy=False)
        z_src = depth_src[ys_sel.astype(np.int64), xs_sel.astype(np.int64)].astype(np.float32)

        fx_s = float(intr_src[0, 0])
        fy_s = float(intr_src[1, 1])
        cx_s = float(intr_src[0, 2])
        cy_s = float(intr_src[1, 2])
        x_cam = (xs_sel - cx_s) / max(fx_s, 1.0e-6) * z_src
        y_cam = (ys_sel - cy_s) / max(fy_s, 1.0e-6) * z_src
        cam_src = np.stack([x_cam, y_cam, z_src], axis=1)
        cam_src_h = np.concatenate([cam_src, np.ones((cam_src.shape[0], 1), dtype=np.float32)], axis=1)

        world = (pose_src @ cam_src_h.T).T
        w2c_tgt = _world_to_camera(pose_tgt)
        cam_tgt = (w2c_tgt @ world.T).T[:, :3]
        z_tgt = cam_tgt[:, 2]

        fx_t = float(intr_tgt[0, 0])
        fy_t = float(intr_tgt[1, 1])
        cx_t = float(intr_tgt[0, 2])
        cy_t = float(intr_tgt[1, 2])
        with np.errstate(divide="ignore", invalid="ignore"):
            xs_proj = fx_t * (cam_tgt[:, 0] / z_tgt) + cx_t
            ys_proj = fy_t * (cam_tgt[:, 1] / z_tgt) + cy_t

        inside = (
            np.isfinite(xs_proj)
            & np.isfinite(ys_proj)
            & np.isfinite(z_tgt)
            & (z_tgt > 1.0e-6)
            & (xs_proj >= 0.0)
            & (ys_proj >= 0.0)
            & (xs_proj <= float(image_tgt.shape[1] - 1))
            & (ys_proj <= float(image_tgt.shape[0] - 1))
        )
        tgt_depth_at_proj = _sample_depth(depth_tgt, ys_proj, xs_proj)
        depth_ref = np.maximum(
            np.maximum(np.abs(tgt_depth_at_proj), np.abs(z_tgt)),
            1.0,
        )
        depth_ok = np.isfinite(tgt_depth_at_proj) & (
            np.abs(tgt_depth_at_proj - z_tgt) <= self.depth_tolerance * depth_ref
        )
        visible_tgt = inside & depth_ok

        src_norm = _normalize_xy(xs_sel, ys_sel, image_src.shape[1], image_src.shape[0])
        tgt_norm = _normalize_xy(xs_proj, ys_proj, image_tgt.shape[1], image_tgt.shape[0])
        tgt_norm = np.where(np.isfinite(tgt_norm), tgt_norm, src_norm)
        tgt_norm = np.clip(tgt_norm, 0.0, 1.0)
        if src_idx == 0:
            target_points = np.stack([src_norm, tgt_norm], axis=1)
            occluded = np.stack(
                [
                    np.zeros(src_norm.shape[0], dtype=bool),
                    ~visible_tgt,
                ],
                axis=1,
            )
        else:
            target_points = np.stack([tgt_norm, src_norm], axis=1)
            occluded = np.stack(
                [
                    ~visible_tgt,
                    np.zeros(src_norm.shape[0], dtype=bool),
                ],
                axis=1,
            )

        query_points = np.stack(
            [
                np.full(src_norm.shape[0], float(src_idx), dtype=np.float32),
                src_norm[:, 0],
                src_norm[:, 1],
            ],
            axis=1,
        )
        return query_points.astype(np.float32), target_points.astype(np.float32), occluded

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index % len(self.records)]
        if self.deterministic_sampling:
            rng = np.random.default_rng(self.seed + int(index))
        else:
            rng = np.random.default_rng()

        depth0 = _resolve_path(self.root, record["depth0"])
        depth1 = _resolve_path(self.root, record["depth1"])
        image0 = _resolve_image_from_depth(self.root, record.get("image0"), record.get("depth0"))
        image1 = _resolve_image_from_depth(self.root, record.get("image1"), record.get("depth1"))
        if image0 is None or image1 is None:
            raise FileNotFoundError(
                f"Could not resolve MegaDepth images for {record.get('pair_name', index)}"
            )

        image0 = _load_rgb_image(str(image0))
        image1 = _load_rgb_image(str(image1))
        depth0 = _load_depth_map(str(depth0))
        depth1 = _load_depth_map(str(depth1))

        if depth0.shape[:2] != image0.shape[:2]:
            depth0 = _resize_depth(depth0, image0.shape[:2])
        if depth1.shape[:2] != image1.shape[:2]:
            depth1 = _resize_depth(depth1, image1.shape[:2])

        num_points_main = max(int(self.num_points), 1)
        if self.bidirectional:
            num_points_src0 = num_points_main // 2
            num_points_src1 = num_points_main - num_points_src0
        else:
            num_points_src0 = num_points_main
            num_points_src1 = 0

        query_parts: List[np.ndarray] = []
        track_parts: List[np.ndarray] = []
        occluded_parts: List[np.ndarray] = []

        if num_points_src0 > 0:
            q0, t0, o0 = self._sample_direction(
                src_idx=0,
                record=record,
                image_src=image0,
                image_tgt=image1,
                depth_src=depth0,
                depth_tgt=depth1,
                intr_src=record["intrinsics0"],
                intr_tgt=record["intrinsics1"],
                pose_src=record["pose0"],
                pose_tgt=record["pose1"],
                num_points=num_points_src0,
                rng=rng,
            )
            query_parts.append(q0)
            track_parts.append(t0)
            occluded_parts.append(o0)

        if num_points_src1 > 0:
            q1, t1, o1 = self._sample_direction(
                src_idx=1,
                record=record,
                image_src=image1,
                image_tgt=image0,
                depth_src=depth1,
                depth_tgt=depth0,
                intr_src=record["intrinsics1"],
                intr_tgt=record["intrinsics0"],
                pose_src=record["pose1"],
                pose_tgt=record["pose0"],
                num_points=num_points_src1,
                rng=rng,
            )
            query_parts.append(q1)
            track_parts.append(t1)
            occluded_parts.append(o1)

        query_points = np.concatenate(query_parts, axis=0)
        target_points = np.concatenate(track_parts, axis=0)
        occluded = np.concatenate(occluded_parts, axis=0)

        out_h, out_w = self.resolution
        image0 = _resize_image(image0, self.resolution)
        image1 = _resize_image(image1, self.resolution)
        video = np.stack([image0, image1], axis=0)

        video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
        if video.max() > 1.5:
            video = video / 255.0

        query_points_t = torch.from_numpy(query_points).float()
        target_points_t = torch.from_numpy(target_points).float()
        occluded_t = torch.from_numpy(occluded.astype(bool))

        if self.enabled_augmentation and self.spatial_aug is not None:
            outputs = self.spatial_aug(video, target_points_t, occluded_t, query_points_t)
            if len(outputs) == 4:
                video, target_points_t, occluded_t, query_points_t = outputs
            else:
                video, target_points_t, occluded_t = outputs
        if self.enabled_augmentation and self.temporal_aug is not None:
            video, target_points_t, occluded_t, query_points_t = self.temporal_aug(
                video, target_points_t, occluded_t, query_points_t
            )

        # Match the current video resolution so evaluation metrics use the same scale.
        original_size = torch.tensor([int(video.shape[-2]), int(video.shape[-1])], dtype=torch.int32)

        return {
            "video": video,
            "query_points": query_points_t,
            "target_points": target_points_t,
            "occluded": occluded_t,
            "video_name": record["pair_name"],
            "original_size": original_size,
            "scene_id": record.get("scene_id", ""),
        }
