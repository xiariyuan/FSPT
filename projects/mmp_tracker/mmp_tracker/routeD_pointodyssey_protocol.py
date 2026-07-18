"""Deterministic long-occlusion PointOdyssey protocol for Route-D."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DURATION_BUCKETS = (
    ("d1_3", 1, 3),
    ("d4_15", 4, 15),
    ("d16_63", 16, 63),
    ("d64_255", 64, 255),
    ("d256_plus", 256, 10**9),
)


@dataclass(frozen=True)
class PointOdysseyEvent:
    scene: str
    point_index: int
    last_visible_frame: int
    reappearance_frame: int
    invisibility_duration: int
    duration_bucket: str
    window_start: int
    window_end: int
    window_length: int


def stable_key(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


def duration_bucket(duration: int) -> str:
    for name, low, high in DURATION_BUCKETS:
        if low <= int(duration) <= high:
            return name
    raise ValueError(f"duration outside supported range: {duration}")


def discover_scenes(root: Path, split: str) -> list[Path]:
    directory = root / split
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(
        path for path in directory.iterdir()
        if path.is_dir() and (path / "anno.npz").is_file()
    )


def select_scenes(
    scenes: Iterable[Path], *, seed: int, role: str, count: int
) -> list[Path]:
    rows = sorted(scenes, key=lambda path: stable_key(seed, role, path.name))
    if count <= 0 or count > len(rows):
        raise ValueError(f"invalid {role} scene count {count}/{len(rows)}")
    return rows[:count]


def _window_for_event(
    *,
    frame_count: int,
    last_visible: int,
    reappearance: int,
    pre_context: int = 32,
    post_context: int = 96,
) -> tuple[int, int, int] | None:
    required = reappearance - last_visible + pre_context + post_context
    window_length = 512 if required <= 512 else 1024 if required <= 1024 else None
    if window_length is None:
        return None
    desired_start = last_visible - pre_context
    latest_start = reappearance + post_context - window_length
    start = max(0, min(desired_start, latest_start if latest_start > 0 else desired_start))
    start = min(start, max(frame_count - window_length, 0))
    end = min(start + window_length, frame_count)
    if not (start <= last_visible < reappearance < end):
        start = max(0, min(last_visible, reappearance + post_context - window_length))
        end = min(start + window_length, frame_count)
    if not (start <= last_visible < reappearance < end):
        return None
    return int(start), int(end), int(window_length)


def extract_record_breaking_events(
    scene: str,
    visible: np.ndarray,
    trajectories_xy: np.ndarray,
) -> list[PointOdysseyEvent]:
    """Extract official-AJ_RD-eligible reappearance events from one scene."""
    vis = np.asarray(visible, dtype=bool)
    tracks = np.asarray(trajectories_xy)
    if vis.ndim != 2 or tracks.shape != vis.shape + (2,):
        raise ValueError("expected visible (T,N) and trajectories (T,N,2)")
    frames, points = vis.shape
    duration = np.zeros((frames, points), dtype=np.int32)
    for frame in range(1, frames):
        duration[frame] = np.where(~vis[frame - 1], duration[frame - 1] + 1, 0)
    reappearance = np.zeros_like(vis)
    reappearance[1:] = vis[1:] & ~vis[:-1]
    indices = np.argwhere(reappearance)  # official flattened (frame, point) order
    max_seen: dict[int, int] = {}
    events: list[PointOdysseyEvent] = []
    for frame, point in indices.tolist():
        current = int(duration[frame, point])
        if current <= max_seen.get(point, -1):
            continue
        max_seen[point] = current
        last_visible = int(frame - current - 1)
        if last_visible < 0:
            continue
        if not (
            np.isfinite(tracks[last_visible, point]).all()
            and np.isfinite(tracks[frame, point]).all()
        ):
            continue
        window = _window_for_event(
            frame_count=frames,
            last_visible=last_visible,
            reappearance=int(frame),
        )
        if window is None:
            continue
        start, end, length = window
        events.append(
            PointOdysseyEvent(
                scene=scene,
                point_index=int(point),
                last_visible_frame=last_visible,
                reappearance_frame=int(frame),
                invisibility_duration=current,
                duration_bucket=duration_bucket(current),
                window_start=start,
                window_end=end,
                window_length=length,
            )
        )
    return events


def balanced_event_selection(
    events: Iterable[PointOdysseyEvent],
    *,
    seed: int,
    quota_per_bucket: int,
) -> list[PointOdysseyEvent]:
    by_bucket: dict[str, list[PointOdysseyEvent]] = {
        name: [] for name, _, _ in DURATION_BUCKETS
    }
    for event in events:
        by_bucket[event.duration_bucket].append(event)
    selected = []
    for bucket, rows in by_bucket.items():
        ordered = sorted(
            rows,
            key=lambda event: stable_key(
                seed,
                event.scene,
                bucket,
                event.point_index,
                event.reappearance_frame,
            ),
        )
        selected.extend(ordered[:quota_per_bucket])
    return sorted(
        selected,
        key=lambda event: (event.scene, event.reappearance_frame, event.point_index),
    )


def event_counts(events: Iterable[PointOdysseyEvent]) -> dict[str, Any]:
    rows = list(events)
    return {
        "total": len(rows),
        "by_bucket": {
            name: sum(event.duration_bucket == name for event in rows)
            for name, _, _ in DURATION_BUCKETS
        },
        "by_dmin": {
            str(value): sum(event.invisibility_duration >= value for event in rows)
            for value in (1, 4, 16, 64, 256)
        },
        "window_length": {
            "512": sum(event.window_length == 512 for event in rows),
            "1024": sum(event.window_length == 1024 for event in rows),
        },
    }
