#!/usr/bin/env python3
"""Build deterministic sparse PointOdyssey caches using official CoTracker3."""
from __future__ import annotations

import argparse
import json
import math
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_safe_redetection_cache import (
    SAFE_REDETECTION_CACHE_SCHEMA,
    SAFE_REDETECTION_INDEX_SCHEMA,
    canonical_json_sha256,
    file_sha256,
    tensor_sha256,
    verify_event_cache,
)


def deterministic(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def execution_identity(*, allow_dirty: bool) -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
    ).strip()
    if status and not allow_dirty:
        raise RuntimeError("formal cache export requires a clean Git worktree")
    paths = {
        "builder": Path(__file__).resolve(),
        "cache_module": REPO_ROOT / "projects/mmp_tracker/mmp_tracker/routeD_safe_redetection_cache.py",
        "model_module": REPO_ROOT / "projects/mmp_tracker/mmp_tracker/routeD_safe_redetection.py",
        "protocol_module": REPO_ROOT / "projects/mmp_tracker/mmp_tracker/routeD_pointodyssey_protocol.py",
    }
    return {
        "git_head": head,
        "git_worktree_clean": not bool(status),
        "git_status_porcelain": status,
        "source_sha256": {key: file_sha256(path) for key, path in paths.items()},
    }


def resolve_repository_artifact(reference: str) -> Path:
    path = Path(reference)
    resolved = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"protocol evidence escapes repository root: {reference}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def verify_protocol_prerequisites(protocol: dict[str, Any]) -> dict[str, str]:
    prerequisites = protocol["official_prerequisites"]
    baseline = resolve_repository_artifact(prerequisites["cotracker3_alignment_summary"])
    parity = resolve_repository_artifact(prerequisites["official_ajrd_parity"])
    if file_sha256(baseline) != prerequisites["cotracker3_alignment_summary_sha256"]:
        raise RuntimeError("official CoTracker3 alignment artifact hash drift")
    if file_sha256(parity) != prerequisites["official_ajrd_parity_sha256"]:
        raise RuntimeError("official AJ_RD parity artifact hash drift")
    baseline_payload = json.loads(baseline.read_text())
    parity_payload = json.loads(parity.read_text())
    if not prerequisites["cotracker3_alignment_pass"] or not baseline_payload["gate"]["pass"]:
        raise RuntimeError("official baseline gate failed")
    if (
        prerequisites["official_ajrd_max_difference"] != 0.0
        or not parity_payload["pass"]
        or parity_payload["maximum_absolute_scalar_difference"] != 0.0
    ):
        raise RuntimeError("official AJ_RD parity is not exact")
    if baseline_payload["official_source"]["commit"] != prerequisites["official_cotracker3_source_commit"]:
        raise RuntimeError("official CoTracker3 source commit drift")
    if parity_payload["official_commit"] != prerequisites["official_ajrd_source_commit"]:
        raise RuntimeError("official AJ_RD source commit drift")
    return {
        "cotracker3_alignment_summary": baseline.relative_to(REPO_ROOT).as_posix(),
        "official_ajrd_parity": parity.relative_to(REPO_ROOT).as_posix(),
    }


def import_official_predictor(source_root: Path):
    source = str(source_root.resolve())
    sys.path = [row for row in sys.path if "baselines/cotracker" not in row]
    if source in sys.path:
        sys.path.remove(source)
    sys.path.insert(0, source)
    from cotracker.predictor import CoTrackerOnlinePredictor
    return CoTrackerOnlinePredictor


def load_frame(path: Path, size: int = 256) -> tuple[torch.Tensor, tuple[int, int]]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    height, width = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(image).permute(2, 0, 1).float(), (height, width)


def load_clip(scene: Path, start: int, end: int, size: int = 256) -> tuple[torch.Tensor, tuple[int, int]]:
    paths = [scene / "rgbs" / f"rgb_{frame:05d}.jpg" for frame in range(start, end)]
    rows = []
    original = None
    for path in paths:
        tensor, shape = load_frame(path, size)
        original = shape if original is None else original
        if shape != original:
            raise ValueError("frame dimensions changed inside scene")
        rows.append(tensor)
    return torch.stack(rows).unsqueeze(0), original


def selected_event_frames(event: dict[str, Any]) -> list[int]:
    start = int(event["window_start"])
    end = int(event["window_end"])
    last = int(event["last_visible_frame"])
    reentry = int(event["reappearance_frame"])
    frames = {max(start, min(end - 1, last + offset)) for offset in (-32, -16, -8, -4, -2, 0)}
    occlusion = list(range(last + 1, reentry))
    if occlusion:
        sample_count = min(8, len(occlusion))
        for index in np.linspace(0, len(occlusion) - 1, sample_count, dtype=int).tolist():
            frames.add(occlusion[index])
        frames.update(occlusion[-2:])
    frames.update(range(reentry, min(reentry + 16, end)))
    return sorted(frame for frame in frames if start <= frame < end)


def run_online(predictor, video: torch.Tensor, query_rel_xy: torch.Tensor) -> None:
    query = torch.tensor(
        [[[float(query_rel_xy[0]), float(query_rel_xy[1]), float(query_rel_xy[2])]]],
        device=video.device,
        dtype=torch.float32,
    )
    predictor(
        video_chunk=video[:, :1],
        is_first_step=True,
        queries=query,
        add_support_grid=False,
        grid_size=0,
    )
    with torch.no_grad():
        for start in range(0, video.shape[1] - predictor.step, predictor.step):
            chunk = video[:, start : start + predictor.step * 2]
            predictor(
                video_chunk=chunk,
                is_first_step=False,
                add_support_grid=False,
                grid_size=0,
            )
    state = predictor.model.online_coords_predicted
    if state is None or state.shape[1] < video.shape[1]:
        raise RuntimeError("incomplete online tracker state")


def compute_sparse_fmaps(predictor, frames: torch.Tensor, device: str) -> torch.Tensor:
    frames = frames.to(device)
    count, channels = frames.shape[:2]
    interp_h, interp_w = predictor.interp_shape
    resized = F.interpolate(frames, size=(interp_h, interp_w), mode="bilinear", align_corners=True)
    normalized = 2.0 * (resized / 255.0) - 1.0
    outputs = []
    with torch.no_grad():
        for start in range(0, count, 8):
            outputs.append(predictor.model.fnet(normalized[start : start + 8]))
    return F.normalize(torch.cat(outputs).float(), dim=1, eps=1e-12)


def resolve_scene(root: Path, role: str, name: str) -> Path:
    split = "train" if role == "fit" else "val"
    scene = root / split / name
    if not (scene / "anno.npz").is_file():
        raise FileNotFoundError(scene)
    return scene


def build_event(
    *,
    predictor,
    event: dict[str, Any],
    scene: Path,
    protocol_sha: str,
    role: str,
    device: str,
    output_path: Path,
    checkpoint_sha: str,
    official_source_commit: str,
    execution: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    start = int(event["window_start"])
    protocol_end = int(event["window_end"])
    end = min(protocol_end, int(event["reappearance_frame"]) + 16)
    if end <= int(event["reappearance_frame"]):
        raise RuntimeError("event inference clip does not include reappearance")
    video, (height, width) = load_clip(scene, start, end)
    with np.load(scene / "anno.npz", allow_pickle=False) as data:
        tracks = data["trajs_2d"][:, int(event["point_index"])].astype(np.float32)
        visible = (data["visibs"][:, int(event["point_index"])] & data["valids"][:, int(event["point_index"])]).astype(bool)
    before = np.where(visible[start : int(event["last_visible_frame"]) + 1])[0]
    if before.size == 0:
        raise RuntimeError("event has no visible query frame in window")
    query_abs = start + int(before[0])
    query_rel = query_abs - start
    query_xy_orig = tracks[query_abs]
    if not np.isfinite(query_xy_orig).all():
        raise RuntimeError("non-finite query coordinate")
    scale_x = 255.0 / float(max(width - 1, 1)); scale_y = 255.0 / float(max(height - 1, 1))
    query_xy = torch.tensor([query_xy_orig[0] * scale_x, query_xy_orig[1] * scale_y])
    video = video.to(device)
    run_online(predictor, video, torch.tensor([query_rel, query_xy[0], query_xy[1]]))

    selected_abs = sorted(set(selected_event_frames(event)) | {query_abs})
    selected_rel = torch.tensor([frame - start for frame in selected_abs], dtype=torch.long)
    selected_video = video[0, selected_rel].detach().cpu()
    fmaps = compute_sparse_fmaps(predictor, selected_video, device)

    count = end - start
    raw_coord = predictor.model.online_coords_predicted[0, :count, 0].float()
    raw_vis = predictor.model.online_vis_predicted[0, :count, 0].float()
    raw_conf = predictor.model.online_conf_predicted[0, :count, 0].float()
    interp_h, interp_w = predictor.interp_shape
    native = raw_coord.clone()
    native[..., 0] *= 255.0 / float(max(interp_w - 1, 1))
    native[..., 1] *= 255.0 / float(max(interp_h - 1, 1))
    vis_prob = torch.sigmoid(raw_vis)
    conf_prob = torch.sigmoid(raw_conf)
    native_visible = vis_prob * conf_prob > 0.6

    gt_xy = torch.from_numpy(tracks[selected_abs]).float()
    gt_xy[..., 0] *= scale_x; gt_xy[..., 1] *= scale_y
    gt_visible = torch.from_numpy(visible[selected_abs]).bool()
    age = np.zeros(len(visible), dtype=np.int64)
    for frame in range(1, len(visible)):
        age[frame] = age[frame - 1] + 1 if not visible[frame - 1] else 0

    tensors = {
        "feature_maps_f16": fmaps.detach().cpu().half().contiguous(),
        "frame_indices_abs": torch.tensor(selected_abs, dtype=torch.long),
        "frame_indices_rel": selected_rel,
        "native_coords_xy_px": native[selected_rel].detach().cpu(),
        "native_visibility_probability": vis_prob[selected_rel].detach().cpu(),
        "native_confidence_probability": conf_prob[selected_rel].detach().cpu(),
        "native_visibility": native_visible[selected_rel].detach().cpu(),
        "gt_coords_xy_px": gt_xy,
        "gt_visible": gt_visible,
        "occlusion_age": torch.from_numpy(age[selected_abs]).long(),
        "query_frame_rel": torch.tensor(query_rel, dtype=torch.long),
        "query_coord_xy_px": query_xy.float(),
        "reappearance_frame_rel": torch.tensor(int(event["reappearance_frame"]) - start, dtype=torch.long),
        "last_visible_frame_rel": torch.tensor(int(event["last_visible_frame"]) - start, dtype=torch.long),
    }
    payload = {
        "schema_version": SAFE_REDETECTION_CACHE_SCHEMA,
        "provenance": {
            "protocol_sha256": protocol_sha,
            "role": role,
            "event": event,
            "scene": str(scene),
            "scene_annotation_sha256": file_sha256(scene / "anno.npz"),
            "official_source_commit": official_source_commit,
            "checkpoint_sha256": checkpoint_sha,
            "execution": execution,
            "input_raster": 256,
            "query_independence": "one event / one query / one forward stream",
            "ground_truth_use": "event selection and labels only; not native inference or feature extraction",
            "selected_frame_rule": "frozen sparse pre/occlusion/reappearance schedule",
            "locked_data_read": {"internal_holdout": False, "pointodyssey_test": False, "kinetics_1144": False},
        },
        "tensors": tensors,
        "tensor_sha256": {key: tensor_sha256(value) for key, value in tensors.items()},
        "runtime": {
            "seconds": time.time() - started,
            "clip_frames": count,
            "protocol_window_frames": protocol_end - start,
            "inference_end_rule": "min(protocol_window_end, reappearance+16)",
            "selected_frames": len(selected_abs),
            "original_height": height,
            "original_width": width,
            "interp_shape": list(predictor.interp_shape),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    verify_event_cache(output_path, expected_protocol_sha256=protocol_sha, expected_role=role)
    return {
        "event_identity_sha256": event["event_identity_sha256"],
        "scene": event["scene"],
        "duration_bucket": event["duration_bucket"],
        "invisibility_duration": event["invisibility_duration"],
        "sidecar": str(output_path),
        "sidecar_sha256": file_sha256(output_path),
        "selected_frames": len(selected_abs),
        "clip_frames": count,
        "seconds": payload["runtime"]["seconds"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=str(REPO_ROOT / "configs/routeD_safe_redetection_pointodyssey_protocol_v0.json"))
    ap.add_argument("--role", choices=("fit", "model_validation"), required=True)
    ap.add_argument("--data-root", default="/gemini/code/FSPT/datasets/pointodyssey")
    ap.add_argument("--official-source-root", default=str(REPO_ROOT / "outputs/official_cotracker3_alignment_20260718/vendor/co-tracker-src"))
    ap.add_argument("--checkpoint", default="/gemini/code/FSPT/weights/scaled_online.pth")
    ap.add_argument("--output-root", default=str(REPO_ROOT / "outputs/routeD_safe_redetection_cache_20260718"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--start-offset", type=int, default=0)
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--index-name", default="")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    deterministic(17018)
    execution = execution_identity(allow_dirty=args.allow_dirty)
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    protocol_sha = file_sha256(protocol_path)
    prerequisite_artifacts = verify_protocol_prerequisites(protocol)
    all_events = list(protocol["partitions"][args.role]["events"])
    if args.start_offset < 0 or args.start_offset >= len(all_events):
        raise ValueError("start-offset outside event partition")
    events = all_events[args.start_offset :]
    if args.max_events > 0:
        events = events[: args.max_events]
    output_dir = Path(args.output_root).resolve() / args.role
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    expected_checkpoint = "205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218"
    if file_sha256(checkpoint) != expected_checkpoint:
        raise RuntimeError("official checkpoint hash drift")
    Predictor = import_official_predictor(Path(args.official_source_root))
    predictor = Predictor(checkpoint=str(checkpoint), window_len=16).to(args.device).eval()
    rows = []
    for local_ordinal, event in enumerate(events):
        ordinal = args.start_offset + local_ordinal
        path = output_dir / f"event_{ordinal:05d}_{event['event_identity_sha256'][:12]}.pt"
        if args.resume and path.exists():
            try:
                verify_event_cache(path, expected_protocol_sha256=protocol_sha, expected_role=args.role)
                rows.append({
                    "event_identity_sha256": event["event_identity_sha256"],
                    "scene": event["scene"],
                    "duration_bucket": event["duration_bucket"],
                    "invisibility_duration": event["invisibility_duration"],
                    "sidecar": str(path),
                    "sidecar_sha256": file_sha256(path),
                    "resumed": True,
                })
                continue
            except Exception:
                path.unlink(missing_ok=True)
        print(json.dumps({"stage":"event_start","ordinal":ordinal,"total":len(events),"scene":event["scene"],"duration":event["invisibility_duration"]}), flush=True)
        scene = resolve_scene(Path(args.data_root), args.role, event["scene"])
        row = build_event(
            predictor=predictor,
            event=event,
            scene=scene,
            protocol_sha=protocol_sha,
            role=args.role,
            device=args.device,
            output_path=path,
            checkpoint_sha=expected_checkpoint,
            official_source_commit=protocol["official_prerequisites"].get("official_cotracker3_source_commit", "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"),
            execution=execution,
        )
        rows.append(row)
        print(json.dumps({"stage":"event_complete", **row}), flush=True)
        torch.cuda.empty_cache()
    full_scale = args.max_events == 0
    index = {
        "schema_version": SAFE_REDETECTION_INDEX_SCHEMA,
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "role": args.role,
        "expected_count": len(all_events) if full_scale else len(events),
        "completed_count": len(rows),
        "complete": len(rows) == (len(all_events) if full_scale else len(events)),
        "full_scale": full_scale,
        "events": rows,
        "execution": execution,
        "official_prerequisite_artifacts": prerequisite_artifacts,
        "locked_data_read": {"internal_holdout": False, "pointodyssey_test": False, "kinetics_1144": False},
    }
    index["payload_sha256"] = canonical_json_sha256(index)
    default_index_name = "cache_index.json" if full_scale else "smoke_cache_index.json"
    index_path = output_dir / (args.index_name or default_index_name)
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({"stage":"complete","index":str(index_path),"count":len(rows),"full_scale":full_scale},indent=2), flush=True)


if __name__ == "__main__":
    main()
