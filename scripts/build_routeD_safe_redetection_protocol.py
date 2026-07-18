#!/usr/bin/env python3
"""Build the immutable PointOdyssey long-occlusion protocol for Route-D."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_pointodyssey_protocol import (
    DURATION_BUCKETS,
    balanced_event_selection,
    discover_scenes,
    event_counts,
    extract_record_breaking_events,
    select_scenes,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def repository_relative(path: Path) -> str:
    """Return a portable POSIX path rooted at the repository checkout."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"official evidence must live inside the repository: {resolved}") from exc


def scene_identity(scene: Path, *, split: str) -> dict[str, Any]:
    annotation = scene / "anno.npz"
    rgb_dir = scene / "rgbs"
    frames = sorted(rgb_dir.glob("rgb_*.jpg")) if rgb_dir.is_dir() else []
    filelist = [f"{path.name}|{path.stat().st_size}" for path in frames]
    samples = []
    if frames:
        for index in sorted({0, len(frames) // 2, len(frames) - 1}):
            samples.append({"name": frames[index].name, "sha256": sha256(frames[index])})
    return {
        "scene": scene.name,
        "annotation": f"{split}/{scene.name}/anno.npz",
        "annotation_size": annotation.stat().st_size,
        "annotation_sha256": sha256(annotation),
        "rgb_frame_count": len(frames),
        "rgb_filelist_sha256": hashlib.sha256("\n".join(filelist).encode()).hexdigest(),
        "sampled_rgb_sha256": samples,
    }


def load_scene_events(scene: Path):
    with np.load(scene / "anno.npz", allow_pickle=False) as data:
        tracks = data["trajs_2d"]
        visible = data["visibs"].astype(bool) & data["valids"].astype(bool)
    return extract_record_breaking_events(scene.name, visible, tracks)


def rows_for_role(
    scenes: list[Path], *, role: str, seed: int, quota: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    identities = []
    events = []
    for ordinal, scene in enumerate(scenes):
        print(json.dumps({"stage": "scene_start", "role": role, "ordinal": ordinal, "scene": scene.name}), flush=True)
        source_split = "train" if role == "fit" else "val"
        identity = scene_identity(scene, split=source_split)
        all_events = load_scene_events(scene)
        selected = balanced_event_selection(
            all_events, seed=seed, quota_per_bucket=quota
        )
        identity["all_eligible_event_counts"] = event_counts(all_events)
        identity["selected_event_counts"] = event_counts(selected)
        identities.append(identity)
        for event in selected:
            row = dict(event.__dict__)
            row["role"] = role
            row["scene_annotation_sha256"] = identity["annotation_sha256"]
            row["event_identity_sha256"] = canonical_sha(row)
            events.append(row)
        print(json.dumps({"stage": "scene_complete", "role": role, "scene": scene.name, "all": len(all_events), "selected": len(selected)}), flush=True)
    return identities, events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/gemini/code/FSPT/datasets/pointodyssey")
    ap.add_argument("--seed", type=int, default=17018)
    ap.add_argument("--fit-scenes", type=int, default=24)
    ap.add_argument("--validation-scenes", type=int, default=8)
    ap.add_argument("--fit-quota-per-bucket", type=int, default=2)
    ap.add_argument("--validation-quota-per-bucket", type=int, default=4)
    ap.add_argument("--official-baseline-summary", default=str(REPO_ROOT / "docs/generated/OFFICIAL_COTRACKER3_DAVIS_FIRST_ALIGNMENT_2026-07-18.json"))
    ap.add_argument("--ajrd-parity", default=str(REPO_ROOT / "docs/generated/OFFICIAL_TAPNEXTPP_AJRD_PARITY_2026-07-18.json"))
    ap.add_argument("--output", default=str(REPO_ROOT / "configs/routeD_safe_redetection_pointodyssey_protocol_v0.json"))
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    baseline_path = Path(args.official_baseline_summary).resolve()
    parity_path = Path(args.ajrd_parity).resolve()
    baseline = json.loads(baseline_path.read_text())
    parity = json.loads(parity_path.read_text())
    if not baseline["gate"]["pass"] or not parity["pass"]:
        raise RuntimeError("official baseline or AJ_RD parity gate is not satisfied")

    train = discover_scenes(root, "train")
    val = discover_scenes(root, "val")
    test = discover_scenes(root, "test")
    fit = select_scenes(train, seed=args.seed, role="fit", count=args.fit_scenes)
    validation = select_scenes(
        val, seed=args.seed, role="model_validation", count=args.validation_scenes
    )
    validation_names = {scene.name for scene in validation}
    locked_holdout = [scene for scene in val if scene.name not in validation_names]

    fit_id, fit_events = rows_for_role(
        fit, role="fit", seed=args.seed, quota=args.fit_quota_per_bucket
    )
    val_id, val_events = rows_for_role(
        validation,
        role="model_validation",
        seed=args.seed,
        quota=args.validation_quota_per_bucket,
    )
    print(json.dumps({"stage": "lock_holdout", "scenes": len(locked_holdout)}), flush=True)
    holdout_id = [scene_identity(scene, split="val") for scene in locked_holdout]
    print(json.dumps({"stage": "lock_test", "scenes": len(test)}), flush=True)
    test_id = [scene_identity(scene, split="test") for scene in test]

    protocol = {
        "schema_version": "routeD_safe_redetection_pointodyssey_protocol_v0",
        "date": "2026-07-18",
        "seed": args.seed,
        "purpose": "Long-occlusion and reappearance training for a frozen CoTracker3 backbone",
        "official_prerequisites": {
            "cotracker3_alignment_summary": repository_relative(baseline_path),
            "cotracker3_alignment_summary_sha256": sha256(baseline_path),
            "cotracker3_alignment_pass": True,
            "official_ajrd_parity": repository_relative(parity_path),
            "official_ajrd_parity_sha256": sha256(parity_path),
            "official_ajrd_source_commit": parity["official_commit"],
            "official_ajrd_source_sha256": parity["official_source_sha256"],
            "official_ajrd_max_difference": parity["maximum_absolute_scalar_difference"],
            "official_cotracker3_source_commit": baseline["official_source"]["commit"],
        },
        "selection_contract": {
            "scene_selection": "ascending sha256(seed|role|scene_name)",
            "event_definition": "official AJ_RD record-breaking reappearance events",
            "duration_buckets": [list(row) for row in DURATION_BUCKETS],
            "fit_quota_per_scene_per_bucket": args.fit_quota_per_bucket,
            "model_validation_quota_per_scene_per_bucket": args.validation_quota_per_bucket,
            "event_selection": "ascending sha256(seed|scene|bucket|point|reappearance)",
            "window_lengths": [512, 1024],
            "pre_occlusion_context_frames": 32,
            "post_reappearance_context_frames": 96,
        },
        "partitions": {
            "fit": {"scenes": fit_id, "events": fit_events, "counts": event_counts([type("E", (), row)() for row in fit_events])},
            "model_validation": {"scenes": val_id, "events": val_events, "counts": event_counts([type("E", (), row)() for row in val_events])},
            "locked_internal_holdout": {
                "scenes": holdout_id,
                "events_parsed": False,
                "use": "one-time evaluation only after all fit/model-validation gates",
            },
            "locked_pointodyssey_test": {
                "scenes": test_id,
                "events_parsed": False,
                "use": "untouched external synthetic test; no model selection",
            },
        },
        "data_exposure": {
            "fit_annotations_parsed": True,
            "model_validation_annotations_parsed": True,
            "locked_internal_holdout_annotations_parsed": False,
            "locked_pointodyssey_test_annotations_parsed": False,
            "tapvid_davis_role": "official baseline alignment and later frozen development benchmark; already historically exposed",
            "tapvid_kinetics_official_1144": "frozen; must not rerun or tune",
        },
    }
    protocol["payload_sha256"] = canonical_sha(protocol)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({
        "stage": "complete",
        "output": str(output),
        "sha256": sha256(output),
        "fit_scenes": len(fit_id),
        "fit_events": len(fit_events),
        "validation_scenes": len(val_id),
        "validation_events": len(val_events),
        "locked_holdout_scenes": len(holdout_id),
        "locked_test_scenes": len(test_id),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
