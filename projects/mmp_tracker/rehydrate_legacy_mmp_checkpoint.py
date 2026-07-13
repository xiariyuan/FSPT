from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import yaml

from projects.mmp_tracker.mmp_tracker import MMPTracker
from projects.mmp_tracker.mmp_tracker.uncertainty_head import (
    MMP_UNCERTAINTY_VALUE_NAMES,
    build_mmp_uncertainty_features,
)
from projects.mmp_tracker.train_mmp import config_from_dict


ALLOWED_LOCALGLOBAL_MISSING_PREFIXES: Tuple[str, ...] = (
    "candidate_scorer.",
    "candidate_gate.",
    "candidate_ranker.",
    "commit_selector.",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            return json.load(handle)
        return yaml.safe_load(handle)


def _all_allowed(keys: Iterable[str], prefixes: Tuple[str, ...]) -> bool:
    return all(any(key.startswith(prefix) for prefix in prefixes) for key in keys)


def rehydrate_checkpoint(
    config_path: Path,
    source_checkpoint_path: Path,
    output_checkpoint_path: Path,
) -> Dict[str, object]:
    config = load_config(config_path)
    model_config = config_from_dict(config)
    variant = str(model_config.variant).strip().lower()
    commit_mode = str(model_config.tracking.commit_mode).strip().lower()

    if variant != "localglobal":
        raise ValueError(
            "Safe legacy rehydration is currently restricted to variant='localglobal'."
        )
    if commit_mode != "heuristic":
        raise ValueError(
            "Safe legacy rehydration currently requires commit_mode='heuristic'."
        )
    if "commit_probability" in MMP_UNCERTAINTY_VALUE_NAMES:
        raise RuntimeError(
            "The strict MVP feature contract unexpectedly includes commit_probability."
        )

    source_payload = torch.load(source_checkpoint_path, map_location="cpu")
    source_state = source_payload.get("model", source_payload)
    if not isinstance(source_state, dict):
        raise ValueError("Source checkpoint does not contain a model state_dict.")

    model = MMPTracker(model_config)
    incompatible = model.load_state_dict(source_state, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)

    if unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected_keys}")
    if not missing_keys:
        raise RuntimeError(
            "Source checkpoint already loads strictly; rehydration is unnecessary."
        )
    if not _all_allowed(missing_keys, ALLOWED_LOCALGLOBAL_MISSING_PREFIXES):
        raise RuntimeError(
            "Checkpoint has non-audited missing keys: "
            + json.dumps(missing_keys, indent=2)
        )

    full_state = model.state_dict()
    for key in missing_keys:
        full_state[key] = torch.zeros_like(full_state[key])

    model.load_state_dict(full_state, strict=True)

    fresh = MMPTracker(model_config)
    fresh.load_state_dict(full_state, strict=True)
    fresh.eval()

    torch.manual_seed(20260713)
    video = torch.randn(1, 4, 3, 32, 32)
    query_points = torch.tensor(
        [[[0.0, 0.25, 0.25], [0.0, 0.75, 0.70]]], dtype=torch.float32
    )
    with torch.no_grad():
        tracks_a, visibility_a, info_a = model.eval()(video, query_points, return_info=True)
        tracks_b, visibility_b, info_b = fresh(video, query_points, return_info=True)
        features_a, _ = build_mmp_uncertainty_features(info_a, visibility_a)
        features_b, _ = build_mmp_uncertainty_features(info_b, visibility_b)

    audit = {
        "tracks_max_abs_diff": float((tracks_a - tracks_b).abs().max().item()),
        "visibility_max_abs_diff": float(
            (visibility_a - visibility_b).abs().max().item()
        ),
        "features_max_abs_diff": float((features_a - features_b).abs().max().item()),
    }
    if any(value != 0.0 for value in audit.values()):
        raise RuntimeError(f"Strict reload determinism audit failed: {audit}")

    provenance = {
        "format_version": 1,
        "kind": "beliefcal_mmp_legacy_rehydration",
        "source_checkpoint": str(source_checkpoint_path.resolve()),
        "source_checkpoint_sha256": sha256_file(source_checkpoint_path),
        "source_config": str(config_path.resolve()),
        "source_config_sha256": sha256_file(config_path),
        "rehydration_git_head": git_head(),
        "variant": variant,
        "commit_mode": commit_mode,
        "missing_keys_neutralized_to_zero": missing_keys,
        "unexpected_keys": unexpected_keys,
        "allowed_missing_prefixes": list(ALLOWED_LOCALGLOBAL_MISSING_PREFIXES),
        "uncertainty_feature_names": list(MMP_UNCERTAINTY_VALUE_NAMES),
        "commit_probability_used_as_feature": False,
        "strict_reload": True,
        "strict_reload_audit": audit,
        "research_scope": (
            "Frozen localglobal trajectory/visibility and strict MVP-1 uncertainty "
            "features only. Neutralized heads are outside the executed localglobal "
            "trajectory path and outside the uncertainty feature contract."
        ),
    }

    output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "model": full_state,
        "epoch": source_payload.get("epoch"),
        "metrics": source_payload.get("metrics", {}),
        "provenance": provenance,
    }
    torch.save(output_payload, output_checkpoint_path)
    provenance["output_checkpoint"] = str(output_checkpoint_path.resolve())
    provenance["output_checkpoint_sha256"] = sha256_file(output_checkpoint_path)

    manifest_path = output_checkpoint_path.with_suffix(
        output_checkpoint_path.suffix + ".manifest.json"
    )
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, ensure_ascii=False)

    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a strict, audited checkpoint from a legacy localglobal MMP checkpoint."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    args = parser.parse_args()

    result = rehydrate_checkpoint(
        Path(args.config), Path(args.source_checkpoint), Path(args.output_checkpoint)
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
