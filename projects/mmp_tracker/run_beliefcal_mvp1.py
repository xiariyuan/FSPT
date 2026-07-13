from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, Mapping

import torch
import yaml

from projects.mmp_tracker.mmp_tracker import MMPTracker
from projects.mmp_tracker.mmp_tracker.uncertainty_head import (
    MMP_UNCERTAINTY_VALUE_NAMES,
)
from projects.mmp_tracker.mmp_tracker.beliefcal_runner import (
    extract_beliefcal_cache_batch,
    load_beliefcal_cache,
    merge_beliefcal_cache_batches,
    run_beliefcal_mvp1_experiment,
    save_beliefcal_cache,
    save_experiment_result,
    synthetic_beliefcal_cache,
)
from projects.mmp_tracker.train_mmp import (
    config_from_dict,
    iterate_limited,
    make_loader,
    resolve_dataset,
)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        if path.endswith(".json"):
            return json.load(handle)
        return yaml.safe_load(handle)


def iterate_loader_window(loader, start_batch: int = 0, limit: int | None = None):
    """Yield a deterministic window from any iterable DataLoader.

    DataLoader implements ``__iter__`` but is not itself an iterator, so batch
    skipping must happen on one shared iterator. Recreating an iterator after
    skipping would silently restart iterable datasets from batch zero.
    """
    start_batch = max(0, int(start_batch))
    iterator = iter(loader)
    skipped = 0
    while skipped < start_batch:
        try:
            next(iterator)
        except StopIteration:
            return
        skipped += 1
    for relative_index, batch in iterate_limited(iterator, limit):
        yield skipped + relative_index, batch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except Exception:
        return None


def normalize_dataset_family(name: object) -> str:
    normalized = str(name or "unknown").lower().replace("-", "_").strip()
    if normalized.startswith("tapvid_"):
        normalized = normalized[len("tapvid_") :]
    aliases = {
        "rgbstacking": "rgb_stacking",
        "stacking": "rgb_stacking",
    }
    return aliases.get(normalized, normalized or "unknown")

def uncertainty_feature_names() -> list[str]:
    values = list(MMP_UNCERTAINTY_VALUE_NAMES)
    return values + [f"{name}_available" for name in values]


def _resolve_annotation_path(data_config: Mapping[str, object]) -> Path | None:
    annotation_file = data_config.get("annotation_file")
    if annotation_file is None:
        return None
    annotation_path = Path(str(annotation_file))
    if annotation_path.is_absolute():
        return annotation_path
    return Path(str(data_config.get("root", "."))) / annotation_path


def dataset_provenance(
    config: Mapping[str, object], split_key: str
) -> Dict[str, object]:
    data_section = config.get("data", {})
    if not isinstance(data_section, Mapping):
        raise ValueError("config.data must be a mapping.")
    raw_data_config = data_section.get(split_key, {})
    if not isinstance(raw_data_config, Mapping):
        raise ValueError(f"config.data.{split_key} must be a mapping.")
    data_config = dict(raw_data_config)
    root = Path(str(data_config.get("root", "."))).resolve()
    annotation_path = _resolve_annotation_path(data_config)
    annotation_sha256 = None
    annotation_payload = None
    source_files = []

    if annotation_path is not None and annotation_path.exists():
        annotation_path = annotation_path.resolve()
        annotation_sha256 = sha256_file(annotation_path)
        try:
            annotation_payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        except Exception:
            annotation_payload = None

    if isinstance(annotation_payload, dict):
        for entry in annotation_payload.get("source_files", []):
            if isinstance(entry, dict) and entry.get("path"):
                source_files.append(str(Path(str(entry["path"])).resolve()))
        if not source_files:
            for entry in annotation_payload.get("shards", []):
                rel = entry.get("path") if isinstance(entry, dict) else entry
                if rel:
                    candidate = Path(str(rel))
                    if not candidate.is_absolute():
                        candidate = root / candidate
                    source_files.append(str(candidate.resolve()))

    dataset_family = normalize_dataset_family(data_config.get("dataset"))
    identity_payload = {
        "dataset": data_config.get("dataset"),
        "dataset_family": dataset_family,
        "root": str(root),
        "split": data_config.get("split", split_key),
        "annotation_path": str(annotation_path) if annotation_path is not None else None,
        "annotation_sha256": annotation_sha256,
    }
    return {
        "data_config": data_config,
        "dataset_family": dataset_family,
        "dataset_identity": sha256_json(identity_payload),
        "dataset_identity_payload": identity_payload,
        "annotation_manifest": str(annotation_path) if annotation_path is not None else None,
        "annotation_manifest_sha256": annotation_sha256,
        "source_files": sorted(set(source_files)),
    }


def _legacy_checkpoint_audit(
    config: Mapping[str, object],
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> Dict[str, object]:
    model_config = config.get("model", {})
    if not isinstance(model_config, Mapping):
        raise RuntimeError("config.model must be a mapping for legacy checkpoint audit.")
    tracking = model_config.get("tracking", {})
    if not isinstance(tracking, Mapping):
        tracking = {}
    variant = str(model_config.get("variant", "")).strip().lower()
    commit_mode = str(tracking.get("commit_mode", "heuristic")).strip().lower()
    allowed_prefixes = (
        "candidate_scorer.",
        "candidate_gate.",
        "candidate_ranker.",
        "commit_selector.",
    )

    errors = []
    if variant != "localglobal":
        errors.append(f"legacy audit only supports variant=localglobal, got {variant!r}")
    if commit_mode != "heuristic":
        errors.append(
            f"legacy audit requires commit_mode=heuristic, got {commit_mode!r}"
        )
    if unexpected_keys:
        errors.append(f"unexpected keys: {unexpected_keys}")
    unsafe_missing = [
        key
        for key in missing_keys
        if not any(key.startswith(prefix) for prefix in allowed_prefixes)
    ]
    if unsafe_missing:
        errors.append(f"non-whitelisted missing keys: {unsafe_missing}")

    audit = {
        "passed": not errors,
        "variant": variant,
        "commit_mode": commit_mode,
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
        "allowed_missing_prefixes": list(allowed_prefixes),
        "commit_probability_used_as_beliefcal_feature": False,
        "errors": errors,
    }
    if errors:
        raise RuntimeError("Legacy checkpoint audit failed:\n" + json.dumps(audit, indent=2))
    return audit


def audit_cache_bundle(
    cache_paths: Mapping[str, str | Path],
    require_strict_checkpoint: bool = True,
    require_distinct_dataset_families: bool = True,
) -> Dict[str, object]:
    roles = ("train", "calibration", "validation", "test")
    missing_roles = [role for role in roles if role not in cache_paths]
    if missing_roles:
        raise ValueError(f"Missing cache roles: {missing_roles}")

    caches = {}
    manifests = {}
    errors = []
    warnings = []
    for role in roles:
        cache, manifest = load_beliefcal_cache(cache_paths[role])
        caches[role] = cache
        manifests[role] = manifest
        if manifest.get("role") not in (None, role):
            errors.append(f"{role}: manifest role={manifest.get('role')!r}")
        load_mode = manifest.get("checkpoint_load_mode")
        if require_strict_checkpoint and load_mode != "strict":
            errors.append(
                f"{role}: checkpoint_load_mode={load_mode!r}, expected 'strict'"
            )
        residual = cache["gt_px"] - cache["mu_px"] - cache["errors_px"]
        max_residual = float(residual.abs().max().item()) if residual.numel() else 0.0
        if max_residual > 1e-5:
            errors.append(
                f"{role}: gt_px != mu_px + errors_px within tolerance "
                f"(max_abs={max_residual})"
            )
        unique_samples = int(torch.unique(cache["sample_id"]).numel())
        manifest_videos = manifest.get("videos")
        if manifest_videos is not None and unique_samples != int(manifest_videos):
            errors.append(
                f"{role}: unique sample_id count={unique_samples}, "
                f"manifest videos={manifest_videos}"
            )
        dirty = manifest.get("git_dirty")
        if dirty is True:
            errors.append(f"{role}: cache was generated from a dirty git worktree")
        elif dirty is None:
            warnings.append(f"{role}: git dirty state was not recorded")

    feature_hashes = {
        str(manifest.get("feature_order_sha256"))
        for manifest in manifests.values()
        if manifest.get("feature_order_sha256")
    }
    if not feature_hashes:
        warnings.append("no feature-order hash recorded in cache manifests")
    elif len(feature_hashes) != 1:
        errors.append(f"feature-order hash mismatch: {sorted(feature_hashes)}")

    checkpoint_hashes = {
        str(manifest.get("checkpoint_sha256"))
        for manifest in manifests.values()
        if manifest.get("checkpoint_sha256")
    }
    if len(checkpoint_hashes) != 1:
        errors.append(
            f"all roles must use one checkpoint hash, got {sorted(checkpoint_hashes)}"
        )

    config_hashes = {
        str(manifest.get("config_sha256"))
        for manifest in manifests.values()
        if manifest.get("config_sha256")
    }
    if len(config_hashes) > 1:
        warnings.append(
            f"role caches use different full config hashes: {sorted(config_hashes)}"
        )

    model_hashes = {
        str(manifest.get("model_config_sha256"))
        for manifest in manifests.values()
        if manifest.get("model_config_sha256")
    }
    if len(model_hashes) != 1:
        errors.append(
            f"all roles must use one model config hash, got {sorted(model_hashes)}"
        )

    identity_owner = {}
    family_owner = {}
    source_owner = {}
    for role, manifest in manifests.items():
        dataset = manifest.get("dataset")
        if not isinstance(dataset, Mapping):
            warnings.append(f"{role}: no dataset provenance recorded")
            continue
        family = dataset.get("dataset_family")
        if family is None:
            payload = dataset.get("dataset_identity_payload", {})
            if isinstance(payload, Mapping):
                family = normalize_dataset_family(payload.get("dataset"))
            warnings.append(f"{role}: dataset family was inferred from legacy provenance")
        family = str(family)
        if family in family_owner:
            message = (
                f"dataset-family collision: {family_owner[family]} and {role} "
                f"both use family={family!r}"
            )
            if require_distinct_dataset_families:
                errors.append(message)
            else:
                warnings.append("engineering override: " + message)
        else:
            family_owner[family] = role

        identity = dataset.get("dataset_identity")
        if identity is None:
            warnings.append(f"{role}: no dataset identity recorded")
        elif identity in identity_owner:
            errors.append(
                f"dataset identity collision: {identity_owner[identity]} and {role}"
            )
        else:
            identity_owner[identity] = role
        for source in dataset.get("source_files", []):
            source = str(source)
            if source in source_owner:
                errors.append(
                    f"source-file leakage: {source_owner[source]} and {role} use {source}"
                )
            else:
                source_owner[source] = role

    report = {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "roles": {
            role: {
                "path": str(Path(cache_paths[role]).resolve()),
                "cache_sha256": sha256_file(cache_paths[role]),
                "rows": int(caches[role]["features"].shape[0]),
                "videos": int(torch.unique(caches[role]["sample_id"]).numel()),
                "checkpoint_load_mode": manifests[role].get(
                    "checkpoint_load_mode"
                ),
                "dataset_family": (
                    manifests[role].get("dataset", {}).get("dataset_family")
                    if isinstance(manifests[role].get("dataset"), Mapping)
                    else None
                ),
                "dataset_identity": (
                    manifests[role].get("dataset", {}).get("dataset_identity")
                    if isinstance(manifests[role].get("dataset"), Mapping)
                    else None
                ),
            }
            for role in roles
        },
    }
    if errors:
        raise RuntimeError(json.dumps(report, indent=2))
    return report


def command_make_cache(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MMPTracker(config_from_dict(config)).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    try:
        model.load_state_dict(state, strict=True)
        checkpoint_load_mode = "strict"
    except RuntimeError as exc:
        if not args.allow_legacy_checkpoint:
            raise
        incompatible = model.load_state_dict(state, strict=False)
        legacy_audit = _legacy_checkpoint_audit(
            config,
            list(incompatible.missing_keys),
            list(incompatible.unexpected_keys),
        )
        checkpoint_load_mode = {
            "mode": "legacy_strict_false_audited",
            "audit": legacy_audit,
            "original_error": str(exc),
        }
        print(json.dumps(checkpoint_load_mode, indent=2))
    model.eval()

    train_flag = args.role == "train"
    dataset, configured_limit = resolve_dataset(config, args.split_key, train=train_flag)
    loader = make_loader(dataset, batch_size=int(args.batch_size), train=False)
    limit = args.max_batches if args.max_batches is not None else configured_limit

    batches = []
    next_sample_id = 0
    with torch.no_grad():
        for batch_index, batch in iterate_loader_window(
            loader, start_batch=args.start_batch, limit=limit
        ):
            video = batch["video"].to(device)
            query_points = batch["query_points"].to(device)
            gt_tracks = batch["target_points"].to(device)
            occluded = batch["occluded"].to(device)
            pred_tracks, pred_visibility, info = model(video, query_points, return_info=True)
            sample_ids = torch.arange(
                next_sample_id,
                next_sample_id + video.shape[0],
                dtype=torch.long,
                device=device,
            )
            next_sample_id += int(video.shape[0])
            batches.append(
                extract_beliefcal_cache_batch(
                    pred_tracks,
                    pred_visibility,
                    info,
                    gt_tracks,
                    occluded,
                    sample_ids=sample_ids,
                )
            )
    cache = merge_beliefcal_cache_batches(batches)
    feature_names = uncertainty_feature_names()
    manifest = {
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "feature_names": feature_names,
        "feature_order_sha256": sha256_json(feature_names),
        "role": args.role,
        "checkpoint_load_mode": checkpoint_load_mode,
        "config": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "model_config_sha256": sha256_json(config.get("model", {})),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_provenance": (
            checkpoint.get("provenance", {}) if isinstance(checkpoint, dict) else {}
        ),
        "split_key": args.split_key,
        "dataset": dataset_provenance(config, args.split_key),
        "start_batch": args.start_batch,
        "max_batches": limit,
        "rows": int(cache["features"].shape[0]),
        "videos": int(next_sample_id),
    }
    save_beliefcal_cache(cache, args.output, manifest=manifest)
    cache_sha256 = sha256_file(args.output)
    Path(str(args.output) + ".sha256").write_text(
        cache_sha256 + "\n", encoding="utf-8"
    )
    print(json.dumps({**manifest, "cache_sha256": cache_sha256}, indent=2))


def command_fit_caches(args: argparse.Namespace) -> None:
    train_cache, train_manifest = load_beliefcal_cache(args.train_cache)
    calibration_cache, calibration_manifest = load_beliefcal_cache(args.calibration_cache)
    validation_cache, validation_manifest = load_beliefcal_cache(args.validation_cache)
    test_cache, test_manifest = load_beliefcal_cache(args.test_cache)
    cache_audit = audit_cache_bundle(
        {
            "train": args.train_cache,
            "calibration": args.calibration_cache,
            "validation": args.validation_cache,
            "test": args.test_cache,
        },
        require_distinct_dataset_families=not args.allow_shared_dataset_family,
    )
    result = run_beliefcal_mvp1_experiment(
        train_cache,
        calibration_cache,
        validation_cache,
        test_cache,
        seeds=tuple(args.seeds),
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    result["cache_manifests"] = {
        "train": train_manifest,
        "calibration": calibration_manifest,
        "validation": validation_manifest,
        "test": test_manifest,
    }
    result["cache_audit"] = cache_audit
    result["git_head"] = git_head()
    save_experiment_result(result, args.output)
    print(Path(args.output) / "beliefcal_metrics.json")


def command_audit_caches(args: argparse.Namespace) -> None:
    report = audit_cache_bundle(
        {
            "train": args.train_cache,
            "calibration": args.calibration_cache,
            "validation": args.validation_cache,
            "test": args.test_cache,
        },
        require_strict_checkpoint=not args.allow_non_strict_checkpoint,
        require_distinct_dataset_families=not args.allow_shared_dataset_family,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def command_synthetic_smoke(args: argparse.Namespace) -> None:
    train_cache = synthetic_beliefcal_cache(4096, 101, 0)
    calibration_cache = synthetic_beliefcal_cache(2048, 102, 100000)
    validation_cache = synthetic_beliefcal_cache(2048, 103, 200000)
    test_cache = synthetic_beliefcal_cache(4096, 104, 300000)
    result = run_beliefcal_mvp1_experiment(
        train_cache,
        calibration_cache,
        validation_cache,
        test_cache,
        seeds=tuple(args.seeds),
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    result["git_head"] = git_head()
    result["synthetic_smoke"] = True
    save_experiment_result(result, args.output)
    global_nll = result["baselines"]["global_scalar"]["metrics"]["overall"]["nll"]
    for run in result["learned_runs"]:
        learned_nll = run["metrics"]["overall"]["nll"]
        print(f"seed={run['seed']} global_nll={global_nll:.6f} learned_nll={learned_nll:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BeliefCal-MMP strict MVP-1 pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_cache = subparsers.add_parser("make-cache")
    make_cache.add_argument("--config", required=True)
    make_cache.add_argument("--checkpoint", required=True)
    make_cache.add_argument("--split-key", required=True)
    make_cache.add_argument(
        "--role",
        choices=("train", "calibration", "validation", "test"),
        required=True,
    )
    make_cache.add_argument("--output", required=True)
    make_cache.add_argument("--device", default="cuda")
    make_cache.add_argument("--batch-size", type=int, default=1)
    make_cache.add_argument("--max-batches", type=int, default=None)
    make_cache.add_argument("--start-batch", type=int, default=0)
    make_cache.add_argument("--allow-legacy-checkpoint", action="store_true")
    make_cache.set_defaults(func=command_make_cache)

    audit = subparsers.add_parser("audit-caches")
    audit.add_argument("--train-cache", required=True)
    audit.add_argument("--calibration-cache", required=True)
    audit.add_argument("--validation-cache", required=True)
    audit.add_argument("--test-cache", required=True)
    audit.add_argument("--output", default=None)
    audit.add_argument("--allow-non-strict-checkpoint", action="store_true")
    audit.add_argument(
        "--allow-shared-dataset-family",
        action="store_true",
        help="Engineering-only override; formal protocol requires distinct families.",
    )
    audit.set_defaults(func=command_audit_caches)

    fit = subparsers.add_parser("fit-caches")
    fit.add_argument("--train-cache", required=True)
    fit.add_argument("--calibration-cache", required=True)
    fit.add_argument("--validation-cache", required=True)
    fit.add_argument("--test-cache", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    fit.add_argument("--hidden-dim", type=int, default=128)
    fit.add_argument("--epochs", type=int, default=30)
    fit.add_argument("--batch-size", type=int, default=4096)
    fit.add_argument("--learning-rate", type=float, default=1e-3)
    fit.add_argument("--device", default="cpu")
    fit.add_argument(
        "--allow-shared-dataset-family",
        action="store_true",
        help="Engineering-only override; formal protocol requires distinct families.",
    )
    fit.set_defaults(func=command_fit_caches)

    smoke = subparsers.add_parser("synthetic-smoke")
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--seeds", nargs="+", type=int, default=[17])
    smoke.add_argument("--hidden-dim", type=int, default=64)
    smoke.add_argument("--epochs", type=int, default=15)
    smoke.add_argument("--batch-size", type=int, default=512)
    smoke.add_argument("--learning-rate", type=float, default=2e-3)
    smoke.add_argument("--device", default="cpu")
    smoke.set_defaults(func=command_synthetic_smoke)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
