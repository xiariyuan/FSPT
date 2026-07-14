from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, Mapping, Sequence

import torch
import yaml

from projects.mmp_tracker.mmp_tracker import MMPTracker
from projects.mmp_tracker.mmp_tracker.uncertainty_head import (
    MMP_UNCERTAINTY_VALUE_NAMES,
)
from projects.mmp_tracker.mmp_tracker.beliefcal_runner import (
    apply_scalar_variance_calibration,
    evaluate_variance_method,
    extract_beliefcal_cache_batch,
    fit_scalar_variance_calibration,
    load_beliefcal_cache,
    merge_beliefcal_cache_batches,
    predict_learned_variance,
    run_beliefcal_mvp1_experiment,
    save_beliefcal_cache,
    save_experiment_result,
    synthetic_beliefcal_cache,
    train_uncertainty_head,
)
from projects.mmp_tracker.mmp_tracker.conditional_calibration import (
    CONDITIONAL_CALIBRATION_L2,
    CONDITIONAL_CALIBRATION_LEARNING_RATE,
    CONDITIONAL_CALIBRATION_MAX_SCALE,
    CONDITIONAL_CALIBRATION_MIN_SCALE,
    CONDITIONAL_CALIBRATION_STEPS,
    CONDITIONAL_SELECTION_RELATIVE_NLL,
    apply_conditional_shared_scale_calibration,
    fit_conditional_shared_scale_calibration,
    select_conditional_candidate,
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
    dataset_family = normalize_dataset_family(data_config.get("dataset"))
    annotation_path = _resolve_annotation_path(data_config)
    if annotation_path is None and dataset_family == "davis":
        default_davis_annotation = root / "tapvid_davis.pkl"
        if default_davis_annotation.exists():
            annotation_path = default_davis_annotation
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

    if annotation_path is not None and annotation_path.exists() and not isinstance(
        annotation_payload, dict
    ):
        source_files.append(str(annotation_path.resolve()))

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
    require_distinct_dataset_families: bool = False,
    require_complete_provenance: bool = False,
    required_roles: Sequence[str] = ("train", "calibration", "validation", "test"),
) -> Dict[str, object]:
    roles = tuple(str(role) for role in required_roles)
    allowed_roles = {"train", "calibration", "validation", "test"}
    if not roles or any(role not in allowed_roles for role in roles):
        raise ValueError(f"Invalid required_roles={roles!r}")
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
            message = f"{role}: git dirty state was not recorded"
            if require_complete_provenance:
                errors.append(message)
            else:
                warnings.append(message)
        elif require_complete_provenance and dirty is not False:
            errors.append(f"{role}: git_dirty must be exactly false")

        if require_complete_provenance:
            required_manifest_fields = (
                "role",
                "checkpoint_sha256",
                "model_config_sha256",
                "feature_order_sha256",
                "config_sha256",
                "git_head",
            )
            for field in required_manifest_fields:
                if not manifest.get(field):
                    errors.append(f"{role}: missing required manifest field {field}")

    feature_hashes = {
        str(manifest.get("feature_order_sha256"))
        for manifest in manifests.values()
        if manifest.get("feature_order_sha256")
    }
    if not feature_hashes:
        message = "no feature-order hash recorded in cache manifests"
        if require_complete_provenance:
            errors.append(message)
        else:
            warnings.append(message)
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
        message = f"role caches use different full config hashes: {sorted(config_hashes)}"
        if require_complete_provenance:
            errors.append(message)
        else:
            warnings.append(message)
    elif require_complete_provenance and len(config_hashes) != 1:
        errors.append(
            f"all roles must use one recorded full config hash, got {sorted(config_hashes)}"
        )

    git_heads = {
        str(manifest.get("git_head"))
        for manifest in manifests.values()
        if manifest.get("git_head")
    }
    if require_complete_provenance and len(git_heads) != 1:
        errors.append(f"all roles must use one git head, got {sorted(git_heads)}")
    elif len(git_heads) > 1:
        warnings.append(f"role caches use different git heads: {sorted(git_heads)}")

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
            message = f"{role}: no dataset provenance recorded"
            if require_complete_provenance:
                errors.append(message)
            else:
                warnings.append(message)
            continue
        family = dataset.get("dataset_family")
        if family is None:
            payload = dataset.get("dataset_identity_payload", {})
            if isinstance(payload, Mapping):
                family = normalize_dataset_family(payload.get("dataset"))
            message = f"{role}: dataset family was inferred from legacy provenance"
            if require_complete_provenance:
                errors.append(f"{role}: missing explicit dataset_family")
            else:
                warnings.append(message)
        family = str(family) if family is not None else ""
        if family in family_owner:
            message = (
                f"dataset-family collision: {family_owner[family]} and {role} "
                f"both use family={family!r}"
            )
            if require_distinct_dataset_families:
                errors.append(message)
            else:
                warnings.append("shared-family diagnostic: " + message)
        else:
            family_owner[family] = role

        identity = dataset.get("dataset_identity")
        if identity is None:
            message = f"{role}: no dataset identity recorded"
            if require_complete_provenance:
                errors.append(message)
            else:
                warnings.append(message)
        elif identity in identity_owner:
            errors.append(
                f"dataset identity collision: {identity_owner[identity]} and {role}"
            )
        else:
            identity_owner[identity] = role
        source_files = dataset.get("source_files", [])
        if not isinstance(source_files, (list, tuple)) or not source_files:
            message = f"{role}: no source files recorded"
            if require_complete_provenance:
                errors.append(message)
            else:
                warnings.append(message)
            source_files = []
        annotation_manifest = dataset.get("annotation_manifest")
        annotation_hash = dataset.get("annotation_manifest_sha256")
        if require_complete_provenance and not annotation_manifest:
            errors.append(f"{role}: missing annotation_manifest")
        if require_complete_provenance and not annotation_hash:
            errors.append(f"{role}: missing annotation_manifest_sha256")
        for source in source_files:
            source = str(source)
            if require_complete_provenance and not Path(source).exists():
                errors.append(f"{role}: source file does not exist: {source}")
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
        "policy": {
            "require_strict_checkpoint": bool(require_strict_checkpoint),
            "require_distinct_dataset_families": bool(
                require_distinct_dataset_families
            ),
            "require_complete_provenance": bool(require_complete_provenance),
        },
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
        require_distinct_dataset_families=args.require_distinct_dataset_families,
        require_complete_provenance=args.require_complete_provenance,
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
        feature_profile=args.feature_profile,
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


def command_fit_pretest_heads(args: argparse.Namespace) -> None:
    """Fit uncertainty heads and scalar calibration without loading a test cache."""
    if git_dirty() is True:
        raise RuntimeError(
            "Refusing pretest fitting from a dirty git worktree"
        )
    cache_paths = {
        "train": args.train_cache,
        "calibration": args.calibration_cache,
        "validation": args.validation_cache,
    }
    cache_audit = audit_cache_bundle(
        cache_paths,
        require_strict_checkpoint=True,
        require_distinct_dataset_families=args.require_distinct_dataset_families,
        require_complete_provenance=args.require_complete_provenance,
        required_roles=("train", "calibration", "validation"),
    )
    train_cache, train_manifest = load_beliefcal_cache(args.train_cache)
    calibration_cache, calibration_manifest = load_beliefcal_cache(
        args.calibration_cache
    )
    validation_cache, validation_manifest = load_beliefcal_cache(
        args.validation_cache
    )

    learned_runs = []
    for seed in args.seeds:
        state = train_uncertainty_head(
            train_cache,
            validation_cache,
            seed=int(seed),
            hidden_dim=int(args.hidden_dim),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            device=args.device,
            feature_profile=args.feature_profile,
        )
        calibration_raw = predict_learned_variance(
            calibration_cache, state, device=args.device
        )
        validation_raw = predict_learned_variance(
            validation_cache, state, device=args.device
        )
        scalar_state = fit_scalar_variance_calibration(
            calibration_cache["errors_px"],
            calibration_raw,
            min_std_px=float(state["min_std_px"]),
            max_std_px=float(state["max_std_px"]),
        )
        scalar_validation = apply_scalar_variance_calibration(
            validation_raw, scalar_state
        )
        learned_runs.append(
            {
                "seed": int(seed),
                "feature_profile": state["feature_profile"],
                "selected_feature_names": state["selected_feature_names"],
                "best_val_nll": float(state["best_val_nll"]),
                "raw_validation_metrics": evaluate_variance_method(
                    validation_cache, validation_raw
                ),
                "calibration_state": scalar_state,
                "scalar_validation_metrics": evaluate_variance_method(
                    validation_cache, scalar_validation
                ),
                "state": state,
            }
        )

    result = {
        "format_version": 1,
        "kind": "beliefcal_mvp1_pretest_head_fit",
        "protocol_amendments": ["A1", "A2"],
        "test_loaded_during_fit": False,
        "feature_profile": str(args.feature_profile),
        "training_hyperparameters": {
            "hidden_dim": int(args.hidden_dim),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "seeds": [int(seed) for seed in args.seeds],
        },
        "learned_runs": learned_runs,
        "cache_manifests": {
            "train": train_manifest,
            "calibration": calibration_manifest,
            "validation": validation_manifest,
        },
        "cache_audit": cache_audit,
        "git_head": git_head(),
        "git_dirty": git_dirty(),
    }
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "beliefcal_pretest_result.pt"
    torch.save(result, result_path)
    summary = {
        "format_version": result["format_version"],
        "kind": result["kind"],
        "protocol_amendments": result["protocol_amendments"],
        "test_loaded_during_fit": result["test_loaded_during_fit"],
        "feature_profile": result["feature_profile"],
        "training_hyperparameters": result["training_hyperparameters"],
        "cache_audit": cache_audit,
        "git_head": result["git_head"],
        "git_dirty": result["git_dirty"],
        "learned_runs": [
            {
                "seed": run["seed"],
                "feature_profile": run["feature_profile"],
                "selected_feature_names": list(run["selected_feature_names"]),
                "best_val_nll": run["best_val_nll"],
                "raw_validation_metrics": run["raw_validation_metrics"],
                "calibration_state": run["calibration_state"],
                "scalar_validation_metrics": run["scalar_validation_metrics"],
            }
            for run in learned_runs
        ],
    }
    summary_path = output_dir / "beliefcal_pretest_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(result_path)
    print(summary_path)


def command_audit_caches(args: argparse.Namespace) -> None:
    cache_paths = {
        "train": args.train_cache,
        "calibration": args.calibration_cache,
        "validation": args.validation_cache,
        "test": args.test_cache,
    }
    try:
        report = audit_cache_bundle(
            cache_paths,
            require_strict_checkpoint=not args.allow_non_strict_checkpoint,
            require_distinct_dataset_families=args.require_distinct_dataset_families,
            require_complete_provenance=args.require_complete_provenance,
        )
    except RuntimeError as exc:
        try:
            report = json.loads(str(exc))
        except json.JSONDecodeError:
            raise
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        raise
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _load_learned_run(result_path: str | Path, run_index: int):
    payload = torch.load(Path(result_path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("Invalid BeliefCal experiment result")
    learned_runs = payload.get("learned_runs")
    if not isinstance(learned_runs, list) or not learned_runs:
        raise ValueError("Experiment result contains no learned runs")
    run_index = int(run_index)
    if run_index < 0 or run_index >= len(learned_runs):
        raise ValueError(f"run_index={run_index} outside [0,{len(learned_runs) - 1}]")
    learned_run = learned_runs[run_index]
    if not isinstance(learned_run, Mapping) or "state" not in learned_run:
        raise ValueError("Selected learned run has no serialized state")
    return payload, learned_run


def _formal_manifest_issues(manifest: Mapping[str, object], role: str) -> list[str]:
    issues = []
    required = (
        "checkpoint_sha256",
        "model_config_sha256",
        "feature_order_sha256",
        "git_head",
        "git_dirty",
    )
    for key in required:
        if manifest.get(key) is None:
            issues.append(f"{role}: missing {key}")
    if manifest.get("git_dirty") is not False:
        issues.append(f"{role}: git_dirty must be false")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        issues.append(f"{role}: missing dataset provenance")
    else:
        if dataset.get("dataset_identity") is None:
            issues.append(f"{role}: missing dataset_identity")
        source_files = dataset.get("source_files")
        if not isinstance(source_files, list) or not source_files:
            issues.append(f"{role}: missing source_files")
    return issues


def _overall_nll(metrics: Mapping[str, Mapping[str, float]]) -> float:
    overall = metrics.get("overall")
    if not isinstance(overall, Mapping) or "nll" not in overall:
        raise ValueError("Metrics contain no overall NLL")
    return float(overall["nll"])


def command_fit_conditional_calibration(args: argparse.Namespace) -> None:
    """Fit A2 calibration candidates without accepting or loading a test cache."""
    if git_dirty() is True:
        raise RuntimeError(
            "Refusing to freeze an A2 calibration bundle from a dirty git worktree"
        )
    result, learned_run = _load_learned_run(args.result, args.run_index)
    if result.get("test_loaded_during_fit") is not False:
        raise RuntimeError(
            "A2 bundle fitting requires a source result that certifies test-free fitting"
        )
    calibration_cache, calibration_manifest = load_beliefcal_cache(
        args.calibration_cache
    )
    validation_cache, validation_manifest = load_beliefcal_cache(
        args.validation_cache
    )
    if calibration_manifest.get("role") not in (None, "calibration"):
        raise ValueError("Calibration cache manifest has the wrong role")
    if validation_manifest.get("role") not in (None, "validation"):
        raise ValueError("Validation cache manifest has the wrong role")

    cache_audit = result.get("cache_audit", {})
    audit_roles = cache_audit.get("roles", {}) if isinstance(cache_audit, Mapping) else {}
    for role, path in (
        ("calibration", args.calibration_cache),
        ("validation", args.validation_cache),
    ):
        expected = audit_roles.get(role, {}) if isinstance(audit_roles, Mapping) else {}
        expected_hash = expected.get("cache_sha256") if isinstance(expected, Mapping) else None
        if expected_hash and str(expected_hash) != sha256_file(path):
            raise RuntimeError(f"{role} cache hash differs from the source experiment")

    learned_state = learned_run["state"]
    seed = int(learned_run["seed"])
    calibration_raw = predict_learned_variance(
        calibration_cache, learned_state, device=args.device
    )
    validation_raw = predict_learned_variance(
        validation_cache, learned_state, device=args.device
    )
    scalar_state = fit_scalar_variance_calibration(
        calibration_cache["errors_px"],
        calibration_raw,
        min_std_px=float(learned_state["min_std_px"]),
        max_std_px=float(learned_state["max_std_px"]),
    )
    conditional_state = fit_conditional_shared_scale_calibration(
        calibration_cache,
        calibration_raw,
        initial_scale=float(scalar_state["scale"]),
        seed=seed,
        device=args.device,
        min_std_px=float(learned_state["min_std_px"]),
        max_std_px=float(learned_state["max_std_px"]),
        steps=CONDITIONAL_CALIBRATION_STEPS,
        learning_rate=CONDITIONAL_CALIBRATION_LEARNING_RATE,
        l2_penalty=CONDITIONAL_CALIBRATION_L2,
        min_scale=CONDITIONAL_CALIBRATION_MIN_SCALE,
        max_scale=CONDITIONAL_CALIBRATION_MAX_SCALE,
    )

    scalar_calibration = apply_scalar_variance_calibration(
        calibration_raw, scalar_state
    )
    scalar_validation = apply_scalar_variance_calibration(validation_raw, scalar_state)
    conditional_calibration = apply_conditional_shared_scale_calibration(
        calibration_cache, calibration_raw, conditional_state, device=args.device
    )
    conditional_validation = apply_conditional_shared_scale_calibration(
        validation_cache, validation_raw, conditional_state, device=args.device
    )
    calibration_metrics = {
        "shared_scalar": evaluate_variance_method(
            calibration_cache, scalar_calibration
        ),
        "conditional_affine": evaluate_variance_method(
            calibration_cache, conditional_calibration
        ),
    }
    validation_metrics = {
        "shared_scalar": evaluate_variance_method(validation_cache, scalar_validation),
        "conditional_affine": evaluate_variance_method(
            validation_cache, conditional_validation
        ),
    }
    selection = select_conditional_candidate(
        _overall_nll(validation_metrics["shared_scalar"]),
        _overall_nll(validation_metrics["conditional_affine"]),
        required_relative_improvement=CONDITIONAL_SELECTION_RELATIVE_NLL,
    )

    formal_issues = _formal_manifest_issues(
        calibration_manifest, "calibration"
    ) + _formal_manifest_issues(validation_manifest, "validation")
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "format_version": 1,
        "kind": "beliefcal_mvp1c_frozen_calibration_bundle",
        "protocol_amendment": "A2",
        "git_head": git_head(),
        "git_dirty": git_dirty(),
        "source_result_path": str(Path(args.result).resolve()),
        "source_result_sha256": sha256_file(args.result),
        "learned_run_index": int(args.run_index),
        "learned_seed": seed,
        "feature_profile": learned_run.get("feature_profile", "full"),
        "learned_state": learned_state,
        "scalar_state": scalar_state,
        "conditional_state": conditional_state,
        "selection": selection,
        "calibration_metrics": calibration_metrics,
        "validation_metrics": validation_metrics,
        "calibration_cache": {
            "path": str(Path(args.calibration_cache).resolve()),
            "sha256": sha256_file(args.calibration_cache),
            "manifest": calibration_manifest,
        },
        "validation_cache": {
            "path": str(Path(args.validation_cache).resolve()),
            "sha256": sha256_file(args.validation_cache),
            "manifest": validation_manifest,
        },
        "formal_eligible_before_test": not formal_issues,
        "formal_eligibility_issues": formal_issues,
        "test_loaded_during_fit": False,
    }
    bundle_path = output_dir / "conditional_calibration_bundle.pt"
    torch.save(bundle, bundle_path)
    summary = {
        key: value
        for key, value in bundle.items()
        if key
        not in {
            "learned_state",
            "scalar_state",
            "conditional_state",
            "calibration_cache",
            "validation_cache",
        }
    }
    summary["scalar_state"] = scalar_state
    summary["conditional_state"] = {
        key: value
        for key, value in conditional_state.items()
        if key not in {"input_mean", "input_std", "model_state"}
    }
    summary["calibration_cache_sha256"] = bundle["calibration_cache"]["sha256"]
    summary["validation_cache_sha256"] = bundle["validation_cache"]["sha256"]
    summary_path = output_dir / "conditional_calibration_selection.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(bundle_path)
    print(summary_path)


def _dataset_provenance_collisions(
    frozen_manifests: Sequence[Mapping[str, object]],
    test_manifest: Mapping[str, object],
) -> list[str]:
    issues = []
    test_dataset = test_manifest.get("dataset")
    if not isinstance(test_dataset, Mapping):
        return ["test: missing dataset provenance"]
    test_identity = test_dataset.get("dataset_identity")
    test_sources = {str(path) for path in test_dataset.get("source_files", [])}
    for manifest in frozen_manifests:
        dataset = manifest.get("dataset")
        if not isinstance(dataset, Mapping):
            continue
        if test_identity is not None and test_identity == dataset.get("dataset_identity"):
            issues.append("test dataset_identity collides with calibration/validation")
        overlap = test_sources & {str(path) for path in dataset.get("source_files", [])}
        for path in sorted(overlap):
            issues.append(f"test source-file collision: {path}")
    return issues


def command_evaluate_frozen_conditional(args: argparse.Namespace) -> None:
    """Evaluate a previously frozen A2 selection; this command fits nothing."""
    bundle = torch.load(Path(args.bundle), map_location="cpu")
    if not isinstance(bundle, Mapping) or bundle.get("kind") != "beliefcal_mvp1c_frozen_calibration_bundle":
        raise ValueError("Invalid frozen A2 calibration bundle")
    if bundle.get("test_loaded_during_fit") is not False:
        raise RuntimeError("Frozen bundle does not certify test-free fitting")
    test_cache, test_manifest = load_beliefcal_cache(args.test_cache)
    if test_manifest.get("role") not in (None, "test"):
        raise ValueError("Test cache manifest has the wrong role")

    frozen_cache_records = [bundle["calibration_cache"], bundle["validation_cache"]]
    for record in frozen_cache_records:
        if sha256_file(args.test_cache) == record["sha256"]:
            raise RuntimeError("Test cache is identical to a fitting cache")
    frozen_manifests = [record["manifest"] for record in frozen_cache_records]
    provenance_issues = _dataset_provenance_collisions(
        frozen_manifests, test_manifest
    )
    if provenance_issues:
        raise RuntimeError(json.dumps(provenance_issues, indent=2))

    reference_manifest = frozen_manifests[0]
    for key in ("checkpoint_sha256", "model_config_sha256", "feature_order_sha256"):
        reference = reference_manifest.get(key)
        observed = test_manifest.get(key)
        if reference is not None and observed is not None and reference != observed:
            raise RuntimeError(f"test {key} differs from frozen calibration bundle")

    raw_variance = predict_learned_variance(
        test_cache, bundle["learned_state"], device=args.device
    )
    selected = bundle["selection"]["selected"]
    if selected == "conditional_affine":
        variance = apply_conditional_shared_scale_calibration(
            test_cache,
            raw_variance,
            bundle["conditional_state"],
            device=args.device,
        )
    elif selected == "shared_scalar":
        variance = apply_scalar_variance_calibration(
            raw_variance, bundle["scalar_state"]
        )
    else:
        raise ValueError(f"Unsupported frozen selection: {selected!r}")

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "format_version": 1,
        "kind": "beliefcal_mvp1c_frozen_test_evaluation",
        "protocol_amendment": "A2",
        "bundle_path": str(Path(args.bundle).resolve()),
        "bundle_sha256": sha256_file(args.bundle),
        "test_cache_path": str(Path(args.test_cache).resolve()),
        "test_cache_sha256": sha256_file(args.test_cache),
        "selected": selected,
        "metrics": evaluate_variance_method(test_cache, variance),
        "test_manifest": test_manifest,
        "formal_eligible": bool(bundle.get("formal_eligible_before_test"))
        and not _formal_manifest_issues(test_manifest, "test"),
        "formal_eligibility_issues": list(
            bundle.get("formal_eligibility_issues", [])
        )
        + _formal_manifest_issues(test_manifest, "test"),
        "git_head": git_head(),
        "git_dirty": git_dirty(),
    }
    output_path = output_dir / "conditional_calibration_test_metrics.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output_path)


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
        "--require-complete-provenance",
        action="store_true",
        help=(
            "Require per-role git state, checkpoint/model/config/feature hashes, "
            "dataset identity, annotation manifest hash, and source files."
        ),
    )
    audit.add_argument(
        "--require-distinct-dataset-families",
        action="store_true",
        help=(
            "Optional stress-test policy. Exact item/video separation is always "
            "required; shared family is otherwise reported as a warning."
        ),
    )
    audit.set_defaults(func=command_audit_caches)

    pretest_fit = subparsers.add_parser("fit-pretest-heads")
    pretest_fit.add_argument("--train-cache", required=True)
    pretest_fit.add_argument("--calibration-cache", required=True)
    pretest_fit.add_argument("--validation-cache", required=True)
    pretest_fit.add_argument("--output", required=True)
    pretest_fit.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    pretest_fit.add_argument("--hidden-dim", type=int, default=128)
    pretest_fit.add_argument("--epochs", type=int, default=30)
    pretest_fit.add_argument("--batch-size", type=int, default=4096)
    pretest_fit.add_argument("--learning-rate", type=float, default=1e-3)
    pretest_fit.add_argument("--device", default="cpu")
    pretest_fit.add_argument(
        "--feature-profile",
        choices=("full", "drop_inert_mmp"),
        default="full",
    )
    pretest_fit.add_argument(
        "--require-complete-provenance", action="store_true"
    )
    pretest_fit.add_argument(
        "--require-distinct-dataset-families", action="store_true"
    )
    pretest_fit.set_defaults(func=command_fit_pretest_heads)

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
        "--require-complete-provenance",
        action="store_true",
        help="Refuse fitting unless all cache manifests satisfy the formal A1 provenance contract.",
    )
    fit.add_argument(
        "--feature-profile",
        choices=("full", "drop_inert_mmp"),
        default="full",
        help="Engineering feature ablation; full remains the preregistered contract.",
    )
    fit.add_argument(
        "--require-distinct-dataset-families",
        action="store_true",
        help=(
            "Optional stress-test policy. Exact item/video separation is always "
            "required; shared family is otherwise reported as a warning."
        ),
    )
    fit.set_defaults(func=command_fit_caches)

    conditional_fit = subparsers.add_parser("fit-conditional-calibration")
    conditional_fit.add_argument("--result", required=True)
    conditional_fit.add_argument("--run-index", type=int, default=0)
    conditional_fit.add_argument("--calibration-cache", required=True)
    conditional_fit.add_argument("--validation-cache", required=True)
    conditional_fit.add_argument("--output", required=True)
    conditional_fit.add_argument("--device", default="cpu")
    conditional_fit.set_defaults(func=command_fit_conditional_calibration)

    conditional_eval = subparsers.add_parser("evaluate-frozen-conditional")
    conditional_eval.add_argument("--bundle", required=True)
    conditional_eval.add_argument("--test-cache", required=True)
    conditional_eval.add_argument("--output", required=True)
    conditional_eval.add_argument("--device", default="cpu")
    conditional_eval.set_defaults(func=command_evaluate_frozen_conditional)

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
