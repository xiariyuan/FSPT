from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict

import torch
import yaml

from projects.mmp_tracker.mmp_tracker import MMPTracker
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


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


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
        checkpoint_load_mode = {
            "mode": "legacy_strict_false",
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "original_error": str(exc),
        }
        print(json.dumps(checkpoint_load_mode, indent=2))
    model.eval()

    train_flag = args.split_key == "train"
    dataset, configured_limit = resolve_dataset(config, args.split_key, train=train_flag)
    loader = make_loader(dataset, batch_size=int(args.batch_size), train=False)
    limit = args.max_batches if args.max_batches is not None else configured_limit

    batches = []
    next_sample_id = 0
    with torch.no_grad():
        for _, batch in iterate_limited(loader, limit):
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
    manifest = {
        "git_head": git_head(),
        "checkpoint_load_mode": checkpoint_load_mode,
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split_key": args.split_key,
        "max_batches": limit,
        "rows": int(cache["features"].shape[0]),
        "videos": int(next_sample_id),
    }
    save_beliefcal_cache(cache, args.output, manifest=manifest)
    print(json.dumps(manifest, indent=2))


def command_fit_caches(args: argparse.Namespace) -> None:
    train_cache, train_manifest = load_beliefcal_cache(args.train_cache)
    calibration_cache, calibration_manifest = load_beliefcal_cache(args.calibration_cache)
    validation_cache, validation_manifest = load_beliefcal_cache(args.validation_cache)
    test_cache, test_manifest = load_beliefcal_cache(args.test_cache)
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
    result["git_head"] = git_head()
    save_experiment_result(result, args.output)
    print(Path(args.output) / "beliefcal_metrics.json")


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
    make_cache.add_argument("--split-key", choices=("train", "val"), required=True)
    make_cache.add_argument("--output", required=True)
    make_cache.add_argument("--device", default="cuda")
    make_cache.add_argument("--batch-size", type=int, default=1)
    make_cache.add_argument("--max-batches", type=int, default=None)
    make_cache.add_argument("--allow-legacy-checkpoint", action="store_true")
    make_cache.set_defaults(func=command_make_cache)

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
