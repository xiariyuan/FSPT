#!/usr/bin/env python3
"""Serial runner for Route A experiments.

This runner launches the Route A long-occlusion main line one step at a time on
a single GPU. It skips steps whose evaluation summary already exists, launches
training when a checkpoint is missing, and launches evaluation when a checkpoint
exists.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

DEFAULT_CHAIN = [
    {
        "name": "fspt_routeA_stage3_relocal_accept_visiblebank_l30_eval256_from_kinetics_guardrail",
        "config": "configs/fspt_routeA_stage3_relocal_accept_visiblebank_l30_eval256_from_kinetics_guardrail.yaml",
        "checkpoint": "checkpoints/fspt_routeA_stage3_relocal_accept_visiblebank_l30_eval256_from_kinetics_guardrail/best_aj.pth",
        "eval_output": "outputs/fspt_routeA_stage3_relocal_accept_visiblebank_l30_eval256_from_kinetics_guardrail",
        "eval_config": "configs/fspt_routeA_stage3_relocal_accept_visiblebank_l30_eval256.yaml",
    },
    {
        "name": "fspt_routeA_stage3_relocal_accept_visiblebank_posterior_l30_eval256",
        "config": "configs/fspt_routeA_stage3_relocal_accept_visiblebank_posterior_l30_eval256.yaml",
        "checkpoint": "checkpoints/fspt_routeA_stage3_relocal_accept_visiblebank_posterior_l30_eval256/best_aj.pth",
        "eval_output": "outputs/fspt_routeA_stage3_relocal_accept_visiblebank_posterior_l30_eval256",
        "eval_config": "configs/fspt_routeA_stage3_relocal_accept_visiblebank_posterior_l30_eval256.yaml",
    },
    {
        "name": "fspt_routeA_stage3_verifier_v1_from_posterior",
        "config": "configs/fspt_routeA_stage3_verifier_v1.yaml",
        "checkpoint": "checkpoints/fspt_routeA_stage3_verifier_v1_from_posterior/best_aj.pth",
        "eval_output": "outputs/fspt_routeA_stage3_verifier_v1_from_posterior",
        "eval_config": "configs/fspt_routeA_stage3_verifier_v1.yaml",
    },
]


@dataclass
class Step:
    name: str
    config: Path
    checkpoint: Path
    eval_output: Path
    eval_config: Path


def _as_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _step_from_dict(d: dict) -> Step:
    return Step(
        name=d["name"],
        config=_as_path(d["config"]),
        checkpoint=_as_path(d["checkpoint"]),
        eval_output=_as_path(d["eval_output"]),
        eval_config=_as_path(d["eval_config"]),
    )


def _run(cmd: List[str], env: Optional[dict] = None) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=str(PROJECT_ROOT))


def _base_env(gpu: str) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    return env


def _summary_path(step: Step) -> Path:
    return step.eval_output / "summary.json"


def _has_summary(step: Step) -> bool:
    return _summary_path(step).exists()


def _has_checkpoint(step: Step) -> bool:
    return step.checkpoint.exists() and step.checkpoint.stat().st_size > 0


def _latest_checkpoint(step: Step) -> Path:
    candidates = [
        step.checkpoint,
        step.checkpoint.with_name("best.pth"),
        step.checkpoint.with_name("latest.pth"),
        step.checkpoint.with_name("early_stop_best.pth"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(f"No checkpoint found under {step.checkpoint.parent}")


def _train_step(step: Step, gpu: str = "0", epochs: Optional[int] = None) -> None:
    cmd = [
        PYTHON,
        str(PROJECT_ROOT / "train.py"),
        "--config",
        str(step.config),
        "--experiment.name",
        step.name,
        "--paths.output_dir",
        str(PROJECT_ROOT / "outputs"),
        "--paths.checkpoint_dir",
        str(PROJECT_ROOT / "checkpoints"),
        "--logging.log_dir",
        str(PROJECT_ROOT / "logs"),
    ]
    if epochs is not None:
        cmd += ["--training.epochs", str(int(epochs))]
    _run(cmd, env=_base_env(gpu))


def _eval_step(step: Step, gpu: str = "0") -> None:
    ckpt = _latest_checkpoint(step)
    step.eval_output.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        str(PROJECT_ROOT / "evaluate.py"),
        "--checkpoint",
        str(ckpt),
        "--config",
        str(step.eval_config),
        "--dataset",
        "davis",
        "--output-dir",
        str(step.eval_output),
        "--compare-base",
    ]
    _run(cmd, env=_base_env(gpu))


def _iter_steps(chain: Iterable[dict], from_step: Optional[str]) -> List[Step]:
    steps = [_step_from_dict(d) for d in chain]
    if not from_step:
        return steps
    found = False
    filtered: List[Step] = []
    for step in steps:
        if step.name == from_step:
            found = True
        if found:
            filtered.append(step)
    if not filtered:
        raise SystemExit(f"Unknown --from-step {from_step}")
    return filtered


def _read_summary_mean(summary_path: Path, key: str = "AJ_delta") -> Optional[float]:
    if not summary_path.exists():
        return None
    try:
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    results = obj.get("results", {})
    if not isinstance(results, dict):
        return None
    davis_delta = results.get("davis_delta", {})
    if not isinstance(davis_delta, dict):
        return None
    metric = davis_delta.get(key, {})
    if not isinstance(metric, dict):
        return None
    value = metric.get("mean", None)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _read_summary_metric(summary_path: Path, group: str, key: str) -> Optional[float]:
    if not summary_path.exists():
        return None
    try:
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    results = obj.get("results", {})
    if not isinstance(results, dict):
        return None
    section = results.get(group, {})
    if not isinstance(section, dict):
        return None
    metric = section.get(key, {})
    if not isinstance(metric, dict):
        return None
    value = metric.get("mean", None)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Serial Route A runner")
    ap.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES value (single GPU only)")
    ap.add_argument("--from-step", type=str, default=None, help="Start from this step name")
    ap.add_argument("--only-eval", action="store_true", help="Skip training and only run evaluation for steps with checkpoints")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = ap.parse_args()

    steps = _iter_steps(DEFAULT_CHAIN, args.from_step)
    for idx, step in enumerate(steps):
        print(f"\n=== {step.name} ===")
        print(f"config: {step.config}")
        print(f"checkpoint: {step.checkpoint}")
        print(f"eval_output: {step.eval_output}")

        if _has_summary(step):
            print("[skip] summary.json already exists")
            continue

        if not args.only_eval and not _has_checkpoint(step):
            print("[train] checkpoint missing; launching training")
            if args.dry_run:
                print("dry-run: would train")
            else:
                _train_step(step, gpu=args.gpu)
        else:
            print("[train] checkpoint exists or only-eval requested; skipping training")

        if args.dry_run:
            print("dry-run: would evaluate")
            continue

        if not _has_checkpoint(step):
            raise FileNotFoundError(f"Cannot evaluate {step.name}: checkpoint missing")
        print("[eval] launching evaluation")
        _eval_step(step, gpu=args.gpu)

        if idx < len(steps) - 1:
            longocc30 = _read_summary_metric(_summary_path(step), "davis_delta", "AJ_longocc30")
            if longocc30 is None:
                longocc30 = _read_summary_metric(_summary_path(step), "davis", "AJ_longocc30")
            if longocc30 is not None and longocc30 < -1.0e-4:
                print(f"[stop] {step.name} regressed on AJ_longocc30; stop before next step")
                break

    print("\nAll requested Route A steps completed.")


if __name__ == "__main__":
    main()
