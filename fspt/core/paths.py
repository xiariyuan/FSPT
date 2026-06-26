from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()


def data_root() -> Path:
    return env_path("FSPT_DATA_ROOT", str(repo_root() / "datasets"))


def output_root() -> Path:
    return env_path("FSPT_OUTPUT_ROOT", str(repo_root() / "outputs"))


def baseline_root() -> Path:
    return env_path("FSPT_BASELINE_ROOT", str(repo_root() / "baselines"))
