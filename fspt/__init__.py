"""FSPT package root."""

from __future__ import annotations

from .paths import baseline_root, caches_dir, data_root, output_root, outputs_dir, repo_root, resolve_repo_path

__all__ = [
    "__version__",
    "baseline_root",
    "caches_dir",
    "data_root",
    "output_root",
    "outputs_dir",
    "repo_root",
    "resolve_repo_path",
]

__version__ = "0.0.0"
