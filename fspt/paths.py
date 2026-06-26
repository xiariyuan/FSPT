from __future__ import annotations

from pathlib import Path

from .core.paths import baseline_root, data_root, env_path, output_root, repo_root


def resolve_repo_path(*parts: str | Path) -> Path:
    if len(parts) == 1:
        candidate = Path(parts[0]).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (repo_root() / candidate).resolve()
    joined = Path(*parts).expanduser()
    if joined.is_absolute():
        return joined.resolve()
    return (repo_root() / joined).resolve()


def outputs_dir(*parts: str | Path) -> Path:
    root = output_root()
    return (root / Path(*parts)).resolve() if parts else root


def caches_dir(*parts: str | Path) -> Path:
    root = repo_root() / "caches"
    return (root / Path(*parts)).resolve() if parts else root


__all__ = [
    "baseline_root",
    "caches_dir",
    "data_root",
    "env_path",
    "output_root",
    "outputs_dir",
    "repo_root",
    "resolve_repo_path",
]
