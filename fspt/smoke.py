"""Minimal smoke entrypoint for the fspt package."""
from __future__ import annotations

from . import __version__


def main() -> int:
    from .paths import caches_dir, data_root, output_root, outputs_dir, repo_root, resolve_repo_path

    root = repo_root()
    readme = resolve_repo_path("README.md")
    coords_ok = False
    metrics_ok = False
    schema_ok = False
    try:
        from .coords import yx_norm_to_xy_pixel as _yx_norm_to_xy_pixel

        coords_ok = callable(_yx_norm_to_xy_pixel)
    except Exception:
        pass
    try:
        from .reentry_metrics import aggregate_reappearance_ajrd as _aggregate_reappearance_ajrd

        metrics_ok = callable(_aggregate_reappearance_ajrd)
    except Exception:
        pass
    try:
        from .io.attempt0_schema import load_attempt0_cache as _load_attempt0_cache

        schema_ok = callable(_load_attempt0_cache)
    except Exception:
        pass

    print(f"fspt {__version__}")
    print(f"repo_root={root}")
    print(f"repo_root.name={root.name}")
    print(f"readme_exists={readme.exists()}")
    print(f"data_root={data_root()}")
    print(f"output_root={output_root()}")
    print(f"outputs_dir={outputs_dir()}")
    print(f"caches_dir={caches_dir()}")
    print(f"coords_wrapper={coords_ok}")
    print(f"reentry_metrics_wrapper={metrics_ok}")
    print(f"attempt0_schema_wrapper={schema_ok}")

    if root.name != "FSPT":
        raise SystemExit(f"repo_root().name expected 'FSPT', got {root.name!r}")
    if not readme.exists():
        raise SystemExit(f"README.md not found at {readme}")
    if not coords_ok:
        raise SystemExit("fspt.coords wrapper failed")
    if not metrics_ok:
        raise SystemExit("fspt.reentry_metrics wrapper failed")
    if not schema_ok:
        raise SystemExit("fspt.io.attempt0_schema wrapper failed")

    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
