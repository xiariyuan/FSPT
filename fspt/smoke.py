"""Smoke entrypoint for the fspt package.

Validates that the package is importable, path helpers resolve correctly,
and key wrapper modules are accessible.
"""
from __future__ import annotations

from . import __version__


def main() -> int:
    from .paths import caches_dir, data_root, output_root, outputs_dir, repo_root, resolve_repo_path

    root = repo_root()
    readme = resolve_repo_path("README.md")
    mainline = resolve_repo_path("CURRENT_MAINLINE.md")
    coords_ok = False
    metrics_ok = False
    schema_ok = False
    find_events_ok = False
    compute_aj_ok = False

    try:
        from .coords import yx_norm_to_xy_pixel as _fn

        coords_ok = callable(_fn)
    except Exception:
        pass
    try:
        from .coords import find_reentry_events as _fn

        find_events_ok = callable(_fn)
    except Exception:
        pass
    try:
        from .reentry_metrics import aggregate_reappearance_ajrd as _fn

        metrics_ok = callable(_fn)
    except Exception:
        pass
    try:
        from .reentry_metrics import compute_reappearance_segment_aj as _fn

        compute_aj_ok = callable(_fn)
    except Exception:
        pass
    try:
        from .io.attempt0_schema import load_attempt0_cache as _fn

        schema_ok = callable(_fn)
    except Exception:
        pass

    print(f"fspt {__version__}")
    print(f"repo_root={root}")
    print(f"repo_root.name={root.name}")
    print(f"readme_exists={readme.exists()}")
    print(f"mainline_exists={mainline.exists()}")
    print(f"data_root={data_root()}")
    print(f"output_root={output_root()}")
    print(f"outputs_dir={outputs_dir()}")
    print(f"caches_dir={caches_dir()}")
    print(f"coords_wrapper={coords_ok}")
    print(f"find_reentry_events={find_events_ok}")
    print(f"reentry_metrics_wrapper={metrics_ok}")
    print(f"compute_reappearance_segment_aj={compute_aj_ok}")
    print(f"attempt0_schema_wrapper={schema_ok}")

    errors: list[str] = []
    if root.name != "FSPT":
        errors.append(f"repo_root().name expected 'FSPT', got {root.name!r}")
    if not readme.exists():
        errors.append(f"README.md not found at {readme}")
    if not mainline.exists():
        errors.append(f"CURRENT_MAINLINE.md not found at {mainline}")
    if not coords_ok:
        errors.append("fspt.coords wrapper failed")
    if not find_events_ok:
        errors.append("fspt.coords.find_reentry_events wrapper failed")
    if not metrics_ok:
        errors.append("fspt.reentry_metrics wrapper failed")
    if not compute_aj_ok:
        errors.append("fspt.reentry_metrics.compute_reappearance_segment_aj wrapper failed")
    if not schema_ok:
        errors.append("fspt.io.attempt0_schema wrapper failed")

    if errors:
        for err in errors:
            print(f"FAIL: {err}")
        raise SystemExit(1)

    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
