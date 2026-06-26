from pathlib import Path


def test_fspt_package_imports():
    import fspt

    assert fspt.__version__ == "0.0.0"


def test_fspt_smoke_entrypoint():
    from fspt.smoke import main

    assert main() == 0


def test_mainline_docs_exist():
    root = Path(__file__).parent.parent
    assert (root / "CURRENT_MAINLINE.md").exists()
    assert (root / "docs/current_redetection_route_closure_2026-06-26.md").exists()


def test_wrapper_imports():
    from fspt.core import coords as fspt_coords
    from fspt.core import paths as fspt_paths
    from fspt.metrics import reentry as fspt_reentry
    from fspt.io import attempt0_schema as fspt_schema

    assert hasattr(fspt_coords, "find_reentry_events")
    assert hasattr(fspt_reentry, "compute_reappearance_segment_aj")
    assert hasattr(fspt_schema, "load_attempt0_cache")
    assert fspt_paths.repo_root().name == "FSPT"


def test_top_level_imports():
    from fspt.paths import caches_dir, data_root, output_root, outputs_dir, repo_root, resolve_repo_path
    from fspt.coords import find_reentry_events, yx_norm_to_xy_pixel
    from fspt.reentry_metrics import aggregate_reappearance_ajrd, compute_reappearance_segment_aj

    assert repo_root().name == "FSPT"
    assert resolve_repo_path("README.md").exists()
    assert callable(yx_norm_to_xy_pixel)
    assert callable(find_reentry_events)
    assert callable(compute_reappearance_segment_aj)
    assert callable(aggregate_reappearance_ajrd)
    assert callable(outputs_dir)
    assert callable(caches_dir)
    assert callable(data_root)
    assert callable(output_root)
