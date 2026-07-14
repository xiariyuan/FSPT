import tempfile
import unittest
from pathlib import Path

from projects.mmp_tracker.mmp_tracker.beliefcal_runner import (
    save_beliefcal_cache,
    synthetic_beliefcal_cache,
)
from projects.mmp_tracker.run_beliefcal_mvp1 import (
    audit_cache_bundle,
    dataset_provenance,
    iterate_loader_window,
)


class RestartSensitiveIterable:
    def __init__(self, values):
        self.values = list(values)
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        return iter(self.values)


class TestBeliefCalCacheCLI(unittest.TestCase):
    def test_start_batch_uses_single_iterator(self):
        loader = RestartSensitiveIterable(range(6))
        observed = list(iterate_loader_window(loader, start_batch=2, limit=3))
        self.assertEqual(observed, [(2, 2), (3, 3), (4, 4)])
        self.assertEqual(loader.iter_calls, 1)

    def test_start_batch_beyond_end_is_empty(self):
        loader = RestartSensitiveIterable(range(2))
        observed = list(iterate_loader_window(loader, start_batch=5, limit=1))
        self.assertEqual(observed, [])
        self.assertEqual(loader.iter_calls, 1)

    def test_davis_default_annotation_is_recorded_for_provenance(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            annotation = Path(tmp_dir) / "tapvid_davis.pkl"
            annotation.write_bytes(b"synthetic-davis-annotation")
            config = {
                "data": {
                    "test": {
                        "dataset": "tapvid_davis",
                        "root": tmp_dir,
                        "split": "test",
                    }
                }
            }
            provenance = dataset_provenance(config, "test")
            self.assertEqual(provenance["dataset_family"], "davis")
            self.assertEqual(
                provenance["annotation_manifest"], str(annotation.resolve())
            )
            self.assertEqual(provenance["source_files"], [str(annotation.resolve())])
            self.assertIsNotNone(provenance["annotation_manifest_sha256"])

    def test_shared_dataset_family_is_warning_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = {}
            for index, role in enumerate(("train", "calibration", "validation", "test")):
                path = Path(tmp_dir) / f"{role}.pt"
                cache = synthetic_beliefcal_cache(32, seed=100 + index, sample_offset=1000 * index)
                manifest = {
                    "role": role,
                    "checkpoint_sha256": "same-checkpoint",
                    "model_config_sha256": "same-model",
                    "checkpoint_load_mode": "strict",
                    "feature_order_sha256": "same-features",
                    "git_dirty": False,
                    "rows": 32,
                    "videos": int(cache["sample_id"].unique().numel()),
                    "dataset": {
                        "dataset_family": "shared-family",
                        "dataset_identity": f"identity-{role}",
                        "source_files": [str(Path(tmp_dir) / f"{role}.pkl")],
                    },
                }
                save_beliefcal_cache(cache, path, manifest=manifest)
                paths[role] = path
            report = audit_cache_bundle(paths)
            self.assertTrue(report["passed"])
            self.assertTrue(
                any("dataset-family collision" in warning for warning in report["warnings"])
            )
            with self.assertRaises(RuntimeError):
                audit_cache_bundle(paths, require_distinct_dataset_families=True)


if __name__ == "__main__":
    unittest.main()
