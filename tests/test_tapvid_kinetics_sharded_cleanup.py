import pickle
import tempfile
import unittest
from pathlib import Path

import datasets.tapvid_kinetics_sharded as kinetics_module
from datasets.tapvid_kinetics_sharded import TAPVidKineticsShardedIterableDataset


class TestTAPVidKineticsShardedCleanup(unittest.TestCase):
    def test_partially_consumed_generator_survives_global_teardown(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            shard_path = Path(tmp_dir) / "tiny.pkl"
            with shard_path.open("wb") as handle:
                pickle.dump([{"value": 1}], handle)

            dataset = TAPVidKineticsShardedIterableDataset.__new__(
                TAPVidKineticsShardedIterableDataset
            )
            dataset.max_retries = 0
            dataset._iter_assigned_shards = lambda: iter([(0, shard_path)])
            dataset._prepare_sample = (
                lambda sample, sample_idx, shard_idx: {
                    "sample": sample,
                    "sample_idx": sample_idx,
                    "shard_idx": shard_idx,
                }
            )

            iterator = iter(dataset)
            self.assertEqual(next(iterator)["sample"]["value"], 1)

            original_skip_sample = kinetics_module._SkipSample
            try:
                # Simulate module globals being cleared during interpreter
                # shutdown while the generator is suspended at `yield`.
                kinetics_module._SkipSample = None
                iterator.close()
            finally:
                kinetics_module._SkipSample = original_skip_sample


if __name__ == "__main__":
    unittest.main()
