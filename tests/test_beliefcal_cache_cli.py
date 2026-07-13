import unittest

from projects.mmp_tracker.run_beliefcal_mvp1 import iterate_loader_window


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


if __name__ == "__main__":
    unittest.main()
