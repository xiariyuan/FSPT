import unittest

from scripts.build_tapvid_kinetics_subset import selected_positions


class TestBuildTAPVidKineticsSubset(unittest.TestCase):
    def test_selection_is_deterministic_unique_and_bin_balanced(self):
        first = selected_positions(115, 5, 17)
        second = selected_positions(115, 5, 17)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(len(set(first)), 5)
        for bin_index, position in enumerate(first):
            start = (bin_index * 115) // 5
            end = ((bin_index + 1) * 115) // 5
            self.assertGreaterEqual(position, start)
            self.assertLess(position, end)

    def test_selection_rejects_more_samples_than_available(self):
        with self.assertRaises(ValueError):
            selected_positions(4, 5, 17)


if __name__ == "__main__":
    unittest.main()
