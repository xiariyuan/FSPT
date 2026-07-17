import unittest

from scripts.audit_routeD_kinetics_full_eval import (
    official_required_wording,
    require,
)


class TestRouteDKineticsFullAudit(unittest.TestCase):
    def test_require_accepts_true(self):
        require(True, "unused")

    def test_require_rejects_false(self):
        with self.assertRaisesRegex(ValueError, "broken"):
            require(False, "broken")

    def test_official_required_wording_uses_verified_csv_counts(self):
        wording = official_required_wording(1147, 1144, 3)
        self.assertIn("1,144 of the 1,147", wording)
        self.assertIn("3 CSV segments", wording)
        self.assertNotIn("1,189", wording)
        self.assertIn("not describe it", wording)


if __name__ == "__main__":
    unittest.main()
