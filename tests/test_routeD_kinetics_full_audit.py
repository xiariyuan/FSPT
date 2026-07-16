import unittest

from scripts.audit_routeD_kinetics_full_eval import require


class TestRouteDKineticsFullAudit(unittest.TestCase):
    def test_require_accepts_true(self):
        require(True, "unused")

    def test_require_rejects_false(self):
        with self.assertRaisesRegex(ValueError, "broken"):
            require(False, "broken")


if __name__ == "__main__":
    unittest.main()
