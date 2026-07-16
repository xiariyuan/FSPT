from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.audit_tapvid_release_package import audit_release_package


class TestTapVidReleasePackageAudit(unittest.TestCase):
    def test_exact_release_members_pass(self):
        filenames = (
            "tapvid_kinetics.csv",
            "README.md",
            "train.txt",
            "val.txt",
            "test.txt",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "local"
            root.mkdir()
            zip_path = Path(tmp) / "release.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                for index, filename in enumerate(filenames):
                    content = f"content-{index}\n".encode()
                    (root / filename).write_bytes(content)
                    archive.writestr(f"tapvid_kinetics/{filename}", content)
            result = audit_release_package(
                zip_path=zip_path,
                root=root,
                source_url="https://example.invalid/release.zip",
                content_length=zip_path.stat().st_size,
                etag="test-etag",
                last_modified="test-date",
            )
            self.assertTrue(result["pass"])
            self.assertTrue(
                all(entry["exact_match"] for entry in result["members"].values())
            )

    def test_modified_local_member_fails(self):
        filenames = (
            "tapvid_kinetics.csv",
            "README.md",
            "train.txt",
            "val.txt",
            "test.txt",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "local"
            root.mkdir()
            zip_path = Path(tmp) / "release.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                for filename in filenames:
                    content = b"same\n"
                    (root / filename).write_bytes(content)
                    archive.writestr(filename, content)
            (root / "val.txt").write_bytes(b"changed\n")
            result = audit_release_package(
                zip_path=zip_path,
                root=root,
                source_url="https://example.invalid/release.zip",
                content_length=zip_path.stat().st_size,
                etag="test-etag",
                last_modified="test-date",
            )
            self.assertFalse(result["pass"])
            self.assertFalse(result["members"]["val.txt"]["exact_match"])


if __name__ == "__main__":
    unittest.main()
