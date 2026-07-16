#!/usr/bin/env python3
"""Verify local TAP-Vid release files byte-for-byte against the official zip."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_release_package(
    zip_path: Path,
    root: Path,
    source_url: str,
    content_length: int,
    etag: str,
    last_modified: str,
) -> dict:
    expected_files = (
        "tapvid_kinetics.csv",
        "README.md",
        "train.txt",
        "val.txt",
        "test.txt",
    )
    if zip_path.stat().st_size != content_length:
        raise ValueError(
            f"Official zip size {zip_path.stat().st_size} != {content_length}"
        )

    member_audit = {}
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        by_basename = {}
        for member in members:
            basename = Path(member).name
            if basename:
                by_basename.setdefault(basename, []).append(member)
        for filename in expected_files:
            candidates = by_basename.get(filename, [])
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected exactly one {filename} in release zip; found {candidates}"
                )
            member = candidates[0]
            archive_bytes = archive.read(member)
            local_path = root / filename
            local_bytes = local_path.read_bytes()
            archive_hash = sha256_bytes(archive_bytes)
            local_hash = sha256_bytes(local_bytes)
            member_audit[filename] = {
                "archive_member": member,
                "archive_bytes": len(archive_bytes),
                "local_path": str(local_path),
                "local_bytes": len(local_bytes),
                "archive_sha256": archive_hash,
                "local_sha256": local_hash,
                "exact_match": archive_bytes == local_bytes,
            }

    passed = all(entry["exact_match"] for entry in member_audit.values())
    return {
        "kind": "tapvid_official_release_package_audit",
        "pass": passed,
        "source_url": source_url,
        "http_metadata": {
            "content_length": content_length,
            "etag": etag,
            "last_modified": last_modified,
        },
        "official_zip": str(zip_path),
        "official_zip_bytes": zip_path.stat().st_size,
        "official_zip_sha256": sha256(zip_path),
        "dataset_root": str(root),
        "members": member_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--content-length", type=int, required=True)
    parser.add_argument("--etag", required=True)
    parser.add_argument("--last-modified", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    zip_path = Path(args.zip).resolve()
    root = Path(args.dataset_root).resolve()
    result = audit_release_package(
        zip_path=zip_path,
        root=root,
        source_url=args.source_url,
        content_length=args.content_length,
        etag=args.etag,
        last_modified=args.last_modified,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
