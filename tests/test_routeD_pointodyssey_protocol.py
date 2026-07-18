from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from projects.mmp_tracker.mmp_tracker.routeD_pointodyssey_protocol import (
    balanced_event_selection,
    extract_record_breaking_events,
    event_counts,
)


def test_record_breaking_event_extraction_and_buckets():
    visible = np.array(
        [
            [1, 1], [0, 1], [1, 0], [1, 0], [0, 1],
            [0, 1], [1, 1], [0, 1], [1, 1],
        ],
        dtype=bool,
    )
    tracks = np.zeros(visible.shape + (2,), dtype=np.float32)
    events = extract_record_breaking_events("scene", visible, tracks)
    rows = [(e.point_index, e.reappearance_frame, e.invisibility_duration) for e in events]
    assert rows == [(0, 2, 1), (1, 4, 2), (0, 6, 2)]
    assert event_counts(events)["by_dmin"] == {"1": 3, "4": 0, "16": 0, "64": 0, "256": 0}


def test_balanced_selection_is_deterministic():
    visible = np.ones((80, 6), dtype=bool)
    for point, start, length in [(0, 2, 1), (1, 3, 2), (2, 4, 4), (3, 5, 8), (4, 6, 16), (5, 7, 32)]:
        visible[start : start + length, point] = False
    tracks = np.zeros(visible.shape + (2,), dtype=np.float32)
    events = extract_record_breaking_events("scene", visible, tracks)
    first = balanced_event_selection(events, seed=17, quota_per_bucket=1)
    second = balanced_event_selection(events, seed=17, quota_per_bucket=1)
    assert first == second
    assert len(first) == 3


def test_frozen_protocol_uses_portable_verified_official_evidence():
    repo_root = Path(__file__).resolve().parents[1]
    protocol_path = repo_root / "configs/routeD_safe_redetection_pointodyssey_protocol_v0.json"
    protocol = json.loads(protocol_path.read_text())
    prerequisites = protocol["official_prerequisites"]
    for reference_key, hash_key in (
        ("cotracker3_alignment_summary", "cotracker3_alignment_summary_sha256"),
        ("official_ajrd_parity", "official_ajrd_parity_sha256"),
    ):
        reference = Path(prerequisites[reference_key])
        assert not reference.is_absolute()
        artifact = (repo_root / reference).resolve()
        artifact.relative_to(repo_root)
        assert artifact.is_file()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == prerequisites[hash_key]
    payload = dict(protocol)
    expected = payload.pop("payload_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert actual == expected
