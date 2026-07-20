import json
import pickle
from pathlib import Path

from scripts.preprocess_kubric_tfrecord_excluding_manifest_v1 import (
    load_excluded_identities,
    plan_selected_identities,
    split_record_lengths,
)


def _manifest(tmp_path: Path, identities):
    samples = [
        {"source_tfrecord": name, "source_record_index": index}
        for name, index in identities
    ]
    shard = tmp_path / "train_00000.pkl"
    with shard.open("wb") as handle:
        pickle.dump(samples, handle, protocol=4)
    import hashlib
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest = tmp_path / "train.index.json"
    manifest.write_text(
        json.dumps(
            {
                "shards": [
                    {
                        "path": shard.name,
                        "num_samples": len(samples),
                        "sha256": digest,
                    }
                ]
            }
        )
    )
    return manifest


def test_exclusion_manifest_and_planned_membership(tmp_path):
    manifest = _manifest(tmp_path, [("a", 0), ("a", 1), ("b", 0)])
    excluded, authority = load_excluded_identities([manifest])
    assert excluded == {("a", 0), ("a", 1), ("b", 0)}
    assert authority[0]["identities_added"] == 3
    files = [tmp_path / "a", tmp_path / "b", tmp_path / "c"]
    planned = plan_selected_identities(files, [3, 2, 2], excluded, 4)
    assert planned == [("a", 2), ("b", 1), ("c", 0), ("c", 1)]


def test_dataset_info_split_lengths(tmp_path):
    path = tmp_path / "dataset_info.json"
    path.write_text(
        json.dumps(
            {
                "splits": [
                    {"name": "train", "shardLengths": ["2", "3"]},
                    {"name": "test", "shardLengths": ["1"]},
                ]
            }
        )
    )
    assert split_record_lengths(path, "train") == [2, 3]


def test_preregistered_config_pins_exact_selection_and_implementations():
    import hashlib
    import yaml

    config = yaml.safe_load(
        Path("configs/routeD_temporal_identity_top1_data_gate3c2_v0.yaml").read_text()
    )
    assert config["expected_selection"]["selected_identity_digest"] == (
        "ae7f8c4231dc81b52327020c43914def5ec6a4666bc374bd2a5539d1be5bcf37"
    )
    for authority in config["implementation"].values():
        actual = hashlib.sha256(Path(authority["path"]).read_bytes()).hexdigest()
        assert actual == authority["sha256"]
