import json
from pathlib import Path

import yaml

from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import canonical_json_sha256
from scripts.diagnose_routeD_temporal_identity_full_population_gate3c1f2_v0 import (
    VIDEO_SCHEMA,
    _aggregate,
    _validate_config,
    jaccard_components,
)

CONFIG = Path(
    "configs/routeD_temporal_identity_full_population_gate3c1f2_failure_diagnostic_v0.yaml"
)


def test_jaccard_components_match_official_semantics():
    assert jaccard_components(
        gt_visible=True, predicted_visible=True, error_px=2.0, threshold_px=4.0
    ) == {"gt_positive": 1, "point_correct": 1, "true_positive": 1, "false_positive": 0}
    assert jaccard_components(
        gt_visible=True, predicted_visible=False, error_px=2.0, threshold_px=4.0
    ) == {"gt_positive": 1, "point_correct": 1, "true_positive": 0, "false_positive": 0}
    assert jaccard_components(
        gt_visible=True, predicted_visible=True, error_px=5.0, threshold_px=4.0
    ) == {"gt_positive": 1, "point_correct": 0, "true_positive": 0, "false_positive": 1}
    assert jaccard_components(
        gt_visible=False, predicted_visible=True, error_px=0.0, threshold_px=4.0
    ) == {"gt_positive": 0, "point_correct": 0, "true_positive": 0, "false_positive": 1}


def test_diagnostic_config_is_exact_and_selects_only_sealed_action_videos():
    config, _, reference = _validate_config(CONFIG.resolve())
    action_sources = sorted(
        source_index
        for source_index, row in reference.items()
        if row["scientific"]["top1_action_rows"] > 0
    )
    assert action_sources == config["action_source_indices"]
    assert len(action_sources) == 44
    assert sum(reference[index]["scientific"]["top1_action_rows"] for index in action_sources) == 89
    assert all(value is False for value in config["locked_data"].values())
    assert not Path(config["output"]["work_root"]).exists()
    assert not Path(config["output"]["result"]).exists()


def _synthetic_record(source_index: int):
    views = {
        "native": {"AJ": 0.2, "<avg": 0.3, "OA": 0.8},
        "actual_modified": {"AJ": 0.21, "<avg": 0.32, "OA": 0.81},
        "modified_coordinates_native_visibility": {"AJ": 0.22, "<avg": 0.32, "OA": 0.8},
        "native_coordinates_modified_visibility": {"AJ": 0.19, "<avg": 0.3, "OA": 0.81},
        "native_coordinates_gt_visibility_oracle": {"AJ": 0.25, "<avg": 0.3, "OA": 1.0},
        "modified_coordinates_gt_visibility_oracle": {"AJ": 0.28, "<avg": 0.32, "OA": 1.0},
    }
    frame = {
        "source_index": source_index,
        "video_name": str(source_index),
        "video_AJ_sign": "positive",
        "action_ordinal": 0,
        "point_index": 0,
        "category": "failure",
        "frame": 15,
        "gt_visible": True,
        "native_visibility_probability": 0.1,
        "native_confidence_probability": 0.1,
        "native_joint_probability": 0.01,
        "native_visible": False,
        "modified_visibility_probability": 0.9,
        "modified_confidence_probability": 0.9,
        "modified_joint_probability": 0.81,
        "modified_visible": True,
        "native_error_px": 10.0,
        "modified_error_px": 2.0,
        "error_reduction_px": 8.0,
        "recovered_visible_false_negative": True,
        "new_visible_false_negative": False,
        "removed_occluded_false_positive": False,
        "new_occluded_false_positive": False,
        "jaccard_components": {},
    }
    for view, error, visible in (
        ("native", 10.0, False),
        ("actual_modified", 2.0, True),
        ("modified_coordinates_native_visibility", 2.0, False),
        ("native_coordinates_modified_visibility", 10.0, True),
        ("native_coordinates_gt_visibility_oracle", 10.0, True),
        ("modified_coordinates_gt_visibility_oracle", 2.0, True),
    ):
        frame["jaccard_components"][view] = {
            str(threshold): jaccard_components(
                gt_visible=True,
                predicted_visible=visible,
                error_px=error,
                threshold_px=threshold,
            )
            for threshold in (1, 2, 4, 8, 16)
        }
    scientific = {
        "source_index": source_index,
        "video_name": str(source_index),
        "sealed_digest_checks": {"all": True},
        "action_decisions_exact": True,
        "action_rows": 1,
        "view_metrics": views,
        "frame_records_digest": canonical_json_sha256([frame]),
        "frame_records": [frame],
        "formal_video_scientific_sha256": str(source_index),
    }
    scientific["scientific_payload_sha256"] = canonical_json_sha256(scientific)
    result = {
        "schema_version": VIDEO_SCHEMA,
        "date": "2026-07-20",
        "status": "completed",
        "source_index": source_index,
        "sample_metadata": {},
        "scientific": scientific,
    }
    result["result_payload_sha256"] = canonical_json_sha256(result)
    return result


def test_diagnostic_aggregate_separates_coordinate_and_visibility_effects():
    config = yaml.safe_load(CONFIG.read_text())
    records = [_synthetic_record(index) for index in range(4)]
    result = _aggregate(config=config, records=records)
    scientific = result["scientific"]
    assert scientific["sealed_pipeline_exact"] is True
    assert scientific["visibility_transitions"]["recovered_visible_false_negative"] == 4
    assert scientific["paired_comparisons"][
        "modified_coordinates_native_visibility_minus_native"
    ]["AJ"]["mean"] > 0.0
    assert scientific["paired_comparisons"][
        "native_coordinates_modified_visibility_minus_native"
    ]["AJ"]["mean"] < 0.0
    assert scientific["pooled_affected_frame_jaccard"]["failure"][
        "actual_modified"
    ]["average_jaccard"] > scientific["pooled_affected_frame_jaccard"]["failure"][
        "native"
    ]["average_jaccard"]
