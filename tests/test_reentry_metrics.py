"""Tests for reentry_metrics aggregation functions, especially summary_key behavior."""
import pytest


def test_aggregate_reappearance_ajrd_summary_key_selects_requested_space():
    """Test that summary_key="ajrd_summary_256" selects 256-space AJ_RD."""
    from utils.reentry_metrics import aggregate_reappearance_ajrd

    row = {
        "ajrd_summary": {
            "aj_rd": 0.1,
            "ajrd_by_dmin": {"1": {"mean": 0.1}, "4": {"mean": 0.12}},
        },
        "ajrd_summary_256": {
            "aj_rd": 0.9,
            "ajrd_by_dmin": {"1": {"mean": 0.9}, "4": {"mean": 0.92}},
        },
    }

    original = aggregate_reappearance_ajrd([row], d_mins=[1, 4], summary_key="ajrd_summary")
    space256 = aggregate_reappearance_ajrd([row], d_mins=[1, 4], summary_key="ajrd_summary_256")

    assert original["aj_rd"] == 0.1
    assert space256["aj_rd"] == 0.9
    assert original["by_dmin"]["1"]["mean"] == 0.1
    assert space256["by_dmin"]["1"]["mean"] == 0.9
    assert original["by_dmin"]["4"]["mean"] == 0.12
    assert space256["by_dmin"]["4"]["mean"] == 0.92


def test_aggregate_reappearance_ajrd_summary_key_overrides_top_level_fields():
    """Test that summary_key takes precedence over top-level aj_rd/ajrd_by_dmin."""
    from utils.reentry_metrics import aggregate_reappearance_ajrd

    row = {
        "aj_rd": 0.2,
        "ajrd_by_dmin": {"1": {"mean": 0.2}},
        "ajrd_summary_256": {
            "aj_rd": 0.9,
            "ajrd_by_dmin": {"1": {"mean": 0.9}},
        },
    }

    result = aggregate_reappearance_ajrd([row], d_mins=[1], summary_key="ajrd_summary_256")

    assert result["aj_rd"] == 0.9
    assert result["by_dmin"]["1"]["mean"] == 0.9


def test_aggregate_reappearance_ajrd_fallback_to_top_level():
    """Test that top-level aj_rd is used when summary_key dict is missing."""
    from utils.reentry_metrics import aggregate_reappearance_ajrd

    row = {
        "aj_rd": 0.3,
        "ajrd_by_dmin": {"1": {"mean": 0.3}},
    }

    result = aggregate_reappearance_ajrd([row], d_mins=[1], summary_key="ajrd_summary_256")

    assert result["aj_rd"] == 0.3
    assert result["by_dmin"]["1"]["mean"] == 0.3


def test_aggregate_reappearance_ajrd_default_summary_key():
    """Test that default summary_key='ajrd_summary' works."""
    from utils.reentry_metrics import aggregate_reappearance_ajrd

    row = {
        "ajrd_summary": {
            "aj_rd": 0.5,
            "ajrd_by_dmin": {"1": {"mean": 0.5}},
        },
    }

    result = aggregate_reappearance_ajrd([row], d_mins=[1])

    assert result["aj_rd"] == 0.5
    assert result["by_dmin"]["1"]["mean"] == 0.5


def test_aggregate_reappearance_ajrd_handles_none_values():
    """Test that None values are handled gracefully."""
    from utils.reentry_metrics import aggregate_reappearance_ajrd

    rows = [
        {
            "ajrd_summary": {"aj_rd": 0.5, "ajrd_by_dmin": {"1": {"mean": 0.5}}},
        },
        {
            "ajrd_summary": {"aj_rd": None, "ajrd_by_dmin": {"1": {"mean": None}}},
        },
        {
            "ajrd_summary": {"aj_rd": 0.7, "ajrd_by_dmin": {"1": {"mean": 0.7}}},
        },
    ]

    result = aggregate_reappearance_ajrd(rows, d_mins=[1])

    assert result["aj_rd"] == 0.6  # (0.5 + 0.7) / 2
    assert result["n_valid_samples"] == 2
