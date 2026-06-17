from pathlib import Path


def _read_text(rel_path: str) -> str:
    root = Path(__file__).parent.parent
    return (root / rel_path).read_text(encoding="utf-8")


def test_config_has_two_view_consistency_knobs():
    text = _read_text("configs/fspt_base.yaml")
    assert "two_view_consistency:" in text
    assert "visibility_weight:" in text
    assert "min_in_bounds_fraction:" in text
    assert "max_resample:" in text
    assert "visibility_schedule:" in text
    assert "auto_disable_dataset_temporal:" in text
    assert "temporal:" in text
    assert "reverse_prob:" in text
    assert "roll_prob:" in text
    assert "max_points:" in text
    assert "mask_weighting:" in text
    assert "oom_safe:" in text
    assert "teacher_disable_autocast:" in text
    assert "unit:" in text
    assert "start_step:" in text
    assert "end_step:" in text


def test_config_has_refiner_prior_features_knobs():
    text = _read_text("configs/fspt_cotracker_refine.yaml")
    assert "prior_features:" in text
    assert "detach_delta:" in text
