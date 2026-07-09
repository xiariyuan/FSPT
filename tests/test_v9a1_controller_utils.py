import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path('scripts/v9a1_controller_calibration_aware_prototype.py')
spec = importlib.util.spec_from_file_location('v9a1', SCRIPT)
v9a1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v9a1)


def test_video_from_meta_prefers_video_name():
    assert v9a1.video_from_meta({'video_name': 'parkour', 'video_idx': 3}) == 'parkour'


def test_video_from_meta_falls_back_to_video_idx():
    assert v9a1.video_from_meta({'video_idx': 7}) == 'video_7'


def test_risk_coverage_curve_keeps_high_score_low_loss_first():
    scores = np.array([0.9, 0.1, 0.8, 0.2], dtype=float)
    losses = np.array([0.0, 10.0, 1.0, 9.0], dtype=float)
    out = v9a1.risk_coverage_curve(scores, losses, coverages=[0.5, 1.0])
    assert out['coverage_0.50_mean_loss'] == 0.5
    assert out['coverage_1.00_mean_loss'] == 5.0


def test_choose_threshold_by_metric_returns_valid_threshold():
    y = np.array([0, 0, 1, 1], dtype=int)
    s = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    thr = v9a1.choose_threshold_by_metric(y, s, metric='f1')
    assert 0.2 < thr <= 0.9


def test_build_video_groups_from_meta_list():
    metas = [{'video_name': 'a'}, {'video_name': 'a'}, {'video_name': 'b'}]
    groups = v9a1.build_video_groups(metas)
    assert groups.tolist() == ['a', 'a', 'b']


def test_label_array_uses_preferred_key():
    data = {'a': np.array([0, 1]), 'b': np.array([1, 1])}
    y, name = v9a1.label_array(data, ['b', 'a'])
    assert name == 'b'
    assert y.tolist() == [1, 1]


def test_select_feature_matrix_uses_X():
    data = {'X': np.ones((3, 2), dtype=np.float32), 'feature_names': np.array(['x', 'y'], dtype=object)}
    X, names = v9a1.select_feature_matrix(data)
    assert X.shape == (3, 2)
    assert names == ['x', 'y']


def test_video_from_meta_supports_video_id():
    assert v9a1.video_from_meta({'video_id': 'bike-packing'}) == 'bike-packing'
