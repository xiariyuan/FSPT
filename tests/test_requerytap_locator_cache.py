import numpy as np
from scripts.build_requerytap_locator_cache import select_points

def test_point_selection_is_deterministic_and_limited():
    valid=np.ones(100,dtype=bool)
    a=select_points('scene',valid,16,17);b=select_points('scene',valid,16,17)
    assert np.array_equal(a,b);assert len(a)==16;assert len(set(a.tolist()))==16

def test_point_selection_respects_valid_mask():
    valid=np.zeros(20,dtype=bool);valid[[2,5,11]]=True
    assert set(select_points('x',valid,10,17).tolist())=={2,5,11}
