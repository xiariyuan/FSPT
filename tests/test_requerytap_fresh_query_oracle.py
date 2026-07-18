from scripts.audit_tapnextpp_fresh_query_oracle import evaluate_gate

def row(scene,gain,frames=.8):
    return {'scene':scene,'status':'complete','delta':{'fresh_GT_gain_vs_native':gain,'fresh_GT_improved_frame_fraction':frames}}

def test_gate_passes_strong_17_scene_signal():
    rows=[row(str(i),3.0) for i in range(17)]
    gate=evaluate_gate(rows,17)
    assert gate['pass']
    assert not gate['training_allowed']

def test_gate_fails_on_incomplete_or_harmful_scene():
    assert not evaluate_gate([row(str(i),3.0) for i in range(16)],17)['pass']
    rows=[row(str(i),3.0) for i in range(16)]+[row('bad',-1.1)]
    gate=evaluate_gate(rows,17)
    assert not gate['pass']
    assert not gate['checks']['fresh_GT_no_scene_regression_worse_than_1px']
