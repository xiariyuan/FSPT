from dataclasses import replace
from scripts.train_requerytap_locator import Pair,fixed_dev_subset,evaluate_gate

def test_fixed_dev_subset_is_deterministic():
    records=[{'_meta':{'scene':'a'}},{'_meta':{'scene':'b'}}]
    pairs=[Pair(i%2,i%3,i%5,40+i%3,100+i) for i in range(30)]
    assert fixed_dev_subset(pairs,records,10,17)==fixed_dev_subset(pairs,records,10,17)

def test_gate_passes_strong_metrics():
    records=[];scenes=['a','b','c','d','e']
    for scene in scenes:records.append({'_meta':{'scene':scene,'split':'dev','raw_baseline':{'median_error_px':50.0}}})
    metrics={'median_error_px':8.0,'hit16':.8,'per_scene':{scene:{'median_error_px':8.0} for scene in scenes}}
    manifest={'dev_scenes':scenes,'gate_C1':{'learned_median_error_px_max':12,'learned_median_hit16_min':.6,'positive_error_reduction_scenes_min':5,'scene_bootstrap_error_reduction_ci_lower_gt':0,'worst_scene_error_reduction_min':-2}}
    assert evaluate_gate(metrics,records,manifest)['pass']

def test_gate_fails_one_scene_regression():
    scenes=['a','b','c','d','e'];records=[{'_meta':{'scene':s,'split':'dev','raw_baseline':{'median_error_px':10.0}}} for s in scenes]
    per={s:{'median_error_px':5.0} for s in scenes};per['e']={'median_error_px':13.0}
    metrics={'median_error_px':6.0,'hit16':.8,'per_scene':per};manifest={'dev_scenes':scenes,'gate_C1':{'learned_median_error_px_max':12,'learned_median_hit16_min':.6,'positive_error_reduction_scenes_min':5,'scene_bootstrap_error_reduction_ci_lower_gt':0,'worst_scene_error_reduction_min':-2}}
    gate=evaluate_gate(metrics,records,manifest);assert not gate['pass'];assert not gate['checks']['positive_error_reduction_5_of_5']
