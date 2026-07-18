import numpy as np
from scripts.audit_requerytap_position_deconfounding import candidate_gate, incremental_gate

def row(scene,base,frame0,max8):
    return {'scene':scene,'status':'complete','encoded_frame0':{'median_error_px':base},'encoded_frame0_error_rows':[30,40,50,60],'metrics':{'conv_frame0':{'median_error_px':frame0},'conv_max8':{'median_error_px':max8}},'error_rows':{'conv_frame0':[5,6,7,8],'conv_max8':[4,5,6,7]}}

def manifest():
    return {'seed':17018,'scenes':['a','b','c','d','e'],'candidate_gate':{'positive_median_error_reduction_scenes_min':5,'median_scene_error_reduction_px_min':10,'scene_bootstrap_reduction_ci_lower_gt':0,'aggregate_hit16_absolute_gain_min':.1,'worst_scene_error_reduction_min':-2},'multiview_incremental_gate':{'positive_median_error_reduction_scenes_min':4,'median_scene_error_reduction_px_min':5,'worst_scene_error_reduction_min':-2}}

def test_candidate_gate_passes_strong_position_free_variant():
    rows=[row(s,50,25,15) for s in manifest()['scenes']]
    assert candidate_gate(rows,'conv_max8',manifest())['pass']

def test_incremental_gate_requires_multiview_gain():
    rows=[row(s,50,25,15) for s in manifest()['scenes']]
    assert incremental_gate(rows,manifest())['pass']
    rows[0]['metrics']['conv_max8']['median_error_px']=40
    assert not incremental_gate(rows,manifest())['pass']
