import torch
from scripts.audit_requerytap_multiview_identity import aggregate_scores,evaluate_gate

def test_aggregation_respects_mask_and_shapes():
    memory=torch.tensor([[[1.,0.],[0.,1.]],[[1.,0.],[1.,0.]]]);valid=torch.tensor([[True,False],[True,True]]);target=torch.tensor([[1.,0.],[0.,1.]])
    out=aggregate_scores(memory,valid,target)
    assert all(value.shape==(2,2) for value in out.values())
    assert torch.allclose(out['max8'][0],out['frame0'][0])

def test_gate_passes_strong_primary():
    manifest={'primary_variant':'max8','scenes':['a','b','c','d','e'],'decision_if_pass':'PASS','decision_if_fail':'FAIL','gate_D':{'positive_median_error_reduction_scenes_min':5,'median_scene_error_reduction_px_min':10,'scene_bootstrap_reduction_ci_lower_gt':0,'aggregate_hit16_absolute_gain_min':.1,'worst_scene_error_reduction_min':-2}}
    rows=[]
    for scene in manifest['scenes']:rows.append({'scene':scene,'status':'complete','metrics':{'frame0':{'median_error_px':50},'max8':{'median_error_px':20}},'error_rows':{'frame0':[30,40,50,60],'max8':[5,6,7,8]}})
    assert evaluate_gate(rows,manifest)['pass']
