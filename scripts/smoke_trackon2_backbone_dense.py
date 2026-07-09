#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent.parent
TRACKON=ROOT/'baselines'/'track_on'
sys.path.insert(0,str(TRACKON))

class Args:
    input_size=[384,512]
    M=24
    D=256
    K=16
    decoder_layer_num=3
    predicton_head_layer_num=3
    rerank_layer_num=3
    vit_backbone='dinov2_b'
    vit_upsample_factor=1.0
    grad_checkpoint=False
    M_i=72
    delta_v=0.8
    memory_update_policy='unconditional'

OUT=ROOT/'outputs/paper_discovery_2026-06-27/b2wa_trackon_dense_pilot'
OUT.mkdir(parents=True,exist_ok=True)
VIDEO=ROOT/'outputs/paper_discovery_2026-06-27/b2wa_video_cache/rgb_dev10/rgb_stacking_000000.npz'
CKPT=ROOT/'baselines/track_on/checkpoints_trackon2_dinov2.pt'


def load_backbone():
    from model.trackon import Track_On2
    model=Track_On2(Args())
    sd=torch.load(CKPT,map_location='cpu')
    miss,unexp=model.load_state_dict(sd,strict=False)
    return model.backbone, {'missing':len(miss),'unexpected':len(unexp),'missing_sample':list(miss)[:10],'unexpected_sample':list(unexp)[:10]}

def prep_frame():
    frame=np.load(VIDEO)['video'][0]
    x=torch.from_numpy(frame).permute(2,0,1).float()[None,None]
    if float(x.max())>1.5: x=x/255.0
    x=F.interpolate(x.reshape(1,3,256,256),size=(384,512),mode='bilinear',align_corners=False).reshape(1,1,3,384,512)
    return x

def main():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    res={'device':device}
    try:
        bb,info=load_backbone()
        bb.eval().to(device)
        x=prep_frame().to(device)
        with torch.no_grad():
            f4,f8,f16,f32=bb(x)
        res.update({'ok':True,'load_info':info,'shapes':{'f4':list(f4.shape),'f8':list(f8.shape),'f16':list(f16.shape),'f32':list(f32.shape)}})
    except Exception as e:
        res.update({'ok':False,'error':repr(e)})
    (OUT/'smoke_trackon2_backbone_dense.json').write_text(json.dumps(res,indent=2))
    print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__': main()
