#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

OUT = Path('outputs/paper_discovery_2026-06-27/b2wa_dino_pilot')
OUT.mkdir(parents=True, exist_ok=True)
VIDEO = Path('outputs/paper_discovery_2026-06-27/b2wa_video_cache/rgb_dev10/rgb_stacking_000000.npz')


def prep_image(frame: np.ndarray, device: str):
    x = torch.from_numpy(frame).permute(2, 0, 1).float()[None]
    if float(x.max()) > 1.5:
        x = x / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    x = (x - mean) / std
    # 256 is not a multiple of 14, use 252 to keep DINO patch grid clean.
    x = F.interpolate(x, size=(252,252), mode='bilinear', align_corners=False)
    return x.to(device)


def try_timm_dino(device: str):
    import timm
    name = 'vit_small_patch14_dinov2'
    model = timm.create_model(name, pretrained=True, dynamic_img_size=True)
    model.eval().to(device)
    frame = np.load(VIDEO)['video'][0]
    x = prep_image(frame, device)
    with torch.no_grad():
        # timm ViT: forward_features returns tokens or dict depending model.
        y = model.forward_features(x)
    if isinstance(y, dict):
        toks = y.get('x_norm_patchtokens') or y.get('x') or next(iter(y.values()))
    else:
        toks = y
    shape = list(toks.shape) if hasattr(toks, 'shape') else str(type(toks))
    return {'backend':'timm', 'model':name, 'ok':True, 'feature_shape':shape}


def try_open_clip(device: str):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai', device=device)
    model.eval()
    frame = np.load(VIDEO)['video'][0]
    from PIL import Image
    img = Image.fromarray(frame.astype('uint8'))
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(x)
    return {'backend':'open_clip', 'model':'ViT-B-16/openai', 'ok':True, 'feature_shape':list(feat.shape)}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    result = {'device': device, 'attempts': []}
    for fn in [try_timm_dino, try_open_clip]:
        try:
            r = fn(device)
            result['attempts'].append(r)
            result['selected'] = r
            break
        except Exception as e:
            result['attempts'].append({'backend':fn.__name__, 'ok':False, 'error':repr(e)})
    (OUT / 'smoke_dino_loader.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)

if __name__ == '__main__':
    main()
