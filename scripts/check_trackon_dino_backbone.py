#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def inspect_dir(path: Path):
    files = sorted([p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file()]) if path.is_dir() else []
    wanted = ['config.json', 'model.safetensors', 'pytorch_model.bin']
    return {
        'path': str(path),
        'exists': path.is_dir(),
        'num_files': len(files),
        'has_config_json': 'config.json' in files,
        'has_model_safetensors': 'model.safetensors' in files,
        'has_pytorch_model_bin': 'pytorch_model.bin' in files,
        'top_files': files[:30],
    }


def main():
    root = Path(__file__).resolve().parent.parent
    candidates = []
    env = os.environ.get('TRACKON_DINOV2_MODEL_PATH', '').strip()
    if env:
        candidates.append(Path(env))
    candidates.extend([
        root / 'checkpoints' / 'hf' / 'facebook_dinov2_base',
        root / 'checkpoints' / 'hf' / 'dinov2-base',
        root / 'checkpoints' / 'dinov2-base',
    ])
    reports = [inspect_dir(p) for p in candidates]
    ok = any(r['exists'] and r['has_config_json'] and (r['has_model_safetensors'] or r['has_pytorch_model_bin']) for r in reports)
    out = root / 'outputs' / 'paper_discovery_2026-06-27' / 'external_baseline_smoke' / 'dino_backbone_check.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {'ok': ok, 'TRACKON_DINOV2_MODEL_PATH': env, 'reports': reports}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps({'out_json': str(out), **payload}, indent=2, ensure_ascii=False))
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
