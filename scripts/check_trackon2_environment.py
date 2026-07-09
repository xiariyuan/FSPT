#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def find_spec(name: str):
    try:
        spec = importlib.util.find_spec(name)
        return {'found': bool(spec), 'origin': spec.origin if spec else None}
    except Exception as e:
        return {'found': False, 'error': repr(e)}


def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f'ERROR: {e}'


def main():
    root = Path(__file__).resolve().parent.parent
    info = {
        'python_executable': sys.executable,
        'python_version': sys.version.split()[0],
        'platform': platform.platform(),
        'cwd': os.getcwd(),
        'repo_root': str(root),
        'modules': {m: find_spec(m) for m in ['torch', 'torchvision', 'mmengine', 'mmcv', 'mmcv.ops']},
        'conda_env': os.environ.get('CONDA_DEFAULT_ENV'),
        'which_python': run(['which', 'python']),
        'conda_env_list': run(['conda', 'env', 'list']) if find_spec('conda') else run(['bash','-lc','conda env list || true']),
    }
    try:
        import torch
        info['torch'] = {
            'version': torch.__version__,
            'cuda_version': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except Exception as e:
        info['torch_error'] = repr(e)
    try:
        from mmcv.ops import MultiScaleDeformableAttention  # noqa: F401
        info['mmcv_ops_ok'] = True
    except Exception as e:
        info['mmcv_ops_ok'] = False
        info['mmcv_ops_error'] = repr(e)
    try:
        sys.path.insert(0, str(root / 'baselines' / 'track_on'))
        from model.trackon_predictor import Predictor  # noqa: F401
        info['trackon2_import_ok'] = True
    except Exception as e:
        info['trackon2_import_ok'] = False
        info['trackon2_import_error'] = repr(e)
    out = root / 'outputs' / 'paper_discovery_2026-06-27' / 'external_baseline_smoke' / 'trackon2_env_check.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps({'out_json': str(out), 'mmcv_ops_ok': info['mmcv_ops_ok'], 'trackon2_import_ok': info['trackon2_import_ok'], 'info': info}, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
