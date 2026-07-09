#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

STRESSES={
 'translate_L16': 'outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/stress_dataset.pt',
 'occluder_L16': 'outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/occluder_L16/stress_dataset.pt',
}
PRED={
 'translate_L16': {
  'offline':'outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/predictions/cotracker3_offline_translate_L16.pt',
  'online':'outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/predictions/cotracker3_online_translate_L16.pt',
  'b2':'outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/predictions/b2_w16_p2_translate_L16.pt',
 },
 'occluder_L16': {
  'offline':'outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/occluder_L16/predictions/cotracker3_offline_occluder_L16.pt',
  'online':'outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/occluder_L16/predictions/cotracker3_online_occluder_L16.pt',
  'b2':'outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/occluder_L16/predictions/b2_w16_p2_occluder_L16.pt',
 },
}
VARIANTS=[
 ('sg0_uncond',0,'unconditional'),
 ('sg20_vismem',20,'visibility_selective'),
 ('sg0_vismem',0,'visibility_selective'),
]
ROOT=Path('outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_variant_256q')
ENV_PREFIX='DINOV3_LOCAL_DIR=/gemini/code/FSPT/third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m '
rows=[]
for stress, ds in STRESSES.items():
    for name, sg, mem in VARIANTS:
        outdir=ROOT/stress/name
        outdir.mkdir(parents=True, exist_ok=True)
        cache=outdir/f'trackon2_dinov3_{stress}_{name}_256q.pt'
        report=outdir/f'trackon2_dinov3_{stress}_{name}_256q_report.json'
        compare=outdir/f'compare_{stress}_{name}_256q.json'
        cmd=(f"{ENV_PREFIX}python scripts/export_trackon2_reentry_stress_cache.py "
             f"--config baselines/track_on/config/test.yaml "
             f"--checkpoint baselines/track_on/checkpoints_trackon2_dinov3.pt "
             f"--stress-dataset {ds} "
             f"--out-cache {cache} --out-report {report} "
             f"--max-videos 1 --max-queries 256 --support-grid-size {sg} --memory-policy {mem}")
        print('RUN', stress, name, flush=True)
        subprocess.run(cmd, shell=True, check=True)
        items=' '.join([f"{k}={v}" for k,v in PRED[stress].items()] + [f"trackon2={cache}"])
        cmd2=(f"python scripts/eval_external_baseline_smoke_cache.py --max-records 1 --max-queries 256 "
              f"--out-json {compare} --items {items}")
        subprocess.run(cmd2, shell=True, check=True)
        data=json.load(open(compare))['rows']
        tr=[r for r in data if r['name']=='trackon2'][0]
        b2=[r for r in data if r['name']=='b2'][0]
        rep=json.load(open(report))
        rows.append({'stress':stress,'variant':name,'support_grid_size':sg,'memory_policy':mem,'trackon2':tr,'b2':b2,'report':rep})
summary={'rows':rows}
ROOT.mkdir(parents=True,exist_ok=True)
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
print(json.dumps({'out':str(ROOT/'summary.json'),'rows':[{'stress':r['stress'],'variant':r['variant'],'AJ_RD':r['trackon2']['AJ_RD_256'],'AJ':r['trackon2']['AJ_256'],'OA':r['trackon2']['OA_256'],'vis_rate':r['report']['mean_vis_rate']} for r in rows]},indent=2,ensure_ascii=False))
