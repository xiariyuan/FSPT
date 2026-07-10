#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/gemini/code/FSPT')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / 'outputs/paper_discovery_2026-07-05'
OUTDIR = BASE / 'v9a2_anchor_uncertainty_reacquisition'
CONTROLLER = OUTDIR / 'v9a2_dynamic_horizon_controller_report.json'
OUT_JSON = OUTDIR / 'v9a2_dynamic_horizon_video_stability_audit.json'
OUT_DOC = ROOT / 'docs/v9a2_dynamic_horizon_video_stability_audit_result_2026-07-09.md'

from scripts.v9a2_dynamic_horizon_controller import JOINT, event_accept_mask, load_joint
from scripts.eval_cotracker3_online_v8c0_causal_recovery_baselines import CANDIDATE, NATIVE, align_candidate, clone_records, standard_and_ajrd
from scripts.eval_cotracker3_online_v8c04_apply_fine_risk_verifier import apply_touched_mask, per_video_delta, summarize_pv


def metric_delta(metric: dict, base: dict) -> dict:
    keys = ['AJ', 'OA', 'delta_avg', 'delta_4px', 'AJ_RD', 'AJ_RD_256']
    return {k: None if metric.get(k) is None or base.get(k) is None else float(metric[k] - base[k]) for k in keys}


def eval_variant(name: str, data: dict, native: dict, cand_by: dict, accept: np.ndarray, native_metric: dict, native_pv: list[dict]) -> dict:
    recs, stats = apply_touched_mask(native, cand_by, data['metas'], accept.astype(bool))
    metric, pv = standard_and_ajrd(recs)
    pv_rows = per_video_delta(native_pv, pv)
    ext = data['is_w16_extension'].astype(bool)
    y_good = data['y_candidate_good'].astype(bool)
    y_bad = data['y_candidate_bad'].astype(bool)
    y_false = data['y_false_visible'].astype(bool)
    y_worse = data['y_candidate_worse_px'].astype(bool)
    return {
        'variant': name,
        'metric': metric,
        'delta_vs_native': metric_delta(metric, native_metric),
        'apply_stats': stats,
        'per_video_summary': summarize_pv(pv_rows),
        'per_video_rows': pv_rows,
        'extension_stats': {
            'accepted_extension': int(np.sum(accept & ext)),
            'accepted_ext_good': int(np.sum(accept & ext & y_good)),
            'accepted_ext_bad': int(np.sum(accept & ext & y_bad)),
            'accepted_ext_false_visible': int(np.sum(accept & ext & y_false)),
            'accepted_ext_worse': int(np.sum(accept & ext & y_worse)),
        },
    }


def pv_delta_key(row: dict) -> float | None:
    for key in ['AJ_RD_256_delta', 'delta_AJ_RD_256', 'AJ_RD_256']:
        if key in row and row[key] is not None:
            return float(row[key])
    # fallback: find a plausible key
    for key, val in row.items():
        if 'AJ_RD_256' in key and isinstance(val, (int, float)):
            return float(val)
    return None


def main() -> None:
    data = load_joint(JOINT)
    report = json.loads(CONTROLLER.read_text())
    native = torch.load(NATIVE, map_location='cpu', weights_only=False)
    cand = torch.load(CANDIDATE, map_location='cpu', weights_only=False)
    ok, info, cand_by = align_candidate(native, cand)
    if not ok:
        raise RuntimeError(info)
    native_metric, native_pv = standard_and_ajrd(clone_records(native['records']))

    common = data['is_common_w8'].astype(bool)
    ext = data['is_w16_extension'].astype(bool)
    y_good = data['y_candidate_good'].astype(bool)
    y_worse = data['y_candidate_worse_px'].astype(bool)

    scores = {}
    thresholds = {}
    for block in report['training']:
        for overall in block['overall']:
            name = f"{overall['feature_set']}_{overall['model']}"
            if name in block['full_scores']:
                scores[name] = np.asarray(block['full_scores'][name], dtype=float)
                thresholds[name] = float(overall['threshold_f1_oof'])

    specs = {
        'w8_preserve_common_only': common,
        'w16_accept_all_joint': common | ext,
        'oracle_ext_candidate_good': common | (ext & y_good),
        'learned_all_logreg_event_max_fixed_0.05': event_accept_mask(data, scores['all_logreg'], 0.05, 'event_max'),
        'learned_all_logreg_event_max_oof_f1': event_accept_mask(data, scores['all_logreg'], thresholds['all_logreg'], 'event_max'),
        'learned_anchor_logreg_event_max_fixed_0.01': event_accept_mask(data, scores['anchor_logreg'], 0.01, 'event_max'),
    }
    variants = [eval_variant(name, data, native, cand_by, accept, native_metric, native_pv) for name, accept in specs.items()]
    by_name = {v['variant']: v for v in variants}

    # Compare learned best against W16 per video.
    learned = by_name['learned_all_logreg_event_max_fixed_0.05']
    w16 = by_name['w16_accept_all_joint']
    def map_pv(v):
        return {str(r.get('video_id', r.get('video', r.get('name', 'unknown')))): r for r in v['per_video_rows']}
    lw = map_pv(learned); ww = map_pv(w16)
    comparison = []
    skipped_undefined = []
    for vid in sorted(set(lw) & set(ww)):
        lval = pv_delta_key(lw[vid]); wval = pv_delta_key(ww[vid])
        if lval is None or wval is None:
            skipped_undefined.append(vid)
            continue
        comparison.append({'video_id': vid, 'learned_AJ_RD_256_delta': lval, 'w16_AJ_RD_256_delta': wval, 'learned_minus_w16': float(lval - wval)})
    comparison_sorted = sorted(comparison, key=lambda r: r['learned_minus_w16'])
    out = {
        'variants': variants,
        'learned_vs_w16_video_comparison': comparison_sorted,
        'summary': {
            'defined_videos': int(len(comparison)),
            'skipped_undefined_videos': skipped_undefined,
            'skipped_undefined_count': int(len(skipped_undefined)),
            'learned_better_videos': int(sum(r['learned_minus_w16'] > 1e-9 for r in comparison)),
            'learned_worse_videos': int(sum(r['learned_minus_w16'] < -1e-9 for r in comparison)),
            'learned_equal_videos': int(sum(abs(r['learned_minus_w16']) <= 1e-9 for r in comparison)),
            'mean_learned_minus_w16': float(np.mean([r['learned_minus_w16'] for r in comparison])) if comparison else None,
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    def fmt(x): return '' if x is None else f'{float(x):+.4f}'
    lines = ['# V9-A2.3c Video Stability Audit', '', '## Variant summary', '', '| Variant | AJ Δ | OA Δ | AJ_RD_256 Δ | Accepted ext/good/bad/false/worse | Pos/Neg/Zero |', '|---|---:|---:|---:|---:|---|']
    for v in variants:
        d = v['delta_vs_native']; pv = v['per_video_summary']; es = v['extension_stats']
        ext_stats = f"{es['accepted_extension']}/{es['accepted_ext_good']}/{es['accepted_ext_bad']}/{es['accepted_ext_false_visible']}/{es['accepted_ext_worse']}"
        lines.append(f"| {v['variant']} | {fmt(d['AJ'])} | {fmt(d['OA'])} | {fmt(d['AJ_RD_256'])} | {ext_stats} | {pv['positive']}/{pv['negative']}/{pv['zero']} |")
    lines += ['', '## Learned fixed 0.05 vs W16 by video', '', f"Defined AJ_RD_256 videos={out['summary']['defined_videos']}; skipped undefined={out['summary']['skipped_undefined_count']}. Summary: better={out['summary']['learned_better_videos']}, worse={out['summary']['learned_worse_videos']}, equal={out['summary']['learned_equal_videos']}, mean_diff={out['summary']['mean_learned_minus_w16']:+.6f}", '', '| Worst videos learned-minus-W16 | Δ |', '|---|---:|']
    for r in comparison_sorted[:10]:
        lines.append(f"| {r['video_id']} | {r['learned_minus_w16']:+.6f} |")
    lines += ['', '| Best videos learned-minus-W16 | Δ |', '|---|---:|']
    for r in comparison_sorted[-10:][::-1]:
        lines.append(f"| {r['video_id']} | {r['learned_minus_w16']:+.6f} |")
    OUT_DOC.write_text('\n'.join(lines))
    print(json.dumps({'ok': True, 'json': str(OUT_JSON), 'doc': str(OUT_DOC), **out['summary']}))


if __name__ == '__main__':
    main()
