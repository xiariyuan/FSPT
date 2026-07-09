#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

ROOT = Path('/gemini/code/FSPT')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / 'outputs/paper_discovery_2026-07-05'
OUTDIR = BASE / 'v9a2_anchor_uncertainty_reacquisition'
W8_DATASET = BASE / 'cotracker3_online_v8c02_fine_risk_verifier_dataset/v8c02_fine_risk_dist_nc_le64_w8.npz'
W16_DATASET = BASE / 'cotracker3_online_v8c02_fine_risk_verifier_dataset/v8c02_fine_risk_dist_nc_le64_w16.npz'

from scripts.eval_cotracker3_online_v8c0_causal_recovery_baselines import (
    NATIVE,
    CANDIDATE,
    align_candidate,
    clone_records,
    standard_and_ajrd,
)
from scripts.eval_cotracker3_online_v8c04_apply_fine_risk_verifier import (
    apply_touched_mask,
    label_stats,
    load_dataset,
    per_video_delta,
    summarize_pv,
)


def row_key(meta: dict) -> tuple[str, int, int]:
    return (str(meta['video_id']), int(meta['query_idx']), int(meta['frame_tau']))


def event_key(meta: dict) -> tuple[str, int, int]:
    return (str(meta['video_id']), int(meta['query_idx']), int(meta.get('first_event_t', meta['frame_tau'])))


def metric_delta(metric: dict, base: dict) -> dict:
    keys = ['AJ', 'OA', 'delta_avg', 'delta_4px', 'AJ_RD', 'AJ_RD_256']
    return {k: (None if metric.get(k) is None or base.get(k) is None else float(metric[k] - base[k])) for k in keys}


def evaluate_variant(name: str, native: dict, cand_by: dict, metas: list[dict], labels: dict, accept: np.ndarray, native_metric: dict, native_pv: list[dict]) -> dict:
    recs, stats = apply_touched_mask(native, cand_by, metas, accept.astype(bool))
    metric, pv = standard_and_ajrd(recs)
    pv_rows = per_video_delta(native_pv, pv)
    return {
        'variant': name,
        'metric': metric,
        'delta_vs_native': metric_delta(metric, native_metric),
        'apply_stats': stats,
        'label_stats_on_accepted': label_stats(labels, accept.astype(bool)),
        'per_video_summary': summarize_pv(pv_rows),
    }


def label_arr(labels: dict, key: str, n: int) -> np.ndarray:
    if key not in labels:
        return np.zeros(n, dtype=np.float32)
    return np.asarray(labels[key], dtype=np.float32)


def build_masks(ds8: dict, ds16: dict) -> dict[str, np.ndarray]:
    metas16 = ds16['metas']
    n16 = len(metas16)
    keys8 = {row_key(m) for m in ds8['metas']}
    common = np.array([row_key(m) in keys8 for m in metas16], dtype=bool)
    ext = ~common
    labels = ds16['labels']
    good = label_arr(labels, 'candidate_good', n16).astype(bool)
    bad = label_arr(labels, 'candidate_bad', n16).astype(bool)
    false_visible = label_arr(labels, 'false_visible', n16).astype(bool)
    worse = label_arr(labels, 'candidate_worse_px', n16).astype(bool)
    damage16 = label_arr(labels, 'damage16', n16).astype(bool)
    utility = label_arr(labels, 'utility', n16)

    masks: dict[str, np.ndarray] = {}
    masks['native_reject_all_on_w16_rows'] = np.zeros(n16, dtype=bool)
    masks['w8_common_only_on_w16_rows'] = common.copy()
    masks['w16_accept_all'] = np.ones(n16, dtype=bool)
    masks['w8_plus_w16only_candidate_good_oracle'] = common | (ext & good)
    masks['w8_plus_w16only_utility_positive_oracle'] = common | (ext & (utility > 0))
    masks['w8_plus_w16only_non_false_visible_oracle'] = common | (ext & ~false_visible)
    masks['w8_plus_w16only_non_damage16_oracle'] = common | (ext & ~damage16)
    masks['w8_plus_w16only_safe_no_damage_oracle'] = common | (ext & ~false_visible & ~bad & ~damage16)
    masks['w8_plus_w16only_good_not_worse_oracle'] = common | (ext & good & ~worse)
    masks['w8_plus_w16only_good_and_utility_positive_oracle'] = common | (ext & good & (utility > 0))

    # Event-level label oracle: extend an event to W16 when extension utility sum is positive.
    ev_to_indices: dict[tuple[str, int, int], list[int]] = {}
    for i, m in enumerate(metas16):
        if ext[i]:
            ev_to_indices.setdefault(event_key(m), []).append(i)
    event_utility_accept = common.copy()
    event_good_accept = common.copy()
    event_safe_accept = common.copy()
    for idxs in ev_to_indices.values():
        idx = np.asarray(idxs, dtype=np.int64)
        if float(np.sum(utility[idx])) > 0:
            event_utility_accept[idx] = True
        if float(np.sum(good[idx])) > float(np.sum(bad[idx])):
            event_good_accept[idx] = True
        if np.any(good[idx]) and not np.any(damage16[idx]) and float(np.mean(false_visible[idx])) <= 0.25:
            event_safe_accept[idx] = True
    masks['event_level_w16_if_extension_utility_positive_oracle'] = event_utility_accept
    masks['event_level_w16_if_good_gt_bad_oracle'] = event_good_accept
    masks['event_level_w16_if_any_good_low_false_no_damage_oracle'] = event_safe_accept
    return masks


def common_label_consistency(ds8: dict, ds16: dict) -> dict:
    key_to_8 = {row_key(m): i for i, m in enumerate(ds8['metas'])}
    out = {'checked_common': 0, 'mismatches': {}}
    labels8 = ds8['labels']; labels16 = ds16['labels']
    keys = sorted(set(labels8) & set(labels16))
    for key in keys:
        mismatches = 0
        max_abs = 0.0
        for i16, m in enumerate(ds16['metas']):
            k = row_key(m)
            if k not in key_to_8:
                continue
            i8 = key_to_8[k]
            v8 = float(labels8[key][i8]); v16 = float(labels16[key][i16])
            diff = abs(v8 - v16)
            max_abs = max(max_abs, diff)
            if diff > 1e-6:
                mismatches += 1
        out['mismatches'][key] = {'count': int(mismatches), 'max_abs': float(max_abs)}
    out['checked_common'] = int(len(key_to_8))
    return out


def summarize_masks(ds16: dict, masks: dict[str, np.ndarray]) -> dict:
    out = {}
    labels = ds16['labels']
    n = len(ds16['metas'])
    for name, m in masks.items():
        stats = {'accepted': int(np.sum(m)), 'total': int(n), 'accept_rate': float(np.mean(m))}
        for key in ['candidate_good', 'candidate_bad', 'false_visible', 'candidate_worse_px', 'repair16', 'damage16']:
            arr = label_arr(labels, key, n)
            if np.sum(m) > 0:
                stats[key + '_mean_on_accepted'] = float(np.mean(arr[m]))
                stats[key + '_sum_on_accepted'] = float(np.sum(arr[m]))
            else:
                stats[key + '_mean_on_accepted'] = None
                stats[key + '_sum_on_accepted'] = 0.0
        if 'utility' in labels and np.sum(m) > 0:
            arr = label_arr(labels, 'utility', n)
            stats['utility_mean_on_accepted'] = float(np.mean(arr[m]))
            stats['utility_sum_on_accepted'] = float(np.sum(arr[m]))
        out[name] = stats
    return out


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    native = torch.load(NATIVE, map_location='cpu', weights_only=False)
    cand = torch.load(CANDIDATE, map_location='cpu', weights_only=False)
    ok, align_info, cand_by = align_candidate(native, cand)
    if not ok:
        raise RuntimeError(f'candidate alignment failed: {align_info}')
    ds8 = load_dataset(W8_DATASET)
    ds16 = load_dataset(W16_DATASET)
    masks = build_masks(ds8, ds16)
    native_metric, native_pv = standard_and_ajrd(clone_records(native['records']))

    rows = []
    for name, mask in masks.items():
        rows.append(evaluate_variant(name, native, cand_by, ds16['metas'], ds16['labels'], mask, native_metric, native_pv))
    rows_sorted = sorted(rows, key=lambda r: -999 if r['delta_vs_native'].get('AJ_RD_256') is None else r['delta_vs_native']['AJ_RD_256'], reverse=True)
    report = {
        'script': 'scripts/v9a2_action_space_oracle_audit.py',
        'w8_dataset': str(W8_DATASET),
        'w16_dataset': str(W16_DATASET),
        'native_metric': native_metric,
        'common_label_consistency': common_label_consistency(ds8, ds16),
        'mask_label_summary': summarize_masks(ds16, masks),
        'rows': rows,
        'best_by_AJ_RD_256': rows_sorted,
    }
    out = OUTDIR / 'v9a2_action_space_oracle_audit.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({'ok': True, 'out': str(out), 'best': rows_sorted[0]['variant'], 'best_AJ_RD_256_delta': rows_sorted[0]['delta_vs_native']['AJ_RD_256']}))


if __name__ == '__main__':
    main()
