#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / 'outputs/paper_discovery_2026-07-05'
OUTDIR = BASE / 'v9a2_anchor_uncertainty_reacquisition'
DEFAULT_DATASET = BASE / 'cotracker3_online_v8c02_fine_risk_verifier_dataset/v8c02_fine_risk_dist_nc_le64_w8.npz'
DAVIS_PKL = ROOT / 'datasets/tapvid_davis/tapvid_davis.pkl'

from scripts.eval_cotracker3_online_v8c0_causal_recovery_baselines import NATIVE, CANDIDATE, align_candidate, npy
from scripts.v9a1_controller_calibration_aware_prototype import load_v8c02_dataset, video_from_meta


def norm_yx_to_pixel(yx: np.ndarray, h: int, w: int) -> tuple[float, float]:
    """Convert normalized yx to raw pixel coordinates without clipping.

    Out-of-bound coordinates are informative for risk/uncertainty, so clipping
    must happen only inside patch intersection logic, not here.
    """
    y = float(yx[0]) * float(h - 1)
    x = float(yx[1]) * float(w - 1)
    return y, x


def pixel_oob_features(y: float, x: float, h: int, w: int) -> list[float]:
    y_low = max(0.0, -float(y))
    y_high = max(0.0, float(y) - float(h - 1))
    x_low = max(0.0, -float(x))
    x_high = max(0.0, float(x) - float(w - 1))
    dist = float(np.sqrt(y_low * y_low + y_high * y_high + x_low * x_low + x_high * x_high))
    return [float(dist > 0.0), y_low, y_high, x_low, x_high, dist]


def extract_patch(img: np.ndarray, y: float, x: float, radius: int) -> tuple[np.ndarray, float]:
    r = int(radius)
    h, w = img.shape[:2]
    cy = int(round(y)); cx = int(round(x))
    y_start = cy - r; y_end = cy + r + 1
    x_start = cx - r; x_end = cx + r + 1
    y0 = max(0, y_start); y1 = min(h, y_end)
    x0 = max(0, x_start); x1 = min(w, x_end)
    patch = np.zeros((2 * r + 1, 2 * r + 1, 3), dtype=np.float32)
    valid = np.zeros((2 * r + 1, 2 * r + 1), dtype=bool)
    if y1 <= y0 or x1 <= x0:
        return patch, 0.0
    py0 = y0 - y_start; px0 = x0 - x_start
    patch[py0:py0 + (y1 - y0), px0:px0 + (x1 - x0)] = img[y0:y1, x0:x1].astype(np.float32) / 255.0
    valid[py0:py0 + (y1 - y0), px0:px0 + (x1 - x0)] = True
    return patch, float(np.mean(valid))


def patch_desc(patch: np.ndarray, valid_frac: float) -> np.ndarray:
    flat = patch.reshape(-1, 3)
    gray = np.mean(patch, axis=2)
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy)
    return np.array([
        valid_frac,
        float(np.mean(flat[:, 0])), float(np.mean(flat[:, 1])), float(np.mean(flat[:, 2])),
        float(np.std(flat[:, 0])), float(np.std(flat[:, 1])), float(np.std(flat[:, 2])),
        float(np.mean(gray)), float(np.std(gray)), float(np.mean(grad)), float(np.std(grad)),
    ], dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))


def pair_features(a: np.ndarray, va: float, b: np.ndarray, vb: float) -> list[float]:
    da = patch_desc(a, va); db = patch_desc(b, vb)
    diff = da - db
    return [
        cosine(da, db),
        float(np.linalg.norm(diff)),
        float(np.mean(np.abs(a - b))),
        float(np.mean((a - b) ** 2)),
        va,
        vb,
        min(va, vb),
    ]


def load_davis_pkl(path: Path) -> dict[str, Any]:
    with path.open('rb') as f:
        return pickle.load(f)


def record_maps(native: dict, cand: dict) -> tuple[dict[str, dict], dict[str, dict]]:
    ok, info, cand_by = align_candidate(native, cand)
    if not ok:
        raise RuntimeError(f'candidate alignment failed: {info}')
    native_by = {str(r['video_id']): r for r in native['records']}
    return native_by, cand_by


def build_row_features(meta: dict, native_rec: dict, cand_rec: dict, video: np.ndarray, radii: list[int]) -> tuple[np.ndarray, dict]:
    q = int(meta['query_idx'])
    tau = int(meta['frame_tau'])
    first_event_t = int(meta.get('first_event_t', tau))
    last_t = int(meta.get('last_native_visible_t', max(0, first_event_t - 1)))
    qpts = npy(native_rec['query_points'], np.float32)
    ntracks = npy(native_rec['pred_tracks'], np.float32)
    ctracks = npy(cand_rec['pred_tracks'], np.float32)
    nvis = npy(native_rec['pred_visibility'], bool)
    nscore = npy(native_rec.get('pred_vis_score', np.zeros_like(nvis, dtype=np.float32)), np.float32)
    nconf = npy(native_rec.get('pred_conf_prob_component', np.zeros_like(nscore, dtype=np.float32)), np.float32)
    h, w = int(video.shape[1]), int(video.shape[2])
    T = int(video.shape[0])
    tau = max(0, min(T - 1, tau)); first_event_t = max(0, min(T - 1, first_event_t)); last_t = max(0, min(T - 1, last_t))
    query_t = int(round(float(qpts[q, 0]))); query_t = max(0, min(T - 1, query_t))
    pre_t = last_t
    preocc_found = False
    # Strict pre-occlusion anchor: never use the event trigger frame itself.
    # The event frame may already be low-confidence / invisible / drifting.
    search_start = min(first_event_t - 1, T - 1)
    if search_start >= 0:
        for t in range(search_start, -1, -1):
            if bool(nvis[q, t]):
                pre_t = t
                preocc_found = True
                break
    q_yx = np.asarray([qpts[q, 1], qpts[q, 2]], dtype=np.float32)
    last_yx = ntracks[q, last_t]
    pre_yx = ntracks[q, pre_t]
    cand_yx = ctracks[q, tau]
    native_yx = ntracks[q, tau]

    feats: list[float] = []
    names: list[str] = []

    def add_oob(name: str, yx: np.ndarray) -> None:
        py, px = norm_yx_to_pixel(yx, h, w)
        vals = pixel_oob_features(py, px, h, w)
        feats.extend(vals)
        names.extend([f'{name}_oob_flag', f'{name}_oob_y_low_px', f'{name}_oob_y_high_px', f'{name}_oob_x_low_px', f'{name}_oob_x_high_px', f'{name}_oob_distance_px'])

    for pname, pyx in [('candidate', cand_yx), ('native', native_yx), ('query_anchor', q_yx), ('last_anchor', last_yx), ('preocc_anchor', pre_yx)]:
        add_oob(pname, pyx)

    def safe_arr_value(arr: np.ndarray, t: int, default: float = 0.0) -> float:
        try:
            return float(arr[q, max(0, min(arr.shape[1] - 1, int(t)))])
        except Exception:
            return float(default)

    def motion_at(t: int) -> float:
        t = max(0, min(T - 1, int(t)))
        if t <= 0:
            return 0.0
        return float(np.linalg.norm(ntracks[q, t] - ntracks[q, t - 1]))

    last_cand_disagree = float(np.linalg.norm(ntracks[q, last_t] - ctracks[q, last_t]))
    pre_cand_disagree = float(np.linalg.norm(ntracks[q, pre_t] - ctracks[q, pre_t]))
    last_pre_dist = float(np.linalg.norm(last_yx - pre_yx))
    reliability_vals = [
        float(bool(nvis[q, last_t])),
        float(bool(nvis[q, pre_t])),
        safe_arr_value(nscore, last_t),
        safe_arr_value(nscore, pre_t),
        safe_arr_value(nconf, last_t),
        safe_arr_value(nconf, pre_t),
        last_cand_disagree,
        pre_cand_disagree,
        motion_at(last_t),
        motion_at(pre_t),
        last_pre_dist,
        float(max(0, tau - last_t)),
        float(max(0, first_event_t - pre_t)),
        float(preocc_found),
        float(max(0, first_event_t - pre_t)),
        safe_arr_value(nscore, pre_t),
    ]
    reliability_names = [
        'last_anchor_native_visible',
        'preocc_anchor_native_visible',
        'last_anchor_vis_score',
        'preocc_anchor_vis_score',
        'last_anchor_conf_prob',
        'preocc_anchor_conf_prob',
        'last_anchor_native_candidate_disagree',
        'preocc_anchor_native_candidate_disagree',
        'last_anchor_motion_jump',
        'preocc_anchor_motion_jump',
        'last_preocc_anchor_distance',
        'tau_minus_last_anchor',
        'event_minus_preocc_anchor',
        'preocc_anchor_found',
        'preocc_anchor_age',
        'preocc_anchor_score',
    ]
    feats.extend(reliability_vals)
    names.extend(reliability_names)

    anchors = [('query', query_t, q_yx), ('last', last_t, last_yx), ('preocc', pre_t, pre_yx)]
    for radius in radii:
        cy, cx = norm_yx_to_pixel(cand_yx, h, w)
        cand_patch, cand_valid = extract_patch(video[tau], cy, cx, radius)
        for aname, at, ayx in anchors:
            ay, ax = norm_yx_to_pixel(ayx, h, w)
            a_patch, a_valid = extract_patch(video[at], ay, ax, radius)
            pf = pair_features(a_patch, a_valid, cand_patch, cand_valid)
            feats.extend(pf)
            names.extend([f'r{radius}_{aname}_cand_cos', f'r{radius}_{aname}_cand_desc_l2', f'r{radius}_{aname}_cand_mad', f'r{radius}_{aname}_cand_mse', f'r{radius}_{aname}_valid', f'r{radius}_cand_valid', f'r{radius}_{aname}_pair_valid_min'])
    dist_nc = float(np.linalg.norm(native_yx - cand_yx))
    event_age = float(max(0, tau - first_event_t))
    invis_age = float(max(0, tau - last_t))
    motion = float(np.linalg.norm(ntracks[q, tau] - ntracks[q, max(0, tau - 1)])) if tau > 0 else 0.0
    sigma_proxy = float(np.sqrt(1e-6 + 0.0025 + 0.0005 * event_age + 0.0005 * invis_age + motion * motion))
    norm_dist = dist_nc / max(sigma_proxy, 1e-6)
    feats.extend([dist_nc, event_age, invis_age, motion, sigma_proxy, norm_dist])
    names.extend(['dist_native_candidate_norm_yx', 'event_age', 'invis_age', 'native_motion_prev_norm_yx', 'sigma_proxy', 'normalized_distance'])
    info = {
        'video_id': str(meta['video_id']),
        'query_idx': q,
        'frame_tau': tau,
        'first_event_t': first_event_t,
        'query_t': query_t,
        'last_t': last_t,
        'preocc_t': pre_t,
        'preocc_found': bool(preocc_found),
        'preocc_strictly_before_event': bool(pre_t < first_event_t),
        'feature_names': names,
        'candidate_yx': [float(cand_yx[0]), float(cand_yx[1])],
        'native_yx': [float(native_yx[0]), float(native_yx[1])],
    }
    return np.asarray(feats, dtype=np.float32), info


def build_smoke(max_rows: int, radii: list[int]) -> dict:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = load_v8c02_dataset(DEFAULT_DATASET)
    native = torch.load(NATIVE, map_location='cpu', weights_only=False)
    cand = torch.load(CANDIDATE, map_location='cpu', weights_only=False)
    native_by, cand_by = record_maps(native, cand)
    davis = load_davis_pkl(DAVIS_PKL)
    metas = data['metas'][:max_rows]
    rows = []
    X = []
    names = None
    for m in metas:
        vid = video_from_meta(m)
        feat, info = build_row_features(m, native_by[vid], cand_by[vid], davis[vid]['video'], radii)
        X.append(feat)
        names = info.pop('feature_names')
        rows.append(info)
    X_arr = np.stack(X, axis=0) if X else np.zeros((0, 0), dtype=np.float32)
    report = {
        'max_rows': int(max_rows),
        'n_rows': int(X_arr.shape[0]),
        'feature_dim': int(X_arr.shape[1]) if X_arr.ndim == 2 else 0,
        'feature_names': names or [],
        'finite_rate': float(np.mean(np.isfinite(X_arr))) if X_arr.size else None,
        'mean_first_features': np.mean(X_arr[:, :min(8, X_arr.shape[1])], axis=0).tolist() if X_arr.size else [],
        'rows': rows[:5],
    }
    (OUTDIR / 'v9a2_feature_smoke.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    np.savez_compressed(OUTDIR / 'v9a2_feature_smoke.npz', X_anchor=X_arr, feature_names=np.asarray(names or [], dtype=object), meta_json=np.asarray([json.dumps(r, ensure_ascii=False) for r in rows], dtype=object))
    return report



def stratified_indices(metas: list[dict], rows_per_video: int = 2) -> list[int]:
    by_video: dict[str, list[int]] = {}
    for i, m in enumerate(metas):
        by_video.setdefault(video_from_meta(m), []).append(i)
    out: list[int] = []
    for vid in sorted(by_video):
        idxs = by_video[vid]
        if len(idxs) <= rows_per_video:
            out.extend(idxs)
        else:
            pick = np.linspace(0, len(idxs) - 1, rows_per_video, dtype=np.int64)
            out.extend([idxs[int(j)] for j in pick])
    return sorted(set(out))


def build_stratified_smoke(rows_per_video: int, radii: list[int], *, suffix: str = "") -> dict:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = load_v8c02_dataset(DEFAULT_DATASET)
    native = torch.load(NATIVE, map_location='cpu', weights_only=False)
    cand = torch.load(CANDIDATE, map_location='cpu', weights_only=False)
    native_by, cand_by = record_maps(native, cand)
    davis = load_davis_pkl(DAVIS_PKL)
    idxs = stratified_indices(data['metas'], rows_per_video=rows_per_video)
    rows = []
    X = []
    names = None
    for idx in idxs:
        m = data['metas'][idx]
        vid = video_from_meta(m)
        feat, info = build_row_features(m, native_by[vid], cand_by[vid], davis[vid]['video'], radii)
        X.append(feat)
        names = info.pop('feature_names')
        info['source_index'] = int(idx)
        rows.append(info)
    X_arr = np.stack(X, axis=0).astype(np.float32) if X else np.zeros((0, 0), dtype=np.float32)
    valid_keys = [i for i, n in enumerate(names or []) if n.endswith('_valid') or n.endswith('_pair_valid_min') or n.endswith('_cand_valid')]
    valid_values = X_arr[:, valid_keys] if valid_keys and X_arr.size else np.zeros((0, 0), dtype=np.float32)
    oob_flag_keys = [i for i, n in enumerate(names or []) if n.endswith('_oob_flag')]
    oob_dist_keys = [i for i, n in enumerate(names or []) if n.endswith('_oob_distance_px')]
    oob_flags = X_arr[:, oob_flag_keys] if oob_flag_keys and X_arr.size else np.zeros((0, 0), dtype=np.float32)
    oob_dists = X_arr[:, oob_dist_keys] if oob_dist_keys and X_arr.size else np.zeros((0, 0), dtype=np.float32)
    videos = sorted(set(r['video_id'] for r in rows))
    per_video = {}
    for vid in videos:
        local = [r for r in rows if r['video_id'] == vid]
        per_video[vid] = {
            'n': len(local),
            'frame_tau_min': int(min(r['frame_tau'] for r in local)),
            'frame_tau_max': int(max(r['frame_tau'] for r in local)),
            'last_t_ne_preocc_count': int(sum(r['last_t'] != r['preocc_t'] for r in local)),
        }
    report = {
        'rows_per_video': int(rows_per_video),
        'n_rows': int(X_arr.shape[0]),
        'n_videos': int(len(videos)),
        'videos': videos,
        'feature_dim': int(X_arr.shape[1]) if X_arr.ndim == 2 else 0,
        'finite_rate': float(np.mean(np.isfinite(X_arr))) if X_arr.size else None,
        'valid_feature_min': float(np.min(valid_values)) if valid_values.size else None,
        'valid_feature_mean': float(np.mean(valid_values)) if valid_values.size else None,
        'valid_feature_p05': float(np.quantile(valid_values, 0.05)) if valid_values.size else None,
        'oob_flag_mean': float(np.mean(oob_flags)) if oob_flags.size else None,
        'oob_any_row_rate': float(np.mean(np.any(oob_flags > 0.5, axis=1))) if oob_flags.size else None,
        'oob_distance_max': float(np.max(oob_dists)) if oob_dists.size else None,
        'oob_distance_mean': float(np.mean(oob_dists)) if oob_dists.size else None,
        'normalized_distance_min': float(np.min(X_arr[:, -1])) if X_arr.size else None,
        'normalized_distance_max': float(np.max(X_arr[:, -1])) if X_arr.size else None,
        'normalized_distance_mean': float(np.mean(X_arr[:, -1])) if X_arr.size else None,
        'per_video': per_video,
        'rows': rows[:10],
        'feature_names': names or [],
    }
    tag = f'_{suffix}' if suffix else ''
    (OUTDIR / f'v9a2_feature_smoke_stratified{tag}.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    np.savez_compressed(
        OUTDIR / f'v9a2_feature_smoke_stratified{tag}.npz',
        X_anchor=X_arr,
        feature_names=np.asarray(names or [], dtype=object),
        meta_json=np.asarray([json.dumps(r, ensure_ascii=False) for r in rows], dtype=object),
        source_indices=np.asarray(idxs, dtype=np.int64),
    )
    return report


def row_key(meta: dict) -> tuple[str, int, int]:
    return (video_from_meta(meta), int(meta['query_idx']), int(meta['frame_tau']))


def event_key(meta: dict) -> tuple[str, int, int]:
    return (video_from_meta(meta), int(meta['query_idx']), int(meta.get('first_event_t', meta['frame_tau'])))


def load_npz_with_metas(path: Path) -> dict:
    d = load_v8c02_dataset(path)
    d['key_to_index'] = {row_key(m): i for i, m in enumerate(d['metas'])}
    return d


def label_mean(data: dict, indices: list[int], key: str) -> float | None:
    if key not in data or not indices:
        return None
    return float(np.mean(np.asarray(data[key])[indices].astype(float)))


def label_sum(data: dict, indices: list[int], key: str) -> int | None:
    if key not in data or not indices:
        return None
    return int(np.sum(np.asarray(data[key])[indices].astype(float)))


def subset_stats(data: dict, indices: list[int]) -> dict:
    labels = ['y_candidate_good', 'y_candidate_bad', 'y_false_visible', 'y_candidate_worse_px', 'y_repair16', 'y_damage16', 'y_gt_visible', 'y_candidate_safe16']
    out = {'n': int(len(indices))}
    for key in labels:
        out[key + '_mean'] = label_mean(data, indices, key)
        out[key + '_sum'] = label_sum(data, indices, key)
    if 'y_utility' in data and indices:
        out['y_utility_mean'] = float(np.mean(np.asarray(data['y_utility'])[indices].astype(float)))
        out['y_utility_sum'] = float(np.sum(np.asarray(data['y_utility'])[indices].astype(float)))
    return out


def audit_w8_w16_action_space() -> dict:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    w8_path = BASE / 'cotracker3_online_v8c02_fine_risk_verifier_dataset/v8c02_fine_risk_dist_nc_le64_w8.npz'
    w16_path = BASE / 'cotracker3_online_v8c02_fine_risk_verifier_dataset/v8c02_fine_risk_dist_nc_le64_w16.npz'
    w8 = load_npz_with_metas(w8_path)
    w16 = load_npz_with_metas(w16_path)
    keys8 = set(w8['key_to_index'])
    keys16 = set(w16['key_to_index'])
    common = sorted(keys8 & keys16)
    w8_only = sorted(keys8 - keys16)
    w16_only = sorted(keys16 - keys8)
    common8_idx = [w8['key_to_index'][k] for k in common]
    common16_idx = [w16['key_to_index'][k] for k in common]
    w8_only_idx = [w8['key_to_index'][k] for k in w8_only]
    w16_only_idx = [w16['key_to_index'][k] for k in w16_only]

    by_video = {}
    for key in w16_only:
        vid = key[0]
        by_video.setdefault(vid, []).append(w16['key_to_index'][key])
    per_video_w16_only = {}
    for vid, idxs in sorted(by_video.items()):
        per_video_w16_only[vid] = subset_stats(w16, idxs)
    report = {
        'w8_path': str(w8_path),
        'w16_path': str(w16_path),
        'n_w8': int(len(keys8)),
        'n_w16': int(len(keys16)),
        'n_common': int(len(common)),
        'n_w8_only': int(len(w8_only)),
        'n_w16_only': int(len(w16_only)),
        'w8_all_stats': subset_stats(w8, list(range(len(w8['metas'])))),
        'w16_all_stats': subset_stats(w16, list(range(len(w16['metas'])))),
        'common_as_w8_stats': subset_stats(w8, common8_idx),
        'common_as_w16_stats': subset_stats(w16, common16_idx),
        'w8_only_stats': subset_stats(w8, w8_only_idx),
        'w16_only_stats': subset_stats(w16, w16_only_idx),
        'per_video_w16_only': per_video_w16_only,
        'top_w16_only_good_videos': sorted(per_video_w16_only.items(), key=lambda kv: (-1 if kv[1].get('y_candidate_good_sum') is None else -kv[1].get('y_candidate_good_sum'), kv[0]))[:10],
        'top_w16_only_damage_videos': sorted(per_video_w16_only.items(), key=lambda kv: (-1 if kv[1].get('y_damage16_sum') is None else -kv[1].get('y_damage16_sum'), kv[0]))[:10],
    }
    (OUTDIR / 'v9a2_w8_w16_action_space_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report



def build_full_feature_dataset(dataset_path: Path, out_npz: Path, radii: list[int], max_rows: int = 0, reference_w8: Path | None = None) -> dict:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = load_v8c02_dataset(dataset_path)
    native = torch.load(NATIVE, map_location='cpu', weights_only=False)
    cand = torch.load(CANDIDATE, map_location='cpu', weights_only=False)
    native_by, cand_by = record_maps(native, cand)
    davis = load_davis_pkl(DAVIS_PKL)
    metas_all = data['metas']
    if max_rows and max_rows > 0:
        metas = metas_all[:int(max_rows)]
        row_indices = list(range(int(max_rows)))
    else:
        metas = metas_all
        row_indices = list(range(len(metas_all)))

    ref_keys: set[tuple[str, int, int]] | None = None
    if reference_w8 is not None and reference_w8.exists():
        ref = load_v8c02_dataset(reference_w8)
        ref_keys = {row_key(m) for m in ref['metas']}

    X_anchor_rows: list[np.ndarray] = []
    out_meta: list[str] = []
    row_key_json: list[str] = []
    event_key_json: list[str] = []
    row_type: list[str] = []
    is_common_w8: list[int] = []
    is_w16_extension: list[int] = []
    names: list[str] | None = None

    for local_i, (src_i, m) in enumerate(zip(row_indices, metas)):
        vid = video_from_meta(m)
        feat, info = build_row_features(m, native_by[vid], cand_by[vid], davis[vid]['video'], radii)
        if names is None:
            names = info.pop('feature_names')
        else:
            info.pop('feature_names', None)
        key = row_key(m)
        ev = event_key(m)
        if ref_keys is None:
            rtype = 'common_w8'
            common = True
            ext = False
        else:
            common = key in ref_keys
            ext = not common
            rtype = 'common_w8' if common else 'w16_extension'
        info.update({'source_index': int(src_i), 'row_key': key, 'event_key': ev, 'row_type': rtype, 'is_common_w8': bool(common), 'is_w16_extension': bool(ext)})
        X_anchor_rows.append(feat)
        out_meta.append(json.dumps(info, ensure_ascii=False))
        row_key_json.append(json.dumps(key, ensure_ascii=False))
        event_key_json.append(json.dumps(ev, ensure_ascii=False))
        row_type.append(rtype)
        is_common_w8.append(int(common))
        is_w16_extension.append(int(ext))
        if (local_i + 1) % 250 == 0:
            print(json.dumps({'processed': local_i + 1, 'total': len(metas), 'dataset': str(dataset_path)}), flush=True)

    X_anchor = np.stack(X_anchor_rows, axis=0).astype(np.float32) if X_anchor_rows else np.zeros((0, 0), dtype=np.float32)
    X_base_full = np.asarray(data['X'], dtype=np.float32)
    X_base = X_base_full[row_indices]
    base_names = [str(x) for x in np.asarray(data.get('feature_names', [f'f{i}' for i in range(X_base.shape[1])])).tolist()]
    anchor_names = names or []
    X_all = np.concatenate([X_base, X_anchor], axis=1).astype(np.float32) if len(X_base) else np.zeros((0, len(base_names) + len(anchor_names)), dtype=np.float32)
    all_names = [f'base::{n}' for n in base_names] + [f'anchor::{n}' for n in anchor_names]

    save_kwargs = {
        'X_base': X_base,
        'X_anchor': X_anchor,
        'X_all': X_all,
        'base_feature_names': np.asarray(base_names, dtype=object),
        'anchor_feature_names': np.asarray(anchor_names, dtype=object),
        'all_feature_names': np.asarray(all_names, dtype=object),
        'meta_json': np.asarray(out_meta, dtype=object),
        'source_meta_json': np.asarray([json.dumps(m, ensure_ascii=False) for m in metas], dtype=object),
        'row_key_json': np.asarray(row_key_json, dtype=object),
        'event_key_json': np.asarray(event_key_json, dtype=object),
        'row_type': np.asarray(row_type, dtype=object),
        'is_common_w8': np.asarray(is_common_w8, dtype=np.int8),
        'is_w16_extension': np.asarray(is_w16_extension, dtype=np.int8),
        'source_dataset': np.asarray(str(dataset_path), dtype=object),
        'reference_w8': np.asarray(str(reference_w8) if reference_w8 is not None else '', dtype=object),
    }
    for k, v in data.items():
        if k.startswith('y_'):
            save_kwargs[k] = np.asarray(v)[row_indices]
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **save_kwargs)

    valid_keys = [i for i, n in enumerate(anchor_names) if n.endswith('_valid') or n.endswith('_pair_valid_min') or n.endswith('_cand_valid')]
    oob_flag_keys = [i for i, n in enumerate(anchor_names) if n.endswith('_oob_flag')]
    valid_values = X_anchor[:, valid_keys] if valid_keys and X_anchor.size else np.zeros((0, 0), dtype=np.float32)
    oob_flags = X_anchor[:, oob_flag_keys] if oob_flag_keys and X_anchor.size else np.zeros((0, 0), dtype=np.float32)
    report = {
        'dataset': str(dataset_path),
        'out_npz': str(out_npz),
        'n_rows': int(X_all.shape[0]),
        'feature_dim_base': int(X_base.shape[1]) if X_base.ndim == 2 else 0,
        'feature_dim_anchor': int(X_anchor.shape[1]) if X_anchor.ndim == 2 else 0,
        'feature_dim_all': int(X_all.shape[1]) if X_all.ndim == 2 else 0,
        'finite_rate_X_all': float(np.mean(np.isfinite(X_all))) if X_all.size else None,
        'finite_rate_X_anchor': float(np.mean(np.isfinite(X_anchor))) if X_anchor.size else None,
        'valid_feature_min': float(np.min(valid_values)) if valid_values.size else None,
        'valid_feature_mean': float(np.mean(valid_values)) if valid_values.size else None,
        'valid_feature_p05': float(np.quantile(valid_values, 0.05)) if valid_values.size else None,
        'oob_flag_mean': float(np.mean(oob_flags)) if oob_flags.size else None,
        'oob_any_row_rate': float(np.mean(np.any(oob_flags > 0.5, axis=1))) if oob_flags.size else None,
        'row_type_counts': {t: int(row_type.count(t)) for t in sorted(set(row_type))},
        'labels': {},
    }
    for k, v in save_kwargs.items():
        if k.startswith('y_'):
            arr = np.asarray(v).astype(float)
            report['labels'][k] = {'mean': float(np.mean(arr)) if arr.size else None, 'sum': float(np.sum(arr)) if arr.size else 0.0}
    return report


def audit_feature_consistency(w8_npz: Path, w16_npz: Path, out_json: Path) -> dict:
    w8 = np.load(w8_npz, allow_pickle=True)
    w16 = np.load(w16_npz, allow_pickle=True)
    keys8 = [json.loads(str(x)) for x in w8['row_key_json'].tolist()]
    keys16 = [json.loads(str(x)) for x in w16['row_key_json'].tolist()]
    map8 = {tuple(k): i for i, k in enumerate(keys8)}
    map16 = {tuple(k): i for i, k in enumerate(keys16)}
    common = sorted(set(map8) & set(map16))
    X8a = w8['X_anchor']; X16a = w16['X_anchor']
    X8b = w8['X_base']; X16b = w16['X_base']
    diffs_anchor = []
    diffs_base = []
    mismatched_rows_anchor = 0
    mismatched_rows_base = 0
    for k in common:
        i8 = map8[k]; i16 = map16[k]
        da = np.abs(X8a[i8] - X16a[i16])
        db = np.abs(X8b[i8] - X16b[i16])
        diffs_anchor.append(da)
        diffs_base.append(db)
        mismatched_rows_anchor += int(float(np.max(da)) > 1e-6)
        mismatched_rows_base += int(float(np.max(db)) > 1e-6)
    A = np.stack(diffs_anchor, axis=0) if diffs_anchor else np.zeros((0, 0), dtype=np.float32)
    B = np.stack(diffs_base, axis=0) if diffs_base else np.zeros((0, 0), dtype=np.float32)
    report = {
        'w8_npz': str(w8_npz),
        'w16_npz': str(w16_npz),
        'n_w8': int(len(keys8)),
        'n_w16': int(len(keys16)),
        'n_common': int(len(common)),
        'anchor_max_abs_diff': float(np.max(A)) if A.size else None,
        'anchor_mean_abs_diff': float(np.mean(A)) if A.size else None,
        'anchor_mismatched_rows': int(mismatched_rows_anchor),
        'base_max_abs_diff': float(np.max(B)) if B.size else None,
        'base_mean_abs_diff': float(np.mean(B)) if B.size else None,
        'base_mismatched_rows': int(mismatched_rows_base),
        'feature_dim_anchor_w8': int(X8a.shape[1]),
        'feature_dim_anchor_w16': int(X16a.shape[1]),
        'feature_dim_base_w8': int(X8b.shape[1]),
        'feature_dim_base_w16': int(X16b.shape[1]),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report

def binary_overlap(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    inter = int(np.sum(a & b))
    union = int(np.sum(a | b))
    a_sum = int(np.sum(a)); b_sum = int(np.sum(b))
    return {
        'a_sum': a_sum,
        'b_sum': b_sum,
        'intersection': inter,
        'union': union,
        'precision_a_given_b': None if b_sum == 0 else float(inter / b_sum),
        'recall_b_given_a': None if a_sum == 0 else float(inter / a_sum),
        'jaccard': None if union == 0 else float(inter / union),
    }


def audit_label_overlap() -> dict:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    w16_path = BASE / 'cotracker3_online_v8c02_fine_risk_verifier_dataset/v8c02_fine_risk_dist_nc_le64_w16.npz'
    w8_path = BASE / 'cotracker3_online_v8c02_fine_risk_verifier_dataset/v8c02_fine_risk_dist_nc_le64_w8.npz'
    w16 = load_npz_with_metas(w16_path)
    w8 = load_npz_with_metas(w8_path)
    keys8 = set(w8['key_to_index'])
    w16_only = np.array([row_key(m) not in keys8 for m in w16['metas']], dtype=bool)
    labels = w16
    good = np.asarray(labels['y_candidate_good']).astype(bool)
    utility_positive = np.asarray(labels['y_utility']) > 0
    good_not_worse = good & ~np.asarray(labels['y_candidate_worse_px']).astype(bool)
    safe_no_damage = ~np.asarray(labels['y_false_visible']).astype(bool) & ~np.asarray(labels['y_candidate_bad']).astype(bool) & ~np.asarray(labels['y_damage16']).astype(bool)
    non_false_visible = ~np.asarray(labels['y_false_visible']).astype(bool)
    non_damage16 = ~np.asarray(labels['y_damage16']).astype(bool)
    sets = {
        'candidate_good': good,
        'utility_positive': utility_positive,
        'good_not_worse': good_not_worse,
        'safe_no_damage': safe_no_damage,
        'non_false_visible': non_false_visible,
        'non_damage16': non_damage16,
    }
    subsets = {'all_w16': np.ones_like(w16_only, dtype=bool), 'w16_only': w16_only, 'common_w8': ~w16_only}
    report = {'w16_path': str(w16_path), 'w8_path': str(w8_path), 'subsets': {}}
    for subset_name, mask in subsets.items():
        sub = {'n': int(np.sum(mask)), 'set_counts': {}, 'pairwise': {}}
        for name, arr in sets.items():
            sub['set_counts'][name] = int(np.sum(arr & mask))
        names_list = list(sets.keys())
        for i, a_name in enumerate(names_list):
            for b_name in names_list[i + 1:]:
                sub['pairwise'][f'{a_name}__vs__{b_name}'] = binary_overlap(sets[a_name][mask], sets[b_name][mask])
        report['subsets'][subset_name] = sub
    (OUTDIR / 'v9a2_label_overlap_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--max-rows', type=int, default=20)
    ap.add_argument('--radii', type=int, nargs='*', default=[5, 9, 17])
    ap.add_argument('--smoke-stratified', action='store_true')
    ap.add_argument('--rows-per-video', type=int, default=2)
    ap.add_argument('--audit-action-space', action='store_true')
    ap.add_argument('--smoke-stratified-v2', action='store_true')
    ap.add_argument('--smoke-stratified-v3', action='store_true')
    ap.add_argument('--audit-label-overlap', action='store_true')
    ap.add_argument('--build-full', action='store_true')
    ap.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    ap.add_argument('--out-npz', type=Path, default=None)
    ap.add_argument('--reference-w8', type=Path, default=BASE / 'cotracker3_online_v8c02_fine_risk_verifier_dataset/v8c02_fine_risk_dist_nc_le64_w8.npz')
    ap.add_argument('--audit-feature-consistency', action='store_true')
    ap.add_argument('--w8-features', type=Path, default=OUTDIR / 'v9a2_features_dist_nc_le64_w8_v3.npz')
    ap.add_argument('--w16-features', type=Path, default=OUTDIR / 'v9a2_features_dist_nc_le64_w16_v3.npz')
    args = ap.parse_args()
    if args.build_full:
        out_npz = args.out_npz
        if out_npz is None:
            raise SystemExit('--out-npz is required with --build-full')
        ref = args.reference_w8 if args.reference_w8 and args.reference_w8.exists() else None
        report = build_full_feature_dataset(args.dataset, out_npz, args.radii, max_rows=args.max_rows, reference_w8=ref)
        report_path = out_npz.with_suffix('.report.json')
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps({'ok': True, 'out_npz': str(out_npz), 'report': str(report_path), 'n_rows': report['n_rows'], 'feature_dim_all': report['feature_dim_all'], 'finite_rate': report['finite_rate_X_all'], 'row_type_counts': report['row_type_counts']}))
        return
    if args.audit_feature_consistency:
        report = audit_feature_consistency(args.w8_features, args.w16_features, OUTDIR / 'v9a2_w8_w16_feature_consistency_audit.json')
        print(json.dumps({'ok': True, 'n_common': report['n_common'], 'anchor_max_abs_diff': report['anchor_max_abs_diff'], 'base_max_abs_diff': report['base_max_abs_diff'], 'anchor_mismatched_rows': report['anchor_mismatched_rows'], 'base_mismatched_rows': report['base_mismatched_rows']}))
        return
    if args.smoke:
        report = build_smoke(args.max_rows, args.radii)
        print(json.dumps({'ok': True, 'n_rows': report['n_rows'], 'feature_dim': report['feature_dim'], 'finite_rate': report['finite_rate']}))
        return
    if args.smoke_stratified:
        report = build_stratified_smoke(args.rows_per_video, args.radii)
        print(json.dumps({'ok': True, 'n_rows': report['n_rows'], 'n_videos': report['n_videos'], 'feature_dim': report['feature_dim'], 'finite_rate': report['finite_rate'], 'valid_p05': report['valid_feature_p05']}))
        return
    if args.smoke_stratified_v2:
        report = build_stratified_smoke(args.rows_per_video, args.radii, suffix='v2')
        print(json.dumps({'ok': True, 'n_rows': report['n_rows'], 'n_videos': report['n_videos'], 'feature_dim': report['feature_dim'], 'finite_rate': report['finite_rate'], 'valid_p05': report['valid_feature_p05'], 'oob_any_row_rate': report['oob_any_row_rate'], 'oob_distance_max': report['oob_distance_max']}))
        return
    if args.smoke_stratified_v3:
        report = build_stratified_smoke(args.rows_per_video, args.radii, suffix='v3')
        strict_rate = None
        if report.get('rows'):
            strict_rate = sum(1 for r in report['rows'] if r.get('preocc_strictly_before_event')) / float(len(report['rows']))
        print(json.dumps({'ok': True, 'n_rows': report['n_rows'], 'n_videos': report['n_videos'], 'feature_dim': report['feature_dim'], 'finite_rate': report['finite_rate'], 'valid_p05': report['valid_feature_p05'], 'oob_any_row_rate': report['oob_any_row_rate'], 'oob_distance_max': report['oob_distance_max'], 'strict_preocc_rate_in_preview': strict_rate}))
        return
    if args.audit_label_overlap:
        report = audit_label_overlap()
        w16_only = report['subsets']['w16_only']
        print(json.dumps({'ok': True, 'out': str(OUTDIR / 'v9a2_label_overlap_audit.json'), 'w16_only_n': w16_only['n'], 'w16_only_counts': w16_only['set_counts']}))
        return
    if args.audit_action_space:
        report = audit_w8_w16_action_space()
        print(json.dumps({'ok': True, 'n_w8': report['n_w8'], 'n_w16': report['n_w16'], 'n_w16_only': report['n_w16_only'], 'w16_only_good_mean': report['w16_only_stats']['y_candidate_good_mean'], 'w16_only_damage16_mean': report['w16_only_stats']['y_damage16_mean']}))
        return
    raise SystemExit('Use --build-full, --audit-feature-consistency, --smoke, --smoke-stratified, --smoke-stratified-v2, --smoke-stratified-v3, --audit-label-overlap, or --audit-action-space.')


if __name__ == '__main__':
    main()
