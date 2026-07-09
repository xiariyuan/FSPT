#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path('/gemini/code/FSPT')
COTRACKER_ROOT = ROOT / 'baselines/cotracker'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(COTRACKER_ROOT))

from cotracker.predictor import CoTrackerOnlinePredictor
from cotracker.datasets.tap_vid_datasets import TapVidDataset
from cotracker.models.core.model_utils import bilinear_sampler

DEFAULT_NATIVE = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_true_streaming_v3_decoupled_subset_eval/full30_overlap_soft_tau055_w2_j4_min1_confirm4_p081/cotracker3_true_streaming_native_subset.pt'
DEFAULT_LABELS = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v6a3_reentry_utility/cotracker3_v6a3_reentry_utility_labels.npz'
DEFAULT_DAVIS = ROOT / 'datasets/tapvid_davis/tapvid_davis.pkl'
DEFAULT_CKPT = ROOT / 'baselines/cotracker/checkpoints/scaled_online.pth'
DEFAULT_OUT = ROOT / 'outputs/paper_discovery_2026-07-05/cotracker3_online_v7a_internal_corr/v7a_internal_corr_first3_smoke.npz'


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def to_uint8_video(sample_video: torch.Tensor) -> np.ndarray:
    v = sample_video.detach().cpu().numpy()
    if v.shape[1] in (1, 3):
        v = np.transpose(v, (0, 2, 3, 1))
    return np.clip(v, 0, 255).astype(np.uint8)


def load_video_tensor(ds: TapVidDataset, video_id: str, device: str, interp_shape: tuple[int, int]) -> torch.Tensor:
    idx = ds.video_names.index(video_id)
    sample = ds[idx]
    v = sample.video.detach().float()
    # TapVidDataset returns TCHW in [0,255]. Convert to B,T,C,H,W.
    if v.shape[1] not in (1, 3):
        raise RuntimeError(f'Unexpected video shape {tuple(v.shape)}')
    v = v[None].to(device)
    B, T, C, H, W = v.shape
    v2 = v.reshape(B * T, C, H, W)
    v2 = F.interpolate(v2, tuple(interp_shape), mode='bilinear', align_corners=True)
    v2 = v2.reshape(B, T, C, interp_shape[0], interp_shape[1])
    return v2


def compute_fmaps(model, video: torch.Tensor, chunk: int = 32) -> list[torch.Tensor]:
    # Mirrors cotracker3_online.py fnet + normalization + pyramid construction.
    B, T, C, H, W = video.shape
    fmaps_list = []
    with torch.no_grad():
        for s in range(0, T, chunk):
            vc = video[:, s:s + chunk]
            f = model.fnet(vc.reshape(-1, C, H, W))
            t = vc.shape[1]
            fmaps_list.append(f.reshape(B, t, f.shape[1], f.shape[2], f.shape[3]))
        fmaps = torch.cat(fmaps_list, dim=1)
        # Normalize along channel dim.
        fmaps = fmaps.permute(0, 1, 3, 4, 2)
        fmaps = fmaps / torch.sqrt(torch.clamp(torch.sum(torch.square(fmaps), dim=-1, keepdim=True), min=1e-12))
        fmaps = fmaps.permute(0, 1, 4, 2, 3).contiguous()
        pyramid = [fmaps]
        cur = fmaps
        for _ in range(model.corr_levels - 1):
            B, T, D, H_, W_ = cur.shape
            pooled = F.avg_pool2d(cur.reshape(B * T, D, H_, W_), 2, stride=2)
            cur = pooled.reshape(B, T, D, pooled.shape[-2], pooled.shape[-1])
            pyramid.append(cur)
    return pyramid


def sample_center_feat(fmap: torch.Tensor, frame_idx: int, yx_norm: np.ndarray, interp_hw: tuple[int, int]) -> torch.Tensor:
    # fmap: B,T,D,Hf,Wf. yx_norm in normalized 0..1 image coords. Return D unit-ish vector.
    B, T, D, Hf, Wf = fmap.shape
    y = float(yx_norm[0]) * (interp_hw[0] - 1) / 8.0  # stride=8 for level0; caller passes level-scaled fmap if needed.
    x = float(yx_norm[1]) * (interp_hw[1] - 1) / 8.0
    yy = int(round(max(0, min(Hf - 1, y))))
    xx = int(round(max(0, min(Wf - 1, x))))
    return fmap[0, frame_idx, :, yy, xx]


def extract_event_features(model, fmaps_pyr: list[torch.Tensor], pred: np.ndarray, score: np.ndarray, vis: np.ndarray, event: dict, *, interp_hw: tuple[int, int], post: int = 4) -> tuple[list[float], list[str]]:
    q = int(event['query_idx'])
    t0 = int(event['frame_t'])
    support_t = int(event.get('support_t', max(0, t0 - 1)))
    last_t = int(event.get('last_visible_t', support_t))
    T = pred.shape[1]
    names: list[str] = []
    feats: list[float] = []

    # Use level-0 fmaps for direct support-current cosine. We avoid large corr_volume export in smoke.
    fmap0 = fmaps_pyr[0]
    B, Tm, D, Hf, Wf = fmap0.shape
    support_feat = sample_center_feat(fmap0, max(0, min(T - 1, support_t)), pred[q, max(0, min(T - 1, support_t))], interp_hw)
    last_feat = sample_center_feat(fmap0, max(0, min(T - 1, last_t)), pred[q, max(0, min(T - 1, last_t))], interp_hw)
    support_feat = F.normalize(support_feat.float(), dim=0)
    last_feat = F.normalize(last_feat.float(), dim=0)

    cos_support_seq = []
    cos_last_seq = []
    corr_peak_seq = []
    corr_margin_seq = []
    corr_entropy_seq = []
    corr_center_minus_peak_seq = []

    # local patch around native coordinate, dot with support feature. This approximates internal support-current correlation.
    radius = 3
    for off in range(0, post + 1):
        tt = min(T - 1, t0 + off)
        coord = pred[q, tt]
        y0 = float(coord[0]) * (interp_hw[0] - 1) / 8.0
        x0 = float(coord[1]) * (interp_hw[1] - 1) / 8.0
        vals = []
        center_val = None
        fmap_frame = fmap0[0, tt].float()  # D,H,W
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                yy = int(round(max(0, min(Hf - 1, y0 + dy))))
                xx = int(round(max(0, min(Wf - 1, x0 + dx))))
                f = F.normalize(fmap_frame[:, yy, xx], dim=0)
                val = float(torch.dot(f, support_feat).detach().cpu())
                vals.append(val)
                if dy == 0 and dx == 0:
                    center_val = val
        arr = np.asarray(vals, np.float32)
        arr_sorted = np.sort(arr)[::-1]
        peak = float(arr_sorted[0])
        top2 = float(arr_sorted[1]) if len(arr_sorted) > 1 else peak
        margin = peak - top2
        # entropy over softmaxed correlation values, lower = sharper peak.
        exp = np.exp((arr - arr.max()) * 10.0)
        prob = exp / max(float(exp.sum()), 1e-12)
        entropy = float(-(prob * np.log(prob + 1e-12)).sum() / math.log(len(prob)))
        center = float(center_val if center_val is not None else arr[len(arr)//2])
        cur_feat = sample_center_feat(fmap0, tt, coord, interp_hw)
        cur_feat = F.normalize(cur_feat.float(), dim=0)
        cos_support = float(torch.dot(cur_feat, support_feat).detach().cpu())
        cos_last = float(torch.dot(cur_feat, last_feat).detach().cpu())
        cos_support_seq.append(cos_support)
        cos_last_seq.append(cos_last)
        corr_peak_seq.append(peak)
        corr_margin_seq.append(margin)
        corr_entropy_seq.append(entropy)
        corr_center_minus_peak_seq.append(center - peak)
        for key, val in [
            (f'cos_support_dt{off}', cos_support),
            (f'cos_last_dt{off}', cos_last),
            (f'corr_peak_dt{off}', peak),
            (f'corr_margin_dt{off}', margin),
            (f'corr_entropy_dt{off}', entropy),
            (f'corr_center_minus_peak_dt{off}', center - peak),
        ]:
            names.append(key); feats.append(float(val))

    def add_stat(prefix: str, seq: list[float]):
        a = np.asarray(seq, np.float32)
        for k, v in [
            ('mean', float(a.mean())), ('max', float(a.max())), ('min', float(a.min())), ('std', float(a.std())),
            ('slope', float(a[-1] - a[0])), ('recovery', float(a.max() - a[0]))
        ]:
            names.append(f'{prefix}_{k}'); feats.append(v)

    add_stat('cos_support', cos_support_seq)
    add_stat('cos_last', cos_last_seq)
    add_stat('corr_peak', corr_peak_seq)
    add_stat('corr_margin', corr_margin_seq)
    add_stat('corr_entropy', corr_entropy_seq)
    add_stat('corr_center_minus_peak', corr_center_minus_peak_seq)

    # Also include native temporal score context for comparison, but separate feature names make audit possible.
    for off in range(0, post + 1):
        tt = min(T - 1, t0 + off)
        names.append(f'native_score_dt{off}'); feats.append(float(score[q, tt]))
        names.append(f'native_vis_dt{off}'); feats.append(1.0 if bool(vis[q, tt]) else 0.0)
    return feats, names


def summarize_feature_auc(X: np.ndarray, y: np.ndarray, names: list[str]) -> list[dict]:
    rows = []
    if y.sum() <= 0 or y.sum() >= len(y):
        return rows
    for j, name in enumerate(names):
        vals = X[:, j]
        if len(np.unique(vals)) < 2:
            continue
        ap = average_precision_score(y, vals)
        auc = roc_auc_score(y, vals)
        apn = average_precision_score(y, -vals)
        aucn = roc_auc_score(y, -vals)
        if apn > ap:
            rows.append({'feature': name, 'direction': '-', 'ap': float(apn), 'auc': float(max(auc, 1 - auc))})
        else:
            rows.append({'feature': name, 'direction': '+', 'ap': float(ap), 'auc': float(max(auc, 1 - auc))})
    rows.sort(key=lambda r: (r['ap'], r['auc']), reverse=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--native-cache', default=str(DEFAULT_NATIVE))
    ap.add_argument('--labels', default=str(DEFAULT_LABELS))
    ap.add_argument('--davis-pkl', default=str(DEFAULT_DAVIS))
    ap.add_argument('--checkpoint', default=str(DEFAULT_CKPT))
    ap.add_argument('--out-npz', default=str(DEFAULT_OUT))
    ap.add_argument('--max-videos', type=int, default=3)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = np.load(args.labels, allow_pickle=True)
    all_meta = [json.loads(str(m)) for m in labels['meta_json'].tolist()]
    y_early4_all = np.asarray(labels['y_reentry_early4'], np.float32).astype(bool)
    y_early8_all = np.asarray(labels['y_reentry_early8'], np.float32).astype(bool)
    y_useful_all = np.asarray(labels['y_useful_open_t'], np.float32).astype(bool) if 'y_useful_open_t' in labels else None
    native = torch.load(args.native_cache, map_location='cpu', weights_only=False)
    rec_by_vid = {str(r['video_id']): r for r in native['records']}
    first_videos = [str(r['video_id']) for r in native['records'][:args.max_videos]]
    keep_idx = [i for i, m in enumerate(all_meta) if m['video_id'] in first_videos]
    keep_meta = [all_meta[i] for i in keep_idx]
    print({'videos': first_videos, 'n_events': len(keep_idx)}, flush=True)

    ds = TapVidDataset(str(args.davis_pkl), dataset_type='davis', resize_to=[256, 256], queried_first=True)
    predictor = CoTrackerOnlinePredictor(checkpoint=str(args.checkpoint)).to(args.device).eval()
    model = predictor.model
    interp_hw = tuple(predictor.interp_shape)

    X_rows = []
    feature_names = None
    used_meta = []
    for vid in first_videos:
        ev_indices = [i for i in keep_idx if all_meta[i]['video_id'] == vid]
        if not ev_indices:
            continue
        print({'video': vid, 'events': len(ev_indices)}, flush=True)
        video = load_video_tensor(ds, vid, args.device, interp_hw)
        with torch.no_grad():
            fmaps_pyr = compute_fmaps(model, video)
        rec = rec_by_vid[vid]
        pred = npy(rec['pred_tracks'], np.float32)
        vis = npy(rec['pred_visibility'], bool)
        score = npy(rec.get('pred_vis_score', vis.astype(np.float32)), np.float32)
        for idx in ev_indices:
            feats, names = extract_event_features(model, fmaps_pyr, pred, score, vis, all_meta[idx], interp_hw=interp_hw, post=4)
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise RuntimeError('feature name mismatch')
            X_rows.append(feats)
            used_meta.append(json.dumps({'source_index': idx, **all_meta[idx]}, ensure_ascii=False))
        del video, fmaps_pyr
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()

    X = np.asarray(X_rows, np.float32)
    y4 = y_early4_all[keep_idx[:len(used_meta)]]
    y8 = y_early8_all[keep_idx[:len(used_meta)]]
    yu = y_useful_all[keep_idx[:len(used_meta)]] if y_useful_all is not None else np.zeros(len(used_meta), bool)
    np.savez_compressed(out, X=X, feature_names=np.asarray(feature_names, dtype=object), meta_json=np.asarray(used_meta, dtype=object), y_reentry_early4=y4.astype(np.float32), y_reentry_early8=y8.astype(np.float32), y_useful_open_t=yu.astype(np.float32), videos=np.asarray(first_videos, dtype=object), labels_source=str(args.labels))

    audits = {
        'early4': {'n_pos': int(y4.sum()), 'base_rate': float(y4.mean()) if len(y4) else 0.0, 'top_features': summarize_feature_auc(X, y4, feature_names)[:20]},
        'early8': {'n_pos': int(y8.sum()), 'base_rate': float(y8.mean()) if len(y8) else 0.0, 'top_features': summarize_feature_auc(X, y8, feature_names)[:20]},
        'useful_open_t': {'n_pos': int(yu.sum()), 'base_rate': float(yu.mean()) if len(yu) else 0.0, 'top_features': summarize_feature_auc(X, yu, feature_names)[:20]},
    }
    report = {'script': 'scripts/export_cotracker3_online_v7a_internal_corr_features.py', 'out_npz': str(out), 'videos': first_videos, 'n_events': int(X.shape[0]), 'feature_dim': int(X.shape[1]) if X.ndim == 2 else 0, 'audits': audits, 'sec': round(time.time() - t0, 2)}
    out.with_suffix('.report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
