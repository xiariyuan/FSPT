#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / 'outputs/paper_discovery_2026-07-05'
OUTDIR = BASE / 'v9a25_dinov3_identity'
JOINT = BASE / 'v9a2_anchor_uncertainty_reacquisition/v9a2_joint_w8_common_plus_w16_extension_v3.npz'
DINO_DIR = ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m'
DAVIS_PKL = ROOT / 'datasets/tapvid_davis/tapvid_davis.pkl'

from scripts.eval_cotracker3_online_v8c0_causal_recovery_baselines import NATIVE, CANDIDATE, align_candidate, npy


def load_davis(path: Path) -> dict[str, Any]:
    with path.open('rb') as f:
        return pickle.load(f)


def prep(frames: list[np.ndarray], device: str) -> torch.Tensor:
    """Resize each frame independently before stacking.

    DAVIS videos have heterogeneous spatial resolutions, so stacking raw frames
    before resize is invalid across videos.
    """
    resized = []
    for frame in frames:
        x = torch.from_numpy(np.asarray(frame)).permute(2, 0, 1).float().unsqueeze(0)
        if float(x.max()) > 1.5:
            x = x / 255.0
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        resized.append(x[0])
    x = torch.stack(resized, dim=0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((x - mean) / std).to(device)


def sample_feat(feat: torch.Tensor, yx: list[float] | np.ndarray) -> np.ndarray:
    y = float(yx[0]); x = float(yx[1])
    grid = torch.tensor([[[[x * 2.0 - 1.0, y * 2.0 - 1.0]]]], device=feat.device, dtype=feat.dtype)
    v = F.grid_sample(feat.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=False)[0, :, 0, 0]
    v = F.normalize(v, dim=0)
    return v.detach().cpu().float().numpy()


def normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-9)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-9))


def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def offset_yx(yx: np.ndarray, dy: float, dx: float) -> list[float]:
    return [float(np.clip(float(yx[0]) + dy, 0.0, 1.0)), float(np.clip(float(yx[1]) + dx, 0.0, 1.0))]


def row_features(d: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    q = d['query']; l = d['last']; p = d['preocc']; c = d['candidate']; n = d['native']
    ql = normalize(q + l)
    qp = normalize(q + p)
    lp = normalize(l + p)
    mem = normalize(q + l + p)
    anchors = {'query': q, 'last': l, 'preocc': p, 'ql_mem': ql, 'qp_mem': qp, 'lp_mem': lp, 'all_mem': mem}
    feats: list[float] = []
    names: list[str] = []
    for name, a in anchors.items():
        feats.extend([cos(a, c), l2(a, c), cos(a, n), l2(a, n)])
        names.extend([f'{name}_cand_cos', f'{name}_cand_l2', f'{name}_native_cos', f'{name}_native_l2'])
    feats.extend([cos(q, l), cos(q, p), cos(l, p), cos(c, n), l2(c, n)])
    names.extend(['query_last_cos', 'query_preocc_cos', 'last_preocc_cos', 'candidate_native_cos', 'candidate_native_l2'])
    anchor_c = np.asarray([cos(q, c), cos(l, c), cos(p, c)], dtype=np.float32)
    anchor_n = np.asarray([cos(q, n), cos(l, n), cos(p, n)], dtype=np.float32)
    feats.extend([
        float(np.max(anchor_c)), float(np.mean(anchor_c)), float(np.std(anchor_c)),
        float(np.max(anchor_n)), float(np.mean(anchor_n)), float(np.std(anchor_n)),
        cos(mem, c) - cos(mem, n),
        float(np.max(anchor_c) - np.max(anchor_n)),
    ])
    names.extend(['anchor_cand_cos_max', 'anchor_cand_cos_mean', 'anchor_cand_cos_std', 'anchor_native_cos_max', 'anchor_native_cos_mean', 'anchor_native_cos_std', 'all_mem_cand_minus_native_cos', 'max_anchor_cand_minus_native_cos'])
    negs = [d[k] for k in ['neg_u1','neg_d1','neg_l1','neg_r1','neg_u2','neg_d2','neg_l2','neg_r2']]
    for name, a in {'query': q, 'last': l, 'preocc': p, 'all_mem': mem}.items():
        scores = np.asarray([cos(a, x) for x in negs], dtype=np.float32)
        cand_score = cos(a, c)
        feats.extend([float(np.max(scores)), float(np.mean(scores)), float(np.std(scores)), float(cand_score - np.max(scores))])
        names.extend([f'{name}_local_neg_cos_max', f'{name}_local_neg_cos_mean', f'{name}_local_neg_cos_std', f'{name}_cand_local_margin'])
    return np.asarray(feats, dtype=np.float32), names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--joint', type=Path, default=JOINT)
    ap.add_argument('--out', type=Path, default=OUTDIR / 'v9a25_dinov3_identity_features.npz')
    ap.add_argument('--max-rows', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    z = np.load(args.joint, allow_pickle=True)
    ext_all = np.where(z['is_w16_extension'].astype(bool))[0]
    ext_indices = ext_all[:args.max_rows] if args.max_rows > 0 else ext_all
    metas = [json.loads(str(z['meta_json'][i])) for i in ext_indices]

    native = torch.load(NATIVE, map_location='cpu', weights_only=False)
    cand = torch.load(CANDIDATE, map_location='cpu', weights_only=False)
    ok, info, cand_by = align_candidate(native, cand)
    if not ok:
        raise RuntimeError(info)
    native_by = {str(r['video_id']): r for r in native['records']}

    t0 = time.perf_counter()
    davis = load_davis(DAVIS_PKL)
    davis_load_s = time.perf_counter() - t0

    need: dict[tuple[str, int], list[tuple[int, str, list[float]]]] = {}
    row_point_meta: list[dict] = []
    token_off = 16.0 / 223.0
    for ri, m in enumerate(metas):
        vid = str(m['video_id']); qi = int(m['query_idx']); tau = int(m['frame_tau'])
        nr = native_by[vid]; cr = cand_by[vid]
        qpts = npy(nr['query_points'], np.float32)
        nt = npy(nr['pred_tracks'], np.float32)
        ct = npy(cr['pred_tracks'], np.float32)
        query_t = int(m['query_t']); last_t = int(m['last_t']); pre_t = int(m['preocc_t'])
        qyx = [float(qpts[qi, 1]), float(qpts[qi, 2])]
        lyx = nt[qi, last_t].tolist(); pyx = nt[qi, pre_t].tolist()
        cyx = ct[qi, tau].tolist(); nyx = nt[qi, tau].tolist()
        pts: dict[str, tuple[int, list[float]]] = {
            'query': (query_t, qyx), 'last': (last_t, lyx), 'preocc': (pre_t, pyx),
            'candidate': (tau, cyx), 'native': (tau, nyx),
        }
        for name, (dy, dx) in {
            'neg_u1': (-token_off, 0), 'neg_d1': (token_off, 0), 'neg_l1': (0, -token_off), 'neg_r1': (0, token_off),
            'neg_u2': (-2*token_off, 0), 'neg_d2': (2*token_off, 0), 'neg_l2': (0, -2*token_off), 'neg_r2': (0, 2*token_off),
        }.items():
            pts[name] = (tau, offset_yx(np.asarray(cyx), dy, dx))
        row_point_meta.append({'video_id': vid, 'query_idx': qi, 'frame_tau': tau, 'points': pts})
        for name, (ft, yx) in pts.items():
            need.setdefault((vid, int(ft)), []).append((ri, name, yx))

    from transformers import AutoModel
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.perf_counter()
    model = AutoModel.from_pretrained(str(DINO_DIR), local_files_only=True).eval().to(device)
    model_load_s = time.perf_counter() - t0
    num_reg = int(getattr(model.config, 'num_register_tokens', 4))
    patch_size = int(getattr(model.config, 'patch_size', 16))
    image_size = int(getattr(model.config, 'image_size', 224))
    grid_hw = image_size // patch_size

    feat_store: list[dict[str, np.ndarray]] = [{} for _ in metas]
    keys = sorted(need.keys())
    videos: dict[str, np.ndarray] = {}
    encode_start = time.perf_counter()
    for start in range(0, len(keys), args.batch_size):
        batch_keys = keys[start:start + args.batch_size]
        frames = []
        for vid, ft in batch_keys:
            if vid not in videos:
                videos[vid] = np.asarray(davis[vid]['video'])
            frames.append(videos[vid][ft])
        x = prep(frames, device)
        with torch.no_grad():
            out = model(pixel_values=x)
            tokens = out.last_hidden_state[:, 1 + num_reg:, :]
            fmap = tokens.reshape(tokens.shape[0], grid_hw, grid_hw, tokens.shape[-1]).permute(0, 3, 1, 2).contiguous()
        for bi, key in enumerate(batch_keys):
            f = fmap[bi]
            for ri, name, yx in need[key]:
                feat_store[ri][name] = sample_feat(f, yx)
        print(json.dumps({'encoded_frames': min(start + args.batch_size, len(keys)), 'total_frames': len(keys), 'device': device}), flush=True)
    encode_s = time.perf_counter() - encode_start

    rows = []
    names: list[str] | None = None
    for d in feat_store:
        x, n = row_features(d)
        rows.append(x)
        if names is None:
            names = n
        elif names != n:
            raise RuntimeError('feature-name mismatch')
    X = np.stack(rows, axis=0).astype(np.float32) if rows else np.zeros((0, 0), dtype=np.float32)
    if not np.all(np.isfinite(X)):
        raise RuntimeError('non-finite DINO feature')

    np.savez_compressed(
        args.out,
        X_dino=X,
        feature_names=np.asarray(names or [], dtype=object),
        joint_indices=np.asarray(ext_indices, dtype=np.int64),
        row_key_json=z['row_key_json'][ext_indices],
        event_key_json=z['event_key_json'][ext_indices],
        meta_json=z['meta_json'][ext_indices],
        y_candidate_good=z['y_candidate_good'][ext_indices],
        y_candidate_bad=z['y_candidate_bad'][ext_indices],
        y_false_visible=z['y_false_visible'][ext_indices],
        y_candidate_worse_px=z['y_candidate_worse_px'][ext_indices],
    )
    report = {
        'joint': str(args.joint), 'out': str(args.out), 'model': str(DINO_DIR), 'device': device,
        'n_rows': int(X.shape[0]), 'feature_dim': int(X.shape[1]) if X.ndim == 2 else 0,
        'finite_rate': float(np.mean(np.isfinite(X))) if X.size else None,
        'unique_frames': int(len(keys)), 'videos': sorted(set(m['video_id'] for m in metas)),
        'davis_load_seconds': float(davis_load_s), 'model_load_seconds': float(model_load_s),
        'encode_seconds': float(encode_s), 'encode_ms_per_unique_frame': float(1000.0 * encode_s / max(1, len(keys))),
        'feature_names': names or [],
    }
    report_path = args.out.with_suffix('.report.json')
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({'ok': True, **{k: report[k] for k in ['out','n_rows','feature_dim','finite_rate','unique_frames','device','encode_seconds']}, 'report': str(report_path)}))


if __name__ == '__main__':
    main()
