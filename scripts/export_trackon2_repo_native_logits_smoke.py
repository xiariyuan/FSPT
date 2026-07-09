#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils import data

ROOT = Path('/gemini/code/FSPT')
TRACKON_DIR = ROOT / 'baselines/track_on'
OLD_NPZ = ROOT / 'outputs/trackon2_dinov3_davis_cache/davis/trackon2/000000.npz'
OUT_DIR = ROOT / 'outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/repo_native_logit_smoke/davis/trackon2'
OUT_NPZ = OUT_DIR / '000000.npz'
OUT_REPORT = ROOT / 'outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/repo_native_logit_smoke_report.json'
DAVIS_PKL = ROOT / 'datasets/tapvid_davis/tapvid_davis.pkl'
CKPT = TRACKON_DIR / 'checkpoints_trackon2_dinov3.pt'
CONFIG = TRACKON_DIR / 'config/test.yaml'
DINOV3_LOCAL_DIR = ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m'

if DINOV3_LOCAL_DIR.is_dir():
    os.environ['DINOV3_LOCAL_DIR'] = str(DINOV3_LOCAL_DIR)

# Use repo-native import paths.
os.chdir(str(TRACKON_DIR))
sys.path.insert(0, str(TRACKON_DIR))
sys.path.append(str(ROOT))

from dataset.tapvid import TAPVid  # noqa: E402
from model.trackon_predictor import Predictor  # noqa: E402
from utils.coord_utils import get_points_on_a_grid  # noqa: E402
from utils.train_utils import load_args_from_yaml  # noqa: E402


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


class LogitPredictor(Predictor):
    """Local-only Predictor subclass that preserves default Predictor behavior.

    It mirrors Predictor.forward/forward_frame and returns logits only through the
    explicit forward_with_logits method. The original Predictor class is not
    modified.
    """

    @torch.no_grad()
    def forward_frame_with_logits(self, frame, new_queries=None):
        device = frame.device
        _, _, H, W = frame.shape

        f4_t, f8_t, f16_t, f32_t, f_fused_t = self.model.extract_frame_features(frame)

        if new_queries is not None and new_queries.shape[0] > 0:
            self.init_queries((f_fused_t, device), new_queries, H, W)

        if self.q_init is None or self.N == 0:
            return (
                torch.empty(0, 2, device=device),
                torch.empty(0, dtype=torch.bool, device=device),
                torch.empty(0, dtype=torch.float32, device=device),
            )

        frame_features = (f4_t, f8_t, f16_t, f32_t, f_fused_t)
        p, v_logit, q_new = self.model.track_frame(
            self.q_init[:self.N],
            self.temporal_mask[:self.N],
            self.point_memory[:self.N],
            frame_features,
            H,
            W,
        )

        v_t = (v_logit.sigmoid() >= self.delta_v)
        if new_queries is not None and new_queries.shape[0] > 0:
            new_count = int(new_queries.shape[0])
            new_query_mask = torch.zeros(self.N, dtype=torch.bool, device=device)
            new_query_mask[self.N - new_count:self.N] = True
        else:
            new_query_mask = torch.zeros(self.N, dtype=torch.bool, device=device)

        if self.model.memory_update_policy == 'visibility_selective':
            write_mask = v_t | new_query_mask
        elif self.model.memory_update_policy == 'unconditional':
            write_mask = torch.ones(self.N, dtype=torch.bool, device=device)
        else:
            raise ValueError(f'Unknown memory_update_policy: {self.model.memory_update_policy}')

        updated_memory, updated_mask = self.model._update_point_memory(
            self.point_memory[:self.N],
            self.temporal_mask[:self.N],
            q_new,
            write_mask,
        )
        self.point_memory[:self.N] = updated_memory
        self.temporal_mask[:self.N] = updated_mask
        self.t += 1

        return p, v_t, v_logit.float()

    @torch.no_grad()
    def forward_with_logits(self, video, queries):
        _, T, _, H, W = video.shape
        device = video.device

        queries = queries.squeeze(0)
        N_orig = queries.shape[0]
        query_times = queries[:, 0].long()
        query_coords = queries[:, 1:]

        if self.support_grid_size > 0:
            extra = get_points_on_a_grid(self.support_grid_size, (H, W), device).squeeze(0)
            extra_queries = torch.cat([torch.zeros(extra.shape[0], 1, device=device), extra], dim=1)
            queries = torch.cat([queries, extra_queries], dim=0)
            query_times = queries[:, 0].long()
            query_coords = queries[:, 1:]

        N_total = queries.shape[0]
        self.reset()
        self.initial_capacity = N_total

        pred_trajectory = torch.zeros(1, T, N_orig, 2, device='cpu', dtype=torch.float32)
        pred_visibility = torch.zeros(1, T, N_orig, device='cpu', dtype=torch.bool)
        pred_vis_logit = torch.zeros(1, T, N_orig, device='cpu', dtype=torch.float32)
        tracking_to_original: list[int] = []

        for t in range(T):
            frame = video[0, t].unsqueeze(0)
            new_queries_mask = (query_times == t)
            new_queries_this_frame = None

            if new_queries_mask.any():
                new_queries_this_frame = query_coords[new_queries_mask]
                new_indices = new_queries_mask.nonzero(as_tuple=True)[0]
                tracking_to_original.extend(new_indices.tolist())

            if self.N > 0 or new_queries_this_frame is not None:
                p_t, v_t, vlogit_t = self.forward_frame_with_logits(frame, new_queries=new_queries_this_frame)

                if self.N > 0:
                    tracking_to_original_tensor = torch.tensor(tracking_to_original, dtype=torch.long, device='cpu')
                    orig_mask = tracking_to_original_tensor < N_orig
                    if orig_mask.any():
                        orig_indices = tracking_to_original_tensor[orig_mask]
                        pred_trajectory[0, t, orig_indices] = p_t[orig_mask].cpu()
                        pred_visibility[0, t, orig_indices] = v_t[orig_mask].cpu()
                        pred_vis_logit[0, t, orig_indices] = vlogit_t[orig_mask].cpu()

                    del p_t, v_t, vlogit_t, tracking_to_original_tensor, orig_mask
                    if new_queries_this_frame is not None:
                        del new_queries_this_frame
            del frame

        pred_vis_conf = torch.sigmoid(pred_vis_logit)
        return pred_trajectory, pred_visibility, pred_vis_logit, pred_vis_conf




def prepare_tapvid_data_local(trajectory, visibility, query_points_i, device):
    """Local copy of evaluation.evaluator.prepare_tapvid_data to avoid wandb import.

    Inputs are exactly from repo-native TAPVid DataLoader.
    """
    queries = query_points_i.clone().float()
    queries = torch.stack([queries[:, :, 0], queries[:, :, 2], queries[:, :, 1]], dim=2).to(device)
    traj = trajectory.clone()
    query_points = query_points_i.clone().cpu().numpy()
    gt_tracks = traj.permute(0, 2, 1, 3).cpu().numpy()
    gt_occluded = torch.logical_not(visibility.clone().permute(0, 2, 1)).cpu().numpy()
    return queries, gt_tracks, gt_occluded, query_points

def summarize_diff(old: np.ndarray, new: np.ndarray, *, bool_like: bool = False) -> dict:
    if old.shape != new.shape:
        return {'same_shape': False, 'old_shape': list(old.shape), 'new_shape': list(new.shape)}
    if bool_like:
        diff = old.astype(bool) != new.astype(bool)
        return {
            'same_shape': True,
            'diff_count': int(diff.sum()),
            'diff_rate': float(diff.mean()) if diff.size else 0.0,
        }
    d = np.abs(old.astype(np.float64) - new.astype(np.float64))
    return {
        'same_shape': True,
        'max_abs_diff': float(d.max()) if d.size else 0.0,
        'mean_abs_diff': float(d.mean()) if d.size else 0.0,
        'p95_abs_diff': float(np.percentile(d, 95)) if d.size else 0.0,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_args = load_args_from_yaml(str(CONFIG))
    model_args.grad_checkpoint = False
    model_args.M_i = 24  # repo-native DAVIS setting in evaluation/eval.py
    model = LogitPredictor(model_args, checkpoint_path=str(CKPT), support_grid_size=20).to(device).eval()

    dataset = TAPVid(None, data_root=str(DAVIS_PKL), dataset_type='davis')
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    video, trajectory, visibility, query_points_i = next(iter(loader))

    query_points_i = query_points_i.to(device, non_blocking=True)
    trajectory = trajectory.to(device, non_blocking=True)
    visibility = visibility.to(device, non_blocking=True)
    video = video.to(device, non_blocking=True)

    queries, gt_tracks, gt_occluded, query_points = prepare_tapvid_data_local(
        trajectory, visibility, query_points_i, device
    )

    t0 = time.time()
    with torch.no_grad():
        if device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                tracks, vis, vlogit, vconf = model.forward_with_logits(video.clone(), queries.clone())
        else:
            tracks, vis, vlogit, vconf = model.forward_with_logits(video.clone(), queries.clone())
    infer_sec = time.time() - t0

    np.savez(
        OUT_NPZ,
        tracks=tracks.cpu().numpy(),
        visibility=vis.cpu().numpy(),
        visibility_logit=vlogit.cpu().numpy(),
        visibility_conf=vconf.cpu().numpy(),
    )

    old = np.load(OLD_NPZ)
    new = np.load(OUT_NPZ)
    report = {
        'script': 'scripts/export_trackon2_repo_native_logits_smoke.py',
        'old_npz': str(OLD_NPZ),
        'new_npz': str(OUT_NPZ),
        'dataset': str(DAVIS_PKL),
        'checkpoint': str(CKPT),
        'config': str(CONFIG),
        'dinov3_local_dir': os.environ.get('DINOV3_LOCAL_DIR', ''),
        'device': str(device),
        'model_args': {
            'M': int(model_args.M),
            'M_i': int(model_args.M_i),
            'delta_v': float(model_args.delta_v),
            'memory_update_policy': str(model_args.memory_update_policy),
            'support_grid_size': 20,
        },
        'infer_sec': round(float(infer_sec), 3),
        'keys_new': sorted(list(new.files)),
        'required_keys_present': all(k in new.files for k in ['tracks', 'visibility', 'visibility_logit', 'visibility_conf']),
        'diff_tracks': summarize_diff(old['tracks'], new['tracks']),
        'diff_visibility': summarize_diff(old['visibility'], new['visibility'], bool_like=True),
        'visibility_conf_range': {
            'min': float(new['visibility_conf'].min()),
            'max': float(new['visibility_conf'].max()),
            'mean': float(new['visibility_conf'].mean()),
        },
        'visibility_logit_range': {
            'min': float(new['visibility_logit'].min()),
            'max': float(new['visibility_logit'].max()),
            'mean': float(new['visibility_logit'].mean()),
        },
        'visibility_equals_conf_ge_0p8': bool(np.all(new['visibility'].astype(bool) == (new['visibility_conf'] >= 0.8))),
        'old_shapes': {k: list(old[k].shape) for k in old.files},
        'new_shapes': {k: list(new[k].shape) for k in new.files},
    }
    report['pass_strict'] = bool(
        report['required_keys_present']
        and report['diff_tracks'].get('same_shape')
        and report['diff_tracks'].get('max_abs_diff', 1.0) <= 1e-4
        and report['diff_visibility'].get('same_shape')
        and report['diff_visibility'].get('diff_count', 1) == 0
        and report['visibility_equals_conf_ge_0p8']
    )
    OUT_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
