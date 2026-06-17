#!/usr/bin/env python3

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_world_state_stage0 import discover_sequences, find_reentry_queries, load_sequence, project_3d_to_2d


def load_rgb_frame(seq_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    rgb_path = seq_path / "rgbs" / f"rgb_{frame_idx:05d}.jpg"
    if not rgb_path.exists():
        return None
    img = cv2.imread(str(rgb_path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def extract_crop(
    image: np.ndarray,
    center_xy: np.ndarray,
    crop_size: int,
) -> np.ndarray:
    center = (float(center_xy[0]), float(center_xy[1]))
    crop = cv2.getRectSubPix(image, (crop_size, crop_size), center)
    return crop


class DINOFeatureExtractor:
    def __init__(self, weights_path: Path, device: torch.device):
        import timm

        self.device = device
        self.model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=False,
            features_only=True,
            out_indices=[3],
            img_size=518,
        ).to(device)

        sd = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        model_keys = set(self.model.state_dict().keys())
        if model_keys and not any(k.startswith("model.") for k in sd):
            sd = {"model." + k: v for k, v in sd.items()}
        pos_key = "model.pos_embed"
        if pos_key in sd and pos_key in self.model.state_dict():
            target_shape = self.model.state_dict()[pos_key].shape
            if sd[pos_key].shape != target_shape:
                old = sd[pos_key]
                cls_tok = old[:, :1]
                spatial = old[:, 1:]
                old_grid = int(spatial.shape[1] ** 0.5)
                new_grid = int((target_shape[1] - 1) ** 0.5)
                dim = spatial.shape[-1]
                spatial = spatial.reshape(1, old_grid, old_grid, dim).permute(0, 3, 1, 2)
                spatial = F.interpolate(spatial, size=(new_grid, new_grid), mode="bilinear", align_corners=False)
                spatial = spatial.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, dim)
                sd[pos_key] = torch.cat([cls_tok, spatial], dim=1)
        self.model.load_state_dict(sd, strict=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def feature_map(self, image_rgb: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(image_rgb).float().permute(2, 0, 1)[None] / 255.0
        x = F.interpolate(x, size=(518, 518), mode="bilinear", align_corners=False).to(self.device)
        feats = self.model(x)[-1]
        feats = F.normalize(feats.float(), dim=1)
        return feats[0].cpu()


def xy_to_feat_idx(local_xy: np.ndarray, crop_size: int, feat_hw: int) -> Tuple[int, int]:
    scale = feat_hw / float(crop_size)
    fx = int(np.clip(math.floor(local_xy[0] * scale), 0, feat_hw - 1))
    fy = int(np.clip(math.floor(local_xy[1] * scale), 0, feat_hw - 1))
    return fy, fx


def extract_template(
    feat_map: torch.Tensor,
    center_xy: np.ndarray,
    crop_size: int,
    radius: int,
) -> torch.Tensor:
    _, h, w = feat_map.shape
    fy, fx = xy_to_feat_idx(center_xy, crop_size, h)
    y0 = max(0, fy - radius)
    y1 = min(h, fy + radius + 1)
    x0 = max(0, fx - radius)
    x1 = min(w, fx + radius + 1)
    template = feat_map[:, y0:y1, x0:x1]
    pad_l = max(0, radius - fx)
    pad_r = max(0, fx + radius + 1 - w)
    pad_t = max(0, radius - fy)
    pad_b = max(0, fy + radius + 1 - h)
    if pad_l or pad_r or pad_t or pad_b:
        template = F.pad(template, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
    return template


def template_score_map(template: torch.Tensor, search_feat: torch.Tensor) -> torch.Tensor:
    c, th, tw = template.shape
    _, h, w = search_feat.shape
    search = F.pad(search_feat.unsqueeze(0), (tw // 2, tw // 2, th // 2, th // 2), mode="replicate")
    patches = F.unfold(search, kernel_size=(th, tw))
    patches = patches.transpose(1, 2)
    patches = F.normalize(patches, dim=-1)
    template_vec = F.normalize(template.reshape(1, -1), dim=-1)
    scores = torch.matmul(patches, template_vec.t()).reshape(h, w)
    return scores


def rgb_patch_ncc(
    query_crop: np.ndarray,
    reentry_crop: np.ndarray,
    query_center_xy: np.ndarray,
    reentry_center_xy: np.ndarray,
    patch_size: int,
) -> float:
    q_patch = extract_crop(query_crop, query_center_xy, patch_size).astype(np.float32)
    r_patch = extract_crop(reentry_crop, reentry_center_xy, patch_size).astype(np.float32)
    q = q_patch.reshape(-1) - q_patch.mean()
    r = r_patch.reshape(-1) - r_patch.mean()
    qn = float(np.linalg.norm(q))
    rn = float(np.linalg.norm(r))
    if qn < 1e-6 or rn < 1e-6:
        return 0.0
    return float(np.dot(q, r) / (qn * rn))


def dense_match_local(
    extractor: DINOFeatureExtractor,
    query_img: np.ndarray,
    reentry_img: np.ndarray,
    query_xy: np.ndarray,
    center_xy: np.ndarray,
    query_crop_size: int,
    search_crop_size: int,
) -> Tuple[np.ndarray, float]:
    q_crop = extract_crop(query_img, query_xy, query_crop_size)
    r_crop = extract_crop(reentry_img, center_xy, search_crop_size)

    q_feat = extractor.feature_map(q_crop)
    r_feat = extractor.feature_map(r_crop)
    _, hr, wr = r_feat.shape

    query_templates = []
    q_center_xy = np.array([query_crop_size / 2.0, query_crop_size / 2.0], dtype=np.float32)
    for radius in [1, 2]:
        query_templates.append(extract_template(q_feat, q_center_xy, query_crop_size, radius))

    score_maps = [template_score_map(tmpl, r_feat) for tmpl in query_templates]
    sims = torch.stack(score_maps, dim=0).mean(dim=0)

    topk = min(10, sims.numel())
    top_vals, top_idx = torch.topk(sims.reshape(-1), k=topk)
    patch_scale = search_crop_size / float(wr)
    q_rgb_center = np.array([query_crop_size / 2.0, query_crop_size / 2.0], dtype=np.float32)

    best_score = -1e9
    best_xy = center_xy.copy()
    for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
        by = idx // wr
        bx = idx % wr
        offset_xy = np.array([(bx + 0.5) * patch_scale, (by + 0.5) * patch_scale], dtype=np.float32)
        pred_xy = center_xy - (search_crop_size / 2.0) + offset_xy
        local_xy = offset_xy
        rgb_score = rgb_patch_ncc(
            query_crop=q_crop,
            reentry_crop=r_crop,
            query_center_xy=q_rgb_center,
            reentry_center_xy=local_xy,
            patch_size=32,
        )
        dist_norm = np.linalg.norm(offset_xy - np.array([search_crop_size / 2.0, search_crop_size / 2.0], dtype=np.float32))
        dist_norm = float(dist_norm / (search_crop_size / 2.0 + 1e-6))
        combined = float(val) + 0.15 * rgb_score - 0.10 * dist_norm
        if combined > best_score:
            best_score = combined
            best_xy = pred_xy

    return best_xy, float(best_score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal DINO local matching for world-state stage 2")
    parser.add_argument("--data-root", type=str, default="/gemini/code/FSPT/datasets/pointodyssey")
    parser.add_argument("--splits", type=str, default="val")
    parser.add_argument("--min-occ-length", type=int, default=20)
    parser.add_argument("--min-camera-motion", type=float, default=0.30)
    parser.add_argument("--max-sequences", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--depth-noise-sigma", type=float, default=0.10)
    parser.add_argument("--query-crop-size", type=int, default=112)
    parser.add_argument("--search-crop-size", type=int, default=224)
    parser.add_argument("--weights", type=str, default="/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth")
    parser.add_argument("--output", type=str, default="/gemini/code/FSPT/outputs/world_state_stage2_causal_dino_quick.json")
    args = parser.parse_args()

    rng = np.random.default_rng(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DINOFeatureExtractor(Path(args.weights), device)

    root = Path(args.data_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    sequences = discover_sequences(root, splits)
    if args.max_sequences > 0:
        sequences = sequences[: args.max_sequences]

    results: List[Dict] = []
    for seq_path in sequences:
        seq = load_sequence(seq_path)
        queries = find_reentry_queries(seq, min_occ_length=args.min_occ_length)
        trajs_2d = seq["trajs_2d"]
        trajs_3d = seq["trajs_3d"]
        intrinsics = seq["intrinsics"]
        extrinsics = seq["extrinsics"]

        count = 0
        for q in queries:
            if count >= args.max_samples:
                break
            t_q = q.query_frame
            t_re = q.reentry_frame
            i = q.point_idx

            e_rel = extrinsics[t_re] @ np.linalg.inv(extrinsics[t_q])
            cam_motion = float(np.linalg.norm(e_rel[:3, :3] - np.eye(3)))
            if cam_motion < args.min_camera_motion:
                continue

            query_img = load_rgb_frame(seq_path, t_q)
            reentry_img = load_rgb_frame(seq_path, t_re)
            if query_img is None or reentry_img is None:
                continue

            hold_3d = trajs_3d[t_q, i].astype(np.float32)
            pt_cam = extrinsics[t_q][:3, :3] @ hold_3d + extrinsics[t_q][:3, 3]
            z_depth = float(pt_cam[2])
            if not np.isfinite(z_depth) or z_depth <= 1e-6:
                continue
            eps = float(np.clip(rng.normal(0.0, args.depth_noise_sigma), -0.35, 0.35))
            noisy_depth = z_depth * math.exp(eps)

            pixels_h = np.array([trajs_2d[t_q, i, 0], trajs_2d[t_q, i, 1], 1.0], dtype=np.float32)
            k_inv = np.linalg.inv(intrinsics[t_q])
            pt_cam_noisy = (k_inv @ pixels_h) * noisy_depth
            e_inv = np.linalg.inv(extrinsics[t_q])
            noisy_world = (e_inv[:3, :3] @ pt_cam_noisy) + e_inv[:3, 3]

            baseline_xy = project_3d_to_2d(noisy_world, intrinsics[t_re], extrinsics[t_re]).astype(np.float32)
            gt_xy = trajs_2d[t_re, i].astype(np.float32)
            pred_xy, score = dense_match_local(
                extractor=extractor,
                query_img=query_img,
                reentry_img=reentry_img,
                query_xy=trajs_2d[t_q, i].astype(np.float32),
                center_xy=baseline_xy,
                query_crop_size=args.query_crop_size,
                search_crop_size=args.search_crop_size,
            )

            baseline_err = float(np.linalg.norm(baseline_xy - gt_xy))
            pred_err = float(np.linalg.norm(pred_xy - gt_xy))
            results.append(
                {
                    "seq": seq_path.name,
                    "query_frame": int(t_q),
                    "reentry_frame": int(t_re),
                    "point_idx": int(i),
                    "occ_length": int(q.occ_length),
                    "camera_motion": cam_motion,
                    "baseline_err": baseline_err,
                    "pred_err": pred_err,
                    "score": score,
                }
            )
            count += 1

    if not results:
        raise RuntimeError("No valid results collected.")

    baseline = np.array([r["baseline_err"] for r in results], dtype=np.float32)
    pred = np.array([r["pred_err"] for r in results], dtype=np.float32)
    summary = {
        "n": len(results),
        "baseline_median_px": float(np.median(baseline)),
        "pred_median_px": float(np.median(pred)),
        "better_frac": float(np.mean(pred < baseline)),
        "baseline_lt4px": float(np.mean(baseline < 4.0)),
        "pred_lt4px": float(np.mean(pred < 4.0)),
        "mean_score": float(np.mean([r["score"] for r in results])),
    }

    output = {
        "config": vars(args),
        "summary": summary,
        "results": results[:100],
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
