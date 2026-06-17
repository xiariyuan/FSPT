#!/usr/bin/env python3
"""
Build a local selector dataset from the precomputed recovery-anchor dataset v3.

Data source:
  outputs/recovery_anchor_dataset_v3_full/
    - index_train.json
    - index_val.json
    - per-sample NPZ with support_descriptor + search_feature_map

This bypasses expensive online forward passes and uses the existing train/val
recovery-event split directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def normalize_feat_map(feat_map: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(feat_map, axis=0, keepdims=True)
    norm = np.clip(norm, 1e-8, None)
    return feat_map / norm


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec
    return vec / norm


def topk_cosine_candidates(
    support_descriptor: np.ndarray,
    search_feature_map: np.ndarray,
    topk: int,
) -> List[Dict]:
    desc = normalize_vec(support_descriptor.astype(np.float32))
    fmap = normalize_feat_map(search_feature_map.astype(np.float32))
    scores = (fmap * desc[:, None, None]).sum(axis=0)  # (Hf, Wf)
    flat_idx = np.argsort(scores.reshape(-1))[::-1][:topk]
    h, w = scores.shape

    out = []
    top1_score = float(scores.reshape(-1)[flat_idx[0]]) if len(flat_idx) else 0.0
    for rank, idx in enumerate(flat_idx):
        fy, fx = divmod(int(idx), w)
        out.append(
            {
                "rank_by_dino": rank,
                "fx": fx,
                "fy": fy,
                "score": float(scores[fy, fx]),
                "top1_score": top1_score,
            }
        )
    return out


def feat_to_global_xy(
    fx: int,
    fy: int,
    fmap_w: int,
    fmap_h: int,
    base_xy: np.ndarray,
    search_crop_size: float,
) -> np.ndarray:
    local_x = (float(fx) + 0.5) / float(fmap_w)
    local_y = (float(fy) + 0.5) / float(fmap_h)
    offset_x = (local_x - 0.5) * float(search_crop_size)
    offset_y = (local_y - 0.5) * float(search_crop_size)
    return np.array([base_xy[0] + offset_x, base_xy[1] + offset_y], dtype=np.float32)


def build_split(anchor_root: Path, split: str, topk: int) -> Dict:
    index_path = anchor_root / f"index_{split}.json"
    samples = json.load(open(index_path))
    out_rows = []
    per_vid = {}

    for meta in samples:
        npz_path = anchor_root / meta["file"]
        if not npz_path.exists():
            continue
        data = np.load(npz_path, allow_pickle=False)
        if "support_descriptor" not in data or "search_feature_map" not in data:
            continue

        support_descriptor = data["support_descriptor"].astype(np.float32)
        search_feature_map = data["search_feature_map"].astype(np.float32)
        fmap_c, fmap_h, fmap_w = search_feature_map.shape

        base_xy = np.asarray(meta["base_xy"], dtype=np.float32)
        gt_xy = np.asarray(meta["gt_xy"], dtype=np.float32)
        base_err = float(meta["base_error_px"])
        search_crop_size = float(meta["search_crop_size"])

        cands_raw = topk_cosine_candidates(support_descriptor, search_feature_map, topk=topk)
        if len(cands_raw) < 2:
            continue

        candidates = []
        top1_xy = None
        cand_errors = []
        scores = [c["score"] for c in cands_raw]
        shifts = []
        oracle_idx = 0
        oracle_err = float("inf")

        for j, c in enumerate(cands_raw):
            cand_xy = feat_to_global_xy(c["fx"], c["fy"], fmap_w, fmap_h, base_xy, search_crop_size)
            cand_err = float(np.linalg.norm(cand_xy - gt_xy))
            cand_shift_px = float(np.linalg.norm(cand_xy - base_xy))
            shifts.append(cand_shift_px)
            cand_errors.append(cand_err)
            if j == 0:
                top1_xy = cand_xy
            if cand_err < oracle_err:
                oracle_err = cand_err
                oracle_idx = j
            candidates.append(
                {
                    "rank_by_dino": j,
                    "cand_xy_norm": [
                        float(np.clip(cand_xy[0] / float(meta["image_W"]), 0.0, 1.0)),
                        float(np.clip(cand_xy[1] / float(meta["image_H"]), 0.0, 1.0)),
                    ],
                    "cand_error_px": round(cand_err, 2),
                    "cand_score": round(float(c["score"]), 4),
                    "cand_margin_to_top2": 0.0,  # filled below
                    "cand_shift_px_from_base": round(cand_shift_px, 2),
                    "cand_shift_norm_from_base": round(
                        float(
                            np.linalg.norm(
                                np.array(
                                    [
                                        cand_xy[0] / float(meta["image_W"]),
                                        cand_xy[1] / float(meta["image_H"]),
                                    ],
                                    dtype=np.float32,
                                )
                                - np.array(meta["base_xy_norm"], dtype=np.float32)
                            )
                        ),
                        4,
                    ),
                    "cand_dist_to_base_px": round(cand_shift_px, 2),
                    "cand_dist_to_top1_px": 0.0,  # filled below
                    "cand_is_oracle": False,
                    "cand_better_than_base": bool(cand_err < base_err),
                    "cand_better_by_2px": bool(cand_err < base_err - 2),
                    "cand_worse_than_base_by_2px": bool(cand_err > base_err + 2),
                    "cand_crop_radius_px": int(search_crop_size // 2),
                }
            )

        if top1_xy is None:
            continue

        score_ranks = np.argsort(np.argsort(scores))[::-1]
        shift_ranks = np.argsort(np.argsort(shifts))
        score_mean = float(np.mean(scores))
        top1_score = float(scores[0])
        top2_score = float(scores[1])
        score_probs = np.exp(np.asarray(scores, dtype=np.float64) - np.max(scores))
        score_probs = score_probs / max(1e-8, float(score_probs.sum()))
        score_entropy = float(-(score_probs * np.log(score_probs + 1e-8)).sum())

        for j, cand in enumerate(candidates):
            cand["cand_is_oracle"] = bool(j == oracle_idx)
            cand["cand_margin_to_top2"] = round(float(top1_score - top2_score) if j == 0 else round(float(top1_score - scores[j]), 4), 4)
            cand_xy = np.array(cand["cand_xy_norm"], dtype=np.float32) * np.array(
                [float(meta["image_W"]), float(meta["image_H"])], dtype=np.float32
            )
            cand["cand_dist_to_top1_px"] = round(float(np.linalg.norm(cand_xy - top1_xy)), 2)
            cand["score_rank_normalized"] = round(float(score_ranks[j]) / max(1, len(candidates) - 1), 3)
            cand["shift_rank_normalized"] = round(float(shift_ranks[j]) / max(1, len(candidates) - 1), 3)
            cand["candidate_score_minus_top1"] = round(float(cand["cand_score"] - top1_score), 4)
            cand["candidate_score_minus_mean"] = round(float(cand["cand_score"] - score_mean), 4)
            cand["candidate_error_minus_base"] = round(float(cand["cand_error_px"] - base_err), 2)

        oracle_better = bool(candidates[oracle_idx]["cand_better_by_2px"])
        label = int(oracle_idx) if oracle_better else int(topk)
        top1_err = float(candidates[0]["cand_error_px"])

        row = {
            "sample_id": int(meta["sample_id"]),
            "video_name": meta["video_name"],
            "point_idx": int(meta["point_idx"]),
            "t_query": int(meta["query_frame"]),
            "t_last_visible": int(meta["support_frames"][-1]) if meta.get("support_frames") else int(meta["query_frame"]),
            "t_reentry": int(meta["reentry_frame"]),
            "occ_length": int(meta["occ_length"]),
            "base_xy_norm": meta["base_xy_norm"],
            "gt_xy_norm": meta["gt_xy_norm"],
            "base_error_px": round(base_err, 2),
            "group_base16": bool(base_err > 16),
            "group_base32": bool(base_err > 32),
            "crop_radius_px": int(search_crop_size // 2),
            "gt_in_crop": True,
            "num_candidates": len(candidates),
            "oracle_index": int(oracle_idx),
            "oracle_error_px": round(oracle_err, 2),
            "top1_index_by_dino": 0,
            "top1_error_px": round(top1_err, 2),
            "oracle_better_by_2px": oracle_better,
            "top1_better_by_2px": bool(top1_err < base_err - 2),
            "label": label,
            "is_abstain": bool(label == topk),
            "has_positive_candidate": oracle_better,
            "top1_score": round(top1_score, 4),
            "top1_margin": round(float(top1_score - top2_score), 4),
            "top1_shift_px": round(float(shifts[0]), 2),
            "score_std_topk": round(float(np.std(scores)), 4),
            "score_range_topk": round(float(np.max(scores) - np.min(scores)), 4),
            "score_entropy_topk": round(score_entropy, 4),
            "peak_sharpness_topk": round(float(top1_score - np.mean(scores[1:])), 4),
            "shift_std_topk": round(float(np.std(shifts)), 2),
            "shift_range_topk": round(float(np.max(shifts) - np.min(shifts)), 2),
            "cand_error_min_topk": round(float(np.min(cand_errors)), 2),
            "candidates": candidates,
        }
        out_rows.append(row)

        vid = row["video_name"]
        per_vid.setdefault(vid, []).append(row)

    summary = {
        "split": split,
        "n_samples": len(out_rows),
        "n_sequences": len(per_vid),
        "topk": topk,
        "crop_radius": 64,
        "label_distribution": {str(k): sum(1 for r in out_rows if r["label"] == k) for k in range(topk + 1)},
        "abstain_ratio": round(sum(1 for r in out_rows if r["is_abstain"]) / max(1, len(out_rows)), 3),
        "oracle_better_by_2px_frac": round(sum(1 for r in out_rows if r["oracle_better_by_2px"]) / max(1, len(out_rows)), 3),
        "top1_better_by_2px_frac": round(sum(1 for r in out_rows if r["top1_better_by_2px"]) / max(1, len(out_rows)), 3),
        "base_median": round(float(np.median([r["base_error_px"] for r in out_rows])), 2) if out_rows else None,
        "oracle_median": round(float(np.median([r["oracle_error_px"] for r in out_rows])), 2) if out_rows else None,
        "top1_median": round(float(np.median([r["top1_error_px"] for r in out_rows])), 2) if out_rows else None,
        "base16": {
            "n": sum(1 for r in out_rows if r["group_base16"]),
            "oracle_better_by_2px_frac": round(sum(1 for r in out_rows if r["group_base16"] and r["oracle_better_by_2px"]) / max(1, sum(1 for r in out_rows if r["group_base16"])), 3),
        },
        "base32": {
            "n": sum(1 for r in out_rows if r["group_base32"]),
            "oracle_better_by_2px_frac": round(sum(1 for r in out_rows if r["group_base32"] and r["oracle_better_by_2px"]) / max(1, sum(1 for r in out_rows if r["group_base32"])), 3),
        },
    }
    return {"rows": out_rows, "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-root", type=str, default="outputs/recovery_anchor_dataset_v3_full")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    anchor_root = Path(args.anchor_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_summary = {}
    for split in ("train", "val"):
        pack = build_split(anchor_root, split=split, topk=args.topk)
        with open(out_dir / f"{split}.jsonl", "w") as f:
            for row in pack["rows"]:
                f.write(json.dumps(row) + "\n")
        merged_summary[split] = pack["summary"]

    with open(out_dir / "summary.json", "w") as f:
        json.dump(merged_summary, f, indent=2)

    print(json.dumps(merged_summary, indent=2))


if __name__ == "__main__":
    main()
