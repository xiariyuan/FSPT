#!/usr/bin/env python3
"""
Train a gated local selector with a vector-based positive-only ranker.

Difference from train_local_selector_v2.py:
  - Gate stays event-level and unchanged in spirit.
  - Ranker no longer relies on scalar DINO score / tabular features only.
  - Each candidate is represented by its full 384D DINO vector from the
    precomputed search feature map, combined with support_descriptor and a
    small amount of geometry metadata.

Goal:
  Determine whether full candidate vectors contain discriminative signal that
  scalar cosine scores miss.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EVENT_FEATURE_KEYS = [
    "occ_length",
    "base_error_px",
    "top1_score",
    "top1_margin",
    "top1_shift_px",
    "score_std_topk",
    "score_range_topk",
    "score_entropy_topk",
    "peak_sharpness_topk",
    "shift_std_topk",
    "shift_range_topk",
    "cand_error_min_topk",
    "num_candidates",
]

GEOM_FEATURE_KEYS = [
    "cand_margin_to_top2",
    "cand_shift_px_from_base",
    "cand_shift_norm_from_base",
    "cand_dist_to_base_px",
    "cand_dist_to_top1_px",
    "score_rank_normalized",
    "shift_rank_normalized",
    "candidate_score_minus_top1",
    "candidate_score_minus_mean",
]


def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_samples(jsonl_path: Path) -> List[Dict]:
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_event_feature(key: str, value: float) -> float:
    if key == "occ_length":
        return value / 300.0
    if key in ("base_error_px", "top1_shift_px", "shift_std_topk", "shift_range_topk", "cand_error_min_topk"):
        return value / 256.0
    if key == "num_candidates":
        return value / 5.0
    return value


def normalize_geom_feature(key: str, value: float) -> float:
    if key in ("cand_shift_px_from_base", "cand_dist_to_base_px", "cand_dist_to_top1_px"):
        return value / 256.0
    return value


def extract_event_vector(sample: Dict) -> List[float]:
    return [normalize_event_feature(k, _safe_float(sample.get(k, 0.0))) for k in EVENT_FEATURE_KEYS]


def extract_geom_vector(sample: Dict, cand: Dict) -> List[float]:
    vec = [normalize_geom_feature(k, _safe_float(cand.get(k, 0.0))) for k in GEOM_FEATURE_KEYS]
    vec.append(float(cand.get("rank_by_dino", 0)) / max(1.0, float(sample.get("num_candidates", 5) - 1)))
    vec.append(normalize_event_feature("occ_length", _safe_float(sample.get("occ_length", 0.0))))
    vec.append(normalize_event_feature("base_error_px", _safe_float(sample.get("base_error_px", 0.0))))
    vec.append(normalize_event_feature("top1_shift_px", _safe_float(sample.get("top1_shift_px", 0.0))))
    return vec


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def normalize_feat_map(feat_map: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(feat_map, axis=0, keepdims=True)
    norm = np.clip(norm, 1e-8, None)
    return (feat_map / norm).astype(np.float32)


def topk_candidate_vectors(
    support_descriptor: np.ndarray,
    search_feature_map: np.ndarray,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    supp = normalize_vec(support_descriptor.astype(np.float32))
    fmap = normalize_feat_map(search_feature_map.astype(np.float32))
    scores = (fmap * supp[:, None, None]).sum(axis=0)
    flat_idx = np.argsort(scores.reshape(-1))[::-1][:topk]
    h, w = scores.shape
    cand_vecs = []
    cand_scores = []
    for idx in flat_idx:
        fy, fx = divmod(int(idx), w)
        cand_vecs.append(fmap[:, fy, fx].astype(np.float32))
        cand_scores.append(float(scores[fy, fx]))
    return np.stack(cand_vecs, axis=0), np.asarray(cand_scores, dtype=np.float32)


class EventGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class VectorRanker(nn.Module):
    def __init__(self, geom_dim: int, hidden_dim: int = 64):
        super().__init__()
        input_dim = 384 * 2 + geom_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, support_vec: torch.Tensor, cand_vecs: torch.Tensor, geom_vecs: torch.Tensor) -> torch.Tensor:
        # support_vec: (B, 384), cand_vecs: (B, K, 384), geom_vecs: (B, K, G)
        supp = support_vec.unsqueeze(1).expand_as(cand_vecs)
        x = torch.cat([cand_vecs * supp, cand_vecs - supp, geom_vecs], dim=-1)
        return self.net(x).squeeze(-1)


def fit_norm(train_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0)
    sigma = train_x.std(axis=0) + 1e-6
    return mu.astype(np.float32), sigma.astype(np.float32)


def apply_norm(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return ((x - mu) / sigma).astype(np.float32)


def build_event_arrays(samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray([extract_event_vector(s) for s in samples], dtype=np.float32)
    y = np.asarray([1.0 if s.get("has_positive_candidate", False) else 0.0 for s in samples], dtype=np.float32)
    return X, y


def train_gate(train_samples: List[Dict], val_samples: List[Dict], device: torch.device, epochs: int, lr: float, hidden: int):
    train_x, train_y = build_event_arrays(train_samples)
    val_x, val_y = build_event_arrays(val_samples)
    mu, sigma = fit_norm(train_x)
    train_x = apply_norm(train_x, mu, sigma)
    val_x = apply_norm(val_x, mu, sigma)

    model = EventGate(train_x.shape[1], hidden_dim=hidden).to(device)
    pos_frac = float(train_y.mean()) if len(train_y) else 0.0
    pos_weight = torch.tensor([(1.0 - pos_frac) / max(pos_frac, 1e-6)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = optim.Adam(model.parameters(), lr=lr)

    Xt = torch.from_numpy(train_x).to(device)
    yt = torch.from_numpy(train_y).to(device)

    best_state = None
    best_metric = -1.0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = criterion(model(Xt), yt)
        loss.backward()
        opt.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_probs = torch.sigmoid(model(torch.from_numpy(val_x).to(device))).cpu().numpy()
            val_preds = val_probs >= 0.5
            recall = float(((val_preds == 1) & (val_y == 1)).sum() / max(1, (val_y == 1).sum()))
            precision = float(((val_preds == 1) & (val_y == 1)).sum() / max(1, (val_preds == 1).sum()))
            metric = 0.7 * recall + 0.3 * precision
            if metric > best_metric:
                best_metric = metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"mu": mu.tolist(), "sigma": sigma.tolist(), "best_metric": best_metric, "train_positive_rate": pos_frac}


def load_anchor_map(anchor_root: Path, split: str) -> Dict[int, Path]:
    index_path = anchor_root / f"index_{split}.json"
    metas = json.load(open(index_path))
    return {int(meta["sample_id"]): anchor_root / meta["file"] for meta in metas}


def build_positive_ranker_tensors(
    samples: List[Dict],
    anchor_map: Dict[int, Path],
) -> Dict[str, np.ndarray]:
    supps = []
    cand_vecs = []
    geom_vecs = []
    oracle_idx = []
    sample_ids = []

    for s in samples:
        if not s.get("has_positive_candidate", False):
            continue
        sid = int(s["sample_id"])
        path = anchor_map.get(sid)
        if path is None or not path.exists():
            continue

        data = np.load(path, allow_pickle=False)
        support = normalize_vec(data["support_descriptor"].astype(np.float32))
        cands_vec, cands_scores = topk_candidate_vectors(
            support_descriptor=data["support_descriptor"].astype(np.float32),
            search_feature_map=data["search_feature_map"].astype(np.float32),
            topk=int(s.get("num_candidates", 5)),
        )

        stored_cands = s.get("candidates", [])
        if len(stored_cands) != len(cands_vec):
            continue

        # Sanity-check rank ordering against stored scalar scores.
        score_ok = True
        for j, cand in enumerate(stored_cands):
            if abs(float(cands_scores[j]) - _safe_float(cand.get("cand_score", 0.0))) > 5e-3:
                score_ok = False
                break
        if not score_ok:
            continue

        supps.append(support.astype(np.float32))
        cand_vecs.append(cands_vec.astype(np.float32))
        geom_vecs.append(np.asarray([extract_geom_vector(s, cand) for cand in stored_cands], dtype=np.float32))
        oracle_idx.append(int(s.get("oracle_index", 0)))
        sample_ids.append(sid)

    if not supps:
        raise RuntimeError("No positive ranker samples could be built from anchor dataset.")

    return {
        "support_vec": np.stack(supps, axis=0),
        "cand_vecs": np.stack(cand_vecs, axis=0),
        "geom_vecs": np.stack(geom_vecs, axis=0),
        "oracle_idx": np.asarray(oracle_idx, dtype=np.int64),
        "sample_id": np.asarray(sample_ids, dtype=np.int64),
    }


def train_vector_ranker(
    train_pack: Dict[str, np.ndarray],
    val_pack: Dict[str, np.ndarray],
    device: torch.device,
    epochs: int,
    lr: float,
    hidden: int,
):
    geom_mu, geom_sigma = fit_norm(train_pack["geom_vecs"].reshape(-1, train_pack["geom_vecs"].shape[-1]))
    train_geom = apply_norm(train_pack["geom_vecs"], geom_mu, geom_sigma)
    val_geom = apply_norm(val_pack["geom_vecs"], geom_mu, geom_sigma)

    model = VectorRanker(geom_dim=train_geom.shape[-1], hidden_dim=hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    x_s_train = torch.from_numpy(train_pack["support_vec"]).to(device)
    x_c_train = torch.from_numpy(train_pack["cand_vecs"]).to(device)
    x_g_train = torch.from_numpy(train_geom).to(device)
    y_train = torch.from_numpy(train_pack["oracle_idx"]).to(device)

    x_s_val = torch.from_numpy(val_pack["support_vec"]).to(device)
    x_c_val = torch.from_numpy(val_pack["cand_vecs"]).to(device)
    x_g_val = torch.from_numpy(val_geom).to(device)
    y_val = torch.from_numpy(val_pack["oracle_idx"]).to(device)

    best_state = None
    best_metrics = None
    best_key = None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        train_scores = model(x_s_train, x_c_train, x_g_train)
        loss = F.cross_entropy(train_scores, y_train)
        loss.backward()
        opt.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_scores = model(x_s_val, x_c_val, x_g_val)
                val_pred = torch.argmax(val_scores, dim=1)
                acc = float((val_pred == y_val).float().mean().item())
                ce = float(F.cross_entropy(val_scores, y_val).item())
                key = (acc, -ce)
                if best_key is None or key > best_key:
                    best_key = key
                    best_metrics = {"best_acc": acc, "best_ce": ce}
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "geom_mu": geom_mu.tolist(),
        "geom_sigma": geom_sigma.tolist(),
        **(best_metrics or {"best_acc": 0.0, "best_ce": None}),
    }


def _subset_metrics(
    mask: np.ndarray,
    base_errs: np.ndarray,
    top1_errs: np.ndarray,
    oracle_candidate_errs: np.ndarray,
    oracle_selective_errs: np.ndarray,
    final_errs: np.ndarray,
    gate_accepts: np.ndarray,
) -> Dict:
    if mask.sum() == 0:
        return {"n": 0}

    be = base_errs[mask]
    te = top1_errs[mask]
    oce = oracle_candidate_errs[mask]
    ose = oracle_selective_errs[mask]
    fe = final_errs[mask]

    base_median = float(np.median(be))
    final_median = float(np.median(fe))
    return {
        "n": int(mask.sum()),
        "base_median": round(base_median, 2),
        "top1_median": round(float(np.median(te)), 2),
        "oracle_candidate_median": round(float(np.median(oce)), 2),
        "oracle_selective_median": round(float(np.median(ose)), 2),
        "final_median": round(final_median, 2),
        "final_better_2px_frac": round(float((fe < be - 2).mean()), 3),
        "oracle_better_2px_frac": round(float((oce < be - 2).mean()), 3),
        "gate_accept_rate": round(float(gate_accepts[mask].mean()), 3),
        "final_median_improvement_frac": round(float((base_median - final_median) / max(base_median, 1e-6)), 3),
    }


def evaluate_pipeline(
    gate_model: EventGate,
    gate_norm: Dict,
    ranker_model: VectorRanker,
    ranker_norm: Dict,
    val_samples: List[Dict],
    val_anchor_map: Dict[int, Path],
    device: torch.device,
    gate_threshold: float,
) -> Tuple[Dict, List[Dict]]:
    gate_model.eval()
    ranker_model.eval()

    base_errs = np.asarray([_safe_float(s["base_error_px"]) for s in val_samples], dtype=np.float32)
    top1_errs = np.asarray([_safe_float(s["top1_error_px"]) for s in val_samples], dtype=np.float32)
    oracle_candidate_errs = np.asarray([_safe_float(s["oracle_error_px"]) for s in val_samples], dtype=np.float32)
    oracle_selective_errs = np.minimum(base_errs, np.where(oracle_candidate_errs < base_errs - 2.0, oracle_candidate_errs, base_errs))

    gate_mu = np.asarray(gate_norm["mu"], dtype=np.float32)
    gate_sigma = np.asarray(gate_norm["sigma"], dtype=np.float32)
    geom_mu = np.asarray(ranker_norm["geom_mu"], dtype=np.float32)
    geom_sigma = np.asarray(ranker_norm["geom_sigma"], dtype=np.float32)

    final_errs = []
    gate_probs = []
    gate_accepts = []
    ranker_correct = 0
    ranker_total = 0
    details = []

    with torch.no_grad():
        for s in val_samples:
            event_x = np.asarray(extract_event_vector(s), dtype=np.float32)
            event_x = (event_x - gate_mu) / gate_sigma
            prob = torch.sigmoid(gate_model(torch.from_numpy(event_x).unsqueeze(0).to(device))).item()
            gate_probs.append(prob)

            base_err = _safe_float(s["base_error_px"])
            oracle_err = _safe_float(s["oracle_error_px"])
            sample_detail = {
                "sample_id": s.get("sample_id"),
                "video_name": s.get("video_name"),
                "point_idx": s.get("point_idx"),
                "group_base16": bool(s.get("group_base16", False)),
                "group_base32": bool(s.get("group_base32", False)),
                "has_positive_candidate": bool(s.get("has_positive_candidate", False)),
                "base_error_px": base_err,
                "oracle_candidate_error_px": oracle_err,
                "gate_prob": round(float(prob), 6),
                "gate_threshold": gate_threshold,
            }

            if prob < gate_threshold:
                final_errs.append(base_err)
                gate_accepts.append(False)
                sample_detail.update(
                    {
                        "gate_accept": False,
                        "chosen_source": "base",
                        "chosen_candidate_rank_by_dino": None,
                        "chosen_candidate_error_px": None,
                        "ranker_pred_is_oracle": False,
                        "final_error_px": base_err,
                        "final_better_by_2px": False,
                    }
                )
                details.append(sample_detail)
                continue

            path = val_anchor_map.get(int(s["sample_id"]))
            cands = s.get("candidates", [])
            if path is None or not path.exists() or not cands:
                final_errs.append(base_err)
                gate_accepts.append(False)
                sample_detail.update(
                    {
                        "gate_accept": False,
                        "chosen_source": "base_missing_anchor",
                        "chosen_candidate_rank_by_dino": None,
                        "chosen_candidate_error_px": None,
                        "ranker_pred_is_oracle": False,
                        "final_error_px": base_err,
                        "final_better_by_2px": False,
                    }
                )
                details.append(sample_detail)
                continue

            data = np.load(path, allow_pickle=False)
            support = normalize_vec(data["support_descriptor"].astype(np.float32))
            cand_vecs, cand_scores = topk_candidate_vectors(
                support_descriptor=data["support_descriptor"].astype(np.float32),
                search_feature_map=data["search_feature_map"].astype(np.float32),
                topk=len(cands),
            )
            geom = np.asarray([extract_geom_vector(s, cand) for cand in cands], dtype=np.float32)
            geom = apply_norm(geom, geom_mu, geom_sigma)
            scores = ranker_model(
                torch.from_numpy(support).unsqueeze(0).to(device),
                torch.from_numpy(cand_vecs).unsqueeze(0).to(device),
                torch.from_numpy(geom).unsqueeze(0).to(device),
            ).cpu().numpy()[0]

            pred_idx = int(np.argmax(scores))
            oracle_idx = int(s.get("oracle_index", -1))
            if s.get("has_positive_candidate", False):
                ranker_correct += int(pred_idx == oracle_idx)
                ranker_total += 1

            chosen_err = _safe_float(cands[pred_idx].get("cand_error_px", base_err))
            final_errs.append(chosen_err)
            gate_accepts.append(True)
            sample_detail.update(
                {
                    "gate_accept": True,
                    "chosen_source": "candidate",
                    "chosen_candidate_rank_by_dino": int(cands[pred_idx].get("rank_by_dino", pred_idx)),
                    "chosen_candidate_error_px": chosen_err,
                    "ranker_pred_is_oracle": bool(pred_idx == oracle_idx),
                    "final_error_px": chosen_err,
                    "final_better_by_2px": bool(chosen_err < base_err - 2.0),
                    "ranker_scores": [round(float(x), 6) for x in scores.tolist()],
                    "candidate_scalar_scores": [round(float(x), 6) for x in cand_scores.tolist()],
                }
            )
            details.append(sample_detail)

    final_errs = np.asarray(final_errs, dtype=np.float32)
    gate_accepts = np.asarray(gate_accepts, dtype=bool)
    positive_mask = np.asarray([bool(s.get("has_positive_candidate", False)) for s in val_samples], dtype=bool)
    base16_mask = np.asarray([bool(s.get("group_base16", False)) for s in val_samples], dtype=bool)
    base32_mask = np.asarray([bool(s.get("group_base32", False)) for s in val_samples], dtype=bool)

    results = {
        "n": int(len(val_samples)),
        "base_median": round(float(np.median(base_errs)), 2),
        "top1_median": round(float(np.median(top1_errs)), 2),
        "oracle_candidate_median": round(float(np.median(oracle_candidate_errs)), 2),
        "oracle_selective_median": round(float(np.median(oracle_selective_errs)), 2),
        "final_median": round(float(np.median(final_errs)), 2),
        "final_better_2px_frac": round(float((final_errs < base_errs - 2).mean()), 3),
        "oracle_better_2px_frac": round(float((oracle_candidate_errs < base_errs - 2).mean()), 3),
        "gate_accept_rate": round(float(gate_accepts.mean()), 3),
        "gate_positive_recall": round(float(gate_accepts[positive_mask].mean()), 3) if positive_mask.any() else 0.0,
        "ranker_top1_accuracy_on_positive": round(float(ranker_correct / max(1, ranker_total)), 3),
        "positive": _subset_metrics(
            positive_mask, base_errs, top1_errs, oracle_candidate_errs, oracle_selective_errs, final_errs, gate_accepts
        ),
        "base16": _subset_metrics(
            base16_mask, base_errs, top1_errs, oracle_candidate_errs, oracle_selective_errs, final_errs, gate_accepts
        ),
        "base32": _subset_metrics(
            base32_mask, base_errs, top1_errs, oracle_candidate_errs, oracle_selective_errs, final_errs, gate_accepts
        ),
        "overall_final_median_diff_px": round(float(np.median(final_errs) - np.median(base_errs)), 2),
    }
    return results, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--anchor-root", type=str, default="outputs/recovery_anchor_dataset_v3_full")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_samples(Path(args.dataset_dir) / "train.jsonl")
    val_samples = load_samples(Path(args.dataset_dir) / "val.jsonl")
    anchor_root = Path(args.anchor_root)
    train_anchor_map = load_anchor_map(anchor_root, "train")
    val_anchor_map = load_anchor_map(anchor_root, "val")

    gate_model, gate_norm = train_gate(
        train_samples=train_samples,
        val_samples=val_samples,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
    )

    train_pack = build_positive_ranker_tensors(train_samples, train_anchor_map)
    val_pack = build_positive_ranker_tensors(val_samples, val_anchor_map)
    ranker_model, ranker_norm = train_vector_ranker(
        train_pack=train_pack,
        val_pack=val_pack,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
    )

    results, details = evaluate_pipeline(
        gate_model=gate_model,
        gate_norm=gate_norm,
        ranker_model=ranker_model,
        ranker_norm=ranker_norm,
        val_samples=val_samples,
        val_anchor_map=val_anchor_map,
        device=device,
        gate_threshold=args.gate_threshold,
    )

    payload = {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "anchor_root": str(anchor_root.resolve()),
        "output_dir": str(out_dir.resolve()),
        "seed": args.seed,
        "train_n": len(train_samples),
        "val_n": len(val_samples),
        "train_positive_n": int(sum(1 for s in train_samples if s.get("has_positive_candidate", False))),
        "val_positive_n": int(sum(1 for s in val_samples if s.get("has_positive_candidate", False))),
        "gate_norm": gate_norm,
        "ranker_norm": {k: v for k, v in ranker_norm.items() if k not in ("geom_mu", "geom_sigma")},
        "results": results,
        "config": {
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden": args.hidden,
            "gate_threshold": args.gate_threshold,
        },
        "reference_stage2c": {
            "positive_final_median": 20.33,
            "base16_final_median": 22.87,
            "base16_final_better_2px_frac": 0.526,
            "overall_final_median_diff_px": -0.09,
        },
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(out_dir / "per_sample_predictions.jsonl", "w") as f:
        for row in details:
            f.write(json.dumps(row) + "\n")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
