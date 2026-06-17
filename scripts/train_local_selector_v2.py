#!/usr/bin/env python3
"""
Train a two-stage local selector on base-centered local DINO candidates.

Stage-2b formulation:
  1. Event gate: predict whether this event has any candidate better than base by >2px
  2. Candidate ranker: on positive events only, rank top-k candidates and select the best

This avoids the failure mode of the tiny K+1 classifier, where the abstain class
dominates and swamps the rare positive candidate classes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
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

CANDIDATE_FEATURE_KEYS = [
    "cand_score",
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


def normalize_candidate_feature(key: str, value: float) -> float:
    if key in ("cand_shift_px_from_base", "cand_dist_to_base_px", "cand_dist_to_top1_px"):
        return value / 256.0
    return value


def extract_event_vector(sample: Dict) -> List[float]:
    return [normalize_event_feature(k, _safe_float(sample.get(k, 0.0))) for k in EVENT_FEATURE_KEYS]


def extract_candidate_vector(sample: Dict, cand: Dict) -> List[float]:
    vec = extract_event_vector(sample)
    vec.extend(normalize_candidate_feature(k, _safe_float(cand.get(k, 0.0))) for k in CANDIDATE_FEATURE_KEYS)
    vec.append(float(cand.get("rank_by_dino", 0)) / max(1.0, float(sample.get("num_candidates", 5) - 1)))
    return vec


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


class CandidateRanker(nn.Module):
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


def build_event_arrays(samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray([extract_event_vector(s) for s in samples], dtype=np.float32)
    y = np.asarray([1.0 if s.get("has_positive_candidate", False) else 0.0 for s in samples], dtype=np.float32)
    return X, y


def build_ranker_rows(samples: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats = []
    labels = []
    sample_ids = []
    for sample_idx, s in enumerate(samples):
        if not s.get("has_positive_candidate", False):
            continue
        oracle_idx = int(s.get("oracle_index", -1))
        for cand in s.get("candidates", []):
            feats.append(extract_candidate_vector(s, cand))
            labels.append(1.0 if int(cand.get("rank_by_dino", -1)) == oracle_idx else 0.0)
            sample_ids.append(sample_idx)
    if not feats:
        return np.zeros((0, len(EVENT_FEATURE_KEYS) + len(CANDIDATE_FEATURE_KEYS) + 1), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(feats, dtype=np.float32), np.asarray(labels, dtype=np.float32), np.asarray(sample_ids, dtype=np.int64)


def fit_norm(train_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0)
    sigma = train_x.std(axis=0) + 1e-6
    return mu, sigma


def apply_norm(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (x - mu) / sigma


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


def train_ranker(train_samples: List[Dict], val_samples: List[Dict], device: torch.device, epochs: int, lr: float, hidden: int):
    train_x, train_y, train_sid = build_ranker_rows(train_samples)
    val_x, val_y, val_sid = build_ranker_rows(val_samples)
    if len(train_x) == 0 or len(val_x) == 0:
        raise RuntimeError("Ranker dataset is empty; no positive events available.")

    mu, sigma = fit_norm(train_x)
    train_x = apply_norm(train_x, mu, sigma)
    val_x = apply_norm(val_x, mu, sigma)

    model = CandidateRanker(train_x.shape[1], hidden_dim=hidden).to(device)
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
                val_scores = model(torch.from_numpy(val_x).to(device)).cpu().numpy()
            correct = 0
            total = 0
            for sid in np.unique(val_sid):
                mask = val_sid == sid
                scores = val_scores[mask]
                labels = val_y[mask]
                if not len(scores):
                    continue
                pred = int(np.argmax(scores))
                gt = int(np.argmax(labels))
                correct += int(pred == gt)
                total += 1
            acc = float(correct / max(1, total))
            if acc > best_metric:
                best_metric = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"mu": mu.tolist(), "sigma": sigma.tolist(), "best_metric": best_metric, "train_positive_rate": pos_frac}


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
    ranker_model: CandidateRanker,
    ranker_norm: Dict,
    val_samples: List[Dict],
    device: torch.device,
    gate_threshold: float = 0.5,
    return_details: bool = False,
) -> Tuple[Dict, Optional[List[Dict]]]:
    gate_model.eval()
    ranker_model.eval()

    base_errs = np.asarray([_safe_float(s["base_error_px"]) for s in val_samples], dtype=np.float32)
    top1_errs = np.asarray([_safe_float(s["top1_error_px"]) for s in val_samples], dtype=np.float32)
    oracle_candidate_errs = np.asarray([_safe_float(s["oracle_error_px"]) for s in val_samples], dtype=np.float32)
    oracle_selective_errs = np.minimum(base_errs, np.where(oracle_candidate_errs < base_errs - 2.0, oracle_candidate_errs, base_errs))
    final_errs = []
    gate_probs = []
    gate_accepts = []
    ranker_correct = 0
    ranker_total = 0
    details = [] if return_details else None

    gate_mu = np.asarray(gate_norm["mu"], dtype=np.float32)
    gate_sigma = np.asarray(gate_norm["sigma"], dtype=np.float32)
    rank_mu = np.asarray(ranker_norm["mu"], dtype=np.float32)
    rank_sigma = np.asarray(ranker_norm["sigma"], dtype=np.float32)

    with torch.no_grad():
        for s in val_samples:
            event_x = np.asarray(extract_event_vector(s), dtype=np.float32)
            event_x = (event_x - gate_mu) / gate_sigma
            prob = torch.sigmoid(gate_model(torch.from_numpy(event_x).unsqueeze(0).to(device))).item()
            gate_probs.append(prob)
            sample_base_error = _safe_float(s["base_error_px"])
            sample_oracle_error = _safe_float(s["oracle_error_px"])
            sample_oracle_selective_error = sample_oracle_error if sample_oracle_error < sample_base_error - 2.0 else sample_base_error

            sample_detail = None
            if return_details:
                sample_detail = {
                    "sample_id": s.get("sample_id"),
                    "video_name": s.get("video_name"),
                    "point_idx": s.get("point_idx"),
                    "t_query": s.get("t_query"),
                    "t_last_visible": s.get("t_last_visible"),
                    "t_reentry": s.get("t_reentry"),
                    "occ_length": s.get("occ_length"),
                    "group_base16": bool(s.get("group_base16", False)),
                    "group_base32": bool(s.get("group_base32", False)),
                    "has_positive_candidate": bool(s.get("has_positive_candidate", False)),
                    "base_error_px": sample_base_error,
                    "top1_error_px": _safe_float(s["top1_error_px"]),
                    "oracle_candidate_error_px": sample_oracle_error,
                    "oracle_selective_error_px": sample_oracle_selective_error,
                    "gate_prob": round(float(prob), 6),
                    "gate_threshold": gate_threshold,
                }

            if prob < gate_threshold:
                final_errs.append(sample_base_error)
                gate_accepts.append(False)
                if return_details and sample_detail is not None:
                    sample_detail.update(
                        {
                            "gate_accept": False,
                            "chosen_source": "base",
                            "chosen_candidate_rank_by_dino": None,
                            "chosen_candidate_error_px": None,
                            "ranker_pred_is_oracle": False,
                            "final_error_px": sample_base_error,
                            "final_better_by_2px": False,
                            "ranker_scores": [],
                            "candidates": [],
                        }
                    )
                    details.append(sample_detail)
                continue

            cands = s.get("candidates", [])
            if not cands:
                final_errs.append(sample_base_error)
                gate_accepts.append(False)
                if return_details and sample_detail is not None:
                    sample_detail.update(
                        {
                            "gate_accept": False,
                            "chosen_source": "base_no_candidates",
                            "chosen_candidate_rank_by_dino": None,
                            "chosen_candidate_error_px": None,
                            "ranker_pred_is_oracle": False,
                            "final_error_px": sample_base_error,
                            "final_better_by_2px": False,
                            "ranker_scores": [],
                            "candidates": [],
                        }
                    )
                    details.append(sample_detail)
                continue

            rank_x = np.asarray([extract_candidate_vector(s, c) for c in cands], dtype=np.float32)
            rank_x = (rank_x - rank_mu) / rank_sigma
            scores = ranker_model(torch.from_numpy(rank_x).to(device)).cpu().numpy()
            pred_idx = int(np.argmax(scores))
            oracle_idx = int(s.get("oracle_index", -1))
            ranker_correct += int(pred_idx == oracle_idx and s.get("has_positive_candidate", False))
            ranker_total += int(s.get("has_positive_candidate", False))
            chosen_error = _safe_float(cands[pred_idx].get("cand_error_px", s["base_error_px"]))
            final_errs.append(chosen_error)
            gate_accepts.append(True)
            if return_details and sample_detail is not None:
                sample_detail.update(
                    {
                        "gate_accept": True,
                        "chosen_source": "candidate",
                        "chosen_candidate_rank_by_dino": int(cands[pred_idx].get("rank_by_dino", pred_idx)),
                        "chosen_candidate_error_px": chosen_error,
                        "ranker_pred_is_oracle": bool(pred_idx == oracle_idx),
                        "final_error_px": chosen_error,
                        "final_better_by_2px": bool(chosen_error < sample_base_error - 2.0),
                        "ranker_scores": [round(float(x), 6) for x in scores.tolist()],
                        "candidates": [
                            {
                                "rank_by_dino": int(c.get("rank_by_dino", idx)),
                                "cand_error_px": _safe_float(c.get("cand_error_px")),
                                "cand_score": _safe_float(c.get("cand_score")),
                                "cand_better_by_2px": bool(c.get("cand_better_by_2px", False)),
                                "ranker_score": round(float(scores[idx]), 6),
                                "is_oracle": bool(idx == oracle_idx),
                            }
                            for idx, c in enumerate(cands)
                        ],
                    }
                )
                details.append(sample_detail)

    final_errs = np.asarray(final_errs, dtype=np.float32)
    gate_probs = np.asarray(gate_probs, dtype=np.float32)
    gate_accepts = np.asarray(gate_accepts, dtype=bool)
    positive_mask = np.asarray([bool(s.get("has_positive_candidate", False)) for s in val_samples], dtype=bool)
    base16_mask = np.asarray([bool(s.get("group_base16", False)) for s in val_samples], dtype=bool)
    base32_mask = np.asarray([bool(s.get("group_base32", False)) for s in val_samples], dtype=bool)

    oracle_gain = float(np.median(base_errs) - np.median(oracle_selective_errs))
    final_gain = float(np.median(base_errs) - np.median(final_errs))
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
        "gate_positive_recall": round(float((gate_accepts[positive_mask]).mean()), 3) if positive_mask.any() else 0.0,
        "ranker_top1_accuracy_on_positive": round(float(ranker_correct / max(1, ranker_total)), 3),
        "oracle_gap_utilization_positive": round(final_gain / oracle_gain, 3) if oracle_gain > 0 else 0.0,
        "positive": _subset_metrics(
            positive_mask,
            base_errs,
            top1_errs,
            oracle_candidate_errs,
            oracle_selective_errs,
            final_errs,
            gate_accepts,
        ),
        "base16": _subset_metrics(
            base16_mask,
            base_errs,
            top1_errs,
            oracle_candidate_errs,
            oracle_selective_errs,
            final_errs,
            gate_accepts,
        ),
        "base32": _subset_metrics(
            base32_mask,
            base_errs,
            top1_errs,
            oracle_candidate_errs,
            oracle_selective_errs,
            final_errs,
            gate_accepts,
        ),
        "gate_prob_median": round(float(np.median(gate_probs)), 3),
        "overall_final_median_diff_px": round(float(np.median(final_errs) - np.median(base_errs)), 2),
        "overall_final_median_improvement_frac": round(float((np.median(base_errs) - np.median(final_errs)) / max(np.median(base_errs), 1e-6)), 3),
    }
    return results, details


def summarize_stage2b(results: Dict) -> Tuple[int, List[str]]:
    base16 = results.get("base16", {})
    passed = 0
    criteria = []

    c1 = base16.get("final_better_2px_frac", 0) >= 0.55
    passed += int(c1)
    criteria.append(f"base>16 final better_2px={base16.get('final_better_2px_frac', 0):.3f} >= 0.55: {'PASS' if c1 else 'FAIL'}")

    if base16.get("n", 0) > 0:
        imp = float(base16.get("final_median_improvement_frac", 0.0))
        c2 = imp >= 0.10
        passed += int(c2)
        criteria.append(f"base>16 median improvement={imp:.1%} >= 10%: {'PASS' if c2 else 'FAIL'}")
    else:
        criteria.append("base>16 median improvement: FAIL (no samples)")

    diff = float(results.get("overall_final_median_diff_px", 0.0))
    c3 = diff <= 1.0
    passed += int(c3)
    criteria.append(f"overall final median diff={diff:.2f}px <= 1px: {'PASS' if c3 else 'FAIL'}")
    return passed, criteria


def parse_thresholds(spec: str) -> List[float]:
    thresholds = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        thresholds.append(float(part))
    thresholds = sorted(set(round(x, 6) for x in thresholds if 0.0 <= x <= 1.0))
    if not thresholds:
        raise ValueError("No valid thresholds parsed.")
    return thresholds


def run_threshold_sweep(
    gate_model: EventGate,
    gate_norm: Dict,
    ranker_model: CandidateRanker,
    ranker_norm: Dict,
    val_samples: List[Dict],
    device: torch.device,
    thresholds: List[float],
) -> Dict:
    rows = []
    for th in thresholds:
        metrics, _ = evaluate_pipeline(
            gate_model,
            gate_norm,
            ranker_model,
            ranker_norm,
            val_samples,
            device,
            gate_threshold=th,
            return_details=False,
        )
        passed, criteria = summarize_stage2b(metrics)
        row = {
            "gate_threshold": round(float(th), 6),
            "criteria_passed": passed,
            "criteria": criteria,
            "overall_final_median_diff_px": metrics["overall_final_median_diff_px"],
            "overall_final_median": metrics["final_median"],
            "overall_base_median": metrics["base_median"],
            "overall_gate_accept_rate": metrics["gate_accept_rate"],
            "base16_n": metrics["base16"]["n"],
            "base16_final_better_2px_frac": metrics["base16"].get("final_better_2px_frac", 0.0),
            "base16_final_median": metrics["base16"].get("final_median"),
            "base16_base_median": metrics["base16"].get("base_median"),
            "base16_final_median_improvement_frac": metrics["base16"].get("final_median_improvement_frac", 0.0),
            "base32_final_better_2px_frac": metrics["base32"].get("final_better_2px_frac", 0.0),
            "positive_gate_accept_rate": metrics["positive"].get("gate_accept_rate", 0.0),
            "gate_positive_recall": metrics["gate_positive_recall"],
            "ranker_top1_accuracy_on_positive": metrics["ranker_top1_accuracy_on_positive"],
        }
        rows.append(row)

    def row_key(row: Dict) -> Tuple:
        return (
            row["criteria_passed"],
            row["base16_final_better_2px_frac"],
            row["base16_final_median_improvement_frac"],
            -abs(row["overall_final_median_diff_px"]),
            -row["overall_gate_accept_rate"],
        )

    recommended = max(rows, key=row_key)
    return {
        "thresholds": rows,
        "recommended_threshold": recommended["gate_threshold"],
        "recommended_row": recommended,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--sweep-thresholds", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--export-per-sample", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.dataset_dir) / "train.jsonl"
    val_path = Path(args.dataset_dir) / "val.jsonl"
    train_samples = load_samples(train_path)
    val_samples = load_samples(val_path)
    sweep_thresholds = parse_thresholds(args.sweep_thresholds)

    gate_model, gate_norm = train_gate(train_samples, val_samples, device, epochs=args.epochs, lr=args.lr, hidden=args.hidden)
    ranker_model, ranker_norm = train_ranker(train_samples, val_samples, device, epochs=args.epochs, lr=args.lr, hidden=args.hidden)
    results, details = evaluate_pipeline(
        gate_model,
        gate_norm,
        ranker_model,
        ranker_norm,
        val_samples,
        device,
        gate_threshold=args.gate_threshold,
        return_details=args.export_per_sample,
    )
    sweep = run_threshold_sweep(
        gate_model,
        gate_norm,
        ranker_model,
        ranker_norm,
        val_samples,
        device,
        thresholds=sweep_thresholds,
    )
    passed, criteria = summarize_stage2b(results)
    recommended_threshold = float(sweep["recommended_threshold"])
    recommended_results = None
    recommended_details = None
    if abs(recommended_threshold - float(args.gate_threshold)) > 1e-9:
        recommended_results, recommended_details = evaluate_pipeline(
            gate_model,
            gate_norm,
            ranker_model,
            ranker_norm,
            val_samples,
            device,
            gate_threshold=recommended_threshold,
            return_details=args.export_per_sample,
        )

    if args.export_per_sample and details is not None:
        with open(out_dir / f"per_sample_gate_{args.gate_threshold:.2f}.jsonl", "w") as f:
            for row in details:
                f.write(json.dumps(row) + "\n")
    if args.export_per_sample and recommended_details is not None:
        with open(out_dir / f"per_sample_gate_{recommended_threshold:.2f}.jsonl", "w") as f:
            for row in recommended_details:
                f.write(json.dumps(row) + "\n")
    with open(out_dir / "threshold_sweep.json", "w") as f:
        json.dump(sweep, f, indent=2)

    payload = {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "output_dir": str(out_dir.resolve()),
        "train_n": len(train_samples),
        "val_n": len(val_samples),
        "gate_norm": gate_norm,
        "ranker_norm": ranker_norm,
        "results": results,
        "recommended_results": recommended_results,
        "threshold_sweep": sweep,
        "stage2b_verdict": "PASS" if passed >= 2 else "FAIL",
        "stage2b_criteria_passed": passed,
        "stage2b_criteria_details": criteria,
        "config": {
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden": args.hidden,
            "gate_threshold": args.gate_threshold,
            "sweep_thresholds": sweep_thresholds,
            "export_per_sample": args.export_per_sample,
        },
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
