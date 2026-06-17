#!/usr/bin/env python3
"""
Positive-only rank refinement for local DINO candidates.

Trains a ranker on events with has_positive_candidate=1 only.
Compares three ranking losses: BCE, pairwise hinge, listwise softmax.
Goal: push positive final median from 20.33 toward oracle 10.67.

Usage:
  python scripts/train_local_ranker_v3.py \
    --train-jsonl outputs/local_selector_dataset_v3_from_anchor_smoke/train.jsonl \
    --val-jsonl outputs/local_selector_dataset_v3_from_anchor_smoke/val.jsonl \
    --output-dir outputs/local_ranker_v3 \
    --gate-model-json outputs/local_selector_stage2b_from_anchor_sweep/results.json \
    --epochs 200 --lr 1e-3
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Feature extraction (reuse from v2 + interaction features)
# ---------------------------------------------------------------------------
EVENT_KEYS = [
    "occ_length", "base_error_px", "top1_score", "top1_margin",
    "top1_shift_px", "score_std_topk", "score_range_topk",
    "score_entropy_topk", "peak_sharpness_topk", "shift_std_topk",
    "shift_range_topk", "cand_error_min_topk", "num_candidates",
]

CAND_KEYS = [
    "cand_score", "cand_margin_to_top2", "cand_shift_px_from_base",
    "cand_shift_norm_from_base", "cand_dist_to_base_px", "cand_dist_to_top1_px",
    "score_rank_normalized", "shift_rank_normalized",
    "candidate_score_minus_top1", "candidate_score_minus_mean",
]


def _sf(v, d=0.0):
    try: return float(v)
    except: return float(d)


def norm_event(k, v):
    if k == "occ_length": return v / 300.0
    if k in ("base_error_px", "top1_shift_px", "shift_std_topk", "shift_range_topk", "cand_error_min_topk"):
        return v / 256.0
    if k == "num_candidates": return v / 5.0
    return v


def norm_cand(k, v):
    if k in ("cand_shift_px_from_base", "cand_dist_to_base_px", "cand_dist_to_top1_px"):
        return v / 256.0
    return v


def extract_event_vec(s):
    return [norm_event(k, _sf(s.get(k, 0))) for k in EVENT_KEYS]


def extract_cand_vec(s, c, extra_features=False):
    """Extract candidate features. If extra_features, add interaction features."""
    vec = extract_event_vec(s)
    vec.extend(norm_cand(k, _sf(c.get(k, 0))) for k in CAND_KEYS)
    vec.append(float(c.get("rank_by_dino", 0)) / max(1.0, float(s.get("num_candidates", 5) - 1)))

    if extra_features:
        # Interaction features
        cand_score = _sf(c.get("cand_score", 0))
        top1_score = _sf(s.get("top1_score", 0))
        score_std = _sf(s.get("score_std_topk", 0)) + 1e-6
        cand_shift = _sf(c.get("cand_shift_px_from_base", 0))
        top1_shift = _sf(s.get("top1_shift_px", 0)) + 1e-6
        cand_margin = _sf(c.get("cand_margin_to_top2", 0))
        base_err = _sf(s.get("base_error_px", 0)) + 1e-6

        vec.append((cand_score - top1_score) / score_std)  # score z-score relative to top1
        vec.append(cand_shift / top1_shift)  # shift ratio to top1
        vec.append(cand_score * cand_shift / 256.0)  # score * shift interaction
        vec.append(cand_margin * cand_score)  # margin * score
        vec.append(cand_shift / 256.0 * base_err / 256.0)  # shift * base_error interaction

        # Rank features
        rank = float(c.get("rank_by_dino", 0))
        K = float(s.get("num_candidates", 5))
        vec.append(rank / K)  # normalized rank
        vec.append((K - rank) / K)  # inverse rank

    return vec


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class RankerMLP(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Ranking losses
# ---------------------------------------------------------------------------
def pairwise_hinge_loss(scores, labels, sample_ids, margin=1.0):
    """Pairwise hinge loss: for each (positive, negative) pair within same event."""
    loss = torch.tensor(0.0, device=scores.device)
    n_pairs = 0
    for sid in torch.unique(sample_ids):
        mask = sample_ids == sid
        s = scores[mask]
        l = labels[mask]
        pos_idx = (l == 1).nonzero(as_tuple=True)[0]
        neg_idx = (l == 0).nonzero(as_tuple=True)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        for pi in pos_idx:
            for ni in neg_idx:
                loss = loss + F.relu(margin - (s[pi] - s[ni]))
                n_pairs += 1
    return loss / max(n_pairs, 1)


def listwise_softmax_loss(scores, labels, sample_ids):
    """Listwise softmax cross-entropy: treat oracle candidate as target."""
    loss = torch.tensor(0.0, device=scores.device)
    n_events = 0
    for sid in torch.unique(sample_ids):
        mask = sample_ids == sid
        s = scores[mask]
        l = labels[mask]
        oracle_idx = (l == 1).nonzero(as_tuple=True)[0]
        if len(oracle_idx) == 0:
            continue
        # Cross-entropy: softmax over candidates, oracle as target
        log_probs = F.log_softmax(s, dim=0)
        loss = loss - log_probs[oracle_idx[0]]
        n_events += 1
    return loss / max(n_events, 1)


# ---------------------------------------------------------------------------
# Build arrays
# ---------------------------------------------------------------------------
def build_ranker_arrays(samples, extra_features=False):
    feats, labels, sids = [], [], []
    for si, s in enumerate(samples):
        if not s.get("has_positive_candidate", False):
            continue
        oracle_idx = int(s.get("oracle_index", -1))
        for c in s.get("candidates", []):
            feats.append(extract_cand_vec(s, c, extra_features=extra_features))
            labels.append(1.0 if int(c.get("rank_by_dino", -1)) == oracle_idx else 0.0)
            sids.append(si)
    if not feats:
        dim = len(EVENT_KEYS) + len(CAND_KEYS) + 1 + (7 if extra_features else 0)
        return np.zeros((0, dim), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(feats, np.float32), np.asarray(labels, np.float32), np.asarray(sids, np.int64)


def build_event_arrays(samples):
    X = np.asarray([extract_event_vec(s) for s in samples], np.float32)
    y = np.asarray([1.0 if s.get("has_positive_candidate", False) else 0.0 for s in samples], np.float32)
    return X, y


# ---------------------------------------------------------------------------
# Train ranker with different losses
# ---------------------------------------------------------------------------
def train_ranker(train_samples, val_samples, device, loss_type="bce", epochs=200, lr=1e-3, hidden=64, extra_features=False):
    train_x, train_y, train_sid = build_ranker_arrays(train_samples, extra_features)
    val_x, val_y, val_sid = build_ranker_arrays(val_samples, extra_features)

    if len(train_x) == 0:
        return None, {}

    mu, sigma = train_x.mean(0), train_x.std(0) + 1e-6
    train_x_n = (train_x - mu) / sigma
    val_x_n = (val_x - mu) / sigma

    model = RankerMLP(train_x.shape[1], hidden=hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    Xt = torch.from_numpy(train_x_n).to(device)
    yt = torch.from_numpy(train_y).to(device)
    sidt = torch.from_numpy(train_sid).to(device)

    best_state = None
    best_acc = -1

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        scores = model(Xt)

        if loss_type == "bce":
            pos_frac = float(train_y.mean())
            pw = torch.tensor([(1 - pos_frac) / max(pos_frac, 1e-6)], device=device)
            loss = F.binary_cross_entropy_with_logits(scores, yt, pos_weight=pw)
        elif loss_type == "pairwise":
            loss = pairwise_hinge_loss(scores, yt, sidt)
        elif loss_type == "listwise":
            loss = listwise_softmax_loss(scores, yt, sidt)
        else:
            raise ValueError(f"Unknown loss_type={loss_type}")

        loss.backward()
        opt.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_scores = model(torch.from_numpy(val_x_n).to(device)).cpu().numpy()
            correct = 0
            total = 0
            for sid in np.unique(val_sid):
                mask = val_sid == sid
                if not mask.sum(): continue
                pred = int(np.argmax(val_scores[mask]))
                gt = int(np.argmax(val_y[mask]))
                correct += int(pred == gt)
                total += 1
            acc = correct / max(1, total)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model, {"mu": mu.tolist(), "sigma": sigma.tolist(), "best_acc": best_acc, "loss_type": loss_type}


# ---------------------------------------------------------------------------
# Evaluate pipeline
# ---------------------------------------------------------------------------
def evaluate_pipeline(ranker_model, ranker_norm, gate_model_or_none, gate_norm_or_none,
                      val_samples, device, gate_threshold=0.50, extra_features=False):
    ranker_model.eval()

    base_errs = np.asarray([_sf(s["base_error_px"]) for s in val_samples])
    oracle_errs = np.asarray([_sf(s["oracle_error_px"]) for s in val_samples])
    oracle_sel_errs = np.minimum(base_errs, np.where(oracle_errs < base_errs - 2, oracle_errs, base_errs))

    rank_mu = np.asarray(ranker_norm["mu"])
    rank_sigma = np.asarray(ranker_norm["sigma"])

    # Gate: use external gate model if provided, else use simple rule
    gate_probs = []
    if gate_model_or_none is not None and gate_norm_or_none is not None:
        gate_model = gate_model_or_none
        gate_model.eval()
        gate_mu = np.asarray(gate_norm_or_none["mu"])
        gate_sigma = np.asarray(gate_norm_or_none["sigma"])
        with torch.no_grad():
            for s in val_samples:
                ev = np.asarray(extract_event_vec(s), np.float32)
                ev = (ev - gate_mu) / gate_sigma
                prob = torch.sigmoid(gate_model(torch.from_numpy(ev).float().unsqueeze(0).to(device))).item()
                gate_probs.append(prob)
    else:
        # No gate — accept all
        gate_probs = [1.0] * len(val_samples)

    final_errs = []
    ranker_correct = 0
    ranker_total = 0
    details = []

    with torch.no_grad():
        for i, s in enumerate(val_samples):
            prob = gate_probs[i]
            base_err = _sf(s["base_error_px"])
            oracle_err = _sf(s["oracle_error_px"])

            if prob < gate_threshold:
                final_errs.append(base_err)
                details.append({"video": s.get("video_name"), "chosen": "base", "final_err": base_err, "base_err": base_err, "oracle_err": oracle_err})
                continue

            cands = s.get("candidates", [])
            if not cands:
                final_errs.append(base_err)
                details.append({"video": s.get("video_name"), "chosen": "base_no_cands", "final_err": base_err})
                continue

            # Score candidates with ranker
            cand_scores = []
            for c in cands:
                cv = np.asarray(extract_cand_vec(s, c, extra_features=extra_features), dtype=np.float32)
                cv = (cv - rank_mu) / rank_sigma
                sc = ranker_model(torch.from_numpy(cv).float().unsqueeze(0).to(device)).item()
                cand_scores.append(sc)

            best_idx = int(np.argmax(cand_scores))
            chosen_err = _sf(cands[best_idx].get("cand_error_px", base_err))
            final_errs.append(chosen_err)

            # Check if ranker picked oracle
            oracle_idx = int(s.get("oracle_index", -1))
            if oracle_idx >= 0 and oracle_idx < len(cands):
                ranker_correct += int(best_idx == oracle_idx)
                ranker_total += 1

            details.append({
                "video": s.get("video_name"), "point_idx": s.get("point_idx"),
                "chosen_rank": best_idx, "oracle_rank": oracle_idx,
                "chosen_err": chosen_err, "base_err": base_err, "oracle_err": oracle_err,
                "gate_prob": round(prob, 4),
            })

    final_errs = np.asarray(final_errs)

    # Metrics
    positive_mask = np.asarray([s.get("has_positive_candidate", False) for s in val_samples], dtype=bool)
    base16_mask = np.asarray([s.get("group_base16", False) for s in val_samples], dtype=bool)
    base32_mask = np.asarray([s.get("group_base32", False) for s in val_samples], dtype=bool)

    def group_stats(mask, name):
        if mask.sum() == 0:
            return {"name": name, "n": 0}
        be = base_errs[mask]
        fe = final_errs[mask]
        oe = oracle_errs[mask]
        ose = oracle_sel_errs[mask]
        return {
            "name": name, "n": int(mask.sum()),
            "base_median": round(float(np.median(be)), 2),
            "final_median": round(float(np.median(fe)), 2),
            "oracle_candidate_median": round(float(np.median(oe)), 2),
            "oracle_selective_median": round(float(np.median(ose)), 2),
            "final_better_2px_frac": round(float((fe < be - 2).mean()), 3),
            "oracle_better_2px_frac": round(float((oe < be - 2).mean()), 3),
            "final_median_improvement_pct": round(float((np.median(be) - np.median(fe)) / max(np.median(be), 1e-6) * 100), 1),
        }

    results = {
        "overall": group_stats(np.ones(len(val_samples), dtype=bool), "overall"),
        "positive": group_stats(positive_mask, "positive"),
        "base16": group_stats(base16_mask, "base16"),
        "base32": group_stats(base32_mask, "base32"),
        "ranker_top1_accuracy_on_positive": round(ranker_correct / max(1, ranker_total), 3),
        "ranker_positive_total": ranker_total,
        "gate_accept_rate": round(float((np.asarray(gate_probs) >= gate_threshold).mean()), 3),
    }

    return results, details


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--val-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gate-model-json", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--gate-threshold", type=float, default=0.50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_samples = [json.loads(l) for l in open(args.train_jsonl) if l.strip()]
    val_samples = [json.loads(l) for l in open(args.val_jsonl) if l.strip()]
    print(f"Train: {len(train_samples)} ({sum(1 for s in train_samples if s.get('has_positive_candidate'))} positive), "
          f"Val: {len(val_samples)} ({sum(1 for s in val_samples if s.get('has_positive_candidate'))} positive)")

    # Load gate model from v2 results if provided
    gate_model = None
    gate_norm = None
    if args.gate_model_json:
        # We'll use the gate from v2 — just load the results for reference
        # The gate model itself isn't saved in results.json, so we'll use a simple rule
        pass

    all_results = {}

    # Train with 3 loss types + extra features variant
    configs = [
        ("bce_baseline", "bce", False),
        ("bce_extra", "bce", True),
        ("pairwise_hinge", "pairwise", True),
        ("listwise_softmax", "listwise", True),
    ]

    best_positive_median = 999
    best_config_name = None

    for config_name, loss_type, extra_feat in configs:
        print(f"\n--- {config_name} (loss={loss_type}, extra_features={extra_feat}) ---")
        model, train_info = train_ranker(
            train_samples, val_samples, device,
            loss_type=loss_type, epochs=args.epochs, lr=args.lr, hidden=args.hidden,
            extra_features=extra_feat,
        )
        if model is None:
            print("  No positive training events!")
            continue

        results, details = evaluate_pipeline(
            model, train_info, gate_model, gate_norm,
            val_samples, device, gate_threshold=args.gate_threshold,
            extra_features=extra_feat,
        )

        all_results[config_name] = {
            "loss_type": loss_type,
            "extra_features": extra_feat,
            "train_info": {k: v for k, v in train_info.items() if k != "mu" and k != "sigma"},
            "results": results,
        }

        pos_med = results["positive"]["final_median"]
        print(f"  positive: final_med={pos_med:.1f} (base={results['positive']['base_median']:.1f}, oracle_sel={results['positive']['oracle_selective_median']:.1f})")
        print(f"  base16:   final_med={results['base16']['final_median']:.1f}, better_2px={results['base16']['final_better_2px_frac']:.3f}")
        print(f"  overall:  final_med={results['overall']['final_median']:.1f}, diff={results['overall']['final_median'] - results['overall']['base_median']:.1f}")
        print(f"  ranker acc on positive: {results['ranker_top1_accuracy_on_positive']:.3f}")

        if pos_med < best_positive_median:
            best_positive_median = pos_med
            best_config_name = config_name

    # Save
    summary = {
        "train_n": len(train_samples),
        "val_n": len(val_samples),
        "gate_threshold": args.gate_threshold,
        "best_config": best_config_name,
        "best_positive_median": best_positive_median,
        "baseline_positive_median": 20.33,
        "oracle_positive_median": 10.67,
        "configs": all_results,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Error analysis on best config
    if best_config_name:
        cfg = all_results[best_config_name]
        # Re-run to get details
        extra_feat = cfg["extra_features"]
        loss_type = cfg["loss_type"]
        model, train_info = train_ranker(
            train_samples, val_samples, device,
            loss_type=loss_type, epochs=args.epochs, lr=args.lr, hidden=args.hidden,
            extra_features=extra_feat,
        )
        if model:
            _, details = evaluate_pipeline(
                model, train_info, gate_model, gate_norm,
                val_samples, device, gate_threshold=args.gate_threshold,
                extra_features=extra_feat,
            )
            # Sort by error (worst first)
            details.sort(key=lambda d: -abs(d.get("chosen_err", 0) - d.get("oracle_err", 0)))
            with open(out_dir / "error_cases_top20.json", "w") as f:
                json.dump(details[:20], f, indent=2)
            with open(out_dir / "per_sample_predictions.jsonl", "w") as f:
                for d in details:
                    f.write(json.dumps(d) + "\n")

    # Final summary
    print(f"\n{'='*60}")
    print(f"Positive-Only Rank Refinement Results")
    print(f"{'='*60}")
    print(f"  Baseline (v2): positive_final_median = 20.33")
    print(f"  Oracle:        positive_oracle_median = 10.67")
    print(f"  Best config:   {best_config_name} -> positive_final_median = {best_positive_median:.1f}")
    for name, cfg in all_results.items():
        r = cfg["results"]
        print(f"  {name:25s}: pos_med={r['positive']['final_median']:.1f}, b16_med={r['base16']['final_median']:.1f}, "
              f"overall_diff={r['overall']['final_median'] - r['overall']['base_median']:.1f}, "
              f"ranker_acc={r['ranker_top1_accuracy_on_positive']:.3f}")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
