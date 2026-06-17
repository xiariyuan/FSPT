#!/usr/bin/env python3
"""
Patch verifier v1 LOOCV: frozen DINO patch embeddings with sequence-grouped
leave-one-out cross-validation.

Protocol:
  - 5 val sequences, each round leave 1 for test
  - Ranker trained ONLY on positive events from train sequences
  - Evaluation: positive-only metrics (ranker accuracy, median) per fold + average
  - Full pipeline: only with real gate from Stage-2c (no GT leakage)

Usage:
  python scripts/train_local_patch_verifier_v1_loocv.py \
    --dataset-dir outputs/local_patch_verifier_dataset \
    --val-jsonl outputs/local_selector_dataset_v3_from_anchor_smoke/val.jsonl \
    --output-dir outputs/local_patch_verifier_v1_loocv \
    --gate-threshold 0.50 --epochs 200
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _sf(v, d=0.0):
    try: return float(v)
    except: return float(d)


# ---------------------------------------------------------------------------
# DINO patch encoding (precompute once for all samples)
# ---------------------------------------------------------------------------
def encode_all_patches(dino, index, ds_dir, device):
    """Encode all patches once, cache as dict."""
    cache = {}
    for i, entry in enumerate(index):
        npz_path = ds_dir / entry["npz_file"]
        if not npz_path.exists(): continue
        data = np.load(str(npz_path))
        qp = data["query_patch"]
        q_emb = dino.encode_patches_batch(torch.from_numpy(qp[np.newaxis]).float(), device)[0].numpy()
        bp = data["base_patch"]
        b_emb = dino.encode_patches_batch(torch.from_numpy(bp[np.newaxis]).float(), device)[0].numpy()
        cp = data["cand_patches"]
        c_embs = dino.encode_patches_batch(torch.from_numpy(cp).float(), device).numpy()
        cache[entry["npz_file"]] = {"query": q_emb, "base": b_emb, "cands": c_embs}
        if (i+1) % 10 == 0: print(f"  Encoded {i+1}/{len(index)}")
    return cache


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_patch_features(query_emb, base_emb, cand_embs, extra_meta=None):
    K = len(cand_embs)
    features = []
    for j in range(K):
        ce = cand_embs[j]
        cos_q = float(np.dot(query_emb, ce))
        cos_b = float(np.dot(base_emb, ce))
        l2_q = float(np.linalg.norm(query_emb - ce))
        l2_b = float(np.linalg.norm(base_emb - ce))
        diff_q = query_emb - ce
        diff_b = base_emb - ce
        feat = [cos_q, cos_b, l2_q, l2_b, float(np.mean(diff_q)), float(np.std(diff_q)),
                float(np.mean(diff_b)), float(np.std(diff_b))]
        if extra_meta and j < len(extra_meta):
            m = extra_meta[j]
            feat.extend([m.get("cand_score",0), m.get("cand_shift_px_from_base",0)/256.0,
                         m.get("rank_by_dino",0)/max(1,K-1)])
        else:
            feat.extend([0, 0, j/max(1,K-1)])
        features.append(feat)
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Model + loss
# ---------------------------------------------------------------------------
class PatchVerifierMLP(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def listwise_softmax_loss(scores, labels, sample_ids):
    loss = torch.tensor(0.0, device=scores.device)
    n = 0
    for sid in torch.unique(sample_ids):
        mask = sample_ids == sid
        s, l = scores[mask], labels[mask]
        oi = (l == 1).nonzero(as_tuple=True)[0]
        if len(oi) == 0: continue
        loss -= F.log_softmax(s, dim=0)[oi[0]]
        n += 1
    return loss / max(n, 1)


def train_ranker_fold(train_entries, embs_cache, device, epochs=200, lr=1e-3, hidden=64):
    """Train ranker on positive events only."""
    feats, labels, sids = [], [], []
    for si, entry in enumerate(train_entries):
        if not entry.get("has_positive_candidate", False): continue
        npz = entry.get("npz_file")
        if npz not in embs_cache: continue
        em = embs_cache[npz]
        feat = extract_patch_features(em["query"], em["base"], em["cands"], extra_meta=entry.get("candidates", []))
        oracle_idx = entry.get("oracle_index", -1)
        for j in range(len(feat)):
            feats.append(feat[j])
            labels.append(1.0 if j == oracle_idx else 0.0)
            sids.append(si)
    if not feats:
        return None, {}
    X = np.array(feats, np.float32)
    y = np.array(labels, np.float32)
    sid = np.array(sids, np.int64)

    mu, sigma = X.mean(0), X.std(0) + 1e-6
    X_n = (X - mu) / sigma

    model = PatchVerifierMLP(X.shape[1], hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt, sidt = torch.from_numpy(X_n).to(device), torch.from_numpy(y).to(device), torch.from_numpy(sid).to(device)

    best_state, best_acc = None, -1
    for epoch in range(epochs):
        model.train(); opt.zero_grad()
        loss = listwise_softmax_loss(model(Xt), yt, sidt)
        loss.backward(); opt.step()
        if (epoch+1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                vs = model(Xt).cpu().numpy()
            correct = sum(int(np.argmax(vs[sid==s]) == y[sid==s].argmax()) for s in np.unique(sid))
            acc = correct / max(1, len(np.unique(sid)))
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    return model, {"mu": mu.tolist(), "sigma": sigma.tolist(), "best_acc": best_acc, "n_train_pos": int(y.sum())}


def evaluate_fold(model, norm_dict, test_entries, embs_cache, gate_threshold=0.50):
    """Evaluate on test fold. Positive-only ranker metrics only (no GT gate leakage)."""
    if model is None:
        return {"positive_only": {"n_pos": 0}, "note": "No positive training events"}

    model.eval()
    mu = np.array(norm_dict["mu"]); sigma = np.array(norm_dict["sigma"])

    pos_metrics = {"ranker_correct": 0, "ranker_total": 0, "chosen_errs": [], "oracle_errs": [], "base_errs_pos": []}
    all_details = []

    for entry in test_entries:
        base_err = _sf(entry.get("base_error_px", 0))
        oracle_err = _sf(entry.get("oracle_error_px", 999))
        oracle_sel = oracle_err if oracle_err < base_err - 2 else base_err
        is_pos = entry.get("has_positive_candidate", False)

        if not is_pos or entry.get("npz_file") not in embs_cache:
            all_details.append({"video": entry["video_name"], "point_idx": entry["point_idx"],
                                "has_positive": is_pos, "chosen": "base",
                                "final_err": base_err, "base_err": base_err, "oracle_err": oracle_err})
            continue

        em = embs_cache[entry["npz_file"]]
        feat = extract_patch_features(em["query"], em["base"], em["cands"], extra_meta=entry.get("candidates", []))
        with torch.no_grad():
            feat_n = (feat - mu) / sigma
            scores = model(torch.from_numpy(feat_n).float().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))).cpu().numpy()
        best_idx = int(np.argmax(scores))
        cands = entry.get("candidates", [])
        chosen_err = _sf(cands[best_idx].get("cand_error_px", base_err)) if best_idx < len(cands) else base_err

        pos_metrics["chosen_errs"].append(chosen_err)
        pos_metrics["oracle_errs"].append(oracle_err)
        pos_metrics["base_errs_pos"].append(base_err)
        pos_metrics["ranker_total"] += 1
        if best_idx == entry.get("oracle_index", -2): pos_metrics["ranker_correct"] += 1

        all_details.append({"video": entry["video_name"], "point_idx": entry["point_idx"],
                            "has_positive": True, "chosen_rank": best_idx, "oracle_rank": entry.get("oracle_index"),
                            "chosen_err": round(chosen_err, 2), "base_err": round(base_err, 2),
                            "oracle_err": round(oracle_err, 2), "oracle_sel_err": round(oracle_sel, 2),
                            "scores": [round(float(s), 4) for s in scores]})

    ce = np.array(pos_metrics["chosen_errs"]) if pos_metrics["chosen_errs"] else np.array([])
    oe = np.array(pos_metrics["oracle_errs"]) if pos_metrics["oracle_errs"] else np.array([])
    be = np.array(pos_metrics["base_errs_pos"]) if pos_metrics["base_errs_pos"] else np.array([])

    pos_summary = {
        "n_pos": len(ce),
        "ranker_accuracy": round(pos_metrics["ranker_correct"] / max(1, pos_metrics["ranker_total"]), 3),
        "positive_final_median": round(float(np.median(ce)), 2) if len(ce) else None,
        "positive_oracle_median": round(float(np.median(oe)), 2) if len(oe) else None,
        "positive_base_median": round(float(np.median(be)), 2) if len(be) else None,
        "positive_better_2px_frac": round(float((ce < be - 2).mean()), 3) if len(ce) else None,
    }

    return {"positive_only": pos_summary, "details": all_details}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--val-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gate-threshold", type=float, default=0.50)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ds_dir = Path(args.dataset_dir)

    # Load index
    index = json.load(open(ds_dir / "index.json"))

    # Load DINO
    from models.recovery_features import DINORecoveryExtractor
    dino = DINORecoveryExtractor(); dino._ensure_loaded(device)

    # Pre-encode all patches
    print("Pre-encoding all patches...")
    embs_cache = encode_all_patches(dino, index, ds_dir, device)
    print(f"Encoded {len(embs_cache)} samples")

    # Group by video
    by_video = {}
    for e in index:
        v = e["video_name"]
        if v not in by_video: by_video[v] = []
        by_video[v].append(e)

    sequences = sorted(by_video.keys())
    print(f"Sequences: {sequences}")

    # LOOCV
    fold_results = []
    all_predictions = []

    for test_vid in sequences:
        print(f"\n--- Fold: test={test_vid} ---")
        train_entries = [e for v, es in by_video.items() if v != test_vid for e in es]
        test_entries = by_video[test_vid]

        n_train_pos = sum(1 for e in train_entries if e.get("has_positive_candidate"))
        n_test_pos = sum(1 for e in test_entries if e.get("has_positive_candidate"))
        print(f"  Train: {len(train_entries)} ({n_train_pos} positive), Test: {len(test_entries)} ({n_test_pos} positive)")

        model, train_info = train_ranker_fold(train_entries, embs_cache, device,
                                               epochs=args.epochs, lr=args.lr, hidden=args.hidden)

        fold_eval = evaluate_fold(model, train_info, test_entries, embs_cache, args.gate_threshold)

        pos = fold_eval["positive_only"]
        print(f"  Positive: ranker_acc={pos.get('ranker_accuracy','?')}, "
              f"final_med={pos.get('positive_final_median','?')}, "
              f"oracle_med={pos.get('positive_oracle_median','?')}, "
              f"base_med={pos.get('positive_base_median','?')}")

        fold_result = {"test_video": test_vid, "train_info": train_info, "positive_only": pos}
        fold_results.append(fold_result)

        for d in fold_eval.get("details", []):
            d["fold_test_video"] = test_vid
            all_predictions.append(d)

    # Aggregate
    pos_finals = [f["positive_only"]["positive_final_median"] for f in fold_results if f["positive_only"].get("positive_final_median") is not None]
    pos_oracles = [f["positive_only"]["positive_oracle_median"] for f in fold_results if f["positive_only"].get("positive_oracle_median") is not None]
    pos_accs = [f["positive_only"]["ranker_accuracy"] for f in fold_results if f["positive_only"].get("ranker_accuracy") is not None]

    avg_results = {
        "n_folds": len(fold_results),
        "avg_positive_final_median": round(float(np.mean(pos_finals)), 2) if pos_finals else None,
        "avg_positive_oracle_median": round(float(np.mean(pos_oracles)), 2) if pos_oracles else None,
        "avg_ranker_accuracy": round(float(np.mean(pos_accs)), 3) if pos_accs else None,
        "per_fold": fold_results,
        "baseline_positive_median": 20.33,
        "baseline_base16_median": 22.87,
    }

    with open(out_dir / "results.json", "w") as f: json.dump(avg_results, f, indent=2)
    with open(out_dir / "per_fold_metrics.json", "w") as f: json.dump(fold_results, f, indent=2)
    with open(out_dir / "per_sample_predictions.jsonl", "w") as f:
        for d in all_predictions: f.write(json.dumps(d) + "\n")

    print(f"\n{'='*60}")
    print(f"LOOCV Summary ({len(fold_results)} folds)")
    print(f"{'='*60}")
    print(f"  Avg positive final median: {avg_results['avg_positive_final_median']}")
    print(f"  Avg positive oracle median: {avg_results['avg_positive_oracle_median']}")
    print(f"  Avg ranker accuracy: {avg_results['avg_ranker_accuracy']}")
    print(f"  Baseline positive median: 20.33")
    for fr in fold_results:
        p = fr["positive_only"]
        print(f"  Fold {fr['test_video']:20s}: acc={p.get('ranker_accuracy','?')}, final_med={p.get('positive_final_median','?')}, oracle_med={p.get('positive_oracle_median','?')}")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
