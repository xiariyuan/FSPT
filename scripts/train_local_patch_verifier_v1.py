#!/usr/bin/env python3
"""
Patch verifier v1: frozen DINO patch embeddings for candidate ranking.

Encodes each patch (base, query, candidate) with frozen DINOv2.
Computes pairwise features: cosine(query, cand), cosine(base, cand), etc.
Trains a small MLP ranker on positive events only.

Usage:
  python scripts/train_local_patch_verifier_v1.py \
    --dataset-dir outputs/local_patch_verifier_dataset \
    --train-index outputs/local_selector_dataset_v3_from_anchor_smoke/train.jsonl \
    --val-index outputs/local_selector_dataset_v3_from_anchor_smoke/val.jsonl \
    --output-dir outputs/local_patch_verifier_v1 \
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
# DINO patch encoding
# ---------------------------------------------------------------------------
def encode_patches_dino(dino_extractor, patches_uint8, device, batch_size=16):
    """Encode (N, H, W, 3) uint8 patches to (N, 384) normalized embeddings."""
    if len(patches_uint8) == 0:
        return np.zeros((0, 384), dtype=np.float32)
    # Convert to tensor
    t = torch.from_numpy(patches_uint8).float()  # (N, H, W, 3)
    if t.dim() == 3:
        t = t.unsqueeze(0)
    # Use encode_patches_batch
    emb = dino_extractor.encode_patches_batch(t, device, batch_size=batch_size)
    return emb.numpy()  # (N, 384)


# ---------------------------------------------------------------------------
# Feature extraction for ranker
# ---------------------------------------------------------------------------
def extract_patch_features(query_emb, base_emb, cand_embs, extra_meta=None):
    """Extract pairwise features between query/base/candidate embeddings.

    Returns feature vector per candidate.
    """
    K = len(cand_embs)
    features = []

    for j in range(K):
        ce = cand_embs[j]

        # Cosine similarities
        cos_query = float(np.dot(query_emb, ce))
        cos_base = float(np.dot(base_emb, ce))

        # L2 distances
        l2_query = float(np.linalg.norm(query_emb - ce))
        l2_base = float(np.linalg.norm(base_emb - ce))

        # Differences
        diff_query = query_emb - ce
        diff_base = base_emb - ce

        feat = [
            cos_query,
            cos_base,
            l2_query,
            l2_base,
            float(np.mean(diff_query)),
            float(np.std(diff_query)),
            float(np.mean(diff_base)),
            float(np.std(diff_base)),
        ]

        # Add scalar features from selector if available
        if extra_meta and j < len(extra_meta):
            m = extra_meta[j]
            feat.extend([
                m.get("cand_score", 0),
                m.get("cand_shift_px_from_base", 0) / 256.0,
                m.get("rank_by_dino", 0) / max(1, K - 1),
            ])
        else:
            feat.extend([0, 0, j / max(1, K - 1)])

        features.append(feat)

    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class PatchVerifierMLP(nn.Module):
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
# Losses
# ---------------------------------------------------------------------------
def listwise_softmax_loss(scores, labels, sample_ids):
    loss = torch.tensor(0.0, device=scores.device)
    n = 0
    for sid in torch.unique(sample_ids):
        mask = sample_ids == sid
        s = scores[mask]
        l = labels[mask]
        oi = (l == 1).nonzero(as_tuple=True)[0]
        if len(oi) == 0:
            continue
        loss = loss - F.log_softmax(s, dim=0)[oi[0]]
        n += 1
    return loss / max(n, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--train-index", type=str, required=True)
    parser.add_argument("--val-index", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gate-threshold", type=float, default=0.50)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_dir = Path(args.dataset_dir)

    # Load DINO
    from models.recovery_features import DINORecoveryExtractor
    dino = DINORecoveryExtractor()
    dino._ensure_loaded(device)

    # Load dataset index
    index = json.load(open(ds_dir / "index.json"))
    idx_by_key = {}
    for e in index:
        key = (e["video_name"], e["point_idx"], e["t_reentry"])
        idx_by_key[key] = e

    # Load train/val sample IDs
    train_sel = [json.loads(l) for l in open(args.train_index) if l.strip()]
    val_sel = [json.loads(l) for l in open(args.val_index) if l.strip()]

    # Match to patch dataset
    def match_samples(sel_samples):
        matched = []
        for s in sel_samples:
            key = (s["video_name"], s["point_idx"], s["t_reentry"])
            if key in idx_by_key:
                matched.append(idx_by_key[key])
        return matched

    train_matched = match_samples(train_sel)
    val_matched = match_samples(val_sel)
    print(f"Matched: train={len(train_matched)}/{len(train_sel)}, val={len(val_matched)}/{len(val_sel)}")

    # Encode all patches
    print("Encoding patches with DINO...")
    all_patch_embs = {}  # npz_file -> {"query": emb, "base": emb, "cands": [emb, ...]}

    for i, entry in enumerate(index):
        npz_path = ds_dir / entry["npz_file"]
        if not npz_path.exists():
            continue
        data = np.load(str(npz_path))

        # Encode query patch
        qp = data["query_patch"]  # (H, W, 3)
        q_emb = encode_patches_dino(dino, qp[np.newaxis], device)[0]  # (384,)

        # Encode base patch
        bp = data["base_patch"]
        b_emb = encode_patches_dino(dino, bp[np.newaxis], device)[0]

        # Encode candidate patches
        cp = data["cand_patches"]  # (K, H, W, 3)
        c_embs = encode_patches_dino(dino, cp, device)  # (K, 384)

        all_patch_embs[entry["npz_file"]] = {
            "query": q_emb,
            "base": b_emb,
            "cands": c_embs,
        }

        if (i + 1) % 10 == 0:
            print(f"  Encoded {i+1}/{len(index)}")

    print(f"Encoded {len(all_patch_embs)} samples")

    # Build feature arrays for positive events
    def build_arrays(matched_entries, all_embs):
        feats, labels, sids = [], [], []
        for si, entry in enumerate(matched_entries):
            if not entry.get("has_positive_candidate", False):
                continue
            npz_file = entry.get("npz_file")
            if npz_file not in all_embs:
                continue
            embs = all_embs[npz_file]
            cands = entry.get("candidates", [])
            feat = extract_patch_features(
                embs["query"], embs["base"], embs["cands"],
                extra_meta=cands,
            )
            K = len(feat)
            oracle_idx = entry.get("oracle_index", -1)
            for j in range(K):
                feats.append(feat[j])
                labels.append(1.0 if j == oracle_idx else 0.0)
                sids.append(si)
        if not feats:
            return np.zeros((0, 11), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64)
        return np.array(feats, np.float32), np.array(labels, np.float32), np.array(sids, np.int64)

    train_x, train_y, train_sid = build_arrays(train_matched, all_patch_embs)
    val_x, val_y, val_sid = build_arrays(val_matched, all_patch_embs)
    print(f"Ranker: train={len(train_x)} rows ({len(np.unique(train_sid))} events), val={len(val_x)} rows ({len(np.unique(val_sid))} events)")

    if len(train_x) == 0 or len(val_x) == 0:
        print("Not enough positive events for training!")
        return

    # Normalize
    mu, sigma = train_x.mean(0), train_x.std(0) + 1e-6
    train_x_n = (train_x - mu) / sigma
    val_x_n = (val_x - mu) / sigma

    # Train
    model = PatchVerifierMLP(train_x.shape[1], hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    Xt = torch.from_numpy(train_x_n).to(device)
    yt = torch.from_numpy(train_y).to(device)
    sidt = torch.from_numpy(train_sid).to(device)

    best_state = None
    best_acc = -1

    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        scores = model(Xt)
        loss = listwise_softmax_loss(scores, yt, sidt)
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

    # Evaluate full pipeline
    def eval_pipeline(matched_entries, all_embs, ranker_model, ranker_mu, ranker_sigma):
        ranker_model.eval()
        base_errs = []
        final_errs = []
        oracle_errs = []
        oracle_sel_errs = []
        ranker_correct = 0
        ranker_total = 0
        details = []

        for entry in matched_entries:
            base_err = _sf(entry.get("base_error_px", 0))
            oracle_err = _sf(entry.get("oracle_error_px", 999))
            oracle_sel = oracle_err if oracle_err < base_err - 2 else base_err

            base_errs.append(base_err)
            oracle_errs.append(oracle_err)
            oracle_sel_errs.append(oracle_sel)

            # Gate: use simple rule (same as v2: gate_prob not available here, use has_positive_candidate as proxy)
            # In real integration, gate would be the trained gate model
            # Here we simulate: accept if has_positive_candidate
            has_pos = entry.get("has_positive_candidate", False)

            npz_file = entry.get("npz_file")
            if not has_pos or npz_file not in all_embs:
                final_errs.append(base_err)
                details.append({"video": entry.get("video_name"), "chosen": "base", "final_err": base_err, "base_err": base_err, "oracle_err": oracle_err})
                continue

            embs = all_embs[npz_file]
            cands = entry.get("candidates", [])
            feat = extract_patch_features(embs["query"], embs["base"], embs["cands"], extra_meta=cands)

            with torch.no_grad():
                feat_n = (feat - ranker_mu) / ranker_sigma
                scores = ranker_model(torch.from_numpy(feat_n).float().to(device)).cpu().numpy()

            best_idx = int(np.argmax(scores))
            cands_list = entry.get("candidates", [])
            if best_idx < len(cands_list):
                chosen_err = _sf(cands_list[best_idx].get("cand_error_px", base_err))
            else:
                chosen_err = base_err
            final_errs.append(chosen_err)

            oracle_idx = entry.get("oracle_index", -1)
            if oracle_idx >= 0:
                ranker_correct += int(best_idx == oracle_idx)
                ranker_total += 1

            details.append({
                "video": entry.get("video_name"), "point_idx": entry.get("point_idx"),
                "chosen_rank": best_idx, "oracle_rank": oracle_idx,
                "chosen_err": round(chosen_err, 2), "base_err": round(base_err, 2),
                "oracle_err": round(oracle_err, 2), "oracle_sel_err": round(oracle_sel, 2),
            })

        base_errs = np.array(base_errs)
        final_errs = np.array(final_errs)
        oracle_errs = np.array(oracle_errs)
        oracle_sel_errs = np.array(oracle_sel_errs)

        pos_mask = np.array([e.get("has_positive_candidate", False) for e in matched_entries], dtype=bool)
        b16_mask = np.array([_sf(e.get("base_error_px", 0)) > 16 for e in matched_entries], dtype=bool)
        b32_mask = np.array([_sf(e.get("base_error_px", 0)) > 32 for e in matched_entries], dtype=bool)

        def gstats(mask, name):
            if mask.sum() == 0:
                return {"name": name, "n": 0}
            be, fe, oe, ose = base_errs[mask], final_errs[mask], oracle_errs[mask], oracle_sel_errs[mask]
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

        return {
            "overall": gstats(np.ones(len(matched_entries), dtype=bool), "overall"),
            "positive": gstats(pos_mask, "positive"),
            "base16": gstats(b16_mask, "base16"),
            "base32": gstats(b32_mask, "base32"),
            "ranker_top1_accuracy_on_positive": round(ranker_correct / max(1, ranker_total), 3),
            "ranker_positive_total": ranker_total,
        }, details

    results, details = eval_pipeline(val_matched, all_patch_embs, model, mu, sigma)

    # Save
    output = {
        "dataset_dir": str(ds_dir.resolve()),
        "train_n": len(train_matched),
        "val_n": len(val_matched),
        "results": results,
        "baseline_positive_median": 20.33,
        "baseline_base16_median": 22.87,
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)

    with open(out_dir / "per_sample_predictions.jsonl", "w") as f:
        for d in details:
            f.write(json.dumps(d) + "\n")

    # Error cases
    details_sorted = sorted(details, key=lambda d: -abs(d.get("chosen_err", 0) - d.get("oracle_err", 0)))
    with open(out_dir / "error_cases_top20.json", "w") as f:
        json.dump(details_sorted[:20], f, indent=2)

    # Print
    print(f"\n{'='*60}")
    print(f"Patch Verifier v1 Results")
    print(f"{'='*60}")
    for group in ["overall", "positive", "base16", "base32"]:
        g = results.get(group, {})
        if g.get("n", 0) == 0:
            continue
        print(f"  {group:12s} (n={g['n']:>3d}): base={g['base_median']:.1f}, final={g['final_median']:.1f}, "
              f"oracle_sel={g['oracle_selective_median']:.1f}, better2px={g['final_better_2px_frac']:.3f}")
    print(f"  ranker acc on positive: {results['ranker_top1_accuracy_on_positive']:.3f}")
    print(f"\n  Baseline: positive=20.33, base16=22.87")
    print(f"  Oracle:   positive=10.67")
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
