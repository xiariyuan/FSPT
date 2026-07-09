# V9-A2.2 Action-Space Oracle Upper-Bound Audit Result

Date: 2026-07-08

---

## 1. Purpose

V9-A2.2 checks whether W16 extension rows provide real trajectory-level upside beyond CVRRM W8, before training a dynamic horizon controller.

Question:

```text
Does W8 + selective W16 extension have enough oracle upper bound to justify V9-A2.3 dynamic horizon learning?
```

---

## 2. Artifacts

Script:

```text
scripts/v9a2_action_space_oracle_audit.py
```

Output:

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_action_space_oracle_audit.json
```

---

## 3. Label consistency check

Common W8/W16 rows:

```text
common rows = 1456
```

Result:

```text
No label mismatches were found on common rows.
```

Interpretation:

```text
The W8 and W16 touched-frame datasets can be safely joined by (video_id, query_idx, frame_tau).
```

---

## 4. Main oracle results

| Variant | AJ Δ | OA Δ | delta_avg Δ | delta_4px Δ | AJ_RD Δ | AJ_RD_256 Δ | Accepted | Pos/Neg/Zero |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| W8 accept-all | +0.0964 | +0.9462 | +0.3971 | +0.5085 | +0.0153 | +0.0274 | 1456 | 16/4/5 |
| W16 accept-all | +0.0810 | +0.9701 | +0.4339 | +0.5593 | +0.0162 | +0.0286 | 2013 | 15/5/5 |
| W8 + W16-only candidate_good oracle | +0.1955 | +1.0204 | +0.5073 | +0.6152 | +0.0191 | +0.0321 | 1574 | 17/3/5 |
| W8 + W16-only utility_positive oracle | +0.1955 | +1.0204 | +0.5073 | +0.6152 | +0.0191 | +0.0321 | 1574 | 17/3/5 |
| W8 + W16-only safe_no_damage oracle | +0.1933 | +1.0256 | +0.5073 | +0.6152 | +0.0190 | +0.0321 | 1845 | 17/3/5 |
| Event-level utility-positive W16 oracle | +0.1883 | +1.0126 | +0.4970 | +0.6037 | +0.0195 | +0.0320 | 1632 | 17/3/5 |

Key result:

```text
Best oracle AJ_RD_256 Δ = +0.0321
CVRRM W8 AJ_RD_256 Δ    = +0.0274
CVRRM W16 AJ_RD_256 Δ   = +0.0286
```

This shows a real upper-bound gap:

```text
W8 + selective W16 extension has meaningful trajectory-level upside.
```

---

## 5. Why this matters

V9-A1 failed because it only filtered W8 touched frames:

```text
Filtering W8 can reduce risk, but it cannot create new re-entry recovery opportunities.
```

V9-A2.2 shows that W16-only rows contain useful recoveries and that oracle selection can convert them into higher full trajectory metrics.

Therefore, the correct next action space is:

```text
W8 keep
W16 extend
high-risk veto
uncertainty/anchor-guided continuation
```

not:

```text
W8-only accept/reject.
```

---

## 6. Event-level oracle result

Event-level policies also improve over W8/W16 baselines:

```text
Event-level utility-positive W16 oracle:
AJ_RD_256 Δ = +0.0320
AJ Δ         = +0.1883
OA Δ         = +1.0126
Pos/Neg/Zero = 17/3/5
```

Interpretation:

```text
The dynamic horizon idea does not require perfect frame-level oracle selection.
Even event-level W8-vs-W16 selection has a strong upper bound.
```

This is important because ReEntryBeliefTrack's phase duration is naturally event/horizon-level, not isolated frame-level.

---

## 7. Cautions

These are oracle strategies using GT-derived labels.

They are not deployable methods.

They only prove:

```text
There is learnable/recoverable action-space headroom if anchor/uncertainty features can predict safe W16 extensions.
```

They do not prove:

```text
A learned V9-A2 controller will achieve this bound.
```

---

## 8. Decision

V9-A2.2 passes.

The dynamic horizon route is worth continuing.

Next:

```text
V9-A2.3 Full W8+W16 Anchor/Uncertainty Feature Build + Dynamic Horizon Controller
```

Minimum goals for V9-A2.3:

```text
1. Build aligned V9-A2 features for W8 and W16 rows.
2. Train video-heldout controller to predict W16 extension safety / utility.
3. Apply back to full trajectories.
4. Compare against W8, W16, and oracle upper bound.
```

Success standard:

```text
Strong pass: learned dynamic horizon AJ_RD_256 > +0.0286 and AJ/OA improve over W16.
Useful pass: AJ_RD_256 close to W16 but negative videos reduced and AJ/OA improved.
Fail: only tabular AP/AUC improves, with no apply-back gain.
```
