# Current Negative Result Closure (2026-06-25)

## Status

The previous pseudo-label rollout route is closed.

### Frozen decisions

- Do not continue DINO or local-feature pseudo-label rollout.
- Do not start P4.
- Do not train from scratch.
- Do not treat V2 local smoke as a GO signal.
- Do not use DINO frozen template matching as a teacher.
- Do not continue feature-based pseudo-label rollout.

### Current state labels

- `P0 = BLOCKED_METRIC_RECONCILIATION`
- `P1 = WEAK_DIAGNOSTIC_ONLY`
- `P2 = PARTIAL_LOCAL_FEATURE_SMOKE`
- `P3A = SMOKE_STOP`
- `P4 = DO_NOT_START`

## What remains open

The narrower idea is still alive:

- freeze a strong baseline as the coarse prior
- let a re-entry visual re-detection module intervene only at re-entry
- optimize for AJ_RD, not for global tracker replacement

This direction is not yet proven viable. It must pass upper-bound audits before any training is allowed.

## Required audits

1. **Search-window oracle**
   - Measure whether GT falls within 16 / 32 / 64 / 128 / 256 px around the baseline re-entry prediction.
   - Gate: if `coverage@128 < 70%`, the direction is not worth training.

2. **Candidate oracle**
   - Measure whether GT appears in top-k candidate sets from grid, local feature matching, and multi-frame variants.
   - Gate: if `top10@16px < 50%`, selector training is not justified.

3. **Oracle AJ_RD gain**
   - Measure corrected AJ_RD after oracle replacement when the GT is inside the candidate set.
   - Gate: if gain is `< +5pp`, the direction should be abandoned or redesigned.

4. **Label audit**
   - Compare any pseudo-label source against CT-offline raw.
   - Gate: if pseudo-labels are worse than CT-offline raw, do not train on them.

## Decision rule

- If all upper-bound audits pass, proceed to a small re-entry module design.
- If any upper-bound audit fails, stop this idea and search for a new one.
