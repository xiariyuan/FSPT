# RGB-Stacking Held-Out-10 B2-W16 Failure Mode Audit — 2026-06-29

## Decision

Held-out B2-W16 is positive for re-entry reliability but has a real standard-AJ tradeoff. The main standard-AJ loss comes from false-trigger tracks, not from true re-entry triggers.

This is a useful negative result for a stronger paper: B2-W16 improves re-entry on held-out videos, but false triggers remain the main bottleneck for making the method CCF-B / CAS-Q2 competitive.

## Protocol reminder

```text
Held-out analysis only. Do not tune B2-W16 on these videos.
```

No rule should be tuned on these held-out videos. This audit is diagnostic only.

## Failed videos

Standard-AJ drop > 2 points vs offline:

```text
rgb_stacking_000010
rgb_stacking_000015
rgb_stacking_000017
```

## Overall class-level effects

| class | n | mean full delta | median full delta | severe < -0.10 | harmful < -0.05 | helpful > 0.05 | mean re-entry delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| true_trigger | 1481 | -0.008365 | -0.003117 | 212 | 374 | 228 | 0.059598 |
| false_trigger | 1325 | -0.037199 | -0.003938 | 286 | 371 | 172 | None |
| missed_reentry | 233 | 0.0 | 0.0 | 0 | 0 | 0 | 0.0 |
| no_trigger | 8172 | 0.0 | 0.0 | 0 | 0 | 0 | None |

Key reading:

```text
true triggers: improve re-entry segment on average, but mildly hurt full-track AJ
false triggers: produce the main standard-AJ cost
missed/no-trigger tracks: unchanged by construction
```

## Failed-video breakdown

| video | B2-offline AJ_RD | B2-offline AJ | trigger precision | trigger recall | true trigger n | false trigger n | false-trigger mean full delta | true-trigger mean full delta | true-trigger mean re-entry delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rgb_stacking_000010 | -0.0036 | -3.7765 | 0.442085 | 0.845018 | 229 | 289 | -0.141184 | -0.034789 | 0.0059 |
| rgb_stacking_000015 | 0.0355 | -2.1038 | 0.757576 | 0.961538 | 125 | 40 | -0.141307 | -0.008141 | 0.036666 |
| rgb_stacking_000017 | 0.0622 | -3.3177 | 0.627907 | 0.992647 | 135 | 80 | -0.049112 | -0.029801 | 0.058351 |

## Interpretation

The three standard-drop videos have different degrees of re-entry success, but the common problem is false-trigger cost:

- `rgb_stacking_000010`: worst case. B2-W16 does not improve AJ_RD over offline and loses -3.7765 AJ. False triggers are highly damaging: mean full-track delta -0.141184.
- `rgb_stacking_000015`: B2-W16 improves AJ_RD by +0.0355 but loses -2.1038 AJ. True re-entry triggers help re-entry, but false triggers are expensive.
- `rgb_stacking_000017`: B2-W16 improves AJ_RD by +0.0622 but loses -3.3177 AJ. True triggers help re-entry, but both false and true triggers reduce full-track AJ.

This supports the current paper framing:

```text
B2-W is a re-entry reliability intervention with a controllable standard-AJ tradeoff.
The remaining bottleneck is false-trigger cost, not the basic local-intervention mechanism.
```

## CCF-B / CAS-Q2 implication

For a stronger target, the current story is promising but not yet enough to claim a polished robust tracker. The paper can be credible if framed as reliability/tradeoff analysis, but to target CCF-B / CAS-Q2 more safely, the next method-development work should reduce false-trigger cost on the development data and then evaluate on a new held-out split.

Do not use this held-out audit to tune a new rule. Safer next protocol:

```text
1. Treat heldout-10 as consumed evaluation evidence.
2. Develop any false-trigger-cost reducer only on DAVIS + RGB dev first-10.
3. Evaluate the improved method on fresh RGB heldout-20/heldout-30, e.g. videos 20-39 or 20-49.
```

## Artifacts

```text
scripts/audit_rgb_heldout_b2w16_failure_modes.py
outputs/paper_discovery_2026-06-27/rgb_stacking_heldout/b2_w16_heldout10_failure_modes/summary.json
outputs/paper_discovery_2026-06-27/rgb_stacking_heldout/b2_w16_heldout10_failure_modes/track_rows.jsonl
```
