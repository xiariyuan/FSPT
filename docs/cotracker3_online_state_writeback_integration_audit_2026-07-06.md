# CoTracker3 Online State-Writeback Integration Audit — 2026-07-06

## Question

Verify whether the current CoTracker3 online state-level ReEntry integration is technically correct, and whether it is methodologically reasonable.

## Code-level finding: state exists and writeback reaches the right fields

CoTracker3 true online keeps these state tensors:

```text
online_track_feat
online_track_support
online_coords_predicted
online_vis_predicted
online_conf_predicted
online_ind
```

The next window copies previous state into the current initialization:

```text
coords_prev -> coords_init
vis_prev    -> vis_init
conf_prev   -> conf_init
```

In `scripts/eval_cotracker3_true_streaming_reentry_subset.py`, the state-writeback variant modifies:

```text
model.model.online_vis_predicted[0, t, q]
model.model.online_conf_predicted[0, t, q]
```

before the next window is run.

Therefore:

```text
The integration is structurally correct: it writes into the tensors that CoTracker3 online uses in later windows.
```

## Important protocol clarification

The old cache:

```text
outputs/attempt0_2026-06-15_recovery/prediction_caches/cotracker3_baseline_davis_first_input_bridge.pt
```

is not the same execution mode as the true streaming `CoTrackerOnlinePredictor` loop used for state-writeback.

The old cache is best treated as:

```text
CoTracker3 online-architecture / full-sequence evaluation cache
```

The state-writeback experiment is:

```text
CoTracker3 true-streaming windowed online evaluation
```

The true-streaming native baseline must be used as the control for state-writeback experiments.

## 3-video subset result

Subset: first three DAVIS videos.

Parameters:

```text
tau_low=0.55
pre_window=2
max_jump_px=16
state_prob=0.97
```

| Variant | AJ | OA | delta_avg | AJ_RD | AJ_RD_256 |
|---|---:|---:|---:|---:|---:|
| true-streaming native | 68.8956 | 86.9745 | 84.5307 | 0.3082 | 0.4941 |
| output-only | 68.7978 | 86.7901 | 84.5307 | 0.3118 | 0.5004 |
| state-writeback | 68.6243 | 86.6221 | 84.5243 | 0.3130 | 0.4993 |

Delta vs true-streaming native:

| Variant | ΔAJ | ΔOA | ΔAJ_RD | ΔAJ_RD_256 |
|---|---:|---:|---:|---:|
| output-only | -0.0977 | -0.1844 | +0.0036 | +0.0063 |
| state-writeback | -0.2713 | -0.3523 | +0.0048 | +0.0052 |

This does not pass the gate:

```text
AJ >= native - 0.10
OA >= native - 0.10
AJ_RD >= native + 0.01
```

## Opened-frame quality audit

Compared to true-streaming native:

```text
output-only opened frames: 26
GT-visible among opened: 8
GT-occluded among opened: 18
GT-visible precision: 30.77%

state-writeback opened frames: 48
GT-visible among opened: 15
GT-occluded among opened: 33
GT-visible precision: 31.25%
```

Interpretation:

```text
The rule opens many false-visible frames. This explains the AJ/OA drop.
```

## Is the current integration correct?

Yes, at the structural code level:

```text
1. It uses the true streaming CoTrackerOnlinePredictor loop.
2. It resets online state per run through is_first_step=True.
3. It writes into online_vis_predicted and online_conf_predicted before later windows.
4. These tensors are copied into later vis_init/conf_init.
5. The native/output/state variants are compared under the same true-streaming subset protocol.
```

But it is not yet a paper-ready method result:

```text
1. The current rule is delayed/window-level, not zero-latency.
2. It edits vis/conf logits but not track feature memory.
3. It writes some frames that are outside the next overlap, which only changes final output rather than future state.
4. The frame-open precision is too low (~31%).
5. It gives small AJ_RD gains but hurts AJ/OA more than allowed.
```

## Is the current integration reasonable?

Reasonable as a diagnostic smoke:

```text
It verifies that state-level visibility writeback is possible and can affect later online outputs.
```

Not reasonable yet as the final method:

```text
The gate is too weak. It should not be expanded to 30 videos or claimed as a success.
```

## Recommended fixes before rerun

1. Restrict writeback to the overlap region that will actually initialize the next window.
2. Add GT-free stronger gates:

```text
confidence high threshold
confidence rising trend
coordinate local smoothness
multi-frame consistency
native re-entry confirmation
no writeback for isolated one-frame candidates
```

3. Separate two variants explicitly:

```text
short-latency output correction
short-latency state writeback
```

4. Track opened-frame precision without using it for inference, only for audit.

## Current decision

```text
Do not run 30-video state-writeback yet.
Do not claim CoTracker3 online state-level improvement.
Proceed to safer gate design and overlap-only writeback smoke.
```
