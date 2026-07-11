# V9-A5.1c Smoke Review

Date: 2026-07-10

Protocol:

```text
clip: ani:0
frames: 0-7
queries: 32
rows: 256
full-frame hash skipped for smoke only
formal three-sequence gates disabled
```

Integrity:

```text
V9-A5.1b row-key mismatch: 0
official replay max_abs: 0
risk mismatch: 0
candidate score max_abs: 0
raw candidate error max_abs: 0
refined candidate error max_abs: 0
hybrid oracle max_abs: 0
official p/v/q parity max_abs: 0
beam width/shape checks: pass for all five policies
last-three-grid signature violations: 0 for all five policies
system top1/oracle formulas independently reproduced for all policies
```

Smoke-only visible means:

```text
official final: 1.8393 px
native_dynamic_B4 top1: 1.8588 px
native_dynamic_B4 oracle: 1.7542 px
native_dynamic_B4 raw-state minimum: 1.8636 px
```

These values are not a scientific result because the smoke contains only one clip and eight frames. They establish only that the history-preserving state, finite cost window, replay parity, pruning, persistence and readout formulas execute correctly.
