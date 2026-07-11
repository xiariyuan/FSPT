# V9-A5.2 Independent-State Smoke Review

Date: 2026-07-11

## 1. Scope

This review covers only the one-event integrity smoke.

It does not evaluate the frozen 72-event scientific gate.

Frozen smoke event:

```text
clip: ani:0
query index: 20
split frame: 2
horizons: 1 / 4 / 8
active tracker rows: 432
memory size: 24
feature dimension: 256
```

The event is the earliest frozen `ani:0` event under `(frame_tau, query_idx, selection_sha256)`. It was selected for runtime efficiency after the 72-event manifest was committed. No GT field was used.

## 2. Integrity conclusion

All smoke integrity gates pass.

```text
official diagnostic p/v/q_new parity max_abs: 0
Branch A full p/v/u/q_new/q_pre/C1/C2 parity max_abs: 0
Branch A q_init/memory/mask parity: exact
A/B/official q_init, memory and mask storage: disjoint
Branch B mutation changes Branch A: false
A->B versus B->A call-order max_abs: 0
online native-risk mismatch: 0
V9-A5.1b candidate replay max_abs: 1.52587890625e-05
shared/independent B2 formulas: exact
all numeric outputs: finite
branch-generation functions contain no GT/error input token
```

The complete numerical smoke was run twice. All NPZ arrays, gates, parity records, candidate replay values, horizon reports and split numerical diagnostics reproduced exactly. Runtime and CUDA storage addresses are intentionally not compared.

## 3. Provenance hashes

```text
smoke script:
1e0a0324bf0170bf7f9db1d91de51075bcd690410f4014b534c834628da3ec37

smoke JSON:
d8639e39c1d2356ad1992e1869fa05d54dcc52143a8426aafcd363645e940834

smoke NPZ:
2dbce4a9782c7226d06e82804817c10ffce95d8df429acfbccbe3dad3d9c8a37
```

## 4. The branch is a real model-state branch

At the event-memory write:

```text
singleton q2_B versus official q_new_A:
  L2:              10.2380
  max_abs:          2.3016
  cosine distance:  0.1104
```

Only the target row receives `q2_B`; all other 431 rows receive official `q_new_A`. The full branch memory then evolves independently.

Future target-state divergence remains nonzero:

```text
horizon 1 q_new L2: 8.9140
horizon 4 q_new L2: 2.7783
horizon 8 q_new L2: 2.0466
```

The target perturbation also propagates through query attention:

```text
horizon 1 non-target evaluated-query q_new mean L2: 0.0308
horizon 1 support-grid q_new mean L2:                0.0271
```

This confirms that complete 432-query branch cloning was necessary. A target-only rollout would not represent the real architecture.

## 5. Future proposal generation changes

Branch A and Branch B do not retain identical future C1 candidate sets.

```text
horizon 1:
  top16 overlap 14/16
  top64 overlap 54/64

horizon 4:
  top16 overlap 15/16
  top64 overlap 59/64

horizon 8:
  top16 overlap 15/16
  top64 overlap 62/64
```

C1 L2 divergence is also nonzero:

```text
horizon 1: 6.9746
horizon 4: 7.6801
horizon 8: 8.7188
```

Therefore the independent memory state changes future proposal generation, not only the final readout.

## 6. Important compression finding

Despite substantial state and correlation-map divergence, final coordinate divergence is small in this event:

```text
horizon 1: 0.0963 px
horizon 4: 0.1470 px
horizon 8: 0.0529 px
```

This is an important warning:

```text
state divergence != useful output divergence
candidate-set divergence != deterministic final improvement
```

The formal audit must therefore retain both mechanistic non-degeneracy and capacity-matched error gates. It may not claim success from q_new/C1 divergence alone.

## 7. Smoke error values are not scientific evidence

All three smoke horizon target rows are GT-invisible/invalid under the committed evaluation mask.

The saved coordinate errors only verify formulas and replay. They must not be used to infer whether Branch B helps or hurts.

For completeness, the formula outputs were:

```text
horizon 1 independent minus shared B2: +0.3631 px
horizon 4 independent minus shared B2:  0.0000 px
horizon 8 independent minus shared B2: +5.2782 px
```

These values are excluded from any route decision because the target is not visible-valid at those horizons.

## 8. Formal-run requirements confirmed by the smoke

The 72-event implementation must reuse the verified mechanisms:

```text
full 432-row TrackerState clone
split after event-frame forward and before memory write
Branch A official q_new write
Branch B target singleton q2 write
separate full-state forward calls
frame-feature reuse
unconditional independent memory updates
shared-state capacity-2 control
independent-state capacity-2 output
```

It must preserve the following all-event integrity checks:

```text
Branch A full output/state parity
storage disjointness
online event/risk parity
candidate replay parity
all requested event-horizon keys exactly once
all values finite
no GT in generation/rollout functions
```

## 9. Decision

```text
SMOKE_PASS
```

The one-event implementation is mechanically valid and exactly reproducible. The frozen 72-event audit may now be implemented.

This result does not authorize training or DAVIS access and is not evidence that independent state improves tracking accuracy.
