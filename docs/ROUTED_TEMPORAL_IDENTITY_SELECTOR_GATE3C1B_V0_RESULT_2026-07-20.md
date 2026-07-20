# Route-D query-closure identity selector Gate 3C1B v0 result — 2026-07-20

## Formal status

```text
COMPLETED_PASS
AUTHORIZE_GATE3C1C_FUTURE_ROLLOUT_PREREGISTRATION
```

The frozen zero-parameter query-closure identity selector passed both the
independent checkpoint-selection gate and the one-shot fit-only internal audit,
with exact fresh-process replay on both partitions.

## Frozen selector

Candidate zero remains mandatory. Eight non-native candidates are retained by:

```text
score = -2 * z(reverse cycle error at the query frame)
        + z(query identity cosine at the query frame)
        + z(mean query identity cosine over frames 0--15)
        + z(minimum query identity cosine over frames 0--15)
```

The rule has no trainable parameter, optimizer, loss, epoch, learned residual,
or checkpoint-dependent hyperparameter.

## Checkpoint-selection result — indices 384--447

| Metric | Static M1 top-8 | Query-closure identity | Difference |
|---|---:|---:|---:|
| 12 px retained support | 58.7956% | 67.5119% | +8.7163 points |
| 8 px retained support | 35.4992% | 44.0571% | +8.5578 points |
| median minimum distance | 10.2961 px | 8.8242 px | -1.4719 px |

- videos: 64;
- videos with natural failures: 63;
- failure rows: 631;
- equal-video support-gain mean: +8.2868 points;
- equal-video bootstrap 95% CI: [+4.1980, +12.4154] points;
- nonnegative-video fraction: 79.3651%;
- exact replay: pass;
- formal decision: `AUTHORIZE_GATE3C1B_FIT_ONLY_AUDIT_448_511`.

Cycle-only reaches 60.2219% and identity-only reaches 61.0143%, so neither
single component explains the 67.5119% primary result.

## One-shot fit-only internal audit — indices 448--511

| Metric | Static M1 top-8 | Query-closure identity | Difference |
|---|---:|---:|---:|
| 12 px retained support | 58.5657% | 64.0106% | +5.4449 points |
| 8 px retained support | 36.5206% | 42.3639% | +5.8433 points |
| median minimum distance | 10.2883 px | 9.0194 px | -1.2689 px |

- videos: 64;
- videos with natural failures: 63;
- failure rows: 753;
- equal-video support-gain mean: +8.1055 points;
- equal-video bootstrap 95% CI: [+2.8084, +13.5213] points;
- nonnegative-video fraction: 73.0159%;
- exact replay: pass;
- formal decision: `AUTHORIZE_GATE3C1C_FUTURE_ROLLOUT_PREREGISTRATION`.

Cycle-only falls below the static rule at 55.7769%; identity-only reaches
60.1594%. The full closure-plus-identity interaction remains necessary on the
one-shot audit.

## Integrity and isolation

- checkpoint cache exactly covers 384--447 with 64 validated sidecars;
- audit cache exactly covers 448--511 with 64 validated sidecars;
- the audit cache builder verified the checkpoint exact-replay authorization
  before opening index 448;
- all selected-index, primary-score, method-metric, and video-record digests
  replay exactly on both partitions;
- original model-validation indices 48--63 remain unread;
- calibration, final holdout, DAVIS, Kinetics, and official Kinetics 1,144
  remain unread or not rerun.

## Claim boundary

Gate 3C1B proves that the causal rule preserves a better nine-candidate
frame-15 shortlist on two independent development partitions. It does not prove
which retained coordinate can be selected without a teacher, that committing a
coordinate improves frames 16--23, that memory writeback is useful, or that any
external tracking metric improves.

The next authorized action is only to preregister Gate 3C1C future rollout with
a frozen teacher-nearest shortlist oracle and explicit coordinate-only versus
coordinate-plus-memory state-action controls.

Machine-readable authority:

`docs/generated/ROUTED_TEMPORAL_IDENTITY_SELECTOR_GATE3C1B_V0_RESULT_2026-07-20.json`
