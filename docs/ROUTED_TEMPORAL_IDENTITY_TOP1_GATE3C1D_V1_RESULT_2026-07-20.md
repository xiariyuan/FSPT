# Route-D two-stage causal top-1 Gate 3C1D v1 result — 2026-07-20

## Formal decision

```text
COMPLETED_PASS
EXACT_REPLAY_PASS_ON_ALL_THREE_RENEWED_PARTITIONS
AUTHORIZE_GATE3C1E_CAUSAL_TOP1_FUTURE_ROLLOUT_PREREGISTRATION
```

Gate 3C1D v1 freezes a four-model causal selector trained only on the already
exposed old development pool. Candidate ranking uses minimum predicted expected
distance. A non-native action additionally requires independent 12-pixel support,
predicted positive value, and low harmful-action probability.

The checkpoint-selected policy is unchanged thereafter:

```text
support probability >= 0.30
predicted value      >= 1.0 px
predicted harm       <= 0.20
```

No renewed audit or model-validation row was used for model fitting, policy
selection, threshold adjustment, or feature redesign.

## Renewed raw-record-disjoint results

| Partition | Rows | Coverage | Action precision <=12 px | Native error | Policy error | Reduction | Video 95% CI | All-row harm | Nonnegative videos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| checkpoint 0--255 | 2,772 | 29.7258% | 66.6262% | 35.2583 px | 31.5271 px | **+3.7312 px** | [+3.4295,+4.3760] | 0.6854% | 98.0237% |
| fit-only audit 256--383 | 1,371 | 24.8724% | 62.1701% | 35.7574 px | 32.5394 px | **+3.2180 px** | [+2.9634,+4.1052] | 0.0729% | 100.0000% |
| model validation 384--511 | 1,379 | 28.0638% | 62.2739% | 34.3669 px | 30.8805 px | **+3.4864 px** | [+3.1016,+4.4182] | 0.7252% | 97.6000% |

On final renewed model validation, mean commit error falls by `3.4864 px`, or
`10.144%` relative to native. Native 12-pixel recovery is `1.2328%`; the frozen
policy output reaches `18.4917%`, an absolute increase of `17.2589` percentage
points. These are failure-row commit metrics, not complete-video TAP-Vid AJ/OA.

## Generalization conclusion

The v1 mechanism survives two independent confirmation stages without tuning:

- error-reduction confidence intervals are strictly positive on checkpoint,
  audit, and model validation;
- action precision remains above the frozen 60% confirmation gate;
- aggregate harmful actions remain below 1% of all failure rows in every stage;
- at least 97.6% of failure videos have nonnegative average commit benefit.

This resolves the v0 precision/coverage frontier at the component level. It does
not yet establish an end-to-end tracking gain because frame-15 state must still
be written back and rolled through future frames.

## Exact replay and sealed caches

| Artifact | File SHA256 | Payload SHA256 |
|---|---|---|
| checkpoint cache index | `cdbcf3ee221f37f332c80715dec885bf0a000ffd396f1381759f865bc0d0236d` | `ef8c843e5523c7f81f744a15a06d165c1463caf196a9aaa229092f85f469b634` |
| audit cache index | `837552bd0dcbb7500f983f68bb19f66d5090f7fe6da1a939c1afa8e0de0ecbb9` | `dd924f35cf91eeb31f8777056dddb15b8b3c2a616b66addb745ccb1e7a9f57f8` |
| model-validation cache index | `d091d9bf5be463a36d312f72b8dc7a76df9a054263f3abd694d0bbd92f2fa513` | `f2f4d4958bbe21acf354a1ad2f09e83da79041057013aaf0ceb5f12a0af25fb6` |
| checkpoint replay result | `a3540bd2317c256b924ce085b4911cc84f4efa75aa95c1b034ae5563e21d5388` | `fc7fce78c7474d36d02f4c92d022c482bea9aaf4a3f33d20d03799af19b0fec0` |
| audit replay result | `28ea3cd3d829167b6a98cd2e4129a68768dbb2ff0475abbbf60da0a767683f08` | `b62ccba67d42cb6d381a9f01d55c7b7a54a4455c4efd49391dfbb2f56844201a` |
| model-validation replay result | `104527b0615723858c90aa137107bac6a7307931bd99fced33af88526f9c1c29` | `5a95b02681827041f941c8ac459e23e31c64dc12ef26f46213310e04a1f10bad` |

Each replay independently reconstructs the target features and predictions. The
checkpoint replay also independently retrains all four models and reproduces the
OOF scientific digests and selected policy exactly.

## Claim boundary and next gate

Allowed claim:

```text
The frozen causal two-stage selector reproducibly reduces commit error on three
raw-record-disjoint Kubric partitions while keeping harmful intervention sparse.
```

Forbidden claim at this stage:

```text
The method improves or exceeds CoTracker3 paper AJ, delta_avg, or OA.
```

Gate 3C1E must now preregister and evaluate coordinate plus deterministic
four-level memory writeback using this frozen deployable selector. Only after a
complete-trajectory run under the same official TAP-Vid dataset, query, raster,
visibility, and evaluator contracts may the result be placed beside the
CoTracker3 paper baseline.

Calibration, final holdout, DAVIS, Kinetics, and official Kinetics 1,144 remain
locked for the new v1 route.
