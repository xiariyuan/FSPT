# Route-D full-population causal entry Gate 3C1F0 v0 result — 2026-07-20

## Formal decision

```text
COMPLETED_PASS
EXACT_REPLAY_PASS
AUTHORIZE_GATE3C1F0_RAW_DISJOINT_FULL_POPULATION_DATA
```

## Frozen entry rule

```text
entry probability >= 0.93
native joint visibility-confidence probability <= 0.02
```

Candidate generation remains forbidden until this rule accepts a point.

## Five-fold source-video OOF

```text
rows:                    376
failure / clean:         164 / 212
AUC / AP:                0.8514 / 0.8090
action rows:             24
coverage:                6.3830%
failure recall:          14.6341%
clean false-apply rate:   0.0000%
action precision:       100.0000%
```

## Independent fit-only validation

```text
rows:                    160
failure / clean:          80 / 80
AUC / AP:                0.7570 / 0.7864
action rows:             12
coverage:                7.5000%
failure recall:          15.0000%
clean false-apply rate:   0.0000%
action precision:       100.0000%
```

Every preregistered OOF and validation gate passes.

## Replay integrity

Primary and fresh-process replay exactly match on:

- reconstructed 130-D train and validation features;
- labels, native joint probabilities, and source-video groups;
- all five-fold OOF probabilities;
- final validation probabilities;
- train and validation point-record membership;
- frozen operating point;
- complete scientific payload.

The primary and replay joblib files are also byte-identical in this run, but
scientific equality is defined by the pinned digests rather than pickle bytes.
Only the primary bundle is authorized downstream.

```text
primary bundle SHA256:
ac25440f3f15e15033e73f783af172796afb218c1964c736d758928fb5fb003f

primary result file SHA256:
369bddbaa459844a499c39a87b574c09fc9dd6dead19f154196edc4e7badc9dd
primary payload SHA256:
ab6697077e69d8d6fd0c74f18ee5a21cfb12967626bef778ba02e4d9e2805b8f

replay result file SHA256:
43f457abde9ba563322864e8f802fa0a156d6a952a3e23d403a69f06c9da28df
replay payload SHA256:
0957fb7f74b0576ef28940b52a746d578b4eda1494558aee79b63a4010e9178e

scientific payload SHA256:
f6a3f3f2fa6e28589cc9fe033bf917af4e8ec87cfc5d24f0cacda10ab38536d7
```

## Claim boundary

This result establishes that a conservative, low-cost, fully causal entry model
can identify a small high-risk subset without applying to any clean row in the
available OOF or fit-only validation populations.

It does not establish complete-video generalization. The 16-video design pilot
remains prior-exposed and its AJ interval crosses zero. The next authorized step
is therefore a new raw-record-disjoint Kubric population that excludes all 1,024
raw identities used by Gate 3C0 and Gate 3C2.

DAVIS, Kinetics, final holdout, and official Kinetics 1,144 remain locked.
