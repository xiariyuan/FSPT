# Route-D geometry representation audit Gate 3A v1 result — 2026-07-19

## Formal status

```text
COMPLETED
INTEGRITY_PASS
STOP_CURRENT_CANDIDATES_AND_REDESIGN_TEMPORAL_IDENTITY_MATCHING
```

No representation passed all preregistered gates. Model-validation source
indices 48--63 and all external datasets remained unread.

## Integrity

```text
preregister commit:                    8b28fdc
config SHA256:                         817d194d9541394ca953852cd4eb5cf0556b4662cb4963696f0ba8c4cb44d478
causal-input index SHA256:             09221a787a358a1a629896db92bfdad2df64685f0a8580182e11281a09fbb55a
causal-input combined tensor digest:   52192852b5bde4cc736804f5bfcf31dd6b8c626c332448f59c22e9c0dcf8189d
candidate combined digest:             f730242abc7ff6b015a8a6d6b0fa34c92852e84c5581afce2c47ba597171d53a
candidate qualification SHA256:        e373d6751cb02961d98eaec6f73b17a578fd6f6e504fce5c881c45a051e09e8e
primary report SHA256:                 844ae8b5749066eef6878a2af8937fc9a436453ffada5a9e8c6f0fea5eea533b
replay report SHA256:                  844ae8b5749066eef6878a2af8937fc9a436453ffada5a9e8c6f0fea5eea533b
summary payload SHA256:                be25d4906af6c178dc225d449e867e9aa72e3971bc24816ce352ca90ddc9a36d
```

The allowlisted causal-input cache contains 16 videos and 80 frozen failure
points. All 16 sidecars were checked to contain no teacher/future tensor key.
Primary and independent candidate caches are nested-exact for all tensors and
metadata. The complete primary and replay teacher-audit JSON reports are exactly
equal.

Two failed GPU bootstrap attempts occurred before process initialization. They
produced no candidate or teacher result and did not change the config. Completed
runs used the frozen preregistration unchanged.

## Formal result

| Representation | Recall <=12 px | Median commit error | Future error reduction | Utility gain | Utility CI lower | Positive points | Severe-rate reduction | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| M0 pooled-native Gate 2 | 0.3750 | 15.2138 px | +30.6952 px | +0.14246 | +0.07815 | 0.9625 | +0.43103 | no |
| M1 geometry-native | 0.3875 | 16.7337 px | +28.8307 px | +0.13133 | +0.08627 | 0.9875 | +0.43259 | no |
| M2 geometry-immutable-query | 0.3875 | 16.7337 px | +28.8307 px | +0.13133 | +0.08604 | 0.9875 | +0.43259 | no |

Every representation passed every future-rollout gate but failed both commit
support gates:

```text
required recall within 12 px: >= 0.75
observed:                       0.3750--0.3875

required median commit error: <= 8 px
observed:                      15.2138--16.7337 px
```

The large future gains do not rescue the gate. They are produced by a
teacher-nearest oracle over a weak candidate set and therefore cannot establish
a usable causal selector.

## What the result rules out

1. The Gate 2 global softmax expectation is not the only problem. Its top-eight
   discrete modes also have insufficient recovery-basin coverage.
2. Retaining the 7 x 7 support geometry does not fix top-eight candidate support.
3. Re-extracting an "immutable query" support is not an independent memory
   intervention in this implementation.
4. Training a selector over the current nine candidates is not authorized. A
   selector cannot select a missing candidate.

## Post-gate capacity diagnosis

This section was computed only after the formal decision and is explicitly not a
preregistered pass result. It is used only to choose the next fit-only design.

### M1 and M2 are effectively the same representation

```text
M1/M2 score-map cosine mean:             0.999999974
M1/M2 score-map maximum absolute diff:   0.00000644
native/query support cosine mean:        0.999999455
native/query support cosine minimum:     0.999998927
```

CoTracker's native online support is already essentially the original query
support. M2 therefore duplicates M1 up to quantization/re-extraction noise. The
next route must add genuinely new temporal evidence, not another name for the
same support tensor.

### Correct regions exist deeper in the geometry-map ranking

| Nonnative top-K | M0 recall <=12 | M0 median | M1 recall <=12 | M1 median |
|---:|---:|---:|---:|---:|
| 8 | 0.3750 | 15.2138 px | 0.3875 | 16.7337 px |
| 16 | 0.5125 | 11.8592 px | 0.5625 | 11.2053 px |
| 32 | 0.6125 | 9.6894 px | 0.6500 | 8.7665 px |
| 64 | 0.7250 | 6.8834 px | 0.7875 | 7.6683 px |

At 64 nonnative proposals M1 crosses both original commit-support thresholds.
This does not authorize a 65-way selector: broad spatial coverage and an oracle
choice can inflate recall while making causal selection harder. It does show
that the next problem is primarily ranking/identity disambiguation over a broad
coarse pool, rather than complete absence of correlation support.

## Next authorized step

Build a fit-only temporal identity feasibility gate:

1. freeze the M1 top-64 coarse proposal bank;
2. for each proposal, use only frames 0--15 to form a candidate-conditioned
   temporal tracklet;
3. score whether the tracklet returns to the point's legitimate original query
   or another pre-occlusion reliable anchor;
4. retain eight proposals by that temporal score;
5. audit 12-pixel support before any learned selector or future rollout.

The first implementation should be a weight-free reverse-cycle consistency
control using frozen CoTracker3. It directly tests the missing temporal identity
mechanism and avoids adding another learned head before causal signal is proven.

No additional dataset or pretrained weight is needed yet. If the weight-free
temporal gate cannot materially raise top-eight support on fit-only data, then
the next escalation should be a genuinely independent appearance encoder or
segmentation-conditioned identity representation; only at that point should new
weights be requested.
