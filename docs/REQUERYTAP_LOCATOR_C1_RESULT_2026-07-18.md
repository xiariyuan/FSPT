# ReQueryTAP Single-Frame Locator C1 Result — 2026-07-18

## Status

The preregistered fit-only C1 training is complete.

```text
best epoch:                 4
train pairs:                42,615
full dev pairs:             17,160
trainable locator params:   230,018
```

## Final dev result

```text
mean coordinate error:      52.7701 px
median coordinate error:    31.0888 px
hit@4:                       1.38%
hit@8:                       5.17%
hit@16:                     18.73%
hit@32:                     50.96%
```

The model substantially improved over raw cosine in four scenes, but failed to
reach the frozen absolute localization requirements and regressed on one scene:

```text
animal:                                   +56.73 px median reduction
ani13_new_f:                              +51.26 px
r0_new_f_:                                +34.69 px
r2_new_:                                  +34.67 px
scene_recording_20210910_S05_S06_0_ego2: -13.09 px
```

Frozen gates:

```text
median error <= 12 px:                FAIL
hit@16 >= 60%:                        FAIL
positive reduction 5/5 scenes:        FAIL
bootstrap reduction CI lower > 0:     PASS
worst reduction >= -2 px:             FAIL
```

Decision:

```text
STOP_SINGLE_FRAME_IDENTITY_LOCATOR
DO NOT RUN PREDICTED-RESPAWN C2
NO MODEL-VALIDATION OR LOCKED DATA
```

## Interpretation

A learned projection of one frame-0 point feature is not a sufficiently stable
exact-point identity representation.  It reduces gross ambiguity on several
scenes but suffers severe cross-scene/domain failure and remains far too coarse
for safe token respawn.

The only allowed redesign is a fit-only information audit of multi-view
visible-tracklet identity memory.  It must establish that multiple reliable
pre-occlusion observations add cross-scene localization information before any
new locator is trained.
