# V9-A5.1a Fixed-K16 Shared-State Beam Baseline Freeze

Date: 2026-07-10

## 1. Purpose

This document freezes the independently completed full-stream K16 beam audit as `V9-A5.1a`.

It is a negative baseline, not the final closure of all temporal multi-hypothesis directions.

## 2. Original execution provenance

```text
original worktree: /gemini/code/FSPT_v9a51_clean
original branch: v9a51-fullstream-beam-20260710
original base HEAD: 7c24367cf8aeb1467946d2ced3b294299032538e
```

The original artifacts were copied byte-for-byte into the current branch and renamed for stable classification. Their contents were not edited.

## 3. Frozen artifact hashes

```text
script SHA256:
4c49df7c743ba14229f30786d13e2776348f9bb66f14aa40b0b6215c94bc5ee8

result JSON SHA256:
902952433e00ed91f584a4ce74efe6469bd6ee87b23ff1f7a69426e20bf5d0e5

result NPZ SHA256:
6606afbebe56b7737bb8b50758cfc206815321b2ae22f958e48ec136f9a519cc
```

The copied script hash exactly matches the hash stored in the original result JSON.

## 4. Integrity status

```text
processed query-frame rows: 27,648
GT-visible rows: 20,218
RGB frames hash verified: 864
official p/v/q parity max_abs: 0
sampled candidate-set parity: <= 1e-6
sampled teacher score max_abs: 0
sampled teacher top1 match: 1.0
beam update receives no GT or error input
all outputs finite
saved NPZ reproduces reported primary means
```

## 5. Result after independent frame-0 exclusion

Frame 0 is not responsible for the negative result.

For visible rows with `frame_tau > 0`:

```text
teacher C1 top1 mean:              9.4394 px
beam4 motion+latent top1 mean:      9.8841 px
mean difference:                  +0.4447 px
better / worse / equal:            2879 / 5158 / 11893

teacher top4 oracle mean:           6.3934 px
beam4 motion+latent oracle mean:    7.5775 px
mean difference:                  +1.1841 px
```

The fair same-capacity width-8 comparison is also negative:

```text
teacher top8 oracle mean:           5.3604 px
beam8 motion+latent oracle mean:    6.2353 px
mean difference:                  +0.8749 px
```

## 6. Valid conclusion

The following tested configuration fails:

```text
fixed frozen C1 top16 candidates
shared official TrackOn2 query/memory proposal state
raw C1 patch-center coordinates as beam output
teacher/motion/previous-latent ordinal Borda
path cost accumulated from frame 0 without forgetting
one retained history per current candidate index
primary comparison against C1 teacher top1
```

Do not repeat this exact configuration or only increase its beam width.

## 7. Limits of the conclusion

This baseline does not test:

```text
risk-gated fused K64 candidates
candidate-conditioned C2/offset refinement
official-final fallback on non-risk frames
distinct incoming histories at the same current coordinate
finite-window path costs
independent per-hypothesis query/memory states
```

V9-A5C.0 later showed large hard-row fused K16-to-K64 recall headroom, so this K16 baseline cannot close the upstream proposal-expansion route.

## 8. Next gate

The next experiment is `V9-A5.1b candidate-conditioned downstream refinement`.

It must first determine whether one frozen C1 candidate hypothesis can drive a meaningful frozen C2/prediction-head refined coordinate. No new beam should be implemented unless the risk-gated refined candidate oracle improves the official final tracker under the predeclared three-sequence and clip-block gates.
