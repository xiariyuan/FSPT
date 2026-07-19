# Route-D state-reinstatement coordinate basin Gate 2.5 result — 2026-07-19

## Formal status

```text
COMPLETED_VALID
INDEPENDENT_REPLAY_EXACT
REPORT_EMPIRICAL_REINSTATEMENT_BASIN_AND_PREREGISTER_GATE3A_V1
```

Gate 2.5 completed on the design-exposed source indices `0--7`. It trained no
parameters, did not load the learned Gate 2 checkpoint, and did not read source
indices `8--63`, DAVIS, Kinetics, calibration, or any final holdout.

The central result is unambiguous:

> Deterministic four-level state re-extraction has a large and continuous
> recovery basin. With native visibility/confidence unchanged, every frozen
> radius from 0 through 12 input pixels passes all downstream utility gates.

The learned Gate 2 failure was therefore not caused by a requirement for
near-exact subpixel localization or teacher probability state. It was caused by
failure to produce a sufficiently accurate target location from the current
representation.

## Integrity

```text
videos:                         8 / 8
failure points:                 49
primary/replay full JSON:       exact
minimum track-feature cosine:   0.9999997616
minimum track-support cosine:   0.9999994040
cache index SHA256:             a10ca52650ef09d4e354ff548179e0bb6ac04ca822bd5df9df4b538cbd2f7d3d
cache tensor digest:            a4d732b2af5ebe5cbca499fd016193f978cac793a0f0b8ccfcf2b3c810179efc
primary report SHA256:          4bbf2399aaaf3e27bfcb84927fde256688a53f26d3fddc6d111f829973f5453a
replay report SHA256:           4bbf2399aaaf3e27bfcb84927fde256688a53f26d3fddc6d111f829973f5453a
generated summary SHA256:       69ba0ae6a7fa6fc6961b16fbce3f15c99770474945fa42cafde41ba1dd7e91c1
summary payload SHA256:         2ea608b4b69258f12590a5edfe03c902493392c962d32d5b92d873caad782b8e
```

The float32 design-only teacher coordinate prevents a quantized nonzero
displacement at radius zero. The center feature/support parity confirms that the
audit action reconstructs the intended fresh-query memory.

## Primary native-probability result

Visibility and confidence remain native in this primary variant. Only the
commit coordinate and four memory levels are replaced.

| Nominal radius | Future error reduction | Utility gain | Severe-rate reduction | Positive rows | Utility CI lower | Gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 0 px  | +56.0430 px | +0.6154 | +0.8761 | 1.0000 | +0.5000 | pass |
| 1 px  | +55.6710 px | +0.5535 | +0.8745 | 1.0000 | +0.4656 | pass |
| 2 px  | +55.1667 px | +0.4638 | +0.8729 | 1.0000 | +0.4066 | pass |
| 4 px  | +53.8951 px | +0.3577 | +0.8555 | 1.0000 | +0.3275 | pass |
| 8 px  | +51.0527 px | +0.2348 | +0.8196 | 1.0000 | +0.2198 | pass |
| 12 px | +47.9553 px | +0.1722 | +0.7597 | 1.0000 | +0.1638 | pass |
| 16 px | +44.4651 px | +0.0876 | +0.3834 | 0.9949 | +0.0741 | fail utility mean |

At radius zero, future mean error is `6.5175 px`, threshold utility is `0.6160`,
and severe-16px rate is `0.1210`. The native future mean error for these rows is
approximately `62.5604 px`.

All eight directions remain positive at 12 px. The weakest direction is
northwest with utility gain `+0.1441` and mean error reduction `+47.0201 px`,
still above every frozen gate. The result is therefore not driven by one easy
perturbation direction.

## Probability control

Writing exact fresh-query teacher visibility/confidence does not enlarge the
basin:

```text
native-probability largest contiguous passing radius:  12 px
teacher-probability largest contiguous passing radius: 12 px
```

At radius zero, teacher probability changes utility gain only from `+0.6154` to
`+0.6188` and future error reduction from `+56.0430` to `+56.0629 px`.
Probability restoration is therefore not the limiting mechanism for the next
candidate audit.

## Scientific decision

Gate 3A v1 may now use `12 px` as the primary empirically justified
state-reinstatement recall radius, while also reporting nested recall at
`4 px` and `8 px`. The old `4/8 px` thresholds remain useful diagnostics but no
longer define the actual recovery boundary.

Gate 3A v1 must test representation support before selector training:

1. retain frozen Gate 2 pooled-native logits as the M0 control;
2. add the immutable original-query 49-token geometry map;
3. freeze every candidate set before reading the commit teacher coordinate;
4. select the nearest frozen candidate only for candidate-support oracle audit;
5. apply the same deterministic full-state action and continuation;
6. keep source indices `48--63` and all external datasets locked.

This result authorizes only a separately preregistered fit-only representation
audit. It does not authorize selector training, action training, model
validation, external evaluation, or a paper-level performance claim.
