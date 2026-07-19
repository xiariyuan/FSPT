# Route-D CSRR Gate 2 failure diagnostic — 2026-07-19

## Diagnostic decision

```text
STOP_ACTION_ONLY_FIX
AUTHORIZE_FIT_ONLY_DISCRETE_HYPOTHESIS_SUPPORT_AUDIT
```

This diagnostic uses only the already exposed Gate 2 fit partitions and the
frozen primary/replay outputs. It does not read source indices `48–63` or any
external benchmark.

## 1. The learned restoration signal is real but rarely becomes accurate

On 80 fit-internal natural failure rows, forced learned full-state restoration
reduces future mean error on 56 points and harms 24 points. The point-equal
error-reduction distribution is:

```text
mean:                 +7.91259 px
median:              +11.07840 px
p10:                 -21.66518 px
p25:                  -5.53428 px
p75:                 +26.35456 px
p90:                 +39.08367 px
minimum:             -86.41399 px
maximum:             +56.55078 px
harm <= -16 px:       10 / 80
positive videos:      11 / 16
```

Threshold utility is much sparser:

```text
mean utility gain:    +0.04527
median utility gain:   0.00000
positive:              17 / 80
zero:                  62 / 80
negative:               1 / 80
maximum:               +0.60000
```

The model often moves a track substantially closer while leaving it outside the
strict `1/2/4/8/16 px` accuracy thresholds. This explains why mean error nearly
passes while utility remains far below its gate.

## 2. Source 47 is important but not the primary blocker

Source 47 contributes one catastrophic row:

```text
full-state error reduction:       -86.41399 px
full-state utility gain:           -0.02857
```

Excluding that already observed row changes the aggregate to:

```text
mean-error reduction:              +9.10659 px
threshold-utility gain:            +0.04620
severe-rate reduction:             +0.15032
```

The narrow mean-error and severe-rate gates would pass, but utility would still
remain far below `+0.12`. The branch therefore cannot be repaired by treating
source 47 as an isolated outlier.

## 3. An action-only fix is mathematically insufficient

The Gate 2 union contains 80 failure and 80 clean rows. On the 80 failure rows,
the sum of positive full-state utility gains is `3.650000043`; the only negative
utility contribution is `-0.028571431`.

Even a non-causal GT oracle that applies full-state restoration exactly on rows
with positive future utility has the following upper bound:

```text
perfect oracle union utility gain:       +0.022812500
Gate 2 required union utility gain:       +0.050000000
```

For comparison, applying to all failure rows gives `+0.022633929`. Selective
action can remove the single negative utility row, but it cannot create the
missing accurate restorations.

The same perfect oracle can achieve a union mean-error reduction of
`+7.255485 px`, so the impossibility is specific to accurate-track utility, not
large-error reduction.

Therefore no classifier, threshold calibration, or risk head built on the
current forced-restoration outcomes can pass the frozen union utility gate.
Restoration quality must improve first.

## 4. The Gate 2 action supervision is misaligned

`apply_target=1` is assigned when native future error on frames `16–23` is at
least `16 px`; `apply_target=0` is assigned when it is at most `4 px`. The model
only observes frames `8–15`.

Consequences:

1. the target is defined by future native failure rather than by whether the
   learned action is beneficial;
2. all 80 failure rows are labelled positive although learned full-state action
   worsens mean error on 24 of them;
3. clean/failure separability must be inferred from causal pre-commit evidence,
   but no explicit counterfactual benefit or harmful-risk target is supplied.

The gate dynamics support this diagnosis:

```text
epoch -1: failure apply 0.8500, clean false apply 0.8625
epoch  0: failure apply 0.0750, clean false apply 0.0750
epoch  1 onward: both 0.0000
```

Action BCE remains near an uninformative binary baseline throughout training:

```text
epoch 0:   0.69385
epoch 1:   0.69104
epoch 6:   0.67839
epoch 10:  0.66883
```

The thresholded no-op collapse occurs while the restorer continues to improve,
so the two failures must be handled as separate stages.

## 5. Current matching/readout is the primary restoration bottleneck

Gate 1 exact fresh-state transplant provides the reference mechanism:

```text
native future error:               60.71294 px
GT fresh full-state error:           5.85432 px
GT fresh full-state utility:         0.63849
full-state utility gain:             0.63804
```

Gate 2 learned full-state restoration reaches only:

```text
native future error:               55.36197 px
learned full-state error:          47.44938 px
learned full-state utility:         0.04719
full-state utility gain:            0.04527
```

The complete-state mechanism remains powerful when the commit location/state is
correct. The learned branch fails mainly before that point.

Two v0 design choices are high-risk:

1. all 49 native support tokens are attention-pooled into one query vector before
   full-frame matching, discarding support geometry and multi-token agreement;
2. a global softmax expectation over all `64 x 64` cells converts a potentially
   multimodal map into one coordinate, which can lie between valid peaks.

Gate 3A therefore tests discrete proposal support before introducing another
learned selector or action policy.

## 6. Required stage ordering

```text
Gate 3A: frozen top-K candidate-support audit
    pass -> Gate 3B discrete candidate selector training
    fail -> redesign support query / temporal matching

Gate 3B forced full-state gates pass
    -> only then construct counterfactual beneficial/harmful action labels
    -> train a separate risk-controlled action policy
```

Joint restorer/action training and action-only threshold tuning are not
authorized by this diagnostic.
