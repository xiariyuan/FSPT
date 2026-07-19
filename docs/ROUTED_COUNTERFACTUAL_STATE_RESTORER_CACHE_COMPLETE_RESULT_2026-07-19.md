# Route-D CSRR complete teacher-cache result — 2026-07-19

## Formal decision

```text
ALLOW_GATE2_RESTORER_TRAINING
```

Both frozen Gate 2 cache partitions are complete, all 40 sidecars have been
reloaded and verified, and the four fixed anchors match both independent smoke
builds at the nested tensor-digest level.

This result authorizes only fit-only CSRR training on source indices `8–31` with
checkpoint selection on `32–47`. It does not authorize reading model-validation
`48–63` or any later dataset.

## Complete partitions

### Gradient training

```text
source indices:                    8–31
videos:                            24 / 24
failure rows:                      164
clean no-op rows:                  212
index SHA256:                      ab93e070adc4d623aeada811a88b16def3f4bb2137946f3728327806ae331517
combined sidecar SHA256:           f03951aab00d72e4d76814946ccf8b2e2818f77e1dbbd8f06ab5c9c6cca11d92
combined tensor digest:            a3617f21239e45c232a6af420fe475a2723538b57fa8e6f120b00ced6ae58df9
on-disk size:                      approximately 356 MiB
```

Sources `22`, `25`, and `31` contain no point meeting the frozen `>=16 px`
natural-failure threshold. They retain their valid clean rows; no failure row was
backfilled from the excluded `(4,16)` interval.

### Fit-internal validation

```text
source indices:                    32–47
videos:                            16 / 16
failure rows:                      80
clean no-op rows:                  80
failure-support videos:            16 / 16
clean-support videos:              16 / 16
index SHA256:                      4216898d32c65a60360087bd7c4026d2db7d6877bfa06a934932a429112e4bef
combined sidecar SHA256:           ddafd347ed5868f67c104f548f053e07be8224ca34ee230ab84256c1aad315fa
combined tensor digest:            54bd9d4eb547790b9183f0acf6efd140db40e4c0e24ab35aa26b016b2cf6a49e
on-disk size:                      approximately 229 MiB
```

The preregistered validation data-support gates all pass:

```text
failure points >= 32:              80
failure videos >= 8:               16
clean points >= 32:                80
clean videos >= 8:                 16
```

## Quantization

```text
train maximum absolute error:      0.0004751682
train minimum cosine:              0.9999960661
validation maximum error:          0.0004826784
validation minimum cosine:         0.9999971986
frozen gates:                      <=0.001 and >=0.99999
```

The float32 exact rollout state is not quantized. These measurements apply to the
float16 CSRR model/teacher view.

## Fixed anchor replay

For sources `8`, `31`, `32`, and `47`, the formal cache tensor digest equals both
the primary and independent replay smoke digest. All four checks pass.

This establishes that partition mode changes only the sidecar partition metadata,
not the generated native state, row identities, teacher state, feature pyramid,
or scoring tensors.

## Full verification

All formal checks pass:

```text
train membership complete and exact:             true
validation membership complete and exact:        true
all 40 sidecars reload and tensor hashes verify: true
train/validation identities disjoint:             true
four fixed anchor digests exact:                  true
quantization gates:                               true
validation data-support gates:                    true
```

Canonical summary SHA256:

```text
b8a4750ad84cf080bde6d5d30c6e58dc12ec42b97d23e7eb3c945903ccccf4ec
```

## Authorized training boundary

CSRR training may now use only:

```text
train cache:                    8–31
fit-internal validation cache:  32–47
seed:                           17
frozen architecture/loss/gates from Gate 2 config
```

The training runner must independently reproduce the selected checkpoint and all
validation outputs in a fresh process. If any fit-internal scientific or safety
gate fails, the branch stops before model-validation.

## Locked data

Source indices `48–63`, calibration, final holdout, DAVIS, and Kinetics remain
unread. The official 1,144-video Kinetics result was not rerun or retuned.
