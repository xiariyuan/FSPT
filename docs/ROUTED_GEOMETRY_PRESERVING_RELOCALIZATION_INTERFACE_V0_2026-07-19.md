# Route-D geometry-preserving relocalization interface v0 — 2026-07-19

## Status

```text
IMPLEMENTED_WITH_UNIT_TESTS_ADDED
TEST_EXECUTION_PENDING_PROJECT_PYTORCH_ENVIRONMENT
NOT_PREREGISTERED_AS_GATE3A_V1
NO_TRACKING_RESULT
```

Interface source hashes:

```text
implementation: 784e14d35a9b8e797800d80be4f330eba33d3a21071dd39f4a23b2f7ac350a47
tests:          65caa23f890d68ba3bba93c270c589754faee440389352652b0954baddd4ed6c
query anchor:   73b524583316716cd0ea81623fe17d7458c92b30f77f030d0e6691ecfcd1fefd
anchor tests:   4c720974ec7d49e903054506448d3dff3ed6ae69ff7d750bbf257f006a287efc
```

This is an implementation interface, not an authorized experiment. Candidate
distance gates remain intentionally unspecified until the Gate 2.5 coordinate
basin audit completes.

## Contract

The interface replaces Gate 2's learned attention pooling of 49 support tokens
with fixed token-wise geometric matching:

```text
anchor support: [row, 49, channel]
target feature: [1, channel, height, width]
output map:     [row, height, width]
```

For every candidate center, each anchor token is compared by cosine similarity
with the target token at the corresponding spatial offset. CoTracker's x-major
support ordering is explicitly converted to PyTorch unfold's y-major kernel
ordering. Candidate patches use replicate border padding, matching deterministic
state re-extraction's coordinate clamping.

Token evidence is aggregated with a frozen symmetric trimmed mean. No support
attention, channel projection, selector, action head or trainable parameter is
present. Four pyramid levels are resized to a common grid and combined by fixed
normalized weights. Independently frozen anchor maps may be combined by median,
minimum or mean; median is the proposed three-anchor default, but no anchor-bank
protocol is frozen yet.

## Tests

The source tests cover:

- exact CoTracker-to-unfold token order;
- recovery of the center of an asymmetric random patch;
- border-clamping parity;
- robustness to a corrupted support token;
- multi-level and multi-anchor shape/consensus contracts.

## Explicit nonclaims

The interface does not yet establish:

- that immutable query support is available in the current Gate 2 cache;
- that geometry-preserving maps improve candidate recall;
- that multi-anchor consensus is causal and safe;
- any candidate K, NMS radius or commit-error gate;
- any selector, state action or natural-union improvement.

After Gate 2.5 passes integrity packaging, Gate 3A v1 must separately freeze a
new immutable-anchor cache, representation controls, candidate extraction, and
recall thresholds derived only from the measured recovery basin.

## Immutable query-anchor interface

The accompanying causal-anchor helper extracts each point's four-level feature
and support memory at its original query frame and coordinate. Before calling
CoTracker's `get_track_feat`, every feature-pyramid level is physically
truncated after the latest requested query frame. Consequently, changing any
later observed feature cannot change the immutable anchor.

This helper defines only the original-query anchor. “Last trusted” and
“pre-risk” anchor update rules remain deliberately unimplemented because their
write conditions require a separately frozen causal protocol.
