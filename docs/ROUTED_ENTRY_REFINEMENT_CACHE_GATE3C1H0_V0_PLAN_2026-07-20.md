# Route-D Entry Refinement Cache Gate 3C1H0 v0 Plan — 2026-07-20

## Motivation

Gate 3C1F2 produced only 89 actions across 5,596 eligible points. Gate 3C1G1 showed that broad action-frame visibility replacement does not generalize. A post-failure activation oracle also shows that perfect gating of all `native=occluded -> modified=visible` transitions can improve action-video AJ by at most `+0.1107` points, below the frozen `+0.15`-point requirement. Visibility calibration alone is therefore insufficient; safe action coverage must increase.

The original Gate 3C1F0 entry model was trained on extreme failure/clean rows. On the full population it admitted many previously unseen `other` rows. Gate 3C1H0 builds a realistic full-population entry-development cache from the already exposed third Kubric population.

## Frozen population

```text
videos: 128
eligible points: 5,596
query frame: < 8
feature dimension: 130
current frozen entry rows: 537
```

This population is the completed Gate 3C1F2 confirmation population and is now development-exposed. It remains raw-record-disjoint from the earlier 1,024 Kubric identities. No fourth population or external data is read.

## Exact parent reproduction

Before any row is stored, every video must reproduce the committed Gate 3C1F2 primary sidecar for:

```text
eligible_indices
entry_features
entry_probability
entry_mask
native_coordinates
```

Any mismatch stops the cache build.

## Stored tensors

Causal model inputs:

```text
130-D observed-history entry features
source/video identity
point identity
query frame
frozen entry probability
native joint visibility-confidence
current entry mask
```

Labels/evaluation tensors, stored separately:

```text
commit visibility
future visible-frame count
future mean native error
failure / clean / ambiguous / other category
evaluable flag
```

The category contract is unchanged from Gate 3C1F2:

```text
failure:   commit visible, >=4 future visible frames, mean future error >=16 px
clean:     commit visible, >=4 future visible frames, mean future error <=4 px
ambiguous: evaluable but between 4 and 16 px
other:     commit invisible or fewer than 4 future visible frames
```

GT and future-derived values are prohibited from model features.

## Output contract

```text
output root:
outputs/routeD_entry_refinement_cache_gate3c1h0_v0_20260720

expected videos: 128
expected rows: 5,596
expected feature dim: 130
```

Every sidecar and tensor is hash-addressed. The final index must verify all sidecars, payloads, tensor digests, support counts, and locked-data flags.

## Decision

A complete and exact cache authorizes only:

```text
AUTHORIZE_GATE3C1H1_ENTRY_REFINEMENT_PREREGISTRATION
```

It does not authorize a new entry threshold, candidate generation, a fourth raw-population confirmation, DAVIS, Kinetics, final holdout, or official Kinetics 1,144.
