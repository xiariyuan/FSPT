# V9-A Route Closure Matrix

Date: 2026-07-11

| Route | Primary question | Data | Fair baseline | Gate/result | Final decision |
|---|---|---|---|---|---|
| V9-A4.5 | Can a bounded residual reranker improve the frozen candidate ordering without damaging safe rows? | PointOdyssey synthetic holdout; ani held out after sequence split | Frozen teacher C1 ranking on the same rows/candidates | Heldout student ranking is worse than the frozen teacher. | **Fixed-topK residual reranking adaptation is closed.** |
| V9-A5.0 | Does causal prior state contain useful temporal information, and is self-state error propagation the blocker? | PointOdyssey canonical 4,878-row visible candidate pool | Frozen teacher/current-frame selection | Sampled self-state fails; causal past-oracle/past-GT state has large sequence-consistent headroom. | **Temporal information exists, but single-state error propagation blocks deployable use.** |
| V9-A5C.0 | Is useful <=4px candidate recall materially higher at fused K64 than at fused K16? | PointOdyssey canonical 4,878-row visible candidate pool | Fused C1 top16 on the same correlation map | All three sequences pass; hard-row headroom is large. | **Synthetic upstream candidate availability exists, but does not establish deployable selection.** |
| V9-A5.1a | Can fixed-K16 temporal path scoring preserve or improve current-frame candidate reachability? | 9 PointOdyssey clips; 27,648 query-frame rows | Current-frame teacher top1/topB at equal capacity | Both deterministic and same-capacity reachability fail. | **The exact fixed-K16/raw-C1/shared-state beam configuration is closed.** |
| V9-A5.1b | Can a frozen fused C1 candidate drive a useful singleton-conditioned C2/head refined coordinate? | 9 PointOdyssey clips; 27,648 query-frame rows | Official final TrackOn2 output with risk-gated fallback | GT-only refined candidate oracle passes strongly; frozen score-top1 remains unreliable. | **Candidate refinement has oracle headroom only; it authorizes a temporal candidate-set audit, not deployment.** |
| V9-A5.1c | Does finite-window history-preserving beam pruning add value beyond a same-frame same-capacity candidate set? | 9 PointOdyssey clips; 27,648 query-frame rows | Same-frame frozen-score topB candidate oracle at matched B/K | System-level min(official,beam) oracle passes versus official; deterministic top1 fails. | **Capacity-matched review reverses the temporal claim: primary B4 is worse than frame-local top4.** |
| V9-A5.2 | Does a complete independent 432-query TrackOn2 state produce useful future outputs beyond a shared-state capacity-2 control? | 72 frozen first-risk events across 9 PointOdyssey clips; horizons 1/4/8 | Shared-state capacity-2 oracle: official final plus current-state score-top1 refined output | State and C1 sets diverge, but horizon-8, pooled, early8, and useful-novelty gates fail. | **The tested independent score-top1 singleton state branch is closed.** |
| V9-A6.0 | Are K16-miss/K64-hit candidates close enough to the K16 score boundary for a small identity-preserving residual? | 760 visible K16-miss/K64-hit opportunities from the 4,878-row canonical pool | Original fused top16 boundary; GT-directed target boost is an optimistic magnitude lower bound | Only 67/760 opportunities convert at 2*g_ref; every gate fails. | **Small bounded pre-topK correlation residual promotion is closed; the evaluated pool is visible-only.** |

## Project conclusion

Substantial candidate-level oracle headroom exists beyond fused top16, but the tested fixed-pool reranking, shared-state temporal, independent-state temporal, and small bounded pre-topK score-shift routes do not convert it into sequence-consistent deployable gains under the committed synthetic gates.

This matrix is a project evidence closure, not a universal impossibility theorem for all tracking architectures.
