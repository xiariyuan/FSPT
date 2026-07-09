# Paper Draft: Limitations and Future Work — 2026-07-04

## Limitations

The main limitation of ReEntry is that re-entry improvement does not automatically imply global benchmark improvement under every query protocol. Under DAVIS first/input evaluation, ReEntry improves AJ, OA, and AJ_RD over the CoTracker3 offline base. However, under the stricter DAVIS strided/original protocol, the default ReEntry variants improve AJ_RD while reducing standard AJ/OA. This shows that aggressive visibility correction can introduce false positives or poorly timed visible predictions that are penalized by global TAP-Vid metrics.

A second limitation is that the current ReEntry pipeline primarily modifies visibility decisions while keeping the base tracker coordinates fixed. As a result, coordinate-based metrics such as δ_avg often remain unchanged. This is useful for isolating the visibility/re-entry effect, but it also means that ReEntry does not correct localization errors directly.

A third limitation is protocol coverage. We report DAVIS first/input and DAVIS strided/original official-style local evaluations, and a full local RGB-Stacking full50 evaluation. However, we do not claim an official leaderboard submission, nor do we claim full TAP-Vid benchmark coverage across all subsets such as Kinetics and Kubric. RGB-Stacking full50 should be interpreted as a full local standard evaluation until explicitly audited or rerun under strict official first/strided query protocol.

Finally, the current V25 official-safe threshold shows only a small positive gain under DAVIS strided/original. This confirms that conservative thresholding can avoid metric degradation, but it is not a strong leaderboard-style improvement.

## Future work

The most direct future direction is **metric-aware selective ReEntry**. Instead of selecting visibility corrections only based on re-entry confidence, a future V26 variant should estimate whether a proposed correction is likely to preserve or improve standard AJ/OA while improving AJ_RD. This requires training or deriving a frame-level decision rule whose objective combines re-entry utility and standard metric safety.

A second direction is coordinate-aware recovery. Since the current method preserves base coordinates, it cannot improve δ_avg. Combining ReEntry with coordinate refinement or re-detection localization could improve both visibility and localization metrics.

A third direction is broader official-protocol coverage. The next protocol supplement should audit or rerun RGB-Stacking full50 under explicit TAP-Vid first or strided query mode. Larger-scale Kinetics and Kubric evaluations may further strengthen benchmark coverage, but they should be considered after the current paper tables and result narrative are finalized.

## Paper-ready paragraph

```text
ReEntry targets a specific failure mode rather than serving as a universal benchmark booster. Our results show strong gains in first-query DAVIS and RGB-Stacking re-entry metrics, but also reveal that aggressive visibility correction can harm global AJ/OA under strided/original evaluation. This motivates future metric-aware selection that explicitly balances re-entry recovery against standard TAP-Vid metric safety.
```
