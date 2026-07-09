# B2-WV P2-safe CV Minimal Report — 2026-06-29

Development-only experiment. No RGB fresh20-49 used.

## Goal

Start from B2-W16-P2 and only shorten or veto selected P2 candidate windows. This tests whether a conservative learned policy can recover standard AJ without losing much AJ_RD.

## Main result

P2-safe learned policies are not ready to replace B2-W16-P2.

## DAVIS

B2-W16-P2: AJ_RD_256=0.6251, AJ_256=69.0119
p2_short_re99: AJ_RD_256=0.6166, AJ_256=69.2530, delta_vs_P2=(-0.0085 AJ_RD, +0.2411 AJ)
p2_short_re97: AJ_RD_256=0.6092, AJ_256=69.3941, delta_vs_P2=(-0.0159 AJ_RD, +0.3822 AJ)
p2_veto_re99:  AJ_RD_256=0.6086, AJ_256=69.3638, delta_vs_P2=(-0.0165 AJ_RD, +0.3519 AJ)

## RGB dev10

B2-W16-P2: AJ_RD_256=0.4863, AJ_256=79.9035
p2_short_re99: AJ_RD_256=0.4450, AJ_256=79.9942, delta_vs_P2=(-0.0413 AJ_RD, +0.0907 AJ)
p2_short_re97: AJ_RD_256=0.4379, AJ_256=80.0413, delta_vs_P2=(-0.0484 AJ_RD, +0.1378 AJ)
p2_veto_re99:  AJ_RD_256=0.4191, AJ_256=79.9787, delta_vs_P2=(-0.0672 AJ_RD, +0.0752 AJ)

## Interpretation

The P2-safe policy can recover some standard AJ, especially on DAVIS, but the AJ_RD cost is still too large. RGB dev10 is particularly sensitive. Therefore, learned runtime-feature policies should not be promoted as the main method.

## Decision

Keep B2-W16-P2 as the main method.
Use B2-WV as future work / appendix.
A real method upgrade likely needs appearance-level verification, e.g. query/last-visible/candidate patch similarity, not just trajectory and visibility features.

## Artifacts

scripts/eval_b2wv_p2safe_cv.py
outputs/paper_discovery_2026-06-27/b2wv_p2safe_cv/summary.json
outputs/paper_discovery_2026-06-27/b2wv_p2safe_cv/oof_rows.jsonl
