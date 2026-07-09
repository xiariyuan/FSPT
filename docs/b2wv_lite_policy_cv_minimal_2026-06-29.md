# B2-WV-Lite Policy CV Minimal Report — 2026-06-29

Development-only experiment on DAVIS + RGB dev0-9. No RGB fresh20-49 used.

## Result

B2-WV-Lite learned window policy is not ready to replace B2-W16-P2.

## DAVIS

B2-W16-P2: AJ_RD_256=0.6251, AJ_256=69.0119
argmax: AJ_RD_256=0.6046, AJ_256=69.4980, delta_vs_P2=(-0.0205 AJ_RD, +0.4861 AJ)
preserve95_dyn: AJ_RD_256=0.6061, AJ_256=69.4896, delta_vs_P2=(-0.0190 AJ_RD, +0.4777 AJ)
strict95_dyn: AJ_RD_256=0.5790, AJ_256=69.8558, delta_vs_P2=(-0.0461 AJ_RD, +0.8439 AJ)

## RGB dev10

B2-W16-P2: AJ_RD_256=0.4863, AJ_256=79.9035
argmax: AJ_RD_256=0.4508, AJ_256=79.9760, delta_vs_P2=(-0.0355 AJ_RD, +0.0725 AJ)
preserve95_dyn: AJ_RD_256=0.4488, AJ_256=79.9619, delta_vs_P2=(-0.0375 AJ_RD, +0.0584 AJ)
strict95_dyn: AJ_RD_256=0.3957, AJ_256=80.0616, delta_vs_P2=(-0.0906 AJ_RD, +0.1581 AJ)

## Interpretation

The learned policy recovers some standard AJ but loses too much AJ_RD. This confirms that naive learned action selection is not sufficient. B2-W16-P2 remains the main method. B2-WV remains a promising future direction because the oracle has strong headroom, especially on DAVIS, but the learned policy needs a stronger AJ_RD-preserving objective or visual/appearance verification.

## Artifacts

scripts/eval_b2wv_lite_policy_cv.py
outputs/paper_discovery_2026-06-27/b2wv_lite_policy_cv/summary.json
outputs/paper_discovery_2026-06-27/b2wv_lite_policy_cv/oof_window_policy_rows.jsonl
