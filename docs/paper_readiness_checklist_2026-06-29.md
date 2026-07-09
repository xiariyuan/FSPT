# Paper Readiness Checklist — 2026-06-29

## Current recommendation

Stop large experimental runs for now. The paper has enough core evidence for a serious CCF-B / CAS-Q2 style submission draft. Further improvement should focus on writing quality, figure polish, and protocol clarity. New experiments should only be started if they address a concrete reviewer-risk item.

## Readiness score

```text
Core idea clarity:          8.5 / 10
DAVIS evidence:             8.5 / 10
RGB validation:             8.0 / 10
Supplementary baseline:     7.0 / 10
Failure analysis:           8.0 / 10
Figures:                    7.0 / 10
Writing readiness:          6.5 / 10
Overall paper readiness:    7.5 / 10
```

Interpretation: the experimental backbone is strong enough to consolidate. The main gap is no longer evidence; it is polish, positioning, and final writing.

## Artifact existence audit

| item | exists | path | size |
|---|---:|---|---:|
| Main paper draft | yes | `docs/full_paper_draft_v0_1_b2w_reentry_reliability_2026-06-29.md` | 24877 |
| Paper skeleton | yes | `docs/paper_skeleton_b2w_reentry_reliability_2026-06-29.md` | 17258 |
| Experiment tables v2 | yes | `docs/paper_experiment_tables_consolidation_v2_2026-06-29.md` | 7704 |
| Qualitative figure manifest | yes | `docs/paper_qualitative_figure_manifest_2026-06-29.md` | 4828 |
| Upgrade plan | yes | `docs/paper_upgrade_strengthening_plan_2026-06-29.md` | 6182 |
| Baseline parity audit | yes | `docs/baseline_parity_audit_tapnext_trackon2_2026-06-29.md` | 6718 |
| TrackOn2 plug-in experiment | yes | `docs/first_input_trackon2_b2w_plugin_experiment_2026-06-29.md` | 4461 |
| B2-RV cache-level dev CV | yes | `docs/b2_rv_cache_level_dev_cv_2026-06-29.md` | 4388 |
| DAVIS B2-W16-P2 audit | yes | `docs/davis_b2w16p2_unified_audit_2026-06-29.md` | 4533 |
| RGB fresh20-49 aggregate | yes | `docs/rgb_stacking_fresh20_49_aggregate_eval_2026-06-29.md` | 3006 |
| Figure 1 SVG | yes | `outputs/paper_discovery_2026-06-27/paper_figures_selected/figure1_b2w_method_diagram.svg` | 12731 |
| Figure 1 PNG | yes | `outputs/paper_discovery_2026-06-27/paper_figures_selected/figure1_b2w_method_diagram.png` | 110401 |
| Figure 1 PDF | yes | `outputs/paper_discovery_2026-06-27/paper_figures_selected/figure1_b2w_method_diagram.pdf` | 164178 |
| Figure manifest JSON | yes | `outputs/paper_discovery_2026-06-27/paper_figures_selected/figure_manifest.json` | 7504 |
| RGB fresh20-49 summary | yes | `outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/summary.json` | 20132 |
| DAVIS unified audit summary | yes | `outputs/paper_discovery_2026-06-27/davis_b2w16p2_unified_audit/summary.json` | 192536 |
| TrackOn2 plug-in summary | yes | `outputs/paper_discovery_2026-06-27/first_input_trackon2_b2w/first_input_trackon2_b2w_per_video_summary.json` | 27645 |

## Main evidence locked

### DAVIS main result

```text
CoTracker3 offline: AJ_RD_256=0.5546, AJ_256=70.0510
global B1:          AJ_RD_256=0.6279, AJ_256=47.4046
B2-W16-P2:          AJ_RD_256=0.6251, AJ_256=69.0119
GT-window oracle:   AJ_RD_256=0.6290, AJ_256=70.1821
```

### RGB fresh20-49 main validation

```text
CoTracker3 offline: AJ_RD_256=0.3816, AJ_256=79.5944
CoTracker3 online:  AJ_RD_256=0.4121, AJ_256=44.6933
B2-W16-P2:          AJ_RD_256=0.4454, AJ_256=79.0664
B2-W16-P2 vs offline: AJ_RD_256 +0.0638, AJ_256 -0.5280
Per-video AJ_RD improved vs offline: 24 / 30
```

### TrackOn2 plug-in supplementary result

```text
TrackOn2 first-input:       AJ_RD_256=0.5444, AJ_256=67.0406
B2-W16-P2 TrackOn2 base:    AJ_RD_256=0.5509, AJ_256=67.1260
Gain:                       +0.0065 AJ_RD_256, +0.0854 AJ_256
Placement: appendix / supplementary only
```

## What is done

- [x] Main DAVIS table.
- [x] Main RGB fresh20-49 validation.
- [x] DAVIS finite-window ablation.
- [x] DAVIS occlusion-length bucket analysis.
- [x] Trigger / false-trigger taxonomy.
- [x] RGB fresh/final split protocol and aggregation.
- [x] TrackOn2/TAPNext baseline parity audit.
- [x] TrackOn2 first-input plug-in generality experiment.
- [x] Qualitative figure selection for success / avoid-damage / failure.
- [x] Figure 1 method diagram.
- [x] Full paper draft v0.1.
- [x] B2-RV exploratory verifier documented as future work.

## What still needs work before submission

### P0 — Must do

- [ ] Polish Abstract and Introduction into conference-paper style.
- [ ] Add proper citations and Related Work details from current papers.
- [ ] Convert selected qualitative figures into final multi-panel figures with consistent labels.
- [ ] Make sure every table states its protocol: strided-original vs first-input/input-resolution.
- [ ] Add a short explicit paragraph: “We do not claim SOTA over TrackOn2/TAPNext because parity differs.”
- [ ] Check all metric names are consistent: AJ_RD_256, AJ_256, OA_256, delta_avg_256.

### P1 — Strongly recommended

- [ ] Create a final `figures/` directory with publish-ready PNG/PDF files.
- [ ] Add appendix table for B2-RV showing why it is not the main method.
- [ ] Add appendix table for baseline parity audit.
- [ ] Add per-video stability table in appendix for RGB fresh20-49 and TrackOn2 plug-in.
- [ ] Run one final script that verifies all table numbers are reproducible from JSON artifacts.

### P2 — Optional if server/time remains

- [ ] Window-level B2-RV on development only.
- [ ] Proper TrackOn2 strided-original export parity fix.
- [ ] Additional untouched RGB split only if a new method is developed.
- [ ] Additional benchmark beyond DAVIS/RGB if targeting CCF-A.

## What should not be done now

- [ ] Do not tune on RGB fresh20-49.
- [ ] Do not claim B2-W generally beats TrackOn2/TAPNext.
- [ ] Do not put non-parity-established TrackOn2/TAPNext strided-original caches into the main table.
- [ ] Do not replace B2-W16-P2 with B2-RV unless a window-level verifier becomes stable.
- [ ] Do not keep changing the main method name; use B2-W16-P2 consistently.

## Recommended next action

The next best action is not another experiment. It is to polish the full paper draft into a submission-style draft, starting with Abstract + Introduction + Method. If any experiment is run, it should be only the final reproducibility-number check, not new method tuning.

## Current target assessment

```text
CCF-B / CAS-Q2: plausible if writing and figures are polished.
CCF-A: still requires stronger novelty or extra validated benchmark / stable learned verifier / fully parity-validated strong baseline comparison.
```


## Added statistical robustness audit

Completed:

```text
docs/statistical_robustness_audit_b2w_2026-06-29.md
```

Main takeaway: DAVIS and RGB fresh20-49 AJ_RD gains are robust by per-video bootstrap/sign-test. TrackOn2 plug-in remains supplementary because its CI crosses zero.
