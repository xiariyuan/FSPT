# Paper Qualitative Figure Manifest — 2026-06-29

## Decision

Selected three main qualitative figures and three appendix alternates. These figures support the final paper story: B2-W recovers re-entry, avoids global/online damage, and still has false-trigger limitations.

## Color legend

```text
GT = green
fixed_offline = yellow
global_B1_or_override = cyan
B2 = magenta
```

## Main paper figures

| figure | role | video/query | fixed AJ | B1/global AJ | B2 AJ | selected image | why selected |
|---|---|---|---:|---:|---:|---|---|
| Figure 2 | main_success_reentry_recovery | bmx-trees q=0 | 0.1554 | 0.7101 | 0.7076 | `outputs/paper_discovery_2026-06-27/paper_figures_selected/images/figure_2__main_success_reentry_recovery__success_reentry_recovery__bmx-trees__q0.png` | Clear re-entry recovery: fixed AJ 0.1554, B2 AJ 0.7076, large +0.5522 gain; trigger before GT re-entry. |
| Figure 3 | main_avoids_global_damage | drift-straight q=83 | 1.0000 | 0.0105 | 1.0000 | `outputs/paper_discovery_2026-06-27/paper_figures_selected/images/figure_3__main_avoids_global_damage__b2_avoids_b1_global_damage__drift-straight__q83.png` | Strongest avoid-global-damage case: fixed=1.000, global B1=0.0105, B2=1.000. |
| Figure 4 | main_failure_false_trigger | dog q=15 | 0.9200 | 0.2963 | 0.2784 | `outputs/paper_discovery_2026-06-27/paper_figures_selected/images/figure_4__main_failure_false_trigger__harmful_false_trigger__dog__q15.png` | Most severe ordinary harmful false-trigger case: fixed=0.9200, B2=0.2784, drop -0.6416. |

## Appendix alternates

| figure | role | video/query | fixed AJ | B1/global AJ | B2 AJ | selected image | why selected |
|---|---|---|---:|---:|---:|---|---|
| Appendix A | alternate_success_reentry_recovery | car-roundabout q=1 | 0.0000 | 0.6064 | 0.6064 | `outputs/paper_discovery_2026-06-27/paper_figures_selected/images/appendix_a__alternate_success_reentry_recovery__success_reentry_recovery__car-roundabout__q1.png` | Very clean fixed failure: fixed=0.0000, B2=0.6064. |
| Appendix B | alternate_avoids_global_damage | soapbox q=267 | 1.0000 | 0.0484 | 1.0000 | `outputs/paper_discovery_2026-06-27/paper_figures_selected/images/appendix_b__alternate_avoids_global_damage__b2_avoids_b1_global_damage__soapbox__q267.png` | B2 remains fixed-like while global B1 fails: fixed=1.000, B1=0.0484, B2=1.000. |
| Appendix C | targeted_harmful_failure | shooting q=47 | 0.5467 | 0.0968 | 0.0899 | `outputs/paper_discovery_2026-06-27/paper_figures_selected/images/appendix_c__targeted_harmful_failure__targeted_harmful_false_trigger__shooting__q47.png` | Targeted failure from worst false-trigger videos: fixed=0.5467, B2=0.0899. |

## Proposed captions

### Figure 2 — Re-entry recovery

B2-W recovers a point after re-entry. The fixed offline base loses the point after occlusion, while the re-entry branch re-localizes it; B2-W activates the override only around the predicted re-entry window and recovers the trajectory. This illustrates why the method improves AJ_RD.

### Figure 3 — Avoiding global damage

B2-W avoids the standard-tracking damage caused by global re-entry fusion. The global branch drifts badly on a normally tracked point, whereas B2-W keeps the base prediction because no re-entry trigger is activated. This explains why B2-W preserves standard AJ while global B1 collapses.

### Figure 4 — Failure case

A harmful false trigger. The base tracker remains reliable, but the override branch becomes visible and is activated despite no true re-entry event. B2-W therefore inherits the override error and loses standard tracking accuracy. This illustrates the remaining false-trigger-cost limitation.

## Contact sheet

Quick review image: `outputs/paper_discovery_2026-06-27/paper_figures_selected/main_figure_contact_sheet.png`

## Paper placement

```text
Figure 1: method diagram, still needs drawing
Figure 2: use selected re-entry recovery
Figure 3: use selected avoids-global-damage
Figure 4: use selected false-trigger failure
Appendix: include alternates if space permits
```


## Figure 1 method diagram

Generated method diagram:

```text
outputs/paper_discovery_2026-06-27/paper_figures_selected/figure1_b2w_method_diagram.svg
outputs/paper_discovery_2026-06-27/paper_figures_selected/figure1_b2w_method_diagram.png
outputs/paper_discovery_2026-06-27/paper_figures_selected/figure1_b2w_method_diagram.pdf
```

Caption draft:

B2-W keeps a standard-strong base tracker by default and activates the re-entry-strong override branch only when the base has been invisible and the override becomes persistently visible. The override is copied only inside a short local window `[t-pre, t+W]`, after which the output returns to the base tracker. This local intervention improves re-entry reliability while avoiding the standard-tracking damage caused by global override.
