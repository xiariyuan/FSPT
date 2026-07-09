# DAVIS B2 Finite-Window Ablation — 2026-06-29

## Decision

Finite-window B2 also works on DAVIS. `post16` and `post32` are only slightly below full-post on AJ_RD, while preserving standard AJ. `post64` is effectively identical to full-post.

This removes the main method-unification risk: finite-window override is not only an RGB-specific patch. It is compatible with the DAVIS B2 story as an ablation under the same local-intervention framework.

## Metrics

| method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | delta AJ_RD vs full | delta AJ vs full | note |
|---|---:|---:|---:|---:|---:|---:|---|
| fixed_offline | 0.5546 | 70.051 | 92.1544 | 82.3206 |  |  | base |
| global_B1_vis4_gated288 | 0.6279 | 47.4046 | 72.6762 | 76.4913 |  |  | global override |
| B2_full_post | 0.6278 | 68.9702 | 91.1851 | 82.4604 |  |  | previous DAVIS mainline |
| post16 | 0.6266 | 68.9512 | 91.1924 | 82.448 | -0.0012 | -0.0190 | finite-window |
| post32 | 0.6265 | 68.9596 | 91.1865 | 82.4479 | -0.0013 | -0.0106 | finite-window |
| post64 | 0.6277 | 68.9706 | 91.1851 | 82.4607 | -0.0001 | +0.0004 | finite-window |

## Key observations

```text
post16: AJ_RD_256=0.6266, AJ_256=68.9512, delta_AJRD_vs_full=-0.0012, delta_AJ_vs_full=-0.0190
post32: AJ_RD_256=0.6265, AJ_256=68.9596, delta_AJRD_vs_full=-0.0013, delta_AJ_vs_full=-0.0106
post64: AJ_RD_256=0.6277, AJ_256=68.9706, delta_AJRD_vs_full=-0.0001, delta_AJ_vs_full=+0.0004
```

Interpretation:

- `post16` is already very close to full-post on DAVIS: only -0.0012 AJ_RD_256 and -0.0190 standard AJ.
- `post32` is similarly close: -0.0013 AJ_RD_256 and -0.0106 standard AJ.
- `post64` is almost the same as full-post: -0.0001 AJ_RD_256 and +0.0004 standard AJ.
- Therefore the paper can safely describe B2 as a localized finite-window intervention family, with full-post as a limiting case.

## Method implication

Before this ablation, there was a risk that RGB used `post16` while DAVIS used full-post, making the method look dataset-specific. This result reduces that risk.

A safe paper framing is:

```text
B2 is a local re-entry override framework.
The override window W controls how long the re-entry branch is trusted.
Short finite windows are more robust on RGB; longer/full windows are equivalent on DAVIS.
```

## Artifacts

```text
scripts/run_b2_mainline_eval.py
outputs/paper_discovery_2026-06-27/teacher_expansion/b2_finite_window_davis/post16/
outputs/paper_discovery_2026-06-27/teacher_expansion/b2_finite_window_davis/post32/
outputs/paper_discovery_2026-06-27/teacher_expansion/b2_finite_window_davis/post64/
```
