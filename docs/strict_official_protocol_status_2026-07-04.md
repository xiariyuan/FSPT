# Strict Official Protocol Status — 2026-07-04

## 1. Why we changed priority

The correct evaluation order should be:

```text
1. First establish official-protocol local evaluation.
2. Then use bridge / fresh / stress results as diagnostic support.
3. Then write paper narrative.
```

Earlier bridge/full50 results were useful for method discovery, but the paper should be anchored by official-style protocol results.

---

## 2. Official-style local evaluations now completed

### A. TAPVid-DAVIS first/input

Artifact:

```text
docs/official_protocol_davis_first_input_comparison_2026-07-04.md
```

Protocol:

```text
Dataset: TAPVid-DAVIS
Videos: 30
Queries: 650
Query mode: first
Metric resolution: input / 256
Metrics: AJ / OA / delta_avg / AJ_RD
```

Main result vs CoTracker3 offline:

```text
V24-DINOScore:
  dAJ    = +2.1874
  dOA    = +3.7363
  dAJ_RD = +0.0407
```

Status:

```text
Completed. This is the cleanest positive official-style DAVIS result.
```

---

### B. TAPVid-DAVIS strided/original

Artifacts:

```text
docs/official_protocol_audit_2026-07-04.md
docs/official_protocol_v25_threshold_sweep_2026-07-04.md
```

Protocol:

```text
Dataset: TAPVid-DAVIS
Videos: 30
Query mode: strided
Metric resolution: original
Metrics: AJ / OA / delta_avg / AJ_RD
```

Default ReEntry:

```text
Improves AJ_RD but reduces standard AJ/OA.
```

V25-safe threshold=0.80:

```text
AJ    = 51.5648 vs offline 51.5385, dAJ = +0.0263
OA    = 92.2106 vs offline 92.1543, dOA = +0.0563
AJ_RD = 0.3900  vs offline 0.3870,  dAJ_RD = +0.0030
```

Status:

```text
Completed. Shows protocol sensitivity and official-safe operating point.
```

---

### C. TAPVid RGB-Stacking full50 strided/input

Artifact:

```text
docs/official_protocol_rgb_full50_strided_comparison_2026-07-04.md
```

Protocol:

```text
Dataset: TAPVid RGB-Stacking
Videos: 50, rgb_stacking_000000 -- rgb_stacking_000049
Queries: 60,829
Audited query mode: strided
Query frames: 0, 5, 10, ..., 245
Metric resolution: input / 256
Metrics: AJ / OA / delta_avg / AJ_RD
```

Main result vs CoTracker3 offline:

```text
V1:
  dAJ    = -0.5043
  dOA    = +1.4387
  dAJ_RD = +0.0797

V22Q:
  dAJ    = -0.3589
  dOA    = +1.4110
  dAJ_RD = +0.0783

V24-DINOScore:
  dAJ    = -0.3371
  dOA    = +1.4018
  dAJ_RD = +0.0777
```

Status:

```text
Completed. This is now an official-style strided local evaluation over the full RGB-Stacking 50-video subset.
```

---

## 3. What we can now safely claim

Safe claim:

```text
We evaluate on TAPVid-DAVIS and TAPVid RGB-Stacking using official-style local protocols. On DAVIS first/input, ReEntry improves AJ/OA/AJ_RD over CoTracker3 offline. On RGB-Stacking full50 strided/input, ReEntry improves AJ_RD and OA with a small AJ trade-off. Under DAVIS strided/original, ReEntry requires a conservative threshold to preserve AJ/OA.
```

Also safe:

```text
The RGB-Stacking result uses all 50 official RGB-Stacking videos and an audited strided query pattern.
```

Not safe:

```text
Official leaderboard submission.
Full TAP-Vid benchmark submission.
Universal improvement across all official protocols.
```

---

## 4. Remaining gaps

### Gap 1: TAPVid-Kinetics

Current status:

```text
Raw data exists, but no generated prediction caches were found earlier.
```

Cost:

```text
High. Requires running trackers over Kinetics videos.
```

Recommendation:

```text
Do not block paper draft on Kinetics unless the goal becomes broad benchmark competition.
```

### Gap 2: TAPVid-Kubric

Current status:

```text
Raw data exists, but no generated prediction caches were found earlier.
```

Cost:

```text
High. Requires generating predictions and potentially handling TFRecord pipeline.
```

Recommendation:

```text
Supplementary only; not needed for current ReEntry failure-mode paper.
```

### Gap 3: RGB-Stacking first-query official-style

Current status:

```text
Not yet run as a separate first-query evaluation.
```

Recommendation:

```text
Optional. The strided full50 official-style result is already strong because query frames are audited as 0:5:245.
```

---

## 5. Next paper action

Now that official-style evaluation is anchored, the next action should be:

```text
1. Generate final paper tables from official-style results.
2. Generate a Results section draft.
3. Move bridge/fresh/stress results to supporting diagnostic sections.
```

Recommended output files:

```text
docs/final_paper_tables_2026-07-04.md
docs/paper_results_section_draft_2026-07-04.md
```

---

## 6. One-sentence final position

```text
The current evaluation package is no longer just local bridge evidence: it now includes official-style local evaluations on TAPVid-DAVIS first/input, TAPVid-DAVIS strided/original, and TAPVid RGB-Stacking full50 strided/input. This is protocol-safe for a paper, while still not an official leaderboard submission.
```
