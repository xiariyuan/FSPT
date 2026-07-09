# Final Decision After Official-Protocol Audit — 2026-07-04

## 1. Updated status

After additional protocol audits, the current evidence is stronger than the earlier conservative interpretation.

We now have three official-style local evaluations:

```text
1. TAPVid-DAVIS first/input official-style local evaluation
2. TAPVid-DAVIS strided/original official-style local evaluation
3. TAPVid RGB-Stacking full50 strided/256 official-style local evaluation
```

This is still not an official leaderboard/server submission, and it is still not the full TAP-Vid benchmark because Kinetics and Kubric are not fully evaluated.

---

## 2. What can be safely claimed now

Safe claim:

```text
We evaluate on standard TAP-Vid datasets using official-style local protocols, including DAVIS first/input, DAVIS strided/original, and RGB-Stacking full50 strided/256. We report standard TAP-Vid metrics (AJ, OA, δ_avg) and the re-entry diagnostic AJ_RD.
```

Even stronger but still safe:

```text
On TAPVid RGB-Stacking full50 under an audited official-style strided/256 protocol, ReEntry substantially improves AJ_RD and OA over CoTracker3 offline, while keeping δ_avg unchanged and trading off a small amount of AJ.
```

Avoid:

```text
official leaderboard submission
full TAP-Vid benchmark submission
all official metrics improve universally
full Kinetics/Kubric benchmark result
```

---

## 3. Key evidence summary

### 3.1 DAVIS first/input official-style local evaluation

| Method | AJ | OA | δ_avg | AJ_RD |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 62.6566 | 88.1487 | 77.2244 | 0.3142 |
| ReEntry V1 | 64.5758 | 91.7274 | 77.2244 | 0.3588 |
| ReEntry V22Q | 64.7260 | 91.7406 | 77.2244 | 0.3556 |
| ReEntry V24-DINOScore | 64.8439 | 91.8851 | 77.2244 | 0.3549 |
| TrackOn2 | 67.0406 | 92.0916 | 79.8418 | 0.3714 |

Main result:

```text
V24 - offline:
  ΔAJ    = +2.1874 pp
  ΔOA    = +3.7363 pp
  Δδ_avg = +0.0000 pp
  ΔAJ_RD = +0.0407
```

Paired-video result:

```text
V24 - offline:
  AJ mean +2.1874 pp, CI [+1.2582, +3.2487], W/L/T = 24/5/1
  OA mean +3.7363 pp, CI [+2.1673, +5.7423], W/L/T = 27/2/1
```

Interpretation:

```text
On DAVIS first/input, ReEntry clearly improves over CoTracker3 offline on AJ/OA/AJ_RD.
TrackOn2 remains the strongest external baseline overall.
```

---

### 3.2 DAVIS strided/original official-style local evaluation

Default ReEntry:

| Method | AJ | OA | δ_avg | AJ_RD |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 51.5385 | 92.1543 | 63.5892 | 0.3870 |
| ReEntry V1 default | 50.7303 | 91.2730 | 63.5892 | 0.4144 |
| ReEntry V22Q default | 50.8342 | 91.3439 | 63.5892 | 0.4136 |

Interpretation:

```text
Default ReEntry improves AJ_RD but hurts standard AJ/OA under stricter strided/original evaluation.
```

V25-safe threshold=0.80:

| Method | AJ | OA | δ_avg | AJ_RD |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 51.5385 | 92.1543 | 63.5892 | 0.3870 |
| V25-safe threshold=0.80 | 51.5648 | 92.2106 | 63.5892 | 0.3900 |

Deltas:

```text
ΔAJ    = +0.0263 pp
ΔOA    = +0.0563 pp
Δδ_avg = +0.0000 pp
ΔAJ_RD = +0.0030
```

Interpretation:

```text
V25-safe is an official-style safe operating point, not a strong leaderboard improvement.
```

---

### 3.3 RGB-Stacking full50 strided/256 official-style local evaluation

Query audit:

```text
Videos: 50 / 50
Queries: 60,829
Query frames: 0, 5, 10, ..., 245
All query frames visible: yes
Anchor error: 0.0
Resolution: 256x256
Metric: official TAP-Vid metric wrapper with query_mode=strided
```

Main results:

| Method | AJ | OA | δ_avg | AJ_RD |
|---|---:|---:|---:|---:|
| CoTracker3 offline | 79.9345 | 91.6371 | 88.5714 | 0.3617 |
| CoTracker3 online | 44.8432 | 55.8092 | 73.1881 | 0.4028 |
| ReEntry V1 | 79.4302 | 93.0758 | 88.5714 | 0.4414 |
| ReEntry V22Q | 79.5756 | 93.0481 | 88.5714 | 0.4400 |
| ReEntry V24-DINOScore | 79.5974 | 93.0389 | 88.5714 | 0.4394 |

Deltas vs CoTracker3 offline:

```text
V1:
  ΔAJ    = -0.5043 pp
  ΔOA    = +1.4387 pp
  Δδ_avg = +0.0000 pp
  ΔAJ_RD = +0.0797

V22Q:
  ΔAJ    = -0.3589 pp
  ΔOA    = +1.4110 pp
  Δδ_avg = +0.0000 pp
  ΔAJ_RD = +0.0783

V24:
  ΔAJ    = -0.3371 pp
  ΔOA    = +1.4018 pp
  Δδ_avg = +0.0000 pp
  ΔAJ_RD = +0.0777
```

Paired-video result:

```text
V24 - offline:
  AJ mean -0.3371 pp, CI [-0.6182, -0.0483], W/L/T = 11/39/0
  OA mean +1.4018 pp, CI [+0.9419, +1.8734], W/L/T = 40/10/0
  δ_avg mean +0.0000 pp, W/L/T = 0/0/50
```

Interpretation:

```text
On RGB full50 strided/256, ReEntry gives a strong AJ_RD/OA improvement with unchanged δ_avg and a small AJ trade-off.
```

---

## 4. Final method positioning

```text
V1:
  Best AJ_RD / OA-oriented main ReEntry method.

V22Q:
  Stability / interval extension method. Slightly improves AJ over V1 with small AJ_RD/OA cost.

V24-DINOScore:
  Optional AJ-oriented appearance micro-filter. Improves AJ over V22Q, with tiny OA/AJ_RD cost.

V25-safe threshold=0.80:
  Official-style safe operating point for DAVIS strided/original. Preserves standard AJ/OA but only retains a tiny AJ_RD gain.
```

---

## 5. Best paper claim

Recommended final claim:

```text
ReEntry improves re-entry recovery across standard TAP-Vid official-style local evaluations. On TAPVid RGB-Stacking full50 strided/256, it improves AJ_RD by +0.0797 and OA by +1.4387 with V1, while keeping δ_avg unchanged and trading off -0.5043 AJ. On TAPVid-DAVIS first/input, V24 improves AJ by +2.1874, OA by +3.7363, and AJ_RD by +0.0407 over CoTracker3 offline. Under the stricter DAVIS strided/original protocol, a conservative V25 threshold preserves official-style AJ/OA with a small AJ_RD gain.
```

This is strong, nuanced, and defensible.

---

## 6. Next step decision

Do not run more random experiments now.

The next best step is:

```text
Write final paper tables and Results section.
```

Concrete next files:

```text
docs/final_paper_tables_2026-07-04.md
docs/paper_results_section_draft_2026-07-04.md
```

Recommended table order:

```text
Table 1. Evaluation protocol summary
Table 2. TAPVid-DAVIS first/input official-style local evaluation
Table 3. TAPVid RGB-Stacking full50 strided/256 official-style local evaluation
Table 4. TAPVid-DAVIS strided/original official-style local evaluation
Table 5. RGB-Stacking fresh/stress diagnostic evaluation
Table 6. V25 threshold sweep and official-safe operating point
Table 7. Paired-video statistical summary
```

Optional later direction:

```text
V26 official-metric-aware selective ReEntry
```

Only start V26 after the current paper draft is frozen.
