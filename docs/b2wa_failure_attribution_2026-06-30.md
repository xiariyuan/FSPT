# B2-WA / B2-WV Failure Attribution Audit — 2026-06-30

## Scope

RGB dev10 only, 2000 stratified candidate windows. RGB fresh20-49 was not used.

## Geometry-risk buckets

| bucket | n | harmful rate | helpful rate | best reject | best long |
|---|---:|---:|---:|---:|---:|
| low | 667 | 0.0390 | 0.3433 | 0.2894 | 0.4243 |
| medium | 666 | 0.0886 | 0.7342 | 0.2312 | 0.3694 |
| high | 667 | 0.2729 | 0.5008 | 0.4528 | 0.2429 |

## Harmful-window attribution tags

| tag | n in all rows | harmful rate in tag | harmful rows covered | coverage of all harmful |
|---|---:|---:|---:|---:|
| large_geometry_gap | 677 | 0.2570 | 174 | 0.6517 |
| low_base_override_agreement | 897 | 0.1973 | 177 | 0.6629 |
| base_recovers_or_stays_visible | 645 | 0.0574 | 37 | 0.1386 |
| override_unstable | 886 | 0.1659 | 147 | 0.5506 |
| ambiguous_low_geometry | 13 | 0.0000 | 0 | 0.0000 |

## Global AUC comparison

| target | traj | RGB app | CLIP | ResNet dense | all visual | combined |
|---|---:|---:|---:|---:|---:|---:|
| harmful_w16 | 0.7374 | 0.4865 | 0.4254 | 0.4729 | 0.4865 | 0.7142 |
| best_reject | 0.6580 | 0.5393 | 0.5086 | 0.5383 | 0.5307 | 0.6547 |
| helpful_w16 | 0.6905 | 0.5662 | 0.5281 | 0.4928 | 0.5398 | 0.6907 |
| best_long | 0.6934 | 0.5189 | 0.5136 | 0.4907 | 0.4913 | 0.6871 |
| window_has_reentry | 0.8376 | 0.6862 | 0.5559 | 0.5089 | 0.6986 | 0.8266 |

## AUC by geometry-risk bucket: harmful_w16

| bucket | traj | RGB app | CLIP | ResNet dense | all visual | combined |
|---|---:|---:|---:|---:|---:|---:|
| low | 0.7764 | 0.7792 | 0.7709 | 0.8065 | 0.7437 | 0.7441 |
| medium | 0.6062 | 0.6412 | 0.6421 | 0.5988 | 0.5875 | 0.5852 |
| high | 0.6466 | 0.4318 | 0.4823 | 0.4295 | 0.4311 | 0.5734 |

## Interpretation

Global harmful_w16 combined gain over trajectory-only: `-0.023245`.

Medium-risk harmful_w16 combined gain over trajectory-only: `-0.021053`.


## Decision

Generic appearance features do not provide a meaningful harmful-window improvement globally or in the medium-risk bucket. The next stronger-method direction should shift from B2-WA appearance verification to B2-WT: temporal / geometry risk-controlled adaptive early-stop.


## Artifacts

```text
outputs/paper_discovery_2026-06-27/b2wa_failure_attribution/summary.json
outputs/paper_discovery_2026-06-27/b2wa_failure_attribution/rows_with_attribution.jsonl
```
