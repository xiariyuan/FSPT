# ReEntry-TAP Stress Trigger Taxonomy and False-trigger Cost — 2026-07-01

## Goal

Analyze where B2-W16-P2 triggers under ReEntry-TAP stress, separated by event provenance:

```text
natural re-entry
stress-induced re-entry
mixed re-entry
false trigger tracks
```

This supplements the main result by explaining why occluder stress has cleaner triggers than translate stress.

## Dev stress trigger taxonomy

| Split | Family | L | precision | recall | natural recall | stress-induced recall | false-trigger rate | false-trigger mean ΔAJ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dev0-9 | translate | 8 | 0.683453 | 0.918508 | 0.909607 | 0.956061 | 0.316547 | +2.5859 |
| dev0-9 | translate | 16 | 0.678363 | 0.929617 | 0.918286 | 0.974436 | 0.321637 | +3.0599 |
| dev0-9 | translate | 32 | 0.678227 | 0.924964 | 0.908326 | 0.970545 | 0.321773 | +3.7145 |
| dev0-9 | occluder | 8 | 0.850898 | 0.958913 | 0.931620 | 0.981893 | 0.149102 | +0.7856 |
| dev0-9 | occluder | 16 | 0.849594 | 0.940071 | 0.906328 | 0.965445 | 0.150406 | +0.6822 |
| dev0-9 | occluder | 32 | 0.843718 | 0.912342 | 0.904423 | 0.917873 | 0.156282 | +1.5108 |

## Frozen fresh20-29 trigger taxonomy

| Split | Family | L | precision | recall | natural recall | stress-induced recall | false-trigger rate | false-trigger mean ΔAJ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fresh20-29 | translate | 16 | 0.650822 | 0.908598 | 0.895417 | 0.948187 | 0.349178 | +3.9190 |
| fresh20-29 | occluder | 16 | 0.876132 | 0.911462 | 0.900223 | 0.923709 | 0.123868 | +2.3778 |

## Interpretation

The trigger signal is high-recall across both stress families.

Key observations:

```text
1. Stress-induced trigger recall is very high.
   dev translate: 0.956–0.974
   dev occluder:  0.918–0.982
   fresh translate L16: 0.948
   fresh occluder L16:  0.924

2. Occluder triggers are cleaner than translate triggers.
   dev translate false-trigger rate: about 32%
   dev occluder false-trigger rate:  about 15%
   fresh translate L16 false-trigger rate: 34.9%
   fresh occluder L16 false-trigger rate:  12.4%

3. False triggers under stress are not necessarily harmful.
   Mean single-query ΔAJ on false-trigger tracks is positive in this audit.
   This differs from the natural DAVIS false-trigger audit where many false triggers are harmful.
```

The positive false-trigger mean under stress should be interpreted carefully. A track labeled as a false trigger has no GT re-entry event, but a short override can still improve the standard trajectory if the base tracker is locally uncertain. Therefore, false-trigger count alone is insufficient; standard AJ deltas remain the authoritative measure of cost.

## Paper wording

Safe statement:

```text
The B2-W16-P2 trigger has high recall for stress-induced re-entry events. Occluder stress produces cleaner triggers than translate stress, with roughly half the false-trigger rate. Under stress, many false-trigger windows are low-cost or even beneficial, so we report both trigger taxonomy and final AJ cost.
```

Avoid claiming:

```text
All false triggers are harmless.
```

Because natural-data audits show that false triggers can be harmful, and this depends on dataset/stress distribution.

## Artifacts

```text
scripts/audit_reentry_stress_trigger_taxonomy.py
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/trigger_taxonomy_stress_summary.json
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_29/trigger_taxonomy_stress_summary.json
```
