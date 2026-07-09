# ReEntry-TAP Event Provenance and Stress-Induced-only AJ_RD — 2026-06-30

## Goal

Audit whether ReEntry-TAP improvements come from stress-created re-entry events, rather than only from natural re-entry already present in RGB-Stacking.

Each eligible re-entry event is labeled as:

```text
natural:         occlusion already exists in original GT visibility
stress_induced:  original point is visible, but stress transform/occluder makes it invisible
mixed:           both natural and stress-induced occlusion occur inside the same invisible interval
other:           fallback; should be zero
```

This audit covers both stress families:

```text
translate_exit_reenter L = 8, 16, 32
moving_occluder       L = 8, 16, 32
```

Artifacts:

```text
scripts/audit_reentry_stress_event_provenance.py
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/event_provenance_combined_summary.json
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/*_L*/event_provenance/event_rows.jsonl
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/*_L*/event_provenance/provenance_ajrd_summary.json
```

## Event provenance: eligible events

| Stress | natural | stress-induced | mixed | stress-induced rate |
|---|---:|---:|---:|---:|
| translate L8 | 3056 | 660 | 29 | 0.1762 |
| translate L16 | 3019 | 665 | 29 | 0.1791 |
| translate L32 | 2841 | 679 | 77 | 0.1888 |
| occluder L8 | 2626 | 3424 | 139 | 0.5532 |
| occluder L16 | 1985 | 3357 | 266 | 0.5986 |
| occluder L32 | 1470 | 3178 | 486 | 0.6190 |

## Interpretation of event provenance

Translate stress is a mixed benchmark: most eligible events are natural re-entry, but a stable subset of about 18% is stress-induced out-of-frame re-entry.

Moving-occluder stress is much cleaner as a stress-induced benchmark: 55%--62% of eligible events are directly created by the occluder. This makes occluder stress especially valuable for defending ReEntry-TAP as a true stress protocol rather than only a reweighting of natural RGB re-entry.

## Stress-induced-only AJ_RD_256

### Translate stress-induced events only

| Stress | offline | online | B2-W16-P2 | B2-offline | B2-online |
|---|---:|---:|---:|---:|---:|
| translate L8 | 0.6980 | 0.7345 | 0.7408 | +0.0428 | +0.0063 |
| translate L16 | 0.6763 | 0.7204 | 0.7242 | +0.0479 | +0.0038 |
| translate L32 | 0.6340 | 0.6771 | 0.6912 | +0.0572 | +0.0141 |

### Moving-occluder stress-induced events only

| Stress | offline | online | B2-W16-P2 | B2-offline | B2-online |
|---|---:|---:|---:|---:|---:|
| occluder L8 | 0.7587 | 0.7538 | 0.7828 | +0.0241 | +0.0290 |
| occluder L16 | 0.7520 | 0.7410 | 0.7772 | +0.0252 | +0.0362 |
| occluder L32 | 0.7420 | 0.6954 | 0.7705 | +0.0285 | +0.0751 |

## Natural-event AJ_RD_256

B2-W16-P2 also improves on natural events, which explains part of the full AJ_RD gain:

| Stress | offline natural | B2 natural | B2-offline |
|---|---:|---:|---:|
| translate L8 | 0.3980 | 0.4927 | +0.0947 |
| translate L16 | 0.4007 | 0.4911 | +0.0904 |
| translate L32 | 0.3952 | 0.4900 | +0.0948 |
| occluder L8 | 0.4001 | 0.4830 | +0.0829 |
| occluder L16 | 0.3797 | 0.4768 | +0.0971 |
| occluder L32 | 0.3846 | 0.4776 | +0.0930 |

This is not a flaw. It shows that ReEntry-TAP stress data contains both natural and stress-induced re-entry, and B2-W16-P2 improves both. However, the stress-induced-only tables above confirm that the gains are not solely due to natural pre-existing re-entry events.

## Key conclusion

The provenance audit closes an important potential reviewer objection.

Potential objection:

```text
The stress benchmark may simply be measuring natural RGB re-entry already present in the data.
```

Answer:

```text
No. On stress-induced-only eligible re-entry events, B2-W16-P2 still improves AJ_RD over offline across all translate and occluder severities.
```

For translate stress-induced re-entry:

```text
B2-W16-P2 improves AJ_RD_256 by +0.0428 to +0.0572 over offline.
```

For moving-occluder stress-induced re-entry:

```text
B2-W16-P2 improves AJ_RD_256 by +0.0241 to +0.0285 over offline.
```

Moreover, for occluder stress-induced events, online becomes worse than offline as L increases, while B2-W16-P2 remains clearly better than both:

```text
occluder L32 stress-induced:
offline = 0.7420
online  = 0.6954
B2-W16  = 0.7705
```

## Paper-level implication

This audit makes ReEntry-TAP substantially stronger. We can safely claim:

```text
Controlled stress creates a measurable set of new re-entry events, and B2-W16-P2 improves re-entry recovery on these stress-induced events while preserving standard tracking in the full stress evaluation.
```

Safe wording:

```text
We separate natural and stress-induced re-entry events and find that the proposed intervention improves AJ_RD on stress-induced events across both out-of-frame and occluder stress families.
```

Avoid overclaiming:

```text
Do not say every re-entry event in the stress benchmark is synthetic/stress-induced.
```

A precise statement is:

```text
Translate stress contains a mixed set of natural and stress-induced events; moving-occluder stress provides a cleaner stress-induced evaluation, with 55%--62% eligible events directly caused by the occluder.
```

## Next recommended step

The next best step is not another stress type. It is to prepare paper-ready figures/tables:

```text
1. Translate severity curve: AJ_RD and AJ vs L.
2. Moving-occluder severity curve: AJ_RD and AJ vs L.
3. Stress-induced-only AJ_RD table.
4. ReEntry-TAP construction diagram.
```

After these figures are ready, decide whether to run a frozen stress validation on RGB fresh20-49.
