# P3A Pseudo-Label Quality Audit

Mode: `smoke`
Verdict: `SMOKE_STOP`
Reason: smoke quality is too weak to justify full P3A rollout

## Coverage

- Labels: `62`
- Videos: `1`
- Tracks: `62`
- Re-entry events: `62`
- Long-occ (>=16) ratio: `0.4839`
- Max video share: `1.0`

## Error Statistics

| Metric | Pseudo-label | CT-offline | Delta |
|---|---:|---:|---:|
| median px | 46.15 | 5.99 | -28.9 |
| mean px | 39.95 | 10.97 | -28.98 |
| <4px | 0.0 | 0.2581 | -0.2581 |
| <8px | 0.0 | 0.7258 | -0.7258 |
| <16px | 0.0806 | 0.8065 | -0.7259 |

- Better than CT-offline: `0.0323`

## Consistency

- FB available: `False`
- FB pass rate: `None`
- Flow available: `False`
- Flow pass rate: `None`

## Quality Flags

- `better_than_ct_offline`: `2`
- `within_16px`: `5`
