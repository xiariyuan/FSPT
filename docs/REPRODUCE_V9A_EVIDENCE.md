# Reproduce and Verify the V9-A Evidence Chain

## Artifact verification from the final checkout

```bash
python scripts/verify_v9a_evidence_chain.py
```

This CPU-side verification checks commit ancestry, SHA256 hashes, JSON parsing, NPZ shapes, key formulas, gate decisions, and evidence-index consistency. It does not rerun TrackOn2 GPU inference.

## Full GPU reproduction versus artifact verification

The committed evidence can be verified from the final checkout. Full GPU recomputation additionally requires the immutable TrackOn2 checkpoint, DINOv3 weights, PointOdyssey annotations/frames, the recorded historical execution state, and the original absolute-path layout or an environment wrapper that maps equivalent roots.

Several formal scripts were executed from a historical HEAD before the execution script/result commit was created. The result JSON stores the execution HEAD and script SHA256. Therefore exact GPU reproduction should reconstruct the recorded historical code tree and inject the hash-matched execution script, rather than assuming the final checkout can rerun every script unchanged.

## Frozen versus generated artifacts

- Frozen scientific artifacts: route result JSON/NPZ, execution scripts, design/review documents.
- Generated evidence artifacts: evidence index, route matrix, verification report.
- Do not modify frozen scripts to replace absolute paths; use an external wrapper or matching mount layout.

## Paper-use boundary

- V9-A5C.0/V9-A6.0 are visible-frame candidate audits by construction.
- GT-only oracle/union results must be labeled as upper bounds.
- V9-A5.1c final interpretation must use the same-capacity comparator, not the raw min(official,beam) oracle pass.
- No official leaderboard or universal-improvement claim is authorized.

Evidence index generated from `bf4aed14c8b3dfa656991392bf27e27dd3982c01`.
