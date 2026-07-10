# V9-A2.6 TrackOn2 Internal Proxy Integrity Audit

Date: 2026-07-10

## Provenance boundary

TrackOn2 256-space M24 support-grid20 rerun is an internal-state proxy. It is not claimed to be the exact latent state that generated the historical first-input cache.

## Integrity checks

| Check | Result |
|---|---|
| full_shape | PASS |
| full_finite | PASS |
| full_indices_match_joint | PASS |
| full_row_keys_match_joint | PASS |
| unique_feature_names | PASS |
| old_smoke_matches_full_prefix | PASS |
| enhanced_smoke_matches_old_smoke | PASS |
| enhanced_smoke_matches_full_prefix | PASS |
| full_first_active_frame_parity | PASS |
| enhanced_long_sequence_parity | PASS |

## Official-forward parity

- Full build: 20 videos, first-active-frame max errors {'p_max_abs': 0.0, 'v_max_abs': 0.0, 'q_max_abs': 0.0}.
- Enhanced smoke target frames: [{'video_id': 'bike-packing', 'frame': 25, 'active_queries': 423}, {'video_id': 'bike-packing', 'frame': 45, 'active_queries': 423}, {'video_id': 'bike-packing', 'frame': 68, 'active_queries': 425}].
- Enhanced smoke max errors: {'p_max_abs': 0.0, 'v_max_abs': 0.0, 'q_max_abs': 0.0}.

## Proxy alignment

- Proxy-to-old-candidate internal-model distance: {'mean': 3.4609868176803125, 'median': 1.3962668180465698, 'p95': 10.488231086730934, 'max': 162.7106475830078, 'std': 12.30236295975087}.
- Proxy-to-native internal-model distance: {'mean': 9.160660164949068, 'median': 2.4022958278656006, 'p95': 48.62225112915031, 'max': 190.54364013671875, 'std': 24.736384250452787}.

## Decision

**PASS_PROXY_FEATURE_LINEAGE_AND_FORWARD_PARITY**

The patched builder preserves the original 64-dimensional features exactly on the smoke prefix, the full matrix is strictly aligned to all 557 extension rows, and the copied diagnostic forward matches official TrackOn2 position/visibility/query tensors exactly at both first-active and long-sequence target frames. The historical-latent provenance limitation remains explicit.
