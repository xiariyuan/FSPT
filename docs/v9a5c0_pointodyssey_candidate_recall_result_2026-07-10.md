# V9-A5C.0 PointOdyssey Candidate-Recall / Correlation-Map Oracle Audit

Date: 2026-07-10

Rows audited: 4878; clips: 9; device: cuda.

## Global recall@4px

| Map | K1 | K4 | K8 | K16 | K32 | K64 |
|---|---:|---:|---:|---:|---:|---:|
| fused | 0.5260 | 0.6427 | 0.7048 | 0.7729 | 0.8473 | 0.9287 |
| c4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0002 | 0.0002 |
| c8 | 0.4863 | 0.6415 | 0.7132 | 0.7692 | 0.8229 | 0.8854 |
| c16 | 0.2390 | 0.4342 | 0.5092 | 0.5558 | 0.5859 | 0.5980 |
| c32 | 0.0816 | 0.1189 | 0.1248 | 0.1347 | 0.1398 | 0.1412 |
| union_raw_equal | 0.5670 | 0.5670 | 0.6626 | 0.7505 | 0.8311 | 0.8831 |

## Per-sequence primary headroom

| Sequence | Fused K16 R@4 | Fused K64 R@4 | Raw-union K64 R@4 (diag.) | Fused headroom | Pass |
|---|---:|---:|---:|---:|---|
| ani | 0.7199 | 0.9058 | 0.8435 | +0.1859 | True |
| animal3 | 0.8555 | 0.9304 | 0.9213 | +0.0749 | True |
| r4_new_f | 0.7697 | 0.9545 | 0.8999 | +0.1849 | True |

## Nearest-GT grid rank

| Map | Median rank | P90 rank | <=16 | <=32 | <=64 | Nearest distance median | Margin median |
|---|---:|---:|---:|---:|---:|---:|---:|
| fused | 6.0 | 120.0 | 0.6347 | 0.7214 | 0.8149 | 0.930 | -0.0854 |
| c4 | 12273.5 | 12288.0 | 0.0000 | 0.0000 | 0.0000 | 0.930 | -0.2636 |
| c8 | 3.0 | 194.0 | 0.6804 | 0.7411 | 0.7997 | 1.847 | -0.0234 |
| c16 | 2.0 | 15.0 | 0.9080 | 0.9588 | 0.9879 | 3.658 | -0.0054 |
| c32 | 2.0 | 11.0 | 0.9258 | 0.9750 | 0.9928 | 7.253 | -0.0012 |

## Integrity

```json
{
  "input_audit": {
    "pass": true,
    "head": "7c24367cf8aeb1467946d2ced3b294299032538e",
    "branch": "v9a45-conservative-residual-20260710",
    "tracked_status": "",
    "checked_inputs": [
      {
        "path": "/gemini/code/FSPT/outputs/paper_discovery_2026-07-05/v9a38_pointodyssey_pool/v9a38_pointodyssey_hypothesis_pool.npz",
        "size_bytes": 7727883,
        "sha256": "137cbec62d7086f53617acc1541b3df146f1664f70f72bfb6b1f2de0176d964f"
      },
      {
        "path": "/gemini/code/FSPT/baselines/track_on/checkpoints_trackon2_dinov3.pt",
        "size_bytes": 93901966,
        "sha256": "0e319c279cbdf51a5fc761b47dc1969520e8cfccfb57dc5a019a8c56e1039cd4"
      },
      {
        "path": "/gemini/code/FSPT/third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m/model.safetensors",
        "size_bytes": 114794096,
        "sha256": "208146e499dace99e4c9376ddb8a26f77d64c31c46c4dc4b86ff8bc63b0235e2"
      },
      {
        "path": "/gemini/code/FSPT/third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m/config.json",
        "size_bytes": 742,
        "sha256": "6f4ac67fea1761fe684d2a7db3139bab2d0dfdf94c05063d5992717c4c1da0ac"
      },
      {
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/ani/anno.npz",
        "size_bytes": 214730314,
        "sha256": "cc20ff10b815d607a09a48895a8d76e9cb8169c108ddb9fdacb2955dc26c2368"
      },
      {
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/animal3/anno.npz",
        "size_bytes": 197918569,
        "sha256": "84110e093f48da96393ddaddeaad15352b11cfa80a298b56ecdaba4365e89dfa"
      },
      {
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/r4_new_f/anno.npz",
        "size_bytes": 98527880,
        "sha256": "382d1edca0d6d88279ed1a61ee63c9ba27b7c5418f2269f4592c1658fd7d8e77"
      },
      {
        "path": "/gemini/code/FSPT_v9a45_clean/baselines/track_on/config/test.yaml",
        "size_bytes": 249,
        "sha256": "34c08c71be511b2cda6aaafb6e2e3af7e21c6fd5197de98a54fa4b9ac7a78c8e"
      },
      {
        "path": "/gemini/code/FSPT_v9a45_clean/docs/v9a45_pointodyssey_frame_manifest_2026-07-10.json",
        "size_bytes": 148902,
        "sha256": "2cc649b8cd6815c0184e9880ed49a5024996c4d8d4d5eda943ffd7bf49bbde74"
      }
    ],
    "frame_manifest": {
      "path": "/gemini/code/FSPT_v9a45_clean/docs/v9a45_pointodyssey_frame_manifest_2026-07-10.json",
      "file_count": 844,
      "total_bytes": 133075204,
      "aggregate_sha256": "0ba38d4fb90cf5a7afd05794eed95fdecd57e8e80b62b79218e0b46a5af13f37"
    },
    "imported_trackon_code_paths": {
      "model.trackon_predictor": "/gemini/code/FSPT_v9a45_clean/baselines/track_on/model/trackon_predictor.py",
      "model.trackon": "/gemini/code/FSPT_v9a45_clean/baselines/track_on/model/trackon.py",
      "model.reranking": "/gemini/code/FSPT_v9a45_clean/baselines/track_on/model/reranking.py",
      "utils.coord_utils": "/gemini/code/FSPT_v9a45_clean/baselines/track_on/utils/coord_utils.py",
      "utils.train_utils": "/gemini/code/FSPT_v9a45_clean/baselines/track_on/utils/train_utils.py"
    },
    "script": {
      "path": "/gemini/code/FSPT_v9a45_clean/scripts/v9a5c0_candidate_recall_correlation_oracle.py",
      "sha256": "627ba83754456c9895bc96f68c1360a43337c52e865decac824c49288e4e70aa"
    }
  },
  "rows_processed": 4878,
  "fused_recompute_max_abs": 0.0,
  "official_p_max_abs": 0.0,
  "official_v_max_abs": 0.0,
  "official_q_max_abs": 0.0,
  "pool_c1_ordered_max_abs": 5.960464477539063e-08,
  "pool_c1_set_hausdorff_max": 8.429369557916289e-08,
  "audit_fused_top16_official_ordered_max_abs": 0.1882353127002716,
  "audit_fused_top16_official_set_hausdorff_max": 0.0,
  "fused_k16_pool_oracle_error_max_abs": 4.57763671875e-05,
  "min_error_monotonic_in_k": true,
  "candidate_count_monotonic_in_k": true,
  "ms_corr_proj_weight_order_c4_c8_c16_c32": [
    -1.298351526260376,
    1.9004112482070923,
    1.9091821908950806,
    3.735589027404785
  ],
  "ms_corr_proj_bias": [
    8.884317398071289
  ]
}
```

## Post-run verified boundary and subset review

This section is rendered from the saved JSON/NPZ without rerunning TrackOn2. The execution-script hash in the integrity block remains unchanged.

### K-boundary score gaps

| Map | K16 gap median | K16 exact/near tie | K64 gap median | K64 exact/near tie |
|---|---:|---:|---:|---:|
| fused | 0.00314331 | 0.0002/0.0002 | 0.000960827 | 0.0002/0.0008 |
| c4 | 0.000416156 | 0.0000/0.0016 | 0.000122011 | 0.0000/0.0062 |
| c8 | 0.000694372 | 0.0000/0.0012 | 0.000207175 | 0.0002/0.0039 |
| c16 | 0.000779726 | 0.0000/0.0018 | 0.000283442 | 0.0000/0.0025 |
| c32 | 0.00127728 | 0.0000/0.0010 | 0.00093041 | 0.0000/0.0008 |

Fused exact boundary ties occur on only 0.0205% of rows at K16 and K64. The top16 ordered-coordinate difference is therefore a tie-order diagnostic; the candidate set is exactly identical and the recall headroom is not a tie artifact.

### Hard/easy concentration

| Subset | N | Fused K16 R@4 | Fused K64 R@4 | Headroom |
|---|---:|---:|---:|---:|
| hard | 2419 | 0.5420 | 0.8561 | +0.3142 |
| easy | 2459 | 1.0000 | 1.0000 | +0.0000 |

Easy rows are already saturated at 100% recall@4 with fused top16. The +31.42 percentage-point hard-row gain shows that candidate expansion should be event/risk activated rather than globally increasing K on every row.

Raw c4 is reported only as a diagnostic. Its learned `ms_corr_proj` coefficient is negative, so raw-cosine descending top-K is not a fair contribution-ranked candidate source and is excluded from the primary fused-map gate.

## Decision

POINTODYSSEY_CANDIDATE_RECALL_HEADROOM_PASS: every sequence has at least two percentage points of fused-map recall@4px between top16 and top64. Freeze this as synthetic upstream headroom. Do not automatically read DAVIS or train an adapter; combine this result with the V9-A5.0 temporal-state audit to choose the next full-stream synthetic experiment.
