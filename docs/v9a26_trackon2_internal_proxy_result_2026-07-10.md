# V9-A2.6 TrackOn2 Internal-State Proxy Result

## Provenance

```text
TrackOn2 256-space M24 support-grid20 internal-state proxy; not exact old-cache latent state
```

Primary: v9a26_all_plus_internal_logreg_event_max_oof_f1

No dense trajectory threshold search was used.

## Classifier OOF

| Feature set / model | Features | AP | AUC | Brier | OOF-F1 threshold |
|---|---:|---:|---:|---:|---:|
| internal_only / logreg | 63 | 0.3262 | 0.5972 | 0.2282 | 0.292621 |
| internal_only / extratrees | 63 | 0.2391 | 0.5318 | 0.1806 | 0.189101 |
| base_plus_internal / logreg | 91 | 0.2341 | 0.4994 | 0.2958 | 0.000001 |
| base_plus_internal / extratrees | 91 | 0.2123 | 0.5062 | 0.1808 | 0.108035 |
| all_plus_internal / logreg | 206 | 0.2030 | 0.4930 | 0.3225 | 0.016976 |
| all_plus_internal / extratrees | 206 | 0.2301 | 0.5328 | 0.1763 | 0.294658 |

## Top single internal features

| Feature | Best AUC | Direction |
|---|---:|---|
| qinit_qnew_l2 | 0.6943 | high |
| c1_softmax_entropy_norm | 0.6928 | high |
| qnew_norm | 0.6890 | high |
| c1_std | 0.6830 | low |
| c1_native_score | 0.6794 | low |
| c2_softmax_entropy_norm | 0.6785 | high |
| c2_native_score | 0.6778 | low |
| rerank_u_min | 0.6747 | high |
| c1_old_candidate_score | 0.6698 | low |
| c1_max | 0.6694 | low |
| uncertainty_logit | 0.6677 | high |
| uncertainty_sigmoid | 0.6677 | high |
| rerank_s_max | 0.6671 | low |
| c2_std | 0.6670 | low |
| c2_old_candidate_score | 0.6667 | low |

## Baselines

| Variant | AJ Δ | OA Δ | delta_avg Δ | delta_4px Δ | AJ_RD Δ | AJ_RD_256 Δ | Ext accepted/good/bad/false/worse | Pos/Neg/Zero |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| w8_preserve_common_only | +0.0964 | +0.9462 | +0.3971 | +0.5085 | +0.0153 | +0.0274 | 0/0/0/0/0 | 16/4/5 |
| w16_accept_all_joint | +0.0810 | +0.9701 | +0.4339 | +0.5593 | +0.0162 | +0.0286 | 557/118/168/67/248 | 15/5/5 |
| frozen_v9a2_fixed0.05 | +0.1025 | +0.9711 | +0.4416 | +0.5848 | +0.0175 | +0.0294 | 481/109/141/61/190 | 17/3/5 |
| frozen_v9a2_oof_f1 | +0.0975 | +0.9746 | +0.4391 | +0.5704 | +0.0169 | +0.0293 | 515/116/153/62/216 | 17/3/5 |
| oracle_ext_candidate_good | +0.1955 | +1.0204 | +0.5073 | +0.6152 | +0.0191 | +0.0321 | 118/118/0/0/4 | 17/3/5 |

## Internal proxy variants

| Variant | AJ Δ | OA Δ | delta_avg Δ | delta_4px Δ | AJ_RD Δ | AJ_RD_256 Δ | Ext accepted/good/bad/false/worse | Pos/Neg/Zero |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| v9a26_internal_only_logreg_event_max_oof_f1 | +0.0664 | +0.9711 | +0.4143 | +0.5275 | +0.0148 | +0.0271 | 394/91/149/56/169 | 16/4/5 |
| v9a26_internal_only_logreg_event_max_fixed0.05 | +0.0833 | +0.9701 | +0.4350 | +0.5562 | +0.0163 | +0.0285 | 525/115/160/66/225 | 15/5/5 |
| v9a26_internal_only_extratrees_event_max_oof_f1 | +0.0851 | +0.9746 | +0.4309 | +0.5593 | +0.0165 | +0.0289 | 480/109/163/64/210 | 15/5/5 |
| v9a26_base_plus_internal_logreg_event_max_oof_f1 | +0.0824 | +0.9735 | +0.4339 | +0.5593 | +0.0162 | +0.0286 | 555/118/166/65/248 | 15/5/5 |
| v9a26_base_plus_internal_logreg_event_max_fixed0.05 | +0.0681 | +0.9613 | +0.4141 | +0.5275 | +0.0145 | +0.0269 | 492/97/155/61/224 | 15/5/5 |
| v9a26_base_plus_internal_extratrees_event_max_oof_f1 | +0.0810 | +0.9701 | +0.4339 | +0.5593 | +0.0162 | +0.0286 | 557/118/168/67/248 | 15/5/5 |
| v9a26_all_plus_internal_logreg_event_max_oof_f1 | +0.0844 | +0.9781 | +0.4273 | +0.5429 | +0.0159 | +0.0280 | 485/106/152/61/204 | 17/3/5 |
| v9a26_all_plus_internal_logreg_event_max_fixed0.05 | +0.0786 | +0.9625 | +0.4138 | +0.5412 | +0.0162 | +0.0282 | 444/96/139/58/187 | 17/3/5 |
| v9a26_all_plus_internal_extratrees_event_max_oof_f1 | +0.0805 | +0.9701 | +0.4315 | +0.5673 | +0.0150 | +0.0275 | 243/63/109/58/103 | 14/6/5 |

## Paired video-level AJ_RD_256

| Comparison | N | Mean | Median | Bootstrap 95% CI | Better/Worse/Equal | Exact sign-flip p |
|---|---:|---:|---:|---|---|---:|
| primary_vs_frozen_fixed0.05 | 25 | -0.001224 | +0.000000 | [-0.003855, +0.000207] | 3/3/19 | 0.6250 |
| primary_vs_w16 | 25 | -0.000559 | +0.000000 | [-0.003179, +0.001278] | 4/4/17 | 0.8828 |

## Decision

TrackOn2 internal proxy features do not improve the frozen V9-A2 policy under the predeclared OOF protocol.
The selector route is saturated; proceed to V9-A3 multi-hypothesis candidate generation / trainable reacquisition.

## Structured feature-family audit

```text
visibility/uncertainty OOF AP/AUC: 0.4368 / 0.6304
query-update OOF AP/AUC: 0.4230 / 0.6740
5-fold memory-consistency AJ_RD_256 Δ: +0.030291
5-fold memory-consistency paired CI vs frozen: [-0.000400,+0.006235]
```

## Leave-one-video-out robustness

```text
LOGO memory-consistency AJ_RD_256 Δ: +0.028353
LOGO query-update AJ_RD_256 Δ: +0.030038
LOGO query-update aggregate vs frozen: +0.000659
LOGO query-update paired CI: [-0.000499,+0.006168]
No one of the 13 predeclared internal families has a positive paired-CI lower bound.
```

## Integrity audit

```text
557 x 64 finite internal proxy matrix; strict joint-index and row-key alignment.
Old smoke == enhanced smoke == full prefix, bit-for-bit.
Official TrackOn2 position/logit/query max_abs error: 0 on 20 first-active checks and 3 long-sequence checks.
Proxy-to-old-candidate distance p95: 10.49 internal-model pixels.
The proxy is not claimed to be the exact historical-cache latent state.
```

## Final V9-A2.6 decision

Selector-side internal features contain real heldout signal, but the promising aggregate rows are not stable under paired uncertainty and LOGO. Stop selector threshold/feature/fusion tuning. Proceed to V9-A3.0 top-K multi-hypothesis action-space oracle.
