# ReEntry-TAP External Baseline Smoke — 2026-07-01

## Goal

Attempt the external-baseline sprint for a stronger paper version:

```text
1. TAPNext / TAPNext-style baseline on ReEntry-TAP stress
2. TrackOn2 baseline on ReEntry-TAP stress
```

The purpose is not to add weak or non-comparable numbers to the main table, but to determine whether an external long-term / re-detection tracker can be evaluated under the same ReEntry-TAP stress protocol.

---

## 1. TAPNext stress smoke

### New exporter

Implemented:

```text
scripts/export_tapnext_reentry_stress_cache.py
```

This adapts the previous RGB-Stacking TAPNext exporter to ReEntry-TAP `stress_dataset.pt` files.

Input:

```text
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16/stress_dataset.pt
```

Output:

```text
outputs/paper_discovery_2026-06-27/external_baseline_smoke/tapnext_translate_L16_dev0/tapnext_translate_L16_dev0.pt
outputs/paper_discovery_2026-06-27/external_baseline_smoke/tapnext_translate_L16_dev0/tapnext_translate_L16_dev0_report.json
```

Smoke status:

```text
video = rgb_stacking_000000_translate_L16
records = 1
queries = 1116
export_sec = 24.255
vis_rate = 0.1030
peak_mem_mb = 2214.6
```

### Same-video comparison: dev0 translate_L16

| Method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | queries |
|---|---:|---:|---:|---:|---:|
| CoTracker3 offline | 0.6667 | 76.0920 | 88.5175 | 87.0434 | 1116 |
| CoTracker3 online | 0.7061 | 46.4554 | 61.2806 | 67.9842 | 1116 |
| B2-W16-P2 | 0.7239 | 77.2732 | 90.6263 | 87.4162 | 1116 |
| TAPNext | 0.0311 | 10.7467 | 28.5831 | 19.7590 | 1116 |

### Interpretation

TAPNext export path works technically, but the available local TAPNext checkpoint / adapter is not a strong comparable baseline under this ReEntry-TAP stress protocol.

Do **not** add this TAPNext result to the main comparison table as a strong baseline.

Safe wording if needed:

```text
We implemented a TAPNext stress exporter, but the available local TAPNext checkpoint did not establish parity under the ReEntry-TAP stress protocol, so we do not use it as a main baseline.
```

---

## 2. TrackOn2 stress smoke

### New exporter attempt

Implemented:

```text
scripts/export_trackon2_reentry_stress_cache.py
```

Planned smoke:

```text
stress: dev0 translate_L16
queries: 256-query smoke
checkpoint: baselines/track_on/checkpoints_trackon2_dinov3.pt
config: baselines/track_on/config/test.yaml
```

### Current blocker

The smoke is blocked by missing `mmcv` in the current execution environment:

```text
ModuleNotFoundError: No module named 'mmcv'
```

Failure path:

```text
from model.trackon_predictor import Predictor
from model.trackon import Track_On2
from model.modules import MHA_Block, SimpleFPN
from mmcv.ops import MultiScaleDeformableAttention
```

This is an environment/dependency blocker, not a ReEntry-TAP metric failure.

### Important context

TrackOn2 is not invalid as a baseline. We already have a parity-valid first-query/input-resolution DAVIS bridge:

```text
docs/first_input_trackon2_b2w_plugin_experiment_2026-06-29.md
```

Key result from that existing valid protocol:

```text
TrackOn2 first-input:
AJ_RD_256 = 0.5444
AJ_256    = 67.0406

B2-W16-P2, TrackOn2 base + CoTracker3 override:
AJ_RD_256 = 0.5509
AJ_256    = 67.1260

Gain over TrackOn2:
AJ_RD_256 +0.0065
AJ_256    +0.0854
```

This can be used as a **supplementary external plug-in generality result**, not as a main ReEntry-TAP stress baseline.

Safe wording:

```text
Under a parity-valid first-query/input-resolution DAVIS protocol, the same local override idea also gives a small positive gain on top of a reproduced TrackOn2 baseline.
```

Unsafe wording:

```text
Do not claim B2 generally beats TrackOn2 under all protocols.
Do not mix first-query/input-resolution TrackOn2 numbers with the main strided-original RGB/ReEntry-TAP tables.
```

---

## Current decision

External baseline status:

```text
TAPNext ReEntry-TAP stress smoke: runnable but weak / not parity-established.
TrackOn2 ReEntry-TAP stress smoke: blocked by missing mmcv dependency.
TrackOn2 first-input DAVIS bridge: valid supplemental external plug-in result already available.
```

Recommended use in paper:

```text
Main paper:
  Use CoTracker3 offline/online/B2/ReEntry-Guard for the primary ReEntry-TAP method story.

Appendix / supplementary:
  Add TrackOn2 first-input plug-in result to show the local override idea can give positive gains on a reproduced external baseline.

Do not include TAPNext stress smoke as a strong baseline.
```

---

## Next possible actions

### Option A: Environment fix for TrackOn2 stress

Install / activate the environment containing:

```text
mmcv with MultiScaleDeformableAttention
```

Then rerun:

```text
scripts/export_trackon2_reentry_stress_cache.py
```

### Option B: Use existing TrackOn2 bridge as supplement

No new environment work needed. Add the existing TrackOn2 first-input plug-in experiment to the paper appendix.

### Option C: Focus on ReEntry-Guard method story

Since ReEntry-Guard v2 already improves beyond B2 and the external stress baseline is blocked, continue improving the main method paper draft and use TrackOn2 first-input as supplementary evidence.

## 3. Environment checker and supplement summary

Added environment checker:

```text
scripts/check_trackon2_environment.py
outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_env_check.json
```

Current checks confirm:

```text
base env: CUDA PyTorch exists, mmcv missing, TrackOn2 import fails.
trackon2_mmcv env: mmengine exists, CPU PyTorch, mmcv missing, TrackOn2 import fails.
```

Added TrackOn2 supplement summarizer:

```text
scripts/summarize_trackon2_first_input_supplement.py
outputs/paper_discovery_2026-06-27/first_input_trackon2_b2w/first_input_trackon2_supplement_summary.json
docs/reentry_tap_trackon2_first_input_supplement_2026-07-01.md
```
