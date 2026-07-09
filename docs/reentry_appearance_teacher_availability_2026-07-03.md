# ReEntry Appearance / Teacher Availability Inspection

## Recommendation

Best next route: `dino_patch_features`

Project contains DINO-like checkpoint references and ViT tooling is available.

Signals:
- `has_dino_checkpoint`: `True`
- `has_vit_tooling`: `True`
- `has_external_teacher_reference`: `True`
- `has_cotracker_reference`: `True`

## Environment

Python: `3.11.8 (main, Feb 26 2024, 21:39:34) [GCC 11.2.0]`
Platform: `Linux-5.4.0-42-generic-x86_64-with-glibc2.35`

Torch details:

```json
{
  "torch_version": "2.2.2+cu121",
  "cuda_available": true,
  "cuda_version": "12.1",
  "device_count": 1,
  "device_name_0": "B1.gpu.large"
}
```

## Python packages

| package | available | version | file | error |
|---|---|---|---|---|
| torch | True | 2.2.2+cu121 | /root/miniconda3/lib/python3.11/site-packages/torch/__init__.py | None |
| torchvision | True | 0.17.2+cu121 | /root/miniconda3/lib/python3.11/site-packages/torchvision/__init__.py | None |
| timm | True | 1.0.3 | /root/miniconda3/lib/python3.11/site-packages/timm/__init__.py | None |
| transformers | True | 4.56.1 | /root/miniconda3/lib/python3.11/site-packages/transformers/__init__.py | None |
| PIL | True | 10.0.1 | /root/miniconda3/lib/python3.11/site-packages/PIL/__init__.py | None |
| cv2 | True | 4.9.0 | /root/miniconda3/lib/python3.11/site-packages/cv2/__init__.py | None |
| sklearn | True | 1.4.2 | /root/miniconda3/lib/python3.11/site-packages/sklearn/__init__.py | None |
| numpy | True | 1.26.4 | /root/miniconda3/lib/python3.11/site-packages/numpy/__init__.py | None |
| scipy | True | 1.13.0 | /root/miniconda3/lib/python3.11/site-packages/scipy/__init__.py | None |

## Top checkpoint candidates

| path | suffix | size MB | hits |
|---|---|---|---|
| /gemini/code/FSPT/baselines/track_on/checkpoints_trackon2_dinov3.pt | .pt | 89.552 | dino, dinov3, trackon, track_on |
| /gemini/code/FSPT/baselines/track_on/checkpoints_trackon2_dinov2.pt | .pt | 89.55 | dino, dinov2, trackon, track_on |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_translate_L16_dev0_full/trackon2_dinov3_translate_L16_dev0_full.pt | .pt | 9.086 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_translate_L16_dev0_full/visibility_variants/trackon2_dinov3_translate_L16_dev0_full_all_visible.pt | .pt | 9.086 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_translate_L16_dev0_full/visibility_variants/trackon2_dinov3_translate_L16_dev0_full_gt_visibility.pt | .pt | 9.086 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_translate_L16_dev0_full/visibility_variants/trackon2_dinov3_translate_L16_dev0_full_original.pt | .pt | 9.086 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_translate_L16_dev0_full/visibility_variants/trackon2_dinov3_translate_L16_dev0_full_query_visible_fill.pt | .pt | 9.086 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_occluder_L16_dev0_full/trackon2_dinov3_occluder_L16_dev0_full.pt | .pt | 8.78 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_occluder_L16_dev0_full/visibility_variants/trackon2_dinov3_occluder_L16_dev0_full_all_visible.pt | .pt | 8.78 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_occluder_L16_dev0_full/visibility_variants/trackon2_dinov3_occluder_L16_dev0_full_gt_visibility.pt | .pt | 8.78 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_occluder_L16_dev0_full/visibility_variants/trackon2_dinov3_occluder_L16_dev0_full_original.pt | .pt | 8.78 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_occluder_L16_dev0_full/visibility_variants/trackon2_dinov3_occluder_L16_dev0_full_query_visible_fill.pt | .pt | 8.78 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_translate_L16_dev0_256q_unitinput/trackon2_dinov3_translate_L16_dev0_256q_unitinput.pt | .pt | 2.177 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_variant_256q/translate_L16/sg0_uncond/trackon2_dinov3_translate_L16_sg0_uncond_256q.pt | .pt | 2.167 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_translate_L16_dev0_256q/trackon2_dinov3_translate_L16_dev0_256q.pt | .pt | 2.166 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_translate_L16_dev0_256q_dv0.5/trackon2_dinov3_translate_L16_dev0_256q_dv0.5.pt | .pt | 2.166 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_translate_L16_dev0_256q_dv0.6/trackon2_dinov3_translate_L16_dev0_256q_dv0.6.pt | .pt | 2.166 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_translate_L16_dev0_256q_dv0.7/trackon2_dinov3_translate_L16_dev0_256q_dv0.7.pt | .pt | 2.166 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_translate_L16_dev0_256q_visselect/trackon2_dinov3_translate_L16_dev0_256q_visselect.pt | .pt | 2.165 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_variant_256q/translate_L16/sg20_vismem/trackon2_dinov3_translate_L16_sg20_vismem_256q.pt | .pt | 2.165 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_variant_256q/translate_L16/sg0_vismem/trackon2_dinov3_translate_L16_sg0_vismem_256q.pt | .pt | 2.164 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_occluder_L16_dev0_256q/trackon2_dinov3_occluder_L16_dev0_256q.pt | .pt | 2.155 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_variant_256q/occluder_L16/sg0_uncond/trackon2_dinov3_occluder_L16_sg0_uncond_256q.pt | .pt | 2.153 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_variant_256q/occluder_L16/sg0_vismem/trackon2_dinov3_occluder_L16_sg0_vismem_256q.pt | .pt | 2.153 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_occluder_L16_dev0_256q_visselect/trackon2_dinov3_occluder_L16_dev0_256q_visselect.pt | .pt | 2.152 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_dinov3_variant_256q/occluder_L16/sg20_vismem/trackon2_dinov3_occluder_L16_sg20_vismem_256q.pt | .pt | 2.152 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt | .pt | 0.946 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis.pt | .pt | 0.933 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_kinetics_cache/kinetics/trackon2/000000.npz | .npz | 0.065 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_kinetics_cache/kinetics/trackon2/000003.npz | .npz | 0.065 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_kinetics_cache/kinetics/trackon2/000004.npz | .npz | 0.065 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_kinetics_cache/kinetics/trackon2/000005.npz | .npz | 0.065 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_kinetics_cache/kinetics/trackon2/000006.npz | .npz | 0.065 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_kinetics_cache/kinetics/trackon2/000007.npz | .npz | 0.065 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_kinetics_cache/kinetics/trackon2/000009.npz | .npz | 0.065 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_kinetics_cache/kinetics/trackon2/000002.npz | .npz | 0.061 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_kinetics_cache/kinetics/trackon2/000008.npz | .npz | 0.058 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_kinetics_cache/kinetics/trackon2/000001.npz | .npz | 0.052 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000025.npz | .npz | 0.022 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000029.npz | .npz | 0.022 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000008.npz | .npz | 0.02 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000003.npz | .npz | 0.019 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000002.npz | .npz | 0.018 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000016.npz | .npz | 0.018 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000024.npz | .npz | 0.018 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000014.npz | .npz | 0.017 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000022.npz | .npz | 0.017 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000000.npz | .npz | 0.015 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000010.npz | .npz | 0.015 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000005.npz | .npz | 0.013 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000011.npz | .npz | 0.012 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000012.npz | .npz | 0.011 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000015.npz | .npz | 0.011 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000018.npz | .npz | 0.011 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000019.npz | .npz | 0.011 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000020.npz | .npz | 0.011 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000021.npz | .npz | 0.011 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000026.npz | .npz | 0.011 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000027.npz | .npz | 0.01 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000006.npz | .npz | 0.009 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000007.npz | .npz | 0.009 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000023.npz | .npz | 0.009 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000028.npz | .npz | 0.009 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000004.npz | .npz | 0.008 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000017.npz | .npz | 0.008 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000009.npz | .npz | 0.006 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000013.npz | .npz | 0.004 | dino, dinov3, trackon |
| /gemini/code/FSPT/outputs/trackon2_dinov3_davis_cache/davis/trackon2/000001.npz | .npz | 0.003 | dino, dinov3, trackon |
| /gemini/code/FSPT/baselines/hf/facebook_dinov2_small/model.safetensors | .safetensors | 84.162 | dino, dinov2 |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/first_input_trackon2_b2w/b2_w16_base_invis_t_cotracker_base_trackon2_override_first_input.pt | .pt | 0.958 | trackon, cotracker |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/first_input_trackon2_b2w/b2_w16_trackon2_override/b2_w16_cotracker_base_trackon2_override_first_input.pt | .pt | 0.958 | trackon, cotracker |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/first_input_trackon2_b2w/b2_w16_k2_cotracker_base_trackon2_override_first_input.pt | .pt | 0.957 | trackon, cotracker |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/first_input_trackon2_b2w/b2_w16_p2_cotracker_base_trackon2_override_first_input.pt | .pt | 0.957 | trackon, cotracker |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/first_input_trackon2_b2w/b2_w16_p2_trackon2_base_cotracker_override_first_input.pt | .pt | 0.947 | trackon, cotracker |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/first_input_trackon2_b2w/b2_w16_trackon2_base_cotracker_override/b2_w16_trackon2_base_cotracker_override_first_input.pt | .pt | 0.947 | trackon, cotracker |
| /gemini/code/FSPT/checkpoints/tapnext/bootstapnext_ckpt.npz | .npz | 740.986 | tapnext |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_49/translate_L16/predictions/cotracker3_online_translate_L16.pt | .pt | 208.669 | cotracker |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_49/translate_L16/predictions/cotracker3_online_translate_L16_fresh20_49.pt | .pt | 208.669 | cotracker |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_49/translate_L16/predictions/cotracker3_offline_translate_L16.pt | .pt | 208.577 | cotracker |
| /gemini/code/FSPT/outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_49/translate_L16/predictions/cotracker3_offline_translate_L16_fresh20_49.pt | .pt | 208.577 | cotracker |

## Code references

### `/gemini/code/FSPT/baselines/cotracker/CODE_OF_CONDUCT.md`
Path hits: `cotracker`

### `/gemini/code/FSPT/baselines/cotracker/CONTRIBUTING.md`
Path hits: `cotracker`
- L1: `# CoTracker`
- L27: `By contributing to CoTracker, you agree that your contributions will be licensed`

### `/gemini/code/FSPT/baselines/cotracker/LICENSE.md`
Path hits: `cotracker`

### `/gemini/code/FSPT/baselines/cotracker/README.md`
Path hits: `cotracker`
- L1: `# CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos`
- L7: `### [Project Page](https://cotracker3.github.io/) | [Paper #1](https://arxiv.org/abs/2307.07635) | [Paper #2](https://arxiv.org/abs/2410.11831) |  [X Thread](https://twitter.com/n_karaev/status/1742638906355470772) | [Bi`
- L12: `<a href="https://huggingface.co/spaces/facebook/cotracker">`
- L18: `**CoTracker** is a fast transformer-based model that can track any point in a video. It brings to tracking some of the benefits of Optical Flow.`
- L20: `CoTracker can track:`

### `/gemini/code/FSPT/baselines/cotracker/demo.py`
Path hits: `cotracker`
- L13: `from cotracker.utils.visualizer import Visualizer, read_video_from_path`
- L14: `from cotracker.predictor import CoTrackerPredictor`
- L37: `# default="./checkpoints/cotracker.pth",`
- L39: `help="CoTracker model parameters",`
- L56: `help="Pass it if you wish to use CoTracker2, CoTracker++ is the default now",`

### `/gemini/code/FSPT/baselines/cotracker/hubconf.py`
Path hits: `cotracker`
- L9: `_COTRACKER2_URL = (`
- L10: `"https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth"`
- L12: `_COTRACKER2v1_URL = (`
- L13: `"https://huggingface.co/facebook/cotracker/resolve/main/cotracker2v1.pth"`
- L15: `_COTRACKER3_SCALED_OFFLINE_URL = (`

### `/gemini/code/FSPT/baselines/cotracker/online_demo.py`
Path hits: `cotracker`
- L13: `from cotracker.utils.visualizer import Visualizer`
- L14: `from cotracker.predictor import CoTrackerOnlinePredictor`
- L31: `help="CoTracker model parameters",`
- L47: `model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)`
- L49: `model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")`

### `/gemini/code/FSPT/baselines/cotracker/setup.py`
Path hits: `cotracker`
- L10: `name="cotracker",`

### `/gemini/code/FSPT/baselines/cotracker/train_on_kubric.py`
Path hits: `cotracker`
- L25: `from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline`
- L26: `from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeOnline`
- L28: `from cotracker.utils.visualizer import Visualizer`
- L29: `from cotracker.models.core.model_utils import get_uniformly_sampled_pts`
- L30: `from cotracker.evaluation.core.evaluator import Evaluator`

### `/gemini/code/FSPT/baselines/cotracker/train_on_real_data.py`
Path hits: `cotracker`
- L27: `from cotracker.models.bootstap_predictor import TAPIRPredictor`
- L28: `from cotracker.models.core.cotracker.cotracker import CoTracker2`
- L29: `from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline`
- L30: `from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeOnline`
- L32: `from cotracker.utils.visualizer import Visualizer`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/__init__.py`
Path hits: `cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/predictor.py`
Path hits: `cotracker`
- L10: `from cotracker.models.core.model_utils import smart_cat, get_points_on_a_grid`
- L11: `from cotracker.models.build_cotracker import build_cotracker`
- L14: `class CoTrackerPredictor(torch.nn.Module):`
- L25: `model = build_cotracker(`
- L212: `class CoTrackerOnlinePredictor(torch.nn.Module):`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/version.py`
Path hits: `cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/cotracker_eval_backup/__init__.py`
Path hits: `cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/cotracker_eval_backup/evaluate.py`
Path hits: `cotracker`
- L18: `from cotracker.datasets.utils import collate_fn`
- L19: `from cotracker.models.evaluation_predictor import EvaluationPredictor`
- L21: `from cotracker.evaluation.core.evaluator import Evaluator`
- L22: `from cotracker.models.build_cotracker import build_cotracker`
- L36: `# The default value is the path to a specific CoTracker model checkpoint.`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/cotracker_eval_backup/configs/eval_dynamic_replica.yaml`
Path hits: `cotracker`
- L3: `exp_dir: ./outputs/cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/cotracker_eval_backup/configs/eval_tapvid_davis_first.yaml`
Path hits: `cotracker`
- L3: `exp_dir: ./outputs/cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/cotracker_eval_backup/configs/eval_tapvid_davis_strided.yaml`
Path hits: `cotracker`
- L3: `exp_dir: ./outputs/cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/cotracker_eval_backup/configs/eval_tapvid_kinetics_first.yaml`
Path hits: `cotracker`
- L3: `exp_dir: ./outputs/cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/cotracker_eval_backup/configs/eval_tapvid_robotap_first.yaml`
Path hits: `cotracker`
- L3: `exp_dir: ./outputs/cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/cotracker_eval_backup/configs/eval_tapvid_stacking_first.yaml`
Path hits: `cotracker`
- L3: `exp_dir: ./outputs/cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/cotracker_eval_backup/configs/eval_tapvid_stacking_strided.yaml`
Path hits: `cotracker`
- L3: `exp_dir: ./outputs/cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/cotracker_eval_backup/core/__init__.py`
Path hits: `cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/cotracker_eval_backup/core/eval_utils.py`
Path hits: `cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/cotracker_eval_backup/core/evaluator.py`
Path hits: `cotracker`
- L15: `from cotracker.datasets.utils import dataclass_to_cuda_`
- L16: `from cotracker.utils.visualizer import Visualizer`
- L17: `from cotracker.models.core.model_utils import reduce_masked_mean`
- L18: `from cotracker.evaluation.core.eval_utils import compute_tapvid_metrics`
- L19: `from cotracker.predictor import CoTrackerOnlinePredictor`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/cotracker_eval_backup/multirun/2026-03-02/22-39-03/multirun.yaml`
Path hits: `cotracker`
- L132: `cwd: /gemini/code/FSPT/baselines/cotracker/cotracker/evaluation`
- L137: `- path: /gemini/code/FSPT/baselines/cotracker/cotracker/evaluation/configs`
- L155: `exp_dir: ./outputs/cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/datasets/__init__.py`
Path hits: `cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/datasets/dataclass_utils.py`
Path hits: `cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/datasets/dr_dataset.py`
Path hits: `cotracker`
- L17: `from cotracker.datasets.utils import CoTrackerData`
- L18: `from cotracker.datasets.dataclass_utils import load_dataclass`
- L162: `return CoTrackerData(`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/datasets/kubric_movif_dataset.py`
Path hits: `cotracker`
- L14: `from cotracker.datasets.utils import CoTrackerData`
- L17: `from cotracker.models.core.model_utils import smart_cat`
- L20: `class CoTrackerDataset(torch.utils.data.Dataset):`
- L30: `super(CoTrackerDataset, self).__init__()`
- L79: `sample = CoTrackerData(`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/datasets/real_dataset.py`
Path hits: `cotracker`
- L15: `from cotracker.datasets.utils import CoTrackerData`
- L18: `from cotracker.models.core.model_utils import smart_cat`
- L21: `from cotracker.datasets.utils import collate_fn, collate_fn_train, dataclass_to_cuda_`
- L203: `sample = CoTrackerData(`
- L271: `sample = CoTrackerData(`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/datasets/tap_vid_datasets.py`
Path hits: `cotracker`
- L18: `from cotracker.datasets.utils import CoTrackerData`
- L35: `"""Package a set of frames and tracks for use in TAPNet evaluations.`
- L78: `"""Package a set of frames and tracks for use in TAPNet evaluations.`
- L235: `return CoTrackerData(`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/datasets/utils.py`
Path hits: `cotracker`
- L16: `class CoTrackerData:`
- L47: `return CoTrackerData(`
- L77: `CoTrackerData(`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/models/__init__.py`
Path hits: `cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/models/bootstap_predictor.py`
Path hits: `cotracker`
- L9: `from tapnet.torch.tapir_model import TAPIR`
- L17: `class TAPIRPredictor(torch.nn.Module):`
- L23: `checkpoint = "./tapnet/bootstapir_checkpoint.pt"`
- L24: `model = TAPIR(pyramid_level=1, extra_convs=True)`
- L26: `checkpoint = "./tapnet/tapir_checkpoint_panning.pt"`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/models/build_cotracker.py`
Path hits: `cotracker`
- L9: `from cotracker.models.core.cotracker.cotracker import CoTracker2`
- L10: `from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline`
- L11: `from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeOnline`
- L14: `def build_cotracker(`
- L18: `return build_cotracker()`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/models/evaluation_predictor.py`
Path hits: `cotracker`
- L11: `from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline`
- L12: `from cotracker.models.core.model_utils import (`
- L22: `from cotracker.models.core.model_utils import bilinear_sampler`
- L28: `cotracker_model: CoTrackerThreeOffline,`
- L46: `self.model = cotracker_model`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/models/core/__init__.py`
Path hits: `cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/models/core/embeddings.py`
Path hits: `cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/models/core/model_utils.py`
Path hits: `cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/models/core/cotracker/__init__.py`
Path hits: `cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/models/core/cotracker/blocks.py`
Path hits: `cotracker`
- L16: `from cotracker.models.core.model_utils import bilinear_sampler`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/models/core/cotracker/cotracker.py`
Path hits: `cotracker`
- L11: `from cotracker.models.core.model_utils import sample_features4d, sample_features5d`
- L12: `from cotracker.models.core.embeddings import (`
- L18: `from cotracker.models.core.cotracker.blocks import (`
- L29: `class CoTracker2(nn.Module):`
- L38: `super(CoTracker2, self).__init__()`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/models/core/cotracker/cotracker3_offline.py`
Path hits: `cotracker`
- L10: `from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeBase, posenc`
- L15: `class CoTrackerThreeOffline(CoTrackerThreeBase):`
- L17: `super(CoTrackerThreeOffline, self).__init__(**args)`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/models/core/cotracker/cotracker3_online.py`
Path hits: `cotracker`
- L10: `from cotracker.models.core.model_utils import sample_features5d, bilinear_sampler`
- L11: `from cotracker.models.core.embeddings import get_1d_sincos_pos_embed_from_grid`
- L13: `from cotracker.models.core.cotracker.blocks import Mlp, BasicEncoder`
- L14: `from cotracker.models.core.cotracker.cotracker import EfficientUpdateFormer`
- L42: `class CoTrackerThreeBase(nn.Module):`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/models/core/cotracker/losses.py`
Path hits: `cotracker`
- L9: `from cotracker.models.core.model_utils import reduce_masked_mean`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/utils/__init__.py`
Path hits: `cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/utils/train_utils.py`
Path hits: `cotracker`
- L13: `from cotracker.datasets.utils import collate_fn, collate_fn_train`
- L15: `from cotracker.datasets.dr_dataset import DynamicReplicaDataset`
- L16: `from cotracker.models.evaluation_predictor import EvaluationPredictor`
- L35: `from cotracker.datasets.tap_vid_datasets import TapVidDataset`
- L39: `from cotracker.datasets.dr_dataset import DynamicReplicaDataset`

### `/gemini/code/FSPT/baselines/cotracker/cotracker/utils/visualizer.py`
Path hits: `cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker.egg-info/SOURCES.txt`
Path hits: `cotracker`
- L4: `cotracker/__init__.py`
- L5: `cotracker/predictor.py`
- L6: `cotracker/version.py`
- L7: `cotracker.egg-info/PKG-INFO`
- L8: `cotracker.egg-info/SOURCES.txt`

### `/gemini/code/FSPT/baselines/cotracker/cotracker.egg-info/dependency_links.txt`
Path hits: `cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker.egg-info/requires.txt`
Path hits: `cotracker`

### `/gemini/code/FSPT/baselines/cotracker/cotracker.egg-info/top_level.txt`
Path hits: `cotracker`
- L1: `cotracker`

### `/gemini/code/FSPT/baselines/cotracker/docs/source/conf.py`
Path hits: `cotracker`
- L2: `exec(open("../../cotracker/version.py", "r").read())`
- L4: `project = "CoTracker"`

### `/gemini/code/FSPT/baselines/cotracker/gradio_demo/app.py`
Path hits: `cotracker`
- L1: `# This Gradio demo code is from https://github.com/cvlab-kaist/locotrack/blob/main/demo/demo.py`
- L2: `# We updated it to work with CoTracker3 models. We thank authors of LocoTrack`
- L123: `icon = np.clip(icon / (radius * 2 * sharpness), 0, 1)`
- L354: `model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")`
- L418: `gr.Markdown("# 🎨 CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos")`

### `/gemini/code/FSPT/baselines/cotracker/gradio_demo/requirements.txt`
Path hits: `cotracker`

### `/gemini/code/FSPT/baselines/cotracker/multirun/2026-03-02/22-52-14/multirun.yaml`
Path hits: `cotracker`
- L119: `- checkpoint=/gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth`
- L124: `override_dirname: checkpoint=/gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth,dataset_name=tapvid_davis_strided,dataset_root=/gemini/code/datasets,exp_dir=/gemini/code/FSPT/outputs/`
- L138: `cwd: /gemini/code/FSPT/baselines/cotracker`
- L143: `- path: /gemini/code/FSPT/baselines/cotracker/cotracker/evaluation/configs`
- L164: `checkpoint: /gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth`

### `/gemini/code/FSPT/baselines/cotracker/multirun/2026-03-02/22-53-44/multirun.yaml`
Path hits: `cotracker`
- L119: `- checkpoint=/gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth`
- L124: `override_dirname: checkpoint=/gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth,dataset_name=tapvid_davis_strided,dataset_root=/gemini/code/datasets,exp_dir=/gemini/code/FSPT/outputs/cotr_eval`
- L138: `cwd: /gemini/code/FSPT/baselines/cotracker`
- L143: `- path: /gemini/code/FSPT/baselines/cotracker/cotracker/evaluation/configs`
- L164: `checkpoint: /gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth`

### `/gemini/code/FSPT/baselines/cotracker/multirun/2026-03-02/22-54-55/multirun.yaml`
Path hits: `cotracker`
- L119: `- checkpoint=/gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth`
- L124: `override_dirname: checkpoint=/gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth,dataset_name=tapvid_davis_strided,dataset_root=/gemini/code/datasets,exp_dir=/gemini/code/FSPT/outputs/`
- L138: `cwd: /gemini/code/FSPT/baselines/cotracker`
- L143: `- path: /gemini/code/FSPT/baselines/cotracker/cotracker/evaluation/configs`
- L164: `checkpoint: /gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth`

### `/gemini/code/FSPT/baselines/cotracker/multirun/2026-03-02/22-58-16/multirun.yaml`
Path hits: `cotracker`
- L119: `- checkpoint=/gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth`
- L126: `override_dirname: checkpoint=/gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth,dataset_name=tapvid_davis_strided,dataset_root=/gemini/code/datasets,exp_dir=/gemini/code/FSPT/outputs/,offline_model=True`
- L140: `cwd: /gemini/code/FSPT/baselines/cotracker`
- L145: `- path: /gemini/code/FSPT/baselines/cotracker/cotracker/evaluation/configs`
- L166: `checkpoint: /gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth`

### `/gemini/code/FSPT/baselines/cotracker/multirun/2026-03-02/23-01-26/multirun.yaml`
Path hits: `cotracker`
- L119: `- checkpoint=/gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth`
- L126: `override_dirname: checkpoint=/gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth,dataset_name=tapvid_davis_strided,dataset_root=/gemini/code/datasets,exp_dir=/gemini/code/FSPT/outputs/,offline_model=True`
- L140: `cwd: /gemini/code/FSPT/baselines/cotracker`
- L145: `- path: /gemini/code/FSPT/baselines/cotracker/cotracker/evaluation/configs`
- L166: `checkpoint: /gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth`

### `/gemini/code/FSPT/baselines/cotracker/multirun/2026-03-02/23-02-39/multirun.yaml`
Path hits: `cotracker`
- L119: `- checkpoint=/gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth`
- L126: `override_dirname: checkpoint=/gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth,dataset_name=tapvid_davis_strided,dataset_root=/gemini/code/datasets,exp_dir=/gemini/code/FSPT/outputs/cotr_eval_offline_2`
- L140: `cwd: /gemini/code/FSPT/baselines/cotracker`
- L145: `- path: /gemini/code/FSPT/baselines/cotracker/cotracker/evaluation/configs`
- L166: `checkpoint: /gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth`

### `/gemini/code/FSPT/baselines/cotracker/multirun/2026-03-02/23-06-44/multirun.yaml`
Path hits: `cotracker`
- L119: `- checkpoint=/gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth`
- L126: `override_dirname: checkpoint=/gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth,dataset_name=tapvid_davis_strided,dataset_root=/gemini/code/datasets,exp_dir=/gemini/code/FSPT/outputs/cotr_eval_offline_2`
- L140: `cwd: /gemini/code/FSPT/baselines/cotracker`
- L145: `- path: /gemini/code/FSPT/baselines/cotracker/cotracker/evaluation/configs`
- L166: `checkpoint: /gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth`

### `/gemini/code/FSPT/baselines/cotracker/multirun/2026-03-02/23-13-24/multirun.yaml`
Path hits: `cotracker`
- L119: `- checkpoint=/gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth`
- L126: `override_dirname: checkpoint=/gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth,dataset_name=tapvid_davis_strided,dataset_root=/gemini/code/datasets,exp_dir=/gemini/code/FSPT/outputs/,offline_model=True`
- L140: `cwd: /gemini/code/FSPT/baselines/cotracker`
- L145: `- path: /gemini/code/FSPT/baselines/cotracker/cotracker/evaluation/configs`
- L166: `checkpoint: /gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth`

### `/gemini/code/FSPT/baselines/cotracker/multirun/2026-03-02/23-19-15/multirun.yaml`
Path hits: `cotracker`
- L119: `- checkpoint=/gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth`
- L126: `override_dirname: checkpoint=/gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth,dataset_name=tapvid_davis_strided,dataset_root=/gemini/code/datasets,exp_dir=/gemini/code/FSPT/outputs/cotr_eval_offline_2`
- L140: `cwd: /gemini/code/FSPT/baselines/cotracker`
- L145: `- path: /gemini/code/FSPT/baselines/cotracker/cotracker/evaluation/configs`
- L166: `checkpoint: /gemini/code/FSPT/baselines/cotracker/checkpoints/scaled_offline.pth`

### `/gemini/code/FSPT/baselines/cotracker/tests/test_bilinear_sample.py`
Path hits: `cotracker`
- L11: `from cotracker.models.core.model_utils import bilinear_sampler`

### `/gemini/code/FSPT/baselines/hf/facebook_dinov2_small/config.json`
Path hits: `dino, dinov2`
- L3: `"Dinov2Model"`
- L15: `"model_type": "dinov2",`

### `/gemini/code/FSPT/baselines/hf/facebook_dinov2_small/preprocessor_config.json`
Path hits: `dino, dinov2`

### `/gemini/code/FSPT/baselines/track_on/README.md`
Path hits: `track_on`
- L1: `# Track-On: Online Point Tracking with Memory`
- L3: `### [Project Page](https://kuis-ai.github.io/track_on_r/) | [Track-On-R](https://arxiv.org/abs/2603.12217) | [Track-On2](https://arxiv.org/abs/2509.19115) | [Track-On](https://arxiv.org/abs/2501.18487)`
- L6: `Official implementation of the **Track-On family of online point tracking models**.`
- L8: `Track-On is an **online point tracking model** that processes videos frame-by-frame using a compact transformer memory. **Track-On2** improves the architecture for stronger performance and efficiency, while **Track-On-R*`
- L11: `<img src="media/teaser.png" alt="Track-On Overview" width="600" />`

### `/gemini/code/FSPT/baselines/track_on/demo.py`
Path hits: `track_on`
- L2: `Track-On2 demo script`
- L23: `from model.trackon_predictor import Predictor`
- L29: `p = argparse.ArgumentParser(description="Track-On2 demo")`
- L32: `p.add_argument("--ckpt", required=True, type=str, help="Path to Track-On2 checkpoint .pth")`

### `/gemini/code/FSPT/baselines/track_on/main.py`
Path hits: `track_on`
- L33: `from model.trackon import Track_On2`
- L79: `nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
- L85: `nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
- L140: `# From CoTracker`
- L178: `model = Track_On2(args).to(args.gpu)`

### `/gemini/code/FSPT/baselines/track_on/main_real_world_ft.py`
Path hits: `track_on`
- L34: `from model.trackon import Track_On2`
- L41: `from model.trackon_predictor import Predictor as Trackon_Predictor`
- L42: `from ensemble.bootstapir.bootstapir_predictor import TAPIRPredictor`
- L43: `from ensemble.tapnext.tapnext_predictor import TAPNextPredictor`
- L44: `from ensemble.cotracker import CoTracker_Predictor`

### `/gemini/code/FSPT/baselines/track_on/main_verifier.py`
Path hits: `track_on`
- L10: `from ensemble.locotrack.locotrack_predictor import LocoTrackPredictor`
- L37: `from ensemble.cotracker import CoTracker_Predictor`
- L38: `from ensemble.bootstapir.bootstapir_predictor import TAPIRPredictor`
- L39: `from ensemble.tapnext.tapnext_predictor import TAPNextPredictor`
- L40: `from model.trackon_predictor import Predictor as Trackon_Predictor`

### `/gemini/code/FSPT/baselines/track_on/read_args.py`
Path hits: `track_on`
- L11: `parser = argparse.ArgumentParser("Track-On2")`
- L34: `parser.add_argument('--vit_backbone', type=str, choices=["dinov2_s", "dinov2_b", "dinov3_s", "dinov3_s_plus", "dinov3_b"], default="dinov2_s", help="Type of ViT backbone of ViT Adapter")`
- L35: `parser.add_argument('--vit_upsample_factor', type=float, default=1.0, help="Upsample factor for ViT inputs")`

### `/gemini/code/FSPT/baselines/track_on/requirements.txt`
Path hits: `track_on`

### `/gemini/code/FSPT/baselines/track_on/config/test.yaml`
Path hits: `track_on`
- L8: `vit_backbone: "dinov3_s_plus"`
- L9: `vit_upsample_factor: 1.143`

### `/gemini/code/FSPT/baselines/track_on/config/test_dinov2.yaml`
Path hits: `dino, dinov2, track_on`
- L8: `vit_backbone: "dinov2_b"`
- L9: `vit_upsample_factor: 1.0`

### `/gemini/code/FSPT/baselines/track_on/config/train.yaml`
Path hits: `track_on`
- L6: `model_save_path: "checkpoints/track_on2"`
- L22: `vit_backbone: "dinov3_s_plus"`
- L23: `vit_upsample_factor: 1.143`

### `/gemini/code/FSPT/baselines/track_on/config/train_real_world.yaml`
Path hits: `track_on`
- L1: `# Fine-tuned + Ensemble Track-On`
- L2: `trackon2_config_path: "./config/test.yaml"`
- L3: `trackon2_checkpoint_path: /path/to/trackon2/checkpoint.pt  # TODO: set path to Track-On2 checkpoint`
- L16: `bootstapnext_checkpoint_path: /path/to/bootstapnext/checkpoint.pt          # TODO: set path to BootsTAPNext checkpoint`
- L17: `bootstapir_checkpoint_path: /path/to/bootstapir/checkpoint.pt              # TODO: set path to BootsTAPIR checkpoint`

### `/gemini/code/FSPT/baselines/track_on/config/train_verifier.yaml`
Path hits: `track_on`
- L7: `trackon2_config_path: "./config/test.yaml"`
- L8: `trackon2_checkpoint_path: /path/to/trackon2/checkpoint.pt        # TODO: set path to Track-On2 checkpoint`
- L9: `bootstapnext_checkpoint_path: /path/to/bootstapnext/checkpoint.pt  # TODO: set path to BootsTAPNext checkpoint`
- L10: `bootstapir_checkpoint_path: /path/to/bootstapir/checkpoint.pt      # TODO: set path to BootsTAPIR checkpoint`

### `/gemini/code/FSPT/baselines/track_on/dataset/README.md`
Path hits: `track_on`
- L7: `## Track-On2 Pretraining Dataset`
- L9: `We use the **TAP-Vid Kubric Movi-F** split from [CoTracker3](https://huggingface.co/datasets/facebook/CoTracker3_Kubric).`
- L11: `Download the dataset from [Hugging Face](https://huggingface.co/datasets/facebook/CoTracker3_Kubric) and place it under:`
- L49: `## Track-On Real-World Fine-Tuning Dataset`
- L127: `- **[TAP-Vid DAVIS](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid#downloading-tap-vid-davis-and-tap-vid-rgb-stacking)**`

### `/gemini/code/FSPT/baselines/track_on/dataset/dynamic_replica.py`
Path hits: `track_on`
- L18: `# This code is based on the previous versions of the CoTracker repository:`
- L19: `# For reference, check: https://github.com/facebookresearch/co-tracker/blob/3716e362497e15e4fb8ec46898dcfd8afbca89e3/cotracker/datasets/dr_dataset.py`

### `/gemini/code/FSPT/baselines/track_on/dataset/ego_points.py`
Path hits: `track_on`

### `/gemini/code/FSPT/baselines/track_on/dataset/epic_k.py`
Path hits: `track_on`
- L10: `return np.clip(x, 0.0, 255.0, out=x)`
- L29: `hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)`
- L58: `NumPy-first dataset with clip-consistent photometric + spatial aug.`
- L147: `# --------- photometric (clip-level) ----------`
- L148: `def _photometric_augment_clip(self, video_thwc: np.ndarray) -> np.ndarray:`

### `/gemini/code/FSPT/baselines/track_on/dataset/movi_f.py`
Path hits: `track_on`
- L21: `# These classes are adapted from CoTracker repo:`
- L22: `#   https://github.com/facebookresearch/co-tracker/blob/main/cotracker/datasets/kubric_movif_dataset.py`
- L111: `x0 = np.clip(xc - dx / 2, 0, W).round().astype(np.int32)`
- L112: `x1 = np.clip(xc + dx / 2, 0, W).round().astype(np.int32)`
- L113: `y0 = np.clip(yc - dy / 2, 0, H).round().astype(np.int32)`

### `/gemini/code/FSPT/baselines/track_on/dataset/point_odyssey.py`
Path hits: `track_on`

### `/gemini/code/FSPT/baselines/track_on/dataset/real_world_dataset.py`
Path hits: `track_on`
- L11: `return np.clip(x, 0.0, 255.0, out=x)`
- L30: `hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)`
- L315: `def _photometric_augment_clip(self, video_thwc: np.ndarray) -> np.ndarray:`
- L316: `"""Apply clip-consistent photometric augmentation."""`
- L570: `# Sample crop parameters (consistent across clip)`

### `/gemini/code/FSPT/baselines/track_on/dataset/syn_real_dataset.py`
Path hits: `track_on`

### `/gemini/code/FSPT/baselines/track_on/dataset/tapvid.py`
Path hits: `track_on`
- L1: `# ==== Below is based on https://github.com/facebookresearch/co-tracker/blob/main/cotracker/datasets/tap_vid_datasets.py ====`

### `/gemini/code/FSPT/baselines/track_on/ensemble/README.md`
Path hits: `track_on`
- L17: `| TAPIR | TAPIR: Tracking Any Point with per-frame Initialization and temporal Refinement | [Project](https://deepmind-tapir.github.io/) |`
- L18: `| BootsTAPIR | BootsTAP: Bootstrapped Training for Tracking-Any-Point | [Project](https://bootstap.github.io/) |`
- L19: `| TAPNext | TAPNext: Tracking Any Point (TAP) as Next Token Prediction | [Project](https://tap-next.github.io/) |`
- L20: `| BootsTAPNext | TAPNext: Tracking Any Point (TAP) as Next Token Prediction | [Project](https://tap-next.github.io/) |`
- L21: `| CoTracker3 | CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos | [Project](https://cotracker3.github.io/) |`

### `/gemini/code/FSPT/baselines/track_on/ensemble/cotracker.py`
Path hits: `track_on, cotracker`
- L4: `class CoTracker_Predictor(nn.Module):`
- L7: `version = "cotracker3_online" if windowed else "cotracker3_offline"`

### `/gemini/code/FSPT/baselines/track_on/ensemble/ensemble_predictor.py`
Path hits: `track_on`
- L62: `# IMPORTANT NOTE: I realized that locotrack models sometimes produce NaN values in their predictions when using mixed precision (float16)`
- L63: `#                 Turn off amp for locotrack models, even if you want to use it for other models.`

### `/gemini/code/FSPT/baselines/track_on/ensemble/alltracker/alltracker.py`
Path hits: `track_on`

### `/gemini/code/FSPT/baselines/track_on/ensemble/alltracker/alltracker_predictor.py`
Path hits: `track_on`

### `/gemini/code/FSPT/baselines/track_on/ensemble/alltracker/blocks.py`
Path hits: `track_on`
- L831: `"""ViT weight initialization, original timm impl (for reproducibility)"""`
- L1035: `"""ViT weight initialization, original timm impl (for reproducibility)"""`

### `/gemini/code/FSPT/baselines/track_on/ensemble/bootstapir/bootstapir_predictor.py`
Path hits: `tapir, track_on`
- L3: `from ensemble.bootstapir.tapir_model import TAPIR`
- L10: `# wget -P . https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt`
- L20: `class TAPIRPredictor(torch.nn.Module):`
- L25: `if "bootstapir" in checkpoint_path:`
- L26: `self.model = TAPIR(pyramid_level=1, extra_convs=True)`

### `/gemini/code/FSPT/baselines/track_on/ensemble/bootstapir/nets.py`
Path hits: `tapir, track_on`

### `/gemini/code/FSPT/baselines/track_on/ensemble/bootstapir/tapir_model.py`
Path hits: `tapir, track_on`
- L16: `"""TAPIR models definition."""`
- L23: `from ensemble.bootstapir import nets`
- L24: `from ensemble.bootstapir import utils`
- L71: `class TAPIR(nn.Module):`
- L72: `"""TAPIR model."""`

### `/gemini/code/FSPT/baselines/track_on/ensemble/bootstapir/utils.py`
Path hits: `tapir, track_on`
- L285: `typically match the training resolution, which is (256, 256) for TAPIR.`

### `/gemini/code/FSPT/baselines/track_on/ensemble/locotrack/cmdtop.py`
Path hits: `locotrack, track_on`
- L6: `from ensemble.locotrack import utils`
