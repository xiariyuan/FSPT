# ReEntry Appearance Frame Access Inspection

RGB root: `/gemini/code/datasets/tapvid_rgb_stacking`
Patch sizes: `[9, 17, 33]`

## natural

### rgb_stacking_000020

Resolver: `TAPVidRGBStackingDataset`
Video shape THWC: `[250, 256, 256, 3]`
Record original size: `[256, 256]`
Size matches record: `True`; frame count matches: `True`
Query idx/t: `0` / `0`; candidate t: `148`
Query pixel yx: `[68.398, 36.191]`; base candidate pixel yx: `[68.523, 36.139]`; gt candidate pixel yx: `[68.555, 35.982]`
Base visible at candidate: `True`; GT visible: `True`

| patch | query valid | base valid | gt valid | query mean | base mean | gt mean |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 1.0 | 1.0 | 1.0 | 0.3102 | 0.3092 | 0.3092 |
| 17 | 1.0 | 1.0 | 1.0 | 0.3126 | 0.3126 | 0.3126 |
| 33 | 1.0 | 1.0 | 1.0 | 0.302 | 0.3024 | 0.3024 |

### rgb_stacking_000021

Resolver: `TAPVidRGBStackingDataset`
Video shape THWC: `[250, 256, 256, 3]`
Record original size: `[256, 256]`
Size matches record: `True`; frame count matches: `True`
Query idx/t: `0` / `0`; candidate t: `74`
Query pixel yx: `[82.012, 153.066]`; base candidate pixel yx: `[82.721, 153.135]`; gt candidate pixel yx: `[82.091, 153.104]`
Base visible at candidate: `True`; GT visible: `True`

| patch | query valid | base valid | gt valid | query mean | base mean | gt mean |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 1.0 | 1.0 | 1.0 | 0.1925 | 0.1935 | 0.1896 |
| 17 | 1.0 | 1.0 | 1.0 | 0.2276 | 0.2283 | 0.2265 |
| 33 | 1.0 | 1.0 | 1.0 | 0.2413 | 0.2405 | 0.2411 |

## translate_L16

### rgb_stacking_000020_translate_L16

Resolver: `stress_dataset_translate`
Video shape THWC: `[250, 256, 256, 3]`
Record original size: `[256, 256]`
Size matches record: `True`; frame count matches: `True`
Query idx/t: `0` / `0`; candidate t: `148`
Query pixel yx: `[68.398, 36.191]`; base candidate pixel yx: `[68.531, 36.221]`; gt candidate pixel yx: `[68.555, 35.982]`
Base visible at candidate: `True`; GT visible: `True`

| patch | query valid | base valid | gt valid | query mean | base mean | gt mean |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 1.0 | 1.0 | 1.0 | 0.3102 | 0.3092 | 0.3092 |
| 17 | 1.0 | 1.0 | 1.0 | 0.3126 | 0.3126 | 0.3126 |
| 33 | 1.0 | 1.0 | 1.0 | 0.302 | 0.3024 | 0.3024 |

### rgb_stacking_000021_translate_L16

Resolver: `stress_dataset_translate`
Video shape THWC: `[250, 256, 256, 3]`
Record original size: `[256, 256]`
Size matches record: `True`; frame count matches: `True`
Query idx/t: `0` / `0`; candidate t: `82`
Query pixel yx: `[82.012, 153.066]`; base candidate pixel yx: `[82.655, 153.158]`; gt candidate pixel yx: `[82.048, 153.106]`
Base visible at candidate: `True`; GT visible: `True`

| patch | query valid | base valid | gt valid | query mean | base mean | gt mean |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 1.0 | 1.0 | 1.0 | 0.1925 | 0.1935 | 0.1898 |
| 17 | 1.0 | 1.0 | 1.0 | 0.2276 | 0.2283 | 0.2265 |
| 33 | 1.0 | 1.0 | 1.0 | 0.2413 | 0.2405 | 0.2411 |

## occluder_L16

### rgb_stacking_000020_occluder_L16

Resolver: `stress_dataset_occluder`
Video shape THWC: `[250, 256, 256, 3]`
Record original size: `[256, 256]`
Size matches record: `True`; frame count matches: `True`
Query idx/t: `0` / `0`; candidate t: `148`
Query pixel yx: `[68.398, 36.191]`; base candidate pixel yx: `[68.55, 36.168]`; gt candidate pixel yx: `[68.555, 35.982]`
Base visible at candidate: `True`; GT visible: `True`

| patch | query valid | base valid | gt valid | query mean | base mean | gt mean |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 1.0 | 1.0 | 1.0 | 0.3102 | 0.3092 | 0.3092 |
| 17 | 1.0 | 1.0 | 1.0 | 0.3126 | 0.3126 | 0.3126 |
| 33 | 1.0 | 1.0 | 1.0 | 0.302 | 0.3024 | 0.3024 |

### rgb_stacking_000021_occluder_L16

Resolver: `stress_dataset_occluder`
Video shape THWC: `[250, 256, 256, 3]`
Record original size: `[256, 256]`
Size matches record: `True`; frame count matches: `True`
Query idx/t: `0` / `0`; candidate t: `66`
Query pixel yx: `[82.012, 153.066]`; base candidate pixel yx: `[82.675, 153.199]`; gt candidate pixel yx: `[82.154, 153.098]`
Base visible at candidate: `True`; GT visible: `True`

| patch | query valid | base valid | gt valid | query mean | base mean | gt mean |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 1.0 | 1.0 | 1.0 | 0.1925 | 0.1935 | 0.1896 |
| 17 | 1.0 | 1.0 | 1.0 | 0.2276 | 0.2284 | 0.2266 |
| 33 | 1.0 | 1.0 | 1.0 | 0.2413 | 0.2405 | 0.2411 |
