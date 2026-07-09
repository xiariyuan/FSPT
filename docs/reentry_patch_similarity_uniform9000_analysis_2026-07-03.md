# ReEntry Patch Similarity Feature Analysis

Samples: `9000`
Patch feature dim: `51`; numeric dim: `49`

## `y_safe16` patch features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| query_base_s17_ncc | 0.5493 | - | 0.7796 | 0.6542 | 0.7219 |
| query_base_s33_ncc | 0.5459 | - | 0.7799 | 0.5760 | 0.6262 |
| last_visible_base_s9_ncc | 0.5459 | - | 0.7733 | 0.6819 | 0.7504 |
| last_visible_base_s17_ncc | 0.5413 | - | 0.7777 | 0.6688 | 0.7340 |
| query_base_s9_ncc | 0.5405 | - | 0.7736 | 0.6673 | 0.7394 |
| last_visible_base_s17_std_color_l2 | 0.5241 | - | 0.7661 | 0.1125 | 0.1202 |
| query_base_s33_mean_color_l2 | 0.5232 | + | 0.7599 | 0.1265 | 0.1206 |
| query_base_s9_grad_mad | 0.5222 | - | 0.7662 | 0.0273 | 0.0282 |
| last_visible_base_s33_std_color_l2 | 0.5218 | - | 0.7633 | 0.0883 | 0.0917 |
| last_visible_base_s33_cosine | 0.5192 | + | 0.7655 | 0.8672 | 0.8628 |
| query_candidate_age_norm | 0.5191 | - | 0.7574 | 0.4358 | 0.4497 |
| last_visible_base_s33_mse | 0.5186 | - | 0.7665 | 0.0456 | 0.0470 |
| last_visible_base_s33_ncc | 0.5178 | - | 0.7662 | 0.5991 | 0.6296 |
| query_base_s9_std_color_l2 | 0.5166 | - | 0.7608 | 0.1016 | 0.1061 |
| query_base_s33_std_color_l2 | 0.5160 | + | 0.7595 | 0.1184 | 0.1145 |
| query_base_s9_cosine | 0.5152 | + | 0.7625 | 0.9380 | 0.9395 |
| last_visible_base_s33_mean_color_l2 | 0.5147 | - | 0.7659 | 0.1005 | 0.1019 |
| last_visible_base_s33_mad | 0.5145 | - | 0.7656 | 0.1200 | 0.1227 |
| query_base_s17_std_color_l2 | 0.5140 | - | 0.7608 | 0.1401 | 0.1450 |
| query_base_s33_grad_mad | 0.5137 | + | 0.7574 | 0.0319 | 0.0307 |

## `y_safe16` numeric reference features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| frame_override_border_dist_norm | 0.8315 | + | 0.9224 | 0.4848 | 0.2662 |
| frame_override_y_norm | 0.8308 | + | 0.9171 | 0.2474 | 0.1361 |
| frame_base_border_dist_norm | 0.7990 | + | 0.9105 | 0.4830 | 0.2985 |
| frame_base_y_norm | 0.7987 | + | 0.9077 | 0.2460 | 0.1518 |
| event_gate_prob | 0.7852 | + | 0.8856 | 0.4408 | 0.1886 |
| frame_override_speed_norm | 0.7806 | - | 0.9251 | 0.0236 | 0.0482 |
| frame_base_speed_norm | 0.7670 | - | 0.9175 | 0.0229 | 0.0444 |
| frame_base_override_dist_norm | 0.7609 | - | 0.8600 | 0.0303 | 0.1297 |
| frame_base_override_ydiff_norm | 0.7591 | + | 0.8649 | 0.0054 | -0.0627 |
| frame_speed_diff_norm | 0.7344 | - | 0.8742 | 0.0056 | 0.0135 |

## `y_gt_visible` patch features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| query_base_s17_ncc | 0.5467 | - | 0.7894 | 0.6559 | 0.7201 |
| last_visible_base_s17_ncc | 0.5450 | - | 0.7896 | 0.6696 | 0.7348 |
| last_visible_base_s9_ncc | 0.5440 | - | 0.7839 | 0.6832 | 0.7498 |
| query_base_s33_ncc | 0.5423 | - | 0.7878 | 0.5778 | 0.6232 |
| query_base_s9_ncc | 0.5379 | - | 0.7842 | 0.6687 | 0.7386 |
| query_base_s33_mean_color_l2 | 0.5262 | + | 0.7728 | 0.1266 | 0.1200 |
| query_base_s33_std_color_l2 | 0.5235 | + | 0.7743 | 0.1187 | 0.1133 |
| query_base_s9_grad_mad | 0.5219 | - | 0.7764 | 0.0273 | 0.0282 |
| last_visible_base_s33_ncc | 0.5188 | - | 0.7768 | 0.5996 | 0.6298 |
| last_visible_base_s17_std_color_l2 | 0.5187 | - | 0.7739 | 0.1130 | 0.1190 |
| query_candidate_age_norm | 0.5183 | - | 0.7680 | 0.4363 | 0.4491 |
| query_base_s17_mean_color_l2 | 0.5179 | + | 0.7733 | 0.1244 | 0.1174 |
| query_base_s33_grad_mad | 0.5168 | + | 0.7700 | 0.0319 | 0.0306 |
| last_visible_base_s33_std_color_l2 | 0.5166 | - | 0.7707 | 0.0885 | 0.0912 |
| query_base_s9_std_color_l2 | 0.5159 | - | 0.7710 | 0.1018 | 0.1057 |
| last_visible_base_s17_mean_color_l2 | 0.5155 | + | 0.7779 | 0.1033 | 0.0934 |
| last_visible_base_s17_mad | 0.5146 | + | 0.7764 | 0.0884 | 0.0832 |
| query_base_s17_mad | 0.5146 | + | 0.7734 | 0.1010 | 0.0980 |
| last_visible_base_s33_cosine | 0.5137 | + | 0.7727 | 0.8668 | 0.8640 |
| query_base_s9_cosine | 0.5133 | + | 0.7725 | 0.9378 | 0.9402 |

## `y_gt_visible` numeric reference features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| frame_override_border_dist_norm | 0.8393 | + | 0.9281 | 0.4839 | 0.2582 |
| frame_override_y_norm | 0.8386 | + | 0.9229 | 0.2468 | 0.1322 |
| event_gate_prob | 0.8103 | + | 0.9051 | 0.4435 | 0.1677 |
| frame_base_border_dist_norm | 0.8052 | + | 0.9160 | 0.4822 | 0.2920 |
| frame_base_y_norm | 0.8049 | + | 0.9132 | 0.2455 | 0.1487 |
| frame_override_speed_norm | 0.7948 | - | 0.9371 | 0.0235 | 0.0496 |
| frame_base_speed_norm | 0.7834 | - | 0.9317 | 0.0228 | 0.0459 |
| frame_base_override_dist_norm | 0.7779 | - | 0.8769 | 0.0302 | 0.1350 |
| frame_base_override_ydiff_norm | 0.7643 | + | 0.8726 | 0.0052 | -0.0656 |
| segment_prob_min | 0.7558 | + | 0.9009 | 0.5511 | 0.3503 |

## `y_safe8` patch features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| query_base_s17_ncc | 0.5458 | - | 0.7406 | 0.6512 | 0.7193 |
| last_visible_base_s9_ncc | 0.5440 | - | 0.7354 | 0.6779 | 0.7501 |
| query_base_s33_ncc | 0.5436 | - | 0.7402 | 0.5739 | 0.6239 |
| query_base_s9_ncc | 0.5412 | - | 0.7353 | 0.6638 | 0.7372 |
| last_visible_base_s17_std_color_l2 | 0.5349 | - | 0.7255 | 0.1109 | 0.1230 |
| last_visible_base_s17_ncc | 0.5343 | - | 0.7383 | 0.6664 | 0.7304 |
| query_base_s33_mean_color_l2 | 0.5239 | + | 0.7198 | 0.1269 | 0.1204 |
| last_visible_age_norm | 0.5238 | + | 0.7278 | 0.2038 | 0.1868 |
| last_visible_base_s33_std_color_l2 | 0.5220 | - | 0.7201 | 0.0881 | 0.0915 |
| query_base_s9_std_color_l2 | 0.5194 | - | 0.7153 | 0.1007 | 0.1075 |
| query_base_s33_grad_mad | 0.5187 | + | 0.7231 | 0.0321 | 0.0304 |
| query_base_s17_std_color_l2 | 0.5183 | - | 0.7182 | 0.1396 | 0.1456 |
| last_visible_base_s33_ncc | 0.5174 | - | 0.7264 | 0.5975 | 0.6291 |
| last_visible_base_s33_cosine | 0.5171 | + | 0.7194 | 0.8672 | 0.8635 |
| last_visible_base_s17_cosine | 0.5162 | + | 0.7201 | 0.9189 | 0.9169 |
| query_base_s9_cosine | 0.5162 | + | 0.7158 | 0.9386 | 0.9379 |
| last_visible_base_s33_mean_color_l2 | 0.5162 | - | 0.7216 | 0.1003 | 0.1023 |
| last_visible_base_s33_mse | 0.5144 | - | 0.7196 | 0.0457 | 0.0467 |
| query_base_s9_mean_color_l2 | 0.5129 | - | 0.7214 | 0.1211 | 0.1183 |
| last_visible_base_s9_std_color_l2 | 0.5128 | - | 0.7181 | 0.0877 | 0.0895 |

## `y_safe8` numeric reference features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| frame_override_border_dist_norm | 0.8586 | + | 0.9210 | 0.4996 | 0.2625 |
| frame_override_y_norm | 0.8580 | + | 0.9154 | 0.2550 | 0.1339 |
| frame_base_border_dist_norm | 0.8331 | + | 0.9098 | 0.4978 | 0.2899 |
| frame_base_y_norm | 0.8328 | + | 0.9069 | 0.2537 | 0.1472 |
| event_gate_prob | 0.7919 | + | 0.8707 | 0.4548 | 0.1916 |
| frame_override_speed_norm | 0.7488 | - | 0.8940 | 0.0237 | 0.0444 |
| frame_base_speed_norm | 0.7436 | - | 0.8924 | 0.0228 | 0.0416 |
| segment_prob_min | 0.7403 | + | 0.8422 | 0.5592 | 0.3681 |
| frame_base_acc_norm | 0.7340 | - | 0.8821 | 0.0127 | 0.0202 |
| frame_speed_diff_norm | 0.7261 | - | 0.8438 | 0.0053 | 0.0132 |

## `y_utility` patch features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| query_base_s33_ncc | 0.5399 | - | 0.4449 | 0.5614 | 0.6073 |
| query_base_s17_ncc | 0.5345 | - | 0.4327 | 0.6424 | 0.6911 |
| query_base_s33_std_color_l2 | 0.5306 | + | 0.4273 | 0.1217 | 0.1145 |
| query_base_s9_ncc | 0.5292 | - | 0.4283 | 0.6530 | 0.7076 |
| last_visible_base_s9_cosine | 0.5269 | + | 0.4264 | 0.9460 | 0.9459 |
| last_visible_base_s9_grad_mad | 0.5232 | - | 0.4255 | 0.0258 | 0.0269 |
| last_visible_age_norm | 0.5229 | + | 0.4280 | 0.2079 | 0.1925 |
| last_visible_base_s9_mse | 0.5226 | - | 0.4267 | 0.0212 | 0.0201 |
| last_visible_base_s33_ncc | 0.5226 | - | 0.4347 | 0.5860 | 0.6210 |
| query_base_s33_mean_color_l2 | 0.5211 | + | 0.4257 | 0.1295 | 0.1220 |
| query_base_s33_valid_min | 0.5208 | - | 0.4127 | 0.9569 | 0.9649 |
| last_visible_base_s17_grad_mad | 0.5181 | - | 0.4252 | 0.0305 | 0.0315 |
| last_visible_base_s9_std_color_l2 | 0.5179 | - | 0.4201 | 0.0865 | 0.0895 |
| last_visible_base_s9_mad | 0.5179 | - | 0.4237 | 0.0684 | 0.0672 |
| last_visible_base_s9_ncc | 0.5171 | - | 0.4240 | 0.6680 | 0.7205 |
| last_visible_base_s17_cosine | 0.5166 | + | 0.4203 | 0.9178 | 0.9187 |
| last_visible_base_s33_grad_mad | 0.5155 | - | 0.4211 | 0.0330 | 0.0338 |
| last_visible_base_s9_mean_color_l2 | 0.5152 | - | 0.4231 | 0.1033 | 0.0964 |
| query_candidate_age_norm | 0.5149 | + | 0.4157 | 0.4458 | 0.4350 |
| last_visible_base_s17_ncc | 0.5143 | - | 0.4284 | 0.6584 | 0.7037 |

## `y_utility` numeric reference features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| frame_base_speed_norm | 0.9155 | - | 0.8828 | 0.0069 | 0.0429 |
| frame_override_speed_norm | 0.9147 | - | 0.8701 | 0.0073 | 0.0452 |
| frame_base_y_norm | 0.8990 | + | 0.7837 | 0.2990 | 0.1700 |
| frame_override_y_norm | 0.8982 | + | 0.7790 | 0.2980 | 0.1656 |
| frame_base_border_dist_norm | 0.8956 | + | 0.7519 | 0.5845 | 0.3356 |
| frame_override_border_dist_norm | 0.8946 | + | 0.7466 | 0.5808 | 0.3266 |
| event_gate_prob | 0.8864 | + | 0.8058 | 0.6022 | 0.2239 |
| frame_base_acc_norm | 0.8731 | - | 0.8463 | 0.0052 | 0.0215 |
| frame_base_override_dist_norm | 0.8232 | - | 0.6766 | 0.0183 | 0.0808 |
| frame_override_acc_norm | 0.8161 | - | 0.7600 | 0.0056 | 0.0183 |

## `y_safe4` patch features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| query_base_s33_ncc | 0.5399 | - | 0.4449 | 0.5614 | 0.6073 |
| query_base_s17_ncc | 0.5345 | - | 0.4327 | 0.6424 | 0.6911 |
| query_base_s33_std_color_l2 | 0.5306 | + | 0.4273 | 0.1217 | 0.1145 |
| query_base_s9_ncc | 0.5292 | - | 0.4283 | 0.6530 | 0.7076 |
| last_visible_base_s9_cosine | 0.5269 | + | 0.4264 | 0.9460 | 0.9459 |
| last_visible_base_s9_grad_mad | 0.5232 | - | 0.4255 | 0.0258 | 0.0269 |
| last_visible_age_norm | 0.5229 | + | 0.4280 | 0.2079 | 0.1925 |
| last_visible_base_s9_mse | 0.5226 | - | 0.4267 | 0.0212 | 0.0201 |
| last_visible_base_s33_ncc | 0.5226 | - | 0.4347 | 0.5860 | 0.6210 |
| query_base_s33_mean_color_l2 | 0.5211 | + | 0.4257 | 0.1295 | 0.1220 |
| query_base_s33_valid_min | 0.5208 | - | 0.4127 | 0.9569 | 0.9649 |
| last_visible_base_s17_grad_mad | 0.5181 | - | 0.4252 | 0.0305 | 0.0315 |
| last_visible_base_s9_std_color_l2 | 0.5179 | - | 0.4201 | 0.0865 | 0.0895 |
| last_visible_base_s9_mad | 0.5179 | - | 0.4237 | 0.0684 | 0.0672 |
| last_visible_base_s9_ncc | 0.5171 | - | 0.4240 | 0.6680 | 0.7205 |
| last_visible_base_s17_cosine | 0.5166 | + | 0.4203 | 0.9178 | 0.9187 |
| last_visible_base_s33_grad_mad | 0.5155 | - | 0.4211 | 0.0330 | 0.0338 |
| last_visible_base_s9_mean_color_l2 | 0.5152 | - | 0.4231 | 0.1033 | 0.0964 |
| query_candidate_age_norm | 0.5149 | + | 0.4157 | 0.4458 | 0.4350 |
| last_visible_base_s17_ncc | 0.5143 | - | 0.4284 | 0.6584 | 0.7037 |

## `y_safe4` numeric reference features

| feature | AUC | dir | AP | pos mean | neg mean |
|---|---|---|---|---|---|
| frame_base_speed_norm | 0.9155 | - | 0.8828 | 0.0069 | 0.0429 |
| frame_override_speed_norm | 0.9147 | - | 0.8701 | 0.0073 | 0.0452 |
| frame_base_y_norm | 0.8990 | + | 0.7837 | 0.2990 | 0.1700 |
| frame_override_y_norm | 0.8982 | + | 0.7790 | 0.2980 | 0.1656 |
| frame_base_border_dist_norm | 0.8956 | + | 0.7519 | 0.5845 | 0.3356 |
| frame_override_border_dist_norm | 0.8946 | + | 0.7466 | 0.5808 | 0.3266 |
| event_gate_prob | 0.8864 | + | 0.8058 | 0.6022 | 0.2239 |
| frame_base_acc_norm | 0.8731 | - | 0.8463 | 0.0052 | 0.0215 |
| frame_base_override_dist_norm | 0.8232 | - | 0.6766 | 0.0183 | 0.0808 |
| frame_override_acc_norm | 0.8161 | - | 0.7600 | 0.0056 | 0.0183 |
