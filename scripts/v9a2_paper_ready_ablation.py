@@
     threshold_ablation = ['w16_accept_all_joint', 'v9a2_all_logreg_event_max_fixed_0.05', 'v9a2_all_logreg_event_max_oof_f1']
+
+    # Full predeclared feature-set x controller-mode x threshold-protocol grid.
+    # This is not a threshold search: each learned score uses only fixed0.05 or
+    # its single OOF-F1 threshold, and W16 accept-all is reported separately.
+    factorial_ablation = []
+    for score_name in ['base_logreg', 'anchor_logreg', 'all_logreg']:
+        info = existing[score_name]
+        feature_set = score_name.split('_', 1)[0]
+        for mode in ['frame', 'event_max', 'event_mean']:
+            for protocol, threshold in [('fixed_0.05', 0.05), ('oof_f1', info['threshold_f1'])]:
+                variant = f'factorial_{feature_set}_{mode}_{protocol}'
+                add(variant, event_accept_mask(data, info['scores'], float(threshold), mode))
+                factorial_ablation.append({
+                    'variant': variant,
+                    'feature_set': feature_set,
+                    'mode': mode,
+                    'protocol': protocol,
+                    'threshold': float(threshold),
+                    'classifier': {k: info[k] for k in ['ap', 'auc', 'brier', 'threshold_f1']},
+                })
@@
         'threshold_ablation': [variants[x] for x in threshold_ablation],
+        'factorial_ablation': [{**x, 'result': variants[x['variant']]} for x in factorial_ablation],