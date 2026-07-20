import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = json.load(open('docs/generated/ROUTED_TEMPORAL_IDENTITY_FULL_POPULATION_GATE3C1F2_FAILURE_DIAGNOSTIC_V0_2026-07-20.json'))
P = json.load(open('docs/generated/ROUTED_TEMPORAL_IDENTITY_FULL_POPULATION_GATE3C1F2_V0_REPLAY_2026-07-20.json'))
action = {}
for video in P['video_records']:
    source = video['scientific']['source_index']
    for row in video['action_records']:
        action[(source, row['point_index'])] = row
rows = []
for video in D['video_records']:
    for frame in video['scientific']['frame_records']:
        row = action[(frame['source_index'], frame['point_index'])]
        nv = frame['native_visibility_probability']
        nc = frame['native_confidence_probability']
        nj = frame['native_joint_probability']
        mv = frame['modified_visibility_probability']
        mc = frame['modified_confidence_probability']
        mj = frame['modified_joint_probability']
        features = [
            (frame['frame'] - 15) / 8,
            nv, nc, nj, mv, mc, mj,
            mv - nv, mc - nc, mj - nj,
            float(frame['native_visible']), float(frame['modified_visible']),
            row['entry_probability'], row['native_joint_probability'],
            row['selected_support_probability'], row['predicted_value_px'] / 64,
            row['predicted_harm_probability'], row['selected_slot'] / 8,
            row['output_candidate_index'] / 128,
        ]
        error = frame['modified_error_px']
        gt_visible = int(frame['gt_visible'])
        utility = int(gt_visible and error < 16)
        rows.append((features, gt_visible, utility, frame['source_index'], frame['category'], frame['modified_visible'], error))
X = np.asarray([row[0] for row in rows], dtype=np.float32)
y_visible = np.asarray([row[1] for row in rows], dtype=np.int64)
y_utility = np.asarray([row[2] for row in rows], dtype=np.int64)
groups = np.asarray([row[3] for row in rows], dtype=np.int64)
categories = np.asarray([row[4] for row in rows])
baseline = np.asarray([row[5] for row in rows], dtype=bool)
error = np.asarray([row[6] for row in rows], dtype=np.float64)
print('shape', X.shape, 'videos', len(np.unique(groups)), 'gt_visible', int(y_visible.sum()), 'utility', int(y_utility.sum()), 'baseline_visible', int(baseline.sum()))

def pooled(mask):
    values = []
    gt = y_visible.astype(bool)
    for threshold in (1, 2, 4, 8, 16):
        within = error < threshold
        tp = (gt & within & mask).sum()
        fp = (((~gt) | (~within)) & mask).sum()
        values.append(tp / (gt.sum() + fp))
    return float(np.mean(values)), values

print('baseline_pooled_AJ', pooled(baseline))
models = {
    'hgb': HistGradientBoostingClassifier(max_iter=180, learning_rate=.04, max_leaf_nodes=15, l2_regularization=3, min_samples_leaf=20, random_state=17, early_stopping=False),
    'log': make_pipeline(StandardScaler(), LogisticRegression(C=.3, max_iter=3000, class_weight='balanced', random_state=17)),
}
cv = GroupKFold(5)
output = {}
for target_name, target in [('gt_visible', y_visible), ('utility16', y_utility)]:
    output[target_name] = {}
    for model_name, model in models.items():
        prediction = np.zeros(len(target), dtype=np.float64)
        for train, test in cv.split(X, target, groups):
            model.fit(X[train], target[train])
            prediction[test] = model.predict_proba(X[test])[:, 1]
        auc = roc_auc_score(target, prediction)
        ap = average_precision_score(target, prediction)
        grid = []
        for threshold in np.arange(.05, .951, .025):
            mask = prediction >= threshold
            aj, _ = pooled(mask)
            oa = float((mask == y_visible).mean())
            visible_recall = float(mask[y_visible == 1].mean())
            occluded_fp = float(mask[y_visible == 0].mean())
            utility_precision = float(y_utility[mask].mean()) if mask.any() else 0.0
            grid.append([aj, oa, visible_recall, occluded_fp, utility_precision, int(mask.sum()), float(threshold)])
        print('\n', target_name, model_name, 'AUC', auc, 'AP', ap)
        for record in sorted(grid, reverse=True)[:10]:
            print(record)
        output[target_name][model_name] = {'prediction': prediction.tolist(), 'auc': auc, 'ap': ap, 'grid': grid}
open('/tmp/gate3c1g0_visibility_probe.json', 'w').write(json.dumps(output, indent=2))
best = max(output['utility16']['hgb']['grid'])
threshold = best[-1]
prediction = np.asarray(output['utility16']['hgb']['prediction'])
mask = prediction >= threshold
print('\nbest_utility_hgb', best)
for category in ('failure', 'ambiguous', 'other'):
    selected = categories == category
    occluded = selected & (y_visible == 0)
    visible = selected & (y_visible == 1)
    print(category, 'rows', int(selected.sum()), 'gt_visible_rate', float(y_visible[selected].mean()), 'pred_visible_rate', float(mask[selected].mean()), 'occluded_fp', float(mask[occluded].mean()) if occluded.any() else 0.0, 'visible_recall', float(mask[visible].mean()) if visible.any() else 0.0)
