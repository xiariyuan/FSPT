#!/usr/bin/env python3
from __future__ import annotations

import json, math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

APP = Path('outputs/paper_discovery_2026-06-27/b2wa_appearance_pilot/rgb_dev10_patch_rows.jsonl')
CLIP = Path('outputs/paper_discovery_2026-06-27/b2wa_clip_pilot/rgb_dev10_clip_patch_rows.jsonl')
DENSE = Path('outputs/paper_discovery_2026-06-27/b2wa_resnet_dense_pilot/rgb_dev10_resnet_dense_rows.jsonl')
OUT = Path('outputs/paper_discovery_2026-06-27/b2wa_failure_attribution')
DOC = Path('docs/b2wa_failure_attribution_2026-06-30.md')

TRAJ = [
    'trigger_t_norm','query_age_norm','candidate_window_index','base_invis_run','override_persist_len_cap16',
    'override_visible_frac_next4','override_visible_frac_next8','override_visible_frac_next16',
    'base_visible_frac_next4','base_visible_frac_next8','base_visible_frac_next16',
    'vis_agreement_frac_next16','override_visibility_transitions_next16','base_visibility_transitions_next16',
    'base_override_dist_t','base_override_dist_mean_next4','base_override_dist_mean_next8','base_override_dist_mean_next16','base_override_dist_max_next16',
    'base_speed_prev4','base_speed_prev8','override_speed_next4','override_speed_next8','override_speed_next16'
]

RISK_FEATURES = [
    ('base_override_dist_mean_next16', 'high'),
    ('base_override_dist_max_next16', 'high'),
    ('base_override_dist_t', 'high'),
    ('vis_agreement_frac_next16', 'low'),
    ('base_visible_frac_next16', 'low'),
    ('base_visible_frac_next8', 'low'),
    ('override_speed_next16', 'high'),
]


def load_rows(p: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in p.open() if l.strip()]

def key(r: Dict[str, Any]) -> Tuple[str, int, int, int]:
    return (str(r['video_id']), int(r['record_index']), int(r['query_idx']), int(r['trigger_t']))

def fnum(v: Any) -> float:
    if v is None:
        return np.nan
    try:
        x = float(v)
    except Exception:
        return np.nan
    return x if math.isfinite(x) else np.nan

def qnorm(x: np.ndarray, high: bool = True) -> np.ndarray:
    x = x.astype(float).copy()
    med = np.nanmedian(x) if np.any(np.isfinite(x)) else 0.0
    x[~np.isfinite(x)] = med
    lo, hi = np.nanpercentile(x, [5, 95])
    if hi <= lo:
        z = np.zeros_like(x)
    else:
        z = np.clip((x - lo) / (hi - lo), 0, 1)
    return z if high else 1 - z

def label(rows: List[Dict[str, Any]], target: str) -> np.ndarray:
    if target == 'harmful_w16':
        return np.asarray([bool(r['w16_harmful_full']) for r in rows], bool)
    if target == 'helpful_w16':
        return np.asarray([bool(r['w16_helpful_target']) for r in rows], bool)
    if target == 'best_reject':
        return np.asarray([r['best_action'] == 'reject' for r in rows], bool)
    if target == 'best_long':
        return np.asarray([r['best_action'] in ('W8','W16') for r in rows], bool)
    if target == 'window_has_reentry':
        return np.asarray([bool(r['window_has_reentry']) for r in rows], bool)
    raise ValueError(target)

def auc(y: np.ndarray, s: np.ndarray):
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y.astype(int), s))

def get_feat(r: Dict[str, Any], name: str) -> float:
    if name.startswith('app::'):
        return fnum(r.get('appearance_features', {}).get(name[5:]))
    if name.startswith('clip::'):
        return fnum(r.get('clip_features', {}).get(name[6:]))
    if name.startswith('dense::'):
        return fnum(r.get('resnet_dense_features', {}).get(name[7:]))
    return fnum(r.get('traj_features', {}).get(name))

def feature_names(rows: List[Dict[str, Any]], kind: str) -> List[str]:
    if kind == 'traj':
        return TRAJ
    if kind == 'app':
        return ['app::' + k for k in sorted(rows[0].get('appearance_features', {}).keys())]
    if kind == 'clip':
        return ['clip::' + k for k in sorted(rows[0].get('clip_features', {}).keys())]
    if kind == 'dense':
        return ['dense::' + k for k in sorted(rows[0].get('resnet_dense_features', {}).keys())]
    if kind == 'all_visual':
        return feature_names(rows,'app') + feature_names(rows,'clip') + feature_names(rows,'dense')
    if kind == 'combined':
        return TRAJ + feature_names(rows,'all_visual')
    raise ValueError(kind)

def matrix(rows: List[Dict[str, Any]], kind: str) -> Tuple[np.ndarray, List[str]]:
    ns = feature_names(rows, kind)
    X=[]
    for r in rows:
        X.append([get_feat(r,n) for n in ns])
    X=np.asarray(X,dtype=np.float32)
    for j in range(X.shape[1]):
        col=X[:,j]
        med=np.nanmedian(col) if np.any(np.isfinite(col)) else 0.0
        col[~np.isfinite(col)] = med
        X[:,j]=col
    return X, ns

def model_auc(rows: List[Dict[str, Any]], kind: str, target: str) -> Dict[str, Any]:
    if len(rows) < 50:
        return {'auc': None, 'reason': 'too_few_rows', 'n': len(rows)}
    y = label(rows, target)
    if len(np.unique(y)) < 2:
        return {'auc': None, 'reason': 'single_class', 'positive': int(y.sum()), 'n': len(rows)}
    groups = np.asarray([str(r['video_id']) for r in rows])
    uniq = np.unique(groups)
    if len(uniq) < 2:
        # fallback in one video bucket: single-feature/logit CV not meaningful
        return {'auc': None, 'reason': 'single_group', 'positive': int(y.sum()), 'n': len(rows)}
    n_splits=min(5,len(uniq))
    X,ns=matrix(rows,kind)
    pred=np.zeros(len(rows))
    folds=[]
    for i,(tr,te) in enumerate(GroupKFold(n_splits=n_splits).split(X,y,groups),1):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf=HistGradientBoostingClassifier(max_iter=120,learning_rate=.06,max_leaf_nodes=31,l2_regularization=.01,random_state=500+i)
        clf.fit(X[tr],y[tr].astype(int))
        p=clf.predict_proba(X[te])[:,1]
        pred[te]=p
        folds.append({'fold':i,'n_test':int(len(te)),'auc':round(auc(y[te],p),6)})
    if not folds:
        return {'auc': None, 'reason': 'no_valid_folds', 'positive': int(y.sum()), 'n': len(rows)}
    return {'auc': round(auc(y,pred),6), 'folds': folds, 'positive': int(y.sum()), 'negative': int((~y).sum()), 'n': len(rows)}

def bucket_summary(rows: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    n=len(rows)
    if n == 0:
        return {'n':0}
    h=label(rows,'harmful_w16')
    hp=label(rows,'helpful_w16')
    br=label(rows,'best_reject')
    bl=label(rows,'best_long')
    best_counts={a: int(sum(1 for r in rows if r['best_action']==a)) for a in ['reject','W4','W8','W16']}
    def mean_feat(feat):
        vals=np.asarray([get_feat(r,feat) for r in rows],float)
        return None if not np.any(np.isfinite(vals)) else round(float(np.nanmean(vals)),6)
    return {
        'bucket': name,
        'n': n,
        'harmful_count': int(h.sum()),
        'harmful_rate': round(float(h.mean()),6),
        'helpful_count': int(hp.sum()),
        'helpful_rate': round(float(hp.mean()),6),
        'best_reject_rate': round(float(br.mean()),6),
        'best_long_rate': round(float(bl.mean()),6),
        'best_action_counts': best_counts,
        'feature_means': {k: mean_feat(k) for k,_ in RISK_FEATURES},
    }

def single_feature_aucs(rows: List[Dict[str, Any]], target: str, feats: List[str]) -> List[Dict[str, Any]]:
    y=label(rows,target)
    out=[]
    for f in feats:
        x=np.asarray([get_feat(r,f) for r in rows],float)
        m=np.isfinite(x)
        if m.sum()<10 or len(np.unique(y[m]))<2:
            continue
        med=np.nanmedian(x) if np.any(m) else 0.0
        x[~m]=med
        a=auc(y,x)
        out.append({'feature':f,'auc_positive_high':round(a,6),'best_auc':round(max(a,1-a),6),'direction':'high' if a>=0.5 else 'low'})
    out.sort(key=lambda z:z['best_auc'],reverse=True)
    return out

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    app_rows=load_rows(APP)
    clip_by={key(r): r for r in load_rows(CLIP)} if CLIP.exists() else {}
    dense_by={key(r): r for r in load_rows(DENSE)} if DENSE.exists() else {}
    rows=[]
    for r in app_rows:
        k=key(r)
        rr=dict(r)
        if k in clip_by:
            rr['clip_features']=clip_by[k].get('clip_features',{})
        if k in dense_by:
            rr['resnet_dense_features']=dense_by[k].get('resnet_dense_features',{})
        rows.append(rr)

    # geometry risk score
    feat_arrays={}
    needed_features = sorted(set([f for f,_ in RISK_FEATURES] + [
        'base_visible_frac_next4','base_visible_frac_next8','base_visible_frac_next16',
        'override_visibility_transitions_next16','override_speed_next16'
    ]))
    for f in needed_features:
        vals=np.asarray([get_feat(r,f) for r in rows],float)
        feat_arrays[f]=vals
    risk=np.zeros(len(rows),float)
    for f,direction in RISK_FEATURES:
        risk += qnorm(feat_arrays[f], high=(direction=='high'))
    risk /= len(RISK_FEATURES)
    q33,q66=np.percentile(risk,[33.333,66.667])
    for r,sc in zip(rows,risk):
        r['geometry_risk_score']=float(sc)
        r['geometry_risk_bucket']='low' if sc<=q33 else ('medium' if sc<=q66 else 'high')

    # Additional attribution masks based on quartiles
    dist16=feat_arrays['base_override_dist_mean_next16']; distmax=feat_arrays['base_override_dist_max_next16']; distt=feat_arrays['base_override_dist_t']
    vis=feat_arrays['vis_agreement_frac_next16']; basevis4=feat_arrays['base_visible_frac_next4']; basevis8=feat_arrays['base_visible_frac_next8']; basevis16=feat_arrays['base_visible_frac_next16']
    ospeed=feat_arrays['override_speed_next16']; otrans=feat_arrays['override_visibility_transitions_next16']
    def q(x,p): return float(np.nanpercentile(x[np.isfinite(x)],p))
    thresholds={
        'dist16_q75': q(dist16,75), 'dist16_q90': q(dist16,90),
        'distmax_q75': q(distmax,75), 'distt_q75': q(distt,75),
        'vis_q25': q(vis,25), 'basevis4_q75': q(basevis4,75), 'basevis8_q75': q(basevis8,75), 'basevis16_q75': q(basevis16,75),
        'ospeed_q75': q(ospeed,75), 'otrans_q75': q(otrans,75),
    }
    for i,r in enumerate(rows):
        cats=[]
        if dist16[i]>=thresholds['dist16_q75'] or distmax[i]>=thresholds['distmax_q75'] or distt[i]>=thresholds['distt_q75']:
            cats.append('large_geometry_gap')
        if vis[i]<=thresholds['vis_q25']:
            cats.append('low_base_override_agreement')
        # Use an absolute threshold here. Quantile thresholds can collapse to zero because most triggered windows
        # keep the base invisible, which would make this tag meaningless.
        if basevis4[i] >= 0.25 or basevis8[i] >= 0.25 or basevis16[i] >= 0.25:
            cats.append('base_recovers_or_stays_visible')
        if ospeed[i]>=thresholds['ospeed_q75'] or otrans[i]>=thresholds['otrans_q75']:
            cats.append('override_unstable')
        if not cats:
            cats.append('ambiguous_low_geometry')
        r['attribution_tags']=cats

    # bucket summaries
    buckets={}
    for b in ['low','medium','high']:
        br=[r for r in rows if r['geometry_risk_bucket']==b]
        buckets[b]=bucket_summary(br,b)

    tag_summary={}
    for tag in ['large_geometry_gap','low_base_override_agreement','base_recovers_or_stays_visible','override_unstable','ambiguous_low_geometry']:
        tr=[r for r in rows if tag in r['attribution_tags']]
        tag_summary[tag]=bucket_summary(tr,tag)

    harmful=[r for r in rows if r['w16_harmful_full']]
    harmful_tag_counts={tag: int(sum(1 for r in harmful if tag in r['attribution_tags'])) for tag in tag_summary}
    harmful_tag_rates={tag: round(c/max(len(harmful),1),6) for tag,c in harmful_tag_counts.items()}

    # AUC by buckets for visual usefulness
    targets=['harmful_w16','best_reject','helpful_w16','best_long']
    model_by_bucket={}
    for b in ['low','medium','high']:
        br=[r for r in rows if r['geometry_risk_bucket']==b]
        model_by_bucket[b]={}
        for t in targets:
            model_by_bucket[b][t]={kind: model_auc(br,kind,t) for kind in ['traj','app','clip','dense','all_visual','combined']}

    model_global={}
    for t in targets+['window_has_reentry']:
        model_global[t]={kind: model_auc(rows,kind,t) for kind in ['traj','app','clip','dense','all_visual','combined']}

    top_visual_medium={}
    med=[r for r in rows if r['geometry_risk_bucket']=='medium']
    visual_feats=feature_names(rows,'all_visual')
    for t in targets:
        top_visual_medium[t]=single_feature_aucs(med,t,visual_feats)[:20]

    # risky but appearance-solvable proxy: medium bucket best_reject/harmful counts
    recommendation=[]
    harm_global_traj=model_global['harmful_w16']['traj']['auc']
    harm_global_comb=model_global['harmful_w16']['combined']['auc']
    med_harm=model_by_bucket['medium']['harmful_w16']
    med_gain=None
    if med_harm['traj']['auc'] is not None and med_harm['combined']['auc'] is not None:
        med_gain=med_harm['combined']['auc']-med_harm['traj']['auc']
    if harm_global_comb is not None and harm_global_traj is not None and harm_global_comb - harm_global_traj >= 0.03:
        recommendation.append('appearance_continue_global')
    if med_gain is not None and med_gain >= 0.03:
        recommendation.append('appearance_continue_medium_risk')
    if not recommendation:
        recommendation.append('do_not_continue_generic_appearance')
        recommendation.append('prefer_temporal_geometry_policy_B2WT')

    summary={
        'n_rows': len(rows),
        'n_harmful_w16': int(sum(r['w16_harmful_full'] for r in rows)),
        'geometry_risk_quantiles': {'q33': round(float(q33),6),'q66': round(float(q66),6)},
        'thresholds': {k: round(float(v),6) for k,v in thresholds.items()},
        'bucket_summary': buckets,
        'tag_summary': tag_summary,
        'harmful_tag_counts': harmful_tag_counts,
        'harmful_tag_rates': harmful_tag_rates,
        'model_auc_global': model_global,
        'model_auc_by_geometry_bucket': model_by_bucket,
        'top_visual_single_features_medium_bucket': top_visual_medium,
        'recommendation': recommendation,
        'interpretation': {
            'harmful_w16_global_traj_auc': harm_global_traj,
            'harmful_w16_global_combined_auc': harm_global_comb,
            'harmful_w16_global_combined_gain': None if harm_global_traj is None or harm_global_comb is None else round(harm_global_comb-harm_global_traj,6),
            'medium_bucket_harmful_combined_gain': None if med_gain is None else round(float(med_gain),6),
        }
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    with (OUT/'rows_with_attribution.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps({
                'video_id': r['video_id'], 'record_index': r['record_index'], 'query_idx': r['query_idx'], 'trigger_t': r['trigger_t'],
                'best_action': r['best_action'], 'w16_harmful_full': r['w16_harmful_full'], 'w16_helpful_target': r['w16_helpful_target'],
                'geometry_risk_score': r['geometry_risk_score'], 'geometry_risk_bucket': r['geometry_risk_bucket'], 'attribution_tags': r['attribution_tags'],
                'traj_features': {k: r['traj_features'].get(k) for k,_ in RISK_FEATURES},
            }, ensure_ascii=False)+'\n')

    # Markdown report
    def auc_line(target: str, scope: Dict[str, Any]) -> str:
        vals=[]
        for k in ['traj','app','clip','dense','all_visual','combined']:
            a=scope[target][k]['auc']
            vals.append('NA' if a is None else f'{a:.4f}')
        return ' | '.join(vals)
    md=[]
    md.append('# B2-WA / B2-WV Failure Attribution Audit — 2026-06-30\n')
    md.append('## Scope\n')
    md.append('RGB dev10 only, 2000 stratified candidate windows. RGB fresh20-49 was not used.\n')
    md.append('## Geometry-risk buckets\n')
    md.append('| bucket | n | harmful rate | helpful rate | best reject | best long |\n|---|---:|---:|---:|---:|---:|')
    for b in ['low','medium','high']:
        x=buckets[b]
        md.append(f"| {b} | {x['n']} | {x['harmful_rate']:.4f} | {x['helpful_rate']:.4f} | {x['best_reject_rate']:.4f} | {x['best_long_rate']:.4f} |")
    md.append('\n## Harmful-window attribution tags\n')
    md.append('| tag | n in all rows | harmful rate in tag | harmful rows covered | coverage of all harmful |\n|---|---:|---:|---:|---:|')
    for tag,x in tag_summary.items():
        md.append(f"| {tag} | {x.get('n',0)} | {x.get('harmful_rate',0.0):.4f} | {harmful_tag_counts[tag]} | {harmful_tag_rates[tag]:.4f} |")
    md.append('\n## Global AUC comparison\n')
    md.append('| target | traj | RGB app | CLIP | ResNet dense | all visual | combined |\n|---|---:|---:|---:|---:|---:|---:|')
    for t in ['harmful_w16','best_reject','helpful_w16','best_long','window_has_reentry']:
        md.append(f'| {t} | {auc_line(t,model_global)} |')
    md.append('\n## AUC by geometry-risk bucket: harmful_w16\n')
    md.append('| bucket | traj | RGB app | CLIP | ResNet dense | all visual | combined |\n|---|---:|---:|---:|---:|---:|---:|')
    for b in ['low','medium','high']:
        md.append(f"| {b} | {auc_line('harmful_w16', model_by_bucket[b])} |")
    md.append('\n## Interpretation\n')
    md.append(f"Global harmful_w16 combined gain over trajectory-only: `{summary['interpretation']['harmful_w16_global_combined_gain']}`.\n")
    md.append(f"Medium-risk harmful_w16 combined gain over trajectory-only: `{summary['interpretation']['medium_bucket_harmful_combined_gain']}`.\n")
    md.append('\n## Decision\n')
    if 'prefer_temporal_geometry_policy_B2WT' in recommendation:
        md.append('Generic appearance features do not provide a meaningful harmful-window improvement globally or in the medium-risk bucket. The next stronger-method direction should shift from B2-WA appearance verification to B2-WT: temporal / geometry risk-controlled adaptive early-stop.\n')
    else:
        md.append('There is some evidence that appearance helps in a subset; continue with true correspondence features.\n')
    md.append('\n## Artifacts\n')
    md.append('```text\noutputs/paper_discovery_2026-06-27/b2wa_failure_attribution/summary.json\noutputs/paper_discovery_2026-06-27/b2wa_failure_attribution/rows_with_attribution.jsonl\n```\n')
    DOC.write_text('\n'.join(md))
    print(json.dumps({
        'wrote': str(OUT/'summary.json'),
        'doc': str(DOC),
        'recommendation': recommendation,
        'global_harmful_gain': summary['interpretation']['harmful_w16_global_combined_gain'],
        'medium_harmful_gain': summary['interpretation']['medium_bucket_harmful_combined_gain'],
        'bucket_summary': buckets,
        'harmful_tag_rates': harmful_tag_rates,
    }, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
