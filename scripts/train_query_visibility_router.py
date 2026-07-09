#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys, warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch

try:
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
except Exception as e:
    raise SystemExit(f"sklearn import failed: {e!r}")

TEACHERS = ["cotracker3_online", "cotracker3_offline", "trackon2"]
CACHE_PATHS = {
    "cotracker3_online": "outputs/redetection_ladder_2026-06-17/caches/cotracker3_online_strided_original.pt",
    "cotracker3_offline": "outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt",
    "trackon2": "caches/trackon2_strided_original.pt",
}
MAJ_JSON = "outputs/paper_discovery_2026-06-27/visibility_fine/all_median__vis_majority_ajrd.json"
UNION_JSON = "outputs/paper_discovery_2026-06-27/visibility_fine/all_median__vis_union_ajrd.json"
GATED_JSON = "outputs/paper_discovery_2026-06-27/visibility_fine/all_median__vis_gated_mean144_ajrd.json"


def load_cache(path: str):
    return torch.load(path, map_location="cpu", weights_only=False)

def npy(x, dtype=None):
    if isinstance(x, torch.Tensor): x=x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)

def px_scale(osz):
    h,w=float(osz[0]),float(osz[1])
    return np.array([max(h-1,1), max(w-1,1)], dtype=np.float32)

def finite(pred): return np.all(np.isfinite(pred), axis=-1)

def masked_median(pred, mask):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(np.where(mask[...,None], pred, np.nan), axis=0).astype(np.float32)

def fill(primary, fallback):
    bad=~np.all(np.isfinite(primary),axis=-1)
    if np.any(bad):
        primary=primary.copy(); primary[bad]=fallback[bad]
    return primary.astype(np.float32)

def all_median_tracks(pred_stack):
    fm=finite(pred_stack)
    med=masked_median(pred_stack,fm)
    return fill(med, med)

def union_vis(vis_stack): return (vis_stack.sum(axis=0)>=1).astype(bool)
def majority_vis(vis_stack): return (vis_stack.sum(axis=0)>=2).astype(bool)

def pairwise_dist(pred_stack, osz):
    pts=pred_stack*px_scale(osz)
    d01=np.linalg.norm(pts[0]-pts[1],axis=-1)
    d02=np.linalg.norm(pts[0]-pts[2],axis=-1)
    d12=np.linalg.norm(pts[1]-pts[2],axis=-1)
    return d01,d02,d12

def speed(track, osz):
    pts=track*px_scale(osz)
    if pts.shape[1] < 2:
        return np.zeros(pts.shape[:2], dtype=np.float32)
    d=np.linalg.norm(np.diff(pts,axis=1),axis=-1)
    return np.concatenate([d[:,:1], d], axis=1).astype(np.float32)

def safe_stats(x):
    x=np.asarray(x, dtype=np.float32)
    x=x[np.isfinite(x)]
    if x.size==0: return [0,0,0,0,0]
    return [float(np.mean(x)), float(np.median(x)), float(np.percentile(x,75)), float(np.percentile(x,95)), float(np.max(x))]

def validate(caches):
    ref=caches[TEACHERS[0]]["records"]
    for t in TEACHERS[1:]:
        recs=caches[t]["records"]
        assert len(recs)==len(ref)
        for i,(a,b) in enumerate(zip(ref,recs)):
            assert a["video_id"]==b["video_id"], (t,i)
            for k in ["query_points","gt_tracks","gt_visibility","original_size","model_input_size"]:
                assert np.allclose(npy(a[k]),npy(b[k]),atol=1e-6,rtol=1e-6), (t,i,k)

def json_query_map(path: str) -> Dict[Tuple[str,int], Dict[str,Any]]:
    obj=json.load(open(path))
    m={}
    for vid in obj.get("per_video", []):
        video_id=vid["video_id"]
        for q in vid.get("per_query", []):
            m[(video_id, int(q["query_idx"]))]=q
    return m

def q_ajrd(q: Dict[str,Any]) -> float:
    return float((q.get("ajrd_summary_256") or {}).get("aj_rd") or 0.0)

def extract_rows(caches, maj_map, union_map, gated_map):
    rows=[]
    ref_records=caches[TEACHERS[0]]["records"]
    for ridx, ref in enumerate(ref_records):
        video_id=str(ref["video_id"])
        pred_stack=np.stack([npy(caches[t]["records"][ridx]["pred_tracks"], np.float32) for t in TEACHERS], axis=0)
        vis_stack=np.stack([npy(caches[t]["records"][ridx]["pred_visibility"], bool) for t in TEACHERS], axis=0)
        osz=npy(ref["original_size"], np.float32)
        qpts=npy(ref["query_points"], np.float32)
        N,T=vis_stack.shape[1:]
        d01,d02,d12=pairwise_dist(pred_stack, osz)
        dis_mean=(d01+d02+d12)/3.0
        dis_max=np.maximum(np.maximum(d01,d02),d12)
        speeds=np.stack([speed(pred_stack[i], osz) for i in range(3)], axis=0)
        union=union_vis(vis_stack); majority=majority_vis(vis_stack)
        count=vis_stack.sum(axis=0)
        for qi in range(N):
            key=(video_id, qi)
            if key not in maj_map or key not in union_map: continue
            maj=q_ajrd(maj_map[key]); uni=q_ajrd(union_map[key]); gat=q_ajrd(gated_map.get(key, {})) if gated_map else 0.0
            qt=int(round(float(qpts[qi,0]))) if qpts.ndim==2 else 0
            post=slice(max(qt,0), T)
            # deployable query-level features
            feat=[]
            # query timing
            feat += [qt/max(T-1,1), float(T), float(qpts[qi,1]), float(qpts[qi,2])]
            # visibility proportions per teacher/post/all and counts
            for arr_slice in [slice(0,T), post]:
                vv=vis_stack[:,qi,arr_slice]
                cc=count[qi,arr_slice]
                uu=union[qi,arr_slice]; mm=majority[qi,arr_slice]
                feat += [float(np.mean(vv[i])) for i in range(3)]
                feat += [float(np.mean(uu)), float(np.mean(mm)), float(np.mean(cc==1)), float(np.mean(cc==2)), float(np.mean(cc==3))]
                # transitions
                feat += [float(np.mean(np.abs(np.diff(vv[i].astype(np.float32))))) if vv.shape[1]>1 else 0.0 for i in range(3)]
            # disagreement stats all/post/union-visible/majority-visible
            for mask in [np.ones(T,dtype=bool), np.arange(T)>=qt, union[qi], majority[qi]]:
                feat += safe_stats(dis_mean[qi,mask])
                feat += safe_stats(dis_max[qi,mask])
                feat += safe_stats(d01[qi,mask]); feat += safe_stats(d02[qi,mask]); feat += safe_stats(d12[qi,mask])
            # speed stats per teacher all/post/visible
            for i in range(3):
                for mask in [np.ones(T,dtype=bool), np.arange(T)>=qt, vis_stack[i,qi]]:
                    feat += safe_stats(speeds[i,qi,mask])
            # run-length style features from teacher visibility, no GT
            for base in [majority[qi], union[qi]]:
                # longest invisible run after query
                inv=(~base[qt:]).astype(np.int32)
                best=cur=0
                for v in inv:
                    if v: cur+=1; best=max(best,cur)
                    else: cur=0
                first_vis=-1
                for tt in range(qt,T):
                    if base[tt]: first_vis=tt-qt; break
                feat += [float(best)/max(T-qt,1), float(first_vis if first_vis>=0 else T)/max(T-qt,1)]
            label_union = 1 if uni > maj + 1e-8 else 0
            # label best among majority/union/gated: 0 majority, 1 union, 2 gated
            vals=[maj,uni,gat]
            label3=int(np.argmax(vals))
            rows.append({
                "video_id":video_id,"record_index":ridx,"query_idx":qi,"feature":feat,
                "aj_majority":maj,"aj_union":uni,"aj_gated":gat,
                "label_union":label_union,"label3":label3,"gain_union_vs_majority":uni-maj,
            })
    return rows

def make_model(kind:str):
    if kind=="logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
    if kind=="rf":
        return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=8, class_weight="balanced_subsample", random_state=0, n_jobs=-1)
    if kind=="hgb":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=0.1, random_state=0)
    raise ValueError(kind)

def build_pred_cache(caches, rows, pred_by_key, out_path: Path):
    refp=caches[TEACHERS[0]]
    records=[]
    row_by=( {(r["record_index"], r["query_idx"]): r for r in rows} )
    for ridx, ref in enumerate(refp["records"]):
        pred_stack=np.stack([npy(caches[t]["records"][ridx]["pred_tracks"], np.float32) for t in TEACHERS], axis=0)
        vis_stack=np.stack([npy(caches[t]["records"][ridx]["pred_visibility"], bool) for t in TEACHERS], axis=0)
        tracks=all_median_tracks(pred_stack)
        union=union_vis(vis_stack); maj=majority_vis(vis_stack)
        out_vis=maj.copy()
        N=out_vis.shape[0]
        for qi in range(N):
            key=(str(ref["video_id"]), qi)
            pred=int(pred_by_key.get(key, 0))
            out_vis[qi] = union[qi] if pred==1 else maj[qi]
        rr=dict(ref)
        rr["pred_tracks"]=tracks
        rr["pred_visibility"]=out_vis.astype(bool)
        rr["model_name"]="learned_query_visibility_router"
        records.append(rr)
    out=dict(refp)
    out["model_name"]="learned_query_visibility_router"
    out["records"]=records
    torch.save(out,out_path)

def eval_cache(cache_path: Path, json_path: Path):
    subprocess.run([sys.executable,"scripts/eval_aj_rd_from_cache.py","--cache-path",str(cache_path),"--output-json",str(json_path)], check=True)
    return json.load(open(json_path))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/paper_discovery_2026-06-27/learned_reliability/query_router")
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    caches={t:load_cache(p) for t,p in CACHE_PATHS.items()}
    validate(caches)
    rows=extract_rows(caches, json_query_map(MAJ_JSON), json_query_map(UNION_JSON), json_query_map(GATED_JSON))
    X=np.asarray([r["feature"] for r in rows], dtype=np.float32)
    y=np.asarray([r["label_union"] for r in rows], dtype=np.int64)
    videos=sorted(set(r["video_id"] for r in rows))
    json.dump({"n_rows":len(rows),"n_features":int(X.shape[1]),"videos":videos,"label_union_rate":float(np.mean(y))}, open(out/"dataset_summary.json","w"), indent=2)
    print(json.dumps(json.load(open(out/"dataset_summary.json")), indent=2), flush=True)
    results=[]
    for kind in ["logreg","rf","hgb"]:
        print("=== model",kind,"===", flush=True)
        pred_by_key={}
        y_true=[]; y_pred=[]; held_rows=[]
        for vid in videos:
            tr=[i for i,r in enumerate(rows) if r["video_id"]!=vid]
            te=[i for i,r in enumerate(rows) if r["video_id"]==vid]
            model=make_model(kind)
            # sample weights emphasize cases where union/majority differ
            w=np.asarray([abs(rows[i]["gain_union_vs_majority"])+0.02 for i in tr], dtype=np.float32)
            try:
                model.fit(X[tr], y[tr], **({"logisticregression__sample_weight":w} if kind=="logreg" else {"sample_weight":w}))
            except TypeError:
                model.fit(X[tr], y[tr])
            p=model.predict(X[te]).astype(int)
            for idx,pp in zip(te,p):
                r=rows[idx]
                pred_by_key[(r["video_id"], r["query_idx"])] = int(pp)
                y_true.append(int(y[idx])); y_pred.append(int(pp)); held_rows.append({**{k:v for k,v in r.items() if k!="feature"}, "pred_union":int(pp)})
        cache_path=out/f"learned_query_router_{kind}.pt"
        json_path=out/f"learned_query_router_{kind}_ajrd.json"
        build_pred_cache(caches, rows, pred_by_key, cache_path)
        m=eval_cache(cache_path, json_path)
        row={
            "kind":kind,
            "lvo_accuracy":float(accuracy_score(y_true,y_pred)),
            "lvo_balanced_accuracy":float(balanced_accuracy_score(y_true,y_pred)),
            "pred_union_rate":float(np.mean(y_pred)),
            "true_union_rate":float(np.mean(y_true)),
            "true_AJ_RD_256":m.get("true_AJ_RD_256"),
            "true_AJ_RD":m.get("true_AJ_RD"),
            "proxy":m.get("first_reentry_frame_proxy"),
            "long20_lt4px":(m.get("long_occ_ge20") or {}).get("lt4px"),
            "long20_lt8px":(m.get("long_occ_ge20") or {}).get("lt8px"),
            "dmin16":(m.get("aj_rd_by_dmin_256") or {}).get("16"),
        }
        results.append(row)
        json.dump({"row":row,"heldout_rows":held_rows}, open(out/f"learned_query_router_{kind}_cvpred.json","w"), indent=2, ensure_ascii=False)
        print(json.dumps(row, indent=2), flush=True)
    baseline={
        "fixed_best":0.5546,
        "old_masked_median_majority":0.5871,
        "all_median_union":0.6171,
        "heuristic_gated_mean144":0.6189,
        "oracle":0.6509,
    }
    summary={"baseline":baseline,"results":sorted(results,key=lambda r:r["true_AJ_RD_256"], reverse=True)}
    json.dump(summary, open(out/"summary.json","w"), indent=2, ensure_ascii=False)
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
