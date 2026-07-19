#!/usr/bin/env python3
"""Verify primary/replay P0l evaluations and package the frozen final decision."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_cmcp_final_holdout import (
    FINAL_HOLDOUT_RESULT_SCHEMA,
    evaluate_final_holdout_gate,
    load_final_holdout_config,
)
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256

DEFAULT_CONFIG = REPO_ROOT / "configs/routeD_cmcp_final_holdout_v0.yaml"
DEFAULT_PRIMARY = REPO_ROOT / "outputs/routeD_cmcp_final_holdout_20260719/evaluation/primary.json"
DEFAULT_REPLAY = REPO_ROOT / "outputs/routeD_cmcp_final_holdout_20260719/evaluation/replay.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs/generated/ROUTED_STRONG_BACKBONE_FINAL_HOLDOUT_SUMMARY_2026-07-19.json"


def _read(path: Path) -> dict[str, Any]: return json.loads(path.read_text())


def verify_eval_replay(primary: dict[str,Any], replay: dict[str,Any]) -> dict[str,bool]:
    fields=("schema_version","seed","config_sha256","cache_index_sha256","cache_index_payload_sha256","checkpoint_sha256","combined_model_state_sha256","adapter_state_sha256","cmcp_state_sha256","comparator_state_sha256","normalization","metrics","model_selection_on_final_holdout","calibration_read","external_data_read")
    checks={key:primary[key]==replay[key] for key in fields}
    if not all(checks.values()): raise RuntimeError(f"P0l replay mismatch: {checks}")
    return checks


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default=str(DEFAULT_CONFIG)); ap.add_argument("--primary",default=str(DEFAULT_PRIMARY)); ap.add_argument("--replay",default=str(DEFAULT_REPLAY)); ap.add_argument("--output",default=str(DEFAULT_OUTPUT)); args=ap.parse_args()
    config=load_final_holdout_config(args.config); pp=Path(args.primary).resolve(); rp=Path(args.replay).resolve(); primary=_read(pp); replay=_read(rp)
    for label,row in (("primary",primary),("replay",replay)):
        if row["schema_version"]!=FINAL_HOLDOUT_RESULT_SCHEMA: raise RuntimeError(f"bad {label} schema")
        if row["config_sha256"]!=config["_config_sha256"]: raise RuntimeError(f"bad {label} config")
        if row["calibration_read"] or any(row["external_data_read"].values()): raise RuntimeError(f"locked data read in {label}")
        if row["model_selection_on_final_holdout"]: raise RuntimeError("final holdout used for selection")
    replay_checks=verify_eval_replay(primary,replay)
    metrics=primary["metrics"]; gate=evaluate_final_holdout_gate(metrics,config,exact_replay=True)
    gains=[float(row["selected_AJ_gain_points"]) for row in metrics["per_video"]]
    summary={
        "schema_version":"routeD_strong_backbone_final_holdout_summary_v0","date":"2026-07-19","status":"completed_pass" if gate["pass"] else "completed_fail",
        "formal_decision":gate["decision"],"claim_boundary":"Frozen output-only variant C on the one-time identity-disjoint 16-video Kubric final holdout. No calibration, checkpoint selection, DAVIS read, or Kinetics rerun.",
        "model":{"checkpoint_sha256":primary["checkpoint_sha256"],"combined_model_state_sha256":primary["combined_model_state_sha256"]},
        "cache":{"index":primary["cache_index"],"index_sha256":primary["cache_index_sha256"],"payload_sha256":primary["cache_index_payload_sha256"]},
        "metrics":metrics,"gate":gate,
        "per_video_summary":{"videos":len(gains),"positive":sum(x>0 for x in gains),"negative":sum(x<0 for x in gains),"min":min(gains),"median":statistics.median(gains),"mean":statistics.mean(gains),"max":max(gains)},
        "replay":{"exact":True,"checks":replay_checks,"primary_sha256":file_sha256(pp),"replay_sha256":file_sha256(rp)},
        "calibration_read":False,"external_data_read":primary["external_data_read"],
    }
    output=Path(args.output).resolve(); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"output":str(output),"sha256":file_sha256(output),"decision":gate["decision"],"AJ_gain_points":metrics["selected_gain_points"]["AJ"],"AJ_CI":metrics["paired_video_selected_AJ_gain_CI"],"gate_pass":gate["pass"]},indent=2))

if __name__=="__main__": main()
