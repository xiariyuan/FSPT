#!/usr/bin/env python3
"""Merge and verify sharded Route-D safe-redetection caches."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_safe_redetection_cache import (
    SAFE_REDETECTION_INDEX_SCHEMA,
    canonical_json_sha256,
    file_sha256,
    verify_event_cache,
)


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--protocol',default=str(REPO_ROOT/'configs/routeD_safe_redetection_pointodyssey_protocol_v0.json'))
    ap.add_argument('--role',choices=('fit','model_validation'),required=True)
    ap.add_argument('--cache-root',default=str(REPO_ROOT/'outputs/routeD_safe_redetection_cache_20260718'))
    args=ap.parse_args()
    protocol_path=Path(args.protocol).resolve(); protocol=json.loads(protocol_path.read_text()); protocol_sha=file_sha256(protocol_path)
    events=protocol['partitions'][args.role]['events']; directory=Path(args.cache_root).resolve()/args.role
    rows=[]
    execution_identity = None
    for ordinal,event in enumerate(events):
        path=directory/f"event_{ordinal:05d}_{event['event_identity_sha256'][:12]}.pt"
        payload=verify_event_cache(path,expected_protocol_sha256=protocol_sha,expected_role=args.role)
        if payload['provenance']['event']['event_identity_sha256'] != event['event_identity_sha256']:
            raise RuntimeError(f'event identity drift at {ordinal}')
        current_execution = payload['provenance'].get('execution')
        if not current_execution or not current_execution.get('git_worktree_clean'):
            raise RuntimeError(f'missing/dirty execution provenance at {ordinal}')
        if execution_identity is None:
            execution_identity = current_execution
        elif current_execution != execution_identity:
            raise RuntimeError(f'execution identity drift at {ordinal}')
        rows.append({
            'event_identity_sha256':event['event_identity_sha256'],
            'scene':event['scene'],
            'duration_bucket':event['duration_bucket'],
            'invisibility_duration':event['invisibility_duration'],
            'sidecar':str(path),
            'sidecar_sha256':file_sha256(path),
            'selected_frames':payload['runtime']['selected_frames'],
            'clip_frames':payload['runtime']['clip_frames'],
        })
        if (ordinal+1)%20==0: print(json.dumps({'role':args.role,'verified':ordinal+1,'total':len(events)}),flush=True)
    index={
        'schema_version':SAFE_REDETECTION_INDEX_SCHEMA,
        'protocol':str(protocol_path),
        'protocol_sha256':protocol_sha,
        'role':args.role,
        'expected_count':len(events),
        'completed_count':len(rows),
        'complete':True,
        'full_scale':True,
        'events':rows,
        'execution':execution_identity,
        'locked_data_read':{'internal_holdout':False,'pointodyssey_test':False,'kinetics_1144':False},
    }
    index['payload_sha256']=canonical_json_sha256(index)
    path=directory/'cache_index.json'; path.write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps({'index':str(path),'sha256':file_sha256(path),'count':len(rows)},indent=2),flush=True)

if __name__=='__main__': main()
