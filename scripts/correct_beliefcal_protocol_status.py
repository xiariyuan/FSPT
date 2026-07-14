#!/usr/bin/env python3
"""Create qualification-only status sidecars without modifying historical artifacts."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from projects.mmp_tracker.run_beliefcal_mvp1 import load_protocol_deviation, protocol_claim_status, sha256_file

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-root',required=True)
    p.add_argument('--deviation',required=True)
    args=p.parse_args()
    root=Path(args.run_root).resolve()
    dev=load_protocol_deviation(args.deviation)
    status=protocol_claim_status(dev)
    summary={**status,'run_root':str(root),'historical_artifacts_preserved':True}
    (root/'qualification_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (root/'PROTOCOL_STATUS.md').write_text('# BeliefCal-MMP Protocol Status\n\nEvidence tier: qualification only\n\nProtocol deviation recorded. Historical artifacts preserved.\n',encoding='utf-8')
    print(root/'qualification_summary.json')
if __name__=='__main__': main()
