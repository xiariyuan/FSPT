#!/usr/bin/env python3
"""
Fetch Track-On official repo and install locally for PRT adapter baseline.
This script **does NOT modify** the main FSPT repo.
It clones the external Track-On repo, installs PyTorch deps, and pre-downloads
checkpoints automatically.
Configuration path pattern:
  baselines/track-on/      <- external repo clone
  baselines/track-on/checkpoints/  <- pretrained weights
  scripts/eval_trackon_prt.py  <- PRT evaluation interface (new file)
"""

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

BASELINES = Path(__file__).resolve().parent.parent / "baselines"
TRACKON_ROOT = BASELINES / "track-on"
CHECKPOINT_DIR = TRACKON_ROOT / "checkpoints"


def run(cmd: str, cwd=None, capture=False):
    print("[CMD]", cmd)
    out = subprocess.run(shlex.split(cmd), cwd=cwd, capture_output=capture)
    if out.returncode != 0:
        raise RuntimeError(f"Command failed\\n{cmd}\\n----\\nstdout:\\n{out.stdout.decode(errors='ignore')}\\nstderr:\\n{out.stderr.decode(errors='ignore')}")
    return out


def main():
    parser = argparse.ArgumentParser("Track-On PRT adapter setup")
    parser.add_argument("--force", action="store_true", help="force fresh clone/build")
    parser.add_argument("--no-clone", action="store_true", help="skip git clone")
    parser.add_argument("--no-install", action="store_true", help="skip pip install")
    args = parser.parse_args()

    # Ensure conda python
    py_path = Path(sys.executable)
    if "conda" not in str(py_path.resolve()):
        print("!! WARNING: Not running inside conda env. pip install may need --user.")

    # Create directories
    BASELINES.mkdir(parents=True, exist_ok=True)

    # 1) Clone Track-On repo if requested
    if not TRACKON_ROOT.exists() or args.force:
        if TRACKON_ROOT.exists():
            shutil.rmtree(TRACKON_ROOT)
        if not args.no_clone:
            print("Cloning Track-On repo...", flush=True)
            run(
                f"git clone --depth=1 https://github.com/aharley/track-on {TRACKON_ROOT}",
                capture=False,
            )
        else:
            raise RuntimeError("Track-On directory missing, but --no-clone passed.")
    else:
        print("Track-On repo already exists.")

    # 2) Install dependencies inside conda python if requested
    if not args.no_install:
        print("Installing Track-On dependencies...")
        # Install via pip (repo already has simple requirements)
        run(
            "pip install -r requirements.txt",
            cwd=str(TRACKON_ROOT),
            capture=False,
        )

    # 3) Download checkpoint automatically
    ckpt_path = CHECKPOINT_DIR / "offline.pth"
    if not ckpt_path.exists():
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        print("Downloading Track-On offline checkpoint...")
        # Use wget or curl fallback to Python as last resort
        try:
            run("wget -q https://github.com/aharley/track-on/releases/download/v1/offline.pth -O offline.pth",
                cwd=str(CHECKPOINT_DIR), capture=False)
        except Exception:
            # last resort
            try:
                import urllib.request
                dest = str(ckpt_path)
                urllib.request.urlretrieve(
                    "https://github.com/aharley/track-on/releases/download/v1/offline.pth",
                    dest,
                )
                print("Checkpoint downloaded via urllib.")
            except Exception as e:
                raise RuntimeError("Unable to download checkpoint. May require manual download from:") from trackon.cotracker.predictor import CoTrackerPredictor
                print("  https://github.com/aharley/track-on/releases")

    print("\\n================ TRACK-ON READY ================")
    print(f"Repo : {TRACKON_ROOT}")
    print(f"CKPT : {ckpt_path}")
    print("Smoke test with:")
    print("  python scripts/eval_trackon_prt.py --max-sequences 1 --max-queries-per-seq 4")


if __name__ == "__main__":
    main()
