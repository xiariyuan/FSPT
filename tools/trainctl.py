#!/usr/bin/env python3
"""FSPT training control center v3.2.

Core commands:
  python3 tools/trainctl.py start NAME -- COMMAND...
  python3 tools/trainctl.py status NAME
  python3 tools/trainctl.py stop NAME
  python3 tools/trainctl.py list
  python3 tools/trainctl.py logs NAME --lines 100
  python3 tools/trainctl.py gpu

v3.2 commands:
  python3 tools/trainctl.py loss NAME [--json] [--csv OUT]
  python3 tools/trainctl.py compare [--metric loss]
  python3 tools/trainctl.py checkpoints [NAME]
  python3 tools/trainctl.py summary NAME
  python3 tools/trainctl.py events [NAME]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
CHECKPOINT_DIRS = [ROOT / "checkpoints", ROOT / "weights", ROOT / "outputs", EXPERIMENTS]
CKPT_EXTS = {".pt", ".pth", ".ckpt", ".safetensors"}
EVENT_PREFIX = "events.out.tfevents"

LOSS_PATTERNS = [
    re.compile(r"(?:^|[ ,;])(?:total_)?loss(?:[:=]|\s+)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.I),
    re.compile(r"(?:^|[ ,;])train_loss(?:[:=]|\s+)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.I),
]
LR_PATTERN = re.compile(r"(?:^|[ ,;])lr(?:[:=]|\s+)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", re.I)
STEP_PATTERN = re.compile(r"(?:^|[ ,;])(?:step|iter|iteration|global_step)(?:[:=]|\s+)(\d+)", re.I)
EPOCH_PATTERN = re.compile(r"(?:^|[ ,;])epoch(?:[:=]|\s+)(\d+)", re.I)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path)


def safe_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name).strip("._-")
    if not safe:
        raise SystemExit("invalid experiment name")
    return safe


def run_dir(name: str) -> Path:
    return EXPERIMENTS / safe_name(name)


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def is_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_log(name: str) -> Path:
    rd = run_dir(name)
    preferred = rd / "train.log"
    if preferred.exists():
        return preferred
    logs = sorted(rd.rglob("*.log"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    if logs:
        return logs[0]
    return preferred


def parse_loss_log(log_file: Path) -> list[dict[str, Any]]:
    if not log_file.exists():
        return []
    records: list[dict[str, Any]] = []
    last_step = 0
    with log_file.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            loss_value = None
            for pat in LOSS_PATTERNS:
                m = pat.search(line)
                if m:
                    try:
                        loss_value = float(m.group(1))
                    except ValueError:
                        loss_value = None
                    break
            if loss_value is None or not math.isfinite(loss_value):
                continue
            sm = STEP_PATTERN.search(line)
            em = EPOCH_PATTERN.search(line)
            lm = LR_PATTERN.search(line)
            step = int(sm.group(1)) if sm else last_step + 1
            last_step = max(last_step, step)
            rec = {
                "line": line_no,
                "step": step,
                "loss": loss_value,
                "epoch": int(em.group(1)) if em else None,
                "lr": float(lm.group(1)) if lm else None,
                "text": line.strip()[:300],
            }
            records.append(rec)
    return records


def loss_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    losses = [float(r["loss"]) for r in records]
    best_idx = min(range(len(losses)), key=lambda i: losses[i])
    last = records[-1]
    first = records[0]
    best = records[best_idx]
    return {
        "count": len(records),
        "first_loss": first["loss"],
        "last_loss": last["loss"],
        "best_loss": best["loss"],
        "best_step": best["step"],
        "last_step": last["step"],
        "delta": last["loss"] - first["loss"],
    }


def experiment_dirs() -> list[Path]:
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    return [p for p in sorted(EXPERIMENTS.iterdir()) if p.is_dir() and not p.name.startswith("__")]


def list_checkpoints(base: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not base.exists():
        return out
    for path in base.rglob("*"):
        if path.is_file() and path.suffix.lower() in CKPT_EXTS:
            st = path.stat()
            out.append({"path": rel(path), "size": st.st_size, "mtime": st.st_mtime})
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out


def list_events(base: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not base.exists():
        return out
    for path in base.rglob("*"):
        if path.is_file() and path.name.startswith(EVENT_PREFIX):
            st = path.stat()
            out.append({"path": rel(path), "size": st.st_size, "mtime": st.st_mtime})
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out


def print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_start(args: argparse.Namespace) -> int:
    if not args.command:
        print("missing command after --", file=sys.stderr)
        return 2
    rd = run_dir(args.name)
    rd.mkdir(parents=True, exist_ok=True)
    pid_file = rd / "pid.txt"
    log_file = rd / "train.log"
    meta_file = rd / "meta.json"

    old_pid = read_pid(pid_file)
    if is_alive(old_pid):
        print_json({"ok": False, "error": "already running", "name": args.name, "pid": old_pid, "log": rel(log_file)})
        return 1

    workdir = (ROOT / args.workdir).resolve() if not Path(args.workdir).is_absolute() else Path(args.workdir).resolve()
    try:
        workdir.relative_to(ROOT.resolve())
    except ValueError:
        print(f"workdir outside root: {workdir}", file=sys.stderr)
        return 2

    command = " ".join(args.command)
    with log_file.open("ab") as log:
        log.write((f"\n===== START {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n").encode())
        log.write((f"cwd: {workdir}\ncmd: {command}\n\n").encode())
        proc = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=str(workdir),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
    pid_file.write_text(str(proc.pid), encoding="utf-8")
    write_json(meta_file, {"name": args.name, "pid": proc.pid, "cmd": command, "workdir": str(workdir), "log": rel(log_file), "started_at": time.time()})
    print_json({"ok": True, "name": args.name, "pid": proc.pid, "log": rel(log_file), "workdir": rel(workdir)})
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    rd = run_dir(args.name)
    pid = read_pid(rd / "pid.txt")
    out = {"ok": True, "name": args.name, "exists": rd.exists(), "pid": pid, "alive": is_alive(pid), "run_dir": rel(rd), "meta": load_json(rd / "meta.json")}
    print_json(out)
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    rd = run_dir(args.name)
    pid = read_pid(rd / "pid.txt")
    if not pid:
        print_json({"ok": False, "error": "pid not found", "name": args.name})
        return 1
    try:
        os.killpg(pid, signal.SIGTERM)
        stopped = True
    except ProcessLookupError:
        stopped = False
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
            stopped = True
        except Exception as exc:
            print_json({"ok": False, "error": str(exc), "name": args.name, "pid": pid})
            return 1
    print_json({"ok": True, "name": args.name, "pid": pid, "stopped": stopped})
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    runs = []
    for rd in experiment_dirs():
        pid = read_pid(rd / "pid.txt")
        records = parse_loss_log(rd / "train.log")
        runs.append({"name": rd.name, "pid": pid, "alive": is_alive(pid), "run_dir": rel(rd), "loss": loss_stats(records)})
    print_json({"ok": True, "runs": runs})
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    log_file = find_log(args.name)
    if not log_file.exists():
        print(f"log not found: {log_file}", file=sys.stderr)
        return 1
    proc = subprocess.run(["tail", "-n", str(args.lines), str(log_file)], text=True, capture_output=True)
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    return proc.returncode


def cmd_gpu(args: argparse.Namespace) -> int:
    query_cmd = ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"]
    try:
        proc = subprocess.run(query_cmd, text=True, capture_output=True, timeout=10)
        if proc.returncode == 0 and args.json:
            gpus = []
            for line in proc.stdout.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    gpus.append({"index": parts[0], "name": parts[1], "memory_used_mb": parts[2], "memory_total_mb": parts[3], "utilization_gpu_pct": parts[4], "temperature_c": parts[5]})
            print_json({"ok": True, "gpus": gpus})
            return 0
        if proc.returncode == 0:
            print(proc.stdout, end="")
            return 0
        fallback = subprocess.run(["nvidia-smi"], text=True, capture_output=True, timeout=10)
        print(fallback.stdout, end="")
        if fallback.stderr:
            print(fallback.stderr, file=sys.stderr, end="")
        return fallback.returncode
    except FileNotFoundError:
        print_json({"ok": False, "error": "nvidia-smi not found"})
        return 1
    except Exception as exc:
        print_json({"ok": False, "error": str(exc)})
        return 1


def cmd_loss(args: argparse.Namespace) -> int:
    log_file = find_log(args.name)
    records = parse_loss_log(log_file)
    stats = loss_stats(records)
    out = {"ok": True, "name": args.name, "log": rel(log_file), "stats": stats, "records": records[-args.last:] if args.last else records}
    if args.csv:
        csv_path = (ROOT / args.csv).resolve() if not Path(args.csv).is_absolute() else Path(args.csv).resolve()
        try:
            csv_path.relative_to(ROOT.resolve())
        except ValueError:
            print(f"csv path outside root: {csv_path}", file=sys.stderr)
            return 2
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["line", "step", "loss", "epoch", "lr", "text"])
            writer.writeheader()
            writer.writerows(records)
        out["csv"] = rel(csv_path)
    if args.json:
        print_json(out)
    else:
        print(f"experiment: {args.name}")
        print(f"log: {rel(log_file)}")
        print_json(stats)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    rows = []
    for rd in experiment_dirs():
        records = parse_loss_log(rd / "train.log")
        st = loss_stats(records)
        if st.get("count", 0):
            rows.append({"name": rd.name, **st})
    key = args.sort_by
    rows.sort(key=lambda r: (r.get(key) is None, r.get(key, float("inf"))))
    print_json({"ok": True, "sort_by": key, "experiments": rows})
    return 0


def cmd_checkpoints(args: argparse.Namespace) -> int:
    bases = [run_dir(args.name)] if args.name else CHECKPOINT_DIRS
    checkpoints: list[dict[str, Any]] = []
    for base in bases:
        checkpoints.extend(list_checkpoints(base))
    checkpoints.sort(key=lambda x: x["mtime"], reverse=True)
    print_json({"ok": True, "count": len(checkpoints[:args.max]), "checkpoints": checkpoints[:args.max]})
    return 0


def cmd_events(args: argparse.Namespace) -> int:
    bases = [run_dir(args.name)] if args.name else [EXPERIMENTS, ROOT / "runs", ROOT / "outputs"]
    events: list[dict[str, Any]] = []
    for base in bases:
        events.extend(list_events(base))
    print_json({"ok": True, "count": len(events), "events": events})
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    rd = run_dir(args.name)
    pid = read_pid(rd / "pid.txt")
    log_file = find_log(args.name)
    records = parse_loss_log(log_file)
    checkpoints = list_checkpoints(rd)[:10]
    events = list_events(rd)[:10]
    out = {
        "ok": True,
        "name": args.name,
        "alive": is_alive(pid),
        "pid": pid,
        "run_dir": rel(rd),
        "log": rel(log_file),
        "loss": loss_stats(records),
        "recent_checkpoints": checkpoints,
        "tensorboard_events": events,
        "meta": load_json(rd / "meta.json"),
    }
    print_json(out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="FSPT training control center v3.2")
    sub = parser.add_subparsers(dest="action", required=True)

    p = sub.add_parser("start")
    p.add_argument("name")
    p.add_argument("--workdir", default=".")
    p.add_argument("command", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_start)

    p = sub.add_parser("status")
    p.add_argument("name")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("stop")
    p.add_argument("name")
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("list")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("logs")
    p.add_argument("name")
    p.add_argument("--lines", type=int, default=100)
    p.set_defaults(func=cmd_logs)

    p = sub.add_parser("gpu")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_gpu)

    p = sub.add_parser("loss")
    p.add_argument("name")
    p.add_argument("--json", action="store_true")
    p.add_argument("--csv")
    p.add_argument("--last", type=int, default=0)
    p.set_defaults(func=cmd_loss)

    p = sub.add_parser("compare")
    p.add_argument("--sort-by", choices=["best_loss", "last_loss", "delta", "last_step"], default="best_loss")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("checkpoints")
    p.add_argument("name", nargs="?")
    p.add_argument("--max", type=int, default=50)
    p.set_defaults(func=cmd_checkpoints)

    p = sub.add_parser("events")
    p.add_argument("name", nargs="?")
    p.set_defaults(func=cmd_events)

    p = sub.add_parser("summary")
    p.add_argument("name")
    p.set_defaults(func=cmd_summary)

    args = parser.parse_args()
    if getattr(args, "command", None) and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
