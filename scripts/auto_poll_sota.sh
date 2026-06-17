#!/usr/bin/env bash
set -euo pipefail

cd /gemini/code/FSPT
LOG_FILE="${1:-logs/train_resume_qframefix_fix2.log}"
OUT_FILE="${2:-logs/auto_sota_compare.log}"
INTERVAL="${3:-300}"

SOTA_AJ="0.689"
SOTA_OA="0.916"
SOTA_AVG="0.824"

mkdir -p logs

echo "[$(date '+%F %T')] monitor started: log=${LOG_FILE}, interval=${INTERVAL}s" >> "${OUT_FILE}"

while true; do
  ~/miniconda3/bin/python - "$LOG_FILE" "$OUT_FILE" "$SOTA_AJ" "$SOTA_OA" "$SOTA_AVG" <<'PY'
import datetime, os, re, subprocess, sys

log_file, out_file, sota_aj, sota_oa, sota_avg = sys.argv[1:6]
sota_aj = float(sota_aj)
sota_oa = float(sota_oa)
sota_avg = float(sota_avg)

now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

running = False
try:
    rc = subprocess.run("pgrep -f 'python.*train.py|train.py.*python' >/dev/null", shell=True).returncode
    running = (rc == 0)
except Exception:
    running = False

progress_text = "progress N/A"
last_eval = None
runtime_err = False

def parse_eval_from_text(raw_text):
    lines = raw_text.split('\n')
    parsed = None
    for i, line in enumerate(lines):
        me = re.search(r'Epoch\s+(\d+)\s+Evaluation:', line)
        if not me:
            continue
        d = {'epoch': int(me.group(1))}
        for j in range(i + 1, min(i + 140, len(lines))):
            mk = re.search(r'\s+(AJ|OA|<avg|<4px):\s*([0-9.]+)', lines[j])
            if mk:
                d[mk.group(1)] = float(mk.group(2))
        parsed = d
    return parsed

raw = ''
if os.path.exists(log_file):
    raw = open(log_file, 'rb').read().replace(b'\r', b'\n').decode('utf-8', 'ignore')
    runtime_err = ('RuntimeError:' in raw) or ('Traceback (most recent call last)' in raw)

    mprog = list(re.finditer(r'Epoch\s+(\d+):[^\n]*?\|\s*(\d+)/(\d+)\s*\[', raw, flags=re.MULTILINE))
    if mprog:
        m = mprog[-1]
        ep = int(m.group(1))
        cur = int(m.group(2))
        tot = int(m.group(3))
        pct = (100.0 * cur / max(tot, 1))
        progress_text = f"Epoch {ep} {cur}/{tot} ({pct:.2f}%)"

    last_eval = parse_eval_from_text(raw)

if last_eval is None:
    fallback_logs = [
        'logs/train_resume_qframefix_manual.log',
        'logs/train_formal_resume.log',
        'logs/train_formal.log',
    ]
    for fb in fallback_logs:
        if not os.path.exists(fb):
            continue
        raw_fb = open(fb, 'rb').read().replace(b'\r', b'\n').decode('utf-8', 'ignore')
        parsed = parse_eval_from_text(raw_fb)
        if parsed is not None:
            last_eval = parsed
            break

status = 'RUNNING' if running else 'STOPPED'
err_flag = 'ERR' if runtime_err else 'OK'

if last_eval is not None and all(k in last_eval for k in ('AJ', 'OA', '<avg')):
    daj = last_eval['AJ'] - sota_aj
    doa = last_eval['OA'] - sota_oa
    davg = last_eval['<avg'] - sota_avg
    eval_text = (
        f"E{last_eval['epoch']} AJ={last_eval['AJ']:.4f} OA={last_eval['OA']:.4f} <avg={last_eval['<avg']:.4f}; "
        f"ΔSOTA AJ={daj:+.4f} OA={doa:+.4f} <avg={davg:+.4f}"
    )
else:
    eval_text = "eval N/A"

line = f"[{now}] {status}/{err_flag} | {progress_text} | {eval_text}"
with open(out_file, 'a', encoding='utf-8') as f:
    f.write(line + '\n')
print(line)
PY
  sleep "$INTERVAL"
done
