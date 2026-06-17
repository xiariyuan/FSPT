#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash projects/mmp_tracker/watch_remote_metrics.sh <output_dir> [interval_seconds]" >&2
  exit 1
fi

OUT_DIR="$1"
INTERVAL="${2:-30}"
METRICS_FILE="$OUT_DIR/epoch_metrics.jsonl"

while true; do
  echo "============================================================"
  echo "Time: $(date '+%F %T')"
  echo "Output dir: $OUT_DIR"
  echo "Metrics file: $METRICS_FILE"
  echo "============================================================"

  if [[ -f "$METRICS_FILE" ]]; then
    python - "$METRICS_FILE" <<'PY'
import json, sys, pathlib
path = pathlib.Path(sys.argv[1])
rows = []
for line in path.read_text(encoding='utf-8').splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        rows.append(json.loads(line))
    except Exception:
        pass

if not rows:
    print('No parsed metrics yet.')
    raise SystemExit(0)

latest = rows[-1]
best_aj = max(rows, key=lambda x: float(x.get('AJ', float('-inf'))))
best_oa = max(rows, key=lambda x: float(x.get('OA', float('-inf'))))
best_err = min(rows, key=lambda x: float(x.get('avg_error_px', float('inf'))))

def fmt(row, keys):
    return ' | '.join(f"{k}={row.get(k)}" for k in keys)

print('Latest:')
print(fmt(latest, ['epoch', 'loss', 'train_coordinate', 'train_local_heatmap', 'train_visibility', 'AJ', 'OA', 'avg_error_px']))
if 'train_long_occ_focus' in latest:
    print(f"train_long_occ_focus={latest.get('train_long_occ_focus')}")
print('')
print('Best AJ:')
print(fmt(best_aj, ['epoch', 'AJ', 'OA', 'avg_error_px']))
print('Best OA:')
print(fmt(best_oa, ['epoch', 'AJ', 'OA', 'avg_error_px']))
print('Best avg_error_px:')
print(fmt(best_err, ['epoch', 'AJ', 'OA', 'avg_error_px']))
print('')
print('Last 5 rows:')
for row in rows[-5:]:
    print(fmt(row, ['epoch', 'loss', 'AJ', 'OA', 'avg_error_px']))
PY
  else
    echo "No metrics file yet."
  fi

  echo ""
  echo "Sleeping ${INTERVAL}s... Press Ctrl+C to stop."
  sleep "$INTERVAL"
done
