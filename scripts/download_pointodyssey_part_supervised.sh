#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/datasets/pointodyssey"
PART_NAME="train.tar.gz.partaa"
EXPECTED_SIZE="34359738368"
BASE_URL="https://hf-mirror.com/datasets/aharley/pointodyssey/resolve/main/train.tar.gz.partaa"
SLEEP_SECONDS=5
MAX_CONNECTIONS=8
SPLIT=8
MIN_SPLIT_SIZE="16M"
MAX_ATTEMPTS=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_pointodyssey_part_supervised.sh [options]

Options:
  --part NAME            Output filename. Default: train.tar.gz.partaa
  --url URL              Base download URL. Default: hf-mirror partaa URL
  --size BYTES           Expected file size in bytes
  --connections N        aria2 max connections per server. Default: 8
  --split N              aria2 split value. Default: 8
  --min-split-size SIZE  aria2 min split size. Default: 16M
  --sleep SEC            Sleep between aria2 restarts. Default: 5
  --max-attempts N       Stop after N aria2 runs. 0 means infinite. Default: 0
  --out DIR              Output directory. Default: <repo>/datasets/pointodyssey
  -h, --help             Show help

Notes:
  - This wrapper re-runs aria2c to refresh hf-mirror redirect URLs after expiry.
  - Completion is determined by actual file size on disk.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --part)
      PART_NAME="$2"
      shift 2
      ;;
    --url)
      BASE_URL="$2"
      shift 2
      ;;
    --size)
      EXPECTED_SIZE="$2"
      shift 2
      ;;
    --connections)
      MAX_CONNECTIONS="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --min-split-size)
      MIN_SPLIT_SIZE="$2"
      shift 2
      ;;
    --sleep)
      SLEEP_SECONDS="$2"
      shift 2
      ;;
    --max-attempts)
      MAX_ATTEMPTS="$2"
      shift 2
      ;;
    --out)
      OUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! command -v aria2c >/dev/null 2>&1; then
  echo "aria2c is required." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

human_size() {
  local bytes="$1"
  python3 - "$bytes" <<'PY'
import sys
value = int(sys.argv[1])
units = ["B", "KiB", "MiB", "GiB", "TiB"]
size = float(value)
for unit in units:
    if size < 1024 or unit == units[-1]:
        if unit == "B":
            print(f"{int(size)} {unit}")
        else:
            print(f"{size:.2f} {unit}")
        break
    size /= 1024.0
PY
}

current_size() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    stat -c '%s' "${path}" 2>/dev/null || stat -f '%z' "${path}"
  else
    echo "0"
  fi
}

TARGET_PATH="${OUT_DIR}/${PART_NAME}"
CONTROL_PATH="${TARGET_PATH}.aria2"
attempt=0

is_complete() {
  local target="$1"
  local control="$2"
  if [[ ! -f "${target}" ]]; then
    return 1
  fi
  if [[ -f "${control}" ]]; then
    return 1
  fi
  return 0
}

while true; do
  cur="$(current_size "${TARGET_PATH}")"
  if is_complete "${TARGET_PATH}" "${CONTROL_PATH}"; then
    echo "[done] ${PART_NAME} ($(human_size "${EXPECTED_SIZE}"))"
    exit 0
  fi

  attempt=$((attempt + 1))
  echo "[attempt ${attempt}] ${PART_NAME}"
  echo "  progress: $(human_size "${cur}") / $(human_size "${EXPECTED_SIZE}")"
  echo "  aria2: connections=${MAX_CONNECTIONS} split=${SPLIT} min_split=${MIN_SPLIT_SIZE}"

  aria2c \
    --continue=true \
    --allow-overwrite=false \
    --auto-file-renaming=false \
    --max-tries=0 \
    --retry-wait=3 \
    --timeout=30 \
    --connect-timeout=30 \
    --summary-interval=5 \
    --console-log-level=notice \
    --file-allocation=none \
    --max-connection-per-server="${MAX_CONNECTIONS}" \
    --split="${SPLIT}" \
    --min-split-size="${MIN_SPLIT_SIZE}" \
    --dir "${OUT_DIR}" \
    --out "${PART_NAME}" \
    "${BASE_URL}" || true

  cur="$(current_size "${TARGET_PATH}")"
  echo "  after: $(human_size "${cur}") / $(human_size "${EXPECTED_SIZE}")"

  if is_complete "${TARGET_PATH}" "${CONTROL_PATH}"; then
    echo "[done] ${PART_NAME} ($(human_size "${EXPECTED_SIZE}"))"
    exit 0
  fi

  if [[ "${MAX_ATTEMPTS}" -gt 0 && "${attempt}" -ge "${MAX_ATTEMPTS}" ]]; then
    echo "[stop] max attempts reached with partial file kept for resume"
    exit 1
  fi

  echo "  sleeping ${SLEEP_SECONDS}s before aria2 restart"
  sleep "${SLEEP_SECONDS}"
done
