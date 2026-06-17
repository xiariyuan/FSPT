#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/datasets/megadepth"
SLEEP_SECONDS=30
MAX_ATTEMPTS=0
LOCK_FILE="${OUT_DIR}/.megadepth_download.lock"
MAX_CONNECTIONS=16
SPLIT=16
MIN_SPLIT_SIZE="4M"

MEGA_URL="https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz"
MEGA_FILE="MegaDepth_v1.tar.gz"
MEGA_SIZE="213461598257"

LIST_URL_1="https://www.cs.cornell.edu/projects/megadepth/dataset/data_lists/train_val_list.tar.gz"
LIST_FILE_1="train_val_list.tar.gz"
LIST_SIZE_1="3205151"

LIST_URL_2="https://www.cs.cornell.edu/projects/megadepth/dataset/data_lists/test_lists.tar.gz"
LIST_FILE_2="test_lists.tar.gz"
LIST_SIZE_2="740518"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_megadepth_resume.sh [options]

Options:
  --out DIR          Output directory. Default: <repo>/datasets/megadepth
  --sleep SEC        Sleep between retry attempts. Default: 30
  --max-attempts N   Stop after N attempts. 0 means infinite. Default: 0
  --connections N    aria2 max connections per server. Default: 16
  --split N          aria2 split value. Default: 16
  --min-split-size   aria2 min split size. Default: 4M
  -h, --help         Show this help

Notes:
  - Uses aria2c when available for HTTP range-based resume and parallel segments.
  - A watchdog loop re-launches the transfer after transient failures.
  - Small MegaDepth list files are downloaded once before the large archive loop.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT_DIR="$2"
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

mkdir -p "${OUT_DIR}"

if command -v flock >/dev/null 2>&1; then
  exec 9>"${LOCK_FILE}"
  if ! flock -n 9; then
    echo "Another MegaDepth download appears to be running." >&2
    exit 1
  fi
fi

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

download_small_file() {
  local filename="$1"
  local expected_size="$2"
  local url="$3"
  local target="${OUT_DIR}/${filename}"
  local cur
  cur="$(current_size "${target}")"
  if [[ "${cur}" == "${expected_size}" ]]; then
    echo "[skip] ${filename}"
    return 0
  fi
  echo "[download] ${filename} ($(human_size "${expected_size}"))"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c \
      --continue=true \
      --allow-overwrite=false \
      --auto-file-renaming=false \
      --max-tries=0 \
      --retry-wait=3 \
      --timeout=30 \
      --connect-timeout=30 \
      --summary-interval=30 \
      --console-log-level=warn \
      --file-allocation=none \
      --max-connection-per-server=1 \
      --split=1 \
      --min-split-size=1M \
      --dir "${OUT_DIR}" \
      --out "${filename}" \
      "${url}"
  else
    wget \
      -c \
      --tries=0 \
      --retry-connrefused \
      --waitretry=10 \
      --timeout=30 \
      --read-timeout=30 \
      --no-http-keep-alive \
      --user-agent="Mozilla/5.0" \
      -O "${target}" \
      "${url}"
  fi
}

download_small_file "${LIST_FILE_1}" "${LIST_SIZE_1}" "${LIST_URL_1}"
download_small_file "${LIST_FILE_2}" "${LIST_SIZE_2}" "${LIST_URL_2}"

attempt=0
target="${OUT_DIR}/${MEGA_FILE}"

while true; do
  cur="$(current_size "${target}")"
  if [[ "${cur}" == "${MEGA_SIZE}" ]]; then
    echo "[done] ${MEGA_FILE} ($(human_size "${MEGA_SIZE}"))"
    exit 0
  fi

  attempt=$((attempt + 1))
  echo "[attempt ${attempt}] ${MEGA_FILE}"
  echo "  progress: $(human_size "${cur}") / $(human_size "${MEGA_SIZE}")"
  echo "  aria2: connections=${MAX_CONNECTIONS} split=${SPLIT} min_split=${MIN_SPLIT_SIZE}"

  if command -v aria2c >/dev/null 2>&1; then
    aria2c \
      --continue=true \
      --allow-overwrite=false \
      --auto-file-renaming=false \
      --max-tries=0 \
      --retry-wait=3 \
      --timeout=30 \
      --connect-timeout=30 \
      --summary-interval=30 \
      --console-log-level=notice \
      --file-allocation=none \
      --max-connection-per-server="${MAX_CONNECTIONS}" \
      --split="${SPLIT}" \
      --min-split-size="${MIN_SPLIT_SIZE}" \
      --dir "${OUT_DIR}" \
      --out "${MEGA_FILE}" \
      "${MEGA_URL}" || true
  else
    wget \
      -c \
      --tries=0 \
      --retry-connrefused \
      --waitretry=10 \
      --timeout=30 \
      --read-timeout=30 \
      --no-http-keep-alive \
      --user-agent="Mozilla/5.0" \
      --progress=dot:giga \
      -O "${target}" \
      "${MEGA_URL}" || true
  fi

  cur="$(current_size "${target}")"
  echo "  after: $(human_size "${cur}") / $(human_size "${MEGA_SIZE}")"

  if [[ "${cur}" == "${MEGA_SIZE}" ]]; then
    echo "[done] ${MEGA_FILE} ($(human_size "${MEGA_SIZE}"))"
    exit 0
  fi

  if [[ "${MAX_ATTEMPTS}" -gt 0 && "${attempt}" -ge "${MAX_ATTEMPTS}" ]]; then
    echo "[stop] max attempts reached with partial file kept for resume"
    exit 1
  fi

  echo "  sleeping ${SLEEP_SECONDS}s before retry"
  sleep "${SLEEP_SECONDS}"
done
