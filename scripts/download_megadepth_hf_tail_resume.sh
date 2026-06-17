#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/datasets/megadepth"
TARGET_FILE="${OUT_DIR}/MegaDepth_v1.tar.gz"
CONTROL_FILE="${TARGET_FILE}.aria2"
LOCK_FILE="${OUT_DIR}/.megadepth_hf_tail_resume.lock"

BASE_URL="https://hf-mirror.com/datasets/ssbai/MegaDepth_v1/resolve/main"
PART_PREFIX="MegaDepth_v1.tar.gz_part"
FULL_PART_SIZE="12884901888"
LAST_PART_SIZE="7303168049"
LAST_PART_INDEX="16"
EXPECTED_TOTAL="213461598257"

VERIFY_BYTES="65536"
SLEEP_SECONDS="10"
MAX_ATTEMPTS="0"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_megadepth_hf_tail_resume.sh [options]

Options:
  --out DIR          Output directory. Default: <repo>/datasets/megadepth
  --sleep SEC        Sleep between retry attempts. Default: 10
  --max-attempts N   Stop after N curl attempts. 0 means infinite. Default: 0
  -h, --help         Show this help

Notes:
  - This script assumes the local MegaDepth_v1.tar.gz is already a valid prefix.
  - Before appending bytes, it verifies the current EOF against the matching
    hf-mirror shard to avoid mixing mismatched content.
  - It appends remaining bytes directly to the existing archive, so it can
    resume safely after interruptions without redownloading the whole file.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT_DIR="$2"
      TARGET_FILE="${OUT_DIR}/MegaDepth_v1.tar.gz"
      CONTROL_FILE="${TARGET_FILE}.aria2"
      LOCK_FILE="${OUT_DIR}/.megadepth_hf_tail_resume.lock"
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
    echo "Another hf-mirror MegaDepth tail resume appears to be running." >&2
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

part_size() {
  local index="$1"
  if [[ "${index}" -lt "${LAST_PART_INDEX}" ]]; then
    echo "${FULL_PART_SIZE}"
  else
    echo "${LAST_PART_SIZE}"
  fi
}

part_url() {
  local index="$1"
  printf '%s/%s%02d' "${BASE_URL}" "${PART_PREFIX}" "${index}"
}

sha256_stream() {
  python3 -c '
import hashlib, sys
h = hashlib.sha256()
while True:
    chunk = sys.stdin.buffer.read(1024 * 1024)
    if not chunk:
        break
    h.update(chunk)
print(h.hexdigest())
'
}

verify_eof_against_remote() {
  local total="$1"
  if [[ "${total}" -eq 0 ]]; then
    echo "[verify] empty file, nothing to check"
    return 0
  fi

  local last_byte=$((total - 1))
  local index=$((last_byte / FULL_PART_SIZE))
  local shard_start=$((index * FULL_PART_SIZE))
  local shard_offset_end=$((total - shard_start))
  local verify_len="${VERIFY_BYTES}"

  if [[ "${shard_offset_end}" -lt "${verify_len}" ]]; then
    verify_len="${shard_offset_end}"
  fi

  local range_start=$((shard_offset_end - verify_len))
  local range_end=$((shard_offset_end - 1))
  local local_hash remote_hash

  local_hash="$(tail -c "${verify_len}" "${TARGET_FILE}" | sha256_stream)"
  remote_hash="$(
    curl \
      -L \
      --fail \
      --connect-timeout 20 \
      --max-time 180 \
      --header 'Accept-Encoding: identity' \
      --range "${range_start}-${range_end}" \
      "$(part_url "${index}")" | sha256_stream
  )"

  if [[ "${local_hash}" != "${remote_hash}" ]]; then
    echo "[verify] mismatch at EOF" >&2
    echo "  part: ${index}" >&2
    echo "  range: ${range_start}-${range_end}" >&2
    echo "  local:  ${local_hash}" >&2
    echo "  remote: ${remote_hash}" >&2
    return 1
  fi

  echo "[verify] part ${index} range ${range_start}-${range_end} sha256 ok"
}

download_from_current_offset() {
  local index="$1"
  local total_before="$2"
  local shard_start=$((index * FULL_PART_SIZE))
  local shard_total
  shard_total="$(part_size "${index}")"
  local shard_offset=$((total_before - shard_start))

  if [[ "${shard_offset}" -ge "${shard_total}" ]]; then
    echo "[skip] part $(printf '%02d' "${index}") already complete"
    return 0
  fi

  local remaining=$((shard_total - shard_offset))
  echo "[download] part $(printf '%02d' "${index}")"
  echo "  offset:    $(human_size "${shard_offset}") / $(human_size "${shard_total}")"
  echo "  remaining: $(human_size "${remaining}")"

  local attempts=0
  while true; do
    local current
    current="$(current_size "${TARGET_FILE}")"
    local current_offset=$((current - shard_start))
    if [[ "${current_offset}" -ge "${shard_total}" ]]; then
      echo "  done: part $(printf '%02d' "${index}")"
      return 0
    fi

    attempts=$((attempts + 1))
    local range_start="${current_offset}"
    echo "  attempt ${attempts}: bytes ${range_start}-"

    if curl \
      -L \
      --fail \
      --connect-timeout 20 \
      --speed-time 60 \
      --speed-limit 1024 \
      --header 'Accept-Encoding: identity' \
      --range "${range_start}-" \
      "$(part_url "${index}")" >> "${TARGET_FILE}"; then
      verify_eof_against_remote "$(current_size "${TARGET_FILE}")"
      echo "  done: part $(printf '%02d' "${index}")"
      return 0
    fi

    local after
    after="$(current_size "${TARGET_FILE}")"
    if [[ "${after}" -gt "${current}" ]]; then
      verify_eof_against_remote "${after}"
      echo "  progress: $(human_size "${after}") / $(human_size "${EXPECTED_TOTAL}")"
    else
      echo "  no progress on this attempt"
    fi

    if [[ "${MAX_ATTEMPTS}" -gt 0 && "${attempts}" -ge "${MAX_ATTEMPTS}" ]]; then
      echo "[stop] max attempts reached with partial file preserved" >&2
      return 1
    fi

    echo "  sleeping ${SLEEP_SECONDS}s before retry"
    sleep "${SLEEP_SECONDS}"
  done
}

if [[ ! -f "${TARGET_FILE}" ]]; then
  echo "Target file not found: ${TARGET_FILE}" >&2
  exit 1
fi

current="$(current_size "${TARGET_FILE}")"
if [[ "${current}" -gt "${EXPECTED_TOTAL}" ]]; then
  echo "Target file is larger than expected total size." >&2
  exit 1
fi

if [[ -f "${CONTROL_FILE}" ]]; then
  mv -f "${CONTROL_FILE}" "${CONTROL_FILE}.cornell.bak"
  echo "[info] moved stale aria2 control file to ${CONTROL_FILE}.cornell.bak"
fi

echo "[start] MegaDepth hf-mirror tail resume"
echo "  progress: $(human_size "${current}") / $(human_size "${EXPECTED_TOTAL}")"
verify_eof_against_remote "${current}"

while [[ "${current}" -lt "${EXPECTED_TOTAL}" ]]; do
  index=$((current / FULL_PART_SIZE))
  download_from_current_offset "${index}" "${current}"
  current="$(current_size "${TARGET_FILE}")"
done

if [[ "${current}" -ne "${EXPECTED_TOTAL}" ]]; then
  echo "[error] final size mismatch: ${current} != ${EXPECTED_TOTAL}" >&2
  exit 1
fi

verify_eof_against_remote "${current}"
echo "[done] MegaDepth_v1.tar.gz ($(human_size "${EXPECTED_TOTAL}"))"
