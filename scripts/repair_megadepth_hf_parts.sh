#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/datasets/megadepth"
TARGET_FILE="${OUT_DIR}/MegaDepth_v1.tar.gz"
PART_DIR="${OUT_DIR}/.hf_tail_parts"
LOCK_FILE="${OUT_DIR}/.megadepth_hf_repair.lock"

BASE_URL="https://hf-mirror.com/datasets/ssbai/MegaDepth_v1/resolve/main"
PART_PREFIX="MegaDepth_v1.tar.gz_part"
FULL_PART_SIZE="12884901888"
LAST_PART_SIZE="7303168049"
LAST_PART_INDEX="16"

REPAIR_START_INDEX="${REPAIR_START_INDEX:-8}"
REPAIR_END_INDEX="${REPAIR_END_INDEX:-16}"
MAX_CONNECTIONS="${MAX_CONNECTIONS:-16}"
SPLIT="${SPLIT:-16}"
MIN_SPLIT_SIZE="${MIN_SPLIT_SIZE:-4M}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/repair_megadepth_hf_parts.sh [options]

Options:
  --out DIR            Output directory. Default: <repo>/datasets/megadepth
  --start N            First part index to repair. Default: 8
  --end N              Last part index to repair. Default: 16
  --connections N      aria2 max connections per server. Default: 16
  --split N            aria2 split value. Default: 16
  --min-split-size S   aria2 min split size. Default: 4M

This script re-downloads selected hf-mirror shard files into
  <out>/.hf_tail_parts/
verifies them with range checks, and overwrites the corresponding byte ranges
inside MegaDepth_v1.tar.gz in place.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT_DIR="$2"
      TARGET_FILE="${OUT_DIR}/MegaDepth_v1.tar.gz"
      PART_DIR="${OUT_DIR}/.hf_tail_parts"
      LOCK_FILE="${OUT_DIR}/.megadepth_hf_repair.lock"
      shift 2
      ;;
    --start)
      REPAIR_START_INDEX="$2"
      shift 2
      ;;
    --end)
      REPAIR_END_INDEX="$2"
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

mkdir -p "${OUT_DIR}" "${PART_DIR}"

if command -v flock >/dev/null 2>&1; then
  exec 9>"${LOCK_FILE}"
  if ! flock -n 9; then
    echo "Another MegaDepth repair job is already running." >&2
    exit 1
  fi
fi

if [[ ! -f "${TARGET_FILE}" ]]; then
  echo "Target file not found: ${TARGET_FILE}" >&2
  exit 1
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

download_part() {
  local index="$1"
  local expected="$2"
  local out_name="$(printf '%s%02d' "${PART_PREFIX}" "${index}")"
  local out_path="${PART_DIR}/${out_name}"
  local url
  url="$(part_url "${index}")"

  if [[ -f "${out_path}" ]]; then
    local existing
    existing="$(current_size "${out_path}")"
    if [[ "${existing}" -eq "${expected}" ]]; then
      echo "[reuse] ${out_name} ($(human_size "${expected}"))"
      return 0
    fi
  fi

  echo "[download] ${out_name}"
  echo "  url: ${url}"
  echo "  expected: $(human_size "${expected}")"

  while true; do
    local existing
    existing="$(current_size "${out_path}")"
    if [[ "${existing}" -eq "${expected}" ]]; then
      echo "  done: ${out_name}"
      return 0
    fi
    if [[ "${existing}" -gt "${expected}" ]]; then
      echo "  error: ${out_name} is larger than expected (${existing} > ${expected})" >&2
      return 1
    fi

    local remaining=$((expected - existing))
    echo "  progress: $(human_size "${existing}") / $(human_size "${expected}")"
    echo "  remaining: $(human_size "${remaining}")"

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
        --dir "${PART_DIR}" \
        --out "${out_name}" \
        "${url}"
    else
      curl \
        -L \
        --fail \
        --retry 10 \
        --retry-delay 3 \
        --retry-all-errors \
        --continue-at - \
        --connect-timeout 20 \
        --speed-time 60 \
        --speed-limit 1024 \
        --header 'Accept-Encoding: identity' \
        "${url}" \
        -o "${out_path}"
    fi
  done
}

verify_part() {
  local index="$1"
  local expected="$2"
  local out_name="$(printf '%s%02d' "${PART_PREFIX}" "${index}")"
  local out_path="${PART_DIR}/${out_name}"
  local sample=65536
  if [[ "${expected}" -lt "${sample}" ]]; then
    sample="${expected}"
  fi

  local first_local first_remote last_local last_remote
  first_local="$(head -c "${sample}" "${out_path}" | sha256_stream)"
  first_remote="$(
    curl \
      -L \
      --fail \
      --retry 5 \
      --retry-delay 3 \
      --retry-all-errors \
      --connect-timeout 20 \
      --max-time 600 \
      --header 'Accept-Encoding: identity' \
      --range "0-$((sample - 1))" \
      "$(part_url "${index}")" | sha256_stream
  )"
  if [[ "${first_local}" != "${first_remote}" ]]; then
    echo "[verify] first block mismatch for ${out_name}" >&2
    return 1
  fi

  if [[ "${expected}" -gt "${sample}" ]]; then
    local last_start=$((expected - sample))
    local last_end=$((expected - 1))
    last_local="$(tail -c "${sample}" "${out_path}" | sha256_stream)"
    last_remote="$(
      curl \
        -L \
        --fail \
        --retry 5 \
        --retry-delay 3 \
        --retry-all-errors \
        --connect-timeout 20 \
        --max-time 600 \
        --header 'Accept-Encoding: identity' \
        --range "${last_start}-${last_end}" \
        "$(part_url "${index}")" | sha256_stream
    )"
    if [[ "${last_local}" != "${last_remote}" ]]; then
      echo "[verify] last block mismatch for ${out_name}" >&2
      return 1
    fi
  fi

  echo "[verify] ${out_name} spot-check ok"
}

overwrite_part() {
  local index="$1"
  local expected="$2"
  local out_name="$(printf '%s%02d' "${PART_PREFIX}" "${index}")"
  local out_path="${PART_DIR}/${out_name}"
  local offset=$((index * FULL_PART_SIZE))

  echo "[patch] ${out_name}"
  echo "  offset: $(human_size "${offset}")"
  echo "  size:   $(human_size "${expected}")"

  python3 - "${TARGET_FILE}" "${out_path}" "${offset}" "${expected}" <<'PY'
import shutil, sys
target_path, part_path, offset, expected = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(target_path, 'r+b') as target, open(part_path, 'rb') as part:
    target.seek(offset)
    remaining = expected
    while remaining > 0:
        chunk = part.read(min(8 * 1024 * 1024, remaining))
        if not chunk:
            raise RuntimeError(f"Unexpected EOF while patching {part_path}")
        target.write(chunk)
        remaining -= len(chunk)
PY
}

echo "[start] MegaDepth hf-mirror repair"
echo "  target: ${TARGET_FILE}"
echo "  range:  ${REPAIR_START_INDEX}..${REPAIR_END_INDEX}"

for ((idx=REPAIR_START_INDEX; idx<=REPAIR_END_INDEX; idx++)); do
  expected="$(part_size "${idx}")"
  download_part "${idx}" "${expected}"
  verify_part "${idx}" "${expected}"
  overwrite_part "${idx}" "${expected}"
done

echo "[done] repaired selected MegaDepth shard ranges in place"
