#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/datasets/megadepth"
TARGET_FILE="${OUT_DIR}/MegaDepth_v1.tar.gz"
CONTROL_FILE="${TARGET_FILE}.aria2"
LOCK_FILE="${OUT_DIR}/.megadepth_hf_tail_fast.lock"
TMP_DIR="${OUT_DIR}/.hf_tail_parts"

BASE_URL="https://hf-mirror.com/datasets/ssbai/MegaDepth_v1/resolve/main"
PART_PREFIX="MegaDepth_v1.tar.gz_part"
FULL_PART_SIZE="12884901888"
LAST_PART_SIZE="7303168049"
LAST_PART_INDEX="16"
EXPECTED_TOTAL="213461598257"

VERIFY_BYTES="65536"
MAX_CONNECTIONS="16"
SPLIT="16"
MIN_SPLIT_SIZE="4M"
SLEEP_SECONDS="10"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_megadepth_hf_tail_fast.sh [options]

Options:
  --out DIR           Output directory. Default: <repo>/datasets/megadepth
  --connections N     aria2 max connections per server. Default: 16
  --split N           aria2 split value. Default: 16
  --min-split-size S  aria2 min split size. Default: 4M
  --sleep SEC         Sleep between retries. Default: 10
  -h, --help          Show this help

Notes:
  - Verifies the local EOF against hf-mirror before downloading anything.
  - Downloads remaining hf-mirror shards into temporary files with aria2c.
  - Appends those temporary files back into the existing MegaDepth archive in order.
  - Safe to rerun after interruptions; completed temp files are reused.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT_DIR="$2"
      TARGET_FILE="${OUT_DIR}/MegaDepth_v1.tar.gz"
      CONTROL_FILE="${TARGET_FILE}.aria2"
      LOCK_FILE="${OUT_DIR}/.megadepth_hf_tail_fast.lock"
      TMP_DIR="${OUT_DIR}/.hf_tail_parts"
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

mkdir -p "${OUT_DIR}" "${TMP_DIR}"

if command -v flock >/dev/null 2>&1; then
  exec 9>"${LOCK_FILE}"
  if ! flock -n 9; then
    echo "Another hf-mirror MegaDepth fast tail resume appears to be running." >&2
    exit 1
  fi
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required." >&2
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

verify_target_eof() {
  local total="$1"
  if [[ "${total}" -eq 0 ]]; then
    echo "[verify] empty target, nothing to check"
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
    echo "[verify] target EOF mismatch" >&2
    echo "  part: ${index}" >&2
    echo "  range: ${range_start}-${range_end}" >&2
    echo "  local:  ${local_hash}" >&2
    echo "  remote: ${remote_hash}" >&2
    return 1
  fi

  echo "[verify] target EOF part ${index} range ${range_start}-${range_end} sha256 ok"
}

verify_segment_file() {
  local seg_file="$1"
  local index="$2"
  local start="$3"
  local seg_size="$4"

  if [[ ! -f "${seg_file}" ]]; then
    echo "[verify] missing segment file: ${seg_file}" >&2
    return 1
  fi

  local actual
  actual="$(current_size "${seg_file}")"
  if [[ "${actual}" -ne "${seg_size}" ]]; then
    echo "[verify] segment size mismatch for ${seg_file}: ${actual} != ${seg_size}" >&2
    return 1
  fi

  local first_len="${VERIFY_BYTES}"
  if [[ "${seg_size}" -lt "${first_len}" ]]; then
    first_len="${seg_size}"
  fi
  local first_end=$((start + first_len - 1))
  local first_local first_remote
  first_local="$(head -c "${first_len}" "${seg_file}" | sha256_stream)"
  first_remote="$(
    curl \
      -L \
      --fail \
      --connect-timeout 20 \
      --max-time 180 \
      --header 'Accept-Encoding: identity' \
      --range "${start}-${first_end}" \
      "$(part_url "${index}")" | sha256_stream
  )"
  if [[ "${first_local}" != "${first_remote}" ]]; then
    echo "[verify] first block mismatch for ${seg_file}" >&2
    return 1
  fi

  if [[ "${seg_size}" -gt "${VERIFY_BYTES}" ]]; then
    local last_start_in_seg=$((seg_size - VERIFY_BYTES))
    local last_range_start=$((start + last_start_in_seg))
    local last_range_end=$((start + seg_size - 1))
    local last_local last_remote
    last_local="$(tail -c "${VERIFY_BYTES}" "${seg_file}" | sha256_stream)"
    last_remote="$(
      curl \
        -L \
        --fail \
        --connect-timeout 20 \
        --max-time 180 \
        --header 'Accept-Encoding: identity' \
        --range "${last_range_start}-${last_range_end}" \
        "$(part_url "${index}")" | sha256_stream
    )"
    if [[ "${last_local}" != "${last_remote}" ]]; then
      echo "[verify] last block mismatch for ${seg_file}" >&2
      return 1
    fi
  fi

  echo "[verify] segment $(basename "${seg_file}") sha256 spot-check ok"
}

launch_part_download() {
  local index="$1"
  local start="$2"
  local out_name="$3"
  local expected="$4"
  local url
  url="$(part_url "${index}")"
  local out_path="${TMP_DIR}/${out_name}"
  local control_path="${out_path}.aria2"

  if [[ -f "${out_path}" ]]; then
    local existing
    existing="$(current_size "${out_path}")"
    if [[ "${existing}" -eq "${expected}" ]]; then
      echo "[reuse] ${out_name} ($(human_size "${expected}"))"
      return 0
    fi
  fi

  rm -f "${control_path}"

  echo "[spawn] ${out_name}"
  echo "  part: $(printf '%02d' "${index}")"
  echo "  range start: $(human_size "${start}")"
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

    local absolute_start=$((start + existing))
    echo "  curl range: ${absolute_start}-"

    if curl \
      -L \
      --fail \
      --retry 10 \
      --retry-delay 3 \
      --retry-all-errors \
      --connect-timeout 20 \
      --speed-time 60 \
      --speed-limit 1024 \
      --header 'Accept-Encoding: identity' \
      --range "${absolute_start}-" \
      "${url}" >> "${out_path}"; then
      continue
    fi

    local after
    after="$(current_size "${out_path}")"
    if [[ "${after}" -gt "${existing}" ]]; then
      echo "  progress: $(human_size "${after}") / $(human_size "${expected}")"
    else
      echo "  no progress; sleeping ${SLEEP_SECONDS}s before retry"
      sleep "${SLEEP_SECONDS}"
    fi
  done
}

append_segment_file() {
  local seg_file="$1"
  local expected_target_start="$2"

  local target_now
  target_now="$(current_size "${TARGET_FILE}")"
  if [[ "${target_now}" -lt "${expected_target_start}" ]]; then
    echo "Target file is shorter than expected append start." >&2
    return 1
  fi

  local seg_total
  seg_total="$(current_size "${seg_file}")"
  local already_appended=$((target_now - expected_target_start))

  if [[ "${already_appended}" -ge "${seg_total}" ]]; then
    echo "[skip append] $(basename "${seg_file}") already applied"
    return 0
  fi

  echo "[append] $(basename "${seg_file}")"
  echo "  already appended: $(human_size "${already_appended}") / $(human_size "${seg_total}")"

  python3 - "${seg_file}" "${TARGET_FILE}" "${already_appended}" <<'PY'
import shutil, sys
src_path, dst_path, skip = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src_path, 'rb') as src, open(dst_path, 'ab') as dst:
    src.seek(skip)
    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
PY
}

if [[ ! -f "${TARGET_FILE}" ]]; then
  echo "Target file not found: ${TARGET_FILE}" >&2
  exit 1
fi

if [[ -f "${CONTROL_FILE}" ]]; then
  mv -f "${CONTROL_FILE}" "${CONTROL_FILE}.cornell.bak"
  echo "[info] moved stale aria2 control file to ${CONTROL_FILE}.cornell.bak"
fi

current="$(current_size "${TARGET_FILE}")"
if [[ "${current}" -gt "${EXPECTED_TOTAL}" ]]; then
  echo "Target file is larger than expected total size." >&2
  exit 1
fi

echo "[start] MegaDepth hf-mirror fast tail resume"
echo "  progress: $(human_size "${current}") / $(human_size "${EXPECTED_TOTAL}")"
verify_target_eof "${current}"

declare -a JOB_INDEXES=()
declare -a JOB_STARTS=()
declare -a JOB_EXPECTEDS=()
declare -a JOB_OUTPUTS=()
declare -a JOB_PIDS=()

while [[ "${current}" -lt "${EXPECTED_TOTAL}" ]]; do
  index=$((current / FULL_PART_SIZE))
  shard_start=$((index * FULL_PART_SIZE))
  shard_total="$(part_size "${index}")"
  offset=$((current - shard_start))
  expected=$((shard_total - offset))
  out_name="$(printf '%s.from_%d' "${PART_PREFIX}${index}" "${offset}")"

  JOB_INDEXES+=("${index}")
  JOB_STARTS+=("${offset}")
  JOB_EXPECTEDS+=("${expected}")
  JOB_OUTPUTS+=("${out_name}")

  current=$((shard_start + shard_total))
done

echo "[plan] ${#JOB_INDEXES[@]} remaining shard downloads"

for i in "${!JOB_INDEXES[@]}"; do
  launch_part_download "${JOB_INDEXES[$i]}" "${JOB_STARTS[$i]}" "${JOB_OUTPUTS[$i]}" "${JOB_EXPECTEDS[$i]}" &
  JOB_PIDS+=("$!")
done

for pid in "${JOB_PIDS[@]}"; do
  wait "${pid}"
done

for i in "${!JOB_INDEXES[@]}"; do
  verify_segment_file \
    "${TMP_DIR}/${JOB_OUTPUTS[$i]}" \
    "${JOB_INDEXES[$i]}" \
    "${JOB_STARTS[$i]}" \
    "${JOB_EXPECTEDS[$i]}"
done

for i in "${!JOB_INDEXES[@]}"; do
  expected_target_start=$(( JOB_INDEXES[$i] * FULL_PART_SIZE + JOB_STARTS[$i] ))
  append_segment_file "${TMP_DIR}/${JOB_OUTPUTS[$i]}" "${expected_target_start}"
done

final_size="$(current_size "${TARGET_FILE}")"
if [[ "${final_size}" -ne "${EXPECTED_TOTAL}" ]]; then
  echo "[error] final size mismatch: ${final_size} != ${EXPECTED_TOTAL}" >&2
  exit 1
fi

verify_target_eof "${final_size}"
echo "[done] MegaDepth_v1.tar.gz ($(human_size "${EXPECTED_TOTAL}"))"
