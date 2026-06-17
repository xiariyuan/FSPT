#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/datasets/pointodyssey"
PROFILE="required"
POLL_SECONDS=30

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_pointodyssey_cn_queue.sh [options]

Options:
  --profile NAME   One of: required, all. Default: required
  --out DIR        Output directory. Default: <repo>/datasets/pointodyssey
  --poll SEC       Poll interval while waiting for an active download. Default: 30
  -h, --help       Show this help

Profiles:
  required
    train.tar.gz.partaa
    train.tar.gz.partab
    train.tar.gz.partac
    train.tar.gz.partad
    val.tar.gz

  all
    required profile plus:
    test.tar.gz
    sample.tar.gz

Behavior:
  - If a file is already being downloaded by another aria2 process, this script waits.
  - If a file is complete, this script skips it.
  - Otherwise it launches the supervised downloader for that file.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --out)
      OUT_DIR="$2"
      shift 2
      ;;
    --poll)
      POLL_SECONDS="$2"
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

run_item() {
  local part_name="$1"
  local expected_size="$2"
  local url="$3"
  local target="${OUT_DIR}/${part_name}"
  local control="${target}.aria2"

  if [[ -f "${target}" && ! -f "${control}" ]]; then
    echo "[skip] ${part_name} looks complete"
    return 0
  fi

  if [[ -f "${control}" ]]; then
    echo "[wait] ${part_name} is already active"
    while [[ -f "${control}" ]]; do
      sleep "${POLL_SECONDS}"
    done
    if [[ -f "${target}" && ! -f "${control}" ]]; then
      echo "[done] ${part_name} completed by existing downloader"
      return 0
    fi
  fi

  echo "[start] ${part_name}"
  bash "${ROOT_DIR}/scripts/download_pointodyssey_part_supervised.sh" \
    --out "${OUT_DIR}" \
    --part "${part_name}" \
    --size "${expected_size}" \
    --url "${url}" \
    --connections 16 \
    --split 16 \
    --min-split-size 4M \
    --sleep 5
}

run_required() {
  run_item "train.tar.gz.partaa" "34359738368" "https://hf-mirror.com/datasets/aharley/pointodyssey/resolve/main/train.tar.gz.partaa"
  run_item "train.tar.gz.partab" "34359738368" "https://hf-mirror.com/datasets/aharley/pointodyssey/resolve/main/train.tar.gz.partab"
  run_item "train.tar.gz.partac" "34359738368" "https://hf-mirror.com/datasets/aharley/pointodyssey/resolve/main/train.tar.gz.partac"
  run_item "train.tar.gz.partad" "31254171521" "https://hf-mirror.com/datasets/aharley/pointodyssey/resolve/main/train.tar.gz.partad"
  run_item "val.tar.gz" "20399261433" "https://hf-mirror.com/datasets/aharley/pointodyssey/resolve/main/val.tar.gz"
}

run_optional() {
  run_item "test.tar.gz" "26521309703" "https://hf-mirror.com/datasets/aharley/pointodyssey/resolve/main/test.tar.gz"
  run_item "sample.tar.gz" "3324284510" "https://hf-mirror.com/datasets/aharley/pointodyssey/resolve/main/sample.tar.gz"
}

case "${PROFILE}" in
  required)
    run_required
    ;;
  all)
    run_required
    run_optional
    ;;
  *)
    echo "Unsupported profile: ${PROFILE}" >&2
    exit 1
    ;;
esac
