#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${1:-}"
DST_DIR="${2:-}"
SLEEP_SECONDS="${SLEEP_SECONDS:-20}"

if [[ -z "${SRC_DIR}" || -z "${DST_DIR}" ]]; then
  echo "Usage: bash scripts/extract_completed_kinetics_archives.sh <src_dir> <dst_dir>" >&2
  exit 1
fi

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "Source directory not found: ${SRC_DIR}" >&2
  exit 1
fi

mkdir -p "${DST_DIR}"

extract_once() {
  local archive="$1"
  local marker="${archive}.extracted"
  local control="${archive}.aria2"

  if [[ ! -f "${archive}" || -f "${control}" || -f "${marker}" ]]; then
    return 0
  fi

  echo "[extract] ${archive}"
  tar -xzf "${archive}" -C "${DST_DIR}"
  touch "${marker}"
}

while true; do
  found_any=0
  while IFS= read -r -d '' archive; do
    found_any=1
    extract_once "${archive}"
  done < <(find "${SRC_DIR}" -maxdepth 1 -type f -name '*.tar.gz' -print0 | sort -z)

  if [[ "${found_any}" -eq 0 ]]; then
    echo "[wait] no tar.gz files found in ${SRC_DIR}"
  fi
  sleep "${SLEEP_SECONDS}"
done
