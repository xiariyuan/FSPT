#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URLS_FILE="${ROOT_DIR}/references/papers/paper_urls.tsv"
OUT_DIR="${ROOT_DIR}/references/papers"

if [[ ! -f "${URLS_FILE}" ]]; then
  echo "paper URL list not found: ${URLS_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

download_with_curl() {
  local name="$1"
  local url="$2"
  local out="${OUT_DIR}/${name}.pdf"
  echo "[curl] ${name}"
  curl -L --fail --retry 3 --retry-delay 2 -C - "${url}" -o "${out}"
}

download_with_wget() {
  local name="$1"
  local url="$2"
  local out="${OUT_DIR}/${name}.pdf"
  echo "[wget] ${name}"
  wget -c -O "${out}" "${url}"
}

while IFS=$'\t' read -r name url; do
  [[ -z "${name}" ]] && continue
  if command -v curl >/dev/null 2>&1; then
    if ! download_with_curl "${name}" "${url}"; then
      if command -v wget >/dev/null 2>&1; then
        download_with_wget "${name}" "${url}"
      else
        exit 1
      fi
    fi
  elif command -v wget >/dev/null 2>&1; then
    download_with_wget "${name}" "${url}"
  else
    echo "Neither curl nor wget is available." >&2
    exit 1
  fi
done < "${URLS_FILE}"

echo
echo "Downloaded papers:"
ls -lh "${OUT_DIR}"/*.pdf
