#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASETS_DIR="${WORKSPACE_DIR}/datasets"
SLEEP_SECONDS=10
LOCK_FILE="${DATASETS_DIR}/.remaining_direct_downloads.lock"
DOWNLOAD_DAVIS_FRAMES="${DOWNLOAD_DAVIS_FRAMES:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_remaining_direct_datasets.sh [--out DIR]

Downloads all currently-missing non-Google-Drive datasets that can be fetched
directly with the current environment:
  - DAVIS 2017 480p frames
  - TAP-Vid-Kinetics annotations
  - TAP-Vid RGB-Stacking
  - Kinetics metadata files

It skips:
  - existing non-empty files
  - Kubric TFDS / official Kubric point-track annotations, because this
    environment currently lacks the required tooling and the official tracks are
    not directly available here.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      DATASETS_DIR="$2"
      LOCK_FILE="${DATASETS_DIR}/.remaining_direct_downloads.lock"
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

mkdir -p "${DATASETS_DIR}"

if command -v flock >/dev/null 2>&1; then
  exec 9>"${LOCK_FILE}"
  if ! flock -n 9; then
    echo "Another remaining-direct-datasets download job is already running." >&2
    exit 1
  fi
fi

download_with_aria2() {
  local out_dir="$1"
  local filename="$2"
  local url="$3"
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
    --max-connection-per-server=8 \
    --split=8 \
    --min-split-size=4M \
    --dir "${out_dir}" \
    --out "${filename}" \
    "${url}"
}

download_with_wget() {
  local out_dir="$1"
  local filename="$2"
  local url="$3"
  wget -c -O "${out_dir}/${filename}" "${url}"
}

download_with_curl() {
  local out_dir="$1"
  local filename="$2"
  local url="$3"
  curl -L --fail --retry 10 --retry-delay 3 -C - "${url}" -o "${out_dir}/${filename}"
}

get_remote_size() {
  local url="$1"
  curl -fsSLI --retry 3 --retry-delay 2 --max-time 30 "${url}" 2>/dev/null \
    | tr -d '\r' \
    | awk 'tolower($1) == "content-length:" { print $2 }' \
    | tail -n 1
}

download_file() {
  local out_dir="$1"
  local filename="$2"
  shift 2
  local urls=("$@")
  local target="${out_dir}/${filename}"
  mkdir -p "${out_dir}"

  local url remote_size="" local_size=0
  for url in "${urls[@]}"; do
    remote_size="$(get_remote_size "${url}" || true)"
    if [[ -s "${target}" ]]; then
      local_size="$(stat -c%s "${target}")"
      if [[ -n "${remote_size}" && "${local_size}" -eq "${remote_size}" ]]; then
        echo "[skip] ${target}"
        return 0
      fi
      if [[ -n "${remote_size}" && "${local_size}" -gt "${remote_size}" ]]; then
        echo "[redownload] removing oversized local file ${target} (${local_size} > ${remote_size})"
        rm -f "${target}" "${target}.aria2" "${target}.aria2__temp"
        local_size=0
      fi
    fi

    if [[ "${local_size}" -gt 0 ]]; then
      if [[ -n "${remote_size}" ]]; then
        echo "[resume] ${filename} (${local_size}/${remote_size} bytes)"
      else
        echo "[resume] ${filename} (${local_size} bytes, remote size unknown)"
      fi
    else
      echo "[download] ${filename}"
    fi
    echo "  source: ${url}"
    if command -v aria2c >/dev/null 2>&1; then
      if download_with_aria2 "${out_dir}" "${filename}" "${url}"; then
        if [[ -n "${remote_size}" ]]; then
          local_size="$(stat -c%s "${target}")"
          if [[ "${local_size}" -eq "${remote_size}" ]]; then
            return 0
          fi
          echo "  incomplete after transfer (${local_size}/${remote_size} bytes)"
        elif [[ -s "${target}" ]]; then
          return 0
        fi
      fi
    elif command -v wget >/dev/null 2>&1; then
      if download_with_wget "${out_dir}" "${filename}" "${url}"; then
        if [[ -n "${remote_size}" ]]; then
          local_size="$(stat -c%s "${target}")"
          if [[ "${local_size}" -eq "${remote_size}" ]]; then
            return 0
          fi
          echo "  incomplete after transfer (${local_size}/${remote_size} bytes)"
        elif [[ -s "${target}" ]]; then
          return 0
        fi
      fi
    else
      if download_with_curl "${out_dir}" "${filename}" "${url}"; then
        if [[ -n "${remote_size}" ]]; then
          local_size="$(stat -c%s "${target}")"
          if [[ "${local_size}" -eq "${remote_size}" ]]; then
            return 0
          fi
          echo "  incomplete after transfer (${local_size}/${remote_size} bytes)"
        elif [[ -s "${target}" ]]; then
          return 0
        fi
      fi
    fi
    echo "  failed, retrying with next source after ${SLEEP_SECONDS}s"
    sleep "${SLEEP_SECONDS}"
  done

  echo "All sources failed for ${filename}" >&2
  return 1
}

extract_zip_if_needed() {
  local zip_path="$1"
  local marker_path="$2"
  local keep_zip="${3:-0}"

  if [[ -e "${marker_path}" ]]; then
    echo "[skip extract] ${marker_path}"
    return 0
  fi

  echo "[extract] ${zip_path}"
  unzip -qo "${zip_path}" -d "$(dirname "${zip_path}")"

  if [[ "${keep_zip}" != "1" ]]; then
    rm -f "${zip_path}"
  fi
}

flatten_single_nested_dir() {
  local root_dir="$1"
  local nested_name="$2"
  local nested_dir="${root_dir}/${nested_name}"

  if [[ ! -d "${nested_dir}" ]]; then
    return 0
  fi

  find "${nested_dir}" -mindepth 1 -maxdepth 1 -exec mv -n {} "${root_dir}/" \;
  rmdir "${nested_dir}" 2>/dev/null || true
}

echo "[start] downloading remaining direct datasets"

# 1. DAVIS 2017 frames
#
# The TAP-Vid DAVIS annotation package already embeds video data in the pkl on
# this server, so the original frame archive is optional. Keep it off by
# default to avoid a very slow redundant download.
DAVIS_DIR="${DATASETS_DIR}/tapvid_davis"
mkdir -p "${DAVIS_DIR}"
if [[ "${DOWNLOAD_DAVIS_FRAMES}" -eq 1 ]]; then
  if [[ ! -d "${DAVIS_DIR}/DAVIS/JPEGImages/480p" ]]; then
    download_file \
      "${DAVIS_DIR}" \
      "davis.zip" \
      "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
    extract_zip_if_needed "${DAVIS_DIR}/davis.zip" "${DAVIS_DIR}/DAVIS/JPEGImages/480p"
  else
    echo "[skip] DAVIS/JPEGImages/480p already exists"
  fi
else
  echo "[skip] DAVIS frames (set DOWNLOAD_DAVIS_FRAMES=1 to enable)"
fi

# 2. TAP-Vid DAVIS annotations (only if missing)
if [[ ! -s "${DAVIS_DIR}/tapvid_davis.pkl" ]]; then
  download_file \
    "${DAVIS_DIR}" \
    "tapvid_davis.pkl" \
    "https://storage.googleapis.com/dm-tapnet/tapvid_davis.pkl"
else
  echo "[skip] tapvid_davis.pkl already exists"
fi

# 3. TAP-Vid Kinetics
KIN_ANN_DIR="${DATASETS_DIR}/tapvid_kinetics"
mkdir -p "${KIN_ANN_DIR}"
flatten_single_nested_dir "${KIN_ANN_DIR}" "tapvid_kinetics"
if [[ ! -s "${KIN_ANN_DIR}/tapvid_kinetics.csv" ]]; then
  download_file \
    "${KIN_ANN_DIR}" \
    "tapvid_kinetics.zip" \
    "https://storage.googleapis.com/dm-tapnet/tapvid_kinetics.zip"
  extract_zip_if_needed "${KIN_ANN_DIR}/tapvid_kinetics.zip" "${KIN_ANN_DIR}/tapvid_kinetics.csv"
  flatten_single_nested_dir "${KIN_ANN_DIR}" "tapvid_kinetics"
else
  echo "[skip] tapvid_kinetics.csv already exists"
fi
echo "[info] TAP-Vid Kinetics official package provides csv/txt split files, not tapvid_kinetics.pkl"

# 4. TAP-Vid RGB-Stacking
RGB_DIR="${DATASETS_DIR}/tapvid_rgb_stacking"
mkdir -p "${RGB_DIR}"
flatten_single_nested_dir "${RGB_DIR}" "tapvid_rgb_stacking"
if [[ ! -s "${RGB_DIR}/tapvid_rgb_stacking.pkl" ]]; then
  download_file \
    "${RGB_DIR}" \
    "tapvid_rgb_stacking.zip" \
    "https://storage.googleapis.com/dm-tapnet/tapvid_rgb_stacking.zip"
  extract_zip_if_needed "${RGB_DIR}/tapvid_rgb_stacking.zip" "${RGB_DIR}/tapvid_rgb_stacking.pkl"
  flatten_single_nested_dir "${RGB_DIR}" "tapvid_rgb_stacking"
else
  echo "[skip] tapvid_rgb_stacking.pkl already exists"
fi

# 5. Kinetics metadata
KIN_META_DIR="${DATASETS_DIR}/kinetics"
mkdir -p "${KIN_META_DIR}"
download_file \
  "${KIN_META_DIR}" \
  "k700_2020_val_path.txt" \
  "https://s3.amazonaws.com/kinetics/700_2020/val/k700_2020_val_path.txt"
download_file \
  "${KIN_META_DIR}" \
  "val.csv" \
  "https://s3.amazonaws.com/kinetics/700_2020/annotations/val.csv"
download_file \
  "${KIN_META_DIR}" \
  "k700_2020_downloader.sh" \
  "https://ghproxy.vip/https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/main/k700_2020_downloader.sh" \
  "https://gh-proxy.com/https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/main/k700_2020_downloader.sh" \
  "https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/main/k700_2020_downloader.sh"
download_file \
  "${KIN_META_DIR}" \
  "k700_2020_train_path.txt" \
  "https://s3.amazonaws.com/kinetics/700_2020/train/k700_2020_train_path.txt"
download_file \
  "${KIN_META_DIR}" \
  "train.csv" \
  "https://s3.amazonaws.com/kinetics/700_2020/annotations/train.csv"

echo "[info] skipped Kubric TFDS / official Kubric point tracks in current environment"
echo "  reason: tensorflow_datasets and gsutil are not available here"

echo "[done] remaining direct datasets download script finished"
