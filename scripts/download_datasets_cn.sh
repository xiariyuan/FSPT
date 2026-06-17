#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/datasets"
PROFILE="required"
INCLUDE_OPTIONAL=0
DRY_RUN=0
GROUPS_CSV=""
CONTINUE_ON_ERROR=0
FAILED_ITEMS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_datasets_cn.sh [options]

Options:
  --out DIR            Download root directory. Default: <repo>/datasets
  --profile NAME       One of: required, all. Default: required
  --groups CSV         Limit downloads to selected groups:
                       megadepth,tapvid,kinetics,pointodyssey
  --with-optional      Include optional files for stronger experiments
  --continue-on-error  Keep processing other files when one file fails
  --dry-run            Print planned downloads without downloading
  -h, --help           Show this help

Notes:
  - Verified CN-friendly mirrors:
    - PointOdyssey via hf-mirror.com
    - raw.githubusercontent.com via ghproxy.vip / gh-proxy.com
  - No verified CN mirror was found for:
    - TAP-Vid files on storage.googleapis.com
    - MegaDepth files on cs.cornell.edu
    - Kinetics S3 manifests on s3.amazonaws.com
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT_DIR="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --groups)
      GROUPS_CSV="$2"
      shift 2
      ;;
    --with-optional)
      INCLUDE_OPTIONAL=1
      shift
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

if command -v aria2c >/dev/null 2>&1; then
  DOWNLOADER="aria2c"
elif command -v wget >/dev/null 2>&1; then
  DOWNLOADER="wget"
elif command -v curl >/dev/null 2>&1; then
  DOWNLOADER="curl"
else
  echo "aria2c, wget, or curl is required." >&2
  exit 1
fi

group_enabled() {
  local group="$1"
  if [[ -z "${GROUPS_CSV}" ]]; then
    return 0
  fi
  local padded=",${GROUPS_CSV},"
  [[ "${padded}" == *",${group},"* ]]
}

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

download_with_aria2() {
  local out_dir="$1"
  local filename="$2"
  local url="$3"
  aria2c \
    --continue=true \
    --max-connection-per-server=16 \
    --split=16 \
    --min-split-size=4M \
    --dir="${out_dir}" \
    --out="${filename}" \
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
  curl -L --fail --retry 3 --retry-delay 2 -C - "${url}" -o "${out_dir}/${filename}"
}

download_file() {
  local group="$1"
  local out_subdir="$2"
  local filename="$3"
  local size_bytes="$4"
  shift 4
  local urls=("$@")

  local target_dir="${OUT_DIR}/${out_subdir}"
  local target_path="${target_dir}/${filename}"
  local expected_human
  expected_human="$(human_size "${size_bytes}")"

  mkdir -p "${target_dir}"

  if [[ -f "${target_path}" ]]; then
    local current_size
    current_size="$(stat -c '%s' "${target_path}" 2>/dev/null || stat -f '%z' "${target_path}")"
    if [[ "${current_size}" == "${size_bytes}" ]]; then
      echo "[skip] ${group}/${filename} (${expected_human})"
      return 0
    fi
  fi

  echo "[plan] ${group}/${filename}"
  echo "       size: ${expected_human}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    local url
    for url in "${urls[@]}"; do
      echo "       source: ${url}"
    done
    return 0
  fi

  local url
  for url in "${urls[@]}"; do
    echo "       try: ${url}"
    if [[ "${DOWNLOADER}" == "aria2c" ]]; then
      if download_with_aria2 "${target_dir}" "${filename}" "${url}"; then
        return 0
      fi
    elif [[ "${DOWNLOADER}" == "wget" ]]; then
      if download_with_wget "${target_dir}" "${filename}" "${url}"; then
        return 0
      fi
    else
      if download_with_curl "${target_dir}" "${filename}" "${url}"; then
        return 0
      fi
    fi
    echo "       failed: ${url}" >&2
  done

  echo "All sources failed for ${filename}" >&2
  return 1
}

run_download() {
  local group="$1"
  local out_subdir="$2"
  local filename="$3"
  shift 3

  if ! group_enabled "${group}"; then
    return 0
  fi

  if download_file "${group}" "${out_subdir}" "${filename}" "$@"; then
    return 0
  fi

  FAILED_ITEMS+=("${group}/${filename}")
  if [[ "${CONTINUE_ON_ERROR}" -eq 1 ]]; then
    return 0
  fi
  return 1
}

download_required_megadepth() {
  run_download \
    "megadepth" \
    "megadepth" \
    "MegaDepth_v1.tar.gz" \
    "213461598257" \
    "https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz"

  run_download \
    "megadepth" \
    "megadepth" \
    "train_val_list.tar.gz" \
    "3205151" \
    "https://www.cs.cornell.edu/projects/megadepth/dataset/data_lists/train_val_list.tar.gz"

  run_download \
    "megadepth" \
    "megadepth" \
    "test_lists.tar.gz" \
    "740518" \
    "https://www.cs.cornell.edu/projects/megadepth/dataset/data_lists/test_lists.tar.gz"
}

download_required_tapvid() {
  run_download \
    "tapvid" \
    "tapvid" \
    "tapvid_davis.zip" \
    "1668710491" \
    "https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip"

  run_download \
    "tapvid" \
    "tapvid" \
    "tapvid_kinetics.zip" \
    "25018959" \
    "https://storage.googleapis.com/dm-tapnet/tapvid_kinetics.zip"
}

download_required_kinetics() {
  run_download \
    "kinetics" \
    "kinetics" \
    "k700_2020_val_path.txt" \
    "46900" \
    "https://s3.amazonaws.com/kinetics/700_2020/val/k700_2020_val_path.txt"

  run_download \
    "kinetics" \
    "kinetics" \
    "val.csv" \
    "1480292" \
    "https://s3.amazonaws.com/kinetics/700_2020/annotations/val.csv"

  run_download \
    "kinetics" \
    "kinetics" \
    "k700_2020_downloader.sh" \
    "2165" \
    "https://ghproxy.vip/https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/main/k700_2020_downloader.sh" \
    "https://gh-proxy.com/https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/main/k700_2020_downloader.sh" \
    "https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/main/k700_2020_downloader.sh"
}

download_required_pointodyssey() {
  run_download \
    "pointodyssey" \
    "pointodyssey" \
    "train.tar.gz.partaa" \
    "34359738368" \
    "https://hf-mirror.com/datasets/aharley/pointodyssey/resolve/main/train.tar.gz.partaa" \
    "https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/train.tar.gz.partaa?download=true"

  run_download \
    "pointodyssey" \
    "pointodyssey" \
    "train.tar.gz.partab" \
    "34359738368" \
    "https://hf-mirror.com/datasets/aharley/pointodyssey/resolve/main/train.tar.gz.partab" \
    "https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/train.tar.gz.partab?download=true"

  run_download \
    "pointodyssey" \
    "pointodyssey" \
    "train.tar.gz.partac" \
    "34359738368" \
    "https://hf-mirror.com/datasets/aharley/pointodyssey/resolve/main/train.tar.gz.partac" \
    "https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/train.tar.gz.partac?download=true"

  run_download \
    "pointodyssey" \
    "pointodyssey" \
    "train.tar.gz.partad" \
    "31254171521" \
    "https://hf-mirror.com/datasets/aharley/pointodyssey/resolve/main/train.tar.gz.partad" \
    "https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/train.tar.gz.partad?download=true"

  run_download \
    "pointodyssey" \
    "pointodyssey" \
    "val.tar.gz" \
    "20399261433" \
    "https://hf-mirror.com/datasets/aharley/pointodyssey/resolve/main/val.tar.gz" \
    "https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/val.tar.gz?download=true"
}

download_required() {
  download_required_megadepth
  download_required_tapvid
  download_required_kinetics
  download_required_pointodyssey
}

download_optional() {
  run_download \
    "kinetics" \
    "kinetics" \
    "k700_2020_train_path.txt" \
    "49700" \
    "https://s3.amazonaws.com/kinetics/700_2020/train/k700_2020_train_path.txt"

  run_download \
    "kinetics" \
    "kinetics" \
    "train.csv" \
    "21770281" \
    "https://s3.amazonaws.com/kinetics/700_2020/annotations/train.csv"

  run_download \
    "pointodyssey" \
    "pointodyssey" \
    "test.tar.gz" \
    "26521309703" \
    "https://hf-mirror.com/datasets/aharley/pointodyssey/resolve/main/test.tar.gz" \
    "https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/test.tar.gz?download=true"

  run_download \
    "pointodyssey" \
    "pointodyssey" \
    "sample.tar.gz" \
    "3324284510" \
    "https://hf-mirror.com/datasets/aharley/pointodyssey/resolve/main/sample.tar.gz" \
    "https://huggingface.co/datasets/aharley/pointodyssey/resolve/main/sample.tar.gz?download=true"
}

case "${PROFILE}" in
  required)
    download_required
    ;;
  all)
    download_required
    download_optional
    ;;
  *)
    echo "Unsupported profile: ${PROFILE}" >&2
    exit 1
    ;;
esac

if [[ "${INCLUDE_OPTIONAL}" -eq 1 && "${PROFILE}" != "all" ]]; then
  download_optional
fi

if [[ "${#FAILED_ITEMS[@]}" -gt 0 ]]; then
  echo
  echo "Failed downloads:"
  printf '  %s\n' "${FAILED_ITEMS[@]}"
  exit 1
fi

if [[ "${DRY_RUN}" -eq 0 ]]; then
  cat <<EOF

PointOdyssey train parts were downloaded separately.
Merge them after download:
  cd "${OUT_DIR}/pointodyssey"
  cat train.tar.gz.partaa train.tar.gz.partab train.tar.gz.partac train.tar.gz.partad > train.tar.gz
EOF
fi
