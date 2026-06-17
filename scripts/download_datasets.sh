#!/bin/bash
# =============================================================================
# FSPT - TAP-Vid 数据集下载脚本
# 在远程服务器上运行此脚本
# =============================================================================

set -euo pipefail

# 基础目录（默认使用服务器工作区 /gemini/code）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${1:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
BASE_DIR="$(cd "$BASE_DIR" && pwd)"
DATASETS_DIR="$BASE_DIR/datasets"

echo "============================================"
echo "FSPT TAP-Vid Datasets Download Script"
echo "Target directory: $DATASETS_DIR"
echo "============================================"

mkdir -p "$DATASETS_DIR"
cd "$DATASETS_DIR"

flatten_single_nested_dir() {
    local root_dir="$1"
    local nested_name="$2"
    local nested_dir="$root_dir/$nested_name"

    if [ ! -d "$nested_dir" ]; then
        return 0
    fi

    find "$nested_dir" -mindepth 1 -maxdepth 1 -exec mv -n {} "$root_dir/" \;
    rmdir "$nested_dir" 2>/dev/null || true
}

# -----------------------------------------------------------------------------
# 1. TAP-Vid Kubric (合成数据，用于训练)
# -----------------------------------------------------------------------------
echo ""
echo "[1/4] Downloading TAP-Vid Kubric..."
KUBRIC_DIR="$DATASETS_DIR/tapvid_kubric"
mkdir -p "$KUBRIC_DIR"
cd "$KUBRIC_DIR"
DOWNLOAD_KUBRIC_TFDS="${DOWNLOAD_KUBRIC_TFDS:-0}"
DOWNLOAD_DAVIS_FRAMES="${DOWNLOAD_DAVIS_FRAMES:-0}"

if [ "$DOWNLOAD_KUBRIC_TFDS" -eq 1 ]; then
    # 下载Kubric数据集（TFDS，仅调试用）
    # 注意：这是合成数据，需要从Google Cloud Storage下载
    echo "Downloading Kubric TFDS data (movi_e/256x256)..."
    mkdir -p ./movi_e/256x256
    if command -v gsutil &> /dev/null; then
        gsutil -m cp -r "gs://kubric-public/tfds/movi_e/256x256/1.0.0" ./movi_e/256x256/ 2>/dev/null || {
            echo "gsutil download failed."
            echo "Please download manually from: https://github.com/deepmind/tapnet#downloading-the-dataset"
        }
    else
        echo "gsutil not available, skipping TFDS download."
        echo "Please download manually from: https://github.com/deepmind/tapnet#downloading-the-dataset"
    fi
else
    echo "Skipping Kubric TFDS download (debug only)."
    echo "Set DOWNLOAD_KUBRIC_TFDS=1 to enable TFDS download."
fi

if [ ! -f "$KUBRIC_DIR/tapvid_kubric_train.pkl" ]; then
    echo "Warning: tapvid_kubric_train.pkl not found."
    echo "TFDS does not include point tracks by default."
    echo "For debugging, set data.train.use_tfds=true and allow_synthetic_tracks=true."
    echo "For real training, provide official Kubric pickle annotations."
elif [ ! -s "$KUBRIC_DIR/tapvid_kubric_train.pkl" ]; then
    echo "Warning: tapvid_kubric_train.pkl is empty; please re-download."
fi

cd "$DATASETS_DIR"
echo "TAP-Vid Kubric setup complete!"

# -----------------------------------------------------------------------------
# 2. TAP-Vid DAVIS (真实视频测试集)
# -----------------------------------------------------------------------------
echo ""
echo "[2/4] Downloading TAP-Vid DAVIS..."
DAVIS_DIR="$DATASETS_DIR/tapvid_davis"
mkdir -p "$DAVIS_DIR"
cd "$DAVIS_DIR"

# The DAVIS annotation package already embeds video data in the pkl on this
# server. Keep the original frame archive optional to avoid a long redundant
# download unless you explicitly need it.
if [ "$DOWNLOAD_DAVIS_FRAMES" -eq 1 ]; then
    if [ ! -d "$DAVIS_DIR/DAVIS/JPEGImages/480p" ]; then
        echo "Downloading DAVIS 2017 videos..."
        wget -c https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip -O davis.zip
        unzip -q davis.zip
        rm davis.zip
    else
        echo "DAVIS frames already exist, skipping..."
    fi
else
    echo "Skipping DAVIS 2017 frames (set DOWNLOAD_DAVIS_FRAMES=1 to enable)."
fi

# 下载TAP-Vid DAVIS标注（pkl）
if [ ! -s "$DAVIS_DIR/tapvid_davis.pkl" ]; then
    echo "Downloading TAP-Vid DAVIS annotations..."
    wget -c https://storage.googleapis.com/dm-tapnet/tapvid_davis.pkl -O tapvid_davis.pkl
else
    echo "tapvid_davis.pkl already exists, skipping..."
fi

cd "$DATASETS_DIR"
echo "TAP-Vid DAVIS setup complete!"

# -----------------------------------------------------------------------------
# 3. TAP-Vid Kinetics (真实视频测试集)
# -----------------------------------------------------------------------------
echo ""
echo "[3/4] Setting up TAP-Vid Kinetics..."
KINETICS_DIR="$DATASETS_DIR/tapvid_kinetics"
mkdir -p "$KINETICS_DIR"
cd "$KINETICS_DIR"
flatten_single_nested_dir "$KINETICS_DIR" "tapvid_kinetics"

if [ ! -s "$KINETICS_DIR/tapvid_kinetics.csv" ]; then
    echo "Downloading TAP-Vid Kinetics annotations..."
    wget -c https://storage.googleapis.com/dm-tapnet/tapvid_kinetics.zip -O tapvid_kinetics.zip
    unzip -q tapvid_kinetics.zip
    rm tapvid_kinetics.zip
    flatten_single_nested_dir "$KINETICS_DIR" "tapvid_kinetics"
fi

if [ -f "$KINETICS_DIR/tapvid_kinetics.csv" ] && [ ! -s "$KINETICS_DIR/tapvid_kinetics.csv" ]; then
    echo "Warning: tapvid_kinetics.csv is empty; please re-download."
else
    echo "TAP-Vid Kinetics annotation package is ready."
fi

echo ""
echo "NOTE: The current official TAP-Vid Kinetics zip provides csv/txt split files,"
echo "      not a ready-to-use tapvid_kinetics.pkl."
echo "      Kinetics clips can be populated from the official Kinetics-700 tar archives"
echo "      under /gemini/code/datasets/kinetics/raw_*_targz and linked with scripts/link_tapvid_kinetics_videos.py."

cd "$DATASETS_DIR"
echo "TAP-Vid Kinetics setup complete!"

# -----------------------------------------------------------------------------
# 4. RGB-Stacking (机器人操作场景，当前代码默认不使用)
# -----------------------------------------------------------------------------
echo ""
echo "[4/4] Setting up TAP-Vid RGB-Stacking (optional)..."
RGBS_DIR="$DATASETS_DIR/tapvid_rgb_stacking"
mkdir -p "$RGBS_DIR"
cd "$RGBS_DIR"
flatten_single_nested_dir "$RGBS_DIR" "tapvid_rgb_stacking"

if [ ! -s "$RGBS_DIR/tapvid_rgb_stacking.pkl" ]; then
    echo "Downloading TAP-Vid RGB-Stacking (optional)..."
    wget -c https://storage.googleapis.com/dm-tapnet/tapvid_rgb_stacking.zip -O tapvid_rgb_stacking.zip
    unzip -q tapvid_rgb_stacking.zip
    rm tapvid_rgb_stacking.zip
    flatten_single_nested_dir "$RGBS_DIR" "tapvid_rgb_stacking"

    cd "$DATASETS_DIR"
    echo "TAP-Vid RGB-Stacking downloaded successfully! (optional)"
else
    cd "$DATASETS_DIR"
    echo "TAP-Vid RGB-Stacking already exists, skipping..."
fi

# -----------------------------------------------------------------------------
# 创建数据集索引文件
# -----------------------------------------------------------------------------
echo ""
echo "Creating dataset index..."
cat > "$DATASETS_DIR/dataset_info.json" << 'EOF'
{
    "tapvid_kubric": {
        "path": "tapvid_kubric",
        "type": "synthetic",
        "usage": "training",
        "num_videos": "~10000",
        "resolution": "256x256"
    },
    "tapvid_davis": {
        "path": "tapvid_davis",
        "type": "real",
        "usage": "testing",
        "num_videos": 30,
        "resolution": "480p"
    },
    "tapvid_kinetics": {
        "path": "tapvid_kinetics",
        "type": "real",
        "usage": "testing",
        "num_videos": 1189,
        "resolution": "variable"
    },
    "tapvid_rgb_stacking": {
        "path": "tapvid_rgb_stacking",
        "type": "real",
        "usage": "optional",
        "num_videos": 50,
        "resolution": "256x256",
        "optional": true
    }
}
EOF

# -----------------------------------------------------------------------------
# 总结
# -----------------------------------------------------------------------------
echo ""
echo "============================================"
echo "Dataset Download Complete!"
echo "============================================"
echo ""
echo "Downloaded datasets:"
ls -la "$DATASETS_DIR"
echo ""
echo "Dataset sizes:"
du -sh "$DATASETS_DIR"/* 2>/dev/null || echo "(sizes will be available after download)"
echo ""
echo "Next steps:"
echo "1. Verify dataset integrity"
echo "2. For Kinetics: download videos using yt-dlp"
echo "3. For Kubric: install gsutil and download training data"
