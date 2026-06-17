#!/bin/bash
# =============================================================================
# FSPT - 基线方法代码下载脚本
# 在远程服务器上运行此脚本
# =============================================================================

set -euo pipefail

# 基础目录（默认使用项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${1:-$(cd "$SCRIPT_DIR/.." && pwd)}"
BASE_DIR="$(cd "$BASE_DIR" && pwd)"
BASELINES_DIR="$BASE_DIR/baselines"

echo "============================================"
echo "FSPT Baselines Download Script"
echo "Target directory: $BASELINES_DIR"
echo "============================================"

mkdir -p "$BASELINES_DIR"
cd "$BASELINES_DIR"

# -----------------------------------------------------------------------------
# 1. CoTracker3 (ICCV 2025 - 当前SOTA)
# -----------------------------------------------------------------------------
echo ""
echo "[1/5] Downloading CoTracker3..."
if [ -d "cotracker" ] && [ ! -d "cotracker/.git" ]; then
    if [ -z "$(ls -A cotracker 2>/dev/null || true)" ]; then
        rmdir cotracker || true
    else
        echo "Warning: cotracker exists but is not a git repo. Skipping to avoid overwriting."
        echo "  Remove $BASELINES_DIR/cotracker manually if this is a placeholder directory."
    fi
fi
if [ ! -d "cotracker/.git" ]; then
    git clone https://github.com/facebookresearch/co-tracker.git cotracker
    cd cotracker
    # 安装依赖
    pip install -e .
    # 下载预训练权重
    mkdir -p checkpoints
    cd checkpoints
    # CoTracker3 online model
    wget -c https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth -O scaled_online.pth
    # CoTracker3 offline model
    wget -c https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth -O scaled_offline.pth
    cd ../..
    echo "CoTracker3 downloaded successfully!"
else
    echo "CoTracker3 already exists, skipping..."
fi

# -----------------------------------------------------------------------------
# 2. LocoTrack (ECCV 2024 - 高效追踪)
# -----------------------------------------------------------------------------
echo ""
echo "[2/5] Downloading LocoTrack..."
if [ -d "locotrack" ] && [ ! -d "locotrack/.git" ]; then
    if [ -z "$(ls -A locotrack 2>/dev/null || true)" ]; then
        rmdir locotrack || true
    else
        echo "Warning: locotrack exists but is not a git repo. Skipping to avoid overwriting."
        echo "  Remove $BASELINES_DIR/locotrack manually if this is a placeholder directory."
    fi
fi
if [ ! -d "locotrack/.git" ]; then
    git clone https://github.com/cvlab-kaist/locotrack.git locotrack
    cd locotrack
    # 安装依赖
    pip install -e .
    # 下载预训练权重
    mkdir -p checkpoints
    # 权重需要从Hugging Face下载
    echo "Note: Download LocoTrack weights from https://huggingface.co/cvlab-kaist/locotrack"
    cd ..
    echo "LocoTrack downloaded successfully!"
else
    echo "LocoTrack already exists, skipping..."
fi

# -----------------------------------------------------------------------------
# 3. TAPIR (ICCV 2023 - 经典baseline)
# -----------------------------------------------------------------------------
echo ""
echo "[3/5] Downloading TAPIR..."
if [ -d "tapir" ] && [ ! -d "tapir/.git" ]; then
    if [ -z "$(ls -A tapir 2>/dev/null || true)" ]; then
        rmdir tapir || true
    else
        echo "Warning: tapir exists but is not a git repo. Skipping to avoid overwriting."
        echo "  Remove $BASELINES_DIR/tapir manually if this is a placeholder directory."
    fi
fi
if [ ! -d "tapir/.git" ]; then
    git clone https://github.com/deepmind/tapnet.git tapir
    cd tapir
    # 安装依赖
    pip install -r requirements.txt
    # 下载预训练权重
    mkdir -p checkpoints
    cd checkpoints
    # TAPIR模型权重
    wget -c https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.npy -O tapir_checkpoint.npy
    # Boosted TAPIR
    wget -c https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.npy -O bootstapir_checkpoint.npy
    cd ../..
    echo "TAPIR downloaded successfully!"
else
    echo "TAPIR already exists, skipping..."
fi

# -----------------------------------------------------------------------------
# 4. PIPs (ECCV 2022 - 粒子追踪)
# -----------------------------------------------------------------------------
echo ""
echo "[4/5] Downloading PIPs..."
if [ -d "pips" ] && [ ! -d "pips/.git" ]; then
    if [ -z "$(ls -A pips 2>/dev/null || true)" ]; then
        rmdir pips || true
    else
        echo "Warning: pips exists but is not a git repo. Skipping to avoid overwriting."
        echo "  Remove $BASELINES_DIR/pips manually if this is a placeholder directory."
    fi
fi
if [ ! -d "pips/.git" ]; then
    git clone https://github.com/aharley/pips.git pips
    cd pips
    # 下载预训练权重
    # 权重需要从Google Drive下载
    echo "Note: Download PIPs weights manually from the repo README"
    cd ..
    echo "PIPs downloaded successfully!"
else
    echo "PIPs already exists, skipping..."
fi

# -----------------------------------------------------------------------------
# 5. AllTracker (ICCV 2025 - 高分辨率稠密追踪)
# -----------------------------------------------------------------------------
echo ""
echo "[5/5] Downloading AllTracker..."
if [ -d "alltracker" ] && [ ! -d "alltracker/.git" ]; then
    if [ -z "$(ls -A alltracker 2>/dev/null || true)" ]; then
        rmdir alltracker || true
    else
        echo "Warning: alltracker exists but is not a git repo. Skipping to avoid overwriting."
        echo "  Remove $BASELINES_DIR/alltracker manually if this is a placeholder directory."
    fi
fi
if [ ! -d "alltracker/.git" ]; then
    git clone https://github.com/aharley/alltracker.git alltracker
    cd alltracker
    echo "AllTracker downloaded successfully!"
    cd ..
else
    echo "AllTracker already exists, skipping..."
fi

# -----------------------------------------------------------------------------
# 总结
# -----------------------------------------------------------------------------
echo ""
echo "============================================"
echo "Download Complete!"
echo "============================================"
echo ""
echo "Downloaded baselines:"
ls -la "$BASELINES_DIR"
echo ""
echo "Next steps:"
echo "1. Check each baseline's README for additional setup"
echo "2. Download model weights manually if needed"
echo "3. Run verification scripts to test each baseline"
