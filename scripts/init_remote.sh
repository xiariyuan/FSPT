#!/bin/bash
# =============================================================================
# FSPT - 远程服务器初始化脚本
# 在远程GPU服务器上运行此脚本进行一键初始化
# 
# 使用方法:
#   bash init_remote.sh
# =============================================================================

set -euo pipefail

echo "============================================"
echo "FSPT Remote Server Initialization"
echo "============================================"

# 配置（可通过环境变量覆盖）
PROJECT_DIR="${PROJECT_DIR:-$HOME/FSPT}"
ENV_NAME="${ENV_NAME:-fspt}"
FMTRACK_DIR="${FMTRACK_DIR:-$HOME/FMtrack-main/FM-Track}"
TACO_DIR="${TACO_DIR:-$HOME/taco}"

if [ ! -d "$(dirname "$PROJECT_DIR")" ]; then
    echo "Parent directory does not exist: $(dirname "$PROJECT_DIR")"
    exit 1
fi

# -----------------------------------------------------------------------------
# 1. 创建项目目录
# -----------------------------------------------------------------------------
echo ""
echo "[1/7] Creating project directory structure..."

mkdir -p "$PROJECT_DIR"/{docs,papers,models,datasets,configs,scripts,experiments,logs,outputs,checkpoints,assets,util}
mkdir -p "$PROJECT_DIR"/baselines/{cotracker,locotrack,tapir,pips,alltracker}

cd "$PROJECT_DIR"
echo "Project directory created at: $PROJECT_DIR"

# -----------------------------------------------------------------------------
# 2. 设置Conda环境
# -----------------------------------------------------------------------------
echo ""
echo "[2/7] Setting up Conda environment..."

# 初始化conda
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
else
    echo "Conda initialization script not found."
    exit 1
fi

# 检查环境是否存在
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment $ENV_NAME already exists, activating..."
else
    echo "Creating new environment: $ENV_NAME"
    conda create -n $ENV_NAME python=3.10 -y
fi

conda activate $ENV_NAME

# 安装PyTorch
echo "Installing PyTorch with CUDA..."
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121 -q

# 安装核心依赖
echo "Installing core dependencies..."
pip install numpy scipy pillow opencv-python einops timm transformers -q
pip install omegaconf hydra-core pyyaml tqdm matplotlib wandb tensorboard -q
pip install kornia mediapy imageio imageio-ffmpeg -q
pip install open-clip-torch ftfy regex -q

# 安装CLIP
pip install git+https://github.com/openai/CLIP.git -q

echo "Environment setup complete!"

# -----------------------------------------------------------------------------
# 3. 从FM-Track复制模块
# -----------------------------------------------------------------------------
echo ""
echo "[3/7] Copying modules from FM-Track..."

if [ -d "$FMTRACK_DIR" ]; then
    # 复制核心模块
    for file in learnable_freq_decomposition.py freq_temporal_transformer.py freq_aware_trajectory_modeling.py; do
        if [ -f "$FMTRACK_DIR/models/motip/$file" ]; then
            cp "$FMTRACK_DIR/models/motip/$file" "$PROJECT_DIR/models/"
        else
            echo "Warning: $FMTRACK_DIR/models/motip/$file not found"
        fi
    done
    
    # 复制工具函数
    for file in box_ops.py misc.py; do
        if [ -f "$FMTRACK_DIR/util/$file" ]; then
            cp "$FMTRACK_DIR/util/$file" "$PROJECT_DIR/util/"
        else
            echo "Warning: $FMTRACK_DIR/util/$file not found"
        fi
    done
    
    echo "FM-Track modules copied!"
else
    echo "Warning: FM-Track directory not found at $FMTRACK_DIR"
fi

# -----------------------------------------------------------------------------
# 4. 从TACO复制模块
# -----------------------------------------------------------------------------
echo ""
echo "[4/7] Copying modules from TACO..."

if [ -d "$TACO_DIR" ]; then
    # 复制CLIP工具
    if [ -f "$TACO_DIR/util/clip_utils.py" ]; then
        cp "$TACO_DIR/util/clip_utils.py" "$PROJECT_DIR/util/"
    else
        echo "Warning: $TACO_DIR/util/clip_utils.py not found"
    fi
    
    # 复制TCL/HAD参考代码
    if [ -f "$TACO_DIR/models/tcl.py" ]; then
        cp "$TACO_DIR/models/tcl.py" "$PROJECT_DIR/models/ref_tcl.py"
    else
        echo "Warning: $TACO_DIR/models/tcl.py not found"
    fi
    if [ -f "$TACO_DIR/models/had.py" ]; then
        cp "$TACO_DIR/models/had.py" "$PROJECT_DIR/models/ref_had.py"
    else
        echo "Warning: $TACO_DIR/models/had.py not found"
    fi
    
    echo "TACO modules copied!"
else
    echo "Warning: TACO directory not found at $TACO_DIR"
fi

# -----------------------------------------------------------------------------
# 5. 下载基线代码
# -----------------------------------------------------------------------------
echo ""
echo "[5/7] Downloading baseline code..."

cd "$PROJECT_DIR/baselines"

# CoTracker3
if [ ! -d "cotracker/.git" ]; then
    echo "Cloning CoTracker3..."
    git clone --depth 1 https://github.com/facebookresearch/co-tracker.git cotracker 2>/dev/null || true
fi

# LocoTrack
if [ ! -d "locotrack/.git" ]; then
    echo "Cloning LocoTrack..."
    git clone --depth 1 https://github.com/cvlab-kaist/locotrack.git locotrack 2>/dev/null || true
fi

# TAPIR
if [ ! -d "tapir/.git" ]; then
    echo "Cloning TAPIR..."
    git clone --depth 1 https://github.com/deepmind/tapnet.git tapir 2>/dev/null || true
fi

echo "Baseline code downloaded!"

# -----------------------------------------------------------------------------
# 6. 创建初始模型文件
# -----------------------------------------------------------------------------
echo ""
echo "[6/7] Creating initial model files..."

cd "$PROJECT_DIR"

# 创建 models/__init__.py（仅当不存在时）
if [ ! -f "models/__init__.py" ]; then
cat > models/__init__.py << 'EOF'
"""
FSPT Models
"""
# 尝试导入复用的模块
try:
    from .learnable_freq_decomposition import (
        LearnableFrequencyDecomposition,
        LearnableFrequencyFilter
    )
    print("✓ LearnableFrequencyDecomposition loaded")
except ImportError as e:
    print(f"✗ LearnableFrequencyDecomposition not available: {e}")

try:
    from .freq_temporal_transformer import (
        FrequencyAwarePositionalEncoding,
        BandSpecificTemporalAttention
    )
    print("✓ FrequencyTemporalTransformer loaded")
except ImportError as e:
    print(f"✗ FrequencyTemporalTransformer not available: {e}")
EOF
fi

# 创建 util/__init__.py（仅当不存在时）
if [ ! -f "util/__init__.py" ]; then
cat > util/__init__.py << 'EOF'
"""
FSPT Utilities
"""
EOF
fi

echo "Initial model files created!"

# -----------------------------------------------------------------------------
# 7. 验证安装
# -----------------------------------------------------------------------------
echo ""
echo "[7/7] Verifying installation..."

cd "$PROJECT_DIR"

python << 'EOF'
import sys
print("=" * 50)
print("FSPT Installation Verification")
print("=" * 50)

errors = []

# 检查核心库
print("\n[Core Libraries]")
for lib in ["torch", "torchvision", "numpy", "cv2", "einops", "timm"]:
    try:
        __import__(lib)
        print(f"  ✓ {lib}")
    except ImportError:
        print(f"  ✗ {lib}")
        errors.append(lib)

# 检查CLIP
print("\n[Vision-Language]")
try:
    import clip
    print("  ✓ CLIP")
except ImportError:
    print("  ✗ CLIP")
    errors.append("clip")

try:
    import open_clip
    print("  ✓ OpenCLIP")
except ImportError:
    print("  ✗ OpenCLIP")
    errors.append("open_clip")

# 检查CUDA
print("\n[Hardware]")
import torch
if torch.cuda.is_available():
    print(f"  ✓ CUDA: {torch.cuda.get_device_name(0)}")
    print(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  ✗ CUDA not available")
    errors.append("cuda")

# 检查复用模块
print("\n[Reused Modules]")
import os
models_dir = "models"
for module in ["learnable_freq_decomposition.py", "freq_temporal_transformer.py"]:
    if os.path.exists(os.path.join(models_dir, module)):
        print(f"  ✓ {module}")
    else:
        print(f"  ✗ {module} (not found)")

# 总结
print("\n" + "=" * 50)
if errors:
    print(f"Some issues found: {errors}")
    print("Please fix these before proceeding.")
else:
    print("All checks passed! FSPT is ready.")
print("=" * 50)

sys.exit(0 if not errors else 1)
EOF

# -----------------------------------------------------------------------------
# 完成
# -----------------------------------------------------------------------------
echo ""
echo "============================================"
echo "FSPT Initialization Complete!"
echo "============================================"
echo ""
echo "Project location: $PROJECT_DIR"
echo "Conda environment: $ENV_NAME"
echo ""
echo "Next steps:"
echo "1. Activate environment: conda activate $ENV_NAME"
echo "2. Download datasets: bash scripts/download_datasets.sh"
echo "3. Download model weights: bash scripts/download_baselines.sh"
echo "4. Start experimenting!"
echo ""
echo "Quick test:"
echo "  cd $PROJECT_DIR"
echo "  python -c 'from models import *'"
echo ""
