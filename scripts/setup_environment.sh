#!/bin/bash
# =============================================================================
# FSPT - 环境配置脚本
# 在远程服务器上运行此脚本
# =============================================================================

set -euo pipefail

# 项目目录（默认使用项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${1:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"
ENV_NAME="${2:-fspt}"

echo "============================================"
echo "FSPT Environment Setup Script"
echo "Project directory: $PROJECT_DIR"
echo "Conda environment: $ENV_NAME"
echo "============================================"

# -----------------------------------------------------------------------------
# 1. 创建Conda环境
# -----------------------------------------------------------------------------
echo ""
echo "[1/6] Creating Conda environment..."

# 检查conda是否可用
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 创建环境
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment $ENV_NAME already exists."
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n $ENV_NAME
        conda create -n $ENV_NAME python=3.10 -y
    fi
else
    conda create -n $ENV_NAME python=3.10 -y
fi

# 激活环境
CONDA_BASE="$(conda info --base)"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "Failed to locate conda.sh under $CONDA_BASE"
    exit 1
fi
conda activate $ENV_NAME

# -----------------------------------------------------------------------------
# 2. 安装PyTorch
# -----------------------------------------------------------------------------
echo ""
echo "[2/6] Installing PyTorch with CUDA..."

# PyTorch 2.2 with CUDA 12.1
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# 验证CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# -----------------------------------------------------------------------------
# 3. 安装核心依赖
# -----------------------------------------------------------------------------
echo ""
echo "[3/6] Installing core dependencies..."

if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip install -r "$PROJECT_DIR/requirements.txt"
else
    echo "Warning: requirements.txt not found at $PROJECT_DIR/requirements.txt"
    echo "Installing minimal dependencies..."
    pip install numpy scipy pillow opencv-python einops timm transformers omegaconf hydra-core pyyaml tqdm matplotlib wandb tensorboard
fi

# -----------------------------------------------------------------------------
# 4. 安装CLIP
# -----------------------------------------------------------------------------
echo ""
echo "[4/6] Installing CLIP..."

pip install git+https://github.com/openai/CLIP.git
pip install open-clip-torch

# 验证CLIP
python -c "import clip; print('CLIP installed successfully')"

# -----------------------------------------------------------------------------
# 5. 安装Deformable Attention (可选，用于高效注意力)
# -----------------------------------------------------------------------------
echo ""
echo "[5/6] Installing Deformable Attention..."

cd "$PROJECT_DIR"
if [ ! -d "third_party/deformable_attention" ]; then
    mkdir -p third_party
    cd third_party
    git clone https://github.com/fundamentalvision/Deformable-DETR.git deformable_attention
    cd deformable_attention/models/ops
    python setup.py build install
    cd "$PROJECT_DIR"
fi

# -----------------------------------------------------------------------------
# 6. 安装评估工具
# -----------------------------------------------------------------------------
echo ""
echo "[6/6] Installing evaluation tools..."

# TAP-Vid评估代码
pip install mediapy
pip install tensorflow tensorflow-datasets  # 用于Kubric数据加载

# -----------------------------------------------------------------------------
# 总结
# -----------------------------------------------------------------------------
echo ""
echo "============================================"
echo "Environment Setup Complete!"
echo "============================================"
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To verify the setup:"
echo "  python $PROJECT_DIR/verify_project.py"
echo ""
echo "Next steps:"
echo "1. Download baseline code: bash scripts/download_baselines.sh"
echo "2. Download datasets: bash scripts/download_datasets.sh"
echo "3. Start experimenting!"
