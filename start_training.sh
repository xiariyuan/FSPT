#!/bin/bash
# FSPT 训练启动脚本

# 激活conda环境
source /root/miniconda3/bin/activate

# 进入项目目录
cd /gemini/code/FSPT

# 启动训练
python train.py --config configs/fspt_server.yaml "$@"
