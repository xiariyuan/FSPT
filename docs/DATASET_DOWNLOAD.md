# FSPT 数据集下载指南

## 目录
1. [TAP-Vid-DAVIS (必需，约500MB)](#1-tap-vid-davis)
2. [TAP-Vid-Kubric / MOVi-E (必需，约150GB)](#2-tap-vid-kubric--movi-e)
3. [TAP-Vid-Kinetics (可选，约50GB)](#3-tap-vid-kinetics)
4. [PointOdyssey (可选，约185GB)](#4-pointodyssey)
5. [中国镜像加速方案](#5-中国镜像加速方案)

---

> 建议在所有命令前先设置路径变量，避免硬编码：
> ```bash
> PROJECT_DIR="${PROJECT_DIR:-$HOME/FSPT}"
> DATA_DIR="${DATA_DIR:-$PROJECT_DIR/datasets}"
> ```

> 跨平台一键下载（推荐）：
> ```bash
> cd "$PROJECT_DIR"
> python scripts/download_all_datasets.py --root "$DATA_DIR"
> # 可选：下载Kubric TFDS调试数据（不含点标注）
> # python scripts/download_all_datasets.py --root "$DATA_DIR" --kubric-tfds
> ```

## 1. TAP-Vid-DAVIS

**大小**: ~500 MB (pickle文件)  
**用途**: 验证集/测试集，30个真实视频

### 官方下载链接
```
https://storage.googleapis.com/dm-tapnet/tapvid_davis.pkl
```

### 服务器下载命令
```bash
# 推荐设置项目目录
PROJECT_DIR="${PROJECT_DIR:-$HOME/FSPT}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/datasets}"

# 创建数据目录
mkdir -p "$DATA_DIR/tapvid_davis"

# 下载 (需要科学上网或代理)
wget -O "$DATA_DIR/tapvid_davis/tapvid_davis.pkl" \
    https://storage.googleapis.com/dm-tapnet/tapvid_davis.pkl

# 或使用curl
curl -o "$DATA_DIR/tapvid_davis/tapvid_davis.pkl" \
    https://storage.googleapis.com/dm-tapnet/tapvid_davis.pkl

# 如果需要代理
export https_proxy=http://your_proxy:port
wget -O "$DATA_DIR/tapvid_davis/tapvid_davis.pkl" \
    https://storage.googleapis.com/dm-tapnet/tapvid_davis.pkl
```

### 验证下载
```bash
# 检查文件大小 (应约500MB)
ls -lh "$DATA_DIR/tapvid_davis/tapvid_davis.pkl"

# 验证pickle文件完整性
python -c "import pickle; d=pickle.load(open('${DATA_DIR}/tapvid_davis/tapvid_davis.pkl','rb')); print(f'Videos: {len(d)}')"
# 应输出: Videos: 30
```

---

## 2. TAP-Vid-Kubric / MOVi-E

**大小**: ~100-150 GB (TFRecord格式)  
**用途**: 训练集，9750个合成视频 + 250个验证视频

### 方法一：直接从GCS加载 (推荐，无需下载)

```python
import tensorflow_datasets as tfds

# 直接从Google Cloud Storage流式加载
# 无需下载到本地，但需要网络访问GCS
train_ds = tfds.load(
    "movi_e/256x256",
    data_dir="gs://kubric-public/tfds",
    split="train"
)

val_ds = tfds.load(
    "movi_e/256x256", 
    data_dir="gs://kubric-public/tfds",
    split="validation"
)
```

### 方法二：下载到本地

```bash
# 1. 安装依赖
pip install tensorflow tensorflow-datasets google-cloud-storage

# 2. 创建目录
mkdir -p "$DATA_DIR/tapvid_kubric"

# 3. 使用gsutil下载 (需要安装Google Cloud SDK)
# 安装gsutil: https://cloud.google.com/storage/docs/gsutil_install

# 下载256x256版本（推荐，与代码配置一致）
gsutil -m cp -r gs://kubric-public/tfds/movi_e/256x256 "$DATA_DIR/tapvid_kubric/movi_e/"
```

> 注意：TFDS 不包含点追踪标注。若仅用于调试，可设置 `data.train.use_tfds=true` 且 `allow_synthetic_tracks=true`；
> 正式训练请提供官方 Kubric pickle 标注（如 `tapvid_kubric_train.pkl`）。

### 方法三：Python脚本下载

```python
import tensorflow_datasets as tfds
import os

# 设置本地数据目录
data_dir = os.path.join(os.environ.get("DATA_DIR", "datasets"), "tapvid_kubric")
os.makedirs(data_dir, exist_ok=True)

# 下载并保存到本地
# 这会自动从GCS下载数据
ds = tfds.load(
    "movi_e/256x256",
    data_dir=data_dir,
    download=True,
    split="train"
)

print(f"Dataset downloaded to: {data_dir}")
```

### GCS直连地址
```
gs://kubric-public/tfds/movi_e/256x256/
```

---

## 3. TAP-Vid-Kinetics

**大小**: ~50 GB (需自行下载YouTube视频)  
**用途**: 测试集，1189个真实视频

### 步骤1：下载标注文件（pkl）

```bash
mkdir -p "$DATA_DIR/tapvid_kinetics"

# 下载并解压pkl标注
wget -O "$DATA_DIR/tapvid_kinetics/tapvid_kinetics.zip" \
    https://storage.googleapis.com/dm-tapnet/tapvid_kinetics.zip
cd "$DATA_DIR/tapvid_kinetics"
unzip tapvid_kinetics.zip
rm tapvid_kinetics.zip
```

### 步骤2：下载Kinetics视频（如pkl不包含视频）

需要先获取Kinetics-700-2020验证集视频：

```bash
# 克隆官方下载工具
git clone https://github.com/cvdfoundation/kinetics-dataset.git
cd kinetics-dataset

# 下载验证集视频 (需要较长时间)
bash download.sh val 700-2020
```

### 步骤3：生成TAP-Vid格式（可选）

```bash
cd "$PROJECT_DIR"

# 克隆tapnet仓库获取处理脚本
git clone https://github.com/google-deepmind/tapnet.git

# 运行生成脚本
#
# 仅当你需要重新生成pkl时，先下载CSV标注：
# wget -O $DATA_DIR/tapvid_kinetics/tapvid_kinetics.csv \
#     https://storage.googleapis.com/dm-tapnet/tapvid_kinetics.csv
python tapnet/tapnet/tapvid/generate_tapvid.py \
    --csv_path=$DATA_DIR/tapvid_kinetics/tapvid_kinetics.csv \
    --output_base_path=$DATA_DIR/tapvid_kinetics/ \
    --video_root_path=/path/to/kinetics_videos/ \
    --alsologtostderr
```

---

## 4. PointOdyssey

**大小**: 184.72 GB (压缩), ~300 GB (解压后)  
**用途**: 扩展训练，159个长视频，~20K轨迹/视频

### 官方下载链接

| 文件 | 大小 | 链接 |
|------|------|------|
| sample.tar.gz | 3.32 GB | [Google Drive](https://drive.google.com/drive/folders/1W6wxsbKbTdtV8-2TwToqa_QgLqRY3ft0) |
| test.tar.gz | 26.5 GB | [Google Drive](https://drive.google.com/drive/folders/1W6wxsbKbTdtV8-2TwToqa_QgLqRY3ft0) |
| train.tar.gz.partaa | 34.4 GB | [HuggingFace](https://huggingface.co/datasets/aharley/pointodyssey/tree/main) |
| train.tar.gz.partab | 34.4 GB | [HuggingFace](https://huggingface.co/datasets/aharley/pointodyssey/tree/main) |
| train.tar.gz.partac | 34.4 GB | [HuggingFace](https://huggingface.co/datasets/aharley/pointodyssey/tree/main) |
| train.tar.gz.partad | 31.3 GB | [HuggingFace](https://huggingface.co/datasets/aharley/pointodyssey/tree/main) |
| val.tar.gz | 20.4 GB | [HuggingFace](https://huggingface.co/datasets/aharley/pointodyssey/tree/main) |

### 服务器下载命令 (使用HuggingFace)

```bash
# 创建目录
mkdir -p "$DATA_DIR/pointodyssey"
cd "$DATA_DIR/pointodyssey"

# 安装huggingface_hub
pip install huggingface_hub

# 方法1: 使用huggingface-cli下载整个数据集
huggingface-cli download --repo-type dataset aharley/pointodyssey --local-dir .

# 方法2: 使用Python API
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='aharley/pointodyssey',
    repo_type='dataset',
    local_dir=os.path.join(os.environ.get("DATA_DIR", "datasets"), "pointodyssey"),
    resume_download=True
)
"

# 解压训练集 (分卷压缩)
cat train.tar.gz.part* > train.tar.gz
tar -xzf train.tar.gz

# 解压测试集和验证集
tar -xzf test.tar.gz
tar -xzf val.tar.gz
```

---

## 5. 中国镜像加速方案

### 方案一：HuggingFace镜像 (hf-mirror.com) - 推荐

**适用于**: PointOdyssey等HuggingFace托管的数据集

```bash
# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 然后正常使用huggingface-cli
huggingface-cli download --repo-type dataset aharley/pointodyssey --local-dir ./pointodyssey
```

或者使用hfd工具：
```bash
# 下载hfd.sh
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh

# 下载数据集
./hfd.sh aharley/pointodyssey --dataset --tool aria2c -x 4
```

### 方案二：代理设置

```bash
# 设置HTTP代理
export http_proxy=http://your_proxy:port
export https_proxy=http://your_proxy:port

# 然后执行下载命令
wget https://storage.googleapis.com/dm-tapnet/tapvid_davis.pkl
```

### 方案三：使用aria2多线程加速

```bash
# 安装aria2
apt-get install aria2

# 多线程下载
aria2c -x 16 -s 16 -k 1M \
    https://storage.googleapis.com/dm-tapnet/tapvid_davis.pkl \
    -d ${DATA_DIR:-datasets}/tapvid_davis/ \
    -o tapvid_davis.pkl
```

### 方案四：学校/实验室网络

如果你的学校/实验室有教育网出口或科研网络：
- 可能已经有Google Cloud、AWS的直连
- 联系网络中心询问是否有数据集镜像服务
- 部分高校有CERNET海外加速服务

---

## 快速开始脚本

将以下脚本保存为 `download_datasets.sh` 并在服务器上运行：

```bash
#!/bin/bash

# FSPT数据集下载脚本
# 使用方法: bash download_datasets.sh [proxy_url]

PROXY=${1:-""}
BASE_DIR="${BASE_DIR:-datasets}"

# 设置代理(如果提供)
if [ -n "$PROXY" ]; then
    export http_proxy=$PROXY
    export https_proxy=$PROXY
    echo "Using proxy: $PROXY"
fi

# 1. 下载TAP-Vid-DAVIS
echo "=== Downloading TAP-Vid-DAVIS ==="
mkdir -p $BASE_DIR/tapvid_davis
wget -c -O $BASE_DIR/tapvid_davis/tapvid_davis.pkl \
    https://storage.googleapis.com/dm-tapnet/tapvid_davis.pkl

# 验证
python -c "import pickle; d=pickle.load(open('$BASE_DIR/tapvid_davis/tapvid_davis.pkl','rb')); print(f'DAVIS Videos: {len(d)}')"

# 2. 下载TAP-Vid-Kinetics标注
echo "=== Downloading TAP-Vid-Kinetics Annotations ==="
mkdir -p $BASE_DIR/tapvid_kinetics
wget -c -O $BASE_DIR/tapvid_kinetics/tapvid_kinetics.zip \
    https://storage.googleapis.com/dm-tapnet/tapvid_kinetics.zip
unzip -q $BASE_DIR/tapvid_kinetics/tapvid_kinetics.zip -d $BASE_DIR/tapvid_kinetics
rm $BASE_DIR/tapvid_kinetics/tapvid_kinetics.zip

# 3. PointOdyssey (使用HuggingFace镜像)
echo "=== Downloading PointOdyssey (optional) ==="
export HF_ENDPOINT=https://hf-mirror.com
pip install -q huggingface_hub
mkdir -p $BASE_DIR/pointodyssey

# 只下载sample用于测试
huggingface-cli download --repo-type dataset aharley/pointodyssey \
    --include "sample.tar.gz" \
    --local-dir $BASE_DIR/pointodyssey

echo "=== Download Complete ==="
echo "TAP-Vid-DAVIS: $BASE_DIR/tapvid_davis/"
echo "TAP-Vid-Kinetics annotations: $BASE_DIR/tapvid_kinetics/"
echo "PointOdyssey sample: $BASE_DIR/pointodyssey/"
echo ""
echo "Note: For Kubric (MOVi-E), use tfds.load() directly from GCS"
echo "      or run: gsutil -m cp -r gs://kubric-public/tfds/movi_e/256x256 $BASE_DIR/tapvid_kubric/movi_e/"
```

---

## 下载优先级建议

| 优先级 | 数据集 | 大小 | 用途 | 下载难度 |
|--------|--------|------|------|----------|
| 🔴 必须 | TAP-Vid-DAVIS | 500 MB | 验证/测试 | 简单 |
| 🔴 必须 | Kubric (MOVi-E) | 100-150 GB | 训练 | 中等 (GCS流式或gsutil) |
| 🟡 建议 | TAP-Vid-Kinetics | 50 GB | 真实场景测试 | 复杂 (需下载YouTube视频) |
| 🟢 可选 | PointOdyssey | 185 GB | 长时序训练 | 中等 (HuggingFace镜像) |

---

## 常见问题

### Q1: GCS连接超时？
```bash
# 使用代理
export https_proxy=http://proxy:port
# 或使用gsutil的代理设置
gsutil -o Boto:proxy=proxy_host -o Boto:proxy_port=port cp ...
```

### Q2: HuggingFace下载断开？
```bash
# 使用--resume-download参数自动续传
huggingface-cli download --resume-download ...
```

### Q3: 如何只下载部分数据测试？
```python
# Kubric只加载100个样本
ds = tfds.load("movi_e/256x256", split="train[:100]", ...)
```

### Q4: 磁盘空间不足？
优先下载DAVIS (500MB) 验证代码，Kubric可以使用GCS流式加载不占本地空间。
