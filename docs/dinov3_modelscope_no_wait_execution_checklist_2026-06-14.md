# DINOv3 ModelScope No-Wait Execution Checklist (2026-06-14)

## 0. 目标

这份清单只解决一个问题：

**不等待 Hugging Face 审批，直接用国内可访问源准备 DINOv3，并为 Track-On-R 提供本地 backbone。**

这不是最终的 Attempt 0 全流程文档。

这份清单的任务边界只有 4 件事：

1. 从 `ModelScope` 获取 `DINOv3`
2. 校验本地模型目录是否完整
3. 让 `track_on` 优先从本地目录加载 DINOv3
4. 如果本地 DINOv3 路线失败，立即回退到 `Track-On2 DINOv2`

---

## 1. 前提判断

当前已知事实：

1. `Hugging Face` 在当前机器上网络不可达，不能假设随时可用
2. `DINOv3` 官方 HF 模型属于 gated / license-controlled 路线
3. `ModelScope` 上存在同名模型：
   - `facebook/dinov3-vits16plus-pretrain-lvd1689m`
4. 当前机器已安装：
   - `modelscope`
   - `transformers`
   - `huggingface_hub`

已确认：

1. `ModelScope` 模型 API 可访问
2. 返回元数据中 `ApprovalMode = 0`
3. `modelscope.hub.snapshot_download.snapshot_download(...)` 可直接调用

---

## 2. 目标模型 ID

只允许使用这个精确模型 ID：

`facebook/dinov3-vits16plus-pretrain-lvd1689m`

不要替换成：

1. 社区重打包版
2. ONNX 版
3. 蒸馏版
4. 7B 版
5. 其他 DINOv3 变体

因为 `Track-On-R` 期望的是指定 backbone，不是“任意 DINOv3”。

---

## 3. 输出物

这条线结束时，必须交付：

1. `dinov3_modelscope_manifest.json`
2. `dinov3_local_layout_report.json`
3. `dinov3_load_smoke.json`
4. `trackon_local_backbone_patch_notes.md`
5. `fallback_decision.md`

如果只完成下载，但没有 load smoke，不算完成。

---

## 4. 推荐目录

统一放在：

`third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m/`

不要散落在：

1. 外部仓库目录
2. 临时下载目录
3. `/tmp`
4. 模糊命名的 cache 目录

理由：

1. 方便写 manifest
2. 方便后续 Claude / 人工复现
3. 避免路径丢失

---

## 5. Step-by-Step 执行

### Step 1. 建立权重目录

运行：

```bash
mkdir -p third_party_weights/dinov3
```

验收：

1. 目录存在
2. 磁盘空间充足

建议先检查：

```bash
df -h .
```

如果剩余空间明显不足，不要开始下载。

### Step 2. 用 ModelScope 下载模型

执行：

```bash
python - <<'PY'
from pathlib import Path
from shutil import copytree, rmtree
from modelscope.hub.snapshot_download import snapshot_download

target = Path("third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m").resolve()
target.parent.mkdir(parents=True, exist_ok=True)

cache_path = Path(snapshot_download("facebook/dinov3-vits16plus-pretrain-lvd1689m")).resolve()
print("snapshot_cache:", cache_path)

if target.exists():
    rmtree(target)
copytree(cache_path, target)
print("final_path:", target)
PY
```

验收：

1. 终端打印 `snapshot_cache`
2. 终端打印 `final_path`
3. `third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m/` 非空

如果失败：

1. 记录异常
2. 不要反复重试超过 3 次
3. 直接进入 `Step 11 fallback`

### Step 3. 记录下载 manifest

执行：

```bash
python - <<'PY'
import json
from pathlib import Path
from datetime import datetime
from modelscope.hub.api import HubApi

model_id = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
root = Path("third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m").resolve()
api = HubApi()
info = api.get_model(model_id)

manifest = {
    "model_id": model_id,
    "download_source": "modelscope",
    "download_time": datetime.now().isoformat(timespec="seconds"),
    "local_path": str(root),
    "approval_mode": info.get("ApprovalMode"),
    "frameworks": info.get("Frameworks"),
    "architectures": info.get("Architectures"),
    "downloads": info.get("Downloads"),
}

out = Path("outputs/dinov3_modelscope_manifest.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
print(out)
PY
```

验收：

1. 生成 `outputs/dinov3_modelscope_manifest.json`
2. 内容至少包含：
   - `model_id`
   - `download_source`
   - `local_path`
   - `approval_mode`

### Step 4. 检查目录布局

执行：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m").resolve()

expected_candidates = [
    "config.json",
    "configuration_dinov3.py",
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "preprocessor_config.json",
]

present = {}
for name in expected_candidates:
    present[name] = (root / name).exists()

all_files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())

report = {
    "root": str(root),
    "present": present,
    "num_files": len(all_files),
    "sample_files": all_files[:100],
}

out = Path("outputs/dinov3_local_layout_report.json")
out.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
print(out)
PY
```

验收：

1. 生成 `outputs/dinov3_local_layout_report.json`
2. 至少要有：
   - `config.json`
   - 一份权重文件
3. `num_files > 0`

如果缺少 `config.json` 或权重文件：

1. 不要继续接 `track_on`
2. 直接进入 `Step 11 fallback`

### Step 5. 做 transformers 本地加载 smoke

执行：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m").resolve()
out = Path("outputs/dinov3_load_smoke.json")

report = {
    "local_path": str(root),
    "config_ok": False,
    "model_ok": False,
    "error": "",
}

try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(str(root), trust_remote_code=True, local_files_only=True)
    report["config_ok"] = True
    report["config_class"] = cfg.__class__.__name__
except Exception as e:
    report["error"] = f"config_load_failed: {repr(e)}"
    out.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(out)
    raise

try:
    from transformers import AutoModel
    model = AutoModel.from_pretrained(str(root), trust_remote_code=True, local_files_only=True)
    report["model_ok"] = True
    report["model_class"] = model.__class__.__name__
except Exception as e:
    report["error"] = f"model_load_failed: {repr(e)}"
    out.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(out)
    raise

out.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
print(out)
PY
```

验收：

1. 生成 `outputs/dinov3_load_smoke.json`
2. `config_ok = true`
3. `model_ok = true`

如果这里只能 load config，不能 load model：

1. 说明目录可能不完整，或 `transformers` / remote code 不兼容
2. 先记录错误
3. 只允许再做一次针对性修复
4. 修不通就 fallback

### Step 6. 冻结一个固定本地路径变量

后续所有 Track-On-R 相关工作，只认这一个环境变量：

```bash
export DINOV3_LOCAL_DIR=/gemini/code/FSPT/third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m
```

同时把这行写入一个记录文件：

`outputs/dinov3_env_hint.txt`

内容至少包含：

```bash
export DINOV3_LOCAL_DIR=/gemini/code/FSPT/third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m
```

### Step 7. 检查 track_on 的 DINOv3 加载入口

目标：

1. 找到 DINOv3 的 `from_pretrained(...)` 或等价入口
2. 确认是否硬编码为 HF repo id

执行原则：

1. 不要先乱改代码
2. 先 `rg` 搜索：
   - `dinov3`
   - `from_pretrained`
   - `facebook/dinov3`
   - `AutoModel`

目标是明确：

1. 哪个文件负责 backbone 构建
2. 哪个参数可注入本地路径
3. 是否能不改代码，仅通过 config 覆盖

### Step 8. 优先尝试“不改代码”的本地路径注入

如果 `track_on` 支持通过 config / args 指定 backbone path：

1. 直接把值改为：
   - `$DINOV3_LOCAL_DIR`
2. 不要走 HF repo id

验收：

1. 最小 import smoke 可过
2. backbone 构建成功
3. 无 HF 网络访问

### Step 9. 如果必须改代码，只做最小 patch

只有在 `track_on` 没有现成本地路径参数时，才允许做 patch。

Patch 原则：

1. 优先读取环境变量 `DINOV3_LOCAL_DIR`
2. 若环境变量存在，则：
   - `from_pretrained(DINOV3_LOCAL_DIR, trust_remote_code=True, local_files_only=True)`
3. 若环境变量不存在，再走原始 repo id

伪代码应接近：

```python
dinov3_source = os.environ.get("DINOV3_LOCAL_DIR", "facebook/dinov3-vits16plus-pretrain-lvd1689m")
model = AutoModel.from_pretrained(
    dinov3_source,
    trust_remote_code=True,
    local_files_only=os.path.isdir(dinov3_source),
)
```

要求：

1. 只改 DINOv3 加载入口
2. 不连带改训练逻辑
3. 写清 patch notes

### Step 10. 做 Track-On-R 本地 backbone smoke

最低 smoke 标准：

1. 仅构建模型，不跑全量评测
2. 成功 import
3. 成功初始化 backbone
4. 无 HF 访问错误

必须产出：

1. `outputs/trackon_local_backbone_patch_notes.md`
2. `outputs/trackon_local_backbone_smoke.txt`

如果这一步通过，才允许继续 Track-On-R repo-native eval。

---

## 6. Timebox 与 Kill Criteria

这条“ModelScope DINOv3 本地化”路线最多只给：

**0.5 - 1 天**

触发 fallback 的条件：

1. `snapshot_download` 下载失败且无法在 3 次内修复
2. 本地目录缺关键文件
3. `AutoConfig.from_pretrained(...)` 失败
4. `AutoModel.from_pretrained(...)` 失败
5. `track_on` backbone patch 后仍强依赖 HF
6. 修 patch 超过半天还没跑通

一旦触发其中任意一条：

1. 停止继续耗时
2. 写 `outputs/fallback_decision.md`
3. 立即切 `Track-On2 DINOv2`

---

## 7. 明确 fallback 路线

如果本地 DINOv3 路线失败，立即执行：

1. 切 `Track-On2 DINOv2`
2. 完成：
   - env smoke
   - DAVIS repo-native eval
   - adapter sanity
   - unified rescoring
3. 并行继续 TAPNext++

不要出现“DINOv3 没好，整周空转”。

---

## 8. Claude 的执行纪律

Claude 在执行这份清单时必须遵守：

1. 先下载，再 smoke，再 patch，再 smoke
2. 每一步都要有工件输出
3. 不允许连续几个小时只在网络 / 权限问题上打转
4. 一旦触发 fallback，就立刻转 `Track-On2 DINOv2`
5. 不要因为 DINOv3 卡住 Attempt 0 主线

---

## 9. 最低可执行命令摘要

### 下载

```bash
python - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download
print(snapshot_download("facebook/dinov3-vits16plus-pretrain-lvd1689m"))
PY
```

### 本地 config smoke

```bash
python - <<'PY'
from transformers import AutoConfig
AutoConfig.from_pretrained(
    "third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m",
    trust_remote_code=True,
    local_files_only=True,
)
print("config_ok")
PY
```

### 本地 model smoke

```bash
python - <<'PY'
from transformers import AutoModel
AutoModel.from_pretrained(
    "third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m",
    trust_remote_code=True,
    local_files_only=True,
)
print("model_ok")
PY
```
