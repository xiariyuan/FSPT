# FSPT 代码复用规划

## 一、从 FM-Track 复用的模块

### 源代码位置
```
<server>:/path/to/FMtrack-main/FM-Track/models/motip/
```

### 1.1 可学习频率分解模块 (核心复用)

**源文件**: `learnable_freq_decomposition.py`

**复用内容**:
| 类/函数 | 行数 | 复用方式 | 修改点 |
|---------|------|----------|--------|
| `LearnableFrequencyFilter` | ~200行 | 直接复用 | 修改输入维度适配点追踪 |
| `LearnableFrequencyDecomposition` | ~150行 | 直接复用 | 添加点级别mask支持 |
| `compute_filter_frequency_orthogonality` | ~50行 | 直接复用 | 无需修改 |
| `compute_feature_orthogonality` | ~50行 | 直接复用 | 无需修改 |

**适配修改**:
```python
# 原始接口 (FM-Track，用于MOT)
# 输入: (B, G, T, N, C) - Batch, Group, Time, NumObjects, Channels

# 新接口 (FSPT，用于点追踪)
# 输入: (B, T, N, C) - Batch, Time, NumPoints, Channels

# 修改forward函数签名
def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
    """
    Args:
        x: (B, T, N, C) 点特征序列
        mask: (B, T, N) 可选的点级别mask
    """
    # 添加虚拟的Group维度以复用原有逻辑
    x = x.unsqueeze(1)  # (B, 1, T, N, C)
    output, info = self._original_forward(x, mask)
    output = output.squeeze(1)  # (B, T, N, C)
    return output, info
```

### 1.2 频率感知时序Transformer

**源文件**: `freq_temporal_transformer.py`

**复用内容**:
| 类/函数 | 行数 | 复用方式 | 修改点 |
|---------|------|----------|--------|
| `FrequencyAwarePositionalEncoding` | ~80行 | 直接复用 | 无需修改 |
| `BandSpecificTemporalAttention` | ~150行 | 直接复用 | 适配点追踪的序列长度 |

**关键设计复用**:
```python
# 不同频带使用不同的注意力窗口
# 低频带(idx=0): 使用大窗口 (30帧) - 捕获长程运动趋势
# 高频带(idx=3): 使用小窗口 (5帧) - 捕获精细位移

window_ratio = 1.0 - (band_idx / max(num_bands - 1, 1)) * 0.7  # [0.3, 1.0]
window_size = max(3, int(max_window_size * window_ratio))
```

### 1.3 频率引导关联

**源文件**: `freq_guided_association.py`

**复用内容**:
| 类/函数 | 行数 | 复用方式 | 修改点 |
|---------|------|----------|--------|
| 频率特征融合逻辑 | ~100行 | 参考设计 | 重写适配点匹配 |
| 频带重要性计算 | ~50行 | 直接复用 | 无需修改 |

---

## 二、从 TACO 复用的模块

### 源代码位置
```
<server>:/path/to/taco/
```

### 2.1 CLIP特征提取

**源文件**: `ovtr/models/ovtr.py`

**复用内容**:
| 类/函数 | 行数 | 复用方式 | 修改点 |
|---------|------|----------|--------|
| CLIP embedding加载 | ~50行 | 直接复用 | 适配点级别特征 |
| `patch2query` | ~20行 | 参考设计 | 修改投影维度 |
| 文本/图像embedding处理 | ~100行 | 参考设计 | 简化为纯图像特征 |

**关键代码片段**:
```python
# 从OVTR复用的CLIP特征加载
from util.clip_utils import load_embeddings

# 加载预训练CLIP图像编码器
self.text_embeddings = text_embeddings.t()  # 文本特征
self.image_embeddings = image_embeddings.t()  # 图像特征

# 投影到追踪特征空间
self.patch2query = nn.Linear(512, 256)  # CLIP dim -> tracking dim
```

### 2.2 时序对比学习 (TCL)

**源文件**: `models/tcl.py`

**复用内容**:
| 类/函数 | 行数 | 复用方式 | 修改点 |
|---------|------|----------|--------|
| `TCL` 类 | ~300行 | 参考设计 | 适配点追踪任务 |
| 负样本挖掘策略 | ~100行 | 直接复用 | 修改为点级别负样本 |
| 时序衰减机制 | ~50行 | 直接复用 | 无需修改 |

**参考设计**:
```python
# TCL的核心思想：同一轨迹的点在时间上应该相似
# 不同轨迹的点应该区分

# 正样本：同一点在不同时间帧的特征
# 负样本：不同点在任意时间帧的特征

# 可以用于点追踪的训练正则化
class PointTemporalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        self.temperature = temperature
        
    def forward(self, point_features, point_ids):
        """
        point_features: (B, T, N, C)
        point_ids: (B, N) 点ID
        """
        # 同一点ID的特征应该相似
        # 不同点ID的特征应该区分
        pass
```

### 2.3 难度感知蒸馏 (HAD)

**源文件**: `models/had.py`

**复用内容**:
| 类/函数 | 行数 | 复用方式 | 修改点 |
|---------|------|----------|--------|
| 难度评估逻辑 | ~100行 | 参考设计 | 定义点追踪的"难度" |
| 自适应权重计算 | ~50行 | 直接复用 | 无需修改 |
| `CurriculumScheduler` | ~80行 | 直接复用 | 无需修改 |

**适配点追踪**:
```python
# HAD的难度评估因子 (MOT版本)
# - 置信度 (confidence)
# - 熵 (entropy)
# - 遮挡程度 (occlusion)

# 点追踪的难度因子
# - 运动幅度 (快速运动更难)
# - 遮挡时长 (长期遮挡更难)
# - 语义模糊度 (低语义区域更难)
# - 重复纹理 (匹配歧义更难)

class PointDifficultyEstimator(nn.Module):
    def __init__(self, alpha=0.3, beta=0.3, gamma=0.2, delta=0.2):
        self.alpha = alpha  # 运动因子
        self.beta = beta    # 遮挡因子
        self.gamma = gamma  # 语义因子
        self.delta = delta  # 纹理因子
        
    def forward(self, motion, occlusion, semantic_clarity, texture_uniqueness):
        difficulty = (
            self.alpha * motion.norm(dim=-1) +
            self.beta * occlusion.float() +
            self.gamma * (1 - semantic_clarity) +
            self.delta * (1 - texture_uniqueness)
        )
        return difficulty
```

---

## 三、需要新开发的模块

### 3.1 点追踪骨干网络

**文件**: `models/backbone.py`

```python
class PointTrackingBackbone(nn.Module):
    """
    点追踪骨干网络
    
    结合:
    1. CNN特征提取 (用于几何特征)
    2. CLIP特征提取 (用于语义特征, 冻结)
    """
    def __init__(self, cnn_type='resnet50', clip_model='ViT-B/16'):
        super().__init__()
        
        # 几何特征提取
        self.cnn = timm.create_model(cnn_type, pretrained=True, features_only=True)
        
        # 语义特征提取 (冻结)
        self.clip, _ = clip.load(clip_model)
        for param in self.clip.parameters():
            param.requires_grad = False
            
    def forward(self, images):
        """
        images: (B, T, C, H, W)
        
        Returns:
            geo_features: (B, T, H', W', C_geo)
            sem_features: (B, T, H', W', C_sem)
        """
        B, T = images.shape[:2]
        images_flat = images.reshape(B * T, *images.shape[2:])
        
        # 几何特征
        geo_features = self.cnn(images_flat)[-1]
        
        # 语义特征
        with torch.no_grad():
            sem_features = self.clip.encode_image(images_flat)
            
        return geo_features, sem_features
```

### 3.2 语义增强频率分解

**文件**: `models/freq_semantic_fusion.py`

```python
class SemanticEnhancedLFD(nn.Module):
    """
    语义增强的可学习频率分解
    
    核心创新：语义特征调制频率响应
    """
    def __init__(self, geo_dim=256, sem_dim=512, num_bands=4):
        super().__init__()
        
        # 复用FM-Track的频率分解
        self.lfd = LearnableFrequencyDecomposition(
            dim=geo_dim, num_bands=num_bands
        )
        
        # 语义调制网络
        self.semantic_modulator = nn.Sequential(
            nn.Linear(sem_dim, geo_dim * 2),
            nn.GELU(),
            nn.Linear(geo_dim * 2, geo_dim * num_bands),
            nn.Sigmoid()
        )
        
        # 跨模态融合
        self.cross_attention = nn.MultiheadAttention(geo_dim, num_heads=8)
        
    def forward(self, geo_feat, sem_feat):
        """
        geo_feat: (B, T, N, C_geo) 几何特征
        sem_feat: (B, T, N, C_sem) 语义特征
        """
        # 频率分解
        freq_output, band_features = self.lfd(geo_feat)
        
        # 语义调制
        modulation = self.semantic_modulator(sem_feat)
        
        # 对每个频带应用调制
        modulated_bands = {}
        for i in range(self.num_bands):
            band_mod = modulation[..., i*self.geo_dim:(i+1)*self.geo_dim]
            modulated_bands[f'band_{i}'] = band_features[f'band_{i}'] * band_mod
            
        return freq_output, modulated_bands
```

### 3.3 频率感知遮挡预测器

**文件**: `models/occlusion_predictor.py`

```python
class FrequencyAwareOcclusionPredictor(nn.Module):
    """
    频率感知的遮挡预测器
    
    核心思想：
    1. 高频异常检测遮挡
    2. 低频轨迹外推预测遮挡期间位置
    3. 语义一致性传播遮挡信息
    """
    def __init__(self, dim=256, num_bands=4):
        super().__init__()
        
        # 遮挡检测器
        self.occlusion_detector = nn.Sequential(
            nn.Linear(dim * num_bands, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 低频轨迹外推器
        self.trajectory_predictor = nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.predictor_proj = nn.Linear(dim * 2, 2)  # 预测位移
        
        # 语义一致性约束
        self.semantic_similarity = nn.CosineSimilarity(dim=-1)
        
    def forward(self, band_features, semantic_feat, positions):
        """
        band_features: dict of (B, T, N, C) per band
        semantic_feat: (B, T, N, C_sem)
        positions: (B, T, N, 2) 当前位置
        
        Returns:
            occlusion_prob: (B, T, N, 1)
            predicted_positions: (B, T, N, 2)
        """
        B, T, N, C = band_features['band_0'].shape
        
        # 合并所有频带
        all_bands = torch.cat([band_features[f'band_{i}'] 
                               for i in range(len(band_features))], dim=-1)
        
        # 遮挡检测
        occlusion_prob = self.occlusion_detector(all_bands)
        
        # 低频轨迹外推
        low_freq = band_features['band_0']
        low_freq_flat = low_freq.reshape(B * N, T, C)
        motion_hidden, _ = self.trajectory_predictor(low_freq_flat)
        predicted_motion = self.predictor_proj(motion_hidden)
        predicted_motion = predicted_motion.reshape(B, N, T, 2).permute(0, 2, 1, 3)
        
        predicted_positions = positions + predicted_motion
        
        # 语义一致性传播
        # 计算语义相似度矩阵
        sem_flat = semantic_feat.reshape(B * T, N, -1)
        sem_sim = self.semantic_similarity(
            sem_flat.unsqueeze(2),  # (B*T, N, 1, C)
            sem_flat.unsqueeze(1)   # (B*T, 1, N, C)
        )  # (B*T, N, N)
        
        # 使用语义相似度传播遮挡信息
        occ_flat = occlusion_prob.reshape(B * T, N, 1)
        occ_propagated = torch.bmm(sem_sim.softmax(dim=-1), occ_flat)
        occlusion_refined = occ_propagated.reshape(B, T, N, 1)
        
        return occlusion_refined, predicted_positions
```

### 3.4 主追踪器

**文件**: `models/point_tracker.py`

```python
class FSPTTracker(nn.Module):
    """
    FSPT: Frequency-Semantic Point Tracker
    
    完整的点追踪模型
    """
    def __init__(self, config):
        super().__init__()
        
        # 骨干网络
        self.backbone = PointTrackingBackbone(
            cnn_type=config.backbone,
            clip_model=config.clip_model
        )
        
        # 语义增强频率分解
        self.freq_semantic = SemanticEnhancedLFD(
            geo_dim=config.geo_dim,
            sem_dim=config.sem_dim,
            num_bands=config.num_bands
        )
        
        # 频率感知时序建模
        self.temporal_transformer = FrequencyTemporalTransformer(
            dim=config.geo_dim,
            num_bands=config.num_bands,
            num_layers=config.num_layers
        )
        
        # 遮挡预测器
        self.occlusion_predictor = FrequencyAwareOcclusionPredictor(
            dim=config.geo_dim,
            num_bands=config.num_bands
        )
        
        # 位置预测头
        self.position_head = nn.Sequential(
            nn.Linear(config.geo_dim, config.geo_dim),
            nn.GELU(),
            nn.Linear(config.geo_dim, 2)  # 预测 (dx, dy)
        )
        
    def forward(self, video, query_points):
        """
        video: (B, T, C, H, W)
        query_points: (B, N, 3) - (t, x, y) 查询点
        
        Returns:
            tracks: (B, N, T, 2) 预测轨迹
            visibility: (B, N, T) 可见性
        """
        B, T, C, H, W = video.shape
        N = query_points.shape[1]
        
        # 1. 特征提取
        geo_feat, sem_feat = self.backbone(video)
        
        # 2. 采样查询点特征
        point_geo = self.sample_features(geo_feat, query_points)  # (B, T, N, C_geo)
        point_sem = self.sample_features(sem_feat, query_points)  # (B, T, N, C_sem)
        
        # 3. 语义增强频率分解
        freq_output, band_features = self.freq_semantic(point_geo, point_sem)
        
        # 4. 时序建模
        temporal_feat = self.temporal_transformer(freq_output, band_features)
        
        # 5. 遮挡预测
        # 初始位置
        positions = self.initialize_positions(query_points, T)  # (B, T, N, 2)
        visibility, predicted_positions = self.occlusion_predictor(
            band_features, point_sem, positions
        )
        
        # 6. 位置更新
        position_delta = self.position_head(temporal_feat)
        tracks = positions + position_delta
        
        # 对遮挡点使用预测位置
        tracks = torch.where(
            visibility < 0.5,
            predicted_positions,
            tracks
        )
        
        return tracks, visibility.squeeze(-1)
```

---

## 四、复用优先级

### 高优先级（必须复用）
1. ✅ `LearnableFrequencyDecomposition` - 核心创新模块
2. ✅ `FrequencyAwarePositionalEncoding` - 频率感知位置编码
3. ✅ CLIP特征加载逻辑

### 中优先级（推荐复用）
4. ⬜ `BandSpecificTemporalAttention` - 频带注意力
5. ⬜ TCL负样本挖掘策略
6. ⬜ HAD难度评估框架

### 低优先级（参考设计）
7. ⬜ 频率引导关联逻辑
8. ⬜ 课程学习调度器

---

## 五、代码迁移步骤

### Step 1: 复制核心模块
```bash
# 在服务器上执行
PROJECT_DIR="${PROJECT_DIR:-$HOME/FSPT}"
FMTRACK_DIR="${FMTRACK_DIR:-/path/to/FMtrack-main/FM-Track}"
cd "$PROJECT_DIR/models"

# 复制FM-Track模块
cp "$FMTRACK_DIR/models/motip/learnable_freq_decomposition.py" ./
cp "$FMTRACK_DIR/models/motip/freq_temporal_transformer.py" ./

# 复制工具函数
cp "$FMTRACK_DIR/util/box_ops.py" "$PROJECT_DIR/util/"
```

### Step 2: 修改导入路径
```python
# 修改 learnable_freq_decomposition.py
# 原始:
# from util.misc import ...

# 修改为:
# from ..util.misc import ...
```

### Step 3: 适配接口
```python
# 创建适配器包装
class LFDAdapter(LearnableFrequencyDecomposition):
    def forward(self, x, mask=None):
        # 点追踪输入: (B, T, N, C)
        # FM-Track输入: (B, G, T, N, C)
        x = x.unsqueeze(1)
        output, info = super().forward(x, mask)
        output = output.squeeze(1)
        return output, info
```

### Step 4: 验证复用
```python
# 测试脚本
def test_lfd_adapter():
    adapter = LFDAdapter(dim=256, num_bands=4)
    x = torch.randn(2, 100, 256, 256)  # (B, T, N, C)
    output, info = adapter(x)
    assert output.shape == x.shape
    print("LFD Adapter test passed!")
```
