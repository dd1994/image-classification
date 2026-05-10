# FungiCLEF 2023 — 基于深度学习的真菌物种识别方案分析

> **论文**: A Deep Learning based Solution to FungiCLEF2023  
> **作者**: Feiran Hu, Peng Wang, Yangyang Li, Chenlong Duan, Zijian Zhu, Yong Li, Xiu-Shen Wei  
> **单位**: 南京理工大学 计算机科学与工程学院  
> **成绩**: 私榜第3名 (F1 54.34%)  
> **代码**: <https://github.com/xiaoxsparraw/CLEF2023>

---

## 一、论文概述

### 1.1 任务背景

FungiCLEF2023 是 LifeCLEF2023 的一部分，与 FGVC10 workshop（CVPR 2023）联合举办，目标是利用图像和元数据识别真菌物种。

### 1.2 核心挑战

| 挑战 | 描述 |
|------|------|
| **细粒度图像识别** | 不同真菌物种间视觉差异极小，类间方差小而类内方差大 |
| **开集识别 (Open-Set)** | 测试集包含训练集中未出现的未知物种 |
| **元数据利用** | 如何有效融合地理、栖息地、基质等元数据 |
| **长尾分布** | 物种样本量极度不均衡（少数头部类占据大量样本） |
| **模型体积限制** | 模型大小不能超过 1GB |

### 1.3 数据集概况

- **训练集**: 295,938 张图像，1,604 个物种（DF20 数据集，主要来自丹麦）
- **验证集**: 30,131 个观测，60,832 张图像，2,713 个物种（含未知物种）
- **测试集**: 30,130 个观测，60,225 张图像
- **元数据**: 国家代码、经纬度、位置名称、栖息地、基质类型
- **一个观测可能包含多张图像**，最终需以观测为单位做预测

### 1.4 评估指标

5 个复合指标的加权组合：

1. **标准分类错误** — 未知物种应正确标注为 "unknown"
2. **有毒/无毒混淆成本** — 可食/有毒混淆惩罚更高
3. **用户导向损失** — 综合分类错误 + 毒性混淆
4. **未知物种混淆成本** — 漏报未知物种比误分类惩罚更重
5. **稀有物种加权** — 按训练集中物种频率的倒数加权

---

## 二、模型架构

### 2.1 Backbone：VOLO (Vision Outlooker)

最终选择 **VOLO-d4-448** 作为主干网络：

- 结合了 **Outlooker Attention**（局部精细编码）+ **Transformer**（全局依赖建模）
- ImageNet 预训练权重来自 timm 库
- 输入分辨率：**448×448**（实验表明高于此分辨率指标趋于平台）
- 其他测试过的 backbone：ResNet50、BEiT-v2-B、ConvNeXt-v2-L

### 2.2 分类头

FungiCLEF 2023 使用标准单层 FC 头（backbone → Linear(backbone_dim, 1604)）。

SnakeCLEF 2023 中使用了更复杂的 **金字塔元数据融合头 (Pyramid Meta Head)**（见下文 §6）。

---

## 三、识别率提升技巧（含超参数）

### 3.1 Seesaw Loss — 长尾分布解决方案

**来源**: [Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021)](https://arxiv.org/abs/2008.10032)

**原理**: 在 Softmax 分母中对来自"强类"（样本多）的 logit 施加惩罚权重，减轻头部类对尾部类的压制。

**公式**:

$$\mathcal{S}_{ij} = \begin{cases} (\frac{n_j}{n_i})^p, & n_i > n_j \\ 1, & \text{otherwise} \end{cases}$$

其中 $n_i, n_j$ 为类别样本数，$p$ 控制惩罚强度。

**代码实现** (`FungiCLEF2023/train_seesawloss.py` 第 224-252 行):

```python
class SeesawLossWithLogits(nn.Module):
    def __init__(self, class_counts: np.array, p: float = 0.8):
        # 构建类别间惩罚矩阵 S
        class_counts = torch.FloatTensor(class_counts)
        conditions = class_counts[:, None] > class_counts[None, :]
        trues = (class_counts[None, :] / class_counts[:, None]) ** p
        falses = torch.ones(len(class_counts), len(class_counts))
        self.s = torch.where(conditions, trues, falses)
```

**对比实验**:
| Loss | Backbone | F1 |
|------|----------|-----|
| CrossEntropy | ResNet50 | 43.19% |
| Weighted CE | ResNet50 | 40.64% (不升反降!) |
| Seesaw Loss | VOLO | 52.65% |

> ⚠️ **关键发现**: 加权交叉熵在此任务中**反而降低**了性能。Seesaw Loss 通过动态调整类别间的关系矩阵取得了显著提升。

### 3.2 数据增强策略

#### 3.2.1 基础增强（Albumentations）

```python
# 来自 train_seesawloss.py 的 get_train_transforms()
Compose([
    RandomResizedCrop(512, 512, interpolation=cv2.INTER_CUBIC, scale=(0.5, 1.3)),
    Transpose(p=0.5),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    ShiftScaleRotate(p=0.3),
    PiecewiseAffine(p=0.5),
    RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=1.0),
    OneOf([
        OpticalDistortion(distort_limit=1.0),
        GridDistortion(num_steps=5, distort_limit=1.),
    ], p=0.5),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

**关键参数**:
- `RandomResizedCrop` scale = `(0.5, 1.3)` — 较大的裁剪范围，增强尺度鲁棒性
- `RandomBrightnessContrast` p=1.0 — 始终应用亮度和对比度增强
- `PiecewiseAffine` p=0.5 — 模拟非刚性形变（对真菌形态变化很有帮助）

#### 3.2.2 混合增强 (Mix-based Augmentation)

**实现** (`get_mixed_data()` 函数):

| 方法 | 策略 | 参数 |
|------|------|------|
| **Mixup** | 图像线性插值 | α=2.0 (Beta分布) |
| **CutMix** | 矩形区域替换 | β=1.0 |
| **TokenMix** | 随机 patch 替换 | patch=16, 随机 mask |
| **RandomMix** | 随机选择以上三种之一 | — |

```python
CFG = {
    'mix_type': 'randommix',   # 每 batch 随机选择 mixup/cutmix/tokenmix
    'mix_prob': 0.8,           # 80% 的 batch 应用混合增强
    'patch': 16,               # TokenMix 的 patch 大小
}
```

**效果**: Seesaw Loss + CutMix 将 F1 从 52.65% → 53.93% (+1.28%)

#### 3.2.3 测试时增强 (TTA)

```python
CFG = {'tta': 5}  # predict.py

# 推理时对每张图做 5 次随机增强，取平均概率
def tta_inference(model, dataloader, tta=5):
    for i in range(tta):
        probs, obs_ids = inference(model, dataloader)
        # 累加 5 次概率后取平均
    return probs / tta
```

TTA 变换（`get_inference_transforms_last()`）:
- `RandomResizedCrop(448, scale=(0.6, 1.2))` + 翻转 + 亮度对比度增强

### 3.3 训练配置总结

| 超参数 | 值 | 说明 |
|--------|-----|------|
| **Model** | volo_d4_448 | Vision Outlooker |
| **Input size** | 448×448 (最终) | 实验 224/384/448/512 |
| **Epochs** | 15 | 短训练周期 |
| **Batch size** | 32 (train) / 64 (valid) | 4 × RTX 3090, DataParallel |
| **Optimizer** | AdamW | — |
| **Learning rate** | 2×10⁻⁵（论文）/ 1.5×10⁻⁵（代码） | — |
| **Min LR** | 1×10⁻⁸ | Cosine 退火最低点 |
| **Weight decay** | 2×10⁻⁵ | — |
| **Warmup** | 1 epoch, LR factor 0.01 | Linear warmup |
| **LR schedule** | Cosine Annealing | Warmup 后的 14 个 epoch |
| **Label smoothing** | 0.1 | CrossEntropy 验证用 |
| **AMP** | ✅ (GradScaler) | 混合精度训练 |
| **Seed** | 42 | 固定随机种子 |

### 3.4 差异化学习率 (DifferLR)

代码中支持但**未启用**（默认 `differLR=False`）:

```python
CFG = {
    'differLR': False,
    'bacbone_lr_factor': 0.2,  # backbone 的学习率是 head 的 0.2 倍
}
```

当 `differLR=True` 时，backbone 参数使用更小的学习率（预训练权重只需微调），head 使用完整 LR。

---

## 四、训练策略

### 4.1 训练流程

```
1. 从 metadata/*.csv 读取图像路径和标签
2. 加载 ImageNet 预训练 backbone (timm)
3. 替换分类头为 num_classes=1604
4. DataParallel 包装 (4 GPU)
5. SeesawLoss + RandomMix 训练 15 epochs
6. 每 epoch 验证, 保存最佳 checkpoint
```

### 4.2 关键实现细节

**1) 多图观测融合**: 一个物种观测可能包含多张图片，推理时对同一 `observationID` 的所有图像预测取平均。

```python
# predict.py - average_probs_by_obs_id()
def average_probs_by_obs_id(probs, obs_id_lst):
    id2probs = {}; id2count = {}
    for obs_id, probs in zip(obs_id_lst, probs):
        id2probs[obs_id] = id2probs.get(obs_id, 0) + probs
        id2count[obs_id] = id2count.get(obs_id, 0) + 1
    # 取平均
    for obs_id in id2probs:
        id2probs[obs_id] /= id2count[obs_id]
```

**2) 已知/未知区域分开推理**: 注意到训练集只覆盖丹麦地区，验证/测试集中有来自其他地区的未知物种。代码区分了 `known_locality` 和 `unknown_locality`，分别处理阈值。

**3) 开集后处理** (简单阈值法):

```python
# predict.py - handle_openset()
def handle_openset(class_id_lst, prob_lst, prob_threshold=None):
    for class_id, prob in zip(class_id_lst, prob_lst):
        predict_label = class_id
        if prob < prob_threshold:    # 最大概率低于阈值 → 标记为未知(-1)
            predict_label = -1
```

在验证集上搜索最佳阈值（按 accuracy 或 macro-F1），用于测试集预测。

### 4.3 消融实验（来自论文 Table 2）

| Backbone | Resolution | 技巧 | Metric (F1) |
|----------|-----------|------|-------------|
| ResNet50 | 224 | CE loss | 43.19% |
| ResNet50 | 224 | Weighted CE | 40.64% ↓ |
| BEiT-v2-B | 224 | Stronger backbone | 50.40% |
| BEiT-v2-B | 224 | + metadata | 51.49% |
| **VOLO** | **448** | + seesaw loss | 52.65% |
| **VOLO** | **448** | + cutmix | 53.93% |
| **VOLO** | **448** | + open-set | **55.46%** (最终) |
| ConvNeXt-v2-L | 512 | + seesaw + cutmix + open-set | 55.35% |

**关键结论**:
1. Backbone 升级带来最大收益 (ResNet→BEiT: +7.21%)
2. Seesaw Loss + CutMix 组合有效 (+1.28%)
3. 开集后处理贡献 +1.53%
4. 元数据在**弱模型**上有帮助，但在**强模型**上无效（甚至有害）
5. 分辨率从 224→448 有提升，再往上趋于平台

---

## 五、推理优化

### 5.1 推理 Pipeline

```
测试图像
  │
  ├── 根据 locality 划分 known/unknown 区域
  │
  ├── TTA × 5 (随机裁剪+翻转+颜色增强)
  │
  ├── 模型推理 → Softmax → 概率向量
  │
  ├── 按 observationID 聚合多图 → 平均概率
  │
  └── 开集阈值处理 → 低置信度样本标记为 -1 (unknown)
       │
       └── 输出 submission.csv
```

### 5.2 推理配置

| 参数 | 值 |
|------|-----|
| GPU | 8 × (推理只在单 GPU with DataParallel) |
| Batch size | 256×4 = 1024 |
| TTA passes | 5 |
| Workers | 32 |
| Known 区域阈值 | ~0.27 (F1最佳) |
| Unknown 区域阈值 | ~0.23 (F1最佳) |

---

## 六、SnakeCLEF 2023 额外技巧（同期同代码库）

虽非 FungiCLEF 的直接方法，但同团队在同竞赛系统中的 SnakeCLEF 方案提供了可借鉴的设计：

### 6.1 元数据特征提取（CLIP 文本编码 + PCA）

`SnakeCLEF2023/extract_metadata_feature.py`:

```python
# 1. 用 CLIP ViT-L/14@336px 将元数据文本（地点编码等）编码为向量
model, _ = clip.load('ViT-L/14@336px', device=device)
text_features = model.encode_text(clip.tokenize(code_lst))

# 2. PCA 降维保留 99% 方差
pca = PCA(n_components=0.99, svd_solver='full')
pca_features = pca.transform(features)

# 3. 保存为 numpy 文件供训练/推理使用
```

### 6.2 金字塔元数据融合模型

`SnakeCLEF2023/train_pyramid_meta.py` — `metamodel`:

```python
class metamodel(nn.Module):
    def forward(self, x, meta_feature):
        outs, mid_feature = self.backbone(x)          # backbone 返回 (global, mid)
        outs = self.dropout(outs)
        meta_feature = self.meta_batchnorm(meta_feature)
        features = torch.cat((outs, meta_feature, mid_feature), dim=-1)  # 三层融合
        outs = self.head(features)                     # 2fc: 2300→hidden→1784
        return outs
```

**设计要点**:
- Backbone 同时输出全局特征 + 中间层特征 (384维)
- 拼接 `[global_feat, meta_feat, mid_feat]` 三路信息
- 2 层 FC head (hidden_size=2300, dropout=0.6)
- Meta 特征先过 BatchNorm 再拼接

### 6.3 先验模型 (Prior Model)

`SnakeCLEF2023/train_prior.py`:

```python
class priormodel(nn.Module):
    def __init__(self, meta_feature_dim, classes_num):
        self.metaBN = nn.BatchNorm1d(meta_feature_dim)
        self.fc1 = nn.Linear(meta_feature_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.3)    # 仅 0.3, 比主模型低
        self.fc3 = nn.Linear(256, classes_num)
```

- 仅用元数据特征 (不依赖图像) 预测物种分布
- 使用 **平衡采样器** (`ImbalancedDatasetSampler`) 解决长尾问题
- 自定义 **PriorLoss**: `BCE(logits, targets, weight=α) + BCE(random_noise, zeros)`
  - 对正类加权 α=10，强制模型对随机噪声输出零 → 提升校准
- 训练 60 epochs, lr=1e-5, batch=512

**推理时使用**: 当主模型预测置信度低于阈值时，乘上先验模型的输出概率重新预测 → 利用地理分布先验修正不可靠预测。

### 6.4 ImbalancedDatasetSampler

`SnakeCLEF2023/imbalanced.py`:

```python
class ImbalancedDatasetSampler(Sampler):
    # 按类别频率的倒数计算采样权重
    weights = 1.0 / label_to_count[df["label"]]
    # torch.multinomial 带权重有放回采样
    return (indices[i] for i in torch.multinomial(weights, num_samples, replacement=True))
```

---

## 七、对 image-classification 项目的借鉴

### 7.1 可直接采纳的技巧

#### (A) Seesaw Loss for 长尾分布

```python
# 可在项目中实现为独立 loss 模块
# class_counts 从训练集统计得出
loss = SeesawLossWithLogits(class_counts, p=0.8)
```

**适用场景**: 当各类别样本量差异超过 10:1 时。

#### (B) RandomMix 数据增强策略

```python
# 每个 batch 随机选择 mixup/cutmix/tokenmix 之一
if np.random.rand() < 0.8:
    imgs, target_a, target_b, lam = get_mixed_data(imgs, labels, 'randommix')
    loss = loss_fn(preds, target_a) * lam + loss_fn(preds, target_b) * (1 - lam)
```

#### (C) 差异化学习率 (DifferLR)

```python
backbone_params = list(map(id, model.module.backbone.parameters()))
head_params = filter(lambda p: id(p) not in backbone_params, model.parameters())
optimizer = AdamW([
    {'params': backbone_params, 'lr': base_lr * 0.2},
    {'params': head_params, 'lr': base_lr}
], weight_decay=2e-5)
```

**作用**: 保护预训练权重不被过度更新，head 以更高 LR 快速适应新任务。

#### (D) 多图观测聚合 (适用于多视角/多帧场景)

如果项目涉及同一目标的多张图像输入，可按 observation/instance ID 聚合概率取平均。

### 7.2 可借鉴的参数配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 分辨率 | ≥448 | 细粒度任务需要足够空间分辨率 |
| Epochs | 15-20 | 小数据集可减到 10，大数据集可增到 30 |
| LR | 2×10⁻⁵ | 预训练 backbone 的合理起点 |
| Weight decay | 2×10⁻⁵ | 较低的 weight decay 匹配较低 LR |
| Warmup | 1 epoch | Linear warmup from 1% LR |
| LR schedule | Cosine Annealing | 平滑退火至 min_lr=1e-8 |
| Label smoothing | 0.1 | 验证时使用，训练时用 Seesaw Loss |
| Mix prob | 0.8 | 80% batch 应用混合增强 |
| Batch size | 32-64 (per GPU) | FP16 混合精度可保证显存 |
| Dropout | 仅在 head 使用 | 主干网络不加 dropout |

### 7.3 值得注意的"反模式"

1. **Weighted CE 可能有害**: 论文实验表明简单按类别频率倒数加权 CE 在此任务中反而下降 2.55%。优先使用 Seesaw Loss 或 Focal Loss。

2. **元数据并非总是有用**: 在强 backbone (BEiT/VOLO) 上元数据反而无效。仅在 baseline 较弱时有帮助。

3. **大分辨率收益递减**: 224→448 有显著提升，但 448→512 基本无差别。

### 7.4 可扩展的增强方向

| 方向 | 来源 | 实施建议 |
|------|------|---------|
| VOLO 替换为更强架构 | 本文思路 | 尝试 ConvNeXt-v2, BEiT-v3, EVA-02 |
| TokenMix 用于 ViT 族 backbone | 本文代码 | ViT/DeiT/Swin 适用 patch 级混合 |
| 先验模型 + 概率乘积融合 | SnakeCLEF 代码 | 适用于有地理/时间先验的场景 |
| 开集识别阈值搜索 | 本文 predict.py | 在验证集上网格搜索最佳置信度阈值 |

---

## 八、代码文件映射

| 功能 | 文件路径 |
|------|---------|
| FungiCLEF 训练 | `FungiCLEF2023/train_seesawloss.py` |
| FungiCLEF 推理 | `FungiCLEF2023/predict.py` |
| SnakeCLEF 训练(meta融合) | `SnakeCLEF2023/train_pyramid_meta.py` |
| SnakeCLEF 先验模型训练 | `SnakeCLEF2023/train_prior.py` |
| SnakeCLEF 推理 | `SnakeCLEF2023/predict.py` |
| 元数据特征提取 | `SnakeCLEF2023/extract_metadata_feature.py` |
| 平衡采样器 | `SnakeCLEF2023/imbalanced.py` |
| Seesaw Loss 实现 | `FungiCLEF2023/train_seesawloss.py:224-252` |
| Mix 增强实现 | `FungiCLEF2023/train_seesawloss.py:156-190` |

---

*分析日期: 2026-05-10 | 基于论文 + 代码的完整分析*
