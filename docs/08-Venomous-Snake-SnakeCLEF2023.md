# 08 - Venomous Snake Species Classification (SnakeCLEF2023)

> **论文:** Watch out Venomous Snake Species: A Solution to SnakeCLEF2023  
> **作者:** Feiran Hu, Peng Wang, Yangyang Li, Chenlong Duan, Zijian Zhu, Fei Wang, Faen Zhang, Yong Li, Xiu-Shen Wei  
> **机构:** 南京理工大学 + 青岛创新奇智科技  
> **竞赛:** CLEF2023 / FGVC10 @ CVPR2023 — **第1名** (Private Leaderboard: 91.31%)  
> **代码:** https://github.com/xiaoxsparraw/CLEF2023  
> **arXiv:** 2307.09748v1

---

## 1. 论文概述

### 1.1 任务背景

SnakeCLEF2023 是 LifeCLEF2023 的一部分，目标是根据图像和元数据（地理位置信息）识别蛇的种类。核心挑战：

| 挑战 | 说明 |
|------|------|
| **细粒度图像识别** | 1784 种蛇类，许多外观极其相似 |
| **元数据利用** | 观察地点的国家/地区代码，需有效融入模型 |
| **长尾分布** | 头部种类最多 1262 条观察（2079 张图），尾部种类仅 3 条观察 |
| **毒蛇识别** | 误判毒蛇为无毒蛇的惩罚极高（权重 5.0） |
| **模型大小限制** | 模型大小 ≤ 1GB |

### 1.2 数据集规模

- **训练集:** 103,404 条观察记录，182,261 张高分辨率图像
- **类别数:** 1,784 种蛇类
- **地理区域:** 214 个不同地区

### 1.3 评估指标

复合指标 M，结合 F1-score 和毒蛇混淆惩罚：

```
M = (w1*F1 + w2*(100-P1) + w3*(100-P2) + w4*(100-P3) + w5*(100-P4)) / Σwi
```

其中权重为 w1=1.0, w2=1.0, w3=2.0, **w4=5.0**, w5=2.0。关键设计：
- **P3 (毒蛇→无害蛇，权重5.0):** 最严重的错误，给予最高惩罚
- **P2 (无害蛇→毒蛇，权重2.0):** 次严重但实际应用中可接受

### 1.4 实验结果（消融实验）

| Backbone | Resolution | Metric (%) | 改进策略 |
|----------|-----------|------------|---------|
| ResNet50 | 224×224 | 72.22 | baseline |
| BEiT-v2-L | 224×224 | 82.59 | 更强 backbone |
| BEiT-L | 384×384 | 88.74 | +CutMix |
| EVA-L | 336×336 | 86.82 | +CutMix |
| Swin-v2-L | 384×384 | 88.19 | +CutMix |
| VOLO | 448×448 | 88.50 | +CutMix |
| ConvNeXt-v2-L | 384×384 | 88.98 | +SeesawLoss + RandomMix |
| ConvNeXt-v2-L | 384×384 | 89.47 | +SeesawLoss + CutMix |
| ConvNeXt-v2-L | 512×512 | 90.86 | +SeesawLoss + CutMix + **Metadata** |
| ConvNeXt-v2-L | 512×512 | 91.98 | +Metadata + **Middle-level Feature** |
| ConvNeXt-v2-L | 512×512 | **93.65** | +Middle Feature + **Post-processing** |

> 注：Public Leaderboard 93.65%，Private Leaderboard 91.31%（仍为第1名）

---

## 2. 识别率提升技巧（含超参数）

### 2.1 架构设计：多层级特征融合 + 元数据注入

**核心思路：** 将 ConvNeXt-v2 的中间层特征、最终层特征和 CLIP 编码的元数据特征拼接后送入 MLP 分类器。

#### 模型架构 (`SnakeCLEF2023/train_pyramid_meta.py`)

```python
class metamodel(nn.Module):
    def __init__(self, model_arch, feature_dim, meta_feature_dim, num_classes):
        self.backbone = timm.create_model(model_arch, num_classes=0, pretrained=True)
        self.dropout = nn.Dropout(0.6)           # 高 dropout 防过拟合
        self.meta_batchnorm = nn.BatchNorm1d(meta_feature_dim)  # 元数据 BN 归一化
        # 2fc head: 拼接 backbone特征 + 元数据特征 + 中间层特征
        self.head = nn.Sequential(
            nn.Linear(feature_dim + meta_feature_dim + 384, 2300),  # +384 = 中间层特征维度
            nn.ReLU(),
            nn.Linear(2300, num_classes)
        )
    
    def forward(self, x, meta_feature):
        outs, mid_feature = self.backbone(x)  # 同时获取最终特征和中间层特征
        outs = self.dropout(outs)
        meta_feature = self.meta_batchnorm(meta_feature)
        features = torch.cat((outs, meta_feature, mid_feature), dim=-1)
        return self.head(features)
```

**关键参数：**
| 参数 | 值 | 说明 |
|------|-----|------|
| `model_arch` | `convnextv2_large.fcmae_ft_in22k_in1k_384` | ImageNet-22K 预训练 + ImageNet-1K 微调 |
| `img_size` | 512 | 输入分辨率 |
| `dropout` | 0.6 | 高 dropout 应对 1784 类过拟合 |
| `hidden_size` | 2300 | 分类头隐藏维度 |
| `head` | `2fc` | 两层全连接 + ReLU |
| `metaBN` | True | 元数据特征 BatchNorm |

### 2.2 元数据特征提取 (`SnakeCLEF2023/extract_metadata_feature.py`)

**方法：CLIP Text Encoder + PCA 降维**

```python
model_arch = 'ViT-L/14@336px'  # 使用 CLIP ViT-L 的文本编码器
model, _ = clip.load(model_arch, device=device)
text = clip.tokenize(code_lst).to(device)  # 国家/地区代码作为文本输入
text_features = model.encode_text(text)     # 提取文本特征

# PCA 降维，保留 99% 方差
pca = PCA(n_components=0.99, svd_solver='full')
pca.fit(features)
pca_features = pca.transform(features)
```

**流程：**
1. 收集训练集和验证集的所有国家/地区代码 (`code` 字段)
2. 用 CLIP Text Encoder (ViT-L/14@336px) 将每个代码编码为文本特征
3. PCA 降维（保留 99% 方差），获得压缩后的元数据特征
4. 保存 `code2feature.npy` 映射字典和 `clip_ViT-L_14_336px_train/val.npy` 特征文件

### 2.3 长尾分布处理：Seesaw Loss

**论文引用:** Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021)

**公式:**
```
L_seesaw(z) = -Σ yi log(σ̂i)
σ̂i = e^zi / (Σ_{j≠i} S_ij * e^zj + e^zi)
```

其中 S_ij 基于类别频率动态调整，p=0.8（惩罚参数）。

**代码实现 (`SnakeCLEF2023/train_pyramid_meta.py`):**

```python
class SeesawLossWithLogits(nn.Module):
    def __init__(self, class_counts: np.array, p: float = 0.8):
        class_counts = torch.FloatTensor(class_counts)
        conditions = class_counts[:, None] > class_counts[None, :]
        trues = (class_counts[None, :] / class_counts[:, None]) ** p
        falses = torch.ones(len(class_counts), len(class_counts))
        self.s = torch.where(conditions, trues, falses)  # 类别间动态缩放矩阵
        self.eps = 1.0e-6

    def forward(self, logits, targets):
        targets = F.one_hot(targets, num_classes).float()
        max_element, _ = logits.max(axis=-1)
        logits = logits - max_element[:, None]  # 数值稳定性
        numerator = torch.exp(logits)
        denominator = ((1 - targets)[:, None, :] * self.s[None, :, :] 
                       * torch.exp(logits)[:, None, :]).sum(axis=-1) + torch.exp(logits)
        sigma = numerator / (denominator + self.eps)
        loss = (-targets * torch.log(sigma + self.eps)).sum(-1)
        return loss.mean()
```

**关键点：**
- `class_counts` 从训练集的 `class_id` 列通过 `np.bincount` 统计得到
- `p=0.8` 控制对稀有类的惩罚强度
- 训练时使用 Seesaw Loss，验证时使用带 label smoothing 的 CrossEntropyLoss

### 2.4 数据增强

#### 训练增强 (`get_train_transforms`)

```python
Compose([
    RandomResizedCrop(512, 512, interpolation=cv2.INTER_CUBIC, scale=(0.5, 1.3)),
    Transpose(p=0.5),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    ShiftScaleRotate(p=0.3),
    PiecewiseAffine(p=0.5),
    RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=1.0),
    OneOf([OpticalDistortion(distort_limit=1.0),
           GridDistortion(num_steps=5, distort_limit=1.0)], p=0.5),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

#### 数据混合增强 (在 `train_one_epoch_mix` 中)

| 混合类型 | 说明 | 概率 |
|---------|------|------|
| `mixup` | α=2.0 的 Beta 分布混合 | 90% (mix_prob) |
| `cutmix` | β=1.0 的矩形区域替换 | 同上 |
| `tokenmix` | 基于 patch (16×16) 级别的随机替换 | 同上 |
| `randommix` | 随机从 mixup/cutmix 中选一个 | 同上 |

**代码关键片段：**
```python
if np.random.rand(1) < CFG['mix_prob']:  # 0.9 概率进行混合
    imgs, target_a, target_b, lam = get_mixed_data(imgs, image_labels, mix_type)
    loss = loss_fn(image_preds, target_a) * lam + loss_fn(image_preds, target_b) * (1. - lam)
```

### 2.5 毒蛇后处理 (`SnakeCLEF2023/predict.py`)

**核心思路：** 对模型置信度较低的预测，若 top-5 中包含毒蛇类别，则强制分类为毒蛇。

```python
threhold = 1.1  # 置信度阈值（softmax 最大值低于此值视为不确定）
venomous_class = set(is_venomous_df['class_id'][is_venomous_df['MIVS'] == 1])

# 对最低置信度的 6% 样本进行毒蛇检查
sorted_idx = np.argsort(max_prior_and_probs)
low_credict_num = int(len(predict_df) * 0.06)  # 约 6% 样本
for j in sorted_idx[0:low_credict_num]:
    top = torch.topk(torch.tensor(softmaxscore[j]), 5)  # 取 top-5 预测
    for idx in top.indices:
        if idx.item() in venomous_class:  # 若 top-5 中有毒蛇
            class_id_lst[j] = idx.item()   # 直接预测为毒蛇
            break
```

**设计理念：** 论文明确指出——"误判毒蛇为无害蛇的代价远高于误判无害蛇为毒蛇"。这是一个 precision/recall 的权衡：愿意接受更多的无害蛇被误判为毒蛇（false positive），但绝不能漏判毒蛇（false negative）。

---

## 3. 训练策略

### 3.1 主模型训练 (`train_pyramid_meta.py`)

| 超参数 | 值 | 说明 |
|--------|-----|------|
| **GPUs** | 4 × NVIDIA RTX 3090 | DataParallel |
| **Epochs** | 15 | 总训练轮数 |
| **Warmup Epochs** | 1 | 使用 linear warmup |
| **Warmup LR Factor** | 0.01 | 从 lr×0.01 线性增长 |
| **初始 LR** | 7.5×10⁻⁵ (`1.5e-4/2`) | 论文描述为 2×10⁻⁵，代码实际为 7.5×10⁻⁵ |
| **Min LR** | 1×10⁻⁸ | Cosine 退火终点 |
| **Optimizer** | AdamW | weight_decay=2×10⁻⁵ |
| **LR Schedule** | SequentialLR(Warmup→CosineAnnealing) | T_max=14 epochs |
| **Batch Size (train)** | 32 | 每 GPU |
| **Batch Size (val)** | 64 |  |
| **混合精度** | AMP (GradScaler) | `torch.cuda.amp.autocast` |
| **Label Smoothing** | 0.1 | 仅用于验证 loss，不用于训练 |
| **Seed** | 42 | 固定随机种子 |
| **DifferLR** | False | 不区分 backbone/head 学习率（最终方案） |
| **Backbone LR Factor** | 0.2 | 如果使用 DifferLR，backbone 学习率为 head 的 1/5 |

### 3.2 Prior Model 训练 (`train_prior.py`)

**作用：** 利用地理位置信息计算类别先验概率，在推理阶段与主模型预测联合。

**模型结构：**
```python
class priormodel(nn.Module):
    def __init__(self, meta_feature_dim, classes_num):
        self.metaBN = nn.BatchNorm1d(meta_feature_dim)
        self.fc1 = nn.Linear(meta_feature_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, classes_num)
```

**训练配置：**

| 超参数 | 值 |
|--------|-----|
| Epochs | 60 |
| Batch Size | 512 |
| LR | 1×10⁻⁵ |
| Min LR | 1×10⁻⁶ |
| Weight Decay | 2×10⁻⁵ |
| Warmup Epochs | 1 |
| Optimizer | AdamW |
| 采样策略 | `ImbalancedDatasetSampler` (balanced sampling) |

**损失函数 (Prior Loss):**
```python
class priorloss(nn.Module):
    def __init__(self, alpha=10):
        self.alpha = alpha  # 正样本权重
    
    def forward(self, inputs, targets, r):
        # 对正样本（在该地点被观察到的种类）加权
        weight = torch.where(targets == 1, alpha * ones, ones)
        loss_loc = F.binary_cross_entropy_with_logits(inputs, targets, weight)
        # 随机地点应预测全 0（未观察到）
        loss_r = F.binary_cross_entropy_with_logits(r, zeros)
        return loss_loc + loss_r
```

**关键设计：**
- 使用 `ImbalancedDatasetSampler` 对训练数据做平衡采样
- 构造随机地点数据 r，要求模型对随机地点输出全 0（提高泛化性）
- `alpha=10` 增强正样本的重要性

### 3.3 增量训练策略

论文提到在最后 3 个 epoch 切换为 RWWCE (Real-World Weighted Cross-Entropy) loss 并使用更低的 LR。代码中未见此部分显式实现，但这是其描述的策略。

**RWWCE 损失函数：** 对不同类型的错误赋予不同权重，惩罚毒蛇→无害蛇的错误最重（权重 5.0）。

### 3.4 FungiCLEF2023 训练配置

为完整起见，记录 FungiCLEF 的训练配置（代码 `FungiCLEF2023/train_seesawloss.py`）：

| 参数 | 值 |
|------|-----|
| Backbone | `convnextv2_large.fcmae_ft_in22k_in1k_384` |
| Classes | 1604 |
| Img Size | 512 |
| Epochs | 15 |
| LR | 1.5×10⁻⁵ |
| Mix Prob | 0.8 |
| Label Smoothing | 0.1 |
| Weight Decay | 2×10⁻⁵ |

---

## 4. 推理优化

### 4.1 TTA (Test Time Augmentation)

```python
'tta': 1  # 实际只用 1 次（论文描述使用了 TTA）

# 推理时的增强（比训练时更轻量）
def get_inference_transforms_last():
    return Compose([
        RandomResizedCrop(512, 512, scale=(0.6, 1.2)),  # 比训练 scale 窄
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomBrightnessContrast(limit=(-0.2, 0.2), p=0.5),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
```

### 4.2 多阶段推理流程 (`SnakeCLEF2023/predict.py`)

完整的推理 pipeline 包含以下步骤：

```
阶段1: 主模型推理
├── 对每个 observation 的每张图片进行推理
├── TTA（多次推理取平均）
└── 对同一 observation 的多张图片概率取平均

阶段2: Prior Model 后处理
├── 对置信度 < 1.1 的样本（即大部分样本）
├── 用 Prior Model 计算地理先验概率
├── 联合概率: S' = Softmax(Prior) ⊙ Softmax(MainModel)
└── 更新预测

阶段3: 毒蛇安全后处理
├── 对置信度最低的 6% 样本
├── 检查 top-5 预测是否包含毒蛇
├── 如有毒蛇 → 强制预测为毒蛇
└── 平衡 precision/recall，偏向 recall
```

### 4.3 Observation 级别聚合

```python
def unique_observation_id(observation_id_lst, array):
    id2count = {}  # 统计每个 observation 的图片数
    id2probs = {}  # 累积概率
    for i, obs_id in enumerate(observation_id_lst):
        sum_probs = sum([lst[i] for lst in array])  # 多模型取平均
        avg_obs_id_probs = sum_probs / len(array)
        if obs_id in id2count:
            id2count[obs_id] += 1
            id2probs[obs_id] += avg_obs_id_probs
        else:
            id2count[obs_id] = 1
            id2probs[obs_id] = avg_obs_id_probs
    # 取平均
    for obs_id in id2probs:
        id2probs[obs_id] /= id2count[obs_id]
```

### 4.4 FungiCLEF2023 推理特点

- TTA = 5 次（比 SnakeCLEF 多）
- 区分已知/未知地点，分别设定 openset 阈值
- openset 处理：如果预测置信度低于阈值 → 标记为 -1 (未知)
- 使用 `best_f1_threshold` 在验证集上搜索最佳阈值

---

## 5. 对 image-classification 项目的借鉴

### 5.1 可直接复用的技巧

| 技巧 | 适用场景 | 优先级 | 实现难度 |
|------|---------|--------|---------|
| **Seesaw Loss** | 长尾分布数据集 | ⭐⭐⭐ | 低（独立 loss 模块） |
| **数据混合增强 (MixUp/CutMix/TokenMix/RandomMix)** | 所有细粒度分类 | ⭐⭐⭐ | 低（已在训练循环中） |
| **多层次特征融合 (Middle+Final features)** | 细粒度分类 | ⭐⭐⭐ | 中（需修改 backbone forward） |
| **CLIP 元数据编码 + PCA** | 有辅助文本/类别信息 | ⭐⭐ | 中（需 CLIP 依赖） |
| **Dropout = 0.6** | 类别多/样本少的任务 | ⭐⭐ | 低 |
| **毒蛇式后处理（top-k 安全策略）** | 误判成本不对称的任务 | ⭐⭐ | 低 |
| **Geographic Prior Model** | 有地理位置信息的分类 | ⭐⭐ | 中 |
| **AMP 混合精度训练** | 所有任务 | ⭐⭐⭐ | 低（PyTorch 内置） |
| **Observation 级别聚合** | 同一实体有多张图片 | ⭐ | 低 |
| **Seesaw Loss + Label Smoothing CrossEntropy 配合** | 长尾分类 | ⭐⭐ | 低 |

### 5.2 推荐的超参数配置（起始值）

```python
# 长尾细粒度分类的推荐起始配置
RECOMMENDED_CFG = {
    # 模型
    'backbone': 'convnextv2_large',     # 或 convnextv2_base 权衡速度
    'pretrained': True,                  # 始终使用预训练
    'img_size': 384,                     # 从 384 开始，逐步提升到 512
    'dropout': 0.5,                      # 高 dropout 防过拟合
    'head_hidden': 2048,                 # 分类头隐藏维度

    # 数据增强
    'random_resized_crop_scale': (0.5, 1.3),
    'mix_type': 'cutmix',                # 论文推荐 CutMix 优于 RandomMix
    'mix_prob': 0.8,                     # 混合增强概率

    # 损失函数
    'loss': 'seesaw',                    # 长尾时用 Seesaw
    'seesaw_p': 0.8,                     # Seesaw 惩罚参数
    'label_smoothing': 0.1,              # 验证时使用

    # 优化器
    'optimizer': 'AdamW',
    'lr': 2e-5,                          # 论文值，代码为 7.5e-5
    'weight_decay': 2e-5,
    'warmup_epochs': 1,
    'warmup_lr_factor': 0.01,
    'min_lr': 1e-8,
    'epochs': 15,

    # 训练
    'train_bs': 32,                      # 4×3090 配置
    'amp': True,                         # 混合精度

    # 后处理（毒蛇式安全策略）
    'post_process_topk': 5,
    'post_process_threshold_ratio': 0.06,
}
```

### 5.3 核心洞察

1. **ConvNeXt-v2 是最佳 backbone 选择：** 在实验中超越了 ResNet、BEiT、EVA、Swin-v2、VOLO。尤其 `fcmae_ft_in22k_in1k` 预训练变体（FCMAE 自监督预训练 + ImageNet-22K 微调）效果最好。

2. **分辨率提升持续有效：** 224→384→512，每步都有显著增益。512×512 是 4×3090 的内存上限。

3. **CutMix 优于其他混合策略：** 论文明确指出 CutMix 在 SnakeCLEF 上优于 RandomMix。

4. **元数据的价值巨大：** 加入 location metadata 提升了 +1.88%（88.98%→90.86%）。可推广：任何有辅助信息（地点、时间、环境等）的 FGVC 任务都可尝试此思路。

5. **中间层特征对细粒度至关重要：** 拼接中间层特征提升了 +1.12%（90.86%→91.98%）。细粒度分类的判别线索往往在中间层。

6. **毒蛇后处理是竞赛取胜的关键：** 但需注意——这是针对特定评估指标（毒蛇误判高惩罚）的优化。对于通用分类任务，此策略需要根据实际需求调整。

7. **Seesaw Loss 优于 CrossEntropy：** 在长尾场景下是显著改进，且实现简单。通过类别频率动态调整 softmax 分母中负类的贡献。

8. **Prior Model 的设计很有借鉴意义：** 用 CLIP 编码地理位置→三隐藏层 MLP 预测物种在该地点的出现概率→与视觉模型联合推理。这种"视觉+先验"的框架适用于任何有观测元数据的分类任务。

### 5.4 代码文件索引

| 文件 | 说明 |
|------|------|
| `code/SnakeCLEF2023/train_pyramid_meta.py` | 主模型训练（金字塔+元数据） |
| `code/SnakeCLEF2023/train_prior.py` | Prior Model 训练 |
| `code/SnakeCLEF2023/predict.py` | 完整推理 pipeline |
| `code/SnakeCLEF2023/extract_metadata_feature.py` | CLIP+PCA 元数据特征提取 |
| `code/SnakeCLEF2023/imbalanced.py` | 不平衡数据集采样器 |
| `code/FungiCLEF2023/train_seesawloss.py` | FungiCLEF 训练（Seesaw Loss + 数据混合） |
| `code/FungiCLEF2023/predict.py` | FungiCLEF 推理（含 openset 处理） |
| `code/README.md` | 使用说明 |
