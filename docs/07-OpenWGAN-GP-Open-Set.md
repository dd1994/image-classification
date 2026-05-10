# OpenWGAN-GP: 细粒度开放集真菌分类

> 论文：OpenWGAN-GP for Fine-Grained Open-Set Fungi Classification (FungiCLEF 2024 第 1 名)
> 作者：Jack N. Etheredge (Twosense)
> 代码：`D:\fgvc-survey\OpenWGAN-GP-for-Fine-Grained-Open-Set-Fungi-Classification\code\`
> 核心代码模块：`fungiclef-effnet/`

---

## 1. 论文概述

### 1.1 任务定义

FungiCLEF 2024 是一个**长尾分布、细粒度、开放集**分类挑战，同时增加**毒蘑菇/食用菌混淆惩罚**：

- **数据集**：Danish Fungi 2020，295,938 张训练图像，1,604 个已知物种 + 未知开放集类别
- **Track 1**：标准分类损失（含 unknown 类）
- **Track 2**：毒/食用混淆损失——Poisonous→Edible 误分类惩罚 ×100
- **Track 3**：Track 1 + Track 2（用户导向评分）
- **F1**：macro-averaged F1 score

### 1.2 核心创新：OpenWGAN-GP

本质上是将 **WGAN-GP** 的训练稳定性改进引入 **OpenGAN** 框架：

1. 先训练一个闭集分类器（closed-set classifier）
2. 用该分类器的**倒数第二层特征**训练一个轻量级 MLP 判别器
3. 判别器做二分类：closed-set（已知物种） vs. open-set（未知物种）
4. 同时训练一个生成器来**合成额外的 open-set 特征**
5. 用 WGAN-GP（Wasserstein loss + Gradient Penalty）替代原始 GAN 训练范式
6. **判别器用 LayerNorm 替代 BatchNorm**（这是 WGAN-GP 的关键建议）

### 1.3 最终成绩

| 指标 | Public LB | Private LB | 排名 |
|------|-----------|------------|------|
| Track 1 ↓ | 0.2394 | 0.2436 | 🥇 1st |
| Track 2 ↓ | 0.1681 | 0.1613 | 🥉 3rd |
| Track 3 ↓ | 0.4075 | 0.4075 | 🥈 2nd |
| F1 ↑ | 49.81% | 56.79% | 🥇 1st |
| Accuracy ↑ | 76.06% | 75.64% | 🥇 1st |

---

## 2. 识别率提升技巧（含超参数）

### 2.1 损失函数设计

#### 2.1.1 Seesaw Loss（核心长尾损失）

文件：`fungiclef-effnet/losses.py` → `SeesawLoss` 类

```python
# 配置（config.yaml）
train:
  loss_function: "seesaw"
  
# 实现参数
SeesawLoss(num_classes=1604, p=0.8, q=2.0, eps=1e-2)
```

**原理**：
- **Mitigation factor (p=0.8)**：根据 tail/head 类别样本比例降低对 tail 类的惩罚
- **Compensation factor (q=2.0)**：当模型对某错分类别过于自信时，增加该错分类别的惩罚
- 双因子动态调整，比 Focal Loss 更适合长尾细粒度任务

**公式**：
$$
S_{ij} = M_{ij} \cdot C_{ij}
$$
其中 $M_{ij}$ 是缓解因子（基于类别实例比），$C_{ij}$ 是补偿因子（基于 softmax 评分比）

> **对 image-classification 项目的借鉴**：4 万类大长尾分类场景下，Seesaw Loss 比 Focal Loss 更有效。Seesaw Loss 自动适应类别不均衡，不需要手动设定 class weights。

#### 2.1.2 Custom Poison Loss（毒蘑菇加权交叉熵）

```python
# losses.py → CompositeLoss 类
use_poison_loss: True
poison_class_weight: [1.0, 100.0]  # edible:poisonous = 1:100
```

**实现细节**：
- 将所有毒蘑菇类别的 softmax 概率求和 → $P_{poisonous}$
- 将所有食用菌类别的 softmax 概率求和 → $P_{edible}$
- 做二分类加权交叉熵，毒蘑菇权重 ×100

> **借鉴**：任何有**类别组级别不对称代价**的分类任务都可以用这个思路。比如珍贵物种 vs 常见物种，保护级 vs 非保护级。

#### 2.1.3 总损失组合

```
Total Loss = Seesaw Loss + Poison Loss (weighted BCE)
```

### 2.2 LogitNorm：Logit 归一化

文件：`fungiclef-effnet/losses.py` → `CompositeLoss.forward()`

```python
use_logitnorm: True
logitnorm_t: 0.01  # temperature for CIFAR-100
```

**实现**：
```python
if self.use_logitnorm:
    norms = torch.norm(closed_outputs, p=2, dim=-1, keepdim=True) + 1e-7
    closed_outputs = torch.div(closed_outputs, norms) / self.logitnorm_t
```

**作用**：
- 对 logits 做 L2 归一化，改善 embedding 空间中类间分离
- 减少过置信预测，类似 Temperature Scaling 的效果
- 论文实验显示：LogitNorm 提升 Track 1 和 F1，但略微降低 Track 2（可能与 Poison Loss 不完全兼容）
- 对 OpenWGAN-GP 的 embedding 质量有显著帮助

> **借鉴**：细粒度分类 + 开放集检测场景下推荐使用 LogitNorm。纯闭集分类可以单独评估。

### 2.3 模型架构与集成

#### 2.3.1 模型选择

| 模型 | 分辨率 | 预训练 | 元数据 | Batch Size | 参数量 |
|------|--------|--------|--------|------------|--------|
| Metaformer-0 | 384×384 | iNaturalist2021 | ✅ | 32 | 轻量 |
| Metaformer-2 | 384×384 | iNaturalist2021 | ✅ | 12 | 中等 |
| CAFormer-S18 | 384→576 | ImageNet-21K | ❌ | 40 | 轻量 |

文件：`fungiclef-effnet/closedset_model.py` → `build_model()`

架构特点：
- **Metaformer = Conv blocks + Multi-Head Self-Attention Transformer blocks**
- **CAFormer** 与 Metaformer 结构同源但不用元数据
- 都是 **hybrid CNN-Transformer** 架构，计算效率优于纯 ViT

**关键设计理念**：用多个计算效率高的模型集成（ensemble），优于单个大模型。
- 论文引用了 "Wisdom of Committees" [23] 证明：多个轻量模型集成在训练和推理成本上都优于单个大模型

#### 2.3.2 数据划分策略

不同模型使用不同的 train/val 划分（A/B/C/D splits），增加集成多样性：

```
CAFormer-S18 (A)  ← data split A
CAFormer-S18 (B)  ← data split B  
CAFormer-S18 (C)  ← data split C
Metaformer-0 (D)  ← data split D
Metaformer-2 (D)  ← data split D
```

最佳集成组合：**Metaformer-0 (D) + Metaformer-2 (D) + CAFormer-S18 (C)**

### 2.4 元数据融合

文件：`fungiclef-effnet/datasets.py` → `encode_metadata_row()`

```python
meta_dims=[4, 34, 32, 31]  # temporal(4) + country(34) + substrate(32) + habitat(31)
```

**编码方式**：
- **月份/日期**：周期性编码 → `[sin(2π·month/12), cos(2π·month/12), sin(2π·day/31), cos(2π·day/31)]`
- **国家代码、基质、栖息地**：one-hot 编码
- 通过可训练的 embedding 投影到与图像特征相同的维度后融合

> **借鉴**：如果有地理位置、季节、时间等辅助信息，用周期性编码 + 可训练 embedding 融合是成熟方案。

### 2.5 训练数据增强

文件：`fungiclef-effnet/datasets.py` → `get_train_transform()`

```yaml
# config.yaml train_aug:
trivial_aug: True          # TrivialAugmentWide
auto_aug: False
random_aug: False
gridmask_prob: 0.2         # GridMask 概率 20%
random_erasing_prob: 0.0
```

**增强 Pipeline（顺序）**：
1. **Bicubic Resize** 到 768（2倍于 crop 尺寸）
2. **RandomCrop** `384×384`（方形裁剪）
3. **TrivialAugmentWide**（自动选择增强操作和强度）
4. **HorizontalFlip**（50% 概率，通过默认 transform）
5. **GridMask**（20% 概率）— 文件：`augmentations.py`

> **关键细节**：Resize 到 2 倍尺寸再做 RandomCrop 是重要技巧。先用大尺寸 resize 保留更多细节，再随机裁剪目标尺寸，增加多样性。

### 2.6 训练超参数汇总

```yaml
# config.yaml train:
epochs: 200
lr: 1e-3                          # 前5个epoch只训练分类头
lr_after_unfreeze: 5e-5           # 解冻全部层后的学习率
optimizer: AdamW
weight_decay: 0.05
dropout_rate: 0.2
max_norm: 1.0                     # 梯度裁剪
early_stop_thresh: 10             # 连续10 epoch不改善则停止
fine_tune_after_n_epochs: 5       # 前5个epoch只训练分类头

# 学习率调度
lr_scheduler: "reducelronplateau"
lr_scheduler_patience: 4          # 连续4 epoch不改善则降低LR
lr_reduction_factor: 0.1          # LR降低为原来的1/10

# 数据
image_resize: 384
validation_frac: 0.1
undersample: False
oversample: False
equal_undersampled_val: True      # 验证集每类4个样本
balanced_sampler: False

# 数据加载
num_dataloader_workers: 12
worker_timeout_s: 360
```

> **借鉴（两阶段训练）**：
> - Stage 1（epoch 1-5）：冻结骨干网络，只训练分类头，lr=1e-3
> - Stage 2（epoch 6+）：解冻全部层，lr=5e-5，开启 ReduceLROnPlateau
> - 这是细粒度迁移学习的标准策略

---

## 3. 训练策略

### 3.1 两阶段闭集训练

文件：`fungiclef-effnet/train.py`

```
Phase 1 (epoch 1-5):
  - 冻结骨干网络参数（pretrained weights）
  - 只训练分类头
  - AdamW, lr=1e-3
  - 不启用 LR finder

Phase 2 (epoch 6-200):
  - 解冻全部参数
  - 手动降低 lr 到 5e-5
  - ReduceLROnPlateau，patience=4，factor=0.1
  - Early Stopping，patience=10
  - 梯度裁剪 max_norm=1.0
```

**检查点保存逻辑**：当 validation loss 改善时保存最佳模型。

### 3.2 OpenWGAN-GP 训练策略

文件：`fungiclef-effnet/train_opengan_from_cached_embeddings.py`

#### 3.2.1 整体流程

```
1. 训练闭集分类器 (train.py)
2. 用闭集分类器提取 penultimate layer 特征 → 缓存为 .h5 文件
3. 训练 OpenWGAN-GP 判别器 + 生成器
4. 基于验证集 macro-F1 选择最佳判别器
5. 组合为推理模型
```

对应代码入口：`create_opengan_discriminator.py` → `train_and_select_discriminator()`

#### 3.2.2 特征提取

文件：`create_embeddings_openset_recognition.py`

```yaml
open-set-recognition:
  openset_n_train: 2000         # 开放集训练样本数
  openset_n_val: 200            # 开放集验证样本数
  closedset_n_train: 50000      # 闭集训练样本数
  openset_oversample_rate: 1    # 开放集过采样倍数
  closedset_oversample_rate: 1  # 闭集过采样倍数
```

**最佳采样策略**（论文 Table 3）：
- **闭集**：weighted undersampling → ~60K 样本
- **开放集**：3× oversampling + 训练增强 → ~57K 样本
- 两个数据集样本数接近时效果最佳

#### 3.2.3 WGAN-GP 训练

```yaml
open-set-recognition:
  dlr: 1e-4                     # 判别器学习率
  glr: 1e-4                     # 生成器学习率
  epochs: 100
  batch_size: 128
  noise_vector_size: 100        # 噪声向量维度（生成器输入）
  hidden_dim_g: 64              # 生成器隐藏维度倍数
  hidden_dim_d: 64              # 判别器隐藏维度倍数
  openset_label: 1              # open-set 标签（同时用作 real label）
  closedset_label: 0
```

**WGAN-GP 关键实现**（`train_opengan_from_cached_embeddings.py`）：

```python
# Adam 优化器（WGAN-GP 推荐 betas）
optimizerD = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.0, 0.9))
optimizerG = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.0, 0.9))

# WGAN-GP loss
d_real = discriminator(real).view(-1)
d_generated = discriminator(fake.detach()).view(-1)
gradient_penalty = calc_gradient_penalty(discriminator, real, fake, device, gp_weight=10.0)
dis_loss = d_generated.mean() - d_real.mean() + gradient_penalty

# 生成器 loss
gen_loss = -discriminator(fake).view(-1).mean()

# 判别器更新 2 次，生成器更新 1 次（critic_iter=2）
```

**核心改进 vs. 原始 OpenGAN**：
1. **BatchNorm → LayerNorm**（判别器，`openset_recognition_models.py` → `LayerNormDiscriminator`）
2. **BCE loss → Wasserstein loss + Gradient Penalty**
3. **Open-set 标签 = Real 标签**（生成器学习生成 open-set 特征，而非 closed-set 特征）
4. **ROC-AUC 选模 → Macro-F1 选模**
5. **β1=0.0**（WGAN-GP 推荐，稳定性更好）

#### 3.2.4 判别器架构

`openset_recognition_models.py` → `LayerNormDiscriminator`：
```
Linear(D → H*8) → LayerNorm → LeakyReLU(0.2)
→ Linear(H*8 → H*4) → LayerNorm → LeakyReLU(0.2)
→ Linear(H*4 → H*2) → LayerNorm → LeakyReLU(0.2)
→ Linear(H*2 → H) → LayerNorm → LeakyReLU(0.2)
→ Linear(H → 1) → Sigmoid
```
其中 D = embedding 维度，H = hidden_dim（默认 64）

#### 3.2.5 判别器选择

文件：`choose_openset_recognition_discriminator.py`

```python
# 使用 macro-F1（而非 ROC-AUC）作为选择指标
f1_macro = f1_score(labels, (preds > 0.5).astype(int), average='macro')
if f1_macro > best_f1:
    best_f1 = f1_macro
    best_discriminator_path = discriminator_path
```

> **设计理由**：用 macro-F1 选模后，集成时可以直接平均概率而无需校准阈值。

#### 3.2.6 隐藏维度消融实验（论文 Table 4）

| 隐藏维度倍数 dim | Track1↓ | Track2↓ | Track3↓ | F1↑ |
|:---:|:---:|:---:|:---:|:---:|
| 64 | 0.3614 | 0.2109 | 0.5723 | 47.56 |
| 26 | 0.3622 | 0.1907 | **0.5529** | **47.76** |
| 32 | 0.2838 | 0.4481 | 0.7319 | 46.24 |

**结论**：小 hidden dim (26-32) 可能略好，但差异不大；64 作为保守默认值。

---

## 4. 推理优化

### 4.1 推理 Pipeline

文件：`create_opengan_discriminator.py` → `CompositeOpenGANInferenceModel`

```python
def forward(self, image, metadata_row=None):
    # Step 1: 提取 penultimate layer 特征
    intermediate_features = self.model.forward_features(image, ...)
    final_features = self.model.forward_head(intermediate_features, pre_logits=True)
    
    # Step 2: 闭集分类
    model_probas = self.model.forward_head(intermediate_features, pre_logits=False)
    model_preds = torch.argmax(model_probas, dim=1)
    
    # Step 3: 开放集判别
    opengan_probas = self.opengan_discriminator(final_features)
    
    return model_preds, opengan_probas
```

### 4.2 Test-Time Augmentation

论文 Table 5 结果（Metaformer-0）：

| Multi-instance | HFlip | Image Size | Track3↓ | F1↑ |
|:---:|:---:|:---:|:---:|:---:|
| ❌ | ❌ | 384 | 0.6093 | 41.64 |
| ✅ | ❌ | 384 | 0.6041 | 41.44 |
| ❌ | ✅ | 384 | 0.4852 | **45.14** |
| ✅ | ✅ | 384 | **0.4795** | **45.14** |

**TTAs 效果排序**（重要性递减）：
1. **Multi-instance averaging**（最大收益）— 同一观察的多张图片平均概率
2. **Horizontal flip averaging**（显著收益）— 原图 + 水平翻转平均
3. **FixRes**（提高推理分辨率）— CAFormer-S18 从 384 到 576，改善 Track 1 但可能降低 Track 2
4. **Multi-crop averaging**（收益较小）

### 4.3 推理 Pipeline 完整流程

```
观测 (Observation)
  ├─ 实例 1: [原图, 水平翻转] → 6 张图
  ├─ 实例 2: [原图, 水平翻转]
  └─ 实例 3: [原图, 水平翻转]
         ↓
  所有图片输入每个闭集分类器
         ↓
  闭集概率平均 → 集成平均 → 最终闭集概率
         ↓
  倒数第二层特征 → 开放集判别器平均 → 开放集概率
         ↓
  如果 top-1 预测是毒蘑菇 → 忽略开放集判别（不标为 unknown）
  如果 top-1 预测是食用菌 → 使用开放集判别决定是否为 unknown
```

### 4.4 "Ignore Poison Pred" 策略

**核心规则**：如果闭集分类器预测为毒蘑菇，**即使 OpenWGAN-GP 判定为 unknown 也不采纳**。

**原因**：评价指标中 unknown 隐式被认为是 edible，毒蘑菇 → unknown = 毒蘑菇 → edible，惩罚 ×100。

> **借鉴**：任何具有类别组不对称惩罚的任务都应考虑这种"安全优先"的决策覆盖策略。

---

## 5. 关键代码文件索引

| 文件 | 功能 |
|------|------|
| `train.py` | 闭集分类器训练主脚本 |
| `losses.py` | SeesawLoss、CompositeLoss（含 poison loss + LogitNorm） |
| `datasets.py` | 数据集类、增强 transform、元数据编码 |
| `augmentations.py` | GridMask 实现 |
| `closedset_model.py` | 模型构建（Metaformer / CAFormer）、预训练权重加载 |
| `openset_recognition_models.py` | 判别器（LayerNormDiscriminator/Discriminator）、生成器 |
| `create_embeddings_openset_recognition.py` | 从闭集分类器提取特征并缓存为 .h5 |
| `train_opengan.py` | 原始 OpenGAN 训练（实时提取特征） |
| `train_opengan_from_cached_embeddings.py` | **OpenWGAN-GP 训练（WGAN loss + GP）** |
| `choose_openset_recognition_discriminator.py` | 基于 macro-F1 选择最佳判别器 |
| `create_opengan_discriminator.py` | 整合训练+选模+组合为推理模型的编排脚本 |
| `evaluate.py` | 评估入口（含温度校准、OpenWGAN-GP 评估） |
| `temperature_scaling.py` | Temperature Scaling 实现 |
| `competition_metrics.py` | FungiCLEF 官方评估指标 |
| `paths.py` | 路径配置 |
| `conf/config.yaml` | Hydra 配置文件（所有超参数集中管理） |

---

## 6. 对 image-classification 项目的借鉴

### 6.1 可直接引入的技术

| 技术 | 适用场景 | 实现难度 | 优先级 |
|------|----------|----------|--------|
| **Seesaw Loss** | 4万类长尾分类 | ⭐⭐ 中等 | 🔴 高 |
| **LogitNorm** | 细粒度 + 开放集 | ⭐ 简单 | 🟡 中 |
| **两阶段训练** | 任何迁移学习 | ⭐ 简单 | 🔴 高 |
| **TrivialAugmentWide** | 通用数据增强 | ⭐ 简单 | 🟡 中 |
| **GridMask** | 细粒度/遮挡场景 | ⭐ 简单 | 🟢 低 |
| **Bicubic Resize(2×)+RandomCrop** | 增强多样性 | ⭐ 简单 | 🟡 中 |
| **Multi-instance averaging** | 多图同物场景 | ⭐⭐ 中等 | 🟢 低 |
| **Horizontal Flip TTA** | 通用 | ⭐ 简单 | 🔴 高 |
| **FixRes** | Transformer 架构 | ⭐ 简单 | 🟡 中 |
| **类组加权损失** | 有类别组代价 | ⭐⭐ 中等 | 🟢 低 |

### 6.2 长尾分类核心建议

```
Seesaw Loss > Focal Loss > 各类加权 CrossEntropy
```

**推荐的 Seesaw Loss 配置**：
```python
SeesawLoss(num_classes=N_CLASSES, p=0.8, q=2.0, eps=1e-2)
```

### 6.3 开放集检测（如果需要）

如果需要检测"不属于训练类的"输入，OpenWGAN-GP 是一个轻量方案：

1. 训练闭集分类器完成后，提取 penultimate layer 特征
2. 收集一些已知的 out-of-distribution 样本作为开放集
3. 训练 MLP 判别器（WGAN-GP 模式）
4. 推理时用判别器输出阈值 0.5 判断 known/unknown

**最小判别器配置**：
```python
LayerNormDiscriminator(nc=embedding_dim, hidden_dim=32)  # 小 hidden dim 足够
```

### 6.4 训练策略建议

```yaml
# 推荐的两阶段训练配置
train:
  epochs: 200
  fine_tune_after_n_epochs: 5    # 前5 epoch 只训练分类头
  lr: 1e-3                       # 第一阶段 LR
  lr_after_unfreeze: 5e-5        # 第二阶段 LR
  lr_scheduler: "reducelronplateau"
  lr_scheduler_patience: 4
  early_stop_thresh: 10
  optimizer: AdamW
  weight_decay: 0.05
  dropout_rate: 0.2
  max_norm: 1.0                  # 梯度裁剪
```

### 6.5 集成策略

- **同架构多划分** > 单模型 — 用不同 train/val split 训练同一架构的多个副本
- **轻量模型集成** > 单个大模型 — 训练和推理成本更低
- **简单平均集成** 已足够有效，无需复杂加权

### 6.6 open-set 训练的采样策略

当开放集样本远少于闭集时：
- 闭集：**weighted undersampling**
- 开放集：**3× oversampling + data augmentation**
- 目标：让两个数据集样本数接近

### 6.7 WGAN-GP 训练要点

如果用 GAN 做特征生成：
1. 判别器用 **LayerNorm** 而非 BatchNorm
2. Adam 优化器设 **betas=(0.0, 0.9)**
3. **Gradient Penalty weight = 10.0**
4. 判别器更新 2-5 次，生成器更新 1 次（critic_iter）
5. 生成器的学习率约为判别器的 1-2 倍
6. 开放集标签应与 "real" 标签一致（生成器学习开放集分布）

---

## 7. 局限性与注意事项

1. **OpenWGAN-GP 需要开放集样本**：无法完全无监督检测，需要一定数量的 out-of-distribution 样本
2. **评价指标与安全目标的矛盾**：当前指标假设 unknown = edible，实际应假设 unknown = poisonous
3. **LogitNorm + Poison Loss 可能存在冲突**：论文实验显示 LogitNorm 提升 Track 1 但降低 Track 2
4. **推理分辨率提升不总有效**：FixRes 对 CAFormer-S18 的 Track 2 和 F1 有负面影响
5. **EfficientNet 在此任务上不如 Metaformer/CAFormer**：hybrid Conv-Transformer 架构更适合细粒度
