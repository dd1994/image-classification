# 03 — Bag of Tricks and a Strong Baseline for FGVC

> **论文**: *Bag of Tricks and a Strong Baseline for Fine-Grained Visual Classification*
> Jun Yu, Hao Chang, Keda Lu, Guochen Xie, Liwen Zhang, et al. (USTC)
> CVPR 2022 Workshop — FungiCLEF 2022 Challenge, **第二名**
>
> **代码仓库**: https://github.com/wujiekd/Bag-of-Tricks-and-a-Strong-Baseline-for-Fungi-Fine-Grained-Classification
> **框架**: MindSpore 1.8.1

---

## 1. 论文概述

### 1.1 任务与数据

- **任务**: 真菌细粒度图像分类（FGVC），属于超大类下的子类识别
- **数据集**: FungiCLEF 2022
  - 训练集：295,938 张图像，1,604 个物种（class）
  - 测试集：59,420 个观测（observation），118,676 张图像，3,134 个物种（含开放集，需预测 -1）
  - 核心难点：(1) **细粒度特征** — 类间差异小、类内差异大；(2) **长尾分布** — 极度类别不平衡；(3) **开放集** — 测试集含有训练时未见过的物种
- **评估指标**: Macro F1-Score（不受类别频率影响，适合长尾分布）

### 1.2 方法论框架

论文的核心思路是**系统性地探索图像分类各组件在 FGVC 任务上的组合效果**，而非提出新的模型架构。整体框架如图：

```
原始图像 + 元数据
    ↓
[数据增强] → AutoAugment / RandAugment / TrivialAugment / CutMix
    ↓
[骨干网络] → CNN (EfficientNet) / Transformer (Swin, BEiT)
    ↓
[损失函数] → CE / FocalLoss / SeesawLoss / Label Smoothing
    ↓
[精细化] → PIM (注意力插件) / Two-Stage Training (不平衡→平衡微调)
    ↓
[后处理] → 元数据条件概率 / TTA / 模型融合
    ↓
最终预测
```

### 1.3 最终成绩

| 阶段 | 排名 | F1 Score |
|------|------|----------|
| Public Leaderboard | 第3 (实际第2) | 83.12% |
| Private Leaderboard | 第3 (实际第2) | 79.06% |

---

## 2. 识别率提升技巧（含超参数）

论文的核心贡献是对各类"trick"进行了系统消融实验，所有实验基于 **SwinTransformer** 作为 baseline（Public F1 = 75.01%）。

### 2.1 数据增强策略

#### 基础增强配置（所有实验共用）

| 参数 | 值 | 说明 |
|------|-----|------|
| `hflip` | 0.5 | 随机水平翻转概率 50% |
| `vflip` | 0.5 | 随机垂直翻转概率 50% |
| `scale` | [0.8, 1.0] | 随机裁剪缩放范围 |
| `color_jitter` | 0.2 ~ 0.4 | 颜色抖动强度（基础训练 0.2，高级增强 0.4） |
| `reprob` | 0.25 | 随机擦除概率 |
| `smoothing` | 0.1 | 标签平滑 |

> ⚠️ 注意：此处的 `scale=[0.8, 1.0]` 与 ImageNet 标准的 `[0.08, 1.0]` 不同。论文针对 FGVC 任务采用了**更保守的裁剪范围**，这对于细粒度任务至关重要——过小的裁剪可能丢失关键的判别性区域。

#### 增强方法消融实验 (Table 2)

所有实验基于 SwinTransformer baseline (F1=75.01%)：

| 增强方法 | F1 Score | 提升 | 说明 |
|----------|----------|------|------|
| Baseline（基础增强） | 75.01% | — | hflip+vflip+scale+color_jitter |
| + **RandAugment** | 75.42% | +0.41% | NAS-based，需搜索 |
| + **TrivialAugment** | 75.50% | +0.49% | 无参数，单操作/图 |
| + **CutMix + Random Erasing** | 75.59% | +0.58% | CutMix α=1.0 + reprob=0.25 |

**关键洞察**:
- **TrivialAugment (TA)** 被最终采用，因其无参数、搜索成本几乎为零且效果最好
- CutMix（α=1.0）与 Random Erasing（概率 0.25）的组合带来最大提升
- **AutoAugment 未被使用**：需要单独搜索阶段，计算成本高

#### TrivialAugment 具体机制

```
对每张图像：
  1. 均匀随机选择一个增强操作（从预定义的图像处理操作集合中）
  2. 均匀随机采样一个强度值 m
  3. 仅应用这一种增强操作
```

操作集合包括：平移、旋转、剪切、锐度、亮度、对比度、色调、均衡化、太阳化、反转等。与 RandAugment（每图 2 个操作）不同，TA 每图仅 1 个操作。

### 2.2 二阶段训练与注意力机制 (Table 3)

| Stage2 方法 | F1 Score | 提升 |
|-------------|----------|------|
| Baseline | 75.01% | — |
| **PIM** (Plugin Module) | 75.89% | +0.88% |
| **DRS** (Deferred Rebalancing) | 75.82% | +0.81% |

#### PIM — 即插即用细粒度注意力模块

- **全称**: Plugin Module for Fine-Grained Visual Classification
- **原理**: 从 backbone 的中间层输出特征图 → 弱监督选择器过滤 → 保留最具判别力的区域 → 组合器融合特征 → 分类
- **特点**: 可插入任意 CNN 或 Transformer backbone，输出像素级特征图，自动找到最具判别力的图像区域
- **遗憾**: 由于计算资源限制，最终方案中**未使用 PIM**

#### Two-Stage Training（长尾分布的核心解决方案）

这是论文**最终采用的最关键策略**，专门针对 Fungi 数据集的长尾分布：

**Stage 1 — 不平衡训练 (Unbalanced Training)**:
- 使用全部原始数据（保持不均衡分布）
- 不使用任何重采样/重加权
- 目的：学习**好的特征表示**（CNN 在不均衡数据上仍可学到强特征）

**Stage 2 — 均衡微调 (Balanced Fine-tuning)**:

论文探索了两种方案：

1. **DRS (Deferred Rebalancing via Resampling)**:
   - Stage 1 正常训练 → Stage 2 使用重采样构造均衡子集微调
   - 通过采样使每个类别出现频率大致相等

2. **DRW (Deferred Rebalancing via Reweighting)**:
   - Stage 1 正常训练 → Stage 2 使用重加权损失函数微调
   - 对尾部类别赋予更高权重

最终采用 **DRS**（效果略优于 DRW，且实现更简单）。

**代码中的实际训练管线**（5 步顺序训练）：

| 步骤 | 脚本 | 关键参数 | 说明 |
|------|------|----------|------|
| ① 基础训练 | `train.py` | lr=0.01, 300 epochs, sched=plateau, freeze_layer=2 | 9:1 划分训练/验证，无 CutMix/高级增强 |
| ② 增强微调 | `train.py` | lr=0.001, cutmix=1, color_jitter=0.4, reprob=0.25, aa=trivial, warmup_epochs=0 | 加载步骤①最佳 checkpoint，添加全部数据增强 |
| ③ 损失函数微调 | `train.py` | lr=0.001, Focalloss, 其余同② | 加载步骤②最佳 checkpoint，切换 FocalLoss |
| ④ 全量数据训练 | `train_all.py` | lr=0.001, epochs=24, sched=multistep, decay_rate=0.1 | 加载步骤③最佳 checkpoint，训练集+验证集合并 |
| ⑤ 均衡微调 (DRS) | `train_all.py` | lr=0.001, epochs=5, balanced data | 加载步骤④最佳 checkpoint，使用均衡子集微调 |

### 2.3 损失函数 (Table 4)

| 损失函数 | F1 Score | 提升 | 说明 |
|----------|----------|------|------|
| Baseline (CE + Label Smoothing=0.1) | 75.01% | — | 标准交叉熵 + 标签平滑 |
| **FocalLoss** | **75.89%** | **+0.88%** | 降低易分样本权重，聚焦难分样本 |
| SeesawLoss | 75.45% | +0.44% | 动态平衡正负样本梯度 |

#### FocalLoss 参数

```python
FocalLossWithSmoothing(
    num_classes=1604,
    gamma=2,        # 聚焦参数，越大越聚焦难分样本
    ignore_index=0,
    alpha=0.25      # 类别权重平衡因子
)
```

**Focal Loss 公式**:
```
FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
```
- `p_t`：模型对正确类别的预测概率
- `γ=2`：高置信度易分样本的 loss 被大幅压制
- `α=0.25`：进一步平衡正负样本

**为什么 FocalLoss 对 FGVC 有效**: 细粒度分类中，部分类别之间视觉差异极小（"难分样本"），模型对它们的预测置信度低。Focal Loss 自动让模型更关注这些困难样本的梯度。

#### SeesawLoss

```python
SeesawLossWithLogits(class_counts=[0..1603])
```

通过动态降低头部类别对尾部类别施加的负样本梯度权重，实现正负样本梯度的相对平衡。效果不如 FocalLoss。

### 2.4 骨干网络对比 (Table 5)

| Backbone | 输入尺寸 | Private F1 | Public F1 |
|----------|----------|------------|-----------|
| EfficientNet-B6 | 600×600 | 76.57% | 81.58% |
| EfficientNet-B7 | 600×600 | 76.91% | 80.73% |
| SwinTransformer-Large | 384×384 | 76.96% | 80.02% |
| SwinTransformer-Base | 384×384 | 76.79% | 79.98% |
| **BEiT-Large** | **512×512** | **77.64%** | **80.48%** |

> ⚠️ 上表是使用**完整训练策略**（FocalLoss + 数据增强 + 二阶段训练）后的结果。BEiT 作为单模型最优，但 Swin 系列整体表现更好。

**关键发现**:
1. **Transformer > CNN**: Swin/BEiT 整体优于 EfficientNet（Private 上更明显）
2. **BEiT 最佳**: BERT-style 预训练（masked image modeling）优于有监督预训练
3. 输入分辨率：CNN 用 600×600，Transformer 用 384~512
4. 所有模型使用 **freeze_layer=2**（冻结前 2 个 stage），ImageNet-1K 预训练权重初始化

### 2.5 元数据利用 (Table 6)

FungiCLEF 2022 提供了丰富元数据（月、日、国家、位置层级、基质、栖息地等）：

| 元数据使用方法 | F1 Score | 效果 |
|---------------|----------|------|
| Baseline（仅图像） | 75.01% | — |
| 条件概率后处理 | 73.70% | ❌ 下降 |
| MLP 交互学习 | 74.55% | ❌ 下降 |
| MetaFormer | 训练困难/梯度爆炸 | ❌ 失败 |

#### 唯一有效的元数据技巧

在测试集的 `substrate`（基质）属性中，发现了一个训练集中**不存在的值** "spiders"（蜘蛛）。将 `substrate=spiders` 的测试图像类别预测强制设为 `-1`（开放集/未知类），获得了小幅提升。

**结论**: 由于训练集和测试集的元数据分布差异大，元数据在 Fungi 数据集上未能有效利用。

### 2.6 模型集成 (Table 7)

| 融合层级 | 方法 | F1 Score |
|----------|------|----------|
| Softmax 输出层 | 简单平均 (Simple Average) | 80.98% |
| 特征输出层 | Concat + MLP | 80.91% |

**最终方案**: 使用**多种 CNN 和 Transformer 模型**提取特征/输出，在 Softmax 层进行**简单平均融合**（简单且效果最好）。

最终提交的模型集成方案使用：
- SwinTransformer Large/Base
- EfficientNet B6/B7
- BEiT Large

### 2.7 推理优化

#### TTA (Test Time Augmentation)

```python
# test.py 中的实现
TTA = True
# 10-Crop 验证
crop_pct = 1.0  # 使用完整图像进行多裁剪
```

**10-Crop 流程**:
1. 原始图像 + 水平翻转
2. 四个角裁剪 + 中心裁剪（5 个位置 × 2 种翻转 = 10 个 crops）
3. 对 10 个 crops 的 softmax 输出取平均

#### 观测级聚合 (Observation-level Aggregation)

FungiCLEF 2022 的特殊性：一个观测（observation）可能包含多张图像。测试需聚合同一观测的所有图像预测：

```python
# 按 ObservationId 分组取平均
group_scores = scores.groupby(['ObservationId']).mean().reset_index()
```

#### 开放集处理

对不在训练集 1604 类中的物种，预测 `-1`。

---

## 3. 训练策略总结

### 3.1 完整训练参数

| 参数 | 值 | 适用阶段 |
|------|-----|----------|
| 优化器 | SGD | 全部 |
| 动量 (momentum) | 0.9 | 全部 |
| 权重衰减 (weight_decay) | 2e-5 | 全部 |
| 初始学习率 | 0.01 | Stage 1 |
| 微调学习率 | 0.001 | Stage 2-5 |
| Batch Size | 4~64（按 GPU 显存） | 全部 |
| 学习率调度 | Plateau (Stage 1) / MultiStep (Stage 4-5) | 分阶段 |
| decay_rate | 0.5 (Stage 1) / 0.1 (Stage 4) | 分阶段 |
| Warmup Epochs | 3 (Stage 1) / 0 (Stage 2-5) | 分阶段 |
| Mixed Precision | Apex AMP (O1) | 全部 |
| 随机种子 | 42 | 全部 |

### 3.2 学习率调度策略

**Stage 1 (基础训练) — ReduceLROnPlateau**:
```yaml
sched: plateau
patience_epochs: 1    # 验证 loss 1 个 epoch 不降即降 lr
decay_rate: 0.5       # 降至 50%
```

**Stage 4-5 (全量数据训练) — MultiStepLR**:
```yaml
sched: multistep
decay_epochs: 24       # 在指定 epoch 降 lr（实际使用 checkpoint-hist 全保存）
decay_rate: 0.1        # 降至 10%
```

### 3.3 Freeze Layer 策略

```python
freeze_layer = 2  # 冻结 backbone 前 2 个 stage
```

- SwinTransformer: 4 个 stage，冻结前 2 个（patch embedding + stage 1-2）
- EfficientNet: 冻结前 2~6 层
- 目的：保留 ImageNet 预训练的低层特征，仅微调高层

### 3.4 数据加载

- 使用 MindSpore 框架的 `mindcv` 工具包
- GPU: Tesla A100
- 多卡训练: `torch.distributed.launch --nproc_per_node=4`
- SyncBN 用于多卡训练

---

## 4. 推理优化

### 4.1 测试配置

| 参数 | 值 |
|------|-----|
| TTA | 10-Crop |
| crop_pct | 1.0（全图裁剪） |
| 输入尺寸 | 与训练一致 (384/512/600) |
| 聚合方式 | 观测级 softmax 平均 |
| 开放集处理 | substrate=spiders → -1 |

### 4.2 推理流程

1. 加载最终 checkpoint（通常为第 3~5 个 epoch 的均衡微调 checkpoint）
2. 对每张测试图像执行 10-Crop
3. 多模型分别推理并保存 logits
4. 在模型集成级别（`model_ensemble.ipynb`）执行 softmax 平均融合
5. 按 ObservationId 聚合多张图像预测
6. 提交 CSV（ObservationId, ClassId）

---

## 5. 对 image-classification 项目的借鉴

### 5.1 直接可用的 Trick（高优先级）

| # | Trick | 价值 | 实施难度 | 建议 |
|---|-------|------|----------|------|
| 1 | **Two-Stage Training (DRS)** | ⭐⭐⭐⭐⭐ | 中 | 长尾分布数据集必用。先不平衡训练学特征，再用均衡子集微调分类头 |
| 2 | **FocalLoss (γ=2, α=0.25)** | ⭐⭐⭐⭐⭐ | 低 | 替代标准 CE。对任何类不平衡 + 困难样本场景有效 |
| 3 | **TrivialAugment** | ⭐⭐⭐⭐ | 低 | 替代 RandAugment，无参数，搜索成本为零 |
| 4 | **冻结前几层 (freeze_layer)** | ⭐⭐⭐⭐ | 低 | 迁移学习必备，ImageNet 预训练低层特征泛化性好 |
| 5 | **CutMix (α=1.0) + RandomErasing (p=0.25)** | ⭐⭐⭐⭐ | 低 | 常规增强之外的最大提升来源 |
| 6 | **Label Smoothing (ε=0.1)** | ⭐⭐⭐ | 极低 | 一行代码，稳定训练，防止过拟合 |
| 7 | **保守的 RandomResizedCrop (scale=[0.8,1.0])** | ⭐⭐⭐⭐ | 极低 | FGVC 场景下比标准 [0.08,1.0] 好得多——微区别在细节里 |

### 5.2 架构选择建议

- **Transformer > CNN**：Swin Transformer/BEiT 在 FGVC 上优于 EfficientNet。如果计算资源充足，优先考虑 ViT 系列
- **BEiT 预训练方式**：Masked Image Modeling (MIM) 比 ImageNet 有监督预训练更适合 FGVC
- **多模型集成**：CNN + Transformer 互补，简单平均优于复杂融合

### 5.3 训练策略建议

```
┌──────────────────────────────────────────────────────────────┐
│ 推荐 FGVC 训练管线 (基于本文最佳实践)                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: 预训练 Backbone + 基础增强 + CE Loss                 │
│          • lr=0.01, plateau scheduler                        │
│          • freeze first 2-3 stages                           │
│          • hflip, vflip, scale=[0.8,1.0], color_jitter=0.2   │
│          • Label Smoothing=0.1                               │
│          • 验证集 9:1 划分                                    │
│                                                              │
│  Step 2: 添加高级增强 → 微调                                  │
│          • lr=0.001, 加载 Step 1 checkpoint                  │
│          • + TrivialAugment                                  │
│          • + CutMix (α=1.0)                                  │
│          • + RandomErasing (p=0.25)                          │
│          • + color_jitter=0.4                                │
│                                                              │
│  Step 3: 切换 FocalLoss → 微调                               │
│          • lr=0.001, 加载 Step 2 checkpoint                  │
│          • FocalLoss(γ=2, α=0.25)                            │
│                                                              │
│  Step 4: 全量数据训练                                         │
│          • lr=0.001, MultiStep scheduler                     │
│          • 训练集 + 验证集合并                                 │
│          • epochs=20~24, warmup=0                            │
│                                                              │
│  Step 5: 均衡微调 (DRS)                                      │
│          • lr=0.001, epochs=5                                │
│          • 使用均衡采样子集 (每类 N 张)                        │
│          • 仅微调分类头 (可选)                                 │
│                                                              │
│  推理: TTA 10-Crop → Softmax Average → Ensemble              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 5.4 需要注意的坑

| 陷阱 | 说明 |
|------|------|
| **元数据不可靠** | 训练集和测试集的元数据分布可能差异巨大，不要盲目相信 |
| **过度调参** | 增强参数应简单化，TrivialAugment 的无参数设计是优势 |
| **PIM 计算开销大** | 注意力插件效果不错但耗时，实际工程中可能不划算 |
| **输入尺寸** | FGVC 需要更大的输入尺寸（384+），标准 224 可能不够 |
| **开放集** | 生产环境必须考虑未知类别的处理 |

### 5.5 关键超参数速查表

| 超参数 | 推荐值 | 适用场景 |
|--------|--------|----------|
| FocalLoss gamma | 2.0 | 通用长尾 |
| FocalLoss alpha | 0.25 | 通用长尾 |
| Label Smoothing | 0.1 | 通用 |
| CutMix alpha | 1.0 | FGVC + 不平衡 |
| Random Erasing prob | 0.25 | FGVC |
| RandomResizedCrop scale | [0.8, 1.0] | FGVC 特化 |
| Color Jitter | 0.2 (基础) / 0.4 (高级) | 渐进式 |
| Weight Decay | 2e-5 | 通用 |
| Freeze Layers | 前 2 个 stage | 迁移学习 |
| TTA | 10-Crop | 测试时 |
| Batch Size | 尽可能大 (32-64 per GPU) | 稳定训练 |

---

## 6. 总结

这篇论文的标题恰如其分——它确实是 FGVC 领域的 **"Bag of Tricks"**，没有提出任何新的网络架构，而是系统地评估了数据增强、损失函数、训练策略、骨干网络、元数据利用等各方面因素。核心结论：

1. **Two-Stage Training (DRS)** 是解决长尾 FGVC 的最有效方法
2. **FocalLoss** 在 FGVC 场景下显著优于标准交叉熵
3. **TrivialAugment + CutMix + RandomErasing** 是最佳增强组合
4. **Transformer 骨干优于 CNN**，BEiT 的 MIM 预训练特别有效
5. **模型集成** 简单平均即最佳
6. **元数据在本任务上未能有效利用**（训练/测试分布差异）

这对 `image-classification` 项目的直接指导意义：应该将上述训练管线（Step 1→5）作为 FGVC 实验的默认配置，并在此基础上进行改进。
