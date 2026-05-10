# 00-MASTER-SUMMARY — 图像分类调研总汇与差距分析


> 基于 9 篇细粒度/长尾/开放集图像分类论文的系统性分析 (``D:\fgvc-survey`文件夹里包含所有论文和代码)
> 目标项目：`D:\image-classification` — PyTorch Lightning + SwinV2 base, 44K 物种细粒度分类

---

## 一、调研总览

### 1.1 9 篇论文概览表

| # | 论文简称 | 竞赛/场景 | 数据集规模 | 核心方法 | 最终成绩 | 年份 |
|---|---------|----------|-----------|---------|---------|------|
| 01 | **FungiCLEF 2022 1st** | 真菌细粒度+开放集+长尾 | 296K 训练, 1604 类 | MetaFormer+SeesawLoss+Ensemble+后处理 | Private F1: 80.43% (🥇) | 2022 |
| 02 | **FungiCLEF 2023 3rd** | 真菌细粒度+开放集+长尾 | 296K 训练, 1604 类 | VOLO+SeesawLoss+RandomMix+TTA | Private F1: 54.34% (🥉) | 2023 |
| 03 | **Bag of Tricks FGVC** | 真菌细粒度+长尾 | 296K 训练, 1604 类 | FocalLoss+Two-Stage DRS+TrivialAugment+CutMix+BEiT/Swin | Private F1: 79.06% (🥈) | 2022 |
| 04 | **Entropy-Guided Open-Set** | 真菌细粒度+开放集+长尾 | 296K 训练, 1604 类 | MetaFormer+元数据融合+熵引导开放集+毒蘑菇损失 | Private F1: 58.36% (🥇) | 2023 |
| 05 | **Long-Tailed FGVC** | 蛇类细粒度+长尾+代价敏感 | 182K 训练, 1784 类 | CAFormer+SeesawLoss+VenomLoss+高分辨率推理+多划分Ensemble | Track1: 79.96 (🥈) | 2024 |
| 06 | **Metaformer+ArcFace+Contrastive** | 蛇类细粒度+长尾+多模态 | 180K 训练, 1785 类 | MetaFG+ArcFace+SimCLR对比学习+Meta融合 | Track1: 88.30% (🥉) | 2023 |
| 07 | **OpenWGAN-GP** | 真菌细粒度+开放集+长尾+毒蘑菇 | 296K 训练, 1604 类 | CAFormer/Metaformer+SeesawLoss+LogitNorm+WGAN-GP开放集+GridMask | Private F1: 56.79% (🥇) | 2024 |
| 08 | **Venomous Snake SnakeCLEF2023** | 蛇类细粒度+长尾+毒蛇代价 | 182K 训练, 1784 类 | ConvNeXt-v2+SeesawLoss+金字塔元数据融合+Prior Model+CutMix | Private: 91.31% (🥇) | 2023 |
| 09 | **Large Kernel ViT CoLKANet** | 蛇+真菌细粒度+长尾+开放集 | 296K+318K 训练 | CoLKANet+TrivialAugmentWide+LabelAwareSmoothing+EMA+5-fold Ensemble | F1: 85.4%/78.9% | 2022 |

### 1.2 关键共性

- **8/9 篇论文来自竞赛** (CLEF LifeCLEF, FGVC Workshop at CVPR)，实践驱动，消融实验充分
- **全部面临长尾分布**，多数同时面临细粒度+开放集
- **7/9 篇使用 Transformer 或 Hybrid 架构** (Swin, BEiT, MetaFormer, VOLO, ConvNeXt-v2, CAFormer, CoLKANet)
- **6/9 篇使用 Seesaw Loss** 作为核心长尾解决方案
- **全部使用 Model Ensemble** 提升最终性能

---

## 二、核心技巧跨论文对比

### 2.1 损失函数

| 技巧名称 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 入选次数 |
|---------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:------:|
| **Seesaw Loss** | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | **6** |
| CrossEntropy + Label Smoothing | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | **6** |
| Focal Loss | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 1 |
| ArcFace Loss | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | 2 |
| Poison/Venom Cost Matrix Loss | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | **4** |
| LogitNorm | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | 2 |
| NT-Xent Contrastive (SimCLR) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | 1 |
| LabelAwareSmoothing | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | 1 |
| Weighted CE (类频率倒数) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 0 (❌ 反模式) |

### 2.2 数据增强

| 技巧名称 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 入选次数 |
|---------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:------:|
| **TrivialAugment(Wide)** | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | **4** |
| RandAugment | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | 3 |
| AutoAugment | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 0 (成本高) |
| **CutMix** | 最终不用 | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | **5** |
| Mixup | 最终不用 | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | **4** |
| TokenMix | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | 2 |
| RandomMix (随机选) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | 2 |
| **Random Erasing** | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | **5** |
| GridMask | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | 1 |
| **Color Jitter** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | **6** |
| Horizontal Flip | ✅ | ✅ | ✅ | ✅ | 隐式 | ✅ | ✅ | ✅ | ✅ | **9 (全部)** |
| Vertical Flip | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | **4** |
| PiecewiseAffine | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | 2 |
| Bicubic Resize (2×) + RandomCrop | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | 2 |

### 2.3 训练策略

| 技巧名称 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 入选次数 |
|---------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:------:|
| **Cosine LR Schedule** | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | **6** |
| **Linear Warmup** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | **7** |
| ReduceLROnPlateau | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | 3 |
| MultiStep LR | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 1 |
| **Gradient Accumulation** | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | 2 |
| **Gradient Clipping** | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | **5** |
| **Mixed Precision (AMP/FP16)** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | **8** |
| **Two-Stage Training (freeze→unfreeze)** | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | 3 |
| DifferLR (backbone LR < head LR) | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | 2 |
| Freeze Layers (前N层冻结) | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | 2 |
| **Weight Decay 分离 (bias/norm=0)** | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | 3 |
| EMA (指数移动平均) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | 1 |
| 5-Fold Cross Validation | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | 1 |
| Large Batch Size | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 1 |

### 2.4 模型架构

| 技巧名称 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 入选次数 |
|---------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:------:|
| **Hybrid Conv+Transformer** (MetaFormer/CAFormer/CoLKANet) | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | **6** |
| ConvNeXt/ConvNeXt-v2 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | 3 |
| Swin Transformer | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | 2 |
| BEiT | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 1 |
| VOLO | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | 2 |
| EfficientNet | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | 2 |
| ViT | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | 2 |
| 元信息融合 (Meta Fusion) | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | **5** |
| 金字塔特征融合 (Mid+Final) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | 1 |
| Dropout | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ (0.6!) | ✅ | **5** |

### 2.5 推理优化

| 技巧名称 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 入选次数 |
|---------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:------:|
| **Model Ensemble (概率/投票平均)** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **9 (全部)** |
| **TTA (Multi-Crop/Flip)** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | **7** |
| High-Res Inference (FixRes) | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | 3 |
| Multi-Instance Averaging (多图同物) | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | **8** |
| 后处理 (Post-Process) | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | **5** |
| 伪标签 (Pseudo Labeling) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | 2 |
| 输出归一化 (Ensemble前) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | 1 |
| Open-Set 阈值/熵检测 | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | **5** |

---

## 三、高共识技巧（多篇论文共同推荐）

以下技巧在 **≥3 篇论文**中被使用或强烈推荐，具有最高置信度。

### 3.1 Model Ensemble — 全票通过的必杀技 ⭐⭐⭐⭐⭐

> 引用: 全部 9 篇论文

**描述**: 训练多个不同配置的模型（不同 backbone、不同数据划分、不同 loss、不同预训练），推理时将它们的 logit/probability 做算术平均。

**推荐做法**:
- **简单平均优于复杂加权** (01, 03, 04, 05, 09)
- **多样性来源**: 不同 backbone 架构 (CNN+Transformer 混合, 01, 03)、不同 train/val split (05, 09)、不同 loss 配置 (09 LabelAware+CE混合)、不同预训练 (01)
- **轻量模型集成 > 单大模型** (07 "Wisdom of Committees")
- 建议 3-6 个模型集成

**实现优先级**: 🔴 **P0** — 仅需训练多个 checkpoint + 后处理脚本

---

### 3.2 Seesaw Loss — 长尾分类的标准答案 ⭐⭐⭐⭐⭐

> 引用: 01, 02, 04, 05, 07, 08 (6/9 篇)

**描述**: 动态平衡长尾分布中头部类和尾部类的训练梯度。通过 Mitigation Factor（按类频率比降低尾部类的惩罚）和 Compensation Factor（对过自信的错分类增强惩罚）双因子实现。

**推荐超参数**:
| 参数 | 值 | 说明 |
|------|-----|------|
| `p` (mitigation factor 幂指数) | 0.8 | 全部论文一致 |
| `q` (compensation factor 幂指数) | 2.0 | 全部论文一致 |
| `eps` (数值稳定) | 1e-2 | 全部论文一致 |

**⚠️ 关键约束**:
- **不能与 Mixup/CutMix 混用** (01 论文 Table 2 明确实验: 加 Mixup 反而降低性能)
- 替代方案: 若要用混合增强 → 换 FocalLoss (03) 或 LabelAwareSmoothing (09)
- **远优于 Weighted CE** (02 论文: Weighted CE 反降 2.55%; 05 论文: Seesaw 比 Focal+Balanced 高 F1 +7)
- 需要传入各类别样本数统计

**代码参考**: `蛇论文 01/models/custom_loss.py` 或 `蛇论文 05/losses.py`

**实现优先级**: 🔴 **P0** — 独立 loss 模块, 一天可集成

---

### 3.3 TTA (Test Time Augmentation) — 推理免费午餐 ⭐⭐⭐⭐

> 引用: 01, 02, 03, 04, 06, 08, 09 (7/9 篇)

**描述**: 推理时对每张图像做多次增强取平均概率，提升预测稳定性。

**推荐做法** (按优先级):
1. **Horizontal Flip TTA** (05, 07) — 最简单, 原图+水平翻转平均
2. **Five-Crop TTA** (01, 04) — 四角+中心, 5 crops
3. **Multi-Scale TTA** (01) — 多尺度 (1.1×, 1.143×, 1.2×) × TenCrop = 30 crops
4. **Random Crop TTA** (02, 09) — 5-13 次随机裁剪取平均

**⚠️ 注意事项**:
- TTA 可能过拟合公共榜 (09 发现 SnakeCLEF 私榜无效)
- Multi-crop 显著增加推理时间 (01: train bs=18 → inference bs=6)
- 优先用 HFlip TTA (计算成本最低, 收益明确 05)

**实现优先级**: 🟡 **P1** — 修改推理代码, 1-2天

---

### 3.4 TrivialAugmentWide — 无参数数据增强标配 ⭐⭐⭐⭐

> 引用: 03, 05, 07, 09 (4/9 篇), 对比 02 的复杂手工增强组合

**描述**: 每张图像随机选择一个增强操作 + 随机强度，无参数、零搜索成本。论文实验优于 RandAugment (03) 和手工增强组合 (09)。

**推荐配置**:
```python
# timm 调用方式
from timm.data import create_transform
transform = create_transform(
    input_size=384,  # 或 448
    is_training=True,
    auto_augment='ta_wide',  # TrivialAugmentWide
    color_jitter=0.4,
    re_prob=0.25,
    re_mode='pixel',
    interpolation='bicubic',
)
```

**对比 AutoAugment**: 03 论文明确 "AutoAugment 未被使用: 需要单独搜索阶段, 计算成本高"

**实现优先级**: 🔴 **P0** — 替换现有增强配置, 半天

---

### 3.5 Random Erasing — 细粒度正则化标配 ⭐⭐⭐⭐

> 引用: 01, 03, 04, 06, 09 (5/9 篇), 但 05 明确反对

**描述**: 随机擦除图像矩形区域 (pixel-level 或 block-level)。

**推荐超参数**:
| 参数 | 多数论文值 | 说明 |
|------|-----------|------|
| `re_prob` | 0.25 | 多数论文一致 |
| `re_mode` | 'pixel' | 像素级擦除 |
| `re_count` | 1 | 每图 1 个擦除区域 |

**⚠️ 争议**: 05 论文 (Table 9) 发现 Random Erasing **降低** Track1 Private 2.31, 认为 "擦除关键细粒度特征". 但 6/9 论文使用并受益。

**建议**: 保守概率 (0.1, 参考 09) 或作为可选项，通过实验决定。

**实现优先级**: 🔴 **P0** — 已在项目中 (可能已启用)

---

### 3.6 Cosine LR + Warmup — 训练稳定性标配 ⭐⭐⭐⭐

> 引用: 01, 02, 04, 06, 08, 09 (6/9 篇)

**描述**: Cosine 退火 + 线性预热的学习率调度。

**推荐超参数**:
| 参数 | 推荐值 | 来源 |
|------|--------|------|
| Warmup Epochs | 1-3 (Transformer), 3-20 (CNN) | 多数论文 |
| Warmup Start LR | base_lr × 0.01 | 02, 06, 09 |
| Min LR | 1e-8 ~ 5e-7 | 01, 02, 04 |
| LR 缩放 | `base_lr × batch_size × num_gpus / 512` | 01, 04, 06 |

**实现优先级**: 🔴 **P0** — 项目大概率已实现, 检查参数即可

---

### 3.7 Label Smoothing — 防过拟合标配 ⭐⭐⭐⭐

> 引用: 01, 02, 03, 06, 07, 08, 09 (7/9 篇)

**描述**: 将 one-hot 标签平滑为 (1-ε, ε/(C-1), ...)，防止模型过自信。

**推荐超参数**: `ε = 0.1` (全部论文一致)

**⚠️ 注意事项**:
- Seesaw Loss 时通常不使用 Label Smoothing (01, 04 论文: `LABEL_SMOOTHING: 0.0`), 因为 Seesaw 已有类间平衡
- 使用 LabelAwareSmoothing (09) 可以针对长尾场景做差异化平滑

**实现优先级**: 🔴 **P0** — 一行代码

---

### 3.8 Horizontal Flip (基础增强) — 100% 论文使用 ⭐⭐⭐⭐⭐

> 引用: 全部 9 篇论文

**描述**: 训练时 50% 概率水平翻转。细粒度任务中不仅做训练增强，还做推理 TTA (05, 07)。

**实现优先级**: ✅ 项目已实现

---

### 3.9 Mixed Precision (AMP) — 训练加速标配 ⭐⭐⭐⭐

> 引用: 01, 02, 03, 04, 06, 07, 08, 09 (8/9 篇)

**描述**: FP16 混合精度训练，降低显存、加速训练。

**实现优先级**: ✅ 项目大概率已实现 (PyTorch Lightning 内置)

---

### 3.10 Gradient Clipping — 稳定深层训练 ⭐⭐⭐

> 引用: 01, 04, 05, 06, 07 (5/9 篇)

**推荐超参数**: `max_norm = 1.0` (05, 07) 或 `5.0` (01, 04, 06)

**实现优先级**: 🟡 **P1** — PyTorch Lightning 中 `gradient_clip_val`

---

### 3.11 Post-Process (后处理) — 竞赛/生产必备 ⭐⭐⭐

> 引用: 01, 02, 04, 08 (5/9 篇，含 openset 处理)

**描述**: 推理后的额外处理步骤：

1. **Open-Set 检测**: 熵阈值 (04) 或 logit 阈值 (01, 02)
2. **长尾类抢救**: 对稀有类的低置信度预测做放宽 (01: `less_cls` 列表 + 宽松阈值; 04: 熵 < 6 而非 < 4)
3. **安全覆盖**: 对高风险类别不做未知类标记 (04: 毒蘑菇不判为 unknown; 08: 低置信度时取 top-5 中的毒蛇)

**实现优先级**: 🟡 **P1** — 依赖具体任务需求

---

### 3.12 Color Jitter — 基础色彩增强 ⭐⭐⭐

> 引用: 01, 02, 03, 04, 06, 08 (6/9 篇)

**推荐超参数**: 0.2 (基础) ~ 0.4 (高级)

**实现优先级**: ✅ 项目大概率已实现

---

## 四、当前项目短板分析

> 项目现状: SwinV2 base, ~8M images, 44K species, two-stage resolution (448→512), ArcFace (optional), TrivialAugmentWide + RandomErasing + CutMix/MixUp, AdamW + CosineAnnealing + LinearWarmup

### 4.1 损失函数

| 项目现状 | 缺失技巧 | 证据 | 重要性 |
|---------|---------|------|--------|
| ✅ CrossEntropyLoss + ArcFace (可选) | **Seesaw Loss** | 6/9 论文核心长尾方案; 44K 类严重长尾, 远优于 ArcFace 处理长尾 | 🔴 **极高** |
| - | **Focal Loss** (备选) | 03 论文在 FGVC 上 FocalLoss > SeesawLoss | 🟡 中 |
| - | **LabelAwareSmoothing** | 09 论文长尾场景下优于标准 Label Smoothing | 🟢 中低 |
| - | **Poison/Venom 代价矩阵 Loss** | 04, 05, 07, 08 论文对有不对称误判代价的场景显著有效 | 🟢 低 (视任务有无安全需求) |

**核心差距**: 当前项目 44K 类的长尾分布极度严重 (50-1000 per class)，Seesaw Loss 是**最直接的提升手段**。

### 4.2 数据增强

| 项目现状 | 缺失技巧 | 证据 | 重要性 |
|---------|---------|------|--------|
| ✅ TrivialAugmentWide | — | — | — |
| ✅ RandomErasing | — | — | — |
| ✅ CutMix/MixUp | ⚠️ **若引入 Seesaw Loss, 需禁用 CutMix/MixUp** | 01 论文 Table 2: Seesaw+Mixup 反降性能 | 🔴 重要 |
| - | **Vertical Flip** | 02, 03, 06, 08 论文使用, 对真菌/蛇类对称性生物有效 | 🟢 低 |
| - | **GridMask** | 07 论文使用概率 0.2 | 🟢 低 |
| - | **Bicubic Resize(2×) + RandomCrop** | 05, 07 论文先用大尺寸 resize 保留细节再随机裁剪 | 🟡 中 |
| - | **PiecewiseAffine** | 02, 08 论文模拟非刚性形变 | 🟢 低 |

**核心差距**: 增强组合已较完善。主要矛盾在 Seesaw Loss 与 Mixup/CutMix 的互斥。

### 4.2.1 RandomResizedCrop 与 RandomErasing 参数专题调研

#### RandomResizedCrop Scale

| # | 论文 | Scale | 来源 | 备注 |
|---|------|-------|------|------|
| 01 | FungiCLEF 2022 🥇 | `(0.08, 1.0)` | timm 默认 | `config.py` 中不显式设置 scale，`create_transform` 走 ImageNet 标准 |
| 02 | FungiCLEF 2023 🥉 | **`(0.5, 1.3)`** | Albumentations 显式 | `train_seesawloss.py:148`: `RandomResizedCrop(scale=(0.5, 1.3))`，允许放大到 1.3× |
| 03 | Bag-of-Tricks 🥈 | **`(0.8, 1.0)`** | YAML 显式覆盖 | 所有 config YAML 中 `scale: [0.8, 1.0]`，**9 篇中最保守** |
| 04 | Entropy Open-Set 🥇 | `(0.08, 1.0)` | timm 默认 | 同 #01，`config.py` 无 scale 覆盖 |
| 05 | Long-Tailed FGVC 🥈 | — | 未显式设置 | CAFormer + timm pipeline，走默认 |
| 06 | Metaformer+ArcFace 🥉 | `(0.08, 1.0)` | timm 默认 | 同 #01/#04 |
| 07 | OpenWGAN-GP 🥇 | — | TrivialAugmentWide | torchvision v2 pipeline，RRC 由 TAW 内部决定 |
| 08 | SnakeCLEF 2023 🥇 | **`(0.5, 1.3)`** | Albumentations 显式 | `train_pyramid_meta.py:157`: 同 #02 |
| 09 | CoLKANet | **`(0.5, 1)`** | Albumentations 显式 | `train_5fold.py:347`: `scale=(0.5,1)`，不许放大 |
| **当前项目** | image-classification | **`(0.3, 1.0)`** | `data_module.py` | `rrc_scale_min: 0.3` |

**论文明显分为两派**：
- **激进派** (01/04/06)：`(0.08, 1.0)` — ImageNet 标准，允许裁到原图 8% 大小，信任模型对极端裁剪的鲁棒性
- **保守派** (02/03/08/09)：`(0.5~0.8, 1.0~1.3)` — 细粒度特征小（如菌褶纹理、蛇鳞排列），过激裁剪会直接丢掉关键辨识特征

**Bag-of-Tricks (#03) 最极端**，用 `(0.8, 1.0)` 几乎不缩小。这是消融实验选出来的值，论文观点：FGVC 场景下判别性区域可能只占图像的 10-20%，过度缩小等于随机猜测。

**建议**:
- 当前项目 `(0.3, 1.0)` 处于中间位置
- 🟡 P0 建议实验 **`(0.5, 1.0)`**，这是 #09 CoLKANet 值和 #02/#08 的融合值，也是细粒度领域更安全的选择
- 如果后续用 Seesaw Loss 禁掉了 Mixup/CutMix，可以考虑用更保守的 `(0.6, 1.0)` 来补偿正则化损失

#### RandomErasing Scale

| # | 论文 | prob | scale | 来源 | 备注 |
|---|------|------|-------|------|------|
| 01 | FungiCLEF 2022 🥇 | 0.25 | `(0.02, 0.33)` | timm 默认 | `config.py:142`: `REPROB=0.25` |
| 02 | FungiCLEF 2023 🥉 | — | — | — | 不用 RE，用 Albumentations Cutout/CoarseDropout 替代 |
| 03 | Bag-of-Tricks 🥈 | 0.25 | `(0.02, 0.33)` | mindcv 默认 | `transforms_factory.py`: `re_scale=(0.02, 0.33)` |
| 04 | Entropy Open-Set 🥇 | 0.25 | `(0.02, 0.33)` | timm 默认 | 同 #01 |
| 05 | Long-Tailed FGVC 🥈 | **0 (禁用)** | — | 论文消融实验 | ⚠️ Table 9: RE 降低 Track1 Private **2.31 点**！ |
| 06 | Metaformer+ArcFace 🥉 | 0.25 | `(0.02, 0.33)` | timm 默认 | 同 #01/#04 |
| 07 | OpenWGAN-GP 🥇 | **0 (禁用)** | — | `config.yaml` | `random_erasing_prob: 0.0`，显式禁用 |
| 08 | SnakeCLEF 2023 🥇 | — | — | — | Albumentations 增强链，无 RE（用 PiecewiseAffine 替代） |
| 09 | CoLKANet | 0.0/0.1 | — | `presets.py`/`train_5fold.py` | presets 默认 0，train 设为 0.1，只用 p 不含 scale |
| **当前项目** | image-classification | 0.25 | **`(0.02, 0.2)`** | `data_module.py` | `re_scale_min/max`，比多数论文更保守的 max_area |

**RandomErasing 在细粒度分类中存在争议**：
- 5/9 论文用了（prob=0.25, 默认 scale），沿用 ImageNet 惯例
- 但 🥈论文 #05 明确实验表明 RE **降低** 性能（擦除区恰好覆盖了蛇的鳞片排列等关键细粒度特征）
- 🥇论文 #07 也禁用了 RE
- #09 用极低概率 0.1 作为折中

当前项目 `re_scale_max=0.2` 比多数论文的 `0.33` 更保守（最大擦除 20% 面积 vs 33%），这个选择本身偏安全。

**建议**:
- 🔴 P0 建议做一次 **RE ON vs OFF 对照实验**
- 如果 offline 验证集中 RE 没有明显正收益，就关掉（参考 #05 经验，不要因为"大家都用"就默认开启）
- 如果保留，降到 `prob=0.1`（参考 #09）

### 4.3 训练策略

| 项目现状 | 缺失技巧 | 证据 | 重要性 |
|---------|---------|------|--------|
| ✅ 两阶段分辨率 (448→512) | **Two-Stage Training (freeze→unfreeze)** | 03, 05, 07 论文: 先冻结 backbone 训分类头, 再解冻全模型 | 🔴 **高** |
| ✅ CosineAnnealing + LinearWarmup | **ReduceLROnPlateau** (替代 Cosine) | 05, 07 论文: patience=4, factor=0.1, 比 Cosine 更稳定 | 🟡 中 |
| - | **EMA (指数移动平均)** | 09 论文: ema_decay=0.99998, ema_steps=32, 单模型稳定提升 | 🟡 中 |
| - | **Gradient Clipping** | 5/9 论文使用 max_norm=1.0~5.0 | 🟡 中 |
| - | **Early Stopping** | 05, 07 论文: patience=10, 节省计算资源 | 🟢 低 |

**核心差距**: 训练策略基本合理，但缺少冻结→解冻的两阶段流程和 EMA。

### 4.4 学习率调度

| 项目现状 | 可改进 | 证据 | 重要性 |
|---------|--------|------|--------|
| ✅ CosineAnnealingLR + LinearWarmup | ✅ 无需大改 | 6/9 论文使用此组合 | — |
| - | 调整 Warmup Epochs | 当前值未知, 论文推荐 1 (Transformer) | 🟢 低 |
| - | 调整 Min LR | 论文推荐 1e-7~5e-7 (01, 04) 或 1e-8 (02) | 🟢 低 |

**核心差距**: 调度器选择合理，微调参数即可。

### 4.5 模型架构

| 项目现状 | 替代/改进方向 | 证据 | 重要性 |
|---------|-------------|------|--------|
| ✅ SwinV2 base | **CAFormer** (timm 直接调用) | 05 论文: CAFormer > Metaformer (无元数据), 性价比最高 | 🟡 中 |
| - | **ConvNeXt-v2 Large** | 08 论文: 超越 BEiT/EVA/Swin/VOLO, 蛇类 93.65% | 🟡 中 |
| - | **MetaFormer 系列** (如需要元数据) | 01, 04, 06 论文: 混合 Conv+Transformer, 支持元信息 token 注入 | 🟢 低 |
| - | **多层级特征融合** (金字塔头) | 08 论文: 拼接中间层+最终层特征, +1.12% | 🟡 中 |
| - | **CoLKANet** (大核注意力) | 09 论文提出, 但需从零实现 | 🟢 低 |

**核心差距**: SwinV2 是合理选择。CAFormer 或 ConvNeXt-v2 可作为实验对比项。金字塔特征融合头是低成本改进。

### 4.6 推理优化

| 项目现状 | 缺失技巧 | 证据 | 重要性 |
|---------|---------|------|--------|
| ❌ 无 TTA | **Horizontal Flip TTA** | 05, 07 论文: 计算成本最低, 收益明确 (+1~2%) | 🔴 **高** |
| ❌ 无 TTA | **Five-Crop TTA** | 01, 04 论文标准做法 | 🟡 中 |
| ❌ 无 TTA | **高分辨率推理 (1.5×)** | 05 论文: 单模型最大提升 (Track1 +2.23) | 🔴 **高** |
| - | **Model Ensemble** | 全部 9 篇论文, 竞赛必杀技 | 🔴 **极高** |
| - | **Multi-Instance Averaging** | 多图同物场景 (若有) | 🟢 低 |

**核心差距**: **当前项目无任何 TTA 和 Model Ensemble**, 这是最大的推理短板。参考 05 论文 Table 8:
```
基础 (384 no-TTA) → +高分辨率推理(576) → +HFlip TTA → +Multi-Instance → +4xEnsemble
Track1: 76.16 → 78.39 (+2.23) → 79.92 (+1.53) → 79.94 → 81.2 (+1.26)
总提升: +5.04
```

### 4.7 长尾处理

| 项目现状 | 更好的方案 | 证据 | 重要性 |
|---------|----------|------|--------|
| `max_per_class=1000` 硬截断 | **Seesaw Loss** (不截断, 用全量数据) | 6/9 论文: 不改数据分布, 在 loss 层处理长尾 | 🔴 **极高** |
| - | **Two-Stage DRS** (不平衡训练→均衡微调) | 03 论文: 先用全量数据学特征, 再用均衡子集微调分类头 | 🟡 中 |
| - | **Balanced Sampler** (可选) | 08 论文 `ImbalancedDatasetSampler`: 按类别频率倒数采样 | 🟢 低 |
| - | **LabelAwareSmoothing** (长尾定制平滑) | 09 论文: 头类平滑多 (0.3), 尾类不平滑 (0.0) | 🟢 低 |

**核心差距**: 硬截断 `max_per_class=1000` 丢失了头部类的样本多样性信息。Seesaw Loss 可以在保留全量数据的同时自动应对长尾。

---

## 五、改进路线图

### P0（立即可做，1-2 天）— 最易实现、最有把握提升

| # | 任务 | 预期收益 | 参考论文 | 实施难度 | 具体步骤 |
|---|------|---------|---------|---------|---------|
| P0.1 | **替换/新增 Seesaw Loss** (禁用 Mixup/CutMix) | 长尾 F1 +3~7% | 01, 02, 04, 05, 07, 08 | 🟢 低 | ① 复制 `losses.py` 中 SeesawLoss 类; ② 在配置中添加 `loss=seesaw` 选项; ③ 与 Seesaw 共用时自动禁用 mixup/cutmix; ④ 对比实验 |
| P0.2 | **添加 Horizontal Flip TTA** | Track +1~2% | 05, 07 | 🟢 低 | ① 推理时同时传原图+水平翻转; ② 两个 logit 取平均; ③ 作为 Lightning `predict_step` 选项 |
| P0.3 | **RRC scale 调为 (0.5, 1.0) + RandomErasing ON/OFF 对照实验** | 细粒度特征保护 +1~2% | 02, 03, 05, 08, 09 | 🟢 低 | ① `rrc_scale_min` 从 0.3 → 0.5; ② RE prob=0 vs prob=0.25 对照实验; ③ 详见 §4.2.1 专题调研 |
| P0.4 | **检查 Gradient Clipping + Weight Decay 分离** | 训练稳定性 | 01, 05, 09 | 🟢 低 | ① `gradient_clip_val=1.0` (Lightning); ② 确认 bias/norm 参数 weight_decay=0 |
| P0.5 | **添加 Early Stopping** | 节省训练时间 | 05, 07 | 🟢 低 | ① `EarlyStopping(monitor='val_loss', patience=10)` |

- [ ] P0.1 Seesaw Loss 集成 + 禁用 Mixup/CutMix
- [ ] P0.2 Horizontal Flip TTA
- [ ] P0.3 RRC scale → (0.5, 1.0) + RE ON/OFF 实验
- [ ] P0.4 Gradient Clipping + Weight Decay 分离
- [ ] P0.5 Early Stopping

### P1（短期，1-2 周）— 需要较多修改但有明确收益

| # | 任务 | 预期收益 | 参考论文 | 实施难度 | 具体步骤 |
|---|------|---------|---------|---------|---------|
| P1.1 | **Two-Stage Training (freeze→unfreeze)** | 特征质量 +1~2% | 03, 05, 07 | 🟡 中 | ① Stage1 (5-10 epoch): 冻结 backbone, lr=1e-3, 只训分类头; ② Stage2: 解冻全部, lr=5e-5; ③ Lightning 中通过 configure_optimizers 分阶段实现 |
| P1.2 | **高分辨率推理 (1.5×)** | 单模型最大提升 +2~3% | 05, 07 | 🟢 低 | ① 训练用 448/512, 推理用 ~672 (1.5×); ② 仅改推理 transform; ③ 不需要重新训练, 论文认为 FixRes Fine-tuning 不利 |
| P1.3 | **Model Ensemble 框架** | +3~5% | 全部 9 篇 | 🟡 中 | ① 训练 3-4 个模型变体 (不同 seed/backbone/split/loss); ② 推理脚本做 logit 平均; ③ WandB 记录各模型独立指标 |
| P1.4 | **Focal Loss 对比实验** | 若 Seesaw 不适用 | 03 | 🟢 低 | ① `FocalLoss(gamma=2.0, alpha=0.25)`; ② 可与 Mixup/CutMix 共用 (03 论文就是这么组合的); ③ 若数据集极度长尾且想保留混合增强, FocalLoss > SeesawLoss |
| P1.5 | **EMA 指数移动平均** | 稳定提升 | 09 | 🟢 低 | ① `ema_decay=0.99998`; ② 每 32 步更新; ③ Lightning `ModelCheckpoint` + EMA wrapper 或手动实现 |
| P1.6 | **金字塔特征融合头** (Mid+Final features) | 细粒度 +1% | 08 | 🟡 中 | ① 修改 SwinV2 的 forward 返回中间层特征 (384 维 from stage 3); ② 拼接 `[final_feat, mid_feat]` → 2-FC head; ③ dropout=0.5~0.6 |

- [ ] P1.1 Two-Stage Training (freeze→unfreeze)
- [ ] P1.2 高分辨率推理 (1.5× training resolution)
- [ ] P1.3 Model Ensemble 框架 (3-4 模型)
- [ ] P1.4 Focal Loss 对比实验
- [ ] P1.5 EMA 指数移动平均
- [ ] P1.6 金字塔特征融合头

### P2（中期，1 个月+）— 需要架构变更或大量实验

| # | 任务 | 预期收益 | 参考论文 | 实施难度 | 具体步骤 |
|---|------|---------|---------|---------|---------|
| P2.1 | **CAFormer/ConvNeXt-v2 替代 SwinV2** | 架构收益 +2~5% | 05, 08 | 🔴 高 | ① `timm.create_model('caformer_s18.sail_in22k_ft_in1k_384')`; ② `timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k_384')`; ③ 完整对比实验 (控制其他变量不变) |
| P2.2 | **5-Fold Cross Validation + Ensemble** | +3~5% | 09 | 🔴 高 | ① StratifiedKFold(n_splits=5); ② 每 fold 独立训练; ③ 5 个模型 ensemble; ④ 计算成本 ×5 |
| P2.3 | **LabelAwareSmoothing** | 长尾场景额外 +1.6% (Ensemble时) | 09 | 🟡 中 | ① 统计各类别样本数; ② `smooth_head=0.3, smooth_tail=0.0, shape='concave'`; ③ 部分 fold 用 CE + 部分用 LabelAwareSmoothing → Ensemble |
| P2.4 | **元信息融合** (如有地理/时间/属性数据) | +3.8% (04: InternImage vs MetaFormer+Meta) | 01, 04, 08 | 🔴 高 | ① 编码元信息 (周期编码+OneHot); ② 训练 embedding 头; ③ 在 Transformer 层作为额外 token 注入; ④ 需要修改模型架构 |
| P2.5 | **Open-Set 检测** (如果需要拒绝未知类) | 开放集鲁棒性 | 04, 07 | 🔴 高 | ① 熵引导阈值 (04) 或 OpenWGAN-GP (07); ② 需要一定量 out-of-distribution 样本; ③ 大工程 |
| P2.6 | **SimCLR 对比预训练** | 特征质量 +7.4% (06) | 06 | 🔴 高 | ① 自监督预训练阶段; ② NT-XentLoss(τ=0.07); ③ 训练时间加倍 |

- [ ] P2.1 CAFormer/ConvNeXt-v2 架构实验
- [ ] P2.2 5-Fold Cross Validation + Ensemble
- [ ] P2.3 LabelAwareSmoothing
- [ ] P2.4 元信息融合 (视数据可用性)
- [ ] P2.5 Open-Set 检测 (视需求)
- [ ] P2.6 SimCLR 对比预训练

---

## 六、推荐超参数配置

### 6.1 损失函数

| 参数 | 当前项目默认值 | 推荐值 | 修改原因 | 来源 |
|------|-------------|--------|---------|------|
| 主损失函数 | CrossEntropyLoss | **SeesawLoss** (p=0.8, q=2.0, eps=1e-2) | 44K 类严重长尾, 6/9 论文首选 | 01, 04, 05, 07, 08 |
| ArcFace | 可选 | 保留作为备选 (若用 Seesaw 则二选一) | Seesaw + ArcFace 组合无论文验证 | 01, 06 |
| FocalLoss (备选) | 无 | gamma=2.0, alpha=0.25 (若保留 Mixup/CutMix) | 03 论文 FocalLoss + CutMix 组合有效 | 03 |
| Label Smoothing | 未知 | ε=0.1 (仅在用 CE/Focal 时; Seesaw 时禁用) | 7/9 论文标配; Seesaw 已有类间平衡 | 01, 03, 07 |

### 6.2 数据增强

| 参数 | 当前项目默认值 | 推荐值 | 修改原因 | 来源 |
|------|-------------|--------|---------|------|
| Auto Augment | TrivialAugmentWide | **保持** TrivialAugmentWide | ✅ 已是最优选择 | 03, 05, 07, 09 |
| Random Erasing prob | 未知 | **0.25** (若不用 05 的反对意见) 或 **0.1** (保守) | 多数论文 0.25, 05 反对, 09 用 0.1 | 01, 03, 04, 06, 09 |
| CutMix alpha | 未知 | **若 Seesaw Loss: 0 (禁用)**; 若 FocalLoss: 1.0 | Seesaw 与 CutMix 互斥 | 01, 03 |
| Mixup alpha | 未知 | **若 Seesaw Loss: 0 (禁用)**; 若 FocalLoss: 0.2~0.8 | 同上 | 01, 03, 06, 09 |
| CutMix/MixUp prob | 未知 | 若使用: **0.8~1.0** | 多数论文高概率混合 | 05, 09 |
| Color Jitter | 未知 | **0.4** | 多数论文值 | 01, 02, 03, 04, 06, 08 |
| RandomResizedCrop scale | `(0.3, 1.0)` | **(0.5, 1.0)** | FGVC 保守派 (02/08/09) 共识, 当前项目偏激进 | 02, 08, 09 |
| RandomErasing prob | 0.25 | **做 ON/OFF 对照实验**; 若保留降到 **0.1** | 05 论文明确反对 (降 2.31 点), 09 用 0.1 折中 | 05, 09 |
| RandomErasing scale | `(0.02, 0.2)` | `(0.02, 0.33)` 或保持当前 `(0.02, 0.2)` | 当前偏保守但问题不大 | 01, 03, 04, 06 |
| Vertical Flip | 无 | **0.5** (对生物对称性有用) | 02, 03, 06, 08 | 02 |
| Interpolation | 未知 | **bicubic** | 全部论文标配 (Albumentations: `cv2.INTER_CUBIC`) | 01, 02, 03, 05 |

### 6.3 训练策略

| 参数 | 当前项目默认值 | 推荐值 | 修改原因 | 来源 |
|------|-------------|--------|---------|------|
| 训练阶段 | 两阶段分辨率 (448→512) | **Two-Stage freeze→unfreeze**: Stage1 (5-10 epoch) freeze backbone lr=1e-3, Stage2 unfreeze lr=5e-5 | 保护预训练权重, 03/05/07 标配 | 03, 05, 07 |
| Optimizer | AdamW (lr=1e-4, wd=2e-5) | **AdamW (lr=5e-5, wd=0.05)** | 论文普遍更低 lr + 更高 wd | 01, 04, 05, 06, 07 |
| LR Scheduler | CosineAnnealingLR | **保持 CosineAnnealingLR** (备选 ReduceLROnPlateau) | ✅ 已是标配, 6/9 论文用 | 01, 02, 04, 06, 09 |
| Warmup | LinearLR | **Linear Warmup, 1-3 epochs, start=lr×0.01** | 确认预热时长和起始 LR | 01, 02, 06, 09 |
| Min LR | 未知 | **1e-7 ~ 5e-7** | 论文推荐范围 | 01, 04, 09 |
| Gradient Clipping | 未知 | **max_norm = 1.0~5.0** | 5/9 论文使用 | 01, 04, 05, 06, 07 |
| EMA | 无 | **ema_decay=0.99998, update every 32 steps** | 09 论文稳定提升 | 09 |
| Batch Size | 未知 | 尽可能大 (≥256 effective) | 01 论文: 大 BS 稳定训练 | 01 |
| Early Stopping | 无 | **patience=10 epochs** | 节省计算资源 | 05, 07 |
| Epochs | 未知 | **保留当前配置** (透明实验决定) | 01 论文: 32-64 最优点, 过长反降 | 01 |

### 6.4 模型架构

| 参数 | 当前项目默认值 | 推荐值 | 修改原因 | 来源 |
|------|-------------|--------|---------|------|
| Backbone | SwinV2 base | **保持 SwinV2** (短期); 实验 CAFormer S18/S36 (中期) | SwinV2 合理, CAFormer 05 论文性价比最高 | 05, 08 |
| 输入分辨率 | 448→512 | **保持** (推理提高到 1.5× → 672~768) | 训练保持, 推理提分 | 05, 07 |
| Dropout | 未知 | **分类头前 0.2~0.5** | 44K 类极高过拟合风险; 08 论文用 0.6 | 05, 07, 08 |
| Drop Path | 未知 | **0.1** | 多数论文值 | 01, 04, 06 |
| 分类头 | 单层 Linear | **2-FC head (backbone_dim→2300→num_classes)** 或 **金字塔头 (mid+final concat)** | 08 论文提升 +1.12% | 08 |
| 预训练权重 | 未知 | **保持 ImageNet-22K 预训练** (timm `swinv2_base_window12to24_192to384.ms_in22k_ft_in1k`) | ✅ 已是最优选择 | 03, 05, 08 |

### 6.5 推理优化

| 参数 | 当前项目默认值 | 推荐值 | 修改原因 | 来源 |
|------|-------------|--------|---------|------|
| TTA | **无** | **HFlip TTA** (原图+水平翻转平均) | 05, 07 论文: 最低成本明确收益 | 05, 07 |
| 推理分辨率 | 512 (同训练) | **1.5× → 768** (不重新训练) | 05 论文: 单模型最大提升, FixRes Fine-tune 无益 | 05 |
| Ensemble | **无** | **3-4 模型 simple average** | 全部 9 篇论文, 竞赛必杀技 | 01-09 |
| Post-Process | 无 | **无** (闭集任务暂不需要; 开放集参考 04/07) | — | — |

---

## 七、核心参考文献速查

| # | 论文/方案 | 最适合查阅的场景 |
|---|----------|----------------|
| 01 | FungiCLEF 2022 1st (MetaFormer + Seesaw + 后处理) | Seesaw Loss 完整实现、后处理逻辑、TTA 配置 |
| 03 | Bag of Tricks FGVC (FocalLoss + Two-Stage + TrivialAugment) | Two-Stage DRS 训练流程、FocalLoss 配置、增强消融 |
| 04 | Entropy-Guided Open-Set (MetaFormer + 熵 + Seesaw) | 熵引导开放集检测、元信息融合、毒蘑菇代价Loss |
| 05 | Long-Tailed FGVC (CAFormer + Seesaw + VenomLoss + TTA) | 高分辨率推理、HFlip TTA、EMA、多划分 Ensemble、Seesaw 移植 |
| 07 | OpenWGAN-GP (Seesaw + LogitNorm + WGAN-GP) | LogitNorm、Two-Stage Training、GridMask、WGAN-GP 开放集 |
| 08 | Venomous Snake (ConvNeXt-v2 + 金字塔头 + Prior Model) | 金字塔特征融合头、CLIP 元数据编码、毒蛇后处理 |
| 09 | Large Kernel ViT (CoLKANet + LabelAwareSmoothing + EMA) | LabelAwareSmoothing、EMA、5-fold CV、输出归一化 Ensemble |

---

## 八、总结

### 8.1 三句话结论

1. **Seesaw Loss + 禁用 Mixup/CutMix** 是 44K 类长尾分类最直接有效的改进 (6/9 论文首选)
2. **HFlip TTA + 1.5× 推理分辨率 + 3-4 模型 Ensemble** 是推理端最快、最稳的提升路线 (3 天可落地)
3. **Two-Stage Training (freeze→unfreeze) + EMA + Gradient Clipping** 是训练端的最佳实践标配

### 8.2 执行优先级一句话

```
P0 (本周): Seesaw Loss + HFlip TTA + 训练超参检查
P1 (两周): Two-Stage Training + 高分辨率推理 + Model Ensemble (3模型)
P2 (一个月): CAFormer 实验 + 5-fold CV + LabelAwareSmoothing
```

---

> 📌 最后更新: 2026-05-10 | 🦎 壁虎  
> 基于 9 篇论文的完整代码+论文分析
