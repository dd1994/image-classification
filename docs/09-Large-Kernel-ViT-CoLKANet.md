# 09 — Large Kernel ViT / CoLKANet

> 论文：*When Large Kernel Meets Vision Transformer: A Solution for SnakeCLEF & FungiCLEF*  
> 作者：Yang Shen, Xuhao Sun, Zijian Zhu (南京理工大学)  
> 会议：CLEF 2022 Working Notes  
> 代码：`D:\fgvc-survey\When-Large-Kernel-Meets-Vision-Transformer-SnakeCLEF-FungiCLEF\code\`

---

## 1. 论文概述

### 1.1 背景与任务

论文针对 LifeCLEF 2022 的两个细粒度识别挑战赛：

| 竞赛 | 类别数 | 训练样本 | 标签特点 | Openset |
|------|--------|----------|----------|---------|
| SnakeCLEF | 1,572 种蛇 | 318,532 张 | 长尾分布（头类 6,472 张，尾类仅 5 张） | 无 |
| FungiCLEF | 1,604 种菌类 | 295,938 张 | 长尾分布，含丰富元数据 | 约 10% openset |

评价指标为 **Macro F1-Score**（类别权重相等，天然偏向长尾性能）。

### 1.2 核心贡献

1. **提出 CoLKANet**：将 Large Kernel Attention (LKA) 与 Vision Transformer 融合的新型 backbone
2. **系统整理 FGVC 竞赛 trick 全集**：数据增强、损失函数、EMA、TTA、FixRes 等 8 项关键技巧
3. **公开了 Leaderboard Overfitting 的坑**：公共榜按类别抽样而非按样本抽样，导致拟合公共榜的策略在私榜无效

### 1.3 最终成绩

| 竞赛 | Macro F1 (Private) | 最终 Ensemble 方案 |
|------|---------------------|-------------------|
| SnakeCLEF | **85.4%** | ConvNeXt 448 + 5-fold ConvNeXt 384 + 5-fold VOLO + CoLKANet + 5-fold Swin + ViT |
| FungiCLEF | **78.9%** | ConvNeXt 448 + 5-fold ConvNeXt 384 + 5-fold VOLO + CoLKANet + 5-fold Swin |

---

## 2. CoLKANet 架构详解

### 2.1 设计理念

> "Earlier convolution helps transformer see better."

核心思路：用 Large Kernel Attention 替代 CoAtNet 中的早期 CNN 阶段，让网络在浅层获得更大的感受野，然后将深层留给 Transformer 的 self-attention。

### 2.2 整体结构

```
Input (3, 384, 384)
  │
  ├─ s0: Block (ConvBlock ×3)      → [192, 96, 96]     # LKA 阶段
  ├─ s1: Block (ConvBlock ×3)      → [192, 48, 48]     # LKA 阶段
  ├─ s2: Block (ConvBlock ×12)     → [384, 24, 24]     # LKA 阶段
  ├─ s3: Transformer ×24           → [768, 12, 12]     # Transformer 阶段
  ├─ s4: Transformer ×2            → [1536, 6, 6]      # Transformer 阶段
  │
  ├─ AvgPool(6, 1) → [1536]
  └─ Linear(1536, num_classes)
```

超参数（CoLKANet-L，文件 `CoLKANet.py` 行 316-319）：

```python
num_blocks = [3, 3, 12, 24, 2]    # 各阶段的 block 数
channels   = [192, 192, 384, 768, 1536]  # 各阶段的通道数
mlp_ratios = [8, 8, 4]            # 前 3 个 ConvBlock 阶段的 MLP 扩展比
# 后 2 个 Transformer 阶段的 FFN hidden_dim = inp × 4（硬编码）
```

### 2.3 Large Kernel Attention (LKA) 机制

**文件：** `CoLKANet.py` 第 255-275 行

LKA 模块将一个标准大核卷积分解为三部分，等价于 13×13 卷积：

```python
class LKA(nn.Module):
    def __init__(self, dim):
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)         # 5×5 DW-Conv
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9,
                                      groups=dim, dilation=3)               # 7×7 DW-D-Conv (dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)                                 # 1×1 Conv

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)           # 局部空间混合
        attn = self.conv_spatial(attn) # 长距离空间混合（dilation 带来大感受野）
        attn = self.conv1(attn)        # 通道混合
        return u * attn                # 逐元素注意力门控
```

**分解图（论文 Fig.3）：**
- 5×5 depth-wise conv → 5×5 depth-wise dilation conv (rate=3) → 1×1 conv
- 等价感受野：13×13，但参数量远小于直接做 13×13 卷积

### 2.4 KernelAttention 封装

```python
class KernalAttention(nn.Module):      # 文件 CoLKANet.py 行 278-290
    def forward(self, x):
        shortcut = x
        x = self.proj_1(x)             # 1×1 Conv（通道投影）
        x = self.activation(x)         # GELU
        x = self.spatial_gating_unit(x) # LKA → 空间注意力门控
        x = self.proj_2(x)             # 1×1 Conv（通道投影回来）
        x = x + shortcut               # 残差连接
        return x
```

### 2.5 ConvBlock（LKA Block）

```python
class ConvBlock(nn.Module):            # 文件 CoLKANet.py 行 293-327
    # 结构：
    #   x → BatchNorm → KernelAttention → LayerScale(1e-2) → DropPath → (+x)
    #   → BatchNorm → Mlp(DWConv)      → LayerScale(1e-2) → DropPath → (+x)
```

关键细节：
- **Layer Scale 初始化值：** `1e-2`（`layer_scale_init_value`）
- **MLP 结构：** 1×1 Conv → 3×3 DWConv → GELU → Dropout → 1×1 Conv → Dropout
- **归一化：** BatchNorm2d
- **激活：** GELU
- **DropPath：** 用于随机深度正则化

### 2.6 Transformer Block

```python
class Transformer(nn.Module):          # 文件 CoLKANet.py 行 120-143
    # 结构：
    #   x → Rearrange → LayerNorm → Attention(relative bias) → Rearrange → (+x)
    #   → Rearrange → LayerNorm → FeedForward              → Rearrange → (+x)
```

关键细节：
- **自注意力：** 标准 QKV 自注意力 + 相对位置偏置（`relative_bias_table`）
- **Head 数：** 8，`dim_head=32` → `inner_dim=256`
- **FFN 扩展比：** ×4（硬编码 `hidden_dim = inp * 4`）
- **归一化：** LayerNorm
- **下采样：** 第一个 Transformer block 用 `MaxPool2d(3, 2, 1)` 做空间下采样，配合 1×1 投影

### 2.7 Overlap Patch Embedding

```python
class OverlapPatchEmbed(nn.Module):    # 文件 CoLKANet.py 行 329-356
    # s0 阶段：patch_size=7, stride=2  → 384→192 降为 192→96
    # s1 阶段：patch_size=3, stride=2  → 96→48
    # s2 阶段：patch_size=3, stride=2  → 48→24
    # 使用重叠 patch 而非 ViT 的 non-overlapping patch
```

### 2.8 与 ResNet / ViT / CoAtNet 的对比

| 特性 | ResNet | ViT | CoAtNet | CoLKANet |
|------|--------|-----|---------|----------|
| 早期阶段 | 小核 Conv (3×3) | Patch Embed + Attention | MBConv | **LKA + ConvBlock** |
| 大感受野 | 靠深度堆叠 | 全局 self-attention | MBConv | **LKA 13×13** |
| 位置编码 | 无显式 | 可学习/正弦 | 相对偏置 | 相对偏置 |
| 浅层特征 | 局部细节 | 全局但低效 | 局部 | **局部+长距离（dilation）** |

---

## 3. 识别率提升技巧（含超参数）

### 3.1 数据增强

论文从初始的 Albumentations 组合切换到 **TrivialAugmentWide**，单模型提升约 **+0.5%** Macro F1。

**代码中使用的训练增强（文件 `presets.py` 行 13-42）：**

```python
ClassificationPresetTrain(
    crop_size=384,
    auto_augment_policy="ta_wide",          # TrivialAugmentWide
    random_erase_prob=0.1,                  # Random Erasing
)
```

**同时使用 batch-level 增强（文件 `train_5fold.py` 行 220-225）：**

```python
mixupcutmix = torchvision.transforms.RandomChoice([
    transforms.RandomMixup(num_classes=CFG['class_num'], p=1.0, alpha=0.2),
    transforms.RandomCutmix(num_classes=CFG['class_num'], p=1.0, alpha=1.0)
])
```

**Mixup/CutMix 超参数汇总：**

| 参数 | 值 | 说明 |
|------|-----|------|
| Mixup alpha | 0.2 | Beta 分布参数，小值弱混合 |
| CutMix alpha | 1.0 | Beta 分布参数 |
| CutMix 概率 | 100%（通过 RandomChoice） | 每个 batch 随机选 Mixup 或 CutMix |
| Random Erasing prob | 0.1 | |

### 3.2 标签感知平滑（Label-Aware Smoothing）

**文件：** `train_5fold.py` 第 80-97 行

针对长尾分布优化的 Label Smoothing，不同类别使用不同的平滑因子：

```python
class LabelAwareSmoothing(nn.Module):
    def __init__(self, smooth_head=0.3, smooth_tail=0.0, shape='concave'):
        # smooth_head=0.3: 头类（样本最多的类别）平滑因子
        # smooth_tail=0.0: 尾类（样本最少的类别）平滑因子
        # shape='concave': 使用 sin 函数从尾到头递增平滑量
```

**公式（论文 Eq.2）：**
- 头类样本多，预测容易过自信 → 更大的平滑因子（0.3）来抑制过自信
- 尾类样本少，预测已经欠自信 → 更小的平滑因子（0.0），不进一步削弱

**实际使用的组合策略：**
- 部分 fold 使用 `LabelAwareSmoothing`，部分 fold 使用标准 `CrossEntropyLoss`
- 训练 `loss_tr = LabelAwareSmoothing()`，验证 `loss_fn = CrossEntropyLoss()`
- Ensemble 时混合两种 loss 训练的模型可获得 **+1.6%** 提升（FungiCLEF）

### 3.3 指数移动平均（EMA）

**文件：** `train_5fold.py` 第 73-77 行

```python
class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param
        super().__init__(model, device, ema_avg)
```

**超参数：**
| 参数 | 值 | 说明 |
|------|-----|------|
| ema_decay | 0.99998 | 接近 1，非常慢的衰减 |
| ema_steps | 32 | 每 32 步更新一次 EMA 模型 |

```python
# 自适应 alpha 计算（train_5fold.py 行 ~330）
adjust = CFG['train_bs'] * CFG['ema_steps'] / CFG['epochs']   # = 9*32/15 = 19.2
alpha = 1.0 - CFG['ema_decay']                                 # = 0.00002
alpha = min(1.0, alpha * adjust)                               # = min(1.0, 0.000384)
model_ema = ExponentialMovingAverage(model.module, device=device, decay=1.0 - alpha)
```

### 3.4 Test Time Augmentation (TTA)

- 对测试图像做 **8-13 次**不同裁剪，取平均
- 单模型提升约 **+0.6%**
- ⚠️ 注意：在 SnakeCLEF 私榜上无效（过拟合公共榜）

### 3.5 FixRes 策略（CNN 专用）

**作用：** 解决训练分辨率和推理分辨率不匹配的问题

- 训练时使用较低分辨率，推理时提高分辨率
- 缩放比例：`0.758` 和 `0.875`（文献推荐）
- ConvNeXt 在 SnakeCLEF 上提升 **+0.8%**，FungiCLEF 上提升 **+0.2%**
- ⚠️ Transformer 固定图像尺寸，无法使用 FixRes

### 3.6 输出归一化（用于 Ensemble）

**文件：** 论文 Section 4.4

```python
Norm(f(x)) = (1 / max(f(x))) ^ alpha * f(x)
# alpha = 0.15 或 0.20 效果最好
```

将各模型的 logits 缩放到同一尺度再 ensemble，避免某个模型因输出幅度过大而主导结果。提升约 **+0.1%**。

### 3.7 混淆矩阵辅助 Ensemble

- 只在 5-fold 模型上使用（没有全量训练模型的混淆矩阵）
- 分析每个模型在各类别上的表现，指导 ensemble 权重
- 提升约 **+0.2%**

### 3.8 Pseudo Labelling（仅 SnakeCLEF 有效）

- 对尾类（训练样本 < 100 的类别）生成伪标签
- 用聚类方法在测试集上生成伪标签
- 在训练+伪标签数据上微调模型
- 公共榜提升 **+0.9%**，私榜无效（过拟合）

### 3.9 Openset Recognition 策略

- 使用简单的 softmax 概率阈值策略（论文 6.2 节引用 [37]）
- 阈值根据公开榜反推（约 60-70% openset 在公共榜子集）
- 未使用元数据（尝试过 MetaFormer 但无效）

---

## 4. 训练策略

### 4.1 超参数完整配置

**文件：** `train_5fold.py` 第 39-72 行 `CFG` 字典

```python
CFG = {
    'root_dir': '/root/dataset/train/DF20-train_val',
    'fold_num': 5,              # 5 折交叉验证
    'seed': 68,                 # 固定随机种子
    'model_arch': 'swinv2',     # 实际使用多种 backbone
    'img_size': 384,            # 输入分辨率
    'resize_size': 384,
    'crop_size': 384,
    'warmup_epochs': 3,         # Warmup 轮数（CNN 使用，ViT 不用）
    'epochs': 15,               # 总训练轮数
    'train_bs': 9,              # 单 GPU batch size（3 GPU 合计 27）
    'valid_bs': 18,
    'T_0': 15,                  # CosineAnnealingWarmRestarts 周期
    'lr': 1.5e-4,               # 初始学习率（论文主文本为 1.2e-4）
    'min_lr': 1e-5/7,           # 最小学习率 ≈ 1.43e-6
    'lr_warmup_decay': 0.01,    # Warmup 初始学习率比例
    'weight_decay': 2e-5,       # 权重衰减
    'num_workers': 24,
    'accum_iter': 1,
    'smoothing': 0.1,           # 标准 Label Smoothing（与 LabelAware 并列）
    'cutmix_prob': 0.8,
    'ema_decay': 0.99998,
    'ema_steps': 32,
}
```

### 4.2 不同 Backbone 的配置差异

| Backbone | 分辨率 | 预训练 | 学习率 | Weight Decay | Label Smoothing | Epochs | Warmup |
|----------|--------|--------|--------|-------------|-----------------|--------|--------|
| Swin-L (in22k) | 384×384 | ImageNet-22K | 1.2e-4 | 2e-5 | 0.1 / LabelAware | 15 | ✗ |
| VOLO-D4 | 448×448 | ImageNet-1K? | 1.2e-4 | 2e-5 | 0.1 / LabelAware | 15 | ✗ |
| ConvNeXt-L (in22k) | 384×384 / 448×448 | ImageNet-22K | 1.2e-4 | 2e-5 | 0.1 / LabelAware | 15 | ✓ (3 epochs) |
| ViT (MAE) | 384×384 | MAE pretrained | 1.2e-4 | 2e-5 | 0.1 | 15 | ✗ |
| CoLKANet | 384×384 | 无预训练？ | 1.2e-4 | 2e-5 | LabelAware | 15 | ✗ |

### 4.3 优化器与调度器

```python
# 优化器：AdamW
optimizer = torch.optim.AdamW(parameters, lr=CFG['lr'], weight_decay=CFG['weight_decay'])

# 学习率调度：Cosine Annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'])

# 权重衰减分离策略（文件 train_5fold.py split_normalization_params 函数）
# 将 BatchNorm/LayerNorm/GroupNorm 参数的 weight_decay 设为 0.0
# 其余参数使用 CFG['weight_decay'] = 2e-5
```

### 4.4 训练流程关键点

1. **5 折交叉验证**：`StratifiedKFold(n_splits=5, shuffle=True)`
2. **混合精度训练**：`GradScaler` + `autocast()`
3. **多 GPU**：`DataParallel` (GPU 0,1,2)
4. **Loss 混合策略**：部分 fold 使用 LabelAwareSmoothing，部分使用 CrossEntropyLoss
5. **EMA 更新**：每 32 步更新一次
6. **最佳模型保存**：按验证集 Macro F1 保存最佳 checkpoint
7. **全量数据训练**：部分模型（ViT, CoLKANet）使用全部训练数据，不使用 5-fold

### 4.5 预训练模型加载（文件 `train_5fold.py` load_pretrained 函数）

关键技巧：
- **相对位置偏置表插值**：当输入分辨率不同于预训练分辨率时，用 `bicubic` 插值调整 `relative_position_bias_table`
- **绝对位置编码插值**：同上
- **分类头适配**：ImageNet-22K (21841 类) → 目标类别数，通过 `map22kto1k.txt` 映射或重新初始化

---

## 5. 推理优化

### 5.1 单模型推理

```python
# 评估 Transform（文件 presets.py ClassificationPresetEval）
transforms.Resize(resize_size=256, interpolation=BICUBIC)
transforms.CenterCrop(crop_size=384)
```

### 5.2 TTA

- 8-13 次随机裁剪 → 取平均 logits
- 注意：论文发现 TTA 在 SnakeCLEF 私榜无效（过拟合公共榜）

### 5.3 多模型 Ensemble

**SnakeCLEF 最终方案（6 个模型组）：**
```
ConvNeXt 448 (全量) + 5-fold ConvNeXt 384 + 5-fold VOLO + CoLKANet + 5-fold Swin + ViT
→ 85.4% Macro F1
```

**FungiCLEF 最终方案（5 个模型组）：**
```
ConvNeXt 448 (全量) + 5-fold ConvNeXt 384 + 5-fold VOLO + CoLKANet + 5-fold Swin
→ 78.9% Macro F1
```

**Ensemble 技巧：**
- 输出归一化（alpha=0.15-0.20）
- 混淆矩阵辅助选择模型
- 混合 LabelAware + CrossEntropy 训练的模型
- 多观察值（每观察多张图）取平均

### 5.4 多观察值处理

- 一个 observation 可能包含多张图像
- 对每张图像独立预测，取平均作为该 observation 的最终预测

---

## 6. 对 image-classification 项目的借鉴

### 6.1 可直接采用的技巧（高优先级）

| 技巧 | 预期收益 | 实现难度 | 说明 |
|------|---------|---------|------|
| **TrivialAugmentWide** | +0.5% | 低 | 替代复杂的手工增强组合，`presets.py` 已实现 |
| **EMA** | 稳定提升 | 低 | ema_decay=0.99998, ema_steps=32，代码可直接复用 |
| **LabelAwareSmoothing** | 长尾场景显著 | 中 | 需统计各类别样本数，`train_5fold.py` 有完整实现 |
| **Mixup + CutMix (RandomChoice)** | 强正则化 | 低 | `transforms.py` 已实现，alpha_mixup=0.2, alpha_cutmix=1.0 |
| **Weight Decay 分离** | 稳定训练 | 低 | BN/LN/GN 参数 weight_decay=0，其余=2e-5 |
| **FixRes** | +0.2~0.8% | 低 | 仅 CNN backbone 可用；训低推高 |
| **TTA** | +0.6%（单模型） | 中 | 注意过拟合风险 |
| **5-fold CV Ensemble** | +3~5% | 高 | 计算成本大，但提升显著 |

### 6.2 可参考的架构设计（中优先级）

| 设计 | 说明 | 适用场景 |
|------|------|---------|
| **LKA 模块** | 5×5 DW → 7×7 Dilation DW → 1×1，等价 13×13 核 | 可插入任何 CNN 的 bottleneck |
| **早期 Conv + 后期 Transformer 的混合结构** | 浅层用 LKA 捕获局部细节，深层用 Transformer 做全局建模 | 细粒度分类天然适合 |
| **Overlap Patch Embedding** | 重叠 patch 嵌入，保留更多空间信息 | 替代 ViT 的 non-overlapping patch |
| **Layer Scale** | 初始值 1e-2，稳定深层训练 | 可用于任何 Transformer/ConvBlock |
| **相对位置偏置** | 比绝对位置编码更好的泛化能力 | 多分辨率训练时需 bicubic 插值 |

### 6.3 训练配置模板

```python
# 建议在 image-classification 项目中采用的默认配置
DEFAULT_FGVC_CONFIG = {
    # 数据增强
    "auto_augment": "ta_wide",         # TrivialAugmentWide
    "random_erase_prob": 0.1,
    "mixup_alpha": 0.2,
    "cutmix_alpha": 1.0,

    # 优化
    "optimizer": "adamw",
    "lr": 1.2e-4,                      # 或根据 batch size 线性缩放
    "min_lr": 1e-6,
    "weight_decay": 2e-5,
    "lr_schedule": "cosine",

    # 正则化
    "label_smoothing": 0.1,            # 标准分类
    "label_aware_smoothing": True,     # 长尾分类（smooth_head=0.3, smooth_tail=0.0）
    "ema_decay": 0.99998,
    "ema_steps": 32,

    # 训练
    "epochs": 15,                      # 论文中训练时间较短
    "warmup_epochs": 0,                # Transformer 不加 warmup
    "mixed_precision": "fp16",
    "gradient_accumulation": 1,

    # 推理
    "tta_crops": 8,                    # TTA 裁剪数
    "fixres": False,                   # CNN 专用
}
```

### 6.4 长尾分类专用流程

```
Step 1: 统计各类别样本数 cls_num_list
Step 2: 使用 LabelAwareSmoothing(smooth_head=0.3, smooth_tail=0.0, shape='concave')
Step 3: 部分实验组使用 CrossEntropy，部分使用 LabelAwareSmoothing
Step 4: Ensemble 时混合两种 loss 训练的模型（可获得额外 +1.6%）
Step 5: 5-fold CV 训练，生成混淆矩阵指导 Ensemble 权重
```

### 6.5 竞赛经验教训

1. **公共榜过拟合是真实风险**：SnakeCLEF/FungiCLEF 的公共榜按类别抽样而非样本抽样，导致在特定类别上加权可以在公共榜提分但在私榜无效
2. **Pseudo Labelling 要谨慎**：只在 SnakeCLEF 有效，FungiCLEF 无效（openset 问题）
3. **多样性优先**：不追求单一模型极致调优，而是训练多种不同模型做 ensemble（不同 backbone、不同 loss、不同 fold）
4. **TTA 可能过拟合**：SnakeCLEF 上 TTA 在公共榜有效（+0.6%），在私榜完全无效
5. **Openset 用简单阈值即可**：论文发现 openset 识别与闭集准确率高度相关，不需要复杂方法

### 6.6 代码文件映射

| 文件 | 内容 | 关键函数/类 |
|------|------|-----------|
| `CoLKANet.py` | CoLKANet 完整实现 | `LKA`, `KernalAttention`, `ConvBlock`, `Transformer`, `Block`, `CoLKANet`, `colkanet_l()` |
| `train_5fold.py` | 5 折训练主流程 | `CFG`, `LabelAwareSmoothing`, `ExponentialMovingAverage`, `split_normalization_params`, `load_pretrained` |
| `presets.py` | Torchvision 增强预设 | `ClassificationPresetTrain`, `ClassificationPresetEval` |
| `transforms.py` | Mixup/CutMix 实现 | `RandomMixup`, `RandomCutmix` |

---

## 7. 关键引用

论文中引用的关键技术及其原始出处：

| 技术 | 原始论文 | arxiv |
|------|---------|-------|
| VAN (LKA 来源) | Guo et al., "Visual Attention Network" | arxiv:2202.09741 |
| CoAtNet | Dai et al., "CoAtNet: Marrying Convolution and Attention" | NeurIPS 2021 |
| RepLKNet (大核设计) | Ding et al., "Scaling Up Your Kernels to 31×31" | arxiv:2203.06717 |
| ConvNeXt | Liu et al., "A ConvNet for the 2020s" | arxiv:2201.03545 |
| Swin Transformer | Liu et al., "Swin Transformer" | arxiv:2103.14030 |
| VOLO | Yuan et al., "Vision Outlooker" | arxiv:2106.13112 |
| Label-Aware Smoothing | Zhong et al., "Improving Calibration for Long-Tailed Recognition" | CVPR 2021 |
| TrivialAugment | Müller & Hutter, "TrivialAugment" | ICCV 2021 |
| FixRes | Touvron et al., "Fixing the Train-Test Resolution Discrepancy" | NeurIPS 2019 |
| OSR baseline | Vaze et al., "Open-Set Recognition: A Good Closed-Set Classifier Is All You Need" | arxiv:2110.06207 |
