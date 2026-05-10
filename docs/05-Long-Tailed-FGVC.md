# 05-Long-Tailed-FGVC — 长尾细粒度图像识别技术总结

> 基于论文：*Generalizable Training Techniques for Fine-Grained Long-Tailed Image Recognition: Transferring Methods Optimized for FungiCLEF 2024 to SnakeCLEF 2024*
> 作者：Jack N. Etheredge (Twosense)
> 会议：CLEF 2024
> 代码：https://github.com/Jack-Etheredge/snakeclef2024
>
> **核心价值**：该论文提供了一套经过充分消融实验验证的、可跨任务泛化的长尾细粒度图像识别训练+推理方法论。在 SnakeCLEF 2024（1,784类蛇类，182,261张图像，严重长尾分布）中获得所有指标的**第2名**。

---

## 1. 论文概述

### 1.1 任务背景

SnakeCLEF 2024 是一个**细粒度 + 长尾**图像识别挑战：
- **1,784 种蛇类**，182,261 张训练图像
- 类别严重不平衡（少数常见种大量图像，多数稀有种仅几张）
- 需要区分毒蛇与无毒蛇（非对称惩罚：毒蛇→无毒蛇误判 cost=5，无毒蛇→毒蛇误判 cost=2，同毒性间误判 cost=1~2）
- **测试集不提供地理元数据**（无法利用地理位置过拟合）
- 评价指标：Track1（加权平均准确率+F1）、Track2（毒蛇混乱损失）、Macro F1

### 1.2 方法核心思想

作者开发了一套**通用训练+推理方法论**，先用于 FungiCLEF 2024（真菌识别，含开放集），然后**不做大改动直接迁移到 SnakeCLEF**，验证了方法的泛化性。核心思路：

1. **Seesaw Loss** 处理长尾分布（无需重采样）
2. **Custom Venom Loss** 嵌入领域代价矩阵
3. **CAFormer 架构**（Metaformer 的改进版本）
4. **多种 Test-Time Augmentation** 提升推理鲁棒性
5. **多数据划分 Ensemble** 增加模型多样性

---

## 2. 识别率提升技巧（含超参数）

### 2.1 模型架构选择

| 模型 | 参数量 | 预训练权重 | 推荐场景 |
|------|--------|-----------|---------|
| **CAFormer-S18** | ~18M | ImageNet-21K | 主力模型，性价比最高 |
| **CAFormer-S36** | ~36M | ImageNet-21K | 强单模型基线，ensemble 加分项 |
| Metaformer-0 | ~??M | iNaturalist2021 | CAFormer 全面优于它 |

**关键发现**：CAFormer > Metaformer（所有指标），且 CAFormer 在**元数据不可用时**优势更明显。

**模型构建代码位置**：`snakeclef/closedset_model.py:build_model()`
- 使用 `timm.create_model()` 加载
- 在分类头前添加 Dropout 层
- 支持递归更新已有 Dropout 层的概率

```text
# 代码路径: snakeclef/closedset_model.py
model_id 示例:
  - "caformer_s18.sail_in22k_ft_in1k_384"
  - "caformer_s36.sail_in22k_ft_in1k_384"
  - "MetaFG_0" (Metaformer-0, 需单独加载 iNaturalist2021 预训练权重)
```

### 2.2 Loss 函数（最重要）

#### 2.2.1 Seesaw Loss（首选长尾分类 Loss）

**论文**：[Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021)](https://arxiv.org/abs/2008.10032)
**代码位置**：`snakeclef/losses.py:class SeesawLoss`

```
超参数：
  - p=0.8 (mitigation factor，控制对尾类惩罚的降低程度)
  - q=2.0 (compensation factor，控制对误分类的惩罚增强程度)  
  - eps=1e-2 (数值稳定性)
```

**原理**：
- **Mitigation Factor**：根据类别样本数比例 `(N_i / N_j)^p` 动态降低尾类的惩罚权重
- **Compensation Factor**：对预测概率高的错误类别施加额外惩罚 `(score_j / score_gt)^q`
- 无需重采样，无需类别权重，自动适应长尾

**消融实验结论**（Table 3）：
- Seesaw vs Focal+BalancedSampling：**F1 提升 ~7 个点**（27.50 → 20.91）
- Seesaw 在所有指标上大幅优于 Balanced Focal Loss

#### 2.2.2 Custom Venom Loss（领域代价矩阵）

**代码位置**：`snakeclef/losses.py:class CompositeLoss`

```
代价矩阵设计（论文 Section 3.3.2）：
  cost(预测正确)         = 0
  cost(无毒→无毒误判)     = 1
  cost(无毒→有毒误判)     = 2
  cost(有毒→有毒误判)     = 2
  cost(有毒→无毒误判)     = 5  ← 最严重，惩罚最重
```

**实现方式**：
1. 构建 n_classes × n_classes 的代价矩阵
2. 将 Softmax 概率与目标类别对应的代价向量逐元素相乘
3. 求和得到 Venom Loss
4. 与 Seesaw Loss 加权组合：`total_loss = seesaw_loss + poison_loss_weight * venom_loss`

**消融实验结论**（Table 13）：
- 加入 Venom Loss 后 Track1 提升 **2.68**（74.46→77.14），Track2 降低 **185**（1375→1190），F1 提升 **1.97**（23.17→25.14）
- **关键发现**：领域代价矩阵不仅改善特定安全指标，还提升了整体 F1

**注意**：
- Class-weighted Venom Loss **有害**（Table 5），不加权重更好
- Sub-center ArcFace Loss 加入后**降低所有指标**（Table 2）

#### 2.2.3 LogitNorm

**代码位置**：`snakeclef/losses.py:CompositeLoss.forward()` 开头

```python
if self.use_logitnorm:
    norms = torch.norm(outputs, p=2, dim=-1, keepdim=True) + 1e-7
    outputs = torch.div(outputs, norms) / self.logitnorm_t  # t=0.01
```

**消融实验结论**（Table 12）：
- 对于 SnakeCLEF（闭集分类）：轻微负面或无影响
- 对于 FungiCLEF（开放集分类）：**显著正面**
- **建议**：如果是纯闭集任务，可以不用 LogitNorm

#### 2.2.4 其他尝试过的 Loss（不推荐）

| Loss | 结果 | 原因 |
|------|------|------|
| Focal Loss + Balanced Sampling | 明显更差 | Seesaw 的 mitigation+compensation 机制更优 |
| Sub-center ArcFace | 降低所有指标 | 多聚类中心的假设可能不适用于需要稠密分类的场景 |
| Class-weighted Venom Loss | 降低所有指标 | 原始 Venom Loss 不需要额外加权 |

### 2.3 数据增强（Training Augmentation）

**代码位置**：`snakeclef/datasets.py:get_train_transform()` 和配置文件 `snakeclef/conf/config.yaml`

```
训练增强 Pipeline（按顺序）：
  1. Resize to 768 (bicubic interpolation, antialias=True)
  2. RandomCrop 384×384 (square)
  3. TrivialAugmentWide (bicubic interpolation)
  4. ToImage + ToDtype(float32, scale=True)
  5. Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])  # ImageNet 标准归一化
```

**关键超参数**（`config.yaml`）：

```yaml
train_aug:
  trivial_aug: True       # 使用 TrivialAugmentWide
  auto_aug: False          # 不用 AutoAugment
  random_aug: False        # 不用 RandAugment
  gridmask_prob: null      # 不用 GridMask

train:
  image_resize: 384        # 训练分辨率（推理用更高分辨率）
  dropout_rate: 0.2        # 标准值（部分实验用 0.4）
```

**重要发现**：

| 增强方式 | 效果 | 说明 |
|----------|------|------|
| TrivialAugmentWide | ✅ 推荐 | 无需调参，效果稳定 |
| **MixUp / CutMix / RandoMix** | ❌ 故意不用 | 细粒度任务中类间差异小，混合增强破坏判别特征 |
| **Random Erasing** | ❌ 有害 | Table 9：Track1 私有榜降低 2.31，可能擦除了关键细粒度特征 |
| GridMask | ❌ 不用 | 类似 Random Erasing 的问题 |
| Horizontal Flip (p=0.5) | 训练时未显式加 | 但在推理时作为 TTA 使用 |

### 2.4 优化器与训练配置

**代码位置**：`snakeclef/train.py:train_model()`

```yaml
# 关键训练超参数 (config.yaml)
train:
  optimizer: AdamW
  weight_decay: 0.05       # AdamW 的解耦权重衰减
  max_norm: 1.0            # 梯度裁剪阈值
  
  # 学习率（两阶段）
  lr: 1e-3                 # 初始 LR（仅训练分类头，backbone 冻结）
  lr_after_unfreeze: 5e-5  # backbone 解冻后的 LR
  
  # 学习率调度
  lr_scheduler: "reducelronplateau"
  lr_scheduler_patience: 4   # 连续4个epoch不改善则LR×0.1
  
  # 训练控制
  epochs: 200
  early_stop_thresh: 10    # 连续10个epoch不改善则停止
  fine_tune_after_n_epochs: 5  # 前5个epoch只训练分类头，之后解冻全部
  
  # Batch Size（根据模型调整）
  batch_size: 24           # CAFormer-S36
  # batch_size: 32          # Metaformer-0
  # batch_size: 40          # CAFormer-S18
  
  num_dataloader_workers: 16
  worker_timeout_s: 360
```

**训练流程**（`train.py:train_model()`）：

```
Phase 1 (epoch 1-5): 
  - Backbone 冻结，仅训练分类头
  - LR = 1e-3, dropout_rate

Phase 2 (epoch 6-200):
  - 解冻全部参数
  - LR 降至 5e-5 (手动) 或通过 LR finder 自动确定
  - 重新创建优化器和调度器
  
每 epoch 后：
  - ReduceLROnPlateau 根据 val_loss 调整 LR
  - 如 val_loss 为历史最佳 → 保存 checkpoint
  - 如连续 early_stop_thresh=10 个 epoch 不改善 → 提前停止
```

**权重衰减特殊处理**（`train.py:add_weight_decay()`）：
- 1维参数（bias、LayerNorm 等）weight_decay=0
- 其他参数 weight_decay=0.05

### 2.5 数据划分策略

**4种数据划分**（论文 Section 3.6）：

```
原始数据：训练集(前) + 验证集(后) + 稀有类补充集

将训练+验证合并后：
  Split A: 前 90% 训练 / 后 10% 验证（最接近原始划分）
  Split B: 后 90% 训练 / 前 10% 验证
  Split C: 中间 90% 训练 / 两端各 5% 验证
  Split D: 原始训练/验证划分（不做合并）

尾类特殊处理：如果某类 < 4 个样本，全部用于训练
```

**为何重要**：Table 10 显示不同划分间差异（Track1: 78.06~79.47）**大于**很多技术改进（LogitNorm、Random Erasing、HFlip 等），说明数据划分对单模型影响显著。因此用多个划分训练 ensemble 可有效降低方差。

---

## 3. 训练策略总结

### 3.1 推荐训练配置（针对长尾细粒度任务）

```
模型:        CAFormer-S18 (或更大模型根据资源)
预训练:      ImageNet-21K (timm)
损失函数:     SeesawLoss(p=0.8, q=2.0) + 可选领域代价Loss
数据增强:     Resize(768) → RandomCrop(384) → TrivialAugmentWide
Dropout:      0.2 (分类头前)
优化器:       AdamW, weight_decay=0.05
学习率:       1e-3 (分类头, 5 epoch) → 5e-5 (全模型)
调度器:       ReduceLROnPlateau(patience=4)
早停:         patience=10 epochs
梯度裁剪:     max_norm=1.0
```

### 3.2 不应做的事

| ❌ 避免 | 原因 |
|--------|------|
| MixUp / CutMix 等混合增强 | 细粒度任务类间差异小，破坏判别特征 |
| Random Erasing | 擦除关键细粒度特征，伤害泛化 |
| Balanced Focal Loss | Seesaw Loss 在所有指标上显著更优 |
| Class-weighted Venom Loss | 反而降低性能 |
| Sub-center ArcFace | 不利于稠密分类场景 |
| LogitNorm（仅闭集） | 对闭集任务无帮助 |

### 3.3 Progressive Learning

代码中实现了 Progressive Learning（`train.py:get_progression_params()`），但最终配置中 `train_progressively: False`，未实际使用。如果要用：

```yaml
progressive-learning:
  start_image_size: 384
  end_image_size: 384
  start_dropout: 0.1
  end_dropout: 0.3
  start_batch_size: 32
  end_batch_size: 32
  progression_epochs: 100
```

即训练过程中逐步增加分辨率、Dropout 和 Batch Size。

---

## 4. 推理优化

### 4.1 Test-Time Augmentation（TTA）组合

**代码参考**：`snakeclef/test-time-augmentations.ipynb`

```
推荐推理 Pipeline（按重要性排序）：

1. 提高推理分辨率  384→576  【最重要！单独使用提升最大】
2. Multi-Instance 平均      【多个视角的图像取平均概率】
3. Horizontal Flip 平均      【原图 + 水平翻转 取平均】
4. Multi-Crop 平均           【3个重叠crop 取平均】
5. 多模型 Ensemble            【不同数据划分训练的模型取概率平均】
```

**消融实验关键数据**（Tables 7, 8）：

| 配置 | Track1↑ | Track2↓ | F1↑ |
|------|---------|---------|-----|
| 基础 (384 no-TTA) | 76.16 | 1251 | 23.29 |
| + 推理分辨率 576 | 78.39 | 1109 | 26.75 |
| + HFlip | 79.92 | 1023 | 30.89 |
| + Multi-Instance | 79.87 | 1025 | 30.57 |
| + HFlip + Multi-Instance | 79.94 | 1024 | 31.23 |
| + 4x Ensemble (best) | **81.2** | **945** | **33.35** |

### 4.2 推理分辨率选择

| 推理分辨率 | 效果 | 计算成本 |
|-----------|------|---------|
| 384 (训练分辨率) | 基线 | 1x |
| 576 | **显著提升，性价比最高** | ~2.25x 像素 |
| 768 | F1 最好，但 Track1/2 略低于 576 | ~4x 像素 |

**发现**：576×576 是性价比最优的推理分辨率，768×768 只在 F1 上有边际提升。

### 4.3 Ensemble 策略

**最佳 Ensemble 组成**（Table 6）：

```
方案1 (公开榜最优):
  CAFormer-S18 × 4:
    - Split A (无 Random Erasing)
    - Split B (有 Random Erasing)  
    - Split C (有 Random Erasing)
    - Split D (无 Random Erasing)
  推理: 576, hflip, multi-instance
  结果: Track1=81.2, Track2=945, F1=33.35

方案2 (私有榜 Track1/Track2 最优):
  CAFormer-S18 × 3 + CAFormer-S36 × 1:
    - Split B, C (RE), Split D
    - Split D
  结果: Track1=79.96, Track2=2481, F1=30.2
```

**Ensemble 方式**：简单概率平均 → 选最大概率类别（不需要复杂加权）

**重要对比**（Table 11）：
- 多分辨率 inference 组合（576+652）**不如**同等计算预算下增加 ensemble 模型数量
- 大 ensemble > 多分辨率平均

### 4.4 FixRes Fine-tuning

论文测试了 FixRes（在目标推理分辨率下微调分类头，不使用训练增强），结果：
- **Table 1**：FixRes **降低了**所有指标（Track1: 79.41→78.08）
- **结论**：不需要 FixRes fine-tuning，直接高分辨率推理即可

### 4.5 Multi-Instance 推理

当每个观察（observation）有多张图像时：
1. 每张图的预测概率独立计算
2. 取所有实例的概率平均值
3. 选择最大概率类别

这是**单模型提升最大**的 TTA（Table 7），但计算成本也最高（需要处理所有实例图像）。

---

## 5. 对 image-classification 项目的借鉴

### 5.1 直接可用的技术

| 技术 | 优先级 | 实施难度 | 预期收益 |
|------|--------|---------|---------|
| **Seesaw Loss** | 🔴 最高 | 低（复制 losses.py） | F1 +5~7 点（长尾场景） |
| **高分辨率推理 (1.5x)** | 🔴 最高 | 低（改推理参数） | Track1 +2~3 |
| **TrivialAugmentWide** | 🟡 高 | 低（改 transform） | 提升鲁棒性 |
| **HFlip TTA** | 🟡 高 | 低（加推理逻辑） | +1~2 |
| **领域代价矩阵 Loss** | 🟢 中 | 中（需定义代价） | 安全指标大幅改善 |
| **多划分 Ensemble** | 🟢 中 | 中（多次训练） | +2~4 |
| **CAFormer 架构** | 🟢 中 | 低（timm直接调用） | 强于EfficientNet/Metaformer |
| **LogitNorm** | 🔵 低 | 低（3行代码） | 仅开放集有益 |

### 5.2 Seesaw Loss 实现要点

从 `snakeclef/losses.py` 直接移植：

```python
class SeesawLoss(torch.nn.Module):
    def __init__(self, num_classes, p=0.8, q=2.0, eps=1e-2, device='cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.p = p       # mitigation factor power
        self.q = q       # compensation factor power
        self.eps = eps
        # accumulate 记录每个类别累计出现次数（跨 batch 累加）
        self.register_buffer('accumulate', 
                             torch.zeros(self.num_classes, dtype=torch.float))
    
    def forward(self, output, target):
        # 更新每个类别的累计样本数
        for unique in target.unique():
            self.accumulate[unique] += (target == unique.item()).sum()
        
        onehot_target = one_hot(target, self.num_classes)
        seesaw_weights = output.new_ones(onehot_target.size())
        
        # Mitigation: 根据类频率比降低尾类惩罚
        if self.p > 0:
            matrix = self.accumulate[None, :].clamp(min=1) / \
                     self.accumulate[:, None].clamp(min=1)
            index = (matrix < 1.0).float()
            sample_weights = matrix.pow(self.p) * index + (1 - index)
            mitigation_factor = sample_weights[target.long(), :]
            seesaw_weights = seesaw_weights * mitigation_factor
        
        # Compensation: 增强对高分误分类的惩罚
        if self.q > 0:
            scores = softmax(output.detach(), dim=1)
            self_scores = scores[torch.arange(0, len(scores)), target.long()]
            score_matrix = scores / self_scores[:, None].clamp(min=self.eps)
            index = (score_matrix > 1.0).float()
            compensation_factor = score_matrix.pow(self.q) * index + (1 - index)
            seesaw_weights = seesaw_weights * compensation_factor
        
        output = output + (seesaw_weights.log() * (1 - onehot_target))
        return cross_entropy(output, target, reduction='none').mean()
```

**注意事项**：
- `accumulate` buffer 在训练中跨 batch 累加，记录的是**运行中的累计**（非整个 epoch 的统计）
- `reduction='none'` 后用 `.mean()`，不要改成 `reduction='mean'`
- 数值稳定性：`.clamp(min=1)` 防止除零，`eps=1e-2` 防止分母过小

### 5.3 自定义领域代价 Loss 实现要点

如果有类似"某些误判比另一些更严重"的需求，可以仿照 Venom Loss：

```python
def create_domain_cost_loss(n_classes, cost_matrix_config):
    """
    cost_matrix_config: dict 定义各种混淆类型的代价
    例如: {
        "dangerous_to_safe": 5.0,
        "safe_to_dangerous": 2.0,
        "same_category_misclass": 1.0,
    }
    """
    cost_matrix = torch.ones((n_classes, n_classes))
    # 填充代价矩阵...
    cost_matrix[torch.arange(n_classes), torch.arange(n_classes)] = 0  # 正确=0
    
    def domain_loss(outputs, labels):
        sm_outs = softmax(outputs, dim=-1)
        costs = cost_matrix[labels, :]
        return (costs * sm_outs).sum(axis=-1).mean()
    
    return domain_loss
```

### 5.4 Training Config 模板

基于 `config.yaml` 的推荐配置（可直接用于 image-classification 项目）：

```yaml
train:
  model_id: "caformer_s18.sail_in22k_ft_in1k_384"
  epochs: 200
  lr: 1e-3
  lr_after_unfreeze: 5e-5
  pretrained: true
  early_stop_thresh: 10
  loss_function: "seesaw"
  balanced_sampler: false
  max_norm: 1.0
  image_resize: 384
  dropout_rate: 0.2
  weight_decay: 0.05
  fine_tune_after_n_epochs: 5
  lr_scheduler: "reducelronplateau"
  lr_scheduler_patience: 4
  use_venom_loss: true         # 替换为你的领域loss
  use_logitnorm: false         # 闭集关
  use_class_weights_venom_loss: false
  use_lr_finder: false
  batch_size: 40               # CAFormer-S18
  num_dataloader_workers: 16
  worker_timeout_s: 360

train_aug:
  trivial_aug: true
  auto_aug: false
  random_aug: false
  gridmask_prob: null
```

### 5.5 推理配置模板

```python
# 推荐推理设置
INFERENCE_CONFIG = {
    "image_size": 576,          # 1.5x 训练分辨率
    "hflip_tta": True,          # 水平翻转平均
    "multi_instance_avg": True, # 多实例平均（如有）
    "multi_crop_tta": False,    # 成本高，优先增加 ensemble
    "multi_res_tta": False,     # 不如大 ensemble
    "ensemble_models": [        # 多模型概率平均
        "model_split_A.pth",
        "model_split_B.pth", 
        "model_split_C.pth",
        "model_split_D.pth",
    ],
    "ensembling_method": "probability_average",  # 简单平均即可
}
```

### 5.6 不要照搬的部分

1. **Random Erasing**：论文明确发现对细粒度有害，不建议（除非你有实验支持）
2. **MixUp/CutMix**：论文故意不用，细粒度场景下混合增强破坏判别特征
3. **FixRes Fine-tuning**：论文发现没用，直接高分辨率推理即可
4. **Sub-center ArcFace**：论文发现降低性能
5. **Balanced Focal Loss**：Seesaw 在所有指标上显著更好

### 5.7 计算资源参考

论文所有实验在**单张 NVIDIA RTX 4090 (24GB)** 上完成：
- CAFormer-S18 batch_size=40
- CAFormer-S36 batch_size=24
- Metaformer-0 batch_size=32
- 推理：576×576 分辨率下，4×ensemble + hflip + multi-instance 可行

---

## 6. 关键代码文件索引

| 文件 | 作用 | 核心内容 |
|------|------|---------|
| `snakeclef/losses.py` | Loss 实现 | SeesawLoss, CompositeLoss (Venom+LogitNorm) |
| `snakeclef/train.py` | 训练主流程 | 两阶段训练、LR调度、早停 |
| `snakeclef/closedset_model.py` | 模型构建 | build_model, load_pretrained_metaformer |
| `snakeclef/datasets.py` | 数据加载 | CustomImageDataset, 增强pipeline |
| `snakeclef/conf/config.yaml` | 配置 | 所有超参数默认值 |
| `snakeclef/evaluate.py` | 评估 | 推理、提交生成 |
| `snakeclef/augmentations.py` | 自定义增强 | GridMask 等 |

---

## 7. 论文核心发现速查表

| 发现 | 证据 | 置信度 |
|------|------|--------|
| Seesaw Loss >> Focal Loss+Balanced | Table 3: F1 +7 | ⭐⭐⭐ 高 |
| 高分辨率推理是最大单次提升 | Table 8: Track1 +2 | ⭐⭐⭐ 高 |
| Random Erasing 对细粒度有害 | Table 9: Track1 -2.31 | ⭐⭐⭐ 高 |
| CAFormer > Metaformer (无元数据时) | Table 4: 全面领先 | ⭐⭐ 中 |
| 领域代价值 Loss 不仅提升安全指标，还提升 F1 | Table 13 | ⭐⭐⭐ 高 |
| 大 Ensemble > 多分辨率平均 | Table 11 | ⭐⭐ 中 |
| FixRes Fine-tuning 无益 | Table 1 | ⭐⭐ 中 |
| Class-weighted Venom Loss 有害 | Table 5 | ⭐⭐ 中 |
| Sub-center ArcFace 有害 | Table 2 | ⭐⭐ 中 |
| LogitNorm 对闭集无帮助 | Table 12 | ⭐⭐ 中 |
| 数据划分影响 > 多项技术改进 | Table 10 | ⭐⭐⭐ 高 |

---

*最后更新: 2026-05-10 | 基于 SnakeCLEF 2024 论文及 GitHub 代码库分析*
