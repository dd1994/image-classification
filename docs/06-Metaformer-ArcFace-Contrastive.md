# 06 - MetaFormer + ArcFace + Contrastive Learning (SnakeCLEF2023 第三名方案)

> 论文: *Metaformer Model with ArcFaceLoss and Contrastive Learning for SnakeCLEF2023 Fine-Grained Classification*
> 作者: Zhennan Shi, Huazhen Chen, Chang Liu, Jun Qiu (北京信息科技大学 / 天津大学)
> 成绩: **Track1 88.30% (第三名)** | Track2 Score 1613 (第三名)
> GitHub: https://github.com/BAOfanTing/SnakeCLEF2023

---

## 1. 论文概述

### 1.1 任务背景

SnakeCLEF2023 是一个**细粒度蛇类物种识别**竞赛，属于 LifeCLEF2023 的一部分。核心挑战：

| 挑战 | 描述 |
|------|------|
| **大类内差异** | 同一蛇种成年/亚成年体外观差异巨大（如 Red-crowned Crane） |
| **小类间差异** | 不同蛇种外观极其相似（如 Coral Snake vs Milk Snake 几乎相同的黑白红环纹） |
| **长尾分布** | 1785 个类别，样本数从 2000+ 到个位数，严重不平衡 |
| **多图像观测** | 每个 observation_id 对应多张图片，需汇总预测 |

数据集规模：训练集约 180K 样本，测试集约 14K 样本（较 2022 年减 10 万训练样本但增 200+ 类别，难度更大）。

### 1.2 方案总览

该方案采用**三条技术路线组合**：

```
MetaFormer Backbone (多模态融合)
    + ArcFace Loss (长尾处理)
    + SimCLR Contrastive Learning (自监督预训练)
    + TTA + 多模型集成 + 后处理投票
```

### 1.3 消融实验结果

| 模型 | 验证集 Acc | 参数量 | 输入尺寸 |
|------|-----------|--------|----------|
| EfficientNet-B7 | 67.5% | 66M | 384×384 |
| MetaFG-0 | 72.6% | 28M | 384×384 |
| **MetaFG-2** | **76.4%** | 81M | 384×384 |
| MetaFG-2 + SimCLR | **83.8%** | 81M | 512×512 |

关键发现：SimCLR 对比学习将准确率从 76.4% 提升至 83.8%（+7.4 个百分点），是最有效的单一改进。

---

## 2. 识别率提升技巧

### 2.1 MetaFormer 多模态架构

#### 架构设计（5 阶段网络）

```
Stage 0: Conv Stem (3×3 conv × 3) → 下采样 4×
Stage 1: MBConv Blocks × N (含 Squeeze-and-Excitation)
Stage 2: MBConv Blocks × N (含 Squeeze-and-Excitation)
Stage 3: Transformer Blocks × N (含相对位置偏置) + cls_token + meta_tokens
Stage 4: Transformer Blocks × N (含相对位置偏置) + cls_token + meta_tokens
```

**三种模型规模** (`code/main/models/MetaFG.py`):

| 模型 | Conv Embed Dims | Attn Embed Dims | Conv Depths | Attn Depths | Heads |
|------|-----------------|-----------------|-------------|-------------|-------|
| MetaFG-0 | [64, 96, 192] | [384, 768] | [2, 2, 3] | [5, 2] | 8 |
| MetaFG-1 | [64, 96, 192] | [384, 768] | [2, 2, 6] | [14, 2] | 8 |
| **MetaFG-2** | [128, 128, 256] | [512, 1024] | [2, 2, 6] | [14, 2] | 8 |

#### 多模态融合机制 (`code/main/models/MetaFG_meta.py`)

Meta 信息包括 **Code（国家代码）、Endemic（是否特有）、Binomial name（双名法名称）**，处理流程：

```
1. One-hot 编码
2. Non-Linear Embedding: Linear → ReLU → LayerNorm → ResNormLayer
3. 将 meta tokens 与 visual tokens + cls_token 拼接
4. 输入 Transformer Stage 3/4 进行联合注意力计算
```

**关键代码路径**: `code/main/models/MetaFG_meta.py` 的 `forward_features()` 方法：
- `extra_token_num` 控制 cls_token + meta_tokens 总数（configs 中设为 2 或 3）
- `META_DIMS` 配置各 meta 特征的维度（如 `[217]` 表示单个 217 维特征，`[4, 3]` 表示两个特征分别 4 维和 3 维）
- Meta 特征通过独立的 `meta_head_1`/`meta_head_2` 分别投影到 Stage 3 和 Stage 4 的 token 维度
- 采用 `ResNormLayer`（残差 + LayerNorm）增强 meta embedding 的非线性表达能力

**Meta Masking 策略** (正则化技巧):
- 训练时以线性递减的概率随机将 meta 置零（`mask_prob - cur_epoch/total_epoch`）
- 迫使模型不完全依赖 meta 信息，增强泛化能力
- 代码在 `MetaFG_Meta.forward()` 中实现

#### 关键代码特征

`code/main/models/MetaFG_meta.py`:
```python
# Meta 信息编码器
meta_head_1 = nn.Sequential(
    nn.Linear(meta_dim, attn_embed_dims[0]),
    nn.ReLU(inplace=True),
    nn.LayerNorm(attn_embed_dims[0]),
    ResNormLayer(attn_embed_dims[0]),
)  # 用于 Stage 3

# 将 meta tokens 拼入 extra_tokens
extra_tokens_1 = [self.cls_token_1] + [meta_1, meta_2, ...]
extra_tokens_2 = [self.cls_token_2] + [meta_1_proj, meta_2_proj, ...]
```

### 2.2 ArcFace Loss 实现

#### 原理

ArcFace（Additive Angular Margin Loss）在特征空间的角度层面添加 margin，增大类间距离、缩小类内距离：

```
L_arcface = -log( exp(s * cos(θ_y + m)) / (exp(s * cos(θ_y + m)) + Σ_j≠y exp(s * cos(θ_j))) )
```

- `s` (scale): 特征归一化后的缩放因子
- `m` (margin): 角度 margin，强制类间分离

#### 代码实现 (`code/main/main_CallArcloss.py`)

使用 `pytorch_metric_learning` 库的内置实现：

```python
from pytorch_metric_learning import losses

# ArcFace 初始化
criterion = losses.ArcFaceLoss(
    embedding_size=1784,    # 特征维度=类别数
    num_classes=1784        # 类别数
).to(torch.device('cuda'))

# 为 ArcFace 参数单独设置优化器
loss_optimizer = torch.optim.Adam(criterion.parameters(), lr=0.01)
```

**训练循环中的调用方式**:
```python
_, outputs = model(samples, meta)           # outputs: (feat, logits)
loss = criterion(outputs, targets.argmax(dim=1))  # ArcFace 直接在 logits 上计算
loss_optimizer.zero_grad()
loss.backward()
loss_optimizer.step()
```

**关键注意事项**: 
- 代码中 `main.py`（基础版）使用 `LabelSmoothingCrossEntropy`（smoothing=0.1）
- `main_CallArcloss.py`（ArcFace版）专门添加了 ArcFace loss 优化器
- ArcFace 的参数单独用一个 Adam 优化器训练（lr=0.01），不与 backbone 共享 optimizer
- 当 Mixup > 0 时，使用 `SoftTargetCrossEntropy` + ArcFaceLoss 双重 loss

### 2.3 SimCLR 对比学习

#### 原理

SimCLR 通过最大化同一图像不同增强视图之间的相似度来学习有意义的特征表示：

```
1. 对输入图像 x 应用两次随机增强 → x_i, x_j
2. Encoder f(·) → h_i, h_j
3. Projection Head g(·) → z_i, z_j
4. 最大化 z_i 和 z_j 的互信息（InfoNCE Loss）
```

#### 代码实现 (`code/main/main_simclr.py`)

**数据准备**: 数据加载器返回 6 通道图像（原始 3 通道 + 增强 3 通道），在训练循环中拆分：

```python
raw_samples, aug_samples = samples[:, :3, :, :], samples[:, 3:, :, :]
# 扩展到 3 通道（兼容模型输入）
y = torch.zeros(config.DATA.BATCH_SIZE, 3, 384, 384)
y[:, :aug_samples.size(1), :, :] = aug_samples
aug_samples = y
```

**NT-Xent Loss** 使用 `pytorch_metric_learning` 的分布式封装：

```python
from pytorch_metric_learning import losses as pml_losses
from pytorch_metric_learning.utils import distributed as pml_dist

loss_fn = pml_losses.NTXentLoss(temperature=0.07)
loss_fn = pml_dist.DistributedLossWrapper(loss=loss_fn)
```

**混合损失函数**:
```python
# 分类损失 + 对比损失
loss = criterion(temp_outputs, temp_targets) + 0.001 * loss_fn(feats, targets)
```
- 对比损失权重仅为 0.001，作为辅助正则化项
- 温度参数 τ=0.07（NTXentLoss）
- 论文中提到 InfoNCE 温度设为 0.25

### 2.4 数据增强策略

| 增强方法 | 参数 | 代码位置 |
|----------|------|----------|
| RandAugment | `rand-m9-mstd0.5-inc1` | `config.py: AUG.AUTO_AUGMENT` |
| Mixup | α=0.8, prob=1.0 | `config.py: AUG.MIXUP` |
| CutMix | α=1.0, switch_prob=0.5 | `config.py: AUG.CUTMIX` |
| Random Erase | prob=0.25, mode='pixel' | `config.py: AUG.REPROB` |
| Color Jitter | 0.4 | `config.py: AUG.COLOR_JITTER` |
| 随机翻转 | 水平 + 垂直 | 论文 Section 3.5 |
| 随机旋转 | 45° | 论文 Section 3.5 |

### 2.5 Test Time Augmentation (TTA)

```python
# 论文 Section 3.5: TTA 策略
# 对测试图像进行多次增强（扩展、翻转、旋转）
# 取多次预测的均值作为最终预测
```

### 2.6 后处理与集成

#### 单模型后处理 (`code/main/post_process.py`)

每个 observation_id 对应多张图片，后处理策略：

```python
# 对每个 observation_id 的所有图片预测：
# 1. 统计每个类别的投票数
# 2. 如果最高票出现平局，选择置信度最高的类别
for obv_id, class_id_scores in observation_to_classes.items():
    # 取最高票
    select_cls = most_common_vote
    # 平局时取最高置信度
    if tie:
        select_cls = highest_confidence_class
```

#### 模型集成

论文最终方案融合了 **3 个模型**（Section 3.5）：
- MetaFormer-0
- MetaFormer-2
- MetaFormer-2 + SimCLR

集成方式：对每个 observation_id，从 3 个模型的输出 CSV 中选取出现最多的类别作为最终预测。

---

## 3. 训练策略

### 3.1 训练配置总览

| 超参数 | 论文值 | 代码默认值 | 说明 |
|--------|--------|-----------|------|
| **Optimizer** | AdamW | AdamW | `code/main/optimizer.py` |
| **Base LR** | 5e-5 | 5e-4 | 论文更低，代码更高 |
| **Weight Decay** | 0.05 | 0.05 | |
| **Batch Size** | 22 | 128 | 单卡 RTX 3090 最大容量 |
| **Epochs** | 100 | 300 | |
| **Warmup Epochs** | 20 | 20 | |
| **Warmup LR** | 5e-8 | 5e-7 | |
| **Min LR** | 5e-7 | 5e-6 | |
| **LR Schedule** | Cosine | Cosine | `code/main/lr_scheduler.py` |
| **Grad Clip** | 5.0 | 5.0 | |
| **AMP** | O1 | O1 | NVIDIA Apex 混合精度 |
| **Label Smoothing** | - | 0.1 | |
| **Drop Path Rate** | - | 0.1 | |
| **Image Size** | 384 | 384 | MetaFG-2 最终配置 |
| **SimCLR Finetune Size** | 512 | - | 论文提到大尺寸微调效果不佳 |

### 3.2 学习率缩放公式

```python
# code/main/main.py L295-297
linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
```

即：`实际LR = base_lr × batch_size × num_gpus / 512`

论文情况（1×RTX3090, bs=22）：实际 LR = 5e-5 × 22 / 512 ≈ 2.15e-6

### 3.3 训练流程

```
第一阶段: 标准监督训练
  - Backbone: MetaFG_meta_2 (81M)
  - Loss: LabelSmoothingCrossEntropy (smoothing=0.1) 或 ArcFaceLoss
  - 数据增强: RandAugment + Mixup + CutMix
  - Meta masking 线性递减

第二阶段: SimCLR 对比预训练 + 微调
  - 先使用 SimCLR + NT-XentLoss 自监督预训练
  - 再使用监督 loss + 0.001×对比 loss 联合训练
  - 输入尺寸: 512×512（论文发现此尺寸微调效果不佳，但预训练有效）
```

### 3.4 ArcFace 训练注意事项

- ArcFace 参数的优化器独立于 backbone（`loss_optimizer = Adam(lr=0.01)`）
- 与 Mixup 同时使用时需要注意：Mixup 混合了标签，ArcFace 需要整数标签
- 代码中通过 `targets.argmax(dim=1)` 处理 mixup label

### 3.5 关键训练代码文件

| 文件 | 功能 |
|------|------|
| `code/main/main.py` | 标准训练入口（LabelSmoothingCrossEntropy） |
| `code/main/main_CallArcloss.py` | ArcFace Loss 训练入口 |
| `code/main/main_simclr.py` | SimCLR 对比学习训练入口 |
| `code/main/config.py` | 全局配置管理 |
| `code/main/lr_scheduler.py` | Cosine/Linear/Step 学习率调度器 |
| `code/main/optimizer.py` | AdamW 优化器构建 |
| `code/main/models/build.py` | 模型构建工厂 |
| `code/main/models/MetaFG.py` | 基础 MetaFG（仅图像） |
| `code/main/models/MetaFG_meta.py` | 多模态 MetaFG_Meta（图像+meta） |
| `code/main/models/meta_encoder.py` | ResNormLayer 实现 |
| `code/main/models/MHSA.py` | Multi-Head Self-Attention Block |
| `code/main/models/MBConv.py` | MobileNetV2 Inverted Residual Block |
| `code/main/post_process.py` | 后处理（observation_id 投票） |
| `code/main/merge_multiple.py` | 多模型集成 |

---

## 4. 推理优化

### 4.1 推理流程

```
Test Image → 模型 → per-image 预测
    ↓
observation_id 分组
    ↓
每组的图片预测进行多数投票
    ↓ （平局）
最高置信度的预测胜出
    ↓
多模型集成（MetaFG-0 + MetaFG-2 + MetaFG-2-SimCLR）
    ↓
最终 CSV 输出
```

### 4.2 推理优化要点

1. **Mixed Precision (AMP O1)**: 训练使用 `apex.amp` 混合精度，推理可保持 FP16 加速
2. **TTA**: 测试时对单张图片做多次增强取平均（论文提到的 flip + rotation）
3. **多模型集成**: 3 个模型独立推理后投票，无额外计算开销
4. **Meta 信息处理**: 测试时不做 meta masking（`cur_mask_prob=0`）

---

## 5. 对 image-classification 项目的借鉴

### 5.1 可直接采用的技术

| 技术 | 优先级 | 实现难度 | 预期收益 |
|------|--------|----------|----------|
| ArcFace Loss | ⭐⭐⭐⭐⭐ | 低 | 显著提升细粒度分类（类间差异小的场景） |
| Mixup + CutMix | ⭐⭐⭐⭐⭐ | 低 | 已有基础，调整参数即可 |
| RandAugment | ⭐⭐⭐⭐ | 低 | timm 已内置 |
| Cosine LR + Warmup | ⭐⭐⭐⭐⭐ | 低 | 训练稳定性提升 |
| Label Smoothing | ⭐⭐⭐⭐ | 低 | 防止过拟合 |
| 多模型集成投票 | ⭐⭐⭐⭐ | 中 | 稳定提升 1-3% |

### 5.2 条件性采用的技术

| 技术 | 优先级 | 条件 | 说明 |
|------|--------|------|------|
| **SimCLR 对比学习** | ⭐⭐⭐ | 需要数据增强 pipeline 配合 | 显著提升特征质量（+7.4%），但训练时间加倍 |
| **Meta 信息融合** | ⭐⭐⭐ | 数据集需额外结构化信息 | 只有具备地理/类别元数据时才有意义 |
| **TTA** | ⭐⭐ | 需额外的推理时间 | 在大规模推理时性价比需权衡 |
| **EMA (指数移动平均)** | ⭐⭐ | - | 论文未使用但常见于比赛方案 |

### 5.3 ArcFace Loss 集成建议

```python
# 使用 pytorch_metric_learning 库
from pytorch_metric_learning import losses

arcface_loss = losses.ArcFaceLoss(
    embedding_size=num_classes,  # 与类别数一致
    num_classes=num_classes,
    scale=64.0,                  # 论文默认值
    margin=0.5,                  # 论文默认值
)

# 训练时
features, logits = model(images)
loss = arcface_loss(logits, labels)  # 或 combined with CE loss
```

**关键超参数**:
- `scale` (s): 默认 64，控制特征空间的尺度
- `margin` (m): 默认 0.5（弧度），控制类间分离程度
- 建议与 CrossEntropyLoss 结合使用（权重比如 1:0.1）

### 5.4 训练超参数推荐

基于该论文的最佳实践：

```yaml
# 推荐配置
optimizer: AdamW
base_lr: 5e-5              # 按 batch_size/gpu_count 线性缩放
weight_decay: 0.05
lr_scheduler: cosine
warmup_epochs: 20
warmup_lr: 5e-7
min_lr: 5e-6
epochs: 100-300
batch_size: 22-128          # 视 GPU 内存定
grad_clip: 5.0
amp: O1                     # 混合精度

# 数据增强
mixup: 0.8
cutmix: 1.0
mixup_prob: 1.0
auto_augment: rand-m9-mstd0.5-inc1
color_jitter: 0.4
random_erase_prob: 0.25
random_erase_mode: pixel

# 正则化
label_smoothing: 0.1
drop_path_rate: 0.1
```

### 5.5 项目集成路线图

```
阶段 1 (短期):
  ✅ 加入 ArcFace Loss 作为可选 loss 函数
  ✅ 完善 Mixup + CutMix 参数化配置
  ✅ 加入 Cosine Warmup LR Scheduler

阶段 2 (中期):
  加入 SimCLR 对比学习预训练流程
  实现 TTA 推理模式
  添加多模型集成工具

阶段 3 (长期):
  探索多模态 token 融合（如有 metadata）
  研究其他对比学习方法（MoCo v3, DINO, MAE）
  超参数自动搜索
```

### 5.6 核心代码片段参考

**ArcFace + CrossEntropy 联合损失**:
```python
class CombinedLoss(nn.Module):
    def __init__(self, num_classes, arcface_scale=64.0, arcface_margin=0.5, ce_weight=0.1):
        super().__init__()
        self.arcface = losses.ArcFaceLoss(num_classes, num_classes, scale=arcface_scale, margin=arcface_margin)
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight

    def forward(self, embeddings, logits, labels):
        loss_arcface = self.arcface(embeddings, labels)
        loss_ce = self.ce(logits, labels)
        return loss_arcface + self.ce_weight * loss_ce
```

**数据增强配置（参考论文）**:
```python
from timm.data import create_transform
transform = create_transform(
    input_size=384,
    is_training=True,
    auto_augment='rand-m9-mstd0.5-inc1',
    color_jitter=0.4,
    re_prob=0.25,
    re_mode='pixel',
    re_count=1,
)
```

---

## 附录: 论文关键数据汇总

| 项目 | 数值 |
|------|------|
| 数据集类别数 | 1,785 |
| 训练样本数 | ~180,000 |
| 测试样本数 | ~14,000 |
| 最终成绩 (Track1) | 88.30% |
| 最终成绩 (Track2) | 1613 |
| 比赛排名 | 第 3 名 |
| GPU | 1× NVIDIA RTX 3090 |
| 推理模型数 | 3 (集成) |
| 最佳单模型 | MetaFG-2 + SimCLR: 83.8% |
| MetaFG-2 参数量 | 81M |
