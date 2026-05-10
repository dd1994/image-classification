# 04 - Entropy-Guided Open-Set Fine-Grained Fungi Recognition

> **论文**: Entropy-guided Open-set Fine-grained Fungi Recognition  
> **作者**: Huan Ren†, Han Jiang†, Wang Luo†, Meng Meng†, Tianzhu Zhang* (USTC)  
> **会议**: CLEF 2023 / FGVC10 Workshop (CVPR 2023)  
> **排名**: FungiCLEF 2023 **第1名** (Public F1: 58.95% | Private F1: 58.36%)  
> **代码**: https://github.com/RenHuan1999/FungiCLEF2023-UstcAIGroup

---

## 一、论文概述

### 1.1 任务背景

FungiCLEF 2023 竞赛要求对真菌物种进行自动识别，核心挑战有三：

| 挑战 | 描述 |
|------|------|
| **细粒度识别 (Fine-grained)** | 不同物种间视觉差异极小（inter-class variation 小），同类内部差异大（intra-class variation 大） |
| **开放集识别 (Open-set)** | 测试集中存在未在训练集见过的"未知物种"，需正确标记为 `unknown` |
| **长尾分布 (Long-tailed)** | 训练集 1,604 类，最多 1913 张/类，最少仅 31 张/类 |

训练数据来自 Danish Fungi 2020 数据集，含有丰富的元数据（habitat、substrate、时间、地点等）。

### 1.2 核心贡献

1. **MetaFormer 作为细粒度基线**：利用 Conv+Transformer 混合架构融合视觉特征与元数据（meta information）
2. **Seesaw Loss 解决长尾分布**：动态调整不同类别的负样本梯度，平衡 head/tail classes
3. **熵引导的开放集识别器 (Entropy-guided Unknown Identifier)**：用预测熵替代 MSP/MLS，更鲁棒地识别未知类别
4. **毒蘑菇辨识辅助损失**：针对有毒物种设计额外的 poisonous/edible 分类损失

---

## 二、识别率提升技巧（含超参数）

### 2.1 元数据融合（Meta Information Fusion）

**核心思路**: 利用真菌观察的附属信息（地理位置、生境、基质、时间）辅助视觉分类。

**编码方式**:
| 元数据类型 | 维度 | 编码方式 |
|-----------|------|---------|
| Observe date (month, day) | 4 | 周期编码: `[sin(2π·month/12), cos(...), sin(2π·day/31), cos(...)]` |
| CountryCode | 34 | One-hot (34 类别) |
| Substrate | 32 | One-hot (32 类别) |
| Habitat | 31 | One-hot (31 类别) |
| **总计** | **101** | 拼接 → Non-linear Embedding → C-dim (与视觉特征维度一致) |

**融合方式**: 在 Transformer 层（Stage 3 & 4）中，元数据 token 与视觉 patch token、CLS token 一起输入 Relative Attention（相对位置注意力）。

**配置文件路径**: `configs/MetaFG_meta_0_384.yaml`, `configs/MetaFG_meta_2_384.yaml`

```yaml
# configs/MetaFG_meta_2_384.yaml
DATA:
  IMG_SIZE: 384
  ADD_META: True           # 启用元数据
MODEL:
  TYPE: MetaFG
  NAME: MetaFG_meta_2
  EXTRA_TOKEN_NUM: 5       # 1 CLS token + 4 meta tokens
  META_DIMS: [4, 34, 32, 31]  # 对应 4 种元数据各自的维度
  LABEL_SMOOTHING: 0.0
AUG:
  MIXUP: 0.                # 未使用 Mixup
  CUTMIX: 0.               # 未使用 CutMix
  REPROB: 0.3              # Random Erasing 概率
TEST:
  FIVE_CROP: True          # 推理时使用 five-crop
```

**Meta 编码实现** (`models/MetaFG_meta.py`):
```python
# 每种元数据通过独立的 embedding head 映射到 Transformer 维度
for ind, meta_dim in enumerate(meta_dims):
    meta_head_1 = nn.Sequential(
        nn.Linear(meta_dim, attn_embed_dims[0]),   # → 384 或 512
        nn.ReLU(inplace=True),
        nn.LayerNorm(attn_embed_dims[0]),
        ResNormLayer(attn_embed_dims[0]),
    )
    meta_head_2 = nn.Sequential(
        nn.Linear(meta_dim, attn_embed_dims[1]),   # → 768 或 1024
        nn.ReLU(inplace=True),
        nn.LayerNorm(attn_embed_dims[1]),
        ResNormLayer(attn_embed_dims[1]),
    )
```

**Meta Dropout (训练时随机屏蔽元数据)**:
```python
# MetaFG_meta.py 中的 forward 方法
if self.mask_type == 'linear':
    cur_mask_prob = self.mask_prob - self.cur_epoch / self.total_epoch
# 随机将部分样本的元数据置零，增强鲁棒性
mask = torch.ones_like(meta)
mask_index = torch.randperm(meta.size(0))[:int(meta.size(0) * cur_mask_prob)]
mask[mask_index] = 0
meta = mask * meta
```

- 初始遮罩概率 `mask_prob = 1.0`（线性衰减至 0），不使用 Mixup/CutMix
- 效果：论文 Table 7 显示，使用 meta + Seesaw + entropy + ℒpoi 达到 **F1=58.11%, Track1=0.2069**
- 对比纯视觉模型 InternImage-L（不使用 meta 信息）：F1=54.32%，差距约 **3.8 个百分点**

### 2.2 熵引导的未知类别识别 (Entropy-guided Unknown Identifier)

这是本文**最核心的创新**，也是对我们项目最有借鉴价值的部分。

#### 原理对比

| 方法 | 公式 | 问题 |
|------|------|------|
| **MSP** (Maximum Softmax Probability) | `p_max(x) = max(softmax(logits))` | Softmax 归一化丢失了特征幅度信息 |
| **MLS** (Maximum Logit Score) | `l_max(x) = max(logits)` | 已知/未知类别的分布边界不清晰，阈值 τ 难以选择（论文 Figure 3a） |
| **Entropy (本文)** | `e(x) = -Σ p_c(x) · log(p_c(x))` | 已知类别熵低（模型自信），未知类别熵高（不确定性大），分布边界清晰（Figure 3b） |

#### 关键发现 (Figure 3)

- 使用 MLS 时，验证集和测试集的 max logit 分布**没有明显边界**，阈值 τ 需要依赖检验集中未知样本数量的先验知识来设定
- 使用 **Entropy** 且在 `train+val` 上训练后，测试集上熵分布出现**清晰的分界**，阈值选择更直观鲁棒
- **额外使用验证集训练是关键**：验证集包含一些未知类样本，模型可以学习未知类的数据分布

#### 实现细节 (`post_avg_entropy.py`)

```python
from scipy.stats import entropy

# 1. 计算每个样本的熵
probs = torch.softmax(torch_score, dim=-1)
entropy_dict[k] = entropy(probs)   # 自然对数底

# 2. 熵阈值判断
eta = 0.7                          # 容差偏移量
if entropy_k < 4 or (entropy_k < 6 and max_idx in less_cls):
    f.write(f"{k},{max_idx}\n")    # 已知类
else:
    f.write(f"{k},{-1}\n")         # 未知类
```

**超参数**:
| 参数 | 值 | 说明 |
|------|-----|------|
| 熵阈值 τ | **4** | 熵 < 4 → 已知类 |
| 长尾类放宽阈值 | **6** | 稀有类的熵 < 6 也判为已知（因为样本少导致更不确定） |
| 容差 η | **0.7** | 在 post_avg_entropy.py 中用于 less_cls 的放宽 |
| `less_cls` | 37 个稀有类 ID | 预定义的长尾类别列表 |

**Ablation 对比 (Table 6)**:
| Unknown Identifier | F1 ↑ | Track1 ↓ | Track4 ↓ |
|-------------------|------|----------|----------|
| MLS | 54.98 | 0.3381 | 2.8853 |
| **Entropy** | **57.84** | **0.2101** | **1.4815** |

→ Entropy 相比 MLS：F1 提升 **+2.86%**，Track4 (unknown 惩罚) 降低 **~50%**！

### 2.3 Seesaw Loss — 长尾分布均衡

**论文公式** (Section 3.1):
```
L_seesaw(z) = -Σ y_i · log(p_i)
p_i = exp(z_i) / (Σ_{j≠i} S_{ij} · exp(z_j) + exp(z_i))
```
其中 S 是可调平衡因子，包含 mitigation factor（按类别样本数 ratio 压制 head class 对 tail class 的抑制）和 compensation factor（对分数过高的负类进一步惩罚）。

**代码实现** (`models/custom_loss.py`):
```python
class SeesawLoss(nn.Module):
    def __init__(self, p=0.8, q=2.0, num_classes=1604, eps=1e-2):
```

**超参数**:
| 参数 | 值 | 说明 |
|------|-----|------|
| `p` | **0.8** | Mitigation factor 的幂指数 |
| `q` | **2.0** | Compensation factor 的幂指数 |
| `eps` | **1e-2** | 除数平滑项 |
| `num_classes` | **1604** | 训练集已知类别数 |

**使用方式** (`main.py`):
```python
criterion = SeesawLoss(num_classes=1604).cuda()
genus_criterion = SeesawLoss(num_classes=566).cuda()  # 属级别分类也用 Seesaw
```

注意: 代码中 `SeesawLoss.forward()` **内置了** unknown 均匀分布约束 + 毒蘑菇分类损失，不需要额外单独调用。

### 2.4 毒蘑菇辨识增强

**论文公式** (Equation 3-4):
```
l_set(x) = mean(top-k({l_c(x) | c ∈ C_set})),   set ∈ {poi, edi}
L_poi(x) = -log[ exp(l_poi) / (exp(l_poi) + exp(l_edi)) ]
```

**代码实现** (`custom_loss.py` forward 中的 poison 部分):
```python
# 有毒物种掩码
poison_mask = torch.zeros(cls_score.shape[-1])
poison_mask[poison_species] = 1

# Top-5 average for poisonous logits
cls_score_poison = cls_score[poison_inds][:, poison_mask == 1]
topk_scores, _ = torch.topk(cls_score_poison, k=5, dim=1)
cls_score_poison = topk_scores.mean(1)

# Top-5 average for edible logits
cls_score_edible = cls_score[poison_inds][:, poison_mask != 1]
topk_scores, _ = torch.topk(cls_score_edible, k=5, dim=1)
cls_score_edible = topk_scores.mean(1)

# Binary classification loss
cls_score_poi = torch.stack([cls_score_poison, cls_score_edible], dim=1)
loss_cls_classes_poison = -F.log_softmax(cls_score_poi, dim=1)[:, 0]
```

**超参数**: `k=5` (top-k average)

**Ablation 结果 (Table 7)**:
| ℒ_poi | F1 ↑ | Track1 ↓ | Track2 ↓ |
|-------|------|----------|----------|
| ✗ | 57.80 | 0.2088 | **0.2865** |
| ✓ | **58.11** | **0.2069** | **0.2067** |

→ Track2 (毒/食混淆代价) 从 0.2865 降至 0.2067，**降低 28%**

### 2.5 未知类均匀分布约束

当使用 train+val 联合训练时，对验证集中的未知样本施加均匀分布约束：

```python
# custom_loss.py 中 SeesawLoss.forward()
if nov_inds.sum() > 0:
    cls_score_nov = cls_score[nov_inds]
    labels_nov = torch.ones_like(cls_score_nov) / cls_score_nov.shape[1]  # uniform
    loss_cls_classes_nov = -(labels_nov * F.log_softmax(cls_score_nov, dim=1)).sum(dim=-1)
    loss_cls_classes_nov = self.loss_weight * loss_cls_classes_nov.mean()
```

- 目标：让模型对未知样本输出均匀分布（所有已知类概率相等），从而使熵最大化
- 这是熵引导方法能工作的**前提**——必须让模型学会对未知样本"不偏向任何已知类"

---

## 三、训练策略

### 3.1 模型架构

| 模型 | 参数量 | Stage 配置 |
|------|--------|-----------|
| **MetaFormer-0** | ~150 MB | S0:64→S1:96→S2:192→S3:384→S4:768 |
| **MetaFormer-2** | ~393 MB | S0:128→S1:128→S2:256→S3:512→S4:1024 |
| **Ensemble** | ~543 MB | MetaFormer-0 + MetaFormer-2 (< 1GB 限制) |

架构特点：
- Stage 0-2: MBConv 卷积块（下采样 + 局部特征提取）
- Stage 3-4: Multi-Head Self-Attention (MHSA) 块（融合视觉+元数据）
- CLS token 聚合：Stage 3 和 Stage 4 的 CLS token 通过 Conv1d 聚合后输入分类头
- 额外输出 genus (属级) 分类头: `genus_head = nn.Linear(attn_embed_dims[-1], 566)`

### 3.2 训练超参数

| 超参数 | MetaFormer-0 | MetaFormer-2 |
|--------|-------------|-------------|
| **图像尺寸** | 384×384 | 384×384 |
| **Batch Size (per GPU)** | 36 | 18 |
| **GPU 数量** | 4 | 4 |
| **总 Batch Size** | 144 | 72 |
| **梯度累积步数** | - | 4 (等效 BS=288) |
| **学习率** | 5e-5 | 5e-5 |
| **最小学习率** | 5e-7 | 5e-7 |
| **Warmup 学习率** | 5e-8 | 5e-8 |
| **Warmup Epochs** | 1 | 1 |
| **总 Epochs** | 80 | 64 |
| **Weight Decay** | 0.05 | 0.05 |
| **优化器** | AdamW | AdamW |
| **AdamW β** | (0.9, 0.999) | (0.9, 0.999) |
| **AdamW ε** | 1e-8 | 1e-8 |
| **学习率调度** | Cosine Decay | Cosine Decay |
| **Gradient Clipping** | 5.0 | 5.0 |
| **混合精度** | O1 (Apex AMP) | O1 (Apex AMP) |
| **预训练权重** | `metafg_0_inat21_384.pth` | `metafg_2_inat21_384.pth` |

**学习率线性缩放** (`main.py`):
```python
linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
# 梯度累积时进一步缩放
if config.TRAIN.ACCUMULATION_STEPS > 1:
    linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
```

**训练命令** (`run_train.sh`):
```bash
# MetaFG_meta_2_384
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
    --nproc_per_node 4 --master_port 12346 main.py \
    --cfg ./configs/MetaFG_meta_2_384.yaml \
    --batch-size 18 --tag ${EXP_TAG} \
    --lr 5e-5 --min-lr 5e-7 --warmup-lr 5e-8 \
    --epochs 64 --warmup-epochs 1 \
    --dataset fungi \
    --pretrain ./pretrained_model/metafg_2_inat21_384.pth \
    --accumulation-steps 4 --num-workers 18 \
    --opts DATA.IMG_SIZE 384
```

### 3.3 数据增强

| 增强方法 | 参数 | 说明 |
|---------|------|------|
| Random Crop | 8%~100% of original size | 随机裁剪 |
| Resize | Bicubic interpolation | 上采样到 384×384 |
| Horizontal Flip | 50% | 水平翻转 |
| RandAugment | `rand-m9-mstd0.5-inc1` | 光度+几何增强（按 Swin Transformer 配置） |
| Random Erasing | 概率 25%, mode='pixel' | 随机遮挡 |
| **Mixup/CutMix** | **未使用** | 因为元数据与图像一一对应，Mixup 会破坏对应关系 |
| **Test-time: Five-Crop + Multi-Scale** | — | 推理增强 |

### 3.4 训练流程关键步骤

1. **两阶段训练**: 先在 train set 上训练 → 再在 train+val set 上联合训练
   - `train+val`: 验证集包含未知类样本，让模型学习未知类的特征分布
   - 对未知类施加均匀分布约束（详见 2.5）

2. **属级辅助任务**: 同时训练 species (1604类) + genus (566类) 两个分类头
   - 但 genus loss 权重设为 0.0（代码中 `loss = loss + 0.0 * genus_loss`），说明只作为辅助监督信号，不直接参与梯度更新

3. **标签平滑**: 未使用 (`LABEL_SMOOTHING: 0.0`)——因为 Seesaw Loss 已有类间平衡

---

## 四、推理优化

### 4.1 推理流程

**推理命令** (`run_inference.sh`):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
    --nproc_per_node 4 --master_port 12346 main.py --eval \
    --cfg ./configs/MetaFG_meta_2_384.yaml \
    --dataset fungi_test \
    --resume output/MetaFG_meta_2/${EXP_TAG}/latest.pth \
    --batch-size 6 --tag ${EXP_TAG}_epoch64_test \
    --opts DATA.IMG_SIZE 384
```

推理时 batch size 从训练时的 18 降至 **6**（因为使用了 Five-Crop TTA，显存需求增加）。

### 4.2 Test-Time Augmentation (TTA)

**Five-Crop + 平均** (`main.py` validate 函数):
```python
# 输入: [N, 5, C, H, W]  (5 = 四个角 + 中心)
n, crop, c, h, w = images.shape
images = images.reshape(-1, c, h, w)
meta = meta.unsqueeze(1).expand(-1, crop, -1).reshape(n * crop, -1)

# 推理
output = model(images, meta)

# 5 个 crop 的结果取平均
output = output.reshape(n, crop, -1)
output = torch.mean(output, 1)
```

### 4.3 后处理管道 (`post_avg_entropy.py`)

完整推理后处理分两步：

**Step 1: 模型集成平均**
```python
# 1. 从多个 pkl 文件读取各模型输出
results = [pickle.load(open(p, "rb")) for p in result_paths]
# 2. 合并所有 rank 的输出
result = results[0]
for i in range(1, len(results)):
    result.update(results[i])
# 3. 多模型平均
for k, v in result_final.items():
    result_final[k] = v / len(dirs)   # 按模型数平均
```

**Step 2: 熵引导的 Open-set 决策**
```python
# 同一 observation 有多张图片时，先对各张图的 logits 求均值
scores = np.vstack(scores)
score = np.mean(scores, 0)

# 计算熵
probs = torch.softmax(torch.tensor(score), dim=-1)
entropy_val = entropy(probs)   # scipy.stats.entropy (natural log)

# 决策
if entropy_val < 4 or (entropy_val < 6 and max_idx in less_cls):
    pred = max_idx       # 已知类
else:
    pred = -1             # 未知类 (unknown)
```

**推理 batch size**: 6 (受 Five-Crop 影响)

### 4.4 附加后处理 (`post_avg.py` 中的技巧)

在 `post_avg.py` 中还有一个基于 **最大 logit 分数** 的方法，含额外技巧：
```python
thresh = 9.8
# 如果一个样本的 top-1 分数低但 top-2/top-3 是稀有类 → 提升稀有类
if max_idx_v2 in less_cls and max_idx not in less_cls and max_score_v2 > thresh - 0.7:
    max_score = thresh + 1      # 强制大于阈值
    max_idx = max_idx_v2
# 如果方差最大值的分数比均值最大值高 > 15 → 用方差最大值
if max_score <= thresh and len(scores) > 1 and max_score_v2 > 15:
    max_score = max_score_v2
    max_idx = max_idx_v2
```
这个技巧通过逐张图片的 logit 差异来纠正模型的低置信度预测。

### 4.5 多尺度推理（论文提到）

论文 3.1 节提到使用 "multi scale & ten crop" 进行 TTA，但代码中主要使用 Five-Crop。

---

## 五、模型集成策略

### 5.1 集成方案

| 组件 | 大小 | 配置 |
|------|------|------|
| MetaFormer-0 (150MB) | `MetaFG_meta_0_384` | bs=36, epochs=80 |
| MetaFormer-2 (393MB) | `MetaFG_meta_2_384` | bs=18, epochs=64, accu=4 |
| **Ensemble (543MB)** | 以上两者平均 | < 1GB 竞赛限制 |

### 5.2 集成效果 (Table 8)

| Model | F1 ↑ | Track1 ↓ | Track2 ↓ | Track3 ↓ | Track4 ↓ |
|-------|------|----------|----------|----------|----------|
| MetaFormer-0 | 58.11 | 0.2069 | 0.2067 | 0.4136 | 1.3936 |
| MetaFormer-2 | 57.63 | 0.2123 | 0.1943 | 0.4066 | 1.4984 |
| **Ensemble** | **58.95** | **0.2072** | **0.1742** | **0.3814** | **1.4762** |

注意: 集成并不一定在所有 metric 上都优于单模型——Track4 上 MetaFormer-0 反而更好。这说明不同模型在不同子任务上有互补优势。

### 5.3 InternImage 对比 (Table 4)

InternImage-L（纯视觉模型，无元数据）F1=54.32%，而 MetaFormer-2 达 55.94%（都不用 val set 训练），验证了**元数据信息对细粒度识别的重要性**。

---

## 六、对 image-classification 项目的借鉴

以下按**可落地难度**从低到高排列：

### ⭐ Tier 1 — 直接可用 (低成本、高收益)

#### 6.1 熵引导的置信度校准 / 拒识机制

**核心价值**: 用预测熵作为"模型是否确定"的指标，建立拒识 (rejection) 机制。

**可应用于**:
- 推理时对低置信度/高熵样本做标记或人工复核
- 作为置信度分数的替代/补充指标
- 构建 Open-set 或 novelty detection 模块

**实现参考**:
```python
import torch
import numpy as np
from scipy.stats import entropy

def entropy_confidence(logits, threshold=4.0):
    """
    用熵来判断预测是否可信
    Args:
        logits: [N, C] 或 [C]
        threshold: 熵阈值，> threshold → 不可信
    Returns:
        is_confident: bool 或 [N] bool
        entropy_val: 熵值
    """
    probs = torch.softmax(logits, dim=-1)
    if probs.dim() == 1:
        ent_val = entropy(probs.cpu().numpy())
    else:
        ent_val = entropy(probs.cpu().numpy(), axis=1)
    return ent_val < threshold, ent_val
```

**超参数**: 阈值 τ 需根据具体数据集标定（可从验证集的熵分布图中选取分界点）。

#### 6.2 Seesaw Loss 处理长尾分布

当前项目如果存在类别不均衡问题，可考虑使用 Seesaw Loss。

```python
from models.custom_loss import SeesawLoss

criterion = SeesawLoss(
    num_classes=NUM_CLASSES,  # 改为实际类别数
    p=0.8,                     # mitigation factor
    q=2.0,                     # compensation factor
    eps=1e-2
).cuda()
```

**优点**: 不需要修改数据采样策略，直接在 loss 层面处理不均衡。

**代码文件**: `code/models/custom_loss.py` (MIT License, 可复用)

### ⭐ Tier 2 — 中等投入

#### 6.3 元数据/多模态信息融合

如果数据集中有附加信息（地理位置、时间、属性标签等），可以：

1. 对非视觉信息做 embedding（类似论文的 sin/cos 周期编码 + one-hot）
2. 在模型中间层（如 Transformer encoder）以额外 token 形式注入
3. 使用相对注意力机制融合

**关键设计**:
- 元数据 token 与视觉 token 一起进入 self-attention
- 训练时随机 dropout 元数据（论文用线性衰减 mask_prob），增强图像-only 的鲁棒性
- 模型支持 inference 时不提供元数据（代码中 `mask_meta=True` 测试了此场景）

#### 6.4 Five-Crop TTA + 模型平均

```python
# 推理时 Five-Crop + 平均
crops = five_crop(image)  # [5, C, H, W]
outputs = model(crops)
output = torch.mean(outputs, dim=0)  # 平均
```

**注意**:
- Batch size 需要相应缩小（训练 bs=18 → 推理 bs=6）
- 可与多尺度 TTA 结合使用

### ⭐ Tier 3 — 深入借鉴

#### 6.5 多模型集成框架

参考论文的集成管道：
1. 每个模型独立训练和推理 → 保存为 `.pkl` (logits)
2. 后处理脚本读取所有 pkl，对 logits 平均 → 再做 open-set 决策

**代码架构**: `post_avg_entropy.py` 的目录结构设计清晰——每个模型一个子目录，每个 rank 一个 pkl 文件。

#### 6.6 辅助分类头 (Auxiliary Head)

论文使用 genus (属级) 分类头作为辅助监督：
```python
self.head = nn.Linear(attn_embed_dims[-1], num_classes)   # 物种级
self.genus_head = nn.Linear(attn_embed_dims[-1], 566)      # 属级
```
可用于层级分类场景（如：科→属→种）。

#### 6.7 长尾类特殊处理

`less_cls` 列表体现了"对长尾类放宽决策阈值"的策略：
- 稀有类的预测不确定性天然更高
- 对它们使用更宽松的阈值（熵 < 6 而非 < 4）
- 这个思路可以泛化到任何需要处理长尾分布的场景

---

## 七、关键文件索引

| 文件 | 用途 |
|------|------|
| `code/models/MetaFG_meta.py` | **核心模型**: 含元数据融合的 MetaFG-Meta |
| `code/models/custom_loss.py` | **Seesaw Loss** + 毒蘑菇损失 + 未知类均匀约束 |
| `code/post_avg_entropy.py` | **熵引导后处理**: 集成 + 熵阈值决策 |
| `code/post_avg.py` | **MLS 后处理**: 基于最大 logit + 方差纠正 |
| `code/main.py` | 训练/推理主流程 |
| `code/config.py` | 配置系统 (基于 yacs) |
| `code/lr_scheduler.py` | Cosine/Linear/Step 学习率调度器 |
| `code/optimizer.py` | AdamW 优化器 (含 weight decay 分离) |
| `code/utils.py` | 加载/保存 checkpoint、预训练权重处理 |
| `code/logger.py` | 日志系统 |
| `code/configs/MetaFG_meta_2_384.yaml` | MetaFormer-2 配置 |
| `code/configs/MetaFG_meta_0_384.yaml` | MetaFormer-0 配置 |
| `code/run_train.sh` | 训练启动脚本 |
| `code/run_inference.sh` | 推理启动脚本 |
| `code/requirements.txt` | Python 依赖 |

---

## 八、总结：核心要点提炼

1. **熵 > Softmax/LMS 用于 Open-set 识别**：熵的分布边界比 MSP/MLS 更清晰，是更鲁棒的未知样本检测指标（F1 +2.86%）

2. **验证集 + 均匀约束是前提**：必须在包含未知类的数据上训练，并对未知类施加均匀分布约束，熵方法才能生效

3. **元数据价值显著**：纯视觉 InternImage vs 视觉+元数据 MetaFormer，差距 ~3.8% F1

4. **Seesaw Loss 零成本换收益**：不改数据采样、不改模型结构，只换 loss 函数即可缓解长尾问题

5. **毒蘑菇损失 = 针对性辅助监督**：在特定类别上施加额外分类损失，可显著降低该类别的误判风险（Track2 -28%）

6. **小模型集成 > 单大模型**：150MB + 393MB = 543MB 的集成效果优于单一大模型，且满足部署约束

---

> **最后更新**: 2026-05-10  
> **分析者**: 🦎 壁虎  
> **基于**: FungiCLEF 2023 第1名方案, Ren et al., CLEF 2023
