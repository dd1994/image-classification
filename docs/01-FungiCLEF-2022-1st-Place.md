# 01 - FungiCLEF 2022 1st Place Solution 详细分析

> **论文**: _1st Place Solution for FungiCLEF 2022 Competition: Fine-grained Open-set Fungi Recognition_  
> **作者**: Zihua Xiong, Yumeng Ruan, Yifei Hu, Yue Zhang, Yuke Zhu, Sheng Guo, Bing Han (MYbank, Ant Group)  
> **代码**: https://github.com/guoshengcv/fgvc9_fungiclef  
> **基础框架**: https://github.com/dqshuai/MetaFormer (MetaFormer)

---

## 📋 论文概述

### 背景与任务

FungiCLEF 2022 是 CLEF 2022 会议与 CVPR 2022 FGVC9 Workshop 联合举办的竞赛。任务为**细粒度、开放集真菌识别**：

- 给定真菌图像及元信息（栖息地、基质、时间、经纬度等），识别其所属物种
- 测试集包含未见过的开放集类别（open-set），需将其标记为未知类
- 本质上是 **fine-grained + long-tailed + open-set** 三合一挑战

### 数据集规模

| 项目 | 数值 |
|------|------|
| 训练集图像数 | 295,938 |
| 已知物种数（closed-set） | 1,604 |
| 测试集观测数 | 59,420 |
| 测试集图像数 | 118,676 |
| 测试集物种数 | 3,134（含1,604已知 + 开放集物种） |
| 类别分布 | **长尾分布**（long-tailed） |

### 使用的 Backbone 模型及变体

#### 1. MetaFormer（主模型）

MetaFormer 是一种 **CNN + Vision Transformer 混合架构**，同时支持元信息融合：

| 变体 | conv_embed_dims | attn_embed_dims | conv_depths | attn_depths | num_heads |
|------|-----------------|-----------------|-------------|-------------|-----------|
| MetaFG-0 | [64, 96, 192] | [384, 768] | [2, 2, 3] | [5, 2] | 8 |
| MetaFG-1 | [64, 96, 192] | [384, 768] | [2, 2, 6] | [14, 2] | 8 |
| MetaFG-2 | [128, 128, 256] | [512, 1024] | [2, 2, 6] | [14, 2] | 8 |

- **MetaFG_meta_*** 变体支持元信息（meta-information）注入
- 元信息通过 additional tokens 在 Transformer 层注入

**代码位置**:
- 基础模型: `models/MetaFG.py` (第 69-160 行)
- 元信息模型: `models/MetaFG_meta.py` (第 108-308 行)
- 模型注册与构建: `models/build.py` (第 1-55 行)

#### 2. ConvNeXt（辅助模型）

纯卷积架构，用于增加模型集成多样性：

| 变体 | depths | dims |
|------|--------|------|
| convnext_tiny | [3, 3, 9, 3] | [96, 192, 384, 768] |
| convnext_base | [3, 3, 27, 3] | [128, 256, 512, 1024] |
| convnext_large | [3, 3, 27, 3] | [192, 384, 768, 1536] |

**代码位置**: `models/convnext.py` (第 140-180 行)

#### 3. BEiT-ViT（实验性）

配置文件中有 beit_vit 的配置，但论文中未作为主要结果报告。

### 最终成绩

| 指标 | Public Leaderboard | Private Leaderboard |
|------|--------------------|---------------------|
| Mean F1 Score | **83.78%** | **80.43%** (🥇 1st Place) |
| 第二名差距 | 83.26 → 83.78 (+0.52) | 79.38 → 80.43 (+1.05) |

---

## 🔬 识别率提升技巧（按重要性排列）

### 技巧 1: MetaFormer 架构 + 元信息融合 ⭐⭐⭐⭐⭐

**原理**:
MetaFormer 是 Conv + MHSA 混合架构，前两个 stage 用 MBConv（Inverted Residual），后两个 stage 用 MHSA（Multi-Head Self-Attention）。通过 extra_tokens 机制将元信息作为额外 token 注入 Transformer 层。
每种元信息先经过独立的 FC 投影网络（`nn.Linear → ReLU → LayerNorm → ResNormLayer`），再作为 extra_token 附加到 cls_token 之后。

**论文中的超参数**:
- `EXTRA_TOKEN_NUM = 5`（1 个 cls_token + 4 个 meta_token）
- `META_DIMS = [4, 34, 32, 31]`，分别对应：
  - `[4]` = 时间编码（sin/cos of month, sin/cos of day）
  - `[34]` = countryCode one-hot（34个国家）
  - `[32]` = Substrate one-hot（32种基质类型）
  - `[31]` = Habitat one-hot（31种栖息地类型）
- 图像尺寸: `384×384`
- `MLP_RATIO = 4.0`

**代码实现要点**:
- 元信息编码: `data/dataset_fg.py` 第 320-342 行（`encode_temporal_info` + one-hot 编码）
- 元信息投影头: `models/MetaFG_meta.py` 第 138-151 行（`meta_head_1`, `meta_head_2`）
- 元信息注入: `models/MetaFG_meta.py` 第 240-254 行（`forward_features` 方法中拼接到 extra_tokens）
- **训练时 mask 策略**: `models/MetaFG_meta.py` 第 260-268 行 — 训练时按概率 mask 掉元信息，mask_prob 随 epoch 线性衰减：
  ```python
  # mask_type='linear' 时:
  cur_mask_prob = self.mask_prob - self.cur_epoch/self.total_epoch
  ```
  默认 `mask_prob = 1.0`（即初始完全随机 mask，随训练逐渐减少）

---

### 技巧 2: Seesaw Loss（长尾分类损失函数）⭐⭐⭐⭐⭐

**原理**:
Seesaw Loss 动态平衡头部类和尾部类的训练。它改造 Cross Entropy 的 softmax 分母：

\[
L_{seesaw}(z) = -\sum y_i \log(\hat{p}_i), \quad \hat{p}_i = \frac{e^{z_i}}{\sum_{j \neq i} S_{ij} e^{z_j} + e^{z_i}}
\]

其中 \(S_{ij}\) 是动态因子，包含两部分：
1. **Mitigation Factor** (p=0.8): 根据类别累计样本数比例动态调整——对样本数少的尾部类，降低来自头部类的负梯度压制
2. **Compensation Factor** (q=2.0): 根据当前模型对各类别的预测分数动态补偿——模型对某类过度自信时降低其负梯度

**论文中的超参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| p | 0.8 | mitigation factor 指数 |
| q | 2.0 | compensation factor 指数 |
| num_classes | 1606 | Fungi 竞赛类别数 |
| eps | 0.01 (1e-2) | 除法平滑除数 |

**代码实现要点**:
- 文件: `models/custom_loss.py` 第 156-248 行（`SeesawLoss` 类）
- 核心函数: `seesaw_ce_loss()` 第 53-127 行
- 在 `main.py` 第 123-125 行使用：
  ```python
  criterion = SeesawLoss().cuda()  # p=0.8, q=2.0, num_classes=1606
  ```
- **重要**: 论文实验显示 Seesaw Loss + Label Smoothing CE 混用优于单独使用
- Seesaw Loss **不能与 mixup/cutmix 混用**（论文 Table 2 显示加 mixup 反而降低性能）

**Ablation 效果**（MetaFormer-0, 32 epochs, train+val）:

| Loss | Public F1 | Private F1 |
|------|-----------|------------|
| Soft Target CE + Mixup | 71.49% | 67.60% |
| Label Smoothing CE | 79.45% | 75.67% |
| **Seesaw Loss** | **79.79%** | **76.15%** |

---

### 技巧 3: Model Ensemble（模型集成）⭐⭐⭐⭐

**原理**:
训练多个不同配置（不同 backbone、不同 pretrain 数据集、不同尺寸）的模型，推理时将它们的 logit 输出做**算术平均**作为最终 logit。多样性来源于：
1. 不同 backbone 架构（MetaFormer-0/1/2 + ConvNeXt-tiny/base/large）
2. 不同预训练数据集（herbarium / imagenet22k / inaturalist21）
3. 不同训练设定（ArcFace vs 非 ArcFace, 不同 accumulate steps）
4. 是否加入 pseudo label 训练

**论文中的超参数**:
- Ensemble 方式: **logit averaging**（简单平均，非加权）
- 最终方案包含的模型: 多个 MetaFormer + ConvNeXt 模型的总集成

**代码实现要点**:
- 文件: `post_avg.py` 第 1-32 行（平均集成逻辑）
- 流程：各模型推理结果保存为 `result{rank}.pkl` → `post_avg.py` 读取并按 observation_id 合并求平均

**有效性证明**（Table 11）:

| Ensemble 模型 | Public F1 | Private F1 |
|---------------|-----------|------------|
| Ensemble + center crop TTA | 83.20% | 79.51% |
| Ensemble + multi scale & ten crops TTA | 83.26% | 79.38% |

---

### 技巧 4: Post-Process（后处理优化 Open-set 识别）⭐⭐⭐⭐

**原理**:
针对开放集识别中的两个核心问题设计后处理：

1. **开放集 vs 已知类区分**: 开放集样本的预测置信度（最大 logit）通常较低。通过绘制 val/test 的 logit 频率分布确定阈值（约 5.0），但实际上采用自适应阈值 `high_t = 9.8`。

2. **尾部类误分类为开放集**: 尾部类（训练样本很少的类别）容易因低置信度被误判为开放集。解决方案：检查 top-3 预测中是否有"hard tail categories"，若有且超过 low threshold，则纠正。

3. **显微镜图像干扰**: 测试样本中可能同时包含自然光和显微镜图像，直接平均会导致低置信度。若平均 logit < high_t 但存在单张图像 max_logit > 15，则采纳该单张图像的预测。

**论文中的超参数**:

| 参数 | 值 | 含义 |
|------|-----|------|
| high_t | 9.8 | 高置信度阈值（高于此直接采纳 top-1） |
| low_t | high_t - 0.7 = 9.1 | 低置信度阈值（尾部类抢救阈值） |
| max_score_threshold | 15 | 单张图像最大 logit 阈值（显微镜处理） |
| topk_score_diff | 1.2 | Top-1 与 Top-2/3 的分数差距阈值 |
| open-set k 初始 | ~1000 | 初步估计的开放集样本数 |
| open-set k 最终 | ~1500 | 最终调整的开放集样本数 |

**代码实现要点**:
- 文件: `post_avg.py` 第 38-93 行
- 核心逻辑: 对每个 observation，先平均所有图片 logit → 取 top-3 → 应用硬尾类抢救 → 显微镜处理 → 开放集判定
- Hard tail categories 列表: `post_avg.py` 第 39 行（37 个尾部类别 ID）

**Ablation 效果**（Table 12）:

| 后处理版本 | Public F1 | Private F1 | 说明 |
|------------|-----------|------------|------|
| v1 (初始) | 83.26% | 79.38% | ~1000 open-set |
| v2 (+proper threshold) | 83.50% | 79.60% | ~1500 open-set |
| v3 (+pseudo label models) | 83.65% | 79.79% | ~1500 open-set |
| v4 (+tail category post) | **83.78%** | **80.43%** | ~1500 open-set |

---

### 技巧 5: Pseudo Labeling（伪标签）⭐⭐⭐

**原理**:
利用当前最佳 ensemble 模型对测试集样本做预测，选取 top ~50% 高置信度样本，将其预测作为伪标签加入训练集。

**论文中的超参数**:
- 选取比例: **top ~50%**（按置信度排序）
- 额外训练 epochs: 80（对于 MetaFormer-2）或 64（baseline）
- Pseudo label models 参与最终 ensemble

**代码实现要点**:
- 伪标签数据加载: `data/dataset_fg.py` 第 350-368 行（`find_images_and_targets_fungi` 函数末尾读取 `pesudo.csv`）
- 在训练函数中与原始数据合并训练

**效果**（Table 9）:
| Backbone | Pseudo Label | Public F1 | Private F1 |
|----------|-------------|-----------|------------|
| ConvNeXt-large | ❌ | 79.15% | 75.59% |
| ConvNeXt-large | ✅ | 80.65% | 76.61% |
| MetaFormer-2 | ❌ | 82.04% | 77.92% |
| MetaFormer-2 | ✅ | 82.45% | 77.93% |

---

### 技巧 6: Test Time Augmentation (TTA) ⭐⭐⭐

**原理**:
推理时对每张图像多次裁切并取平均，提升预测稳定性和精度。

**论文中对比的 TTA 策略**:
- Center crop
- Five crop
- Multi scale & ten crop（多尺度 + 十裁切）

**Multi Scale & Ten Crop 实现**:
- 三个尺度: 1.1×, 1.143×, 1.2× 原始尺寸
- 每个尺度做 TenCrop
- 最终 30 个 crop 的预测取平均

**最终选择**: Multi scale & ten crop（基于 public set 性能选择）

**代码实现要点**:
- 文件: `data/build.py` 第 131-139 行（`MultiCrop` 类）
- 验证阶段 five crop 处理: `main.py` 第 204-208 行
  ```python
  n, crop, c, h, w = images.shape
  images = images.reshape(-1, c, h, w)
  # ...
  output = output.reshape(n, 10, -1)
  output = torch.mean(output, 1)  # 平均 10 个 crop
  ```

**TTA 效果**（Table 10）:

| TTA 方式 | 单模型 Public F1 | 单模型 Private F1 |
|----------|-----------------|-------------------|
| Center Crop | 81.66% | 77.94% |
| Five Crop | **81.76%** | **78.25%** |
| Center Crop | 80.46% | 77.02% |
| Multi Scale & Ten Crop | 80.20% | 77.31% |

---

### 技巧 7: 大规模 Batch Size + 梯度累积 ⭐⭐

**原理**:
增大有效 batch size 可稳定训练、提升性能。受限于 GPU 显存，通过梯度累积模拟大批量训练。

**论文中的超参数**:

| Backbone | Per-GPU Batch | GPUs | Accum Steps | Effective Batch | Public F1 | Private F1 |
|----------|---------------|------|-------------|-----------------|-----------|------------|
| MetaFormer-0 | 64 | 4 | 3 | 768 | 79.79% | 76.15% |
| MetaFormer-0 | 64 | 4 | 6 | 1536 | 80.22% | 76.90% |
| MetaFormer-1 | 32 | 4 | 3 | 384 | 81.67% | 77.62% |
| MetaFormer-1 | 32 | 4 | 6 | 768 | 81.66% | **77.94%** |

**代码实现要点**:
- 梯度累积: `main.py` 第 174-178 行（仅在 accumulation_steps 整数倍时执行 `optimizer.step()`）
- LR 线性缩放: `main.py` 第 304-310 行：
  ```python
  linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
  linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS  # 梯度累积也缩放LR
  ```

---

### 技巧 8: 预训练数据集选择 ⭐⭐

**原理**:
FGVC 任务中，与目标域相关的预训练数据比通用 ImageNet 预训练效果好。

**论文中的超参数**:

| Pretrain 数据集 | Backbone | Public F1 | Private F1 |
|-----------------|----------|-----------|------------|
| herbarium | MetaFormer-2 | 80.90% | 77.37% |
| imagenet22k | MetaFormer-2 | 81.47% | 77.86% |
| **inaturalist21** | MetaFormer-2 | **82.04%** | **77.92%** |

**策略**: 不只用最好的单模型，而是把不同预训练模型都拉入集成，进一步提升。

**代码实现要点**:
- 预训练权重加载: `utils.py` 第 40-68 行（`load_pretained` 函数）
- 自动丢弃 head 层权重: `config.MODEL.DORP_HEAD = True`（默认）
- 自动丢弃 meta 相关权重: `config.MODEL.DORP_META = True`（默认）
- 命令行指定: `--pretrain ./pretrained_model/metafg_0_inat21_384.pth`

---

### 技巧 9: 类别特征 Embedding 的 ArcFace Loss ⭐

**原理**:
在 classifier head 前加 L2 归一化，使特征分布在超球面上，增加类间距离、减小类内距离。实验中作为 Seesaw Loss 的替代/补充。

**论文中的超参数**:
- `s = 30.0`（scale factor）
- `m = 0.5`（margin）

**代码实现要点**:
- 文件: `main.py` 第 32-61 行（`ArcfaceLoss` 类）
- 配置开关: `config.TRAIN.USE_ARCFACE = True`（配置文件 `MetaFG_meta_0_384_arcface.yaml`）
- 模型中特征归一化: `models/MetaFG_meta.py` 第 275-277 行：
  ```python
  if self.use_arcface:
      x = F.linear(F.normalize(x), F.normalize(self.head.weight))
  ```

> **注意**: 论文最终方案中 ArcFace 作为模型多样性的一种尝试，参与 ensemble 但不作为核心 loss。

---

### 技巧 10: 训练 Epoch 数调优 ⭐

**原理**:
更长的训练不总带来收益。实验发现 32-64 epochs 是最优点。

**论文中的超参数**:

| Epochs | Backbone | Public F1 | Private F1 |
|--------|----------|-----------|------------|
| 32 | MetaFormer-0 | 79.79% | 76.15% |
| 64 | MetaFormer-0 | **80.46%** | **77.01%** |
| 100 | MetaFormer-0 | 80.18% | 76.77% |
| 32 | MetaFormer-2 | 81.18% | 77.56% |
| 48 | MetaFormer-2 | **82.04%** | **77.92%** |
| 64 | MetaFormer-2 | 80.45% | 77.63% |

---

## ⚙️ 训练策略

### 学习率调度策略

| 参数 | 值 | 说明 |
|------|-----|------|
| Scheduler 类型 | **Cosine LR** | CosineLRScheduler |
| Base LR | **5e-5** | 基础学习率 |
| Warmup LR | **5e-8** | 预热起始学习率 |
| Min LR | **5e-7** | 最小学习率（Cosine 结束值） |
| Warmup Epochs | **1** | 预热 epoch 数 |
| LR 缩放规则 | `base_lr × batch_size × world_size / 512 × accumulation_steps` | 随 batch size 线性缩放 |

**代码位置**: `lr_scheduler.py` 第 14-27 行

**学习率变化曲线**（每次 iteration 更新）:
```
Epoch 0: 5e-8 → 线性上升到 scaled_base_lr
Epoch 1-64: scaled_base_lr → Cosine 衰减到 scaled_min_lr
```

### Optimizer 配置

| 参数 | 值 |
|------|-----|
| Optimizer | **AdamW** |
| Weight Decay | **0.05** |
| Betas | **(0.9, 0.999)** |
| Epsilon | **1e-8** |
| Gradient Clipping | **5.0** |
| Mixed Precision | **AMP O1** (NVIDIA Apex) |

**代码位置**: `optimizer.py` 第 1-54 行

**特殊参数分组**:
- bias 和 LayerNorm 参数: `weight_decay = 0`
- 其他参数: `weight_decay = 0.05`

### 数据增强策略

| 增强方式 | 参数值 | 说明 |
|----------|--------|------|
| RandomResizedCrop | size=384 | 随机裁切缩放 |
| Color Jitter | **0.4** | 颜色抖动强度 |
| Auto Augment | **rand-m9-mstd0.5-inc1** | RandAugment 策略 |
| Random Erase | **prob=0.25**, mode='pixel', count=1 | 随机擦除 |
| Interpolation | **bicubic** | 图像插值方式 |
| Mixup | **alpha=0**（最终未使用） | Seesaw Loss 时禁用 |
| CutMix | **alpha=0**（最终未使用） | Seesaw Loss 时禁用 |
| Normalize | ImageNet 均值/标准差 | `(0.485,0.456,0.406)` / `(0.229,0.224,0.225)` |

**代码位置**: `data/build.py` 第 141-168 行（`build_transform`）

### 训练参数总览

| 参数 | MetaFormer-0 | MetaFormer-1 | MetaFormer-2 | ConvNeXt-large |
|------|-------------|-------------|-------------|----------------|
| Image Size | 384 | 384 | 384 | 384 |
| Epochs | 32-64 | 64 | 32-48 | 64-80 |
| Batch Size (per GPU) | 64 | 32 | 24 | 24/10 |
| GPUs | 4-8 | 4 | 4 | 8 |
| Accumulation Steps | 3-6 | 6 | 4-8 | 4-6 |
| Effective Batch | 768-1536 | 768 | 384-768 | 480-1152 |
| Loss | SeesawLoss | SeesawLoss | SeesawLoss | SeesawLoss |
| Label Smoothing | 0.0 | 0.0 | 0.0 | 0.0 |
| Drop Path Rate | 0.1 | 0.1 | 0.1 | 0.1 |
| AMP | O1 | O1 | O1 | O1 |
| Gradient Clip | 5.0 | 5.0 | 5.0 | 5.0 |

### 训练命令示例

**MetaFormer-0 with meta**:
```bash
python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 \
  main.py --cfg ./configs/MetaFG_meta_0_384.yaml \
  --batch-size 64 --lr 5e-5 --min-lr 5e-7 --warmup-lr 5e-8 \
  --epochs 64 --warmup-epochs 1 --dataset fungi \
  --pretrain ./pretrained_model/metafg_0_inat21_384.pth \
  --accumulation-steps 4 --num-workers 16 --opts DATA.IMG_SIZE 384
```

**ConvNeXt-large**:
```bash
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 \
  main.py --cfg ./configs/convnext_large.yaml \
  --batch-size 24 --lr 5e-5 --min-lr 5e-7 --warmup-lr 5e-8 \
  --epochs 80 --warmup-epochs 1 --dataset fungi \
  --pretrain ./pretrained_model/convnext_large_22k_1k_384.pth \
  --accumulation-steps 6 --num-workers 32 --opts DATA.IMG_SIZE 384
```

---

## 🔮 推理优化

### TTA 策略

**最终采用**: **Multi Scale & Ten Crop**

具体流程：
1. 对输入图像做 3 个尺度的 resize: 1.1×, 1.143×, 1.2× 原始尺寸
2. 每个尺度做 TenCrop（四角+中心 + 水平翻转）
3. 共 30 个 crop，分别前向推理
4. 30 个 logit 取平均作为该图片预测

**代码位置**: `data/build.py` 第 131-139 行（`MultiCrop` 类），`main.py` 第 204-208 行（five crop 平均逻辑）

### Model Ensemble（模型集成）

1. 所有模型按上述 TTA 推理
2. 结果保存为 `result{rank}.pkl`
3. `post_avg.py` 读取各模型结果，对同一 observation 的多模型 logit 做算术平均
4. 一个 observation 可能含多张图片（如 sequence 拍摄），先图片间平均，再模型间平均

**代码位置**: `post_avg.py` 第 1-34 行

### 后处理（Post-Average）

在所有 ensemble 完成后的最终阶段进行：

1. **开放集阈值判断**: `max_score ≤ 9.8` → 标记为 `class_id = -1`
2. **硬尾部类抢救**: 检查 top-3 预测是否属于 37 个 hard tail categories
3. **显微镜图像处理**: 若 mean_score ≤ 9.8 但 max_per_image_score > 15，采信高置信度单张图像
4. **最终输出**: `ObservationId, ClassId` 的 CSV 格式

超参数：
- `high_t = 9.8`, `low_t = 9.1 (= high_t - 0.7)`
- `max_score_threshold = 15`（单张图像）
- `topk_diff_threshold = 1.2`

---

## 🎯 对 image-classification 项目的借鉴

### ✅ 可以直接移植的技巧

| 技巧 | 移植难度 | 说明 |
|------|---------|------|
| **Seesaw Loss** | 🟢 低 | 独立 loss 模块，只需 `pip install mmcv` 或直接复制 `custom_loss.py`。对长尾分类任务有直接提升 |
| **Cosine LR Scheduler + Warmup** | 🟢 低 | 标准训练技巧，timm 库原生支持 |
| **Gradient Accumulation** | 🟢 低 | 标准的 PyTorch 训练模式 |
| **Model Ensemble (logit averaging)** | 🟢 低 | 简单的逻辑，独立后处理步骤 |
| **Multi Scale TTA** | 🟢 低 | 独立于训练的推理增强 |
| **AdamW + Weight Decay 分离** | 🟢 低 | bias/norm 层不收 weight decay 是标准做法 |
| **RandAugment** | 🟢 低 | timm 支持的自动增强策略 |
| **AMP 混合精度训练** | 🟢 低 | PyTorch 官方支持 |

### ⚠️ 需要修改才能使用的技巧

| 技巧 | 所需修改 |
|------|---------|
| **MetaFormer Backbone** | 需要完整移植 `models/MetaFG*.py`、`models/MBConv.py`、`models/MHSA.py` 等依赖。如果当前项目已有的 ViT/CNN backbone 表现足够，不必优先移植 |
| **元信息融合** | 需要修改数据加载 pipeline，编码元信息（时间、地理位置、类别属性等）并作为额外 token 注入模型。这需要数据集提供相应元数据 |
| **Post-Process for Open-set** | 需要根据具体数据集的 logit 分布调整阈值。阈值 `high_t=9.8` 和 hard_tail_categories 列表是 FungiCLEF 特定的 |
| **Pseudo Labeling** | 需要额外的推理 → 置信度过滤 → 合并训练流程 |

### ❌ 当前项目已实现的技巧

| 技巧 | 状态 |
|------|------|
| Cosine LR Scheduler | 待确认 |
| AdamW Optimizer | 待确认 |
| Mixed Precision (AMP) | 待确认 |
| RandomResizedCrop + HorizontalFlip | 基本增强，通常已实现 |
| Drop Path / Stochastic Depth | 待确认 |
| Label Smoothing | 待确认 |

### 📊 移植优先级建议

1. **最高优先级** → Seesaw Loss（对长尾分类提升明显，独立模块）
2. **高优先级** → Cosine LR warmup + Gradient Accumulation + AdamW weight decay 分离
3. **中优先级** → TTA (Multi Scale & Ten Crop) + Model Ensemble
4. **低优先级** → MetaFormer backbone 移植（工程量大，且需要预训练权重）
5. **视数据而定** → 元信息融合、Pseudo Labeling、Open-set Post-Process

---

## 📁 关键代码文件索引

| 文件 | 内容 |
|------|------|
| `main.py` | 训练主循环、验证、LR 缩放、ArcFace Loss |
| `config.py` | 配置系统（基于 yacs） |
| `optimizer.py` | AdamW optimizer + 参数分组 |
| `lr_scheduler.py` | Cosine/Linear/Step LR scheduler |
| `utils.py` | 预训练加载、checkpoint 管理、梯度计算 |
| `post_avg.py` | Ensemble 平均 + 后处理 |
| `models/MetaFG_meta.py` | MetaFG + 元信息融合模型 |
| `models/MetaFG.py` | MetaFG 基础模型（不含元信息） |
| `models/convnext.py` | ConvNeXt 模型 |
| `models/custom_loss.py` | Seesaw Loss 实现 |
| `models/build.py` | 模型构建工厂 |
| `data/dataset_fg.py` | Fungi 数据集加载、元信息编码 |
| `data/build.py` | DataLoader + 数据增强 + MultiCrop TTA |
| `configs/MetaFG_meta_0_384.yaml` | MetaFormer-0 带元信息的配置示例 |
| `configs/convnext_large.yaml` | ConvNeXt-large 配置示例 |
| `run.sh` | 训练/测试命令示例 |
