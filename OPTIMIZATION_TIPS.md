# 训练优化建议

## 训练流程分析

### 1. 数据增强 (`data_module.py`)
- **Train**: `RandomResizedCrop(512)` + `TrivialAugmentWide()` + `RandomErasing()`
- **Val/Test**: `Resize(614)` + `CenterCrop(512)` + 标准化
- **问题**: `RandomErasing` 效果和不用完全一样（代码注释已确认），说明 augment strength 不足或参数不对

### 2. 混合增强 (`model.py:36-40`)
- `CutMix` + `MixUp`，概率 80%（前70% epoch）/ 20%（后30%）
- 每次只选一个，存在改进空间

### 3. 优化器 (`model.py:97`)
- `AdamW(weight_decay=2e-5)`，固定学习率 1e-4
- Warmup 3 epochs + CosineAnnealing

### 4. 模型选择
- 当前使用 `swinv2_base_window12to24_192to384.ms_in22k_ft_in1k`
- 预训练权重来自 ImageNet-22k 微调至 ImageNet-1k

---

## 可提升识别率的方向

| 方向 | 具体做法 | 预期收益 |
|------|---------|---------|
| **数据增强** | 用 `RandAugment` / `AutoAugment` 替换 `TrivialAugmentWide`；调整 `RandomErasing` 参数（ratio, probability） | +0.5~2% |
| **CutMix+MixUp 叠加** | 同时应用 CutMix 和 MixUp，而非二选一 | +0.3~1% |
| **标签平滑** | `CrossEntropyLoss(label_smoothing=0.1)` | +0.2~0.5% |
| **类别不平衡** | 当前 `class_counts` 未传入分类器，可用 `WeightedRandomSampler` 或 `loss_weight` | +0.5~2% |
| **更大模型** | 尝试 `swinv2_large_window12to24_192to384` 或 `convnextv2-base-22k-384`（更大感受野） | +1~3% |
| **更高分辨率** | 512→640（更大的 crop size 保留更多细粒度特征） | +0.5~1% |
| **SWA / 指数移动平均** | 添加 `StochasticWeightAveraging` 或 EMA | +0.3~0.8% |
| **知识蒸馏** | 用大模型作为 teacher，小模型为 student | +0.5~1% |
| **优化器调整** | 尝试 `Lion` 优化器（比 AdamW 更省显存）；或使用更大的 weight_decay (0.05) | +0.2~0.5% |
| **学习率调度** | 余弦退火 + 线性 warmup 已有的情况下，可尝试 `cosine Annealing with restarts` 或 `OneCycleLR` | 轻微 |

---

## 最值得尝试的 Top3

1. **加入标签平滑** — 一行代码改动，风险极低
2. **调整 `RandomErasing` 参数或替换增强策略** — 当前该增强形同虚设
3. **处理类别不平衡** — 如果数据存在长尾分布，用 `WeightedRandomSampler` 或 loss weighting 可显著提升少数类的识别率
