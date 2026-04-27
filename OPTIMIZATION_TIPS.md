# 训练优化建议

## 训练流程分析

### 1. 数据增强 (`data_module.py`)
- **Train**: `RandomResizedCrop(512)` + `TrivialAugmentWide()` + `RandomErasing()`
- **Val/Test**: `Resize(614)` + `CenterCrop(512)` + 标准化

### 2. 混合增强 (`model.py:36-40`)
- `CutMix` + `MixUp`，概率 80%（前70% epoch）/ 20%（后30% epoch）
- 每次只选一个

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
| **类别不平衡** | 当前 `class_counts` 未传入分类器，可用 `WeightedRandomSampler` 或 `loss_weight` | +0.5~2% |
| **SWA / 指数移动平均** | 添加 `StochasticWeightAveraging` 或 EMA | +0.3~0.8% |


3. **处理类别不平衡** — 如果数据存在长尾分布，用 `WeightedRandomSampler` 或 loss weighting 可显著提升少数类的识别率


## 已尝试但无效的方法
1. 加入标签平滑: 实测识别率没有上升反而略微下降。可能原因分析：CutMix/MixUp 已经在创建软标签（混合样本的标签本身就是 soft target）。标签平滑 + CutMix/MixUp 叠加可能导致过度正则化，模型学习信号被削弱。
2. 用 `RandAugment` / `AutoAugment` 替换 `TrivialAugmentWide`，基本沒有提升；

## 决定不会尝试的方法
1. 不会更大模型或者更大图片分辨率，因为算力达到上限了。我只使用一张 4090 训练 4 万种动植物分类模型。
2. 知识蒸馏。同理也是因为算力限制。

