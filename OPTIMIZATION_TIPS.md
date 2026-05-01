# 训练识别率提升方法

## 基本信息
支持 4.4 万种国内动植物分类的图像分类模型（大规模细粒度分类），基于 swin v2 base(imageNet 22k 上训练，1k 上微调后的预训练模型)，在 4090 单卡上进行训练，总共训练约 800 万张图片。

## 训练流程分析
### 输入
两阶段训练：前 70% epoch 使用 448px, 后 30% epoch 使用 512px(配置文件示例：前 70% epoch 使用 `config/tiny/swinv2_tiny.json`，后 30% epoch 使用 `config/tiny/swinv2_tiny512.json`)

### 数据增强

**样本级增强** (`data_module.py`，按顺序应用)：
* `RandomResizedCrop(input_size, scale=(0.3, 1.0))` — 随机裁剪缩放，scale 下界 0.3 提供较强尺度多样性
* `TrivialAugmentWide()` — 无参数自动增强，每张图片随机选一种操作+强度
* `RandomHorizontalFlip(p=0.5)` — 50% 水平翻转
* `ToDtype(float32, scale=True)` — 归一化到 [0, 1]
*  `Normalize(ImageNet mean/std)` — ImageNet 标准归一化
* `RandomErasing(p=0.25, scale=(0.02, 0.2))` — 25% 概率随机擦除，模拟遮挡

**批次级增强** (`model.py` → `training_step`)：
- `CutMix` 或 `MixUp(alpha=0.2)`，每次随机二选一（`RandomChoice`）
- 前 70% epoch：**80%** 概率触发（强力正则化）
- 后 30% epoch：**20%** 概率触发（降低干扰，专注真实分布）

**验证/测试增强**：
- `Resize(input_size × 1.2)` → `CenterCrop(input_size)` → `Normalize`
- 不应用任何随机增强，保证评估可复现 

### Backbone 模型选择
- 当前使用 `swinv2_base_window12to24_192to384.ms_in22k_ft_in1k`，约 88M 参数，预训练权重来自 ImageNet-22k 微调至 ImageNet-1k
- 尝试过其它模型但效果不佳，同时因为算力限制不再考虑参数量更大的模型。

### 优化器 (`model.py`)
- AdamW
- Warmup 3 epochs（线性增加） + CosineAnnealing

## 已尝试但无效的方法
1. 加入标签平滑（Label Smoothing）: 实测识别率没有上升反而略微下降。可能原因分析：CutMix/MixUp 已经在创建软标签（混合样本的标签本身就是 soft target）。标签平滑 + CutMix/MixUp 叠加可能导致过度正则化，模型学习信号被削弱。
2. 用 `RandAugment` / `AutoAugment` 替换 `TrivialAugmentWide`，基本沒有提升；
3. StochasticWeightAveraging, 使用之后识别率反而有所下降，原因不知。

## 决定不再尝试的方法
1. 不会更大模型，因为算力达到上限了。我只使用一张 4090 训练 4 万种动植物分类模型。
2. 知识蒸馏。同理也是因为算力限制。
3. Test-Time Augmentation (TTA)。因为实际使用场景时，没有足够的资源进行 TTA

## 待尝试的数据增强方案
* 降低 cutmix 和 mixup 触发概率
* RandomResizedCrop scale 的下限值调高、
* 随机 RandomErasing 的概率调低


