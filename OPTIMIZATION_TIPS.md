# 训练识别率提升方法

## 基本信息
支持 4.4 万种国内动植物分类的图像分类模型，基于 swin v2 base(imageNet 22k 上训练，1k 上微调后的预训练模型)，在 4090 单卡上进行训练，总共训练约 800 万张图片。

## 训练流程分析
### 输入
两阶段训练：前 70% epoch 使用 448px, 后 30% epoch 使用 512px(配置文件示例：前 70% epoch 使用 `config/tiny/swinv2_tiny.json`，后 30% epoch 使用 `config/tiny/swinv2_tiny512.json`)

### 数据增强 (`data_module.py`)
- `RandomResizedCrop(512)` + `TrivialAugmentWide()` + `RandomErasing()`
- `CutMix` + `MixUp`（`model.py` 中实现），前70% epoch 80% 概率触发， 最后 30% epoch 20%概率触发
- cutmix/mixup 每次只能二选一

#### 可优化方向
1. **添加 `RandomHorizontalFlip`**：`TrivialAugmentWide` 包含 14 种增强操作（Rotate, Shear, Translate, AutoContrast, Equalize, Invert, Posterize, Contrast, Brightness, Sharpness, Color），但**不包含水平翻转**，需单独添加：
   ```python
   v2.RandomHorizontalFlip(p=0.5)  # 加在 TrivialAugmentWide 之后
   ```

2. **`RandomResizedCrop` 的 `scale` 参数**：当前默认 `scale=(0.08, 1.0)` 最小裁剪到原图 8%，对细粒度分类偏小。建议 `scale=(0.3, 1.0)`，保留更多原始分辨率的局部细节（羽毛纹理、翅脉、花瓣边缘等判别特征）。

4. Mixup 的 alpha 参数调整：
Mixup：alpha=0.5。过大（如1.0）会产生过多不真实幻象，4万类中更容易混淆判别特征。

CutMix：alpha=1.0（默认），但可以对长尾类别降低混合比例，或使用 TokenCutMix / SnapMix（保留局部语义块）更贴合细粒度场景。   

### Backbone 模型选择
- 当前使用 `swinv2_base_window12to24_192to384.ms_in22k_ft_in1k`，约 88M 参数，预训练权重来自 ImageNet-22k 微调至 ImageNet-1k
- 尝试过其它模型但效果不佳，同时因为算力限制不再考虑参数量更大的模型。

### 优化器 (`model.py`)
- AdamW
- Warmup 3 epochs（线性增加） + CosineAnnealing

## 已尝试但无效的方法
1. 加入标签平滑（Label Smoothing）: 实测识别率没有上升反而略微下降。可能原因分析：CutMix/MixUp 已经在创建软标签（混合样本的标签本身就是 soft target）。标签平滑 + CutMix/MixUp 叠加可能导致过度正则化，模型学习信号被削弱。
2. 用 `RandAugment` / `AutoAugment` 替换 `TrivialAugmentWide`，基本沒有提升；

## 决定不再尝试的方法
1. 不会更大模型，因为算力达到上限了。我只使用一张 4090 训练 4 万种动植物分类模型。
2. 知识蒸馏。同理也是因为算力限制。
3. Test-Time Augmentation (TTA)。因为实际使用场景时，没有足够的资源进行 TTA

## 待尝试
* WeightedRandomSampler 解决类别不平衡问题

