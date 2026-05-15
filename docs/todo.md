# 待尝试识别率技巧

### 损失函数
* Seesaw Loss or (ArcFace Loss+DRS (Deferred Rebalancing via Resampling))

### 数据增强
Vertical Flip 小概率(0.1)尝试
mixp up alpha 设置为 2
RandomResizedCrop scale = `(0.5, 1.3)`

### 训练策略
* 梯度剪裁
* 梯度检查点

### TTA
FLIP TTA

### 模型
* EVA02


* 关闭 ArcFace 看看结果
*  梯度检查点

