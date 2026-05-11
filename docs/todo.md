# 待尝试识别率技巧

### 损失函数
* Seesaw Loss or (ArcFace Loss+重采样)

### 数据增强
Vertical Flip 小概率(0.1)尝试
mixp up alpha 设置为 2

### 训练策略
* 梯度剪裁
* 梯度检查点

### TTA
FLIP TTA

### 模型
* EVA02


1. 关闭 ArcFace 看看结果
2. arcface m 目前是 0.2~0.4， 设置为 0.1~0.3 再看看结果
