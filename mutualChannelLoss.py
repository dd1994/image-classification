import torch.nn as nn
import torch

class MutualChannelLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(MutualChannelLoss, self).__init__()
        self.alpha = alpha
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels, features):
        """
        logits: 模型输出
        labels: 真实标签
        features: 中间特征图 (来自 EfficientNet 的特征)
        """
        # 判别性损失（交叉熵）
        cls_loss = self.cross_entropy_loss(logits, labels)

        # 多样性损失（计算每个通道间的差异）
        b, c, h, w = features.shape  # b = batch size, c = channels, h/w = feature map size
        features = features.view(b, c, -1)  # 变形为 (batch_size, channels, h*w)
        
        # 通道间特征的协方差矩阵
        channel_covariance = torch.bmm(features, features.transpose(1, 2))  # 计算通道间的协方差
        channel_variation = channel_covariance.mean()  # 取均值作为多样性损失

        # 最终的互通道损失
        total_loss = self.alpha * cls_loss + (1 - self.alpha) * channel_variation
        return total_loss