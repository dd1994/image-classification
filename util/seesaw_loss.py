import numpy as np
import torch
from torch import nn


class SeesawLossWithLogits(nn.Module):
    """
    This is unofficial implementation for Seesaw loss,
    which is proposed in the techinical report for LVIS workshop at ECCV 2020.
    For more detail, please refer https://arxiv.org/pdf/2008.10032.pdf.
    Args:
    class_counts: The list which has number of samples for each class.
                  Should have same length as num_classes.
    p: Scale parameter which adjust the strength of panishment.
       Set to 0.8 as a default by following the original paper.
    """

    def __init__(self, class_counts: np.array, p: float = 0.8, num_classes: int = 51):
        super().__init__()

        class_counts = torch.FloatTensor(class_counts)
        conditions = class_counts[:, None] > class_counts[None, :]
        trues = (class_counts[None, :] / class_counts[:, None]) ** p
        falses = torch.ones(len(class_counts), len(class_counts))
        self.s = torch.where(conditions, trues, falses)

        self.num_classes = num_classes

        self.eps = 1.0e-6

    def forward(self, logits, targets):
        # 如果 targets 是独热编码格式，直接使用
        if targets.dim() == 2:  # 检查是否为 [batch_size, num_classes]
            targets = targets.float().to(targets.device)  # 确保是浮点型
        else:
            targets = nn.functional.one_hot(targets, num_classes=self.num_classes).float().to(targets.device)
        max_element, _ = logits.max(axis=-1)
        logits = logits - max_element[:, None]  # to prevent overflow
        self.s = self.s.to(targets.device)
        numerator = torch.exp(logits)
        denominator = (
            (1 - targets)[:, None, :]
            * self.s[None, :, :]
            * torch.exp(logits)[:, None, :]).sum(axis=-1) \
            + torch.exp(logits)

        sigma = numerator / (denominator + self.eps)
        loss = (- targets * torch.log(sigma + self.eps)).sum(-1)
        return loss.mean()