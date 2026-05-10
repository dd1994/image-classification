import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """ArcFace additive angular margin loss.

    Stores learnable class centers (weight matrix) to compute cosine
    similarity with input embeddings, applies angular margin to the
    target class during training, and computes cross-entropy loss.

    Compatible with MixUp/CutMix soft labels: when ``labels.dim() == 2``
    the margin is skipped and only scaled cosine logits are used.

    Args:
        in_features: embedding dimension from backbone.
        num_classes: number of classes.
        s: feature scale factor (default 30.0).
        m: angular margin in radians (default 0.50).
        number_sub_center: K sub-centers per class, 1 = standard ArcFace.
        easy_margin: use SphereFace-style margin (cos > 0) instead of
            standard ArcFace threshold.
        ls_eps: label smoothing epsilon (default 0.0).
    """

    def __init__(self, in_features, num_classes, s=30.0, m=0.50,
                 number_sub_center=1, easy_margin=False, ls_eps=0.0):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.number_sub_center = number_sub_center
        self.easy_margin = easy_margin
        self.ls_eps = ls_eps

        weight = torch.FloatTensor(num_classes * number_sub_center, in_features)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.ce_loss = nn.CrossEntropyLoss()

    def set_margin(self, m):
        """Dynamically update angular margin and dependent constants."""
        self.m = m
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embedding, labels):
        """Compute ArcFace loss and logits.

        Returns:
            (loss, logits): Cross-entropy loss and logits used for
            accuracy computation.
        """
        # Force float32 for numerical stability in margin computation
        embedding = embedding.float()
        weight = self.weight.float()
        cosine = F.linear(F.normalize(embedding), F.normalize(weight))

        if self.number_sub_center > 1:
            cosine = cosine.view(-1, self.num_classes, self.number_sub_center)
            cosine, _ = torch.max(cosine, dim=2)

        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if labels.dim() == 2:
            # Soft labels from MixUp/CutMix — skip margin
            logits = cosine * self.s
            loss = self.ce_loss(logits, labels)
            return loss, logits

        # Apply ArcFace margin
        phi = torch.cos(torch.acos(cosine) + self.m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.num_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits = output * self.s
        loss = self.ce_loss(logits, labels)

        return loss, logits

    def get_logits(self, embedding):
        """Return scaled cosine logits without margin (for inference/validation)."""
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight))

        if self.number_sub_center > 1:
            cosine = cosine.view(-1, self.num_classes, self.number_sub_center)
            cosine, _ = torch.max(cosine, dim=2)

        return cosine * self.s
