import pytorch_lightning as pl
import timm
import torch
from aim.v2.utils import load_pretrained
from timm.models.hiera import Hiera
from torch import nn as nn
from torchvision.transforms import v2
from transformers import ConvNextV2ForImageClassification

from util.get_inat_2019_class_counts import get_inat2019_class_counts
from util.seesaw_loss import SeesawLossWithLogits


class BaseModel(pl.LightningModule):
    def __init__(self, num_classes: int = 51, t_max=20, learning_rate: float = 1e-4, class_counts=None):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()
        self.t_max = t_max
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.class_counts = class_counts
         # 如果 class_counts 没有传入，则初始化为全零的列表
        if class_counts is None:
            self.class_counts = [1] * num_classes
        else:
            self.class_counts = class_counts

        self.loss_tr = SeesawLossWithLogits(class_counts, num_classes=num_classes)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        # 80% 的概率使用 CutMix 或 MixUp
        if torch.rand(1).item() < 0.8:
            cutmix = v2.CutMix(num_classes=self.num_classes)
            mixup = v2.MixUp(num_classes=self.num_classes)
            cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
            # 应用 CutMix 或 MixUp
            images, labels = cutmix_or_mixup(images, labels)

        outputs = self(images)
        loss = self.loss_tr(outputs, labels)

        # 获取预测的类别
        preds = torch.argmax(outputs, dim=1)  # 预测类别索引

        # 处理标签 labels 的形状
        if labels.dim() == 2:  # 如果 labels 是 [16, 23]，表示使用了 CutMix 或 MixUp
            true_labels = torch.argmax(labels, dim=1)  # 获取真实类别索引
        else:  # 如果 labels 是 [16]，表示没有使用 CutMix 或 MixUp
            true_labels = labels  # 直接使用 labels

        acc = (preds == true_labels).float().mean()  # 计算准确率
        self.log('epoch', self.current_epoch, prog_bar=False)
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', acc, prog_bar=True)
        # 记录学习率
        if batch_idx == 0:
            self.log('train/lr', self.trainer.optimizers[0].param_groups[0]['lr'])

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        _, preds = torch.max(outputs, 1)
        top1_acc = (preds == y).float().mean()

        # 计算 Top-3 准确率
        top3_preds = torch.topk(outputs, k=3, dim=1).indices
        top3_acc = (top3_preds == y.view(-1, 1)).sum().float() / len(y)

        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc_top1', top1_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/acc_top3', top3_acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        _, preds = torch.max(outputs, 1)
        top1_acc = (preds == y).float().mean()

        # 计算 Top-3 准确率
        top3_preds = torch.topk(outputs, k=3, dim=1).indices
        top3_acc = (top3_preds == y.view(-1, 1)).sum().float() / len(y)

        self.log('test/loss', loss, prog_bar=True)
        self.log('test/acc_top1', top1_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test/acc_top3', top3_acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=2e-5)
        
              # 添加学习率预热
        warmup_epochs = 1  # 预热的 epoch 数，可以根据需要调整
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1e-6 if epoch < warmup_epochs else 1
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.t_max, eta_min=1e-8
        )

        combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[warmup_epochs]  # 在 warmup_epochs 之后切换到 CosineAnnealingLR
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": combined_scheduler,
                "monitor": "val/loss"
            }
        }


class SwinV2Model(BaseModel):
    def __init__(self, num_classes: int = 51, learning_rate: float = 1e-4, input_size = 448, t_max=20):
        super().__init__(t_max=t_max, num_classes=num_classes,class_counts=get_inat2019_class_counts(num_classes), learning_rate=learning_rate)
        self.model = timm.create_model('swinv2_base_window12to24_192to384.ms_in22k_ft_in1k', pretrained=True)
        self.model.set_input_size([input_size, input_size])
        self.model.head.fc = nn.Linear(self.model.head.fc.in_features, num_classes)

        # checkpoint = torch.load('wandb_logs/identify/zdyuh8p2/checkpoints/swinv2-inat2021-mini-epoch=12-val/acc_top1=0.8633.ckpt')
        #
        # state_dict = checkpoint['state_dict']
        #     # 移除最后一层的权重
        # state_dict.pop('model.head.fc.weight', None)
        # state_dict.pop('model.head.fc.bias', None)
        # self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        return self.model(x)

class ConvNextV2Model(BaseModel):
    # 在小的数据集上和 swin 不相上下，但是在 iNat2019 这样的数据集上只有 84% 的识别率。
    def __init__(self, num_classes: int = 51, learning_rate: float = 1e-4, input_size = 448, t_max=20):
        super().__init__(t_max=t_max,class_counts=get_inat2019_class_counts(num_classes), num_classes=num_classes,learning_rate=learning_rate)
        self.model = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-base-22k-384")
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x).logits

class DinoV2Model(BaseModel):
    def __init__(self, num_classes: int = 51, learning_rate: float = 1e-4, input_size = 448, t_max=20):
        super().__init__(t_max=t_max, class_counts=get_inat2019_class_counts(num_classes), num_classes=num_classes, learning_rate=learning_rate)

        # 引入 DINO V2 模型
        self.model =  torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg_lc')
        self.model.linear_head = nn.Linear(self.model.linear_head.in_features, num_classes)

        # 冻结特征提取层的参数，但不冻结最后一层分类层
        for name, param in self.model.named_parameters():
            if 'linear_head' not in name:  # 确保不冻结分类层
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)


class AIMv2Model(BaseModel):
    def __init__(self, num_classes: int = 51, learning_rate: float = 1e-4, input_size = 448, t_max=20):
        super().__init__(t_max=t_max,num_classes=num_classes, learning_rate=learning_rate)
        self.model = load_pretrained("aimv2-large-patch14-448", backend="torch")
           # 冻结特征提取层的参数
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 展平层
            nn.MaxPool1d(kernel_size=2),  # 添加最大池化层，注意输入输出维度
            nn.Linear(1024 * 1024 // 2, 1024),  # 第一隐藏层
            nn.ReLU(),  # 激活函数
            nn.Dropout(0.5),  # Dropout 层
            nn.Linear(1024, num_classes)  # 输出层，51 个类别
        )

    def forward(self, x):
        features = self.model(x)
        # print(f"Input shape: {x.shape}")
        # for layer in self.classifier:
        #     x = layer(x)
        #     print(f"After {layer.__class__.__name__}: {x.shape}")
        # return x
        return self.classifier(features)

class HieraModel(BaseModel):
    def __init__(self, num_classes: int = 51, learning_rate: float = 1e-4, input_size=448, t_max=20):
        super().__init__(t_max=t_max, learning_rate=learning_rate)
        

        pretrained_model = timm.create_model('hiera_base_plus_224.mae_in1k_ft_in1k', pretrained=True)
        pretrained_model.head.fc = nn.Linear(pretrained_model.head.fc.in_features, num_classes)

        # 修改模型的输入层以支持 448px 输入

        # self.model = Hiera(
        #     img_size=(input_size, input_size),  # 设置输入大小为 448x448
        #     embed_dim=96,  # 嵌入维度
        #     num_heads=1,   # 注意力头数
        #     stages=(2, 3, 16, 3),  # 各个阶段的块数
        #     num_classes=num_classes,  # 类别数
        #     # 其他参数可以根据需要添加
        # )

        print(pretrained_model)
        self.model = Hiera(
            img_size=(input_size, input_size),  # 设置输入大小为 448x448
            embed_dim=112,  # 嵌入维度
            num_heads=2,  # 注意力头数
            stages=(2, 3, 16, 3),  # 各个阶段的块数
            num_classes=num_classes,  # 类���数
            # 其他参数可以根据需要添加
        )
        print(self.model)
        self.model.load_state_dict(pretrained_model.state_dict(), strict=False)

        # self.model.head.fc = nn.Linear(self.model.head.fc.in_features, num_classes)

        # in_features = self.model.head.projection.in_features
        # self.model.head.projection = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # 直接使用 448px 的输入
        return self.model(x)


class EfficientNetV2Model(BaseModel):
    def __init__(self, num_classes: int = 51, learning_rate: float = 1e-4, t_max=20):
        super().__init__(t_max=t_max, learning_rate=learning_rate)
        self.model = timm.create_model('tf_efficientnetv2_l.in21k_ft_in1k', pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
