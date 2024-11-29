import pytorch_lightning as pl
import timm
import torch
from timm.models.hiera import PatchEmbed, Hiera
from torch import nn as nn


class BaseModel(pl.LightningModule):
    def __init__(self, t_max=20, learning_rate: float = 1e-4,):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()
        self.t_max = t_max
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)

        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == y).float() / len(y)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.t_max, eta_min=self.learning_rate*0.001
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss"
            }
        }


class SwinV2Model(BaseModel):
    def __init__(self, num_classes: int = 51, learning_rate: float = 1e-4, input_size = 448, t_max=20):
        super().__init__(t_max=t_max, learning_rate=learning_rate)
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
            num_classes=num_classes,  # 类别数
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
