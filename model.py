import pytorch_lightning as pl
import timm
import torch
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

        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc_top1', top1_acc, prog_bar=True, on_step=False, on_epoch=True)

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

    def forward(self, x):
        return self.model(x)

class HieraModel(BaseModel):
    def __init__(self, num_classes: int = 51, learning_rate: float = 1e-4, input_size = 448, t_max=20):
        super().__init__(t_max=t_max, learning_rate=learning_rate)
        self.model = torch.hub.load("facebookresearch/hiera", model="hiera_base_plus_224", pretrained=True,
                               checkpoint="mae_in1k_ft_in1k" , input_size = (224, 224))
        # adjust_pos_embed_shape(self.model, input_size)
        in_features = self.model.head.projection.in_features
        self.model.head.projection = nn.Linear(in_features, num_classes)

        # , input_size = (224, 224), num_classes = num_classes
        # print(self.model)
        # out_features = self.model.head.projection.out_features
        # self.model.head.projection.out_features = num_classes

    def forward(self, x):
        return self.model(x)

class EfficientNetV2Model(BaseModel):
    def __init__(self, num_classes: int = 51, learning_rate: float = 1e-4, t_max=20):
        super().__init__(t_max=t_max, learning_rate=learning_rate)
        self.model = timm.create_model('tf_efficientnetv2_l.in21k_ft_in1k', pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
