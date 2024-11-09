from pytorch_lightning.cli import LightningCLI
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.distributed.elastic.agent.server.api import logger
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import INaturalist
from pytorch_lightning.loggers import WandbLogger
import wandb
import timm

class INatDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './data', batch_size: int = 32, num_workers: int = 3, input_size: int = 448):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size

        self.transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val_test': transforms.Compose([
                transforms.Resize(int(self.input_size * 1.2)),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = INaturalist(
                root=self.data_dir,
                version='2021_train_mini',
                transform=self.transform['train']
            )
            self.val_dataset = INaturalist(
                root=self.data_dir,
                version='2021_valid',
                transform=self.transform['val_test']
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, persistent_workers=True)
class BaseModel(pl.LightningModule):
    def __init__(self, num_classes: int = 51, learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        
        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == y).float() / len(y)
        self.log('epoch', self.current_epoch, prog_bar=True)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

class SwinV2Model(BaseModel):
    def __init__(self, num_classes: int = 51, learning_rate: float = 1e-4, input_size = 448):
        super().__init__(num_classes, learning_rate)
        self.model = timm.create_model('timm/swinv2_tiny_window16_256.ms_in1k', pretrained=True)
        self.model.set_input_size([input_size, input_size])
        self.model.head.fc = nn.Linear(self.model.head.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class EfficientNetV2Model(BaseModel):
    def __init__(self, num_classes: int = 51, learning_rate: float = 1e-4):
        super().__init__(num_classes, learning_rate)
        self.model = timm.create_model('tf_efficientnetv2_s.in1k', pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def cli_main():
    wandb_logger = WandbLogger(
        project="identify",  # 项目名称
        log_model=True,       # 记录模型检查点
        save_dir='wandb_logs' # 日志保存目录
    )
    cli = LightningCLI(
        model_class=BaseModel,
        datamodule_class=INatDataModule,
        subclass_mode_model=True,  # 启用子类模式
        save_config_callback=None,
        seed_everything_default=42,
        trainer_defaults={
               "logger": {
                "class_path": "pytorch_lightning.loggers.WandbLogger",
                "init_args": {
                    "project": "identify",
                    "log_model": True,
                    "save_dir": "wandb_logs"
                }
            }
        }
    )

if __name__ == '__main__':
    cli_main()