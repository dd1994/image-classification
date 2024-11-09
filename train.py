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
import ctypes
from ctypes import wintypes

# Windows constants
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

def prevent_sleep():
    """防止 Windows 系统休眠"""
    try:
        # 加载 kernel32.dll
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        # 设置函数参数类型
        kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
        kernel32.SetThreadExecutionState.restype = wintypes.DWORD
        
        # 设置系统状态：保持系统运行、显示器开启
        execution_state = ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        kernel32.SetThreadExecutionState(execution_state)
        print("已设置防止系统休眠")
    except Exception as e:
        print(f"设置防止休眠失败: {e}")

def restore_sleep():
    """恢复 Windows 系统休眠设置"""
    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
        kernel32.SetThreadExecutionState.restype = wintypes.DWORD
        
        # 恢复正常状态
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("已恢复系统休眠设置")
    except Exception as e:
        print(f"恢复休眠设置失败: {e}")

class CustomTransform:
    def __call__(self, img):
        # 检查图像的通道数
        if img.mode == 'L':  # 灰度图像
            # 将灰度图像转换为三通道
            img = img.convert('RGB')
        return img

class INatBaseDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './data', batch_size: int = 32, num_workers: int = 3, input_size: int = 448):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size

        self.transform = {
            'train': transforms.Compose([
                CustomTransform(),  # 自定义转换
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val_test': transforms.Compose([
                CustomTransform(),  # 自定义转换
                transforms.Resize(int(self.input_size * 1.2)),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, persistent_workers=True)


class INatDataModule2019(INatBaseDataModule):
    def __init__(self, val_split: float = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_split = val_split  # 验证集比例

    def setup(self, stage=None):
        # 加载整个数据集
        full_dataset = INaturalist(
            root=self.data_dir,
            version='2019',  # 2019 数据集
            transform=self.transform['train']
        )

        # 按比例划分训练和验证集
        train_size = int((1 - self.val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])


class INatDataModule2021(INatBaseDataModule):
    def setup(self, stage=None):
        # 直接加载训练和验证数据集
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

class BaseModel(pl.LightningModule):
    def __init__(self, num_classes: int = 51, learning_rate: float = 1e-4, input_size=448):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()
        
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss"
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
    # 训练开始前设置防止休眠
    prevent_sleep()
    
    try:
        cli = LightningCLI(
            model_class=BaseModel,
            datamodule_class=INatBaseDataModule,
            subclass_mode_model=True,
            subclass_mode_data=True,
            save_config_callback=None,
            seed_everything_default=42,
            trainer_defaults={
                "logger": {
                    "class_path": "pytorch_lightning.loggers.WandbLogger",
                    "init_args": {
                        "project": "identify",
                        "log_model": True,
                        "mode": "offline",
                        "save_dir": "wandb_logs"
                    }
                }
            }
        )
    finally:
        # 训练结束后恢复休眠设置
        restore_sleep()

if __name__ == '__main__':
    cli_main()