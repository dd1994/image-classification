import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import INaturalist
import timm

class INatDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = '../data', batch_size: int = 32, num_workers: int = 3, input_size: int = 448):
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

class SwinV2Model(pl.LightningModule):
    def __init__(self, num_classes: int = 51, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # 模型初始化
        self.model = timm.create_model('timm/swinv2_tiny_window16_256.ms_in1k', pretrained=True)
        self.model.set_input_size([448, 448])
        self.model.head.fc = nn.Linear(self.model.head.fc.in_features, num_classes)
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        
        # 计算准确率
        _, preds = torch.max(outputs, 1)
        acc = torch.sum(preds == y).float() / len(y)
        
        # 记录指标
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss = self.criterion(outputs, y)
        _, preds = torch.max(outputs, 1)
        top1_acc = (preds == y).float().mean()
        
        # 记录指标
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc_top1', top1_acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
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

def main():
    # 数据模块
    data_module = INatDataModule()
    
    # 模型
    model = SwinV2Model()
    
    # 回调函数
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min'
        ),
        ModelCheckpoint(
            monitor='val_acc_top1',
            filename='swinv2-{epoch:02d}-{val_acc_top1:.2f}',
            save_top_k=1,
            mode='max'
        )
    ]
    
    # 日志记录器
    logger = TensorBoardLogger("lightning_logs", name="swinv2")
    
    # 训练器
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='gpu',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        precision=16  # 使用混合精度训练
    )
    
    # 开始训练
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()