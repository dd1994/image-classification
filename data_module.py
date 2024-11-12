import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, default_collate
from torchvision.transforms import v2
from torchvision.datasets import INaturalist
from torchvision.transforms import TrivialAugmentWide

from util.transform import ToRGBTransform


class INatBaseDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './data/tiny', batch_size: int = 32, num_workers: int = 3, input_size: int = 448, num_classes = 51):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        self.num_classes = num_classes

        self.transform = {
            'train': v2.Compose([
                ToRGBTransform(),
                v2.ToImage(), # Convert to tensor, only needed if you had a PIL image
                v2.RandomResizedCrop(self.input_size),
                TrivialAugmentWide(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                # 不要加 Random Erasing，会显著降低识别率，不知道为啥。
            ]),
            'val_test': v2.Compose([
                ToRGBTransform(),
                v2.ToImage(),
                v2.Resize(int(self.input_size * 1.2)),
                v2.CenterCrop(self.input_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, persistent_workers=True)

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
            version='2021_train',
            transform=self.transform['train']
        )
        self.val_dataset = INaturalist(
            root=self.data_dir,
            version='2021_valid',
            transform=self.transform['val_test']
        )
