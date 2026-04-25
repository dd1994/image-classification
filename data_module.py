import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import INaturalist
from torchvision.transforms import TrivialAugmentWide
from torchvision.transforms import v2

from dataSet.SpecialCateDataset import SpecialCateDataset
from util.transform import ToRGBTransform


class INatBaseDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './data/tiny', valid_dir= './data/tiny', id_map_file_path = '', valid_id_map_file_path = '',batch_size: int = 32, num_workers: int = 3, input_size: int = 448, num_classes = 51):
        super().__init__()
        self.data_dir = data_dir
        self.valid_dir = valid_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.id_map_file_path = id_map_file_path
        self.valid_id_map_file_path = valid_id_map_file_path
        self.input_size = input_size
        self.num_classes = num_classes
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.transform = {
            'train': v2.Compose([
                ToRGBTransform(),
                v2.ToImage(), # Convert to tensor, only needed if you had a PIL image
                v2.RandomResizedCrop(self.input_size),
                TrivialAugmentWide(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                v2.RandomErasing()
                # 加不加 Random Erasing，都是完全一样的效果，不知道为啥。
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
                          shuffle=True, num_workers=self.num_workers, persistent_workers=True, prefetch_factor=1)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, persistent_workers=True)
    

class SpecialCateData(INatBaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):
        # 加载整个数据集
        train_dataset = SpecialCateDataset(
            root_dir=self.data_dir,
            transform=self.transform['train'],
            id_map_file_path=self.id_map_file_path
        )

        valid_dataset = SpecialCateDataset(
            root_dir=self.valid_dir,
            transform= self.transform['val_test'],
            id_map_file_path=self.valid_id_map_file_path
        )

        self.train_dataset = train_dataset
        self.val_dataset = valid_dataset

class INatDataModule2019(INatBaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, stage=None):
        # 加载整个数据集
        full_dataset = INaturalist(
            root=self.data_dir,
            version='2019',  # 2019 数据集
            transform=self.transform['train']
        )

        # 按比例划分训练、验证和测试集
        train_size = int(0.8 * len(full_dataset))  # 50% 训练集
        val_size = int(0.1 * len(full_dataset))    # 10% 验证集
        test_size = len(full_dataset) - train_size - val_size  # 剩余 10% 测试集
        
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

        # 应用转换
        self.val_dataset.dataset.transform = self.transform['val_test']
        self.test_dataset.dataset.transform = self.transform['val_test']


class INatDataModule2021Mini(INatBaseDataModule):
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

        self.test_dataset = INaturalist(
            root=self.data_dir,
            version='2021_valid',
            transform=self.transform['val_test']
        )
