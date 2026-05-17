import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Sampler
from torchvision.datasets import INaturalist
from torchvision.transforms import TrivialAugmentWide
from torchvision.transforms import v2

from dataSet.SpecialCateDataset import SpecialCateDataset
from util.transform import ToRGBTransform


class ClassBalancedSampler(Sampler):
    """前 N-3 epoch 做 uniform shuffle，最后 3 epoch 做 class-balanced 采样。

    均衡策略：每类至少保证 floor_count 个有效样本。
    - 所有图片先各出现一次（头类图片全部保留）
    - 尾类（N_c < floor_count）随机补足 floor_count - N_c 份额外副本

    通过 _trainer 引用动态读取当前 epoch。
    """
    def __init__(self, dataset, rebalance_start_epoch, floor_count=100):
        class_counts = {}
        class_indices = {}
        for i in range(len(dataset)):
            label = dataset.index[i][0]
            class_counts[label] = class_counts.get(label, 0) + 1
            class_indices.setdefault(label, []).append(i)

        self.original_num_samples = len(dataset)
        self.rebalance_start_epoch = rebalance_start_epoch
        self.floor_count = floor_count
        self.class_counts = class_counts
        self.class_indices = class_indices
        # epoch 总样本数 = 原始全部 + 尾类额外补足
        self.rebalance_num_samples = self.original_num_samples + sum(
            max(0, floor_count - n_c) for n_c in class_counts.values()
        )
        self._trainer = None

    def __iter__(self):
        epoch = self._trainer.current_epoch if self._trainer is not None else 0
        if self.rebalance_start_epoch >= 0 and epoch >= self.rebalance_start_epoch:
            indices = list(range(self.original_num_samples))
            for label, n_c in self.class_counts.items():
                deficit = self.floor_count - n_c
                if deficit > 0:
                    pool = self.class_indices[label]
                    extras = [pool[i] for i in torch.randint(0, len(pool), (deficit,)).tolist()]
                    indices.extend(extras)
            # shuffle 使得各份副本分散在 epoch 中
            perm = torch.randperm(len(indices)).tolist()
            return iter([indices[i] for i in perm])
        else:
            return iter(torch.randperm(self.original_num_samples).tolist())

    def __len__(self):
        epoch = self._trainer.current_epoch if self._trainer is not None else 0
        if self.rebalance_start_epoch >= 0 and epoch >= self.rebalance_start_epoch:
            return self.rebalance_num_samples
        return self.original_num_samples


class INatBaseDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './data/tiny', valid_dir= './data/tiny', id_map_file_path = '', valid_id_map_file_path = '',batch_size: int = 32, num_workers: int = 3, input_size: int = 448, num_classes = 51, rrc_scale_min: float = 0.3, random_erase_prob: float = 0.25, re_scale_min: float = 0.02, re_scale_max: float = 0.2, rebalance_last_epochs: int = 0, rebalance_floor_count: int = 100):
        super().__init__()
        self.data_dir = data_dir
        self.valid_dir = valid_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.id_map_file_path = id_map_file_path
        self.valid_id_map_file_path = valid_id_map_file_path
        self.input_size = input_size
        self.num_classes = num_classes
        self.rrc_scale_min = rrc_scale_min
        self.random_erase_prob = random_erase_prob
        self.re_scale_min = re_scale_min
        self.re_scale_max = re_scale_max
        self.rebalance_last_epochs = rebalance_last_epochs
        self.rebalance_floor_count = rebalance_floor_count
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        self.transform = {
            'train': v2.Compose([
                ToRGBTransform(),
                v2.ToImage(), # Convert to tensor, only needed if you had a PIL image
                v2.RandomResizedCrop(self.input_size, scale=(self.rrc_scale_min, 1.0)),
                v2.RandomApply([v2.RandomRotation(degrees=45, fill="random")], p=0.2),
                TrivialAugmentWide(),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                v2.RandomErasing(p=self.random_erase_prob, scale=(self.re_scale_min, self.re_scale_max), value='random')
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
        if self.rebalance_last_epochs > 0 and self.trainer is not None:
            start_epoch = self.trainer.max_epochs - self.rebalance_last_epochs
            sampler = ClassBalancedSampler(self.train_dataset, start_epoch, self.rebalance_floor_count)
            sampler._trainer = self.trainer
            return DataLoader(self.train_dataset, batch_size=self.batch_size,
                              sampler=sampler, num_workers=self.num_workers,
                              persistent_workers=True, prefetch_factor=1)
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
