import csv
import os
from typing import Tuple, List

from PIL import Image
from torch.utils.data import Dataset

def save_index_to_species_id_to_csv(index_to_species_id, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'SpeciesID'])  # 写入表头
        for index, species_id in index_to_species_id.items():
            writer.writerow([index, species_id])

class AmphibiansDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.species_ids = sorted([
            species_id for species_id in os.listdir(root_dir) if  not species_id.startswith('.')
        ])
        self.index_to_species_id = {i: species_id for i, species_id in enumerate(self.species_ids)}
        save_index_to_species_id_to_csv(self.index_to_species_id, 'index_to_species_id.csv')

        self.index: List[Tuple[int, str]] = []

        for dir_index, dir_name in enumerate(self.species_ids):
            files = os.listdir(os.path.join(self.root_dir, dir_name))
            for fname in files:
                if not fname.startswith('.'):  # Ignore hidden files
                    self.index.append((dir_index, fname))
        # 遍历每个子文件夹，收集所有图片路径及其对应的标签
        # for idx, species_id in enumerate(self.species_ids):
        #     species_dir = os.path.join(root_dir, species_id)
        #     print(species_dir)
        #     if not os.path.isdir(species_dir):
        #         continue
        #     for image_name in os.listdir(str(species_dir)):
        #         image_path = os.path.join(str(species_dir), image_name)
        #         self.index.append((image_path))
        #         self.labels.append(int(species_id))  # 假设类别标签是根据子文件夹的顺序分配的

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dir_index, fname = self.index[idx]
        image = Image.open(os.path.join(self.root_dir, self.index_to_species_id[dir_index], fname))

        if self.transform:
            image = self.transform(image)
        return image, dir_index