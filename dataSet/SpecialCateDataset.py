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

class SpecialCateDataset(Dataset):
    def __init__(self, root_dir = '', id_map_file_path = '', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.species_ids = []
        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path) and not class_dir.startswith('.'):
                self.species_ids.extend(
                    species_id for species_id in os.listdir(class_path)
                    if os.path.isdir(os.path.join(class_path, species_id)) 
                    and not species_id.startswith('.')
                )

        self.species_ids = sorted(self.species_ids)
        self.index_to_species_id = {i: species_id for i, species_id in enumerate(self.species_ids)}
        print("数据集初始化成功，已写入到 csv 文件")
        save_index_to_species_id_to_csv(self.index_to_species_id, id_map_file_path)
        species_id_to_index = {v: k for k, v in self.index_to_species_id.items()}

        self.index: List[Tuple[int, str]] = []

        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path) and not class_dir.startswith('.'):
                for species_id in os.listdir(class_path):
                    species_path = os.path.join(class_path, species_id)
                    if os.path.isdir(species_path) and not species_id.startswith('.'):
                        dir_index = species_id_to_index[species_id]
                        for fname in os.listdir(species_path):
                            if not fname.startswith('.'):
                                rel_path = os.path.join(class_dir, species_id, fname)
                                self.index.append((dir_index, rel_path))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dir_index, fname = self.index[idx]
        image = Image.open(os.path.join(self.root_dir, fname))

        if self.transform:
            image = self.transform(image)
        return image, dir_index