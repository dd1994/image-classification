# 我在进行动植物图像识别模型训练，有一个 "D:\image-classification\data\train" 文件夹，该文件夹下有 Amphibia，Insetca, Plantea 等子文件夹表示类别，
# 再下一层是物种 id 作为文件夹，再下一层是物种图片，每个物种大约有300多张照片。
# 现在写一段 python 脚本对高度相似的图片进行去重，使用一个已有的模型生成特征图，根据特征图来计算相似度进行去重。
# 我有一个减少复杂度的思路，根据图片的修改时间排序后，每个图片只可能和它的前 5 张和 后 5 张相似，因为他们是同一时间被拍照的。
# 下面是该生成特征图的模型的文档示例

# from PIL import Image
#
# from aim.v2.utils import load_pretrained
# from aim.v1.torch.data import val_transforms
#
# img = Image.open(...)
# model = load_pretrained("aimv2-large-patch14-336", backend="torch")
# transform = val_transforms(img_size=336)
#
# inp = transform(img).unsqueeze(0)
# features = model(inp)
#
# 处理过程最好展示进度

import os
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from aim.v2.utils import load_pretrained
from aim.v1.torch.data import val_transforms

# 配置参数
BASE_DIR = r"D:\image-classification\data\dup_test"
SIMILARITY_THRESHOLD = 0.6  # 相似度阈值，可调整
NEIGHBOR_RANGE = 10  # 前后检查范围
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg')

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_pretrained("aimv2-large-patch14-448", backend="torch").to(device)
transform = val_transforms(img_size=448)


def process_species_directory(species_dir):
    # 获取所有图片并按修改时间排序
    image_paths = []
    for fname in os.listdir(species_dir):
        if fname.lower().endswith(SUPPORTED_EXTENSIONS):
            path = os.path.join(species_dir, fname)
            image_paths.append((path, os.path.getmtime(path)))

    if not image_paths:
        return 0

    # 按修改时间排序
    image_paths.sort(key=lambda x: x[1])
    sorted_paths = [x[0] for x in image_paths]
    
    # 根据图片数量动态设置阈值
    if len(sorted_paths) > 500:
        print(f"图片数量 {len(sorted_paths)} > 500，使用阈值 0.5")
        similarity_threshold = 0.5
    elif len(sorted_paths) > 200:
        print(f"图片数量 {len(sorted_paths)} 201-500，使用阈值 0.6")
        similarity_threshold = 0.6
    else:
        print(f"图片数量 {len(sorted_paths)} ≤ 200，使用阈值 0.8")
        similarity_threshold = 0.8

    # 提取特征
    features = []
    valid_paths = []
    for path in tqdm(sorted_paths, desc="提取特征"):
        try:
            img = Image.open(path).convert('RGB')
            inp = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(inp).cpu().numpy().flatten()
            features.append(feat)
            valid_paths.append(path)
        except Exception as e:
            print(f"处理图片 {path} 失败: {e}")

    if len(features) < 2:
        return 0  # 图片数量不足无需去重

    features = np.array(features)
    to_delete = set()

    # 相似度检测
    total = len(valid_paths)
    for i in tqdm(range(total), desc="检测相似度"):
        if i in to_delete:
            continue

        # 仅检查后续的 NEIGHBOR_RANGE 张图片
        start = i + 1
        end = min(total, i + NEIGHBOR_RANGE + 1)

        for j in range(start, end):
            if j >= total or j in to_delete:
                continue

            # 计算余弦相似度
            vi, vj = features[i], features[j]
            norm = np.linalg.norm(vi) * np.linalg.norm(vj)
            if norm == 0:
                similarity = 0.0
            else:
                similarity = np.dot(vi, vj) / norm

            if similarity > similarity_threshold:
                to_delete.add(j)

    # 执行删除操作
    deleted_count = 0
    for idx in sorted(to_delete, reverse=True):
        try:
            os.remove(valid_paths[idx])
            deleted_count += 1
        except Exception as e:
            print(f"删除 {valid_paths[idx]} 失败: {e}")

    return deleted_count


def main():
    # 遍历所有类别和物种目录
    for class_name in os.listdir(BASE_DIR):
        class_dir = os.path.join(BASE_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        for species_id in os.listdir(class_dir):
            species_dir = os.path.join(class_dir, species_id)
            if not os.path.isdir(species_dir):
                continue

            print(f"\n正在处理: {class_name}/{species_id}")
            deleted = process_species_directory(species_dir)
            print(f"已删除 {deleted} 张重复图片")


if __name__ == "__main__":
    main()