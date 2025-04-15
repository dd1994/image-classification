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
import cv2  # 需要安装 opencv-python 包
import random

# 配置参数
BASE_DIR = r"D:\image-classification\data\train-small"
# ACTIVE = 'Insecta'
similarity_threshold = 0.573
similarity_threshold_plus = 0.583
NEIGHBOR_RANGE = 100  # 前后检查范围
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg')

MAX_IMG_COUNT= 600 #
OVERFLOW_IMG_COUNT = 894 # 植物设置为 990，
batch_size = 380  # 根据GPU显存调整

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_pretrained("aimv2-large-patch14-224-distilled", backend="torch").to(device)
model.eval()
transform = val_transforms(img_size=224)

def calculate_blur_score(image_path):
    """计算图像模糊分数（值越小越模糊）"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return float('inf')  # 无效文件返回最大值避免误删
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    except Exception as e:
        print(f"计算模糊度失败 {image_path}: {e}")
        return float('inf')

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
    
    # 新增动态调整逻辑
    current_max = MAX_IMG_COUNT
    current_threshold = similarity_threshold

   # 修改后的数量检查
    if len(sorted_paths) <= current_max:
        return 0  # 直接跳过处理

    print(f"\n该物种总共有 {len(sorted_paths)} 张图片")

    if len(sorted_paths) <= (current_max + 150):
        # 这种情况下图片没有超出太多，阈值调大一点，不然去重后图片数量太少了。
        current_threshold = similarity_threshold_plus
    # elif len(sorted_paths) > OVERFLOW_IMG_COUNT:
    #     # 因为限制了最多下载 900 张图片，对于达到这个最大值的物种来说，是最常见的物种，为了增加最常见物种的识别率，给它增加 100 张训练图片（用 895 是因为偶尔出现图片下载错误，达不到 900 张）
    #     current_max += 100

    
    # 修改特征提取部分为批处理
    features = []
    valid_paths = []
    
    # 使用批处理加速特征提取
    with torch.no_grad():
        for i in range(0, len(sorted_paths), batch_size):
            batch_paths = sorted_paths[i:i+batch_size]
            batch_images = []
            
            # 预处理批次图像
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    inp = transform(img).unsqueeze(0).to(device)
                    batch_images.append(inp)
                except Exception as e:
                    print(f"处理图片 {path} 失败: {e}")
                    continue
            
            if not batch_images:
                continue
                
            # 批量推理
            batch = torch.cat(batch_images, dim=0)
            batch_features = model(batch).cpu().numpy()
            
            # 保存结果
            for j in range(batch_features.shape[0]):
                features.append(batch_features[j].flatten())
                valid_paths.append(batch_paths[j])

    if len(features) < 2:
        return 0  # 图片数量不足无需去重

    features = np.array(features)
    to_delete = set()


    # 修改相似度计算部分为GPU加速
    features_tensor = torch.tensor(features, device=device)
    total = len(valid_paths)
    
    for i in range(total):
        if i in to_delete:
            continue

        # 计算范围
        start = i + 1
        end = min(total, i + NEIGHBOR_RANGE + 1)
        if start >= end:
            continue

        # 批量计算相似度
        current_feature = features_tensor[i]
        compare_features = features_tensor[start:end]
        
        # 使用矩阵运算加速
        with torch.no_grad():
            norms = torch.norm(current_feature) * torch.norm(compare_features, dim=1)
            similarities = torch.mm(current_feature.unsqueeze(0), compare_features.T).squeeze(0)
            similarities = similarities / norms
        
        # 找出超过阈值的索引
        over_threshold = torch.nonzero(similarities > current_threshold).squeeze(1)
        for idx in over_threshold:
            j = start + idx.item()
            if j not in to_delete:
                # print(f"\n相似图片对 (相似度 {similarities[idx].item():.4f}):")
                # print(f"基准图片: {valid_paths[i]}")
                # print(f"重复图片: {valid_paths[j]}")
                # print("-" * 80)
                to_delete.add(j)

    # 执行删除操作
    deleted_count = 0
    for idx in sorted(to_delete, reverse=True):
        try:
            os.remove(valid_paths[idx])
            deleted_count += 1
        except Exception as e:
            print(f"删除 {valid_paths[idx]} 失败: {e}")
    # return

    # 二次处理：模糊度去重
    remaining_paths = [p for i, p in enumerate(valid_paths) if i not in to_delete]
    if len(remaining_paths) > current_max:
        print(f"相似度去重删除 {len(to_delete)} 张")
        
        # 计算所有剩余图片的模糊度
        blur_scores = []
        for path in remaining_paths:
            score = calculate_blur_score(path)
            blur_scores.append((path, score))
        
        # 按模糊度排序（分数低的模糊图片在前）
        blur_scores.sort(key=lambda x: x[1])
        
        # 第一步：删除最模糊的5%
        total = len(blur_scores)
        to_delete_blur_count = max(1, int(np.ceil(total * 0.05)))  # 至少删除1张
        to_delete_blur = [item[0] for item in blur_scores[:to_delete_blur_count]]
        
        # 执行删除
        for path in to_delete_blur:
            try:
                os.remove(path)
                deleted_count += 1
            except Exception as e:
                print(f"删除模糊图片 {path} 失败: {e}")
        print(f"模糊度筛选删除 {len(to_delete_blur)} 张")
        
        # 更新剩余路径
        remaining_after_blur = [item[0] for item in blur_scores[to_delete_blur_count:]]
        
        # 第二步：如果仍然超过限制，随机删除到保留 current_max 张
        if len(remaining_after_blur) > current_max:
            # print(f"模糊筛选后仍有 {len(remaining_after_blur)} 张，执行随机筛选")
            
            # 随机打乱列表
            random.shuffle(remaining_after_blur)
            
            # 保留前 current_max 张，删除多余的
            to_delete_random = remaining_after_blur[current_max:]
            
            # 执行删除
            for path in to_delete_random:
                try:
                    os.remove(path)
                    deleted_count += 1
                except Exception as e:
                    print(f"删除随机图片 {path} 失败: {e}")
            print(f"随机删除 {len(to_delete_random)} 张")
    print(f"总共删除{deleted_count} 张，剩余 {len(sorted_paths) - deleted_count} 张")
    return deleted_count

# 79,961
def main():
    # 遍历所有类别
    for class_name in os.listdir(BASE_DIR):
        # if class_name != ACTIVE:
        #     continue
        class_dir = os.path.join(BASE_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        # 收集当前类群的所有物种目录
        current_class_species = []
        for species_id in os.listdir(class_dir):
            species_dir = os.path.join(class_dir, species_id)
            if os.path.isdir(species_dir):
                current_class_species.append((species_id, species_dir))

        # 使用当前类群的进度条
        with tqdm(current_class_species, desc=f"处理 {class_name}", unit="物种") as class_pbar:
            for species_id, species_dir in class_pbar:
                class_pbar.set_postfix_str(species_id)
                deleted = process_species_directory(species_dir)
                class_pbar.write(f"\n{class_name}/{species_id}处理完成")
                # if deleted > 0:
                #     class_pbar.write(f"{class_name}/{species_id} 总共删除 {deleted} 张")


if __name__ == "__main__":
    main()