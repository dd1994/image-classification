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
BASE_DIR = r"D:\image-classification\data\dup_test"
ACTIVE = ''

# 不同类群的配置参数
DEFAULT_CONFIG = {
    'similarity_threshold': 0.6,
    'similarity_threshold_plus': 0.64, # 比如 700 张图片，只能删除高度相似的少数图片，所以这个相似度要设置得高
    'similarity_threshold_plus2': 0.62,
    'max_img_count': 600,
    'overflow_img_count': 850,
    'white_list_max_img_count': 1000,
    'white_list': []
}

CLASS_CONFIG = {
    'Amphibia': {
        'white_list': [
            '66330',
            '134932',
            '62345',
            '120791',
            '134709'
        ]
    },
    'Insecta': {},
    'Plantae': {},
    'Reptilia': {
        'white_list': [
            '539936',
            '30282',
            '30492',
            '101504',
            '966797',
            '1545858',
            '30217',
            '30996',
            '30442',
            '28872',
            '28868',
            '30685',
            '30253',
            '29832',
            '29266',
            '29256',
            '28999',
            '30231',
            '1337912'
        ]
    },
    'Arachnida': {},
}

NEIGHBOR_RANGE = 70  # 前后检查范围
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', 'webp')

batch_size = 380  # 根据GPU显存调整

# 获取当前类群的配置
current_config = {**DEFAULT_CONFIG, **CLASS_CONFIG.get(ACTIVE, {})} if ACTIVE in CLASS_CONFIG else DEFAULT_CONFIG
similarity_threshold = current_config['similarity_threshold']
similarity_threshold_plus = current_config['similarity_threshold_plus']
similarity_threshold_plus2 = current_config['similarity_threshold_plus2']
MAX_IMG_COUNT = current_config['max_img_count']
OVERFLOW_IMG_COUNT = current_config['overflow_img_count']
WHITE_LIST = current_config['white_list']
WHITE_LIST_MAX_IMG_COUNT = current_config['white_list_max_img_count']

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

def process_species_directory(species_dir, species_id):
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
    
    # 检查是否在白名单中，如果是则使用白名单最大图片数
    if species_id in WHITE_LIST:
        current_max = WHITE_LIST_MAX_IMG_COUNT
        print(f"物种 {species_id} 在白名单中，使用白名单最大图片数: {current_max}")

   # 修改后的数量检查
    if len(sorted_paths) <= current_max:
        return 0  # 直接跳过处理

    print(f"\n该物种总共有 {len(sorted_paths)} 张图片")

    if len(sorted_paths) <= (current_max + 200):
        current_threshold = similarity_threshold_plus2

    if len(sorted_paths) <= (current_max + 100):
        current_threshold = similarity_threshold_plus

    print(f"使用的相似度阈值: {current_threshold:.4f} (基础: {similarity_threshold}, +100: {similarity_threshold_plus}, +200: {similarity_threshold_plus2})")
    
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
        
        # 第一步：删除最模糊的5%（但确保删除后总数不低于MAX_IMG_COUNT）
        total = len(remaining_paths)
        allowed_to_delete = max(0, total - current_max)  # 允许删除的最大数量
        if allowed_to_delete > 0:
            # 取5%和允许删除量的较小值，且至少删除1张
            to_delete_blur_count = min(max(1, int(np.ceil(total * 0.08))), allowed_to_delete)
        else:
            to_delete_blur_count = 0  # 不允许删除
        
        # 执行删除
        to_delete_blur = remaining_paths[:to_delete_blur_count]
        for path in to_delete_blur:
            try:
                os.remove(path)
                deleted_count += 1
            except Exception as e:
                print(f"删除模糊图片 {path} 失败: {e}")
        print(f"模糊度筛选删除 {len(to_delete_blur)} 张")
        
        # 更新剩余路径
        remaining_after_blur = remaining_paths[to_delete_blur_count:]
        
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
    all_species = []
    for class_name in os.listdir(BASE_DIR):
        if ACTIVE and class_name != ACTIVE:
            continue
        class_dir = os.path.join(BASE_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        for species_id in os.listdir(class_dir):
            species_dir = os.path.join(class_dir, species_id)
            if os.path.isdir(species_dir):
                all_species.append((class_name, species_id, species_dir))

    with tqdm(all_species, desc="处理进度", unit="物种") as pbar:
        for class_name, species_id, species_dir in pbar:
            pbar.set_postfix_str(f"{class_name}/{species_id}")
            process_species_directory(species_dir, species_id)


if __name__ == "__main__":
    main()