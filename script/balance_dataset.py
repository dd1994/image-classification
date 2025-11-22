import os
import os.path
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm

def balance_dataset(root_dir, min_images=80):
    """
    平衡数据集，对图片数量少于min_images的物种进行复制和左右翻转处理
    
    Args:
        root_dir (str): 训练数据根目录，如 "train-pre"
        min_images (int): 最小图片数量阈值，默认为80
    """
    # 首先统计所有需要处理的物种
    all_species = []
    for class_dir in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
            
        for species_dir in os.listdir(class_path):
            species_path = os.path.join(class_path, species_dir)
            if not os.path.isdir(species_path):
                continue
                
            # 获取该物种文件夹下的所有图片文件
            image_files = [f for f in os.listdir(species_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # 如果图片数量少于阈值，则添加到需要处理的列表
            if len(image_files) < min_images:
                all_species.append((class_dir, species_dir, species_path, image_files))
    
    print(f"总共需要处理 {len(all_species)} 个物种")
    
    # 使用tqdm显示总体进度
    with tqdm(total=len(all_species), desc="处理物种进度", unit="物种") as species_pbar:
        # 遍历所有需要处理的物种
        for class_dir, species_dir, species_path, image_files in all_species:
            species_pbar.set_description(f"处理 {class_dir}/{species_dir}")
            
            for img_file in image_files:
                    img_path = os.path.join(species_path, img_file)
                    
                    try:
                        # 打开原始图片
                        with Image.open(img_path) as img:
                            # 创建副本（直接复制）
                            copy_name = f"copy_{img_file}"
                            copy_path = os.path.join(species_path, copy_name)
                            img.save(copy_path)
                            
                            # 创建左右翻转的副本
                            flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                            flip_name = f"copy_flipped_{img_file}"
                            flip_path = os.path.join(species_path, flip_name)
                            flipped_img.save(flip_path)
                            
                    except Exception as e:
                        print(f"\n处理图片 {img_file} 时出错: {e}")
            
            # 更新物种进度条
            species_pbar.update(1)
            final_count = len([f for f in os.listdir(species_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            species_pbar.set_postfix({"图片数量": f"{len(image_files)}→{final_count}"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="平衡数据集，对图片数量少的物种进行数据增强")
    parser.add_argument("--root_dir", type=str, default="D:\\image-classification\\data\\train-pre", 
                       help="训练数据根目录路径")
    parser.add_argument("--min_images", type=int, default=80, 
                       help="最小图片数量阈值")
    
    args = parser.parse_args()
    
    # 确保根目录存在
    if not os.path.exists(args.root_dir):
        print(f"错误: 目录 {args.root_dir} 不存在")
        exit(1)
        
    print(f"开始处理数据集: {args.root_dir}")
    print(f"最小图片数量阈值: {args.min_images}")
    
    # 统计总体信息
    total_species = 0
    total_images = 0
    need_process_species = 0
    
    for class_dir in os.listdir(args.root_dir):
        class_path = os.path.join(args.root_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
            
        for species_dir in os.listdir(class_path):
            species_path = os.path.join(class_path, species_dir)
            if not os.path.isdir(species_path):
                continue
                
            total_species += 1
            image_files = [f for f in os.listdir(species_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            total_images += len(image_files)
            
            if len(image_files) < args.min_images:
                need_process_species += 1
    
    print(f"统计信息: 总共 {total_species} 个物种，{total_images} 张图片")
    print(f"其中 {need_process_species} 个物种需要处理（图片数量少于 {args.min_images} 张）")
    
    balance_dataset(args.root_dir, args.min_images)
    
    print("数据集平衡处理完成!")