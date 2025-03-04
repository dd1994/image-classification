import os
from PIL import Image, ImageDraw
from PIL.Image import Resampling
from tqdm import tqdm


def process_images(root_dir):
    file_list = []
    # 遍历目录并收集未处理的图片
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                if not filename.startswith('processed_'):
                    file_path = os.path.join(dirpath, filename)
                    file_list.append(file_path)

    # 按修改时间排序（最旧优先）
    file_list.sort(key=lambda x: os.path.getmtime(x))

    # 处理每个文件
    for file_path in tqdm(file_list, desc='Processing images'):
        try:
            with Image.open(file_path) as img:
                # 创建绘图对象
                # draw = ImageDraw.Draw(img)
                #
                # # 去除左上水印（坐标系统原点在左上角）
                # left_watermark = [94, 0, 94+120, 135]  # left, top, right, bottom
                # draw.rectangle(left_watermark, fill=(0, 0, 0))  # 用黑色填充
                #
                # # 去除右下水印
                # width, height = img.size
                # right_watermark = [
                #     width - 360,  # left
                #     height - 50,  # top
                #     width,        # right
                #     height        # bottom
                # ]
                # draw.rectangle(right_watermark, fill=(0, 0, 0))

                # 计算目标尺寸（保持宽高比，最长边=800px）
                width, height = img.size
                max_dim = max(width, height)

                # 计算缩放比例（无论放大缩小都应用）
                scale = 800 / max_dim
                new_size = (int(width * scale), int(height * scale))

                # 应用高质量缩放
                img = img.resize(new_size, Resampling.LANCZOS)

                # 处理特殊颜色模式
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')

                # 生成新文件名
                dir_name, file_name = os.path.split(file_path)
                base_name = os.path.splitext(file_name)[0]
                new_file_name = f"processed_{base_name}.jpg"
                new_file_path = os.path.join(dir_name, new_file_name)

                # 保存压缩图片
                img.save(new_file_path, "JPEG", quality=90, optimize=True)

                # 删除原文件（仅在保存成功后执行）
                os.remove(file_path)

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            # 保留原文件以便排查问题


if __name__ == "__main__":
    root_dir = r"D:\image-classification\data\dup_test2"
    process_images(root_dir)

