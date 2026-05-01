import os
from collections import defaultdict
from tqdm import tqdm


def find_duplicate_files_by_name(folder_path):
    """递归查找文件夹中的所有重复文件（按文件名）"""
    file_names = defaultdict(list)
    total_files = sum([len(files) for root, dirs, files in os.walk(folder_path)])

    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_names[file].append(file_path)
                pbar.update(1)

    # 找出所有具有相同文件名的文件
    duplicates = {name: paths for name, paths in file_names.items() if len(paths) > 1}
    return duplicates


def main():
    folder_path = input("请输入要检查的文件夹路径: ")
    duplicates = find_duplicate_files_by_name(folder_path)

    if not duplicates:
        print("没有找到重复的文件。")
    else:
        print("找到以下重复文件:")
        for name, paths in duplicates.items():
            print(f"文件名: {name}")
            for path in paths:
                print(f"  - {path}")


if __name__ == "__main__":

    main()

# {
#     "class_path": "StochasticWeightAveraging",
#     "init_args": {
#         "swa_lrs": 5e-6,
#         "swa_epoch_start": 7
#     }
# }