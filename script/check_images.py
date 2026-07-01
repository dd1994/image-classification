"""
预扫训练图片，检测并移走损坏文件到 data/bad_img/，保持原目录结构。
边遍历边检查，不预收集全量路径，内存友好。
"""
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# 可通过命令行参数指定数据目录: python check_images.py data/valid
DATA_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "./data/train")
BAD_DIR = DATA_DIR.parent / "bad_img"
EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff'}
WORKERS = 16
CHUNK_SIZE = 2000


def check_one(args):
    """检查单张图片，返回 (path_str, is_bad, error_msg)"""
    path_str, data_root_str, bad_root_str = args
    path = Path(path_str)
    try:
        if path.stat().st_size == 0:
            return (path_str, True, "文件大小为 0")
        with Image.open(path) as img:
            img.verify()
        return (path_str, False, None)
    except UnidentifiedImageError:
        return (path_str, True, "不是有效图片格式")
    except (OSError, IOError, SyntaxError) as e:
        return (path_str, True, str(e))
    except Exception as e:
        return (path_str, True, f"{type(e).__name__}: {e}")


def move_bad(src: Path, data_root: Path, bad_root: Path):
    """把损坏文件移到 bad_img/ 下，保持相对目录结构"""
    rel = src.relative_to(data_root)
    dst = bad_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    try:
        src.parent.rmdir()
    except OSError:
        pass


def iter_tasks():
    """生成器：边遍历文件树边产出任务元组，不攒全量列表"""
    data_root_str = str(DATA_DIR)
    bad_root_str = str(BAD_DIR)
    for dirpath, dirnames, filenames in os.walk(DATA_DIR):
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        for fname in filenames:
            if fname.startswith('.'):
                continue
            fpath = Path(dirpath) / fname
            if fpath.suffix.lower() in EXTS:
                yield (str(fpath), data_root_str, bad_root_str)


def main():
    print(f"数据目录: {DATA_DIR}")
    print(f"损坏文件移至: {BAD_DIR}")
    print(f"使用 {WORKERS} 进程, chunk={CHUNK_SIZE}")
    print()

    print("正在统计文件总数...")
    total = sum(
        1 for dirpath, dirnames, filenames in os.walk(DATA_DIR)
        for fname in filenames
        if not fname.startswith('.')
        and Path(fname).suffix.lower() in EXTS
    )
    print(f"共 {total:,} 张图片待检查\n")

    bad_list = []
    start = time.time()

    print("开始并行检查（边遍历边检查）...")
    with Pool(processes=WORKERS) as pool, \
         tqdm(
            total=total, desc="检查进度", unit="张",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}, {rate_fmt}] 坏文件:{postfix}"
         ) as pbar:
        for path_str, is_bad, err in pool.imap_unordered(
            check_one, iter_tasks(), chunksize=CHUNK_SIZE
        ):
            if is_bad:
                bad_list.append((path_str, err))
                pbar.set_postfix_str(str(len(bad_list)))
            pbar.update(1)

    elapsed = time.time() - start
    print(f"\n检查完成! 耗时 {elapsed/60:.1f} 分钟")
    print(f"发现 {len(bad_list)} 张损坏图片")

    if not bad_list:
        print("没有损坏图片，数据集很干净。")
        return

    print(f"\n正在移走 {len(bad_list)} 张损坏图片...")
    moved = 0
    for path_str, err in bad_list:
        try:
            move_bad(Path(path_str), DATA_DIR, BAD_DIR)
            moved += 1
        except Exception as e:
            print(f"  移动失败: {path_str} -> {e}")

    print(f"成功移动 {moved} 张到 {BAD_DIR}")

    log_path = DATA_DIR.parent / "bad_images_log.txt"
    with open(log_path, 'w', encoding='utf-8') as f:
        for path_str, err in bad_list:
            f.write(f"{path_str}\t{err}\n")
    print(f"坏文件清单已保存到: {log_path}")


if __name__ == '__main__':
    main()
