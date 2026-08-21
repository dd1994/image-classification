"""在验证集上评估 fp32 ONNX 的 top-1 / top-3 识别率（校验导出无损）。

在训练机运行（需 torch/torchvision + onnxruntime；验证集 data/valid 与 train_map_enriched.csv 都在训练机）。
预处理复用与训练一致的 torchvision v2 变换。

用法：
  python script/compare_precision.py --data-dir ./data/valid --map train_map_enriched.csv
  快速子集：--limit 5000
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from dataSet.SpecialCateDataset import SpecialCateDataset
from util.transform import ToRGBTransform

import onnxruntime as ort


def make_session(model_path, threads):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_cpu_mem_arena = True
    return ort.InferenceSession(model_path, sess_options=so,
                                providers=['CPUExecutionProvider'])


def load_model_index_map(csv_path):
    """返回 {species_id: model_index}，把验证集图片的 species_id 映射到模型类别序号(0..num_classes-1)。"""
    mapping = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        lower = [name.strip().lower() for name in header]
        idx_col = lower.index('index')
        sid_col = lower.index('speciesid')
        for row in reader:
            if not row or len(row) <= max(idx_col, sid_col):
                continue
            try:
                mapping[row[sid_col].strip()] = int(row[idx_col])
            except ValueError:
                continue
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='./data/valid')
    ap.add_argument('--map', default='train_map_enriched.csv')
    ap.add_argument('--onnx-dir', default='onnx')
    ap.add_argument('--input-size', type=int, default=448)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0, help='只测前 N 张；0=全部')
    args = ap.parse_args()

    model_index = load_model_index_map(args.map)
    print(f'[map] 加载 {len(model_index)} 个 species_id -> model_index')

    # 与训练 val_test 完全一致的变换
    transform = v2.Compose([
        ToRGBTransform(),
        v2.ToImage(),
        v2.Resize(int(args.input_size * 1.2)),
        v2.CenterCrop(args.input_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = SpecialCateDataset(root_dir=args.data_dir, id_map_file_path='_valid_map_tmp.csv',
                            transform=transform)
    print(f'[data] 验证集 {len(ds)} 张图片')

    session = make_session(os.path.join(args.onnx_dir, 'eva02_all_fp32.onnx'), args.threads)
    in_name = session.get_inputs()[0].name

    top1 = top3 = total = skipped = 0
    t0 = time.time()

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    for images, valid_dir_index in dl:
        if args.limit and total >= args.limit:
            break
        # valid 集自身序号 -> species_id -> 模型类别序号（关键：不能直接用 valid_dir_index）
        labels = []
        for vi in valid_dir_index.tolist():
            species_id = ds.index_to_species_id[int(vi)]
            labels.append(model_index.get(species_id, -1))

        x = images.numpy().astype(np.float32)  # [B,3,H,W]
        logits = session.run(None, {in_name: x})[0]

        for b in range(images.shape[0]):
            label = labels[b]
            if label < 0:
                skipped += 1
                continue
            total += 1
            top1 += int(np.argmax(logits[b]) == label)
            top3 += int(label in set(np.argsort(logits[b])[::-1][:3].tolist()))

    elapsed = time.time() - t0
    n = max(total, 1)
    print(f'\n===== 结果 total={total} skipped={skipped} threads={args.threads} =====')
    print(f'top1 = {top1 / n:.4f}  ({top1}/{total})')
    print(f'top3 = {top3 / n:.4f}  ({top3}/{total})')
    print(f'平均 {elapsed / n * 1000:.1f} ms/图')


if __name__ == '__main__':
    main()
