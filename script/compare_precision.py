"""在验证集上对比 fp32 / int8 / hybrid 三档 ONNX 的 top-1 / top-3 识别率。

在训练机运行（需 torch/torchvision + onnxruntime；验证集 data/valid 与 train_map_enriched.csv
都在训练机上）。预处理复用与训练一致的 torchvision v2 变换，保证对比公平。

用法：
  python script/compare_precision.py --data-dir ./data/valid --map train_map_enriched.csv
  快速子集：加 --limit 5000
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


def cosine_logits(emb, W, num_classes, sub, s, chunk=4096):
    logits = np.empty(num_classes, dtype=np.float32)
    for start in range(0, num_classes, chunk):
        end = min(start + chunk, num_classes)
        w = W[start * sub:end * sub].astype(np.float32)
        logits[start:end] = (w @ emb).reshape(-1, sub).max(axis=1) * s
    return logits


def load_model_index_map(csv_path):
    """返回 {species_id: model_index}，把验证集图片的 species_id 映射到模型类别序号(0..44268)。"""
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
    ap.add_argument('--num-classes', type=int, default=44269)
    ap.add_argument('--arcface-s', type=float, default=64.0)
    ap.add_argument('--sub-center', type=int, default=3)
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

    fp32 = make_session(os.path.join(args.onnx_dir, 'eva02_all_fp32.onnx'), args.threads)
    int8 = make_session(os.path.join(args.onnx_dir, 'eva02_all_int8.onnx'), args.threads)
    backbone = make_session(os.path.join(args.onnx_dir, 'eva02_all_backbone.onnx'), args.threads)
    W = np.load(os.path.join(args.onnx_dir, 'arcface_weight_fp16.npy'))

    fp32_in = fp32.get_inputs()[0].name
    int8_in = int8.get_inputs()[0].name
    bb_in = backbone.get_inputs()[0].name

    stats = {m: {'top1': 0, 'top3': 0, 'agree': 0, 'time': 0.0}
             for m in ('fp32', 'int8', 'hybrid')}
    total = 0
    skipped = 0
    examples = []

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

        t0 = time.time()
        fp32_logits = fp32.run(None, {fp32_in: x})[0]
        stats['fp32']['time'] += time.time() - t0
        t0 = time.time()
        int8_logits = int8.run(None, {int8_in: x})[0]
        stats['int8']['time'] += time.time() - t0
        t0 = time.time()
        emb = backbone.run(None, {bb_in: x})[0]
        stats['hybrid']['time'] += time.time() - t0

        for b in range(images.shape[0]):
            label = labels[b]
            if label < 0:
                skipped += 1
                continue
            total += 1

            fp32_top1 = int(np.argmax(fp32_logits[b]))
            fp32_top3 = set(np.argsort(fp32_logits[b])[::-1][:3].tolist())
            int8_top1 = int(np.argmax(int8_logits[b]))
            int8_top3 = set(np.argsort(int8_logits[b])[::-1][:3].tolist())
            h_logits = cosine_logits(emb[b], W, args.num_classes, args.sub_center, args.arcface_s)
            h_top1 = int(np.argmax(h_logits))
            h_top3 = set(np.argsort(h_logits)[::-1][:3].tolist())

            stats['fp32']['top1'] += int(fp32_top1 == label)
            stats['fp32']['top3'] += int(label in fp32_top3)
            stats['fp32']['agree'] += 1  # 自身恒为 1

            stats['int8']['top1'] += int(int8_top1 == label)
            stats['int8']['top3'] += int(label in int8_top3)
            stats['int8']['agree'] += int(int8_top1 == fp32_top1)

            stats['hybrid']['top1'] += int(h_top1 == label)
            stats['hybrid']['top3'] += int(label in h_top3)
            stats['hybrid']['agree'] += int(h_top1 == fp32_top1)

            if len(examples) < 10 and (int8_top1 != fp32_top1 or h_top1 != fp32_top1):
                examples.append((label, fp32_top1, int8_top1, h_top1))

    n = max(total, 1)
    print(f'\n===== 结果 total={total} skipped={skipped} threads={args.threads} =====')
    print(f'{"method":<8}{"top1":>10}{"top3":>10}{"top1==fp32":>12}{"ms/img":>10}')
    for m in ('fp32', 'int8', 'hybrid'):
        s = stats[m]
        print(f'{m:<8}{s["top1"] / n:>10.4f}{s["top3"] / n:>10.4f}'
              f'{s["agree"] / n:>12.4f}{s["time"] / n * 1000:>10.1f}')

    if examples:
        print('\n前若干 top-1 不一致样本 (label, fp32, int8, hybrid):')
        for e in examples:
            print('  ', e)


if __name__ == '__main__':
    main()
