"""CPU 推理示例：用 onnxruntime 跑 EVA02 ONNX 模型，输出 top-K 预测。

只依赖 onnxruntime + numpy + Pillow（不需要 torch），适合 4GB 双核 CPU 机器。

三档精度（--precision）：
  fp32   整模型 fp32（~750MB）
  int8   整模型动态 INT8（~200MB）
  hybrid 主干(int8/fp32 均可用) + ArcFace 头权重 fp16 分块余弦（~290MB，推荐）

用法：
  python script/infer_onnx.py --image <图片路径> --precision hybrid --map train_map_enriched.csv

注意：--num-classes / --arcface-s / --sub-center 需与导出时的模型一致
（默认 44269 / 64 / 3，即 config/large/eva02_all_part2.json）。
"""
import argparse
import csv
import os
import time

import numpy as np
from PIL import Image

import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def make_session(model_path, threads=2):
    """构建针对 transformer(ViT) 优化的 ONNX Runtime 会话。"""
    so = ort.SessionOptions()
    # 图优化全开：融合 LayerNorm / GELU / Attention / SkipLayerNorm / MatMul+Add
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads       # 匹配物理核数（双核=2），不超订阅
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # 单图低延迟
    so.enable_cpu_mem_arena = True          # 复用激活缓冲，降低内存峰值
    return ort.InferenceSession(model_path, sess_options=so,
                                providers=['CPUExecutionProvider'])


def preprocess(img_path, input_size=448):
    """复刻训练/预测的 val_test 变换：ToRGB -> Resize(短边 537) -> CenterCrop(448) -> Normalize。"""
    img = Image.open(img_path).convert('RGB')          # ToRGB
    w, h = img.size
    scale = int(input_size * 1.2) / min(w, h)           # 短边缩到 537，保持宽高比
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    left = (nw - input_size) // 2
    top = (nh - input_size) // 2
    img = img.crop((left, top, left + input_size, top + input_size))  # 中心裁剪 448
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = x.transpose(2, 0, 1)                           # [C,H,W]
    x = (x - MEAN[:, None, None]) / STD[:, None, None]
    return x[None].astype(np.float32)                  # [1,C,H,W]


def cosine_logits(emb, W, num_classes, sub=3, s=64.0, chunk=4096):
    """hybrid 档：归一化 embedding 与 fp16 头权重分块做余弦，返回完整 logits。

    emb: [768] float32 已归一；W: [num_classes*sub, 768] float16 已归一。
    分块避免把 408MB 的 fp32 权重一次性载入内存。
    """
    logits = np.empty(num_classes, dtype=np.float32)
    for start in range(0, num_classes, chunk):
        end = min(start + chunk, num_classes)
        w = W[start * sub:end * sub].astype(np.float32)   # [chunk*sub, 768]
        logits[start:end] = (w @ emb).reshape(-1, sub).max(axis=1) * s
    return logits


def softmax(logits):
    logits = logits - logits.max()
    e = np.exp(logits)
    return e / e.sum()


def topk_from_logits(logits, k):
    probs = softmax(logits)
    idx = probs.argsort()[::-1][:k]
    return [(int(i), float(probs[i])) for i in idx]


def load_mapping(csv_path):
    """通用 CSV 加载：按表头定位 Index/SpeciesID 与名称列，返回 {index: display_name}。

    兼容 2 列(Index,SpeciesID)、4 列(Index,SpeciesID,taxonName,chineseName) 及 enriched 多列。
    """
    mapping = {}
    name_col_names = {'taxonname', 'chinesename', 'vernacularname', 'name'}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        lower = [name.strip().lower() for name in header]
        idx_col = lower.index('index') if 'index' in lower else 0
        sid_col = lower.index('speciesid') if 'speciesid' in lower else 1
        name_cols = [i for i, name in enumerate(lower) if name in name_col_names]
        for row in reader:
            if not row or len(row) <= max(idx_col, sid_col):
                continue
            try:
                index = int(row[idx_col])
            except ValueError:
                continue
            parts = [row[i].strip() for i in name_cols if i < len(row) and row[i].strip()]
            mapping[index] = ' '.join(parts) if parts else f"SpeciesID={row[sid_col].strip()}"
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--onnx-dir', default='onnx')
    ap.add_argument('--precision', choices=['fp32', 'int8', 'hybrid'], default='hybrid')
    ap.add_argument('--map', default='train_map_enriched.csv', help='物种映射 CSV（可缺省）')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--input-size', type=int, default=448)
    ap.add_argument('--num-classes', type=int, default=44269)
    ap.add_argument('--arcface-s', type=float, default=64.0)
    ap.add_argument('--sub-center', type=int, default=3)
    ap.add_argument('--topk', type=int, default=3)
    args = ap.parse_args()

    x = preprocess(args.image, args.input_size)

    t0 = time.time()
    if args.precision in ('fp32', 'int8'):
        model_path = os.path.join(args.onnx_dir, f'eva02_all_{args.precision}.onnx')
        session = make_session(model_path, args.threads)
        logits = session.run(None, {session.get_inputs()[0].name: x})[0][0]
        results = topk_from_logits(logits, args.topk)
    else:  # hybrid
        bb = make_session(os.path.join(args.onnx_dir, 'eva02_all_backbone.onnx'), args.threads)
        emb = bb.run(None, {bb.get_inputs()[0].name: x})[0][0]  # [768] 已归一
        W = np.load(os.path.join(args.onnx_dir, 'arcface_weight_fp16.npy'))
        logits = cosine_logits(emb, W, args.num_classes, args.sub_center, args.arcface_s)
        results = topk_from_logits(logits, args.topk)
    elapsed = (time.time() - t0) * 1000

    mapping = load_mapping(args.map) if os.path.exists(args.map) else {}
    print(f'[{args.precision}] 推理耗时 {elapsed:.0f} ms')
    for i, (cls, prob) in enumerate(results, 1):
        name = mapping.get(cls, f'class_{cls}')
        print(f'  {i}. {name}  ({prob * 100:.2f}%)')


if __name__ == '__main__':
    main()
