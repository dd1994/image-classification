"""从 last.ckpt 重建 EVA02Model 并导出为 ONNX（fp32 整模型 + int8 整模型 + 主干 + fp16 头权重）。

产出到 onnx/ 目录：
  - eva02_all_fp32.onnx       整模型，[B,3,448,448] -> logits [B,44269]
  - eva02_all_int8.onnx       动态 INT8 量化版（需 onnxruntime）
  - eva02_all_backbone.onnx   仅主干 -> 归一化 embedding [B,768]（供 hybrid 档）
  - arcface_weight_fp16.npy   归一化后的 ArcFace 头权重 [44269*3, 768] fp16

在训练机（已装 torch/timm/aim/transformers）运行：
    pip install onnx onnxruntime
    python script/export_onnx.py --ckpt last.ckpt --out-dir onnx
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import EVA02Model


class ArcFaceExportWrapper(nn.Module):
    """整模型导出 wrapper：主干 + ArcFace 头（头权重预先 L2 归一为常量 buffer）。

    等价于 EVA02Model.forward 的 ArcFace 路径（get_logits），但把 F.normalize(weight)
    在构造期算好，避免每次推理对大权重做 L2 归一；导出时该 buffer 作为 initializer 常量折叠。
    """
    def __init__(self, model: EVA02Model):
        super().__init__()
        self.model = model
        self.register_buffer(
            'weight_norm',
            F.normalize(model.arcface_loss.weight.data, dim=1).detach(),
        )
        self.num_classes = model.num_classes
        self.sub_center = model.arcface_loss.number_sub_center
        self.arcface_s = model.arcface_loss.s

    def forward(self, x):
        emb = F.normalize(self.model.forward_features(x), dim=1)  # [B, 768]
        cos = F.linear(emb, self.weight_norm)                     # [B, C*sub]
        cos = cos.view(-1, self.num_classes, self.sub_center)
        logits = cos.max(dim=2).values * self.arcface_s           # [B, C]
        return logits


class BackboneExportWrapper(nn.Module):
    """主干导出 wrapper：输出 L2 归一化的 embedding [B, 768]（供 hybrid 档）。"""
    def __init__(self, model: EVA02Model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return F.normalize(self.model.forward_features(x), dim=1)


def build_model(ckpt_path):
    """从 Lightning checkpoint 重建 EVA02Model（超参从 hyper_parameters 动态读取）。"""
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    hp = checkpoint['hyper_parameters']
    model = EVA02Model(
        num_classes=hp.get('num_classes', 44269),
        input_size=hp.get('input_size', 448),
        use_arcface=hp.get('use_arcface', True),
        arcface_s=hp.get('arcface_s', 64.0),
        arcface_m=hp.get('arcface_m', 0.15),
        arcface_sub_center=hp.get('arcface_sub_center', 3),
        arcface_easy_margin=hp.get('arcface_easy_margin', False),
        arcface_ls_eps=hp.get('arcface_ls_eps', 0.0),
        layer_decay_rate=hp.get('layer_decay_rate', 0.85),
        head_lr_mult=hp.get('head_lr_mult', 1.0),
    )
    state_dict = checkpoint['state_dict']
    # ArcFace 路径下 head 是 nn.Identity，这两个 key 不存在，pop 仅为防御
    state_dict.pop('model.head.weight', None)
    state_dict.pop('model.head.bias', None)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def export_onnx(wrapper, input_size, out_path, input_names, output_names):
    dummy = torch.randn(1, 3, input_size, input_size)
    dynamic_axes = {n: {0: 'batch'} for n in input_names + output_names}
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, out_path,
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,        # 17 才引入原生 LayerNormalization 算子，利于 transformer 融合
            do_constant_folding=True,
        )
    print(f'[exported] {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='last.ckpt')
    ap.add_argument('--out-dir', default='onnx')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model = build_model(args.ckpt)
    num_classes = model.num_classes
    input_size = model.hparams.get('input_size', 448)
    print(f'num_classes={num_classes} input_size={input_size} '
          f'arcface_s={model.arcface_loss.s} sub_center={model.arcface_loss.number_sub_center}')

    # 1) fp32 整模型 -> logits [B, num_classes]
    full = ArcFaceExportWrapper(model)
    export_onnx(full, input_size, os.path.join(args.out_dir, 'eva02_all_fp32.onnx'),
                ['image'], ['logits'])

    # 2) 主干 -> 归一化 embedding [B, 768]
    export_onnx(BackboneExportWrapper(model), input_size,
                os.path.join(args.out_dir, 'eva02_all_backbone.onnx'),
                ['image'], ['embedding'])

    # 3) 保存归一化头权重（fp16 存储、fp32 计算用）
    weight_norm = full.weight_norm.detach().cpu().numpy().astype(np.float16)
    np.save(os.path.join(args.out_dir, 'arcface_weight_fp16.npy'), weight_norm)
    print(f'[saved] arcface_weight_fp16.npy shape={weight_norm.shape} dtype=fp16')

    # 4) 动态 INT8 量化 + 5) 校验（需要 onnxruntime）
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print('[skip] 未安装 onnxruntime，跳过 int8 量化与校验。'
              '请 `pip install onnxruntime` 后重跑以生成 eva02_all_int8.onnx。')
        return

    quantize_dynamic(
        os.path.join(args.out_dir, 'eva02_all_fp32.onnx'),
        os.path.join(args.out_dir, 'eva02_all_int8.onnx'),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    print('[exported] eva02_all_int8.onnx (dynamic int8, per_channel)')

    # 校验：onnx(fp32) vs PyTorch 前向，确认导出无损
    dummy = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        torch_logits = model(dummy).numpy()
    sess = ort.InferenceSession(os.path.join(args.out_dir, 'eva02_all_fp32.onnx'),
                                providers=['CPUExecutionProvider'])
    onnx_logits = sess.run(None, {'image': dummy.numpy()})[0]
    max_err = float(np.abs(torch_logits - onnx_logits).max())
    print(f'[verify] onnx(fp32) vs torch max|Δlogits| = {max_err:.6f}')
    if max_err >= 1e-2:
        raise SystemExit('ONNX 导出误差过大，请检查 opset / SDPA 分解等。')
    print('done.')


if __name__ == '__main__':
    main()
