"""从 last.ckpt 直接重建 EVA02 主干 + ArcFace 头，导出整模型 ONNX（fp32）。

不依赖 script/model.py（其顶部 `from aim.v2.utils import ...` / `from transformers import ...`
会拖入训练机的额外依赖），改用 timm 直接重建主干。前向路径与 EVA02Model 完全一致：

    f   = backbone.forward_features(x)
    emb = backbone.forward_head(f, pre_logits=True)          # fc_norm 后、fc 前的 embedding
    logits = F.normalize(emb) @ F.normalize(weight).T        # ArcFace cosine（weight=[C*sub, 768]）
              -> view(C, sub) -> max -> *s

并内置 transformer 专用优化（onnxruntime.transformers.optimizer，model_type='vit'），
优化后与 torch 参考前向做数值校验，失败自动回退原始导出。

产物：onnx/eva02_all_fp32.onnx（整模型 → logits [B, num_classes]）

运行（需 torch + timm + onnxruntime）：
    python script/export_onnx.py --ckpt last.ckpt --out-dir onnx
"""
import argparse
import os
import tempfile

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = 'eva02_base_patch14_448.mim_in22k_ft_in22k'


class ArcFaceExportWrapper(nn.Module):
    """主干 + ArcFace 头（头权重预归一为常量 buffer）。"""
    def __init__(self, backbone, weight_norm, num_classes, sub_center, arcface_s):
        super().__init__()
        self.backbone = backbone
        self.register_buffer('weight_norm', weight_norm)   # [C*sub, 768] 已归一
        self.num_classes = num_classes
        self.sub_center = sub_center
        self.arcface_s = arcface_s

    def forward(self, x):
        f = self.backbone.forward_features(x)
        emb = F.normalize(self.backbone.forward_head(f, pre_logits=True), dim=1)
        cos = F.linear(emb, self.weight_norm)
        cos = cos.view(-1, self.num_classes, self.sub_center)
        return cos.max(dim=2).values * self.arcface_s


def build_from_checkpoint(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    hp = ck['hyper_parameters']
    num_classes = hp.get('num_classes', 44269)
    input_size = hp.get('input_size', 448)
    sub_center = hp.get('arcface_sub_center', 3)
    arcface_s = hp.get('arcface_s', 64.0)
    sd = ck['state_dict']

    backbone = timm.create_model(MODEL_NAME, pretrained=False, img_size=input_size, num_classes=0)
    backbone_sd = {k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')}
    missing, unexpected = backbone.load_state_dict(backbone_sd, strict=False)
    if missing or unexpected:
        print(f'[warn] backbone key 不匹配: missing={missing} unexpected={unexpected}')
    backbone.eval()

    weight_norm = F.normalize(sd['arcface_loss.weight'].float(), dim=1).detach()
    num_heads = backbone.blocks[0].attn.num_heads
    hidden_size = backbone.num_features
    del ck, sd
    return {
        'backbone': backbone, 'weight_norm': weight_norm, 'num_classes': num_classes,
        'input_size': input_size, 'sub_center': sub_center, 'arcface_s': arcface_s,
        'num_heads': num_heads, 'hidden_size': hidden_size,
    }


def export_raw(wrapper, input_size, out_path, input_names, output_names):
    dummy = torch.randn(1, 3, input_size, input_size)
    dynamic_axes = {n: {0: 'batch'} for n in input_names + output_names}
    with torch.no_grad():
        torch.onnx.export(wrapper, dummy, out_path,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes, opset_version=17,
                          do_constant_folding=True)
    return dummy


def transformer_optimize(src_path, dst_path, num_heads, hidden_size):
    """onnxruntime.transformers 的 transformer 专用图融合。"""
    from onnxruntime.transformers.optimizer import optimize_model
    opt = optimize_model(src_path, model_type='vit', num_heads=num_heads,
                         hidden_size=hidden_size, opt_level=1, use_gpu=False)
    opt.save_model_to_file(dst_path)


def run_logits(onnx_path, dummy, input_name='image'):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    return sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})[0]


def max_abs_err(a, b):
    return float(np.abs(a - b).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='last.ckpt')
    ap.add_argument('--out-dir', default='onnx')
    ap.add_argument('--no-tf-opt', action='store_true',
                    help='跳过 onnxruntime.transformers 专用优化')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    m = build_from_checkpoint(args.ckpt)
    input_size = m['input_size']
    print(f"num_classes={m['num_classes']} input_size={input_size} arcface_s={m['arcface_s']} "
          f"sub_center={m['sub_center']} num_heads={m['num_heads']} hidden={m['hidden_size']}")

    full = ArcFaceExportWrapper(m['backbone'], m['weight_norm'], m['num_classes'],
                                m['sub_center'], m['arcface_s'])
    dummy = torch.randn(1, 3, input_size, input_size)
    with torch.no_grad():
        torch_logits = full(dummy).numpy()

    fp32_path = os.path.join(args.out_dir, 'eva02_all_fp32.onnx')

    with tempfile.TemporaryDirectory() as tmp:
        raw_fp32 = os.path.join(tmp, 'fp32_raw.onnx')
        export_raw(full, input_size, raw_fp32, ['image'], ['logits'])
        if args.no_tf_opt:
            os.replace(raw_fp32, fp32_path)
            print(f'[exported] {fp32_path} (raw, 未做 transformer 优化)')
        else:
            try:
                transformer_optimize(raw_fp32, fp32_path, m['num_heads'], m['hidden_size'])
                err = max_abs_err(run_logits(fp32_path, dummy), torch_logits)
                print(f'[verify] tf-optimized vs torch max|Δlogits| = {err:.6f}')
                if err < 1e-2:
                    print(f'[exported] {fp32_path} (transformer 优化)')
                else:
                    print(f'[warn] transformer 优化后误差 {err:.4f} 超阈，回退原始导出')
                    os.replace(raw_fp32, fp32_path)
            except Exception as e:
                print(f'[warn] transformer 优化失败({e})，回退原始导出')
                os.replace(raw_fp32, fp32_path)

    print('done.')


if __name__ == '__main__':
    main()
