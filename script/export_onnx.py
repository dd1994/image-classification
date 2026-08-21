"""从 last.ckpt 直接重建 EVA02 主干 + ArcFace 头，导出纯 fp32 ONNX（无 transformer 优化）。

不依赖 script/model.py（其顶部 `from aim.v2.utils import ...` / `from transformers import ...`
会拖入训练机的额外依赖），改用 timm 直接重建主干。前向路径与 EVA02Model 完全一致：

    f   = backbone.forward_features(x)
    emb = backbone.forward_head(f, pre_logits=True)          # fc_norm 后、fc 前的 embedding
    logits = F.normalize(emb) @ F.normalize(weight).T        # ArcFace cosine（weight=[C*sub, 768]）
              -> view(C, sub) -> max -> *s

只导纯 `ai.onnx`（opset 17），不含 com.microsoft 算子。曾用 onnxruntime.transformers 做融合优化，
会引入 SkipLayerNormalization 等 com.microsoft 算子导致 OpenVINO 等工具无法加载，已弃用（见 CLAUDE.md）。

产物：onnx/eva02_all_fp32.onnx（整模型 → logits [B, num_classes]，纯 fp32）

运行（需 torch + timm，用 /usr/bin/python3）：
    /usr/bin/python3 script/export_onnx.py --ckpt last.ckpt --out-dir onnx
"""
import argparse
import os

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
    del ck, sd
    return {
        'backbone': backbone, 'weight_norm': weight_norm, 'num_classes': num_classes,
        'input_size': input_size, 'sub_center': sub_center, 'arcface_s': arcface_s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='last.ckpt')
    ap.add_argument('--out-dir', default='onnx')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    m = build_from_checkpoint(args.ckpt)
    input_size = m['input_size']
    print(f"num_classes={m['num_classes']} input_size={input_size} arcface_s={m['arcface_s']} "
          f"sub_center={m['sub_center']}")

    full = ArcFaceExportWrapper(m['backbone'], m['weight_norm'], m['num_classes'],
                                m['sub_center'], m['arcface_s'])
    dummy = torch.randn(1, 3, input_size, input_size)
    out_path = os.path.join(args.out_dir, 'eva02_all_fp32.onnx')
    with torch.no_grad():
        torch.onnx.export(full, dummy, out_path,
                          input_names=['image'], output_names=['logits'],
                          dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}},
                          opset_version=17, do_constant_folding=True)
    print(f'[exported] {out_path} (纯 fp32, 无 transformer 优化)')
    print('done.')


if __name__ == '__main__':
    main()
