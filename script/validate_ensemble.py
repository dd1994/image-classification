"""
Validate a trained checkpoint with Baseline vs Grad-CAM Crop Ensemble.

Usage:
    python script/validate_ensemble.py --config <training_config.json> --checkpoint <path.ckpt>

Reads data params (valid_dir, batch_size, etc.) from the training config file.
"""
import os
import sys
import json
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SwinV2Model
from util.transform import ToRGBTransform
from dataSet.SpecialCateDataset import SpecialCateDataset

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CAM_THRESHOLD = 0.3
MIN_AREA_RATIO = 0.3
MAX_AREA_RATIO = 0.7


def load_model(checkpoint_path, model_cfg, device):
    """Load model from checkpoint, falling back to model_cfg for missing hyper_parameters."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    hp = checkpoint['hyper_parameters']

    def get_param(key):
        return hp.get(key, model_cfg.get(key))

    input_size = get_param('input_size')

    model = SwinV2Model(
        num_classes=get_param('num_classes'),
        input_size=input_size,
        learning_rate=get_param('learning_rate'),
        use_arcface=get_param('use_arcface'),
        arcface_s=get_param('arcface_s'),
        arcface_m=get_param('arcface_m'),
        arcface_sub_center=get_param('arcface_sub_center'),
        arcface_easy_margin=get_param('arcface_easy_margin'),
        arcface_ls_eps=get_param('arcface_ls_eps'),
        use_gradient_checkpointing=get_param('use_gradient_checkpointing'),
    )
    state_dict = checkpoint['state_dict']
    state_dict.pop('model.head.fc.weight', None)
    state_dict.pop('model.head.fc.bias', None)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def build_unnorm_transform(input_size):
    return transforms.Compose([
        ToRGBTransform(),
        transforms.Resize(int(input_size * 1.2)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])


def normalize_tensor(t):
    return transforms.Normalize(MEAN, STD)(t)


def compute_gradcam(model, img_batch, target_class):
    feature_maps = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal feature_maps
        feature_maps = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    target_layer = model.model.norm
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    logits = model(img_batch)
    target_logit = logits[0, target_class]

    model.zero_grad()
    target_logit.backward()

    fwd_handle.remove()
    bwd_handle.remove()

    fm = feature_maps.detach()
    grad = gradients.detach()

    alpha = grad.mean(dim=(1, 2), keepdim=True)
    cam = (alpha * fm).sum(dim=3)
    cam = torch.relu(cam)

    h_img, w_img = img_batch.shape[2], img_batch.shape[3]
    cam_4d = cam.unsqueeze(1)
    cam_upsampled = F.interpolate(cam_4d, size=(h_img, w_img),
                                   mode='bilinear', align_corners=False)
    cam_upsampled = cam_upsampled.squeeze()

    cam_min = cam_upsampled.min()
    cam_max = cam_upsampled.max()
    if cam_max > cam_min:
        cam_norm = (cam_upsampled - cam_min) / (cam_max - cam_min)
    else:
        cam_norm = torch.zeros_like(cam_upsampled)

    return cam_norm.cpu().numpy()


def get_bbox_ratio(cam_heatmap, threshold=CAM_THRESHOLD):
    """Return bbox area ratio [0, 1] from Grad-CAM heatmap. -1 if no activation."""
    binary_mask = (cam_heatmap > threshold).astype(np.uint8)
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not rows.any() or not cols.any():
        return -1.0
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    bbox_area = (y_max - y_min) * (x_max - x_min)
    img_area = cam_heatmap.shape[0] * cam_heatmap.shape[1]
    return bbox_area / img_area


def extract_crop(img_tensor_unnorm, cam_heatmap, input_size,
                  threshold=CAM_THRESHOLD, min_area_ratio=MIN_AREA_RATIO, bbox_expand=0.15):
    binary_mask = (cam_heatmap > threshold).astype(np.uint8)
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)

    h_img, w_img = cam_heatmap.shape
    if not rows.any() or not cols.any():
        y1, y2, x1, x2 = 0, h_img, 0, w_img
    else:
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        area = (y_max - y_min) * (x_max - x_min)
        if area < min_area_ratio * h_img * w_img:
            y1, y2, x1, x2 = 0, h_img, 0, w_img
        else:
            h, w = y_max - y_min, x_max - x_min
            dy, dx = int(h * bbox_expand), int(w * bbox_expand)
            y1 = max(0, y_min - dy)
            y2 = min(h_img, y_max + dy)
            x1 = max(0, x_min - dx)
            x2 = min(w_img, x_max + dx)

    crop = img_tensor_unnorm[:, y1:y2, x1:x2]
    crop_4d = crop.unsqueeze(0)
    crop_resized = F.interpolate(crop_4d, size=(input_size, input_size),
                                  mode='bilinear', align_corners=False)
    return normalize_tensor(crop_resized.squeeze(0))


def main():
    parser = argparse.ArgumentParser(description="Validate with Grad-CAM ensemble")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training JSON config (reads data params from it)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to .ckpt checkpoint')
    parser.add_argument('--batch-size', type=int, default=80,
                        help='Override batch size (default: from config)')
    parser.add_argument('--max-area-ratio', type=float, default=MAX_AREA_RATIO,
                        help=f'Skip crop if bbox area exceeds this ratio (default: {MAX_AREA_RATIO})')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    data_cfg = cfg['data']['init_args']
    model_cfg = cfg['model']['init_args']
    checkpoint_path = args.checkpoint
    data_dir = data_cfg['valid_dir']
    id_map = data_cfg['valid_id_map_file_path']
    batch_size = args.batch_size
    max_area_ratio = args.max_area_ratio
    num_workers = data_cfg['num_workers']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {checkpoint_path}")

    model = load_model(checkpoint_path, model_cfg, device)
    input_size = model.hparams.input_size
    num_classes = model.hparams.num_classes
    print(f"Model: input_size={input_size}, num_classes={num_classes}")

    transform_unnorm = build_unnorm_transform(input_size)
    dataset = SpecialCateDataset(
        root_dir=data_dir,
        transform=transform_unnorm,
        id_map_file_path=id_map,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    print(f"Validation images: {len(dataset)}")

    total = 0
    baseline_top1 = 0
    baseline_top3 = 0
    crop_top1 = 0
    crop_top3 = 0
    ensemble_avg_top1 = 0
    ensemble_avg_top3 = 0
    ensemble_max_top1 = 0
    ensemble_max_top3 = 0
    skip_large_count = 0   # bbox > max_area_ratio
    skip_small_count = 0   # bbox < min_area_ratio or no activation

    for img_unnorm, labels in tqdm(loader, desc="Validating"):
        bs = img_unnorm.size(0)
        img_norm = normalize_tensor(img_unnorm).to(device)
        labels = labels.to(device)

        # --- Baseline (batched) ---
        with torch.no_grad():
            baseline_logits = model(img_norm)

        baseline_preds = baseline_logits.argmax(dim=1)
        baseline_top3_preds = baseline_logits.topk(3, dim=1).indices

        # --- Per-image Grad-CAM + Crop ---
        crop_logits_list = []
        for i in range(bs):
            single_norm = img_norm[i:i+1]
            single_unnorm = img_unnorm[i]

            pred = baseline_preds[i].item()
            cam = compute_gradcam(model, single_norm, pred)

            # Skip crop if: no activation / fills most of frame / too small (would fallback)
            bbox_ratio = get_bbox_ratio(cam)
            if bbox_ratio < 0 or bbox_ratio < MIN_AREA_RATIO:
                crop_logits_list.append(baseline_logits[i:i+1])
                skip_small_count += 1
                continue
            if bbox_ratio > max_area_ratio:
                crop_logits_list.append(baseline_logits[i:i+1])
                skip_large_count += 1
                continue

            crop_norm = extract_crop(single_unnorm, cam, input_size)
            crop_batch = crop_norm.unsqueeze(0).to(device)

            with torch.no_grad():
                cl = model(crop_batch)
            crop_logits_list.append(cl)

        crop_logits = torch.cat(crop_logits_list, dim=0)
        ensemble_avg_logits = (baseline_logits + crop_logits) / 2.0
        ensemble_max_logits = torch.max(baseline_logits, crop_logits)

        crop_preds = crop_logits.argmax(dim=1)
        ensemble_avg_preds = ensemble_avg_logits.argmax(dim=1)
        ensemble_max_preds = ensemble_max_logits.argmax(dim=1)

        crop_top3_batch = crop_logits.topk(3, dim=1).indices
        ensemble_avg_top3_batch = ensemble_avg_logits.topk(3, dim=1).indices
        ensemble_max_top3_batch = ensemble_max_logits.topk(3, dim=1).indices

        for i in range(bs):
            y = labels[i].item()
            total += 1
            baseline_top1 += int(baseline_preds[i].item() == y)
            baseline_top3 += int(y in baseline_top3_preds[i])
            crop_top1 += int(crop_preds[i].item() == y)
            crop_top3 += int(y in crop_top3_batch[i])
            ensemble_avg_top1 += int(ensemble_avg_preds[i].item() == y)
            ensemble_avg_top3 += int(y in ensemble_avg_top3_batch[i])
            ensemble_max_top1 += int(ensemble_max_preds[i].item() == y)
            ensemble_max_top3 += int(y in ensemble_max_top3_batch[i])

    bl_top1 = float(baseline_top1) / total * 100
    bl_top3 = float(baseline_top3) / total * 100
    cp_top1 = float(crop_top1) / total * 100
    cp_top3 = float(crop_top3) / total * 100
    ea_top1 = float(ensemble_avg_top1) / total * 100
    ea_top3 = float(ensemble_avg_top3) / total * 100
    em_top1 = float(ensemble_max_top1) / total * 100
    em_top3 = float(ensemble_max_top3) / total * 100

    print(f"\n{'='*65}")
    print(f"Total validation images: {total}")
    print(f"Crop skipped (bbox > {max_area_ratio:.0%}): {skip_large_count}")
    print(f"Crop skipped (bbox < {MIN_AREA_RATIO:.0%} or no activation): {skip_small_count}")
    print(f"{'='*65}")
    print(f"{'Strategy':<20} {'Top-1':>10} {'Top-3':>10}")
    print(f"{'-'*45}")
    print(f"{'Baseline':<20} {bl_top1:>9.2f}% {bl_top3:>9.2f}%")
    print(f"{'Crop-Only':<20} {cp_top1:>9.2f}% {cp_top3:>9.2f}%")
    print(f"{'Ensemble Avg':<20} {ea_top1:>9.2f}% {ea_top3:>9.2f}%")
    print(f"{'Ensemble Max':<20} {em_top1:>9.2f}% {em_top3:>9.2f}%")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
