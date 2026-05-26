import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from model import SwinV2Model
from util.transform import ToRGBTransform
import csv


def load_index_to_species_id_from_csv(csv_file_path):
    index_to_species_id = {}
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            index, species_id, taxon_name, chinese_name = row
            index_to_species_id[int(index)] = taxon_name + ' ' + chinese_name
    return index_to_species_id


def compute_gradcam(model, img_tensor, target_class):
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

    logits = model(img_tensor)
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

    input_size = img_tensor.shape[2]
    cam_4d = cam.unsqueeze(1)
    cam_upsampled = F.interpolate(cam_4d, size=(input_size, input_size),
                                   mode='bilinear', align_corners=False)
    cam_upsampled = cam_upsampled.squeeze()

    cam_min = cam_upsampled.min()
    cam_max = cam_upsampled.max()
    if cam_max > cam_min:
        cam_norm = (cam_upsampled - cam_min) / (cam_max - cam_min)
    else:
        cam_norm = torch.zeros_like(cam_upsampled)

    return cam_norm.cpu().numpy()


def extract_crop(img_tensor_unnorm, cam_heatmap, normalize_fn, input_size,
                  threshold=0.3, min_area_ratio=0.25, bbox_expand=0.15):
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
    return normalize_fn(crop_resized.squeeze(0))


def print_topk(logits, probs, index_to_species_id, k=3):
    top_probs, top_classes = torch.topk(probs, k)
    for i in range(k):
        class_index = top_classes[0][i].item()
        species_id = index_to_species_id[class_index]
        probability = top_probs[0][i].item()
        print(f"  {i+1}. {species_id}, 概率: {probability * 100:.2f}%")


def main():
    torch.cuda.empty_cache()

    csv_file_path = 'D:\image-classification\script\spider_index_to_species_id.csv'
    index_to_species_id = load_index_to_species_id_from_csv(csv_file_path)

    checkpoint = torch.load('D:\image-classification\wandb_logs\identify\eoz9k2h9\checkpoints\last.ckpt',
                            map_location=torch.device('cuda:0'), weights_only=True)

    hp = checkpoint['hyper_parameters']
    input_size = hp.get('input_size', 512)

    model = SwinV2Model(
        num_classes=hp.get('num_classes', 75),
        input_size=input_size,
        use_arcface=hp.get('use_arcface', True),
        arcface_s=hp.get('arcface_s', 30.0),
        arcface_m=hp.get('arcface_m', 0.3),
        arcface_sub_center=hp.get('arcface_sub_center', 3),
        arcface_easy_margin=hp.get('arcface_easy_margin', False),
        arcface_ls_eps=hp.get('arcface_ls_eps', 0.0),
    )

    state_dict = checkpoint['state_dict']
    state_dict.pop('model.head.fc.weight', None)
    state_dict.pop('model.head.fc.bias', None)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    img_path = r"E:\Downloads\default.jpg"
    image = Image.open(img_path)

    transform_to_tensor = transforms.Compose([
        ToRGBTransform(),
        transforms.Resize(int(input_size * 1.2)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])
    normalize_fn = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    img_tensor_unnorm = transform_to_tensor(image)
    img_tensor_norm = normalize_fn(img_tensor_unnorm)
    img_batch = img_tensor_norm.unsqueeze(0).to(device)

    # Baseline prediction
    with torch.no_grad():
        baseline_logits = model(img_batch)
    baseline_probs = torch.softmax(baseline_logits, dim=1)
    baseline_pred = baseline_logits.argmax(dim=1).item()

    # Grad-CAM + crop + ensemble
    cam = compute_gradcam(model, img_batch, baseline_pred)
    crop_tensor_norm = extract_crop(img_tensor_unnorm, cam, normalize_fn, input_size)
    crop_batch = crop_tensor_norm.unsqueeze(0).to(device)

    with torch.no_grad():
        crop_logits = model(crop_batch)
    ensemble_logits = (baseline_logits + crop_logits) / 2.0
    ensemble_probs = torch.softmax(ensemble_logits, dim=1)

    print("=" * 60)
    print("Baseline Top-3:")
    print_topk(baseline_logits, baseline_probs, index_to_species_id)
    print()
    print("Ensemble (Baseline + Grad-CAM Crop) Top-3:")
    print_topk(ensemble_logits, ensemble_probs, index_to_species_id)
    print("=" * 60)


if __name__ == '__main__':
    main()
