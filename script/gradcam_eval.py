import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from model import SwinV2Model
from util.transform import ToRGBTransform

CHECKPOINT_PATH = "d:/image-classification/wandb_logs/identify/eoz9k2h9/checkpoints/last.ckpt"
VAL_DATA_DIR = "d:/image-classification/data/valid-fgvc-aves-tiny"
ID_MAP_PATH = "d:/image-classification/valid_fgvc-aves-tiny_map.csv"

CAM_THRESHOLD = 0.3
MIN_AREA_RATIO = 0.25
BBOX_EXPAND = 0.15


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    hp = checkpoint['hyper_parameters']

    model = SwinV2Model(
        num_classes=hp.get('num_classes', 75),
        input_size=hp.get('input_size', 512),
        learning_rate=hp.get('learning_rate', 1e-4),
        use_arcface=hp.get('use_arcface', True),
        arcface_s=hp.get('arcface_s', 30.0),
        arcface_m=hp.get('arcface_m', 0.3),
        arcface_sub_center=hp.get('arcface_sub_center', 3),
        arcface_easy_margin=hp.get('arcface_easy_margin', False),
        arcface_ls_eps=hp.get('arcface_ls_eps', 0.0),
        use_gradient_checkpointing=hp.get('use_gradient_checkpointing', False),
    )
    state_dict = checkpoint['state_dict']
    state_dict.pop('model.head.fc.weight', None)
    state_dict.pop('model.head.fc.bias', None)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def build_transforms(input_size):
    transform_to_tensor = transforms.Compose([
        ToRGBTransform(),
        transforms.Resize(int(input_size * 1.2)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return transform_to_tensor, normalize


def build_image_index(val_dir, id_map_path):
    index_to_species_id = {}
    with open(id_map_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            index_to_species_id[int(row[0])] = row[1]
    species_id_to_index = {v: k for k, v in index_to_species_id.items()}

    images = []
    aves_dir = os.path.join(val_dir, 'Aves')
    for species_id in sorted(os.listdir(aves_dir)):
        species_path = os.path.join(aves_dir, species_id)
        if not os.path.isdir(species_path) or species_id.startswith('.'):
            continue
        label = species_id_to_index[species_id]
        for fname in sorted(os.listdir(species_path)):
            if fname.startswith('.'):
                continue
            images.append((os.path.join(species_path, fname), label))
    return images


def compute_gradcam(model, img_tensor, target_class):
    """
    Args:
        model: SwinV2Model in eval mode
        img_tensor: normalized image [1, 3, 512, 512] on device
        target_class: integer class index
    Returns:
        cam_norm: [input_size, input_size] numpy array in [0, 1]
    """
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

    # feature_maps and gradients: [1, H, W, C] NHWC
    fm = feature_maps.detach()
    grad = gradients.detach()

    # alpha = GAP over H,W: [1, 1, 1, C]
    alpha = grad.mean(dim=(1, 2), keepdim=True)

    # cam = sum over channel: [1, H, W]
    cam = (alpha * fm).sum(dim=3)
    cam = torch.relu(cam)

    # Upsample
    h, w = fm.shape[1], fm.shape[2]
    input_size = img_tensor.shape[2]  # 512
    cam_4d = cam.unsqueeze(1)  # [1, 1, H, W]
    cam_upsampled = F.interpolate(cam_4d, size=(input_size, input_size),
                                   mode='bilinear', align_corners=False)
    cam_upsampled = cam_upsampled.squeeze()  # [input_size, input_size]

    cam_min = cam_upsampled.min()
    cam_max = cam_upsampled.max()
    if cam_max > cam_min:
        cam_norm = (cam_upsampled - cam_min) / (cam_max - cam_min)
    else:
        cam_norm = torch.zeros_like(cam_upsampled)

    return cam_norm.cpu().numpy()


def extract_crop(img_tensor_unnorm, cam_heatmap, normalize, input_size,
                  threshold=CAM_THRESHOLD, min_area_ratio=MIN_AREA_RATIO,
                  bbox_expand=BBOX_EXPAND):
    """
    Args:
        img_tensor_unnorm: [3, H, W] tensor in [0,1]
        cam_heatmap: [H, W] numpy array in [0,1]
        normalize: Normalize transform
        input_size: int
    Returns:
        crop_tensor_norm: [3, H, W] normalized tensor
    """
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

    crop = img_tensor_unnorm[:, y1:y2, x1:x2]  # [3, h_crop, w_crop]
    crop_4d = crop.unsqueeze(0)  # [1, 3, h, w]
    crop_resized = F.interpolate(crop_4d, size=(input_size, input_size),
                                  mode='bilinear', align_corners=False)
    crop_norm = normalize(crop_resized.squeeze(0))
    return crop_norm


def infer(model, img_tensor_norm, device):
    batch = img_tensor_norm.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(batch)
    return logits.cpu()


def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(CHECKPOINT_PATH, device)
    input_size = model.hparams.input_size
    print(f"Model loaded, input_size={input_size}, num_classes={model.hparams.num_classes}")

    transform_to_tensor, normalize = build_transforms(input_size)
    image_list = build_image_index(VAL_DATA_DIR, ID_MAP_PATH)
    print(f"Validation images: {len(image_list)}")

    results = {
        'baseline_correct': 0,
        'crop_correct': 0,
        'ensemble_avg_correct': 0,
        'ensemble_max_correct': 0,
        'total': 0,
        'skip_count': 0,
    }

    for img_path, true_label in tqdm(image_list, desc="Evaluating"):
        try:
            pil_img = Image.open(img_path)
            img_tensor_unnorm = transform_to_tensor(pil_img)
            img_tensor_norm = normalize(img_tensor_unnorm)
            img_batch = img_tensor_norm.unsqueeze(0).to(device)

            # Baseline
            baseline_logits = infer(model, img_tensor_norm, device)
            baseline_pred = baseline_logits.argmax(dim=1).item()

            # Grad-CAM using baseline prediction as target
            cam = compute_gradcam(model, img_batch, baseline_pred)

            # Crop and infer
            crop_tensor_norm = extract_crop(img_tensor_unnorm, cam, normalize, input_size)
            crop_logits = infer(model, crop_tensor_norm, device)
            crop_pred = crop_logits.argmax(dim=1).item()

            # Ensemble avg
            ensemble_avg_logits = (baseline_logits + crop_logits) / 2.0
            ensemble_avg_pred = ensemble_avg_logits.argmax(dim=1).item()

            # Ensemble max
            ensemble_max_logits = torch.max(baseline_logits, crop_logits)
            ensemble_max_pred = ensemble_max_logits.argmax(dim=1).item()

            results['total'] += 1
            results['baseline_correct'] += int(baseline_pred == true_label)
            results['crop_correct'] += int(crop_pred == true_label)
            results['ensemble_avg_correct'] += int(ensemble_avg_pred == true_label)
            results['ensemble_max_correct'] += int(ensemble_max_pred == true_label)

        except Exception as e:
            results['skip_count'] += 1
            tqdm.write(f"  Skip {img_path}: {e}")

    # Report
    total = results['total']
    print(f"\n{'='*60}")
    print(f"Total images processed: {total}")
    print(f"Skipped: {results['skip_count']}")
    print(f"{'='*60}")
    print(f"{'Strategy':<25} {'Correct':<10} {'Total':<10} {'Top-1':<10}")
    print(f"{'-'*55}")
    for name, correct in [("Baseline", results['baseline_correct']),
                            ("Crop-Only", results['crop_correct']),
                            ("Ensemble Avg", results['ensemble_avg_correct']),
                            ("Ensemble Max", results['ensemble_max_correct'])]:
        print(f"{name:<25} {correct:<10} {total:<10} {correct/total*100:.2f}%")


if __name__ == '__main__':
    run_evaluation()
