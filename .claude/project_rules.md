# Project Rules for image-classification

## Python Interpreter
python: C:\ProgramData\anaconda3\envs\myenv\python.exe

## Project Overview
This is an image classification project using PyTorch Lightning with SwinV2 vision transformers.
支持识别约 4.4 万 种国内动植物识别，差不多 800 万张训练图片，每个类最多 1000 张，最少 50 张训练图片。

## Key Dependencies
- pytorch_lightning
- timm (for SwinV2, Hiera, EfficientNetV2 models)
- torchvision
- transformers (for ConvNextV2)
- aim (for AIMv2)
- torch



## Project Structure
```
image-classification/
├── config/              # JSON configuration files for different models/datasets
│   ├── large/           # Large model configs
│   ├── medium/          # Medium model configs
│   ├── small/           # Small model configs
│   └── tiny/            # Tiny model configs
├── data/                # Training and validation data
├── dataSet/             # Custom dataset classes
│   └── SpecialCateDataset.py
├── script/              # Utility scripts for data processing
├── trash/               # Deprecated/experimental code
├── util/                # Utility modules (transforms, losses)
├── train.py             # Main training entry point
├── model.py             # Model definitions
├── data_module.py       # DataModule definitions
├── predict.py           # Prediction script
└── train.sh             # Training shell script
```

## Configuration File Format
JSON configs follow LightningCLI format with three main sections:
- **model**: Model class and initialization arguments
- **trainer**: Trainer configuration (epochs, accelerator, callbacks)
- **data**: DataModule class and data paths

Example config structure:
```json
{
  "model": {
    "class_path": "SwinV2Model",
    "init_args": { "num_classes": 168, "learning_rate": 1e-4, "input_size": 448, "t_max": 17 }
  },
  "trainer": {
    "max_epochs": 17,
    "accelerator": "gpu",
    "devices": 1,
    "precision": "16-mixed",
    "accumulate_grad_batches": 8,
    "callbacks": [...]
  },
  "data": {
    "class_path": "SpecialCateData",
    "init_args": { "data_dir": "./data/train-tiny", "batch_size": 15, "num_workers": 5, "input_size": 448 }
  }
}
```

## Available Model Classes (in model.py)
- **BaseModel**: Base class with training/validation/test steps, uses CrossEntropyLoss
- **SwinV2Model**: SwinV2 with timm pretrained backbone (input_size: 448)
- **SwinV2FixResModel**: SwinV2 variant with fixed resolution
- **ConvNextV2Model**: Facebook ConvNextV2 (from transformers)
- **DinoV2Model**: DINO V2 with frozen backbone (linear_head trainable)
- **AIMv2Model**: AIMv2 with custom classifier head
- **HieraModel**: Meta Hiera model (from timm)
- **EfficientNetV2Model**: EfficientNetV2-L from timm
但是除了 SwinV2Model 其他都是测试用的

## Available DataModule Classes (in data_module.py)
- **INatBaseDataModule**: Base class with standard transforms (TrivialAugmentWide, RandomErasing, Normalize)
- **SpecialCateData**: For custom category datasets with id_map_file_path
- **INatDataModule2019**: iNaturalist 2019 dataset (train/val/test split 80/10/10)
- **INatDataModule2021Mini**: iNaturalist 2021 mini dataset (separate train/valid dirs)
目前只有 SpecialCateData 在用。

## Data Directory Structure (SpecialCateData)
```
data_dir/
├── class_dir/           # Top-level category (e.g., "arachnida")
│   └── species_id/      # Species folder
│       └── *.jpg        # Image files
```

## Standard Training Command
```bash
python train.py fit --config ./config/<size>/<model>.json
```

## Training Configuration Defaults
- Optimizer: AdamW (lr=1e-4, weight_decay=2e-5)
- Scheduler: CosineAnnealingLR with linear warmup (3 epochs)
- Loss: CrossEntropyLoss
- Mixed Precision: 16-mixed
- Gradient Accumulation: 8 batches
- Data Augmentation: CutMix/MixUp (80% probability), TrivialAugmentWide, RandomErasing
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

## Logging
- Logger: WandbLogger (offline mode)
- Project: "identify"
- Logs saved to: wandb_logs/identify/<run_id>/checkpoints/

## Wandb Sync (Offline Runs)
离线训练日志同步到 wandb 服务器：

```bash
# 查看所有离线运行
cd wandb_logs; python -m wandb sync --show 20 --include-offline --no-include-online

# 同步最新的离线运行（79gm4hjy，你需要根据时间戳来找出最新的 run_id ）
cd wandb_logs; python -m wandb sync wandb\offline-run-20260423_200036-79gm4hjy

# 同步所有离线运行
cd wandb_logs; python -m wandb sync --sync-all
```

## Callbacks (Standard)
- EarlyStopping: monitor=val/acc_top1, patience=7, mode=max
- ModelCheckpoint: monitor=val/acc_top1, save_top_k=3, save_last=true
- LearningRateMonitor: logging_interval=step

## Image Size Conventions
- Standard input_size: 448
- Validation/test resize: input_size * 1.2 then center crop

## Prediction (predict.py)
- Loads checkpoint from wandb_logs/identify/<run_id>/checkpoints/last.ckpt
- Requires index_to_species_id CSV mapping for species names
- Outputs top-3 predictions with probabilities

## CUDA Memory
- PYTCH_CUDA_ALLOC_CONF=expandable_segments:True for better memory management
- Use torch.cuda.empty_cache() before loading models for prediction

## 消融实验
- 模型使用 swinV2Model 即可。
- 依次使用 data 目录下的 train-tiny/train-mini/train-small 数据集进行试验，参考 config/tiny/swinv2_tiny.json 。
- 使用 train.sh 里的命令来运行实验，先尝试提升 train-tiny 的识别率，每个 epoch 运行可能要 20 分钟。你要监控它的 top1 和 top3 成功率来决定实验结果。
- 每次进行实验时，要使用控制变量法。要列一个计划，写清楚理由。
