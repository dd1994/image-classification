# image-classification 项目规则

## 项目概述
这是一个使用 PyTorch Lightning 和 SwinV2 transformer(base 型号)的图像分类项目。
支持识别约 4.4 万 种国内动植物识别（细粒度分类），差不多 800 万张训练图片，每个类最多 1000 张，最少 50 张训练图片。使用两阶段训练（前 70% epoch 使用 448px, 后 30% epoch 使用 512px）

## 关键依赖
- torch && pytorch_lightning
- timm（用于 SwinV2、Hiera、EfficientNetV2 模型）
- torchvision

## Python 解释器
python: C:\ProgramData\anaconda3\envs\myenv\python.exe


## 项目结构
```
image-classification/
├── config/              # 不同模型/数据集的 JSON 配置文件
│   ├── large/           # 大模型配置
│   ├── medium/          # 中等模型配置
│   ├── small/           # 小模型配置
│   └── tiny/            # 微型模型配置
├── data/                # 训练和验证数据
├── dataSet/             # 自定义数据集类
│   └── SpecialCateDataset.py
├── script/              # 数据处理工具脚本
├── trash/               # 已废弃/实验性代码
├── util/                # 工具模块（数据增强、损失函数）
├── train.py             # 主训练入口
├── model.py             # 模型定义
├── data_module.py       # DataModule 定义
├── predict.py           # 预测脚本
└── train.sh             # 训练 shell 脚本
```

## 配置文件格式
JSON 配置文件遵循 LightningCLI 格式，包含三个主要部分：
- **model**：模型类及初始化参数
- **trainer**：训练器配置（epochs、accelerator、callbacks）
- **data**：DataModule 类及数据路径

配置示例
* `./config/mini/swinv2_mini.json`
* `./config/mini/swinv2_mini512.json`

## 可用的模型类（model.py 中）
- **BaseModel**：基类，包含训练/验证/测试步骤，使用 CrossEntropyLoss
- **SwinV2Model**：使用 timm 预训练主干的 SwinV2（input_size: 448）

除了 SwinV2Model,还有些其他 model 都是测试用的，无需关注。

## 可用的 DataModule 类（data_module.py 中）
- **INatBaseDataModule**：基类，包含标准数据增强（TrivialAugmentWide、RandomErasing、Normalize）
- **SpecialCateData**：目前在用的自定义分类数据集，支持 id_map_file_path


## 数据目录结构（SpecialCateData）
```
data_dir/
├── class_dir/           # 物种大类（如 Insecta/FungiArachnida）
│   └── species_id/      # 物种文件夹
│       └── *.jpg        # 图片文件
```

## 标准训练命令
```bash
python train.py fit --config ./config/<size>/<model>.json
```

## 训练配置默认值
- 优化器：AdamW（lr=1e-4，weight_decay=2e-5）
- 学习率调度器：CosineAnnealingLR + 线性预热（3 个 epoch）
- 损失函数：CrossEntropyLoss
- 混合精度：bf16-mixed
- 梯度累积：8 个批次
- ImageNet 归一化：mean=[0.485, 0.456, 0.406]，std=[0.229, 0.224, 0.225]

## 日志记录
- 日志器：WandbLogger（离线模式）
- 项目名称："identify"
- 日志保存路径：wandb_logs/identify/<run_id>/checkpoints/

## Wandb 同步（离线运行）

运行 sync_wandb.sh 脚本进行同步最近一个 run id, 也可命令行参数制定 run id.


## 回调函数（标准配置）
- EarlyStopping：monitor=val/acc_top1，patience=7，mode=max
- ModelCheckpoint：monitor=val/acc_top1，save_top_k=3，save_last=true
- LearningRateMonitor：logging_interval=step

## 图像尺寸规范
分两阶段进行训练，前 70% epoch:
- 标准 input_size：448px
- 验证/测试resize：input_size * 1.2 然后中心裁剪

后 30% epoch
- 标准 input_size：512px
- 验证/测试resize：input_size * 1.2 然后中心裁剪

## 预测（predict.py）
- 从 wandb_logs/identify/<run_id>/checkpoints/last.ckpt 加载模型
- 需要 index_to_species_id CSV 文件用于物种名称映射
- 输出 top-3 预测结果及概率

## CUDA 内存
- PYTCH_CUDA_ALLOC_CONF=expandable_segments:True 用于更好的内存管理
- 预测加载模型前使用 torch.cuda.empty_cache()

## 消融实验
- 模型使用 swinV2Model 即可。
- 依次使用 data 目录下的 train-tiny/train-mini/train-small 数据集进行试验，参考 config/tiny/swinv2_tiny.json。
- 使用 train.sh 里的命令来运行实验，先尝试提升 train-tiny 的识别率，每个 epoch 运行可能要 20 分钟。你要监控它的 top1 和 top3 成功率来决定实验结果。
- 每次进行实验时，要使用控制变量法。要列一个计划，写清楚理由。

## 注意点
- 该项目在 windows 上进行训练，注意兼容性
