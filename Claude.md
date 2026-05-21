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
│   ├── fgvc-aves-tiny/  # FGVC 鸟类细粒度分类（75 类）
│   ├── large/           # 大模型配置
│   ├── medium/          # 中等模型配置
│   ├── small/           # 小模型配置
│   └── tiny/            # 微型模型配置（168 类）
├── data/                # 训练和验证数据
├── dataSet/             # 自定义数据集类
│   └── SpecialCateDataset.py
├── script/              # 核心脚本（train.py, model.py 等在此目录）
├── trash/               # 已废弃/实验性代码
├── util/                # 工具模块（数据增强、损失函数、ArcFace）
├── script/train.sh      # 训练 shell 脚本（位于 script/ 目录下）
└── wandb_logs/          # Wandb 日志和 checkpoint
```

重要：`train.py`, `model.py`, `data_module.py`, `predict.py` 均在 `script/` 目录下，训练命令需使用 `python script/train.py fit --config ...`

## ArcFace Loss

项目已集成 ArcFace 作为可选 loss（`util/arcface_loss.py`）。ArcFace 通过在角度空间给正确类别施加 margin，提升细粒度分类的类间可分性。

### 使用方式
在 JSON 配置的 `model.init_args` 中添加：
```json
"use_arcface": true,
"arcface_s": 30.0,
"arcface_m": 0.5,
"arcface_sub_center": 1,
"arcface_easy_margin": false,
"arcface_ls_eps": 0.0
```

### 关键实现细节
- ArcFaceLoss 内部持有类别中心权重（`self.weight`），替换了 fc 层（fc 设为 `nn.Identity()`）
- `SwinV2Model.forward_features(x)` 通过 `model.forward_features(x) + model.forward_head(x, pre_logits=True)` 获取 fc 前 embedding
- 当 MixUp/CutMix 激活时（labels 为 2D soft labels），ArcFaceLoss 自动跳过 margin，仅使用 s*cosine
- ArcFaceLoss 权重由 SwinV2Model 管理，通过 `self.arcface_loss` 属性访问

### ArcFace 参数含义
| 参数 | 含义 | 推荐值 |
|------|------|--------|
| arcface_s | 余弦相似度缩放因子 | 30（小数据集）, 50-64（大数据集）|
| arcface_m | 角度 margin（弧度）| 0.5 |
| arcface_sub_center | 每类子中心数 | 1（标准）|
| arcface_easy_margin | 边界处理策略 | false（ArcFace 标准）|
| arcface_ls_eps | Label smoothing | 0.0（过拟合时 0.05-0.1）|

## 已踩过的坑

### 1. `torch.cuda.amp.autocast` 版本兼容
此环境的 PyTorch 版本 `torch.cuda.amp.autocast` 不接受 `device_type` 参数。不要使用该 API 强制 float32，改用显式 `.float()` 转换：
```python
# 错误（此环境不支持）
with torch.cuda.amp.autocast(enabled=False, device_type='cuda'):
    ...

# 正确
embedding = embedding.float()
weight = self.weight.float()
```

### 2. `save_hyperparameters()` 不会设置实例属性
`save_hyperparameters()` 将参数存入 `self.hparams`，但不会自动添加为 `self.xxx` 属性。如需在代码中直接访问 `self.use_arcface`，必须在 `__init__` 中显式设置 `self.use_arcface = use_arcface`。

### 3. timm SwinV2 特征提取
获取 fc 层之前的 embedding 向量：
```python
x = self.model.forward_features(x)          # 返回 feature maps
embedding = self.model.forward_head(x, pre_logits=True)  # 返回 fc 前向量 [B, in_features]
```
`model.head.fc.in_features` 可获取 embedding 维度（SwinV2 base 为 1024）。

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
- **SpecialCateData**：目前在用的自定义分类数据集


## 数据目录结构（SpecialCateData）
```
data_dir/
├── class_dir/           # 物种大类（如 Insecta/Fungi/Arachnida）
│   └── species_id/      # 物种文件夹
│       └── *.jpg        # 图片文件
```

## 标准训练命令
```bash
python script/train.py fit --config ./config/<size>/<model>.json
```

## 日志记录
- 日志器：WandbLogger（离线模式）
- 项目名称："identify"
- 日志保存路径：wandb_logs/identify/<run_id>/checkpoints/

## Wandb 同步（离线运行）

运行 ./script/sync_wandb.sh 脚本进行同步最近一个 run id, 也可命令行参数制定 run id.

## 图像尺寸规范
分两阶段进行训练，前 70% epoch:
- 标准 input_size：448px
- 验证/测试resize：input_size * 1.2 然后中心裁剪

后 30% epoch
- 标准 input_size：512px
- 验证/测试resize：input_size * 1.2 然后中心裁剪

## 预测（predict.py）
- predict.py 位于 script/ 目录下
- 从 wandb_logs/identify/<run_id>/checkpoints/last.ckpt 加载模型
- 需要 index_to_species_id CSV 文件用于物种名称映射
- 输出 top-3 预测结果及概率
- 注意：加载 ArcFace checkpoint 时需从 `checkpoint['hyper_parameters']` 读取 arcface 参数来初始化模型

## CUDA 内存
- PYTCH_CUDA_ALLOC_CONF=expandable_segments:True 用于更好的内存管理
- 预测加载模型前使用 torch.cuda.empty_cache()

## 消融实验
- 模型使用 swinV2Model 即可。
- 先使用 data 目录下的 fgvc-aves-tiny 数据集进行试验，训练配置参考 config\fgvc-aves-tiny。
- 使用 ./script/train.sh 里的命令来运行实验，先尝试提升 fgvc-aves-tiny 的识别率，每个 epoch 运行可能要 20 分钟。你要监控它的 top1 和 top3 成功率来决定实验结果。
- 每次进行实验时，要使用控制变量法。要列一个计划，写清楚理由。

## OpenClaw 运行训练脚本

训练耗时较长（每个 epoch ~20 分钟，完整训练需数小时），在 OpenClaw 中运行时必须注意：

1. **用 Git Bash 直接调用**，不要双层嵌套 bash -c。Git Bash 路径：`C:\Program Files\Git\bin\bash.exe`
2. **必须设置 timeout: 0**（不限时），否则默认 30 分钟自动杀掉进程
3. 工作目录设为项目根目录 `D:\image-classification`

正确命令：
```
exec command: "C:\Program Files\Git\bin\bash.exe" "./script/train.sh"
exec workdir: D:\image-classification
exec timeout: 0
```

注意：不能用 PowerShell 运行 bash 脚本，也不能用 `bash -c "..."` 双层嵌套（会导致内存分配问题、子进程 crash）。直接 `bash.exe ./script/train.sh` 即可。

## 注意点
- 该项目在 windows 上进行训练，注意兼容性
