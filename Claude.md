# image-classification 项目规则

## 项目概述
这是一个使用 PyTorch Lightning 和 EVA02 base 模型（`eva02_base_patch14_448.mim_in22k_ft_in22k`，来自 timm）的图像分类项目。
支持识别约 4.4 万 种国内动植物识别（细粒度分类），差不多 1600 万张训练图片，每个类最多 1000 张，最少 50 张训练图片。使用两阶段训练（前 70% epoch 使用 448px, 后 30% epoch 使用 512px）

## 关键依赖
- torch && pytorch_lightning
- timm（用于 EVA02、Hiera、EfficientNetV2 等模型）
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
- `EVA02Model.forward_features(x)` 通过 `model.forward_features(x) + model.forward_head(x, pre_logits=True)` 获取 fc 前 embedding
- 当 MixUp/CutMix 激活时（labels 为 2D soft labels），ArcFaceLoss 自动跳过 margin，仅使用 s*cosine
- ArcFaceLoss 权重由 EVA02Model 管理，通过 `self.arcface_loss` 属性访问

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

### 3. timm EVA02 特征提取
获取 fc 层之前的 embedding 向量：
```python
x = self.model.forward_features(x)          # 返回 feature maps
embedding = self.model.forward_head(x, pre_logits=True)  # 返回 fc 前向量 [B, in_features]
```
`model.num_features` 可获取 embedding 维度（EVA02 base 为 768）。

## 配置文件格式
JSON 配置文件遵循 LightningCLI 格式，包含三个主要部分：
- **model**：模型类及初始化参数
- **trainer**：训练器配置（epochs、accelerator、callbacks）
- **data**：DataModule 类及数据路径

配置示例
* `./config/mini/eva02_mini.json`
* `./config/mini/eva02_mini_part2.json`
* `./config/fgvc-aves-tiny/eva02_tiny.json`

## 可用的模型类（model.py 中）
- **BaseModel**：基类，包含训练/验证/测试步骤，使用 CrossEntropyLoss
- **EVA02Model**：当前使用的模型，使用 timm `eva02_base_patch14_448.mim_in22k_ft_in22k` 预训练主干（input_size: 448）
- **SwinV2Model**：旧模型（基于 SwinV2 base），保留用于对比实验
- 其他模型类（HieraModel, ConvNextV2Model, DinoV2Model, AIMv2Model, EfficientNetV2Model）均为测试用，无需关注。

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
- 模型使用 EVA02Model 即可。
- 先使用 data 目录下的 fgvc-aves-tiny 数据集进行试验，训练配置参考 config\fgvc-aves-tiny。
- 使用 ./script/train.sh 里的命令来运行实验，先尝试提升 fgvc-aves-tiny 的识别率，每个 epoch 运行可能要 20 分钟。你要监控它的 top1 和 top3 成功率来决定实验结果。
- 每次进行实验时，要使用控制变量法。要列一个计划，写清楚理由。

## 运行完成后如何查看识别率
wandb_logs\identify 下有各个 run id 文件夹，文件里 checkpoint 的文件名就有识别率。比如：`wandb_logs\identify\3dmhbizq\checkpoints\eva02-all-epoch=11-val\acc_top1=0.7127.ckpt`

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

## 从 Wandb 日志查看学习率

Wandb 离线日志（`.wandb` 二进制文件）里记录了两个 LR key：

| Key | 来源 | 频率 | 用途 |
|-----|------|------|------|
| `train/lr` | `training_step` 中手动 `self.log('train/lr', ...)` | 每 epoch 一次（`batch_idx==0`） | 粗略，数据点少 |
| `lr-AdamW` | `LearningRateMonitor` callback | 每个 optimizer step | **推荐**，完整 LR 曲线 |

### 解析方法

`.wandb` 文件是 protobuf 二进制格式，可用正则直接从文件中提取（无需 wandb SDK）。

**推荐做法：只读文件尾部**。`lr-AdamW` 数据在文件持续追加写入，最新 LR 在尾部 32MB 内，无需全量读取：

```python
import re, os

filepath = 'wandb_logs/wandb/<run_dir>/run-<id>.wandb'
file_size = os.path.getsize(filepath)
read_size = min(32 * 1024 * 1024, file_size)

with open(filepath, 'rb') as f:
    if file_size > read_size:
        f.seek(file_size - read_size)
    data = f.read()

# 匹配 lr-AdamW<浮点数> ... global_step<整数>
pattern = re.compile(
    rb'lr-AdamW...([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)'
    rb'.{0,200}'
    rb'global_step...([0-9]+)',
    re.DOTALL
)

LR_MIN_SANE = 1e-10  # 过滤二进制噪声误匹配，真实 LR 不会低于此值

seen = {}
for m in pattern.finditer(data):
    lr = float(m.group(1))
    step = int(m.group(2))
    if LR_MIN_SANE < lr < 1.0:
        seen[step] = lr  # 覆盖写：尾部后出现的值更新，始终保留最新

sorted_steps = sorted(seen.keys())
# latest_step = sorted_steps[-1], latest_lr = seen[latest_step]
```

**关键点：** protobuf 里同一个 step 会有多条重复记录，**不能**用 `step not in seen` 跳过——必须无条件覆盖，因为尾部扫描到的后一条值才是最新的。

**⚠️ 坑：二进制噪声误匹配。** `.wandb` 文件二进制数据中随机字节可能恰好匹配 LR 正则（如 `4.494429e-57`），必须加 `LR_MIN_SANE = 1e-10` 下限过滤。否则误值会走 `lr < 1.0` 进入 seen，导致趋势判断错误（如误判为 cosine 衰减）。

### 全量读取（需要完整曲线时）

对于大文件（数百 MB），用流式分块读取，每 16MB 一块，保留 1000 字节的跨块尾以保证不截断匹配。用 `re.finditer` 而非 `re.findall` 避免内存暴增。

### 验证 Warmup 线性度

对提取的 (step, lr) 做线性回归，检查：
- 截距 ≈ `learning_rate × start_factor`（配置为 `3e-4 × 0.001 = 3e-7`）
- R² ≈ 1.0（线性）
- 每步增量 = `(3e-4 - 3e-7) / warmup_steps`

### 注意事项

- 不是所有 run 都开了 `LearningRateMonitor`（早期 run 可能只有 `train/lr`，每 epoch 才记一次）
- 找最新 run：按 `.wandb` 文件的**修改时间**排序，不要看目录名里的日期（续跑的 run 目录名不变但文件持续更新）
- 用 `re.finditer` 而非 `re.findall` 避免内存暴增

## 注意点
- 该项目在 windows 上进行训练，注意兼容性
