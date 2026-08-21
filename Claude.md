# image-classification 项目规则

## 项目概述
这是一个使用 PyTorch Lightning 和 EVA02 base 模型（`eva02_base_patch14_448.mim_in22k_ft_in22k`，来自 timm）的图像分类项目。
支持识别约 4.4 万 种国内动植物识别（细粒度分类），差不多 1600 万张训练图片，每个类最多 1000 张，最少 50 张训练图片。使用两阶段训练（前 70% epoch 使用强数据增强, 后 30% epoch 减弱数据增强微调）

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

### Wandb 日志
- 日志器：WandbLogger（离线模式）
- 项目名称："identify"
- 日志保存路径：wandb_logs/identify/<run_id>/checkpoints/

### 训练终端日志（train.sh 输出文件）

train.sh 会将训练脚本的 stdout 和 stderr 重定向到 `./logs/train_seed_${seed}.log`。

**实时查看日志**（训练进行中）：
```bash
tail -f ./logs/train_seed_1.log
```

**注意事项**：
- 配置中已设置 `"enable_progress_bar": false`，避免 tqdm 进度条撑爆日志文件
- Python 输出默认有缓冲，可在 train.sh 中加 `PYTHONUNBUFFERED=1` 实现行缓冲实时写入：
  ```bash
  PYTHONUNBUFFERED=1 /c/ProgramData/anaconda3/envs/myenv/python.exe -X faulthandler \
      ./script/train.py fit --config "$CONFIG" ... \
      > "./logs/train_seed_${seed}.log" 2>&1
  ```

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

### 找到最近的 run id
```powershell
Get-ChildItem -Path "D:\image-classification\wandb_logs\identify" -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 5 Name, LastWriteTime
```
按 LastWriteTime 排序，最上面的就是最近活跃的 run。

### 查看 checkpoint 和 val acc
```powershell
# 列出所有 checkpoint（替换 <run_id> 为实际 run id）
Get-ChildItem -Path "D:\image-classification\wandb_logs\identify\<run_id>\checkpoints" |
    Sort-Object LastWriteTime |
    Select-Object Name, @{N='Size (MB)';E={[math]::Round($_.Length/1MB,2)}}, LastWriteTime
```

### 文件命名规则

**训练 checkpoint**（每 ~8 小时自动存）：
- 格式：`eva02-all-epoch=XX-step=NNNNN.ckpt`
- 示例：`eva02-all-epoch=03-step=32508.ckpt`

**Validation checkpoint**（每个 epoch 结束验证完后存）：
- ⚠️ val checkpoint 是**文件夹**，不是文件。文件夹名以 `-val` 结尾
- 文件夹内只有一个 `.ckpt` 文件，文件名包含 top-1 准确率
- 格式：`eva02-all-epoch=XX-val/` (目录) → 内含 `acc_top1=0.XXXX.ckpt`
- 示例：`eva02-all-epoch=03-val/acc_top1=0.7439.ckpt`

**last.ckpt**：最新的 checkpoint，训练崩溃后可从它恢复。

### 踩过的坑
- `swinv2-all-epoch=XX-val` 文件夹在 PowerShell `Get-ChildItem` 不递归时显示为 0 字节，容易误以为文件损坏。实际上是目录，需要 `Get-ChildItem -Recurse` 或进入目录才能看到 `acc_top1=X.XXX.ckpt`。

## OpenClaw 运行训练脚本

训练耗时较长（每个 epoch ~20 分钟，完整训练需数小时），在 OpenClaw 中运行时必须注意：

1. **用 Git Bash 直接调用**，不要双层嵌套 bash -c。Git Bash 路径：`C:\Program Files\Git\bin\bash.exe`
2. **必须设置 timeout: 0**（不限时），否则默认 30 分钟自动杀掉进程
3. 工作目录设为项目根目录 `D:\image-classification`


正确命令（注意必须加 `&` 调用运算符，否则 PowerShell 会把两个连续引号字符串当成语法错误）：
```
exec command: & "C:\Program Files\Git\bin\bash.exe" "./script/train.sh"
exec workdir: D:\image-classification
exec timeout: 0
exec background: true
```

⚠️ 踩过的坑：
- 不能用 PowerShell 运行 bash 脚本，也不能用 `bash -c "..."` 双层嵌套（会导致内存分配问题、子进程 crash）
- **必须用 `&` 前缀**，因为 PowerShell 不认识 `"path1" "arg1"` 两个连续引用字符串的语法
- 第一次没加 `&` 报错 `表达式或语句中包含意外的标记`
- 直接用 `bash.exe ./script/train.sh` 即可

## 从 Wandb 日志查看学习率

使用 `script/check_lr.py` 查看当前 run 的学习率。

### 两种 LR 指标的区别

| Key | 来源 | 频率 | 可靠性 |
|-----|------|------|--------|
| `train/lr` | `training_step` 中 `batch_idx==0` 时 `self.log` | 每 epoch 一次 | ❌ 不可用 — epoch 内 LR 一直在变，但这条只记录 epoch 开头的值 |
| `lr-AdamW/pg{N}` | `LearningRateMonitor` callback | 每 50 optimizer step | ✅ **推荐** — 24 个 param group 各有独立 LR |

### `check_lr.py` 工作原理

`.wandb` 文件是 protobuf 二进制格式，可直接用正则提取（无需 wandb SDK）：

1. 找到文件末尾最新的 `trainer/global_step` 位置
2. 在该位置前后 20KB 窗口内收集所有 `lr-AdamW/pg{N}:<float>` 条目
3. 输出最低层（block.0, lr_scale=0.167）和最高层（head, lr_scale=1.0）的 LR

### Warmup 进度判断

当前配置：`LinearLR(start_factor=0.001)`, 3 epoch warmup, base_lr=3e-4
- 起始 LR: 3e-7, 目标 LR: 3e-4
- head 组的 LR ÷ head_lr_mult = base LR，由此计算 warmup 进度百分比
- 如果 base LR 长时间停在 3e-7 附近且 step 不变 → 训练可能冻结

### ⚠️ 踩过的坑

- **`train/lr` 不可靠**：它只在 `batch_idx==0` 时记录一次，epoch 0 全程显示 3e-7，完全看不出 warmup 在推进
- **全量读取末尾无效**：`lr-AdamW` 数据在 wandb protobuf 中分布在全局，尾部 32MB 可能只包含某个很早 step 的数据。正确做法是从最后一个 `global_step` 位置反向搜索
- **二进制噪声误匹配**：`.wandb` 随机字节可能恰好匹配 LR 正则可读格式（如 `4.494e-57`），必须加 `LR_MIN_SANE = 1e-10` 下限过滤

## 注意点
- 该项目在 windows 上进行训练，注意兼容性

## ONNX 导出与 CPU 推理

### 脚本与产物
- `script/export_onnx.py`：从 `last.ckpt` 直接重建 EVA02 主干并导出整模型。**不要 import `script/model.py`**——
  其顶部 `from aim.v2.utils import ...` / `from transformers import ...` 是训练机独有依赖（aim 不是 PyPI 的 aimstack）。
  直接用 `timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in22k', num_classes=0)` 重建主干（已核实与
  EVA02Model 零缺/零多 key）。产出 `onnx/eva02_all_fp32.onnx`（整模型 → logits `[B,44269]`，~766MB）。
- `script/infer_onnx.py`：单图推理（torch-free），加载 `eva02_all_fp32.onnx` 输出 top-K。
- `script/compare_precision.py`：验证集评估 fp32 ONNX 的 top1/top3（需训练机的 `data/valid` + `train_map_enriched.csv`）。

### transformer 专用优化
- 用 `onnxruntime.transformers.optimizer.optimize_model(model_type='vit', num_heads=12, hidden_size=768, opt_level=1, use_gpu=False)` 做 Attention/LayerNorm/GELU 融合，opset 17 导原生 LayerNormalization 算子；优化后与 torch 前向校验误差 ~1e-5。
- 注意：该优化会引入 com.microsoft 域融合算子，**不能再对其结果做动态量化**（shape 推断失败）。

### 精度结论（已实测）
- 统一用 fp32 整模型（`eva02_all_fp32.onnx`），不做 hybrid / 量化拆分。
- **int8 不可用**：onnxruntime 1.19.2 的 `quantize_dynamic` 量化 EVA02 主干会让 embedding 严重失真（逐通道 cosine≈0.09、逐张量≈0.887），对 4.4 万类细粒度分类直接判错。**不要用动态 int8**。
- FP16 运行时在 CPU 上无内核、不划算；只有上 GPU 才用 `convert_float_to_float16`。

### 推理 session 配置
- `graph_optimization_level=ORT_ENABLE_ALL`、`intra_op_num_threads=物理核数`、`inter_op_num_threads=1`、`ORT_SEQUENTIAL`、`enable_cpu_mem_arena=True`。
