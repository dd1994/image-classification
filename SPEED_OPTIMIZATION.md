# 训练速度优化方案

> 目标：在不影响识别率的前提下，从各个环节提升训练速度。

---

## 0. 当前配置确认

- 硬件：单卡 4090
- 精度：`16-mixed`（FP16），**建议升级为 `bf16-mixed`**
- 模型：`swinv2_base_window12to24_192to384.ms_in22k_ft_in1k`
- 输入分辨率：448 / 512
- 4 万类动植物分类任务

---

## 1. 数据加载

### 1.1 SpecialCateDataset 目录遍历优化

**问题**：每次 `__init__` 都遍历整个目录树，4 万类别场景下单次初始化可能耗时数分钟。

**方案**：将 `self.index` 缓存为 pickle 文件，复用时直接 load。

```python
import pickle
import os

class SpecialCateDataset(Dataset):
    def __init__(self, root_dir='', id_map_file_path='', transform=None, cache_file='dataset_cache.pkl'):
        self.root_dir = root_dir
        self.transform = transform

        cache_path = os.path.join(root_dir, cache_file)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
                self.index = cached['index']
                self.species_ids = cached['species_ids']
                self.index_to_species_id = cached['index_to_species_id']
            return

        # ... 原有的遍历逻辑 ...

        # 缓存结果
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'species_ids': self.species_ids,
                'index_to_species_id': self.index_to_species_id
            }, f)
```

**预期收益**：首次运行正常初始化，之后秒级加载。

### 1.2 DataLoader 配置优化

| 参数 | 当前值 | 建议值 | 收益 |
|------|--------|--------|------|
| `prefetch_factor` | 1 | 4~8 | DataLoader 吞吐量翻倍 |
| `pin_memory` (val/test) | 未设置 | `True` | GPU 数据传输加速 |
| `persistent_workers` | True (train) | 所有 DataLoader 都加 | 避免每个 epoch 重启 workers |
| `num_workers` | 3 | 8 | 充分利用多核 CPU（需注意 open files 限制） |

**修改位置**：`data_module.py` 第 47-57 行

```python
def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size,
                      shuffle=True, num_workers=8, persistent_workers=True,
                      prefetch_factor=4, pin_memory=True)

def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.batch_size,
                      shuffle=False, num_workers=8, persistent_workers=True,
                      prefetch_factor=4, pin_memory=True)

def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.batch_size,
                      shuffle=False, num_workers=8, persistent_workers=True,
                      prefetch_factor=4, pin_memory=True)
```

---

## 2. 精度升级

### 2.1 FP16 → BF16

**问题**：`16-mixed` 使用 FP16，动态范围有限。

**方案**：所有 config 文件中将 `16-mixed` 替换为 `bf16-mixed`。

```json
"precision": "bf16-mixed"
```

**收益**：
- 4090 原生 BF16 硬件加速
- 动态范围更大，训练更稳定
- 同等显存下可增大 batch size

> **注意**：确认 PyTorch ≥ 1.10，CUDA ≥ 11.0。

---

## 3. 模型推理优化

### 3.1 cudnn.benchmark

**位置**：`train.py` 开头（`cli_main` 函数之前）

```python
import torch
torch.backends.cudnn.benchmark = True
```

**收益**：当输入尺寸固定时，cuDNN 自动选择最优卷积算法，~10-30% 加速。

### 3.2 channels_last 内存格式

**位置**：`model.py`，`SwinV2Model.forward`

```python
def forward(self, x):
    x = x.to(memory_format=torch.channels_last)  # 添加
    return self.model(x)
```

**收益**：cuBLAS 对 channels_last 格式有专门优化，~10-30% 加速。

> ⚠️ 注意：如果同时使用 `torch.compile`，两者可能有冲突，建议分开验证。

### 3.3 torch.compile（可选，实验性）

**位置**：`model.py`，`SwinV2Model.__init__` 末尾

```python
self.model = torch.compile(self.model, mode='reduce-overhead')
```

**收益**：PyTorch 2.0+ 可对模型做 JIT 编译优化，~10-30% 加速。

**风险**：
- 首次运行有编译开销（~1-2 分钟）
- 部分模型结构可能不兼容
- 与 channels_last 混用可能有问题

**建议**：先不加此优化，确认其他优化稳定后再尝试。

---

## 4. 其他优化

### 4.1 梯度累积

当前 `accumulate_grad_batches=8`，effective batch = 32×8=256。

如切换 BF16 后显存充裕，可尝试：
- per-GPU batch 增大到 64，`accumulate_grad_batches` 相应减小到 4
- 减少梯度同步次数，提升吞吐量

### 4.2 验证集不做无意义 shuffle

`val_dataloader` 中 `shuffle=False` 已经是正确的，无需修改。

---

## 5. 推荐实施顺序

| 优先级 | 优化项 | 改动位置 | 风险 |
|--------|--------|---------|------|
| **P0** | cudnn.benchmark | train.py | 无 |
| **P0** | prefetch_factor=4, pin_memory=True | data_module.py | 无 |
| **P0** | BF16 混合精度 | config/*.json | 无 |
| **P1** | 数据集路径缓存 | SpecialCateDataset.py | 无 |
| **P1** | num_workers=8 | data_module.py | 低 |
| **P2** | channels_last | model.py | 极低 |
| **P3** | torch.compile | model.py | 中，需测试 |

---

## 6. 验证方法

每次改完后运行训练，观察：

```bash
# 观察 GPU 利用率（应 > 80%）
nvidia-smi

# 或在训练时观察
python train.py fit --config config/tiny/swinv2_tiny512.json
```

**关键指标**：
- GPU-Util 越高越好（表示 GPU 没有饿着）
- GPU-Memory-util 越高说明显存用得越充分
- 相同 epoch 数下 val/acc_top1 是否持平或提升
