import gc
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.cuda
torch.set_float32_matmul_precision('high')  # 启用 TF32，利用 RTX 4090 Tensor Cores 加速
from pytorch_lightning.cli import LightningCLI

from data_module import INatBaseDataModule
from model import BaseModel
from util.win_sleep import prevent_sleep, restore_sleep


def cli_main():
    # Monkey-patch _atomic_save: 用临时文件替代 BytesIO，避免长时间训练后
    # 内存碎片化导致 torch.save 到 BytesIO 时无法分配连续内存而 MemoryError。
    # torch.save 直接写文件是顺序磁盘写入，不需要大块连续内存。
    import lightning_fabric.plugins.io.torch_io as _tio
    import tempfile as _tempfile

    _original_atomic_save = _tio._atomic_save

    def _patched_atomic_save(checkpoint, filepath):
        import os
        from pathlib import Path
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = _tempfile.mkstemp(dir=filepath.parent, prefix='.tmp-', suffix='.ckpt')
        try:
            os.close(fd)
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, str(filepath))
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    _tio._atomic_save = _patched_atomic_save

    # 训练开始前设置防止休眠
    # prevent_sleep()
    #     gc.collect()
    #     torch.cuda.empty_cache()
    # try:
    cli = LightningCLI(
        model_class=BaseModel,
        datamodule_class=INatBaseDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=42,
        trainer_defaults={
            "accumulate_grad_batches": 8,
            "logger": {
                "class_path": "pytorch_lightning.loggers.WandbLogger",
                "init_args": {
                    "mode": "offline",
                    "project": "identify",
                    "log_model": False,
                    "save_dir": "wandb_logs"
                }
            }
        }
    )
    # finally:
        # 训练结束后恢复休眠设置
        # restore_sleep()

if __name__ == '__main__':
    cli_main()