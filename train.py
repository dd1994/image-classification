from pytorch_lightning.cli import LightningCLI

from data_module import INatBaseDataModule
from model import BaseModel
from util.win_sleep import prevent_sleep, restore_sleep


def cli_main():
    # 训练开始前设置防止休眠
    # prevent_sleep()
    
    # try:
        cli = LightningCLI(
            model_class=BaseModel,
            datamodule_class=INatBaseDataModule,
            subclass_mode_model=True,
            subclass_mode_data=True,
            save_config_callback=None,
            seed_everything_default=1,
            trainer_defaults={
                "accumulate_grad_batches": 8,
                "logger": {
                    "class_path": "pytorch_lightning.loggers.WandbLogger",
                    "init_args": {
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