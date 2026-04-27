from pytorch_lightning.callbacks import Callback


class EpochStoppingCallback(Callback):
    def __init__(self, stop_at_epoch: int):
        super().__init__()
        self.stop_at_epoch = stop_at_epoch

    def on_train_epoch_end(self, trainer, *_):
        if trainer.current_epoch == self.stop_at_epoch:
            print(f"\n[EpochStoppingCallback] Reached epoch {self.stop_at_epoch}, stopping training...")
            trainer.should_stop = True
