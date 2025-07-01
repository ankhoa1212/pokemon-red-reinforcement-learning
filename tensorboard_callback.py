from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

# def calculate_values(info):
#     total, 
#     for key, value in info.items():

class TensorBoardCallback(BaseCallback):
    def __init__(self, log_dir, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = None

    def _on_training_start(self) -> None:
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=Path(self.log_dir))

    def _on_step(self) -> bool:
        if self.training_env.env_method("truncated_check", indices=[0])[0]:
            info = self.training_env.get_attr("info")
            for key, value in info.items():
                self.logger.record(f"env_stats/{key}", value)
        return True

    def _on_training_end(self) -> None:
        if self.writer:
            self.writer.close()

    def _on_rollout_start(self) -> None:
        pass

    def _on_rollout_end(self) -> None:
        pass