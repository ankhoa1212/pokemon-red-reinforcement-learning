from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def calculate_values(info):
    sum_dict = {}
    count_dict = {}
    distrib_dict = {}
    scalar_list = []

    for dict in info:
        del dict["TimeLimit.truncated"]
        scalar_dict = {}
        for k, v in dict.items():
            if isinstance(v, (int, float)): 
                sum_dict[k] = sum_dict.get(k, 0) + v
                count_dict[k] = count_dict.get(k, 0) + 1
                distrib_dict.setdefault(k, []).append(v)
                scalar_dict[dict["steps"]] = dict[k] if k != "steps" else None
        if scalar_dict:
            scalar_list.append(scalar_dict)

    mean_dict = {}
    for k in sum_dict:
        mean_dict[k] = sum_dict[k] / count_dict[k]
        distrib_dict[k] = np.array(distrib_dict[k])

    return scalar_list, mean_dict, distrib_dict


class TensorBoardCallback(BaseCallback):
    def __init__(self, log_dir, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = None

    def _on_training_start(self) -> None:
        if self.verbose >= 1:
            print(f"Logging with TensorBoard to {self.log_dir}")
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=Path(self.log_dir))

    def _on_step(self) -> bool:
        truncated = self.training_env.env_method("pre_truncated_check", indices=[0])[0]
        if self.verbose > 1:
            print(f"Truncated check: {truncated}")

        if truncated:
            info = self.training_env.get_attr("info")
            if self.verbose > 1:
                print(f"Info: {info}")
            final_info = [stat[-1] for stat in info]
            _, mean, distributions = calculate_values(final_info)
            # for scalar in scalars:
            #     for key, val in scalar.items():
            #         self.writer.add_scalar(f"env_stats/{key}", val, self.n_calls)

            if self.verbose > 1:
                print("Recording environment mean stats:", mean)
            for key, val in mean.items():
                self.logger.record(f"env_stats/{key}", val)

            if self.verbose > 1:
                print("Recording environment distribution stats:", distributions)
            for key, distrib in distributions.items():
                self.writer.add_histogram(f"env_stats_distribs/{key}", distrib, self.n_calls)
                self.logger.record(f"env_stats_max/{key}", max(distrib))
        return True

    def _on_training_end(self) -> None:
        if self.verbose > 1:
            print(f"Ending training with TensorBoard to {self.log_dir}")
        if self.writer:
            self.writer.close()

    def _on_rollout_start(self) -> None:
        pass

    def _on_rollout_end(self) -> None:
        pass