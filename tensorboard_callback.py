from collections import Counter
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def merge_visit_count_deltas(global_counts, deltas):
    """
    Merges one or more worker-local visit-count deltas into a global
    visit-count table.

    Each delta represents a single worker's counts incremented since its
    last sync (see PokemonRedEnv.pop_visit_count_delta). A given state is
    only ever incremented, between two syncs, by the worker(s) that
    actually observed it during that interval, so when a key appears in
    more than one delta (or already exists in global_counts) the counts
    represent independent new observations and are summed together, not
    deduplicated.

    This is pure and standalone -- it doesn't touch any vectorized-env
    machinery -- so it can be unit tested directly against plain dicts.

    Args:
        global_counts: the current global visit-count table. Not mutated.
        deltas: an iterable of per-worker delta dicts (state hash -> count
            increment observed since that worker's last sync).

    Returns:
        A new Counter: global_counts with every delta's counts added in.
        Returned as a Counter, not a plain dict -- this value gets pushed
        back onto each worker's `visit_counts` via VecEnv.set_attr, and
        PokemonRedEnv.calculate_fitness relies on Counter's zero-default
        behavior (`self.visit_counts[key] += 1`) for keys it hasn't seen
        yet; downgrading to a plain dict here would KeyError on that line
        the next time a worker visits a genuinely new state.
    """
    merged = Counter(global_counts)
    for delta in deltas:
        merged.update(delta)
    return merged


def sync_visit_counts(vec_env, global_counts):
    """
    Runs one pull/merge/push visit-count sync round across every worker in
    a vectorized environment.

    Pulls each worker's local delta via env_method("pop_visit_count_delta")
    (all workers, not just index 0). If every worker's delta is empty (no
    new states since the last sync), every local table already matches
    global_counts from the previous round, so the merge and the
    full-table set_attr broadcast are skipped -- that broadcast cost grows
    with the number of distinct states found so far, and paying it when
    nothing changed is pure waste. Otherwise, merges the deltas into
    global_counts with merge_visit_count_deltas, then broadcasts a copy of
    the resulting global table to every worker via
    set_attr("visit_counts", ...) so each worker's local table converges
    on the shared baseline.

    A distinct Counter copy is set per worker rather than sharing one
    object across the set_attr call: under DummyVecEnv, all workers run
    in this same process, so set_attr("visit_counts", merged) with no
    per-worker copy would hand every worker (and global_counts itself)
    the same mutable object -- every worker's subsequent local increments
    would then double-count directly into global_counts, silently
    inflating visit counts further with every sync. SubprocVecEnv doesn't
    have this problem (each worker is a separate process, so set_attr's
    pickling already copies the value), but the fix must hold for both.

    Args:
        vec_env: a VecEnv-like object (DummyVecEnv, SubprocVecEnv, or a
            duck-typed stub for testing) exposing env_method/set_attr.
        global_counts: the current global visit-count table.

    Returns:
        The updated global visit-count table (also pushed to every
        worker, unless no worker had anything new to report). Never the
        same object as any worker's local table.
    """
    deltas = vec_env.env_method("pop_visit_count_delta")
    if not any(deltas):
        return global_counts
    merged = merge_visit_count_deltas(global_counts, deltas)
    for i in range(vec_env.num_envs):
        vec_env.set_attr("visit_counts", Counter(merged), indices=[i])
    return merged


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
    def __init__(self, log_dir, verbose: int = 0, sync_interval: int = 1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = None
        # How many rollouts to wait between cross-worker visit-count syncs.
        # Named and tunable independently of PPO's n_steps -- see
        # sync_visit_counts / _on_rollout_end.
        self.sync_interval = sync_interval
        self._rollouts_since_sync = 0
        self.global_visit_counts = {}

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
        self._rollouts_since_sync += 1
        if self._rollouts_since_sync >= self.sync_interval:
            self._rollouts_since_sync = 0
            self.global_visit_counts = sync_visit_counts(
                self.training_env, self.global_visit_counts
            )