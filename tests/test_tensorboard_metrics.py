from stable_baselines3.common.vec_env import DummyVecEnv

from tensorboard_callback import TensorBoardCallback
from test_visit_count_merge import make_stub_env


class StubLogger:
    """Minimal stand-in for SB3's Logger, capturing record() calls."""

    def __init__(self):
        self.recorded = {}

    def record(self, key, value):
        self.recorded[key] = value


class StubModel:
    def __init__(self, logger, vec_env):
        self.logger = logger
        self._vec_env = vec_env

    def get_env(self):
        return self._vec_env


def make_callback(vec_env, sync_interval=1):
    callback = TensorBoardCallback(log_dir="unused", sync_interval=sync_interval)
    callback.model = StubModel(StubLogger(), vec_env)
    return callback


# --- Happy path ---------------------------------------------------------

def test_sync_round_with_new_states_logs_cumulative_and_delta():
    vec_env = DummyVecEnv([make_stub_env, make_stub_env])
    try:
        callback = make_callback(vec_env)
        vec_env.env_method("increment", "state_a", indices=[0])
        vec_env.env_method("increment", "state_b", indices=[1])

        callback._on_rollout_end()

        recorded = callback.model.logger.recorded
        assert recorded["env_stats/distinct_states_total"] == 2
        assert recorded["env_stats/distinct_states_new"] == 2
    finally:
        vec_env.close()


# --- Edge case -----------------------------------------------------------

def test_sync_short_circuit_logs_zero_delta_and_unchanged_total():
    vec_env = DummyVecEnv([make_stub_env, make_stub_env])
    try:
        callback = make_callback(vec_env)
        # First round establishes a non-empty global table.
        vec_env.env_method("increment", "state_a", indices=[0])
        callback._on_rollout_end()
        assert callback.model.logger.recorded["env_stats/distinct_states_total"] == 1

        # Second round: no worker has anything new, so sync_visit_counts
        # short-circuits and returns the same table unchanged.
        callback._on_rollout_end()

        recorded = callback.model.logger.recorded
        assert recorded["env_stats/distinct_states_total"] == 1
        assert recorded["env_stats/distinct_states_new"] == 0
    finally:
        vec_env.close()


# --- Integration -----------------------------------------------------------

def test_cumulative_metric_grows_monotonically_across_rollouts():
    vec_env = DummyVecEnv([make_stub_env, make_stub_env])
    try:
        callback = make_callback(vec_env)

        vec_env.env_method("increment", "state_a", indices=[0])
        callback._on_rollout_end()
        first_total = callback.model.logger.recorded["env_stats/distinct_states_total"]
        first_delta = callback.model.logger.recorded["env_stats/distinct_states_new"]

        vec_env.env_method("increment", "state_a", indices=[0])  # revisit, not new
        vec_env.env_method("increment", "state_b", indices=[1])  # genuinely new
        callback._on_rollout_end()
        second_total = callback.model.logger.recorded["env_stats/distinct_states_total"]
        second_delta = callback.model.logger.recorded["env_stats/distinct_states_new"]

        assert first_total == 1
        assert first_delta == 1
        assert second_total == 2
        assert second_delta == 1  # only state_b is genuinely new, not double-counted
    finally:
        vec_env.close()
