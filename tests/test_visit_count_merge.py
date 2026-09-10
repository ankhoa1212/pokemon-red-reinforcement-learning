from collections import Counter

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from tensorboard_callback import merge_visit_count_deltas, sync_visit_counts


class StubCountEnv(gym.Env):
    """
    Minimal gymnasium.Env stub exposing only what the cross-worker
    visit-count sync machinery needs: a settable `visit_counts` table and
    the `pop_visit_count_delta` hook PokemonRedEnv also implements. No
    PyBoy/ROM dependency, so this is cheap to run under both DummyVecEnv
    (in-process) and SubprocVecEnv (real separate processes).
    """

    observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    action_space = spaces.Discrete(1)

    def __init__(self):
        super().__init__()
        self.visit_counts = Counter()
        self._visit_count_delta = Counter()

    def increment(self, key):
        """Test helper mirroring PokemonRedEnv.calculate_fitness's
        bookkeeping: simulates this worker observing a state once."""
        self.visit_counts[key] += 1
        self._visit_count_delta[key] += 1

    def pop_visit_count_delta(self):
        delta = self._visit_count_delta
        self._visit_count_delta = Counter()
        return delta

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


def make_stub_env():
    return StubCountEnv()


# --- Happy path: pure merge logic ------------------------------------------

def test_merge_no_overlapping_keys_produces_union():
    delta_a = {"state_1": 3}
    delta_b = {"state_2": 5}

    merged = merge_visit_count_deltas({}, [delta_a, delta_b])

    assert merged == {"state_1": 3, "state_2": 5}


def test_merge_overlapping_keys_sums_counts():
    delta_a = {"state_1": 2, "state_2": 1}
    delta_b = {"state_1": 4}

    merged = merge_visit_count_deltas({}, [delta_a, delta_b])

    assert merged == {"state_1": 6, "state_2": 1}


# --- Edge case ---------------------------------------------------------

def test_merge_empty_delta_leaves_global_table_unchanged():
    global_counts = {"state_1": 10, "state_2": 2}

    merged = merge_visit_count_deltas(global_counts, [{}])

    assert merged == global_counts
    # Pure function: the input dict itself must not be mutated.
    assert merged is not global_counts


def test_sync_with_every_delta_empty_skips_broadcast():
    global_counts = {"pre_existing": 5}
    vec_env = DummyVecEnv([make_stub_env, make_stub_env])
    try:
        # No worker has incremented anything since the last sync.
        result = sync_visit_counts(vec_env, global_counts)

        # The unchanged table is returned as-is (same object), and nothing
        # is pushed down to the workers -- broadcasting a no-op change
        # would cost O(table size) in IPC for zero new information.
        assert result is global_counts
        for local_table in vec_env.get_attr("visit_counts"):
            assert local_table == {}
    finally:
        vec_env.close()


def test_sync_does_not_alias_the_same_table_across_dummy_workers_and_global():
    """
    Regression test: under DummyVecEnv, all workers share one process, so
    a naive set_attr("visit_counts", merged) with no per-worker copy would
    hand every worker (and global_counts itself) the exact same mutable
    object. A worker's own local increments would then double-count
    directly into that shared object, so a second sync round would report
    inflated counts instead of the true total.
    """
    vec_env = DummyVecEnv([make_stub_env, make_stub_env])
    try:
        vec_env.env_method("increment", "X", indices=[0])
        global_counts = sync_visit_counts(vec_env, {})

        vec_env.env_method("increment", "X", indices=[0])
        global_counts = sync_visit_counts(vec_env, global_counts)

        assert global_counts == {"X": 2}

        local_tables = vec_env.get_attr("visit_counts")
        assert local_tables[0] is not local_tables[1]
        assert local_tables[0] is not global_counts
    finally:
        vec_env.close()


# --- Integration: DummyVecEnv (in-process, all-workers scope) ----------

def test_sync_pulls_from_and_broadcasts_to_every_dummy_worker():
    vec_env = DummyVecEnv([make_stub_env, make_stub_env, make_stub_env])
    try:
        # Each worker independently observes a distinct state -- including
        # workers other than index 0, which _on_step's truncation check
        # narrows to via indices=[0]. sync_visit_counts must not repeat
        # that narrowing.
        vec_env.env_method("increment", "state_worker_0", indices=[0])
        vec_env.env_method("increment", "state_worker_1", indices=[1])
        vec_env.env_method("increment", "state_worker_2", indices=[2])

        global_counts = sync_visit_counts(vec_env, {})

        assert global_counts == {
            "state_worker_0": 1,
            "state_worker_1": 1,
            "state_worker_2": 1,
        }

        # The full merged table must be broadcast back to every worker,
        # not just worker 0.
        local_tables = vec_env.get_attr("visit_counts")
        assert len(local_tables) == 3
        for local_table in local_tables:
            assert local_table == global_counts
    finally:
        vec_env.close()


def test_sync_with_one_worker_delta_empty_only_adds_the_others():
    vec_env = DummyVecEnv([make_stub_env, make_stub_env])
    try:
        # Only worker 0 has observed anything since the last sync; worker
        # 1's delta is empty.
        vec_env.env_method("increment", "state_worker_0", indices=[0])

        global_counts = sync_visit_counts(vec_env, {"pre_existing": 5})

        assert global_counts == {"pre_existing": 5, "state_worker_0": 1}
        local_tables = vec_env.get_attr("visit_counts")
        assert local_tables[0] == global_counts
        assert local_tables[1] == global_counts
    finally:
        vec_env.close()


# --- Integration: real SubprocVecEnv (actual cross-process boundary) ---

def test_sync_makes_one_workers_count_visible_in_another_process():
    """
    Exercises the actual cross-process boundary the merge protocol depends
    on: a count incremented inside worker 0's OS process must become
    visible inside worker 1's separate OS process after a sync round. Uses
    SubprocVecEnv's default start method (forkserver, falling back to
    spawn), the same one training uses -- so this exercises the exact
    process-creation path the protocol depends on, not a test-only
    shortcut.
    """
    vec_env = SubprocVecEnv([make_stub_env, make_stub_env])
    try:
        # Worker 0's OS process observes a state; worker 1 observes a
        # different one, in its own separate OS process.
        vec_env.env_method("increment", "state_from_worker_0", indices=[0])
        vec_env.env_method("increment", "state_from_worker_1", indices=[1])

        global_counts = sync_visit_counts(vec_env, {})

        assert global_counts == {
            "state_from_worker_0": 1,
            "state_from_worker_1": 1,
        }

        # After the sync, each worker process's own local table must
        # contain the *other* worker's count too -- proof the count
        # crossed the process boundary, not just that the main process
        # computed the right merge.
        local_tables = vec_env.get_attr("visit_counts")
        assert local_tables[0] == global_counts
        assert local_tables[1] == global_counts

        # A second round: worker 1 observes a new state, worker 0 observes
        # nothing new. The previously-synced counts must persist and the
        # new one must still cross over correctly (delta-only pulls, not
        # full-table pulls, are what's being exercised here).
        vec_env.env_method("increment", "state_from_worker_1_round_2", indices=[1])
        global_counts = sync_visit_counts(vec_env, global_counts)

        assert global_counts == {
            "state_from_worker_0": 1,
            "state_from_worker_1": 1,
            "state_from_worker_1_round_2": 1,
        }
        local_tables = vec_env.get_attr("visit_counts")
        assert local_tables[0] == global_counts
        assert local_tables[1] == global_counts
    finally:
        vec_env.close()
