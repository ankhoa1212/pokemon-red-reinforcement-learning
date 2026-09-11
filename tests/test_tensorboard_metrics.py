import os
from pathlib import Path

from stable_baselines3.common.logger import Image
from stable_baselines3.common.vec_env import DummyVecEnv

from tensorboard_callback import TensorBoardCallback
from test_stitching import _make_textured_canvas, _write_png
from test_visit_count_merge import make_stub_env


class StubLogger:
    """Minimal stand-in for SB3's Logger, capturing record() calls.

    Also tracks every call whose value is an SB3 Image separately, so
    tests can assert on "exactly one image was logged" without the
    scalar/image calls' keys colliding in `recorded`.
    """

    def __init__(self):
        self.recorded = {}
        self.image_calls = []

    def record(self, key, value):
        self.recorded[key] = value
        if isinstance(value, Image):
            self.image_calls.append((key, value))


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


def make_map_callback(tmp_path, stitch_sync_interval=1):
    """Builds a TensorBoardCallback wired for the master-map stitching
    sync only. sync_interval is left effectively infinite so the
    unrelated visit-count sync branch never fires and never needs a real
    vec_env -- these tests are exercising _sync_master_map in isolation,
    the same way test_map_stitching_sync.py exercises collect_new_screenshots
    et al. in isolation from this callback.
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    env_data_directory = tmp_path / "env_data"
    callback = TensorBoardCallback(
        log_dir="unused",
        sync_interval=1_000_000,
        stitch_sync_interval=stitch_sync_interval,
        master_map_path=str(images_dir / "master_map.png"),
        master_map_backup_path=str(images_dir / "master_map.prev.png"),
        env_data_directory=str(env_data_directory),
        image_directory="images",
    )
    callback.model = StubModel(StubLogger(), vec_env=None)
    return callback, env_data_directory


def _save_worker_screenshot(env_data_directory, worker_id, filename, image):
    """Mirrors PokemonRedEnv.calculate_fitness's save location: each
    worker gets its own numbered subdirectory under env_data_directory,
    with an "images" subdirectory beneath that."""
    worker_dir = Path(env_data_directory) / str(worker_id) / "images"
    worker_dir.mkdir(parents=True, exist_ok=True)
    return _write_png(worker_dir / filename, image)


def _save_overlapping_screenshot_pair(env_data_directory):
    """Writes two crops of the same textured canvas -- guaranteed to
    stitch successfully, per test_stitching.py's identical construction
    -- into two distinct workers' screenshot directories."""
    canvas = _make_textured_canvas()
    path_a = _save_worker_screenshot(env_data_directory, 0, "a.png", canvas[:, 0:400])
    path_b = _save_worker_screenshot(env_data_directory, 1, "b.png", canvas[:, 250:650])
    return path_a, path_b


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


# --- Master map stitching sync ------------------------------------------
#
# _sync_master_map is exercised directly here (via _on_rollout_end) rather
# than through the visit-count sync's DummyVecEnv/StubCountEnv machinery --
# it runs on its own interval, needs no vec_env round-trip at all (see
# collect_new_screenshots), and reads real screenshots off disk, so a
# tmp_path-based env_data_directory is the more faithful fixture.

# --- Happy path ----------------------------------------------------------

def test_stitch_sync_logs_exactly_one_image_on_success(tmp_path):
    callback, env_data_directory = make_map_callback(tmp_path, stitch_sync_interval=1)
    _save_overlapping_screenshot_pair(env_data_directory)

    callback._on_rollout_end()

    logger = callback.model.logger
    assert len(logger.image_calls) == 1
    key, value = logger.image_calls[0]
    assert key == "env_stats/master_map"
    assert isinstance(value, Image)
    assert os.path.exists(callback.master_map_path)


# --- Edge cases ------------------------------------------------------------

def test_stitch_sync_before_threshold_performs_no_attempt(tmp_path):
    callback, env_data_directory = make_map_callback(tmp_path, stitch_sync_interval=2)
    _save_overlapping_screenshot_pair(env_data_directory)

    callback._on_rollout_end()

    logger = callback.model.logger
    assert logger.image_calls == []
    assert not os.path.exists(callback.master_map_path)


def test_stitch_sync_with_zero_new_screenshots_performs_no_attempt(tmp_path):
    callback, env_data_directory = make_map_callback(tmp_path, stitch_sync_interval=1)
    # env_data_directory exists but no worker has saved anything.

    callback._on_rollout_end()

    logger = callback.model.logger
    assert logger.image_calls == []
    assert not os.path.exists(callback.master_map_path)


# --- Error path ------------------------------------------------------------

def test_stitch_sync_with_failed_stitch_writes_nothing_and_does_not_raise(tmp_path):
    callback, env_data_directory = make_map_callback(tmp_path, stitch_sync_interval=1)
    # A single screenshot with no existing master map: fewer than two real
    # images for cv2.Stitcher, which cv2.Stitcher (and U1's stitch_images)
    # is confirmed to reject with a non-OK status rather than raising.
    canvas = _make_textured_canvas()
    _save_worker_screenshot(env_data_directory, 0, "a.png", canvas)

    callback._on_rollout_end()  # must not raise

    logger = callback.model.logger
    assert logger.image_calls == []
    assert not os.path.exists(callback.master_map_path)


# --- Integration -----------------------------------------------------------

def test_stitch_sync_failed_batch_rolls_forward_into_next_success(tmp_path):
    callback, env_data_directory = make_map_callback(tmp_path, stitch_sync_interval=1)
    canvas = _make_textured_canvas()

    # First sync: only one screenshot exists anywhere -- guaranteed to
    # fail (fewer than two real images) and leave it unconsumed (R4).
    path_a = _save_worker_screenshot(env_data_directory, 0, "a.png", canvas[:, 0:400])
    callback._on_rollout_end()
    assert callback.model.logger.image_calls == []
    assert not os.path.exists(callback.master_map_path)

    # Second sync: a second, overlapping screenshot shows up. The batch
    # must include both the leftover from the failed first sync and this
    # new one, which is exactly enough for the stitch to succeed.
    path_b = _save_worker_screenshot(env_data_directory, 1, "b.png", canvas[:, 250:650])
    callback._on_rollout_end()

    logger = callback.model.logger
    assert len(logger.image_calls) == 1
    assert os.path.exists(callback.master_map_path)
    assert callback._already_stitched_screenshots == {str(path_a), str(path_b)}
