from unittest.mock import patch

from stable_baselines3.common.vec_env import DummyVecEnv

from conftest import default_env_settings, make_frame
from main import create_env


# --- Integration ---------------------------------------------------------

def test_create_env_gives_every_dummy_vec_env_instance_the_same_initial_seed(tmp_path):
    """The seed value placed in env_settings["initial_visit_counts"] must
    reach every parallel environment instance constructed via create_env,
    matching the DummyVecEnv-only integration scenario for this wiring."""
    initial_visit_counts = {b"seed-state": 7}
    env_settings = default_env_settings(tmp_path, initial_visit_counts=initial_visit_counts)

    with patch("pokemon_red_env.PyBoy") as mock_pyboy_class:
        mock_pyboy_class.return_value.screen.ndarray = make_frame(100)
        vec_env = DummyVecEnv(
            [
                lambda i=i: create_env(env_settings, env_id=i)
                for i in range(3)
            ]
        )

    try:
        visit_counts_per_env = vec_env.get_attr("visit_counts")
        assert len(visit_counts_per_env) == 3
        for visit_counts in visit_counts_per_env:
            assert visit_counts == initial_visit_counts
    finally:
        vec_env.close()


def test_create_env_seeded_instances_do_not_share_object_identity(tmp_path):
    """Each worker's env_settings closure is pickled/constructed
    independently, so each PokemonRedEnv's visit_counts must be its own
    dict, equal in value but not the same object -- otherwise a mutation in
    one worker would leak into another in-process instance before any real
    merge sync happens."""
    initial_visit_counts = {b"seed-state": 3}
    env_settings = default_env_settings(tmp_path, initial_visit_counts=initial_visit_counts)

    with patch("pokemon_red_env.PyBoy") as mock_pyboy_class:
        mock_pyboy_class.return_value.screen.ndarray = make_frame(50)
        env_a = create_env(env_settings, env_id=0)
        env_b = create_env(env_settings, env_id=1)

    # .unwrapped reaches the underlying PokemonRedEnv directly: plain
    # attribute access on the gym.make()-returned object itself does not
    # forward through Gymnasium's wrapper chain (only get_wrapper_attr
    # does), so tests reading internal attributes must go through it too.
    assert env_a.unwrapped.visit_counts == env_b.unwrapped.visit_counts
    assert env_a.unwrapped.visit_counts is not env_b.unwrapped.visit_counts
