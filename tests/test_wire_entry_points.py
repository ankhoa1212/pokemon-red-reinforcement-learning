from pathlib import Path
from unittest.mock import patch

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from main import create_env

REPO_ROOT = Path(__file__).resolve().parent.parent

SCREEN_HEIGHT = 144
SCREEN_WIDTH = 160


def make_frame(fill_value: int) -> np.ndarray:
    """Builds a synthetic multi-channel screen filled with one value."""
    return np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 4), fill_value, dtype=np.uint8)


def make_env_settings(tmp_path, initial_visit_counts):
    return {
        "game_path": "pokemon_red.gb",
        "debug": False,
        "frame_rate": 24,
        "map": str(REPO_ROOT / "images" / "master_map.png"),
        "output_shape": (144, 160),
        "max_steps": 1000,
        "image_directory": "images/",
        "view": None,
        "env_data_directory": str(tmp_path / "env_data") + "/",
        "start_state_path": str(REPO_ROOT / "start_states" / "fast_off_set_start.state"),
        "save_info": False,
        "initial_visit_counts": initial_visit_counts,
    }


# --- Integration ---------------------------------------------------------

def test_create_env_gives_every_dummy_vec_env_instance_the_same_initial_seed(tmp_path):
    """The seed value placed in env_settings["initial_visit_counts"] (U4's
    wiring) must reach every parallel environment instance constructed via
    create_env, matching the plan's DummyVecEnv-only integration scenario
    for this unit."""
    initial_visit_counts = {b"seed-state": 7}
    env_settings = make_env_settings(tmp_path, initial_visit_counts)

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
    independently (see plan's U4 approach note), so each PokemonRedEnv's
    visit_counts must be its own dict, equal in value but not the same
    object -- otherwise a mutation in one worker would leak into another
    in-process instance before any real merge sync happens."""
    initial_visit_counts = {b"seed-state": 3}
    env_settings = make_env_settings(tmp_path, initial_visit_counts)

    with patch("pokemon_red_env.PyBoy") as mock_pyboy_class:
        mock_pyboy_class.return_value.screen.ndarray = make_frame(50)
        env_a = create_env(env_settings, env_id=0)
        env_b = create_env(env_settings, env_id=1)

    assert env_a.visit_counts == env_b.visit_counts
    assert env_a.visit_counts is not env_b.visit_counts
