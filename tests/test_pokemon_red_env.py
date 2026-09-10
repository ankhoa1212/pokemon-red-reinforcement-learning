from math import sqrt
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import image_checker
from conftest import SCREEN_HEIGHT, SCREEN_WIDTH, default_env_settings, make_frame
from image_checker import hash_screen_state
from pokemon_red_env import PokemonRedEnv


def make_random_frame(seed: int) -> np.ndarray:
    """Builds a synthetic multi-channel screen of random noise, so distinct
    seeds are (with overwhelming probability) genuinely distinct states."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(SCREEN_HEIGHT, SCREEN_WIDTH, 4)).astype(np.uint8)


def build_env(tmp_path, fill_value=100, **setting_overrides):
    """Constructs a PokemonRedEnv with PyBoy mocked out, seeded with a
    synthetic screen frame."""
    settings = default_env_settings(tmp_path, **setting_overrides)
    with patch("pokemon_red_env.PyBoy") as mock_pyboy_class:
        mock_pyboy_class.return_value.screen.ndarray = make_frame(fill_value)
        env = PokemonRedEnv(settings=settings)
    env.last_actions = np.zeros((env.frames_to_track,), dtype=np.uint8)
    return env


def image_dir_for(env) -> Path:
    return Path(env.saved_info_directory) / env.image_directory


# --- Happy path ---------------------------------------------------------

def test_first_visit_yields_max_per_visit_reward(tmp_path):
    env = build_env(tmp_path, fill_value=100)

    reward = env.calculate_fitness()

    assert reward == pytest.approx(1.0)


def test_revisit_yields_strictly_smaller_reward(tmp_path):
    env = build_env(tmp_path, fill_value=100)

    first_reward = env.calculate_fitness()
    second_reward = env.calculate_fitness()

    assert second_reward < first_reward


# --- Integration ---------------------------------------------------------

def test_step_reward_equals_one_over_sqrt_count(tmp_path):
    env = build_env(tmp_path, fill_value=50)
    env.reset()

    _obs, reward, _terminated, _truncated, _info = env.step(0)

    screen = env._get_obs()["screen"]
    state_hash = hash_screen_state(screen)
    count = env.visit_counts[state_hash]
    assert reward == pytest.approx(1 / sqrt(count))


def test_visit_count_survives_reset(tmp_path):
    env = build_env(tmp_path, fill_value=77)
    env.reset()
    env.step(0)

    screen = env._get_obs()["screen"]
    state_hash = hash_screen_state(screen)
    count_before_reset = env.visit_counts[state_hash]

    env.reset()
    env.step(0)

    count_after_reset = env.visit_counts[state_hash]
    assert count_after_reset == count_before_reset + 1
    assert count_after_reset > 1


def test_first_visit_writes_screenshot_revisit_does_not_duplicate(tmp_path):
    env = build_env(tmp_path, fill_value=33)
    env.reset()

    env.step(0)
    screen = env._get_obs()["screen"]
    state_hash = hash_screen_state(screen)
    expected_file = image_dir_for(env) / f"{state_hash.hex()}.png"
    assert expected_file.exists()

    env.step(0)  # revisit: same mocked frame, same hash
    matching_files = list(image_dir_for(env).glob(f"{state_hash.hex()}*"))
    assert len(matching_files) == 1


# --- Edge case -------------------------------------------------------------

def test_many_distinct_states_keep_count_table_correct(tmp_path):
    env = build_env(tmp_path, fill_value=0)
    env.reset()

    num_steps = 50
    for i in range(num_steps):
        env.pyboy.screen.ndarray = make_random_frame(seed=i)
        env.step(0)

    # Every step increments exactly one hash's count by one, so the total
    # of all counts must equal the number of steps taken, regardless of
    # any hash collisions among the random frames.
    assert sum(env.visit_counts.values()) == num_steps
    # With random noise frames, distinct seeds are overwhelmingly likely to
    # hash to distinct state cells.
    assert len(env.visit_counts) > 1


# --- Regression --------------------------------------------------------

def test_compare_images_removed_from_image_checker():
    assert not hasattr(image_checker, "compare_images")


def test_env_has_no_memory_attribute(tmp_path):
    env = build_env(tmp_path)
    assert not hasattr(env, "memory")
