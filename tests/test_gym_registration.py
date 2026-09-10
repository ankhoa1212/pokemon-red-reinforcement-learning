import importlib
from unittest.mock import patch

import gymnasium as gym

from conftest import default_env_settings, make_frame
from pokemon_red_env import ENV_ID


# --- Happy path ---------------------------------------------------------

def test_gym_make_returns_working_pokemon_red_env(tmp_path):
    settings = default_env_settings(tmp_path)
    with patch("pokemon_red_env.PyBoy") as mock_pyboy_class:
        mock_pyboy_class.return_value.screen.ndarray = make_frame(100)
        env = gym.make(ENV_ID, settings=settings)
        obs, info = env.reset()

    assert "screen" in obs
    assert "last_actions" in obs
    env.close()


# --- Edge case -----------------------------------------------------------

def test_reimporting_module_does_not_raise_duplicate_registration_error():
    import pokemon_red_env

    importlib.reload(pokemon_red_env)
    importlib.reload(pokemon_red_env)


# --- Regression ------------------------------------------------------------

def test_direct_construction_still_works_unchanged(tmp_path):
    from pokemon_red_env import PokemonRedEnv

    settings = default_env_settings(tmp_path)
    with patch("pokemon_red_env.PyBoy") as mock_pyboy_class:
        mock_pyboy_class.return_value.screen.ndarray = make_frame(100)
        env = PokemonRedEnv(settings=settings)

    assert env.observation_space["screen"].shape == settings["output_shape"]
