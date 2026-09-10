import importlib
import tempfile
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv

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


def _make_registered_env_in_subprocess():
    """Runs inside a SubprocVecEnv worker subprocess: builds a real env
    through create_env's actual gym.make() path (PyBoy mocked locally,
    since a mock applied in the parent process can't cross the process
    boundary), proving U4's import-time registration reaches this worker
    the same way main.py's real training path relies on."""
    from main import create_env

    settings = default_env_settings(Path(tempfile.mkdtemp()))
    with patch("pokemon_red_env.PyBoy") as mock_pyboy_class:
        mock_pyboy_class.return_value.screen.ndarray = make_frame(100)
        return create_env(settings)


def test_registration_reaches_real_subprocvecenv_workers():
    vec_env = SubprocVecEnv(
        [_make_registered_env_in_subprocess, _make_registered_env_in_subprocess]
    )
    try:
        # If registration hadn't reached these worker processes, gym.make()
        # inside _make_registered_env_in_subprocess would have raised
        # NameNotFound and SubprocVecEnv construction itself would have
        # failed -- reaching this line is already the proof. Confirming a
        # real attribute read closes the loop end-to-end.
        visit_counts = vec_env.get_attr("visit_counts")
        assert visit_counts == [{}, {}]
    finally:
        vec_env.close()


# --- Regression ------------------------------------------------------------

def test_direct_construction_still_works_unchanged(tmp_path):
    from pokemon_red_env import PokemonRedEnv

    settings = default_env_settings(tmp_path)
    with patch("pokemon_red_env.PyBoy") as mock_pyboy_class:
        mock_pyboy_class.return_value.screen.ndarray = make_frame(100)
        env = PokemonRedEnv(settings=settings)

    assert env.observation_space["screen"].shape == settings["output_shape"]
