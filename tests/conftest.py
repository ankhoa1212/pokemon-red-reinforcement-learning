from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Typical Game Boy screen resolution (rows, cols); pyboy's screen.ndarray has
# a trailing color-channel axis that PokemonRedEnv._get_obs slices down to
# one channel.
SCREEN_HEIGHT = 144
SCREEN_WIDTH = 160


def make_frame(fill_value: int) -> np.ndarray:
    """Builds a synthetic multi-channel screen filled with one value, shaped
    like the raw array PyBoy exposes before _get_obs slices it down."""
    return np.full((SCREEN_HEIGHT, SCREEN_WIDTH, 4), fill_value, dtype=np.uint8)


def default_env_settings(tmp_path, **overrides):
    """Builds a PokemonRedEnv settings dict pointed at a pytest tmp_path,
    with sensible defaults for tests that don't care about most fields."""
    settings = {
        "game_path": "pokemon_red.gb",
        "debug": False,
        "frame_rate": 24,
        "output_shape": (144, 160),
        "max_steps": 1000,
        "image_directory": "images/",
        "view": None,
        "env_data_directory": str(tmp_path / "env_data") + "/",
        "start_state_path": str(REPO_ROOT / "start_states" / "fast_off_set_start.state"),
        "save_info": False,
    }
    settings.update(overrides)
    return settings
