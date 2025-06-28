from pokemon_red_env import PokemonRedEnv
from gymnasium.utils.env_checker import check_env

IMAGE_DIRECTORY = 'images/'  # directory to images
SCREENSHOT_FILENAME = 'screenshot.png'
MASTER_MAP_FILENAME = 'master_map.png'  # master map of all areas

map = {}  # map of area names to coordinate and corresponding image files
pos = (0, 0)

def create_env(env_settings):
    env = PokemonRedEnv(settings=env_settings)
    try:
        check_env(env)
    except Exception as e:
        print(f"Environment check failed: {e}")
    return env

# TODO consider multiple environments for parallel training

if __name__ == "__main__":
    env_settings = {
        "game_path": "pokemon_red.gb",
        "debug": False,
        "frames_per_action": 24,
        "map": IMAGE_DIRECTORY + MASTER_MAP_FILENAME,
        "output_shape": (160, 144, 1),
        "max_steps": 1000000
    }
    environment = create_env(env_settings)