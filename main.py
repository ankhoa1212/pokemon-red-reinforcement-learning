from pokemon_red_env import PokemonRedEnv
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

IMAGE_DIRECTORY = 'images/'  # directory to images
SCREENSHOT_FILENAME = 'screenshot.png'
MASTER_MAP_FILENAME = 'master_map.png'  # master map of all areas

def create_env(env_settings):
    env = PokemonRedEnv(settings=env_settings)
    # try:
    #     check_env(env, warn=True, skip_render_check=True)
    # except Exception as e:
    #     print(f"Environment check failed: {e}")
    return env

# TODO consider multiple environments for parallel training

if __name__ == "__main__":
    env_settings = {
        "game_path": "pokemon_red.gb",
        "debug": False,
        "frame_rate": 24,
        "map": IMAGE_DIRECTORY + MASTER_MAP_FILENAME,
        "output_shape": (144, 160),
        "max_steps": 1000,
        "image_directory": IMAGE_DIRECTORY,
        "view": "SDL2"
    }
    environment = create_env(env_settings)
    while True:
        action = environment.action_space.sample()  # sample a random action
        obs, rewards, terminated, truncated, info = environment.step(action)
        environment.render()