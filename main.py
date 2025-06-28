from pokemon_red_env import PokemonRedEnv
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import CheckpointCallback

IMAGE_DIRECTORY = 'images/'  # directory to images
SCREENSHOT_FILENAME = 'screenshot.png'
MASTER_MAP_FILENAME = 'master_map.png'  # master map of all areas

# TODO consider multiple environments for parallel training

def create_env(env_settings, debug=False):
    env = PokemonRedEnv(settings=env_settings)
    if debug:
        try:
            check_env(env)
        except Exception as e:
            print(f"Environment check failed: {e}")
    return env

if __name__ == "__main__":
    episode_length = 1000  # number of steps per episode
    checkpoint_path = 'checkpoints/'  # path to save checkpoints
    env_settings = {
        "game_path": "pokemon_red.gb",
        "debug": False,
        "frame_rate": 24,
        "map": IMAGE_DIRECTORY + MASTER_MAP_FILENAME,
        "output_shape": (144, 160),
        "max_steps": episode_length,
        "image_directory": IMAGE_DIRECTORY,
        "view": "SDL2"
    }
    environment = create_env(env_settings)

    # TODO implement neural network model

    # checkpoint_callback = CheckpointCallback(save_freq=episode_length, save_path=checkpoint_path,
    #                                  name_prefix='trainer')

    # model = PPO.load()
    while True:
        action = environment.action_space.sample()  # sample a random action
        obs, rewards, terminated, truncated, info = environment.step(action)
        environment.render()