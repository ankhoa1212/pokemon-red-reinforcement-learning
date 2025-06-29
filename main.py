from pokemon_red_env import PokemonRedEnv
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import os
import sys

IMAGE_DIRECTORY = 'images/'  # directory to images
CHECKPOINT_DIRECTORY = 'checkpoints/'  # directory to save checkpoints
SCREENSHOT_FILENAME = 'screenshot.png' # screenshot filename
MASTER_MAP_FILENAME = 'master_map.png'  # master map filename

NUM_CPU = os.cpu_count() if os.cpu_count() is not None else 1
NUM_CPU=1
def create_env(env_settings, env_id=0, debug=False, seed=0):
    env = PokemonRedEnv(settings=env_settings)
    set_random_seed(seed, using_cuda=True)
    if debug:
        try:
            check_env(env)
        except Exception as e:
            print(f"Environment check failed: {e}")
    print(f"Environment {env_id} created")
    env.reset(seed)
    return env

if __name__ == "__main__":
    episode_length = 1000  # number of steps per episode
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
    
    try:
        environments = SubprocVecEnv([lambda: create_env(env_settings, env_id=i) for i in range(NUM_CPU)])
    except Exception as e:
        print(f"SubprocVecEnv failed, falling back to DummyVecEnv: {e}")
        environments = DummyVecEnv([lambda: create_env(env_settings, env_id=i) for i in range(NUM_CPU)])

    checkpoint_callback = CheckpointCallback(save_freq=episode_length, save_path=CHECKPOINT_DIRECTORY,
                                     name_prefix='trainer')

    model = PPO("MultiInputPolicy", environments, n_steps=episode_length, batch_size=episode_length*NUM_CPU,verbose=1)  # initialize PPO model
    print(model.policy)  # print the model's policy architecture
    model.learn(total_timesteps=episode_length*NUM_CPU*10, callback=checkpoint_callback, tb_log_name="trainer_ppo", progress_bar=True)
    # while True:
    #     action = environment.action_space.sample()  # sample a random action
    #     obs, rewards, terminated, truncated, info = environment.step(action)
    #     environment.render()