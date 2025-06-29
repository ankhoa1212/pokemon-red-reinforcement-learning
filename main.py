from pokemon_red_env import PokemonRedEnv
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import os
import sys
import glob

IMAGE_DIRECTORY = 'images/'  # directory to images
CHECKPOINT_DIRECTORY = 'checkpoints/'  # directory to save checkpoints
SCREENSHOT_FILENAME = 'screenshot.png' # screenshot filename
MASTER_MAP_FILENAME = 'master_map.png'  # master map filename

NUM_CPU = os.cpu_count() if os.cpu_count() is not None else 1
NUM_CPU-=2
def create_env(env_settings, env_id=0, debug=False, seed=0):
    set_random_seed(seed, using_cuda=True)
    env = PokemonRedEnv(settings=env_settings)
    env.reset(seed+env_id)
    if debug:
        try:
            check_env(env)
        except Exception as e:
            print(f"Environment check failed: {e}")
    print(f"Environment {env_id} created")
    return env

def test():
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
    env = create_env(env_settings)
    latest_checkpoint = None
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIRECTORY, 'trainer_*.zip'))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print(f"Loading model from checkpoint: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=environments)
    while model:
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()

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

    checkpoint_callback = CheckpointCallback(save_freq=episode_length // 2, save_path=CHECKPOINT_DIRECTORY,
                                     name_prefix='trainer')

    latest_checkpoint = None
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIRECTORY, 'trainer_*.zip'))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print(f"Loading model from checkpoint: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=environments)
    else:
        model = PPO("MultiInputPolicy", environments, n_steps=episode_length, batch_size=episode_length*NUM_CPU,verbose=1)  # initialize PPO model

    model.learn(total_timesteps=episode_length*NUM_CPU*30, callback=checkpoint_callback, tb_log_name="trainer_ppo", progress_bar=True)
