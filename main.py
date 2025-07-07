from pokemon_red_env import PokemonRedEnv
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from tensorboard_callback import TensorBoardCallback
import os
import sys
import glob
import uuid
import datetime

IMAGE_DIR = 'images/'             # directory to images
CHECKPOINT_DIR = 'checkpoints/'   # directory to save checkpoints
LOG_DIR = 'logs/'                 # directory to save logs
MODEL_DIR = 'models/'             # directory to save models
ENV_DATA_DIR = 'env_data/'        # directory to save environment data

SCREENSHOT_FILENAME = 'screenshot.png'  # screenshot filename
MASTER_MAP_FILENAME = 'master_map.png'  # master map filename

NUM_CPU = os.cpu_count() if os.cpu_count() is not None else 1

def create_env(env_settings, env_id=0, debug=False, seed=0):
    set_random_seed(seed)
    env = PokemonRedEnv(settings=env_settings)
    env.reset(seed+env_id)
    if debug:
        try:
            check_env(env)
        except Exception as e:
            print(f"Environment check failed: {e}")
    return env      

if __name__ == "__main__":
    episode_length = 10  # number of steps per episode
    run_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4()}"  # unique run identifier with timestamp
    env_settings = {
        "game_path": "pokemon_red.gb",
        "debug": True,
        "frame_rate": 24,
        "map": IMAGE_DIR + MASTER_MAP_FILENAME,
        "output_shape": (144, 160),
        "max_steps": episode_length,
        "image_directory": IMAGE_DIR,
        "view": "SDL2",
        "env_data_directory": ENV_DATA_DIR + run_id + "/",
        "start_state_path": "start_states/fast_off_set_start.state",
        "save_info": True,
    }
    
    try:
        # raise Exception("Forcing DummyVecEnv for testing purposes")
        environments = SubprocVecEnv([lambda: create_env(env_settings, env_id=i) for i in range(NUM_CPU)])
    except Exception as e:
        print(f"SubprocVecEnv failed, falling back to DummyVecEnv: {e}")
        environments = DummyVecEnv([lambda: create_env(env_settings, env_id=i) for i in range(NUM_CPU)])

    checkpoint_callback = CheckpointCallback(save_freq=episode_length//2, save_path=CHECKPOINT_DIR,
                                     name_prefix='trainer', verbose=1)
    # eval_callback = EvalCallback(environments, best_model_save_path=MODEL_DIR)

    callbacks = CallbackList([checkpoint_callback, TensorBoardCallback(CHECKPOINT_DIR, verbose=1)])

    # latest_checkpoint = None
    # checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIRECTORY, 'trainer_*.zip'))
    # if checkpoint_files:
    #     latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    #     print(f"Loading model from checkpoint: {latest_checkpoint}")
    #     model = PPO.load(latest_checkpoint, env=environments)
    #     model.n_steps = episode_length
    #     model.n_envs = NUM_CPU
    #     model.rollout_buffer.buffer_size = episode_length
    #     model.rollout_buffer.n_envs = NUM_CPU
    #     model.rollout_buffer.reset()
    # else:
    #     model = PPO("MultiInputPolicy", environments, n_steps=episode_length, batch_size=episode_length*NUM_CPU,tensorboard_log=LOG_DIRECTORY, verbose=1)  # initialize PPO model

    model = PPO("MultiInputPolicy", environments, n_steps=episode_length, batch_size=2, n_epochs=1, tensorboard_log=CHECKPOINT_DIR, verbose=1, device='cpu')  # initialize PPO model
    model.learn(total_timesteps=episode_length*NUM_CPU*5, callback=callbacks, tb_log_name="trainer_ppo", progress_bar=True)
    model.save(os.path.join(MODEL_DIR, 'trainer_ppo_model.zip'))