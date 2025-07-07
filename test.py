from gc import callbacks
from main import create_env
import os
import glob
from stable_baselines3 import PPO
import argparse
from enum import Enum
from typing import Optional
from pathlib import Path
import uuid
import datetime

IMAGE_DIR = 'images/'             # directory to images
CHECKPOINT_DIR = 'checkpoints/'   # directory to save checkpoints
LOG_DIR = 'logs/'                 # directory to save logs
MODEL_DIR = 'models/'             # directory to save models
ENV_DATA_DIR = 'env_data/'        # directory to save environment data

SCREENSHOT_FILENAME = 'screenshot.png'  # screenshot filename
MASTER_MAP_FILENAME = 'master_map.png'  # master map filename

num_cpu = 1  # Number of CPU cores to use, set to 1 for testing

class RunMode(Enum):
    MANUAL = 1
    LOAD_FROM_FILE = 2
    TRAIN_FROM_SCRATCH = 3

def test(episode_length: Optional[int], run_mode=RunMode.MANUAL, debug=False):
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
    if run_mode == RunMode.MANUAL:
        from pyboy import PyBoy
        test = PyBoy(env_settings['game_path'], window=env_settings['view'], debug=debug, sound_emulated=False)
        # with open(os.path.join('start_states', 'fast_off_shift_start.state'), 'rb') as f:
        #     test.load_state(f)
        while test:
            test.tick()

    elif run_mode == RunMode.LOAD_FROM_FILE:
        env = create_env(env_settings)
        print("Running by loading from file.")
        latest_checkpoint = None
        checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, 'trainer_*.zip'))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            print(f"Loading model from checkpoint: {latest_checkpoint}")
            model = PPO.load(latest_checkpoint, env=env)
        obs, info = env.reset()
        while model:
            action, _states = model.predict(obs, deterministic=False)
            obs, rewards, terminated, truncated, info = env.step(action)
            env.render()
    elif run_mode == RunMode.TRAIN_FROM_SCRATCH:
        env = create_env(env_settings)
        print("Training from scratch.")
        model = PPO("MultiInputPolicy", env, n_steps=episode_length, batch_size=2, n_epochs=1, tensorboard_log=LOG_DIR, verbose=1, device='cpu')  # initialize PPO model
        model.learn(total_timesteps=episode_length*num_cpu*5, callback=callbacks, tb_log_name="trainer_ppo", progress_bar=True)
        model.save(os.path.join(MODEL_DIR, 'trainer_ppo_model.zip'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PPO agent on Pokemon Red environment.")
    parser.add_argument('--episode_length', type=int, default=1000000, help='Maximum steps per episode')
    parser.add_argument('--run_mode', type=RunMode, default=RunMode.MANUAL, help='Select a run mode', choices=list(RunMode))
    parser.add_argument('--debug', type=bool, default=False, help='Enable debug mode')
    args = parser.parse_args()

    test(args.episode_length, run_mode=args.run_mode, debug=args.debug)