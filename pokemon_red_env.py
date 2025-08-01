from gymnasium import spaces, Env
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from PIL import Image
from image_checker import stitch_images, compare_images
import uuid
import pandas as pd
from pathlib import Path
from copy import deepcopy
import os

class PokemonRedEnv(Env):

    def __init__(self, settings=None):
        super().__init__()
        self.id = uuid.uuid4()
        self.game_path = settings["game_path"]
        self.start_state_path = settings["start_state_path"]
        self.image_directory = settings["image_directory"]
        self.env_data_directory = settings["env_data_directory"]
        self.saved_info_directory = self.env_data_directory + str(self.id) + "/"
        self.save_info = settings["save_info"]
        self.debug = settings["debug"]
        self.view = settings["view"]
        self.frame_rate = settings["frame_rate"]
        self.frames_to_track = 1
        self.map = np.array(Image.open(fp=settings["map"]).convert("L"))
        self.steps = 0
        self.max_steps = settings["max_steps"]
        self.memory = []
        self.info = []
        self._fitness = 0
        self._previous_fitness = 0
        self.output_shape = settings["output_shape"]

        Path(self.env_data_directory).mkdir(exist_ok=True)
        Path(self.saved_info_directory).mkdir(exist_ok=True)

        self.actions = [            
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
        ]

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Dict({
            "screen": spaces.Box(low=0, high=255, shape=settings["output_shape"], dtype=np.uint8),
            "last_actions": spaces.MultiDiscrete([len(self.actions)] * self.frames_to_track)
        })

        self.pyboy = PyBoy(self.game_path, window=self.view, sound_emulated=False)
        if not self.debug:
            self.pyboy.set_emulation_speed(6)


    def step(self, action):
        if action is not None:
            self.do_action(action)
        reward=self.calculate_fitness()
        observation=self._get_obs()
        self.steps += 1
        if self.debug:
            print(f"Step: {self.steps}/{self.max_steps}, Fitness: {self._fitness}, Reward: {reward}, Id: {self.id}")
        info = {
            "steps": deepcopy(self.steps),
            "fitness": deepcopy(self._fitness),
            "reward": deepcopy(reward),
            "action": deepcopy(int(action)) if action is not None else None,
            "last_actions": deepcopy(self.last_actions),
        }
        self.info.append(info)

        terminated = False
        truncated = self.truncated_check()
        if (terminated or truncated) and self.save_info:
            pd.DataFrame(self.info).to_csv(
                self.saved_info_directory / Path(f'trainer_info.csv.gz'), compression='gzip', mode='a')
        return observation, reward, terminated, truncated, info

    def truncated_check(self):
        return self.steps >= self.max_steps

    def pre_truncated_check(self):
        return self.steps >= self.max_steps - 1

    def update_actions(self, action):
        self.last_actions = np.roll(self.last_actions, shift=1)
        self.last_actions[0] = action

    def _get_obs(self):
        observation = {
            "screen": self.pyboy.screen.ndarray[:, :, 0].astype(np.uint8),
            "last_actions": self.last_actions}
        return observation

    def do_action(self, action):
        self.pyboy.send_input(self.actions[action])
        self.update_actions(action)
        for i in range(self.frame_rate):
            if i == 8:
                self.pyboy.send_input(self.release_actions[action])
            self.pyboy.tick()

    def calculate_fitness(self):
        self._previous_fitness=self._fitness
        difference = 0
        img = self._get_obs()["screen"]
        img = Image.fromarray(img)
        Path(f"{self.saved_info_directory}{self.image_directory}").mkdir(exist_ok=True)
        if not self.memory:
            img.save(f"{self.saved_info_directory}{self.image_directory}{len(self.memory)}.png")
            self.memory.append(img)
        else:
            ind = 0
            min_difference = 1
            for i, test_image in enumerate(self.memory):
                similarity = compare_images(np.array(img), np.array(test_image))
                test_difference = 1 - similarity
                if test_difference > difference:
                    ind = i
                    difference = test_difference
                else:
                    min_difference = min(min_difference, test_difference)

            if difference > 0.8 and min_difference > 0.2:  # threshold for saving images
                difference = max(0, min(difference, 1))
                img.save(f"{self.saved_info_directory}{self.image_directory}{len(self.memory)}_{ind}_{difference}.png")
                self.memory.append(img)
        self._fitness += difference
        return self._fitness-self._previous_fitness

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)

        initial_state = self.start_state_path
        with open(initial_state, "rb") as f:
            self.pyboy.load_state(f)

        self._fitness=0
        self._previous_fitness=0

        self.last_actions = np.zeros((self.frames_to_track,), dtype=np.uint8)
        self.info = []
        self.memory = []
        self.steps = 0

        return self._get_obs(), {}

    def render(self):
        return self.pyboy.screen.image

    def close(self):
        self.pyboy.stop()

    def save_state(self, filename):
        with open(f"{filename}", "wb") as f:
            f.seek(0)
            self.pyboy.save_state(f)