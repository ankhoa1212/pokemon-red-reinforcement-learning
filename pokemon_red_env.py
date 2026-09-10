from collections import Counter
from gymnasium import spaces, Env
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from PIL import Image
from image_checker import hash_screen_state
import uuid
import pandas as pd
from pathlib import Path
from copy import deepcopy
from math import sqrt

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
        self.visit_counts = Counter(settings.get("initial_visit_counts", {}))
        # Counts incremented locally since the last cross-worker sync (see
        # tensorboard_callback.sync_visit_counts). Cleared by
        # pop_visit_count_delta() each time this worker's delta is pulled.
        self._visit_count_delta = Counter()
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
                self.saved_info_directory / Path('trainer_info.csv.gz'), compression='gzip', mode='a')
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
        screen = self._get_obs()["screen"]
        state_hash = hash_screen_state(screen)
        self.visit_counts[state_hash] += 1
        self._visit_count_delta[state_hash] += 1
        visit_count = self.visit_counts[state_hash]
        reward = 1 / sqrt(visit_count)

        if visit_count == 1:
            img = Image.fromarray(screen)
            image_dir = Path(self.saved_info_directory) / self.image_directory
            image_dir.mkdir(exist_ok=True)
            img.save(image_dir / f"{state_hash.hex()}.png")

        self._fitness += reward
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
        self.steps = 0

        return self._get_obs(), {}

    def pop_visit_count_delta(self):
        """
        Returns the visit counts incremented locally since the last call to
        this method, then clears the local delta tracker.

        This is the "pull" half of the cross-worker visit-count merge
        (see tensorboard_callback.sync_visit_counts): the main process
        calls this via VecEnv.env_method on every worker to collect what
        each worker has newly observed since its last sync, without
        needing to transfer that worker's entire (unboundedly growing)
        local table.

        Returns:
            A dict mapping state hash -> count incremented since the last
            pop, i.e. this worker's delta.
        """
        delta = self._visit_count_delta
        self._visit_count_delta = Counter()
        return delta

    def render(self):
        return self.pyboy.screen.image

    def close(self):
        self.pyboy.stop()

    def save_state(self, filename):
        with open(f"{filename}", "wb") as f:
            f.seek(0)
            self.pyboy.save_state(f)