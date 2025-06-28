# Adopted from https://github.com/NicoleFaye/PyBoy/blob/rl-test/PokemonPinballEnv.py
from gymnasium import spaces, Env
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from PIL import Image
from image_checker import stitch_images, compare_images


class PokemonRedEnv(Env):

    def __init__(self, settings=None):
        super().__init__()
        self.game_path = settings["game_path"]
        self._fitness=0
        self._previous_fitness=0
        self.debug = settings["debug"]
        self.frame_rate = settings["frame_rate"]
        self.map = np.array(Image.open(fp=settings["map"]).convert("L"))
        self.max_steps = settings["max_steps"]
        self.output_shape = settings["output_shape"]
        self.image_directory = settings["image_directory"]
        self.view = settings.get("view", "null")
        self.steps = 0
        self.memory = []

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
        self.observation_space = spaces.Box(low=np.zeros(shape=settings["output_shape"]), high=np.full(shape=settings["output_shape"], fill_value=255), dtype=np.uint8)

        self.pyboy = PyBoy(self.game_path)

        if not self.debug:
            self.pyboy.set_emulation_speed(1)

        self.reset()
        print("Pokemon Red environment initialized.")


    def step(self, action):
        self._do_action(action)

        done = self.steps >= self.max_steps

        self._calculate_fitness()
        reward=self._fitness-self._previous_fitness

        observation=self._get_obs()
        info = {}
        truncated = False

        self.steps += 1
        print(f"Step: {self.steps}/{self.max_steps}, Fitness: {self._fitness}, Reward: {reward}")

        return observation, reward, done, truncated, info

    def _get_obs(self):
        return self.pyboy.screen.ndarray[:, :, 0].astype(np.uint8)

    def _do_action(self, action):
        self.pyboy.send_input(self.actions[action])
        for i in range(self.frame_rate):
            if i == 8:
                self.pyboy.send_input(self.release_actions[action])
            self.pyboy.tick()

    def _calculate_fitness(self):
        self._previous_fitness=self._fitness
        with self.pyboy.screen.image as img:
            if not self.memory:
                self.memory.append(img)
            else:
                difference = 1 - compare_images(np.array(img), np.array(self.memory[-1]))
                self._fitness += difference
                if difference > 0.5:  # threshold for saving images
                    img.save(f"{self.image_directory}/{self.steps}.png")
                    self.memory.append(img)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.pyboy = PyBoy(self.game_path, window=self.view, debug=self.debug)

        self._fitness=0
        self._previous_fitness=0

        return self._get_obs(), {}

    def render(self):
        pass
        # return self._get_obs()

    def close(self):
        self.pyboy.stop()