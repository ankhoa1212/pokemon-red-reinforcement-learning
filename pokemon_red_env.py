from gymnasium import spaces, Env
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from PIL import Image
from image_checker import stitch_images, compare_images
import uuid

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
        self.id = uuid.uuid4()
        self.memory = []
        self.steps = 0
        self.frames_to_track = 1

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

        self.pyboy = PyBoy(self.game_path)
        if not self.debug:
            self.pyboy.set_emulation_speed(0)
        self.reset()


    def step(self, action):
        self.do_action(action)
        reward=self._calculate_fitness()
        done = self.steps >= self.max_steps
        observation=self._get_obs()
        self.steps += 1
        print(f"Step: {self.steps}/{self.max_steps}, Fitness: {self._fitness}, Reward: {reward}, Id: {self.id}")

        info = {}
        truncated = False
        return observation, reward, done, truncated, info

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

    def _calculate_fitness(self):
        self._previous_fitness=self._fitness
        with self.pyboy.screen.image as img:
            if not self.memory:
                self.memory.append(img)
            else:
                difference = 1 - compare_images(np.array(img), np.array(self.memory[-1]))
                self._fitness += difference
                if difference > 0.5:  # threshold for saving images
                    img.save(f"{self.image_directory}{self.id}/{self.steps}.png")
                    self.memory.append(img)
        return self._fitness-self._previous_fitness

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.pyboy = PyBoy(self.game_path, window=self.view, debug=self.debug, sound_emulated=False)

        self._fitness=0
        self._previous_fitness=0

        self.last_actions = np.zeros((self.frames_to_track,), dtype=np.uint8)

        return self._get_obs(), {}

    def render(self):
        return self.pyboy.screen.image

    def close(self):
        self.pyboy.stop()