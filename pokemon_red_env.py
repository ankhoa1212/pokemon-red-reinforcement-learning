# Adopted from https://github.com/NicoleFaye/PyBoy/blob/rl-test/PokemonPinballEnv.py
from gymnasium import spaces, Env
import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from PIL import Image
from image_stitching import stitch_images


class PokemonRedEnv(Env):

    def __init__(self, settings=None):
        super().__init__()
        self.game_path = settings["game_path"]
        self._fitness=0
        self._previous_fitness=0
        self.debug = settings["debug"]
        self.frames_per_action = settings["frames_per_action"]
        self.map = np.array(Image.open(fp=settings["map"]).convert("L"))
        self.max_steps = settings["max_steps"]
        self.output_shape = settings["output_shape"]
        self.steps = 0

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
        self.observation_space = spaces.Box(low=0, high=255, shape=settings["output_shape"], dtype=np.uint8)
        
        self.pyboy = PyBoy(self.game_path)

        if not self.debug:
            self.pyboy.set_emulation_speed(6)

        self.reset()
        print("Pokemon Red environment initialized.")


    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self._do_action(action)

        # TODO consider disabling renderer when not needed to improve speed

        done = self.steps >= self.max_steps

        self._calculate_fitness()
        reward=self._fitness-self._previous_fitness

        observation=self._get_obs()
        info = {}
        truncated = False

        return observation, reward, done, truncated, info

    def _get_obs(self):
        return self.render()

    def _do_action(self, action):
        self.pyboy.send_input(self.actions[action])
        
        release_action = self.release_actions[action]
        for i in range(self.frames_per_action):
            if i == 8:
                self.pyboy.send_input(release_action)
            self.pyboy.tick()

    def _calculate_fitness(self):
        self._previous_fitness=self._fitness

        # NOTE: Only some game wrappers will provide a score
        # If not, you'll have to investigate how to score the game yourself
        new_image = stitch_images(self.render(np_array=True), self.map)
        if new_image:
            self.map = new_image
            self.steps += 1
            self._fitness += 1

    def reset(self, seed=0, **kwargs):
        # restart the game
        self.pyboy = PyBoy(self.game_path)
        with open("pokemon_red.gb.state", 'rb') as f:
            self.pyboy.load_state(f)
        self._fitness=0
        self._previous_fitness=0

        observation=self._get_obs()
        info = {}
        return observation, info

    def render(self, mode='human', np_array=False):
        if np_array:
            return np.array(self.pyboy.screen.ndarray)[:, :, 0]
        return self.pyboy.screen.ndarray[:, :, 0]

    def close(self):
        self.pyboy.stop()