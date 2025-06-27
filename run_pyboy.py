import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
import cv2

IMAGE_DIRECTORY = 'images/'  # directory to images
SCREENSHOT_FILENAME = 'screenshot.png'
MASTER_MAP_FILENAME = 'master_map.png'  # master map of all areas
map = {}  # map of area names to coordinate and corresponding image files
pos = (0, 0)

pyboy = PyBoy('pokemon_red.gb')
pyboy.set_emulation_speed(6)

button_presses = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

button_releases = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START
        ]

def get_screen(pyboy):
    screen = pyboy.screen.ndarray
    return screen.shape

def get_area(pyboy):
    return pyboy.game_area()

while True:
    pil_image = pyboy.screen.image
    pil_image.save(IMAGE_DIRECTORY + SCREENSHOT_FILENAME)
    pyboy.tick()