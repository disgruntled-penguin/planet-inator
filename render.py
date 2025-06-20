import time
from reboundsim import Simulation
import pygame



WIDTH = 1000
HEIGHT = 1000
VIEWPORT_HALF_SIZE = 50 #change this for closer look at inner planets (=3) (have to add zoom)
# VIEWPORT_HALF_SIZE = 31
# VIEWPORT_HALF_SIZE = 14
VIEWPORT = ((-VIEWPORT_HALF_SIZE, -VIEWPORT_HALF_SIZE), (VIEWPORT_HALF_SIZE, VIEWPORT_HALF_SIZE))

SIZE = WIDTH, HEIGHT
VIEWPORT_WIDTH = VIEWPORT[1][0] - VIEWPORT[0][0]
VIEWPORT_HEIGHT = VIEWPORT[1][1] - VIEWPORT[0][1]


def viewport_to_pixels(x, y):
    if x < VIEWPORT[0][0] or x > VIEWPORT[1][0]: 
        return None
    if y < VIEWPORT[0][1] or y > VIEWPORT[1][1]:
        return None

    px = ((x - VIEWPORT[0][0]) * WIDTH) / VIEWPORT_WIDTH
    py = ((VIEWPORT_HEIGHT - (y - VIEWPORT[0][1])) * HEIGHT) / VIEWPORT_HEIGHT

    return px, py


