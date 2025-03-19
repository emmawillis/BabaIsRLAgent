import json
import numpy as np
from game_objects import Object
from level_utils import get_level_from_screenshot

def level_1():
    return get_level_from_screenshot("envs/levels/level_1.png", (33, 18), "imgs")


# just for testing:
def hardcoded_level_1():
    state = np.zeros((33, 18), dtype='uint8')  # Default background

    # two rows of walls
    for x in range(11, 22):
        state[x, 7] = Object.WALL.value
        state[x, 11] = Object.WALL.value
    
    state[12, 9] = Object.BABA.value
    state[20, 9] = Object.FLAG.value

    # column of rocks between baba and flag
    for y in range(8, 11):
        state[16, y] = Object.ROCK.value
    
    # text rules:
    state[11, 5] = Object.BABA_TEXT.value
    state[12, 5] = Object.IS_TEXT.value
    state[13, 5] = Object.YOU_TEXT.value

    state[19, 5] = Object.FLAG_TEXT.value
    state[20, 5] = Object.IS_TEXT.value
    state[21, 5] = Object.WIN_TEXT.value

    state[11, 13] = Object.WALL_TEXT.value
    state[12, 13] = Object.IS_TEXT.value
    state[13, 13] = Object.STOP_TEXT.value

    state[19, 13] = Object.ROCK_TEXT.value
    state[20, 13] = Object.IS_TEXT.value
    state[21, 13] = Object.PUSH_TEXT.value

    return state