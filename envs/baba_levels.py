import numpy as np
from .game_objects import Object
from .level_utils import get_level_from_json, get_level_from_screenshot
import os.path

def level(number, grid_size=(17, 15)):
    json_name = f"envs/levels/level_{number}.json"
    if os.path.isfile(json_name):
        return get_level_from_json(json_name)

    return get_level_from_screenshot(
        f"envs/levels/level_{number}.png", 
        grid_size, 
        "imgs", 
        should_save=True, 
        output_filename=json_name # saving json for faster loading
    )

# just for testing:
def hardcoded_level_1():
    state = np.zeros((17, 15), dtype='uint8')  # Default background

    # two rows of walls
    for x in range(3, 14):
        state[x, 5] = Object.WALL.value
        state[x, 9] = Object.WALL.value
    
    state[3, 6] = Object.BABA.value
    state[12, 7] = Object.FLAG.value

    # column of rocks between baba and flag
    for y in range(6, 9):
        state[8, y] = Object.ROCK.value
    
    # text rules:
    state[3, 3] = Object.BABA_TEXT.value
    state[4, 3] = Object.IS_TEXT.value
    state[5, 3] = Object.YOU_TEXT.value

    state[11, 3] = Object.FLAG_TEXT.value
    state[12, 3] = Object.IS_TEXT.value
    state[13, 3] = Object.WIN_TEXT.value

    state[3, 11] = Object.WALL_TEXT.value
    state[4, 11] = Object.IS_TEXT.value
    state[5, 11] = Object.STOP_TEXT.value

    state[11, 11] = Object.ROCK_TEXT.value
    state[12, 11] = Object.IS_TEXT.value
    state[13, 11] = Object.PUSH_TEXT.value

    return state