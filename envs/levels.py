import numpy as np

from grid_world import Object
def level_1():
        state = np.zeros((33, 18), dtype=np.uint8)  # Default background
    
        rules = []
        # two rows of walls
        for x in range(12, 23):
            state[x, 6] = Object.WALL
            state[x, 12] = Object.WALL
        
        state[13, 10] = Object.BABA
        state[21, 10] = Object.FLAG

        # column of rocks between baba and flag
        for y in range(9, 12):
            state[17, y] = Object.ROCK
        
        # text rules:
        state[12, 6] = Object.BABA_TEXT
        state[13, 6] = Object.IS_TEXT
        state[14, 6] = Object.YOU_TEXT
        rules.add(Object.BABA, "you")

        state[20, 6] = Object.FLAG_TEXT
        state[21, 6] = Object.IS_TEXT
        state[22, 6] = Object.WIN_TEXT
        rules.add(Object.FLAG, "win")

        state[12, 14] = Object.WALL_TEXT
        state[13, 14] = Object.IS_TEXT
        state[14, 14] = Object.STOP_TEXT
        rules.add(Object.WALL, "stop")

        state[20, 14] = Object.ROCK_TEXT
        state[21, 14] = Object.IS_TEXT
        state[22, 14] = Object.PUSH_TEXT
        rules.add(Object.ROCK, "push")

        return state, rules