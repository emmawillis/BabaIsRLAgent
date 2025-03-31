from .game_objects import Object
# import envs
# import pygame
import random
import numpy as np

class Randomizer:
    def __init__(self, level_state):
        self.level_state = level_state

    def reset(self):
        self.new_state = np.zeros((17, 15), dtype='uint8')
        self.objects = {
            Object.BABA.value: 0,
            Object.FLAG.value: 0,
            Object.WALL.value: 0,
            Object.ROCK.value: 0,
            Object.PUSH_TEXT.value: 0,
            Object.STOP_TEXT.value: 0,
            Object.YOU_TEXT.value: 0,
            Object.IS_TEXT.value: 0,
            Object.WIN_TEXT.value: 0,
            Object.BABA_TEXT.value: 0,
            Object.FLAG_TEXT.value: 0,
            Object.ROCK_TEXT.value: 0,
            Object.WALL_TEXT.value: 0,
        }

        # count how many times each object appears
        for row in range(len(self.level_state)):
            for col in range(len(self.level_state[0])):
                obj = self.level_state[row][col]
                if obj != Object.BACKGROUND.value:
                    self.objects[obj] += 1

        self.avail_nouns = []
        for val in [Object.BABA_TEXT.value, Object.FLAG_TEXT.value, Object.WALL_TEXT.value, Object.ROCK_TEXT.value]:
            if self.objects[val] > 0:
                self.avail_nouns.append(val)

        self.avail_adjs = []
        for val in [Object.PUSH_TEXT.value, Object.STOP_TEXT.value, Object.YOU_TEXT.value, Object.WIN_TEXT.value]:
            if self.objects[val] > 0:
                self.avail_adjs.append(val)

        if len(self.avail_nouns) != len(self.avail_adjs):
            return ValueError("The number of nouns and adjectives in the provided level do not match.")
        if len(self.avail_nouns) != self.objects[Object.IS_TEXT.value]:
            return ValueError("The number of nouns/paired adjectives and 'IS' blocks in the provided level do not match.")

    def add_to_state(self, row, col, obj):
        self.new_state[row][col] = obj
        self.objects[obj] -= 1

    def get_random_cell(self, text_string = False):
        if text_string:
            padding = 4
        else:
            padding = 2
        # pad the top and bottom so text isn't trapped
        rand_x = random.randint(1, len(self.new_state) - 2)
        rand_y = random.randint(1, len(self.new_state[0]) - padding)
        while self.new_state[rand_x][rand_y] != Object.BACKGROUND.value:
            rand_x = random.randint(1, len(self.new_state) - 2)
            rand_y = random.randint(1, len(self.new_state[0]) - padding)
        return rand_x, rand_y
    
    def add_obj_to_state(self, obj):
        rand_x, rand_y = self.get_random_cell()
        while self.new_state[rand_x][rand_y] != Object.BACKGROUND.value:
            rand_x, rand_y = self.get_random_cell()
        self.add_to_state(rand_x, rand_y, obj)

    def add_text_string_to_state(self, string: list):
        noun, is_t, adj = string[0], string[1], string[2]
        rand_x, rand_y = self.get_random_cell(text_string=True)
        while self.new_state[rand_x][rand_y + 1] != Object.BACKGROUND.value or self.new_state[rand_x][rand_y + 2] != Object.BACKGROUND.value:
            rand_x, rand_y = self.get_random_cell(text_string=True)
        self.add_to_state(rand_x, rand_y, noun)
        self.add_to_state(rand_x, rand_y + 1, is_t)
        self.add_to_state(rand_x, rand_y + 2, adj)

    def reshuffle_level(self):
        # reset from previous run
        self.reset()
        # create ALL text strings
        while self.avail_nouns:
            noun = random.choice(self.avail_nouns)
            adj = random.choice(self.avail_adjs)
            self.add_text_string_to_state([noun, Object.IS_TEXT.value, adj])
            self.avail_nouns.remove(noun)
            self.avail_adjs.remove(adj)

        # reshuffle the rest of the objects
        for obj in self.objects:
            for _ in range(self.objects[obj]):
                self.add_obj_to_state(obj)


if __name__ == "__main__":
    # randomizer = Randomizer(hardcoded_level_1())
    # randomizer.reshuffle_level()

    # env = envs.BABAWorldEnv(render_mode="human", width=17, height=15, level=randomizer.new_state)
    # env.reset()
    # terminated = False
    # action_map = {
    #     pygame.K_RIGHT: envs.Actions.right.value,
    #     pygame.K_UP: envs.Actions.down.value,
    #     pygame.K_LEFT: envs.Actions.left.value,
    #     pygame.K_DOWN: envs.Actions.up.value 
    # }

    # while not terminated:
    #     action = None
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             terminated = True
    #         if event.type == pygame.KEYDOWN:
    #             if event.key in action_map:
    #                 action = action_map[event.key]
                
    #             if event.key == pygame.K_ESCAPE:
    #                 terminated = True

    #     if action is not None:
    #         result = env.step(action)
    #         terminated = terminated or result[2]

    #     env.render()

    # env.close()
    pass