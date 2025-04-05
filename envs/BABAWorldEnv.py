from .baba_levels import level_grid, hardcoded_level_1
from .level_randomization import Randomizer
import gymnasium as gym
from gymnasium.spaces import Dict, Sequence, Box, Discrete
import pygame
import numpy as np
from .game_objects import Actions, Object, ObjectState

class BABAWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode="human", width=17, height=15, level=1, 
                 train=True, object_to_shuffle: int=Object.BABA.value, 
                 rewards = ["winlose", "nochange", "movetext", "distance"]
        ):
        self.width = width
        self.height = height
        self.level = level
        if "all" in rewards:
            self.rewards = ["winlose", "nochange", "movetext", "distance"]
        else:
            self.rewards = rewards
        
        if train:
            self.randomizer = Randomizer(level_grid(self.level, grid_size=(width, height)), object_to_shuffle)
        else:
            self.randomizer = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            infoObject = pygame.display.Info()
            self.window_width, self.window_height = infoObject.current_w, infoObject.current_h
            self.window_width, self.window_height = 17*50, 15*50
            self.pix_square_size = min(
                self.window_width // width, self.window_height // height
            )
            # self.window = pygame.display.set_mode((self.window_width, self.window_height), pygame.FULLSCREEN)
            self.window = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
        if self.render_mode == "human":
            self.load_images()
        self.objects = {
            Object.BACKGROUND.value: ObjectState(Object.BACKGROUND),
            Object.BABA.value: ObjectState(Object.BABA),
            Object.FLAG.value: ObjectState(Object.FLAG),
            Object.WALL.value: ObjectState(Object.WALL),
            Object.ROCK.value: ObjectState(Object.ROCK),
            Object.PUSH_TEXT.value: ObjectState(Object.PUSH_TEXT, rule_text="push", immutable_rules=["push"]),
            Object.STOP_TEXT.value: ObjectState(Object.STOP_TEXT, rule_text="stop", immutable_rules=["push"]),
            Object.YOU_TEXT.value: ObjectState(Object.YOU_TEXT, rule_text="you", immutable_rules=["push"]),
            Object.IS_TEXT.value: ObjectState(Object.IS_TEXT, immutable_rules=["push"]),
            Object.WIN_TEXT.value: ObjectState(Object.WIN_TEXT, rule_text="win", immutable_rules=["push"]),
            Object.BABA_TEXT.value: ObjectState(Object.BABA_TEXT, paired_object_key=Object.BABA.value, immutable_rules=["push"]),
            Object.FLAG_TEXT.value: ObjectState(Object.FLAG_TEXT, paired_object_key=Object.FLAG.value, immutable_rules=["push"]),
            Object.ROCK_TEXT.value: ObjectState(Object.ROCK_TEXT, paired_object_key=Object.ROCK.value, immutable_rules=["push"]),
            Object.WALL_TEXT.value: ObjectState(Object.WALL_TEXT, paired_object_key=Object.WALL.value, immutable_rules=["push"]),
        }

        # num_objects = np.count_nonzero()
        self.observation_space = Box(low=-1, high=np.array([[width, height, len(self.objects)] for _ in range(width*height)]), shape=(width*height,3), dtype=np.int64)

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }
        self.visited_states = set()
        self.reset()

    def load_images(self):
        babaImg = pygame.image.load('imgs/baba.png')
        wallImg = pygame.image.load('imgs/wall.png')
        rockImg = pygame.image.load('imgs/rock.png')
        tileImg = pygame.image.load('imgs/tile.png')
        flagImg = pygame.image.load('imgs/FLAG.png')
        
        babaImg = pygame.transform.scale(babaImg, (self.pix_square_size, self.pix_square_size))
        wallImg = pygame.transform.scale(wallImg, (self.pix_square_size, self.pix_square_size))
        rockImg = pygame.transform.scale(rockImg, (self.pix_square_size, self.pix_square_size))
        tileImg = pygame.transform.scale(tileImg, (self.pix_square_size, self.pix_square_size))
        flagImg = pygame.transform.scale(flagImg, (self.pix_square_size, self.pix_square_size))
        
        babaTextImg = pygame.image.load('imgs/BABA_text.png')
        wallTextImg = pygame.image.load('imgs/WALL_text.png')
        rockTextImg = pygame.image.load('imgs/ROCK_text.png')
        flagTextImg = pygame.image.load('imgs/FLAG_text.png')
        
        babaTextImg = pygame.transform.scale(babaTextImg, (self.pix_square_size, self.pix_square_size))
        wallTextImg = pygame.transform.scale(wallTextImg, (self.pix_square_size, self.pix_square_size))
        rockTextImg = pygame.transform.scale(rockTextImg, (self.pix_square_size, self.pix_square_size))
        flagTextImg = pygame.transform.scale(flagTextImg, (self.pix_square_size, self.pix_square_size))
        
        isTextImg = pygame.image.load('imgs/IS_text.png')
        youTextImg = pygame.image.load('imgs/YOU_text.png')
        pushTextImg = pygame.image.load('imgs/PUSH_text.png')
        stopTextImg = pygame.image.load('imgs/STOP_text.png')
        winTextImg = pygame.image.load('imgs/WIN_text.png')
        
        isTextImg = pygame.transform.scale(isTextImg, (self.pix_square_size, self.pix_square_size))
        youTextImg = pygame.transform.scale(youTextImg, (self.pix_square_size, self.pix_square_size))
        pushTextImg = pygame.transform.scale(pushTextImg, (self.pix_square_size, self.pix_square_size))
        stopTextImg = pygame.transform.scale(stopTextImg, (self.pix_square_size, self.pix_square_size))
        winTextImg = pygame.transform.scale(winTextImg, (self.pix_square_size, self.pix_square_size))
        
        self.obj_imgs = {
            Object.BACKGROUND.value: tileImg,
            Object.BABA.value: babaImg,
            Object.WALL.value: wallImg,
            Object.ROCK.value: rockImg,
            Object.FLAG.value: flagImg,
            Object.BABA_TEXT.value: babaTextImg,
            Object.WALL_TEXT.value: wallTextImg,
            Object.ROCK_TEXT.value: rockTextImg,
            Object.FLAG_TEXT.value: flagTextImg,
            Object.IS_TEXT.value: isTextImg,
            Object.YOU_TEXT.value: youTextImg,
            Object.PUSH_TEXT.value: pushTextImg,
            Object.STOP_TEXT.value: stopTextImg,
            Object.WIN_TEXT.value: winTextImg,
        }

    def get_objects(self, x, y):
        # dictionary of object:[object_identifier] for all objects at location x,y
        objects = {self.objects[k]:[i for (x_, y_, i) in self.state[k] if x_ == x and y_ == y] 
                   for k in self.state.keys() if (x,y) in self.get_object_type_locations(k)}
        return objects
    
    def set_object_location(self, new_x, new_y, object_identifier, object_type):
        # remove object from old location
        self.state[object_type] = [(x, y, i) for (x, y, i) in self.state[object_type] if i != object_identifier]
        # add object to new location
        self.state[object_type].append((new_x, new_y, object_identifier))
    
    def get_you_objects(self):
        return [(obj_type, obj) for obj_type, obj in self.objects.items() if obj.is_you()]

    def get_win_objects(self):
        return [(obj_type, obj) for obj_type, obj in self.objects.items() if obj.is_win()]

    def get_object_type_locations(self, object_type):
        return [(x,y) for (x,y,_) in self.state[object_type]]
    
    def _get_info(self):
        return {
            # "distance": np.linalg.norm(
            #     self._agent_location - self._target_location, ord=1
            # )
        }
    
    def _get_obs(self):
        observation = [np.array([x,y,k]) for k in self.state.keys() for (x,y,_) in self.state[k]]
        for _ in range(self.width*self.height-len(observation)):
            observation.append(np.array([-1,-1,-1]))
        return np.array(observation)
            
    def _get_reward(self):
        # get all the changes between states
        # convert to regular list for easier comparison
        obs = [list(x) for x in self._get_obs()]
        prev_obs = [list(x) for x in self.prev_obs]
        changes = [x for x in obs if x not in prev_obs]

        # rewards for win/lose conditions
        if "winlose" in self.rewards:
            if self.check_win_condition():
                return 100
            elif self.check_lose_condition():
                return -1000
            
        # punish the agent for not changing the state
        if "nochange" in self.rewards and not changes:
                return -100

        reward = 0

        # reward the agent for moving text
        if "movetext" in self.rewards:
            for loc_and_obj in changes:
                if loc_and_obj[2] in [Object.PUSH_TEXT.value, Object.STOP_TEXT.value, Object.YOU_TEXT.value, Object.WIN_TEXT.value, Object.IS_TEXT.value, Object.BABA_TEXT.value, Object.ROCK_TEXT.value, Object.FLAG_TEXT.value, Object.WALL_TEXT.value]:
                    reward += 10

            for obj in self.objects:
                # if the object has a new rule, reward the agent
                if len(self.objects[obj].user_rules) > len(self.prev_object_rules[obj]):
                    # changing rules for "you" or "win" is a big reward
                    if obj in [Object.YOU_TEXT.value, Object.WIN_TEXT.value]:
                        reward += 50
                    else:
                        reward += 25

        # Distance to nearest WIN object
        if "distance" in self.rewards:
            you_positions = [pos for obj_type in self.get_you_objects()
                            for pos in self.get_object_type_locations(obj_type[0])]
            win_positions = [pos for obj_type in self.get_win_objects()
                            for pos in self.get_object_type_locations(obj_type[0])]
            if you_positions and win_positions:
                distances = [np.linalg.norm(np.array(y) - np.array(w), ord=1)
                            for y in you_positions for w in win_positions]
                min_distance = min(distances)
                reward -= min_distance  # Normalize distance to reward


        return reward


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.state = {k:[] for k in range(1, len(self.objects.keys()))}
        object_identifier = 0
        if self.randomizer:
            if self.randomizer.you_object_value:
                self.randomizer.reshuffle_object()
            else:
                self.randomizer.reshuffle_level()
            state = self.randomizer.new_state
        else:
            state = level_grid(self.level, grid_size=(self.width, self.height))
        # self.state = state
        for x in range(self.width):
            for y in range(self.height):
                obj = state[x, y]
                if obj != Object.BACKGROUND.value:
                    self.state[obj].append((x, y, object_identifier))
                    object_identifier += 1
        self.update_rules()

        if self.render_mode == "human":
            self._render_frame()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def check_win_condition(self):
        # checking if an object has both 'YOU' and 'WIN' properties
        if any(obj.is_win() for (_, obj) in self.get_you_objects()):
            return True
        # checking if any object with 'YOU' property has moved onto an object with 'WIN' property
        you_loc = set()
        win_loc = set()
        for (obj_type, _) in self.get_you_objects():
            you_loc.update(set(self.get_object_type_locations(obj_type)))
        for (obj_type, _) in self.get_win_objects():
            win_loc.update(set(self.get_object_type_locations(obj_type)))
        if you_loc.intersection(win_loc):
            return True
        return False 
    
    def check_lose_condition(self):
        # checking if there are no objects with the property 'YOU'
        if not self.get_you_objects():
            return True
        return False 
    
    # backtracking a move that is not valid
    def backtrack(self, current_location, direction, flagged_objects, backtracked_objects=None):
        recurse = False
        # for the first call
        if backtracked_objects is None:
            backtracked_objects = set()
        # getting the objects in the current location and the previous location based on the direction
        current_objects = self.get_objects(current_location[0], current_location[1])
        previous_location = current_location - direction
        for obj, obj_ids in current_objects.items():
            # only move back objects that are push or you
            if obj.is_push() or obj.is_you():
                # make sure the object actually moved into this square and that it has not already been backtracked
                for obj_id in [obj_id for obj_id in obj_ids if obj_id in flagged_objects and obj_id not in backtracked_objects]:
                    self.set_object_location(previous_location[0], previous_location[1], obj_id, obj.type.value)
                    backtracked_objects.add(obj_id)
                    recurse = True # need to backtrack any items in the previous location if something was there
        if recurse:
            self.backtrack(previous_location, direction, flagged_objects, backtracked_objects)
        return

    def is_valid_move(self, current_location, direction, flagged_objects):
        """Returns True if an object can move in the given direction."""
        next_location = current_location + direction
        check_next = False
        
        # Check if the next position is within grid bounds
        if np.any(next_location < 0) or next_location[0] > self.width - 1 or next_location[1] > self.height - 1:
            self.backtrack(current_location, direction, flagged_objects)
            return 0  # Out of bounds
        
        next_objects = self.get_objects(next_location[0], next_location[1])
        
        for obj, obj_ids in next_objects.items():
            # if there is an object that is a stop object, the move is invalid
            if obj.is_stop() and not obj.is_push():
                self.backtrack(current_location, direction, flagged_objects)
                return 0
            # if there is a push object that is flagged (has already been moved, hence was backtracked), the move is invalid
            elif obj.is_push() and set(obj_ids).intersection(flagged_objects): # if obj is push and is flagged (moved already)
                self.backtrack(current_location, direction, flagged_objects)
                return 0
            elif obj.is_push(): # if there is a push object that has not been flagged, we have to check the next square
                check_next = True
        if check_next:
            return 1
        else:
            return -1

    def handle_move(self, update_locations, direction):
        """Moves object and handles push mechanics."""
        flagged_objects = set()
        next_locations = []
        # first movement, check starting location and move all objects with atrribute 'you' if move is valid
        for current_location in update_locations:
            current_objects = self.get_objects(current_location[0], current_location[1])
            check_next = self.is_valid_move(current_location, direction, flagged_objects)
            if check_next:
                next_location = current_location + direction
                for obj, obj_ids in current_objects.items():
                    non_flagged = [obj_id for obj_id in obj_ids if obj_id not in flagged_objects]
                    if obj.is_you():
                        for obj_id in non_flagged:
                            self.set_object_location(next_location[0], next_location[1], obj_id, obj.type.value)
                        flagged_objects.update(obj_ids)
                        if check_next == 1:
                            next_locations.append(next_location)
            else: # if it is not a valid move, flag the objects here so that they are not moved/act as non-valid square
                for obj, obj_ids in current_objects.items():
                    if obj.is_you() or obj.is_push():
                        flagged_objects.update(obj_ids)
        # go through each location and move all push objects at that location if the next location is a valid move
        # non-valid moves will recursively backtrack movements
        update_locations = next_locations
        while update_locations:
            current_location = update_locations.pop()
            current_objects = self.get_objects(current_location[0], current_location[1])
            check_next = self.is_valid_move(current_location, direction, flagged_objects)
            if check_next:
                for obj, obj_ids in current_objects.items():
                    # go through only the non-flagged push objects (flagged ones have been moved into the current location)
                    non_flagged = [obj_id for obj_id in obj_ids if obj_id not in flagged_objects]
                    if obj.is_push() and non_flagged:
                            next_location = current_location + direction
                            for obj_id in non_flagged:
                                self.set_object_location(next_location[0], next_location[1], obj_id, obj.type.value)
                            flagged_objects.update(non_flagged)
                            if check_next == 1:
                                update_locations.append(next_location)
            else: # if it is not a valid move, flag the objects here so that they are not moved/act as non-valid square
                for obj, obj_ids in current_objects.items():
                    if obj.is_push():
                        flagged_objects.update(obj_ids)
        return

    def update_rules(self):
        # clear rules on all objects
        for obj in self.objects.values():
            obj.clear_rules()
        # go through all the is text locations and look for horizontal and vertical sentences
        for (x,y,_) in self.state[Object.IS_TEXT.value]:
            for obj_n in self.get_objects(x-1, y).keys():
                if obj_n.paired_object_key:
                    for obj_v in self.get_objects(x+1, y).keys():
                        if obj_v.rule_text:
                            self.objects[obj_n.paired_object_key].add_rule(obj_v.rule_text)
            for obj_n in self.get_objects(x, y-1).keys():
                if obj_n.paired_object_key:
                    for obj_v in self.get_objects(x, y+1).keys():
                        if obj_v.rule_text:
                            self.objects[obj_n.paired_object_key].add_rule(obj_v.rule_text)
        return

    def step(self, action):
        self.prev_obs = self._get_obs()
        # Map the action (element of {0,1,2,3}) to the delta in coordinates
        direction = self._action_to_direction[action]
        starting_locations = set()
        # move all objects that are "you":
        for (type_id, you_object) in self.get_you_objects():
            # Find all instances of this object type
            starting_locations.update(set(self.get_object_type_locations(type_id)))

        self.handle_move(starting_locations, direction)
        
        self.prev_object_rules = {obj: self.objects[obj].user_rules for obj in self.objects}
        self.update_rules()

        # An episode is done if any 'you' is on top of any 'win' object
        terminated = self.check_win_condition() or self.check_lose_condition()
        reward = self._get_reward()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
            # self.window = pygame.display.set_mode((self.window_width, self.window_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((0, 0, 0))

       # draw
        for x in range(self.width):
            for y in range(self.height):
                # draw tiles everywhere
                canvas.blit(self.obj_imgs[Object.BACKGROUND.value], pygame.Rect(
                            (np.array([x, y])) * self.pix_square_size, 
                            (self.pix_square_size, self.pix_square_size)
                            )
                        )
        you_objects = [obj_type for (obj_type, _) in self.get_you_objects()]
        win_objects = [obj_type for (obj_type, _) in self.get_win_objects()]
        for obj_type in self.state.keys():
            if obj_type not in you_objects and obj_type not in win_objects:
                for (x, y, _) in self.state[obj_type]:
                    canvas.blit(self.obj_imgs[obj_type], pygame.Rect(
                            (np.array([x, y])) * self.pix_square_size, 
                            (self.pix_square_size, self.pix_square_size)
                            )
                        )
        for obj_type in win_objects:
            for (x, y, _) in self.state[obj_type]:
                canvas.blit(self.obj_imgs[obj_type], pygame.Rect(
                        (np.array([x, y])) * self.pix_square_size, 
                        (self.pix_square_size, self.pix_square_size)
                        )
                    )
        for obj_type in you_objects:
            for (x, y, _) in self.state[obj_type]:
                canvas.blit(self.obj_imgs[obj_type], pygame.Rect(
                        (np.array([x, y])) * self.pix_square_size, 
                        (self.pix_square_size, self.pix_square_size)
                        )
                    )
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    env = BABAWorldEnv(render_mode="human", level=1, width=17, height=15)
    env.reset()
    terminated = False
    action_map = {
        pygame.K_RIGHT: Actions.right.value,
        pygame.K_UP: Actions.down.value,
        pygame.K_LEFT: Actions.left.value,
        pygame.K_DOWN: Actions.up.value 
    }

    while not terminated:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key in action_map:
                    action = action_map[event.key]
                
                if event.key == pygame.K_ESCAPE:
                    terminated = True

        if action is not None:
            result = env.step(action)
            terminated = terminated or result[2]

        env.render()

    env.close()