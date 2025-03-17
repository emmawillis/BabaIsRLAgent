from enum import Enum
from baba_levels import level_1
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from game_objects import Actions, Object, ObjectState

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, width=5, height=5):
        self.width = width
        self.height = height

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
            self.pix_square_size = min(
                self.window_width // width, self.window_height // height
            )
            self.window = pygame.display.set_mode((self.window_width, self.window_height), pygame.FULLSCREEN)

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

        # Observations are dictionaries
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Box(low=0, high=len(self.objects) - 1, shape=(width, height), dtype=np.uint8)

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

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

    def load_images(self):
        self.babaImg = pygame.image.load('imgs/baba.png')
        self.wallImg = pygame.image.load('imgs/wall.png')
        self.rockImg = pygame.image.load('imgs/rock.png')
        self.tileImg = pygame.image.load('imgs/tile.png')
        self.flagImg = pygame.image.load('imgs/FLAG.png')

        self.babaImg = pygame.transform.scale(self.babaImg, (self.pix_square_size, self.pix_square_size))
        self.wallImg = pygame.transform.scale(self.wallImg, (self.pix_square_size, self.pix_square_size))
        self.rockImg = pygame.transform.scale(self.rockImg, (self.pix_square_size, self.pix_square_size))
        self.tileImg = pygame.transform.scale(self.tileImg, (self.pix_square_size, self.pix_square_size))
        self.flagImg = pygame.transform.scale(self.flagImg, (self.pix_square_size, self.pix_square_size))

        self.babaTextImg = pygame.image.load('imgs/BABA_text.png')
        self.wallTextImg = pygame.image.load('imgs/WALL_text.png')
        self.rockTextImg = pygame.image.load('imgs/ROCK_text.png')
        self.flagTextImg = pygame.image.load('imgs/FLAG_text.png')

        self.babaTextImg = pygame.transform.scale(self.babaTextImg, (self.pix_square_size, self.pix_square_size))
        self.wallTextImg = pygame.transform.scale(self.wallTextImg, (self.pix_square_size, self.pix_square_size))
        self.rockTextImg = pygame.transform.scale(self.rockTextImg, (self.pix_square_size, self.pix_square_size))
        self.flagTextImg = pygame.transform.scale(self.flagTextImg, (self.pix_square_size, self.pix_square_size))

        self.isTextImg = pygame.image.load('imgs/IS_text.png')
        self.youTextImg = pygame.image.load('imgs/YOU_text.png')
        self.pushTextImg = pygame.image.load('imgs/PUSH_text.png')
        self.stopTextImg = pygame.image.load('imgs/STOP_text.png')
        self.winTextImg = pygame.image.load('imgs/WIN_text.png')

        self.isTextImg = pygame.transform.scale(self.isTextImg, (self.pix_square_size, self.pix_square_size))
        self.youTextImg = pygame.transform.scale(self.youTextImg, (self.pix_square_size, self.pix_square_size))
        self.pushTextImg = pygame.transform.scale(self.pushTextImg, (self.pix_square_size, self.pix_square_size))
        self.stopTextImg = pygame.transform.scale(self.stopTextImg, (self.pix_square_size, self.pix_square_size))
        self.winTextImg = pygame.transform.scale(self.winTextImg, (self.pix_square_size, self.pix_square_size))

    def get_object(self, x, y):
        return self.objects[self.state[x, y]]
    
    def set_object(self, x, y, object=Object.BACKGROUND):
        self.objects[self.state[x, y]] = object
    
    def get_you_objects(self):
        return [(obj_type, obj) for obj_type, obj in self.objects.items() if obj.is_you()]

    def get_win_objects(self):
        return [(obj_type, obj) for obj_type, obj in self.objects.items() if obj.is_win()]

    def _get_info(self):
        return {
            # "distance": np.linalg.norm(
            #     self._agent_location - self._target_location, ord=1
            # )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # TODO handle levels properly
        state, rules = level_1()
        self.state = state
        for (type, rule) in rules:
            self.objects[type.value].add_rule(rule)

        if self.render_mode == "human":
            self._render_frame()

        observation = self.state
        info = self._get_info()
        return observation, info

    def check_win_condition(self):
        if any(obj.is_win() for (_, obj) in self.get_you_objects()):
            return True

        """Checks if any object with 'YOU' property has moved into a space occupied by an object with 'WIN' property."""        
        # you_positions = [pos for pos in self.state if self.get_object(pos[0], pos[1]).is_you()]
        # win_positions = [pos for pos in self.state if self.get_object(pos[0], pos[1]).is_win()]
        
        # return [pos in win_positions for pos in you_positions].any()
        return False # TODO!!!! objects cant currently overlap

    def is_valid_move(self, moving_object, current_position, direction):
        """Returns True if moving_object can move in the given direction."""
        if moving_object.is_stop() and not moving_object.is_push():
            return False
        
        next_position = current_position + direction

        # Check if the next position is within grid bounds
        if np.any(next_position < 0) or np.any(next_position >= self.observation_space.shape[:2]):
            return False  # Out of bounds

        # Get the object at the next position
        next_object = self.get_object(next_position[0], next_position[1])

        # If the next position is empty, or there is an object with no rules (treated as background), then it's a valid move
        if next_object.is_free() or next_object.is_win():
            return True

        # If the next position object is "PUSH", recursively check if it can move
        if next_object.is_push():
            return self.is_valid_move(next_object, next_position, direction)

        return False  # Default case: Invalid move

    def handle_move(self, moving_object, current_position, direction):
        """Moves object and handles push mechanics."""

        if not self.is_valid_move(moving_object, current_position, direction):
            return  # Invalid move, do nothing

        next_position = current_position + direction
        next_object = self.get_object(next_position[0], next_position[1])

        # If the next object is pushable, move it recursively
        if next_object.is_push():
            self.handle_move(next_object, next_position, direction)

        # update the state with the object's new position
        self.state[next_position[0], next_position[1]] = self.state[current_position[0], current_position[1]]
        self.state[current_position[0], current_position[1]] = Object.BACKGROUND.value  # Clear old position

        return

    def get_coordinates_of_type(self, object_type):
        return [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if self.get_object(x, y).type == object_type
        ]

    def update_rules(self):
        # clear rules on all objects
        for obj in self.objects.values():
            obj.clear_rules()
        
        for x,y in self.get_coordinates_of_type(Object.IS_TEXT): # this is looping over all IS_TEXT objects
            paired_object_key = self.get_object(x-1, y).paired_object_key
            if paired_object_key is not None:
                rule = self.get_object(x+1, y).rule_text
                if rule is not None:
                    self.objects[paired_object_key].add_rule(rule)
            
            paired_object_key = self.get_object(x, y-1).paired_object_key
            if paired_object_key is not None:
                rule = self.get_object(x, y+1).rule_text
                if rule is not None:
                    self.objects[paired_object_key].add_rule(rule)

        return

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the delta in coordinates
        direction = self._action_to_direction[action]

        # move all objects that are "you":
        for (type_id, you_object) in self.get_you_objects():
            # Find all instances of this object type
            object_positions = np.argwhere(self.state == type_id)

            for current_position in object_positions:
                self.handle_move(you_object, current_position, direction)
                self.update_rules()
                # TODO - what if multiple 'you' objects collide? do we want to handle this

        # An episode is done if any 'you' is on top of any 'win' object
        terminated = self.check_win_condition()
        reward = 1 if terminated else 0  # TODO figure out ideal rewards (if move creates a useful rule that should be rewarded)
        observation = self.state
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

        # draw the stops
        for x in range(self.width):
            for y in range(self.height):
                if self.get_object(x,y).type == Object.WALL:
                    canvas.blit(self.wallImg, pygame.Rect(
                        (np.array([x, y])) * self.pix_square_size, 
                        (self.pix_square_size, self.pix_square_size)
                        )
                    )
                elif self.get_object(x,y).type == Object.ROCK:
                    canvas.blit(self.rockImg, pygame.Rect(
                        (np.array([x, y])) * self.pix_square_size, 
                        (self.pix_square_size, self.pix_square_size)
                        )
                    )  
                elif self.get_object(x,y).type == Object.FLAG:
                    canvas.blit(self.flagImg, pygame.Rect(
                        (np.array([x, y])) * self.pix_square_size, 
                        (self.pix_square_size, self.pix_square_size)
                        )
                    )  
                elif self.get_object(x,y).type == Object.BABA:
                        canvas.blit(self.babaImg, pygame.Rect(
                            (np.array([x, y])) * self.pix_square_size, 
                            (self.pix_square_size, self.pix_square_size)
                            )
                        )
                elif self.get_object(x,y).type == Object.BABA_TEXT:
                        canvas.blit(self.babaTextImg, pygame.Rect(
                            (np.array([x, y])) * self.pix_square_size, 
                            (self.pix_square_size, self.pix_square_size)
                            )
                        )
                elif self.get_object(x,y).type == Object.FLAG_TEXT:
                        canvas.blit(self.flagTextImg, pygame.Rect(
                            (np.array([x, y])) * self.pix_square_size, 
                            (self.pix_square_size, self.pix_square_size)
                            )
                        )
                elif self.get_object(x,y).type == Object.IS_TEXT:
                        canvas.blit(self.isTextImg, pygame.Rect(
                            (np.array([x, y])) * self.pix_square_size, 
                            (self.pix_square_size, self.pix_square_size)
                            )
                        )
                elif self.get_object(x,y).type == Object.PUSH_TEXT:
                        canvas.blit(self.pushTextImg, pygame.Rect(
                            (np.array([x, y])) * self.pix_square_size, 
                            (self.pix_square_size, self.pix_square_size)
                            )
                        )
                elif self.get_object(x,y).type == Object.ROCK_TEXT:
                        canvas.blit(self.rockTextImg, pygame.Rect(
                            (np.array([x, y])) * self.pix_square_size, 
                            (self.pix_square_size, self.pix_square_size)
                            )
                        )
                elif self.get_object(x,y).type == Object.STOP_TEXT:
                        canvas.blit(self.stopTextImg, pygame.Rect(
                            (np.array([x, y])) * self.pix_square_size, 
                            (self.pix_square_size, self.pix_square_size)
                            )
                        )
                elif self.get_object(x,y).type == Object.WALL_TEXT:
                        canvas.blit(self.wallTextImg, pygame.Rect(
                            (np.array([x, y])) * self.pix_square_size, 
                            (self.pix_square_size, self.pix_square_size)
                            )
                        )
                elif self.get_object(x,y).type == Object.WIN_TEXT:
                        canvas.blit(self.winTextImg, pygame.Rect(
                            (np.array([x, y])) * self.pix_square_size, 
                            (self.pix_square_size, self.pix_square_size)
                            )
                        )
                elif self.get_object(x,y).type == Object.YOU_TEXT:
                        canvas.blit(self.youTextImg, pygame.Rect(
                            (np.array([x, y])) * self.pix_square_size, 
                            (self.pix_square_size, self.pix_square_size)
                            )
                        )
                
                else:
                    # anything that isn't one of these objects (and is not where baba is), draw as a tile  
                    canvas.blit(self.tileImg, pygame.Rect(
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
    env = GridWorldEnv(render_mode="human", width=33, height=18)
    env.reset()
    terminated = False
    action_map = {
        pygame.K_RIGHT: Actions.right.value,
        pygame.K_UP: Actions.down.value,
        pygame.K_LEFT: Actions.left.value,
        pygame.K_DOWN: Actions.up.value # TODO whats up with up and down
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