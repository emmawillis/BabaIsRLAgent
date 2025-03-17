from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from levels import level_1

class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3

class Object(Enum):
    BACKGROUND = 0
    BABA = 1
    FLAG = 2
    WALL = 3
    ROCK = 4
    PUSH_TEXT = 5
    STOP_TEXT = 6
    YOU_TEXT = 7
    WIN_TEXT = 8
    IS_TEXT = 9
    BABA_TEXT = 10
    ROCK_TEXT = 11
    FLAG_TEXT = 12
    WALL_TEXT = 13

class ObjectState: 
    def __init__(self, type, is_text=False, user_rules=[], immutable_rules=[]):
        self.type = type
        self.is_text = is_text
        self.user_rules = user_rules
        self.immutable_rules = immutable_rules

    def rules(self):
        return self.user_rules + self.immutable_rules
 
    def remove_rule(self, rule): # only removes one instance of the rule
        self.user_rules.remove(rule)
    
    def add_rule(self, new_rule):
        self.user_rules.add(new_rule)

    def is_text(self):
        return self.is_text

    def is_you(self):
        return "you" in self.rules()
    
    def is_win(self):
        return "win" in self.rules()
    
    def is_stop(self):
        return "stop" in self.rules()
    
    def is_push(self):
        return "push" in self.rules()
    
    def is_free(self):
        return not self.rules()

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
  
    def __init__(self, render_mode=None, grid_height=18, grid_width=33):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.window_size = 512  # The size of the PyGame window

        self.objects = {
            Object.BACKGROUND: ObjectState(Object.BACKGROUND),
            Object.BABA: ObjectState(Object.BABA),
            Object.FLAG: ObjectState(Object.FLAG),
            Object.WALL: ObjectState(Object.WALL),
            Object.ROCK: ObjectState(Object.ROCK),
            Object.PUSH_TEXT: ObjectState(Object.PUSH_TEXT, [], immutable_rules=["push"]),
            Object.STOP_TEXT: ObjectState(Object.STOP_TEXT, immutable_rules=["push"]),
            Object.YOU_TEXT: ObjectState(Object.YOU_TEXT, immutable_rules=["push"]),
            Object.IS_TEXT: ObjectState(Object.IS_TEXT, immutable_rules=["push"]),
            Object.BABA_TEXT: ObjectState(Object.BABA_TEXT, immutable_rules=["push"]),
            Object.FLAG_TEXT: ObjectState(Object.FLAG_TEXT, immutable_rules=["push"]),
            Object.ROCK_TEXT: ObjectState(Object.ROCK_TEXT, immutable_rules=["push"]),
            Object.WALL_TEXT: ObjectState(Object.WALL_TEXT, immutable_rules=["push"]),
        }

        self.observation_space = spaces.Box(low=0, high=len(self.objects) - 1, shape=(grid_width, grid_height), dtype=np.uint8)

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

        self.reset()

    def get_object(self, x, y):
        return self.objects[self.state[x, y]]
    
    def set_object(self, x, y, object=Object.BACKGROUND):
        self.objects[self.state[x, y]] = object
    
    def get_you_objects(self):
        return [(obj_type, obj) for obj_type, obj in self.objects.items() if obj.is_you()]

    def get_win_objects(self):
        return [(obj_type, obj) for obj_type, obj in self.objects.items() if obj.is_win()]

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # TODO handle levels properly
        state, rules = level_1()
        self.state = state
        for rule in rules:
            self.objects[rule[0]].add_rule(rule[1])

        if self.render_mode == "human":
            self._render_frame()

        observation = self.state
        info = self._get_info()
        return observation, info

    def check_win_condition(self):
        """Checks if any object with 'YOU' property has moved into a space occupied by an object with 'WIN' property."""        
        you_positions = [pos for pos in self.state if self.objects[pos].is_you()]
        win_positions = [pos for pos in self.state if self.objects[pos].is_win()]
        
        return any(pos in win_positions for pos in you_positions)

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
        if next_object.is_free():
            return True

        # If the next position object is "PUSH", recursively check if it can move
        if next_object.is_push():
            return self.is_valid_move(next_object, next_position, direction)

        return False  # Default case: Invalid move

    def handle_move(self, moving_object, current_position, direction, moved_text=[]):
        """Moves object and handles push mechanics."""

        if not self.is_valid_move(moving_object, current_position, direction):
            return  # Invalid move, do nothing

        next_position = current_position + direction
        next_object = self.state[next_position[0], next_position[1]]

        # If the next object is pushable, move it recursively
        if next_object.is_push():
            moved_text = self.handle_move(next_object, next_position, direction, moved_text)

        # update the state with the object's new position
        self.state[next_position[0], next_position[1]] = self.state[current_position[0], current_position[1]]
        self.state[current_position[0], current_position[1]] = self.object_dict["background"]  # Clear old position

        if next_object.is_text():
            moved_text.add((next_object, next_position))
            
        return moved_text

    def update_rules(self, moved_text):
        # TODO!! UPDATE RULES HERE!! moved_text tracks all text objects that have moved as (ObjectState, (newposition))
        return # Placeholder

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the delta in coordinates
        direction = self._action_to_direction[action]
    
        # move all objects that are "you":
        for (type_id, you_object) in self.get_you_objects():
            # Find all instances of this object type
            object_positions = np.argwhere(self.state == type_id)

            for current_position in object_positions:
                moved_text = self.handle_move(you_object, current_position, direction)
                self.update_rules(moved_text)
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
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
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
    env = GridWorldEnv(render_mode="human")
    env.reset()
    for _ in range(10):
        env.step(env.action_space.sample())
        env.render()
    env.close()