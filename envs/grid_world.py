from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.pix_square_size = (
                    self.window_size / self.size
                )  # The size of a single grid square in pixels

        # images
        self.babaImg = pygame.image.load('imgs/baba.png')
        self.wallImg = pygame.image.load('imgs/wall.png')
        self.rockImg = pygame.image.load('imgs/rock.png')

        self.babaImg = pygame.transform.scale(self.babaImg, (self.pix_square_size, self.pix_square_size))
        self.wallImg = pygame.transform.scale(self.wallImg, (self.pix_square_size, self.pix_square_size))
        self.rockImg = pygame.transform.scale(self.rockImg, (self.pix_square_size, self.pix_square_size))

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "stop": spaces.Dict(
                    {
                        "wall": spaces.Sequence(spaces.Box(0, size - 1, shape=(2,), dtype=int)),
                        "rock": spaces.Sequence(spaces.Box(0, size - 1, shape=(2,), dtype=int)),
                    }
                ),
            }
        )

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

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location, "stop": self._stops}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        # choose locations according to the first level
        self._stops = {
            "wall": np.array([np.array([0, 0]), np.array([1, 1]), np.array([2, 2]), np.array([3, 3]), np.array([4, 4])]),
            "rock": np.array([np.array([0, 1]), np.array([1, 3])]),
        }

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        new_loc = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # ----- STOP PROPERTY -----
        # before updating the movement, check if we are moving into something with STOP
        # combine all the "stop" arrays to do a general check
        all_stops = np.array([loc for object_type in self._stops.keys() for loc in self._stops[object_type]])
        if not any(np.equal(all_stops, new_loc).all(1)):
            self._agent_location = new_loc

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
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
        canvas.fill((0, 0, 0))

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                self.pix_square_size * self._target_location,
                (self.pix_square_size, self.pix_square_size),
            ),
        )

        # draw the agent
        canvas.blit(self.babaImg, pygame.Rect(
                (self._agent_location) * self.pix_square_size, 
                (self.pix_square_size, self.pix_square_size)
            )
        )

        # draw the stops
        for x in range(self.size):
            for y in range(self.size):
                if any(np.equal(self._stops["wall"], np.array([x,y])).all(1)):
                    canvas.blit(self.wallImg, pygame.Rect(
                        (np.array([x, y])) * self.pix_square_size, 
                        (self.pix_square_size, self.pix_square_size)
                        )
                    )
                elif any(np.equal(self._stops["rock"], np.array([x,y])).all(1)):
                    canvas.blit(self.rockImg, pygame.Rect(
                        (np.array([x, y])) * self.pix_square_size, 
                        (self.pix_square_size, self.pix_square_size)
                        )
                    )    

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (255, 255, 255),
                (0, self.pix_square_size * x),
                (self.window_size, self.pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                (255, 255, 255),
                (self.pix_square_size * x, 0),
                (self.pix_square_size * x, self.window_size),
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
    # env = GridWorldEnv(render_mode="human", size=13)
    env = GridWorldEnv(render_mode="human", size=5)
    env.reset()
    terminated = False
    while not terminated:
        result = env.step(env.action_space.sample())
        terminated = result[2]
        env.render()
    env.close()