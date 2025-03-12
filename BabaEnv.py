import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BabaIsYouEnv(gym.Env):
    """Custom Gym environment for Baba Is You."""
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Define action space (UP, DOWN, LEFT, RIGHT)
        self.action_space = spaces.Discrete(4)  

        # Define observation space (33x18 grid with multiple object types)
        self.observation_space = spaces.MultiDiscrete(np.full((33, 18), len(self.object_dict)))

        # Load initial game state (first level)
        self.reset()

        self.object_dict = {
            "Baba": "agent",
            "Wall": "obstacle",
            "Rock": "obstacle",
            "Flag": "goal",
            "Baba_text": "text",
            "Wall_text": "text",
            "Flag_text": "text",
            "Rock_text": "text",
            "Push_text": "text",
            "Stop_text": "text",
            "You_text": "text",
            "Is_text": "text",
            "Win_text": "text",
        } # TODO 

    def reset(self, seed=None, options=None):
        """Reset game to initial state."""
        self.state = self._load_game_config("level1.txt")  # Replace with dynamic game state loading
        self.baba_position = self._find_baba()

        return self.state, {}

    def step(self, action):
        """Apply an action and update game state."""
        prev_position = self.baba_position

        if action == 0:  # UP
            self.baba_position[1] -= 1
        elif action == 1:  # DOWN
            self.baba_position[1] += 1
        elif action == 2:  # LEFT
            self.baba_position[0] -= 1
        elif action == 3:  # RIGHT
            self.baba_position[0] += 1

        # Validate move
        if self._is_collision(self.baba_position):
            self.baba_position = prev_position  # Revert move

        # TODO update observation with any moved blocks

        # Check for win condition
        done = self._check_win()

        # Reward function
        reward = 1 if done else -1  # Encourage efficiency
        # TODO:
            # negative reward for breaking condition that makes game winnable
            # positive reward for making the game winnable

        return self.state, reward, done, False, {}

    def _load_game_config(self, config_path):
        """Load the game initial state."""
        return None # TODO 

    def _find_baba(self):
        """Find Baba's position in the grid."""
        # TODO This should be replaced with actual object detection logic
        return [5, 5]

    def _is_collision(self, position):
        """Check if Baba collides with an obstacle."""
        # TODO Implement collision detection based on grid objects
        return False

    def _check_win(self):
        """Check if Baba has reached the win condition."""
        #TODO  Implement win condition detection
        return False
