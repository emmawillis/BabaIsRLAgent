import envs
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
import numpy as np
from time import sleep
import gymnasium as gym
from gymnasium.spaces import Dict, Sequence, Box, Discrete
import pygame
import numpy as np
from gymnasium.wrappers import TimeLimit


env = envs.BABAWorldEnv()
wrapped_env = TimeLimit(env, max_episode_steps=100)


model = A2C("MlpPolicy", wrapped_env, device="cpu")
model = model.learn(total_timesteps=50000)

env = envs.BABAWorldEnv(render_mode="human")
obs,_ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(int(action))
    env.render()
    print(f"Action: {action}, Reward: {reward}")
    if done:
        obs = env.reset()

# env = envs.BABAWorldEnv(render_mode="human", width=17, height=15)
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