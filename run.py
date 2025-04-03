import envs
from stable_baselines3 import A2C, DQN
from time import sleep
import pygame
from gymnasium.wrappers import TimeLimit
from envs.game_objects import Object


def learn_decaying_epsilon(alg, level: int, train: bool = True, object_to_shuffle: int = None):
    env = envs.BABAWorldEnv(render_mode=None, level=level, train=train, object_to_shuffle=object_to_shuffle)
    wrapped_env = TimeLimit(env, max_episode_steps=int(1e4))

    model = alg("MlpPolicy", wrapped_env, device="cpu")

    # from ChatGPT
    # Train in small steps, reducing epsilon manually
    for _ in range(10):
      # Reassign the environment to the model
        model.learn(total_timesteps=int(1e3), progress_bar=True)  # Train in chunks
        model.exploration_rate = max(0.1, model.exploration_rate * 0.9)  # Decay epsilon manually
    return model

def evaluate_model(model, level: int, train: bool = True, object_to_shuffle: int = None):
    env = envs.BABAWorldEnv(level=level, train=train, object_to_shuffle=object_to_shuffle)
    obs,_ = env.reset()
    done = False
    steps = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(int(action))
        env.render()
        print(f"Action: {action}, Reward: {reward}")
        steps += 1
        if done or steps > 25:
            env.reset()
            steps = 0
            sleep(3)
            print()

def run_manually(level: int=1, train: bool = False, object_to_shuffle: int = None):
    env = envs.BABAWorldEnv(level=level, train=train, object_to_shuffle=object_to_shuffle)
    env.reset()
    terminated = False
    action_map = {
        pygame.K_RIGHT: envs.Actions.right.value,
        pygame.K_UP: envs.Actions.down.value,
        pygame.K_LEFT: envs.Actions.left.value,
        pygame.K_DOWN: envs.Actions.up.value 
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
            print(f"Reward: {result[1]}")
            terminated = terminated or result[2]

        env.render()
    env.close()

if __name__ == "__main__":
    object_to_shuffle = Object.BABA.value
    # run_manually(level=1, train=True, object_to_shuffle=object_to_shuffle)
    
    model = learn_decaying_epsilon(A2C, level=1, train=True, object_to_shuffle=object_to_shuffle)
    evaluate_model(model, level=1, train=False)
