import envs
from stable_baselines3 import A2C, DQN, PPO
from time import sleep
import pygame
from gymnasium.wrappers import TimeLimit
from envs.game_objects import Object
import argparse
import matplotlib.pyplot as plt
import os
from enum import Enum

def learn_decaying_epsilon(alg, level: int, train: bool = True, object_to_shuffle: int = None, rewards = None):
    env = envs.BABAWorldEnv(render_mode=None, level=level, train=train, object_to_shuffle=object_to_shuffle, rewards=rewards)
    wrapped_env = TimeLimit(env, max_episode_steps=int(1e4))

    model = alg("MlpPolicy", wrapped_env, device="cpu")
    reward_log = []

    for epoch in range(10):
        print(f"\nEpoch {epoch + 1} Training...")
        model.learn(total_timesteps=5000, progress_bar=True)

        # Evaluate model after training chunk
        eval_env = envs.BABAWorldEnv(render_mode=None, level=level, train=False, object_to_shuffle=object_to_shuffle)
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 100:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = eval_env.step(int(action))
            total_reward += reward
            steps += 1

        reward_log.append((epoch + 1, total_reward))
        print(f"Epoch {epoch + 1} total reward: {total_reward}")

        # Save and exit early if model wins
        # if total_reward >= 100:
        #     save_path = f"best_model_level{level}_{alg.__name__}.zip"
        #     model.save(save_path)
        #     print(f"\n🎉 Early stopping: agent won! Model saved to '{save_path}'")
        #     break

        # Decay epsilon manually (DQN only)
        if hasattr(model, "exploration_rate"):
            model.exploration_rate = max(0.1, model.exploration_rate * 0.9)

    # Plot reward over epochs
    xs, ys = zip(*reward_log)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker='o', linestyle='-', linewidth=2)
    plt.title('Reward per Training Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Total Reward')
    plt.grid(True)

    for x, y in zip(xs, ys):
        plt.text(x, y + 50, str(y), ha='center', fontsize=8)

    plt.tight_layout()
    plt.show()

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
        
    parser = argparse.ArgumentParser()
    parser.add_argument('level', type=int)
    parser.add_argument('--alg')
    valid_rewards = ["winlose", "nochange", "movetext", "distance", "all"]

    parser.add_argument(
        '--rewards',
        nargs='+',
        choices=valid_rewards,
        required=True,
        help=f"Choose one or more reward functions from: {', '.join(valid_rewards)}"
    )
    args = parser.parse_args()

    alg_2_class = {
        'a2c': A2C,
        'ppo': PPO,
        'dqn': DQN
    }

    model = learn_decaying_epsilon(alg_2_class[args.alg.lower()], level=args.level, train=True, object_to_shuffle=object_to_shuffle, rewards = args.rewards)
    evaluate_model(model, level=args.level, train=False)
