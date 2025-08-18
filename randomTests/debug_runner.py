# debug_runner.py
import gymnasium as gym
import torch
import numpy as np
import yaml

from dynamicrl.algorithms.ppo.ppo_algorithm import PPOAlgorithm

print("--- Starting Debug Runner ---")

try:
    print("Loading configuration from configs/ppo_humanoid.yaml...")
    with open("configs/ppo_humanoid.yaml", 'r') as f:
        config = yaml.safe_load(f)
    print("Configuration loaded successfully.")
except FileNotFoundError:
    print("FATAL: Could not find configs/ppo_humanoid.yaml. Make sure you have created the file.")
    exit()

try:
    env_id = config["env"]["name"]
    env = gym.make(env_id)
    obs_space = env.observation_space
    act_space = env.action_space

    algo = PPOAlgorithm(config=config, obs_space=obs_space, act_space=act_space)
    print("Algorithm and Environment initialized successfully.")
except Exception as e:
    print(f"Error during initialization: {e}")
    exit()

print(f"Filling buffer with {algo.rollout_steps} steps of random data...")
current_obs, _ = env.reset()
current_obs = current_obs.astype(np.float32)

for i in range(algo.rollout_steps):
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    algo.buffer.add(
        obs=current_obs, action=action, reward=reward, done=done,
        log_prob=-1.0, value=0.5
    )
    
    current_obs = next_obs if not done else env.reset()[0]
    current_obs = current_obs.astype(np.float32)

algo.last_observation = current_obs
print("Buffer filled and last_observation is set.")

print("--- Calling update_policy() now... ---")
try:
    algo.update_policy()
    print("\n--- update_policy() completed without errors. ---")
except Exception as e:
    print(f"\n--- CRASH CAUGHT ---")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Details: {e}")

print("\n--- Debug Runner Finished ---")