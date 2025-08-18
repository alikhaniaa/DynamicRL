# main.py
import gymnasium as gym
import yaml
import torch
import numpy as np
import os

from dynamicrl.algorithms.ppo.ppo_algorithm import PPOAlgorithm
from dynamicrl.core.trainer import Trainer

def main():
    """A simple entry point to run a short training test and record a video."""
    

    try:
        with open("configs/ppo_humanoid.yaml", 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("FATAL: Could not find configs/ppo_humanoid.yaml.")
        return

    env_id = config["env"]["name"]
    video_folder = "videos"
    

    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda episode_id: episode_id < 50,  
        name_prefix=f"{env_id}-run"
    )
    
    obs_space = env.observation_space
    act_space = env.action_space
    
    
    algorithm = PPOAlgorithm(config=config, obs_space=obs_space, act_space=act_space)
    trainer = Trainer(config=config, algorithm=algorithm, env=env)
    
    
    trainer.total_timesteps = 10000
    
    print(f"--- Starting a short test run for {trainer.total_timesteps} timesteps ---")
    print(f"A video of the first 3 episodes will be saved to the '{video_folder}/' directory.")
    

    try:
        trainer.train()
        print("\n--- Test run completed successfully! ---")
    except Exception as e:
        print(f"\n--- A CRITICAL ERROR OCCURRED DURING TRAINING ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    main()