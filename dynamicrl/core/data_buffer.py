from typing import Generator
import torch
import numpy as np
from .types import ActionType, Batch, DoneType, LogProbType, ObservationType, RewardType, ValueType

#Buffer for storing trajectories of experience for on-policy algorithms(for now PPO but logic can be applied without much of a refactor)
class RolloutBuffer:
    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int, device: torch.device, num_envs: int):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        
        #pre-allocated tensors for storage
        self.observations = torch.zeros((buffer_size, num_envs, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, num_envs, act_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((buffer_size, num_envs), dtype=torch.float32, device=device)        
        self.ptr = 0
        self.full = False
        
    
    #Add a single transaction to the buffer
    def add(self, obs:ObservationType, action:ActionType, reward:float, done: bool, log_prob: float, value: float):
        if self.ptr >= self.buffer_size:
            raise ValueError("RolloutBuffer is full.")
        
        if len(action.shape) == 1:
            action = np.expand_dims(action, axis=-1)

        self.observations[self.ptr] = torch.as_tensor(obs, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, device=self.device)
        self.rewards[self.ptr] = torch.as_tensor(reward, device=self.device)
        self.dones[self.ptr] = torch.as_tensor(done, device=self.device)
        self.log_probs[self.ptr] = torch.as_tensor(log_prob, device=self.device)
        self.values[self.ptr] = torch.as_tensor(value, device=self.device)
        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.full = True
        
    # Returns all stored data and resets the buffer.
    def get_all(self) -> dict[str, torch.Tensor]:
        
        if not self.full:
            raise ValueError(f"Buffer is not full! Expected {self.buffer_size}, got {self.ptr}")
        
        data = {
            "observations": self.observations.reshape(self.buffer_size * self.num_envs, -1),
            "actions": self.actions.reshape(self.buffer_size * self.num_envs, -1),
            "rewards": self.rewards.flatten(),
            "dones": self.dones.flatten(),
            "log_probs": self.log_probs.flatten(),
            "values": self.values.flatten(),
        }
        return data

    def reset(self):
        self.ptr = 0
        self.full = False
        
    def get_all_flattened(self) -> dict[str, torch.Tensor]:
        if not self.full:
            raise ValueError(f"Buffer not full!")
        
        if self.actions.shape[-1] == 1:
            actions = self.actions.reshape(-1)
        else:
            actions = self.actions.reshape(-1, self.actions.shape[-1])

        data = {
            "observations": self.observations.reshape(-1, self.observations.shape[-1]),
            "actions": actions,
            "rewards": self.rewards.flatten(),
            "dones": self.dones.flatten(),
            "log_probs": self.log_probs.flatten(),
            "values": self.values.flatten(),
        }
        return data
    
    def get_all_unflattened(self) -> dict[str, torch.Tensor]:
        if not self.full:
             raise ValueError(f"Buffer not full!")
        
        return {
            "rewards": self.rewards,
            "dones": self.dones,
            "values": self.values,
        }