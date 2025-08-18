from typing import Generator
import torch
from .types import ActionType, Batch, DoneType, LogProbType, ObservationType, RewardType, ValueType

#Buffer for storing trajectories of experience for on-policy algorithms(for now PPO but logic can be applied without much of a refactor)
class RolloutBuffer:
    def __init__(self, buffer_size: int, obs_dim: int, act_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        
        #pre-allocated tensors for storage
        self.observation = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, act_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)        
        self.ptr = 0
        
    
    #Add a single transaction to the buffer
    def add(self, obs:ObservationType, action:ActionType, reward:float, done: bool, log_prob: float, value: float):
        if self.ptr >= self.buffer_size:
            raise ValueError("RolloutBuffer is full.")

        self.observation[self.ptr] = torch.as_tensor(obs, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, device=self.device)
        self.rewards[self.ptr] = torch.as_tensor(reward, device=self.device)
        self.dones[self.ptr] = torch.as_tensor(done, device=self.device)
        self.log_probs[self.ptr] = torch.as_tensor(log_prob, device=self.device)
        self.values[self.ptr] = torch.as_tensor(value, device=self.device)
        self.ptr += 1
        
    # Returns all stored data and resets the buffer.
    def get_all(self) -> dict[str, torch.Tensor]:
        
        if self.ptr != self.buffer_size:
            raise ValueError(f"Buffer is not full! Expected {self.buffer_size}, got {self.ptr}")
        
        data = {
            "observations": self.observation,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "log_probs": self.log_probs,
            "values": self.values,
        }
        self.ptr = 0
        return data