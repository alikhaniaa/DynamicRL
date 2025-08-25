"""
    Abstract class for all RL algorithms. 
    //TODO after learning off-policy well, try to refactor this.
"""
from abc import ABC, abstractmethod
from typing import Any
from omegaconf import DictConfig
import torch
from .policies import PolicyNetworkBase, ValueNetworkBase
from .data_buffer import RolloutBuffer
from .types import ObservationType

class RLAlgorithm(ABC):
    def __init__(self, cfg: DictConfig, obs_space: Any, act_space: Any):
        self.cfg = cfg # Also change it here
        self.obs_space = obs_space
        self.act_space = act_space
        
        #Initialized params 
        self.policy: PolicyNetworkBase
        self.value_net: ValueNetworkBase
        self.optimizer: torch.optim.Optimizer
        
    #Data collection phase of the algorithm
    #TODO for on policy full rollout, maybe for off policy can only involve with single or fixed steps. check this out later
    @abstractmethod
    def collect_experiences(self, env: Any, current_obs: ObservationType) -> tuple[ObservationType, dict[str, float]]:
        pass
    
    #run the learning update phase of the algorithm
    @abstractmethod
    def update_policy(self) -> dict[str, float]:
        pass
    
    #get a serializable state of the algorithm for checkpoints
    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        pass

    #load the algorithm state from a chackpoint dict
    @abstractmethod
    def load_state(self, state: dict[str, Any]):
        pass