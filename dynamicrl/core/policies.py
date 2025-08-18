"""
    NN module for the agent. absrtact classes to define a contract for policies or value networks
"""
from abc import ABC, abstractmethod
from .types import ActionType, LogProbType, ObservationType, ValueType

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

#Abstract base class for actor(policy network)
class PolicyNetworkBase(ABC, nn.Module):
    
    @abstractmethod
    def get_distribution(self, obs: ObservationType) -> torch.distributions.Distribution:
        pass
    
    #Perform forward pass to sample an action and it's log-probability during data collection
    @abstractmethod
    def forward(self, obs: ObservationType) -> tuple[ActionType, LogProbType]:
        pass
    
#Abstract base class for critic(value network)
class ValueNetworkBase(ABC, nn.Module):
    @abstractmethod
    def predict_value(self, obs: ObservationType) -> ValueType:
        pass
    
#Imp of Actor-Critic network with a shared MLP body designed for --continuous-- action spaces
class ActorCriticMLP(PolicyNetworkBase, ValueNetworkBase):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple[int, ...] = (64, 64)):
        super().__init__()
        
        #shared feature extracture
        shared_layers = []
        input_dim = obs_dim
        for hidden_dim in hidden_sizes:
            shared_layers.append(nn.Linear(input_dim, hidden_dim))
            shared_layers.append(nn.Tanh())
            input_dim = hidden_dim
        self.shared_net = nn.Sequential(*shared_layers)
        
        #Policy(actor)
        self.policy_head = nn.Linear(input_dim, act_dim)
        
        #Value(critic)
        self.value_head = nn.Linear(input_dim, 1)
        
        #Learnable param for the SD of action distribution
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
    #Given observation and recieve Gaussian policy distribution
    def get_distribution(self, obs: ObservationType) -> Normal:
        features = self.shared_net(obs)
        mean = self.policy_head(features)
        std = torch.exp(self.log_std)
        return Normal(mean, std)
    
    #Sample action and logProb for data collection
    def forward(self, obs: ObservationType) -> tuple[ActionType, LogProbType]:
        dist = self.get_distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action, log_prob
    
    def predict_value(self, obs: ObservationType) -> ValueType:
        features = self.shared_net(obs)
        return self.value_head(features).squeeze(-1)
    
    #Evaluate given action to compute their logProbs, policy entropy and state values --during-- PPO updates
    def evaluate_actions(self, obs: ObservationType, action: ActionType) -> tuple[LogProbType, torch.Tensor, ValueType]:
        dist = self.get_distribution(obs)
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        
        features = self.shared_net(obs)
        value = self.value_head(features).squeeze(-1)
        
        return log_prob, entropy, value