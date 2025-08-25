"""
    NN module for the agent. absrtact classes to define a contract for policies or value networks
"""
import logging
from abc import ABC, abstractmethod
from .types import ActionType, LogProbType, ObservationType, ValueType
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

logger = logging.getLogger(__name__)

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
    def __init__(self, obs_dim: int, act_space: gym.spaces.Space, hidden_sizes: tuple[int, ...] = (64, 64)):
        super().__init__()
        
        self.is_continuous = isinstance(act_space, gym.spaces.Box)
        if self.is_continuous:
            logger.info("Initializing ActorCriticMLP for a CONTINIUOUS action space")
        else:
            logger.info("Initializing ActorCriticMLP for a DISCRETE action space")
        
        #shared feature extracture
        shared_layers = []
        input_dim = obs_dim
        for hidden_dim in hidden_sizes:
            shared_layers.append(nn.Linear(input_dim, hidden_dim))
            shared_layers.append(nn.Tanh())
            input_dim = hidden_dim
        self.shared_net = nn.Sequential(*shared_layers)
        
        #Policy(actor)
        if self.is_continuous:
            act_dim = act_space.shape[0]
            self.policy_head = nn.Linear(input_dim, act_dim)
            #Learnable param for the SD of action distribution
            self.log_std = nn.Parameter(torch.zeros(act_dim))
        else:
            act_dim = act_space.n
            self.policy_head = nn.Linear(input_dim, act_dim)
        #Value(critic)
        self.value_head = nn.Linear(input_dim, 1)
                
    #Given observation and recieve Gaussian policy distribution
    def get_distribution(self, obs: ObservationType) -> torch.distributions.Distribution:
        features = self.shared_net(obs)
        if self.is_continuous:
            mean = self.policy_head(features)
            std = torch.exp(self.log_std)
            return Normal(mean, std)
        else: 
            logits = self.policy_head(features)
            return Categorical(logits=logits)
    
    #Sample action and logProb for data collection
    def forward(self, obs: ObservationType) -> tuple[ActionType, LogProbType]:
        dist = self.get_distribution(obs)
        action = dist.sample()
        if self.is_continuous:
            log_prob = dist.log_prob(action).sum(axis=-1)
        else: 
            log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def predict_value(self, obs: ObservationType) -> ValueType:
        features = self.shared_net(obs)
        return self.value_head(features).squeeze(-1)
    
    #Evaluate given action to compute their logProbs, policy entropy and state values --during-- PPO updates
    def evaluate_actions(self, obs: ObservationType, action: ActionType) -> tuple[LogProbType, torch.Tensor, ValueType]:
        dist = self.get_distribution(obs)
        
        if self.is_continuous:
            log_prob = dist.log_prob(action).sum(axis=-1)
        else:
            log_prob = dist.log_prob(action)
        
        entropy = dist.entropy().sum(axis=-1)
        value = self.predict_value(obs)
        
        return log_prob, entropy, value